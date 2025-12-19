import torch
import torch.nn as nn
import numpy as np

class RandomizedParameterMaskAttack:
    def __init__(self, model, loss_fn, eps=0.03,
                 sparsity_level=0.7, dynamic_sparsity=False,
                 input_types=None, target_class_idx=None,
                 use_cuda=True):

        self.model = model
        self.loss_fn = loss_fn
        self.eps = eps
        self.sparsity_level = sparsity_level
        self.dynamic_sparsity = dynamic_sparsity
        self.input_types = input_types or {}
        self.target_class_idx = target_class_idx
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

        # 记录原始状态以便恢复
        self.original_state_dict = None
        self.random_seed_base = 42

    def _create_random_masks_for_all_params(self, iteration_num):

        masks = {}
        rng = np.random.RandomState(self.random_seed_base + iteration_num)

        current_sparsity = self.sparsity_level
        if self.dynamic_sparsity:

            adaptive_factor = min(1.0, iteration_num / 100.0)
            current_sparsity = max(0.05, self.sparsity_level * adaptive_factor)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                shape = param.data.shape


                random_values = rng.uniform(size=np.prod(shape))
                threshold = np.percentile(random_values, current_sparsity * 100)

                binary_mask = torch.from_numpy(random_values <= threshold).float()
                binary_mask = binary_mask.view(shape).to(self.device)

                masks[name] = binary_mask

        return masks

    def _apply_param_masking(self, masks, mode='both'):

        masked_params = {}

        for name, param in self.model.named_parameters():
            if name in masks:
                if mode in ['forward', 'both']:
                    # 保存原始参数并应用掩码
                    masked_params[name] = param.data.clone()
                    param.data = param.data * masks[name]

        return masked_params

    def _restore_parameters(self, masked_params):

        for name, param in self.model.named_parameters():
            if name in masked_params:
                param.data.copy_(masked_params[name])

    def generate_varied_samples(self, *inputs, labels, num_variants=10,
                                batch_size=None, iteration_offset=0):
        all_adv_inputs = []

        for variant_idx in range(num_variants):

            masks = self._create_random_masks_for_all_params(
                variant_idx + iteration_offset
            )


            adv_inputs = self._generate_single_variant(
                inputs, labels, masks, variant_idx
            )

            all_adv_inputs.append(adv_inputs)

        return all_adv_inputs

    def _generate_single_variant(self, inputs, labels, masks, variant_idx):


        self._save_original_state()

        try:

            masked_params = self._apply_param_masking(masks, mode='both')


            inputs_requires_grad = []
            for i, input_data in enumerate(inputs):
                if self.input_types.get(f'input_{i}', True):
                    inputs_requires_grad.append(
                        input_data.detach().clone().requires_grad_(True)
                    )
                else:
                    inputs_requires_grad.append(input_data.detach().clone())


            outputs = self.model(*inputs_requires_grad)


            if self.target_class_idx is not None:
                target_labels = torch.full_like(labels, self.target_class_idx)
                loss = -self.loss_fn(outputs, target_labels)
            else:
                loss = self.loss_fn(outputs, labels)


            self.model.zero_grad()
            loss.backward()

            adv_inputs = []
            for i, (input_data, input_var) in enumerate(zip(inputs, inputs_requires_grad)):
                if hasattr(input_var, 'grad') and input_var.grad is not None:
                    #                    # 应用随机因子进一步增强多样性
                    noise_factor = 0.8 + 0.4 * np.sin(variant_idx * 15)  # 周期性变化的噪音因子

                    gradient_sign = input_var.grad.sign()
                    perturbation = self.eps * gradient_sign * noise_factor

                    adv_input = input_data + perturbation


                    if self.input_types.get(f'input_{i}_clip', False):
                        adv_input = torch.clamp(adv_input, 0, 1)

                    adv_inputs.append(adv_input.detach())
                else:
                    adv_inputs.append(input_data)

            return adv_inputs

        finally:

            self._restore_from_backup()

    def _save_original_state(self):

        if self.original_state_dict is None:
            self.original_state_dict = {
                name: param.data.clone()
                for name, param in self.model.named_parameters()
            }

    def _restore_from_backup(self):
   
        if self.original_state_dict is not None:
            for name, param in self.model.named_parameters():
                if name in self.original_state_dict:
                    param.data.copy_(self.original_state_dict[name])


