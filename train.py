import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from net import cls_model, reg_model
from attack import RandomizedParameterMaskAttack
import argparse
import warnings

warnings.filterwarnings('ignore')

np.set_printoptions(threshold=np.inf, precision=3, suppress=True, linewidth=100)
cluster_mid = [22.5, 18, 16, 13, 11, 9]

parser = argparse.ArgumentParser(description='Deep Co-Training for Reservoir Prediction')
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--batch', '-b', default=14, type=int)
parser.add_argument('--seed', default=66, type=int)
parser.add_argument('--epochs', default=600, type=int)
parser.add_argument('--epsilon', default=0.02, type=float)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--input_types', default={'input_0': True, 'input_1': False}, type=dict)
parser.add_argument('--eps', default=0.01, type=float)
parser.add_argument('--num_variants', default=10, type=float)
args = parser.parse_args()

torch.manual_seed(args.seed)

def calculate_P(pre, sigma=1):
    centers = torch.tensor(cluster_mid,
                        dtype=pre.dtype,
                        device=pre.device).view(1, -1)


    data_cov = torch.var(pre, unbiased=False)

    inv_cov = 1 / data_cov

    diff = pre - centers

    squared_mahal_dist = diff ** 2 * inv_cov

    prob = torch.exp(-squared_mahal_dist / (2 * sigma ** 1))

    return (prob)

def loss_sup(pre_reg, pre_cls, label_reg, label_cls):
    pre_reg = pre_reg.squeeze(1)
    loss1 = ce(pre_cls, label_cls)
    loss2 = se(pre_reg, label_reg)
    return loss1, loss2


class CoTLoss(nn.Module):
    def __init__(self, grad_mask=(1, 1), eps=1e-8):
        super(CoTLoss, self).__init__()
        self.eps = eps
        valid_masks = [(1, 1), (1, 0), (0, 1)]
        if grad_mask not in valid_masks:
            raise ValueError(f"grad_mask should be one of {valid_masks}, got {grad_mask}")
        self.grad_mask = grad_mask
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, pre_reg, pre_cls):
        batch_size = pre_reg.size(0)
        if self.grad_mask[0] == 0:
            pre_reg = pre_reg.detach()
        if self.grad_mask[1] == 0:
            pre_cls = pre_cls.detach()

        mask1 = torch.argmax(pre_reg, dim=1)
        mask2 = torch.argmax(pre_cls, dim=1)
        mask = (abs(mask1 - mask2)).float().view(-1, 1)

        prob_reg = self.softmax(pre_reg)
        prob_cls = self.softmax(pre_cls)
        avg_prob = 0.5 * (prob_reg + prob_cls)

        entropy_avg = avg_prob * torch.log(avg_prob + self.eps)
        entropy_avg = entropy_avg * mask
        loss1 = -torch.sum(entropy_avg)

        ce_reg = prob_reg * self.log_softmax(pre_reg)
        ce_reg = ce_reg * mask
        loss2 = -torch.sum(ce_reg)

        ce_cls = prob_cls * self.log_softmax(pre_cls)
        ce_cls = ce_cls * mask
        loss3 = -torch.sum(ce_cls)

        final_loss = loss1 - 0.5 * (loss2 + loss3)
        final_loss = final_loss / batch_size
        return final_loss


def calculate_accuracy(predictions, labels, threshold=1):
    labels = labels.view(-1, 1)
    assert predictions.shape == labels.shape, "The predicted value and label shape are inconsistent"
    absolute_errors = torch.abs(predictions - labels)
    correct = absolute_errors < threshold
    accuracy = correct.float().mean().item()
    return accuracy


def Generate_data(model_cls, model_reg, data, SC, label_cls, label_reg):
    attacker_cls = RandomizedParameterMaskAttack(
        model=model_cls, loss_fn=nn.CrossEntropyLoss(), eps=args.eps,
        sparsity_level=0.65, dynamic_sparsity=True,
        input_types=args.input_types, target_class_idx=None
    )
    attacker_reg = RandomizedParameterMaskAttack(
        model=model_reg, loss_fn=nn.MSELoss(), eps=args.eps,
        sparsity_level=0.65, dynamic_sparsity=True,
        input_types=args.input_types, target_class_idx=None
    )
    variants_cls = attacker_cls.generate_varied_samples(
        data, SC, labels=label_cls, num_variants=args.num_variants, iteration_offset=100
    )
    variants_reg = attacker_reg.generate_varied_samples(
        data, SC, labels=label_reg, num_variants=args.num_variants, iteration_offset=100
    )

    label_reg_P = torch.argmax(calculate_P(label_reg.reshape(-1, 1)), dim=1)
    selected_variant_cls = []
    selected_variant_reg = []

    for _, adv_sample in enumerate(variants_cls):
        adv_data, adv_SC = adv_sample
        cls_pre = torch.argmax(model_cls(adv_data, adv_SC), dim=1)
        reg_pre = torch.argmax(calculate_P(model_reg(adv_data, adv_SC)), dim=1)
        for j in range(len(adv_data)):
            if cls_pre[j] == label_cls[j] and reg_pre[j] != label_cls[j]:
                selected_variant_cls.append([adv_data[j].squeeze(), adv_SC[j], label_reg[j]])

    for i, adv_sample in enumerate(variants_reg):
        adv_data, adv_SC = adv_sample
        cls_pre = torch.argmax(model_cls(adv_data, adv_SC), dim=1)
        reg_pre = torch.argmax(calculate_P(model_reg(adv_data, adv_SC)), dim=1)
        for j in range(len(adv_data)):
            if cls_pre[j] != label_reg_P[j] and reg_pre[j] == label_reg_P[j]:
                selected_variant_reg.append([adv_data[j].squeeze(), adv_SC[j], label_reg_P[j]])

    return selected_variant_cls, selected_variant_reg


if __name__ == "__main__":
    cls_acc = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model1 = cls_model().to(device)
    model2 = reg_model().to(device)
    acc1, acc2, acc3, total_acc = [0], [0], [0], [0]


    datasets = np.load('datasets.npz')
    datasets_unlabel = np.load('datasets_unlabel.npz')
    # train_data
    train_data = torch.from_numpy(datasets['train_data']).float()
    train_SC = torch.from_numpy(datasets['train_SC']).float()
    train_label_cls = torch.from_numpy(datasets['train_label_cls']).long()
    train_label_reg = torch.from_numpy(datasets['train_label_reg']).float()

    # val_data
    val_data = torch.from_numpy(datasets['val_data']).float()
    val_SC = torch.from_numpy(datasets['val_SC']).float()
    val_label_cls = torch.from_numpy(datasets['val_label_cls']).long()
    val_label_reg = torch.from_numpy(datasets['val_label_reg']).float()

    # data_diff
    diff_data = torch.from_numpy(datasets['diff_data']).float()
    diff_SC = torch.from_numpy(datasets['diff_SC']).float()
    diff_label = torch.from_numpy(datasets['diff_label']).float()

    # unlabel_data
    unlabel_data = torch.from_numpy(datasets_unlabel['unlabel_data']).float()
    unlabel_SC = torch.from_numpy(datasets_unlabel['unlabel_SC']).float()


    dataset_train = TensorDataset(train_data, train_SC, train_label_cls, train_label_reg)
    dataset_val = TensorDataset(val_data, val_SC, val_label_cls, val_label_reg)
    dataset_train_more = TensorDataset(diff_data, diff_SC,  diff_label)
    unlabel_triandata = TensorDataset(unlabel_SC, unlabel_data)



    train_dataloder_reg = DataLoader(dataset_train_more, batch_size=int(len(dataset_train_more) / args.batch),
                                     shuffle=True, drop_last=True)
    train_dataloder2 = DataLoader(unlabel_triandata, batch_size=int(len(unlabel_triandata)/ args.batch) + 1,
                                  shuffle=True, drop_last=True)
    val_dataloder = DataLoader(dataset_val, batch_size=len(dataset_val), shuffle=False)
    val_dataloder2 = DataLoader(unlabel_triandata, batch_size=len(unlabel_triandata), shuffle=False)


    gen_dataloder = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=False)



    optim1 = torch.optim.Adam(model1.parameters(), args.lr)
    optim2 = torch.optim.Adam(model2.parameters(), args.lr)


    se = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    criterion = CoTLoss().to(device)


    cls_GenDataLoder = []
    reg_GenDataLoder = []
    prior_pre = []
    merged_dataset = dataset_train

    model1.train()
    model2.train()

    for i in range(args.epochs):
        print(f"Round {i+1} of training")
        epoch_loss1, epoch_loss2 = [], []

        if (i + 1) % 50 == 0:

            model1.eval()
            model2.eval()
            with torch.no_grad():
                iter_gen = iter(gen_dataloder)
                data_gen, sc_gen, cls_label_gen, reg_label_gen = next(iter_gen)
                data_gen = data_gen.to(device)
                sc_gen = sc_gen.to(device)
                cls_label_gen = cls_label_gen.long().to(device)
                reg_label_gen = reg_label_gen.to(device)

            model1.train()
            model2.train()
            cls_generate, reg_generate = Generate_data(
                model1, model2, data_gen, sc_gen, cls_label_gen, reg_label_gen
            )

            if len(cls_generate) > 0:
                cls_Generate_data = torch.stack([x[0] for x in cls_generate])
                cls_Generate_SC = torch.stack([x[1] for x in cls_generate])
                cls_Generate_label = torch.stack([x[2] for x in cls_generate])
                cls_GenDataLoder = DataLoader(
                    TensorDataset(cls_Generate_data, cls_Generate_SC, cls_Generate_label),
                    batch_size=BATCH_SIZE, shuffle=True, drop_last=True
                )

            if len(reg_generate) > 0:
                reg_Generate_data = torch.stack([x[0] for x in reg_generate])
                reg_Generate_SC = torch.stack([x[1] for x in reg_generate])
                reg_Generate_label = torch.stack([x[2] for x in reg_generate])
                reg_GenDataLoder = DataLoader(
                    TensorDataset(reg_Generate_data, reg_Generate_SC, reg_Generate_label),
                    batch_size=BATCH_SIZE, shuffle=True, drop_last=True
                )


        merged_train_loader = DataLoader(merged_dataset, batch_size=int(len(merged_dataset) / 14),
                                         shuffle=True, drop_last=True)
        iter_main = iter(merged_train_loader)
        iter_reg = iter(train_dataloder_reg)

        # 生成数据迭代器（如果存在）
        iter_gen_cls = iter(cls_GenDataLoder) if cls_GenDataLoder != [] else None
        iter_gen_reg = iter(reg_GenDataLoder) if reg_GenDataLoder != [] else None




        for batch_idx in range(args.batch):
            optim1.zero_grad()

            accumulated_cls_loss = 0.0
            total_cls_samples = 0

            data, SC, cls_label, reg_label = next(iter_main)
            data, SC = data.to(device), SC.to(device)
            cls_label, reg_label = cls_label.to(device), reg_label.to(device)
            batch_size = data.size(0)


            out1 = model1(data, SC)
            out2 = model2(data, SC)
            loss1, loss2 = loss_sup(out2, out1, reg_label, cls_label)


            optim2.zero_grad()
            loss2.backward()
            optim2.step()

            data_reg, SC_reg,  reg_label1 = next(iter_reg)
            data_reg, SC_reg = data_reg.to(device), SC_reg.to(device)
            reg_label1 = reg_label1.to(device)
            batch_size_reg = data_reg.size(0)


            out2_2 = model2(data_reg, SC_reg)
            out1_2 = model1(data_reg, SC_reg)

            loss_reg = se(out2_2.squeeze(1), reg_label1)
            loss_diff = criterion(calculate_P(reg_label1.detach().reshape(-1, 1)), out1_2)


            optim2.zero_grad()
            loss_total_reg = loss_reg
            loss_total_reg.backward()
            optim2.step()


            if iter_gen_cls is not None and iter_gen_reg is not None:
                try:
                    gen_cls_data, gen_cls_SC, gen_reg_label = next(iter_gen_cls)
                    gen_reg_data, gen_reg_SC, gen_cls_label = next(iter_gen_reg)

                    gen_cls_data = gen_cls_data.to(device)
                    gen_cls_SC = gen_cls_SC.to(device)
                    gen_reg_label = gen_reg_label.to(device)
                    gen_reg_data = gen_reg_data.to(device)
                    gen_reg_SC = gen_reg_SC.to(device)
                    gen_cls_label = gen_cls_label.to(device)
                    batch_size_gen = gen_cls_data.size(0)


                    out_gen_cls = model1(gen_reg_data, gen_reg_SC)
                    out_gen_reg = model2(gen_cls_data, gen_cls_SC)

                    loss_gen_reg = se(out_gen_reg.squeeze(1), gen_reg_label)
                    loss_gen_cls = ce(out_gen_cls, gen_cls_label)


                    optim2.zero_grad()
                    loss_gen_reg.backward()
                    optim2.step()


                    accumulated_cls_loss += loss_gen_cls
                except StopIteration:
                    iter_gen_cls = iter(cls_GenDataLoder) if cls_GenDataLoder != [] else None
                    iter_gen_reg = iter(reg_GenDataLoder) if reg_GenDataLoder != [] else None


            final_cls_loss = loss1 + loss_diff + accumulated_cls_loss
            final_cls_loss.backward()
            optim1.step()
            optim1.zero_grad()


        model1.eval()
        model2.eval()
        with torch.no_grad():
            val_data_batch, val_SC_batch, cls_label, reg_label = next(iter(val_dataloder))
            val_data_batch, val_SC_batch = val_data_batch.to(device), val_SC_batch.to(device)
            cls_label, reg_label = cls_label.to(device), reg_label.to(device)

            un_SC, un_data = next(iter(val_dataloder2))
            un_SC, un_data = un_SC.to(device), un_data.to(device)

            out1 = model1(val_data_batch, val_SC_batch)
            out2 = model2(val_data_batch, val_SC_batch)

            out1_U = model1(un_data, un_SC)
            out2_U = model2(un_data, un_SC)


            if (i + 1) % 100 == 0:
                pred1 = torch.argmax(out1_U, dim=1)
                pred2 = torch.argmax(calculate_P(out2_U), dim=1)
                class_diff = torch.abs(pred1 - pred2)
                conf = torch.max(nn.Softmax(dim=1)(out1_U), dim=1)[0]

                if i > 100 and len(prior_pre) > 0:
                    reliable_indices = (
                            ((out2_U - prior_pre).abs() <= 0.25).squeeze() &
                            (class_diff == 0) &
                            (conf >= 0.7)
                    ).nonzero(as_tuple=True)[0]

                    reliable_data = un_data[reliable_indices]
                    reliable_sc = un_SC[reliable_indices]
                    reliable_cls_labels = pred1[reliable_indices]
                    reliable_reg_labels = out2_U[reliable_indices]


                    reliable_reg_labels = reliable_reg_labels.squeeze()


                    pseudo_dataset = TensorDataset(
                        reliable_data.cpu(),
                        reliable_sc.cpu(),
                        reliable_cls_labels.cpu(),
                        reliable_reg_labels.cpu()
                    )


                    original_reg_labels_1d = merged_dataset.tensors[3].squeeze()
                    pseudo_reg_labels_1d = pseudo_dataset.tensors[3].squeeze()

                    original_data = torch.cat([merged_dataset.tensors[0], pseudo_dataset.tensors[0]], dim=0)
                    original_SC = torch.cat([merged_dataset.tensors[1], pseudo_dataset.tensors[1]], dim=0)
                    original_cls_labels = torch.cat([merged_dataset.tensors[2], pseudo_dataset.tensors[2]], dim=0)
                    original_reg_labels = torch.cat([original_reg_labels_1d, pseudo_reg_labels_1d], dim=0)

                    merged_dataset = TensorDataset(
                        original_data,
                        original_SC,
                        original_cls_labels,
                        original_reg_labels
                    )


                prior_pre = out2_U


            out1_pred = torch.argmax(out1, dim=1)
            out2_U_cls = torch.argmax(calculate_P(out2_U), dim=1)
            out1_U_cls = torch.argmax(out1_U, dim=1)

            out2_for_acc = out2.squeeze() if out2.dim() > 1 else out2

            acc_cls = (out1_pred == cls_label).float().mean().item()


            acc2.append(acc_cls)

            print(
                f"Epoch {i + 1} | cls:{acc_cls:.2%}  | Max cls:{max(acc2):.2%} | 当前数据集大小:{len(merged_dataset)}")
            cls_acc.append(acc_cls)
        model1.train()
        model2.train()


    np.save('acc.npy', np.array(cls_acc))