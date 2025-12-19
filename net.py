import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock2D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super(BasicBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.extra = nn.Sequential()
        if output_channels != input_channels:
            self.extra = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(output_channels)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        identity = self.extra(identity)
        out = F.relu(out + identity)

        return out

class SC_model(nn.Module):
    def __init__(self):
        super(SC_model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(9, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16)
        )
        self.bk1 = BasicBlock2D(16, 16)
        self.bk2 = BasicBlock2D(16, 32)
        self.bk3 = BasicBlock2D(32, 64)
        self.bk4 = BasicBlock2D(64, 32)
        self.bk5 = BasicBlock2D(32, 16)
        self.bk6 = BasicBlock2D(16, 9)
        self.bk6 = BasicBlock2D(16, 1)
        self.pool = nn.AvgPool2d(kernel_size=(13, 1), stride=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bk1(out)
        out = self.bk2(out)
        out = self.bk3(out)
        out = self.bk4(out)
        out = self.bk5(out)
        out = self.bk6(out)

        out = self.pool(out)

        return out

class BasicBlock_3D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super(BasicBlock_3D, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm3d(output_channels)
        self.bn2 = nn.BatchNorm3d(output_channels)

        self.extra = nn.Sequential()
        if input_channels != output_channels:
            self.extra = nn.Sequential(
                nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=1, stride=1)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        identity = self.extra(identity)
        out = F.relu(out + identity)

        return out


class seismic_model(nn.Module):
    def __init__(self):
        super(seismic_model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(16)
        )
        self.bk1 = BasicBlock_3D(16, 16)
        self.bk2 = BasicBlock_3D(16, 32)
        self.bk3 = BasicBlock_3D(32, 64)
        self.bk4 = BasicBlock_3D(64, 32)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bk1(x)
        x = self.bk2(x)
        x = self.bk3(x)
        x = self.bk4(x)


        return x

class cls_model(nn.Module):
    def __init__(self):
        super(cls_model, self).__init__()
        self.model1 = seismic_model()
        self.model2 = SC_model()
        self.outlayer = nn.Linear(3744, 6)
        self.SP = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        out1 = self.model1(x)
        out1 = self.dropout(out1)
        out2 = self.model2(y)
        out2 = self.dropout(out2)
        out2 = out2.unsqueeze(1).repeat(1, out1.size(1), 1, 1, 1)
        out = out1 + out2
        out = out.view(out.size(0), -1)
        out = self.outlayer(out)
        return out


class reg_model(nn.Module):
    def __init__(self):
        super(reg_model, self).__init__()
        self.model1 = seismic_model()
        self.model2 = SC_model()
        self.outlayer = nn.Linear(3744, 1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x, y):
        out1 = self.model1(x)
        out1 = self.dropout(out1)
        out2 = self.model2(y)  
        out2 = self.dropout(out2)
        out2 = out2.unsqueeze(1).repeat(1, out1.size(1), 1, 1, 1)
        out = out1 + out2
        out = out.view(out.size(0), -1)
        out = self.outlayer(out)
        return out
