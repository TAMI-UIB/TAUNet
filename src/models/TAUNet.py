import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.utils.AttentionLayers import  TSCA


class TAUNet(nn.Module):
    def __init__(self, input_channels, hidden_channels, n_classes, patch_size=1):
        super(TAUNet, self).__init__()

        self.doble_conv = DobleConv(input_channels+2, 2 * hidden_channels, hidden_channels)
        self.down_block1 = DownBlock(hidden_channels, 2 * hidden_channels, hidden_channels)
        self.down_block2 = DownBlock(hidden_channels, 2 * hidden_channels, hidden_channels, kernel_size=3)
        self.down_block3 = DownBlock(hidden_channels, 2 * hidden_channels, hidden_channels)
        self.mha = TSCA(hidden_channels, num_heads=8, patch_size=1)
        self.up_block1 = UpBlock(hidden_channels, hidden_channels, 2 * hidden_channels, hidden_channels, patch_size=1)
        self.up_block2 = UpBlock(hidden_channels, hidden_channels, 2 * hidden_channels, hidden_channels, patch_size=patch_size, kernel_size=3)
        self.up_block3 = UpBlock(hidden_channels, hidden_channels, 2 * hidden_channels, hidden_channels, patch_size=patch_size)
        self.out = nn.Conv2d(hidden_channels, n_classes, kernel_size=1)

    def forward(self, x):
        fdi, ndvi = self.get_indices(x)
        x = torch.cat([x, fdi, ndvi], dim=1)
        x1 = self.doble_conv(x)
        x2 = self.down_block1(x1)
        x3 = self.down_block2(x2)
        y = self.down_block3(x3)
        y = self.mha(y)
        y = self.up_block1(y, x3)
        y = self.up_block2(y, x2)
        y = self.up_block3(y, x1)
        out=self.out(y)

        return out

    def get_indices(self, x):
        lmbd = (842-750)/(1610-750)
        fdi = x[:,[7],:,:]-( x[:,[5],:,:] + (x[:,[10],:,:]- x[:,[5],:,:]) * lmbd * 10.)
        ndvi = (x[:,[7],:,:]- x[:,[3],:,:]) / torch.max(x[:,[7],:,:]+ x[:,[3],:,:], 1.e-8*torch.ones_like(x[:,[7],:,:]))
        return fdi, ndvi
class DobleConv(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super(DobleConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm2d(mid_channels)
        self.bn2 = nn.InstanceNorm2d(out_channels)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.ReLU(self.bn1(self.conv1(x)))
        x = self.ReLU(self.bn2(self.conv2(x)))
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=2):
        super(DownBlock, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size)
        self.doble_conv = DobleConv(in_channels, mid_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        return self.doble_conv(x) + x


class UpBlock(nn.Module):
    def __init__(self, in_channels, mid_channels1, mid_channels2, out_channels, patch_size, kernel_size=2):
        super(UpBlock, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, mid_channels1, kernel_size=kernel_size, stride=kernel_size)
        self.cmha = TSCA(mid_channels1, num_heads=8, patch_size=patch_size)
        self.doble_conv = DobleConv(mid_channels1 * 2, mid_channels2, out_channels)

    def forward(self, x, conc_layer):
        x1 = self.up(x)

        diffY = conc_layer.size()[2] - x1.size()[2]
        diffX = conc_layer.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x1 = self.cmha(x1, conc_layer)
        x = torch.cat([x1, conc_layer], dim=1)
        return self.doble_conv(x) + x1