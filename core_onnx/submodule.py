import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class BasicConv_IN(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, **kwargs):
        super(BasicConv_IN, self).__init__()

        self.use_relu = relu
        self.use_in = IN

        if is_3d:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs) if deconv else nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.norm = nn.InstanceNorm3d(out_channels) if IN else nn.Identity()
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs) if deconv else nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.norm = nn.InstanceNorm2d(out_channels) if IN else nn.Identity()

        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=False) if relu else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class Conv2x_IN(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, IN=True, relu=True, keep_dispc=False):
        super(Conv2x_IN, self).__init__()
        self.concat = concat
        self.is_3d = is_3d

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
        elif deconv and is_3d:
            kernel = (4, 4, 4)
            stride = 2
            padding = 1
        elif deconv:
            kernel = 4
            stride = 2
            padding = 1
        else:
            kernel = 3
            stride = 2
            padding = 1

        self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN, relu,
                                  kernel_size=kernel, stride=stride, padding=padding)

        mult = 2 if concat and keep_concat else 1
        conv2_in = out_channels * 2 if concat else out_channels
        conv2_out = out_channels * mult

        self.conv2 = BasicConv_IN(conv2_in, conv2_out, False, is_3d, IN, relu,
                                  kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape[-2:] != rem.shape[-2:]:
            x = F.interpolate(x, size=rem.shape[-2:], mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), dim=1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x
