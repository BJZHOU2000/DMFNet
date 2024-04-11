import torch
import math
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# SE注意力机制（Squeeze-and-Excitation Networks）在通道维度增加注意力机制
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
# 非对称的上下文调制 类似于一个注意力机制去进行特征的跨层融合
class AsymBiChaFuse(nn.Module):
    def __init__(self, channels, r=4):
        super(AsymBiChaFuse, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // r)

        self.topdown_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.channels, out_channels=self.bottleneck_channels, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(self.bottleneck_channels, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.bottleneck_channels, out_channels=self.channels, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(self.channels, momentum=0.9),
            nn.Sigmoid()
        )

        self.topdown_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.channels, out_channels=self.bottleneck_channels, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(self.bottleneck_channels, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.bottleneck_channels, out_channels=self.channels, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(self.channels, momentum=0.9),
            nn.Sigmoid()
        )

        self.bottomup_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.bottleneck_channels, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(self.bottleneck_channels, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.bottleneck_channels, out_channels=self.channels, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(self.channels, momentum=0.9),
            nn.Sigmoid()
        )

        self.bottomup_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.bottleneck_channels, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(self.bottleneck_channels, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.bottleneck_channels, out_channels=self.channels, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(self.channels, momentum=0.9),
            nn.Sigmoid()
        )

        self.post_1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(channels, momentum=0.9),
            nn.ReLU(inplace=True)
        )

        self.post_2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(channels, momentum=0.9),
            nn.ReLU(inplace=True)
        )

        self.post = nn.Sequential(
            # nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(channels, momentum=0.9),
            nn.ReLU(inplace=True)
        )

    def forward(self, xh, xl_1, xl_2):
        topdown_wei_1 = self.topdown_1(xh)
        topdown_wei_2 = self.topdown_2(xh)
        bottomup_wei_1 = self.bottomup_1(xl_1)
        bottomup_wei_2 = self.bottomup_2(xl_2)
        xs_1 = 2 * torch.mul(xl_1, topdown_wei_1) + 2 * torch.mul(xh, bottomup_wei_1)
        xs_2 = 2 * torch.mul(xl_2, topdown_wei_2) + 2 * torch.mul(xh, bottomup_wei_2)
        xs = self.post_1(xs_1) + self.post_1(xs_2)
        xs = self.post(xs)
        return xs

class DilationModule(nn.Module):
    def __init__(self, inc, outc, kernel_size, padding, dilation):
        super(DilationModule, self).__init__()
        self.conv1 = nn.Sequential(nn.BatchNorm2d(inc),
                                   nn.Conv2d(inc, outc, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
                                   nn.GroupNorm(32, outc),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(outc),
                                   nn.Conv2d(outc, inc, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
                                   nn.GroupNorm(32, inc),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.BatchNorm2d(inc),
                                   nn.Conv2d(inc, outc, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
                                   nn.GroupNorm(32, outc),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(outc),
                                   nn.Conv2d(outc, inc, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
                                   nn.GroupNorm(32, inc),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.BatchNorm2d(inc),
                                   nn.Conv2d(inc, outc, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
                                   nn.GroupNorm(32, outc),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(outc),
                                   nn.Conv2d(outc, inc, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
                                   nn.GroupNorm(32, inc),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = x + x1
        x2 = self.conv2(x1)
        x2 = x1 + x2
        x3 = self.conv3(x2)
        x3 = x2 + x3
        return x3

class Multiple(nn.Module):
    def __init__(self, in_c, out_c):
        super(Multiple, self).__init__()
        self.dilation1 = DilationModule(in_c, out_c, kernel_size=3, padding=1, dilation=1)
        self.dilation2 = DilationModule(in_c, out_c, kernel_size=3, padding=2, dilation=2)
        self.dilation3 = DilationModule(in_c, out_c, kernel_size=3, padding=4, dilation=4)

        self.conv = nn.Conv2d(in_c * 3, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.dilation1(x)
        x2 = self.dilation2(x)
        x3 = self.dilation3(x)
        x = torch.cat((x1, x2, x3), dim=1)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return self.dropout(x)

class Fusion1(nn.Module):
    def __init__(self, in_ch):
        super(Fusion1, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, in_ch//2, 1, bias=False),
                                   nn.BatchNorm2d(in_ch//2),
                                   nn.ReLU(inplace=True))
        # self.conv2 = nn.Sequential(nn.Conv2d(in_ch, in_ch // 2, 1, bias=False),
        #                            nn.BatchNorm2d(in_ch // 2),
        #                            nn.ReLU(inplace=True))
        self.multiple = Multiple(in_ch//2, in_ch)
        self.conv3 = nn.Sequential(  # nn.Conv2d(in_ch*2, in_ch, 1, bias=False),
                                   nn.BatchNorm2d(in_ch),
                                   nn.ReLU(inplace=True))
        self.se = SELayer(in_ch)
        self.conv4 = nn.Sequential(  # nn.Conv2d(in_ch*2, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(  # nn.Conv2d(in_ch*2, in_ch, 1, bias=False),
                                nn.BatchNorm2d(in_ch),
                              nn.ReLU(inplace=True))

    def forward(self, e):
        # #e1 = self.conv1(e)
        # #s1 = self.conv2(s)
        # out = e + s
        # out = self.multiple(out)
        # e_out = out * e
        # s_out = out * s
        # out = torch.cat((e_out, s_out), dim=1)
        # # out = self.conv3(out)
        
        e1 = self.conv1(e)
        out = self.multiple(e1)
        e_out = out * e
        out = e + e_out
        out = self.conv3(out)

        ca = self.se(out)
        out = out + ca
        out = self.conv4(out)

        out = out + e
        out = self.conv5(out)

        return out

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            downsample=None,
            norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class unet_resnet_18(nn.Module):
    def __init__(self, block=BasicBlock, inplanes=[3, 16, 32, 64, 128, 256], layers=[2, 2, 2, 2]):
        super(unet_resnet_18, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.layer_0 = self._make_layer(block, inplanes[0], inplanes[1], 1, stride=1)
        self.layer_1 = self._make_layer(block, inplanes[1], inplanes[2], layers[0], stride=1)
        self.layer_2 = self._make_layer(block, inplanes[2], inplanes[3], layers[1], stride=1)
        self.layer_3 = self._make_layer(block, inplanes[3], inplanes[4], layers[2], stride=1)

        self.layer_0_0 = self._make_layer(block, inplanes[0], inplanes[1], 1, stride=1)
        self.layer_1_0 = self._make_layer(block, inplanes[1], inplanes[2], layers[0], stride=1)
        self.layer_2_0 = self._make_layer(block, inplanes[2], inplanes[3], layers[1], stride=1)
        self.layer_3_0 = self._make_layer(block, inplanes[3], inplanes[4], layers[2], stride=1)

        # self.skip1 = Fusion1(inplanes[1])
        # self.skip2 = Fusion1(inplanes[2])
        # self.skip3 = Fusion1(inplanes[3])
        self.mul_scales = Fusion1(inplanes[5])

        self.layer_4_0 = self._make_layer(block, inplanes[4] * 2, inplanes[5], layers[3], stride=1)
        self.layer_3_1 = self._make_layer(block, inplanes[5] + 2 * inplanes[4], inplanes[4], layers[2], stride=1)
        self.layer_2_1 = self._make_layer(block, inplanes[4] + 2 * inplanes[3], inplanes[3], layers[2], stride=1)
        self.layer_1_1 = self._make_layer(block, inplanes[3] + 2 * inplanes[2], inplanes[2], layers[2], stride=1)
        self.layer_0_1 = self._make_layer(block, inplanes[2] + 2 * inplanes[1], inplanes[1], 1, stride=1)

        self.fuse_0 = AsymBiChaFuse(inplanes[1])
        self.fuse_1 = AsymBiChaFuse(inplanes[2])
        self.fuse_2 = AsymBiChaFuse(inplanes[3])
        self.fuse_3 = AsymBiChaFuse(inplanes[4])

        self.layer_f_0 = nn.Conv2d(inplanes[1], inplanes[0], kernel_size=1)
        self.layer_f_1 = nn.Conv2d(inplanes[2], inplanes[0], kernel_size=1)
        self.layer_f_2 = nn.Conv2d(inplanes[3], inplanes[0], kernel_size=1)
        self.layer_f_3 = nn.Conv2d(inplanes[4], inplanes[0], kernel_size=1)
        self.layer_f_4 = nn.Conv2d(inplanes[5], inplanes[0], kernel_size=1)

        self.final_layer = nn.Conv2d(inplanes[0] * 5, 1, kernel_size=1)



    def _make_layer(
            self,
            block: BasicBlock,
            inplanes: int,
            planes: int,
            blocks: int,
            stride: int = 1,
    ) -> nn.Sequential:
        downsample = None

        if inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(
            block(
                inplanes, planes, downsample
            )
        )
        for _ in range(1, blocks):
            layers.append(
                block(
                    planes,
                    planes,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, input, input_1):
        x_0 = self.layer_0(input)
        x_1 = self.layer_1(x_0)
        x_2 = self.layer_2(self.pool(x_1))
        x_3 = self.layer_3(self.pool(x_2))

        x_0_0 = self.layer_0_0(input_1)
        x_1_0 = self.layer_1_0(x_0_0)
        x_2_0 = self.layer_2_0(self.pool(x_1_0))
        x_3_0 = self.layer_3_0(self.pool(x_2_0))


        x_4_0 = self.layer_4_0(torch.cat([self.pool(x_3), self.pool(x_3_0)], 1))
        mul = self.mul_scales(x_4_0)

        x_3_1 = self.layer_3_1(torch.cat([x_3, x_3_0, self.up(x_4_0)], 1))
        x_2_1 = self.layer_2_1(torch.cat([x_2, x_2_0, self.up(x_3_1)], 1))
        x_1_1 = self.layer_1_1(torch.cat([x_1, x_1_0, self.up(x_2_1)], 1))
        x_0_1 = self.layer_0_1(torch.cat([x_0, x_0_0, x_1_1], 1))
         
        fuse0 = self.fuse_0(x_0_1, x_0, x_0_0)
        fuse1 = self.fuse_1(x_1_1, x_1, x_1_0)
        fuse2 = self.fuse_2(x_2_1, x_2, x_2_0)
        fuse3 = self.fuse_3(x_3_1, x_3, x_3_0)

        f_0_0 = self.layer_f_0(self.relu(fuse0 + x_0_1))
        f_0_1 = self.layer_f_1(self.relu(fuse1 + x_1_1))
        f_0_2 = self.layer_f_2(self.up(self.relu(fuse2 + x_2_1)))
        f_0_3 = self.layer_f_3(self.up_4(self.relu(fuse3 + x_3_1)))
        f_0_4 = self.layer_f_4(self.up_8(mul))

        final = self.final_layer(torch.cat([f_0_0, f_0_1, f_0_2, f_0_3, f_0_4 ], 1))

        return final
