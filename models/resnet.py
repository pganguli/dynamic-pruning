"""
ResNet variants for CIFAR-10 image classification.

BasicBlock: standard 2-conv residual block with optional downsample shortcut.
ResNet: parametric ResNet builder (ResNet-10, ResNet-18, etc.) where the
  depth and channel widths are controlled by block count per stage.
"""

import math
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class CifarResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(CifarResNet, self).__init__()
        self.in_planes = 16

        out_channels = 16
        self.conv1 = nn.Conv2d(3, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        layers = [self._make_layer(block, out_channels, num_blocks[0], stride=1)]
        if len(num_blocks) >= 2:
            out_channels = 32
            layers.append(
                self._make_layer(block, out_channels, num_blocks[1], stride=2)
            )
        if len(num_blocks) >= 3:
            out_channels = 64
            layers.append(
                self._make_layer(block, out_channels, num_blocks[2], stride=2)
            )
        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(out_channels * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def cifar_resnet10(num_classes):
    return CifarResNet(BasicBlock, [2, 2], num_classes)


def cifar_resnet20(num_classes):
    return CifarResNet(BasicBlock, [3, 3, 3], num_classes)


def cifar_resnet56(num_classes):
    return CifarResNet(BasicBlock, [9, 9, 9], num_classes)
