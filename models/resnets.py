from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

def Conv3x3(in_planes, out_planes, stride=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None,batchn=True):
        super().__init__()
        self.conv1 = Conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes) if batchn else nn.GroupNorm(2,planes)
        self.conv2 = Conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes) if batchn else nn.GroupNorm(2,planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = torch.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(self, inplanes, planes, stride, downsample: Optional[nn.Module] = None,batchn = True):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes) if batchn else nn.GroupNorm(2,planes)
        self.conv2 = Conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes) if batchn else nn.GroupNorm(2,planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion) if batchn else nn.GroupNorm(2,planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# NOTE: The original ResNet implementation below is kept for the
# dynamically-scaled CIFAR-style models used across most of the codebase.
# A dedicated ResNet18 implementation that mirrors the user's version but
# stays compatible with the rest of the project is provided later in this
# file (see ``CifarResNet18``).

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, large_input, width, zero_init_residual=False,batchn=True):
        super().__init__()
        self.inplanes = width

        if batchn:
            norm_layer = nn.BatchNorm2d(self.inplanes)
        else:
            norm_layer = nn.GroupNorm(2,self.inplanes)
        if large_input:

            self.embed = nn.Sequential(
                nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                norm_layer,
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.embed = nn.Sequential(
                nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer,
                nn.ReLU(inplace=True)
            )

        layers_list = []
        for depth, stride, multiplier in layers:
            layers_list.append(self._make_layer(block, width * multiplier, depth, stride=stride,batchn=batchn))
        self.layers = nn.Sequential(*layers_list)

        self.fc = nn.Linear(self.inplanes, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, blocks, stride = 1,batchn=True):
        if stride != 1 or self.inplanes != planes * block.expansion:
            norm_layer = nn.BatchNorm2d(planes * block.expansion) if batchn else nn.GroupNorm(2,planes * block.expansion)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer,
            )
        else:
            downsample = None

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample,batchn
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride = 1,
                    batchn=batchn
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.embed(x)

        x = self.layers(x)

        x = x.mean(-1).mean(-1)
        features = torch.flatten(x, 1)
        x = self.fc(features)

        return x,features

def resnet8(feature_maps, input_shape, num_classes,batchn=False):
    large_input = False
    width=feature_maps

    return ResNet(BasicBlock, [(1, 1, 1), (1, 2, 2), (1, 2, 4)], num_classes, large_input, width,batchn)

class CifarBasicBlock(nn.Module):
    """Basic block mirroring the standard ResNet18 layout.

    The block supports both BatchNorm and GroupNorm depending on the
    ``batchn`` flag propagated by ``CifarResNet18``.
    """

    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int, norm_layer: Callable[[int], nn.Module]):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = norm_layer(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class CifarResNet18(nn.Module):
    """ResNet18 variant compatible with the project's expectations.

    The implementation mirrors the user's reference model while exposing the
    same signature used across ``model_selection`` (feature maps scaling,
    optional BatchNorm/GroupNorm, tuple output ``(logits, features)`` and a
    ``features_dim`` attribute).
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        num_classes: int,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.use_batchnorm = use_batchnorm

        self.in_channels = base_channels
        self.conv1 = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = self._make_norm(base_channels)
        self.relu = nn.ReLU(inplace=True)
        # CIFAR inputs are only 32x32, so the ImageNet-style max-pool right
        # after the stem discards too much spatial information. Keep the
        # canonical ResNet18 stage layout, but skip this early downsampling.
        self.maxpool = nn.Identity()

        self.layer1 = self._make_layer(base_channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(base_channels * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(base_channels * 8, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.features_dim = base_channels * 8
        self.fc = nn.Linear(self.features_dim, num_classes)

    def _make_norm(self, num_features: int) -> nn.Module:
        if self.use_batchnorm:
            return nn.BatchNorm2d(num_features)
        return nn.GroupNorm(2, num_features)

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        layers.append(CifarBasicBlock(self.in_channels, out_channels, stride, self._make_norm))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(CifarBasicBlock(self.in_channels, out_channels, 1, self._make_norm))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        features = torch.flatten(out, 1)
        logits = self.fc(features)
        return logits, features


def _resolve_input_channels(input_shape: Any) -> int:
    """Utility to robustly infer the number of input channels."""

    if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0:
        return int(input_shape[0])
    if isinstance(input_shape, torch.Size) and len(input_shape) > 0:
        return int(input_shape[0])
    if isinstance(input_shape, int):
        return input_shape
    return 3


def resnet18(feature_maps, input_shape, num_classes, batchn=False):
    in_channels = _resolve_input_channels(input_shape)
    # Keep ResNet18 stage depth (2,2,2,2) and enforce a minimum base width of
    # 64 channels for CIFAR-100 use, while still allowing larger custom widths.
    base_channels = max(64, int(feature_maps))
    return CifarResNet18(in_channels, base_channels, num_classes, use_batchnorm=batchn)

def resnet34(feature_maps, input_shape, num_classes,batchn=False):
    large_input = False
    width=feature_maps

    return ResNet(BasicBlock, [(3, 1, 1), (4, 2, 2), (6, 2, 4), (3, 2, 8)], num_classes, large_input, width,batchn)

def resnet50(feature_maps, input_shape, num_classes,batchn=False):
    large_input = False
    width=feature_maps

    return ResNet(Bottleneck, [(3, 1, 1), (4, 2, 2), (6, 2, 4), (3, 2, 8)], num_classes, large_input, width,batchn)

def resnet101(feature_maps, input_shape, num_classes,batchn=False):
    large_input = False
    width=feature_maps

    return ResNet(Bottleneck, [(3, 1, 1), (4, 2, 2), (23, 2, 4), (3, 2, 8)], num_classes, large_input, width,batchn)

def resnet152(feature_maps, input_shape, num_classes,batchn=False):
    large_input = False
    width=feature_maps

    return ResNet(Bottleneck, [(3, 1, 1), (8, 2, 2), (36, 2, 4), (3, 2, 8)], num_classes, large_input, width,batchn)

def resnet20(feature_maps, input_shape, num_classes,batchn=False):
    large_input = False
    width=feature_maps

    return ResNet(BasicBlock, [(3, 1, 1), (3, 2, 2), (3, 2, 4)], num_classes, large_input, width,batchn)

def resnet32(feature_maps, input_shape, num_classes,batchn=False):
    large_input = False
    width=feature_maps

    return ResNet(BasicBlock, [(5, 1, 1), (5, 2, 2), (5, 2, 4)], num_classes, large_input, width,batchn)

def resnet44(feature_maps, input_shape, num_classes,batchn=False):
    large_input = False
    width=feature_maps

    return ResNet(BasicBlock, [(7, 1, 1), (7, 2, 2), (7, 2, 4)], num_classes, large_input, width,batchn)

def resnet56(feature_maps, input_shape, num_classes,batchn=False):
    large_input = False
    width=feature_maps

    return ResNet(BasicBlock, [(9, 1, 1), (9, 2, 2), (9, 2, 4)], num_classes, large_input, width,batchn)

def resnet110(feature_maps, input_shape, num_classes,batchn=False):
    large_input = False
    width=feature_maps

    return ResNet(BasicBlock, [(18, 1, 1), (18, 2, 2), (18, 2, 4)], num_classes, large_input, width,batchn)

def resnet1202(feature_maps, input_shape, num_classes,batchn=False):
    large_input = False
    width=feature_maps

    return ResNet(BasicBlock, [(200, 1, 1), (200, 2, 2), (200, 2, 4)], num_classes, large_input, width,batchn)

def resnettest(feature_maps, input_shape, num_classes,batchn=False):
    large_input = False
    width=feature_maps

    return ResNet(Bottleneck, [(3, 1, 1), (4, 2, 1), (6, 2, 2), (6, 2, 4), (4, 2, 8), (3, 2, 16)], num_classes, False, width,batchn)
