"""CIFAR-oriented EfficientNet backbones.

This module exposes an EfficientNet-B0 implementation that matches the project
contract for backbone models:
- constructor signature ``(feature_maps, input_shape, num_classes, batchn=False)``
- ``forward`` returns ``(logits, features)``
- the module defines ``features_dim`` for downstream projection heads

The implementation keeps the original EfficientNet structure but adapts the stem
stride and pooling so 32x32 inputs are handled without losing resolution too
quickly. Norm layers respect the ``batchn`` flag, switching to GroupNorm when
BatchNorm should be avoided (e.g., for FedBN experiments).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["efficientnet_b0"]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _make_norm(num_channels: int, batchn: bool) -> nn.Module:
    if batchn:
        return nn.BatchNorm2d(num_channels)
    if num_channels % 2 != 0:
        raise ValueError(
            "GroupNorm with 2 groups requires an even channel count; "
            "consider adjusting feature_maps or enabling batch normalization."
        )
    return nn.GroupNorm(2, num_channels)


class SqueezeExcite(nn.Module):
    def __init__(self, in_ch: int, se_ratio: float = 0.25) -> None:
        super().__init__()
        reduced = max(1, int(in_ch * se_ratio))
        self.fc1 = nn.Conv2d(in_ch, reduced, kernel_size=1)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, in_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        s = F.adaptive_avg_pool2d(x, 1)
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        return x * torch.sigmoid(s)


@dataclass(frozen=True)
class MBConvConfig:
    expand_ratio: float
    out_channels: int
    num_blocks: int
    stride: int
    kernel_size: int


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck with Squeeze-Excite."""

    @staticmethod
    def _make_depthwise_conv(channels: int, kernel_size: int, stride: int) -> nn.Conv2d:
        """Return a channel-preserving convolution with standard grouping."""

        if channels <= 0:
            raise ValueError("convolution requires a positive channel count")

        groups = 1
        return nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )

    def __init__(
        self,
        in_ch: int,
        cfg: MBConvConfig,
        *,
        batchn: bool,
        se_ratio: float,
        drop_rate: float,
    ) -> None:
        super().__init__()
        self.use_residual = cfg.stride == 1 and in_ch == cfg.out_channels
        mid_ch = int(in_ch * cfg.expand_ratio)

        layers: Iterable[nn.Module] = []
        if cfg.expand_ratio != 1:
            layers = [
                nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
                _make_norm(mid_ch, batchn),
                nn.SiLU(inplace=True),
            ]
        else:
            mid_ch = in_ch
            layers = []

        layers = list(layers)
        depthwise_conv = self._make_depthwise_conv(mid_ch, cfg.kernel_size, cfg.stride)
        layers += [
            depthwise_conv,
            _make_norm(mid_ch, batchn),
            nn.SiLU(inplace=True),
        ]
        self.pre_se = nn.Sequential(*layers)

        self.se = SqueezeExcite(mid_ch, se_ratio=se_ratio)

        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, cfg.out_channels, kernel_size=1, bias=False),
            _make_norm(cfg.out_channels, batchn),
        )

        self.drop_rate = drop_rate
        self.dropout = (
            nn.Dropout(p=drop_rate, inplace=True) if drop_rate > 0.0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pre_se(x)
        out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            out = self.dropout(out)
            out = out + x
        return out


class EfficientNetB0CIFAR(nn.Module):
    """EfficientNet-B0 backbone adapted for 32x32 inputs."""

    def __init__(
        self,
        feature_maps: int,
        input_shape,
        num_classes: int,
        *,
        batchn: bool = False,
        se_ratio: float = 0.25,
        drop_rate: float = 0.2,
    ) -> None:
        super().__init__()
        width_mult = max(feature_maps, 1) / 16.0  # 16 -> baseline 1.0

        if not input_shape or len(input_shape) < 1:
            raise ValueError("input_shape must provide at least the channel dimension")

        in_channels = int(input_shape[0])
        if in_channels <= 0:
            raise ValueError("input_shape must describe a positive number of channels")

        if int(num_classes) <= 0:
            raise ValueError("num_classes must be a positive integer")

        def scale_channels(channels: int) -> int:
            return max(1, int(channels * width_mult))

        self.stem_out = scale_channels(32)
        stem_conv = nn.Conv2d(
            in_channels, self.stem_out, kernel_size=3, stride=1, padding=1, bias=False
        )
        stem_norm = _make_norm(self.stem_out, batchn)
        stem_act = nn.SiLU(inplace=True)
        self.stem = nn.Sequential(
            OrderedDict(
                (
                    ("conv", stem_conv),
                    ("norm", stem_norm),
                    ("act", stem_act),
                )
            )
        )

        base_cfgs = [
            MBConvConfig(1, 16, 1, 1, 3),
            MBConvConfig(6, 24, 2, 2, 3),
            MBConvConfig(6, 40, 2, 2, 5),
            MBConvConfig(6, 80, 3, 2, 3),
            MBConvConfig(6, 112, 3, 1, 5),
            MBConvConfig(6, 192, 4, 2, 5),
            MBConvConfig(6, 320, 1, 1, 3),
        ]

        block_modules: List[Tuple[str, nn.Module]] = []
        stage_in_channels = self.stem_out
        total_blocks = sum(cfg.num_blocks for cfg in base_cfgs)
        drop_increment = drop_rate / total_blocks if total_blocks > 0 else 0.0
        block_id = 0
        for cfg in base_cfgs:
            scaled_out = scale_channels(cfg.out_channels)
            for i in range(cfg.num_blocks):
                stride = cfg.stride if i == 0 else 1
                block_cfg = MBConvConfig(
                    cfg.expand_ratio,
                    scaled_out,
                    1,
                    stride,
                    cfg.kernel_size,
                )
                drop_p = drop_increment * block_id
                block_id += 1
                block = MBConv(
                    stage_in_channels,
                    block_cfg,
                    batchn=batchn,
                    se_ratio=se_ratio,
                    drop_rate=drop_p,
                )
                block_modules.append((f"mbconv{len(block_modules)}", block))
                stage_in_channels = scaled_out

        self.blocks = nn.Sequential(OrderedDict(block_modules))

        self.head_channels = scale_channels(1280)
        head_conv = nn.Conv2d(
            stage_in_channels, self.head_channels, kernel_size=1, bias=False
        )
        head_norm = _make_norm(self.head_channels, batchn)
        head_act = nn.SiLU(inplace=True)
        self.head = nn.Sequential(
            OrderedDict(
                (
                    ("conv", head_conv),
                    ("norm", head_norm),
                    ("act", head_act),
                )
            )
        )

        self.features = nn.Sequential(
            OrderedDict(
                (
                    ("stem", self.stem),
                    ("blocks", self.blocks),
                    ("head", self.head),
                )
            )
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.head_channels, num_classes)

        self.features_dim = self.head_channels

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.classifier(features)
        return logits, features

    def iter_backbone_convs(self) -> Iterator[Tuple[str, nn.Conv2d]]:
        """Yield backbone convolution layers with their qualified names."""

        for name, module in self.features.named_modules():
            if isinstance(module, nn.Conv2d):
                yield name, module

    def describe_lora_targets(self) -> dict:
        """Return metadata describing layers relevant for LoRA injections."""

        conv_layers: List[dict] = []
        pointwise_layers: List[dict] = []
        for name, module in self.iter_backbone_convs():
            metadata = {
                "name": name,
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "kernel_size": tuple(module.kernel_size),
                "stride": tuple(module.stride),
                "padding": tuple(module.padding),
                "groups": module.groups,
                "apply_lora": True,
            }
            conv_layers.append(metadata)
            if module.kernel_size == (1, 1) and module.groups == 1:
                pointwise_layers.append(metadata.copy())

        classifier_meta = {
            "name": "classifier",
            "in_features": self.classifier.in_features,
            "out_features": self.classifier.out_features,
            "apply_lora": False,
        }

        return {
            "conv_layers": conv_layers,
            "pointwise_conv_layers": pointwise_layers,
            "classifier": classifier_meta,
        }


def efficientnet_b0(feature_maps, input_shape, num_classes, batchn=False):
    return EfficientNetB0CIFAR(
        feature_maps,
        input_shape,
        num_classes,
        batchn=batchn,
    )


if __name__ == "__main__":
    model = efficientnet_b0(16, (3, 32, 32), 100, batchn=True)
    dummy = torch.randn(1, 3, 32, 32)
    logits, feats = model(dummy)
    print(logits.shape, feats.shape)
    summary = model.describe_lora_targets()
    print(f"Total conv layers: {len(summary['conv_layers'])}")
    print(f"Pointwise conv layers: {len(summary['pointwise_conv_layers'])}")
    print(f"Classifier LoRA applied: {summary['classifier']['apply_lora']}")
