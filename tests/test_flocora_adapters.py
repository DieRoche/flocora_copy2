import copy
from types import SimpleNamespace

import pytest

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - environments without torch
    pytest.skip("PyTorch is required for FLoCoRA adapter tests", allow_module_level=True)

from utils.flocora_adapters import (
    FLoCoRAConv2dAdapter,
    FLoCoRALinearAdapter,
    apply_flocora_adapters,
    verify_equivalent_outputs,
)


def test_conv_adapter_matches_original_output():
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 5, kernel_size=3, stride=2, padding=1, bias=True)
    adapter = FLoCoRAConv2dAdapter.from_conv(conv, rank=2, alpha=8)

    sample = torch.randn(2, 3, 16, 16)
    assert torch.allclose(conv(sample), adapter(sample))


def test_apply_flocora_adapters_replaces_nested_modules():
    torch.manual_seed(1)

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 6, kernel_size=3, padding=1, bias=False),
                nn.ReLU(),
            )
            self.depthwise = nn.Conv2d(6, 6, kernel_size=3, padding=1, groups=6, bias=False)
            self.pointwise = nn.Conv2d(6, 8, kernel_size=1)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(8, 4),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.stem(x)
            x = self.depthwise(x)
            x = self.pointwise(x)
            x = self.head(x)
            return x

    base = ToyModel()
    wrapped = copy.deepcopy(base)

    config = SimpleNamespace(
        alpha=4,
        r=2,
        target_modules=[],
        modules_to_save=[],
        rank_pattern=None,
        lora_type="flocora",
        wrap_linear=True,
    )

    apply_flocora_adapters(wrapped, config, wrap_linear=True)

    assert isinstance(wrapped.stem[0], FLoCoRAConv2dAdapter)
    assert isinstance(wrapped.depthwise, FLoCoRAConv2dAdapter)
    assert isinstance(wrapped.pointwise, FLoCoRAConv2dAdapter)
    assert isinstance(wrapped.head[2], FLoCoRALinearAdapter)

    sample = torch.randn(1, 3, 16, 16)
    assert verify_equivalent_outputs(base, wrapped, sample)

    replaced = getattr(wrapped, "_flocora_replaced_modules", ())
    assert "stem.0" in replaced
    assert "head.2" in replaced

