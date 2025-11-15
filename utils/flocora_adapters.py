"""Adapters implementing FLoCoRA-style low-rank updates for convolution and linear layers."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "FLoCoRAConv2dAdapter",
    "FLoCoRALinearAdapter",
    "apply_flocora_adapters",
    "verify_equivalent_outputs",
]


def _as_tuple(value: Union[Sequence[int], int]) -> Tuple[int, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, (list, range)):
        return tuple(int(v) for v in value)
    return (int(value), int(value))


def _deterministic_generator(device: torch.device) -> torch.Generator:
    if device.type == "cuda":
        return torch.Generator(device=device)
    return torch.Generator()


def _parameter_belongs_to_module(parameter_name: str, module_name: str) -> bool:
    if not module_name:
        return False
    if parameter_name == module_name:
        return True
    return parameter_name.startswith(f"{module_name}.")


class _BaseAdapter(nn.Module):
    """Shared logic for FLoCoRA adapters."""

    _NON_SERIALIZED_BUFFERS: Tuple[str, ...] = ("base_weight", "base_bias")

    def __init__(self, alpha: float, rank: int) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.rank = int(rank) if rank is not None else 0
        self.scaling = self.alpha / self.rank if self.rank > 0 else 0.0

    def _reset_lora_parameters(self, *parameters: nn.Parameter) -> None:
        if self.rank <= 0:
            return
        generator = _deterministic_generator(parameters[0].device)
        generator.manual_seed(0)
        std = 1e-4
        with torch.no_grad():
            for idx, parameter in enumerate(parameters):
                if idx == 0:
                    parameter.normal_(mean=0.0, std=std, generator=generator)
                else:
                    parameter.normal_(mean=0.0, std=std, generator=generator)

    def _save_to_state_dict(self, destination, prefix, keep_vars):  # type: ignore[override]
        super()._save_to_state_dict(destination, prefix, keep_vars)
        for name in self._NON_SERIALIZED_BUFFERS:
            destination.pop(f"{prefix}{name}", None)

    def _load_from_state_dict(  # type: ignore[override]
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name in self._NON_SERIALIZED_BUFFERS:
            key = f"{prefix}{name}"
            if key in missing_keys:
                missing_keys.remove(key)

    def named_buffers(  # type: ignore[override]
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ):
        for name, buffer in super().named_buffers(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        ):
            local_name = name.split(".")[-1]
            if local_name in self._NON_SERIALIZED_BUFFERS:
                continue
            yield name, buffer


class FLoCoRAConv2dAdapter(_BaseAdapter):
    """Adapter that wraps ``nn.Conv2d`` and applies a low-rank FLoCoRA update."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        padding: Sequence[int] | int,
        dilation: Sequence[int] | int,
        groups: int,
        bias: bool,
        alpha: float,
        rank: int,
        base_weight: torch.Tensor,
        base_bias: Optional[torch.Tensor],
    ) -> None:
        super().__init__(alpha=alpha, rank=rank)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _as_tuple(kernel_size)
        self.stride = _as_tuple(stride)
        self.padding = _as_tuple(padding)
        self.dilation = _as_tuple(dilation)
        self.groups = int(groups)
        self.has_bias = bool(bias)

        dtype = base_weight.dtype
        device = base_weight.device
        self.register_buffer("base_weight", base_weight.detach().clone())
        bias_tensor = base_bias.detach().clone() if base_bias is not None else torch.zeros(0, dtype=dtype, device=device)
        self.register_buffer("base_bias", bias_tensor)

        in_features = (self.in_channels // self.groups) * self.kernel_size[0] * self.kernel_size[1]
        if self.rank > 0 and in_features > 0 and self.out_channels > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.rank, in_features, dtype=dtype, device=device))
            self.lora_B = nn.Parameter(torch.zeros(self.out_channels, self.rank, dtype=dtype, device=device))
            self._reset_lora_parameters(self.lora_A, self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, *, rank: int, alpha: float) -> "FLoCoRAConv2dAdapter":
        base_weight = conv.weight.detach().clone()
        base_bias = conv.bias.detach().clone() if conv.bias is not None else None
        adapter = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            alpha=alpha,
            rank=rank,
            base_weight=base_weight.to(conv.weight.device, conv.weight.dtype),
            base_bias=base_bias.to(conv.weight.device, conv.weight.dtype) if base_bias is not None else None,
        )
        adapter.to(conv.weight.device)
        adapter.train(conv.training)
        return adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.base_weight
        if self.rank > 0 and self.lora_A is not None and self.lora_B is not None:
            in_features = (self.in_channels // self.groups) * self.kernel_size[0] * self.kernel_size[1]
            update = torch.matmul(self.lora_B, self.lora_A)
            update = update.view(self.out_channels, in_features)
            update = update.view(self.out_channels, self.in_channels // self.groups, *self.kernel_size)
            weight = weight + self.scaling * update
        bias = self.base_bias if self.has_bias else None
        return F.conv2d(
            x,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class FLoCoRALinearAdapter(_BaseAdapter):
    """Adapter that wraps ``nn.Linear`` with a low-rank update."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        bias: bool,
        alpha: float,
        rank: int,
        base_weight: torch.Tensor,
        base_bias: Optional[torch.Tensor],
    ) -> None:
        super().__init__(alpha=alpha, rank=rank)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.has_bias = bool(bias)

        dtype = base_weight.dtype
        device = base_weight.device
        self.register_buffer("base_weight", base_weight.detach().clone())
        bias_tensor = base_bias.detach().clone() if base_bias is not None else torch.zeros(0, dtype=dtype, device=device)
        self.register_buffer("base_bias", bias_tensor)

        if self.rank > 0 and self.in_features > 0 and self.out_features > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.rank, self.in_features, dtype=dtype, device=device))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank, dtype=dtype, device=device))
            self._reset_lora_parameters(self.lora_A, self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, rank: int, alpha: float) -> "FLoCoRALinearAdapter":
        base_weight = linear.weight.detach().clone()
        base_bias = linear.bias.detach().clone() if linear.bias is not None else None
        adapter = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            alpha=alpha,
            rank=rank,
            base_weight=base_weight.to(linear.weight.device, linear.weight.dtype),
            base_bias=base_bias.to(linear.weight.device, linear.weight.dtype) if base_bias is not None else None,
        )
        adapter.to(linear.weight.device)
        adapter.train(linear.training)
        return adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.base_weight
        if self.rank > 0 and self.lora_A is not None and self.lora_B is not None:
            update = torch.matmul(self.lora_B, self.lora_A)
            weight = weight + self.scaling * update
        bias = self.base_bias if self.has_bias else None
        return F.linear(x, weight, bias if self.has_bias else None)


def _coerce_rank(value: Optional[int]) -> int:
    if value is None:
        return 0
    if isinstance(value, dict):
        ranks = [_coerce_rank(v) for v in value.values()]
        return max(ranks) if ranks else 0
    if isinstance(value, (tuple, list, set)):
        ranks = [_coerce_rank(v) for v in value]
        return max(ranks) if ranks else 0
    try:
        rank = int(value)
    except (TypeError, ValueError):
        return 0
    return rank if rank > 0 else 0


def _resolve_rank(lora_config, module_name: str) -> int:
    rank_pattern = getattr(lora_config, "rank_pattern", None)
    if isinstance(rank_pattern, dict) and module_name in rank_pattern:
        rank = _coerce_rank(rank_pattern[module_name])
        if rank > 0:
            return rank
    fallback = getattr(lora_config, "r", 0)
    try:
        fallback = int(fallback)
    except (TypeError, ValueError):
        fallback = 0
    return fallback if fallback > 0 else 0


def _iter_named_modules_with_parents(
    module: nn.Module, prefix: str = ""
) -> Iterable[tuple[nn.Module, str, nn.Module, str]]:
    """Yield ``(parent, name, child, qualified_name)`` for all children recursively."""

    for name, child in module.named_children():
        qualified_name = f"{prefix}.{name}" if prefix else name
        yield module, name, child, qualified_name
        yield from _iter_named_modules_with_parents(child, qualified_name)


def _should_consider_module(
    module_name: str,
    module: nn.Module,
    *,
    target_modules: Optional[set[str]],
    modules_to_save: set[str],
) -> bool:
    if module_name in modules_to_save:
        return False
    if any(module_name.startswith(f"{saved}.") for saved in modules_to_save):
        return False
    if isinstance(module, (FLoCoRAConv2dAdapter, FLoCoRALinearAdapter)):
        return False
    if target_modules is not None and module_name not in target_modules:
        return False
    return True


def apply_flocora_adapters(
    model: nn.Module,
    lora_config,
    *,
    wrap_linear: bool = False,
) -> nn.Module:
    """Replace Conv2d/Linear layers with FLoCoRA adapters according to ``lora_config``.

    Args:
        model: The module whose submodules should be wrapped.
        lora_config: Configuration object providing ``alpha``, ``r``, ``target_modules``
            and optionally ``rank_pattern`` to control the injected ranks.
        wrap_linear: Whether ``nn.Linear`` layers should be considered for wrapping when
            they are not explicitly enumerated in ``target_modules``.

    Returns:
        The model with eligible layers swapped for adapter equivalents. The function
        mutates ``model`` in-place and also stores ``modules_to_save`` on the root module
        for downstream utilities.
    """

    if model is None:
        raise ValueError("model cannot be None when applying FLoCoRA adapters")

    alpha = float(getattr(lora_config, "alpha", 1.0) or 1.0)
    target_modules = getattr(lora_config, "target_modules", None)
    if target_modules:
        target_set: Optional[set[str]] = set(target_modules)
    else:
        rank_pattern = getattr(lora_config, "rank_pattern", None)
        if isinstance(rank_pattern, dict) and rank_pattern:
            target_set = set(rank_pattern.keys())
        else:
            target_set = None

    modules_to_save = set(getattr(lora_config, "modules_to_save", []) or [])
    replaced_modules: list[str] = []

    # Snapshot traversal order before mutating the module tree.
    traversal = list(_iter_named_modules_with_parents(model))

    for parent, child_name, child_module, qualified_name in traversal:
        if not _should_consider_module(
            qualified_name,
            child_module,
            target_modules=target_set,
            modules_to_save=modules_to_save,
        ):
            continue

        rank = _resolve_rank(lora_config, qualified_name)
        if rank <= 0:
            continue

        if isinstance(child_module, nn.Conv2d):
            adapter = FLoCoRAConv2dAdapter.from_conv(
                child_module,
                rank=rank,
                alpha=alpha,
            )
            setattr(parent, child_name, adapter)
            replaced_modules.append(qualified_name)
            continue

        allow_linear = wrap_linear or (target_set is not None and qualified_name in target_set)
        if allow_linear and isinstance(child_module, nn.Linear):
            adapter = FLoCoRALinearAdapter.from_linear(
                child_module,
                rank=rank,
                alpha=alpha,
            )
            setattr(parent, child_name, adapter)
            replaced_modules.append(qualified_name)

    setattr(model, "_flocora_modules_to_save", tuple(sorted(modules_to_save)))
    setattr(model, "_flocora_replaced_modules", tuple(sorted(replaced_modules)))
    return model


def verify_equivalent_outputs(
    original: nn.Module,
    wrapped: nn.Module,
    sample_input: torch.Tensor,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> bool:
    """Return ``True`` if the wrapped layer matches the original for ``sample_input``.

    The function runs both modules in evaluation mode without gradient tracking. It is
    intended for lightweight smoke tests verifying the replacement pipeline.
    """

    if original is None or wrapped is None:
        raise ValueError("original and wrapped modules must be provided")

    original_mode = original.training
    wrapped_mode = wrapped.training

    try:
        original.eval()
        wrapped.eval()
        with torch.no_grad():
            reference = original(sample_input)
            candidate = wrapped(sample_input)
        if isinstance(reference, torch.Tensor) and isinstance(candidate, torch.Tensor):
            return torch.allclose(candidate, reference, rtol=rtol, atol=atol)
        return False
    finally:
        original.train(original_mode)
        wrapped.train(wrapped_mode)

