"""Adapters implementing FLoCoRA-style low-rank updates for convolution and linear layers."""

from typing import Iterable, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "FLoCoRAConv2dAdapter",
    "FLoCoRALinearAdapter",
    "wrap_efficientnet_with_adapters",
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


def _normalize_module_names(modules: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not modules:
        return ()
    normalized: List[str] = []
    seen: Set[str] = set()
    for name in modules:
        if not isinstance(name, str):
            continue
        trimmed = name.strip()
        if not trimmed or trimmed in seen:
            continue
        seen.add(trimmed)
        normalized.append(trimmed)
    return tuple(normalized)


def _parameter_belongs_to_module(parameter_name: str, module_name: str) -> bool:
    if not module_name:
        return False
    if parameter_name == module_name:
        return True
    return parameter_name.startswith(f"{module_name}.")


def _apply_peft_freezing(model: nn.Module, modules_to_save: Tuple[str, ...]) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for module in model.modules():
        if isinstance(module, (FLoCoRAConv2dAdapter, FLoCoRALinearAdapter)):
            if getattr(module, "lora_A", None) is not None:
                module.lora_A.requires_grad = True
            if getattr(module, "lora_B", None) is not None:
                module.lora_B.requires_grad = True

    module_map = dict(model.named_modules())
    for module_name in modules_to_save:
        module = module_map.get(module_name)
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = True


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


def _iter_target_names(lora_config) -> Iterable[str]:
    seen: Set[str] = set()
    for collection_name in ("target_modules", "rank_pattern"):
        collection = getattr(lora_config, collection_name, None)
        if isinstance(collection, dict):
            for name in collection.keys():
                if name not in seen:
                    seen.add(name)
                    yield name
        elif isinstance(collection, (list, tuple, set)):
            for name in collection:
                if isinstance(name, str) and name not in seen:
                    seen.add(name)
                    yield name


def _replace_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    if not module_name:
        raise ValueError("module_name must be a non-empty string")
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent._modules[part]
    key = parts[-1]
    current = parent._modules.get(key)
    if isinstance(current, (FLoCoRAConv2dAdapter, FLoCoRALinearAdapter)):
        return
    parent._modules[key] = new_module


def wrap_efficientnet_with_adapters(model: nn.Module, lora_config) -> nn.Module:
    target_names = tuple(_iter_target_names(lora_config))
    modules_to_save = _normalize_module_names(getattr(lora_config, "modules_to_save", None))

    if target_names:
        alpha = getattr(lora_config, "alpha", 1.0)
        modules = list(model.named_modules())
        target_set = set(target_names)
        for name, module in modules:
            if name not in target_set:
                continue
            if isinstance(module, (FLoCoRAConv2dAdapter, FLoCoRALinearAdapter)):
                continue
            rank = _resolve_rank(lora_config, name)
            if rank <= 0:
                continue
            if isinstance(module, nn.Conv2d):
                adapter = FLoCoRAConv2dAdapter.from_conv(module, rank=rank, alpha=alpha)
            elif isinstance(module, nn.Linear):
                adapter = FLoCoRALinearAdapter.from_linear(module, rank=rank, alpha=alpha)
            else:
                continue
            _replace_module(model, name, adapter)

    if target_names or modules_to_save:
        _apply_peft_freezing(model, modules_to_save)
    setattr(model, "_flocora_target_modules", tuple(dict.fromkeys(target_names)))
    setattr(model, "_flocora_modules_to_save", modules_to_save)
    return model
