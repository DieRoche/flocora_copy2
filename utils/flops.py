"""Utility helpers to estimate floating point operations for training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Tuple

import torch


def _extract_tensor(data: Any) -> torch.Tensor | None:
    """Return the first tensor contained in ``data`` or ``None``.

    Forward hooks receive tuples for both inputs and outputs. This helper
    navigates these containers and returns the first :class:`torch.Tensor`
    instance so that shape information can be used to derive FLOPs.
    """

    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, (tuple, list)):
        for item in data:
            tensor = _extract_tensor(item)
            if tensor is not None:
                return tensor
    return None


@dataclass
class _FlopState:
    """Internal accumulator storing FLOP statistics."""

    batch: float = 0.0
    epoch: float = 0.0
    total: float = 0.0

    def reset_batch(self) -> None:
        self.batch = 0.0

    def reset_epoch(self) -> None:
        self.epoch = 0.0

    def add(self, flops: float) -> None:
        self.batch += flops
        self.epoch += flops
        self.total += flops


class FlopMeter:
    """Attach forward hooks to track floating point operations.

    The meter approximates the number of floating point operations executed
    during a forward pass by inspecting well-known modules such as convolution,
    linear, normalisation, activation and pooling layers. The resulting
    estimates are sufficient to expose the relative compute requirements per
    epoch without introducing heavy profiling overhead.
    """

    def __init__(self, model: torch.nn.Module):
        self._model = model
        self._state = _FlopState()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    # ------------------------------------------------------------------
    # Hooks registration
    # ------------------------------------------------------------------
    def _register_hooks(self) -> None:
        for module in self._model.modules():
            hook = None
            if isinstance(module, torch.nn.Conv2d):
                hook = module.register_forward_hook(self._conv2d_hook)
            elif isinstance(module, torch.nn.Linear):
                hook = module.register_forward_hook(self._linear_hook)
            elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                hook = module.register_forward_hook(self._batchnorm_hook)
            elif isinstance(
                module,
                (
                    torch.nn.ReLU,
                    torch.nn.ReLU6,
                    torch.nn.SiLU,
                    torch.nn.GELU,
                    torch.nn.LeakyReLU,
                    torch.nn.ELU,
                ),
            ):
                hook = module.register_forward_hook(self._activation_hook)
            elif isinstance(
                module,
                (
                    torch.nn.AvgPool2d,
                    torch.nn.MaxPool2d,
                    torch.nn.AdaptiveAvgPool2d,
                ),
            ):
                hook = module.register_forward_hook(self._pooling_hook)

            if hook is not None:
                self._handles.append(hook)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start_epoch(self) -> None:
        self._state.reset_epoch()

    def start_batch(self) -> None:
        self._state.reset_batch()

    def finish_epoch(self) -> float:
        return float(self._state.epoch)

    def finish_batch(self) -> float:
        return float(self._state.batch)

    @property
    def total_flops(self) -> float:
        return float(self._state.total)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # ------------------------------------------------------------------
    # Hook helpers
    # ------------------------------------------------------------------
    def _add_flops(self, flops: float) -> None:
        if not math.isfinite(flops):
            return
        self._state.add(float(flops))

    def _conv2d_hook(
        self,
        module: torch.nn.Conv2d,
        inputs: Tuple[torch.Tensor, ...],
        outputs,
    ) -> None:
        input_tensor = _extract_tensor(inputs)
        output_tensor = _extract_tensor(outputs)
        if input_tensor is None or output_tensor is None:
            return

        batch_size = input_tensor.shape[0]
        out_channels = module.out_channels
        kernel_height, kernel_width = module.kernel_size
        in_channels = module.in_channels
        groups = module.groups

        if output_tensor.dim() < 4:
            return

        out_height = output_tensor.shape[2]
        out_width = output_tensor.shape[3]

        filters_per_channel = out_channels // groups
        conv_per_position_flops = (
            kernel_height
            * kernel_width
            * in_channels
            * filters_per_channel
        )
        active_elements = batch_size * out_height * out_width
        total_flops = conv_per_position_flops * active_elements * 2.0

        if module.bias is not None:
            total_flops += batch_size * out_height * out_width * out_channels

        self._add_flops(total_flops)

    def _linear_hook(
        self,
        module: torch.nn.Linear,
        inputs: Tuple[torch.Tensor, ...],
        outputs,
    ) -> None:
        input_tensor = _extract_tensor(inputs)
        if input_tensor is None:
            return

        batch_size = input_tensor.shape[0]
        in_features = module.in_features
        out_features = module.out_features

        total_flops = batch_size * in_features * out_features * 2.0
        if module.bias is not None:
            total_flops += batch_size * out_features

        self._add_flops(total_flops)

    def _batchnorm_hook(
        self,
        module: torch.nn.modules.batchnorm._BatchNorm,
        inputs: Tuple[torch.Tensor, ...],
        outputs,
    ) -> None:
        output_tensor = _extract_tensor(outputs)
        if output_tensor is None:
            return

        # Two operations per element: normalisation and affine transform.
        self._add_flops(output_tensor.numel() * 2.0)

    def _activation_hook(self, module, inputs, outputs) -> None:  # noqa: D401
        del module, inputs  # Unused.
        output_tensor = _extract_tensor(outputs)
        if output_tensor is None:
            return
        # One comparison/operation per element.
        self._add_flops(float(output_tensor.numel()))

    def _pooling_hook(self, module, inputs, outputs) -> None:
        input_tensor = _extract_tensor(inputs)
        output_tensor = _extract_tensor(outputs)
        if input_tensor is None or output_tensor is None:
            return

        if isinstance(module, torch.nn.AdaptiveAvgPool2d):
            kernel_mul = input_tensor.numel() / max(output_tensor.numel(), 1)
        else:
            kernel_size = getattr(module, "kernel_size", 1)
            if isinstance(kernel_size, tuple):
                kernel_mul = float(kernel_size[0] * kernel_size[1])
            else:
                kernel_mul = float(kernel_size * kernel_size)

        total_flops = float(output_tensor.numel()) * float(kernel_mul)
        self._add_flops(total_flops)


__all__ = ["FlopMeter"]

