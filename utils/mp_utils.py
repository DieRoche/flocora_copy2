import gc
from typing import Any

import torch
from prune import prune
from utils.utils import set_params, get_params,train,inst_model_info
from utils.dataset import (
    get_dataloader,
    dict_tranforms_train,
)
from utils.lora import *
from utils.simple_quant import *


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_device(device_hint):
    """Return a valid torch.device using the provided hint."""
    if isinstance(device_hint, torch.device):
        candidate = device_hint
    else:
        try:
            candidate = torch.device(device_hint)
        except (TypeError, RuntimeError):
            candidate = torch.device("cpu")

    if candidate.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return candidate


def _coerce_rank(rank_value: Any) -> int:
    """Return a non-negative integer rank from potentially nested containers."""

    if isinstance(rank_value, dict):
        ranks = [_coerce_rank(value) for value in rank_value.values()]
        ranks = [rank for rank in ranks if rank > 0]
        if not ranks:
            return 0
        return max(ranks)

    if isinstance(rank_value, (tuple, list, set)):
        ranks = [_coerce_rank(value) for value in rank_value]
        ranks = [rank for rank in ranks if rank > 0]
        if not ranks:
            return 0
        return max(ranks)

    try:
        rank_int = int(rank_value)
    except (TypeError, ValueError):
        return 0

    return rank_int if rank_int > 0 else 0


def _get_module_rank(lora_config, module_name: str) -> int:
    """Return the configured LoRA rank for the provided module name."""

    if lora_config is None:
        return 0

    rank_pattern = getattr(lora_config, "rank_pattern", None)
    if isinstance(rank_pattern, dict) and module_name in rank_pattern:
        rank = _coerce_rank(rank_pattern[module_name])
        if rank > 0:
            return rank

    try:
        fallback_rank = int(getattr(lora_config, "r", 0) or 0)
    except (TypeError, ValueError):
        fallback_rank = 0

    return fallback_rank if fallback_rank > 0 else 0


def _estimate_lora_projection_flops(model: torch.nn.Module, lora_config) -> float:
    """Approximate the FLOPs required to project LoRA adapters to dense weights."""

    if lora_config is None:
        return 0.0

    target_modules = getattr(lora_config, "target_modules", None)
    if not target_modules:
        return 0.0

    modules = dict(model.named_modules())
    total_flops = 0.0

    for module_name in target_modules:
        module = modules.get(module_name)
        if module is None:
            continue

        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim == 0:
            continue

        rank = _get_module_rank(lora_config, module_name)
        if rank <= 0:
            continue

        out_dim = int(weight.shape[0])
        if out_dim <= 0:
            continue

        per_output_features = int(weight[0].numel()) if weight[0].numel() > 0 else 0
        if per_output_features <= 0:
            continue

        total_flops += 2.0 * float(out_dim) * float(per_output_features) * float(rank)

    return float(total_flops)


def _estimate_quantization_flops(model: torch.nn.Module) -> float:
    """Approximate per-round FLOPs spent on fake quantization and dequantization."""

    # Each element is subject to reductions (min/max), affine transformations,
    # rounding, clamping, and a final dequantisation multiply-add. We budget
    # eight floating point operations per trainable element and a small per-
    # channel overhead for scale/zero-point updates.
    PER_ELEMENT_FLOPS = 8.0
    PER_CHANNEL_FLOPS = 6.0

    total_flops = 0.0
    for param in model.parameters():
        if not getattr(param, "requires_grad", False) or param.dim() <= 1:
            continue

        num_elements = float(param.numel())
        if num_elements <= 0.0:
            continue

        total_flops += PER_ELEMENT_FLOPS * num_elements
        total_flops += PER_CHANNEL_FLOPS * float(param.shape[0])

    return float(total_flops)


def mp_fit(info, fl_info,config, parameters, return_dict):

    use_prune = fl_info.prune
    use_prune_srv = fl_info.prune_srv
    device = _resolve_device(fl_info.device if hasattr(fl_info, "device") else "cpu")
    fed_dir = fl_info.fed_dir
    cid = fl_info.cid

    net = inst_model_info(info)
    if fl_info.lora_config is not None :
        net = inject_low_rank(net,fl_info.lora_config)

    if fl_info.apply_quant:
        fakequant_trainable_channel(net,fl_info.quant_bits)

    if parameters is not None:
        if use_prune_srv:
            parameters = prune(parameters,config["prate"])
        set_params(net, parameters,fedbn=info.fedbn)

    net.to(device)

    lr = config["cl_lr"]
    momentum = config["cl_momentum"]
    weight_decay = config["cl_wd"]
    # Load data for this client and get trainloader
    trainloader = get_dataloader(
        fed_dir,
        cid,
        is_train=True,
        batch_size=config["batch_size"],
        workers=fl_info.nworkers,
        transform=dict_tranforms_train[info.dataset_name],
    )

    criterion = torch.nn.CrossEntropyLoss().to(device)

    params = [
        {
            "params": [value],
            "name": key,
            "weight_decay": weight_decay if "bn" not in key else 0.0,
            "param_size": value.size(),
            "nelement": value.nelement(),
        }
        for key, value in net.named_parameters()
    ]

    optimizer = torch.optim.SGD(params, 
                                lr = lr,
                                momentum = momentum,
                                nesterov = momentum > 0.0)

    training_stats = train(
        net,
        trainloader,
        config["epochs"],
        optimizer,
        criterion,
        device,
    )

    epoch_flops = []
    if isinstance(training_stats, dict):
        epoch_flops = training_stats.get("epoch_flops", []) or []

    lora_projection_flops = _estimate_lora_projection_flops(net, getattr(fl_info, "lora_config", None))
    quantization_flops = (
        _estimate_quantization_flops(net) if getattr(fl_info, "apply_quant", False) else 0.0
    )

    compression_flops = float(lora_projection_flops + quantization_flops)
    decompression_flops = float(lora_projection_flops + quantization_flops)

    epoch_flops_total = float(sum(epoch_flops)) if epoch_flops else 0.0
    sum_epoch_including_comp = (
        epoch_flops_total + compression_flops + decompression_flops
    )

    flop_metrics = {
        "flops_by_epoch": epoch_flops_total,
        "flops_compression": compression_flops,
        "flops_decompression": decompression_flops,
        "sum_flops_epoch_includingcompdecomp": sum_epoch_including_comp,
    }
    
    if fl_info.apply_quant: 
        fakequant_trainable_channel(net,fl_info.quant_bits)

    params = get_params(net,fedbn=info.fedbn)

    if use_prune:
        params = prune(params,config["prate"])

    net.to(torch.device("cpu"))
    return_dict["params"] = params
    return_dict["size"] = len(trainloader.dataset)
    return_dict["metrics"] = flop_metrics

    del net, trainloader, optimizer, criterion
    cleanup_memory()
