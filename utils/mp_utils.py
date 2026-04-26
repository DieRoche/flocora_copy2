import gc
import traceback

from typing import Any, Iterable, Optional


import torch
from prune import prune
from utils.utils import (
    set_params,
    get_params,
    train,
    inst_model_info,
    compute_payload_size_bytes,
    estimate_serialization_flops,
    estimate_deserialization_flops,
)
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


def _expected_state_items(model: torch.nn.Module, fedbn: bool):
    """Return ordered `(name, tensor)` pairs expected from serialized parameters."""

    if fedbn:
        return [(name, tensor) for name, tensor in model.state_dict().items() if "bn" not in name]
    return list(model.state_dict().items())


def _validate_parameter_layout(parameters, model: torch.nn.Module, fedbn: bool, model_name: str):
    """Validate incoming parameter list against the instantiated model layout."""

    if parameters is None:
        return

    expected = _expected_state_items(model, fedbn)
    expected_count = len(expected)
    received_count = len(parameters)

    if received_count != expected_count:
        raise ValueError(
            f"Received {received_count} parameter tensors, but model '{model_name}' expects "
            f"{expected_count}. Check that server/client both use the same architecture."
        )

    for idx, (expected_name, expected_tensor) in enumerate(expected):
        received_param = parameters[idx]
        expected_shape = tuple(expected_tensor.shape)
        received_shape = tuple(getattr(received_param, "shape", ()))
        if received_shape != expected_shape:
            raise ValueError(
                f"Parameter shape mismatch at index {idx} ('{expected_name}') for model "
                f"'{model_name}': received {received_shape}, expected {expected_shape}."
            )


def _validate_lora_parameter_layout(parameters, model: torch.nn.Module, model_name: str):
    """Validate incoming LoRA payload against PEFT state layout."""

    if parameters is None:
        return

    expected = get_lora_state_items(model)
    expected_count = len(expected)
    received_count = len(parameters)

    if received_count != expected_count:
        raise ValueError(
            f"Received {received_count} LoRA tensors, but model '{model_name}' expects "
            f"{expected_count}. Check that server/client share the same LoRA config."
        )

    for idx, (expected_name, expected_tensor) in enumerate(expected):
        received_param = parameters[idx]
        expected_shape = tuple(expected_tensor.shape)
        received_shape = tuple(getattr(received_param, "shape", ()))
        if received_shape != expected_shape:
            raise ValueError(
                f"LoRA parameter shape mismatch at index {idx} ('{expected_name}') for model "
                f"'{model_name}': received {received_shape}, expected {expected_shape}."
            )


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


def _iter_module_search_roots(model: torch.nn.Module) -> Iterable[torch.nn.Module]:
    """Yield the module roots that may contain LoRA-wrapped submodules."""

    if model is not None:
        yield model

    base_model = getattr(model, "base_model", None)
    if isinstance(base_model, torch.nn.Module):
        yield base_model
        inner_model = getattr(base_model, "model", None)
        if isinstance(inner_model, torch.nn.Module):
            yield inner_model


def _locate_module(model: torch.nn.Module, module_name: str) -> Optional[torch.nn.Module]:
    """Return the module referenced by ``module_name`` even when wrapped by PEFT."""

    candidate_paths = [module_name]

    if module_name:
        candidate_paths.append(f"base_model.{module_name}")
        candidate_paths.append(f"base_model.model.{module_name}")

    for root in _iter_module_search_roots(model):
        if root is None:
            continue

        for path in candidate_paths:
            try:
                return root.get_submodule(path)
            except (AttributeError, KeyError):
                continue

        modules = dict(root.named_modules())
        if module_name in modules:
            return modules[module_name]

    # Fallback to suffix search to handle duplicated wrappers while
    # preferring longer matches.
    for root in _iter_module_search_roots(model):
        if root is None:
            continue
        for name, module in sorted(
            root.named_modules(), key=lambda item: len(item[0]), reverse=True
        ):
            if name.endswith(module_name):
                return module

    return None


def _extract_lora_ranks(module: torch.nn.Module) -> Iterable[int]:
    """Return the ranks of all active LoRA adapters attached to ``module``."""

    lora_A = getattr(module, "lora_A", None)
    lora_B = getattr(module, "lora_B", None)

    if not isinstance(lora_A, torch.nn.ModuleDict) or not isinstance(
        lora_B, torch.nn.ModuleDict
    ):
        return []

    ranks = []
    for adapter_name, adapter_module in lora_A.items():
        if adapter_name not in lora_B:
            continue

        weight = getattr(adapter_module, "weight", None)
        if weight is None or weight.ndim == 0:
            continue

        rank = int(weight.shape[0]) if weight.shape[0] > 0 else 0
        if rank > 0:
            ranks.append(rank)

    return ranks


def _estimate_lora_projection_flops(model: torch.nn.Module, lora_config) -> float:
    """Approximate the FLOPs required to project LoRA adapters to dense weights."""

    if lora_config is None:
        return 0.0

    target_modules = getattr(lora_config, "target_modules", None)
    if not target_modules:
        return 0.0

    total_flops = 0.0
    rank_pattern = getattr(lora_config, "rank_pattern", None)

    for module_name in target_modules:
        module = _locate_module(model, module_name)
        if module is None:
            continue

        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim == 0:
            continue

        ranks = list(_extract_lora_ranks(module))
        if not ranks:
            if isinstance(rank_pattern, dict) and module_name not in rank_pattern:
                continue

            fallback_rank = _get_module_rank(lora_config, module_name)
            if fallback_rank <= 0:
                continue
            ranks = [fallback_rank]

        out_dim = int(weight.shape[0])
        if out_dim <= 0:
            continue

        per_output_features = int(weight[0].numel()) if weight[0].numel() > 0 else 0
        if per_output_features <= 0:
            continue

        for rank in ranks:
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


def _capture_target_spatial_dims(
    model: torch.nn.Module,
    target_modules: Iterable[str],
    sample_input: torch.Tensor,
) -> dict[str, tuple[int, int]]:
    """Capture output `(H, W)` for target modules using a single forward pass."""

    captured_dims: dict[str, tuple[int, int]] = {}
    hooks = []

    for module_name in target_modules:
        module = _locate_module(model, module_name)
        if module is None:
            continue

        def _hook(_module, _inputs, outputs, name=module_name):
            output_tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            if not isinstance(output_tensor, torch.Tensor):
                return
            if output_tensor.dim() >= 4:
                captured_dims[name] = (int(output_tensor.shape[-2]), int(output_tensor.shape[-1]))
            else:
                captured_dims[name] = (1, 1)

        hooks.append(module.register_forward_hook(_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(sample_input)
    if was_training:
        model.train()

    for hook in hooks:
        hook.remove()

    return captured_dims


def _estimate_lora_training_and_communication(
    model: torch.nn.Module,
    lora_config,
    sample_input: Optional[torch.Tensor],
    num_batches: int,
    num_epochs: int,
) -> tuple[float, float]:
    """Estimate LoRA training FLOPs and communication size per round."""

    if lora_config is None:
        return 0.0, 0.0

    target_modules = getattr(lora_config, "target_modules", None)
    if not target_modules or num_batches <= 0 or num_epochs <= 0:
        return 0.0, 0.0

    spatial_dims = {}
    if sample_input is not None:
        spatial_dims = _capture_target_spatial_dims(model, target_modules, sample_input)

    forward_flops_total = 0.0
    communication_size_total = 0.0

    for module_name in target_modules:
        module = _locate_module(model, module_name)
        if module is None:
            continue

        ranks = list(_extract_lora_ranks(module))
        if not ranks:
            fallback_rank = _get_module_rank(lora_config, module_name)
            if fallback_rank <= 0:
                continue
            ranks = [fallback_rank]

        if isinstance(module, torch.nn.Conv2d):
            cin = int(module.in_channels)
            cout = int(module.out_channels)
            kh, kw = module.kernel_size
            k_sq = int(kh * kw)
            h, w = spatial_dims.get(module_name, (1, 1))
        elif isinstance(module, torch.nn.Linear):
            cin = int(module.in_features)
            cout = int(module.out_features)
            k_sq = 1
            h, w = 1, 1
        else:
            weight = getattr(module, "weight", None)
            if weight is None or weight.ndim < 2:
                continue
            cin = int(weight[0].numel())
            cout = int(weight.shape[0])
            k_sq = 1
            h, w = spatial_dims.get(module_name, (1, 1))

        for rank in ranks:
            r = float(rank)
            forward_flops = (r * float(cin) * float(h) * float(w)) + (
                float(cout) * r * float(k_sq) * float(h) * float(w)
            )
            forward_flops_total += forward_flops
            communication_size_total += (r * float(cin)) + (float(cout) * r * float(k_sq))

    training_flops_total = 3.0 * forward_flops_total * float(num_batches) * float(num_epochs)
    return float(training_flops_total), float(communication_size_total)


def mp_fit(info, fl_info,config, parameters, return_dict):
    try:
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
            if fl_info.lora_config is not None:
                _validate_lora_parameter_layout(
                    parameters,
                    net,
                    model_name=getattr(info, "model", "unknown"),
                )
                if use_prune_srv:
                    parameters = prune(parameters,config["prate"])
                set_lora_params(net, parameters)
            else:
                _validate_parameter_layout(
                    parameters,
                    net,
                    fedbn=info.fedbn,
                    model_name=getattr(info, "model", "unknown"),
                )
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
        sample_input = None
        try:
            sample_batch = next(iter(trainloader))
            if isinstance(sample_batch, (tuple, list)) and len(sample_batch) > 0:
                sample_input = sample_batch[0].to(device)
        except StopIteration:
            sample_input = None

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

        lora_training_flops, lora_communication_size = _estimate_lora_training_and_communication(
            net,
            getattr(fl_info, "lora_config", None),
            sample_input,
            num_batches=len(trainloader),
            num_epochs=int(config["epochs"]),
        )

        lora_projection_flops = _estimate_lora_projection_flops(net, getattr(fl_info, "lora_config", None))
        quantization_flops = (
            _estimate_quantization_flops(net) if getattr(fl_info, "apply_quant", False) else 0.0
        )

        compression_flops = float(lora_projection_flops + quantization_flops)
        decompression_flops = float(lora_projection_flops + quantization_flops)

        epoch_flops_total = float(sum(epoch_flops)) if epoch_flops else 0.0
        if getattr(fl_info, "lora_config", None) is not None and lora_training_flops > 0.0:
            epoch_flops_total = float(lora_training_flops)
        sum_epoch_including_comp = (
            epoch_flops_total + compression_flops + decompression_flops
        )

        flop_metrics = {
            "flops_by_epoch": epoch_flops_total,
            "flops_compression": compression_flops,
            "flops_decompression": decompression_flops,
            "compression_flops_round_clients": compression_flops,
            "decompression_flops_round_clients": decompression_flops,
            "intermediate_communication_processing_flops_round_clients": 0.0,
            "communication_lora_size_round_clients": float(lora_communication_size),
            "sum_flops_epoch_includingcompdecomp": sum_epoch_including_comp,
        }
        
        if fl_info.apply_quant: 
            fakequant_trainable_channel(net,fl_info.quant_bits)

        if fl_info.lora_config is not None:
            params = get_lora_params(net)
        else:
            params = get_params(net,fedbn=info.fedbn)

        if use_prune:
            params = prune(params,config["prate"])

        upload_traffic = compute_payload_size_bytes(params)
        download_traffic = compute_payload_size_bytes(parameters)
        serialization_flops = estimate_serialization_flops(params)
        deserialization_flops = estimate_deserialization_flops(parameters)
        flop_metrics["serialization_flops_round_clients"] = float(serialization_flops)
        flop_metrics["deserialization_flops_round_clients"] = float(deserialization_flops)
        flop_metrics["upload_traffic"] = float(upload_traffic)
        flop_metrics["download_traffic"] = float(download_traffic)
        flop_metrics["overall_traffic"] = float(upload_traffic + download_traffic)

        net.to(torch.device("cpu"))
        return_dict["params"] = params
        return_dict["size"] = len(trainloader.dataset)
        return_dict["metrics"] = flop_metrics

        del net, trainloader, optimizer, criterion
        cleanup_memory()
    except Exception as ex:
        return_dict["error"] = str(ex)
        return_dict["traceback"] = traceback.format_exc()
