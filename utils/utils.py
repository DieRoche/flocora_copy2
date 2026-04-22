import torch
import numpy as np
from collections import OrderedDict, defaultdict
import csv
from utils.models import model_selection
from utils.dcs import *
from models.projector import Project
import math
from functools import reduce
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Any
from argparse import Namespace
from pathlib import Path
from log import logger
from utils.flops import FlopMeter
import args as args_module

SCALING_FACTOR = 1.2

def get_random_guess_perf(dataset):
    if dataset == "cifar10":
        return 1 / 10 * SCALING_FACTOR
    elif dataset == "cifar100":
        return 1 / 100 * SCALING_FACTOR
    elif "imagenet" in dataset:
        return 1 / 1000 * SCALING_FACTOR
    else:
        raise NotImplementedError

def adjust_learning_rate(args, optimizer, len_loader, step):
    max_steps = args.kd_epochs * len_loader
    base_lr = args.kd_lr #* args.batch_size / 256

    warmup_steps = 10 * len_loader
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_tensor_parameters(model,fedbn=False):
    from flwr.common.parameter import ndarrays_to_parameters

    return ndarrays_to_parameters(
        get_params(model,fedbn)
    )

def get_params(model,fedbn=False):
    """Get model weights as a list of NumPy ndarrays."""

    if(fedbn):
        return [val.cpu().numpy() for name, val in model.state_dict().items() if 'bn' not in name]
    else:
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

def count_params(model,trainable = False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def set_params(model, params, fedbn = False, bb_only = False):
    """Set model weights from a list of NumPy   ndarrays."""
    
    # keys = model.state_dict().keys()
    
    if(bb_only):
        keys = model.state_dict().keys()
        params_dict = dict(zip(keys, params))
        linear_keys = [k for k in params_dict.keys() if "linear" in k]
        [params_dict.pop(k) for k in linear_keys] # pop layers linear layers
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict.items()})

        model.load_state_dict(state_dict, strict=False)
    elif(fedbn):
        keys = [k for k in model.state_dict().keys() if 'bn' not in k]
        params_dict = zip(keys, params)
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})

        model.load_state_dict(state_dict, strict=False)
    else:
        keys = model.state_dict().keys()
        params_dict = zip(keys, params)
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})

        model.load_state_dict(state_dict, strict=True)


def pile_str(line, item):
    return "_".join([line, item])


def aggregate_client_metrics(
    metrics: Iterable[Tuple[int, Mapping[str, object]]]
) -> Dict[str, float]:
    """Aggregate only the metric keys required by W&B reporting."""

    totals: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    aggregated: Dict[str, float] = {}

    if not hasattr(aggregate_client_metrics, "_running_total_flops"):
        aggregate_client_metrics._running_total_flops = 0.0  # type: ignore[attr-defined]
    if not hasattr(aggregate_client_metrics, "_running_total_flops_compression"):
        aggregate_client_metrics._running_total_flops_compression = 0.0  # type: ignore[attr-defined]
    allowed_metric_keys = {
        "upload_sparsity",
        "download_sparsity",
        "server_to_client_nonzero",
        "server_to_client_density",
        "nonzero_communication_total",
        "client_to_server_nonzero",
        "client_to_server_density",
        "upload_traffic",
        "download_traffic",
        "distributed_test_accuracy",
        "distributed_loss",
        "flops_by_epoch",
        "flops_compression",
        "flops_decompression",
        "serialization_flops_round_clients",
        "deserialization_flops_round_clients",
        "compression_flops_round_clients",
        "decompression_flops_round_clients",
        "intermediate_communication_processing_flops_round_clients",
        "aggregation_flops_round_server",
        "update_flops_round_server",
        "evaluation_flops_round",
        "serialization_flops_round_server",
        "deserialization_flops_round_server",
        "compression_flops_round_server",
        "decompression_flops_round_server",
        "intermediate_communication_processing_flops_round_server",
    }

    for num_examples, client_metrics in metrics:
        if not isinstance(client_metrics, Mapping):
            continue

        for key, value in client_metrics.items():
            if key == "cid" or key not in allowed_metric_keys:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            totals[key] += numeric_value
            counts[key] += 1

    training_flops_round_clients = float(totals.get("flops_by_epoch", 0.0))
    aggregation_flops_round_server = float(totals.get("aggregation_flops_round_server", 0.0))
    update_flops_round_server = float(totals.get("update_flops_round_server", 0.0))
    evaluation_flops_round = float(totals.get("evaluation_flops_round", 0.0))
    round_flops = float(
        training_flops_round_clients
        + aggregation_flops_round_server
        + update_flops_round_server
        + evaluation_flops_round
    )

    serialization_flops_round_clients = float(totals.get("serialization_flops_round_clients", 0.0))
    serialization_flops_round_server = float(totals.get("serialization_flops_round_server", 0.0))
    deserialization_flops_round_clients = float(totals.get("deserialization_flops_round_clients", 0.0))
    deserialization_flops_round_server = float(totals.get("deserialization_flops_round_server", 0.0))
    compression_flops_round_clients = float(
        totals.get("compression_flops_round_clients", totals.get("flops_compression", 0.0))
    )
    compression_flops_round_server = float(totals.get("compression_flops_round_server", 0.0))
    decompression_flops_round_clients = float(
        totals.get("decompression_flops_round_clients", totals.get("flops_decompression", 0.0))
    )
    decompression_flops_round_server = float(totals.get("decompression_flops_round_server", 0.0))
    intermediate_comm_flops_round_clients = float(
        totals.get("intermediate_communication_processing_flops_round_clients", 0.0)
    )
    intermediate_comm_flops_round_server = float(
        totals.get("intermediate_communication_processing_flops_round_server", 0.0)
    )

    round_flops_compression = float(
        serialization_flops_round_clients
        + serialization_flops_round_server
        + deserialization_flops_round_clients
        + deserialization_flops_round_server
        + compression_flops_round_clients
        + compression_flops_round_server
        + decompression_flops_round_clients
        + decompression_flops_round_server
        + intermediate_comm_flops_round_clients
        + intermediate_comm_flops_round_server
    )
    round_total_flops = float(round_flops + round_flops_compression)

    running_total = getattr(aggregate_client_metrics, "_running_total_flops", 0.0)
    running_total += round_total_flops
    aggregate_client_metrics._running_total_flops = running_total  # type: ignore[attr-defined]

    running_total_compression = getattr(
        aggregate_client_metrics, "_running_total_flops_compression", 0.0
    )
    running_total_compression += round_flops_compression
    aggregate_client_metrics._running_total_flops_compression = running_total_compression  # type: ignore[attr-defined]

    aggregated["round_flops"] = round_flops
    aggregated["round_flops_compression"] = round_flops_compression
    aggregated["total_flops"] = float(running_total)
    aggregated["total_flops_compression"] = float(running_total_compression)
    aggregated["round_training_flops_clients"] = training_flops_round_clients
    aggregated["aggregation_flops_round_server"] = aggregation_flops_round_server
    aggregated["update_flops_round_server"] = update_flops_round_server
    aggregated["evaluation_flops_round"] = evaluation_flops_round
    aggregated["serialization_flops_round_clients"] = serialization_flops_round_clients
    aggregated["serialization_flops_round_server"] = serialization_flops_round_server
    aggregated["deserialization_flops_round_clients"] = deserialization_flops_round_clients
    aggregated["deserialization_flops_round_server"] = deserialization_flops_round_server
    aggregated["compression_flops_round_clients"] = compression_flops_round_clients
    aggregated["compression_flops_round_server"] = compression_flops_round_server
    aggregated["decompression_flops_round_clients"] = decompression_flops_round_clients
    aggregated["decompression_flops_round_server"] = decompression_flops_round_server
    aggregated["intermediate_communication_processing_flops_round_clients"] = (
        intermediate_comm_flops_round_clients
    )
    aggregated["intermediate_communication_processing_flops_round_server"] = (
        intermediate_comm_flops_round_server
    )

    mean_keys = {
        "upload_sparsity",
        "download_sparsity",
        "server_to_client_density",
        "client_to_server_density",
        "distributed_test_accuracy",
        "distributed_loss",
    }
    sum_keys = {
        "server_to_client_nonzero",
        "client_to_server_nonzero",
        "nonzero_communication_total",
    }

    for key in mean_keys:
        if counts.get(key, 0) > 0:
            mean_value = float(totals[key]) / float(counts[key])
            aggregated[key] = mean_value
            if key == "upload_sparsity":
                aggregated["upload_sparsity_mean"] = mean_value
            if key == "download_sparsity":
                aggregated["download_sparsity_mean"] = mean_value

    for key in sum_keys:
        if counts.get(key, 0) > 0:
            aggregated[key] = float(totals[key])

    upload_traffic = float(totals.get("upload_traffic", 0.0))
    download_traffic = float(totals.get("download_traffic", 0.0))
    aggregated["upload_traffic"] = upload_traffic
    aggregated["download_traffic"] = download_traffic
    aggregated["overall_traffic"] = upload_traffic + download_traffic

    return aggregated


def compute_payload_size_bytes(payload: Any) -> float:
    """Compute payload size in bytes for transmitted objects."""

    if payload is None:
        return 0.0

    if isinstance(payload, np.ndarray):
        return float(payload.nbytes)
    if isinstance(payload, torch.Tensor):
        return float(payload.nelement() * payload.element_size())
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return float(len(payload))
    if isinstance(payload, Mapping):
        return float(sum(compute_payload_size_bytes(v) for v in payload.values()))
    if isinstance(payload, (list, tuple)):
        return float(sum(compute_payload_size_bytes(v) for v in payload))

    if hasattr(payload, "tensors") and isinstance(getattr(payload, "tensors"), list):
        return float(sum(len(tensor) for tensor in getattr(payload, "tensors")))

    try:
        np_array = np.asarray(payload)
    except Exception:
        return 0.0
    return float(np_array.nbytes)


def compute_payload_num_elements(payload: Any) -> float:
    """Compute scalar element count for a payload structure."""

    if payload is None:
        return 0.0

    if isinstance(payload, np.ndarray):
        return float(payload.size)
    if isinstance(payload, torch.Tensor):
        return float(payload.nelement())
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return float(len(payload))
    if isinstance(payload, Mapping):
        return float(sum(compute_payload_num_elements(v) for v in payload.values()))
    if isinstance(payload, (list, tuple)):
        return float(sum(compute_payload_num_elements(v) for v in payload))

    if hasattr(payload, "tensors") and isinstance(getattr(payload, "tensors"), list):
        return float(sum(len(tensor) for tensor in getattr(payload, "tensors")))

    try:
        np_array = np.asarray(payload)
    except Exception:
        return 0.0
    return float(np_array.size)


def estimate_serialization_flops(payload: Any) -> float:
    """Estimate serialization FLOPs from payload element and byte volume."""

    total_bytes = compute_payload_size_bytes(payload)
    total_elements = compute_payload_num_elements(payload)
    return float(total_elements + total_bytes)


def estimate_deserialization_flops(payload: Any) -> float:
    """Estimate deserialization FLOPs from payload element and byte volume."""

    total_bytes = compute_payload_size_bytes(payload)
    total_elements = compute_payload_num_elements(payload)
    return float(total_elements + total_bytes)


def estimate_fedavg_aggregation_and_update_flops(
    client_payloads: Sequence[Sequence[np.ndarray]],
) -> Tuple[float, float]:
    """Estimate server aggregation/update FLOPs for elementwise weighted averaging."""

    if not client_payloads:
        return 0.0, 0.0

    num_clients = len(client_payloads)
    first_payload = client_payloads[0]
    aggregation_flops = 0.0
    update_flops = 0.0

    for tensor in first_payload:
        tensor_array = np.asarray(tensor)
        num_elements = float(tensor_array.size)
        if num_elements <= 0.0:
            continue
        aggregation_flops += num_elements * float(max(num_clients - 1, 0))
        update_flops += num_elements

    return float(aggregation_flops), float(update_flops)


def _round_metrics_output_path(runtime_args: Namespace) -> Path:
    base_path = Path(getattr(runtime_args, "path_results", "results/"))
    base_path.mkdir(parents=True, exist_ok=True)
    file_name = getattr(runtime_args, "file_name", "run")
    return base_path / f"{file_name}_round_flops_metrics.csv"


def _persist_round_metrics_log(runtime_args: Namespace, payload: Mapping[str, float]) -> None:
    fieldnames = [
        "round",
        "round_flops",
        "round_training_flops_clients",
        "aggregation_flops_round_server",
        "update_flops_round_server",
        "evaluation_flops_round",
        "round_flops_compression",
        "compression_flops_clients",
        "compression_flops_server",
        "compression_flops_round_clients",
        "compression_flops_round_server",
        "decompression_flops_clients",
        "decompression_flops_server",
        "decompression_flops_round_clients",
        "decompression_flops_round_server",
        "serialization_flops",
        "serialization_flops_round_clients",
        "serialization_flops_round_server",
        "deserialization_flops_round_clients",
        "deserialization_flops_round_server",
        "intermediate_communication_processing_flops_round_clients",
        "intermediate_communication_processing_flops_round_server",
        "total_flops",
        "total_flops_compression",
        "acc_servers_highest",
        "overall_traffic",
        "upload_traffic",
        "download_traffic",
    ]
    output_path = _round_metrics_output_path(runtime_args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = {field: payload.get(field, "") for field in fieldnames}
    file_exists = output_path.exists()
    with output_path.open("a", encoding="utf-8", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def maybe_log_to_wandb(metrics: Mapping[str, float], *, step: Optional[int] = None) -> None:
    """Log metrics to Weights & Biases when the integration is enabled."""

    if not metrics:
        return

    try:
        runtime_args = args_module.get_args()
    except RuntimeError:
        return

    if not getattr(runtime_args, "wandb", False):
        return

    if not hasattr(maybe_log_to_wandb, "_round_cache"):
        maybe_log_to_wandb._round_cache = {}  # type: ignore[attr-defined]
    if not hasattr(maybe_log_to_wandb, "_running_total_flops"):
        maybe_log_to_wandb._running_total_flops = 0.0  # type: ignore[attr-defined]
    if not hasattr(maybe_log_to_wandb, "_running_total_flops_compression"):
        maybe_log_to_wandb._running_total_flops_compression = 0.0  # type: ignore[attr-defined]

    log_payload = dict(metrics)

    if step is not None:
        cache = maybe_log_to_wandb._round_cache  # type: ignore[attr-defined]
        round_state = dict(cache.get(step, {}))
        round_state.update(log_payload)

        training = float(round_state.get("round_training_flops_clients", 0.0))
        aggregation = float(round_state.get("aggregation_flops_round_server", 0.0))
        update = float(round_state.get("update_flops_round_server", 0.0))
        evaluation = float(round_state.get("evaluation_flops_round", 0.0))
        round_flops = float(training + aggregation + update + evaluation)

        serialization = float(
            round_state.get("serialization_flops_round_clients", 0.0)
            + round_state.get("serialization_flops_round_server", 0.0)
        )
        deserialization = float(
            round_state.get("deserialization_flops_round_clients", 0.0)
            + round_state.get("deserialization_flops_round_server", 0.0)
        )
        compression = float(
            round_state.get("compression_flops_round_clients", 0.0)
            + round_state.get("compression_flops_round_server", 0.0)
        )
        decompression = float(
            round_state.get("decompression_flops_round_clients", 0.0)
            + round_state.get("decompression_flops_round_server", 0.0)
        )
        intermediate_comm = float(
            round_state.get("intermediate_communication_processing_flops_round_clients", 0.0)
            + round_state.get("intermediate_communication_processing_flops_round_server", 0.0)
        )
        round_flops_compression = float(
            serialization + deserialization + compression + decompression + intermediate_comm
        )
        round_total = float(round_flops + round_flops_compression)

        previous_round_total = float(round_state.get("_round_total_accounted", 0.0))
        previous_round_compression = float(round_state.get("_round_compression_accounted", 0.0))
        maybe_log_to_wandb._running_total_flops += round_total - previous_round_total  # type: ignore[attr-defined]
        maybe_log_to_wandb._running_total_flops_compression += (  # type: ignore[attr-defined]
            round_flops_compression - previous_round_compression
        )

        round_state["round_flops"] = round_flops
        round_state["round_flops_compression"] = round_flops_compression
        round_state["total_flops"] = float(maybe_log_to_wandb._running_total_flops)  # type: ignore[attr-defined]
        round_state["total_flops_compression"] = float(
            maybe_log_to_wandb._running_total_flops_compression  # type: ignore[attr-defined]
        )
        round_state["compression_flops_clients"] = float(
            round_state.get("compression_flops_round_clients", 0.0)
        )
        round_state["compression_flops_server"] = float(
            round_state.get("compression_flops_round_server", 0.0)
        )
        round_state["decompression_flops_clients"] = float(
            round_state.get("decompression_flops_round_clients", 0.0)
        )
        round_state["decompression_flops_server"] = float(
            round_state.get("decompression_flops_round_server", 0.0)
        )
        round_state["serialization_flops"] = float(
            round_state.get("serialization_flops_round_clients", 0.0)
            + round_state.get("serialization_flops_round_server", 0.0)
        )
        round_state["_round_total_accounted"] = round_total
        round_state["_round_compression_accounted"] = round_flops_compression
        cache[step] = round_state

        for key in (
            "round_flops",
            "round_flops_compression",
            "total_flops",
            "total_flops_compression",
            "round_training_flops_clients",
            "aggregation_flops_round_server",
            "update_flops_round_server",
            "evaluation_flops_round",
            "serialization_flops_round_clients",
            "serialization_flops_round_server",
            "compression_flops_round_clients",
            "compression_flops_round_server",
            "decompression_flops_round_clients",
            "decompression_flops_round_server",
            "deserialization_flops_round_clients",
            "deserialization_flops_round_server",
            "compression_flops_clients",
            "compression_flops_server",
            "decompression_flops_clients",
            "decompression_flops_server",
            "serialization_flops",
            "intermediate_communication_processing_flops_round_clients",
            "intermediate_communication_processing_flops_round_server",
            "acc_servers_highest",
            "overall_traffic",
            "upload_traffic",
            "download_traffic",
        ):
            if key in round_state:
                log_payload[key] = round_state[key]

        log_payload["round"] = float(step)
        _persist_round_metrics_log(runtime_args, log_payload)

    import wandb

    wandb.log(log_payload, step=step)


def _extract_metric_values(
    metric_entries: Optional[Sequence[Tuple[int, float]]]
) -> np.ndarray:
    if not metric_entries:
        return np.array([], dtype=float)
    rounds, values = zip(*metric_entries)
    return np.asarray(values, dtype=float)


def _summarize_series(series: np.ndarray) -> Dict[str, float]:
    if series.size == 0:
        return {}
    series_mean = float(np.mean(series))
    series_std = float(np.std(series))
    return {
        "mean": series_mean,
        "std": series_std,
        "lowest": float(series_mean - series_std),
        "highest": float(series_mean + series_std),
    }


def tell_history(
    hist,
    file_name,
    infos=None,
    path="",
    report_metadata: Optional[Dict[str, float]] = None,
    args: Optional[Namespace] = None,
):
    accuracy_centralized = hist.metrics_centralized.get("accuracy", [])
    acc_cent_values = _extract_metric_values(accuracy_centralized)
    losses_cent = hist.losses_centralized
    losses_dis = hist.losses_distributed

    acc_distributed = hist.metrics_distributed.get("distributed_test_accuracy")
    if acc_distributed is None:
        acc_distributed = hist.metrics_distributed.get("dist_acc", [])

    acc_dis_values = _extract_metric_values(acc_distributed)
    losses_dis_values = _extract_metric_values(losses_dis)

    if infos is None:
        infos = {}

    infos["accuracy_cent"] = acc_cent_values
    infos["accuracy_dist"] = acc_distributed
    infos["losses_cent"] = losses_cent
    infos["losses_dis"] = losses_dis

    report: Dict[str, float] = {}

    training_summary = _summarize_series(losses_dis_values)
    if training_summary:
        report.update(
            {
                "training_loss_lowest": training_summary["lowest"],
                "training_loss_highest": training_summary["highest"],
                "distributed_loss": training_summary["mean"],
            }
        )

    client_acc_summary = _summarize_series(acc_dis_values)
    if client_acc_summary:
        report.update(
            {
                "distributed_test_accuracy": client_acc_summary["mean"],
                "acc_clients_lowest": client_acc_summary["lowest"],
                "acc_clients_highest": client_acc_summary["highest"],
            }
        )

    server_acc_summary = _summarize_series(acc_cent_values)
    if server_acc_summary:
        report.update(
            {
                "acc_servers_lowest": server_acc_summary["lowest"],
                "acc_servers_highest": server_acc_summary["highest"],
            }
        )

    if "cos_mean" in report and "cos_std" in report:
        report["cos"] = float(report["cos_mean"])
        report["cos_lowest"] = float(report["cos_mean"] - report["cos_std"])
        report["cos_highest"] = float(report["cos_mean"] + report["cos_std"])
        report.pop("cos_mean", None)
        report.pop("cos_std", None)

    if report:
        infos["report"] = report

    with open(path + file_name + ".npy", "wb") as f:
        np.save(f, infos)

    if args is not None and args.wandb and report:
        import wandb

        wandb.log(report)


def inst_model_info(model_info: Info, use_proj: bool = False, out_dim: int = -1):
    model = model_selection(model_info.model)
    model = model(
        model_info.feature_maps,
        model_info.input_shape,
        model_info.num_classes,
        model_info.batchn,
    )

    if use_proj:
        model = Project(model, 
                        input_dim=model.features_dim, 
                        out_dim=out_dim
                )

    return model


def inst_model_lora_info(model_info: Info, lora_config : LoraInfo):
    from utils.lora import inject_low_rank
    model = model_selection(model_info.model)
    model = model(
        model_info.feature_maps,
        model_info.input_shape,
        model_info.num_classes,
        model_info.batchn,
    )


    return inject_low_rank(model,lora_config)

def create_all_dirs(path_results: str) -> None:
    Path.mkdir(Path("./data"), parents=True, exist_ok=True)
    Path.mkdir(Path(path_results), parents=True, exist_ok=True)
    Path.mkdir(Path("./checkpoint"), parents=True, exist_ok=True)


def train(net, trainloader, epochs, optimizer, criterion, device):
    """Train the network on the training set and track FLOPs per epoch."""

    net.train()
    flop_meter = FlopMeter(net)
    epoch_flops: list[float] = []

    for epoch_idx in range(epochs):
        flop_meter.start_epoch()
        for images, labels, _ in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            flop_meter.start_batch()
            out, _ = net(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            flop_meter.finish_batch()

        epoch_total_flops = flop_meter.finish_epoch()
        epoch_flops.append(epoch_total_flops)
        logger.info(
            "Epoch %s/%s - approx FLOPs: %.2f",
            epoch_idx + 1,
            epochs,
            epoch_total_flops,
        )

    flop_meter.close()

    return {"epoch_flops": epoch_flops}


def test(model, test_loader, device, track_flops: bool = False):
    if not isinstance(model,list):
        model =  [model]

    for m in model:
        m.eval()
        m.to(device)
    flop_meters = [FlopMeter(m) for m in model] if track_flops else []
    if track_flops:
        for meter in flop_meters:
            meter.start_epoch()
    outputs=[]
    losses = torch.zeros(len(model))
    accuracies = torch.zeros(len(model))
    en_loss, en_accuracy, total,accuracy_top_5 = 0, 0, 0,0
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs=[]
            for model_idx, m in enumerate(model):
                if track_flops:
                    flop_meters[model_idx].start_batch()
                out, _ = m(data)
                if track_flops:
                    flop_meters[model_idx].finish_batch()
                outputs.append(out)
            if(len(model)>1):
                en_output = sum(outputs)/len(model)
            else:
                en_output = outputs[0]

            for i,out in enumerate(outputs):
                losses[i] += criterion(out, target).item() * data.shape[0]
                pred = out.argmax(dim=1, keepdim=True)
                accuracies[i] += pred.eq(target.view_as(pred)).sum().item()

            en_loss += criterion(en_output, target).item() * data.shape[0]
            pred = en_output.argmax(dim=1, keepdim=True)
            en_accuracy += pred.eq(target.view_as(pred)).sum().item()

            total += target.shape[0]
            # preds = output.sort(dim = 1, descending = True)[1][:,:5]
            # for i in range(preds.shape[0]):
            #     if target[i] in preds[i]:
            #         accuracy_top_5 += 1

    # return results

    evaluation_flops = 0.0
    if track_flops:
        for meter in flop_meters:
            evaluation_flops += float(meter.finish_epoch())
            meter.close()

    return {
        "test_loss": en_loss / total,
        "test_acc": en_accuracy / total,
        "test_acc_top_5": accuracy_top_5 / total,
        "losses": losses/total,
        "accuracies": accuracies/total,
        "evaluation_flops": evaluation_flops,
    }

def quick_plot(file_name, threshold=0.7):
    import matplotlib.pyplot as plt

    for i, name in enumerate(file_name):
        vec = np.load(name, allow_pickle=True).item()
        acc = vec["accuracy_cent"]
        rounds = range(len(acc))
        max_idx = acc.argmax()
        plt.plot(rounds, acc, label=f"run {i}")
        round_threshold = np.argmax(acc > threshold)
        print(
            f"Run {i} : Max accuracy {acc[max_idx]} @ round {max_idx+1}, "
            + f"it reaches {threshold} @ round {round_threshold} - {name}"
        )
    plt.legend()
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()


def save_model(file_name, data):
    # file_name must contains path ex: "checkpoint/server.npy"
    obj_data = np.array(data, dtype=object)
    np.save(file_name, obj_data)

def ema(prev_weights,results,decay = 0.9):
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]

    ema_weights = [x*decay + y*(1-decay)
                    for x,y in zip(weights_prime,prev_weights)]
    
    return ema_weights

def load_pretrained(model,model_name,path="pretrained"):

    path_to_pretrained = f"./{path}/{model_name}.pt"
    state_dict = torch.load(path_to_pretrained, map_location=lambda storage, loc: storage)
    model_state_dict = model.state_dict()
    ##Correcting for the number of classes, it should not be a problem
    ##in future implementations, where it would be trained with ssl
    state_dict["fc.weight"] = model_state_dict["fc.weight"]
    state_dict["fc.bias"] = model_state_dict["fc.bias"]

    model.load_state_dict(state_dict)
    print("### Successfully Loaded from pretrained ###")
    return model
