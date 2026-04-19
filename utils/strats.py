from argparse import Namespace
from typing import Any, Dict, Mapping, Optional

import torch
from torch.utils.data import DataLoader
import numpy as np

from log import logger, HFILE
from utils.lora import extract_AB_matrix
from utils.utils import test, save_model, set_params


def _to_serializable(value: Any) -> Any:
    """Convert tensors and other containers to Python-native types."""

    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.numel() == 1:
            return value.item()
        return value.cpu().tolist()

    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]

    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass

    return value


def _build_metrics(ans: Dict[str, Any]) -> Dict[str, Any]:
    """Return a reduced Flower/W&B metric payload with canonical names."""

    serialized = {key: _to_serializable(val) for key, val in ans.items()}
    metrics: Dict[str, Any] = {}

    loss = serialized.get("test_loss")
    accuracy = serialized.get("test_acc")

    if loss is not None:
        metrics["distributed_loss"] = loss
    if accuracy is not None:
        metrics["distributed_test_accuracy"] = accuracy
        metrics["acc_servers_highest"] = accuracy
        metrics["acc_server_highest"] = accuracy

    return metrics


def _ensure_float(value: float) -> float:
    """Return ``value`` as a plain ``float``."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _resolve_clients_per_round(args: Namespace, config: Optional[Mapping[str, Any]]) -> float:
    """Return the configured number of participating clients per round."""

    config = config or {}

    for key in ("clients_per_round", "min_fit_clients", "sample_size"):
        if key in config:
            try:
                value = float(config[key])
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value

    if hasattr(args, "clients_per_round"):
        try:
            value = float(getattr(args, "clients_per_round"))
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            return value

    try:
        pool_size = float(getattr(args, "num_clients", 0.0))
        sample_rate = float(getattr(args, "samp_rate", 0.0))
    except (TypeError, ValueError):
        pool_size = 0.0
        sample_rate = 0.0

    clients_per_round = pool_size * sample_rate
    if clients_per_round <= 0 and pool_size > 0:
        clients_per_round = 1.0

    return max(clients_per_round, 0.0)


def _resolve_communication_steps(
    args: Namespace, config: Optional[Mapping[str, Any]]
) -> float:
    """Return the number of communication steps per round."""

    config = config or {}

    for key in ("communication_steps_per_round", "communication_steps"):
        if key in config:
            try:
                value = float(config[key])
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value

    for attr in (
        "communication_steps_per_round",
        "communication_steps",
    ):
        if hasattr(args, attr):
            try:
                value = float(getattr(args, attr))
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value

    return 1.0


def _compute_model_payload_size(parameters: Any) -> float:
    """Return the payload size of ``parameters`` in bytes."""

    if parameters is None:
        return 0.0

    total_bytes = 0.0
    for param in parameters:
        if param is None:
            continue
        if isinstance(param, torch.Tensor):
            total_bytes += float(param.nelement() * param.element_size())
            continue
        if isinstance(param, np.ndarray):
            total_bytes += float(param.nbytes)
            continue
        try:
            np_array = np.asarray(param)
        except Exception:  # pylint: disable=broad-except
            continue
        total_bytes += float(np_array.nbytes)

    return total_bytes


def _build_traffic_metrics(
    parameters: Any, args: Namespace, config: Optional[Mapping[str, Any]]
) -> Dict[str, float]:
    """Traffic metrics are collected from active client fit payloads."""
    del parameters, args, config
    return {}


class Evaluate:
    def __init__(self, model, test_set, device, args: Namespace):
        self.args = args
        self.test_loader = DataLoader(
            test_set,
            batch_size=256,
            shuffle=False,
            pin_memory=True,
            num_workers=args.nworkers,
        )
        self.model = model
        self.device = device

    def __call__(self, server_round, parameters, config, to_log=None):
        set_params(self.model, parameters, self.args.fedbn)
        self.model.to(self.device)
        ans = test(self.model, self.test_loader, self.device)
        metrics = _build_metrics(ans)
        traffic_metrics = _build_traffic_metrics(parameters, self.args, config)
        if traffic_metrics:
            metrics.update(traffic_metrics)
        loss = metrics.get("distributed_loss")
        accuracy = metrics.get("distributed_test_accuracy")
        logger.info(
            f"Server round {server_round} loss : {loss:.4f} acc : {accuracy:.4f}",
            extra=HFILE,
        )

        if self.args.num_rounds == server_round or server_round % self.args.freq_checkpoint == 0:
            save_model(f"checkpoint/{self.args.file_name}.npy", parameters)
        if server_round != -1 and self.args.wandb:
            if to_log is None:
                to_log = {}
            log_payload = {**to_log, **metrics}
            import wandb

            wandb.log(log_payload)

        return loss, metrics


class EvaluateLora:
    def __init__(self, model, lora_config, test_set, device, args: Namespace):
        self.args = args
        self.test_loader = DataLoader(
            test_set,
            batch_size=256,
            shuffle=False,
            pin_memory=True,
            num_workers=args.nworkers,
        )
        self.model = model
        self.device = device
        self.past_a_matrix, self.past_b_matrix = extract_AB_matrix(model.state_dict())

    def __call__(self, server_round, parameters, config):
        set_params(self.model, parameters, self.args.fedbn)
        to_log = {}

        self.model.to(self.device)
        if self.args.log_a_sim and server_round >= 1:
            current_a_list, current_b_list = extract_AB_matrix(self.model.state_dict())
            cos = torch.nn.CosineSimilarity(dim=0)
            simA = [cos(curr, old).mean() for curr, old in zip(current_a_list, self.past_a_matrix)]
            simB = [cos(curr, old).mean() for curr, old in zip(current_b_list, self.past_b_matrix)]
            for i, (sA, sB) in enumerate(zip(simA, simB)):
                logger.info(f"Similarity for layer {i} --> simA: {sA}, simB: {sB}")
                if self.args.wandb:
                    to_log[f"simA_layer_{i}"] = sA
                    to_log[f"simB_layer_{i}"] = sB
            self.past_a_matrix = current_a_list
            self.past_b_matrix = current_b_list

        ans = test(self.model, self.test_loader, self.device)
        metrics = _build_metrics(ans)
        traffic_metrics = _build_traffic_metrics(parameters, self.args, config)
        if traffic_metrics:
            metrics.update(traffic_metrics)
        loss = metrics.get("distributed_loss")
        accuracy = metrics.get("distributed_test_accuracy")
        logger.info(
            f"Server round {server_round} loss : {loss:.4f} acc : {accuracy:.4f}",
            extra=HFILE,
        )

        if self.args.num_rounds == server_round or server_round % self.args.freq_checkpoint == 0:
            save_model(f"checkpoint/{self.args.file_name}.npy", parameters)
        if server_round != -1 and self.args.wandb:
            log_payload = {**to_log, **metrics}
            import wandb

            wandb.log(log_payload)

        return loss, metrics


def get_evaluate_fn(model, test_set, device, args: Namespace):
    """Return an evaluation function for server-side evaluation."""

    test_loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        pin_memory=True,
        num_workers=args.nworkers,
    )

    def evaluate(server_round, parameters, config):
        set_params(model, parameters, args.fedbn)
        model.to(device)
        ans = test(model, test_loader, device)
        metrics = _build_metrics(ans)
        traffic_metrics = _build_traffic_metrics(parameters, args, config)
        if traffic_metrics:
            metrics.update(traffic_metrics)
        loss = metrics.get("distributed_loss")
        accuracy = metrics.get("distributed_test_accuracy")
        logger.info(
            f"Server round {server_round} loss : {loss:.4f} acc : {accuracy:.4f}",
            extra=HFILE,
        )

        if args.num_rounds == server_round or server_round % args.freq_checkpoint == 0:
            save_model(f"checkpoint/{args.file_name}.npy", parameters)
        if server_round != -1 and args.wandb:
            import wandb

            wandb.log(metrics)

        return loss, metrics

    return evaluate


def get_model_size(model):
    return [p.shape for p in model.parameters()]
