from argparse import Namespace
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

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
    """Return a Flower/W&B friendly metric payload from ``ans``."""

    metrics = {key: _to_serializable(val) for key, val in ans.items()}

    loss = metrics.get("test_loss")
    accuracy = metrics.get("test_acc")

    if loss is not None and "loss" not in metrics:
        metrics["loss"] = loss
    if accuracy is not None:
        metrics.setdefault("accuracy", accuracy)

    return metrics


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
        loss = metrics.get("test_loss")
        accuracy = metrics.get("test_acc")
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
                    to_log[f"simA_l{i}"] = sA
                    to_log[f"simB_l{i}"] = sB
            self.past_a_matrix = current_a_list
            self.past_b_matrix = current_b_list

        ans = test(self.model, self.test_loader, self.device)
        metrics = _build_metrics(ans)
        loss = metrics.get("test_loss")
        accuracy = metrics.get("test_acc")
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
        loss = metrics.get("test_loss")
        accuracy = metrics.get("test_acc")
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
