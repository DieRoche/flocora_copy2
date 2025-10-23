import torch
import numpy as np
from collections import OrderedDict, defaultdict
from utils.models import model_selection
from utils.dcs import *
from models.projector import Project
import math
from functools import reduce
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple
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
    """Aggregate numeric client metrics and expose totals/means.

    Parameters
    ----------
    metrics: Iterable[Tuple[int, Mapping[str, object]]]
        Sequence of ``(num_examples, metrics)`` tuples as provided by Flower.

    Returns
    -------
    Dict[str, float]
        Aggregated metrics including per-key sums (``*_sum``), means (``*_mean``),
        and per-example values (``*_per_example``) when the total number of
        examples is known. The helper also exposes ``aggregated_client_count``
        and ``aggregated_num_examples`` to aid downstream consumers.
    """

    totals: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    aggregated: Dict[str, float] = {}

    special_metric_keys = {
        "flops_by_epoch",
        "flops_compression",
        "flops_decompression",
        "sum_flops_epoch_includingcompdecomp",
        "total_flops",
    }

    if not hasattr(aggregate_client_metrics, "_running_total_flops"):
        aggregate_client_metrics._running_total_flops = 0.0  # type: ignore[attr-defined]

    total_examples = 0
    client_count = 0

    for num_examples, client_metrics in metrics:
        if not isinstance(client_metrics, Mapping):
            continue
        client_count += 1
        try:
            total_examples += int(num_examples)
        except (TypeError, ValueError):
            pass

        for key, value in client_metrics.items():
            if key == "cid":
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            totals[key] += numeric_value
            counts[key] += 1

    if "sum_flops_epoch_includingcompdecomp" in totals:
        running_total = getattr(
            aggregate_client_metrics, "_running_total_flops", 0.0
        )
        running_total += float(totals["sum_flops_epoch_includingcompdecomp"])
        aggregate_client_metrics._running_total_flops = running_total  # type: ignore[attr-defined]
        totals["total_flops"] = running_total
        counts["total_flops"] = 1

    for key, total in totals.items():
        if key in special_metric_keys:
            aggregated[key] = float(total)
            continue
        aggregated[key] = float(total)
        aggregated[f"{key}_sum"] = float(total)
        aggregated[f"{key}_mean"] = float(total) / float(counts[key])
        if total_examples > 0:
            aggregated[f"{key}_per_example"] = float(total) / float(total_examples)

    if client_count > 0:
        aggregated["aggregated_client_count"] = float(client_count)
    if total_examples > 0:
        aggregated["aggregated_num_examples"] = float(total_examples)

    return aggregated


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

    import wandb

    wandb.log(dict(metrics), step=step)


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

    round_indices: Sequence[int] = []
    if losses_dis:
        first_entry = losses_dis[0]
        if isinstance(first_entry, Sequence) and not isinstance(first_entry, (str, bytes)):
            try:
                round_indices = [int(entry[0]) for entry in losses_dis if len(entry) > 0]
            except (TypeError, ValueError):
                round_indices = []
        else:
            round_indices = list(range(1, len(losses_dis) + 1))

    try:
        acc_distributed = hist.metrics_distributed["dist_acc"]
    except KeyError:
        acc_distributed = []

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
                "training_loss_mean": training_summary["mean"],
                "training_loss_std": training_summary["std"],
                "training_loss_lowest": training_summary["lowest"],
                "training_loss_highest": training_summary["highest"],
            }
        )

    client_acc_summary = _summarize_series(acc_dis_values)
    if client_acc_summary:
        report.update(
            {
                "acc_clients_mean": client_acc_summary["mean"],
                "acc_clients_std": client_acc_summary["std"],
                "acc_clients_lowest": client_acc_summary["lowest"],
                "acc_clients_highest": client_acc_summary["highest"],
            }
        )

    server_acc_summary = _summarize_series(acc_cent_values)
    if server_acc_summary:
        report.update(
            {
                "acc_servers_mean": server_acc_summary["mean"],
                "acc_servers_std": server_acc_summary["std"],
                "acc_servers_lowest": server_acc_summary["lowest"],
                "acc_servers_highest": server_acc_summary["highest"],
            }
        )

    per_round_traffic_metrics: Optional[Dict[str, float]] = None

    if report_metadata is not None:
        model_size_bytes = report_metadata.get("model_size_bytes", 0.0) or 0.0
        clients_per_round = report_metadata.get("clients_per_round", 0.0) or 0.0
        num_rounds = report_metadata.get("num_rounds", 0.0) or 0.0

        if model_size_bytes > 0 and clients_per_round > 0 and num_rounds > 0:
            upload_traffic_round = model_size_bytes * clients_per_round
            download_traffic_round = model_size_bytes * clients_per_round
            total_upload_traffic = upload_traffic_round * num_rounds
            total_download_traffic = download_traffic_round * num_rounds
            overall_traffic = total_upload_traffic + total_download_traffic

            communication_steps = report_metadata.get(
                "communication_steps_per_round"
            )
            try:
                communication_steps = float(communication_steps)
            except (TypeError, ValueError):
                communication_steps = 1.0

            if communication_steps <= 0:
                communication_steps = 1.0

            upload_on_wire_round = upload_traffic_round * communication_steps
            download_on_wire_round = download_traffic_round * communication_steps
            total_upload_on_wire = upload_on_wire_round * num_rounds
            total_download_on_wire = download_on_wire_round * num_rounds
            overall_traffic_on_wire = total_upload_on_wire + total_download_on_wire
            total_communication_steps = communication_steps * num_rounds

            traffic_metrics: Dict[str, float] = {
                "upload_traffic": upload_traffic_round,
                "download_traffic": download_traffic_round,
                "upload_traffic_per_client": model_size_bytes,
                "per_user_upload_traffic": model_size_bytes,
                "overall_traffic": overall_traffic,
                "total_upload_traffic": total_upload_traffic,
                "total_download_traffic": total_download_traffic,
            }

            per_round_traffic_metrics = {
                "upload_traffic": upload_traffic_round,
                "download_traffic": download_traffic_round,
                "upload_traffic_per_client": model_size_bytes,
                "per_user_upload_traffic": model_size_bytes,
            }

            if communication_steps != 1.0:
                traffic_metrics.update(
                    {
                        "communication_steps_per_round": communication_steps,
                        "total_communication_steps": total_communication_steps,
                        "upload_traffic_on_wire": upload_on_wire_round,
                        "download_traffic_on_wire": download_on_wire_round,
                        "total_upload_traffic_on_wire": total_upload_on_wire,
                        "total_download_traffic_on_wire": total_download_on_wire,
                        "overall_traffic_on_wire": overall_traffic_on_wire,
                    }
                )

                per_round_traffic_metrics.update(
                    {
                        "communication_steps_per_round": communication_steps,
                        "upload_traffic_on_wire": upload_on_wire_round,
                        "download_traffic_on_wire": download_on_wire_round,
                    }
                )

            if not round_indices:
                try:
                    round_count = int(num_rounds)
                except (TypeError, ValueError):
                    round_count = 0
                if round_count > 0:
                    round_indices = list(range(1, round_count + 1))

            report.update(traffic_metrics)

    if report:
        infos["report"] = report

    with open(path + file_name + ".npy", "wb") as f:
        np.save(f, infos)

    if args is not None and args.wandb and report:
        import wandb

        if per_round_traffic_metrics and round_indices:
            for round_idx in round_indices:
                wandb.log(per_round_traffic_metrics, step=int(round_idx))

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


def test(model, test_loader, device):
    if not isinstance(model,list):
        model =  [model]

    for m in model:
        m.eval()
        m.to(device)
    outputs=[]
    losses = torch.zeros(len(model))
    accuracies = torch.zeros(len(model))
    en_loss, en_accuracy, total,accuracy_top_5 = 0, 0, 0,0
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs=[]
            for m in model:
                out, _ = m(data)
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

    return {
        "test_loss": en_loss / total,
        "test_acc": en_accuracy / total,
        "test_acc_top_5": accuracy_top_5 / total,
        "losses": losses/total,
        "accuracies": accuracies/total,
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