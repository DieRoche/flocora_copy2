import gc

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

    flop_metrics = {
        f"flops_epoch_{idx+1}": float(value)
        for idx, value in enumerate(epoch_flops)
    }
    if epoch_flops:
        flop_metrics["flops_total"] = float(sum(epoch_flops))
    
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
