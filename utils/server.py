from argparse import Namespace
from typing import Optional

import flwr as fl

import args as args_module
from utils.dcs import FlInfo, Info, LoraInfo, ServerInfo
from utils.lora import *
from utils.utils import inst_model_info, resolve_wandb_run_name


def _get_args() -> Optional[Namespace]:
    try:
        return args_module.get_args()
    except RuntimeError:
        return None


def _require_args() -> Namespace:
    args = _get_args()
    if args is None:
        raise RuntimeError("Arguments have not been initialized. Call parse_and_cache_args() first.")
    return args


def start_server(srv_addr, strategy, num_rounds, server_queue):
    """Start the server."""

    args = _get_args()
    if args is not None and args.wandb:
        import wandb

        wandb.init(
            entity=args.entity,
            # set the wandb project where this run will be logged
            project=args.wandb_prj_name,
            name=resolve_wandb_run_name(args),
            # track hyperparameters and run metadata
            config=args,
        )
    if server_queue != None:
        server_queue.put(
            fl.server.start_server(
                server_address=srv_addr,
                config=fl.server.ServerConfig(num_rounds=num_rounds),
                strategy=strategy,
            )
        )
    else:
        return fl.server.start_server(
            server_address=srv_addr,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )


def build_clients(clients_models, input_shape, num_classes):
    args = _require_args()
    cls_models = []
    for cl in clients_models:
        cls_models.append(
            Info(
                model=cl,
                dataset_name=args.dataset,
                feature_maps=args.feature_maps,
                input_shape=input_shape,
                num_classes=num_classes,
                batchn=args.batchn,
            )
        )
    return cls_models


def build_df_lora_config(protos: Info, lora_alpha, lora_r):
    lora_config = []
    for p in protos:
        target_modules, modules_to_save = get_target_save(inst_model_info(p), p.model)

        lc = LoraInfo(
            alpha=lora_alpha,
            r=lora_r,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
        )
        lora_config.append(lc)


def build_prototypes(protos, input_shape, num_classes):
    args = _require_args()
    prototypes = []

    for p in protos:
        prototypes.append(
            Info(
                model=p,
                dataset_name=args.dataset,
                feature_maps=args.feature_maps,
                input_shape=input_shape,
                num_classes=num_classes,
                batchn=args.batchn,
            )
        )

    return prototypes


def start_client(info, fl_info):
    from client import FlowerClient

    client = FlowerClient(info, fl_info)
    client.start_client()
