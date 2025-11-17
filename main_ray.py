import math
import multiprocessing
from argparse import Namespace
from typing import Optional

import torch
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)
from json import dumps
from pathlib import Path

import args as args_module
from torch import device as torch_device
from utils.models import do_model_pool
from utils.dataset import import_dataset, do_fl_partitioning
from utils.utils import *
from utils.file_name import gen_filename
from utils.server import *
from log import logger, HFILE
from utils.strats import Evaluate, EvaluateLora, get_evaluate_fn
from utils.simple_quant import original_msg_size, quant_msg_size

args: Optional[Namespace] = None
client_lr: float = 0.0


def _require_args() -> Namespace:
    if args is None:
        raise RuntimeError("Runtime arguments have not been initialized. Call parse_and_cache_args() first.")
    return args


def fit_config(server_round):
    """Return a configuration with static batch size and (local) epochs."""
    global client_lr
    runtime_args = _require_args()
    if runtime_args.milestones != 0:
        if server_round in runtime_args.milestones:
            client_lr *= runtime_args.lr_step
    logger.debug(f"Client lr {client_lr}, round {server_round}")
    config = {
        "epochs": runtime_args.cl_epochs,  # number of local epochs
        "batch_size": runtime_args.cl_bs,
        "cl_lr": client_lr,
        "cl_momentum": runtime_args.cl_mmt,
        "cl_wd": runtime_args.cl_wd,
        "server_round": server_round,
        "prate": runtime_args.prate,
    }
    return config


def eval_config(server_round):
    """Return a configuration with static batch size and (local) epochs."""

    config = {
        "server_round": server_round,
    }
    return config

def build_server_info(test_set,knn_set=None):
    runtime_args = _require_args()
    return ServerInfo(
        model=runtime_args.model,
        dataset_name=runtime_args.dataset,
        feature_maps=runtime_args.feature_maps,
        input_shape=input_shape,
        num_classes=num_classes,
        batchn=runtime_args.batchn,
        test_set=test_set,
        knn_set=knn_set,
        num_clients=runtime_args.num_clients,
    )

if __name__ == "__main__":
    saddr = "0.0.0.0:8080"
    args = args_module.parse_and_cache_args()
    client_lr = args.cl_lr
    processes = []

    pool_size = args.num_clients

    create_all_dirs(args.path_results)

    # Dataset
    train_path, num_classes, input_shape = import_dataset(args.dataset, is_train=True,skip_gen_training=args.skip_gen_training,path_to_data=args.dataset_path)
    test_set = import_dataset(args.dataset, is_train=False,skip_gen_training=args.skip_gen_training,path_to_data=args.dataset_path)

    if args.file_name == "":
        file_name = gen_filename(args)
        args.file_name = file_name
    else:
        if args.id_exp != "":
            file_name = "exp_" + args.id_exp + "_" + args.file_name
        else:
            file_name = args.file_name

    if args.alpha_inf:
        alpha = float("inf")
    else:
        alpha = args.alpha

    args_dict = vars(args)

    logger.info(f"Starting experiment - {file_name}")
    logger.debug(dumps(args_dict, indent=2), extra=HFILE)

    server_model, clients_models = do_model_pool(model=args.model,pool_size=pool_size)

    fed_dir, class_dst, _ = do_fl_partitioning(
        train_path,
        pool_size=pool_size,
        alpha=alpha,
        num_classes=num_classes,
        val_ratio=args.val_ratio,
        seed=args.seed,        
        is_cinic=args.dataset == "cinic10"
    )

    pclass = [f"{i} : {x}" for i, x in enumerate(class_dst)]
    pclass = "\n".join(pclass)

    logger.info(f"Class distribution : \n{pclass}")

    protos = list(set(clients_models))
    # configure the strategy

    # Common parameters
    kwargs_dict = {
        "fraction_fit": args.samp_rate,
        "fraction_evaluate": 0.0,
        "min_fit_clients": int(pool_size * args.samp_rate),
        "min_evaluate_clients": pool_size,
        "min_available_clients": pool_size,  # All clients should be available
        "initial_parameters": [],
        "on_fit_config_fn": fit_config,
        "on_evaluate_config_fn": eval_config,
        "evaluate_fn": None,
        "drop_random": args.drop_random,
        "fedbn": args.fedbn,
    }

    aggregate_client_metrics._running_total_flops = 0.0  # type: ignore[attr-defined]
    kwargs_dict["fit_metrics_aggregation_fn"] = aggregate_client_metrics

    clients_per_round = max(kwargs_dict["min_fit_clients"], 0)

    device = torch_device("cuda" if torch.cuda.is_available() else "cpu")
    strategy = None
    lora_config = None

    model_size = -1
    trainable_params = 100
    total_nb_params = -1

    if args.strategy == "fedavg":
        from strategies.fedavg import FedAvg

        server_model = server_model(
            args.feature_maps, input_shape, num_classes, batchn=args.batchn
        )
        model_size = original_msg_size(server_model)
        total_nb_params = model_size//4

        if args.checkpoint:
            try:
                from flwr.common import ndarrays_to_parameters

                params = np.load("checkpoint/server.npy", allow_pickle=True)
                kwargs_dict["initial_parameters"] = ndarrays_to_parameters(params)
            except:
                kwargs_dict["initial_parameters"] = get_tensor_parameters(server_model,args.fedbn)
        else:
            kwargs_dict["initial_parameters"] = get_tensor_parameters(server_model,args.fedbn)

        kwargs_dict["evaluate_fn"] = Evaluate(server_model, test_set, device, args)

        del server_model
        strategy = FedAvg(**kwargs_dict)
    elif args.strategy == "fedlora" or  args.strategy == "fedloha":
        from strategies import FedLora
        from utils.lora import inject_low_rank
        server_model = server_model(
            args.feature_maps, input_shape, num_classes, batchn=args.batchn
        )

        target_modules, modules_to_save,rank_pattern = gen_rank_pattern(server_model,r=args.lora_r,mode=args.lora_ablation_mode,ratio= args.loha_ratio)
        
        lora_config = LoraInfo(alpha=args.lora_alpha,
                               r=args.lora_r,
                               target_modules=target_modules,
                               modules_to_save=modules_to_save,
                               lora_type= args.strategy[3:],
                               rank_pattern=rank_pattern)

        if args.from_pretrained:
            try:
                server_model= load_pretrained(server_model,args.model)
            except:
                pass

        server_model = inject_low_rank(server_model,lora_config)

        _trainable,_total = server_model.get_nb_trainable_parameters()
        total_nb_params = _total
        trainable_params = 100 * _trainable / _total

        if args.apply_quant:
            model_size = quant_msg_size(server_model,bits=args.quant_bits)
        else:
            model_size = original_msg_size(server_model)


        kwargs_dict["initial_parameters"] = get_tensor_parameters(server_model,args.fedbn)

        # evaluate = get_evaluate_fn(server_model, test_set, device)
        # kwargs_dict["evaluate_fn"] = get_evaluate_fn(server_model, test_set, device)
        kwargs_dict["evaluate_fn"] = EvaluateLora(server_model, lora_config, test_set, device, args)

        del server_model
        strategy = FedLora(**kwargs_dict)
    elif args.strategy == "fedprox":
        from strategies import FedProx

        server_model = server_model(
            args.feature_maps, input_shape, num_classes, batchn=args.batchn
        )

        kwargs_dict["initial_parameters"] = get_tensor_parameters(server_model,args.fedbn)
        kwargs_dict["evaluate_fn"] = get_evaluate_fn(server_model, test_set, device, args)
        kwargs_dict.update({"proximal_mu" : args.mu})
        agg_fn = kwargs_dict.pop("fit_metrics_aggregation_fn", None)
        del server_model,kwargs_dict["drop_random"],kwargs_dict["fedbn"]
        strategy = FedProx(**kwargs_dict)
        if agg_fn is not None:
            kwargs_dict["fit_metrics_aggregation_fn"] = agg_fn
    else:
        logger.error(f"Unknown strategy {args.strategy}")
        exit(-1)


    def client_fn(cid):

        from client import FlowerClient
        info = Info(
            model=clients_models[int(cid)],
            dataset_name=args.dataset,
            feature_maps=args.feature_maps,
            input_shape=input_shape,
            num_classes=num_classes,
            batchn=args.batchn,
            fedbn=args.fedbn,
        )

        if args.only_cpu:
            client_device = "cpu"
        else:
            client_device = "cuda"

        fl_info = FlInfo(
            exp_name=file_name,
            saddr=saddr,
            device=client_device,
            num_rounds=args.num_rounds,
            cid=str(cid),
            fed_dir=Path(fed_dir),
            no_thread=args.no_thread,
            server_model=args.model,
            prune=args.prune,
            prune_srv=args.prune_srv,
            strategy=args.strategy,
            lora_config = lora_config,
            nworkers=args.nworkers,
            apply_quant=args.apply_quant,
            quant_bits=args.quant_bits,
        )

        return FlowerClient(info,fl_info).to_client()

    # (optional) specify Ray config
    ray_init_args = {"include_dashboard": False}
    total_cpus = max(1, multiprocessing.cpu_count())
    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if args.sequential_clients:
        # Explicitly constrain each client to the requested fractions so that
        # Ray schedules only one client at a time even when multiple are
        # available. Cap the request to the host's visible resources.
        per_client_cpus = min(5.0, float(total_cpus))
        per_client_gpus = 0.0
        if not args.only_cpu and visible_gpus > 0:
            per_client_gpus = min(0.7, float(visible_gpus))
        ray_init_args.update({
            "num_cpus": per_client_cpus,
            "num_gpus": per_client_gpus,
        })
    else:
        per_client_cpus = max(0.0, float(args.ray_cpu))
        if per_client_cpus == 0.0:
            logger.warning(
                "Per-client CPU allocation was resolved to 0. Ray requires a positive value; "
                "falling back to reserving 1 CPU per client."
            )
            per_client_cpus = 1.0
        per_client_cpus = min(per_client_cpus, float(total_cpus))

        if args.only_cpu or visible_gpus == 0:
            per_client_gpus = 0.0
        else:
            if args.ray_gpu is None:
                per_client_gpus = float(visible_gpus)
            else:
                per_client_gpus = max(0.0, float(args.ray_gpu))
            per_client_gpus = min(per_client_gpus, float(visible_gpus))

    client_resources = {"num_cpus": per_client_cpus, "num_gpus": per_client_gpus}

    max_parallel_by_cpu = (
        total_cpus / per_client_cpus if per_client_cpus > 0 else float("inf")
    )
    max_parallel_by_gpu = (
        visible_gpus / per_client_gpus if per_client_gpus > 0 else float("inf")
    )
    est_parallel_clients = int(math.floor(min(max_parallel_by_cpu, max_parallel_by_gpu)))

    logger.debug(f"Client resources resolved to: {client_resources}")
    logger.info(
        "Sequential client scheduling is %s; Ray will run approximately %s client(s) in parallel.",
        "enabled" if args.sequential_clients else "disabled",
        est_parallel_clients,
    )

    if args.wandb:
        import wandb

        wandb.init(
            entity=args.entity,
            # set the wandb project where this run will be logged
            project=args.wandb_prj_name,
            # track hyperparameters and run metadata
            config=args
        )

        wandb.config["model_size_bytes"] = model_size
        wandb.config["total_nb_params"] = total_nb_params
        wandb.config["trainable"] = trainable_params

    # start simulation
    hist =  fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
    )
    report_metadata = {
        "model_size_bytes": model_size if model_size > 0 else 0.0,
        "clients_per_round": float(clients_per_round),
        "num_rounds": float(args.num_rounds),
    }
    tell_history(
        hist,
        file_name,
        infos=args_dict,
        path=args.path_results,
        report_metadata=report_metadata,
        args=args,
    )

    logger.info("The End")
