from argparse import Namespace


def _simplify_model_name(model_name: str) -> str:
    normalized = (model_name or "").strip().lower()
    if normalized.startswith("resnet"):
        return "resnet"
    return normalized or "model"


def _clients_per_round_tag(args: Namespace) -> str:
    num_clients = max(int(getattr(args, "num_clients", 0) or 0), 0)
    sample_rate = float(getattr(args, "samp_rate", 0.0) or 0.0)
    clients_per_round = max(int(round(num_clients * sample_rate)), 0)
    return f"{clients_per_round}cl"


def gen_run_name(args: Namespace) -> str:
    dataset = str(getattr(args, "dataset", "dataset") or "dataset").lower()
    model = _simplify_model_name(str(getattr(args, "model", "model") or "model"))
    return f"flocora_{dataset}_{model}_{_clients_per_round_tag(args)}"


def gen_filename(args: Namespace):
    run_name = gen_run_name(args)
    if args.id_exp != "":
        return f"exp_{args.id_exp}_{run_name}"
    return run_name
