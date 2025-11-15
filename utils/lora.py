import copy
import math
from collections import OrderedDict
from types import MethodType
from typing import Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import (
    LoHaConfig,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch.nn.init import orthogonal_

from utils.dcs import LoraInfo
from utils.flocora_adapters import apply_flocora_adapters, _parameter_belongs_to_module


def _is_peft_model(model: nn.Module) -> bool:
    return hasattr(model, "peft_config") and hasattr(model, "base_model")


def _iter_adapter_tensors(model: nn.Module):
    modules_to_save = getattr(model, "_flocora_modules_to_save", ())
    if modules_to_save is None:
        modules_to_save = ()
    yielded: Set[str] = set()

    for name, parameter in model.named_parameters():
        if "lora_" in name:
            yielded.add(name)
            yield name, parameter
            continue

        if any(_parameter_belongs_to_module(name, module) for module in modules_to_save):
            yielded.add(name)
            yield name, parameter

    if not modules_to_save:
        return

    for name, buffer in model.named_buffers():
        if name in yielded:
            continue
        if any(_parameter_belongs_to_module(name, module) for module in modules_to_save):
            yield name, buffer


def _ensure_trainable_api(model: nn.Module) -> nn.Module:
    if not hasattr(model, "get_nb_trainable_parameters"):
        def get_nb_trainable_parameters(self):
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            return trainable, total

        model.get_nb_trainable_parameters = MethodType(get_nb_trainable_parameters, model)

    if not hasattr(model, "print_trainable_parameters"):
        def print_trainable_parameters(self):
            trainable, total = self.get_nb_trainable_parameters()
            ratio = (100 * trainable / total) if total else 0.0
            print(
                f"trainable params: {trainable} || total params: {total} || trainable%: {ratio:.2f}"
            )

        model.print_trainable_parameters = MethodType(print_trainable_parameters, model)

    return model

def inject_low_rank(model, lora_config: LoraInfo):
    if lora_config is None:
        return _ensure_trainable_api(model)

    lora_type = getattr(lora_config, "lora_type", "lora") or "lora"

    alpha = getattr(lora_config, "alpha", 1)
    r = getattr(lora_config, "r", 0)
    target_modules = getattr(lora_config, "target_modules", None)
    modules_to_save = getattr(lora_config, "modules_to_save", None)
    rank_pattern = getattr(lora_config, "rank_pattern", None)

    if lora_type == "lora":
        peft_config = LoraConfig(
            lora_alpha=alpha,
            lora_dropout=0.0,
            r=r,
            bias="none",
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            rank_pattern=rank_pattern,
        )
    elif lora_type == "loha":
        peft_config = LoHaConfig(
            use_effective_conv2d=True,
            alpha=alpha,
            r=1,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            rank_pattern=rank_pattern,
        )
    elif lora_type == "flocora":
        wrap_linear = bool(getattr(lora_config, "wrap_linear", False))
        wrapped = apply_flocora_adapters(
            model,
            lora_config,
            wrap_linear=wrap_linear,
        )
        return _ensure_trainable_api(wrapped)
    else:
        return _ensure_trainable_api(model)

    wrapped = get_peft_model(model, peft_config)
    return _ensure_trainable_api(wrapped)

def get_lora_params(lora_model):
    if _is_peft_model(lora_model):
        state_dict = get_peft_model_state_dict(model=lora_model)
        return [val.cpu().numpy() for _, val in state_dict.items()]

    adapter_state = OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in _iter_adapter_tensors(lora_model)
    )
    return [tensor.numpy() for tensor in adapter_state.values()]


def set_lora_params(lora_model, adapter_params):
    if _is_peft_model(lora_model):
        lora_state_dict = get_peft_model_state_dict(lora_model)
        keys = lora_state_dict.keys()
        params_dict = zip(keys, adapter_params)
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
        return set_peft_model_state_dict(lora_model, state_dict)

    adapter_params = list(adapter_params)
    named_tensors = list(_iter_adapter_tensors(lora_model))
    if len(adapter_params) != len(named_tensors):
        raise ValueError("Adapter parameter count mismatch")

    for (name, reference), values in zip(named_tensors, adapter_params):
        tensor = torch.from_numpy(np.copy(values)).to(reference.device, dtype=reference.dtype)
        if isinstance(reference, nn.Parameter):
            reference.data.copy_(tensor)
        else:
            reference.copy_(tensor)
    return lora_model

def singular_value(p):
    sv = math.sqrt(p.shape[0] / p.shape[1])
    if p.dim() == 4:
        sv /= math.sqrt(p.shape[2] * p.shape[3])
    return sv

def toggle_grad(model):
    for p in model.parameters():
        p.requires_grad = not p.requires_grad

def reinit_a_ortho(model):
    toggle_grad(model)
    for n,p in model.named_parameters():
        if "lora" in n:
            if p.dim() == 2: 
                orthogonal_(p)
            if p.dim() == 4:
                for kx in range(p.shape[2]):
                    for ky in range(p.shape[3]):
                        orthogonal_(p[:,:,kx,ky])
            p *= singular_value(p)
    toggle_grad(model)

def calc_conv_from_ratio(out_channels,in_channels,kernel_size,ratio): # Return the low-rank of sub-matrices given the compression ratio 
        #adapted from : https://github.com/South-hw/FedPara_ICLR22
        r1 = int(np.ceil(np.sqrt(out_channels))) 
        r2 = int(np.ceil(np.sqrt(in_channels))) 
        r = np.max((r1, r2)) 
        
        num_target_params = out_channels * in_channels \
            * (kernel_size ** 2) * ratio 
        
        r3 = np.sqrt( ((out_channels + in_channels) ** 2) / (4 *(kernel_size ** 4)) \
                        + num_target_params / (2 * (kernel_size ** 2)) ) \
                        - (out_channels + in_channels) / (2 * (kernel_size ** 2)) 
        
        r3 = int(np.ceil(r3))
        return np.max((r, r3)) 

def gen_rank_pattern(model,r,mode=3,ratio=0):

    if mode == 0 : # mode all lora, all frozen
        target_instances = (torch.nn.Conv2d,torch.nn.Linear)
        save_instances = ()
    elif mode == 1 : # normalization
        target_instances = (torch.nn.Conv2d,torch.nn.Linear)
        save_instances = (torch.nn.BatchNorm2d,torch.nn.GroupNorm)
    elif mode == 2 or mode ==3: # normalization + linear
        target_instances = (torch.nn.Conv2d)
        save_instances = (torch.nn.BatchNorm2d,torch.nn.GroupNorm,torch.nn.Linear)
    else:
        print("Unknown mode, options : (0), (1) and (2)")

    target_modules = []
    modules_to_save = []
    rank_pattern = {}

    for name,m in model.named_modules():

        if isinstance(m,target_instances):
            if isinstance(m, torch.nn.Conv2d) and getattr(m, "groups", 1) > 1:
                # Depthwise or grouped convolutions cannot be merged back from LoRA adapters.
                # Keep them trainable by registering them as saved modules instead of targeting
                # them for LoRA.
                if name not in modules_to_save:
                    modules_to_save.append(name)
                continue

            target_modules.append(name)
            if ratio > 0.0 and isinstance(m,(torch.nn.Conv2d)):
                out_channels = m.out_channels
                in_channels = m.in_channels
                kernel_size = m.kernel_size[0]
                rank_pattern[name] = calc_conv_from_ratio(out_channels,in_channels,kernel_size,ratio=ratio)
            else:
                rank_pattern[name] = r

        if isinstance(m,save_instances):
            modules_to_save.append(name)
    if ratio <= 0.0 and mode == 3: # remove lora from the first layer
        ranks = list(rank_pattern.keys())
        # rank_pattern[ranks[0]] = 4*r
        del rank_pattern[ranks[0]]
        modules_to_save.append(ranks[0])
    return target_modules,modules_to_save,rank_pattern

def extract_AB_matrix(state_dict):
    import re
    a_list= []
    b_list= []
    for k in state_dict.keys():
        if re.search(r"lora_(.*)A",k) is not None:
            a_list.append(copy.deepcopy(state_dict[k]).to("cpu"))
        elif re.search(r"lora_(.*)B",k) is not None:
            b_list.append(copy.deepcopy(state_dict[k]).to("cpu"))


    return a_list,b_list   

if __name__ == "__main__":
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.features_dim = 400
            self.linear1 = nn.Linear(16 * 5 * 5, 120)
            self.linear2 = nn.Linear(120, 84)
            self.linear3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            features = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.linear1(features))
            x = F.relu(self.linear2(x))
            x = self.linear3(x)
            return x,features


    model = DummyModel()
    print(model)

    target_modules = ["conv1","conv2","linear1"]
    modules_to_save = ["linear2","linear3"]
    # model = inject_lora(model,16,1,target_modules=target_modules,modules_to_save=modules_to_save)
    # print(model)

    lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.0,
    r=1,
    bias="none",
    target_modules=target_modules,
    modules_to_save=modules_to_save
    )

    lora_config = LoraInfo(alpha = 16,r=1,target_modules=target_modules,modules_to_save=modules_to_save)

    model = inject_low_rank(model,lora_config)
    list_a = extract_AB_matrix(model)
    params = get_lora_params(model)
    set_lora_params(model,params)
    # model.add_adapter