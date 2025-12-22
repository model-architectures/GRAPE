from __future__ import annotations

from typing import Any

import torch.nn as nn


def get_optimizer_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    rmsnorm_params = set()
    for module in model.modules():
        if module.__class__.__name__ in {"RMSNorm", "RMSNormLinear"}:
            weight = getattr(module, "weight", None)
            bias = getattr(module, "bias", None)
            if weight is not None:
                rmsnorm_params.add(weight)
            if bias is not None:
                rmsnorm_params.add(bias)

    decay_params = []
    no_decay_params = []
    for param in model.parameters():
        if param in rmsnorm_params:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups: list[dict[str, Any]] = [{"params": decay_params, "weight_decay": weight_decay}]
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return param_groups

