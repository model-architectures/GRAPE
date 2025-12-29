from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy and PyTorch RNGs.

    Notes:
    - This does not guarantee bitwise-identical training on GPU by itself. For that,
      set deterministic=True (and be aware of potential perf impact / op limitations).
    """
    random.seed(seed)
    # NumPy expects a 32-bit seed.
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)

    if deterministic:
        # Best-effort flags for deterministic kernels.
        # Some CUDA determinism also requires setting CUBLAS_WORKSPACE_CONFIG
        # before CUDA context initialization; we still set a default here.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

