from __future__ import annotations

# Small, dependency-free helpers for deterministic seeding.
#
# We keep this separate from the training scripts so the "stateless" data RNG
# mode can be shared across entrypoints without copy/paste.

UINT64_MASK = 0xFFFFFFFFFFFFFFFF
INT64_MASK = 0x7FFFFFFFFFFFFFFF  # torch manual_seed expects < 2**63


def _splitmix64(x: int) -> int:
    """Deterministic 64-bit mixing (public-domain SplitMix64)."""
    x = (x + 0x9E3779B97F4A7C15) & UINT64_MASK
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & UINT64_MASK
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & UINT64_MASK
    z = z ^ (z >> 31)
    return z & UINT64_MASK


def fold_in_seed(base_seed: int, *values: int) -> int:
    """
    Combine a base seed with additional integers into a torch-friendly seed.

    Returns an int in [0, 2**63-1] so it can be passed to torch.Generator.manual_seed.
    """
    x = base_seed & UINT64_MASK
    for v in values:
        x = _splitmix64(x + (v & UINT64_MASK))
    return x & INT64_MASK

