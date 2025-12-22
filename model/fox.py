# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import List, Tuple, Optional, Any, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig

# from fla.layers.attn import Attention
"""Local implementations for norms, activations, and rotary.
Removes dependency on FLA modules.
"""

# Local simple fused CE wrapper (compat shim)
class FusedCrossEntropyLoss(nn.Module):
    def __init__(self, inplace_backward: bool = False, ignore_index: int = -1):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(input, target)


def swiglu_linear(gate: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
    z = F.silu(gate) * y
    return F.linear(z.to(weight.dtype), weight, bias)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.max_seq_len_cached: Optional[int] = None
        self.cos_cached: Optional[torch.Tensor] = None
        self.sin_cached: Optional[torch.Tensor] = None

    def _build_cache(self, seqlen: int, device, dtype):
        if (
            self.max_seq_len_cached is None or
            seqlen > self.max_seq_len_cached or
            self.cos_cached is None or
            self.cos_cached.device != device
        ):
            t = torch.arange(seqlen, device=device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().to(dtype)
            self.sin_cached = freqs.sin().to(dtype)
            self.max_seq_len_cached = seqlen

    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # x: [B, T, H, D]
        D = x.shape[-1]
        half = D // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], dim=-1).type_as(x)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seqlen_offset: Union[int, torch.Tensor] = 0, max_seqlen: Optional[int] = None):
        B, T, H, D = q.shape
        device = q.device
        dtype = q.dtype
        total_len = max_seqlen if max_seqlen is not None else T if isinstance(seqlen_offset, int) else (T + int(torch.as_tensor(seqlen_offset).max()))
        total_len = int(total_len)
        self._build_cache(total_len, device, dtype)

        if torch.is_tensor(seqlen_offset):
            pos = seqlen_offset.to(torch.long).unsqueeze(1) + torch.arange(T, device=device).unsqueeze(0)  # [B, T]
            cos = self.cos_cached.index_select(0, pos.view(-1)).view(B, T, -1)
            sin = self.sin_cached.index_select(0, pos.view(-1)).view(B, T, -1)
            cos = cos[:, :, None, :]
            sin = sin[:, :, None, :]
        else:
            start = int(seqlen_offset)
            cos = self.cos_cached[start:start+T][None, :, None, :]  # [1, T, 1, D/2]
            sin = self.sin_cached[start:start+T][None, :, None, :]

        q = self._apply_rotary(q, cos, sin)
        k = self._apply_rotary(k, cos, sin)
        return q, k

from einops import rearrange

import triton
import triton.language as tl

from forgetting_transformer import forgetting_attention

from functools import partial

logger = logging.get_logger(__name__)



def maybe_contiguous(x):
    # only when the inner most dimension is contiguous can LDGSTS be used
    # so inner-dimension contiguity is enforced.
    return x.contiguous() if x.stride(-1) != 1 else x

from .rmsnorm import RMSNorm, GroupNorm

@triton.jit
def shift_fwd_kernel(
    X_PTR,
    PREV_WEIGHT_PTR,
    CURR_WEIGHT_PTR,
    OUT_PTR,

    stride_x_b, stride_x_t, stride_x_h, stride_x_d,
    stride_weight_b, stride_weight_t, stride_weight_h,
    T: tl.constexpr, D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """
        everything is (B, T, D)
    """
    b_offset = tl.program_id(axis=0).to(tl.int64)
    t_offset = tl.program_id(axis=1).to(tl.int64) * BLOCK_T
    h_offset = tl.program_id(axis=2).to(tl.int64)


    x_ptr_offset = b_offset * stride_x_b + t_offset * stride_x_t + h_offset * stride_x_h
    X_PTR += x_ptr_offset
    OUT_PTR += x_ptr_offset

    weight_ptr_offset = b_offset * stride_weight_b + t_offset * stride_weight_t + h_offset * stride_weight_h
    CURR_WEIGHT_PTR += weight_ptr_offset
    PREV_WEIGHT_PTR += weight_ptr_offset

    x_ptr = X_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_x_t + tl.arange(0, D)[None, :] * stride_x_d
    t_offset_block = t_offset + tl.arange(0, BLOCK_T)[:, None]
    x_mask = t_offset_block < T

    # Yeah this is correct
    x_prev_ptr = x_ptr - stride_x_t
    t_prev_offset_block = t_offset_block - 1
    x_prev_mask = ((t_prev_offset_block) < T) & (t_prev_offset_block >= 0)

    curr_weight_ptr = CURR_WEIGHT_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_weight_t
    prev_weight_ptr = PREV_WEIGHT_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_weight_t


    x = tl.load(x_ptr, mask=x_mask, other=0.0)
    x_prev = tl.load(x_prev_ptr, mask=x_prev_mask, other=0.0)
    curr_weight = tl.load(curr_weight_ptr, mask=x_mask, other=0.0)
    prev_weight = tl.load(prev_weight_ptr, mask=x_mask, other=0.0)

    result = x * curr_weight.to(tl.float32) + x_prev * prev_weight.to(tl.float32)
    result = result.to(x.dtype)

    out_ptr = OUT_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_x_t + tl.arange(0, D)[None, :] * stride_x_d
    tl.store(out_ptr, result, mask=x_mask)


@triton.jit
def shift_bwd_kernel(
    X_PTR,
    PREV_WEIGHT_PTR,
    CURR_WEIGHT_PTR,

    DOUT_PTR,
    DX_PTR,
    DPREV_WEIGHT_PTR,
    DCURR_WEIGHT_PTR,

    stride_x_b, stride_x_t, stride_x_h, stride_x_d,
    stride_weight_b, stride_weight_t, stride_weight_h,
    T: tl.constexpr, D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """
        everything is (B, T, D)
    """
    b_offset = tl.program_id(axis=0).to(tl.int64)
    t_offset = tl.program_id(axis=1).to(tl.int64) * BLOCK_T
    h_offset = tl.program_id(axis=2).to(tl.int64)


    x_ptr_offset = b_offset * stride_x_b + t_offset * stride_x_t + h_offset * stride_x_h
    X_PTR += x_ptr_offset
    DX_PTR += x_ptr_offset
    DOUT_PTR += x_ptr_offset

    weight_ptr_offset = b_offset * stride_weight_b + t_offset * stride_weight_t + h_offset * stride_weight_h
    CURR_WEIGHT_PTR += weight_ptr_offset
    PREV_WEIGHT_PTR += weight_ptr_offset
    DCURR_WEIGHT_PTR += weight_ptr_offset
    DPREV_WEIGHT_PTR += weight_ptr_offset

    x_ptr = X_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_x_t + tl.arange(0, D)[None, :] * stride_x_d
    t_offset_block = t_offset + tl.arange(0, BLOCK_T)[:, None]
    x_mask = t_offset_block < T

    dout_ptr = DOUT_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_x_t + tl.arange(0, D)[None, :] * stride_x_d

    # Yeah this is correct
    dout_next_ptr = dout_ptr + stride_x_t
    t_next_offset_block = t_offset_block + 1
    x_next_mask = (t_next_offset_block) < T


    # Yeah this is correct
    x_prev_ptr = x_ptr - stride_x_t
    t_prev_offset_block = t_offset_block - 1
    x_prev_mask = ((t_prev_offset_block) < T) & (t_prev_offset_block >= 0)

    curr_weight_ptr = CURR_WEIGHT_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_weight_t
    prev_weight_ptr = PREV_WEIGHT_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_weight_t
    next_prev_weight_ptr = prev_weight_ptr + stride_weight_t


    x = tl.load(x_ptr, mask=x_mask, other=0.0)
    x_prev = tl.load(x_prev_ptr, mask=x_prev_mask, other=0.0)
    dout = tl.load(dout_ptr, mask=x_mask, other=0.0)
    dout_next= tl.load(dout_next_ptr, mask=x_next_mask, other=0.0)

    curr_weight = tl.load(curr_weight_ptr, mask=x_mask, other=0.0)
    next_prev_weight = tl.load(next_prev_weight_ptr, mask=x_next_mask, other=0.0)

    dx =  dout * curr_weight.to(tl.float32) + dout_next * next_prev_weight.to(tl.float32)
    dx = dx.to(x.dtype)

    dcurr_weight = tl.sum(dout.to(tl.float32) * x, axis=1, keep_dims=True)
    dprev_weight = tl.sum(dout.to(tl.float32) * x_prev, axis=1, keep_dims=True)

    dx_ptr = DX_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_x_t + tl.arange(0, D)[None, :] * stride_x_d
    tl.store(dx_ptr, dx, mask=x_mask)
    dcurr_weight_ptr = DCURR_WEIGHT_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_weight_t
    tl.store(dcurr_weight_ptr, dcurr_weight, mask=x_mask)
    dprev_weight_ptr = DPREV_WEIGHT_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_weight_t
    tl.store(dprev_weight_ptr, dprev_weight, mask=x_mask)



class TokenShift(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, prev_weight: torch.Tensor, curr_weight: torch.Tensor):

        B, T, H, D = x.size()
        assert D in {16, 32, 64, 128}
        assert prev_weight.size() == curr_weight.size() == (B, T, H)
        assert prev_weight.stride() == curr_weight.stride()
        x = maybe_contiguous(x)
        out = torch.empty_like(x)

        BLOCK_T = triton.next_power_of_2(min(64, T))

        grid = lambda meta: (B, triton.cdiv(T, meta["BLOCK_T"]), H)
        # NOTE:
        #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
        #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
        #  - Don't forget to pass meta-parameters as keywords arguments.
        shift_fwd_kernel[grid](
            x,
            prev_weight,
            curr_weight,
            out,
            *x.stride(),
            *curr_weight.stride(),
            T=T, D=D,
            BLOCK_T=BLOCK_T,
        )
        ctx.save_for_backward(x, prev_weight, curr_weight)
        # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
        # running asynchronously at this point.
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):

        x, prev_weight, curr_weight = ctx.saved_tensors
        B, T, H, D = x.size()
        assert D in {16, 32, 64, 128}
        assert prev_weight.size() == curr_weight.size() == (B, T, H)
        assert prev_weight.stride() == curr_weight.stride()
        x = maybe_contiguous(x)
        assert dout.stride() == x.stride()
        dx = torch.empty_like(x)
        dcurr_weight = torch.empty_like(curr_weight)
        dprev_weight = torch.empty_like(prev_weight)

        BLOCK_T = triton.next_power_of_2(min(64, T))

        grid = lambda meta: (B, triton.cdiv(T, meta["BLOCK_T"]), H)
        # NOTE:
        #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
        #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
        #  - Don't forget to pass meta-parameters as keywords arguments.
        shift_bwd_kernel[grid](
            x,
            prev_weight,
            curr_weight,
            dout,
            dx,
            dprev_weight,
            dcurr_weight,
            *x.stride(),
            *curr_weight.stride(),
            T=T,
            D=D,
            BLOCK_T=BLOCK_T,
        )
        # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
        # running asynchronously at this point.
        return dx, dprev_weight, dcurr_weight

def token_shift(x, prev_weight, curr_weight):
    return TokenShift.apply(x, prev_weight, curr_weight)


def glu_linear(gate: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
    z = torch.sigmoid(gate) * y
    return F.linear(z.to(weight.dtype), weight, bias)

class FgateDynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = DynamicCache()
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        DynamicCache()
        ```
    """

    def __init__(self) -> None:
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.log_fgate_cache: List[torch.Tensor] = []

        self.key_shift_cache: List[Optional[torch.Tensor]] = []
        self.value_shift_cache: List[Optional[torch.Tensor]] = []

        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    def update_shift_cache(
        self,
        key_shift_state: Optional[torch.Tensor],
        value_shift_state: Optional[torch.Tensor],
        layer_idx: int,
    ):
        assert layer_idx == len(self.key_shift_cache) == len(self.value_shift_cache)
        self.key_shift_cache.append(key_shift_state)
        self.value_shift_cache.append(value_shift_state)


    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx], self.log_fgate_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx], self.log_fgate_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        log_fgate_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        assert log_fgate_states.ndim == 3, f"log_fgate must be (B, H, T), but get {log_fgate_states.size()}"
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.log_fgate_cache.append(log_fgate_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            self.log_fgate_cache[layer_idx] = torch.cat([self.log_fgate_cache[layer_idx], log_fgate_states], dim=-1)

        return self.key_cache[layer_idx], self.value_cache[layer_idx], self.log_fgate_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx], self.log_fgate_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor], ...]] = None,
        num_layers: Optional[int] = None,
    ) -> "FgateDynamicCache":
        """Convert legacy tuple cache into `FgateDynamicCache` (keys, values, log_fgates)."""
        cache = cls()

        if past_key_values is None:
            if num_layers is not None:
                cache.key_shift_cache = [None] * num_layers
                cache.value_shift_cache = [None] * num_layers
            return cache

        for layer_idx, (key_states, value_states, log_fgate_states) in enumerate(past_key_values):
            cache.update(key_states, value_states, log_fgate_states, layer_idx)

            # Initialize shift states (used only if k/v shift is enabled).
            batch_size = key_states.shape[0]
            kv_dim = int(key_states.shape[1]) * int(key_states.shape[-1])
            cache.update_shift_cache(
                key_shift_state=key_states.new_zeros(batch_size, kv_dim),
                value_shift_state=value_states.new_zeros(batch_size, kv_dim),
                layer_idx=layer_idx,
            )

        if num_layers is not None and num_layers > len(cache.key_shift_cache):
            extra = num_layers - len(cache.key_shift_cache)
            cache.key_shift_cache.extend([None] * extra)
            cache.value_shift_cache.extend([None] * extra)

        return cache

    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search."""
        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
            self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]
            self.log_fgate_cache[idx] = self.log_fgate_cache[idx][..., :max_length]

    def batch_split(self, full_batch_size: int, split_size: int) -> List["FgateDynamicCache"]:
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = self.__class__()
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
            current_split.value_cache = [tensor[i : i + split_size] for tensor in self.value_cache]
            current_split.log_fgate_cache = [tensor[i : i + split_size] for tensor in self.log_fgate_cache]
            current_split.key_shift_cache = [
                None if tensor is None else tensor[i : i + split_size] for tensor in self.key_shift_cache
            ]
            current_split.value_shift_cache = [
                None if tensor is None else tensor[i : i + split_size] for tensor in self.value_shift_cache
            ]
            out.append(current_split)
        return out

    @classmethod
    def from_batch_splits(cls, splits: List["FgateDynamicCache"]) -> "FgateDynamicCache":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        cache = cls()
        if not splits:
            return cache

        cache._seen_tokens = splits[0]._seen_tokens

        num_layers_kv = len(splits[0].key_cache)
        cache.key_cache = [torch.cat([current.key_cache[idx] for current in splits], dim=0) for idx in range(num_layers_kv)]
        cache.value_cache = [torch.cat([current.value_cache[idx] for current in splits], dim=0) for idx in range(num_layers_kv)]
        cache.log_fgate_cache = [
            torch.cat([current.log_fgate_cache[idx] for current in splits], dim=0) for idx in range(num_layers_kv)
        ]

        num_layers_shift = len(splits[0].key_shift_cache)
        cache.key_shift_cache = []
        cache.value_shift_cache = []
        for layer_idx in range(num_layers_shift):
            key_shifts = [current.key_shift_cache[layer_idx] for current in splits]
            value_shifts = [current.value_shift_cache[layer_idx] for current in splits]

            cache.key_shift_cache.append(None if any(t is None for t in key_shifts) else torch.cat(key_shifts, dim=0))
            cache.value_shift_cache.append(
                None if any(t is None for t in value_shifts) else torch.cat(value_shifts, dim=0)
            )
        return cache

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.log_fgate_cache[layer_idx] = self.log_fgate_cache[layer_idx].repeat_interleave(repeats, dim=0)
        for layer_idx in range(len(self.key_shift_cache)):
            if self.key_shift_cache[layer_idx] is not None:
                self.key_shift_cache[layer_idx] = self.key_shift_cache[layer_idx].repeat_interleave(repeats, dim=0)
            if self.value_shift_cache[layer_idx] is not None:
                self.value_shift_cache[layer_idx] = self.value_shift_cache[layer_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]
            self.log_fgate_cache[layer_idx] = self.log_fgate_cache[layer_idx][indices, ...]
        for layer_idx in range(len(self.key_shift_cache)):
            if self.key_shift_cache[layer_idx] is not None:
                self.key_shift_cache[layer_idx] = self.key_shift_cache[layer_idx][indices, ...]
            if self.value_shift_cache[layer_idx] is not None:
                self.value_shift_cache[layer_idx] = self.value_shift_cache[layer_idx][indices, ...]
            

class GPTConfig(PretrainedConfig):

    model_type = 'forgetting_transformer-project_fox'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        # Core model dims (match llama-mha-alibi API)
        vocab_size: int = 50304,
        n_layer: int = 24,
        n_head: int = 32,
        n_embd: int = 2048,
        head_dim: Optional[int] = None,
        block_size: int = 2048,
        # Common flags
	        bias: bool = False,
	        dropout: float = 0.0,
	        using_groupnorm: bool = False,
	        use_qk_rmsnorm: bool = True,
        # Init controls (match llama-mha-alibi)
        embedding_init_std: float = 0.02,
        hidden_init_std_factor: float = 0.5,
        # Legacy/fox-specific knobs retained
        hidden_ratio: Optional[float] = 4,
        intermediate_size: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        hidden_act: str = "swish",
        window_size: Optional[int] = None,
        initializer_range: float = 0.02,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        attention_bias: bool = False,
        fuse_norm: bool = True,
        fuse_cross_entropy: bool = True,
        # RoPE (kept for compatibility though fox does not use by default)
        rope_base: float = 10000.0,
        use_rope: bool = False,
        # Output gate (legacy)
        use_output_gate: bool = False,
        ogate_act: str = "sigmoid",
        # Forget gate (legacy)
        fgate_type: str = "full",
        fgate_bias_init: bool = False,
        decay_time_min: Optional[float] = None,
        decay_time_max: Optional[float] = None,
        # Fox encoding toggles
        use_k_shift: bool = False,
        use_v_shift: bool = False,
        # Misc (passed but unused; compat with training script)
        scale_attn_by_inverse_layer_idx: bool = False,
        **kwargs,
    ):
        # Primary dims and counts
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.hidden_size = n_embd
        # Derive head_dim if not given
        self.head_dim = head_dim if head_dim is not None else (n_embd // n_head)
        self.block_size = block_size
        self.num_hidden_layers = n_layer
        self.num_heads = n_head
        self.num_kv_heads = num_kv_heads
        # Training-time misc
        self.bias = bias
        self.dropout = dropout
        self.using_groupnorm = using_groupnorm
        self.use_qk_rmsnorm = use_qk_rmsnorm
        # Init controls
        self.embedding_init_std = embedding_init_std
        self.hidden_init_std_factor = hidden_init_std_factor
        # Legacy/fox params
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.window_size = window_size
        self.max_position_embeddings = block_size
        self.initializer_range = initializer_range
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_norm = fuse_norm
        self.rope_base = rope_base
        self.use_rope = use_rope
        self.use_output_gate = use_output_gate
        self.ogate_act = ogate_act
        self.fgate_type = fgate_type
        self.fgate_bias_init = fgate_bias_init
        self.decay_time_min = decay_time_min
        self.decay_time_max = decay_time_max
        # Map alibi-style flag to fox output norm behavior
        self.use_output_norm = using_groupnorm
        # QK RMSNorm behavior
        self.qk_norm = use_qk_rmsnorm
        # Use per-head RMSNorm params (align with alibi QK RMSNorm semantics)
        self.qk_norm_share_param_across_head = True
        # Fox encoding
        self.use_k_shift = use_k_shift
        self.use_v_shift = use_v_shift
        # Keep model type label
        self.model_type = "fox"

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

class ShiftLinear(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        bias: bool,
        shift_bias: bool = False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        assert self.output_dim % self.num_heads == 0

        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.shift_proj = nn.Linear(input_dim, num_heads, bias=shift_bias)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.input_dim}, {self.output_dim})"
        return s

    def forward(self, x: torch.Tensor, shift_state: Optional[torch.Tensor]) -> torch.Tensor:
        assert x.ndim == 3, "Input must be (B, T, D)"
        _, seq_len, _ = x.size()
        out = self.linear(x)
        # (B, T, H, 1)
        alpha = torch.sigmoid(self.shift_proj(x).float())
        # left, right, top, bottom (B, T=H, D=W)
        # out_prev = nn.functional.pad(out, (0, 0, 1, -1))
        # out_prev = torch.roll(out, shifts=1, dims=1)

        out_per_head = rearrange(out, 'b t (h d) -> b t h d', h=self.num_heads)
        if seq_len > 1:
            # TODO: note in this case cache is not used
            result_per_head = token_shift(out_per_head, alpha, 1.0 - alpha)
        else:
            if shift_state is None:
                result_per_head = out_per_head
            else:
                shift_state_per_head = rearrange(shift_state, 'b (h d) -> b 1 h d', h=self.num_heads)
                result_per_head = (alpha[..., None] * shift_state_per_head + (1 - alpha[..., None]) * out_per_head)

        result_per_head = result_per_head.to(out.dtype)

        if shift_state is not None:
            shift_state.copy_(out[:, -1, :])

        result = rearrange(result_per_head, 'b t h d -> b t (h d)', h=self.num_heads)
        return result

class ForgettingAttentionLayer(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        window_size: Optional[int] = None,
        max_position_embeddings: Optional[int] = None,
        use_rope: bool = False,
        rope_base: float = 10000.0,
        use_output_gate: bool = False,
        ogate_act: str = "sigmoid",
        fgate_type: str = "full",
        fgate_bias_init: bool = False,
        decay_time_min: Optional[float] = None,
        decay_time_max: Optional[float] = None,
        use_output_norm: bool = False,
        norm_eps: float = 1e-5,
        qk_norm: bool = False,
        qk_norm_share_param_across_head: bool = False,
        use_k_shift: bool = False,
        use_v_shift: bool = False,
	        initializer_range: float = 0.02,
	        layer_idx: int = None,
	    ):
        """
        Forgetting Attention layer.

        Arguments:
            - hidden_size: Input dimension and qkv dimension
            - num_heads: Number of heads
            - num_kv_heads: Not used. Should be None 
            - window_size: Not used. Should be None
            - max_position_embeddings: Not used. Should be None
            - use_rope: Whether to use RoPE. Default is False
            - rope_base: the theta hyperparameter in RoPE. This has no effect if
                  use_rope=False
            - use_output_gate: Whether to use output gates. Note that using output gates
                  introduces extra parameters and you may want to reduce parameters from
                  other components (e.g., MLPs)
            - ogate_act: Activation for the output gate. Either "sigmoid" or "silu"
            - fgate_type: Forget gate type. The following are supported:
                - "full": The default data-dependent forget gate
                - "bias_only": The data-independent forget gate
                - "fixed": Forget gates with fixed values
                - "none": Not using forget gates. Equivalent to forget gates with all
                  ones.
            - fgate_bias_init: Whether to use special initalization for the bias terms in 
                  the forget gate. This should only be used with fgate types in 
                  ["bias_only", "fixed"].
            - decay_time_min: T_min for the forget gate bias initialization. See paper
                  for details.
            - decay_time_max: T_max for the forget gate bias initalization. See paper
                  for details.
            - use_output_norm: Whether to use output normalization.
            - norm_eps: Epsilon for the RMSNorms
            - qk_norm: Whether to use qk_norm
            - qk_norm_share_param_across_head: In QK-norm, whether to share the RMSNorm
                scaling parameters across heads. This is just for backward compatibility.
            - use_k_shift: Whether to use data-dependent key shift
            - use_v_shift: Whether to use data-dependent value shift
            - initializer_range: standard deviation for initialization
            - layer_idx: The block index of this layer. Needed for KV-cache
        """
        super().__init__()

        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            raise NotImplementedError("GQA has not been tested.")
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.window_size = window_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        if use_k_shift:
            self.k_proj = ShiftLinear(self.hidden_size, self.kv_dim, self.num_heads, bias=False)
        else:
            self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)

        if use_v_shift:
            self.v_proj = ShiftLinear(self.hidden_size, self.kv_dim, self.num_heads, bias=False)
        else:
            self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)

        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.use_k_shift = use_k_shift
        self.use_v_shift = use_v_shift


        device = next(self.parameters()).device
        # Forget gate
        assert fgate_type in ["full", "bias_only", "fixed", "none"]
        self.fgate_type = fgate_type
        self.fgate_bias_init = fgate_bias_init
        if fgate_type == "full":
            assert not fgate_bias_init
            self.fgate_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        elif fgate_type == "bias_only":
            self.fgate_bias = nn.Parameter(torch.zeros(size=(self.num_heads,), device=device))
            self.fgate_bias._no_weight_decay = True
        elif fgate_type == "fixed":
            assert fgate_bias_init, "You must set fgate_bias_init = True with fixed fgate"
            fgate_bias = torch.zeros(size=(self.num_heads,), device=device)
            self.register_buffer("fgate_bias", fgate_bias)
        elif fgate_type == "none":
            pass
        else:
            raise ValueError(f"Unknown fgate type {fgate_type}")

                

        # Forget gate intialization for data-independent and fixed forget gates
        if fgate_bias_init:
            assert decay_time_min is not None and decay_time_max is not None
            assert decay_time_min > 0 and decay_time_max > 0
            with torch.no_grad():
                log_decay_time = torch.linspace(math.log(decay_time_min), math.log(decay_time_max), steps=self.num_heads)
                decay_time = torch.exp(log_decay_time)
                # Such that t = -1 / logsigmoid(b)
                bias_init = -torch.log(torch.expm1(1 / decay_time))
                self.fgate_bias.copy_(bias_init)
        else:
            assert decay_time_min is None and decay_time_max is None

        if use_output_gate:
            self.ogate_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.ogate_act = ogate_act
            assert ogate_act in ["silu", "sigmoid"]
        else:
            self.ogate_proj = None

        if use_output_norm:
            self.output_norm = RMSNorm(self.head_dim, eps=norm_eps, elementwise_affine=True)
        else:
            self.output_norm = None


        if use_rope:
            self.rotary = RotaryEmbedding(self.head_dim, base=rope_base)
        else:
            self.rotary = None

        self.qk_norm = qk_norm
        self.qk_norm_share_param_across_head = qk_norm_share_param_across_head
        if qk_norm:
            if self.qk_norm_share_param_across_head:
                # Learnable RMSNorm on each head channel, shared across heads
                self.q_norm = RMSNorm(self.head_dim, eps=norm_eps)
                self.k_norm = RMSNorm(self.head_dim, eps=norm_eps)
            else:
                self.q_norm = GroupNorm(num_groups=self.num_heads, hidden_size=self.hidden_size, eps=norm_eps, is_rms_norm=True)
                self.k_norm = GroupNorm(num_groups=self.num_heads, hidden_size=self.hidden_size, eps=norm_eps, is_rms_norm=True)

        self.initializer_range = initializer_range
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        # This will actually be overwritten by outer init.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        We assume that during decoding attention mask is always 1. Otherwise it won't work.
        """
        batch_size, q_len, _ = hidden_states.size()
        if use_cache:
            key_shift_state = past_key_values.key_shift_cache[self.layer_idx]
            value_shift_state = past_key_values.value_shift_cache[self.layer_idx]
        else:
            key_shift_state = value_shift_state = None

        # Shift states are updated in place
        q = self.q_proj(hidden_states)
        if self.use_k_shift:
            k = self.k_proj(hidden_states, key_shift_state)
        else:
            k = self.k_proj(hidden_states)
        if self.use_v_shift:
            v = self.v_proj(hidden_states, value_shift_state)
        else:
            v = self.v_proj(hidden_states)

        if self.qk_norm and (not self.qk_norm_share_param_across_head):
            q = self.q_norm(q).to(q.dtype)
            k = self.k_norm(k).to(k.dtype)

        q = rearrange(q, '... (h d) -> ... h d', h=self.num_heads)
        k = rearrange(k, '... (h d) -> ... h d', h=self.num_kv_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_kv_heads)


        if self.qk_norm and (self.qk_norm_share_param_across_head):
            q = self.q_norm(q).to(q.dtype)
            k = self.k_norm(k).to(k.dtype)


        seqlen_offset, max_seqlen = 0, q.shape[1]
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
                seqlen_offset = (seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1])
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        if self.rotary is not None:
            q, k = self.rotary(q, k, seqlen_offset, max_seqlen)

        if self.fgate_type == "full":
            fgate_logit = self.fgate_proj(hidden_states)
            fgate_logit = rearrange(fgate_logit, "b t h -> b h t")
            log_fgate = torch.nn.functional.logsigmoid(fgate_logit.float())
        elif self.fgate_type == "none":
            log_fgate = torch.zeros((batch_size, self.num_heads, q_len), dtype=torch.float32, device=hidden_states.device)
        else:
            assert self.fgate_type in ["fixed", "bias_only"]
            fgate_logit = torch.broadcast_to(self.fgate_bias, (batch_size, q_len, self.num_heads))
            fgate_logit = rearrange(fgate_logit, "b t h -> b h t")
            log_fgate = torch.nn.functional.logsigmoid(fgate_logit.float())

        k = rearrange(k, 'b t h d -> b h t d')
        if past_key_values is not None:
            k, v, log_fgate = past_key_values.update(k, v, log_fgate, self.layer_idx)
        # k, v = rearrange(k, 'b h t d -> b t h d'), rearrange(v, 'b h t d -> b t h d')
        q = rearrange(q, 'b t h d -> b h t d')

        if self.num_kv_groups > 1:
            assert False
            k = rearrange(k.unsqueeze(-2).repeat(1, 1, 1, self.num_kv_groups, 1), 'b t h g d -> b t (h g) d')
            v = rearrange(v.unsqueeze(-2).repeat(1, 1, 1, self.num_kv_groups, 1), 'b t h g d -> b t (h g) d')

        # Fast path: when no forgetting (fgate all ones) and no padding mask, this reduces to standard causal SDPA
        if self.fgate_type == "none" and attention_mask is None:
            o = F.scaled_dot_product_attention(
                q,  # (B, H, T_q, D)
                k,  # (B, H, T_k, D)
                v,  # (B, H, T_k, D)
                is_causal=True,
            )
            o = rearrange(o, "b h t d -> b t h d")
        # Contains at least one padding token in the sequence
        elif attention_mask is not None:
            B, _, T = log_fgate.size()
            assert attention_mask.size() == (B, T), ((B, T), attention_mask.size())
            seq_start = T - attention_mask.sum(dim=-1)
            o = forgetting_attention(
                q, k, v,
                log_fgate,
                head_first=True,
                seq_start=seq_start,
                sm_scale=1 / math.sqrt(self.head_dim),
            )
            o = rearrange(o, "b h t d -> b t h d")
        else:
            o = forgetting_attention(
                q, k, v,
                log_fgate,
                head_first=True,
                sm_scale=1 / math.sqrt(self.head_dim),
            )
            o = rearrange(o, "b h t d -> b t h d")

        if self.output_norm is not None:
            o = self.output_norm(o)

        o = o.reshape(batch_size, q_len, self.hidden_size)

        if self.ogate_proj is not None:
            # ogate = self.ogate act(self.ogate_proj(hidden_states))
            # o = o * ogate
            # ogate = act_gate(self.ogate_proj(hidden_states), o)
            ogate_logit = self.ogate_proj(hidden_states)
            dtype = ogate_logit.dtype
            if self.ogate_act == "silu":
                o = swiglu_linear(
                    ogate_logit,
                    o,
                    self.c_proj.weight.to(dtype),
                    self.c_proj.bias.to(dtype) if self.c_proj.bias is not None else self.c_proj.bias,
                )
            elif self.ogate_act == "sigmoid":
                o = glu_linear(
                    ogate_logit,
                    o,
                    self.c_proj.weight.to(dtype),
                    self.c_proj.bias.to(dtype) if self.c_proj.bias is not None else self.c_proj.bias,
                )
            else:
                raise ValueError(f"Unknown ogate act {self.ogate_act}")
        else:
            o = self.c_proj(o)

        if not output_attentions:
            attentions = None
        else:
            SAVE_HEADS = [0, 1, 2, 3]
            # (B, H, T, T)
            score = q[:, SAVE_HEADS] @ k[:, SAVE_HEADS].mT
            log_lambda = torch.cumsum(log_fgate, dim=-1)
            decay_bias = (log_lambda[:, SAVE_HEADS, :, None] - log_lambda[:, SAVE_HEADS, None, :]).to(torch.bfloat16)
            # normalized_score = torch.softmax(score, dim=-1)
            attentions = (score, decay_bias)

        return o, attentions, past_key_values

    def init_shift_state(self, batch_size: int):
        param = next(self.parameters())
        state = dict()
        try:
            dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else torch.float32
        except TypeError:
            # Support legacy torch version
            dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else torch.float32
        if self.use_k_shift:
            state['key_shift'] = param.new_zeros(batch_size, self.kv_dim, dtype=dtype)
        else:
            state['key_shift'] = None
        if self.use_v_shift:
            state['value_shift'] = param.new_zeros(batch_size, self.kv_dim, dtype=dtype)
        else:
            state['value_shift'] = None
        return state


class ForgettingTransformerMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[float] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish'
    ) -> ForgettingTransformerMLP:
        super().__init__()

        self.hidden_size = hidden_size
        # the final number of params is `hidden_ratio * hidden_size^2`
        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]
        self.hidden_act = hidden_act
        assert hidden_act in ["swish", "sigmoid"]

    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        # TODO: maybe wrap swiglu_linear in custom_fwd/custom_bwd
        if self.hidden_act == "swish":
            return swiglu_linear(
                gate, y,
                self.down_proj.weight.to(y.dtype),
                self.down_proj.bias.to(y.dtype) if self.down_proj.bias is not None else self.down_proj.bias
            )
        elif self.hidden_act == "sigmoid":
            return glu_linear(
                gate, y,
                self.down_proj.weight.to(y.dtype),
                self.down_proj.bias.to(y.dtype) if self.down_proj.bias is not None else self.down_proj.bias
            )
        else:
            raise ValueError()


class ForgettingTransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = ForgettingAttentionLayer(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            window_size=config.window_size,
            max_position_embeddings=config.max_position_embeddings,
            rope_base=config.rope_base,
            use_rope=config.use_rope,
            use_output_gate=config.use_output_gate,
            ogate_act=config.ogate_act,
            fgate_type=config.fgate_type,
            fgate_bias_init=config.fgate_bias_init,
            decay_time_min=config.decay_time_min,
            decay_time_max=config.decay_time_max,
            use_output_norm = config.use_output_norm,
	            norm_eps=config.norm_eps,
	            qk_norm=config.qk_norm,
	            qk_norm_share_param_across_head=config.qk_norm_share_param_across_head,
	            use_k_shift=config.use_k_shift,
	            use_v_shift=config.use_v_shift,
	            initializer_range=config.initializer_range,
	            layer_idx=layer_idx,
	        )
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = ForgettingTransformerMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act
        )

    def forward_attn(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        # residual handled outside of this
        # residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
        return hidden_states, attentions, past_key_values

    def forward_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ):
        hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        gradient_checkpointing: bool = False
        # **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states


        if gradient_checkpointing:
            forward_attn = partial(torch.utils.checkpoint.checkpoint, self.forward_attn, use_reentrant=False)
            forward_mlp = partial(torch.utils.checkpoint.checkpoint, self.forward_mlp, use_reentrant=False)
        else:
            forward_attn = self.forward_attn
            forward_mlp = self.forward_mlp

        hidden_states, attentions, past_key_values = forward_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions
        )

        hidden_states = forward_mlp(
            hidden_states,
            residual,
        )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attentions,)

        if use_cache:
            outputs += (past_key_values,)

        return outputs



class ForgettingTransformerPreTrainedModel(PreTrainedModel):

    config_class = GPTConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ['ForgettingTransformerBlock']

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
    ):
        # if isinstance(module, (nn.Linear, nn.Conv1d)):
        if isinstance(module, (nn.Linear)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class ForgettingTransformerModel(ForgettingTransformerPreTrainedModel):

    def __init__(self, config: GPTConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([ForgettingTransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache:
            # use_legacy_cache = not isinstance(past_key_values, Cache)
            # if use_legacy_cache:
                # past_key_values = FgateDynamicCache.from_legacy_cache(past_key_values)
            if past_key_values is None:
                past_key_values = FgateDynamicCache()
                for layer_idx, layer in enumerate(self.layers):
                    shift_state = layer.attn.init_shift_state(
                        batch_size=input_ids.size(0),
                    )
                    past_key_values.update_shift_cache(
                        key_shift_state=shift_state["key_shift"],
                        value_shift_state=shift_state["value_shift"],
                        layer_idx=layer_idx
                    )
            else:
                assert isinstance(past_key_values, FgateDynamicCache)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = {} if output_attentions else None
        next_decoder_cache = None

        for layer_id, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                gradient_checkpointing=self.gradient_checkpointing and self.training
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                OUTPUT_ATTN_LAYERS = [0, 7, 15, 23]
                if layer_id in OUTPUT_ATTN_LAYERS:
                    # all_attns += (layer_outputs[1],)
                    all_attns[layer_id] = layer_outputs[1]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            # next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
            next_cache = next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns
        )
        
class GPT(PreTrainedModel):
    config_class = GPTConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.model = ForgettingTransformerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Run base init if any
        self.post_init()

        # Tie token embeddings and LM head
        self.model.embeddings.weight = self.lm_head.weight

        # Re-initialize to align with latest API (llama-mha-alibi)
        emb_std = getattr(config, 'embedding_init_std', 0.02)
        hidden_factor = getattr(config, 'hidden_init_std_factor', 0.5)
        n_embd = getattr(config, 'n_embd', config.hidden_size)
        n_layer = getattr(config, 'n_layer', config.num_hidden_layers)
        hidden_std = hidden_factor / math.sqrt(n_embd)
        proj_std = hidden_factor / math.sqrt(n_embd) / math.sqrt(n_layer)

        with torch.no_grad():
            # Initialize tied embedding/LM head
            self.lm_head.weight.normal_(mean=0.0, std=emb_std)

            # Special init for attention/MLP output projections
            for name, module in self.named_modules():
                # attn output projection
                if isinstance(module, ForgettingAttentionLayer):
                    if hasattr(module, 'c_proj') and hasattr(module.c_proj, 'weight'):
                        module.c_proj.weight.normal_(mean=0.0, std=proj_std)
                # mlp output projection
                if isinstance(module, ForgettingTransformerMLP):
                    if hasattr(module, 'down_proj') and hasattr(module.down_proj, 'weight'):
                        module.down_proj.weight.normal_(mean=0.0, std=proj_std)

            # Initialize all other 2D+ weights (excluding the above and tied embeddings)
            skip_suffixes = (
                'attn.c_proj.weight',
                'mlp.down_proj.weight',
            )
            for name, p in self.named_parameters():
                if p.dim() >= 2:
                    if name.endswith(skip_suffixes) or name.endswith('lm_head.weight') or name.endswith('embeddings.weight'):
                        continue
                    p.normal_(mean=0.0, std=hidden_std)
    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # only last token for `inputs_ids` if the `past_key_values` is passed along.
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard.
            # Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {'input_ids': input_ids.contiguous()}

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
        })
        return model_inputs
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        targets: Optional[torch.LongTensor] = None,
        output_all_seq: Optional[bool] = False,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_logits: Optional[bool] = True,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        hidden_states = outputs[0]

        loss = None
        if targets is not None:
            logits = self.lm_head(hidden_states)
            # Enable model parallelism
            targets = targets.to(logits.device)
            if self.config.fuse_cross_entropy:
                loss_fct = FusedCrossEntropyLoss(inplace_backward=True, ignore_index=-1)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
            # loss = loss.view(*targets.size())
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        elif output_all_seq:
            logits = self.lm_head(hidden_states[:, :, :]) # note: using list [-1] to preserve the time dim
            loss = None
        else:
            logits = self.lm_head(hidden_states[:, [-1], :])
            logits = logits.float()  # use tf32/fp32 for logits
            loss = None
        
        if not return_logits:
            logits = None
            
        return logits, loss
            
        
    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        # assert block_size <= self.config.block_size
        # self.config.block_size = block_size
        # self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        # for block in self.transformer.h:
        #     block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
        pass
                
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        # return n_params
        return n_params

    # 添加保存和加载配置的方法
    def save_pretrained(self, save_directory):
        self.config.save_pretrained(save_directory)
        super().save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = super().from_pretrained(pretrained_model_name_or_path, config=config, *model_args, **kwargs)
        return model
