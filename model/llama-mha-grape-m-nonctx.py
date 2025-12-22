# file: model/llama-mha-grape-nonctx-type1.py
# Non-contextual MS-GRAPE (commuting) for MHA
# - Per-plane (2D) learnable frequencies, initialized with RoPE's log-uniform spectrum (base=10000)
# - Canonical coordinate planes (commuting dictionary); exact relative law like RoPE
# - Drop-in replacement for the Rotary in llama-mha.py
#
# Notes:
# * This implements Section 3.1 (commuting MS-GRAPE) from the paper: a product of block 2x2 rotations.
# * We keep the coordinate planes fixed and only learn frequencies (one scalar per plane, per head or shared).
# * Frequencies are parameterized as exp(log_freq) to keep them strictly positive.

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from .rmsnorm import RMSNorm
from .kv_shift import ShiftLinear
from .init_utils import init_llama_mha_weights

# -----------------------------------------------------------------------------
# Utilities

def apply_rotary_emb(x, cos, sin):
    """
    Apply planar 2D rotations to x in-place fashion (returns a new tensor).
    x:   (B, T, H, D)
    cos: (1, T, H, D//2) or (1, T, 1, D//2)  (broadcastable to x halves)
    sin: (1, T, H, D//2) or (1, T, 1, D//2)
    """
    assert x.ndim == 4, "expected x of shape (B, T, H, D)"
    d_half = x.shape[-1] // 2
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]
    y1 = x1 * cos + x2 * sin
    y2 = (-x1) * sin + x2 * cos
    y = torch.cat([y1, y2], dim=-1)
    return y.type_as(x)

# -----------------------------------------------------------------------------
# MS-GRAPE (commuting): learnable frequencies in canonical coordinate planes

class MSGRAPECommuting(nn.Module):
    """
    Multi-Subspace GRAPE (commuting) with canonical coordinate planes.
    - Keeps the basis planes fixed (like RoPE) => commuting block-diagonal rotations.
    - Makes the per-plane angular frequencies learnable (per-head or shared).
    - Frequencies are parameterized as freq = exp(log_freq) to ensure positivity.

    Given sequence length T:
       angle[t, j] = t * freq[j]               (shared across heads)
       or angle[t, h, j] = t * freq[h, j]      (per-head)

    cos/sin are computed in float32 and cached in fp32 for consistency, and
    are broadcast across batch.
    """
    def __init__(
        self,
        n_head: int,
        head_dim: int,
        base: float = 10000.0,
        learnable: bool = True,
        share_across_heads: bool = True,
        log_freq_scale: float = 1.0,
        cache_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for planar rotations"
        self.n_head = n_head
        self.head_dim = head_dim
        self.d_half = head_dim // 2
        self.base = float(base)
        self.learnable = learnable
        self.share_across_heads = share_across_heads
        if learnable:
            self.log_freq_scale = float(log_freq_scale)
            if self.log_freq_scale <= 0:
                raise ValueError("log_freq_scale must be > 0")
        else:
            self.log_freq_scale = 1.0
        self.cache_dtype = cache_dtype

        # RoPE-style log-uniform initial spectrum on canonical planes
        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (D/2,)
        log_init = inv_freq.log().float() * self.log_freq_scale  # scaled init for lr-scaling trick

        if share_across_heads:
            # One spectrum shared by all heads
            self.log_freq = nn.Parameter(log_init, requires_grad=learnable)  # (D/2,)
        else:
            # Per-head spectrum
            log_init = log_init.unsqueeze(0).repeat(n_head, 1)               # (H, D/2)
            self.log_freq = nn.Parameter(log_init, requires_grad=learnable)  # (H, D/2)

        # simple cache for the last-seen T to avoid re-allocations (optional)
        self._cached_T = None
        self._cached_cos = None
        self._cached_sin = None

    def _apply(self, fn):
        # Keep log_freq in fp32 regardless of module-wide dtype casts.
        super()._apply(fn)
        if getattr(self, "log_freq", None) is not None:
            self.log_freq.data = self.log_freq.data.float()
            if self.log_freq.grad is not None:
                self.log_freq.grad.data = self.log_freq.grad.data.float()
        return self

    @property
    def freq(self):
        # strictly positive frequencies
        scaled_log_freq = self.log_freq / self.log_freq_scale
        if self.share_across_heads:
            # (D/2,) -> (H, D/2) for uniform processing
            return torch.exp(scaled_log_freq).unsqueeze(0).expand(self.n_head, self.d_half)
        else:
            # (H, D/2)
            return torch.exp(scaled_log_freq)

    def get_cos_sin(self, T: int, device: torch.device, dtype: torch.dtype):
        """
        Returns cos, sin with shapes (1, T, H, D/2) in cache_dtype (e.g., float32).
        Always recompute if T changes or buffers are on a different device.
        We recompute every forward to reflect frequency updates during training.
        """
        # Build time indices [0..T-1]
        t = torch.arange(T, device=device, dtype=torch.float32)  # (T,)
        # Frequencies (H, D/2) in float32 for stability
        freq = self.freq.to(device=device, dtype=torch.float32)  # (H, D/2)
        # Angles: (T, H, D/2)
        angles = t[:, None, None] * freq[None, :, :]            # broadcast time

        cos = angles.cos().to(self.cache_dtype)  # (T, H, D/2)
        sin = angles.sin().to(self.cache_dtype)

        # reshape to (1, T, H, D/2) to broadcast across batch
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        return cos, sin

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """
        q, k: (B, T, H, D)
        returns: q', k' with MS-GRAPE commuting rotations applied
        """
        B, T, H, D = q.shape
        assert H == self.n_head and D == self.head_dim
        cos, sin = self.get_cos_sin(T, device=q.device, dtype=q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        return q, k

    def extra_repr(self) -> str:
        share = "shared" if self.share_across_heads else "per-head"
        learn = "learnable" if self.learnable else "frozen"
        return f"MS-GRAPE(commuting): {share} freq, {learn}, base={self.base}, head_dim={self.head_dim}"


# -----------------------------------------------------------------------------
# RMSNorm, MLP, Attention, Blocks (as in llama-mha.py, with GRAPE dropped in)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        assert self.n_head * self.head_dim == self.n_embd, "n_head * head_dim must equal n_embd"

        # QKV projections (no bias by default)
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.use_k_shift = getattr(config, "use_k_shift", False)
        self.use_v_shift = getattr(config, "use_v_shift", False)
        if self.use_k_shift:
            self.c_k = ShiftLinear(self.n_embd, self.n_head * self.head_dim, self.n_head, bias=False)
        else:
            self.c_k = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        if self.use_v_shift:
            self.c_v = ShiftLinear(self.n_embd, self.n_head * self.head_dim, self.n_head, bias=False)
        else:
            self.c_v = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        # Output projection back to embedding dim
        self.c_proj = nn.Linear(self.n_head * self.head_dim, self.n_embd, bias=False)

        # Initialize attn output proj with reduced std: factor/sqrt(n_embd)/sqrt(layers)
        with torch.no_grad():
            factor = getattr(config, 'hidden_init_std_factor', 0.5)
            std = factor / math.sqrt(config.n_embd) / math.sqrt(config.n_layer)
            self.c_proj.weight.normal_(mean=0.0, std=std)

        # === MS-GRAPE (commuting) ===
        self.grape = MSGRAPECommuting(
            n_head=config.n_head,
            head_dim=config.head_dim,
            base=getattr(config, 'grape_base', 10000.0),
            learnable=getattr(config, 'grape_learnable_freq', True),
            share_across_heads=getattr(config, 'grape_share_across_heads', True),
            log_freq_scale=getattr(config, 'grape_log_freq_scale', 16.0),
            cache_dtype=torch.float32,
        )

        # QK RMSNorm (learnable) flag and layers
        self.use_qk_rmsnorm = getattr(config, 'use_qk_rmsnorm', True)
        if self.use_qk_rmsnorm:
            self.q_rms = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
            self.k_rms = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)

        # Optional per-head RMSNorm post attention (groupnorm-like)
        self.using_groupnorm = getattr(config, 'using_groupnorm', False)
        if self.using_groupnorm:
            self.subln = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x):
        B, T, C = x.size()
        # Project to heads
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        if self.use_k_shift:
            k = self.c_k(x, None).view(B, T, self.n_head, self.head_dim)
        else:
            k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        if self.use_v_shift:
            v = self.c_v(x, None).view(B, T, self.n_head, self.head_dim)
        else:
            v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        # Apply commuting MS-GRAPE rotations to Q and K
        q, k = self.grape(q, k)

        # Optional learnable RMSNorm on Q and K (per head)
        if self.use_qk_rmsnorm:
            q = self.q_rms(q)
            k = self.k_rms(k)

        # SDPA expects (B, H, T, D)
        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        if self.using_groupnorm:
            # Apply RMSNorm head-wise on the attention outputs
            y = self.subln(y)

        # Merge heads
        y = y.transpose(1, 2).contiguous().reshape(B, T, self.n_head * self.head_dim)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = math.floor(8 / 3 * config.n_embd)
        self.c_fc1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
        with torch.no_grad():
            factor = getattr(config, 'hidden_init_std_factor', 0.5)
            std = factor / math.sqrt(config.n_embd) / math.sqrt(config.n_layer)
            self.c_proj.weight.normal_(mean=0.0, std=std)

    def forward(self, x):
        x1 = self.c_fc1(x)
        x2 = self.c_fc2(x)
        x = F.silu(x1) * x2
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.ln_1 = RMSNorm(config.n_embd)
        self.ln_2 = RMSNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# -----------------------------------------------------------------------------
# The main GPT model

@dataclass
class GPTConfig(PretrainedConfig):
    model_type = "gpt2"
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6
    n_embd : int = 768
    head_dim: int = 128                 # per-head dim, must divide n_embd
    block_size: int = 1024
    bias: bool = False
    dropout: float = 0.0
    scale_attn_by_inverse_layer_idx: bool = False
    using_groupnorm: bool = False
    use_qk_rmsnorm: bool = True
    use_k_shift: bool = False
    use_v_shift: bool = False

    # Initializations
    embedding_init_std: float = 0.02
    hidden_init_std_factor: float = 0.5

    # --- MS-GRAPE (commuting) knobs ---
    grape_base: float = 10000.0             # RoPE-style base for init
    grape_learnable_freq: bool = True        # make per-plane frequencies learnable
    grape_share_across_heads: bool = True    # share the spectrum across heads (RoPE-like)
    grape_log_freq_scale: float = 16.0       # scale log_freq to reduce its effective lr (learnable only)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class GPT(PreTrainedModel):
    config_class = GPTConfig
    base_model_prefix = "gpt2"
    supports_gradient_checkpointing = True

    def __init__(self, config):
        if not isinstance(self, PreTrainedModel):
            super().__init__()
        else:
            super().__init__(config)
        self.config = config

        assert (config.n_head * config.head_dim) == config.n_embd, \
            f"n_head * head_dim ({config.n_head}*{config.head_dim}) must equal n_embd ({config.n_embd})"

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Final RMSNorm
        self.ln_f = RMSNorm(config.n_embd)

        init_llama_mha_weights(self, config)

    def forward(self, idx, targets=None, return_logits=True, output_all_seq=False):
        x = self.transformer.wte(idx)  # (B, T, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            logits = logits.float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        elif output_all_seq:
            logits = self.lm_head(x)
            loss = None
        else:
            logits = self.lm_head(x[:, [-1], :])  # last position
            logits = logits.float()
            loss = None

        if not return_logits:
            logits = None
        return logits, loss

    def crop_block_size(self, block_size: int):
        # No absolute pos embeddings exist; relative (multiplicative) encoding only.
        # Kept for API parity.
        pass

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12 # A100 bfloat16 peak
        mfu = flops_achieved / flops_promised
        return mfu

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def save_pretrained(self, save_directory):
        self.config.save_pretrained(save_directory)
        super().save_pretrained(save_directory, safe_serialization=False)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = super().from_pretrained(pretrained_model_name_or_path, config=config, *model_args, **kwargs)
        return model
