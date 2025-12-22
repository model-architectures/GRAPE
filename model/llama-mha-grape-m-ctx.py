# file: model/llama-mha-grape-ctx-type1.py
# Contextual GRAPE Type-I (Phase-Modulated, Commuting)
# - Fixed commuting dictionary on canonical coordinate planes (like RoPE planes).
# - Learnable per-plane base frequencies θ_j initialized from RoPE's log-uniform (base=10000).
# - A single scalar, nonnegative per-token increment ω_t = g(x_t); cumulative phase Φ_t = Σ_{τ<t} ω_τ.
# - Each plane rotates by angle Φ_t * θ_j. Exact relative law holds: G_t^T G_j = exp((Φ_j-Φ_t) L).
#
# Implementation details:
# * g(·) is a lightweight linear gate with Softplus to enforce ω_t ≥ 0.
# * We initialize g to produce small ω (~1e-3) so the model starts near non-contextual RoPE.
# * cos/sin are computed per-(B,T,H) because Φ depends on the input; values cached in fp32.
# * Drop-in for llama-mha.py: only the positional module differs.

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
# Helpers

def _softplus_inverse(y: float) -> float:
    """Return x such that softplus(x) = y, for y>0. Using log(expm1(y))."""
    return math.log(math.expm1(float(y)))

def apply_rotary_emb(x, cos, sin):
    """
    Apply planar 2D rotations to x.
    x:   (B, T, H, D)
    cos: broadcastable to (B, T, H, D//2)  [e.g., (B,T,H,D//2) or (1,T,H,D//2)]
    sin: same as cos
    """
    assert x.ndim == 4, "expected x of shape (B, T, H, D)"
    d_half = x.shape[-1] // 2
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]
    # Broadcast multiply
    y1 = x1 * cos + x2 * sin
    y2 = (-x1) * sin + x2 * cos
    y = torch.cat([y1, y2], dim=-1)
    return y.type_as(x)

# -----------------------------------------------------------------------------
# Contextual Type-I: Φ_t (scalar) modulates a commuting MS-GRAPE dictionary

class MSGRAPEContextType1(nn.Module):
    """
    Contextual GRAPE Type-I:
      - Frequencies θ_j (per plane), learnable; initialized from RoPE (base=10000).
      - Single scalar cumulative phase per token Φ_t (broadcast to all heads/planes).
    """
    def __init__(
        self,
        n_head: int,
        head_dim: int,
        base: float = 10000.0,
        learnable_freq: bool = True,
        share_freq_across_heads: bool = True,
        cache_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for planar rotations"
        self.n_head = n_head
        self.head_dim = head_dim
        self.d_half = head_dim // 2
        self.base = float(base)
        self.learnable_freq = learnable_freq
        self.share_freq_across_heads = share_freq_across_heads
        self.cache_dtype = cache_dtype

        # RoPE-style initialization: θ_j = 1 / base**(2j/D)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (D/2,)
        log_init = inv_freq.log().float()

        if share_freq_across_heads:
            self.log_freq = nn.Parameter(log_init, requires_grad=learnable_freq)           # (D/2,)
        else:
            self.log_freq = nn.Parameter(log_init.unsqueeze(0).repeat(n_head, 1),
                                         requires_grad=learnable_freq)                    # (H, D/2)

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
        """Positive per-plane frequencies θ_j; shape (H, D/2)."""
        scaled_log_freq = self.log_freq
        if self.share_freq_across_heads:
            return torch.exp(scaled_log_freq).unsqueeze(0).expand(self.n_head, self.d_half)
        else:
            return torch.exp(scaled_log_freq)

    def cos_sin_from_phi(self, phi: torch.Tensor):
        """
        phi: (B, T, 1) or (B, T, H)  cumulative phase Φ_t
        Returns cos, sin with shape (B, T, H, D/2) in cache_dtype.
        """
        B, T = phi.shape[0], phi.shape[1]
        device = phi.device

        # (H, D/2) -> (1,1,H,D/2)
        freq = self.freq.to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Broadcast Φ to heads if needed
        if phi.size(-1) == 1:
            phi = phi.expand(B, T, self.n_head)  # (B,T,H)

        # Angles: (B,T,H,1) * (1,1,H,D/2) -> (B,T,H,D/2)
        angles = (phi.unsqueeze(-1).to(torch.float32)) * freq
        cos = angles.cos().to(self.cache_dtype)
        sin = angles.sin().to(self.cache_dtype)
        return cos, sin

    def rotate(self, q: torch.Tensor, k: torch.Tensor, phi: torch.Tensor):
        """
        Apply contextual GRAPE rotations to q,k using cumulative phase phi.
        q,k: (B, T, H, D)
        phi: (B, T, 1) or (B, T, H)
        """
        cos, sin = self.cos_sin_from_phi(phi)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        return q, k

    def extra_repr(self) -> str:
        share = "shared" if self.share_freq_across_heads else "per-head"
        learn = "learnable" if self.learnable_freq else "frozen"
        return f"MS-GRAPE(ctx Type-I): {share} freq, {learn}, base={self.base}, head_dim={self.head_dim}"


# -----------------------------------------------------------------------------
# RMSNorm, MLP, Attention, Blocks (drop-in; attention now computes Φ_t)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        assert self.n_head * self.head_dim == self.n_embd, "n_head * head_dim must equal n_embd"

        # QKV projections
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
        # Output projection
        self.c_proj = nn.Linear(self.n_head * self.head_dim, self.n_embd, bias=False)
        with torch.no_grad():
            factor = getattr(config, 'hidden_init_std_factor', 0.5)
            std = factor / math.sqrt(config.n_embd) / math.sqrt(config.n_layer)
            self.c_proj.weight.normal_(mean=0.0, std=std)

        # === Contextual GRAPE Type-I ===
        self.grape = MSGRAPEContextType1(
            n_head=config.n_head,
            head_dim=config.head_dim,
            base=getattr(config, 'grape_base', 10000.0),
            learnable_freq=getattr(config, 'grape_learnable_freq', True),
            share_freq_across_heads=getattr(config, 'grape_share_across_heads', True),
            cache_dtype=torch.float32,
        )

        # Phase gate ω_t = softplus(Wx+b)  (scalar per token by default; can be per-head if enabled)
        self.grape_ctx_per_head = getattr(config, 'grape_ctx_per_head', False)
        out_dim = self.n_head if self.grape_ctx_per_head else 1
        self.omega_proj = nn.Linear(self.n_embd, out_dim, bias=True)
        ctx_omega_scale = getattr(config, "grape_ctx_omega_scale", None)
        if ctx_omega_scale is None:
            ctx_omega_scale = getattr(config, "grape_log_freq_scale", 16.0)
        self.ctx_omega_scale = float(ctx_omega_scale)
        if self.ctx_omega_scale <= 0:
            raise ValueError("grape_ctx_omega_scale must be > 0")

        # Initialize gate near zero increments (e.g., ~1e-3) so training begins close to non-contextual.
        init_omega = max(getattr(config, 'grape_ctx_init_omega', 1e-3), 1e-8)
        with torch.no_grad():
            nn.init.zeros_(self.omega_proj.weight)
            self.omega_proj.bias.fill_(_softplus_inverse(init_omega))

        # QK RMSNorm
        self.use_qk_rmsnorm = getattr(config, 'use_qk_rmsnorm', True)
        if self.use_qk_rmsnorm:
            self.q_rms = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
            self.k_rms = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)

        # Optional per-head RMSNorm on attn outputs
        self.using_groupnorm = getattr(config, 'using_groupnorm', False)
        if self.using_groupnorm:
            self.subln = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)

    def _cumulative_phase(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Φ_t = Σ_{τ<t} ω_τ from input features x (B,T,C).
        Returns phi with shape:
            (B,T,1) if grape_ctx_per_head=False,
            (B,T,H) if grape_ctx_per_head=True.
        """
        omega = F.sigmoid(self.omega_proj(x).float() / self.ctx_omega_scale)  # (B,T,1 or H), ω_t ≥ 0
        # exclusive cumsum along time: cumsum - current
        phi = torch.cumsum(omega, dim=1) - omega
        return phi

    def forward(self, x):
        B, T, C = x.size()

        # Compute cumulative phase from current layer's input (pre-attention)
        phi = self._cumulative_phase(x)  # (B,T,1) or (B,T,H)

        # QKV
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        if self.use_k_shift:
            k = self.c_k(x, None).view(B, T, self.n_head, self.head_dim)
        else:
            k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        if self.use_v_shift:
            v = self.c_v(x, None).view(B, T, self.n_head, self.head_dim)
        else:
            v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        # Apply contextual GRAPE rotations to Q and K
        q, k = self.grape.rotate(q, k, phi)

        # Optional QK RMSNorm
        if self.use_qk_rmsnorm:
            q = self.q_rms(q)
            k = self.k_rms(k)

        # SDPA expects (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        if self.using_groupnorm:
            y = self.subln(y)

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
    head_dim: int = 128
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

    # --- GRAPE (freq) knobs ---
    grape_base: float = 10000.0
    grape_learnable_freq: bool = True
    grape_share_across_heads: bool = True

    # --- Context gate knobs ---
    grape_ctx_per_head: bool = False   # Type-I: scalar Φ by default; set True to make Φ per-head
    grape_ctx_init_omega: float = 1e-3 # initial ω_t via softplus-inverse(bias)
    grape_ctx_omega_scale: float = 16.0 # scale on omega logits; replaces grape_log_freq_scale

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

        init_llama_mha_weights(self, config, exclude_suffixes=("omega_proj.weight",))

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
            logits = self.lm_head(x[:, [-1], :])
            logits = logits.float()
            loss = None

        if not return_logits:
            logits = None
        return logits, loss

    def crop_block_size(self, block_size: int):
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
        flops_promised = 312e12
        return flops_achieved / flops_promised

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
