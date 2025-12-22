# For comparison
# LlaMA using SwiGLU, learnable RMSNorm, and a GRAPE-A key-gated additive bias
# Implements Eq. (add_bias_keygated) in paper.tex.

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from .init_utils import init_llama_mha_weights
from .kv_shift import ShiftLinear
from .rmsnorm import RMSNorm


def _get_alibi_slopes(n_heads: int) -> list[float]:
    """Return head-wise slopes for ALiBi as in the paper implementation."""

    def get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return get_slopes_power_of_2(n_heads)
    closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
    slopes = get_slopes_power_of_2(closest_power_of_2)
    extra = _get_alibi_slopes(2 * closest_power_of_2)
    slopes += extra[0::2][: n_heads - closest_power_of_2]
    return slopes


class KeyGatedAdditiveBias(nn.Module):
    """Add a key-gated relative-position bias: b(i,j) = -(j-i) * ω_h * (u_h^T k_j).

    For causal attention (j <= i), this is equivalent to:
      b(i,j) = (i-j) * ω_h * (u_h^T k_j).
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        *,
        omega_init: float | None = None,
        u_init_std: float | None = None,
        u_l2_norm: bool = True,
        u_l2_eps: float = 1e-6,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.u_l2_norm = bool(u_l2_norm)
        self.u_l2_eps = float(u_l2_eps)

        self.omega = nn.Parameter(torch.empty(n_heads, dtype=torch.float32))
        self.u = nn.Parameter(torch.empty(n_heads, head_dim, dtype=torch.float32))

        with torch.no_grad():
            if omega_init is None:
                # Match ALiBi init: bias ≈ -dist * slope when u^T k ≈ 1.
                slopes = torch.tensor(_get_alibi_slopes(n_heads), dtype=torch.float32)
                self.omega.copy_(-slopes)
            else:
                self.omega.fill_(float(omega_init))
            if u_init_std is None:
                u_init_std = 1.0 / math.sqrt(head_dim)
            self.u.normal_(mean=0.0, std=float(u_init_std))

        self.seq_len_cached: int | None = None
        self.dist_cached: torch.Tensor | None = None  # (1, 1, T, T) float32 on last-used device

    def _get_dist(self, T: int, *, device: torch.device) -> torch.Tensor:
        if T != self.seq_len_cached or self.dist_cached is None or self.dist_cached.device != device:
            self.seq_len_cached = T
            arange = torch.arange(T, device=device)
            # dist[i, j] = i - j; clamp future (j>i) to 0 since is_causal will mask anyway
            dist = (arange.view(T, 1) - arange.view(1, T)).clamp_min(0).to(dtype=torch.float32)
            self.dist_cached = dist.view(1, 1, T, T)
        return self.dist_cached

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        # k is (B, T, H, D)
        B, T, H, D = k.shape
        assert H == self.n_heads, f"Expected H={self.n_heads}, got {H}"
        assert D == self.head_dim, f"Expected D={self.head_dim}, got {D}"

        dist = self._get_dist(T, device=k.device)  # (1, 1, T, T), float32
        # gate[b, t, h] = u_h^T k_{b,t,h,:}
        u = self.u
        if self.u_l2_norm:
            u = F.normalize(u, p=2, dim=-1, eps=self.u_l2_eps)
        gate = (k * u.view(1, 1, H, D).to(dtype=k.dtype, device=k.device)).sum(dim=-1)  # (B, T, H)
        gate = gate.transpose(1, 2).contiguous()  # (B, H, T)

        omega = self.omega.view(1, H, 1, 1).to(dtype=k.dtype, device=k.device)  # (1, H, 1, 1)
        bias = dist.to(dtype=k.dtype) * omega * gate.view(B, H, 1, T)  # (B, H, T, T)
        return bias


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.use_k_shift = getattr(config, "use_k_shift", False)
        self.use_v_shift = getattr(config, "use_v_shift", False)
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        if self.use_k_shift:
            self.c_k = ShiftLinear(
                self.n_embd, self.n_head * self.head_dim, self.n_head, bias=False
            )
        else:
            self.c_k = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        if self.use_v_shift:
            self.c_v = ShiftLinear(
                self.n_embd, self.n_head * self.head_dim, self.n_head, bias=False
            )
        else:
            self.c_v = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        # output projection back to embedding dim
        self.c_proj = nn.Linear(self.n_head * self.head_dim, self.n_embd, bias=False)
        # initialize attn output proj with reduced std: factor/sqrt(n_embd)/sqrt(layers)
        with torch.no_grad():
            factor = getattr(config, "hidden_init_std_factor", 0.5)
            std = factor / math.sqrt(config.n_embd) / math.sqrt(config.n_layer)
            self.c_proj.weight.normal_(mean=0.0, std=std)

        omega_init = getattr(config, "keygated_omega_init", None)
        u_init_std = getattr(config, "keygated_u_init_std", None)
        u_l2_norm = bool(getattr(config, "keygated_u_l2_norm", True))
        u_l2_eps = float(getattr(config, "keygated_u_l2_eps", 1e-6))
        self.keygated_bias = KeyGatedAdditiveBias(
            self.n_head,
            self.head_dim,
            omega_init=omega_init,
            u_init_std=u_init_std,
            u_l2_norm=u_l2_norm,
            u_l2_eps=u_l2_eps,
        )

        self.using_groupnorm = config.using_groupnorm
        # QK RMSNorm (learnable) flag and layers
        self.use_qk_rmsnorm = getattr(config, "use_qk_rmsnorm", True)
        if self.use_qk_rmsnorm:
            self.q_rms = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
            self.k_rms = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
        if self.using_groupnorm:
            # Apply RMSNorm to each head's output dimension
            self.subln = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        if self.use_k_shift:
            k = self.c_k(x, None).view(B, T, self.n_head, self.head_dim)
        else:
            k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        if self.use_v_shift:
            v = self.c_v(x, None).view(B, T, self.n_head, self.head_dim)
        else:
            v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        if self.use_qk_rmsnorm:
            q = self.q_rms(q)
            k = self.k_rms(k)

        # Key-gated additive bias, broadcastable to (B, H, T, T)
        bias = self.keygated_bias(k).float()
        # SDPA ignores attn_mask when is_causal=True, so bake causal mask into bias.
        causal_mask = torch.ones(T, T, dtype=torch.bool, device=bias.device).tril(diagonal=0)
        bias = bias.masked_fill(~causal_mask, float("-inf"))

        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=bias,
            is_causal=False,
        )

        if self.using_groupnorm:
            # Apply RMSNorm directly to each head's output
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
        # initialize MLP output proj with reduced std: factor/sqrt(n_embd)/sqrt(layers)
        with torch.no_grad():
            factor = getattr(config, "hidden_init_std_factor", 0.5)
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
# The main GPT-2 model


@dataclass
class GPTConfig(PretrainedConfig):
    model_type = "gpt2"
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6  # head dim 128 suggested by @Grad62304977
    n_embd: int = 768
    head_dim: int = 128  # Dimension per head
    block_size: int = 1024  # Maximum sequence length
    bias: bool = False  # Use bias in all linear layers
    dropout: float = 0.0  # Dropout rate
    scale_attn_by_inverse_layer_idx: bool = False  # Scale attention by 1/sqrt(layer_idx)
    using_groupnorm: bool = False  # Whether to use Group Layernorm
    use_qk_rmsnorm: bool = True  # Apply learnable RMSNorm to Q and K in attention
    use_k_shift: bool = False
    use_v_shift: bool = False
    # Key-gated additive bias knobs
    keygated_omega_init: float | None = None  # None => ALiBi-style per-head slopes
    keygated_u_init_std: float | None = None
    keygated_u_l2_norm: bool = True
    keygated_u_l2_eps: float = 1e-6
    # Embedding init std (normal init for tied token embedding / LM head)
    embedding_init_std: float = 0.02
    # Factor for hidden (>=2D) param init; actual std = factor / sqrt(n_embd)
    hidden_init_std_factor: float = 0.5

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

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying
        self.ln_f = RMSNorm(config.n_embd)
        init_llama_mha_weights(self, config, exclude_suffixes=("keygated_bias.u",))

    def forward(self, idx, targets=None, return_logits=True, output_all_seq=False):
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            logits = logits.float()  # use tf32/fp32 for logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        elif output_all_seq:
            logits = self.lm_head(x[:, :, :])
            loss = None
        else:
            logits = self.lm_head(x[:, [-1], :])
            logits = logits.float()
            loss = None

        if not return_logits:
            logits = None

        return logits, loss

    def crop_block_size(self, block_size):
        # Placeholder for potential sequence length surgery
        pass

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
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
        model = super().from_pretrained(
            pretrained_model_name_or_path, config=config, *model_args, **kwargs
        )
        return model
