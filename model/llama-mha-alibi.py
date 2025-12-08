# For comparison
# LlaMA using SwiGLU, learnable RMSNorm, and ALiBi positional bias

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


def _get_alibi_slopes(n_heads: int):
    """Return head-wise slopes for ALiBi as in the paper implementation.

    This follows the commonly used recipe that yields a set of decreasing
    geometric slopes across heads, and extends to non-powers-of-two by
    interleaving from the next power-of-two set.
    """
    def get_slopes_power_of_2(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return get_slopes_power_of_2(n_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2)
        extra = _get_alibi_slopes(2 * closest_power_of_2)
        slopes += extra[0::2][: n_heads - closest_power_of_2]
        return slopes


class AlibiBias(nn.Module):
    def __init__(self, n_heads: int):
        super().__init__()
        slopes = torch.tensor(_get_alibi_slopes(n_heads), dtype=torch.float32).view(1, n_heads, 1, 1)
        self.register_buffer("slopes", slopes, persistent=False)
        self.seq_len_cached = None
        self.bias_cached = None

    def forward(self, x):
        # x is (B, T, H, D) or (B, T, C)
        T = x.shape[1]
        if T != self.seq_len_cached or self.bias_cached is None or self.bias_cached.device != x.device:
            self.seq_len_cached = T
            arange = torch.arange(T, device=x.device)
            # dist[i, j] = i - j; clamp future (j>i) to 0 since is_causal will mask anyway
            dist = (arange.view(T, 1) - arange.view(1, T)).clamp_min(0).to(dtype=self.slopes.dtype)
            # bias shape: (1, H, T, T)
            bias = -dist.view(1, 1, T, T) * self.slopes.to(device=x.device)
            self.bias_cached = bias
        return self.bias_cached


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine=True, use_fp32: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_fp32 = use_fp32
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if self.use_fp32:
            output = self._norm(x.float())
        else:
            output = self._norm(x)
        if self.weight is not None:
            if self.use_fp32:
                output = output * self.weight.float()
            else:
                output = output * self.weight
        return output.type_as(x) if self.use_fp32 else output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        # output projection back to embedding dim
        self.c_proj = nn.Linear(self.n_head * self.head_dim, self.n_embd, bias=False)
        # initialize attn output proj with reduced std: factor/sqrt(n_embd)/sqrt(layers)
        with torch.no_grad():
            factor = getattr(config, 'hidden_init_std_factor', 0.5)
            std = factor / math.sqrt(config.n_embd) / math.sqrt(config.n_layer)
            self.c_proj.weight.normal_(mean=0.0, std=std)
        self.alibi = AlibiBias(self.n_head)
        self.using_groupnorm = config.using_groupnorm
        # QK RMSNorm (learnable) flag and layers
        self.use_qk_rmsnorm = getattr(config, 'use_qk_rmsnorm', True)
        if self.use_qk_rmsnorm:
            self.q_rms = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True, use_fp32=config.use_fp32_rmsnorm)
            self.k_rms = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True, use_fp32=config.use_fp32_rmsnorm)
        if self.using_groupnorm:
            # Apply RMSNorm to each head's output dimension
            self.subln = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True, use_fp32=config.use_fp32_rmsnorm)

    def forward(self, x):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        # ALiBi additive bias, broadcastable to (B, H, T, T)
        alibi = self.alibi(q).to(dtype=q.dtype)

        if self.use_qk_rmsnorm:
            q = self.q_rms(q)
            k = self.k_rms(k)

        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=alibi,
            is_causal=True,
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
        self.ln_1 = RMSNorm(config.n_embd, use_fp32=config.use_fp32_rmsnorm)
        self.ln_2 = RMSNorm(config.n_embd, use_fp32=config.use_fp32_rmsnorm)

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
    use_fp32_rmsnorm: bool = True  # Cast to fp32 inside RMSNorm by default
    use_qk_rmsnorm: bool = True  # Apply learnable RMSNorm to Q and K in attention
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
        # initialize the shared embedding/LM head weights with normal_(0, std)
        init_std = getattr(config, 'embedding_init_std', 0.02)
        with torch.no_grad():
            self.lm_head.weight.normal_(mean=0.0, std=init_std)
        self.ln_f = RMSNorm(config.n_embd, use_fp32=config.use_fp32_rmsnorm)

        # Initialize all hidden (>=2D) parameters with normal_(0, hidden_init_std)
        # where hidden_init_std = hidden_init_std_factor / sqrt(n_embd).
        # Excludes tied embedding/LM head weights and specially-initialized c_proj weights.
        hidden_std = getattr(config, 'hidden_init_std_factor', 0.5) / math.sqrt(config.n_embd)
        with torch.no_grad():
            for name, p in self.named_parameters():
                if p.dim() >= 2:
                    if (
                        name.endswith('attn.c_proj.weight') or
                        name.endswith('mlp.c_proj.weight') or
                        name == 'transformer.wte.weight' or
                        name == 'lm_head.weight'
                    ):
                        continue
                    p.normal_(mean=0.0, std=hidden_std)

    def forward(self, idx, targets=None, return_logits=True, output_all_seq=False):
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            logits = logits.float()  # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
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
        model = super().from_pretrained(pretrained_model_name_or_path, config=config, *model_args, **kwargs)
        return model
