# For comparison
# LlaMA using SwiGLU, learnable RMSNorm, and GRAPE additive bias

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from .rmsnorm import RMSNorm
from .kv_shift import ShiftLinear
from .init_utils import init_llama_mha_weights


class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().float()
            self.sin_cached = freqs.sin().float()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

def _get_alibi_slopes(n_heads: int):
    """Return head-wise slopes in the ALiBi style (used as GRAPE head scalars)."""
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
    
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        tie_mode_cfg = getattr(config, 'p_tie_mode', 'none')
        if isinstance(tie_mode_cfg, str):
            tie_mode_cfg = tie_mode_cfg.lower()
        alias_map = {
            'pq': 'pqtie',
            'pk': 'pktie',
            'pv': 'pvtie',
        }
        tie_mode = alias_map.get(tie_mode_cfg, tie_mode_cfg)
        if tie_mode not in ('none', 'pqtie', 'pktie', 'pvtie'):
            raise ValueError(
                f"Unsupported p_tie_mode '{tie_mode}'. Expected one of: 'none', 'pqtie', 'pktie', 'pvtie'."
            )
        self.p_tie_mode = tie_mode
        self.p_is_tied = self.p_tie_mode != 'none'
        assert self.n_embd % self.n_head == 0
        config_p_head_dim = getattr(config, 'p_head_dim', None)
        if self.p_is_tied:
            if config_p_head_dim not in (None, config.head_dim):
                raise ValueError(
                    "p_head_dim must match head_dim when tying P to existing projections."
                )
            self.p_head_dim = self.head_dim
        else:
            if config_p_head_dim is None:
                self.p_head_dim = self.head_dim
            else:
                if not isinstance(config_p_head_dim, int) or isinstance(config_p_head_dim, bool):
                    raise TypeError("p_head_dim must be an integer when specified.")
                if config_p_head_dim <= 0:
                    raise ValueError("p_head_dim must be positive.")
                if config_p_head_dim % 2 != 0:
                    raise ValueError("p_head_dim must be even to support rotary embedding.")
                self.p_head_dim = config_p_head_dim
        # projections to per-head dims
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
        # GRAPE positional projection p (unused when tying to q/k/v)
        self.c_p = None if self.p_is_tied else nn.Linear(self.n_embd, self.n_head * self.p_head_dim, bias=False)
        # output projection back to embedding dim
        self.c_proj = nn.Linear(self.n_head * self.head_dim, self.n_embd, bias=False)
        # initialize attn output proj with reduced std: factor/sqrt(n_embd)/sqrt(layers)
        with torch.no_grad():
            factor = getattr(config, 'hidden_init_std_factor', 0.5)
            std = factor / math.sqrt(config.n_embd) / math.sqrt(config.n_layer)
            self.c_proj.weight.normal_(mean=0.0, std=std)

        # GRAPE rotary for p (base=1)
        rotary_dim = self.head_dim if self.p_is_tied else self.p_head_dim
        self.rotary_position = Rotary(rotary_dim, base=1)

        # Per-head slopes used to scale GRAPE additive bias
        slopes = torch.tensor(_get_alibi_slopes(self.n_head), dtype=torch.float32).view(1, self.n_head, 1, 1)
        self.register_buffer("slopes", slopes, persistent=False)

        self.using_groupnorm = config.using_groupnorm
        # QK RMSNorm (learnable) flag and layers
        self.use_qk_rmsnorm = getattr(config, 'use_qk_rmsnorm', True)
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
        p = None if self.c_p is None else self.c_p(x).view(B, T, self.n_head, self.p_head_dim)

        if self.use_qk_rmsnorm:
            q = self.q_rms(q)
            k = self.k_rms(k)

        if self.p_tie_mode == 'pqtie':
            p = q
        elif self.p_tie_mode == 'pktie':
            p = k
        elif self.p_tie_mode == 'pvtie':
            p = v
        elif p is None:
            raise RuntimeError("GRAPE positional projection is undefined.")

        # GRAPE additive bias via rotary of normalized p
        p_norm = F.rms_norm(p, (p.size(-1),))  # keep original GRAPE p normalization (non-learnable)
        cos_p, sin_p = self.rotary_position(p_norm)
        p_rot = apply_rotary_emb(p_norm, cos_p, sin_p)
        p = p.transpose(1, 2)       # (B, H, T, D)
        p_rot = p_rot.transpose(1, 2)  # (B, H, T, D)

        # compute additive attention bias
        scale_factor = 1.0 / p.size(-1)
        attn_bias = F.logsigmoid((torch.matmul(p, p_rot.transpose(-2, -1)) * scale_factor).float())
        attn_bias = attn_bias * self.slopes.to(device=q.device, dtype=attn_bias.dtype)  # (B, H, T, T)
        temp_mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril(diagonal=0)
        attn_bias = attn_bias.masked_fill(~temp_mask, 0.0)
        attn_bias_sum = attn_bias.sum(dim=-1, keepdim=True)
        attn_bias = (attn_bias_sum - attn_bias.cumsum(dim=-1))
        attn_bias = attn_bias.masked_fill(~temp_mask, float("-inf"))

        # SDPA ignores attn_mask when is_causal=True; attn_bias already includes causal mask.
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_bias,
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
        # Calculate the floored hidden dimension size
        hidden_dim = math.floor(8 / 3 * config.n_embd)

        # Split the linear projection into two parts for SwiGLU
        self.c_fc1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, hidden_dim, bias=False)

        # Output projection
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
        # initialize MLP output proj with reduced std: factor/sqrt(n_embd)/sqrt(layers)
        with torch.no_grad():
            factor = getattr(config, 'hidden_init_std_factor', 0.5)
            std = factor / math.sqrt(config.n_embd) / math.sqrt(config.n_layer)
            self.c_proj.weight.normal_(mean=0.0, std=std)

    def forward(self, x):
        # Apply the first linear layer to produce two projections
        x1 = self.c_fc1(x)
        x2 = self.c_fc2(x)

        # Apply the SwiGLU gating: SILU on one projection, and gate with the other
        x = F.silu(x1) * x2

        # Apply the final output projection
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
    p_tie_mode: str = 'none'  # Options: 'none', 'pqtie', 'pktie', 'pvtie' (alias 'pq'/'pk'/'pv' -> '*tie')
    p_head_dim: Optional[int] = None  # Per-head dim for learnable P projection when not tied
    # Embedding init std (normal init for tied token embedding / LM head)
    embedding_init_std: float = 0.02
    # Factor for hidden (>=2D) param init; actual std = factor / sqrt(n_embd)
    hidden_init_std_factor: float = 0.5
    # Optional legacy fields kept for compatibility
    q_rank: int = 1
    rank: int = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
class GPT(PreTrainedModel):
    config_class = GPTConfig
    base_model_prefix = "gpt2"
    supports_gradient_checkpointing = True

    def __init__(self, config):
        # if self is not a subclass of PreTrinedModel, then we need to call super().__init__()
        # else we can just call super().__init__(config) to handle the config argument
        if not isinstance(self, PreTrainedModel):
            super().__init__()
        else:
            super().__init__(config)
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        self.ln_f = RMSNorm(config.n_embd)
        init_llama_mha_weights(self, config)

    def forward(self, idx, targets=None, return_logits=True, output_all_seq=False):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        elif output_all_seq:
            logits = self.lm_head(x[:, :, :]) # note: using list [-1] to preserve the time dim
            loss = None
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
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
        return n_params
    
    # 添加保存和加载配置的方法
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
