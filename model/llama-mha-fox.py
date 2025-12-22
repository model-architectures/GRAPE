# For comparison
# LlaMA using SwiGLU, learnable RMSNorm, and Fox forgetting attention

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from .rmsnorm import RMSNorm
from .fox import ForgettingAttentionLayer
from .init_utils import init_llama_mha_weights


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.use_k_shift = getattr(config, 'use_k_shift', False)
        self.use_v_shift = getattr(config, 'use_v_shift', False)

        self.core = ForgettingAttentionLayer(
            hidden_size=self.n_embd,
            num_heads=self.n_head,
            num_kv_heads=getattr(config, "num_kv_heads", None),
            window_size=getattr(config, "window_size", None),
            max_position_embeddings=getattr(config, "block_size", None),
            use_rope=getattr(config, "use_rope", False),
            rope_base=getattr(config, "rope_base", 10000.0),
            use_output_gate=getattr(config, "use_output_gate", False),
            ogate_act=getattr(config, "ogate_act", "sigmoid"),
            fgate_type=getattr(config, "fgate_type", "full"),
            fgate_bias_init=getattr(config, "fgate_bias_init", False),
            decay_time_min=getattr(config, "decay_time_min", None),
            decay_time_max=getattr(config, "decay_time_max", None),
            use_output_norm=config.using_groupnorm,
            norm_eps=1e-5,
            qk_norm=getattr(config, "use_qk_rmsnorm", True),
            qk_norm_share_param_across_head=True,
            use_k_shift=self.use_k_shift,
            use_v_shift=self.use_v_shift,
            layer_idx=None,
        )

        # initialize attn output proj with reduced std: factor/sqrt(n_embd)/sqrt(layers)
        with torch.no_grad():
            factor = getattr(config, 'hidden_init_std_factor', 0.5)
            std = factor / math.sqrt(config.n_embd) / math.sqrt(config.n_layer)
            if hasattr(self.core, 'c_proj') and hasattr(self.core.c_proj, 'weight'):
                self.core.c_proj.weight.normal_(mean=0.0, std=std)

    def forward(self, x):
        out, _, _ = self.core(
            hidden_states=x,
            attention_mask=None,
            past_key_values=None,
            output_attentions=False,
            use_cache=False,
        )
        return out


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
        init_llama_mha_weights(self, config)

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
