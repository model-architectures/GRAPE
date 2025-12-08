# For comparison
# LlaMA using SwiGLU, learnable RMSNorm, and RoPE

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


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
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

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
        # projections to per-head dimensions
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        # output projection maps back to embedding dim
        self.c_proj = nn.Linear(self.n_head * self.head_dim, self.n_embd, bias=False)
        # initialize attn output proj with reduced std: factor/sqrt(n_embd)/sqrt(layers)
        with torch.no_grad():
            factor = getattr(config, 'hidden_init_std_factor', 0.5)
            std = factor / math.sqrt(config.n_embd) / math.sqrt(config.n_layer)
            self.c_proj.weight.normal_(mean=0.0, std=std)
        self.rotary = Rotary(self.head_dim)
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
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        # Apply learnable RMSNorm to Q and K if enabled
        if self.use_qk_rmsnorm:
            q = self.q_rms(q)
            k = self.k_rms(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        
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
        # Define RMSNorm layers once in the module
        self.ln_1 = RMSNorm(config.n_embd, use_fp32=config.use_fp32_rmsnorm)
        self.ln_2 = RMSNorm(config.n_embd, use_fp32=config.use_fp32_rmsnorm)

    def forward(self, x):
        # Apply pre-norm before sublayers
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig(PretrainedConfig):
    model_type = "gpt2"  
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768
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
        # weight tying between token embedding and LM head
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        
        # initialize the shared embedding/LM head weights with normal_(0, std)
        init_std = getattr(config, 'embedding_init_std', 0.02)
        with torch.no_grad():
            self.lm_head.weight.normal_(mean=0.0, std=init_std)
        # Final RMSNorm defined in the network
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

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x)
        # Apply final RMSNorm before the LM head
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
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        # return n_params
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
