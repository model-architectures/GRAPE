# file: config/train_llama_mha_grape_nonctx_type1_large_adam_80g8.py
# Large LLaMA-MHA + non-contextual MS-GRAPE (commuting) config

import time
import os
from datetime import datetime

# Wandb configs
wandb_log = True
wandb_project = 'nanogpt-next'

# Model configs
n_layer = 36
n_head = 10
n_embd = 1280
head_dim = 128
dropout = 0.0
bias = False
using_groupnorm = False             # Optional per-head RMSNorm on attn outputs
use_qk_rmsnorm = True               # Apply learnable RMSNorm to Q and K

# --- MS-GRAPE (commuting) knobs ---
grape_base = 10000.0               # RoPE base for log-uniform init
grape_learnable_freq = True        # frequencies are trainable
grape_share_across_heads = True    # share the spectrum across heads (RoPE-like)

# Embedding and hidden init
embedding_init_std = 0.02
hidden_init_std_factor = 0.5

# Training configs
batch_size = 15
block_size = 4096
gradient_accumulation_steps = 60 // batch_size
max_iters = 100000
lr_decay_iters = 100000
eval_interval = 1000
eval_iters = 200
log_interval = 10

# Optimizer configs
optimizer_name = 'adamw'
learning_rate = 1e-3
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
min_lr = 3e-5
schedule = 'cosine'

# System configs
compile = True
model_type = 'llama-mha-grape-m-nonctx'
