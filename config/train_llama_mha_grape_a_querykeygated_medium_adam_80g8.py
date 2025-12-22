import time
import os
from datetime import datetime

# Wandb configs
wandb_log = True
wandb_project = 'nanogpt-next'

# Model configs
n_layer = 24
n_head = 8
n_embd = 1024
head_dim = 128
dropout = 0.0
bias = False
using_groupnorm = False  # Enable Group Layernorm
use_qk_rmsnorm = True    # Apply learnable RMSNorm to Q and K

# GRAPE-A (query+key-gated additive) knobs
# If None, ω is initialized with ALiBi-style per-head slopes.
querykeygated_omega_init = None
keygated_u_l2_norm = True
keygated_u_l2_eps = 1e-6
querygated_v_l2_norm = True
querygated_v_l2_eps = 1e-6

# Embedding init (normal std)
embedding_init_std = 0.02
# Hidden weights init factor (all >=2D tensors), actual std = factor / sqrt(n_embd)
hidden_init_std_factor = 0.5

# Training configs
batch_size = 20
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
model_type = 'llama-mha-grape-a-querykeygated'

