import importlib
import math
import os
import pickle
import time
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from optim_utils import get_optimizer_param_groups

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on Fineweb-edu 100B
# I/O
data_path = "data"
out_dir = 'output/out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False  # disabled by default
wandb_project = 'nanogpt-next'
wandb_run_name = 'gpt2'  # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5  # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 4096
# model
n_layer = 12
n_head = 12
head_dim = 64
tpa_kvrank = 2
tpa_qrank = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
using_groupnorm = False
# KV shifting (optional)
use_k_shift = False
use_v_shift = False
# optimizer
optimizer_name = 'adamw'
learning_rate = 1e-3  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
rho = 0.1
interval = 10
variant = 4
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl'  # 'nccl', 'gloo', etc.
schedule = 'cosine'
model_type = '__base_model_placeholder__'
group_size = 11
# system
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True  # use PyTorch 2.0 to compile the model to be faster
scale_attn_by_inverse_layer_idx = True
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith('_') and isinstance(v, (int, float, bool, str))
]
exec(open('configurator.py').read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------
model_file = importlib.import_module(f'model.{model_type}')
GPTConfig = model_file.GPTConfig
GPT = model_file.GPT

def get_num_params(self, non_embedding=False):
    """
    Return the number of parameters in the model.
    For non-embedding count (default), the position embeddings get subtracted.
    The token embeddings would too, except due to the parameter sharing these
    params are actually used as weights in the final layer, so we include them.
    """
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding:
        n_params -= self.transformer.wpe.weight.numel()
    return n_params


# Get current date and job ID
current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
job_id = os.environ.get('SLURM_JOB_ID', '0')

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    print(
        f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}, "
        f"RANK: {os.environ.get('RANK')}, "
        f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}"
    )
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    world_size = 1
    gradient_accumulation_steps *= 8  # simulate 8 gpus

# Calculate total tokens in billions
tokens_per_iter = batch_size * block_size * gradient_accumulation_steps * world_size
total_tokens_B = tokens_per_iter * max_iters / (1000 ** 3)

# Add after the initial variable declarations
tokens_trained = 0  # track total tokens trained

# Initialize random seed and torch settings
torch.manual_seed(5000 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
# Note: float16 data type will automatically use a GradScaler
ptdtype = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
}[dtype]
if device_type == 'cpu':
    ctx = nullcontext()
else:
    ctx = torch.autocast(device_type=device_type, dtype=ptdtype)

# Poor man's data loader
data_dir = os.path.join(data_path, dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy(
                data[i + 1 : i + 1 + block_size].astype(np.int64)
            )
            for i in ix
        ]
    )
    if device_type == 'cuda':
        # pin arrays x, y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# Attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# Model initialization
# Build model_args with defaults that can be overridden by config globals
model_args = {
    'n_layer': n_layer,
    'n_head': n_head,
    'n_embd': n_embd,
    'block_size': block_size,
    'bias': bias,
    'head_dim': head_dim,
    'tpa_kvrank': tpa_kvrank,
    'tpa_qrank': tpa_qrank,
    'using_groupnorm': using_groupnorm,
    'vocab_size': None,
    'dropout': dropout,
    'scale_attn_by_inverse_layer_idx': scale_attn_by_inverse_layer_idx,
    # Init/normalization knobs (with sensible defaults)
    'embedding_init_std': globals().get('embedding_init_std', 0.02),
    'hidden_init_std_factor': globals().get('hidden_init_std_factor', 0.5),
    'use_qk_rmsnorm': globals().get('use_qk_rmsnorm', True),
    'use_k_shift': globals().get('use_k_shift', False),
    'use_v_shift': globals().get('use_v_shift', False),
    'p_tie_mode': globals().get('p_tie_mode', 'none'),
    'p_head_dim': globals().get('p_head_dim', None),
}  # start with model_args from command line

# Fox-specific toggles (only relevant when using model_type == 'fox')
if model_type == 'fox':
    model_args.update({
        'fgate_type': globals().get('fgate_type', 'full'),
    })

if "gqa" in model_type:
    model_args['group_size'] = group_size

# Pass through any GRAPE-specific hyperparameters provided via config files.
if 'grape' in model_type:
    for key, value in list(globals().items()):
        if key.startswith('grape_'):
            model_args[key] = value

# Pass through any key-gated additive bias hyperparameters provided via config files.
if 'keygated' in model_type:
    for key, value in list(globals().items()):
        if key.startswith('keygated_'):
            model_args[key] = value

# Pass through any query-gated additive bias hyperparameters provided via config files.
if 'querygated' in model_type or 'querykeygated' in model_type:
    for key, value in list(globals().items()):
        if key.startswith('querygated_'):
            model_args[key] = value

# Pass through any query+key-gated additive bias hyperparameters provided via config files.
if 'querykeygated' in model_type:
    for key, value in list(globals().items()):
        if key.startswith('querykeygated_'):
            model_args[key] = value

if init_from == 'scratch':
    # Init a new model from scratch
    print("Initializing a new model from scratch")
    # Determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # Resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # Force these config attributes to be equal otherwise we can't even resume training
    # The rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # Create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # Fix the keys of the state dictionary :(
    # Honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # Initialize from OpenAI GPT-2 weights
    override_args = {'dropout': dropout}
    model = GPT.from_pretrained(init_from, override_args)
    # Read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# Crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size  # so that the checkpoint will have the right value
model.to(device)

# Now calculate non-embedding parameters
param_count = get_num_params(model, non_embedding=False)
param_count_m = param_count / 1_000_000  # convert to millions

# Update wandb run name and out_dir if not resuming
if init_from != 'resume':
    # Update wandb run name
    wandb_run_name = f"M_{model_type}_{int(param_count_m)}m_Opt_{optimizer_name}_LR_{learning_rate}_D_{dataset}_T_{total_tokens_B:.2f}B_time_{current_date}_jobid_{job_id}"
    # Update output directory
    out_dir = f"output/out_{model_type}_{int(param_count_m)}m_Opt_{optimizer_name}_LR_{learning_rate}_D_{dataset}_T_{total_tokens_B:.2f}B_time_{current_date}_jobid_{job_id}"

# Now create the output directory
if master_process:
    os.makedirs(out_dir, exist_ok=True)

# Initialize a GradScaler. If enabled=False, scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Optimizer
param_groups = get_optimizer_param_groups(model, weight_decay)
optimizer = AdamW(
    param_groups,
    lr=learning_rate,
    betas=(beta1, beta2),
    eps=1e-8,
)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    del state_dict
    del checkpoint
# Compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# Wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Learning rate decay scheduler (cosine with warmup)
def get_lr(it, schedule='cosine'):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1

    return min_lr + coeff * (learning_rate - min_lr)

# Logging
if wandb_log and master_process:
    import wandb
    wandb_config = {
        'model_args': model_args,
        'training_args': {
            'batch_size': batch_size,
            'block_size': block_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'max_iters': max_iters,
            'lr_decay_iters': lr_decay_iters,
            'eval_interval': eval_interval,
            'eval_iters': eval_iters,
            'log_interval': log_interval,
        },
        'optimizer_args': {
            'optimizer_name': optimizer_name,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'beta1': beta1,
            'beta2': beta2,
            'grad_clip': grad_clip,
            'decay_lr': decay_lr,
            'warmup_iters': warmup_iters,
            'min_lr': min_lr,
            'schedule': schedule,
        },
    }
    wandb.init(project=wandb_project, name=wandb_run_name, config=wandb_config)

# Training loop
X, Y = get_batch('train')  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
clip_time = 0
while True:

    # Determine and set the learning rate for this iteration
    lr = get_lr(iter_num, schedule=schedule) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )
        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                },
                step=iter_num,
            )
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                print(f"saving checkpoint to {out_dir}")
                # 保存模型和配置
                raw_model.save_pretrained(out_dir)
                # 保存优化器状态(可选)
                optimizer_state = {
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                torch.save(optimizer_state, os.path.join(out_dir, 'optimizer.pt'))

        if iter_num % (eval_interval * 5) == 0:
            checkpoint_dir = os.path.join(out_dir, f'checkpoint-{iter_num}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            # 保存模型和配置
            raw_model.save_pretrained(checkpoint_dir)
            # 保存优化器状态(可选)
            optimizer_state = {
                'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
            }
            torch.save(optimizer_state, os.path.join(checkpoint_dir, 'optimizer.pt'))

    if iter_num == 0 and eval_only:
        break

    # Forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # In DDP training we only need to sync gradients at the last micro step.
            # The official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # Looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
        # Immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # Backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # Clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if total_norm.item() > grad_clip:
            clip_time += 1
    # Step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # Flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    # Update total tokens trained
    tokens_trained += tokens_per_iter
    tokens_trained_B = tokens_trained / 1e9  # Convert to billions

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item()  # loss as float. note: this is a CPU-GPU sync point
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        tokens_per_sec = tokens_per_iter / dt
        tokens_per_sec_M = tokens_per_sec / 1_000_000
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, "
            f"mfu {running_mfu * 100:.2f}%, tps (M) {tokens_per_sec_M:.2f}, "
            f"tokens trained {tokens_trained_B:.2f}B"
        )

        params = [param for _, param in model.named_parameters()]
        total_param_norm = 0.0
        for param in params:
            param_norm = param.data.norm(2)
            total_param_norm += param_norm.item() ** 2
        total_param_norm = total_param_norm ** 0.5

        optimizer_state = optimizer.state_dict()['state']
        momentum_norm = 0
        momentum_norm_sq = 0
        for state in optimizer_state.values():
            momentum_norm += state['exp_avg'].detach().norm(2) ** 2
            momentum_norm_sq += state['exp_avg_sq'].detach().norm(2) ** 2
        momentum_norm = torch.sqrt(momentum_norm).item()
        momentum_norm_sq = torch.sqrt(momentum_norm_sq).item()
        momentum_div = momentum_norm / (np.sqrt(momentum_norm_sq) + 1e-8)
        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": lossf,
                    "lr": lr,
                    "param_norm": total_param_norm,
                    "momentum_norm": momentum_norm,
                    "momentum_norm_sq": momentum_norm_sq,
                    "momentum_div": momentum_div,
                    "train/clip_rate": clip_time / (iter_num + 1),
                    "train/grad_norm": total_norm.item() if grad_clip != 0.0 else 0.0,
                    "train/iter_time_ms": dt * 1000,
                    "train/mfu": running_mfu * 100,
                    "train/tokens_per_sec_M": tokens_per_sec_M,
                    "train/tokens_trained_B": tokens_trained_B,
                    "gpu/memory_allocated_MB": torch.cuda.memory_allocated() / (1024 * 1024),
                    "gpu/max_memory_allocated_MB": torch.cuda.max_memory_allocated() / (1024 * 1024),
                },
                step=iter_num,
            )
    iter_num += 1
    local_iter_num += 1

    # Termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
