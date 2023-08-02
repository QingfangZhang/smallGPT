"""
To run on multiple GPU in one PC, run the following script in Anaconda prompt
torchrun --standalone --nproc_per_node=2 GPT_train.py
if run on kaggle, run the following in a cell
!torchrun --standalone --nproc_per_node=2 /kaggle/input/script/GPT_train.py
"""

import os
import numpy as np
import wandb
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

@dataclass
class GPTConfig:
    vocab_size: int = 50000
    n_step: int = 7
    n_layer: int = 2
    n_head: int = 4
    model_dim: int = 32
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        assert config.model_dim % config.n_head == 0
        self.model_dim = config.model_dim
        self.n_head = config.n_head
        self.linear_in = nn.Linear(config.model_dim, 3 * config.model_dim, bias=config.bias)  # for q, k, v
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.linear_out = nn.Linear(config.model_dim, config.model_dim, bias=config.bias)
        self.dropout = config.dropout
        # self.register_buffer('tril', torch.tril(torch.ones((config.n_step, config.n_step), dtype=torch.bool)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.linear_in(x).split(self.model_dim, dim=-1)
        q = q.reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, C//nh)
        k = k.reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, C//nh)
        v = v.reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, C//nh)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        # attn = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])  # (B, nh, T, T)
        # attn = attn.masked_fill(self.tril == 0, float('-inf'))  # (B, nh, T, T)
        # attn = F.softmax(attn, dim=-1)
        # attn = self.attn_dropout(attn)
        # out = attn @ v  # (B, nh, T, C//nh)
        out = out.transpose(1, 2).reshape(B, T, C)  # (B, T, C)
        return self.resid_dropout(self.linear_out(out))


def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_in = nn.Linear(config.model_dim, 4 * config.model_dim, bias=config.bias)
        self.linear_out = nn.Linear(4 * config.model_dim, config.model_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        y = self.dropout(self.linear_out(new_gelu(self.linear_in(x))))
        return y


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.ln1 = nn.LayerNorm(config.model_dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.model_dim)
        self.ffn = FFN(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


def get_device():
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
    else:
        devices = [torch.device('cpu')]
    return devices


class GPT(nn.Module):
    def __init__(self, config):
        super(GPT, self).__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.model_dim)
        self.pos_emb = nn.Embedding(config.n_step, config.model_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln = nn.LayerNorm(config.model_dim)
        self.linear = nn.Linear(config.model_dim, config.vocab_size, bias=config.bias)
        self.tok_emb.weight = self.linear.weight  # weight-tying  https://paperswithcode.com/method/weight-tying

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('linear_out.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape
        assert T <= self.config.n_step
        x = self.dropout(self.tok_emb(x) + self.pos_emb(torch.arange(T, dtype=torch.long, device=x.device).repeat(B, 1)))
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)

        if targets is not None:
            logits = self.linear(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
        else:
            logits = self.linear(x[:, [-1], :])  # use [-1] to preserve the sequence dim
            loss = None

        return logits, loss

    def configure_optim(self, learning_rate, betas, weight_decay):
        params_dict = {n: p for n, p in self.named_parameters()}
        params_dict = {n: p for n, p in params_dict.items() if p.requires_grad}
        decay_params = [p for n, p in params_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in params_dict.items() if p.dim() < 2]
        optim_group = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_group, lr=learning_rate, betas=betas, fused=True)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.config.n_step else idx[:, -self.config.n_step:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
            prob = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(prob, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data)-n_step, (batch_size, ))
    x = torch.stack([torch.from_numpy(data[i: i+n_step].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[(i+1): (i+1+n_step)].astype(np.int64)) for i in ix])
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            with ctx:
                logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    if it < warmup_iters:  # first linear warmup for warmup_iters steps
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0-1
    return min_lr + coeff * (learning_rate - min_lr)


def get_model_and_optimizer(init_from, model_args):
    if init_from == 'scratch':
        print('initializing a new model from scratch')
        iter_num = 0
        best_val_loss = 1e9
        gptconfig = GPTConfig(**model_args)
        model = GPT(gptconfig)
        run_id = None
    elif init_from == 'resume':
        print(f'resuming training from {ckpt_dir_load}')
        checkpoint = torch.load(os.path.join(ckpt_dir_load, ckpt_file_load), map_location=device)
        for k in ['vocab_size', 'n_step', 'n_layer', 'n_head', 'model_dim', 'bias']:  # dropout can change
            model_args[k] = checkpoint['model_args'][k]
        gptconfig = GPTConfig(**model_args)
        model = GPT(gptconfig)
        model.load_state_dict(checkpoint['model'])
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        run_id = checkpoint['wandb_run_id']

    model.to(device)
    optimizer = model.configure_optim(learning_rate, (beta1, beta2), weight_decay)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None  # free up memory
    if is_compile:
        print('compiling the model ... (takes a minute')
        t0 = time.time()
        model = torch.compile(model)
        t1 = time.time()
        print(f'compile finished, take time {(t1-t0): .2f}s')
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    return model, optimizer, model_args, iter_num, best_val_loss, run_id


# ----------------------------------------------------------------------
init_from = 'resume'  # 'scratch' or 'resume'.
# if 'resume', change random seed, otherwise, same data will be used for training
data_dir = '/kaggle/input/openwebtext'
ckpt_dir_load = '/kaggle/input/checkpoint-ddp'
ckpt_file_load = 'ckpt_run-t83wyh0n.pt'
ckpt_dir_save = '/kaggle/working/checkpoint'

eval_interval = 125 # evaluation loss and save wandb log and checkpoint
log_interval = 10
eval_iters = 25  # for estimate loss, average the loss from eval_iters iterations
batch_size = 12
gradient_accumulation_steps = 20
# model_args ------------------------------------------------------------
vocab_size = 50304
n_step = 512
n_layer = 12
n_head = 12
model_dim = 768
dropout = 0.0  # from pretraining 0 is good, for finetuning try 0.1+
bias = False
model_args = dict(vocab_size=vocab_size, n_step=n_step, n_layer=n_layer, n_head=n_head, model_dim=model_dim,
                  dropout=dropout, bias=bias)
# adamw optimizer --------------------------------------------------------
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value
# learning rate decay settings -----------------------------
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
# system ---------------------------------------------------
device = get_device()[0]
is_compile = False
ptdtype = torch.float16
# ----------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
wandb_config = {k: globals()[k] for k in config_keys}
# ------------------------------------------------------------------------
ddp = int(os.environ.get('RANK', -1)) != -1   # true or false
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
# -------------------------------------------------------------------------
if master_process:
    os.makedirs(ckpt_dir_save, exist_ok=True)
torch.manual_seed(1337 + seed_offset + 90)  # 注意：如果从checkpoint resume，manual_seed不变，训练的数据会一样
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
ctx = torch.cuda.amp.autocast(dtype=ptdtype)

# data, model, optimizer, scaler, wandb -----------------------------------
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

model, optimizer, model_args, iter_num, best_val_loss, run_id = get_model_and_optimizer(init_from, model_args)
scaler = torch.cuda.amp.GradScaler(enabled=(ptdtype == torch.float16))
if master_process:

    if init_from == 'scratch':
        run = wandb.init(project='openwebtext-gpt2', name=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                         config=wandb_config, save_code=True)
    elif init_from == 'resume':
        run = wandb.init(project='openwebtext-gpt2', id=run_id, resume='must', save_code=True,
                         notes=f'resume from iter {iter_num}')

# training ------------------------------------------------------------
x, y = get_batch('train')
# t0 = time.time()
raw_model = model.module if ddp else model
while iter_num <= max_iters:
    iter_num += 1
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(x, y)
            loss = loss / gradient_accumulation_steps
        x, y = get_batch('train')  # immediately async prefetch next batch while model is doing forward on GPU
        scaler.scale(loss).backward()   # gradient scaling if training in fp16
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    if (iter_num % eval_interval == 0 or iter_num == 1) and master_process:
        losses = estimate_loss()
        print(f"iter {iter_num}: train_loss {losses['train']:.4f}, val_loss {losses['val']:.4f}")
        wandb.log({"iter": iter_num, "train/loss": losses['train'], "val/loss": losses['val'], "lr": lr})

        best_val_loss = losses['val']
        checkpoint = {'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(), 'model_args': model_args,
                      'iter_num': iter_num, 'best_val_loss': best_val_loss,
                      'wandb_run_id': run.id, 'wandb_config': wandb_config}
        print(f"saving checkpoint to {ckpt_dir_save}")
        ckpt_file_save = f'ckpt_run-{run.id}.pt'
        torch.save(checkpoint, os.path.join(ckpt_dir_save, ckpt_file_save))
        wandb.save(os.path.join(ckpt_dir_save, ckpt_file_save))

    # t1 = time.time()
    # dt = t1 - t0
    # t0 = t1
    # if iter_num % log_interval == 0:
    #     lossf = loss.item() * gradient_accumulation_steps
    #     print(f"iter {iter_num}: loss {lossf: .4f}, time {dt*1000: .2f}ms")

if ddp:
    destroy_process_group()