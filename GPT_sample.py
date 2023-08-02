import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
import wandb


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


ckpt_file = '/kaggle/input/checkpoint/ckpt-iter13000.pt'
max_new_tokens = 500
temperature = 0.8
top_k = 200
num_samples = 10
is_compile = False

device = get_device()[0]
torch.manual_seed(1337)
torch.cuda.manual_seed_all(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
ctx = torch.cuda.amp.autocast(dtype=torch.float16)


# run = wandb.init(project='openwebtext-gpt2', id='snwgtjui', resume='must')
# checkpoint = torch.load(wandb.restore('ckpt.pt'), map_location=device)
checkpoint = torch.load(ckpt_file, map_location=device)
gptconfig = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconfig)
model.load_state_dict(checkpoint['model'])
model.to(device)
if is_compile:
    model = torch.compile(model)

input_word = input('please input: ')
enc = tiktoken.get_encoding('gpt2')
input_idx = enc.encode(input_word, allowed_special={"<|endoftext|>"})
x = torch.tensor(input_idx, dtype=torch.long, device=device).unsqueeze(dim=0)

model.eval()
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(enc.decode(y[0].tolist()))
            print('-----------------------------')


