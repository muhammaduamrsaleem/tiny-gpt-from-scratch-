"""
grow_model.py
--------------
Continue training your EXISTING tiny_gpt.pt on the original text
PLUS everything from growth_log.txt (your saved conversations).

This is what makes the model "grow" over time — instead of starting
from random numbers, it starts from what it already learned, and
keeps adjusting based on new material.

Run this any time after you've had a few chat sessions:
    python grow_model.py

Honest note: for a model this small, "growth" mostly means it drifts
toward the style/content of whatever you feed it. It won't develop
new reasoning ability — that requires far more scale. But it genuinely
keeps learning from accumulated text, which is real and yours.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

checkpoint_path = 'tiny_gpt.pt'
original_data_file = 'input.txt'
growth_log_path = 'growth_log.txt'

extra_iters = 500          # how many additional training steps to run each growth session
batch_size = 64
eval_interval = 100
learning_rate = 1e-4        # lower than initial training — gentle fine-tuning, not relearning from scratch
eval_iters = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

if not os.path.exists(checkpoint_path):
    raise SystemExit("No tiny_gpt.pt found — run train_tiny_gpt.py first before growing it.")

checkpoint = torch.load(checkpoint_path, map_location=device)
stoi = checkpoint['stoi']
itos = checkpoint['itos']
vocab_size = checkpoint['vocab_size']
cfg = checkpoint['config']
n_embd = cfg['n_embd']
n_head = cfg['n_head']
n_layer = cfg['n_layer']
block_size = cfg['block_size']
dropout = cfg['dropout']

# ---------------------------------------------------------------
# Combine original training text with everything chatted so far
# ---------------------------------------------------------------
with open(original_data_file, 'r', encoding='utf-8') as f:
    text = f.read()

if os.path.exists(growth_log_path):
    with open(growth_log_path, 'r', encoding='utf-8') as f:
        growth_text = f.read()
    print(f"Found {len(growth_text)} characters of conversation history to learn from.")
    text = text + "\n" + growth_text
else:
    print("No growth_log.txt found yet — chat with the model first using chat_with_model.py.")

def encode(s):
    return [stoi[c] for c in s if c in stoi]   # unseen characters are skipped safely

def decode(l):
    return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ---- same architecture as train_tiny_gpt.py ----
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ---------------------------------------------------------------
# Load EXISTING weights (not random!) and keep training
# ---------------------------------------------------------------
model = TinyGPT().to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.train()
print(f"Loaded existing model ({sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters) — continuing its training, not starting over.")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

growth_count = checkpoint.get('growth_sessions', 0) + 1

for it in range(extra_iters):
    if it % eval_interval == 0 or it == extra_iters - 1:
        losses = estimate_loss(model)
        print(f"growth step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save({
    'model_state_dict': model.state_dict(),
    'stoi': stoi,
    'itos': itos,
    'vocab_size': vocab_size,
    'config': cfg,
    'growth_sessions': growth_count,
}, checkpoint_path)

print(f"\nGrowth session #{growth_count} complete. Model updated in place: {checkpoint_path}")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\n--- Sample after this growth session ---")
with torch.no_grad():
    print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))
