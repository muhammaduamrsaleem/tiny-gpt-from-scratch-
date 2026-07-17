"""
chat_with_model.py
--------------------
Load your trained tiny GPT (tiny_gpt.pt) and generate text with it.
This is the "chat" step — no internet, no API, 100% offline,
100% built and trained by you.

Run:
    python chat_with_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

checkpoint_path = 'tiny_gpt.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def encode(s):
    return [stoi[c] for c in s if c in stoi]

def decode(l):
    return ''.join([itos[i] for i in l])

# ---- must match the architecture in train_tiny_gpt.py exactly ----
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
        return logits, None

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
# Load trained weights and chat
# ---------------------------------------------------------------
model = TinyGPT().to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Loaded model with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters\n")
print("Type a starting phrase and it will continue the text in that style.")
print("Type 'quit' to exit.\n")

while True:
    prompt = input("You: ")
    if prompt.strip().lower() == 'quit':
        break
    if prompt.strip() == '':
        prompt = '\n'

    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(context, max_new_tokens=200)
    print("\nModel:", decode(out[0].tolist()))
    print()
