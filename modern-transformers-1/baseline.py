import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import json

# hyperparameters
batch_size = 16
context_size = 128
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
embed_size = 96
num_heads = 8  # head_size = 96/8 = 12
n_layer = 8
dropout = 0.0
# ------------

torch.manual_seed(42)

with open("./input.txt", "r", encoding="utf-8") as f:
    text = f.read()
    text = "".join(text.split("\n"))  # not for tinyshakeseare

chars = sorted(list(set(text)))
vocab_size = len(chars)

# encoder: character to index, decoder: index to character
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Train and test splits at 90%
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i : i + context_size] for i in ix])
    y = torch.stack([data[i + 1 : i + context_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = []
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out.append(losses.mean())
    model.train()
    return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads) -> None:
        super().__init__()
        self.attn = nn.Linear(embed_size, 3 * embed_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(context_size, context_size)).view(
                1, 1, context_size, context_size
            ),
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.rsid_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_size, embed_size)
        self.num_heads = num_heads

    def forward(self, x):
        B, C, E = x.size()  # B for batch size, C for context size, E for embedding size
        NH = self.num_heads  # NH for number of heads
        attn_out = self.attn(x)  # (B,C,E) --> (B,C,3E)
        k, q, v = attn_out.split(embed_size, dim=2)  # (B,C,E)
        k = k.view(B, C, NH, E // NH).transpose(
            1, 2
        )  # (B,C,NH,HS) --> (B,NH,C,HS) # HS for head size
        q = q.view(B, C, NH, E // NH).transpose(1, 2)
        v = v.view(B, C, NH, E // NH).transpose(1, 2)
        wei = (
            q @ k.transpose(-2, -1) * (E // NH) ** -0.5
        )  # (B,NH,C,HS) @ (B,NH,HS,C) --> (B,NH,C,C)
        wei = wei.masked_fill(self.tril[:, :, :C, :C] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)  # (B,NH,C,C)
        out = wei @ v  # (B,NH,C,C) @ (B,NH,C,HS) --> (B,NH,C,HS)
        out = (
            out.transpose(1, 2).contiguous().view(B, C, E)
        )  # concat --> (B,C,E) where E is NH*HS
        proj_out = self.rsid_dropout(self.proj(out))  # (B,C,E)
        return proj_out


class FeedForward(nn.Module):
    def __init__(self, embed_size) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size, bias=False),  # scale hidden size
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, embed_size, num_heads) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(num_heads)
        self.ffwd = FeedForward(embed_size)
        self.ln_mha = nn.LayerNorm(embed_size)
        self.ln_ff = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.mha(self.ln_mha(x))  # pre-norm
        x = x + self.ffwd(self.ln_ff(x))
        return x


class MinGPT(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding_table = nn.Embedding(context_size, embed_size)
        self.blocks = nn.Sequential(
            *[Block(embed_size, num_heads) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, idx, targets=None):
        B, C = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,C,E)
        pos_emb = self.positional_embedding_table(
            torch.arange(C, device=device)
        )  # (C,E)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, C, E = (
                logits.shape
            )  # B for batch size, C for context size, E for embed size
            logits = logits.view(B * C, E)
            targets = targets.view(B * C)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


model = MinGPT()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
baseline_losses = {"train": [], "val": []}

for iter in tqdm(range(max_iters)):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        iter_losses = estimate_loss()
        print(
            f"step {iter}: train loss {iter_losses[0]:.4f}, val loss {iter_losses[1]:.4f}"
        )
        baseline_losses["train"].append(iter_losses[0].item())
        baseline_losses["val"].append(iter_losses[1].item())
    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# save model weights
sp = "./baseline_model_small_tinyshakespeare.pt"
torch.save(model.state_dict(), sp)

# dump exp losses
with open("./baseline_exp_small_tinyshakespeare.json", "w") as f:
    json.dump(baseline_losses, f)
