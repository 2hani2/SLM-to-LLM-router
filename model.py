import torch
import torch.nn as nn
import math

# ── Configuration ──────────────────────────────────────────────
class SLMConfig:
    vocab_size     = 8000    # number of unique tokens the model knows
    embed_dim      = 256     # size of each token's representation
    num_heads      = 8       # attention heads
    num_layers     = 6       # how many transformer blocks stacked
    ff_dim         = 1024    # size of the feedforward layer inside each block
    max_seq_len    = 256     # maximum input length
    dropout        = 0.1     # regularization (prevents overfitting)

# ── Attention ──────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim  = cfg.embed_dim // cfg.num_heads

        self.q = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.k = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.v = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.out = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q(x).view(B, T, H, D).transpose(1, 2)
        k = self.k(x).view(B, T, H, D).transpose(1, 2)
        v = self.v(x).view(B, T, H, D).transpose(1, 2)

        scale  = math.sqrt(D)
        scores = (q @ k.transpose(-2, -1)) / scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)

# ── Transformer Block ──────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn    = MultiHeadAttention(cfg)
        self.norm1   = nn.LayerNorm(cfg.embed_dim)
        self.norm2   = nn.LayerNorm(cfg.embed_dim)
        self.ff      = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.ff_dim),
            nn.GELU(),
            nn.Linear(cfg.ff_dim, cfg.embed_dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x

# ── Full SLM Model ─────────────────────────────────────────────
class SLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg        = cfg
        self.tok_embed  = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_embed  = nn.Embedding(cfg.max_seq_len, cfg.embed_dim)
        self.blocks     = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm       = nn.LayerNorm(cfg.embed_dim)
        self.head       = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        self.dropout    = nn.Dropout(cfg.dropout)

    def forward(self, idx, mask=None):
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device).unsqueeze(0)

        x = self.dropout(self.tok_embed(idx) + self.pos_embed(positions))

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quick Test ─────────────────────────────────────────────────
if __name__ == "__main__":
    cfg   = SLMConfig()
    model = SLM(cfg)

    print(f"Model created successfully!")
    print(f"Total parameters: {model.count_params():,}")

    # fake input — batch of 2 sentences, each 32 tokens long
    dummy = torch.randint(0, cfg.vocab_size, (2, 32))
    out   = model(dummy)

    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print("model.py is working perfectly!")