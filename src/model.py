import torch
import torch.nn as nn
import torch.einsum as einsum
from einops import rearrange
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, inner_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.query_norm = nn.LayerNorm(query_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.num_heads = num_heads

    def forward(self, query, context):
        q = self.to_q(query)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d',
                                          h=self.num_heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.num_heads)
        return self.to_out(out)


class LattentTransformer(nn.Module):
    def __init__(self, inner_dim):
        super(LattentTransformer, self).__init__()
        self.query_norm = nn.LayerNorm(inner_dim)
        self.context_norm = nn.LayerNorm(inner_dim)
        self.to_q = nn.Linear(inner_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(inner_dim, inner_dim * 2, bias=False)

    def forward(self, query, context):
        q = self.to_q(query)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d',
                                          h=self.num_heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.num_heads)
        return self.to_out(out)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, inner_dim, mult=4, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(inner_dim)
        self.net = nn.Sequential(
            nn.Linear(inner_dim, inner_dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim * mult, inner_dim))

    def forward(self, x):
        h = self.norm(x)
        return self.net(h)

