from __future__ import annotations

from functools import reduce
import math

from einops import rearrange
import torch
from torch import nn
from torch.amp import autocast
import torch.nn.functional as F


class FourierFeatures(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features // 2, in_features))

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class RotaryEmbedding(nn.Module):

    def __init__(self, dim: int, base: float = 10000.):
        super().__init__()
        self.register_buffer("inv_freq", 1. / (base ** (torch.arange(0, dim, 2).float() / dim)))

    @autocast("cuda", enabled=False)
    def forward(self, t):
        t = t.to(torch.float32)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1), 1.


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


@autocast("cuda", enabled=False)
def apply_rotary_pos_emb(t, freqs, scale=1):
    out_dtype = t.dtype
    dtype = reduce(torch.promote_types, (t.dtype, freqs.dtype, torch.float32))
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs, t = freqs.to(dtype), t.to(dtype)
    freqs = freqs[-seq_len:, :]
    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, "b n d -> b 1 n d")
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    t, t_unrotated = t.to(out_dtype), t_unrotated.to(out_dtype)
    return torch.cat((t, t_unrotated), dim=-1)


class LayerNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], weight=self.gamma, bias=self.beta, eps=self.eps)


class GLU(nn.Module):

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.act = nn.SiLU()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate)


class FeedForward(nn.Module):

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner = int(dim * mult)
        self.ff = nn.Sequential(GLU(dim, inner), nn.Identity(), nn.Linear(inner, dim), nn.Identity())

    def forward(self, x):
        return self.ff(x)


class Attention(nn.Module):

    def __init__(self, dim: int, dim_heads: int, dim_context: int | None = None):
        super().__init__()
        self.num_heads = dim // dim_heads
        if dim_context is None:
            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        else:
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_kv = nn.Linear(dim_context, dim_context * 2, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)


class TransformerBlock(nn.Module):

    def __init__(self, dim: int, dim_heads: int, dim_context: int):
        super().__init__()
        self.pre_norm = LayerNorm(dim)
        self.self_attn = Attention(dim, dim_heads)
        self.cross_attend_norm = LayerNorm(dim)
        self.cross_attn = Attention(dim, dim_heads, dim_context)
        self.ff_norm = LayerNorm(dim)
        self.ff = FeedForward(dim)
        self.to_scale_shift_gate = nn.Parameter(torch.zeros(6 * dim))


class Transformer(nn.Module):

    def __init__(self, dim: int, depth: int, dim_heads: int, dim_in: int, dim_out: int):
        super().__init__()
        self.dim = dim
        self.project_in = nn.Linear(dim_in, dim, bias=False)
        self.project_out = nn.Linear(dim, dim_out, bias=False)
        self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))
        self.global_cond_embedder = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim * 6))
        self.layers = nn.ModuleList([TransformerBlock(dim, dim_heads, dim) for _ in range(depth)])


def projection(dim_in: int, dim: int, bias: bool = False):
    return nn.Sequential(nn.Linear(dim_in, dim, bias=bias), nn.SiLU(), nn.Linear(dim, dim, bias=bias))


class DiT(nn.Module):

    def __init__(self, io_channels=64, embed_dim=1536, depth=24, num_heads=24, cond_token_dim=1024,
                 global_cond_dim=2048, additional_cond_dim=1024):
        super().__init__()
        self.timestep_features = FourierFeatures(1, 256)
        self.to_timestep_embed = projection(256, embed_dim, bias=True)
        self.to_cond_embed = projection(cond_token_dim, embed_dim)
        self.to_global_embed = projection(global_cond_dim, embed_dim)
        self.to_additional_embed = projection(additional_cond_dim, embed_dim)
        self.transformer = Transformer(embed_dim, depth, embed_dim // num_heads, io_channels, io_channels)
        self.preprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
