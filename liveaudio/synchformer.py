from __future__ import annotations

import einops
from einops import rearrange, repeat
import torch
from torch import einsum, nn


def qkv_attn(q, k, v):
    sim = einsum("b i d, b j d -> b i j", q, k)
    attn = sim.softmax(dim=-1)
    return einsum("b i j, b j d -> b i d", attn, v)


class DividedAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, einops_from, einops_to, **einops_dims):
        h = self.num_heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        q *= self.scale
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))
        cls_out = qkv_attn(cls_q, k, v)
        q_, k_, v_ = map(lambda t: rearrange(t, f"{einops_from} -> {einops_to}", **einops_dims), (q_, k_, v_))
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, "b () d -> (b r) () d", r=r), (cls_k, cls_v))
        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)
        out = qkv_attn(q_, k_, v_)
        out = rearrange(out, f"{einops_to} -> {einops_from}", **einops_dims)
        out = torch.cat((cls_out, out), dim=1)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.proj(out)


class Mlp(nn.Module):

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DividedSpaceTimeBlock(nn.Module):

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = DividedAttention(dim, num_heads)
        self.timeattn = DividedAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, dim * mlp_ratio)
        self.norm3 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x, seq_len: int, num_frames: int):
        time_output = self.timeattn(self.norm3(x), "b (f n) d", "(b n) f d", n=seq_len)
        time_residual = x + time_output
        space_output = self.attn(self.norm1(time_residual), "b (f n) d", "(b f) n d", f=num_frames)
        x = time_residual + space_output
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)


class PatchEmbed3D(nn.Module):

    def __init__(self, img_size=224, in_chans=3, patch_size=16, z_block_size=2, embed_dim=768):
        super().__init__()
        self.height = self.width = img_size // patch_size
        self.z_block_size = z_block_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(z_block_size, patch_size, patch_size),
                              stride=(z_block_size, patch_size, patch_size))

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class SpatialTransformerEncoderLayer(nn.TransformerEncoderLayer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.self_attn.embed_dim))

    def forward(self, x):
        bs, d, t, h, w = x.shape
        x = einops.rearrange(x, "BS D t h w -> (BS t) (h w) D")
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=-2)
        x = super().forward(src=x, src_mask=None)[:, 0, :]
        return einops.rearrange(x, "(BS t) D -> BS t D", BS=bs, t=t)


class MotionFormer(nn.Module):

    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, temporal_resolution=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.temporal_resolution = temporal_resolution
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.patch_embed_3d = PatchEmbed3D(embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.temp_embed = nn.Parameter(torch.zeros(1, temporal_resolution, embed_dim))
        self.blocks = nn.ModuleList([DividedSpaceTimeBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.spatial_attn_agg = SpatialTransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, activation=nn.GELU(), batch_first=True,
            dim_feedforward=mlp_ratio * embed_dim, dropout=0.0, layer_norm_eps=1e-6, norm_first=True)

    def forward(self, x):
        B, S, C, T, H, W = x.shape
        x = x.view(B * S, C, T, H, W)
        x = self.patch_embed_3d(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        npatch = self.patch_embed.num_patches
        cls_embed = self.pos_embed[:, 0, :].unsqueeze(1)
        tile_pos_embed = self.pos_embed[:, 1:, :].repeat(1, self.temporal_resolution, 1)
        tile_temporal_embed = self.temp_embed.repeat_interleave(npatch, 1)
        x = x + torch.cat([cls_embed, tile_pos_embed + tile_temporal_embed], dim=1)
        for blk in self.blocks:
            x = blk(x, seq_len=npatch, num_frames=self.temporal_resolution)
        x = self.norm(x[:, 1:, :])
        t = T // self.patch_embed_3d.z_block_size
        x = x.permute(0, 2, 1).view(B * S, self.embed_dim, t, self.patch_embed_3d.height, self.patch_embed_3d.width)
        x = self.spatial_attn_agg(x)
        return x.view(B, S, *x.shape[1:])


class Synchformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.vfeat_extractor = MotionFormer()

    def forward(self, vis):
        return self.vfeat_extractor(vis.permute(0, 1, 3, 2, 4, 5))

    def load_state_dict(self, state_dict, strict: bool = True):
        state_dict = {k: v for k, v in state_dict.items() if k.startswith("vfeat_extractor.")}
        return super().load_state_dict(state_dict, strict)
