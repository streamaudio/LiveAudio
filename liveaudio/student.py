from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .layout import Layout


def split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    return rearrange(x, "b n (h d) -> b h n d", h=heads)


def attend(q, k, v, mask=None) -> torch.Tensor:
    out = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(), attn_mask=mask)
    return rearrange(out, "b h n d -> b n (h d)")


def x0_from_v(x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return x_t - t * v


class StreamingDiT(nn.Module):

    def __init__(self, dit, layout: Layout):
        super().__init__()
        from stable_audio_tools.models.transformer import apply_rotary_pos_emb
        self.apply_rotary = apply_rotary_pos_emb
        self.dit, self.layout = dit, layout
        self.tr = dit.transformer
        self.register_buffer("clip_group", layout.clip_group.clone(), persistent=False)
        self.register_buffer("token_clip_visible", layout.token_clip_visible.clone(), persistent=False)
        self.heads = self.tr.layers[0].self_attn.num_heads
        for layer in self.tr.layers:
            assert layer.self_attn.qk_norm == "none" and layer.cross_attend and layer.conformer is None
            assert layer.global_cond_dim is not None and not layer.self_attn.feat_scale
        assert dit.global_cond_type == "prepend" and dit.timestep_cond_type == "global" and dit.patch_size == 1
        assert dit.preprocess_conv.kernel_size == (1,) and dit.postprocess_conv.kernel_size == (1,)

    def encode_context(self, context):
        return self.dit.to_cond_embed(context)

    def sync_modulation(self, sync):
        return self.tr.global_cond_embedder(self.dit.to_additional_embed(sync))

    def zero_modulation(self, like):
        return self.tr.global_cond_embedder(like.new_zeros(like.shape[0], 1, self.tr.dim))

    def context_kv(self, context):
        out = []
        for layer in self.tr.layers:
            k, v = layer.cross_attn.to_kv(context).chunk(2, dim=-1)
            out.append((split_heads(k, self.heads), split_heads(v, self.heads)))
        return out

    def context_visible(self, n_ctx: int, n_clip: int, blocks: torch.Tensor, token_index: torch.Tensor) -> torch.Tensor:
        vis = torch.ones(len(blocks), n_ctx, dtype=torch.bool, device=blocks.device)
        if n_clip:
            vis[:, :n_clip] = self.clip_group[None, :n_clip].to(blocks.device) <= blocks[:, None]
            if self.layout.clip_visibility == "token":
                tok = token_index.to(blocks.device)
                audio = tok >= 0
                vis[audio, :n_clip] = self.token_clip_visible[tok[audio], :n_clip].to(blocks.device)
        return vis

    def time_tokens(self, t, global_cond):
        bsz, n = t.shape
        emb = self.dit.to_timestep_embed(self.dit.timestep_features(t.reshape(-1, 1))).view(bsz, n, -1)
        return emb + self.dit.to_global_embed(global_cond)[:, None]

    def audio_in(self, x):
        y = x.transpose(1, 2)
        y = self.dit.preprocess_conv(y) + y
        return self.tr.project_in(y.transpose(1, 2))

    def audio_out(self, h):
        y = self.tr.project_out(h).transpose(1, 2)
        y = self.dit.postprocess_conv(y) + y
        return y.transpose(1, 2)

    def rotary(self, positions):
        freqs, _ = self.tr.rotary_pos_emb.forward(positions.to(self.tr.rotary_pos_emb.inv_freq.device))
        return freqs

    def layer_forward(self, layer, x, mod6, freqs, *, ctx_kv, cross_mask, self_kv=None):
        scale_s, shift_s, gate_s, scale_f, shift_f, gate_f = mod6.chunk(6, dim=-1)
        h = layer.pre_norm(x) * (1 + scale_s) + shift_s
        q, k, v = layer.self_attn.to_qkv(h).chunk(3, dim=-1)
        q, k, v = (split_heads(z, self.heads) for z in (q, k, v))
        q = self.apply_rotary(q.float(), freqs.float()).to(q.dtype)
        k = self.apply_rotary(k.float(), freqs.float()).to(k.dtype)
        kk, vv = (k, v) if self_kv is None else (torch.cat([self_kv[0], k], 2), torch.cat([self_kv[1], v], 2))
        a = layer.self_attn.to_out(attend(q, kk, vv))
        x = x + layer.self_attn_scale(a * torch.sigmoid(1 - gate_s))
        qc = split_heads(layer.cross_attn.to_q(layer.cross_attend_norm(x)), self.heads)
        x = x + layer.cross_attn_scale(layer.cross_attn.to_out(attend(qc, ctx_kv[0], ctx_kv[1], cross_mask)))
        h = layer.ff_norm(x) * (1 + scale_f) + shift_f
        x = x + layer.ff_scale(layer.ff(h) * torch.sigmoid(1 - gate_f))
        return x, (k, v)
