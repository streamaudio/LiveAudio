from __future__ import annotations

import time

import torch
import torch.nn as nn

from .student import StreamingDiT, x0_from_v


def cast_linear_weights_(module: nn.Module, dtype=torch.bfloat16) -> int:
    n = 0
    for mod in module.modules():
        if isinstance(mod, (nn.Linear, nn.Conv1d)):
            mod.weight.data = mod.weight.data.to(dtype)
            if mod.bias is not None:
                mod.bias.data = mod.bias.data.to(dtype)
            n += 1
    return n


def draw_renoise(layout, n_steps: int, generator: torch.Generator) -> torch.Tensor:
    out = torch.zeros(max(n_steps - 1, 1), 1, layout.num_tokens, 64)
    for s, e in zip(layout.starts, layout.starts[1:]):
        for i in range(n_steps - 1):
            out[i, :, s:e] = torch.randn((1, e - s, 64), generator=generator, device="cpu")
    return out


class BlockEngine:

    def __init__(self, model: StreamingDiT, cond, t_steps, cuda_graph=True, dtype=torch.bfloat16):
        self.m, self.layout, self.dtype = model, model.layout, dtype
        self.t_steps = tuple(float(t) for t in t_steps)
        self.cuda_graph = cuda_graph
        dev = cond["context"].device
        self.n_ctx, self.n_clip = cond["context"].shape[1], cond["n_clip"]
        m, layout = model, self.layout
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            ctx_kv = m.context_kv(m.encode_context(cond["context"]))
            sync6, zero6 = m.sync_modulation(cond["sync"]), m.zero_modulation(cond["sync"])
            self.freqs, self.vis = [], []
            for b in range(layout.num_blocks):
                s, e = layout.starts[b], layout.starts[b + 1]
                self.freqs.append(m.rotary(torch.cat([torch.zeros(1, dtype=torch.long), torch.arange(s + 1, e + 1)])))
                blocks = torch.full((1 + e - s,), b, device=dev, dtype=torch.long)
                tok = torch.cat([torch.tensor([-1]), torch.arange(s, e)]).to(dev)
                self.vis.append(m.context_visible(self.n_ctx, self.n_clip, blocks, tok))
        self.ctx_kv = [(k.clone(), v.clone()) for k, v in ctx_kv]
        self.sync6, self.zero6 = sync6.clone(), zero6.clone()
        self.global_cond = cond["global_cond"].clone()
        self.ctx_mask = torch.ones(1, self.n_ctx, dtype=torch.bool, device=dev)
        self.noise = torch.zeros(1, layout.num_tokens, 64, device=dev)
        self.renoise = torch.zeros(max(len(self.t_steps) - 1, 1), 1, layout.num_tokens, 64, device=dev)
        self.out = torch.zeros(1, layout.num_tokens, 64, device=dev)
        heads, dim = m.heads, m.tr.dim
        self.kc = [torch.zeros(1, heads, layout.num_tokens, dim // heads, device=dev, dtype=dtype) for _ in m.tr.layers]
        self.vc = [torch.zeros_like(k) for k in self.kc]
        self.capture_s = self._capture() if cuda_graph else 0.0

    def _forward(self, b, x, t, commit):
        m, layout = self.m, self.layout
        s, e = layout.starts[b], layout.starts[b + 1]
        tt = torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
        h = torch.cat([m.time_tokens(tt.to(x.dtype)[:, None], self.global_cond), m.audio_in(x)], dim=1)
        mod = torch.cat([self.zero6, self.sync6[:, s:e]], dim=1)
        cross_mask = (self.vis[b][None] & self.ctx_mask[:, None, :])[:, None]
        for l, layer in enumerate(m.tr.layers):
            prefix = None if s == 0 else (self.kc[l][:, :, :s], self.vc[l][:, :, :s])
            h, (k, v) = m.layer_forward(layer, h, layer.to_scale_shift_gate + mod, self.freqs[b], ctx_kv=self.ctx_kv[l],
                                        cross_mask=cross_mask, self_kv=prefix)
            if commit:
                self.kc[l][:, :, s:e].copy_(k[:, :, 1:])
                self.vc[l][:, :, s:e].copy_(v[:, :, 1:])
        return m.audio_out(h[:, 1:])

    def _denoise(self, b):
        s, e = self.layout.starts[b], self.layout.starts[b + 1]
        ts = self.t_steps
        x = self.noise[:, s:e]
        for i, t in enumerate(ts):
            tt = torch.full((x.shape[0], 1, 1), t, device=x.device, dtype=x.dtype)
            x0 = x0_from_v(x, self._forward(b, x, t, commit=False), tt)
            if i + 1 == len(ts):
                x = x0
            else:
                x = (1 - ts[i + 1]) * x0 + ts[i + 1] * self.renoise[i, :, s:e]
        self.out[:, s:e].copy_(x)

    def _commit(self, b):
        s, e = self.layout.starts[b], self.layout.starts[b + 1]
        self._forward(b, self.out[:, s:e], 0.0, commit=True)

    @torch.no_grad()
    def _capture(self):
        t0 = time.perf_counter()
        nb = self.layout.num_blocks
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.autocast("cuda", dtype=self.dtype, cache_enabled=False):
            for b in range(nb):
                self._denoise(b)
                self._commit(b)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.pool = torch.cuda.graph_pool_handle()
        self.g_denoise, self.g_commit = [], []
        for b in range(nb):
            for fn, graphs in ((self._denoise, self.g_denoise), (self._commit, self.g_commit)):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self.pool), torch.autocast("cuda", dtype=self.dtype, cache_enabled=False):
                    fn(b)
                graphs.append(g)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    @torch.no_grad()
    def load(self, cond, noise, renoise):
        m = self.m
        assert cond["context"].shape[1] == self.n_ctx and cond["n_clip"] == self.n_clip
        with torch.autocast("cuda", dtype=self.dtype):
            ctx_kv = m.context_kv(m.encode_context(cond["context"]))
            sync6, zero6 = m.sync_modulation(cond["sync"]), m.zero_modulation(cond["sync"])
        pairs = [(d, s) for (dk, dv), (sk, sv) in zip(self.ctx_kv, ctx_kv) for d, s in ((dk, sk), (dv, sv))]
        pairs += [(self.sync6, sync6), (self.zero6, zero6), (self.global_cond, cond["global_cond"]),
                  (self.noise, noise), (self.renoise, renoise)]
        for dst, src in pairs:
            assert dst.shape == src.shape and dst.dtype == src.dtype
            dst.copy_(src, non_blocking=True)

    @torch.no_grad()
    def denoise(self, b):
        if self.cuda_graph:
            self.g_denoise[b].replay()
        else:
            with torch.autocast("cuda", dtype=self.dtype, cache_enabled=False):
                self._denoise(b)
        s, e = self.layout.starts[b], self.layout.starts[b + 1]
        return self.out[:, s:e]

    @torch.no_grad()
    def commit(self, b):
        if self.cuda_graph:
            self.g_commit[b].replay()
        else:
            with torch.autocast("cuda", dtype=self.dtype, cache_enabled=False):
                self._commit(b)
