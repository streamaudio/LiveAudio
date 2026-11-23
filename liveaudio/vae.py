from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn


class Snake(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1)).to(dtype=x.dtype)
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1)).to(dtype=x.dtype)
        s = torch.sin(x * alpha)
        return x + torch.reciprocal(beta + 1e-9) * (s * s)


class CausalConv1d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias)
        self.context = (kernel_size - 1) * dilation
        self.state = None

    def reset(self, batch_size: int):
        w = self.conv.weight
        self.state = torch.zeros(batch_size, w.shape[1], self.context, dtype=w.dtype, device=w.device)

    def forward(self, x):
        if self.context:
            x = torch.cat([self.state, x], dim=-1)
        y = self.conv(x)
        if self.context:
            self.state = x[..., -self.context:].clone()
        return y


class CausalConvTranspose1d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.convtr = nn.ConvTranspose1d(in_channels, out_channels, 2 * stride, stride)
        self.overlap = stride
        self.partial = None

    def reset(self, batch_size: int):
        w = self.convtr.weight
        self.partial = torch.zeros(batch_size, w.shape[1], self.overlap, dtype=w.dtype, device=w.device)

    def forward(self, x):
        y = self.convtr(x)
        y[..., :self.overlap] += self.partial
        tail = y[..., -self.overlap:]
        tail -= self.convtr.bias[:, None]
        self.partial = tail.clone()
        return y[..., :-self.overlap]


class ResidualUnit(nn.Module):

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.layers = nn.Sequential(Snake(channels), CausalConv1d(channels, channels, 7, dilation),
                                    Snake(channels), CausalConv1d(channels, channels, 1))

    def forward(self, x):
        return self.layers(x) + x


class DecoderBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.layers = nn.Sequential(Snake(in_channels), CausalConvTranspose1d(in_channels, out_channels, stride),
                                    ResidualUnit(out_channels, 1), ResidualUnit(out_channels, 3),
                                    ResidualUnit(out_channels, 9))

    def forward(self, x):
        return self.layers(x)


class StreamingDecoder(nn.Module):

    def __init__(self, latent_dim=64, channels=128, c_mults=(1, 2, 4, 8, 16), strides=(2, 4, 5, 5, 8), out_channels=2):
        super().__init__()
        c_mults = [1] + list(c_mults)
        layers = [CausalConv1d(latent_dim, c_mults[-1] * channels, 7)]
        for i in range(len(c_mults) - 1, 0, -1):
            layers.append(DecoderBlock(c_mults[i] * channels, c_mults[i - 1] * channels, strides[i - 1]))
        layers += [Snake(channels), CausalConv1d(channels, out_channels, 7, bias=False)]
        self.layers = nn.Sequential(*layers)

    def forward(self, z):
        return self.layers(z)

    def stateful(self):
        return [m for m in self.modules() if isinstance(m, (CausalConv1d, CausalConvTranspose1d))]

    @contextmanager
    def streaming(self, batch_size: int = 1):
        for m in self.stateful():
            m.reset(batch_size)
        try:
            yield self
        finally:
            for m in self.stateful():
                m.state = m.partial = None

    def load_vae_checkpoint(self, path):
        source = torch.load(str(path), map_location="cpu", weights_only=True)["state_dict"]
        source = {k[len("decoder."):]: v for k, v in source.items() if k.startswith("decoder.")}
        target = {}
        for name, m in self.named_modules():
            if isinstance(m, (CausalConv1d, CausalConvTranspose1d)):
                inner = "conv" if isinstance(m, CausalConv1d) else "convtr"
                v, g = source.pop(f"{name}.weight_v"), source.pop(f"{name}.weight_g")
                target[f"{name}.{inner}.weight"] = torch._weight_norm(v, g, 0)
                if getattr(m, inner).bias is not None:
                    target[f"{name}.{inner}.bias"] = source.pop(f"{name}.bias")
            elif isinstance(m, Snake):
                target[f"{name}.alpha"] = source.pop(f"{name}.alpha")
                target[f"{name}.beta"] = source.pop(f"{name}.beta")
        if source:
            raise RuntimeError(f"unused VAE decoder weights: {sorted(source)[:5]}")
        self.load_state_dict(target, strict=True)
        return self
