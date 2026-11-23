from __future__ import annotations

from math import pi
from pathlib import Path

from einops import rearrange
import torch
from torch import nn

COND_DIM = 1024
MAX_SECONDS = 10
CLIP_FPS = 8
SYNC_RATE = 30
TEXT_TOKENS = 128


class LearnedPositionalEmbedding(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(dim // 2))

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return torch.cat((x, fouriered), dim=-1)


class NumberEmbedder(nn.Module):

    def __init__(self, features: int, dim: int = 256):
        super().__init__()
        self.features = features
        self.embedding = nn.Sequential(LearnedPositionalEmbedding(dim), nn.Linear(dim + 1, features))

    def forward(self, x):
        shape = x.shape
        return self.embedding(rearrange(x, "... -> (...)")).view(*shape, self.features)


class NumberConditioner(nn.Module):

    def __init__(self, min_val: float = 0, max_val: float = 512):
        super().__init__()
        self.min_val, self.max_val = min_val, max_val
        self.embedder = NumberEmbedder(COND_DIM)

    def forward(self, value: float, device):
        x = torch.tensor([float(value)]).to(device).clamp(self.min_val, self.max_val)
        x = (x - self.min_val) / (self.max_val - self.min_val)
        return self.embedder(x).unsqueeze(1)


class AbsolutePositionalEmbedding(nn.Module):

    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.scale = dim ** -0.5
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device)
        return self.emb(pos) * self.scale


class FeatureConditioner(nn.Module):

    def __init__(self, feature_dim: int, length: int, use_abs_pos_emb: bool):
        super().__init__()
        self.length = length
        self.proj_out = nn.Identity() if feature_dim == COND_DIM else nn.Linear(feature_dim, COND_DIM)
        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            self.abs_pos_emb = AbsolutePositionalEmbedding(COND_DIM, length)

    def forward(self, features):
        x = self.proj_out(features)
        if self.use_abs_pos_emb:
            x = x + self.abs_pos_emb(x)
        return x


class TextConditioner(nn.Module):

    def __init__(self):
        super().__init__()
        self.proj_out = nn.Linear(768, COND_DIM)
        self.tokenizer = None
        self.encoder = None

    def load_encoder(self, path, device):
        from transformers import AutoTokenizer, T5EncoderModel
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        encoder = T5EncoderModel.from_pretrained(str(path)).eval().requires_grad_(False).to(torch.float16)
        self.__dict__["encoder"] = encoder.to(device)

    @torch.no_grad()
    def forward(self, text: str, device):
        tokens = self.tokenizer([text], truncation=True, max_length=TEXT_TOKENS, padding="max_length", return_tensors="pt")
        ids = tokens["input_ids"].to(device)
        mask = tokens["attention_mask"].to(device).to(torch.bool)
        embeddings = self.encoder(input_ids=ids, attention_mask=mask)["last_hidden_state"]
        embeddings = self.proj_out(embeddings.to(self.proj_out.weight.dtype))
        return embeddings * mask.unsqueeze(-1).float()


class Conditioner(nn.Module):

    def __init__(self):
        super().__init__()
        self.conditioners = nn.ModuleDict({
            "text_prompt": TextConditioner(),
            "seconds_start": NumberConditioner(),
            "seconds_total": NumberConditioner(),
            "clip_feature": FeatureConditioner(1024, MAX_SECONDS * CLIP_FPS, use_abs_pos_emb=True),
            "sync_feature": FeatureConditioner(768, MAX_SECONDS * SYNC_RATE, use_abs_pos_emb=False),
        })

    def load_text_encoder(self, weights_dir, device):
        self.conditioners["text_prompt"].load_encoder(Path(weights_dir) / "flan-t5-base", device)

    @torch.no_grad()
    def initial(self, prompt, video: bool, seconds: float, device):
        c = self.conditioners
        context = []
        if video:
            context.append(c["clip_feature"](torch.zeros(1, c["clip_feature"].length, 1024, device=device)))
        if prompt:
            context.append(c["text_prompt"](prompt, device))
        start, total = c["seconds_start"](0.0, device), c["seconds_total"](seconds, device)
        context += [start, total]
        sync = c["sync_feature"](torch.zeros(1, c["sync_feature"].length, 768, device=device))
        return {"context": torch.cat(context, dim=1), "n_clip": c["clip_feature"].length if video else 0,
                "global_cond": torch.cat([start, total], dim=-1).squeeze(1), "sync": sync}
