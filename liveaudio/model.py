from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .conditioning import Conditioner
from .dit import DiT
from .layout import Layout
from .vae import StreamingDecoder


def split_state(state, prefix):
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def load_generator(checkpoint, weights_dir, device):
    state = load_file(str(checkpoint))
    with safe_open(str(checkpoint), framework="pt") as f:
        meta = f.metadata()
    layout = Layout(**json.loads(meta["layout"]))
    t_steps = tuple(json.loads(meta["t_steps"]))
    with torch.device("meta"):
        dit, conditioner = DiT(), Conditioner()
    dit.load_state_dict(split_state(state, "model.model."), strict=True, assign=True)
    conditioner.load_state_dict(split_state(state, "conditioner."), strict=True, assign=True)
    dit = dit.eval().requires_grad_(False).to(device)
    conditioner = conditioner.eval().requires_grad_(False).to(device)
    conditioner.load_text_encoder(weights_dir, device)
    return dit, conditioner, layout, t_steps


def load_decoder(weights_dir, device):
    decoder = StreamingDecoder().load_vae_checkpoint(Path(weights_dir) / "soundreactor_vae/sr_causal_ds_1600_dim_64.ckpt")
    decoder = decoder.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    with torch.no_grad(), decoder.streaming(1):
        for _ in range(3):
            decoder(torch.zeros(1, 64, 3, device=device))
    torch.cuda.synchronize()
    return decoder
