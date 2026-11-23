from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .layout import Layout

CONFIG = Path(__file__).with_name("config.json")
VAE_CONFIG = "modules_causal/stable_audio_tools/configs/model_configs/autoencoders/sr_ds_1600_dim_64_causal.json"


def add_third_party(third_party, weights_dir):
    third_party = Path(third_party).resolve()
    for name in ("Omni2Sound", "soundreactor_vae"):
        root = third_party / name
        if not root.is_dir():
            raise FileNotFoundError(f"{root} not found, run scripts/setup_third_party.sh first")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    os.environ["V2A_WEIGHTS_DIR"] = str(Path(weights_dir).resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return third_party


def load_generator(checkpoint, device):
    from stable_audio_tools.models.factory import create_model_from_config
    model = create_model_from_config(json.loads(CONFIG.read_text()))
    model.load_state_dict(load_file(str(checkpoint)), strict=True)
    with safe_open(str(checkpoint), framework="pt") as f:
        meta = f.metadata()
    layout = Layout(**json.loads(meta["layout"]))
    t_steps = tuple(json.loads(meta["t_steps"]))
    model.eval().requires_grad_(False).to(device)
    return model, layout, t_steps


def load_decoder(third_party, weights_dir, device):
    from modules_causal.stable_audio_tools.models import create_model_from_config as create_vae
    from modules_causal.stable_audio_tools.models.autoencoders import StreamingOobleckDecoder
    from modules_causal.stable_audio_tools.models.utils import load_ckpt_state_dict
    root = Path(third_party) / "soundreactor_vae"
    spec = importlib.util.spec_from_file_location("soundreactor_online_utils", root / "utils/online_utils.py")
    online = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(online)
    vae = create_vae(json.loads((root / VAE_CONFIG).read_text()))
    ckpt = Path(weights_dir) / "soundreactor_vae/sr_causal_ds_1600_dim_64.ckpt"
    vae.load_state_dict(load_ckpt_state_dict(str(ckpt)), strict=True)
    vae.eval().requires_grad_(False)
    old = vae.decoder
    decoder = StreamingOobleckDecoder(
        out_channels=old.out_channels, channels=old.channels, latent_dim=old.latent_dim, c_mults=old.input_c_mults,
        strides=old.strides, use_snake=old.use_snake, antialias_activation=old.antialias_activation,
        use_nearest_upsample=old.use_nearest_upsample, final_tanh=old.final_tanh, causal=True)
    decoder.initialize_from_conversion(old)
    online.strip_weight_norm_streaming_(decoder)
    vae.encoder = None
    vae.decoder = decoder.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    vae.to(device)
    with torch.no_grad(), decoder.streaming(1):
        for _ in range(3):
            vae.decode(torch.zeros(1, 64, 3, device=device))
    torch.cuda.synchronize()
    return vae, decoder


@torch.no_grad()
def initial_conditions(model, prompt, seconds, num_clip, num_sync, device):
    video = num_clip > 0
    task = ("V" if video else "") + ("T" if prompt else "") + "A"
    meta = {"task": task, "seconds_start": 0.0, "seconds_total": seconds,
            "sync_feature": torch.zeros(num_sync, 768, device=device)}
    required = {"seconds_start", "seconds_total", "sync_feature"}
    if video:
        meta["clip_feature"] = torch.zeros(num_clip, 1024, device=device)
        required.add("clip_feature")
    if prompt:
        meta["text_prompt"] = prompt
        required.add("text_prompt")
    tensors = model.conditioner([meta], device)
    missing = required - set(tensors)
    if missing:
        raise RuntimeError(f"conditioner produced no output for {sorted(missing)}")
    cond = model.get_conditioning_inputs(tensors)
    return {"context": cond["cross_attn_cond"], "n_clip": tensors["clip_feature"][0].shape[1] if video else 0,
            "global_cond": cond["global_cond"], "sync": cond["additional_cond"]}
