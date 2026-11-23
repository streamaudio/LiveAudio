from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

T5_REPO = "google/flan-t5-base"
T5_FILES = ["config.json", "generation_config.json", "special_tokens_map.json", "spiece.model", "tokenizer.json",
            "tokenizer_config.json", "model.safetensors"]
CLIP_REPO = "apple/DFN5B-CLIP-ViT-H-14-384"
CLIP_FILES = ["open_clip_config.json", "open_clip_pytorch_model.bin"]
SYNCHFORMER_URL = "https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth"
VAE_DRIVE_ID = "1iIxSpfUwxdN2w4e88uDoKM09JYKQCROT"

SHA256 = {
    "flan-t5-base/model.safetensors": "1dfb70afdcedceb9f9fae2f9b68e004ad934361fb35b9b2bd50b45ea90790fc8",
    "DFN5B-CLIP-ViT-H-14-384/open_clip_pytorch_model.bin": "c07a17b547d461c60a3cce5062b26bf8545b13de602c4c59d8490361eb716033",
    "synchformer_state_dict.pth": "8aff082f2df5c3bc52759db0c865c7ee772ae6400b860d1b7e90413f2defb67c",
    "soundreactor_vae/sr_causal_ds_1600_dim_64.ckpt": "f6d772a4779c0ad6641f6f0e429c24bb993394cb30572a39294d3b4261fb7d92",
    "liveaudio.safetensors": "9b52633cec6f4651f890a8a7d870bc611bfc2d497addb58b34805895fe507daf",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(root: Path, name: str):
    expected = SHA256.get(name)
    if expected is None:
        return
    actual = sha256(root / name)
    if actual != expected:
        raise RuntimeError(f"{name}: sha256 {actual}, expected {expected}")
    print(f"verified {name}")


def from_hub(root: Path, repo: str, folder: str, files: list[str]):
    from huggingface_hub import hf_hub_download
    for name in files:
        if not (root / folder / name).is_file():
            hf_hub_download(repo, name, local_dir=str(root / folder))
        verify(root, f"{folder}/{name}")


def main():
    p = argparse.ArgumentParser(description="Download the pretrained encoders and the audio VAE used by LiveAudio.")
    p.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    p.add_argument("--checkpoint-repo", default=None, help="Hugging Face repository that holds liveaudio.safetensors")
    a = p.parse_args()
    root = a.weights_dir
    root.mkdir(parents=True, exist_ok=True)
    from_hub(root, T5_REPO, "flan-t5-base", T5_FILES)
    from_hub(root, CLIP_REPO, "DFN5B-CLIP-ViT-H-14-384", CLIP_FILES)
    target = root / "synchformer_state_dict.pth"
    if not target.is_file():
        with urllib.request.urlopen(SYNCHFORMER_URL) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, 16 << 20)
    verify(root, "synchformer_state_dict.pth")
    target = root / "soundreactor_vae/sr_causal_ds_1600_dim_64.ckpt"
    if not target.is_file():
        import gdown
        target.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=VAE_DRIVE_ID, output=str(target))
    verify(root, "soundreactor_vae/sr_causal_ds_1600_dim_64.ckpt")
    checkpoint = root / "liveaudio.safetensors"
    if not checkpoint.is_file() and a.checkpoint_repo:
        from huggingface_hub import hf_hub_download
        hf_hub_download(a.checkpoint_repo, "liveaudio.safetensors", local_dir=str(root))
    if checkpoint.is_file():
        verify(root, "liveaudio.safetensors")
    else:
        print(f"{checkpoint} is missing, see the README")


if __name__ == "__main__":
    main()
