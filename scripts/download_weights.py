from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_REPO = "YuXuAN0622/LiveAudio"
T5_REPO = "google/flan-t5-base"
T5_FILES = ["config.json", "generation_config.json", "special_tokens_map.json", "spiece.model", "tokenizer.json",
            "tokenizer_config.json", "model.safetensors"]
CLIP_REPO = "apple/DFN5B-CLIP-ViT-H-14-384"
CLIP_FILES = ["open_clip_config.json", "open_clip_pytorch_model.bin"]
SYNCHFORMER_URL = "https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth"
VAE_DRIVE_ID = "1iIxSpfUwxdN2w4e88uDoKM09JYKQCROT"

SHA256 = {
    "liveaudio.safetensors": "9b52633cec6f4651f890a8a7d870bc611bfc2d497addb58b34805895fe507daf",
    "flan-t5-base/model.safetensors": "1dfb70afdcedceb9f9fae2f9b68e004ad934361fb35b9b2bd50b45ea90790fc8",
    "DFN5B-CLIP-ViT-H-14-384/open_clip_pytorch_model.bin": "c07a17b547d461c60a3cce5062b26bf8545b13de602c4c59d8490361eb716033",
    "synchformer_state_dict.pth": "8aff082f2df5c3bc52759db0c865c7ee772ae6400b860d1b7e90413f2defb67c",
    "soundreactor_vae/sr_causal_ds_1600_dim_64.ckpt": "f6d772a4779c0ad6641f6f0e429c24bb993394cb30572a39294d3b4261fb7d92",
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
        raise RuntimeError(f"{root / name}: SHA-256 {actual}, expected {expected}; delete the file and run again")
    print(f"ok  {name}", flush=True)


def from_hub(root: Path, repo: str, folder: str | None, files: list[str]):
    from huggingface_hub import hf_hub_download
    target = root / folder if folder else root
    for name in files:
        if not (target / name).is_file():
            print(f"downloading {repo}/{name}", flush=True)
            hf_hub_download(repo, name, local_dir=str(target))
        verify(root, f"{folder}/{name}" if folder else name)


def from_url(root: Path, name: str, fetch):
    target = root / name
    if not target.is_file():
        print(f"downloading {name}", flush=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        fetch(partial)
        partial.replace(target)
    verify(root, name)


def fetch_synchformer(path: Path):
    with urllib.request.urlopen(SYNCHFORMER_URL) as src, path.open("wb") as dst:
        shutil.copyfileobj(src, dst, 16 << 20)


def fetch_vae(path: Path):
    import gdown
    try:
        result = gdown.download(id=VAE_DRIVE_ID, output=str(path))
    except Exception as error:
        raise RuntimeError("the VAE download from Google Drive failed, see the README for the manual download") from error
    if result is None:
        raise RuntimeError("the VAE download from Google Drive failed, see the README for the manual download")


def main():
    p = argparse.ArgumentParser(description="Download the LiveAudio checkpoint, the pretrained encoders and the audio VAE.")
    p.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    p.add_argument("--checkpoint-repo", default=CHECKPOINT_REPO, help="Hugging Face repository of the LiveAudio checkpoint")
    a = p.parse_args()
    root = a.weights_dir
    root.mkdir(parents=True, exist_ok=True)
    from_hub(root, a.checkpoint_repo, None, ["liveaudio.safetensors"])
    from_hub(root, T5_REPO, "flan-t5-base", T5_FILES)
    from_hub(root, CLIP_REPO, "DFN5B-CLIP-ViT-H-14-384", CLIP_FILES)
    from_url(root, "synchformer_state_dict.pth", fetch_synchformer)
    from_url(root, "soundreactor_vae/sr_causal_ds_1600_dim_64.ckpt", fetch_vae)
    print(f"all weights are in {root}", flush=True)


if __name__ == "__main__":
    main()
