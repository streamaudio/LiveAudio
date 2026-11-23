from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import soundfile as sf

from liveaudio.pipeline import SAMPLE_RATE, LiveAudio

ROOT = Path(__file__).resolve().parent


def mux(video: Path, wav: Path, target: Path):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(target)], check=True)


def main():
    p = argparse.ArgumentParser(description="LiveAudio: streaming audio generation from a video, a video and a text prompt, or a text prompt.")
    p.add_argument("videos", nargs="*", type=Path, help="input video files (the first 10 seconds are used); none for text-to-audio")
    p.add_argument("--prompt", default=None, help="text prompt, used with the video or alone")
    p.add_argument("--seconds", type=float, default=10.0, help="length of the audio for text-to-audio (at most 10)")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--realtime", action="store_true", help="release the video frames at their real arrival times")
    p.add_argument("--mux", action="store_true", help="also write the video with the generated audio")
    p.add_argument("--no-cuda-graph", action="store_true", help="run without CUDA graphs (slower, same output)")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "weights/liveaudio.safetensors")
    p.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if not a.videos and not a.prompt:
        p.error("give at least one video or a --prompt")
    if a.mux and shutil.which("ffmpeg") is None:
        p.error("--mux needs ffmpeg")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    pipe = LiveAudio(a.checkpoint, a.weights_dir, device=a.device, cuda_graph=not a.no_cuda_graph)
    if not a.videos:
        audio, stats = pipe.generate_from_text(a.prompt, seconds=a.seconds, seed=a.seed)
        name = re.sub(r"[^a-z0-9]+", "_", a.prompt.lower()).strip("_")[:60] or "audio"
        wav = a.output_dir / (name + ".wav")
        sf.write(wav, audio.T, SAMPLE_RATE, subtype="FLOAT")
        print(json.dumps(dict(prompt=a.prompt, output=str(wav), **{k: round(v, 3) for k, v in stats.items()})), flush=True)
    for video in a.videos:
        audio, stats = pipe.generate(video, prompt=a.prompt, seed=a.seed, realtime=a.realtime)
        wav = a.output_dir / (video.stem + ".wav")
        sf.write(wav, audio.T, SAMPLE_RATE, subtype="FLOAT")
        if a.mux:
            clipped = a.output_dir / (video.stem + ".mux.wav")
            sf.write(clipped, np.clip(audio.T, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16")
            try:
                mux(video, clipped, a.output_dir / (video.stem + ".mp4"))
            finally:
                clipped.unlink()
        print(json.dumps(dict(video=str(video), output=str(wav), **{k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()})), flush=True)


if __name__ == "__main__":
    main()
