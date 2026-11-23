from __future__ import annotations

import math
from pathlib import Path
import time

import av
import numpy as np
import torch

from .engine import BlockEngine, cast_linear_weights_, draw_renoise
from .frontend import ConditionBuffer, FrameSource, VisualEncoder
from .model import add_third_party, initial_conditions, load_decoder, load_generator
from .student import StreamingDiT

SAMPLE_RATE = 48000
MAX_SECONDS = 10.0


def video_seconds(path) -> float:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None:
            return float(stream.duration * stream.time_base)
        return float(container.duration) / av.time_base


class LiveAudio:

    def __init__(self, checkpoint, weights_dir, third_party, device="cuda:0", cuda_graph=True):
        self.device = torch.device(device)
        self.weights_dir = Path(weights_dir)
        self.cuda_graph = cuda_graph
        third_party = add_third_party(third_party, weights_dir)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model, self.layout, self.t_steps = load_generator(checkpoint, self.device)
        cast_linear_weights_(self.model.model.model)
        self.student = StreamingDiT(self.model.model.model, self.layout).to(self.device).eval()
        self.vae, self.decoder = load_decoder(third_party, weights_dir, self.device)
        self.encoder = VisualEncoder(weights_dir, self.device)
        self.engines = {}

    def _engine(self, cond):
        key = (cond["context"].shape[1], cond["n_clip"])
        if key not in self.engines:
            self.engines[key] = BlockEngine(self.student, cond, self.t_steps, cuda_graph=self.cuda_graph)
        return self.engines[key]

    def _noise(self, seed):
        layout = self.layout
        noise = torch.randn(64, layout.num_tokens, generator=torch.Generator().manual_seed(seed))
        noise = noise.transpose(0, 1)[None].contiguous().to(self.device)
        renoise = draw_renoise(layout, len(self.t_steps), torch.Generator().manual_seed(seed))
        return noise, renoise

    @torch.no_grad()
    def generate_from_text(self, prompt, seconds=MAX_SECONDS, seed=0, on_audio=None):
        layout = self.layout
        seconds = min(float(seconds), MAX_SECONDS)
        cond = initial_conditions(self.model, prompt, MAX_SECONDS, 0, layout.num_tokens, self.device)
        engine = self._engine(cond)
        engine.load(cond, *self._noise(seed))
        torch.cuda.synchronize()
        start = time.perf_counter()
        blocks, pcm = [], []
        with self.decoder.streaming(1):
            for b in range(layout.num_blocks):
                begin = time.perf_counter()
                z = engine.denoise(b)
                engine.commit(b)
                y = self.vae.decode(z.float().transpose(1, 2)).float().cpu()
                end = time.perf_counter()
                blocks.append(dict(audio_ready_s=end - start, processing_ms=(end - begin) * 1000))
                pcm.append(y)
                if on_audio is not None:
                    on_audio(y[0].numpy(), layout.starts[b] / 30.)
        audio = torch.cat(pcm, -1)[0, :, :round(seconds * SAMPLE_RATE)].numpy()
        stats = dict(audio_seconds=seconds, wall_seconds=blocks[-1]["audio_ready_s"],
                     first_audio_ms=blocks[0]["audio_ready_s"] * 1000,
                     block_processing_ms=float(np.mean([x["processing_ms"] for x in blocks])))
        return audio, stats

    @torch.no_grad()
    def generate(self, video, prompt=None, seed=0, realtime=False, on_audio=None):
        layout, device = self.layout, self.device
        seconds = min(video_seconds(video), MAX_SECONDS)
        windows = min(layout.num_sync_windows, int(math.floor(seconds * 25 + 1e-6)) // 2)
        clip_count = min(layout.num_clip_frames, max(1, int(math.floor(seconds * 8 + 1e-6))))
        if windows < 1:
            raise ValueError(f"{video} is too short")
        num_sync = max(1, int(math.floor(seconds * 30 + 1e-6)))
        cond = initial_conditions(self.model, prompt, MAX_SECONDS, clip_count, num_sync, device)
        engine = self._engine(cond)
        torch.cuda.synchronize()
        start = time.perf_counter()
        source = FrameSource(video, start, realtime, sync_frames=2 * windows, clip_frames=clip_count)
        engine.load(cond, *self._noise(seed))
        state = ConditionBuffer(self.model.conditioner, engine, cond, seconds, windows, clip_count)
        frames, sync_buffer, blocks, pcm = [], None, [], []
        b = 0
        try:
            with self.decoder.streaming(1):
                for event in source:
                    if event["kind"] == "clip":
                        state.push_clip(event["index"], self.encoder.encode_clip(event["rgb"]))
                    else:
                        frames.append(event["rgb"])
                        if len(frames) == 2:
                            feature, sync_buffer = self.encoder.encode_sync(frames, sync_buffer)
                            state.push_sync(event["index"] // 2, feature)
                            frames.clear()
                    while b < layout.num_blocks and state.ready(b):
                        begin = time.perf_counter()
                        z = engine.denoise(b).float()
                        engine.commit(b)
                        y = self.vae.decode(z.transpose(1, 2)).float().cpu()
                        end = time.perf_counter()
                        s = layout.starts[b]
                        blocks.append(dict(audio_start_s=s / 30., video_ready_s=state.required_video_time(b),
                                           audio_ready_s=end - start, processing_ms=(end - begin) * 1000))
                        pcm.append(y)
                        if on_audio is not None:
                            on_audio(y[0].numpy(), s / 30.)
                        b += 1
        finally:
            source.close()
        if b != layout.num_blocks:
            raise RuntimeError(f"generated {b} of {layout.num_blocks} blocks")
        audio = torch.cat(pcm, -1)[0, :, :round(seconds * SAMPLE_RATE)].numpy()
        wall = blocks[-1]["audio_ready_s"]
        stats = dict(
            video_seconds=seconds, realtime=realtime, wall_seconds=wall,
            first_audio_ms=blocks[0]["audio_ready_s"] * 1000,
            block_processing_ms=float(np.mean([x["processing_ms"] for x in blocks])),
            playout_delay_ms=max((x["audio_ready_s"] - x["audio_start_s"]) * 1000 for x in blocks),
            frames_causal=all(x["source_pts_s"] <= x["target_s"] + 1e-6 for x in source.trace))
        return audio, stats
