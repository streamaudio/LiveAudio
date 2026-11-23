from __future__ import annotations

import json
from pathlib import Path
import queue
import tempfile
import threading
import time

import av
import numpy as np
import torch
from torchvision.transforms import v2

from .synchformer import Synchformer


class FrameSource:

    def __init__(self, path, start: float, realtime: bool, sync_frames: int, clip_frames: int, capacity=8):
        self.path, self.start, self.realtime = Path(path), start, realtime
        self.events = sorted([(i / 25., "sync", i) for i in range(sync_frames)]
                             + [(i / 8., "clip", i) for i in range(clip_frames)],
                             key=lambda x: (x[0], x[1]))
        self.queue = queue.Queue(capacity)
        self.stop = threading.Event()
        self.trace = []
        self.thread = threading.Thread(target=self._produce, daemon=True)

    def _put(self, value):
        while not self.stop.is_set():
            try:
                self.queue.put(value, timeout=.1)
                return
            except queue.Full:
                pass

    def _produce(self):
        try:
            with av.open(str(self.path)) as container:
                stream = container.streams.video[0]
                stream.codec_context.thread_count = 2
                time_base = float(stream.time_base)
                origin = stream.start_time or 0
                decoded = iter(container.decode(stream))

                def read():
                    for frame in decoded:
                        if frame.pts is not None:
                            return frame, (frame.pts - origin) * time_base
                    return None

                candidate = read()
                if candidate is None:
                    raise RuntimeError(f"no video frames in {self.path}")
                following = None
                cached_frame, cached_rgb = None, None
                for target, kind, index in self.events:
                    if self.stop.is_set():
                        return
                    ready = target + .04
                    if self.realtime:
                        wait = self.start + ready - time.perf_counter()
                        if wait > 0 and self.stop.wait(wait):
                            return
                    begin = time.perf_counter()
                    if following is None:
                        following = read()
                    while following is not None and following[1] <= target + 1e-9:
                        candidate, following = following, read()
                    frame, pts = candidate
                    if pts > target + 1e-6:
                        raise RuntimeError(f"video starts after the requested time: {pts} > {target}")
                    if cached_frame is not frame:
                        cached_frame, cached_rgb = frame, frame.to_ndarray(format="rgb24")
                    end = time.perf_counter()
                    event = dict(kind=kind, index=index, target_s=target, ready_s=ready, source_pts_s=pts,
                                 rgb=cached_rgb, decode_ms=(end - begin) * 1000, delivered_wall_s=end - self.start)
                    self.trace.append({k: v for k, v in event.items() if k != "rgb"})
                    self._put(event)
            self._put(None)
        except BaseException as exc:
            self._put(exc)

    def __iter__(self):
        self.thread.start()
        while True:
            value = self.queue.get(timeout=120)
            if isinstance(value, BaseException):
                raise value
            if value is None:
                return
            yield value

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)


class VisualEncoder:

    def __init__(self, weights_dir, device):
        from open_clip import create_model
        from open_clip.factory import add_model_config
        weights_dir = Path(weights_dir)
        self.sync_transform = v2.Compose([
            v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(224), v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[.5] * 3, std=[.5] * 3)])
        self.clip_transform = v2.Compose([
            v2.Resize((384, 384), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.ToDtype(torch.float32, scale=True)])
        clip_root = weights_dir / "DFN5B-CLIP-ViT-H-14-384"
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "DFN5B_CLIP_ViT_H_14_384.json"
            config.write_text(json.dumps(json.loads((clip_root / "open_clip_config.json").read_text())["model_cfg"]))
            add_model_config(config)
            self.clip = create_model(config.stem, pretrained=str(clip_root / "open_clip_pytorch_model.bin"), pretrained_hf=False)
        self.clip.to(device=device, dtype=torch.float16).eval().requires_grad_(False)
        self.sync = Synchformer()
        self.sync.load_state_dict(torch.load(weights_dir / "synchformer_state_dict.pth", map_location="cpu", weights_only=True))
        self.sync.to(device=device, dtype=torch.float16).eval().requires_grad_(False)
        self.device = device
        self.mean = torch.tensor([.48145466, .4578275, .40821073], device=device, dtype=torch.float32)[None, :, None, None]
        self.std = torch.tensor([.26862954, .26130258, .27577711], device=device, dtype=torch.float32)[None, :, None, None]
        for _ in range(3):
            self.sync(torch.zeros(1, 1, 16, 3, 224, 224, device=device, dtype=torch.float16))
            self.clip.encode_image(torch.zeros(1, 3, 384, 384, device=device, dtype=torch.float16), normalize=True)
        torch.cuda.synchronize()

    def _pixels(self, frames, transform):
        frames = np.ascontiguousarray(frames)
        return transform(torch.from_numpy(frames).to(self.device).permute(0, 3, 1, 2))

    @torch.no_grad()
    def encode_clip(self, rgb):
        pixels = self._pixels(rgb[None], self.clip_transform)
        return self.clip.encode_image(((pixels - self.mean) / self.std).half(), normalize=True).to(torch.float16)

    @torch.no_grad()
    def encode_sync(self, frames, previous):
        new = self._pixels(np.stack(frames), self.sync_transform).to(torch.float16)
        if previous is None:
            buffer = torch.cat([new[:1].expand(14, -1, -1, -1), new], dim=0)
        else:
            buffer = torch.cat([previous[-14:], new], dim=0)
        return self.sync(buffer[None, None])[0, 0, -1:].to(torch.float16), buffer


class ConditionBuffer:

    def __init__(self, conditioner, engine, initial, seconds, num_sync_windows, num_clip_frames):
        self.engine = engine
        self.student, self.layout = engine.m, engine.layout
        layout = self.layout
        self.ctx = initial["context"].clone()
        self.raw_clip = self.ctx.new_zeros(1, layout.num_clip_frames, 1024)
        self.raw_sync = self.ctx.new_zeros(1, layout.num_tokens, 768)
        self.clip_conditioner = conditioner.conditioners["clip_feature"]
        self.sync_conditioner = conditioner.conditioners["sync_feature"]
        self.token_window = torch.as_tensor(layout.token_window, device=self.ctx.device)
        self.num_sync_windows = int(num_sync_windows)
        self.num_clip_frames = int(num_clip_frames)
        self.valid_tokens = min(layout.num_tokens, int(np.floor(seconds * 30 + 1e-6)))
        assert 0 < self.num_sync_windows <= layout.num_sync_windows
        assert 0 < self.num_clip_frames <= layout.num_clip_frames and self.valid_tokens > 0
        self.aligned_window = self.token_window.clamp(max=self.num_sync_windows - 1)
        self.valid_mask = torch.arange(layout.num_tokens, device=self.ctx.device) < self.valid_tokens
        self.seen_clip, self.seen_window = 0, -1

    @torch.no_grad()
    def push_clip(self, index, feature):
        assert index == self.seen_clip
        self.raw_clip[:, index].copy_(feature.reshape(1, 1024).float())
        cc = self.clip_conditioner
        clip = cc.proj_out(self.raw_clip)
        if cc.use_abs_pos_emb:
            clip = clip + cc.abs_pos_emb(clip)
        self.ctx[:, :self.engine.n_clip].copy_(clip)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            kv = self.student.context_kv(self.student.encode_context(self.ctx))
        for (dk, dv), (sk, sv) in zip(self.engine.ctx_kv, kv):
            dk.copy_(sk)
            dv.copy_(sv)
        self.seen_clip += 1

    @torch.no_grad()
    def push_sync(self, window, feature):
        assert window == self.seen_window + 1
        self.raw_sync[:, (self.aligned_window == window) & self.valid_mask] = feature.reshape(1, 1, 768).float()
        sc = self.sync_conditioner
        projected = sc.proj_out(self.raw_sync)
        if sc.use_abs_pos_emb:
            projected = projected + sc.abs_pos_emb(projected)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            modulation = self.student.sync_modulation(projected)
        self.engine.sync6.copy_(modulation)
        self.seen_window = window

    def requirements(self, block):
        layout = self.layout
        s, e = layout.starts[block:block + 2]
        if s < self.valid_tokens:
            window = min(int(layout.token_window[s:min(e, self.valid_tokens)].max()), self.num_sync_windows - 1)
        else:
            window = self.num_sync_windows - 1
        clips = min(int((layout.clip_group <= block).sum()), self.num_clip_frames)
        return window, clips

    def required_video_time(self, block):
        window, clips = self.requirements(block)
        return max((window + 1) * self.layout.hop_s, float(self.layout.clip_arrival[clips - 1]) if clips else 0.)

    def ready(self, block):
        window, clips = self.requirements(block)
        return self.seen_window >= window and self.seen_clip >= clips
