from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch

AUDIO_RATE = 30.0
VIDEO_FPS = 25.0
SLICE_S = 2 / VIDEO_FPS
CLIP_FPS = 8.0
CLIP_ARRIVAL_DELAY = 1 / VIDEO_FPS
EPS = 1e-6


@dataclass(frozen=True)
class Layout:
    num_tokens: int = 300
    block_tokens: int = 3
    num_sync_windows: int = 125
    num_clip_frames: int = 80
    hop_frames: int = 2
    clip_visibility: str = "token"

    def __post_init__(self):
        assert self.hop_frames in (2, 4)
        assert self.clip_visibility in ("block", "token")

    @property
    def hop_s(self) -> float:
        return self.hop_frames / VIDEO_FPS

    @cached_property
    def starts(self) -> tuple[int, ...]:
        return tuple(range(0, self.num_tokens, self.block_tokens)) + (self.num_tokens,)

    @property
    def num_blocks(self) -> int:
        return len(self.starts) - 1

    @cached_property
    def token_window(self) -> np.ndarray:
        k = np.floor(np.arange(self.num_tokens) / AUDIO_RATE / SLICE_S + 1e-9).astype(np.int64)
        return np.minimum(k // (self.hop_frames // 2), self.num_sync_windows - 1)

    @cached_property
    def token_ready(self) -> np.ndarray:
        return self.hop_s * (self.token_window + 1)

    @cached_property
    def block_ready(self) -> np.ndarray:
        return np.array([self.token_ready[s:e].max() for s, e in zip(self.starts, self.starts[1:])])

    @cached_property
    def clip_arrival(self) -> np.ndarray:
        return np.arange(self.num_clip_frames) / CLIP_FPS + CLIP_ARRIVAL_DELAY

    @cached_property
    def clip_group(self) -> torch.Tensor:
        out = []
        for a in self.clip_arrival:
            hit = np.nonzero(self.block_ready + EPS >= a)[0]
            out.append(int(hit[0]) if len(hit) else self.num_blocks)
        return torch.tensor(out, dtype=torch.long)

    @cached_property
    def token_clip_visible(self) -> torch.Tensor:
        return torch.from_numpy(self.clip_arrival[None, :] <= self.token_ready[:, None] + EPS)
