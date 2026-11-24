# LiveAudio

Inference code and checkpoint for **LiveAudio: From Bidirectional to Autoregressive Diffusion for Real-Time Streaming Video-to-Audio Generation**.

Yuxuan Jiang<sup>1</sup>, Zehua Chen<sup>1</sup>, Yusheng Dai<sup>2</sup>, Andong Wang<sup>3</sup>, Weibei Dou<sup>1</sup>, Jun Zhu<sup>1,†</sup>

<sup>1</sup>Tsinghua University, <sup>2</sup>Monash University, <sup>3</sup>Renmin University of China, <sup>†</sup>corresponding author

[Paper](https://arxiv.org/abs/2611.30714) · [Project page](https://streamaudio.github.io/) · [Model](https://huggingface.co/YuXuAN0622/LiveAudio)

LiveAudio generates 48 kHz stereo audio for a video as a stream. The video file is decoded once, frame by frame, and nothing of the video is precomputed. Every 0.1 s block of audio is generated as soon as the frames it depends on have arrived, with four sampling steps and no guidance, and is decoded to a waveform immediately by a causal audio VAE. The same checkpoint covers three tasks: video-to-audio, video-text-to-audio (a text prompt in addition to the video) and text-to-audio (a text prompt only).

## Setup

The code is self-contained; it needs PyTorch and a few pip packages. Tested with Python 3.10, PyTorch 2.1.0 (CUDA 12.1) and one NVIDIA A100 GPU; inference uses about 6 GB of GPU memory.

```bash
conda create -n liveaudio python=3.10 -y
conda activate liveaudio
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`ffmpeg` is needed only for `--mux`.

## Weights

```bash
python scripts/download_weights.py
```

The script downloads the following files into `weights/` and checks their SHA-256:

| File in `weights/` | Source |
| --- | --- |
| `liveaudio.safetensors` | [LiveAudio](https://huggingface.co/YuXuAN0622/LiveAudio) |
| `flan-t5-base/` | [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) |
| `DFN5B-CLIP-ViT-H-14-384/` | [apple/DFN5B-CLIP-ViT-H-14-384](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-384) |
| `synchformer_state_dict.pth` | [MMAudio release v0.1](https://github.com/hkchengrex/MMAudio/releases/tag/v0.1) |
| `soundreactor_vae/sr_causal_ds_1600_dim_64.ckpt` | [SoundReactor VAE](https://github.com/SonyResearch/soundreactor_vae) (Google Drive) |

If the Google Drive download fails, download `sr_causal_ds_1600_dim_64.ckpt` from the [folder linked by SoundReactor VAE](https://drive.google.com/drive/folders/1rlROnePyQU4b8GzjkB07W5vlb297p5_c), place it in `weights/soundreactor_vae/` and run the script again.

## Usage

`examples/` contains four 8-second test videos without sound: `hammer.mp4`, `table_tennis.mp4`, `fireworks.mp4` and `printer.mp4`.

```bash
python infer.py examples/hammer.mp4 examples/table_tennis.mp4 examples/fireworks.mp4 --mux
python infer.py examples/printer.mp4 --prompt "A dot matrix printer whirs to life, emitting a rapid, high-pitched, buzzing printing sound that then ceases." --mux
```

The results are written to `outputs/`. More generally:

```bash
python infer.py video.mp4
python infer.py video.mp4 --prompt "a dog barks twice" --mux
python infer.py a.mp4 b.mp4 --output-dir outputs --realtime
python infer.py --prompt "rain falls on a tin roof"
```

| Option | Meaning |
| --- | --- |
| `--prompt` | text prompt, used together with the video or alone (text-to-audio) |
| `--seconds` | length of the audio for text-to-audio, at most 10 (default 10) |
| `--seed` | random seed, the output is deterministic for a given seed |
| `--realtime` | release each video frame at its arrival time, as in a live stream |
| `--mux` | also write the video with the generated audio (needs `ffmpeg`) |
| `--no-cuda-graph` | run without CUDA graphs; slower, same output |
| `--output-dir` | where the results are written (default `outputs/`) |

For each input the script writes `<name>.wav` (48 kHz stereo, 32-bit float; for text-to-audio the name is taken from the prompt) and prints one line of statistics: `first_audio_ms` is the time from the start of the stream to the first decoded block, `block_processing_ms` the mean time to generate and decode a block of 100 ms, and `playout_delay_ms` the smallest constant delay with which the audio could be played without interruption. The timing fields describe a live stream only with `--realtime` and on a GPU that runs nothing else.

From Python:

```python
from liveaudio.pipeline import LiveAudio

pipe = LiveAudio("weights/liveaudio.safetensors", "weights")
audio, stats = pipe.generate("video.mp4", prompt=None, seed=0)
audio, stats = pipe.generate_from_text("rain falls on a tin roof", seconds=10, seed=0)
```

`audio` is a `(2, samples)` float array at 48 kHz. `generate` and `generate_from_text` also take `on_audio`, a function that receives each decoded block as soon as it is ready.

Notes:

- The model generates up to 10 seconds; for a longer video the first 10 seconds are used. The duration condition of the model is always 10 seconds, and for a shorter video the audio is cut to the length of the video.
- With CUDA graphs, the first input of each kind (video, video with a prompt, prompt only) takes about a minute longer, because the graphs are captured once.
- Without `--realtime` the frames are read as fast as they can be decoded. The generated audio is identical in both modes: a block never uses a frame that would not have arrived yet.

## Code

| File | Content |
| --- | --- |
| `infer.py` | command-line entry point |
| `liveaudio/pipeline.py` | streaming loop: frames or text in, audio blocks out |
| `liveaudio/frontend.py` | one-pass video decoding, online CLIP and Synchformer features, condition buffers |
| `liveaudio/engine.py` | block-by-block few-step sampling with a key-value cache |
| `liveaudio/student.py` | block-causal forward pass of the transformer |
| `liveaudio/layout.py` | timing of audio blocks and video features |
| `liveaudio/dit.py` | transformer modules |
| `liveaudio/conditioning.py` | text, duration and video-feature conditioning |
| `liveaudio/synchformer.py` | Synchformer video encoder |
| `liveaudio/vae.py` | streaming causal VAE decoder |
| `liveaudio/model.py` | loading of the generator and the decoder |

## License

The code and the model weights are released under [CC BY-NC 4.0](LICENSE). The pretrained encoders and the VAE downloaded by `scripts/download_weights.py` are subject to the licenses of their sources.

## Acknowledgements

LiveAudio builds on [Omni2Sound](https://github.com/omni2sound/Omni2Sound) and [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools), from which the transformer and the conditioning are adapted, the [SoundReactor VAE](https://github.com/SonyResearch/soundreactor_vae) for the causal audio VAE, [Synchformer](https://github.com/v-iashin/Synchformer) and [MMAudio](https://github.com/hkchengrex/MMAudio) for the synchronization features, [OpenCLIP](https://github.com/mlfoundations/open_clip) with [DFN5B-CLIP](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-384) for the visual features and [Flan-T5](https://huggingface.co/google/flan-t5-base) for the text encoder. We thank the authors for releasing their code and models.
