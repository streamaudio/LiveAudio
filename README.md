# LiveAudio

Inference code and checkpoint for **LiveAudio: From Bidirectional to Autoregressive Diffusion for Real-Time Streaming Video-to-Audio Generation**.

Audio samples: https://streamaudio.github.io/

LiveAudio generates 48 kHz stereo audio for a video as a stream. The video file is decoded once, frame by frame, and nothing of the video is precomputed. Every 0.1 s block of audio is generated as soon as the frames it depends on have arrived, with four sampling steps and no guidance, and is decoded to a waveform immediately by a causal audio VAE. The same checkpoint covers three tasks: video-to-audio, video-text-to-audio (a text prompt in addition to the video) and text-to-audio (a text prompt only).

## Setup

Tested with Python 3.10, PyTorch 2.1.0 (CUDA 12.1), PyAV 12.3.0 and one NVIDIA A100 GPU. Use a fresh environment: the requirements of Omni2Sound pin the versions of many packages.

```bash
conda create -n liveaudio python=3.10 -y
conda activate liveaudio
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
bash scripts/setup_third_party.sh
pip install -r third_party/Omni2Sound/requirements.txt
pip install -r requirements.txt
```

`scripts/setup_third_party.sh` clones the two code bases the model is built on, at fixed commits, into `third_party/`: [Omni2Sound](https://github.com/omni2sound/Omni2Sound) (transformer and conditioning modules) and [soundreactor_vae](https://github.com/SonyResearch/soundreactor_vae) (causal audio VAE).

## Weights

Place the LiveAudio checkpoint `liveaudio.safetensors` in `weights/`, then download the pretrained encoders and the audio VAE:

```bash
python scripts/download_weights.py
```

| File in `weights/` | Source |
| --- | --- |
| `liveaudio.safetensors` | LiveAudio generator (this release) |
| `flan-t5-base/` | [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) |
| `DFN5B-CLIP-ViT-H-14-384/` | [apple/DFN5B-CLIP-ViT-H-14-384](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-384) |
| `synchformer_state_dict.pth` | [MMAudio release](https://github.com/hkchengrex/MMAudio/releases/tag/v0.1) |
| `soundreactor_vae/sr_causal_ds_1600_dim_64.ckpt` | [soundreactor_vae](https://github.com/SonyResearch/soundreactor_vae) |

The script checks the SHA-256 of the downloaded weights.

## Usage

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

For each video the script writes `<name>.wav` (48 kHz stereo, 32-bit float; for text-to-audio the name is taken from the prompt) and prints one line of statistics: `first_audio_ms` is the time from the start of the stream to the first decoded block, `block_processing_ms` the mean time to generate and decode a block of 100 ms, and `playout_delay_ms` the smallest constant delay with which the audio could be played without interruption. The timing fields describe a live stream only with `--realtime` and on a GPU that runs nothing else.

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
| `liveaudio/model.py` | loading of the generator and of the streaming VAE decoder |

## Acknowledgements

This code builds on [Omni2Sound](https://github.com/omni2sound/Omni2Sound), [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools), the [SoundReactor VAE](https://github.com/SonyResearch/soundreactor_vae), [Synchformer](https://github.com/v-iashin/Synchformer), [MMAudio](https://github.com/hkchengrex/MMAudio), [OpenCLIP](https://github.com/mlfoundations/open_clip) and [DFN](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-384). Their licenses apply to the respective code and weights.
