# Benchmark and demo record

All numbers were measured on 2026-10-02 for this release. Raw records:
[`docs/results/benchmark_runs.json`](results/benchmark_runs.json).

## Setup

| | |
| --- | --- |
| Host | Windows 11 Pro 10.0.26200, Intel Core Ultra 9 285K, 64 GB RAM |
| GPU | NVIDIA GeForce RTX 5090 32 GB, driver 595.95, WDDM; about 2.4 GiB of VRAM was in use by other applications before the runs |
| Software | ComfyUI v0.38.0 (`6b747c0`), Python 3.12.13, PyTorch 2.14.1+cu130, ComfyUI DynamicVRAM enabled (default) |
| Launch flags | `--bf16-unet --bf16-text-enc --fp32-vae --disable-api-nodes --listen 127.0.0.1` |
| Base | `wan2.1_t2v_14B_bf16.safetensors` (Comfy-Org repackage `123acf1`, sha256 `193535c6…`), UMT5 fp16 file run in bf16, Wan2.1 VAE in fp32 |
| Adapters | Few-Step `743fc9a4…` (weight 1.0), CFG `1a311f60…` (weight 0.5); both 400/400 targets applied |
| Prompt | see README; seed 20261002 (B run 2: 20261003), 832×480, 81 frames, 16 fps |
| Workflows | `workflows/api/*.json` built by `scripts/build_workflows.py`; queued over ComfyUI's HTTP API by `scripts/run_comparison.py` |

## Method

- Wall time and per-node time come from ComfyUI's websocket `executing`
  events. Per-step times come from `progress` events, so they carry some event
  delivery jitter.
- A benchmark probe (`benchmarks/comfy_probe`, loaded only for these runs)
  counted sampler evaluations, conditional/unconditional passes and actual
  batched forward calls, and recorded the model timesteps and PyTorch
  allocator peaks.
- nvidia-smi was polled every 0.5 s for total device memory (all processes).
- Each MP4 was checked with ffprobe and a full ffmpeg decode. Five frames per
  video (0, n/4, n/2, 3n/4, n−1) were extracted and inspected by eye.
- s1 and s2 are two separate ComfyUI processes. Each started fresh, but
  the OS file cache was not flushed, so "first run" is not a cold-disk number.
- ComfyUI's DynamicVRAM loads weights lazily inside the sampler. Model loading
  therefore cannot be separated from the first denoising step, and the
  "loader" nodes take well under a second.

## Results (832×480×81)

| Session / order | Run | Recipe | Model passes (cond+uncond) | Forward calls × batch | First step incl. weight staging | Sampler stage | VAE decode | Wall total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s1 #1 (first after start) | B LongLive-Plug | `wan21-14b-plug-4step-unipc` | 4 + 0 | 4 × 1 | 13.4 s | 41.3 s | 8.9 s | 59.0 s |
| s1 #2 (seed+1) | B LongLive-Plug | same | 4 + 0 | 4 × 1 | 9.7 s | 37.8 s | 6.0 s | 44.5 s |
| s1 #3 (after B) | A baseline | `wan21-14b-base-50step-unipc-cfg5` | 50 + 50 | 50 × 2 | 24.9 s | 933.6 s | 6.0 s | 943.0 s |
| s1 #4 | C naive (comparison) | `wan21-14b-base-4step-unipc-cfg5-naive` | 4 + 4 | 4 × 2 | 18.2 s | 78.2 s | 6.1 s | 84.9 s |
| s2 #1 (first after start) | A baseline | `wan21-14b-base-50step-unipc-cfg5` | 50 + 50 | 50 × 2 | 25.8 s | 932.2 s | 6.2 s | 943.5 s |
| s2 #2 | B, alternative recipe | `wan21-14b-plug-4step-euler` | 4 + 0 | 4 × 1 | 16.3 s | 44.6 s | 7.0 s | 58.3 s |
| s2 #3 | B, alternative recipe | `wan21-14b-plug-4step-lcm` | 4 + 0 | 4 × 1 | 10.0 s | 38.4 s | 6.7 s | 45.6 s |
| s2 #4 | B cancel test | `wan21-14b-plug-4step-unipc` | interrupted after step 1 | — | — | — | — | 9.9 s, status `interrupted`, no file saved |

Steady-state step time: about 9.3 s for B (batch 1) and about 18.5 s for A
(cond and uncond batched as one forward of batch 2).

Model timesteps recorded by the probe:
- FlowUniPC recipes: 999, 937, 833, 624 for B, and 999, 995, 991, … for A
  (upstream int64 truncation).
- Euler and LCM recipes: 1000, 937.5, 833.33, 625.

### Memory

| Run | PyTorch peak allocated / reserved | nvidia-smi device total (max, all processes) |
| --- | --- | --- |
| B (s1 #1, #2) | 6.01 / 9.06 GiB, 6.01 / 8.66 GiB | 30.0 GiB, 30.0 GiB |
| A (s1 #3, s2 #1) | 6.10 / 8.41 GiB, 6.10 / 7.94 GiB | 30.2 GiB, 29.8 GiB |
| C (s1 #4) | 6.10 / 8.50 GiB | 31.0 GiB |

DynamicVRAM manages the model weights outside the PyTorch caching allocator
and fills free VRAM, which explains both columns. Neither column is a
minimum requirement. The minimum VRAM and host RAM needed were not measured.

## Correctness checks done with the real model

- Adapter coverage report: both adapters 400/400 targets, application rate
  1.0, no unused keys, alpha 128 / rank 128. The Few-Step alpha came from
  safetensors metadata plus the pinned hash; the CFG alpha from the pinned hash.
- `scripts/verify_merge.py`: ComfyUI's patch path vs upstream
  `merge_lora.merge()` on 7 layers of the real base, bit-identical on CPU
  ([`results/verify_merge_wan21_14b.json`](results/verify_merge_wan21_14b.json)).
- No residue between workflows: A run directly after B in the same server
  (s1 #3) decodes to the same 81 frames as A in a fresh server (s2 #1), with a
  maximum pixel difference of 0.
- Conditional-only: every B run had 0 unconditional passes.

## Quality notes (visual review)

- B (all three 4-step recipes) follows the prompt with a consistent subject,
  scene and tracking motion over 81 frames. Compared with A it is sharper, with
  stronger contrast and saturation and more stylised birch-bark patterns.
  A looks more natural. In B's last frame the fox's leg and its shadow merge
  slightly.
- C (the base model with 4 steps and no adapters) is heavily blurred and
  ghosted. The adapters, not the step count, produce B's quality.
- A and B differ in composition: they are different samplers on the same noise
  and are not expected to match pixel for pixel.

## Demo files (v0.1.0 release assets)

| Asset | Run | sha256 |
| --- | --- | --- |
| `longlive_plug_4step_seed20261002.mp4` | s1 #1 | `d3dee8a9f5d6722b60f1a83159ace1062862f8a98d8142faa627f0ce2365e2bc` |
| `baseline_50step_seed20261002.mp4` | s2 #1 | `d44697a2baeb4cbf92489f12bd61e7b504b294d63c413ff2f8ac8d48a8b04c2f` |
| `naive_4step_seed20261002.mp4` | s1 #4 | `c3af4e517f4083a8aa3b2b25c50995069dd072d64012029b5893403e19c30097` |

The MP4s carry ComfyUI's embedded prompt JSON (node graph and settings; no
paths). The README images are reduced previews of the s1 files.
