# Reproducing the demo and benchmark

## 1. ComfyUI

Tested with ComfyUI `v0.38.0` (`6b747c0`), Python 3.12.13, PyTorch
2.14.1+cu130, Windows 11, RTX 5090 32 GB, 64 GB RAM.

Install this package into `ComfyUI/custom_nodes/ComfyUI-LongLive-Plug` (git
clone or archive). It has no Python dependencies beyond ComfyUI's own.

Start ComfyUI with the upstream precisions (see `docs/SCHEDULER.md` §4):

```
python main.py --bf16-unet --bf16-text-enc --fp32-vae
```

## 2. Model files (pinned revisions)

| Put in | File name used by the workflows | Download |
| --- | --- | --- |
| `models/diffusion_models/` | `wan2.1_t2v_14B_bf16.safetensors` | `Comfy-Org/Wan_2.1_ComfyUI_repackaged` @ `123acf1`, `split_files/diffusion_models/` — sha256 `193535c6450045f718df5f011de6d94d49bd9b13f37ca0412500f050dbbb01a8` |
| `models/text_encoders/` | `umt5_xxl_fp16.safetensors` | same repo, `split_files/text_encoders/` — sha256 `7b8850f1961e1cf8a77cca4c964a358d303f490833c6c087d0cff4b2f99db2af` |
| `models/vae/` | `wan_2.1_vae.safetensors` | same repo, `split_files/vae/` — sha256 `2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b` |
| `models/loras/` | `LongLive-Plug-Wan2.1-T2V-14B-few-step.safetensors` | `Efficient-Large-Model/LongLive-Plug-Wan2.1-T2V-14B-few-step` @ `f125af0`, file `generator_lora_lightx2v.safetensors`, renamed — sha256 `743fc9a44e118a09f932ae5a0420c88b46bb6afdf605230f115e93eb710884ba` |
| `models/loras/` | `LongLive-Plug-Wan2.1-T2V-14B-cfg.safetensors` | `Efficient-Large-Model/LongLive-Plug-Wan2.1-T2V-14B-cfg` @ `32b8aa3`, file `adapter_model.safetensors`, renamed — sha256 `1a311f6030a74e739d9705347079a131bd5a5579bd693c993026d63e36d4bea0` |

Example with the Hugging Face CLI:

```bash
hf download Efficient-Large-Model/LongLive-Plug-Wan2.1-T2V-14B-few-step generator_lora_lightx2v.safetensors \
  --revision f125af0533b94e2cee0163a7a42bb736d59a1d37 --local-dir dl/few-step
hf download Efficient-Large-Model/LongLive-Plug-Wan2.1-T2V-14B-cfg adapter_model.safetensors adapter_config.json \
  --revision 32b8aa3d1db3d178a9c82f3731cc917a34fb7693 --local-dir dl/cfg
```

Renaming is fine: the node recognises the released files by SHA-256 and takes
rank/alpha from the pinned release. If you keep the CFG adapter as
`adapter_model.safetensors` in its own sub-folder, keep `adapter_config.json`
next to it. Note that ComfyUI lists files in sub-folders with OS-specific
separators, so the shipped workflows expect the files at the `loras` root.

## 3. Run

The scripts used below (`scripts/`, `benchmarks/`) are in the GitHub
repository; the Comfy Registry package does not include them.


Open `workflows/wan21_t2v_14b_longlive_plug_4step.json` (LongLive-Plug, 4 steps)
or `workflows/wan21_t2v_14b_baseline_50step.json` (base model, official
50-step recipe) in ComfyUI and queue it. The API-format equivalents are in
`workflows/api/`.

To reproduce the comparison table (same prompt, seed, size and frame count;
writes MP4s, extracted frames and a manifest):

```bash
python scripts/run_comparison.py --server http://127.0.0.1:8188 \
  --variants longlive_plug_4step baseline_50step naive_4step --results-dir results
```

The numbers in `docs/BENCHMARKS.md` additionally used the optional benchmark
probe (`benchmarks/comfy_probe`, copied to `custom_nodes` and enabled with
`LONGLIVE_PLUG_BENCH=1`) to count model passes and read allocator peaks.

## 4. Checks that do not need the GPU

```bash
python -m pytest -q          # set COMFYUI_PATH to a ComfyUI v0.38.0 checkout
python -m ruff check . && python -m ruff format --check .
python scripts/verify_merge.py --comfyui <ComfyUI> --base <wan2.1_t2v_14B_bf16.safetensors> \
  --few-step <few-step> --cfg <cfg> [--longlive-root <NVlabs/LongLive checkout>]
```

`scripts/make_reference_vectors.py --longlive-root <checkout>` regenerates
`tests/data/flow_unipc_reference.json` from the upstream scheduler (needs
`diffusers`).
