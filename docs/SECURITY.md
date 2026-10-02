# Security notes

## What the nodes do

- Read LoRA files that ComfyUI lists in its `loras` folders, via ComfyUI's
  `load_torch_file(..., safe_load=True)` (safetensors, no pickle). File names
  come from ComfyUI's own folder listing, not from free text.
- Read `adapter_config.json` only when it sits next to a file named
  `adapter_model.safetensors` (JSON, at most 1 MiB).
- Hash the selected LoRA files with SHA-256 (`verify_sha256`).
- Patch a clone of the model in memory. Nothing is written to disk.

They start no processes, make no network requests, install nothing, and use
no `eval`/`exec` or dynamic imports. Reports contain file *names* only, never
absolute paths.

## Benchmark probe (`benchmarks/comfy_probe/`)

The probe is **not** part of the node package; ComfyUI only loads it if you
place it in `custom_nodes` yourself. Even then it does nothing unless the
environment variable `LONGLIVE_PLUG_BENCH=1` is set. When enabled it

- wraps `comfy.samplers.calc_cond_batch` and `comfy.model_base.BaseModel.apply_model`
  to count calls (it does not change arguments or results),
- adds `GET /longlive-plug-bench/stats` (counters, sampled timesteps, CUDA
  allocator peaks, device name) and `POST /longlive-plug-bench/reset`.

These routes are unauthenticated like the rest of ComfyUI's API. Use the probe
only on a ComfyUI bound to `127.0.0.1` for benchmarking.

## Scripts

- `scripts/run_comparison.py` and `scripts/export_gui_workflows.py` refuse
  non-loopback servers. The runner calls `nvidia-smi`, `ffprobe` and `ffmpeg`
  with argument lists (no shell) and writes manifests without absolute paths,
  host names or environment variables.
- `scripts/make_reference_vectors.py` and `scripts/verify_merge.py` import
  Python files from an upstream LongLive checkout *that you point them at*.
  This executes that code; use a checkout you trust.

## Reporting

Please open a GitHub issue for security problems that do not expose other
users; otherwise contact the maintainer through GitHub first.
