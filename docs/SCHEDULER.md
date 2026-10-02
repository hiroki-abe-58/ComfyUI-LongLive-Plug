# Adapter and scheduler correctness

What this package checks, against which upstream source, and how.

## 1. Which 4-step recipe is "official"?

For the Wan2.1-T2V-14B adapters, three upstream artifacts describe 4-step
sampling. They agree on the step *points* but not on the step *rule*:

| Recipe id | Upstream source | Step rule | Sigmas (σ) / model timesteps |
| --- | --- | --- | --- |
| `wan21-14b-plug-4step-unipc` (default) | NVlabs/LongLive `fb16a87`: `LongLive-Plug/inference.py` + `configs/wan21_dmd.yaml` (`sampling_steps: 4`, `guidance_scale: 1.0`, `timestep_shift: 5.0`) | Wan `FlowUniPCMultistepScheduler` (order 2, bh2, x0-prediction, final σ = 0) | σ = 0.99980, 0.93727, 0.83306, 0.62469, 0 — model t = 999, 937, 833, 624 |
| `wan21-14b-plug-4step-euler` | HF few-step repo `f125af0`: `inference_config.json` (`denoising_step_list` [1000, 750, 500, 250], `sample_shift` 5, `enable_cfg` false), consumed by LightX2V `WanStepDistillScheduler` (`8a97c75`) | Euler: `x ← x + (σ_next − σ)·v` | σ = 1.0, 0.9375, 0.83333, 0.625, 0 — model t = σ·1000 |
| `wan21-14b-plug-4step-lcm` | HF few-step repo `f125af0`: `training_config.yaml` (Self-Forcing-Plus run, `warp_denoising_step: true`) | x0 prediction, then re-noised with fresh Gaussian noise at the next σ (ComfyUI `lcm`) | same σ as the Euler recipe |

The default follows the LongLive-Plug README ("check the scheduler against the
matching LongLive-Plug few-step recipe"), which is the repository's
`inference.py` with the `wan21_dmd` config. The other two are provided because
they are published next to the adapter. The demo and benchmark use the
default only.

All three use runtime CFG scale 1.0 with a conditional-only forward pass and
the adapter weights Few-Step 1.0 + CFG 0.5 (model cards: "few-step : CFG = 1 : 0.5";
"These are adapter weights, not the inference CFG scale").

The baseline recipe `wan21-14b-base-50step-unipc-cfg5` reproduces Wan2.1
`generate.py` defaults for `t2v-14B` at `9737cba`: UniPC, 50 steps, shift 5.0,
guidance 5.0 with Wan's default negative prompt.

## 2. Verified details of the FlowUniPC path

| Detail | Upstream behaviour | This package | Evidence |
| --- | --- | --- | --- |
| Sigma table | built in float64, stored float32, `final_sigmas_type="zero"` | same computation | `tests/test_flow_unipc.py::test_schedule_matches_upstream` (exact) |
| Model timestep | `int64(σ·1000)` (truncated) while the solver uses float32 σ | the sampler calls the model at `σ_in = t_int/1000`, recovers `v = (x − x0_in)/σ_in`, and forms `x0 = x − σ·v` with the float σ | integration test checks the timesteps seen by the model; real-model probe recorded `[999, 937, 833, 624]` |
| Prediction type | flow (`v`), `x0 = x − σ·v` | ComfyUI's CONST model returns `x − σ_in·v`; converted as above | — |
| Solver | order 2, bh2, ρ_p = 0.5, corrector via 2×2 solve, lower-order warm-up and final step | same arithmetic, float32 CPU scalars | final latents **bit-identical** to the upstream class on CPU for 1, 4, 7 and 50 steps (`tests/data/flow_unipc_reference.json`) |
| Initial noise | `x_T = ε` | ComfyUI would use `σ_0·ε` (σ_0 = 0.9998); `init_noise="upstream"` uses ε exactly (requires an empty latent) | integration test: first model input equals the raw noise |
| CFG | `uncond + g·(cond − uncond)` on `v` | ComfyUI combines denoised outputs, which is the same linear combination | — |
| CFG = 1 | conditional pass only | `BasicGuider`; no unconditional pass | tests count passes; real-model probe: 4 cond, 0 uncond |
| Upstream noise RNG | `torch.randn` in `[B, T, C, H, W]` layout | ComfyUI's `RandomNoise` in `[B, C, T, H, W]` | **not** reproduced: same seed ≠ same noise as the upstream CLI |

The ρ_c solve runs on CPU here, on the sample's device upstream. Results can
differ at float32 rounding level on GPU.

## 3. Adapter application

LongLive-Plug `scripts/merge_lora.py` defines the merge:
`W ← W + Σ weightᵢ·(αᵢ/rᵢ)·Bᵢ@Aᵢ`, accumulated in float32, rounded to the base dtype.

| Check | How |
| --- | --- |
| Key layout | PEFT `base_model.model.` / native Wan / `diffusion_model.` prefixes are normalised; anything that is not a plain `lora_A`/`lora_B` (or `.alpha`) tensor is rejected (DoRA, rsLoRA, rank/alpha patterns, embeddings). |
| Targets | All 400 targets (40 blocks × {self_attn, cross_attn} × {q, k, v, o} + ffn.0, ffn.2), i.e. every `nn.Linear` in a `WanAttentionBlock`, as in upstream `configure_lora_for_model`. Missing or extra targets are an error. |
| Shapes / A-B direction | `B` is `[out, r]`, `A` is `[r, in]`; the base weight must be `[out, in]`. Shape mismatches (e.g. Wan2.2-5B or 1.3B adapters) are an error. |
| Alpha / rank | Taken from `.alpha` tensors, safetensors metadata, `adapter_config.json` (only beside `adapter_model.safetensors`), or the pinned release hash; all available sources must agree. The Few-Step file carries `alpha=128` in its metadata; the CFG file needs its config or the pinned hash. |
| Released files | SHA-256 of both released files are pinned. A pinned file connected to the wrong input is reported as swapped. |
| Coverage | ComfyUI must build and attach exactly one patch per pair (`patches_applied == 400`, `application_rate == 1.0`, `unused_adapter_keys == 0`). |
| Base dtype | Only float base weights are accepted; quantized (fp8 / int8 / GGUF-style) bases are refused, because their patch path is not verified. |
| Clone semantics | The node patches a `ModelPatcher.clone()`. Tests check the input model has no patches, and that unpatching restores every original weight bit-for-bit. |
| Real weights | `scripts/verify_merge.py` compares ComfyUI's patch path (`load_lora` → `calculate_weight` in float32 → round to bf16) with upstream `merge_lora.merge()` on 7 layers of the real bf16 base. Result: **100 % bit-identical** on CPU (`docs/results/verify_merge_wan21_14b.json`). |

## 4. Dtypes

ComfyUI picks precisions automatically. With ComfyUI v0.38.0 on an RTX 5090 it
loaded the bf16 Wan2.1 DiT as **fp16**, UMT5 as fp16 and the VAE in bf16.
Upstream runs the DiT and T5 in bf16 and decodes with an fp32 VAE. The tested
configuration therefore starts ComfyUI with

```
--bf16-unet --bf16-text-enc --fp32-vae
```

Without these flags the workflows still run (a 17-frame smoke run worked), but
that configuration is not the one benchmarked here.
