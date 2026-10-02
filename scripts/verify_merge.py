"""Compare ComfyUI's LoRA patch path with LongLive-Plug's merge rule on real weights.

For selected layers of the real base model this computes

1. the upstream merge: ``merge()`` from ``LongLive-Plug/scripts/merge_lora.py``
   (imported from an upstream checkout when ``--longlive-root`` is given,
   otherwise the identical ``longlive_plug.adapters.reference_merge``), and
2. ComfyUI's path: ``comfy.lora.load_lora`` -> ``comfy.lora.calculate_weight``
   on a float32 copy -> ``comfy.float.stochastic_rounding`` to the base dtype,
   i.e. what ``ModelPatcher.patch_weight_to_device`` does,

on CPU, and reports exact-match rates and the largest difference in units of
the base dtype's spacing (ULP). Only the selected tensors are read.

    python scripts/verify_merge.py --comfyui <ComfyUI dir> --base <wan2.1_t2v_14B_bf16.safetensors> \
        --few-step <few-step.safetensors> --cfg <cfg.safetensors> [--longlive-root <LongLive checkout>]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from longlive_plug.adapters import analyze_adapter, comfy_lora_dict, file_sha256, reference_merge  # noqa: E402
from longlive_plug.profiles import WAN21_T2V_14B  # noqa: E402

DEFAULT_LAYERS = [
    "blocks.0.self_attn.q",
    "blocks.0.cross_attn.v",
    "blocks.0.ffn.0",
    "blocks.19.self_attn.o",
    "blocks.19.ffn.2",
    "blocks.39.cross_attn.k",
    "blocks.39.ffn.0",
]


def load_subset(path: Path, keys: list[str]) -> tuple[dict, dict]:
    with safe_open(str(path), "pt") as f:
        meta = f.metadata() or {}
        return {k: f.get_tensor(k) for k in keys if k in f.keys()}, meta


def adapter_keys(path: Path, layers: list[str]) -> list[str]:
    with safe_open(str(path), "pt") as f:
        names = list(f.keys())
    wanted = {f"{layer}.lora_{s}" for layer in layers for s in "AB"}
    return [k for k in names if any(k.endswith(w + ".weight") or k.endswith(w + ".default.weight") for w in wanted)]


def ulp_diff(a: torch.Tensor, b: torch.Tensor) -> int:
    if a.dtype != torch.bfloat16:
        return int((a.float() - b.float()).abs().max().item() > 0)
    ia = a.view(torch.int16).to(torch.int32)
    ib = b.view(torch.int16).to(torch.int32)
    # Map sign-magnitude to a monotonic integer line.
    ia = torch.where(ia < 0, -32768 - ia, ia)
    ib = torch.where(ib < 0, -32768 - ib, ib)
    return int((ia - ib).abs().max().item())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comfyui", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--few-step", type=Path, required=True)
    p.add_argument("--cfg", type=Path, required=True)
    p.add_argument("--few-step-weight", type=float, default=1.0)
    p.add_argument("--cfg-weight", type=float, default=0.5)
    p.add_argument("--longlive-root", type=Path)
    p.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    sys.path.insert(0, str(args.comfyui))
    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    import comfy.float
    import comfy.lora
    import comfy.utils

    upstream_merge, merge_source = None, "longlive_plug.adapters.reference_merge"
    if args.longlive_root:
        path = args.longlive_root / "LongLive-Plug" / "scripts" / "merge_lora.py"
        spec = importlib.util.spec_from_file_location("upstream_merge_lora", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        upstream_merge, merge_source = mod, f"upstream LongLive-Plug/scripts/merge_lora.py (sha256 {file_sha256(path)[:16]}...)"

    base, _ = load_subset(args.base, [f"{layer}.weight" for layer in args.layers])
    analyses = []
    for role, path, weight in (("few_step", args.few_step, args.few_step_weight), ("cfg", args.cfg, args.cfg_weight)):
        sha = file_sha256(path)
        sd, meta = load_subset(path, adapter_keys(path, args.layers))
        a = analyze_adapter(role, path.name, sd, metadata=meta, sha256=sha, profile=WAN21_T2V_14B)
        analyses.append((a, weight))

    results = []
    for layer in args.layers:
        w = base[f"{layer}.weight"]
        state = {f"{layer}.weight": w.clone()}
        if upstream_merge is not None:
            updates = []
            for a, weight in analyses:
                pair = a.pairs[layer]
                updates.append([(f"{layer}.weight", pair["A"], pair["B"], weight * a.alpha / a.rank)])
            upstream_merge.merge(state, updates)
            ref = state[f"{layer}.weight"]
        else:
            ref = reference_merge(w, [(a.pairs[layer]["A"], a.pairs[layer]["B"], weight * a.alpha / a.rank) for a, weight in analyses])

        patches = []
        for a, weight in analyses:
            sub = type(a)(
                role=a.role,
                file_name=a.file_name,
                sha256=a.sha256,
                pinned=a.pinned,
                key_format=a.key_format,
                pairs={layer: a.pairs[layer]},
                rank=a.rank,
                alpha=a.alpha,
                alpha_source=a.alpha_source,
            )
            lora_sd, to_load = comfy_lora_dict(sub)
            built = comfy.lora.load_lora(lora_sd, to_load, log_missing=True)
            adapter = built[f"diffusion_model.{layer}.weight"]
            patches.append((weight, adapter, 1.0, None, None))
        key = f"diffusion_model.{layer}.weight"
        temp = w.to(torch.float32, copy=True)
        out = comfy.lora.calculate_weight(patches, temp, key)
        comfy_w = comfy.float.stochastic_rounding(out, w.dtype, seed=comfy.utils.string_to_seed(key))

        equal = (comfy_w == ref).float().mean().item()
        changed = (ref != w).float().mean().item()
        results.append(
            {
                "layer": layer,
                "shape": list(w.shape),
                "dtype": str(w.dtype).replace("torch.", ""),
                "fraction_bitwise_equal_to_upstream": equal,
                "max_ulp_diff": ulp_diff(comfy_w, ref),
                "max_abs_diff": (comfy_w.float() - ref.float()).abs().max().item(),
                "fraction_changed_vs_base": changed,
                "delta_rms": (ref.float() - w.float()).pow(2).mean().sqrt().item(),
            }
        )
        print(json.dumps(results[-1]))
    summary = {
        "reference": merge_source,
        "adapters": [
            {"role": a.role, "sha256": a.sha256, "pinned": bool(a.pinned), "alpha": a.alpha, "rank": a.rank, "weight": wgt}
            for a, wgt in analyses
        ],
        "base_file": args.base.name,
        "torch": torch.__version__,
        "device": "cpu",
        "layers": results,
    }
    if args.output:
        args.output.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
