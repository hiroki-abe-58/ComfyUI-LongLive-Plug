"""Generate the public API-format prompts in ``workflows/api/``.

All node types are ComfyUI core nodes (v0.38.0) plus this package's two nodes.
GUI workflows in ``workflows/`` are produced from these prompts by loading them
in the ComfyUI frontend (see docs/REPRODUCE.md) and are checked by the tests.

    python scripts/build_workflows.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from longlive_plug.recipes import WAN_DEFAULT_NEGATIVE_PROMPT  # noqa: E402

BASE_MODEL = "wan2.1_t2v_14B_bf16.safetensors"
TEXT_ENCODER = "umt5_xxl_fp16.safetensors"
VAE = "wan_2.1_vae.safetensors"
FEW_STEP_LORA = "LongLive-Plug-Wan2.1-T2V-14B-few-step.safetensors"
CFG_LORA = "LongLive-Plug-Wan2.1-T2V-14B-cfg.safetensors"

DEMO_PROMPT = (
    "A red fox trots through fresh snow in a quiet birch forest at sunrise, soft golden light filtering "
    "between the white trunks, its breath visible in the cold air, snow crystals sparkling, "
    "low tracking shot, shallow depth of field, cinematic and natural colors."
)
DEMO_SEED = 20261002
DEMO_SIZE = {"width": 832, "height": 480, "length": 81}
FPS = 16.0

RECIPE_BY_VARIANT = {
    "baseline_50step": "wan21-14b-base-50step-unipc-cfg5",
    "longlive_plug_4step": "wan21-14b-plug-4step-unipc",
    "naive_4step": "wan21-14b-base-4step-unipc-cfg5-naive",
    "longlive_plug_4step_euler": "wan21-14b-plug-4step-euler",
    "longlive_plug_4step_lcm": "wan21-14b-plug-4step-lcm",
}


def _common(prompt: str, seed: int, size: dict) -> dict:
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": BASE_MODEL, "weight_dtype": "default"},
            "_meta": {"title": "Load Diffusion Model"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": TEXT_ENCODER, "type": "wan", "device": "default"},
            "_meta": {"title": "Load CLIP"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}, "_meta": {"title": "Load VAE"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}, "_meta": {"title": "Positive Prompt"}},
        "6": {"class_type": "EmptyHunyuanLatentVideo", "inputs": {**size, "batch_size": 1}, "_meta": {"title": "Empty Video Latent"}},
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}, "_meta": {"title": "Noise"}},
        "10": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {"noise": ["7", 0], "guider": ["9", 0], "sampler": ["9", 1], "sigmas": ["9", 2], "latent_image": ["6", 0]},
            "_meta": {"title": "Sampler"},
        },
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}, "_meta": {"title": "VAE Decode"}},
        "12": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["11", 0], "fps": FPS, "bit_depth": "auto", "color_space": "sRGB", "codec": "none"},
            "_meta": {"title": "Create Video"},
        },
        "14": {"class_type": "PreviewAny", "inputs": {"source": ["9", 3]}, "_meta": {"title": "Recipe Report"}},
    }


def build(variant: str, prompt: str = DEMO_PROMPT, seed: int = DEMO_SEED, size: dict | None = None) -> dict:
    size = dict(size or DEMO_SIZE)
    recipe = RECIPE_BY_VARIANT[variant]
    g = _common(prompt, seed, size)
    uses_adapters = variant.startswith("longlive_plug")
    if uses_adapters:
        g["8"] = {
            "class_type": "LongLivePlugApplyAdapters",
            "inputs": {
                "model": ["1", 0],
                "profile": "wan2.1-t2v-14b",
                "few_step_lora": FEW_STEP_LORA,
                "cfg_lora": CFG_LORA,
                "few_step_weight": 1.0,
                "cfg_weight": 0.5,
                "verify_sha256": True,
                "allow_untested_derivative": False,
            },
            "_meta": {"title": "LongLive-Plug Apply Adapter Pair"},
        }
        g["15"] = {"class_type": "PreviewAny", "inputs": {"source": ["8", 1]}, "_meta": {"title": "Adapter Coverage Report"}}
        model_ref = ["8", 0]
    else:
        model_ref = ["1", 0]
    g["9"] = {
        "class_type": "LongLivePlugRecipe",
        "inputs": {"model": model_ref, "positive": ["4", 0], "recipe": recipe, "init_noise": "upstream", "strict_profile": True},
        "_meta": {"title": "LongLive-Plug Sampling Recipe"},
    }
    if recipe.endswith(("cfg5", "cfg5-naive")):
        g["5"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": WAN_DEFAULT_NEGATIVE_PROMPT, "clip": ["2", 0]},
            "_meta": {"title": "Negative Prompt (Wan default)"},
        }
        g["9"]["inputs"]["negative"] = ["5", 0]
    g["13"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["12", 0],
            "filename_prefix": f"longlive_plug/{variant}",
            "format": "mp4",
            "format.codec": "h264",
            "format.codec.encoding": "auto",
            "codec": "auto",
        },
        "_meta": {"title": "Save Video"},
    }
    return dict(sorted(g.items(), key=lambda kv: int(kv[0])))


def main() -> None:
    out_dir = ROOT / "workflows" / "api"
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant in ("baseline_50step", "longlive_plug_4step"):
        path = out_dir / f"wan21_t2v_14b_{variant}.json"
        path.write_text(json.dumps(build(variant), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("wrote", path.relative_to(ROOT))


def variant_prompt(variant: str, **kw) -> dict:
    return copy.deepcopy(build(variant, **kw))


if __name__ == "__main__":
    main()
