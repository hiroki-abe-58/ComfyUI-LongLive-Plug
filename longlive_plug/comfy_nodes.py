"""ComfyUI nodes. Everything ComfyUI-specific lives in this module."""

from __future__ import annotations

import json
import logging
import os

import comfy.lora
import comfy.model_sampling
import comfy.samplers
import comfy.utils
import folder_paths
import torch

from .adapters import (
    AdapterError,
    analyze_adapter,
    comfy_lora_dict,
    file_sha256,
    read_sibling_peft_config,
    validate_against_base,
)
from .flow_unipc import sample_flow_unipc
from .profiles import PROFILES, fingerprint
from .recipes import DEFAULT_PLUG_RECIPE, RECIPES
from .schedules import flow_unipc_schedule

log = logging.getLogger("LongLivePlug")
MARKER_KEY = "longlive_plug"
INIT_NOISE_MODES = ["upstream", "comfy"]


def _state_shapes(model) -> tuple[dict, dict]:
    sd = model.model_state_dict()
    return {k: tuple(v.shape) for k, v in sd.items()}, {k: v.dtype for k, v in sd.items()}


def _check_backbone(profile, shapes, allow_derivative: bool) -> dict:
    fp = fingerprint(profile, shapes)
    if fp["match"] == "mismatch":
        raise AdapterError(f"model is not {profile.display_name}: " + "; ".join(fp["reasons"][:4]))
    if fp["match"] == "derivative" and not allow_derivative:
        raise AdapterError(
            f"model looks like a {profile.display_name} derivative ({'; '.join(fp['reasons'][:3])}). "
            "Derivatives are in LongLive-Plug's transfer scope but untested here; enable allow_untested_derivative to proceed."
        )
    return fp


class LongLivePlugApplyAdapters:
    """Apply the official Few-Step + CFG LoRA pair with full coverage checks."""

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "profile": (list(PROFILES), {"default": "wan2.1-t2v-14b"}),
                "few_step_lora": (loras, {"tooltip": "LongLive-Plug Few-Step LoRA (e.g. generator_lora_lightx2v.safetensors)"}),
                "cfg_lora": (
                    loras,
                    {"tooltip": "LongLive-Plug CFG LoRA (PEFT adapter_model.safetensors; keep adapter_config.json next to it)"},
                ),
                "few_step_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "cfg_weight": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "LoRA weight, not the sampler CFG scale"},
                ),
                "verify_sha256": ("BOOLEAN", {"default": True}),
                "allow_untested_derivative": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "apply"
    CATEGORY = "LongLive-Plug"
    DESCRIPTION = "Applies the LongLive-Plug Few-Step and CFG LoRA pair to a matching Wan backbone and reports key coverage."

    def apply(self, model, profile, few_step_lora, cfg_lora, few_step_weight, cfg_weight, verify_sha256, allow_untested_derivative):
        prof = PROFILES[profile]
        if few_step_lora == cfg_lora:
            raise AdapterError("few_step_lora and cfg_lora point to the same file")
        if model.model_options.get(MARKER_KEY):
            raise AdapterError("LongLive-Plug adapters are already applied to this model")
        shapes, dtypes = _state_shapes(model)
        fp = _check_backbone(prof, shapes, allow_untested_derivative)

        prepared = []
        for role, name, weight in (("few_step", few_step_lora, few_step_weight), ("cfg", cfg_lora, cfg_weight)):
            path = folder_paths.get_full_path_or_raise("loras", name)
            sha = file_sha256(path) if verify_sha256 else None
            sd, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
            analysis = analyze_adapter(
                role,
                os.path.basename(name),
                sd,
                metadata=metadata,
                peft_config=read_sibling_peft_config(path),
                sha256=sha,
                profile=prof,
            )
            if analysis.pinned is not None and analysis.pinned.role != role:
                raise AdapterError(f"the {role} input received the pinned {analysis.pinned.role} adapter; are the inputs swapped?")
            coverage = validate_against_base(analysis, shapes, dtypes, prof)
            prepared.append((analysis, coverage, float(weight)))

        patched = model.clone()
        adapters_report = []
        for analysis, coverage, weight in prepared:
            lora_sd, to_load = comfy_lora_dict(analysis)
            patches = comfy.lora.load_lora(lora_sd, to_load, log_missing=True)
            if len(patches) != len(analysis.pairs):
                raise AdapterError(f"{analysis.role}: ComfyUI built {len(patches)} patches for {len(analysis.pairs)} LoRA pairs")
            applied = patched.add_patches(patches, strength_patch=weight)
            if len(applied) != len(analysis.pairs):
                raise AdapterError(f"{analysis.role}: only {len(applied)}/{len(analysis.pairs)} patches matched model weights")
            pin = analysis.pinned
            adapters_report.append(
                {
                    "role": analysis.role,
                    "file": analysis.file_name,
                    "sha256": analysis.sha256,
                    "pinned_release": f"{pin.repo}@{pin.revision}/{pin.filename}" if pin else None,
                    "key_format": analysis.key_format,
                    "tensor_dtypes": sorted(set(analysis.dtypes)),
                    "lora_pairs": len(analysis.pairs),
                    "rank": analysis.rank,
                    "alpha": analysis.alpha,
                    "alpha_source": analysis.alpha_source,
                    "lora_weight": weight,
                    "effective_scale": weight * analysis.alpha / analysis.rank,
                    **coverage,
                    "patches_applied": len(applied),
                    "application_rate": len(applied) / coverage["targets_expected"],
                }
            )
        recommended = prof.recommended_weights
        conformance = (
            "official"
            if (few_step_weight, cfg_weight) == (recommended["few_step"], recommended["cfg"])
            and all(a["pinned_release"] for a in adapters_report)
            else "custom"
        )
        marker = {
            "profile": prof.profile_id,
            "backbone_match": fp["match"],
            "few_step_weight": float(few_step_weight),
            "cfg_weight": float(cfg_weight),
            "recipe_conformance": conformance,
        }
        patched.model_options[MARKER_KEY] = marker
        report = {
            "node": "LongLivePlugApplyAdapters",
            "profile": prof.profile_id,
            "backbone": {"match": fp["match"], "notes": fp["reasons"]},
            "pre_existing_patched_weights": len(model.patches),
            "adapters": adapters_report,
            "merge_rule": "W + sum_i weight_i * (alpha_i / rank_i) * B_i @ A_i, accumulated in float32, rounded to the base dtype",
            "recipe_conformance": conformance,
        }
        text = json.dumps(report, indent=1, ensure_ascii=False)
        log.info("LongLive-Plug adapters applied: %s", json.dumps({k: report[k] for k in ("profile", "recipe_conformance")}))
        return (patched, text)


def _flow_unipc_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, schedule=None, init_noise="upstream"):
    """ComfyUI sampler function for Wan's FlowUniPC (order 2, bh2)."""
    extra_args = {} if extra_args is None else extra_args
    expected = schedule.sigmas_tensor()
    got = sigmas.detach().float().cpu()
    if got.shape != expected.shape or not torch.equal(got, expected):
        raise ValueError("sigmas do not match the LongLive-Plug recipe; connect the recipe node's SIGMAS output unchanged")
    if init_noise == "upstream":
        noise = getattr(model, "noise", None)
        latent = getattr(model, "latent_image", None)
        if noise is None or latent is None or torch.count_nonzero(latent) != 0:
            raise ValueError("init_noise='upstream' needs an empty latent (text-to-video); use init_noise='comfy' otherwise")
        x = noise.to(device=x.device, dtype=x.dtype)
    s_in = x.new_ones([x.shape[0]])

    def flow_model(xc, i):
        sigma_in = schedule.timesteps[i] / 1000.0
        denoised = model(xc, sigma_in * s_in, **extra_args)
        return (xc - denoised) / sigma_in

    def report(i, x0, xt, sigma):
        if callback is not None:
            callback({"x": xt, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": x0})

    return sample_flow_unipc(flow_model, x, schedule.sigmas, schedule.timesteps, callback=report)


class LongLivePlugRecipe:
    """Verified sampler/sigma/guider presets for the baseline and LongLive-Plug runs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "recipe": (list(RECIPES), {"default": DEFAULT_PLUG_RECIPE}),
                "init_noise": (
                    INIT_NOISE_MODES,
                    {
                        "default": "upstream",
                        "tooltip": "upstream: x_T = noise (Wan/LongLive). comfy: x_T = sigma_0 * noise + (1 - sigma_0) * latent",
                    },
                ),
                "strict_profile": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "negative": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("GUIDER", "SAMPLER", "SIGMAS", "STRING")
    RETURN_NAMES = ("guider", "sampler", "sigmas", "recipe_report")
    FUNCTION = "build"
    CATEGORY = "LongLive-Plug"
    DESCRIPTION = "Builds the guider, sampler and sigmas of an upstream-verified Wan / LongLive-Plug sampling recipe."

    def build(self, model, positive, recipe, init_noise, strict_profile, negative=None):
        r = RECIPES[recipe]
        prof = PROFILES[r.profile_id]
        marker = model.model_options.get(MARKER_KEY)
        if r.requires_adapters and not marker:
            raise ValueError(f"recipe '{recipe}' expects the LongLive-Plug adapter pair; run LongLivePlugApplyAdapters first")
        if not r.requires_adapters and marker:
            raise ValueError(f"recipe '{recipe}' is a base-model recipe but LongLive-Plug adapters are applied to this model")
        if marker and marker["profile"] != r.profile_id:
            raise ValueError(f"adapters were applied with profile {marker['profile']} but the recipe targets {r.profile_id}")
        shapes, _ = _state_shapes(model)
        fp = fingerprint(prof, shapes)
        if fp["match"] == "mismatch" or (
            strict_profile and fp["match"] != "exact" and not (marker and marker["backbone_match"] == fp["match"])
        ):
            raise ValueError(f"model does not match {prof.display_name} ({fp['match']}): {'; '.join(fp['reasons'][:3])}")
        ms = model.get_model_object("model_sampling")
        if not isinstance(ms, comfy.model_sampling.CONST) or float(getattr(ms, "multiplier", 1000)) != 1000.0:
            raise ValueError("expected a flow-matching (CONST) Wan model with timestep multiplier 1000")

        if r.cfg == 1.0:
            from comfy_extras.nodes_custom_sampler import Guider_Basic

            guider = Guider_Basic(model)
            guider.set_conds(positive)
        else:
            if negative is None:
                raise ValueError(f"recipe '{recipe}' uses CFG {r.cfg}; connect a negative conditioning")
            guider = comfy.samplers.CFGGuider(model)
            guider.set_conds(positive, negative)
            guider.set_cfg(r.cfg)

        if r.solver == "flow_unipc":
            schedule = flow_unipc_schedule(r.steps, r.shift)
            sampler = comfy.samplers.KSAMPLER(_flow_unipc_sampler, extra_options={"schedule": schedule, "init_noise": init_noise})
            sigmas = schedule.sigmas_tensor()
        else:
            # sigma_0 == 1.0 here, so ComfyUI's noise scaling already equals x_T = noise.
            sampler = comfy.samplers.sampler_object(r.solver)
            sigmas = torch.tensor(r.sigmas(), dtype=torch.float32)

        report = {
            "node": "LongLivePlugRecipe",
            **r.describe(),
            "init_noise": init_noise if r.solver == "flow_unipc" else "sigma_0 = 1.0 (identical in both modes)",
            "backbone_match": fp["match"],
            "adapters": marker,
            "negative_used": r.cfg != 1.0,
        }
        return (guider, sampler, sigmas, json.dumps(report, indent=1, ensure_ascii=False))


NODE_CLASS_MAPPINGS = {
    "LongLivePlugApplyAdapters": LongLivePlugApplyAdapters,
    "LongLivePlugRecipe": LongLivePlugRecipe,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LongLivePlugApplyAdapters": "LongLive-Plug Apply Adapter Pair",
    "LongLivePlugRecipe": "LongLive-Plug Sampling Recipe",
}
