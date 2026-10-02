"""Sampling recipes with their upstream provenance (ComfyUI-independent)."""

from __future__ import annotations

from dataclasses import dataclass

from .flow_unipc import model_evaluations
from .schedules import flow_unipc_schedule, warped_step_list_sigmas

LONGLIVE = "NVlabs/LongLive@fb16a879f46e604df4a0ca48ce2bf45adfa90352"
FEW_STEP_HF = "Efficient-Large-Model/LongLive-Plug-Wan2.1-T2V-14B-few-step@f125af0533b94e2cee0163a7a42bb736d59a1d37"
WAN21 = "Wan-Video/Wan2.1@9737cba9c1c3c4d04b33fcad41c111989865d315"
LIGHTX2V = "ModelTC/LightX2V@8a97c7591d7252ef491392e83e1eb18617ac9368"


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    profile_id: str
    requires_adapters: bool
    solver: str  # "flow_unipc" | "euler" | "lcm"
    steps: int
    shift: float
    cfg: float
    status: str  # "official" | "comparison"
    source: str
    step_list: tuple[int, ...] | None = None
    note: str = ""

    def sigmas(self) -> tuple[float, ...]:
        if self.solver == "flow_unipc":
            return flow_unipc_schedule(self.steps, self.shift).sigmas
        return warped_step_list_sigmas(self.step_list, self.shift)

    def model_timesteps(self) -> tuple[float, ...]:
        """Timesteps the diffusion model receives at each step."""
        if self.solver == "flow_unipc":
            return tuple(float(t) for t in flow_unipc_schedule(self.steps, self.shift).timesteps)
        return tuple(s * 1000.0 for s in self.sigmas()[:-1])

    def describe(self) -> dict:
        return {
            "recipe": self.recipe_id,
            "status": self.status,
            "source": self.source,
            "profile": self.profile_id,
            "requires_longlive_plug_adapters": self.requires_adapters,
            "solver": self.solver,
            "steps": self.steps,
            "shift": self.shift,
            "runtime_cfg_scale": self.cfg,
            "conditional_only": self.cfg == 1.0,
            "model_evaluations": model_evaluations(self.steps, self.cfg),
            "sigmas": list(self.sigmas()),
            "model_timesteps": list(self.model_timesteps()),
            "timestep_handling": (
                "int64-truncated sigma*1000 (as upstream)" if self.solver == "flow_unipc" else "float32 sigma*1000 (as upstream)"
            ),
            "note": self.note,
        }


RECIPES: dict[str, Recipe] = {
    r.recipe_id: r
    for r in (
        Recipe(
            recipe_id="wan21-14b-plug-4step-unipc",
            profile_id="wan2.1-t2v-14b",
            requires_adapters=True,
            solver="flow_unipc",
            steps=4,
            shift=5.0,
            cfg=1.0,
            status="official",
            source=f"{LONGLIVE}: LongLive-Plug/inference.py + configs/wan21_dmd.yaml (sampling_steps=4, guidance_scale=1.0, timestep_shift=5.0)",
            note="Primary recipe. Use with few-step LoRA 1.0 + CFG LoRA 0.5.",
        ),
        Recipe(
            recipe_id="wan21-14b-plug-4step-euler",
            profile_id="wan2.1-t2v-14b",
            requires_adapters=True,
            solver="euler",
            steps=4,
            shift=5.0,
            cfg=1.0,
            step_list=(1000, 750, 500, 250),
            status="official",
            source=f"{FEW_STEP_HF}: inference_config.json (denoising_step_list, sample_shift=5, enable_cfg=false); step rule from {LIGHTX2V} WanStepDistillScheduler",
            note="Alternative published with the few-step adapter for the LightX2V runtime.",
        ),
        Recipe(
            recipe_id="wan21-14b-plug-4step-lcm",
            profile_id="wan2.1-t2v-14b",
            requires_adapters=True,
            solver="lcm",
            steps=4,
            shift=5.0,
            cfg=1.0,
            step_list=(1000, 750, 500, 250),
            status="official",
            source=f"{FEW_STEP_HF}: training_config.yaml (Self-Forcing-Plus rollout: x0 prediction re-noised with fresh Gaussian noise at warped steps)",
            note="Training-rollout parity; stochastic (re-noising draws depend on the seed).",
        ),
        Recipe(
            recipe_id="wan21-14b-base-50step-unipc-cfg5",
            profile_id="wan2.1-t2v-14b",
            requires_adapters=False,
            solver="flow_unipc",
            steps=50,
            shift=5.0,
            cfg=5.0,
            status="official",
            source=f"{WAN21}: generate.py t2v-14B defaults (sample_solver=unipc, sample_steps=50, sample_shift=5.0, sample_guide_scale=5.0)",
            note="Baseline. Use Wan's default negative prompt.",
        ),
        Recipe(
            recipe_id="wan21-14b-base-4step-unipc-cfg5-naive",
            profile_id="wan2.1-t2v-14b",
            requires_adapters=False,
            solver="flow_unipc",
            steps=4,
            shift=5.0,
            cfg=5.0,
            status="comparison",
            source="Not an upstream recipe: the official base recipe with steps reduced to 4.",
            note="Shows what step reduction alone does without the adapters.",
        ),
    )
}

DEFAULT_PLUG_RECIPE = "wan21-14b-plug-4step-unipc"
DEFAULT_BASE_RECIPE = "wan21-14b-base-50step-unipc-cfg5"

# Wan2.1 shared_config.sample_neg_prompt (verbatim).
WAN_DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
    "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
