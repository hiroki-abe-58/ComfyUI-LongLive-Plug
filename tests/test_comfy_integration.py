"""Tests against a real ComfyUI checkout (CPU, tiny Wan-shaped model)."""

from __future__ import annotations

import json
import sys

import pytest
import torch
from safetensors.torch import save_file
from tiny_wan import tiny_lora, tiny_profile, tiny_state_dict

from longlive_plug.adapters import reference_merge
from longlive_plug.profiles import PinnedAdapter
from longlive_plug.recipes import Recipe
from longlive_plug.schedules import flow_unipc_schedule

pytestmark = pytest.mark.comfy

NODE_IDS = ("LongLivePlugApplyAdapters", "LongLivePlugRecipe")


def _node_module(comfyui, node_id):
    return sys.modules[comfyui["nodes"].NODE_CLASS_MAPPINGS[node_id].__module__]


@pytest.fixture
def env(comfyui, tmp_path, monkeypatch):
    """Tiny model + tiny adapters in a loras folder whose path has spaces and Unicode."""
    import comfy.sd
    import folder_paths

    mod = _node_module(comfyui, "LongLivePlugApplyAdapters")
    lora_dir = tmp_path / "モデル dir with spaces" / "loras"
    lora_dir.mkdir(parents=True)
    prof = tiny_profile()
    fs = tiny_lora(prof, rank=4, seed=1, dtype=torch.bfloat16)
    cfg = tiny_lora(prof, rank=8, seed=2, prefix="base_model.model.")
    save_file(fs, str(lora_dir / "few step.safetensors"), metadata={"rank": "4", "alpha": "4"})
    (lora_dir / "cfg").mkdir()
    save_file(cfg, str(lora_dir / "cfg" / "adapter_model.safetensors"))
    (lora_dir / "cfg" / "adapter_config.json").write_text(
        json.dumps({"r": 8, "lora_alpha": 16, "use_dora": False, "use_rslora": False}), encoding="utf-8"
    )
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "loras", ([str(lora_dir)], folder_paths.supported_pt_extensions))
    folder_paths.filename_list_cache.pop("loras", None)

    monkeypatch.setitem(mod.PROFILES, prof.profile_id, prof)
    recipes = {
        "tiny-plug": Recipe("tiny-plug", prof.profile_id, True, "flow_unipc", 4, 5.0, 1.0, "official", "test"),
        "tiny-base": Recipe("tiny-base", prof.profile_id, False, "flow_unipc", 3, 5.0, 5.0, "official", "test"),
        "tiny-euler": Recipe(
            "tiny-euler", prof.profile_id, True, "euler", 4, 5.0, 1.0, "official", "test", step_list=(1000, 750, 500, 250)
        ),
    }
    for k, v in recipes.items():
        monkeypatch.setitem(mod.RECIPES, k, v)
    base_sd = tiny_state_dict(seed=0)
    model = comfy.sd.load_diffusion_model_state_dict({k: v.clone() for k, v in base_sd.items()})
    cfg_name = "cfg/adapter_model.safetensors"
    return {
        "mod": mod,
        "profile": prof,
        "model": model,
        "base_sd": base_sd,
        "fs": fs,
        "cfg": cfg,
        "fs_name": "few step.safetensors",
        "cfg_name": cfg_name,
        "lora_dir": lora_dir,
    }


def _apply(env, **kw):
    node = env["mod"].LongLivePlugApplyAdapters()
    args = dict(
        model=env["model"],
        profile=env["profile"].profile_id,
        few_step_lora=env["fs_name"],
        cfg_lora=env["cfg_name"],
        few_step_weight=1.0,
        cfg_weight=0.5,
        verify_sha256=True,
        allow_untested_derivative=False,
    )
    args.update(kw)
    return node.apply(**args)


def test_nodes_registered_through_comfy_loader(comfyui):
    nodes = comfyui["nodes"]
    for node_id in NODE_IDS:
        cls = nodes.NODE_CLASS_MAPPINGS[node_id]
        assert node_id in nodes.NODE_DISPLAY_NAME_MAPPINGS
        spec = cls.INPUT_TYPES()
        assert "required" in spec and cls.FUNCTION and cls.RETURN_TYPES
    recipe_spec = nodes.NODE_CLASS_MAPPINGS["LongLivePlugRecipe"].INPUT_TYPES()
    assert "wan21-14b-plug-4step-unipc" in recipe_spec["required"]["recipe"][0]
    assert nodes.NODE_CLASS_MAPPINGS["LongLivePlugRecipe"].RETURN_TYPES == ("GUIDER", "SAMPLER", "SIGMAS", "STRING")


def test_patch_matches_upstream_merge_and_unpatch_restores(env):
    model, base_sd = env["model"], env["base_sd"]
    patched, report = _apply(env)
    rep = json.loads(report)
    assert rep["recipe_conformance"] == "custom"  # tiny adapters are not pinned releases
    by_role = {a["role"]: a for a in rep["adapters"]}
    assert by_role["few_step"]["alpha_source"] == "safetensors_metadata"
    assert by_role["cfg"]["alpha_source"] == "adapter_config.json"
    assert by_role["cfg"]["key_format"] == "peft" and by_role["few_step"]["key_format"] == "native"
    for a in rep["adapters"]:
        assert a["patches_applied"] == a["targets_expected"] == len(env["profile"].expected_targets())
        assert a["application_rate"] == 1.0 and a["unused_adapter_keys"] == 0
    assert by_role["cfg"]["effective_scale"] == pytest.approx(0.5 * 16 / 8)
    # No absolute paths in the report.
    assert str(env["lora_dir"]) not in report and "モデル" not in report

    assert model.model_options.get("longlive_plug") is None  # clone semantics: input untouched
    assert len(model.patches) == 0
    patched.patch_model(device_to=torch.device("cpu"))
    try:
        live = patched.model.state_dict()
        for target in ("blocks.0.self_attn.q", "blocks.1.ffn.2", "blocks.0.cross_attn.o"):
            w = base_sd[target + ".weight"]
            ups = [
                (env["fs"][f"{target}.lora_A.weight"], env["fs"][f"{target}.lora_B.weight"], 1.0 * 4 / 4),
                (
                    env["cfg"][f"base_model.model.{target}.lora_A.weight"],
                    env["cfg"][f"base_model.model.{target}.lora_B.weight"],
                    0.5 * 16 / 8,
                ),
            ]
            ref = reference_merge(w, ups)
            got = live["diffusion_model." + target + ".weight"].float().cpu()
            torch.testing.assert_close(got, ref.float(), rtol=1e-6, atol=1e-7)
            assert not torch.equal(got, w.float())
    finally:
        patched.unpatch_model(device_to=torch.device("cpu"))
    restored = model.model.state_dict()
    for k, v in base_sd.items():
        assert torch.equal(restored["diffusion_model." + k].float().cpu(), v.float()), k


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        ({"cfg_lora": "few step.safetensors"}, "same file"),
        ({"profile": "wan2.1-t2v-14b"}, "is not Wan2.1-T2V-14B"),
    ],
)
def test_apply_rejects_bad_inputs(env, kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        _apply(env, **kwargs)


def test_apply_rejects_unused_keys_and_shape_mismatch(env):
    bad = dict(env["fs"])
    bad["blocks.0.self_attn.q.lora_magnitude_vector"] = torch.ones(256)
    save_file(bad, str(env["lora_dir"] / "dora.safetensors"), metadata={"alpha": "4"})
    with pytest.raises(ValueError, match="unsupported tensor"):
        _apply(env, few_step_lora="dora.safetensors")
    other = tiny_lora(env["profile"], rank=4)
    other["blocks.0.ffn.0.lora_B.weight"] = torch.zeros(999, 4)
    save_file(other, str(env["lora_dir"] / "wrong.safetensors"), metadata={"alpha": "4"})
    with pytest.raises(ValueError, match="shape mismatch"):
        _apply(env, few_step_lora="wrong.safetensors")
    missing_alpha = tiny_lora(env["profile"], rank=4)
    save_file(missing_alpha, str(env["lora_dir"] / "noalpha.safetensors"))
    with pytest.raises(ValueError, match="cannot determine LoRA alpha"):
        _apply(env, few_step_lora="noalpha.safetensors")


def test_swapped_pinned_adapters_are_detected(env, monkeypatch):
    from longlive_plug.adapters import file_sha256

    fs_sha = file_sha256(env["lora_dir"] / env["fs_name"])
    cfg_sha = file_sha256(env["lora_dir"] / "cfg" / "adapter_model.safetensors")
    pinned = (
        PinnedAdapter("few_step", "test/fs", "rev", env["fs_name"], fs_sha, 4, 4.0, "bfloat16", "native"),
        PinnedAdapter("cfg", "test/cfg", "rev", "adapter_model.safetensors", cfg_sha, 8, 16.0, "float32", "peft"),
    )
    prof = tiny_profile(pinned)
    monkeypatch.setitem(env["mod"].PROFILES, prof.profile_id, prof)
    _, report = _apply(env)
    assert json.loads(report)["recipe_conformance"] == "official"
    with pytest.raises(ValueError, match="swapped"):
        _apply(env, few_step_lora=env["cfg_name"], cfg_lora=env["fs_name"])


def _run_sampler(env, model, recipe, cfg_cond=True, seed=7):
    from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

    calls = []

    def wrapper(apply_model, args):
        calls.append(
            {"cond_or_uncond": list(args["cond_or_uncond"]), "t": args["timestep"].float().cpu().tolist(), "x": args["input"].clone()}
        )
        return apply_model(args["input"], args["timestep"], **args["c"])

    m = model.clone()
    m.set_model_unet_function_wrapper(wrapper)
    pos = [[torch.randn(1, 8, 4096, generator=torch.Generator().manual_seed(3)), {}]]
    neg = [[torch.randn(1, 8, 4096, generator=torch.Generator().manual_seed(4)), {}]]
    guider, sampler, sigmas, report = (
        env["mod"].LongLivePlugRecipe().build(m, pos, recipe, "upstream", True, negative=neg if cfg_cond else None)
    )
    latent = torch.zeros(1, 16, 2, 4, 4)
    noise = Noise_RandomNoise(seed).generate_noise({"samples": latent})
    out = guider.sample(noise, latent, sampler, sigmas, seed=seed, disable_pbar=True)
    return out, calls, json.loads(report), noise


def test_plug_recipe_runs_conditional_only_with_upstream_timesteps(env):
    patched, _ = _apply(env)
    out, calls, rep, noise = _run_sampler(env, patched, "tiny-plug", cfg_cond=False)
    sched = flow_unipc_schedule(4, 5.0)
    assert len(calls) == 4 == rep["model_evaluations"]
    assert all(c["cond_or_uncond"] == [0] for c in calls)  # conditional-only forward
    got_t = [c["t"][0] * 1000.0 for c in calls]
    assert got_t == pytest.approx([float(t) for t in sched.timesteps], abs=1e-3)  # int-truncated like upstream
    assert torch.equal(calls[0]["x"].cpu(), noise.cpu())  # x_T = noise (upstream init)
    assert torch.isfinite(out).all() and out.shape == noise.shape
    assert rep["runtime_cfg_scale"] == 1.0 and rep["adapters"]["profile"] == "tiny-wan-test"


def test_base_recipe_runs_cond_and_uncond_and_rejects_patched_models(env):
    out, calls, rep, _ = _run_sampler(env, env["model"], "tiny-base")
    passes = sum(len(c["cond_or_uncond"]) for c in calls)
    assert passes == 6 == rep["model_evaluations"]
    assert torch.isfinite(out).all()
    patched, _ = _apply(env)
    with pytest.raises(ValueError, match="base-model recipe"):
        _run_sampler(env, patched, "tiny-base")
    with pytest.raises(ValueError, match="expects the LongLive-Plug adapter pair"):
        _run_sampler(env, env["model"], "tiny-plug")


def test_euler_steplist_recipe_matches_warped_sigmas(env):
    patched, _ = _apply(env)
    out, calls, rep, noise = _run_sampler(env, patched, "tiny-euler", cfg_cond=False)
    assert [round(c["t"][0], 6) for c in calls] == pytest.approx([1.0, 0.9375, 0.833333, 0.625], abs=1e-6)
    assert torch.equal(calls[0]["x"].cpu(), noise.cpu())
    assert torch.isfinite(out).all()


def test_flow_unipc_rejects_foreign_sigmas(env):
    from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

    patched, _ = _apply(env)
    pos = [[torch.randn(1, 8, 4096), {}]]
    guider, sampler, sigmas, _ = env["mod"].LongLivePlugRecipe().build(patched, pos, "tiny-plug", "upstream", True)
    latent = torch.zeros(1, 16, 1, 4, 4)
    noise = Noise_RandomNoise(1).generate_noise({"samples": latent})
    with pytest.raises(ValueError, match="sigmas do not match"):
        guider.sample(noise, latent, sampler, torch.tensor([1.0, 0.5, 0.0]), seed=1, disable_pbar=True)


def test_upstream_init_refuses_non_empty_latent(env):
    from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

    patched, _ = _apply(env)
    pos = [[torch.randn(1, 8, 4096), {}]]
    guider, sampler, sigmas, _ = env["mod"].LongLivePlugRecipe().build(patched, pos, "tiny-plug", "upstream", True)
    latent = torch.ones(1, 16, 1, 4, 4)
    noise = Noise_RandomNoise(1).generate_noise({"samples": latent})
    with pytest.raises(ValueError, match="empty latent"):
        guider.sample(noise, latent, sampler, sigmas, seed=1, disable_pbar=True)


RUNTIME_FILES = ("__init__.py", "pyproject.toml", "longlive_plug")


@pytest.mark.parametrize("dirname", ["longlive-plug", "LongLive Plug copy ü"])
def test_package_loads_under_any_directory_name(comfyui, tmp_path, dirname):
    """Registry/Manager installs into custom_nodes/<node id>; nothing may depend on the folder name."""
    import asyncio
    import shutil
    from pathlib import Path

    from longlive_plug import comfy_nodes as source

    repo = Path(__file__).resolve().parents[1]
    target = tmp_path / "custom_nodes" / dirname
    target.mkdir(parents=True)
    for name in RUNTIME_FILES:
        src = repo / name
        if src.is_dir():
            shutil.copytree(src, target / name, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(src, target / name)
    nodes = comfyui["nodes"]
    saved = (dict(nodes.NODE_CLASS_MAPPINGS), dict(nodes.NODE_DISPLAY_NAME_MAPPINGS))
    try:
        assert asyncio.run(nodes.load_custom_node(str(target)))
        for name, display in source.NODE_DISPLAY_NAME_MAPPINGS.items():
            cls = nodes.NODE_CLASS_MAPPINGS[name]
            module_file = Path(sys.modules[cls.__module__].__file__).resolve()
            assert module_file.is_relative_to(target.resolve()), module_file
            assert nodes.NODE_DISPLAY_NAME_MAPPINGS[name] == display
        loaded = {
            n
            for n, c in nodes.NODE_CLASS_MAPPINGS.items()
            if Path(sys.modules[c.__module__].__file__).resolve().is_relative_to(target.resolve())
        }
        assert loaded == set(source.NODE_CLASS_MAPPINGS)
    finally:
        nodes.NODE_CLASS_MAPPINGS.clear()
        nodes.NODE_CLASS_MAPPINGS.update(saved[0])
        nodes.NODE_DISPLAY_NAME_MAPPINGS.clear()
        nodes.NODE_DISPLAY_NAME_MAPPINGS.update(saved[1])
