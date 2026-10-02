"""The shipped workflows: structure, references, and ComfyUI's own validation."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_workflows  # noqa: E402

API_DIR = ROOT / "workflows" / "api"
API_FILES = sorted(API_DIR.glob("*.json"))
GUI_FILES = sorted((ROOT / "workflows").glob("*.json"))
MODEL_FILES = {
    "diffusion_models": [build_workflows.BASE_MODEL],
    "text_encoders": [build_workflows.TEXT_ENCODER],
    "vae": [build_workflows.VAE],
    "loras": [build_workflows.FEW_STEP_LORA, build_workflows.CFG_LORA],
}


def test_workflow_files_exist_and_pair_up():
    assert (
        {p.stem for p in API_FILES}
        == {p.stem for p in GUI_FILES}
        == {
            "wan21_t2v_14b_baseline_50step",
            "wan21_t2v_14b_longlive_plug_4step",
        }
    )


@pytest.mark.parametrize("path", API_FILES, ids=lambda p: p.stem)
def test_api_prompt_is_reproducible_from_builder(path):
    variant = path.stem.replace("wan21_t2v_14b_", "")
    assert json.loads(path.read_text(encoding="utf-8")) == build_workflows.build(variant)


@pytest.mark.parametrize("path", API_FILES, ids=lambda p: p.stem)
def test_api_prompt_links_point_to_existing_nodes(path):
    prompt = json.loads(path.read_text(encoding="utf-8"))
    for node_id, node in prompt.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list):
                assert value[0] in prompt, f"{node_id}.{name} -> missing node {value[0]}"
                assert isinstance(value[1], int)


def test_recipes_and_weights_in_shipped_prompts():
    plug = json.loads((API_DIR / "wan21_t2v_14b_longlive_plug_4step.json").read_text(encoding="utf-8"))
    base = json.loads((API_DIR / "wan21_t2v_14b_baseline_50step.json").read_text(encoding="utf-8"))
    apply = next(n for n in plug.values() if n["class_type"] == "LongLivePlugApplyAdapters")["inputs"]
    assert (apply["few_step_weight"], apply["cfg_weight"]) == (1.0, 0.5)
    assert next(n for n in plug.values() if n["class_type"] == "LongLivePlugRecipe")["inputs"]["recipe"] == "wan21-14b-plug-4step-unipc"
    base_recipe = next(n for n in base.values() if n["class_type"] == "LongLivePlugRecipe")["inputs"]
    assert base_recipe["recipe"] == "wan21-14b-base-50step-unipc-cfg5" and "negative" in base_recipe
    assert not any(n["class_type"] == "LongLivePlugApplyAdapters" for n in base.values())
    for p in (plug, base):  # same prompt / seed / size for the comparison
        assert next(n for n in p.values() if n["class_type"] == "RandomNoise")["inputs"]["noise_seed"] == build_workflows.DEMO_SEED
        assert next(n for n in p.values() if n["class_type"] == "EmptyHunyuanLatentVideo")["inputs"] == {
            **build_workflows.DEMO_SIZE,
            "batch_size": 1,
        }


@pytest.mark.parametrize("path", GUI_FILES, ids=lambda p: p.stem)
def test_gui_workflow_links_are_consistent(path):
    wf = json.loads(path.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in wf["nodes"]}
    for link in wf["links"]:
        link_id, src, src_slot, dst, dst_slot, _type = link
        assert link_id in (nodes[src]["outputs"][src_slot].get("links") or [])
        assert nodes[dst]["inputs"][dst_slot]["link"] == link_id
    seeds = [n for n in wf["nodes"] if n["type"] == "RandomNoise"]
    assert seeds and all(n["widgets_values"][1] == "fixed" for n in seeds)
    api = json.loads((API_DIR / path.name).read_text(encoding="utf-8"))
    assert sorted(n["type"] for n in wf["nodes"]) == sorted(n["class_type"] for n in api.values())
    text = path.read_text(encoding="utf-8")
    for value in (build_workflows.FEW_STEP_LORA, build_workflows.BASE_MODEL, str(build_workflows.DEMO_SEED)):
        if "plug" in path.stem or value != build_workflows.FEW_STEP_LORA:
            assert value in text


@pytest.mark.comfy
@pytest.mark.parametrize("path", API_FILES, ids=lambda p: p.stem)
def test_comfyui_validate_prompt_accepts_shipped_prompts(comfyui, path, tmp_path, monkeypatch):
    import execution
    import folder_paths

    for folder, names in MODEL_FILES.items():
        d = tmp_path / folder
        d.mkdir()
        for name in names:
            (d / name).write_bytes(b"")
        monkeypatch.setitem(folder_paths.folder_names_and_paths, folder, ([str(d)], folder_paths.supported_pt_extensions))
        folder_paths.filename_list_cache.pop(folder, None)
    prompt = json.loads(path.read_text(encoding="utf-8"))
    valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt("test", prompt, None))
    # validate_prompt reports valid=True if *any* output validates, so check all of them.
    assert valid and not node_errors, (error, node_errors)
    output_nodes = {nid for nid, n in prompt.items() if n["class_type"] in ("SaveVideo", "PreviewAny")}
    assert set(outputs) == output_nodes


@pytest.mark.comfy
@pytest.mark.parametrize(
    "node_class,field,bad",
    [("LongLivePlugRecipe", "recipe", "no-such-recipe"), ("LongLivePlugApplyAdapters", "cfg_lora", "missing.safetensors")],
)
def test_comfyui_validate_prompt_rejects_bad_values(comfyui, tmp_path, monkeypatch, node_class, field, bad):
    import execution
    import folder_paths

    for folder, names in MODEL_FILES.items():
        d = tmp_path / folder
        d.mkdir()
        for name in names:
            (d / name).write_bytes(b"")
        monkeypatch.setitem(folder_paths.folder_names_and_paths, folder, ([str(d)], folder_paths.supported_pt_extensions))
        folder_paths.filename_list_cache.pop(folder, None)
    prompt = build_workflows.build("longlive_plug_4step")
    next(n for n in prompt.values() if n["class_type"] == node_class)["inputs"][field] = bad
    valid, _error, _outputs, node_errors = asyncio.run(execution.validate_prompt("test", prompt, None))
    assert node_errors  # valid may still be True when another output validates
