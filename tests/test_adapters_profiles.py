import json

import pytest
import torch
from tiny_wan import tiny_lora, tiny_profile, tiny_state_dict

from longlive_plug.adapters import (
    AdapterError,
    analyze_adapter,
    comfy_lora_dict,
    read_sibling_peft_config,
    reference_merge,
    strip_prefix,
    validate_against_base,
)
from longlive_plug.profiles import WAN21_T2V_14B, PinnedAdapter, fingerprint

PROF = tiny_profile()


def shapes_of(sd, prefix="diffusion_model."):
    return {prefix + k: tuple(v.shape) for k, v in sd.items()}, {prefix + k: v.dtype for k, v in sd.items()}


def test_wan21_14b_profile_matches_real_layout():
    targets = WAN21_T2V_14B.expected_targets()
    assert len(targets) == 400 == len(set(targets))  # upstream expected_target_modules: 400
    assert WAN21_T2V_14B.target_shape("blocks.3.ffn.0") == (13824, 5120)
    assert WAN21_T2V_14B.target_shape("blocks.3.ffn.2") == (5120, 13824)
    assert WAN21_T2V_14B.target_shape("blocks.3.cross_attn.v") == (5120, 5120)
    roles = {a.role for a in WAN21_T2V_14B.adapters}
    assert roles == {"few_step", "cfg"}
    assert all(len(a.sha256) == 64 and len(a.revision) == 40 for a in WAN21_T2V_14B.adapters)
    assert WAN21_T2V_14B.recommended_weights == {"few_step": 1.0, "cfg": 0.5}


@pytest.mark.parametrize(
    "key,expected",
    [
        ("base_model.model.blocks.0.ffn.0.lora_A.weight", ("base_model.model.", "blocks.0.ffn.0.lora_A.weight")),
        ("module.base_model.model.x.lora_B.weight", ("module.base_model.model.", "x.lora_B.weight")),
        ("diffusion_model.blocks.1.q.lora_A.weight", ("diffusion_model.", "blocks.1.q.lora_A.weight")),
        ("blocks.0.ffn.0.lora_A.weight", ("", "blocks.0.ffn.0.lora_A.weight")),
    ],
)
def test_strip_prefix(key, expected):
    assert strip_prefix(key) == expected


def test_peft_default_suffix_and_alpha_from_config():
    sd = {k.replace(".weight", ".default.weight"): v for k, v in tiny_lora(PROF, rank=4, prefix="base_model.model.").items()}
    a = analyze_adapter("cfg", "x", sd, peft_config={"r": 4, "lora_alpha": 8})
    assert a.key_format == "peft" and a.rank == 4 and a.alpha == 8.0 and a.alpha_source == "adapter_config.json"
    assert len(a.pairs) == len(PROF.expected_targets())


def test_alpha_sources_must_agree():
    sd = tiny_lora(PROF, rank=4)
    with pytest.raises(AdapterError, match="conflicting alpha"):
        analyze_adapter("few_step", "x", sd, metadata={"alpha": "4"}, peft_config={"r": 4, "lora_alpha": 8})
    with pytest.raises(AdapterError, match="metadata rank"):
        analyze_adapter("few_step", "x", sd, metadata={"alpha": "4", "rank": "8"})
    with pytest.raises(AdapterError, match="cannot determine LoRA alpha"):
        analyze_adapter("few_step", "x", sd)


def test_per_module_alpha_tensors():
    sd = tiny_lora(PROF, rank=4)
    for t in PROF.expected_targets():
        sd[f"{t}.alpha"] = torch.tensor(2.0)
    a = analyze_adapter("few_step", "x", sd)
    assert a.alpha == 2.0 and a.alpha_source == "tensor"
    del sd[f"{PROF.expected_targets()[0]}.alpha"]
    with pytest.raises(AdapterError, match="only some targets"):
        analyze_adapter("few_step", "x", sd)


@pytest.mark.parametrize(
    "mutate,msg",
    [
        (lambda sd: sd.pop(next(k for k in sd if k.endswith("lora_B.weight"))), "incomplete LoRA pair"),
        (lambda sd: sd.__setitem__("blocks.0.self_attn.q.lora_magnitude_vector", torch.ones(1)), "unsupported tensor"),
        (lambda sd: sd.__setitem__("blocks.0.self_attn.q.lora_A.weight", torch.zeros(3, 256)), "invalid LoRA dimensions"),
        (
            lambda sd: sd.__setitem__("base_model.model.blocks.0.self_attn.k.lora_A.weight", torch.zeros(4, 256)),
            "duplicate|mixed key prefixes",
        ),
    ],
)
def test_malformed_adapters_rejected(mutate, msg):
    sd = tiny_lora(PROF, rank=4)
    mutate(sd)
    with pytest.raises(AdapterError, match=msg):
        analyze_adapter("few_step", "x", sd, metadata={"alpha": "4"})


def test_mixed_ranks_and_rslora_rejected():
    sd = tiny_lora(PROF, rank=4)
    t = PROF.expected_targets()[0]
    sd[f"{t}.lora_A.weight"] = torch.zeros(2, 256)
    sd[f"{t}.lora_B.weight"] = torch.zeros(256, 2)
    with pytest.raises(AdapterError, match="mixed ranks"):
        analyze_adapter("few_step", "x", sd, metadata={"alpha": "4"})
    with pytest.raises(AdapterError, match="rsLoRA"):
        analyze_adapter("cfg", "x", tiny_lora(PROF, rank=4), peft_config={"r": 4, "lora_alpha": 4, "use_rslora": True})


def test_pinned_release_supplies_alpha_and_rank_check():
    sd = tiny_lora(PROF, rank=4)
    prof = tiny_profile((PinnedAdapter("cfg", "r", "v", "f", "a" * 64, 4, 4.0, "float32", "native"),))
    a = analyze_adapter("cfg", "x", sd, sha256="a" * 64, profile=prof)
    assert a.alpha == 4.0 and a.alpha_source == "pinned_release" and a.pinned.role == "cfg"
    bad = tiny_profile((PinnedAdapter("cfg", "r", "v", "f", "a" * 64, 8, 8.0, "float32", "native"),))
    with pytest.raises(AdapterError, match="pinned release rank"):
        analyze_adapter("cfg", "x", sd, sha256="a" * 64, profile=bad)


def test_validate_against_base_detects_other_backbones():
    shapes, dtypes = shapes_of(tiny_state_dict())
    a = analyze_adapter("few_step", "x", tiny_lora(PROF, rank=4), metadata={"alpha": "4"})
    cov = validate_against_base(a, shapes, dtypes, PROF)
    assert cov["targets_matched"] == cov["targets_expected"] == len(PROF.expected_targets())
    # An adapter for a wider backbone (e.g. 14B weights on a smaller model) must not apply.
    wide = tiny_profile()
    object.__setattr__(wide, "dim", 512)
    with pytest.raises(AdapterError, match="shape mismatch"):
        validate_against_base(analyze_adapter("x", "x", tiny_lora(wide, rank=4), metadata={"alpha": "4"}), shapes, dtypes, PROF)
    # Quantized base weights are refused.
    q_dtypes = {k: (torch.float8_e4m3fn if k.endswith("self_attn.q.weight") else v) for k, v in dtypes.items()}
    with pytest.raises(AdapterError, match="quantized"):
        validate_against_base(a, shapes, q_dtypes, PROF)
    # Extra targets outside the released layout are refused.
    extra = tiny_lora(PROF, rank=4)
    extra["time_embedding.0.lora_A.weight"] = torch.zeros(4, 256)
    extra["time_embedding.0.lora_B.weight"] = torch.zeros(256, 4)
    with pytest.raises(AdapterError, match="differ from"):
        validate_against_base(analyze_adapter("x", "x", extra, metadata={"alpha": "4"}), shapes, dtypes, PROF)


def test_nonfinite_adapter_rejected():
    shapes, dtypes = shapes_of(tiny_state_dict())
    sd = tiny_lora(PROF, rank=4)
    sd["blocks.1.ffn.2.lora_B.weight"][0, 0] = float("nan")
    a = analyze_adapter("x", "x", sd, metadata={"alpha": "4"})
    with pytest.raises(AdapterError, match="non-finite"):
        validate_against_base(a, shapes, dtypes, PROF)


def test_fingerprint_exact_derivative_mismatch():
    assert fingerprint(PROF, shapes_of(tiny_state_dict())[0])["match"] == "exact"
    der = fingerprint(PROF, shapes_of(tiny_state_dict(extra=True))[0])
    assert der["match"] == "derivative" and any("image cross-attention" in r for r in der["reasons"])
    small = {k: v for k, v in shapes_of(tiny_state_dict())[0].items() if ".blocks.1." not in k}
    assert fingerprint(PROF, small)["match"] == "mismatch"
    assert fingerprint(WAN21_T2V_14B, shapes_of(tiny_state_dict())[0])["match"] == "mismatch"


def test_comfy_lora_dict_layout_and_reference_merge():
    sd = tiny_lora(PROF, rank=4, prefix="base_model.model.")
    a = analyze_adapter("cfg", "x", sd, peft_config={"r": 4, "lora_alpha": 8})
    lora_sd, to_load = comfy_lora_dict(a)
    t = PROF.expected_targets()[5]
    assert to_load[t] == f"diffusion_model.{t}.weight"
    assert float(lora_sd[f"{t}.alpha"]) == 8.0
    assert torch.equal(lora_sd[f"{t}.lora_A.weight"], sd[f"base_model.model.{t}.lora_A.weight"])
    w = torch.randn(PROF.target_shape(t), dtype=torch.bfloat16)
    lo, up = sd[f"base_model.model.{t}.lora_A.weight"], sd[f"base_model.model.{t}.lora_B.weight"]
    merged = reference_merge(w, [(lo, up, 0.5 * 8 / 4)])
    expected = (w.float() + 0.5 * 2.0 * (up.float() @ lo.float())).to(torch.bfloat16)
    assert merged.dtype == torch.bfloat16
    torch.testing.assert_close(merged.float(), expected.float(), rtol=0, atol=2**-6)


def test_sibling_config_only_for_peft_filename(tmp_path):
    d = tmp_path / "dir with spaces ü"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 4}), encoding="utf-8")
    (d / "adapter_model.safetensors").write_bytes(b"")
    (d / "renamed.safetensors").write_bytes(b"")
    assert read_sibling_peft_config(d / "adapter_model.safetensors") == {"r": 4, "lora_alpha": 4}
    assert read_sibling_peft_config(d / "renamed.safetensors") is None
    (d / "adapter_config.json").write_text("{not json", encoding="utf-8")
    assert read_sibling_peft_config(d / "adapter_model.safetensors") is None
