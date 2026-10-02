"""LoRA adapter parsing, validation and coverage reporting (ComfyUI-independent).

The rules follow LongLive-Plug ``scripts/merge_lora.py``: only plain linear
LoRA A/B pairs are accepted, the PEFT ``base_model.model.`` prefix is
stripped, every target must exist in the base with shape ``(B.out, A.in)``,
and ``delta_W = weight * (alpha / rank) * B @ A`` is accumulated in float32.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .profiles import BackboneProfile, PinnedAdapter

KNOWN_PREFIXES = (
    "base_model.model.",
    "model.base_model.model.",
    "module.base_model.model.",
    "diffusion_model.",
)
_LORA_RE = re.compile(r"^(?P<target>.+)\.lora_(?P<side>[AB])(?:\.default)?\.weight$")
_ALPHA_RE = re.compile(r"^(?P<target>.+)\.alpha$")
_CONFIG_MAX_BYTES = 1 << 20


class AdapterError(ValueError):
    """Raised when an adapter cannot be applied faithfully."""


@dataclass
class AdapterAnalysis:
    role: str
    file_name: str
    sha256: str | None
    pinned: PinnedAdapter | None
    key_format: str
    pairs: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    alphas: dict[str, float] = field(default_factory=dict)
    rank: int | None = None
    alpha: float | None = None
    alpha_source: str | None = None
    unsupported_keys: list[str] = field(default_factory=list)
    dtypes: list[str] = field(default_factory=list)

    @property
    def tensor_count(self) -> int:
        return 2 * len(self.pairs) + len(self.alphas)


def strip_prefix(key: str) -> tuple[str, str]:
    for prefix in KNOWN_PREFIXES:
        if key.startswith(prefix):
            return prefix, key[len(prefix) :]
    return "", key


_SHA_CACHE: dict[tuple[str, int, int], str] = {}


def file_sha256(path: str | os.PathLike) -> str:
    """SHA-256 of a file, cached per (realpath, size, mtime)."""
    real = os.path.realpath(path)
    st = os.stat(real)
    key = (real, st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(real, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def read_sibling_peft_config(adapter_path: str | os.PathLike) -> dict | None:
    """Read ``adapter_config.json`` next to a PEFT ``adapter_model.safetensors``.

    Only the PEFT file name is paired with the config, so a renamed adapter
    in a shared folder never picks up another adapter's config.
    """
    if Path(adapter_path).name != "adapter_model.safetensors":
        return None
    cfg = Path(adapter_path).with_name("adapter_config.json")
    if not cfg.is_file() or cfg.stat().st_size > _CONFIG_MAX_BYTES:
        return None
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _parse_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def analyze_adapter(
    role: str,
    file_name: str,
    tensors: dict[str, torch.Tensor],
    *,
    metadata: dict | None = None,
    peft_config: dict | None = None,
    sha256: str | None = None,
    profile: BackboneProfile | None = None,
) -> AdapterAnalysis:
    """Parse an adapter state dict and resolve rank/alpha. Raises AdapterError."""
    pinned = profile.pinned(sha256) if profile else None
    prefixes = Counter()
    a = AdapterAnalysis(role=role, file_name=file_name, sha256=sha256, pinned=pinned, key_format="unknown")
    for key, tensor in tensors.items():
        prefix, name = strip_prefix(key)
        m = _LORA_RE.match(name)
        if m:
            prefixes[prefix] += 1
            side = m.group("side")
            pair = a.pairs.setdefault(m.group("target"), {})
            if side in pair:
                raise AdapterError(f"{role}: duplicate lora_{side} for {m.group('target')}")
            pair[side] = tensor
            a.dtypes.append(str(tensor.dtype).replace("torch.", ""))
            continue
        m = _ALPHA_RE.match(name)
        if m and getattr(tensor, "numel", lambda: 0)() == 1:
            a.alphas[m.group("target")] = float(tensor.item())
            continue
        a.unsupported_keys.append(key)
    if a.unsupported_keys:
        sample = ", ".join(sorted(a.unsupported_keys)[:5])
        raise AdapterError(f"{role}: {len(a.unsupported_keys)} unsupported tensor(s) (not plain LoRA A/B): {sample}")
    if not a.pairs:
        raise AdapterError(f"{role}: no LoRA tensors found in {file_name}")
    if len(prefixes) != 1:
        raise AdapterError(f"{role}: mixed key prefixes {dict(prefixes)}")
    prefix = next(iter(prefixes))
    a.key_format = {"": "native", "diffusion_model.": "comfy"}.get(prefix, "peft")

    ranks = set()
    for target, pair in a.pairs.items():
        if set(pair) != {"A", "B"}:
            raise AdapterError(f"{role}: incomplete LoRA pair for {target}")
        lo, up = pair["A"], pair["B"]
        if lo.ndim != 2 or up.ndim != 2 or lo.shape[0] == 0 or lo.shape[0] != up.shape[1]:
            raise AdapterError(f"{role}: invalid LoRA dimensions for {target}: A{tuple(lo.shape)} B{tuple(up.shape)}")
        if not lo.is_floating_point() or not up.is_floating_point():
            raise AdapterError(f"{role}: non floating-point LoRA tensor for {target}")
        ranks.add(int(lo.shape[0]))
    if len(ranks) != 1:
        raise AdapterError(f"{role}: mixed ranks {sorted(ranks)} are not supported")
    a.rank = ranks.pop()
    if a.alphas and set(a.alphas) != set(a.pairs):
        raise AdapterError(f"{role}: per-module alpha present for only some targets")

    candidates = []
    if a.alphas:
        values = set(a.alphas.values())
        candidates.append(("tensor", values.pop() if len(values) == 1 else None))
    meta_alpha = _parse_float((metadata or {}).get("alpha"))
    if meta_alpha is not None:
        meta_rank = _parse_float((metadata or {}).get("rank"))
        if meta_rank is not None and int(meta_rank) != a.rank:
            raise AdapterError(f"{role}: safetensors metadata rank {meta_rank} != tensor rank {a.rank}")
        candidates.append(("safetensors_metadata", meta_alpha))
    if peft_config:
        if peft_config.get("use_dora") or peft_config.get("use_rslora"):
            raise AdapterError(f"{role}: DoRA / rsLoRA adapters are not supported")
        if peft_config.get("rank_pattern") or peft_config.get("alpha_pattern"):
            raise AdapterError(f"{role}: per-module rank/alpha patterns are not supported")
        cfg_r = peft_config.get("r")
        if cfg_r is not None and int(cfg_r) != a.rank:
            raise AdapterError(f"{role}: adapter_config.json r={cfg_r} != tensor rank {a.rank}")
        cfg_alpha = _parse_float(peft_config.get("lora_alpha"))
        if cfg_alpha is not None:
            candidates.append(("adapter_config.json", cfg_alpha))
    if pinned is not None:
        if pinned.rank != a.rank:
            raise AdapterError(f"{role}: pinned release rank {pinned.rank} != tensor rank {a.rank}")
        candidates.append(("pinned_release", pinned.alpha))
    if not candidates:
        raise AdapterError(
            f"{role}: cannot determine LoRA alpha for {file_name} (no alpha tensors, no safetensors metadata, "
            "no adapter_config.json, and the file hash does not match a pinned release)"
        )
    values = {v for _, v in candidates}
    if None in values or len(values) != 1:
        raise AdapterError(f"{role}: conflicting alpha sources {candidates}")
    a.alpha = values.pop()
    a.alpha_source = "+".join(src for src, _ in candidates)
    if a.alphas:
        a.alphas = {}  # identical per-target alphas folded into a.alpha
    return a


def validate_against_base(
    analysis: AdapterAnalysis,
    base_shapes: dict[str, tuple[int, ...]],
    base_dtypes: dict[str, torch.dtype],
    profile: BackboneProfile | None,
    *,
    prefix: str = "diffusion_model.",
    check_finite: bool = True,
) -> dict:
    """Check every LoRA target against the base model. Raises AdapterError."""
    missing, shape_bad, dtype_bad = [], [], []
    for target, pair in analysis.pairs.items():
        key = prefix + target + ".weight"
        if key not in base_shapes:
            missing.append(target)
            continue
        want = (int(pair["B"].shape[0]), int(pair["A"].shape[1]))
        if tuple(base_shapes[key]) != want:
            shape_bad.append(f"{target}: base {tuple(base_shapes[key])} vs adapter {want}")
        dt = base_dtypes.get(key)
        if dt not in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
            dtype_bad.append(f"{target}: {dt}")
    if missing:
        raise AdapterError(f"{analysis.role}: {len(missing)} LoRA target(s) absent from the base model, e.g. {missing[:3]}")
    if shape_bad:
        raise AdapterError(f"{analysis.role}: {len(shape_bad)} shape mismatch(es), e.g. {shape_bad[:3]} (different backbone?)")
    if dtype_bad:
        raise AdapterError(
            f"{analysis.role}: base weights are not plain float tensors ({dtype_bad[:2]}); quantized bases are not supported by this release"
        )
    expected = set(profile.expected_targets()) if profile else set(analysis.pairs)
    got = set(analysis.pairs)
    missing_expected = sorted(expected - got)
    unexpected = sorted(got - expected)
    if missing_expected or unexpected:
        raise AdapterError(
            f"{analysis.role}: adapter targets differ from the {profile.display_name if profile else 'expected'} release layout "
            f"(missing {len(missing_expected)}, unexpected {len(unexpected)}; e.g. {(missing_expected or unexpected)[:3]})"
        )
    nonfinite = []
    if check_finite:
        for target, pair in analysis.pairs.items():
            if not (torch.isfinite(pair["A"]).all() and torch.isfinite(pair["B"]).all()):
                nonfinite.append(target)
        if nonfinite:
            raise AdapterError(f"{analysis.role}: non-finite LoRA values in {nonfinite[:3]}")
    return {
        "targets_expected": len(expected),
        "targets_in_adapter": len(got),
        "targets_matched": len(got & expected),
        "unused_adapter_keys": 0,
        "shape_mismatches": 0,
        "nonfinite_checked": bool(check_finite),
    }


def comfy_lora_dict(analysis: AdapterAnalysis) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Convert to ComfyUI's generic ``lora_A/lora_B/alpha`` key layout.

    Returns ``(lora_sd, to_load)`` for ``comfy.lora.load_lora``. The explicit
    ``.alpha`` tensor makes ComfyUI apply ``alpha / rank`` (it defaults to 1.0).
    """
    sd, to_load = {}, {}
    for target, pair in analysis.pairs.items():
        sd[f"{target}.lora_A.weight"] = pair["A"]
        sd[f"{target}.lora_B.weight"] = pair["B"]
        sd[f"{target}.alpha"] = torch.tensor(float(analysis.alpha))
        to_load[target] = f"diffusion_model.{target}.weight"
    return sd, to_load


@torch.no_grad()
def reference_merge(weight: torch.Tensor, updates: list[tuple[torch.Tensor, torch.Tensor, float]]) -> torch.Tensor:
    """Upstream ``merge_lora.merge`` arithmetic for one weight.

    ``updates`` is a list of ``(A, B, scale)`` with ``scale = weight * alpha / rank``.
    """
    value = weight.float().clone()
    for lo, up, scale in updates:
        value.addmm_(up.float(), lo.float(), beta=1, alpha=scale)
    return value.to(weight.dtype)
