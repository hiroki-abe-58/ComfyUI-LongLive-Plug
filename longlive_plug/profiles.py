"""Backbone / adapter profiles with pinned upstream revisions.

A profile states what was actually verified. Adding a backbone here is a
claim that its adapters, targets and recipes were checked against upstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ATTN_PROJ = ("q", "k", "v", "o")


@dataclass(frozen=True)
class PinnedAdapter:
    role: str  # "few_step" | "cfg"
    repo: str
    revision: str
    filename: str
    sha256: str
    rank: int
    alpha: float
    dtype: str
    key_format: str  # "native" (no PEFT prefix) | "peft" (base_model.model.)


@dataclass(frozen=True)
class BackboneProfile:
    profile_id: str
    display_name: str
    num_layers: int
    dim: int
    ffn_dim: int
    in_dim: int
    text_dim: int
    out_dim: int
    patch_size: tuple[int, int, int]
    adapters: tuple[PinnedAdapter, ...]
    recommended_weights: dict = field(default_factory=dict)
    base_reference: dict = field(default_factory=dict)

    def expected_targets(self) -> list[str]:
        """Every ``nn.Linear`` inside a ``WanAttentionBlock`` (upstream ``configure_lora_for_model``)."""
        out = []
        for i in range(self.num_layers):
            for attn in ("self_attn", "cross_attn"):
                out.extend(f"blocks.{i}.{attn}.{p}" for p in ATTN_PROJ)
            out.extend((f"blocks.{i}.ffn.0", f"blocks.{i}.ffn.2"))
        return out

    def target_shape(self, target: str) -> tuple[int, int]:
        if target.endswith(".ffn.0"):
            return (self.ffn_dim, self.dim)
        if target.endswith(".ffn.2"):
            return (self.dim, self.ffn_dim)
        return (self.dim, self.dim)

    def pinned(self, sha256: str | None) -> PinnedAdapter | None:
        if not sha256:
            return None
        return next((a for a in self.adapters if a.sha256 == sha256), None)


WAN21_T2V_14B = BackboneProfile(
    profile_id="wan2.1-t2v-14b",
    display_name="Wan2.1-T2V-14B",
    num_layers=40,
    dim=5120,
    ffn_dim=13824,
    in_dim=16,
    text_dim=4096,
    out_dim=16,
    patch_size=(1, 2, 2),
    adapters=(
        PinnedAdapter(
            role="few_step",
            repo="Efficient-Large-Model/LongLive-Plug-Wan2.1-T2V-14B-few-step",
            revision="f125af0533b94e2cee0163a7a42bb736d59a1d37",
            filename="generator_lora_lightx2v.safetensors",
            sha256="743fc9a44e118a09f932ae5a0420c88b46bb6afdf605230f115e93eb710884ba",
            rank=128,
            alpha=128.0,
            dtype="bfloat16",
            key_format="native",
        ),
        PinnedAdapter(
            role="cfg",
            repo="Efficient-Large-Model/LongLive-Plug-Wan2.1-T2V-14B-cfg",
            revision="32b8aa3d1db3d178a9c82f3731cc917a34fb7693",
            filename="adapter_model.safetensors",
            sha256="1a311f6030a74e739d9705347079a131bd5a5579bd693c993026d63e36d4bea0",
            rank=128,
            alpha=128.0,
            dtype="float32",
            key_format="peft",
        ),
    ),
    recommended_weights={"few_step": 1.0, "cfg": 0.5},
    base_reference={
        "upstream": "Wan-AI/Wan2.1-T2V-14B@a064a6c71f5be440641209c07bf2a5ce7a2ff5e4",
        "comfy_repackaged": "Comfy-Org/Wan_2.1_ComfyUI_repackaged@123acf1cc74bccbb9bfff8ac1ee72edc08c2341d",
        "tested_file": "split_files/diffusion_models/wan2.1_t2v_14B_bf16.safetensors",
        "tested_file_sha256": "193535c6450045f718df5f011de6d94d49bd9b13f37ca0412500f050dbbb01a8",
    },
)

PROFILES = {p.profile_id: p for p in (WAN21_T2V_14B,)}


def fingerprint(profile: BackboneProfile, shapes: dict[str, tuple[int, ...]], prefix: str = "diffusion_model.") -> dict:
    """Compare a model state-dict shape map against a profile.

    Returns ``{"match": "exact"|"derivative"|"mismatch", "reasons": [...]}``.
    "derivative" means every adapter target exists with the right shape but the
    model carries extra branches (e.g. I2V image cross-attention, VACE); such
    models are in the upstream transfer scope but are not tested here.
    """
    reasons = []

    def shape(name):
        return shapes.get(prefix + name)

    p = profile.patch_size
    want = {
        "patch_embedding.weight": (profile.dim, profile.in_dim, *p),
        "text_embedding.0.weight": (profile.dim, profile.text_dim),
        "head.head.weight": (profile.out_dim * p[0] * p[1] * p[2], profile.dim),
    }
    hard_fail = False
    for name, expected in want.items():
        got = shape(name)
        if got is None:
            reasons.append(f"missing {name}")
            hard_fail = True
        elif tuple(got) != expected:
            reasons.append(f"{name} shape {tuple(got)} != {expected}")
            if name != "patch_embedding.weight" and name != "head.head.weight":
                hard_fail = True
    block_ids = set()
    for k in shapes:
        if k.startswith(prefix + "blocks."):
            idx = k[len(prefix) + len("blocks.") :].split(".", 1)[0]
            if idx.isdigit():
                block_ids.add(int(idx))
    if len(block_ids) != profile.num_layers:
        reasons.append(f"{len(block_ids)} transformer blocks != {profile.num_layers}")
        hard_fail = True
    for target in profile.expected_targets():
        got = shape(target + ".weight")
        if got is None or tuple(got) != profile.target_shape(target):
            reasons.append(f"adapter target {target} missing or wrong shape ({got})")
            hard_fail = True
            break
    if hard_fail:
        return {"match": "mismatch", "reasons": reasons}
    extra = sorted(
        {k[len(prefix) :].split(".", 1)[0] for k in shapes if k.startswith(prefix)}
        - {"patch_embedding", "text_embedding", "time_embedding", "time_projection", "blocks", "head"}
    )
    has_img_xattn = any(".cross_attn.k_img." in k for k in shapes)
    if extra or has_img_xattn or reasons:
        if extra:
            reasons.append("extra modules: " + ", ".join(extra))
        if has_img_xattn:
            reasons.append("image cross-attention (I2V/FLF2V-style) present")
        return {"match": "derivative", "reasons": reasons}
    return {"match": "exact", "reasons": []}
