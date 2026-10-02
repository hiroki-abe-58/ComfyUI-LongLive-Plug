"""A tiny Wan2.1-T2V-shaped state dict for CPU tests (same key layout as the real model)."""

from __future__ import annotations

import torch

from longlive_plug.profiles import BackboneProfile, PinnedAdapter

DIM, FFN, LAYERS = 256, 512, 2


def tiny_profile(adapters: tuple[PinnedAdapter, ...] = ()) -> BackboneProfile:
    return BackboneProfile(
        profile_id="tiny-wan-test",
        display_name="Tiny Wan (test)",
        num_layers=LAYERS,
        dim=DIM,
        ffn_dim=FFN,
        in_dim=16,
        text_dim=4096,
        out_dim=16,
        patch_size=(1, 2, 2),
        adapters=adapters,
        recommended_weights={"few_step": 1.0, "cfg": 0.5},
    )


def tiny_state_dict(seed: int = 0, dtype=torch.float32, extra: bool = False) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def w(*shape, scale=0.02):
        return (torch.randn(*shape, generator=g) * scale).to(dtype)

    d, f = DIM, FFN
    sd = {
        "patch_embedding.weight": w(d, 16, 1, 2, 2),
        "patch_embedding.bias": w(d),
        "text_embedding.0.weight": w(d, 4096),
        "text_embedding.0.bias": w(d),
        "text_embedding.2.weight": w(d, d),
        "text_embedding.2.bias": w(d),
        "time_embedding.0.weight": w(d, 256),
        "time_embedding.0.bias": w(d),
        "time_embedding.2.weight": w(d, d),
        "time_embedding.2.bias": w(d),
        "time_projection.1.weight": w(6 * d, d),
        "time_projection.1.bias": w(6 * d),
        "head.head.weight": w(64, d),
        "head.head.bias": w(64),
        "head.modulation": w(1, 2, d),
    }
    for i in range(LAYERS):
        p = f"blocks.{i}."
        for attn in ("self_attn", "cross_attn"):
            for proj in ("q", "k", "v", "o"):
                sd[f"{p}{attn}.{proj}.weight"] = w(d, d)
                sd[f"{p}{attn}.{proj}.bias"] = w(d)
            sd[f"{p}{attn}.norm_q.weight"] = torch.ones(d, dtype=dtype)
            sd[f"{p}{attn}.norm_k.weight"] = torch.ones(d, dtype=dtype)
        sd[f"{p}ffn.0.weight"] = w(f, d)
        sd[f"{p}ffn.0.bias"] = w(f)
        sd[f"{p}ffn.2.weight"] = w(d, f)
        sd[f"{p}ffn.2.bias"] = w(d)
        sd[f"{p}modulation"] = w(1, 6, d)
        sd[f"{p}norm3.weight"] = torch.ones(d, dtype=dtype)
        sd[f"{p}norm3.bias"] = torch.zeros(d, dtype=dtype)
    if extra:
        # Image cross-attention branch as in Wan I2V: a "derivative" backbone.
        sd["img_emb.proj.0.weight"] = torch.ones(1280, dtype=dtype)
        sd["img_emb.proj.0.bias"] = torch.zeros(1280, dtype=dtype)
        sd["img_emb.proj.1.weight"] = w(1280, 1280)
        sd["img_emb.proj.1.bias"] = w(1280)
        sd["img_emb.proj.3.weight"] = w(d, 1280)
        sd["img_emb.proj.3.bias"] = w(d)
        sd["img_emb.proj.4.weight"] = torch.ones(d, dtype=dtype)
        sd["img_emb.proj.4.bias"] = torch.zeros(d, dtype=dtype)
        for i in range(LAYERS):
            sd[f"blocks.{i}.cross_attn.k_img.weight"] = w(d, d)
            sd[f"blocks.{i}.cross_attn.k_img.bias"] = w(d)
            sd[f"blocks.{i}.cross_attn.v_img.weight"] = w(d, d)
            sd[f"blocks.{i}.cross_attn.v_img.bias"] = w(d)
            sd[f"blocks.{i}.cross_attn.norm_k_img.weight"] = torch.ones(d, dtype=dtype)
    return sd


def tiny_lora(
    profile: BackboneProfile, rank: int = 4, seed: int = 1, prefix: str = "", dtype=torch.float32, scale: float = 0.05
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for target in profile.expected_targets():
        out_f, in_f = profile.target_shape(target)
        sd[f"{prefix}{target}.lora_A.weight"] = (torch.randn(rank, in_f, generator=g) * scale).to(dtype)
        sd[f"{prefix}{target}.lora_B.weight"] = (torch.randn(out_f, rank, generator=g) * scale).to(dtype)
    return sd
