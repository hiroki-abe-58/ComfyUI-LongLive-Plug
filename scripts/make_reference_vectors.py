"""Generate golden vectors from the *upstream* FlowUniPCMultistepScheduler.

This is a maintainer tool. It needs an upstream checkout and ``diffusers``;
CI only consumes the JSON it writes (``tests/data/flow_unipc_reference.json``).

    python scripts/make_reference_vectors.py --longlive-root <path to NVlabs/LongLive checkout>

The synthetic "model" is a fixed, timestep-dependent nonlinear map, so the
test exercises predictor, corrector, lower-order warmup and the final step.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import torch

CASES = [
    {"name": "longlive_plug_4step", "steps": 4, "shift": 5.0},
    {"name": "wan21_base_50step", "steps": 50, "shift": 5.0},
    {"name": "odd_7step_shift3", "steps": 7, "shift": 3.0},
    {"name": "single_step", "steps": 1, "shift": 5.0},
]
SHAPE = [1, 4, 3, 4, 6]  # [B, C, T, H, W]


def synthetic_flow(x: torch.Tensor, t: float) -> torch.Tensor:
    s = t / 1000.0
    return torch.tanh(1.3 * x) * (0.4 + 0.6 * s) + 0.25 * torch.sin(3.0 * x + s) - 0.1 * s


def load_upstream(longlive_root: Path):
    path = longlive_root / "LongLive-Plug" / "wan_5b" / "utils" / "fm_solvers_unipc.py"
    spec = importlib.util.spec_from_file_location("upstream_fm_solvers_unipc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FlowUniPCMultistepScheduler, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--longlive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "tests" / "data" / "flow_unipc_reference.json")
    args = parser.parse_args()

    scheduler_cls, path = load_upstream(args.longlive_root)
    commit = subprocess.run(
        ["git", "-C", str(args.longlive_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    noise = torch.randn(SHAPE, generator=torch.Generator().manual_seed(20261002), dtype=torch.float32)
    cases = []
    for case in CASES:
        sched = scheduler_cls(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        sched.set_timesteps(case["steps"], device="cpu", shift=case["shift"])
        latent = noise.clone()
        for timestep in sched.timesteps:
            flow = synthetic_flow(latent, float(timestep))
            latent = sched.step(flow, timestep, latent, return_dict=False)[0]
        cases.append(
            {
                **case,
                "sigmas": [float(s) for s in sched.sigmas.tolist()],
                "timesteps": [int(t) for t in sched.timesteps.tolist()],
                "final": [float(v) for v in latent.flatten().tolist()],
            }
        )
    out = {
        "generator": "scripts/make_reference_vectors.py",
        "upstream": {
            "repo": "https://github.com/NVlabs/LongLive",
            "commit": commit,
            "file": "LongLive-Plug/wan_5b/utils/fm_solvers_unipc.py",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "torch": torch.__version__,
        "noise_seed": 20261002,
        "shape": SHAPE,
        "noise": [float(v) for v in noise.flatten().tolist()],
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {args.output.name}")


if __name__ == "__main__":
    main()
