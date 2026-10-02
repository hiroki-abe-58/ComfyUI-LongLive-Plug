"""Noise schedules extracted from the upstream Wan / LongLive-Plug sources.

Every function here reproduces a specific upstream computation, including its
dtype handling, so that the values handed to ComfyUI are the values the
upstream code uses. The upstream sources are cited next to each function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

NUM_TRAIN_TIMESTEPS = 1000


@dataclass(frozen=True)
class FlowUniPCSchedule:
    """Sigmas and model timesteps of Wan's ``FlowUniPCMultistepScheduler``.

    ``sigmas`` has ``steps + 1`` float32 entries (the last one is 0.0, i.e.
    ``final_sigmas_type="zero"``). ``timesteps`` has ``steps`` int64 entries:
    upstream feeds the model ``int(sigma * 1000)`` while the solver itself
    keeps using the float32 sigma.
    """

    steps: int
    shift: float
    sigmas: tuple[float, ...]
    timesteps: tuple[int, ...]

    def sigmas_tensor(self) -> torch.Tensor:
        return torch.tensor(self.sigmas, dtype=torch.float32)


def flow_unipc_schedule(steps: int, shift: float) -> FlowUniPCSchedule:
    """Reproduce ``FlowUniPCMultistepScheduler(shift=1).set_timesteps(steps, shift=shift)``.

    Source: Wan2.1 ``wan/utils/fm_solvers_unipc.py`` (vendored unchanged in
    LongLive-Plug as ``wan_5b/utils/fm_solvers_unipc.py``), as called by
    Wan2.1 ``wan/text2video.py`` and LongLive-Plug ``inference.py``.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not np.isfinite(shift) or shift <= 0:
        raise ValueError("shift must be a positive finite number")
    # __init__: training table built in float64 numpy, stored as float32 torch.
    alphas = np.linspace(1, 1 / NUM_TRAIN_TIMESTEPS, NUM_TRAIN_TIMESTEPS)[::-1].copy()
    train_sigmas = torch.from_numpy(1.0 - alphas).to(dtype=torch.float32)
    # shift=1 at construction is the identity map: 1 * s / (1 + 0 * s).
    sigma_max = train_sigmas[0].item()
    sigma_min = train_sigmas[-1].item()
    # set_timesteps: float64 numpy from here on.
    sigmas = np.linspace(sigma_max, sigma_min, steps + 1).copy()[:-1]
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    timesteps = sigmas * NUM_TRAIN_TIMESTEPS
    sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
    int_timesteps = torch.from_numpy(timesteps).to(dtype=torch.int64)
    return FlowUniPCSchedule(
        steps=steps,
        shift=float(shift),
        sigmas=tuple(float(s) for s in sigmas),
        timesteps=tuple(int(t) for t in int_timesteps.tolist()),
    )


def warped_step_list_sigmas(denoising_step_list: list[int] | tuple[int, ...], shift: float) -> tuple[float, ...]:
    """Sigmas of a warped ``denoising_step_list`` (Self-Forcing / LightX2V convention).

    Sources:
    - LightX2V ``WanStepDistillScheduler.set_denoising_timesteps``: float32
      ``linspace(1, 0, 1001)[:-1]``, shifted in float32, indexed at
      ``1000 - t``.
    - Self-Forcing-Plus ``warp_denoising_step`` uses the same table
      (``FlowMatchScheduler(shift, sigma_min=0, extra_one_step=True)``).

    Returns the per-step sigmas followed by a terminal 0.0.
    """
    if not denoising_step_list:
        raise ValueError("denoising_step_list must not be empty")
    table = torch.linspace(1.0, 0.0, NUM_TRAIN_TIMESTEPS + 1)[:-1]
    table = shift * table / (1 + (shift - 1) * table)
    out = []
    for t in denoising_step_list:
        t = int(t)
        if not 0 < t <= NUM_TRAIN_TIMESTEPS:
            raise ValueError(f"denoising step {t} outside (0, {NUM_TRAIN_TIMESTEPS}]")
        out.append(float(table[NUM_TRAIN_TIMESTEPS - t].item()))
    if any(b >= a for a, b in zip(out, out[1:])):
        raise ValueError("denoising_step_list must be strictly decreasing")
    return tuple(out) + (0.0,)
