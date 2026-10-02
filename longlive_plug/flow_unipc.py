# SPDX-License-Identifier: Apache-2.0
#
# The predictor/corrector arithmetic below is adapted from
# ``FlowUniPCMultistepScheduler`` in Wan2.1 ``wan/utils/fm_solvers_unipc.py``
# (Copyright 2024-2025 The Alibaba Wan Team Authors, Apache-2.0), which is
# itself adapted from diffusers ``scheduling_unipc_multistep.py``
# (Copyright 2024 The HuggingFace Team, Apache-2.0). The same file is vendored
# by NVlabs/LongLive ``LongLive-Plug/wan_5b/utils/fm_solvers_unipc.py``.
# Only the configuration used by Wan/LongLive-Plug is implemented:
# solver_order=2, solver_type="bh2", predict_x0=True,
# prediction_type="flow_prediction", lower_order_final=True,
# final_sigmas_type="zero", no thresholding, no disabled correctors.
"""Stateless re-implementation of Wan's flow-matching UniPC sampler."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

SOLVER_ORDER = 2


def _alpha_sigma(sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return 1 - sigma, sigma


def _lambda(sigma: torch.Tensor) -> torch.Tensor:
    alpha, sigma = _alpha_sigma(sigma)
    return torch.log(alpha) - torch.log(sigma)


def _bh2_coefficients(h: torch.Tensor, rks: list, order: int):
    """Shared ``R``/``b``/``h_phi_1``/``B_h`` construction (predict_x0, bh2)."""
    rks = torch.tensor(rks + [1.0])
    hh = -h
    h_phi_1 = torch.expm1(hh)
    h_phi_k = h_phi_1 / hh - 1
    factorial_i = 1
    b_h = torch.expm1(hh)
    rows, b = [], []
    for i in range(1, order + 1):
        rows.append(torch.pow(rks, i - 1))
        b.append(h_phi_k * factorial_i / b_h)
        factorial_i *= i + 1
        h_phi_k = h_phi_k / hh - 1 / factorial_i
    return torch.stack(rows), torch.tensor(b), h_phi_1, b_h


def _predictor(x, model_outputs, sigmas, step_index, order):
    m0 = model_outputs[-1]
    alpha_t, sigma_t = _alpha_sigma(sigmas[step_index + 1])
    alpha_s0, sigma_s0 = _alpha_sigma(sigmas[step_index])
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    h = (torch.log(alpha_t) - torch.log(sigma_t)) - lambda_s0
    rks, d1s = [], []
    for i in range(1, order):
        rk = (_lambda(sigmas[step_index - i]) - lambda_s0) / h
        rks.append(rk)
        d1s.append((model_outputs[-(i + 1)] - m0) / rk)
    _, _, h_phi_1, b_h = _bh2_coefficients(h, rks, order)
    x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
    if d1s:
        # order == 2: upstream uses the fixed rho_p = 0.5 instead of solving R.
        rhos_p = torch.tensor([0.5], dtype=x.dtype, device=x.device)
        pred_res = torch.einsum("k,bkc...->bc...", rhos_p, torch.stack(d1s, dim=1))
    else:
        pred_res = 0
    return (x_t_ - alpha_t * b_h * pred_res).to(x.dtype)


def _corrector(model_t, last_sample, model_outputs, sigmas, step_index, order):
    m0 = model_outputs[-1]
    x = last_sample
    alpha_t, sigma_t = _alpha_sigma(sigmas[step_index])
    alpha_s0, sigma_s0 = _alpha_sigma(sigmas[step_index - 1])
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    h = (torch.log(alpha_t) - torch.log(sigma_t)) - lambda_s0
    rks, d1s = [], []
    for i in range(1, order):
        rk = (_lambda(sigmas[step_index - (i + 1)]) - lambda_s0) / h
        rks.append(rk)
        d1s.append((model_outputs[-(i + 1)] - m0) / rk)
    r, b, h_phi_1, b_h = _bh2_coefficients(h, rks, order)
    if order == 1:
        rhos_c = torch.tensor([0.5], dtype=x.dtype, device=x.device)
    else:
        rhos_c = torch.linalg.solve(r, b).to(device=x.device, dtype=x.dtype)
    x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
    corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], torch.stack(d1s, dim=1)) if d1s else 0
    d1_t = model_t - m0
    return (x_t_ - alpha_t * b_h * (corr_res + rhos_c[-1] * d1_t)).to(x.dtype)


def sample_flow_unipc(
    flow_model: Callable[[torch.Tensor, int], torch.Tensor],
    x: torch.Tensor,
    sigmas: Sequence[float] | torch.Tensor,
    timesteps: Sequence[int] | Sequence[float],
    callback: Callable[[int, torch.Tensor, torch.Tensor, float], None] | None = None,
) -> torch.Tensor:
    """Run the full UniPC (bh2, order 2) loop.

    ``flow_model(x, i)`` must return the flow prediction ``v`` for step ``i``
    evaluated at model timestep ``timesteps[i]``. ``sigmas`` are the solver
    sigmas (``len(timesteps) + 1`` entries ending in 0). Scalar coefficient
    arithmetic is done on float32 CPU tensors, as in the upstream scheduler.
    """
    sig = torch.as_tensor(sigmas, dtype=torch.float32).cpu()
    n = len(timesteps)
    if sig.ndim != 1 or sig.numel() != n + 1:
        raise ValueError(f"expected {n + 1} sigmas for {n} timesteps, got {tuple(sig.shape)}")
    if float(sig[-1]) != 0.0:
        raise ValueError("FlowUniPC expects a terminal sigma of 0 (final_sigmas_type='zero')")
    if len(set(timesteps)) != n:
        # Upstream resolves its start index by timestep value; duplicates would
        # silently shift the schedule there, so refuse them here.
        raise ValueError("duplicate model timesteps are not supported")

    model_outputs: list[torch.Tensor | None] = [None] * SOLVER_ORDER
    lower_order_nums = 0
    last_sample = None
    this_order = 0
    for i in range(n):
        v = flow_model(x, i)
        x0 = x - sig[i] * v  # convert_model_output (predict_x0, flow_prediction)
        if i > 0 and last_sample is not None:
            x = _corrector(x0, last_sample, model_outputs, sig, i, this_order)
        model_outputs = model_outputs[1:] + [x0]
        this_order = min(SOLVER_ORDER, n - i, lower_order_nums + 1)
        last_sample = x
        x = _predictor(x, model_outputs, sig, i, this_order)
        lower_order_nums = min(lower_order_nums + 1, SOLVER_ORDER)
        if callback is not None:
            callback(i, x0, x, float(sig[i]))
    return x


def model_evaluations(steps: int, cfg: float) -> int:
    """Number of diffusion-model passes (cond + uncond) for a CFG run."""
    return steps * (1 if cfg == 1.0 else 2)
