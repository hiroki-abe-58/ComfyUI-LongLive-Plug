import json
from pathlib import Path

import pytest
import torch

from longlive_plug.flow_unipc import model_evaluations, sample_flow_unipc
from longlive_plug.schedules import flow_unipc_schedule, warped_step_list_sigmas

REF = json.loads((Path(__file__).parent / "data" / "flow_unipc_reference.json").read_text(encoding="utf-8"))


def synthetic_flow(x, t):
    # Must match scripts/make_reference_vectors.py.
    s = t / 1000.0
    return torch.tanh(1.3 * x) * (0.4 + 0.6 * s) + 0.25 * torch.sin(3.0 * x + s) - 0.1 * s


@pytest.mark.parametrize("case", REF["cases"], ids=[c["name"] for c in REF["cases"]])
def test_schedule_matches_upstream(case):
    sched = flow_unipc_schedule(case["steps"], case["shift"])
    assert list(sched.timesteps) == case["timesteps"]
    # float32 values serialised from the same float32 tensor: exact.
    assert list(sched.sigmas) == case["sigmas"]


@pytest.mark.parametrize("case", REF["cases"], ids=[c["name"] for c in REF["cases"]])
def test_sampler_matches_upstream(case):
    sched = flow_unipc_schedule(case["steps"], case["shift"])
    noise = torch.tensor(REF["noise"], dtype=torch.float32).reshape(REF["shape"])
    out = sample_flow_unipc(
        lambda x, i: synthetic_flow(x, float(sched.timesteps[i])),
        noise.clone(),
        sched.sigmas,
        sched.timesteps,
    )
    ref = torch.tensor(case["final"], dtype=torch.float32).reshape(REF["shape"])
    # Same float32 CPU arithmetic as upstream; allow only float32 rounding noise.
    torch.testing.assert_close(out, ref, rtol=0, atol=2e-6)


def test_final_step_returns_x0_prediction():
    sched = flow_unipc_schedule(1, 5.0)
    x = torch.randn(1, 2, 1, 2, 2, generator=torch.Generator().manual_seed(0))
    v = torch.randn_like(x)
    out = sample_flow_unipc(lambda x_, i: v, x.clone(), sched.sigmas, sched.timesteps)
    torch.testing.assert_close(out, x - sched.sigmas[0] * v)


def test_callback_reports_every_step_and_x0():
    sched = flow_unipc_schedule(4, 5.0)
    seen = []
    x = torch.zeros(1, 1, 1, 1, 1)
    sample_flow_unipc(
        lambda x_, i: torch.ones_like(x_), x, sched.sigmas, sched.timesteps, callback=lambda i, x0, xt, s: seen.append((i, s))
    )
    assert [i for i, _ in seen] == [0, 1, 2, 3]
    assert [s for _, s in seen] == pytest.approx(list(sched.sigmas[:-1]))


@pytest.mark.parametrize(
    "sigmas,timesteps,msg",
    [
        ([1.0, 0.5], [999, 500], "expected 3 sigmas"),
        ([1.0, 0.5, 0.1], [999, 500], "terminal sigma of 0"),
        ([1.0, 0.5, 0.0], [500, 500], "duplicate"),
    ],
)
def test_sampler_rejects_inconsistent_schedules(sigmas, timesteps, msg):
    with pytest.raises(ValueError, match=msg):
        sample_flow_unipc(lambda x, i: x, torch.zeros(1, 1, 1, 1, 1), sigmas, timesteps)


def test_schedule_argument_validation():
    with pytest.raises(ValueError):
        flow_unipc_schedule(0, 5.0)
    with pytest.raises(ValueError):
        flow_unipc_schedule(4, 0.0)


def test_warped_step_list_matches_lightx2v_formula():
    # LightX2V/Self-Forcing: sigma = shift*s/(1+(shift-1)*s), s = t/1000 (float32 table).
    sig = warped_step_list_sigmas([1000, 750, 500, 250], 5.0)
    expected = [5 * s / (1 + 4 * s) for s in (1.0, 0.75, 0.5, 0.25)] + [0.0]
    assert sig == pytest.approx(expected, rel=0, abs=1e-6)
    assert sig[0] == 1.0 and sig[-1] == 0.0


@pytest.mark.parametrize("bad", [[], [1000, 1000], [0], [1001], [250, 500]])
def test_warped_step_list_validation(bad):
    with pytest.raises(ValueError):
        warped_step_list_sigmas(bad, 5.0)


def test_model_evaluation_count():
    assert model_evaluations(4, 1.0) == 4
    assert model_evaluations(50, 5.0) == 100
