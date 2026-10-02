"""Benchmark-only probe for ComfyUI (NOT part of the node package).

Load it as a separate custom node in a *test* ComfyUI only, and only with
``LONGLIVE_PLUG_BENCH=1``. It wraps two ComfyUI internals to count work:

- ``comfy.samplers.calc_cond_batch``: one call per sampler model evaluation;
  counts conditional and unconditional passes.
- ``comfy.model_base.BaseModel.apply_model``: one call per actual batched
  diffusion-model forward, with its batch size.

and exposes read-only counters plus CUDA allocator peaks at
``GET /longlive-plug-bench/stats``; ``POST /longlive-plug-bench/reset`` zeroes
them. It never reads files, environment secrets or prompts.
"""

import os
import threading

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if os.environ.get("LONGLIVE_PLUG_BENCH") == "1":
    import comfy.model_base
    import comfy.samplers
    import torch
    from aiohttp import web
    from server import PromptServer

    _lock = threading.Lock()
    _stats = {}

    def _reset():
        with _lock:
            _stats.clear()
            _stats.update(sampler_evaluations=0, cond_passes=0, uncond_passes=0, forward_calls=0, forward_rows=0, timesteps=[])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    _reset()
    _orig_calc = comfy.samplers.calc_cond_batch
    _orig_apply = comfy.model_base.BaseModel.apply_model

    def calc_cond_batch(model, conds, x_in, timestep, model_options, *args, **kwargs):
        with _lock:
            _stats["sampler_evaluations"] += 1
            _stats["cond_passes"] += int(len(conds) > 0 and conds[0] is not None)
            _stats["uncond_passes"] += sum(1 for c in conds[1:] if c is not None)
            _stats["timesteps"].append(round(float(timestep.flatten()[0]) * 1000.0, 4))
        return _orig_calc(model, conds, x_in, timestep, model_options, *args, **kwargs)

    def apply_model(self, x, t, *args, **kwargs):
        with _lock:
            _stats["forward_calls"] += 1
            _stats["forward_rows"] += int(x.shape[0])
        return _orig_apply(self, x, t, *args, **kwargs)

    comfy.samplers.calc_cond_batch = calc_cond_batch
    comfy.model_base.BaseModel.apply_model = apply_model

    routes = PromptServer.instance.routes

    @routes.get("/longlive-plug-bench/stats")
    async def _get_stats(request):
        with _lock:
            out = {k: (list(v) if isinstance(v, list) else v) for k, v in _stats.items()}
        if torch.cuda.is_available():
            out["cuda"] = {
                "max_allocated_bytes": torch.cuda.max_memory_allocated(),
                "max_reserved_bytes": torch.cuda.max_memory_reserved(),
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "device": torch.cuda.get_device_name(0),
            }
        out["torch"] = torch.__version__
        out["torch_cuda"] = torch.version.cuda
        return web.json_response(out)

    @routes.post("/longlive-plug-bench/reset")
    async def _post_reset(request):
        _reset()
        return web.json_response({"ok": True})
