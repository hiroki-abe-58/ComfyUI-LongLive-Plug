"""Queue the public workflows on a running ComfyUI and record what happened.

Example (baseline and LongLive-Plug, same prompt/seed/size):

    python scripts/run_comparison.py --server http://127.0.0.1:8188 \
        --variants longlive_plug_4step baseline_50step --results-dir results

For each run it records per-node wall time from ComfyUI's websocket events,
per-step sampler timestamps, the node reports, optional benchmark-probe
counters (``benchmarks/comfy_probe``), nvidia-smi samples, and validates the
saved MP4 with ffprobe and a full ffmpeg decode. The manifest contains no
absolute paths, host names or environment variables.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode, urlparse

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_workflows import DEMO_PROMPT, DEMO_SEED, RECIPE_BY_VARIANT, build  # noqa: E402

PHASES = {
    "UNETLoader": "load_diffusion_model",
    "CLIPLoader": "load_text_encoder",
    "VAELoader": "load_vae",
    "CLIPTextEncode": "text_encode",
    "LongLivePlugApplyAdapters": "apply_adapters",
    "LongLivePlugRecipe": "recipe",
    "SamplerCustomAdvanced": "sampler",
    "VAEDecode": "vae_decode",
    "CreateVideo": "save",
    "SaveVideo": "save",
}


class GpuSampler(threading.Thread):
    """Polls nvidia-smi for total used memory on GPU 0 (all processes)."""

    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval, self.samples, self._stop = interval, [], threading.Event()
        self.exe = shutil.which("nvidia-smi")

    def run(self):
        while self.exe and not self._stop.is_set():
            try:
                out = subprocess.run(
                    [self.exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                ).stdout.strip()
                self.samples.append(int(out.splitlines()[0]))
            except (subprocess.SubprocessError, ValueError, IndexError):
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_video(path: Path, frames_dir: Path) -> dict:
    ffprobe, ffmpeg = shutil.which("ffprobe"), shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg:
        return {"error": "ffprobe/ffmpeg not found"}
    info = json.loads(
        subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=codec_name,width,height,r_frame_rate,nb_read_frames,pix_fmt:format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    stream = info["streams"][0]
    decode = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"], capture_output=True, text=True)
    n = int(stream.get("nb_read_frames") or 0)
    frames_dir.mkdir(parents=True, exist_ok=True)
    picks = sorted({0, n // 4, n // 2, (3 * n) // 4, max(n - 1, 0)}) if n else []
    extracted = []
    for k in picks:
        out = frames_dir / f"frame_{k:03d}.png"
        r = subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-i", str(path), "-vf", f"select=eq(n\\,{k})", "-frames:v", "1", str(out)], capture_output=True
        )
        if r.returncode == 0 and out.exists():
            extracted.append(out.name)
    num, den = (stream.get("r_frame_rate") or "0/1").split("/")
    return {
        "codec": stream.get("codec_name"),
        "pix_fmt": stream.get("pix_fmt"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": float(num) / float(den or 1),
        "frames_decoded": n,
        "duration_s": float(info.get("format", {}).get("duration") or 0),
        "full_decode_ok": decode.returncode == 0 and not decode.stderr.strip(),
        "extracted_frames": extracted,
    }


async def get_json(session, url):
    async with session.get(url) as r:
        if r.status != 200:
            return None
        return await r.json()


async def run_one(session, server, prompt, label, results_dir: Path, use_probe: bool, cancel_after_steps: int | None = None) -> dict:
    client_id = uuid.uuid4().hex
    ws_url = server.replace("http", "ws", 1) + "/ws?" + urlencode({"clientId": client_id})
    events, step_times = [], []
    if use_probe:
        await session.post(server + "/longlive-plug-bench/reset")
    gpu = GpuSampler()
    gpu.start()
    async with session.ws_connect(ws_url, max_msg_size=0) as ws:
        t0 = time.perf_counter()
        async with session.post(server + "/prompt", json={"prompt": prompt, "client_id": client_id}) as r:
            body = await r.json()
            if r.status != 200:
                gpu.stop()
                return {"label": label, "status": "rejected", "error": body}
        prompt_id = body["prompt_id"]
        status, interrupted_sent = "unknown", False
        while True:
            msg = await ws.receive(timeout=7200)
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            data = json.loads(msg.data)
            d = data.get("data", {})
            if d.get("prompt_id") not in (None, prompt_id):
                continue
            now = time.perf_counter() - t0
            kind = data.get("type")
            if kind == "executing":
                events.append((now, d.get("node")))
                if d.get("node") is None:
                    status = status if status != "unknown" else "success"
                    break
            elif kind == "progress" and d.get("node") == "10":
                step_times.append((now, d.get("value"), d.get("max")))
                if cancel_after_steps and d.get("value") >= cancel_after_steps and not interrupted_sent:
                    await session.post(server + "/interrupt", json={"prompt_id": prompt_id})
                    interrupted_sent = True
            elif kind == "execution_cached":
                events.append((now, ("cached", tuple(d.get("nodes", [])))))
            elif kind == "execution_success":
                status = "success"
                break
            elif kind == "execution_interrupted":
                status = "interrupted"
                break
            elif kind == "execution_error":
                status = "error"
                events.append((now, ("error", d.get("exception_type"), d.get("exception_message", "")[:500])))
                break
        total = time.perf_counter() - t0
    gpu.stop()
    probe = await get_json(session, server + "/longlive-plug-bench/stats") if use_probe else None
    history = (await get_json(session, f"{server}/history/{prompt_id}") or {}).get(prompt_id, {})

    node_time, cached = {}, []
    timeline = [e for e in events if not isinstance(e[1], tuple)]
    for (t_a, node), (t_b, _) in zip(timeline, timeline[1:] + [(total, None)]):
        if node is not None:
            node_time[node] = node_time.get(node, 0.0) + (t_b - t_a)
    for _, ev in events:
        if isinstance(ev, tuple) and ev[0] == "cached":
            cached.extend(ev[1])
    phases = {}
    for node, sec in node_time.items():
        phase = PHASES.get(prompt[node]["class_type"], "other")
        phases[phase] = round(phases.get(phase, 0.0) + sec, 3)
    sampler = {}
    if step_times and "10" in node_time:
        start = next(t for t, n in timeline if n == "10")
        ts = [t for t, _, _ in step_times]
        sampler = {
            "steps_reported": len(ts),
            "until_first_step_done_s": round(ts[0] - start, 3),
            "after_first_step_s": round(ts[-1] - ts[0], 3),
            "per_step_s": [round(b - a, 3) for a, b in zip([start] + ts[:-1], ts)],
        }
    reports = {}
    outputs = history.get("outputs", {})
    for node_id in ("14", "15"):
        texts = (outputs.get(node_id) or {}).get("text")
        if texts:
            try:
                reports[node_id] = json.loads(texts[0])
            except json.JSONDecodeError:
                reports[node_id] = texts[0]
    video = None
    for item in (outputs.get("13") or {}).get("images", []) + (outputs.get("13") or {}).get("video", []):
        if item.get("filename", "").endswith((".mp4", ".webm", ".mkv")):
            video = item
    record = {
        "label": label,
        "status": status,
        "prompt_id": prompt_id,
        "wall_total_s": round(total, 3),
        "phases_s": phases,
        "nodes_cached": sorted(set(cached), key=int),
        "sampler": sampler,
        "probe": probe,
        "nvidia_smi_total_used_mib": {
            "max": max(gpu.samples) if gpu.samples else None,
            "min": min(gpu.samples) if gpu.samples else None,
            "samples": len(gpu.samples),
        },
        "reports": reports,
        "errors": [e[1] for e in events if isinstance(e[1], tuple) and e[1][0] == "error"],
    }
    if video and status == "success":
        query = urlencode({"filename": video["filename"], "subfolder": video.get("subfolder", ""), "type": video.get("type", "output")})
        dest = results_dir / label / video["filename"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        async with session.get(f"{server}/view?{query}") as r:
            dest.write_bytes(await r.read())
        record["video"] = {
            "file": f"{label}/{video['filename']}",
            "sha256": sha256(dest),
            "bytes": dest.stat().st_size,
            **probe_video(dest, dest.parent / "frames"),
        }
    return record


async def main_async(args) -> int:
    parsed = urlparse(args.server)
    if parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        print("refusing a non-loopback server", file=sys.stderr)
        return 2
    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        stats = await get_json(session, args.server + "/system_stats") or {}
        use_probe = (await get_json(session, args.server + "/longlive-plug-bench/stats")) is not None
        env = {
            "comfyui_version": stats.get("system", {}).get("comfyui_version"),
            "python": (stats.get("system", {}).get("python_version") or "").split(" ")[0],
            "pytorch": stats.get("system", {}).get("pytorch_version"),
            "devices": [{"name": d.get("name"), "vram_total": d.get("vram_total")} for d in stats.get("devices", [])],
            "benchmark_probe": use_probe,
        }
        size = {"width": args.width, "height": args.height, "length": args.frames}
        runs = []
        for variant in args.variants:
            for rep in range(args.repeat):
                seed = args.seed + rep * args.seed_stride
                label = f"{variant}{'' if rep == 0 else f'_run{rep + 1}'}"
                prompt = build(variant, prompt=args.prompt, seed=seed, size=size)
                prompt["13"]["inputs"]["filename_prefix"] = f"longlive_plug/{args.tag}_{label}"
                print(f"[{time.strftime('%H:%M:%S')}] queue {label} (recipe={RECIPE_BY_VARIANT[variant]}, seed={seed})", flush=True)
                rec = await run_one(session, args.server, prompt, label, results, use_probe, args.cancel_after_steps)
                rec.update(
                    {"variant": variant, "recipe": RECIPE_BY_VARIANT[variant], "seed": seed, "size": size, "fps": 16.0, "repeat_index": rep}
                )
                runs.append(rec)
                print(json.dumps({k: rec.get(k) for k in ("label", "status", "wall_total_s", "phases_s")}), flush=True)
                (results / f"manifest_{args.tag}.json").write_text(
                    json.dumps({"tag": args.tag, "prompt": args.prompt, "environment": env, "runs": runs}, indent=1, ensure_ascii=False)
                    + "\n",
                    encoding="utf-8",
                )
    return 0 if all(r["status"] in ("success", "interrupted") for r in runs) else 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default="http://127.0.0.1:8188")
    p.add_argument("--variants", nargs="+", default=["longlive_plug_4step", "baseline_50step"], choices=sorted(RECIPE_BY_VARIANT))
    p.add_argument("--prompt", default=DEMO_PROMPT)
    p.add_argument("--seed", type=int, default=DEMO_SEED)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--repeat", type=int, default=1, help="runs per variant in this server session (run 1 = first in order)")
    p.add_argument("--seed-stride", type=int, default=1, help="seed increment between repeats (forces re-sampling with cached loaders)")
    p.add_argument("--cancel-after-steps", type=int, default=None, help="interrupt each run after this many sampler steps (cancel test)")
    p.add_argument("--tag", default="run")
    p.add_argument("--results-dir", default="results")
    sys.exit(asyncio.run(main_async(p.parse_args())))


if __name__ == "__main__":
    main()
