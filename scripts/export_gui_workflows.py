"""Convert ``workflows/api/*.json`` into GUI workflows with the real ComfyUI frontend.

Maintainer tool (needs ``pip install playwright`` + ``playwright install chromium``
and a running ComfyUI with this package installed). For each API prompt it

1. loads it with the frontend's ``app.loadApiJson``,
2. checks that every node type is registered in the frontend,
3. saves ``app.graph.serialize()`` to ``workflows/<name>.json``,
4. converts the graph back with ``app.graphToPrompt()`` and checks that every
   node's inputs equal the original API prompt (round trip),
5. sets seed widgets' control_after_generate to "fixed" so a GUI run reuses
   the published seed,
6. optionally saves a screenshot of the graph.

Nothing is queued.

    python scripts/export_gui_workflows.py --server http://127.0.0.1:8188 [--screenshots docs/img]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]

LOAD_AND_CHECK = """
async ([api, name]) => {
  const app = window.app ?? window.comfyAPI?.app?.app;
  await app.loadApiJson(api, name);
  await new Promise(r => setTimeout(r, 1500));
  const types = window.LiteGraph?.registered_node_types ?? {};
  const nodes = app.graph._nodes ?? app.graph.nodes;
  const missing = nodes.filter(n => !(n.type in types)).map(n => n.type);
  // Keep seeds reproducible: the frontend defaults seed controls to "randomize".
  for (const n of nodes) for (const w of (n.widgets ?? [])) if (w.name === "control_after_generate") w.value = "fixed";
  const workflow = app.graph.serialize();
  const back = await app.graphToPrompt();
  return {workflow, prompt: back.output, missing};
}
"""


def normalise(prompt: dict) -> dict:
    return {nid: {"class_type": n["class_type"], "inputs": n["inputs"]} for nid, n in prompt.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default="http://127.0.0.1:8188")
    p.add_argument("--screenshots", type=Path)
    args = p.parse_args()
    if urlparse(args.server).hostname not in ("127.0.0.1", "localhost", "::1"):
        sys.exit("refusing a non-loopback server")

    failures = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1800, "height": 1000}, locale="en-US")
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(args.server, wait_until="networkidle")
        page.wait_for_function("() => (window.app ?? window.comfyAPI?.app?.app)?.graph !== undefined", timeout=120_000)
        for api_path in sorted((ROOT / "workflows" / "api").glob("*.json")):
            api = json.loads(api_path.read_text(encoding="utf-8"))
            res = page.evaluate(LOAD_AND_CHECK, [api, api_path.stem])
            if res["missing"]:
                failures.append(f"{api_path.name}: frontend lacks node types {res['missing']}")
            original, back = normalise(api), normalise(res["prompt"])
            if original != back:
                diff = sorted(k for k in set(original) | set(back) if original.get(k) != back.get(k))
                failures.append(
                    f"{api_path.name}: GUI round trip differs at nodes {diff}: {[(original.get(k), back.get(k)) for k in diff][:2]}"
                )
            out = ROOT / "workflows" / f"{api_path.stem}.json"
            out.write_text(json.dumps(res["workflow"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(
                f"{api_path.name} -> {out.relative_to(ROOT)} (nodes={len(res['workflow']['nodes'])}, round_trip={'ok' if original == back else 'DIFF'})"
            )
            if args.screenshots:
                args.screenshots.mkdir(parents=True, exist_ok=True)
                page.evaluate(
                    "() => { const a = window.app ?? window.comfyAPI.app.app; a.canvas.ds.scale = 0.6; a.canvas.ds.offset = [40, 40]; a.graph.setDirtyCanvas(true, true); }"
                )
                page.wait_for_timeout(800)
                page.screenshot(path=str(args.screenshots / f"{api_path.stem}.png"))
        browser.close()
    if errors:
        print("page errors:", *errors[:5], sep="\n  ")
    if failures:
        sys.exit("\n".join(failures))


if __name__ == "__main__":
    main()
