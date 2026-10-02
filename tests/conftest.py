"""Test setup.

Pure-logic tests need only torch. Tests marked ``comfy`` import a real ComfyUI
checkout given by the ``COMFYUI_PATH`` environment variable and run on CPU.
They fail (not skip) when ``COMFYUI_PATH`` is missing, so CI cannot pass by
silently skipping them; deselect them explicitly with ``-m "not comfy"``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

_COMFY_STATE: dict = {}


def _boot_comfy():
    if _COMFY_STATE:
        return _COMFY_STATE
    path = os.environ.get("COMFYUI_PATH")
    if not path or not (Path(path) / "comfy").is_dir():
        pytest.fail("COMFYUI_PATH must point to a ComfyUI checkout for tests marked 'comfy'", pytrace=False)
    if path not in sys.path:
        sys.path.insert(0, path)
    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    comfy.cli_args.args.disable_all_custom_nodes = False
    import nodes  # noqa: E402  (ComfyUI's nodes.py)

    asyncio.run(nodes.init_builtin_extra_nodes())
    ok = asyncio.run(nodes.load_custom_node(str(REPO_ROOT)))
    if not ok:
        pytest.fail("ComfyUI failed to load the custom node package", pytrace=False)
    _COMFY_STATE["nodes"] = nodes
    _COMFY_STATE["path"] = Path(path)
    return _COMFY_STATE


@pytest.fixture(scope="session")
def comfyui():
    return _boot_comfy()


def pytest_configure(config):
    config.addinivalue_line("markers", "comfy: needs a ComfyUI checkout (COMFYUI_PATH)")
