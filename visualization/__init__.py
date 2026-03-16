"""Visualization package for the Qt-based pellet injection dashboard."""

from __future__ import annotations

import os
from pathlib import Path


VISUALIZATION_ROOT = Path(__file__).resolve().parent
RUNTIME_CACHE_ROOT = VISUALIZATION_ROOT / ".runtime_cache"
RUNTIME_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
(RUNTIME_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)

os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE_ROOT / "matplotlib"))
