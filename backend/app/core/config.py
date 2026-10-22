from __future__ import annotations

import sys
from pathlib import Path

from pydantic_settings import BaseSettings

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
ROOT_DIR = BACKEND_DIR.parent

# `ml` lives at the repo root, not inside backend/, and isn't installed as
# a package in this venv (pyproject.toml declares it under
# [tool.setuptools.packages.find], but nothing runs `pip install -e .`).
# uvicorn launched from backend/ - the documented command in CLAUDE.md -
# would not find it otherwise, regardless of PYTHONPATH. Inserted once,
# here: every module in this app imports from app.core.config (directly
# or via app.services.inference) before it ever imports from ml, so this
# runs first no matter the entry point.
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

CHECKPOINTS_DIR = ROOT_DIR / "experiments" / "checkpoints"
RESULTS_DIR = ROOT_DIR / "experiments" / "results"
GOLD_DIR = ROOT_DIR / "data" / "gold" / "md17"

# The checkpoint that serves each project phase, and the one file both
# app.services.inference (live predictions) and app.api.benchmarks (the
# three-phase table) read - one mapping, not two that could drift apart.
#
# Two BLIP entries, deliberately: phase2_blip_k1 is the original run
# (train_num_mc_samples=1, plain MSE); phase2_blip is phase2_blip_nll.pt,
# the K=4/NLL run after SCRUM-61's numerical-stability fix. Both are real,
# scored results - showing them side by side is the actual point (does
# more MC samples + a likelihood loss improve calibration? measured
# answer: no, slightly worse - see docs/model_cards and SCRUM-61). The
# K=1 run was trained from a dirty tree (docs/phase2_remaining_work.md);
# labelled honestly in the UI, not hidden. "phase2_blip" (unqualified)
# keeps meaning the current/default BLIP for anything that only wants one.
PHASE_CHECKPOINTS: dict[str, Path] = {
    "phase1_deterministic": CHECKPOINTS_DIR / "phase1_baseline.pt",
    "phase2_blip_k1": CHECKPOINTS_DIR / "phase2_blip.pt",
    "phase2_blip": CHECKPOINTS_DIR / "phase2_blip_nll.pt",
    "phase3_graph_stochastic": CHECKPOINTS_DIR / "phase3_graph_stochastic.pt",
}
DEFAULT_PHASE = "phase1_deterministic"
PHASE_ALIASES: dict[str, str] = {"phase1": "phase1_deterministic", "phase2": "phase2_blip", "phase3": "phase3_graph_stochastic"}

# Only ethanol (DFT) has a trained checkpoint right now. See
# app.services.inference.MD17_MOLECULES_CATALOG for the real catalog, read
# from data/gold/md17/*.npz rather than hand-maintained.
TRAINED_MOLECULE = "ethanol"


class Settings(BaseSettings):
    app_name: str = "Hydrogen Materials Discovery API"
    database_url: str = f"sqlite:///{BACKEND_DIR}/app.db"
    cors_origins: list[str] = ["http://localhost:5173"]

    class Config:
        env_file = ".env"


settings = Settings()
