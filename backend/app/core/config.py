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
BRONZE_DIR = ROOT_DIR / "data" / "bronze" / "md17"
SILVER_DIR = ROOT_DIR / "data" / "silver" / "md17"
CONFIGS_DIR = ROOT_DIR / "experiments" / "configs"
JOB_CONFIGS_DIR = CONFIGS_DIR / "jobs"
RUNS_DIR = ROOT_DIR / "experiments" / "runs"
JOB_LOGS_DIR = ROOT_DIR / "experiments" / "job_logs"

# The checkpoint that serves each project phase, and the one file both
# app.services.inference (live predictions) and app.api.benchmarks (the
# three-phase table) read - one mapping, not two that could drift apart.
#
# "phase1_deterministic" stays MPNN specifically: it's the shared backbone
# BLIP and GraphStochastic both extend, so it's the one Phase 1 result
# that's directly comparable to Phase 2/3 (same architecture, different
# stochasticity). The other four Phase 1 bake-off architectures
# (SchNet/PaiNN/PhysNet/DimeNet++ - same data, same loss, same budget,
# different architecture only) get their own keys so the full bake-off
# shows up in GET /benchmarks too, not just MPNN.
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
    "phase1_schnet": CHECKPOINTS_DIR / "phase1_schnet.pt",
    "phase1_painn": CHECKPOINTS_DIR / "phase1_painn.pt",
    "phase1_physnet": CHECKPOINTS_DIR / "phase1_physnet.pt",
    "phase1_dimenet": CHECKPOINTS_DIR / "phase1_dimenet.pt",
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

# Registry phase (ml/models/registry.py key) -> the vetted config a new
# training job's generated config is derived from (app.services.jobs).
# Only data: molecule/theory/gold_path/use_literature_split and name are
# overridden - every model/train kwarg stays exactly what that phase's
# own maintainer already tuned on ethanol, so launching a job on a new
# molecule can never silently diverge into an unvetted architecture or
# training setting.
JOB_TEMPLATE_CONFIGS: dict[str, Path] = {
    "mpnn": CONFIGS_DIR / "phase1_baseline.yaml",
    "schnet": CONFIGS_DIR / "phase1_schnet.yaml",
    "painn": CONFIGS_DIR / "phase1_painn.yaml",
    "physnet": CONFIGS_DIR / "phase1_physnet.yaml",
    "dimenet": CONFIGS_DIR / "phase1_dimenet.yaml",
    "blip": CONFIGS_DIR / "phase2_blip_nll.yaml",
    "graph_stochastic": CONFIGS_DIR / "phase3_graph_stochastic.yaml",
}


class Settings(BaseSettings):
    app_name: str = "Hydrogen Materials Discovery API"
    database_url: str = f"sqlite:///{BACKEND_DIR}/app.db"
    cors_origins: list[str] = ["http://localhost:5173"]

    class Config:
        env_file = ".env"


settings = Settings()
