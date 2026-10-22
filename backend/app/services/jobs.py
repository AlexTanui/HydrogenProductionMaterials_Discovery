"""Background training jobs: launch ml.training.train as a real
subprocess for any molecule the gold catalog has data for (not just
ethanol), and track it.

In-memory job registry only - a prototype, not a production job queue.
Jobs are lost from this registry on server restart; the run directory,
checkpoint and log file a launched job wrote are not - they are real
files on disk regardless of whether this process remembers the job.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from fastapi import HTTPException

from app.core.config import CHECKPOINTS_DIR, JOB_CONFIGS_DIR, JOB_LOGS_DIR, JOB_TEMPLATE_CONFIGS, ROOT_DIR, RUNS_DIR
from app.services import inference as inference_module

# Ethanol's own locked-protocol budget - reused whenever a molecule has at
# least this many real frames; dropped to the whole split otherwise (see
# _subset_for below). 10000/5000 is tuned for ethanol's 444074-frame train
# block; the four literature-split CCSD(T) datasets (aspirin, ethanol_ccsd_t,
# malonaldehyde, toluene) have only ~889 train frames and would trip
# ml.training.train.stride_subset's own "size must be in 1..n" check if
# asked for 10000 of them.
_DEFAULT_TRAIN_SUBSET = 10000
_DEFAULT_VAL_SUBSET = 5000


@dataclass
class Job:
    id: str
    molecule: str
    theory: str
    phase: str
    config_name: str
    config_path: Path
    pid: int
    log_path: Path
    started_at: str
    # The live subprocess.Popen, when this job was launched by this
    # process (None for one recovered from disk after a restart - see
    # recover_jobs). Liveness MUST be checked via process.poll(), not a
    # bare os.kill(pid, 0): a child that already exited but was never
    # poll()/wait()-ed becomes a zombie, and most POSIX systems keep
    # answering os.kill(pid, 0) for a zombie's PID - polling is also what
    # reaps it. See job_status's own comment.
    process: Optional[subprocess.Popen] = None

    @property
    def checkpoint_path(self) -> Path:
        return CHECKPOINTS_DIR / f"{self.config_name}.pt"


_JOBS: dict[str, Job] = {}


def list_phases() -> list[str]:
    return sorted(JOB_TEMPLATE_CONFIGS)


def _subset_for(n_train: int, n_val: int) -> tuple[Optional[int], Optional[int], Optional[int]]:
    if n_train >= _DEFAULT_TRAIN_SUBSET and n_val >= _DEFAULT_VAL_SUBSET:
        return _DEFAULT_TRAIN_SUBSET, _DEFAULT_VAL_SUBSET, 42
    return None, None, None


def start_job(molecule: str, phase: str, theory: Optional[str] = None, epochs: Optional[int] = None) -> Job:
    if phase not in JOB_TEMPLATE_CONFIGS:
        raise HTTPException(422, f"Unknown phase {phase!r}; choose one of {list_phases()}")

    catalog = inference_module.MD17_MOLECULES_CATALOG
    info = catalog.get(molecule)
    if info is None:
        raise HTTPException(404, f"Unknown molecule {molecule!r}; see GET /molecules")
    if theory is not None and theory != info["theory"]:
        raise HTTPException(
            422,
            f"{molecule!r} is catalogued under theory {info['theory']!r}, not {theory!r} - only the "
            "catalog's own theory is launchable here (GET /molecules doesn't expose theory, but "
            "upload a different theory's gold file under a distinct molecule name to train both)",
        )
    resolved_theory = info["theory"]
    gold_path = Path(info["gold_path"])

    with np.load(gold_path, allow_pickle=True) as data:
        n_train, n_val = len(data["train_idx"]), len(data["val_idx"])
        used_literature_split = (
            bool(data["used_literature_split"]) if "used_literature_split" in data.files else False
        )

    template_path = JOB_TEMPLATE_CONFIGS[phase]
    with open(template_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    job_id = uuid.uuid4().hex[:10]
    config_name = f"job_{job_id}"
    raw["name"] = config_name
    raw["data"]["molecule"] = molecule
    raw["data"]["theory"] = resolved_theory
    raw["data"]["gold_path"] = str(gold_path.relative_to(ROOT_DIR))
    raw["data"]["use_literature_split"] = used_literature_split

    train_subset, val_subset, train_subset_seed = _subset_for(n_train, n_val)
    raw["train"]["train_subset"] = train_subset
    raw["train"]["val_subset"] = val_subset
    raw["train"]["train_subset_seed"] = train_subset_seed
    if epochs is not None:
        if epochs < 1:
            raise HTTPException(422, f"epochs must be a positive integer, got {epochs}")
        raw["train"]["epochs"] = epochs

    JOB_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    config_path = JOB_CONFIGS_DIR / f"{config_name}.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False)

    JOB_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = JOB_LOGS_DIR / f"{config_name}.log"

    # The child dup()s this fd when Popen starts it; closing our own copy
    # right after (the `with` block exiting) is the standard, safe
    # pattern - it does not close the child's independent copy.
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, "-m", "ml.training.train", str(config_path.relative_to(ROOT_DIR))],
            cwd=str(ROOT_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    job = Job(
        id=job_id, molecule=molecule, theory=resolved_theory, phase=phase,
        config_name=config_name, config_path=config_path, pid=process.pid,
        log_path=log_path, started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        process=process,
    )
    _JOBS[job_id] = job
    return job


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False  # recovered after a restart - see recover_jobs(); real pid is gone
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def recover_jobs() -> list[str]:
    """Rebuild what can be known about jobs from a previous process -
    JOB_CONFIGS_DIR/job_*.yaml survives a server restart even though the
    in-memory `_JOBS` dict does not. Recovered jobs get pid=0 (never
    "alive" - see _process_alive): the real PID isn't recoverable, so a
    job whose training process is actually still running shows as
    "failed" here until something else tells this server about it. That
    is an honest gap in an in-memory-only registry (a real limitation,
    documented, not hidden) - but it still reads real on-disk evidence
    (the checkpoint and run record), not a guess, for every job whose
    process has already finished one way or the other, which is most of
    them most of the time. Called once at server startup."""
    if not JOB_CONFIGS_DIR.exists():
        return []
    recovered = []
    for config_path in sorted(JOB_CONFIGS_DIR.glob("job_*.yaml")):
        job_id = config_path.stem.removeprefix("job_")
        if job_id in _JOBS:
            continue
        try:
            with open(config_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            molecule = raw["data"]["molecule"]
            theory = raw["data"]["theory"]
            phase = raw["model"]["phase"]
        except (OSError, KeyError, yaml.YAMLError):
            continue
        config_name = config_path.stem
        started_at = datetime.fromtimestamp(
            config_path.stat().st_mtime, tz=timezone.utc
        ).isoformat(timespec="seconds")
        _JOBS[job_id] = Job(
            id=job_id, molecule=molecule, theory=theory, phase=phase,
            config_name=config_name, config_path=config_path, pid=0,
            log_path=JOB_LOGS_DIR / f"{config_name}.log", started_at=started_at,
        )
        recovered.append(job_id)
    return recovered


def job_status(job: Job) -> dict[str, Any]:
    run_dir = RUNS_DIR / job.config_name
    run_json = run_dir / "run.json"
    history_csv = run_dir / "history.csv"

    # process.poll() when we launched it ourselves - both tells us
    # liveness AND reaps the child the instant it exits, so it never sits
    # around as a zombie that os.kill(pid, 0) would keep reporting as
    # alive. _process_alive(pid) is the fallback only for a job recovered
    # from disk after a restart (job.process is None there, pid is the
    # sentinel 0 - see recover_jobs - so this always resolves to False
    # for those, which is the documented, honest limitation).
    alive = job.process.poll() is None if job.process is not None else _process_alive(job.pid)
    record: dict[str, Any] = {}
    if run_json.exists():
        try:
            record = json.loads(run_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            record = {}  # written mid-update; next poll will see it whole

    last_val_loss = None
    if history_csv.exists():
        try:
            with open(history_csv, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                last_val_loss = float(rows[-1]["val_loss"])
        except (OSError, ValueError, KeyError):
            pass

    checkpoint_ready = job.checkpoint_path.exists()
    if alive:
        status = "running"
    elif checkpoint_ready:
        status = "complete"
    else:
        status = "failed"

    log_tail = ""
    if job.log_path.exists():
        lines = job.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        log_tail = "\n".join(lines[-15:])

    return {
        "id": job.id,
        "molecule": job.molecule,
        "theory": job.theory,
        "phase": job.phase,
        "config_name": job.config_name,
        "started_at": job.started_at,
        "status": status,
        "epochs_run": record.get("epochs_run"),
        "max_epochs": (record.get("protocol") or {}).get("max_epochs"),
        "best_epoch": record.get("best_epoch"),
        "best_val_loss": record.get("best_val_loss"),
        "last_val_loss": last_val_loss,
        "stop_reason": record.get("stop_reason"),
        "checkpoint_ready": checkpoint_ready,
        "log_tail": log_tail,
    }


def list_jobs() -> list[dict[str, Any]]:
    # Most recently started first - the ones someone just launched are
    # the ones worth seeing without scrolling.
    return [job_status(j) for j in sorted(_JOBS.values(), key=lambda j: j.started_at, reverse=True)]


def get_job(job_id: str) -> Job:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job {job_id!r}")
    return job
