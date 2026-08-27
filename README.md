# Hydrogen Production Materials Discovery Using Deep Neural Networks

**PG-S2-47** — Adelaide University, College of Engineering & IT
Agency supervisor: Henry Li (<henry.li@adelaide.edu.au>) · Academic supervisor: Dhika Pratama (<dhika.pratama@adelaide.edu.au>)

Investigating where stochasticity should be introduced in molecular graph
neural networks — model weights or graph representations — to make machine
learning interatomic potentials (MLIPs) know when their energy/force
predictions can be trusted. Developed and validated on MD17/MD22 as the
concrete technical plan behind the project's materials-discovery goal (see
`glossary.md`'s note on how the two relate).

This repo currently holds **structure only** in `ml/` — folders and empty
placeholder files there; `backend/` and `frontend/` have a working
interface-preview build.

**Start here: [glossary.md](glossary.md)** — team structure and
responsibilities, full architecture, folder-by-folder ownership, API
contracts, data schema, experiment config schema, and terminology.

**Then: [ROADMAP.md](ROADMAP.md)** — the 10-week execution plan: phase
sequencing, week-by-week ownership, integration checkpoints, per-phase
definition of done, and the fallback order if a track falls behind.

[CLAUDE.md](CLAUDE.md) has the condensed command/convention reference used
by Claude Code once implementation begins.

## Getting the data

The MD17/MD22 dataset (~1GB) is never committed to this repo. Get it with:

```bash
scripts/download_bronze_data.sh
```

which pulls the public benchmark files into `data/bronze/{md17,md22}/`.
Safe to re-run — it skips files you already have.
