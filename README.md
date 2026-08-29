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

`data/bronze/` (~814MB) is committed via **Git LFS**. Install the LFS
client once (`brew install git-lfs` / `apt install git-lfs`), then
`git lfs install` and clone/pull as normal — the real files come down
automatically. If LFS bandwidth runs out or you'd rather skip it entirely,
`scripts/download_bronze_data.sh` fetches the same files directly from
their public sources instead. See `glossary.md` §3 for the full data
staging story (bronze/silver/gold, why they're split by dataset).
