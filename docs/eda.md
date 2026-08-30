# Exploratory Data Analysis — MD17 (gold stage)

Full analysis with charts: [`notebooks/eda_md17.ipynb`](../notebooks/eda_md17.ipynb)
(executed, outputs embedded — open it to see the actual plots referenced
below). This document is the write-up; the notebook is the evidence.

**Scope:** reads exclusively from `data/gold/md17/` — the same discipline
`ml/data/datasets.py::MD17Dataset` follows — so everything here reflects
exactly what training sees, not raw bronze data. MD22 is not covered yet
(see §5).

---

## 1. What was checked

| # | Question | Method |
|---|---|---|
| 1 | Are energy/force distributions sane (no corruption, outliers, bimodality)? | Per-dataset histograms, all 9 gold datasets |
| 2 | Is the default 5 Å cutoff radius (`ml/data/md17.py::build_graph`) actually doing anything? | Pairwise interatomic distance distributions vs. cutoff, all 9 datasets |
| 3 | Does the same molecule at two theory levels (DFT vs CCSD(T)) look reasonable relative to each other? | Ethanol DFT vs CCSD(T), centered-energy and force-magnitude overlay |
| 4 | Does the train/val/test split logic actually avoid leakage? | Visual check: energy vs. trajectory position, colored by split |

---

## 2. Findings

### Distributions are clean

All 9 MD17 gold datasets (aspirin, azobenzene, benzene, ethanol×2
theories, malonaldehyde, paracetamol, toluene, uracil) have unimodal,
roughly bell-shaped energy and force-magnitude distributions. No
bimodality, no outlier clusters, no signs of corrupted configs beyond
what `validate_sample`'s NaN/shape/unit checks already catch
programmatically — this confirms visually what those checks assert
numerically.

### The 5 Å cutoff radius barely matters for small MD17 molecules — and that's worth knowing before it's reused for MD22

This was never checked against real data before this EDA pass. Measured
fraction of atom pairs within 5 Å, per dataset:

| Dataset | Atoms | % pairs ≤ 5 Å |
|---|---|---|
| ethanol (dft / ccsd_t) | 9 | 100% |
| malonaldehyde (ccsd_t) | 9 | 100% |
| benzene (dft) | 12 | 98% |
| uracil (dft) | 12 | 98% |
| toluene (ccsd_t) | 15 | 92% |
| paracetamol (dft) | 20 | 73% |
| aspirin (ccsd) | 21 | 72% |
| azobenzene (dft) | 24 | 59% |

**Reading this:** for the 9-atom molecules, the "neighbor graph" the MPNN
sees is just the complete graph — every atom is within cutoff of every
other atom, so the cutoff radius provides zero locality structure there.
It only starts doing real graph-sparsification work on the larger MD17
molecules. This isn't a bug — 5 Å is a reasonable default and MD17
molecules are genuinely small — but it means **the same default cannot be
assumed to behave sensibly on MD22**, where the largest system (the
double-walled nanotube) has 370 atoms. A cutoff that's "basically no-op"
on a 9-atom molecule will be meaningfully sparse — possibly too sparse,
possibly fragmenting the graph — on something 40× larger. This should be
resolved deliberately (empirically, the same way this table was produced)
before Phase 1 training touches MD22, not inherited by default.

### Ethanol at DFT vs. CCSD(T): no red flags, but not a real comparison either

Centered energy spread and force-magnitude distributions look broadly
similar between the two theory levels. Important caveat, stated plainly
so it isn't overclaimed later: **these are two independent simulations**
(different sampling/temperature), not the same configurations evaluated
at two levels of theory — so this is not a validation of one method
against the other, just a sanity check that neither dataset looks
anomalous relative to the other.

### Split logic behaves exactly as designed

Visual check (energy vs. trajectory-order index, colored by split)
confirms both split paths in `ml/data/preprocessing.py::silver_to_gold`
produce clean, non-interleaved, contiguous blocks:

- **Trajectory-block split** (ethanol, no shipped split): train/val/test
  are three contiguous chronological segments, in order.
- **Literature split** (aspirin, and the other three sources that ship
  one): the preserved test set occupies exactly the index range the
  original benchmark defined; train/val are carved from the remaining
  pool, still contiguous.

No interleaving in either case — which is what "no leakage" looks like
visually, matching the hard `isdisjoint` assertions already in the code.

---

## 3. What this changes

Nothing in the pipeline code changed as a result of this EDA pass — it's
a validation, not a bug hunt, and it passed. The one actionable item is
the cutoff-radius question above, tracked as an open item in
`docs/data_dictionary.md` §6.

---

## 4. Reproducing this

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace eda_md17.ipynb
```

Requires `data/gold/md17/*.npz` to exist — run
`python -m ml.data.preprocessing` first if it doesn't (see
`docs/data_dictionary.md`).

---

## 5. Not yet covered

- **MD22** — no gold-stage MD22 data exists yet (`ml/data/preprocessing.py`
  only processes `md17/`), so none of this analysis extends there yet.
  Same questions apply, especially the cutoff-radius one, and probably
  matter more given the larger systems.
- **Confounder analysis across theory levels** (DFT functional variants,
  sampling temperature) — flagged in `ROADMAP.md` as data-foundation
  work, not attempted here.
