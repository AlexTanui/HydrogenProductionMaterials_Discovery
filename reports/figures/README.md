# Exported figures

PNGs lifted out of the executed notebooks by
[`scripts/export_notebook_figures.py`](../../scripts/export_notebook_figures.py).
Nothing here is drawn fresh: each file is the exact image embedded in the
notebook's output cells, so a figure in a deck and the same figure in the
notebook can never disagree.

Regenerate after re-running a notebook:

```bash
python -m scripts.export_notebook_figures
```

No data, no Jupyter and no Python are needed to *use* these — pull the
repo and drop the PNG into a slide. The full write-up is
[`docs/eda.md`](../../docs/eda.md).

## MD17 — gold stage (`notebooks/eda_md17.ipynb`)

| File | What it shows | One-line caption for a slide |
|---|---|---|
| `md17_02_energy_distributions.png` | Energy histogram per molecule+theory, 9 panels | All nine MD17 datasets have clean unimodal energy distributions — no corruption, no outlier clusters |
| `md17_03_force_magnitude_distributions.png` | Mean per-atom force magnitude, same 9 panels | Force magnitudes are similarly well behaved across every dataset |
| `md17_04_interatomic_distances_vs_the_5_cutoff.png` | Pairwise distances against the 5 Å cutoff, smallest vs largest molecule | The 5 Å cutoff captures 100% of ethanol's atom pairs but only 59% of azobenzene's |
| `md17_05_same_molecule_two_theory_levels_ethanol.png` | Ethanol at DFT vs CCSD(T), centred energy and force | Neither theory level looks anomalous relative to the other |
| `md17_06_train_val_test_split_integrity_check.png` | Energy vs trajectory position, coloured by split | Train/val/test are contiguous blocks with no interleaving — no leakage |

**The one worth a slide of its own** is `md17_04`. It shows that for the
9-atom molecules the neighbour graph is the complete graph — every atom is
within cutoff of every other — so the cutoff provides no locality
structure at that size. That is not a problem for MD17, but it means the
same 5 Å default cannot be assumed to behave sensibly on MD22's systems of
up to 370 atoms. It is the one open question `docs/eda.md` leaves on the
table.

## Phase 1 bake-off — PhysNet

Produced separately by
[`scripts/plot_physnet_results.py`](../../scripts/plot_physnet_results.py)
from a trained checkpoint, not from a notebook. Regenerate with:

```bash
python -m scripts.plot_physnet_results --config experiments/configs/phase1_physnet.yaml
```

| File | What it shows |
|---|---|
| `physnet_parity.png` | Predicted vs reference energy and force, with the 1:1 line |
| `physnet_error_distribution.png` | Signed error — whether error is noise or bias |
| `physnet_force_error_vs_magnitude.png` | Whether accuracy holds where forces are large |

These are checkpoint-dependent, so they are **not** committed — rerun the
script against your own checkpoint.
