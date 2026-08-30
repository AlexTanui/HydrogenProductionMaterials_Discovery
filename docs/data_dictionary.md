# Data Dictionary

MD17 schema, the bronze → silver → gold pipeline, and why each design
decision below is the way it is. Companion to
[glossary.md](../glossary.md) §3/§5 (folder ownership, high-level schema)
— this document is the detailed version, kept in sync with the actual
implementation in `ml/data/`.

**Status:** MD17 pipeline implemented and verified (9/9 bronze sources
process cleanly end to end). MD22 is not yet covered — see §6.

---

## 1. Where the data comes from

`data/bronze/{md17,md22}/` holds raw, untouched, as-downloaded files.
Nobody manually copies this between machines — get it one of two ways:

- `git lfs install && git clone`/`git pull` — the canonical MD17 set
  (~814MB) is committed via Git LFS.
- `scripts/download_bronze_data.sh` — fetches the same files directly
  from their public sources (quantum-machine.org / sgdml.org), no
  credentials needed. Works even without `git-lfs` installed.

Current MD17 bronze roster (9 files, one per molecule+theory level):

| File | Molecule | Theory | Atoms | Format |
|---|---|---|---|---|
| `md17_ethanol.npz` | ethanol | DFT | 9 | direct npz |
| `ethanol_ccsd_t.zip` | ethanol | CCSD(T) | 9 | zip, pre-split train/test npz |
| `benzene2018_dft.npz` | benzene | DFT | 12 | direct npz |
| `azobenzene_dft.npz` | azobenzene | DFT | 24 | direct npz |
| `paracetamol_dft.npz` | paracetamol | DFT | 20 | direct npz |
| `aspirin_ccsd.zip` | aspirin | CCSD | 21 | zip, pre-split train/test npz |
| `malonaldehyde_ccsd_t.zip` | malonaldehyde | CCSD(T) | 9 | zip, pre-split train/test npz |
| `toluene_ccsd_t.zip` | toluene | CCSD(T) | 15 | zip, pre-split train/test npz |
| `md17_uracil.npz` | uracil | DFT | 12 | direct npz |

**Theory level is part of the dataset's identity.** `ethanol` exists at
both DFT and CCSD(T) — these are never pooled as one dataset; every
silver/gold file is named `{molecule}_{theory}.npz` specifically so
that's structurally impossible to do by accident.

---

## 2. The pipeline: bronze → silver → gold

Implemented in `ml/data/md17.py` (loading/parsing) and
`ml/data/preprocessing.py` (promotion). Run it with:

```bash
python -m ml.data.preprocessing
```

### Bronze → silver (`bronze_to_silver`)

1. **Load.** A direct `.npz` is loaded as-is (`load_npz`). A `.zip` is
   opened and its `train`/`test` members loaded separately
   (`load_split_zip`), then concatenated into one array — but *which*
   configs were originally "test" is preserved as a boolean mask
   (`literature_test_mask`), not discarded. This is what lets gold
   preserve the literature split later instead of erasing it.
2. **Validate** (`validate_sample`): NaN-checks on `R`/`E`/`F`, non-zero
   atom/config counts, and — importantly — that `r_unit`/`e_unit` match
   the expected `Ang`/`kcal/mol` where the source file reports them at
   all. A source with different units would silently produce a model
   that's wrong by a constant factor if this weren't checked.
3. **Write** one `{molecule}_{theory}.npz` per source into
   `data/silver/md17/`, carrying `has_literature_split` and
   `literature_test_mask` forward for the gold step.

### Silver → gold (`silver_to_gold`)

Applies the train/val/test split. Two cases:

- **No literature split exists** (the 5 direct-`.npz` DFT sources): a
  **contiguous trajectory-block split** (default 80/10/10) — first 80% of
  timesteps train, next 10% val, last 10% test. **Never random** —
  adjacent MD frames are near-duplicates, so a random split leaks
  near-identical structures across train/test and inflates apparent
  accuracy.
- **A literature split ships with the source** (aspirin, malonaldehyde,
  toluene, ccsd_t-ethanol): the shipped test set is preserved exactly as
  the original benchmark defined it — a model trained here is directly
  comparable to published results. The *remaining* (non-test) pool is
  then split into train/val using the same 80/(80+10) ratio, still in
  original (contiguous) order.

Either way, gold asserts train/val/test are pairwise disjoint before
writing — a leakage bug here would silently invalidate every downstream
metric, so it's a hard assertion, not a warning.

### Gold → training (`ml/data/datasets.py`)

`MD17Dataset` reads **gold only**, never bronze/silver directly. Per
sample, `ml/data/md17.py::build_graph` constructs the neighbor graph at
train time: nodes = atoms within a cutoff radius (default 5 Å) of each
other, edge feature = **Gaussian/RBF-expanded distance** (16 bins by
default), not raw scalar distance. This matches Gilmer et al.'s MPNN (the
Phase 1 reference architecture) — a raw scalar distance is a poor edge
feature because energy/forces are sharply nonlinear near equilibrium bond
lengths.

---

## 3. Schema reference

**Bronze** (as downloaded) — see `glossary.md` §5 for the raw `z/R/E/F`
shapes.

**Silver** (`data/silver/md17/{molecule}_{theory}.npz`):

| Key | Type | Meaning |
|---|---|---|
| `z`, `R`, `E`, `F` | as bronze | concatenated across train+test if the source was a split zip |
| `molecule`, `theory` | str | dataset identity |
| `has_literature_split` | bool | whether this source shipped a pre-existing train/test split |
| `literature_test_mask` | bool[n_configs] | which configs were in the shipped test set (all-`False` if `has_literature_split` is `False`) |

**Gold** (`data/gold/md17/{molecule}_{theory}.npz`):

| Key | Type | Meaning |
|---|---|---|
| `z`, `R`, `E`, `F` | as silver | unchanged from silver |
| `train_idx`, `val_idx`, `test_idx` | int[] | disjoint index arrays into `R`/`E`/`F` |
| `molecule`, `theory` | str | dataset identity |
| `used_literature_split` | bool | whether `test_idx` came from the shipped split or a trajectory-block split |

---

## 4. Known issues already fixed here

- **`__MACOSX` zip junk broke every zip-based source.** macOS-zipped
  archives carry `__MACOSX/._<name>` AppleDouble resource-fork entries
  that mirror the real filename — including its `.npz` suffix — so a
  naive `endswith(".npz")` filter matches them too. Loading one crashes
  with `UnpicklingError` (214 bytes of binary resource-fork data isn't a
  valid npz). Fixed in `ml/data/md17.py::_is_real_member`.
- **Unknown element symbols no longer silently become atomic number 0.**
  The xyz-parsing fallback path used to default unrecognized symbols to
  `0` (not a real element) instead of failing — now raises.

---

## 5. Verified state

Running `python -m ml.data.preprocessing` against the current bronze
roster: **9/9 sources process successfully.** One data point worth
recording so it isn't mistaken for a bug later: the four CCSD/CCSD(T)
sources (aspirin, malonaldehyde, toluene, ccsd_t-ethanol) all report
exactly **889 train configs** after splitting — confirmed real, not a
bug: each ships exactly 1,000 literature-train configs (a standard
convention for that benchmark family), and `round(1000 × 0.8/0.9) = 889`.

---

## 6. Not yet covered

- **MD22** — `data/bronze/md22/` has 4 sources (AT-AT-CG-CG, DHA,
  buckyball-catcher, double-walled nanotube) but `ml/data/preprocessing.py`
  only processes `md17/` right now. Same pipeline shape should apply;
  the 370-atom nanotube may need the cutoff-radius question flagged in
  `glossary.md` §5 resolved first (a fixed 5 Å radius is much sparser,
  relatively, on a 370-atom system than on a 9-atom one).
- **Confounder/bias audit across theory levels** — `ROADMAP.md` calls
  for this as part of the data-foundation work; not yet done beyond
  keeping theory levels structurally separate.
