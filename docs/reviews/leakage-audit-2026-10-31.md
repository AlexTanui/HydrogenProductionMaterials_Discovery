# Leakage audit of the gold splits (31 Oct 2026, SCRUM-47)

**Question.** Adjacent MD frames are near-duplicates, so a split leaks if a held-out frame has a training frame that is structurally the same configuration. Do the gold splits leak, and by how much would leakage inflate the reported errors?

**Short answer.**
- **Test does not leak into training.** No test frame is within two MD steps of any training frame. Test never touches train, because the 55,509-frame validation block sits between them.
- **Two boundary near-duplicates.** Validation touches train, and test touches validation, with no purge buffer, so each boundary has one frame one MD step from the other block, then a short ramp of a few frames. No benchmark run trained or selected on those frames.
- **Decisions (31 Oct).** The 2-MD-step nearest-neighbour threshold is approved and enforced in the tests. A 10-frame purge buffer for new splits (keeping the existing split) and split provenance in every gold file are proposals for review; nothing in `ml/data/` has changed.
- **Random-split control (part 3b).** On one machine and one commit, a random split scored lower test errors than the block split: energy MAE 0.0587 against 0.0792 kcal/mol, force MAE 0.1795 against 0.2162 kcal/mol/Å. Each split has one training run, so the gap is indicative and could be within run-to-run noise, which has not been measured yet.

Measured at commit `b4a8634` with `scripts/leakage_audit.py`; raw numbers in [reports/leakage/ethanol_dft.json](../../reports/leakage/ethanol_dft.json) and [reports/leakage/ethanol_ccsd_t.json](../../reports/leakage/ethanol_ccsd_t.json). Tests in [tests/ml/test_split_leakage.py](../../tests/ml/test_split_leakage.py); method in [ml/utils/leakage.py](../../ml/utils/leakage.py).

## What's in use

Every benchmarked config (`phase1_*`, `phase2_*`, `phase3_*`) trains on **ethanol (DFT)**, `data/gold/md17/ethanol_dft.npz`. The legacy `dimenet_*` configs use ethanol (CCSD(T)). These are the only two gold files on this machine, and both were audited. Other molecules can now be trained through the app (`/jobs/train`), but no benchmark uses them.

## Method

- **Distance:** RMSD after optimal superposition (Kabsch), in Å, float64.
  - Proper rotations only.
  - Minimised over the 12 relabellings of ethanol's equivalent hydrogens: 2 on one carbon, 3 on the other. A methyl rotation swaps those labels, so without this, physically identical frames look far apart.
  - Bonding is read from frame 0 and was stable on 3,000 random frames.
  - Allowing mirror images, as a sensitivity check, moves the test→train median from 0.077 to 0.073 Å and leaves the minimum unchanged.
- **Search:** exact all-pairs RMSD (55,509 × 444,074) is out of reach.
  - Candidates are shortlisted by interatomic-distance vectors, best over the query's relabellings, and exact RMSD is computed for the 512 closest.
  - Checked against brute force (every train frame, every relabelling): 55/55 exact on a validation sample during development, and 19/20 in the recorded run, the miss by 0.0004 Å.
  - A first, fully permutation-invariant descriptor ranked candidates too poorly (72% misses at k=16 against k=256) and was replaced.
- **Scale:** RMSD between frame *t* and frame *t + lag* inside the train block, which says what "near-duplicate" means in this trajectory. Lag-1 RMSD (0.014 Å) is 30× below the random-pair RMSD (0.42 Å), confirming the file is in trajectory order.

| Lag (MD steps) | 1 | 2 | 3 | 5 | 10 | 20 | 100 | 10,000 | random pairs |
|---|---|---|---|---|---|---|---|---|---|
| Median RMSD (Å) | 0.0141 | 0.0283 | 0.0419 | 0.0678 | 0.1235 | 0.191 | 0.353 | 0.411 | 0.420 |

## Results: ethanol (DFT)

**Split structure.**
- 444,074 / 55,509 / 55,509 frames, contiguous blocks in train < val < test order, exactly what `trajectory_block_split` produces.
- **Gaps between blocks: 0 frames** (train→val and val→test); train→test: 55,509 (the val block).
- **The file records no provenance:** keys are `z R E F train_idx val_idx test_idx molecule theory used_literature_split`. There's no source hash, split method or parameters, or commit. (The block split has no randomness, so it needs no seed.)

**Nearest neighbours** (every query frame against the reference set):

| Query → reference | Frames | Closest (Å) | p1 | p5 | Median | Max | Within 2 MD steps (< 0.0283 Å) |
|---|---|---|---|---|---|---|---|
| test → train | 55,509 | 0.0344 | 0.0533 | 0.0598 | 0.0773 | 0.142 | **0** |
| val → train | 55,509 | **0.0148** | 0.0537 | 0.0605 | 0.0795 | 0.170 | **1** |
| test → val | 55,509 | **0.0143** | 0.0617 | 0.0706 | 0.0919 | 0.195 | 1 |
| test → train frames actually used (10,000) | 55,509 | 0.0457 | 0.0640 | 0.0719 | 0.0919 | 0.158 | 0 |
| val frames used (5,000) → train frames used | 5,000 | 0.0531 | 0.0640 | 0.0731 | 0.0949 | 0.196 | 0 |

"Actually used" means the stride subsets every benchmarked run trained and selected on (`experiments/runs/phase1_baseline/subsets.json`; `benchmark.py` checks all runs share them).

**Reading it.**
1. **Test is clean of adjacency leakage.** Its closest training frame is 0.034 Å away (between 2 and 3 MD steps), and that pair is 181,429 frames apart in time. It's the trajectory revisiting a configuration, not a neighbour across a boundary. The test frames nearest the boundary are 0.083–0.092 Å from train, which is typical.
2. **Each boundary leaks a handful of frames.** Val frame 444,074 is 0.0148 Å from train frame 444,073 (one step), then 0.029, 0.041, 0.052, 0.062 Å for the next four, after which the bulk takes over. Test→val has the same ramp at frame 499,583. With validation used for checkpoint selection, early stopping and (for NLL runs) the variance floor, this is a real but tiny path: about 4 frames of 55,509.
3. **No benchmark run touches those frames.** The stride subsets leave the last 4,074 train frames and the last 509 val frames undrawn (docs/phase1_protocol.md, "Uncovered tail"). That gives 4,114 frames between the used train and val frames and 519 between the used val frames and test: an accidental purge buffer. Against the frames actually used, nothing is closer than 0.046 Å.
4. **MD17 is densely sampled, whatever the split.** Every test frame has a training frame within 0.142 Å, with a median of 0.077 Å (about 5–6 MD steps), although its nearest training frame in time is at least 55,510 steps away. The block split stops adjacency leakage; it doesn't make test frames unlike training frames. A random frame-level split would put test frames one step from training (median 0.014 Å against the full train block, from a 1,000-frame sample of the control split below).

## Results: ethanol (CCSD(T)), literature split

- **Literature split preserved:** 889 / 111 / 1,000 frames (`used_literature_split=True`).
- **The frames are not in trajectory order:** lag-1 RMSD ≈ random-pair RMSD (0.42 vs 0.43 Å), so the lag scale doesn't apply.
- **A few near-duplicates exist:**
  - test → train: closest 0.0126 Å, median 0.111 Å; **6 of 1,000** test frames within 0.028 Å (DFT's 2-step scale), 1 within 0.014 Å;
  - val → train: closest 0.0137 Å.
- That's a property of the published split, which is kept deliberately for comparability with published numbers. Only the legacy `dimenet_*` configs use this file.

## Decisions and proposals (31 Oct)

Only the threshold (1) is enforced. The other two change `ml/data/` and are proposals for review.

1. **Approved: no held-out frame within 2 MD steps of a training frame** (`NN_THRESHOLD_LAG = 2`, i.e. the lag-2 median, 0.028 Å for ethanol DFT).
   - Lag 1 would miss the boundary frame (0.0148 > 0.0141).
   - Lag 3 would flag 12 test frames that are configuration recurrences hundreds of thousands of steps apart, which is the dataset rather than the split.
   - With lag 2: test → train passes (0 frames); val → train fails on frame 444,074, because the existing split is kept (2). The tests record that one case as a strict expected failure, and separately check the frames actually used, which pass for both validation and test.
   - Expressed as a lag rather than in ångström, so it transfers to other molecules.
2. **Proposed: a 10-frame purge buffer between blocks for any new split** (`AGREED_PURGE_BUFFER_FRAMES = 10` once agreed; unset until then, so its test skips).
   - The boundary ramp reaches the bulk's 1st percentile (0.053 Å) after about 4 frames, and lag-10 RMSD (0.12 Å median, 0.09 Å 5th percentile) clears the bulk with margin.
   - Cost: 20 of 555,092 frames.
   - **For `ethanol_dft` as it stands**, adding a buffer changes every run's validation and test frames, so all eight runs would need retraining and re-scoring. The recommendation is not to re-split it: the frames every run actually used already sit 4,114 / 519 frames apart. Record it as a documented exemption backed by the "actually used" rows above, and require the buffer from the splitter for new gold files. This is a change to `ml/data/preprocessing.py`.
3. **Proposed: split provenance in every gold file:** `source_sha256`, `split_method`, `split_params` and `git_commit`, written by `silver_to_gold`. A change to `ml/data/`. The test for it stays a strict expected failure until then.

## Part 3b: random-split control (trained 31 Oct, UTC)

- **What:** `phase1_baseline`'s config, trained twice on one machine (Windows x86, CPU, 24 threads) from one clean commit (`465f769`), one run after the other:
  - `leakctl_mpnn_random_split` on a random frame-level split of the same ethanol (DFT) frames;
  - `leakctl_mpnn_block_split` on the real block split, as the same-machine reference.
- **Why a same-machine reference:** an earlier retrain of the Phase 1 configs moved SchNet's energy MAE from 0.1173 to 0.1714. That retrain changed the machine and thread count (Windows x86, 24 threads → macOS arm64, 4 threads) and the commit (`d9f3814` → `b1c2d89`) together, so the change can't be put down to any one of them. A comparison with `phase1_baseline`, trained elsewhere, could carry the same mix of differences.
- **Files:**
  - `scripts/make_random_split_control.py` (seed 0) writes `data/external/leakage_controls/ethanol_dft_random_split.npz`, which is gitignored and not scanned by the backend. The split's seed, method, source sha256 (`245697fe…`) and commit (`023d40d`) are in the file's `control_*` keys and in the JSON next to it.
  - `experiments/configs/leakctl_mpnn_random_split.yaml` and `leakctl_mpnn_block_split.yaml` differ from `phase1_baseline.yaml` only in `name` and, for the control, `data.gold_path`.
  - Run records: `experiments/runs/leakctl_mpnn_random_split/` and `experiments/runs/leakctl_mpnn_block_split/`. The `leakctl_` names keep them out of the default Phase 1 table. `benchmark.py` refuses to put the two in one table (different gold file, frames and test fingerprint), and accepts the block control next to `phase1_baseline` (same frames, test fingerprint `518af6e3…`).

| Run | Split | Machine, threads, commit | Energy MAE | Force MAE | Energy RMSE | Force RMSE |
|---|---|---|---|---|---|---|
| `leakctl_mpnn_random_split` | random, frame-level (its own 55,509 test frames) | Windows x86, 24, `465f769` | **0.0587** | **0.1795** | 0.0796 | 0.3327 |
| `leakctl_mpnn_block_split` | gold block split | Windows x86, 24, `465f769` | **0.0792** | **0.2162** | 0.1007 | 0.3281 |
| `phase1_baseline` (for reference) | gold block split | macOS arm64, 4, `755d57f` | 0.0779 | 0.1775 | 0.0962 | 0.2774 |

Test set; energy in kcal/mol, force per component in kcal/mol/Å. Both new runs hit the 200-epoch cap (best epochs 199 and 197).

**Reading it.**
1. **On the same machine and commit, the random split scored lower:** energy MAE 26% lower and force MAE 17% lower, the direction leakage would produce. Force RMSE doesn't follow (0.333 against 0.328).
2. **One training run per split, so the gap is indicative and could be within run-to-run noise.** The only repeat available is the block control against `phase1_baseline`: the same config, frames and seeds, but a different machine, thread count and commit. Their losses differ from the first epoch, and they end 1.7% apart on energy MAE and 22% apart on force MAE. The force gap above is no larger than that one difference; the energy gap is larger, but one pair is not a noise estimate. Run-to-run noise is measured separately (Step 4 of the evaluation plan). Until then, neither gap should be quoted as the size of a leakage effect.
3. **The control can't say which change made the gap.** A random split changes two things at once:
   - its test frames have a training frame about one MD step away in the full train block (median 0.014 Å), although against the 10,000 frames actually trained on they are about as far as the block split's (median 0.088 against 0.092 Å);
   - its test frames are spread over the whole trajectory instead of being its last 10%.
4. **Against part 3a's prediction of a small gap** (because the two splits' used training frames are about equally far from test): the force gap fits it; the energy gap is larger than predicted, and whether it survives run-to-run noise is open.

**Cost:** 44 min (control) and 77 min (reference). Epoch times varied between about 11 and 20 s within each run on the same laptop, so wall-clock time doesn't compare the two.

## Reproduce

```
python scripts/leakage_audit.py data/gold/md17/ethanol_dft.npz          # ~3.5 min (GPU shortlist, CPU RMSD)
python scripts/leakage_audit.py data/gold/md17/ethanol_ccsd_t.npz --run none
pytest tests/ml/test_leakage_method.py tests/ml/test_split_leakage.py -rs
python scripts/make_random_split_control.py                             # 3b split only; trains nothing
python -m ml.training.train experiments/configs/leakctl_mpnn_random_split.yaml   # 45-80 min each on CPU
python -m ml.training.train experiments/configs/leakctl_mpnn_block_split.yaml
```
