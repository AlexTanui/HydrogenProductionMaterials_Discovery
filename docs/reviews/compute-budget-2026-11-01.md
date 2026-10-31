# Compute budget and Monte Carlo sample count (SCRUM-49, 1 Nov 2026)

**Short answer.**
- **MC sampling is cheap and scales linearly.** One MC sample costs about one forward-and-force pass: on the Mac's official runs, 0.6 ms per frame for BLIP and 0.33 ms for Phase 3. Time grows exactly with the sample count K, and memory doesn't grow with K at all.
- **The known remaining work fits in a few CPU-hours.** An optional repeat-seed noise study is the only large item, at 10–24 h.
- **Whether K = 20 (the count every stochastic run uses) is enough can't be measured yet.** It needs the two trained checkpoints that aren't on this laptop. The tool is ready (`python -m ml.training.mc_stability`), with a proposed criterion. Theory alone puts K = 20 at the edge of a 0.01 tolerance for a well-calibrated model.

**Hardware.**
- **The official runs:** every Phase 1/2/3 record trained on an Apple arm64 Mac, CPU only, 4 torch threads. Those numbers come from the run records.
- **This laptop:** Windows, Intel x86 (24 logical CPUs), 31 GB RAM, NVIDIA RTX 5070 Ti Laptop GPU (12 GB). Measured with [scripts/profile_compute.py](../../scripts/profile_compute.py) at commit `efb9818`; raw numbers in [reports/compute/](../../reports/compute/).
- **What the laptop timings leave out:** they time the model on pre-built graphs, without the per-frame graph construction a real epoch also does. Real MPNN runs on this laptop at 24 threads averaged 13–23 s per epoch, against 10.4 s projected. Treat laptop training projections as lower bounds.

## 1. Baseline: the Phase 1 model (MPNN, `phase1_baseline`)

| | Mac, 4 threads (run record) | Laptop CPU, 4 threads | Laptop CPU, 24 threads | Laptop GPU |
|---|---|---|---|---|
| Epoch, protocol budget (10,000 train + 5,000 val frames) | 10.6 s | 22.4 s | 10.4 s | 5.5 s |
| 200 epochs | 0.63 h | 1.24 h | 0.58 h | 0.30 h |
| One inference pass (energy + forces), per frame | 0.337 ms | 0.396 ms | 0.199 ms | 0.145 ms |
| Peak memory, training | not recorded | 931 MB RSS | 940 MB RSS | 135 MB GPU |

**All four phases, 200 epochs** (hours):

| Config | Mac (record) | Laptop CPU, 24 threads | Laptop GPU |
|---|---|---|---|
| `phase1_baseline` (MPNN) | 0.63 | 0.58 | 0.30 |
| `phase2_blip` (K = 1 training) | 1.33 | 1.27 | 0.76 |
| `phase2_blip_nll` (K = 4 training) | 3.86 | 5.18 | 2.54 |
| `phase3_graph_stochastic` | 0.73 | 0.65 | 0.44 |

Peak GPU memory during training is at most 426 MB (BLIP-NLL), so the 12 GB GPU isn't a constraint. The protocol fixes CPU training, though, so GPU runs wouldn't be comparable with the existing records.

## 2. Monte Carlo scaling

`predict_distribution` at K samples, in ms per frame (energy + forces, batch 32, 256 val frames):

| K | BLIP, laptop CPU 24 threads | BLIP, laptop GPU | Phase 3, laptop CPU 24 threads | Phase 3, laptop GPU |
|---|---|---|---|---|
| 1 | 0.54 | 0.33 | 0.36 | 0.18 |
| 2 | 1.12 | 0.64 | 0.67 | 0.32 |
| 5 | 2.25 | 1.48 | 1.31 | 0.83 |
| 10 | 5.07 | 2.68 | 2.76 | 1.74 |
| 20 | 9.49 | 5.41 | 5.48 | 3.50 |
| 50 | 24.04 | 13.82 | 14.01 | 8.66 |
| 100 | 47.43 | 27.67 | 27.56 | 17.79 |

- **Time is linear in K.** Cost per sample stays within ±12% from K = 2 to 100, on every device and at 4 threads too ([JSON](../../reports/compute/)).
- **On the Mac,** the records give 0.59–0.65 ms per sample for BLIP and 0.33 ms for Phase 3, measured at K = 20 while scoring all 55,509 test frames.
- **Memory doesn't grow with K.** Samples run one after another, and each sample's graph is freed before the next. On CPU, peak RSS rises 26–69 MB above setup by K = 5, then by at most 5 MB more up to K = 100. On GPU it's flat at 152 MB (BLIP) and 122 MB (Phase 3).
- **Scoring the whole test split at K costs** 55,509 × K × (cost per sample). On the Mac that's:
  - BLIP: 11–12 min at K = 20 (the records show 10.9–12.2 min), about 29 min at K = 50, and about an hour at K = 100.
  - Phase 3: about half that.

## 3. Stability versus cost: blocked on checkpoints; tool ready

[ml/training/mc_stability.py](../../ml/training/mc_stability.py) is the tool.
- **What it does:** on 2,000 validation frames, never test, since this picks a protocol value, it draws 100 stochastic passes per frame for each of 3 seeds. It scores the first K passes for K = 2, 5, 10, 20, 50 and 100 through `UQAccumulator`, the same path `evaluate.py` uses. Each prefix equals `predict_distribution(K)` exactly; a test pins that.
- **Cost per checkpoint:** about 6 min on the Mac for BLIP, and about half that for Phase 3.

**Proposed stability criterion (not agreed):** the smallest K from which every larger K satisfies both of these:
- the energy and force ECE, coverage at 90% and Spearman are each within **0.01** of their value at K = 100 (mean over the 3 seeds);
- the seed-to-seed spread is also within 0.01.

Sharpness and Gaussian NLL are reported but not judged, because both carry units. For scale, the three stochastic runs' recorded ECEs differ by at least 0.07, but their force Spearman values differ by as little as 0.011 (0.262, 0.273, 0.300). A 0.01 tolerance is therefore about the smallest between-run difference the tables show; a looser one could blur that comparison.

**What to expect, from theory alone.** Take a perfectly calibrated model with Gaussian errors. Estimating each frame's std from K samples still lowers coverage, because the error-to-std ratio then follows a Student-t distribution with K − 1 degrees of freedom:

| K | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|
| Coverage at nominal 90% | 82.5% | 86.6% | 88.4% | 89.4% | 89.7% |
| ECE from finite K alone | 0.063 | 0.029 | 0.014 | 0.005 | 0.003 |

So for a well-calibrated model, K = 20 is about 0.011 from K = 100 on ECE: right at the proposed tolerance. For today's BLIP, which is overconfident by more than 10×, the effect of K is negligible next to its miscalibration. The measurement decides.

**Blocked on** `phase2_blip_nll.pt` and `phase3_graph_stochastic.pt` (sha256 `bf6c13fb…` and `9412d664…`, as recorded in their run records). They are not on this laptop.

## 4. Projection for the remaining work

| Item | Mac CPU (official hardware) | Status |
|---|---|---|
| Retrain and score `phase2_blip` (Step 7, SCRUM-69) | 1.5 h | planned |
| Re-run SCRUM-68 calibration for 3 checkpoints, after its fix (55,509 val + 5,000 test frames at K = 20) | 0.5 h | planned, needs checkpoints |
| MC stability study, 2 checkpoints | 0.2 h | needs checkpoints |
| Re-score the 3 stochastic runs if the study picks K > 20 | 1.2 h at K = 50; 2.4 h at K = 100 | conditional |
| Noise study, 2 extra seeds of MPNN, BLIP-NLL and Phase 3 | 10.4 h | optional, decided after Step 4 |
| Noise study, 2 extra seeds of all 8 configs | 24 h | optional |

- **Planned items:** about 2.2 h.
- **Planned plus the K re-score:** at most 4.6 h.
- **With the full noise study:** about 29 h of CPU. Running on both the Mac and this laptop in parallel roughly halves the wall time.
- **How long is available is not in the repo.** The sprint calendar in `docs/sprint_process.md` ends on 26 Oct, so the budget comparison is open.

## 5. If the projection exceeds what's available

These options are in order of what they save for the least lost:
1. **Smaller noise study:** MPNN, BLIP-NLL and Phase 3 only, which is 10 h rather than 24 h. One extra seed instead of two halves either figure.
2. **Run the noise study on the GPU,** 2–3× faster per the table above, as its own table. GPU numerics differ from the CPU records, so the two must not share a table.
3. **Cap K at the smallest stable value up to 50,** rather than 100. Scoring cost is linear in K.
4. **Drop the optional leakage-control repeats** (SCRUM-47), about 3–5 h.

## Reproduce

```
python scripts/profile_compute.py --device cpu --threads 4 24 --out reports/compute/NAME_cpu.json   # ~5 min
python scripts/profile_compute.py --device cuda --out reports/compute/NAME_cuda.json                # ~1 min
python -m ml.training.mc_stability --checkpoint experiments/checkpoints/phase2_blip_nll.pt --out reports/mc_stability/phase2_blip_nll.json
```
