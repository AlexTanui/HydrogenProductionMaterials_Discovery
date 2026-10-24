import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { IconChart } from "../components/icons";
import { BenchmarkResult, fetchBenchmarks } from "../api/client";

const PHASE_LABELS: Record<string, string> = {
  phase1_deterministic: "Phase 1 — MPNN",
  phase1_schnet: "Phase 1 — SchNet",
  phase1_painn: "Phase 1 — PaiNN",
  phase1_physnet: "Phase 1 — PhysNet",
  phase1_dimenet: "Phase 1 — DimeNet++",
  phase2_blip_k1: "Phase 2 — BLIP (K=1)",
  phase2_blip: "Phase 2 — BLIP (K=4 NLL)",
  phase3_graph_stochastic: "Phase 3 — graph-space",
};

// Phase 1's five architectures are a separate comparison from Phase 2/3
// (different question: "which deterministic backbone is most accurate?"
// vs. "where should stochasticity live?") - kept visually distinct so a
// reader doesn't average them into one "Phase 1" bar.
const PHASE1_KEYS = ["phase1_deterministic", "phase1_schnet", "phase1_painn", "phase1_physnet", "phase1_dimenet"];

function CustomTooltip({ active, payload, label, unit }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <div style={{ fontWeight: 600 }}>{label}</div>
      <div className="chart-tooltip-row">
        <span className="chart-tooltip-swatch" style={{ background: payload[0].color }} />
        <span style={{ marginLeft: "auto", fontVariantNumeric: "tabular-nums" }}>
          {Number(payload[0].value).toFixed(4)} {unit}
        </span>
      </div>
    </div>
  );
}

// Phase 1's five architectures in one colour, Phase 2/3 in another - two
// different questions (best deterministic backbone vs. where should
// stochasticity live) sharing one chart should not look like one group.
function colorForPhase(phase: string): string {
  return PHASE1_KEYS.includes(phase) ? "var(--series-blue)" : "var(--series-orange)";
}

function MetricChart({
  title,
  data,
  dataKey,
  unit,
}: {
  title: string;
  data: BenchmarkResult[];
  dataKey: "energy_mae" | "force_mae";
  unit: string;
}) {
  const chartData = data.map((r) => ({ ...r, label: PHASE_LABELS[r.phase] ?? r.phase }));
  return (
    <div className="card">
      <p className="card-title">{title}</p>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid vertical={false} stroke="var(--gridline)" />
          <XAxis
            dataKey="label"
            tick={{ fill: "var(--ink-muted)", fontSize: 11 }}
            axisLine={{ stroke: "var(--baseline)" }}
            tickLine={false}
            angle={-20}
            textAnchor="end"
            height={50}
          />
          <YAxis tick={{ fill: "var(--ink-muted)", fontSize: 12 }} axisLine={false} tickLine={false} width={44} />
          <Tooltip content={<CustomTooltip unit={unit} />} cursor={{ fill: "var(--gridline)", opacity: 0.4 }} />
          <Bar dataKey={dataKey} radius={[4, 4, 0, 0]} maxBarSize={40}>
            {chartData.map((r) => (
              <Cell key={r.phase} fill={colorForPhase(r.phase)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p style={{ margin: "8px 0 0", fontSize: 12, color: "var(--ink-muted)" }}>
        <span style={{ color: "var(--series-blue)" }}>■</span> Phase 1 bake-off (five architectures, same
        data/loss/budget — which backbone is most accurate) &nbsp;
        <span style={{ color: "var(--series-orange)" }}>■</span> Phase 2/3 (same MPNN backbone — where
        should stochasticity live)
      </p>
    </div>
  );
}

export default function Benchmarks() {
  const [results, setResults] = useState<BenchmarkResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchBenchmarks()
      .then(setResults)
      .catch((err) => setError(err.message));
  }, []);

  return (
    <>
      <div className="page-header">
        <h1>Three-phase benchmark</h1>
        <p>Deterministic MPNN vs. BLIP weight-space UQ vs. graph-space UQ, on held-out MD17 trajectory blocks.</p>
      </div>

      {error && <div className="error-block">{error}</div>}

      {!error && results === null && (
        <div className="card">
          <p style={{ margin: 0, color: "var(--ink-secondary)", fontSize: 14 }}>Loading…</p>
        </div>
      )}

      {!error && results?.length === 0 && (
        <div className="card">
          <div className="empty-state">
            <div className="empty-state-icon">
              <IconChart size={28} />
            </div>
            <p style={{ margin: 0, fontWeight: 500 }}>Benchmarks in progress</p>
            <p style={{ margin: "4px 0 0", fontSize: 13, color: "var(--ink-muted)" }}>
              Results will appear here as each phase's model is trained — tracking energy MAE, force
              MAE, ECE, and uncertainty–error correlation.
            </p>
          </div>
        </div>
      )}

      {!error && results && results.length > 0 && (
        <>
          <div className="card">
            <p className="card-title">What these charts are comparing</p>
            <p style={{ margin: 0, fontSize: 14, lineHeight: 1.6, color: "var(--ink-secondary)" }}>
              Two separate questions share this page. <b>Phase 1</b> (blue) is a bake-off: five
              architectures trained under one identical protocol — same data, same loss, same budget —
              so any difference is architecture, not conditions. <b>Phase 2/3</b> (orange) all extend
              the same MPNN backbone and ask a different question: where should stochasticity live, in
              the weights (BLIP) or the graph representations (Phase 3)? Lower is better on every chart
              below.
            </p>
          </div>

          <MetricChart title="Energy MAE by phase" data={results} dataKey="energy_mae" unit="kcal/mol" />
          <MetricChart title="Force MAE by phase" data={results} dataKey="force_mae" unit="kcal/mol/Å" />

          <div className="card">
            <p className="card-title">Uncertainty quantification</p>
            <p style={{ margin: "0 0 10px", fontSize: 13, color: "var(--ink-secondary)" }}>
              Phase 1 has no uncertainty by construction (deterministic — one forward pass, no sampled
              distribution to compute a confidence interval from), so it has no rows here. ECE closer to
              0 is better calibrated; Spearman closer to 1 means uncertainty actually tracks error.
            </p>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Phase</th>
                    <th>ECE</th>
                    <th>Uncertainty–error correlation</th>
                  </tr>
                </thead>
                <tbody>
                  {results
                    .filter((r) => r.ece !== null || r.uncertainty_correlation !== null)
                    .map((r) => (
                      <tr key={r.phase}>
                        <td>{PHASE_LABELS[r.phase] ?? r.phase}</td>
                        <td>{r.ece?.toFixed(4) ?? "—"}</td>
                        <td>{r.uncertainty_correlation?.toFixed(4) ?? "—"}</td>
                      </tr>
                    ))}
                  {results.every((r) => r.ece === null && r.uncertainty_correlation === null) && (
                    <tr>
                      <td colSpan={3} style={{ color: "var(--ink-muted)" }}>
                        No stochastic phase (2 or 3) results yet.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card">
            <p className="card-title">Full metrics — every number ml/utils/uq_metrics.py checks</p>
            <p style={{ margin: "0 0 10px", fontSize: 12, color: "var(--ink-muted)" }}>
              Coverage@90% shows which direction a model is miscalibrated in (below 0.90 =
              overconfident, above = overcautious) — ECE alone collapses that away. Sharpness is the
              model's own typical predicted uncertainty, for reading coverage against how wide its
              intervals actually are. NLL/CRPS are proper scoring rules — lower is better, unlike ECE/
              Spearman.
            </p>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Phase</th>
                    <th>Model</th>
                    <th>Energy MAE</th>
                    <th>Energy RMSE</th>
                    <th>Force MAE</th>
                    <th>Force RMSE</th>
                    <th>Energy ECE</th>
                    <th>Force ECE</th>
                    <th>Energy Spearman</th>
                    <th>Force Spearman</th>
                    <th>Energy cov@90%</th>
                    <th>Force cov@90%</th>
                    <th>Energy sharpness</th>
                    <th>Force sharpness</th>
                    <th>Energy NLL</th>
                    <th>Force NLL</th>
                    <th>Energy CRPS</th>
                    <th>Force CRPS</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((r) => (
                    <tr key={r.phase}>
                      <td>{PHASE_LABELS[r.phase] ?? r.phase}</td>
                      <td>{r.model}</td>
                      <td>{r.energy_mae.toFixed(4)}</td>
                      <td>{r.energy_rmse?.toFixed(4) ?? "—"}</td>
                      <td>{r.force_mae.toFixed(4)}</td>
                      <td>{r.force_rmse?.toFixed(4) ?? "—"}</td>
                      <td>{r.ece?.toFixed(4) ?? "—"}</td>
                      <td>{r.force_ece?.toFixed(4) ?? "—"}</td>
                      <td>{r.uncertainty_correlation?.toFixed(4) ?? "—"}</td>
                      <td>{r.force_uncertainty_correlation?.toFixed(4) ?? "—"}</td>
                      <td>{r.energy_coverage_90 !== null ? `${(r.energy_coverage_90 * 100).toFixed(1)}%` : "—"}</td>
                      <td>{r.force_coverage_90 !== null ? `${(r.force_coverage_90 * 100).toFixed(1)}%` : "—"}</td>
                      <td>{r.energy_sharpness?.toFixed(4) ?? "—"}</td>
                      <td>{r.force_sharpness?.toFixed(4) ?? "—"}</td>
                      <td>{r.energy_nll_gaussian?.toFixed(3) ?? "—"}</td>
                      <td>{r.force_nll_gaussian?.toFixed(3) ?? "—"}</td>
                      <td>{r.energy_crps?.toFixed(4) ?? "—"}</td>
                      <td>{r.force_crps?.toFixed(4) ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {results.some((r) => r.energy_ece_calibrated !== null) && (
            <div className="card">
              <p className="card-title">Post-hoc calibration (SCRUM-68)</p>
              <p style={{ margin: "0 0 10px", fontSize: 13, color: "var(--ink-secondary)" }}>
                A single conformal scale factor, fit offline on the held-out validation split
                (<code>ml/training/calibrate.py</code>), multiplied onto every predicted std at test
                time — no retraining. Shown as before/after rather than replacing the raw numbers
                above, since this is a correction on top of each model's own uncertainty, not a
                different model.
              </p>
              <div className="table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Phase</th>
                      <th>Energy ECE (raw → calibrated)</th>
                      <th>Force ECE (raw → calibrated)</th>
                      <th>Energy cov@90% (raw → calibrated)</th>
                      <th>Force cov@90% (raw → calibrated)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results
                      .filter((r) => r.energy_ece_calibrated !== null)
                      .map((r) => (
                        <tr key={r.phase}>
                          <td>{PHASE_LABELS[r.phase] ?? r.phase}</td>
                          <td>
                            {r.ece?.toFixed(4) ?? "—"} → {r.energy_ece_calibrated?.toFixed(4) ?? "—"}
                          </td>
                          <td>
                            {r.force_ece?.toFixed(4) ?? "—"} → {r.force_ece_calibrated?.toFixed(4) ?? "—"}
                          </td>
                          <td>
                            {r.energy_coverage_90 !== null ? `${(r.energy_coverage_90 * 100).toFixed(1)}%` : "—"} →{" "}
                            {r.energy_coverage_90_calibrated !== null
                              ? `${(r.energy_coverage_90_calibrated * 100).toFixed(1)}%`
                              : "—"}
                          </td>
                          <td>
                            {r.force_coverage_90 !== null ? `${(r.force_coverage_90 * 100).toFixed(1)}%` : "—"} →{" "}
                            {r.force_coverage_90_calibrated !== null
                              ? `${(r.force_coverage_90_calibrated * 100).toFixed(1)}%`
                              : "—"}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
              <p style={{ margin: "10px 0 0", fontSize: 12, color: "var(--ink-muted)" }}>
                Coverage lands near 98–100% rather than exactly 90% after calibration — plausibly
                because MD17's contiguous-trajectory-block splits (required to avoid leaking
                near-duplicate frames, see CLAUDE.md) violate the exchangeability assumption
                conformal prediction relies on between the validation calibration set and test set.
                Reported honestly rather than claimed as a perfect fix — see SCRUM-68.
              </p>
            </div>
          )}
        </>
      )}
    </>
  );
}
