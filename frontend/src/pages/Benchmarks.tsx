import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { IconChart } from "../components/icons";
import { BenchmarkResult, fetchBenchmarks } from "../api/client";

const PHASE_LABELS: Record<string, string> = {
  phase1_deterministic: "Phase 1",
  phase2_blip: "Phase 2 (BLIP)",
  phase3_graph_stochastic: "Phase 3 (graph-space)",
};

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

function MetricChart({
  title,
  data,
  dataKey,
  unit,
  color,
}: {
  title: string;
  data: BenchmarkResult[];
  dataKey: "energy_mae" | "force_mae";
  unit: string;
  color: string;
}) {
  const chartData = data.map((r) => ({ ...r, label: PHASE_LABELS[r.phase] ?? r.phase }));
  return (
    <div className="card">
      <p className="card-title">{title}</p>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid vertical={false} stroke="var(--gridline)" />
          <XAxis
            dataKey="label"
            tick={{ fill: "var(--ink-muted)", fontSize: 12 }}
            axisLine={{ stroke: "var(--baseline)" }}
            tickLine={false}
          />
          <YAxis tick={{ fill: "var(--ink-muted)", fontSize: 12 }} axisLine={false} tickLine={false} width={44} />
          <Tooltip content={<CustomTooltip unit={unit} />} cursor={{ fill: "var(--gridline)", opacity: 0.4 }} />
          <Bar dataKey={dataKey} fill={color} radius={[4, 4, 0, 0]} maxBarSize={40} />
        </BarChart>
      </ResponsiveContainer>
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
          <MetricChart
            title="Energy MAE by phase"
            data={results}
            dataKey="energy_mae"
            unit="kcal/mol"
            color="var(--series-blue)"
          />
          <MetricChart
            title="Force MAE by phase"
            data={results}
            dataKey="force_mae"
            unit="kcal/mol/Å"
            color="var(--series-blue)"
          />

          <div className="card">
            <p className="card-title">Uncertainty quantification</p>
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
            <p className="card-title">All metrics</p>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Phase</th>
                    <th>Model</th>
                    <th>Energy MAE</th>
                    <th>Force MAE</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((r) => (
                    <tr key={r.phase}>
                      <td>{PHASE_LABELS[r.phase] ?? r.phase}</td>
                      <td>{r.model}</td>
                      <td>{r.energy_mae.toFixed(4)}</td>
                      <td>{r.force_mae.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </>
  );
}
