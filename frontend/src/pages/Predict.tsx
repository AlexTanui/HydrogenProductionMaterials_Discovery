import { FormEvent, useEffect, useState } from "react";
import { fetchMolecules, MoleculeInfo, predict, PredictionResponse } from "../api/client";

const PHASE_LABELS: Record<string, string> = {
  phase1_deterministic: "Phase 1 — deterministic MPNN",
  phase2_blip_k1: "Phase 2 — BLIP (K=1, MSE)",
  phase2_blip: "Phase 2 — BLIP (K=4, NLL fix)",
  phase3_graph_stochastic: "Phase 3 — graph-space UQ",
};

const PHASE_OPTIONS = Object.entries(PHASE_LABELS);

// Only ethanol has a trained checkpoint right now (backend/app/services/
// inference.py) - every other molecule in the catalog has real MD17 data
// but no model trained on it, and the API refuses to guess.
const TRAINED_MOLECULE = "ethanol";

export default function Predict() {
  const [molecules, setMolecules] = useState<MoleculeInfo[]>([]);
  const [molecule, setMolecule] = useState("");
  const [frameIndex, setFrameIndex] = useState(0);
  const [phase, setPhase] = useState<string>("phase1_deterministic");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [comparison, setComparison] = useState<Record<string, PredictionResponse | string> | null>(null);
  const [comparing, setComparing] = useState(false);

  useEffect(() => {
    fetchMolecules()
      .then((list) => {
        setMolecules(list);
        // Default to the one molecule that actually has a trained model,
        // if it's in the catalog, rather than whichever happens to sort first.
        const preferred = list.find((m) => m.molecule === TRAINED_MOLECULE);
        if (preferred) setMolecule(preferred.molecule);
        else if (list.length > 0) setMolecule(list[0].molecule);
      })
      .catch((err) => setError(err.message));
  }, []);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!molecule) return;
    setLoading(true);
    setError(null);
    setComparison(null);
    try {
      setResult(await predict(molecule, frameIndex, phase));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  // The actual research question ("where should uncertainty live?") on one
  // frame, side by side: same molecule, same frame, three phases, three
  // real checkpoints - not a canned comparison, a live one.
  async function handleCompare() {
    if (!molecule) return;
    setComparing(true);
    setError(null);
    setResult(null);
    const phases = Object.keys(PHASE_LABELS);
    const entries = await Promise.all(
      phases.map(async (p) => {
        try {
          return [p, await predict(molecule, frameIndex, p)] as const;
        } catch (err) {
          return [p, (err as Error).message] as const;
        }
      }),
    );
    setComparison(Object.fromEntries(entries));
    setComparing(false);
  }

  return (
    <>
      <div className="page-header">
        <h1>Predict energy &amp; forces</h1>
        <p>
          Pick a trajectory frame from the held-out MD17 test split and a trained phase to predict
          total potential energy and force magnitude against. Only <code>ethanol</code> (DFT) has a
          trained checkpoint right now — every other molecule below has real MD17 data but no model
          trained on it yet, so predicting on it is refused rather than guessed.
        </p>
      </div>

      <div className="card">
        <form onSubmit={handleSubmit}>
          <div className="field-row">
            <select
              className="input"
              value={molecule}
              onChange={(e) => setMolecule(e.target.value)}
              style={{ flex: 2 }}
            >
              {molecules.map((m) => (
                <option key={m.molecule} value={m.molecule}>
                  {m.molecule} ({m.num_atoms} atoms){m.molecule !== TRAINED_MOLECULE ? " — not trained" : ""}
                </option>
              ))}
            </select>
            <input
              className="input"
              type="number"
              min={0}
              value={frameIndex}
              onChange={(e) => setFrameIndex(Number(e.target.value))}
              placeholder="Frame index"
              style={{ flex: 1 }}
            />
            <select
              className="input"
              value={phase}
              onChange={(e) => setPhase(e.target.value)}
              style={{ flex: 2 }}
            >
              {PHASE_OPTIONS.map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
            <button className="btn btn-primary" type="submit" disabled={loading || comparing || !molecule}>
              {loading ? "Predicting…" : "Predict"}
            </button>
          </div>
        </form>

        <button
          className="btn"
          type="button"
          onClick={handleCompare}
          disabled={loading || comparing || !molecule}
          style={{ marginTop: 10 }}
        >
          {comparing ? "Comparing all 3 phases…" : "Compare all 3 phases on this frame"}
        </button>

        {error && <div className="error-block">{error}</div>}

        {result && (
          <div className="result-block">
            <p className="stat-label" style={{ margin: 0 }}>
              <code>{result.molecule}</code>, frame {result.frame_index}
            </p>

            <div style={{ display: "flex", gap: 32, marginTop: 10, flexWrap: "wrap" }}>
              <div>
                <p className="stat-label" style={{ margin: "0 0 2px" }}>
                  Predicted energy
                </p>
                <p className="result-value" style={{ fontSize: 20, margin: 0 }}>
                  {result.predicted_energy.toFixed(2)} <span style={{ fontSize: 13, color: "var(--ink-muted)" }}>kcal/mol</span>
                </p>
                {result.true_energy !== null && (
                  <p className="stat-label" style={{ margin: "2px 0 0", fontSize: 12 }}>
                    real DFT value: {result.true_energy.toFixed(2)} (off by{" "}
                    {Math.abs(result.predicted_energy - result.true_energy).toFixed(4)})
                  </p>
                )}
              </div>
              <div>
                <p className="stat-label" style={{ margin: "0 0 2px" }}>
                  Force RMS
                </p>
                <p className="result-value" style={{ fontSize: 20, margin: 0 }}>
                  {result.predicted_force_rms.toFixed(2)} <span style={{ fontSize: 13, color: "var(--ink-muted)" }}>kcal/mol/Å</span>
                </p>
                {result.true_force_rms !== null && (
                  <p className="stat-label" style={{ margin: "2px 0 0", fontSize: 12 }}>
                    real DFT value: {result.true_force_rms.toFixed(2)} (off by{" "}
                    {Math.abs(result.predicted_force_rms - result.true_force_rms).toFixed(4)})
                  </p>
                )}
              </div>
              <div>
                <p className="stat-label" style={{ margin: "0 0 2px" }}>
                  Uncertainty
                </p>
                <p className="result-value" style={{ fontSize: 20, margin: 0 }}>
                  {result.uncertainty === null ? "—" : result.uncertainty.toFixed(4)}
                </p>
                {result.uncertainty === null && (
                  <p className="stat-label" style={{ margin: "2px 0 0", fontSize: 12 }}>
                    deterministic — no uncertainty by construction
                  </p>
                )}
              </div>
            </div>

            <span className="status-pill" style={{ marginTop: 12 }}>
              <span className="badge-dot" />
              {PHASE_LABELS[result.phase] ?? result.phase}
            </span>
          </div>
        )}
      </div>

      {comparison && (
        <div className="card">
          <p className="card-title">
            The research question, live: same frame, three phases
          </p>
          <p style={{ margin: "0 0 10px", fontSize: 13, color: "var(--ink-secondary)" }}>
            <code>{molecule}</code>, frame {frameIndex} — same structure, scored by three separately
            trained checkpoints that differ only in where stochasticity lives.
          </p>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Phase</th>
                  <th>Predicted energy</th>
                  <th>Error vs. real DFT</th>
                  <th>Uncertainty</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(PHASE_LABELS).map(([phaseKey, label]) => {
                  const entry = comparison[phaseKey];
                  if (typeof entry === "string") {
                    return (
                      <tr key={phaseKey}>
                        <td>{label}</td>
                        <td colSpan={3} style={{ color: "var(--status-critical)" }}>
                          {entry}
                        </td>
                      </tr>
                    );
                  }
                  const errorAbs =
                    entry.true_energy !== null ? Math.abs(entry.predicted_energy - entry.true_energy) : null;
                  return (
                    <tr key={phaseKey}>
                      <td>{label}</td>
                      <td>{entry.predicted_energy.toFixed(2)} kcal/mol</td>
                      <td>{errorAbs !== null ? errorAbs.toFixed(4) : "—"}</td>
                      <td>{entry.uncertainty === null ? "— (deterministic)" : entry.uncertainty.toFixed(4)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p style={{ margin: "10px 0 0", fontSize: 12, color: "var(--ink-muted)" }}>
            Phase 2 and 3 both report real uncertainty — but neither is well-calibrated yet: Phase 2
            tends toward overconfidence, Phase 3's first draft toward the opposite (honestly uncertain
            to the point of being uninformative). See Benchmarks for the full picture.
          </p>
        </div>
      )}
    </>
  );
}
