import { FormEvent, useEffect, useState } from "react";
import { fetchMolecules, MoleculeInfo, predict, PredictionResponse } from "../api/client";

const PHASE_LABELS: Record<string, string> = {
  phase1_deterministic: "Phase 1 — deterministic MPNN",
  phase2_blip: "Phase 2 — BLIP (weight-space UQ)",
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
    try {
      setResult(await predict(molecule, frameIndex, phase));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
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
            <button className="btn btn-primary" type="submit" disabled={loading || !molecule}>
              {loading ? "Predicting…" : "Predict"}
            </button>
          </div>
        </form>

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
    </>
  );
}
