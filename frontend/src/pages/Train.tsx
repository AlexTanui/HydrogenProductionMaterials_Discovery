import { Fragment, FormEvent, useEffect, useRef, useState } from "react";
import {
  DataUploadResponse,
  MoleculeInfo,
  TrainJobStatus,
  fetchJobPhases,
  fetchJobs,
  fetchMolecules,
  startTrainingJob,
  uploadData,
} from "../api/client";

const PHASE_LABELS: Record<string, string> = {
  mpnn: "MPNN",
  schnet: "SchNet",
  painn: "PaiNN",
  physnet: "PhysNet",
  dimenet: "DimeNet++",
  blip: "BLIP (Phase 2)",
  graph_stochastic: "GraphStochastic (Phase 3)",
};

const STATUS_COLOR: Record<string, string> = {
  running: "var(--series-orange)",
  complete: "var(--status-good)",
  failed: "var(--status-critical)",
};

export default function Train() {
  const [molecules, setMolecules] = useState<MoleculeInfo[]>([]);
  const [phases, setPhases] = useState<string[]>([]);
  const [jobs, setJobs] = useState<TrainJobStatus[]>([]);
  const [expandedLog, setExpandedLog] = useState<string | null>(null);

  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadMolecule, setUploadMolecule] = useState("");
  const [uploadTheory, setUploadTheory] = useState("");
  const [uploadResult, setUploadResult] = useState<DataUploadResponse | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const [trainMolecule, setTrainMolecule] = useState("");
  const [trainPhase, setTrainPhase] = useState("");
  const [trainEpochs, setTrainEpochs] = useState<string>("");
  const [trainError, setTrainError] = useState<string | null>(null);
  const [launching, setLaunching] = useState(false);

  const pollRef = useRef<number | null>(null);

  async function refresh() {
    try {
      const [m, p, j] = await Promise.all([fetchMolecules(), fetchJobPhases(), fetchJobs()]);
      setMolecules(m);
      setPhases(p);
      setJobs(j);
      if (!trainMolecule && m.length > 0) setTrainMolecule(m[0].molecule);
      if (!trainPhase && p.length > 0) setTrainPhase(p[0]);
    } catch {
      // Transient - the next poll tick will retry. Don't blank the page
      // over one missed request.
    }
  }

  useEffect(() => {
    refresh();
    pollRef.current = window.setInterval(refresh, 3000);
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleUpload(e: FormEvent) {
    e.preventDefault();
    if (!uploadFile) return;
    setUploading(true);
    setUploadError(null);
    setUploadResult(null);
    try {
      const result = await uploadData(uploadFile, uploadMolecule || undefined, uploadTheory || undefined);
      setUploadResult(result);
      await refresh(); // the new molecule should now be in the catalog
    } catch (err) {
      setUploadError((err as Error).message);
    } finally {
      setUploading(false);
    }
  }

  async function handleLaunch(e: FormEvent) {
    e.preventDefault();
    if (!trainMolecule || !trainPhase) return;
    setLaunching(true);
    setTrainError(null);
    try {
      const epochs = trainEpochs.trim() ? Number(trainEpochs) : undefined;
      await startTrainingJob(trainMolecule, trainPhase, undefined, epochs);
      await refresh();
    } catch (err) {
      setTrainError((err as Error).message);
    } finally {
      setLaunching(false);
    }
  }

  return (
    <>
      <div className="page-header">
        <h1>Train</h1>
        <p>
          Launch real training on any molecule in the catalog below, or add a new one first. Training
          runs as a real background process — this page shows its actual progress, not a simulation.
        </p>
      </div>

      <div className="card">
        <p className="card-title">Add new data</p>
        <p style={{ margin: "0 0 10px", fontSize: 13, color: "var(--ink-secondary)" }}>
          Upload a raw MD17-format source (<code>.npz</code> for a single split, <code>.zip</code> for a
          shipped literature train/test split) — it runs through the real bronze → silver → gold
          pipeline (<code>ml/data/preprocessing.py</code>), the same validation and splitting every
          built-in dataset already went through, not a separate path.
        </p>
        <form onSubmit={handleUpload}>
          <div className="field-row">
            <input
              className="input"
              type="file"
              accept=".npz,.zip"
              onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)}
              style={{ flex: 2 }}
            />
            <input
              className="input"
              placeholder="molecule (optional override)"
              value={uploadMolecule}
              onChange={(e) => setUploadMolecule(e.target.value)}
              style={{ flex: 1 }}
            />
            <input
              className="input"
              placeholder="theory (optional override)"
              value={uploadTheory}
              onChange={(e) => setUploadTheory(e.target.value)}
              style={{ flex: 1 }}
            />
            <button className="btn btn-primary" type="submit" disabled={uploading || !uploadFile}>
              {uploading ? "Processing…" : "Upload"}
            </button>
          </div>
        </form>
        {uploadError && <div className="error-block" style={{ marginTop: 10 }}>{uploadError}</div>}
        {uploadResult && (
          <div className="result-block" style={{ marginTop: 10 }}>
            <code>{uploadResult.molecule}</code> ({uploadResult.theory}): {uploadResult.n_train} train /{" "}
            {uploadResult.n_val} val / {uploadResult.n_test} test frames —{" "}
            {uploadResult.used_literature_split ? "preserved the shipped literature split" : "trajectory-block split applied"}.
            Now available below and in Predict/Benchmarks.
          </div>
        )}
      </div>

      <div className="card">
        <p className="card-title">Launch training</p>
        <form onSubmit={handleLaunch}>
          <div className="field-row">
            <select
              className="input"
              value={trainMolecule}
              onChange={(e) => setTrainMolecule(e.target.value)}
              style={{ flex: 2 }}
            >
              {molecules.map((m) => (
                <option key={m.molecule} value={m.molecule}>
                  {m.molecule} ({m.num_atoms} atoms)
                </option>
              ))}
            </select>
            <select
              className="input"
              value={trainPhase}
              onChange={(e) => setTrainPhase(e.target.value)}
              style={{ flex: 2 }}
            >
              {phases.map((p) => (
                <option key={p} value={p}>
                  {PHASE_LABELS[p] ?? p}
                </option>
              ))}
            </select>
            <input
              className="input"
              type="number"
              min={1}
              placeholder="epochs (default 200)"
              value={trainEpochs}
              onChange={(e) => setTrainEpochs(e.target.value)}
              style={{ flex: 1 }}
            />
            <button className="btn btn-primary" type="submit" disabled={launching || !trainMolecule || !trainPhase}>
              {launching ? "Launching…" : "Train"}
            </button>
          </div>
        </form>
        {trainError && <div className="error-block" style={{ marginTop: 10 }}>{trainError}</div>}
      </div>

      <div className="card">
        <p className="card-title">Jobs</p>
        {jobs.length === 0 && (
          <p style={{ margin: 0, fontSize: 13, color: "var(--ink-muted)" }}>
            No jobs launched yet this session.
          </p>
        )}
        {jobs.length > 0 && (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Molecule</th>
                  <th>Phase</th>
                  <th>Status</th>
                  <th>Epoch</th>
                  <th>Best val loss</th>
                  <th>Started</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((j) => (
                  <Fragment key={j.id}>
                    <tr>
                      <td>{j.molecule}</td>
                      <td>{PHASE_LABELS[j.phase] ?? j.phase}</td>
                      <td>
                        <span className="status-pill">
                          <span className="badge-dot" style={{ background: STATUS_COLOR[j.status] }} />
                          {j.status}
                          {j.status === "failed" && !j.checkpoint_ready ? " (no checkpoint written)" : ""}
                        </span>
                      </td>
                      <td>{j.epochs_run ?? "—"}{j.max_epochs ? ` / ${j.max_epochs}` : ""}</td>
                      <td>{j.best_val_loss !== null ? j.best_val_loss.toFixed(4) : "—"}</td>
                      <td style={{ fontSize: 12, color: "var(--ink-muted)" }}>{j.started_at}</td>
                      <td>
                        <button
                          className="btn"
                          type="button"
                          onClick={() => setExpandedLog(expandedLog === j.id ? null : j.id)}
                        >
                          {expandedLog === j.id ? "Hide log" : "Log"}
                        </button>
                      </td>
                    </tr>
                    {expandedLog === j.id && (
                      <tr>
                        <td colSpan={7}>
                          <pre
                            style={{
                              margin: 0,
                              fontSize: 11,
                              whiteSpace: "pre-wrap",
                              color: "var(--ink-secondary)",
                              maxHeight: 220,
                              overflowY: "auto",
                            }}
                          >
                            {j.log_tail || "(no log output yet)"}
                          </pre>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p style={{ margin: "10px 0 0", fontSize: 12, color: "var(--ink-muted)" }}>
          Jobs are tracked in memory for this server process only — a backend restart loses this list,
          though the checkpoint and run record a finished job wrote are real files, unaffected by that.
          Once a job's checkpoint is ready, try it on <b>Predict</b> by selecting its molecule and phase
          <code>job:&lt;id&gt;</code>.
        </p>
      </div>
    </>
  );
}
