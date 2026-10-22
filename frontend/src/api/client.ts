const BASE_URL = "/api";

export interface PredictionResponse {
  molecule: string;
  frame_index: number;
  predicted_energy: number;
  predicted_force_rms: number;
  uncertainty: number | null;
  phase: string;
  true_energy: number | null;
  true_force_rms: number | null;
}

export interface MoleculeInfo {
  molecule: string;
  num_atoms: number;
}

export interface BenchmarkResult {
  phase: string;
  model: string;
  energy_mae: number;
  force_mae: number;
  ece: number | null;
  uncertainty_correlation: number | null;
  energy_rmse: number | null;
  force_rmse: number | null;
  force_ece: number | null;
  force_uncertainty_correlation: number | null;
  energy_coverage_90: number | null;
  force_coverage_90: number | null;
  energy_sharpness: number | null;
  force_sharpness: number | null;
  energy_nll_gaussian: number | null;
  force_nll_gaussian: number | null;
  energy_crps: number | null;
  force_crps: number | null;
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    // FastAPI's error body is {"detail": "..."} - that message is the
    // whole point (e.g. "no trained checkpoint for 'aspirin' yet"), so
    // surface it instead of just the status code.
    let detail = `${response.status}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      // body wasn't JSON (or was empty) - fall back to the status code above.
    }
    throw new Error(detail);
  }
  return response.json();
}

export function predict(
  molecule: string,
  frameIndex: number,
  phase?: string,
): Promise<PredictionResponse> {
  return request<PredictionResponse>("/predictions", {
    method: "POST",
    body: JSON.stringify({ molecule, frame_index: frameIndex, phase: phase ?? null }),
  });
}

export function fetchMolecules(): Promise<MoleculeInfo[]> {
  return request<MoleculeInfo[]>("/molecules");
}

export function fetchBenchmarks(): Promise<BenchmarkResult[]> {
  return request<BenchmarkResult[]>("/benchmarks");
}

export interface TrainJobStatus {
  id: string;
  molecule: string;
  theory: string;
  phase: string;
  config_name: string;
  started_at: string;
  status: "running" | "complete" | "failed";
  epochs_run: number | null;
  max_epochs: number | null;
  best_epoch: number | null;
  best_val_loss: number | null;
  last_val_loss: number | null;
  stop_reason: string | null;
  checkpoint_ready: boolean;
  log_tail: string;
}

export interface DataUploadResponse {
  molecule: string;
  theory: string;
  gold_path: string;
  n_train: number;
  n_val: number;
  n_test: number;
  used_literature_split: boolean;
}

export function fetchJobPhases(): Promise<string[]> {
  return request<string[]>("/jobs/phases");
}

export function fetchJobs(): Promise<TrainJobStatus[]> {
  return request<TrainJobStatus[]>("/jobs");
}

export function fetchJob(jobId: string): Promise<TrainJobStatus> {
  return request<TrainJobStatus>(`/jobs/${jobId}`);
}

export function startTrainingJob(
  molecule: string,
  phase: string,
  theory?: string,
  epochs?: number,
): Promise<TrainJobStatus> {
  return request<TrainJobStatus>("/jobs/train", {
    method: "POST",
    body: JSON.stringify({ molecule, phase, theory: theory ?? null, epochs: epochs ?? null }),
  });
}

// Not routed through request(): a file upload is multipart/form-data, and
// the browser must set that header itself (with the correct boundary) -
// request() always sets Content-Type: application/json, which would
// corrupt this one.
export async function uploadData(
  file: File,
  molecule?: string,
  theory?: string,
): Promise<DataUploadResponse> {
  const form = new FormData();
  form.append("file", file);
  if (molecule) form.append("molecule", molecule);
  if (theory) form.append("theory", theory);

  const response = await fetch(`${BASE_URL}/data/upload`, { method: "POST", body: form });
  if (!response.ok) {
    let detail = `${response.status}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      // not JSON - fall back to the status code
    }
    throw new Error(detail);
  }
  return response.json();
}
