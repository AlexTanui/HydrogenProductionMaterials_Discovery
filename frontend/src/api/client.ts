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
