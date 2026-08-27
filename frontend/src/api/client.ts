const BASE_URL = "/api";

export interface PredictionResponse {
  molecule: string;
  frame_index: number;
  predicted_energy: number;
  predicted_force_rms: number;
  uncertainty: number | null;
  phase: string;
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
    throw new Error(`Request to ${path} failed: ${response.status}`);
  }
  return response.json();
}

export function predict(molecule: string, frameIndex: number): Promise<PredictionResponse> {
  return request<PredictionResponse>("/predictions", {
    method: "POST",
    body: JSON.stringify({ molecule, frame_index: frameIndex }),
  });
}

export function fetchMolecules(): Promise<MoleculeInfo[]> {
  return request<MoleculeInfo[]>("/molecules");
}

export function fetchBenchmarks(): Promise<BenchmarkResult[]> {
  return request<BenchmarkResult[]>("/benchmarks");
}
