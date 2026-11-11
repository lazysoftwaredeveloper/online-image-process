export type BenchmarkRecord = {
  key: string;
  taskId: string;
  taskName: string;
  model: string;
  backend: string;
  dtype: string;
  loadMs: number | null;
  transferBytes: number;
  transferEvents: number;
  transferWindowMs: number | null;
  firstInferenceMs: number | null;
  warmInferenceMs: number[];
  medianWarmMs: number | null;
  loadCount: number;
  updatedAt: string;
  userAgent: string;
  hardwareConcurrency: number | null;
  deviceMemory: number | null;
};

const STORAGE_KEY = 'transformers-js-playground-benchmarks-v1';

export function readBenchmarks(): BenchmarkRecord[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as BenchmarkRecord[]) : [];
  } catch {
    return [];
  }
}

export function median(values: number[]): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function upsertBenchmark(
  current: BenchmarkRecord[],
  incoming: Omit<BenchmarkRecord, 'warmInferenceMs' | 'medianWarmMs' | 'loadCount' | 'updatedAt'> & {
    inferenceMs: number;
    newLoad: boolean;
  },
): BenchmarkRecord[] {
  const existing = current.find((record) => record.key === incoming.key);
  const warmInferenceMs = existing
    ? [...existing.warmInferenceMs, ...(existing.firstInferenceMs === null ? [] : [incoming.inferenceMs])]
    : [];
  const firstInferenceMs = existing?.firstInferenceMs ?? incoming.inferenceMs;
  const loadMs = incoming.newLoad ? incoming.loadMs : existing?.loadMs ?? incoming.loadMs;
  const next: BenchmarkRecord = {
    key: incoming.key,
    taskId: incoming.taskId,
    taskName: incoming.taskName,
    model: incoming.model,
    backend: incoming.backend,
    dtype: incoming.dtype,
    loadMs,
    transferBytes: incoming.newLoad ? incoming.transferBytes : existing?.transferBytes ?? incoming.transferBytes,
    transferEvents: incoming.newLoad ? incoming.transferEvents : existing?.transferEvents ?? incoming.transferEvents,
    transferWindowMs: incoming.newLoad ? incoming.transferWindowMs : existing?.transferWindowMs ?? incoming.transferWindowMs,
    firstInferenceMs,
    warmInferenceMs,
    medianWarmMs: median(warmInferenceMs),
    loadCount: (existing?.loadCount ?? 0) + (incoming.newLoad ? 1 : 0),
    updatedAt: new Date().toISOString(),
    userAgent: incoming.userAgent,
    hardwareConcurrency: incoming.hardwareConcurrency,
    deviceMemory: incoming.deviceMemory,
  };
  const nextRecords = existing
    ? current.map((record) => (record.key === incoming.key ? next : record))
    : [next, ...current];
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(nextRecords));
  } catch {
    // Private browsing or storage quota may disable persistence; the current UI still works.
  }
  return nextRecords;
}

export function clearBenchmarks(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Ignore unavailable storage.
  }
}

export function exportBenchmarks(records: BenchmarkRecord[]): string {
  return JSON.stringify({
    exportedAt: new Date().toISOString(),
    benchmarkMethod: 'Model load + runtime setup; first inference; warm inference; observed transfer telemetry when available.',
    records,
  }, null, 2);
}

export function exportBenchmarksCsv(records: BenchmarkRecord[]): string {
  const headers = [
    'taskId', 'taskName', 'model', 'backend', 'dtype', 'loadMs', 'transferBytes',
    'transferEvents', 'transferWindowMs', 'firstInferenceMs', 'warmInferenceMs',
    'medianWarmMs', 'loadCount', 'updatedAt', 'hardwareConcurrency', 'deviceMemory', 'userAgent',
  ];
  const quote = (value: unknown) => '"' + String(value ?? '').replaceAll('"', '""') + '"';
  const rows = records.map((record) => [
    record.taskId,
    record.taskName,
    record.model,
    record.backend,
    record.dtype,
    record.loadMs,
    record.transferBytes,
    record.transferEvents,
    record.transferWindowMs,
    record.firstInferenceMs,
    record.warmInferenceMs.join('|'),
    record.medianWarmMs,
    record.loadCount,
    record.updatedAt,
    record.hardwareConcurrency,
    record.deviceMemory,
    record.userAgent,
  ].map(quote).join(','));
  return [headers.map(quote).join(','), ...rows].join('\n');
}
