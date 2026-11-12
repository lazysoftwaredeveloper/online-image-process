import { beforeEach, describe, expect, it } from 'vitest';
import { median, upsertBenchmark, type BenchmarkRecord } from './benchmark';

const baseInput = {
  key: 'text-classification|Xenova/distilbert-base-uncased-finetuned-sst-2-english|wasm|q8',
  taskId: 'text-classification',
  taskName: 'Text classification',
  model: 'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
  backend: 'wasm',
  dtype: 'q8',
  loadMs: 1250,
  transferBytes: 4_194_304,
  transferEvents: 8,
  transferWindowMs: 940,
  userAgent: 'Vitest',
  hardwareConcurrency: 8,
  deviceMemory: 8,
};

describe('browser benchmark measurements', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('calculates odd and even medians', () => {
    expect(median([100, 20, 60])).toBe(60);
    expect(median([100, 20])).toBe(60);
    expect(median([])).toBeNull();
  });

  it('records first-load and first-inference metrics separately', () => {
    const records = upsertBenchmark([], {
      ...baseInput,
      inferenceMs: 340,
      newLoad: true,
    });

    expect(records).toHaveLength(1);
    expect(records[0].loadMs).toBe(1250);
    expect(records[0].firstInferenceMs).toBe(340);
    expect(records[0].warmInferenceMs).toEqual([]);
    expect(records[0].loadCount).toBe(1);
    expect(JSON.parse(localStorage.getItem('transformers-js-playground-benchmarks-v1') ?? '[]')).toHaveLength(1);
  });

  it('retains first-load telemetry and adds subsequent warm-run timings', () => {
    const first = upsertBenchmark([], {
      ...baseInput,
      inferenceMs: 340,
      newLoad: true,
    });
    const second = upsertBenchmark(first, {
      ...baseInput,
      loadMs: 0,
      transferBytes: 0,
      transferEvents: 0,
      transferWindowMs: null,
      inferenceMs: 80,
      newLoad: false,
    });
    const third = upsertBenchmark(second, {
      ...baseInput,
      loadMs: 0,
      transferBytes: 0,
      transferEvents: 0,
      transferWindowMs: null,
      inferenceMs: 100,
      newLoad: false,
    });

    expect(third[0].loadMs).toBe(1250);
    expect(third[0].transferBytes).toBe(4_194_304);
    expect(third[0].firstInferenceMs).toBe(340);
    expect(third[0].warmInferenceMs).toEqual([80, 100]);
    expect(third[0].medianWarmMs).toBe(90);
    expect(third[0].loadCount).toBe(1);
  });
});
