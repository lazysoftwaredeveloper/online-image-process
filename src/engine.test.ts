import { describe, expect, it } from 'vitest';
import { runRandomLab, runTensorLab } from './engine';

describe('public Transformers.js utility APIs', () => {
  it('runs tensor operations and returns shape-aware results', () => {
    const result = runTensorLab();
    expect(result.tensor.dims).toEqual([2, 3]);
    expect(result.tensorMath.permutedShape).toEqual([3, 2]);
    expect(result.tensorMath.interpolatedShape).toEqual([1, 4, 4]);
    expect(result.tensorUtilities.quantizedEmbedding).toBeDefined();
  });

  it('reproduces seeded random sequences', () => {
    const first = runRandomLab(42);
    const second = runRandomLab(42);
    expect(first.sequence).toEqual(second.sequence);
    expect(first.shuffled).toEqual(second.shuffled);
    expect(first.weightedChoice).toEqual(second.weightedChoice);
  });
});
