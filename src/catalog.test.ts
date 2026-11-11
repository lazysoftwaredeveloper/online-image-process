import { describe, expect, it } from 'vitest';
import { ADVANCED_APIS, TASKS, UNSUPPORTED_TASKS } from './catalog';

describe('capability catalog', () => {
  it('covers all 25 pipeline IDs plus the sentence-similarity task demo', () => {
    expect(TASKS).toHaveLength(26);
    expect(new Set(TASKS.map((task) => task.id)).size).toBe(26);
    expect(new Set(TASKS.map((task) => task.task)).size).toBe(25);
    expect(TASKS.find((task) => task.id === 'sentence-similarity')?.task).toBe('feature-extraction');
  });

  it('provides practical scope and limitations for every task', () => {
    for (const task of TASKS) {
      expect(task.useCases.length).toBeGreaterThan(0);
      expect(task.scope.length).toBeGreaterThan(20);
      expect(task.limitations.length).toBeGreaterThan(20);
      expect(task.model).toContain('/');
      expect(task.docs.startsWith('https://huggingface.co/docs/transformers.js/')).toBe(true);
    }
  });

  it('documents interactive advanced API families and unsupported task boundaries', () => {
    expect(ADVANCED_APIS).toHaveLength(14);
    expect(new Set(ADVANCED_APIS.map((api) => api.id)).size).toBe(14);
    expect(UNSUPPORTED_TASKS.length).toBeGreaterThanOrEqual(8);
  });
});
