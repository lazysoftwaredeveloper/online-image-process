import { describe, expect, it } from 'vitest';
import { ADVANCED_APIS, TASKS, UNSUPPORTED_TASKS } from './catalog';

describe('capability catalog', () => {
  it('includes all 25 documented pipeline task IDs exactly once', () => {
    expect(TASKS).toHaveLength(25);
    expect(new Set(TASKS.map((task) => task.task)).size).toBe(25);
  });

  it('provides practical scope and limitations for every task', () => {
    for (const task of TASKS) {
      expect(task.useCases.length).toBeGreaterThan(0);
      expect(task.scope.length).toBeGreaterThan(20);
      expect(task.limitations.length).toBeGreaterThan(20);
      expect(task.model).toContain('/');
    }
  });

  it('documents the advanced API families and unsupported task boundaries', () => {
    expect(ADVANCED_APIS.length).toBeGreaterThanOrEqual(8);
    expect(UNSUPPORTED_TASKS.length).toBeGreaterThanOrEqual(8);
  });
});
