import { expect, test } from '@playwright/test';
import { writeFile } from 'node:fs/promises';

const STORAGE_KEY = 'transformers-js-playground-benchmarks-v1';

test('loads a real model and records cold-load plus five warm inferences in Chromium/WASM', async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  const failedRequests: Array<{ url: string; error: string | null }> = [];

  page.on('pageerror', (error) => pageErrors.push(error.message));
  page.on('requestfailed', (request) => {
    failedRequests.push({
      url: request.url(),
      error: request.failure()?.errorText ?? null,
    });
  });

  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Explore tasks' })).toBeVisible();

  // Use a compact, real Hub model so this is an actual browser inference test,
  // not a mocked pipeline or a unit-test substitute.
  await page.getByRole('button', { name: 'Text', exact: true }).click();
  await page.getByRole('button', { name: /Text embeddings/ }).click();
  await page.getByLabel('Backend').selectOption('wasm');
  await page.locator('.run-actions').getByRole('button', { name: /Benchmark/ }).click();

  await expect(page.getByText('Complete', { exact: true })).toBeVisible({ timeout: 11 * 60 * 1000 });
  const output = await page.locator('.json-output').textContent();
  expect(output).toContain('dims');

  const result = await page.evaluate((storageKey) => {
    const raw = localStorage.getItem(storageKey);
    const records = raw ? JSON.parse(raw) : [];
    const record = records.find((item: any) =>
      item.taskId === 'feature-extraction' &&
      item.model === 'onnx-community/all-MiniLM-L6-v2-ONNX' &&
      item.backend === 'wasm'
    );
    return {
      record: record ?? null,
      browser: {
        userAgent: navigator.userAgent,
        platform: navigator.platform,
        language: navigator.language,
        hardwareConcurrency: navigator.hardwareConcurrency ?? null,
        deviceMemory: (navigator as any).deviceMemory ?? null,
        crossOriginIsolated: window.crossOriginIsolated,
      },
      collectedAt: new Date().toISOString(),
    };
  }, STORAGE_KEY);

  expect(result.record).toBeTruthy();
  expect(result.record.loadMs).toBeGreaterThan(0);
  expect(result.record.firstInferenceMs).toBeGreaterThan(0);
  expect(result.record.warmInferenceMs).toHaveLength(5);
  expect(result.record.warmInferenceMs.every((value: number) => value > 0)).toBeTruthy();
  expect(result.record.medianWarmMs).toBeGreaterThan(0);
  expect(result.record.loadCount).toBe(1);
  expect(result.record.transferBytes).toBeGreaterThanOrEqual(0);
  expect(pageErrors).toEqual([]);

  const report = {
    schemaVersion: 1,
    suite: 'real-browser-model-inference',
    environment: result.browser,
    collectedAt: result.collectedAt,
    model: {
      taskId: result.record.taskId,
      taskName: result.record.taskName,
      repository: result.record.model,
      backend: result.record.backend,
      dtype: result.record.dtype,
    },
    measurements: {
      modelLoadAndSetupMs: result.record.loadMs,
      observedTransferBytes: result.record.transferBytes,
      progressEvents: result.record.transferEvents,
      transferEventWindowMs: result.record.transferWindowMs,
      firstInferenceMs: result.record.firstInferenceMs,
      warmInferenceMs: result.record.warmInferenceMs,
      warmMedianMs: result.record.medianWarmMs,
      modelInitializations: result.record.loadCount,
    },
    pageErrors,
    failedRequests,
    notes: [
      'Fresh Playwright Chromium context; model artifacts were not pre-cached in this test context.',
      'Model load + setup includes network transfer, parsing, and ONNX Runtime session initialization.',
      'Transfer bytes are progress-callback telemetry and may not equal exact wire bytes.',
      'WASM timings describe this GitHub-hosted runner and are not representative of Apple Silicon/WebGPU.',
    ],
  };

  await writeFile(testInfo.outputPath('cloud-browser-benchmark.json'), JSON.stringify(report, null, 2));
  await page.screenshot({ path: testInfo.outputPath('cloud-browser-benchmark.png'), fullPage: true });
});
