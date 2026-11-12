# Cloud browser benchmark — 2026-11-12

<badge>Automated browser inference passed</badge>

## Execution

- Repository: `lazysoftwaredeveloper/online-image-process`
- Branch: `feat/transformersjs-playground`
- Tested commit: `d8c2742bcc872f7afe9b0648f850b5c2ffac208c`
- [GitHub Actions Run #18 — push](https://github.com/lazysoftwaredeveloper/online-image-process/actions/runs/43129029999)
- [GitHub Actions Run #19 — pull request](https://github.com/lazysoftwaredeveloper/online-image-process/actions/runs/43129038291)
- [Download JSON and screenshot artifact](https://github.com/lazysoftwaredeveloper/online-image-process/actions/runs/43129029999/artifacts/13905060365)

The Playwright test launched a fresh headless Chromium context, selected **Text embeddings**, forced the **WASM** backend, loaded the actual model from Hugging Face Hub, and executed one first inference plus five warm inferences. This is a real browser test; no pipeline or model inference was mocked.

## Environment

| Field | Value |
|---|---|
| Runner | GitHub-hosted Ubuntu 24.04 |
| Browser | Headless Chromium 141.0.7390.37 |
| Platform | Linux x86_64 |
| Hardware concurrency | 4 |
| Reported device memory | 8 GB |
| Cross-origin isolated | false |
| Task | `feature-extraction` |
| Model | `onnx-community/all-MiniLM-L6-v2-ONNX` |
| Backend | WASM |
| Dtype | auto |

## Measurements

| Metric | Result |
|---|---:|
| Model load + runtime setup | **3,411 ms** |
| Observed file-transfer bytes | **90,854,365 bytes / 86.65 MiB** |
| Progress events | 538 |
| Transfer event window | 1,760.9 ms |
| First inference | **124.5 ms** |
| Warm run 1 | 58.7 ms |
| Warm run 2 | 60.9 ms |
| Warm run 3 | 61.4 ms |
| Warm run 4 | 75.6 ms |
| Warm run 5 | 27.1 ms |
| Warm median | **60.9 ms** |
| Inference output | Tensor shape `[1, 384]` |
| Uncaught browser errors | 0 |
| Failed network requests | 0 |

## Interpretation

1. The model loaded and completed inference successfully in a cloud-hosted Chromium browser using WASM.
2. The first inference was about 2.04× the warm median, consistent with first-run setup/warm-up work. This is one run, not a statistically robust performance study.
3. The reported 86.65 MiB is the sum of observed progress-callback file bytes. It is not guaranteed to equal exact network wire bytes.
4. The 1.76-second progress event window is the interval between observed progress events, not total end-to-end download time.
5. The 3.411-second model load metric includes model initialization and runtime/session setup; it must not be interpreted as pure download time.

## Important limits

- These measurements describe a GitHub-hosted Linux CPU/WASM environment, not Apple Silicon or WebGPU.
- They are not directly comparable to a local Mac or another browser/device.
- Only one model/task combination has been measured in this first cloud run.
- The five warm samples are useful as an initial baseline but are too few to establish a stable cross-machine benchmark.
- A fresh Playwright context was used, but browser/runtime cache behavior can still vary across environments and runner images.

## Next tests

- Add a small representative text, vision, and audio model matrix.
- Run browser-specific tests on Chromium, Firefox, and WebKit where ONNX Runtime support permits.
- Collect local Apple Silicon/WebGPU measurements separately and compare only equivalent model, dtype, input, and browser conditions.
- Repeat each configuration across multiple fresh contexts and report median and percentile ranges.
