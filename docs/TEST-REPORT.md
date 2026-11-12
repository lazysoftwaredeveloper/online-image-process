# Browser performance test report

## Verification and measurement status

This report separates checks that can be run in CI from real model timings that require a browser/device session.

| Check | Status | Evidence |
|---|---|---|
| Catalog coverage tests | **Passed** | 3 test cases; 26 demo cards, 25 unique first-class pipeline task IDs plus sentence similarity |
| Advanced API coverage tests | **Passed** | 14 advanced API areas verified |
| Benchmark aggregation tests | **Passed** | 3 test cases for first-use, warm-run, median, and local-storage behavior |
| Tensor and seeded random tests | **Passed** | 2 test cases exercise public tensor helpers and deterministic random output |
| TypeScript + production build | **Passed** | `tsc --noEmit` and `vite build` |
| Cold model download timings | Not yet measured on a real browser/device | Must clear browser model cache and run the demos |
| Warm inference timings | Not yet measured on a real browser/device | Must repeat the same input after model initialization |
| Cloud Chromium + WASM model inference | Added to CI | Real Hub download + one first inference + five warm inferences; JSON/screenshot artifacts uploaded by the workflow |
| Cross-browser manual smoke tests | Not yet completed | Run on target browsers/devices |

**Automated CI:** [Run #13 — passed](https://github.com/lazysoftwaredeveloper/online-image-process/actions/runs/43035879458). All 3 test files and 8 tests passed. The production build transformed 1,605 modules and completed in 3.23 seconds.

No device-specific model timings are fabricated in this document. Model sizes, network throughput, cache state, GPU support, memory, browser versions, and ONNX Runtime initialization can change first-use and repeat-use timings substantially.

The CI production build emitted a 962.95 kB JavaScript bundle (285.71 kB gzip), a 31.57 kB CSS bundle (7.67 kB gzip), and a 26,861.78 kB ONNX Runtime WASM asset (6,690.35 kB gzip). The WASM binary is a runtime asset and is typically needed when the WASM backend initializes; it is not an individual model's weight download.

## What the application measures

### Pipeline demos

Each pipeline demo exposes these values directly beside its output:

| Metric | Meaning | Caveat |
|---|---|---|
| Model load + setup | Time from pipeline initialization request until the pipeline is ready | Includes downloads on a cache miss, model parsing, ONNX session creation, and runtime setup; it is not pure network time |
| Observed model transfer | Sum of the largest `loaded` byte count reported for each file in progress callbacks | Only available when the library emits file-level progress events; it is observed telemetry, not guaranteed exact wire bytes |
| Progress events | Number of progress callback events observed | Some cached files or runtime phases may emit no events |
| Transfer event window | Time from the first to last observed progress event | Not guaranteed to equal total network download duration |
| First inference | Time to complete the first inference after initialization | May include first-run kernel/shader compilation and warm-up |
| Warm inference | Time for each subsequent run | Input length, image resolution, audio duration, and generated output length affect the result |
| Warm median | Median of recorded warm runs | Compare only matching task, model, input, backend, dtype, and device |

The **Benchmark · 1 + 5 runs** action records one first inference followed by five repeated runs. Results are stored in the browser's Local Storage and can be exported as JSON or CSV. They are not uploaded to an application server.

### Advanced API demos

The API lab displays total execution time for every demo. Model-backed labs additionally return the relevant initialization and operation timings:

| API demo | First-use metric | Repeat-use metric |
|---|---|---|
| Tokenizer / chat template | Tokenizer load + tokenization | Cached tokenizer + tokenization |
| Image processor | Processor load + preprocessing | Cached processor + preprocessing |
| Audio feature extractor | Feature-extractor load + audio preprocessing | Cached extractor + audio preprocessing |
| Direct AutoModel | Processor/model load, preprocessing, and forward pass | Cached model/processor, preprocessing, and forward pass |
| Text generation | Pipeline load + first generation | Cached pipeline + generation |
| Structured output | Pipeline load + constrained generation | Cached pipeline + constrained generation |
| Multimodal embeddings | Processor/model load + text/media embedding comparison | Cached model + embedding comparison |
| AutoConfig / Hub options | Config metadata request duration | Repeated metadata request duration; model weights are not loaded |
| ModelRegistry | Model metadata/cache inspection duration | Repeated metadata/cache inspection duration; model weights are not loaded |
| Image / audio / video I/O | Decode and transform duration | Repeated decode and transform duration; no inference model is loaded |
| Tensor / maths / random | In-memory operation duration | Repeated in-memory operation duration; no model is downloaded |
| Runtime settings | Runtime inspection and settings-application duration | Repeated runtime inspection; no model is downloaded |

A metadata lookup, media decode, tensor operation, or random-number operation is not presented as model inference. Demos without model weights show execution duration rather than an artificial “download” value.

## Cloud browser benchmark

The CI workflow includes a Playwright Chromium test on a fresh browser context. It selects the compact `onnx-community/all-MiniLM-L6-v2-ONNX` feature-extraction model, forces the WASM backend, runs the built-in **Benchmark · 1 + 5 runs** flow, and exports the browser's measured initialization, first-inference, five warm-inference, and transfer-telemetry values.

The workflow uploads `cloud-browser-benchmark.json` and a full-page screenshot as the `cloud-browser-benchmark` artifact. The report includes the browser/runner environment and explicitly identifies that the results are GitHub-hosted CPU/WASM measurements, not Apple Silicon/WebGPU measurements.

The first CI execution validates whether the hosted runner can complete the actual Hub download and browser inference. If the Hugging Face Hub or CDN is temporarily unavailable, the workflow retains Playwright traces/screenshots for diagnosis and retries once.

## Repeatable cold-cache and warm-cache protocol

1. Open the site in a current Chromium-based browser. Record the browser/version, operating system, CPU/GPU, RAM, and network type.
2. For a true cold-cache test, clear the site's browser storage including **Cache Storage** and other site data, then reload. Clearing only Local Storage does not remove downloaded model artifacts.
3. Open one demo at a time and keep the model, input, backend, and dtype unchanged across repetitions.
4. Click **Load model & run**. Record model load + setup, observed transfer bytes/events/window, and first inference.
5. Repeat the same input at least five times. The benchmark panel stores the warm timings and calculates their median.
6. Reload the page without clearing site data and repeat once to observe the browser-cache path.
7. Repeat with WASM and WebGPU separately where supported. Do not compare different input sizes or dtypes as though they were controlled tests.
8. Export JSON/CSV from the Performance report. Keep the export with the device/browser/network details for reproducibility.

## Coverage under test

- 25 unique first-class `pipeline()` task IDs.
- 1 additional sentence-similarity demo implemented with feature-extraction embeddings and cosine similarity.
- 14 advanced API labs, including tokenizers, processors, tensor/math helpers, media I/O, runtime/cache settings, ModelRegistry, direct AutoModel inference, config inspection, streaming generation, structured output, multimodal embeddings, and seeded random utilities.
- Direct model architecture families and platform integrations are catalogued separately where they require model-specific preprocessing or cannot be meaningfully represented as one generic browser demo.

See [FEATURE-COVERAGE.md](./FEATURE-COVERAGE.md) for the full inventory and documented limitations.

## Important interpretation notes

- A page refresh is not a cold-cache test.
- Model load + setup is an end-to-end initialization measurement, not a pure download measurement.
- Progress callback bytes are observed per-file values. They may not include every request or reflect compressed wire bytes.
- A warm run is only comparable when task, model, backend, dtype, input, browser, and device are held constant.
- WebGPU is browser/device dependent. The app may use WASM when Auto mode cannot use WebGPU.
- The large LFM3-VL example is a copyable reference and is not auto-loaded; its multi-gigabyte artifacts should be tested separately on suitable hardware.
