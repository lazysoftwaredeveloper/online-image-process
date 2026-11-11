# Browser performance test report

## Status

**Measurement status: awaiting a real browser/device run.**

This repository does not contain fabricated latency numbers. A model's first-use duration is dominated by model artifact size, network throughput, browser cache state, browser/runtime version, device memory, and WebGPU/WASM support. A single number measured on another machine would not be a reliable promise to visitors.

The live Benchmark panel measures each selected task in the visitor's own browser and stores the result locally. Export the JSON/CSV report after testing to attach a reproducible, device-specific result set to a release.

## What is measured

| Metric | Meaning | Limitation |
|---|---|---|
| Model load + runtime setup | Time from pipeline creation to a ready pipeline | Includes downloads on cache misses, parsing, session creation, and runtime initialization; not pure network time |
| Observed transfer bytes | Sum of the largest loaded byte count reported for each file in progress callbacks | Only available when the library emits file progress events; cache hits may emit no transfer events |
| Observed transfer window | First-to-last progress event span | A progress span is not guaranteed to equal the complete network transfer time |
| First inference | First completed inference after the pipeline is ready | Includes one-time warm-up/kernel compilation costs that may occur on first execution |
| Warm inference | Each subsequent completed inference | Input size and output length still affect runtime |
| Warm median | Median of recorded warm inference runs | Compare only like-for-like model, input, backend, dtype, and device |

The interface calls the initialization metric **Model load + runtime setup** rather than “download time”. It only labels the transfer span as observed telemetry when progress events provide enough evidence. This avoids conflating download, parsing, decompression, and WebGPU shader compilation.

## Repeatable test protocol

1. Use a current Chromium-based browser and note the browser version, OS, CPU/GPU, RAM, and network conditions.
2. For a cold-cache measurement, clear the site's storage and the browser's cached model assets for the Hugging Face Hub, then reload the playground. A normal page refresh is not a cold-cache test.
3. Open one demo at a time. Keep the same backend and dtype for all repetitions.
4. Click **Load model & run** once. Record model load + runtime setup, progress telemetry, and first inference.
5. Run the same input at least five more times. Use the warm median, not the fastest run.
6. Repeat in a new tab without clearing the browser cache to observe cached initialization behavior.
7. Export the benchmark report. Do not compare measurements across different inputs or devices as if they were controlled experiments.

## Per-demo reporting

Every pipeline demo shows:
- model ID and selected backend/dtype;
- current model load + runtime setup time;
- progress events and observed transfer bytes, when available;
- first inference time;
- warm inference time and local median;
- a note that timings are device- and input-dependent.

The app's browser-local benchmark history is not uploaded to a server. Clearing site data also clears this history.

## Current verified status

- Repository source: implemented with a browser-side benchmark harness.
- Automated build/test status: to be confirmed by the GitHub Actions run for the implementation branch.
- Real model inference timings: not yet measured in a controlled browser session.
- Cross-browser compatibility: not yet manually verified.
