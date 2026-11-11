# Transformers.js Playground

A browser-first capability lab for Hugging Face Transformers.js. The goal is to make the library's supported tasks and core developer APIs tangible: users can upload sample inputs, load ONNX models from the Hugging Face Hub, run inference locally, inspect outputs, and record repeatable timing measurements.

## Principles

- **Browser inference, not a hosted inference API.** Model files are fetched from the Hugging Face Hub; inference runs in the visitor's browser through ONNX Runtime.
- **No hidden backend.** The application is a static Vite + React + TypeScript site. User media is processed in the browser.
- **Explicit coverage.** Every documented pipeline() task is represented in the catalog. Direct model APIs and the tokenizer, processor, tensor, media I/O, generation, runtime, and cache concepts are documented in the Advanced API lab.
- **Honest benchmarks.** First-load and warm-inference numbers are measured in the current browser/device and saved locally. No universal latency numbers are hard-coded because model size, network, browser, backend, and hardware materially affect results.
- **Scope labels.** Each demo explains practical use cases, expected input/output, limitations, model ID, and model-loading cost.

## Stack

- Vite
- React + TypeScript
- @huggingface/transformers 4.3.2 (pinned)
- Browser Cache API for model assets (managed by Transformers.js)
- Local Storage for benchmark history and preferences

No application server or inference API is required.

## Run locally

    npm install
    npm run dev
    npm run build
    npm run test

## Benchmark methodology

For each selected task, the browser records:
1. pipeline initialization duration (includes model download when the browser cache misses, model parsing, and runtime/session setup);
2. observed model-file transfer bytes and progress events when exposed by the library;
3. first inference duration;
4. warm inference durations and median across repeated runs;
5. browser, user agent, device memory/hardware concurrency where exposed, selected backend, dtype, and model ID.

The UI labels initialization time as **model load + runtime setup**, not pure network time. A progress callback can report file transfer progress, but it cannot always separate network, decompression, model parsing, and runtime compilation. Cache hits may produce no transfer progress events. The report therefore preserves the raw observed transfer data and does not infer an exact download duration when the library does not expose it.

## Coverage

See docs/FEATURE-COVERAGE.md for the supported pipeline inventory, advanced API coverage, and explicitly unsupported task families.

See docs/TEST-REPORT.md for the test protocol, environment fields, and how to generate a device-specific report from the live benchmark panel.
