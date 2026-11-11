# Feature coverage

This inventory is pinned to Transformers.js 4.3.2. The playground contains all 25 first-class pipeline IDs and a separate sentence-similarity task demo implemented with FeatureExtractionPipeline. It also includes interactive labs for 14 advanced API areas. Direct AutoModel families are catalogued separately because each architecture has its own processor and output contract.

## First-class pipeline tasks

| Area | Pipeline ID | Demo | Example use | Input |
|---|---|---|---|---|
| Audio | audio-classification | Yes | Identify speech, music, animals, or environmental sounds | Audio |
| Audio | automatic-speech-recognition | Yes | Transcribe recorded speech | Audio |
| Vision | background-removal | Yes | Create transparent product/portrait cut-outs | Image |
| Vision | depth-estimation | Yes | Estimate relative depth for image effects and scene understanding | Image |
| Multimodal | document-question-answering | Yes | Ask questions about invoices, forms, and scanned documents | Document image + question |
| Text | feature-extraction | Yes | Create sentence embeddings for semantic search and similarity | Text |
| Text | sentence-similarity (task matrix) | Yes | Compare two sentence embeddings with cosine similarity; implemented via feature-extraction, not a separate pipeline class | Text pair |
| Text | fill-mask | Yes | Predict masked words for language understanding | Text with [MASK] |
| Vision | image-classification | Yes | Classify the main subject of an image | Image |
| Vision | image-feature-extraction | Yes | Create visual embeddings for image search and clustering | Image |
| Vision | image-segmentation | Yes | Produce pixel-level masks for objects or regions | Image |
| Vision | image-to-image | Yes | Super-resolution and image-domain transformation | Image |
| Multimodal | image-to-text | Yes | Generate a caption or description for an image | Image |
| Vision | object-detection | Yes | Locate known object classes with bounding boxes | Image |
| Text | question-answering | Yes | Extract an answer from a supplied text passage | Context + question |
| Text | summarization | Yes | Condense long articles, notes, or reports | Text |
| Text | text-classification | Yes | Sentiment, topic, or intent classification | Text |
| Text | text-generation | Yes | Continue a prompt with a causal language model | Prompt |
| Audio | text-to-audio | Yes | Synthesize speech from text | Text |
| Text | text2text-generation | Yes | Instruction-style transformation and text rewriting | Text |
| Text | token-classification | Yes | Extract entities such as people, places, and organizations | Text |
| Text | translation | Yes | Translate between model-supported language pairs | Text |
| Audio | zero-shot-audio-classification | Yes | Classify audio against user-provided labels | Audio + labels |
| Text | zero-shot-classification | Yes | Classify text against labels not fixed during training | Text + labels |
| Vision | zero-shot-image-classification | Yes | Match an image to user-provided labels | Image + labels |
| Vision | zero-shot-object-detection | Yes | Find objects described by user-provided labels | Image + labels |

The catalog intentionally shows a model ID per demo. Models are separate Hub artifacts: task support does not mean every model, language, dtype, or backend is compatible with every browser.

## Advanced API coverage

| API family | Coverage in the playground | Notes |
|---|---|---|
| AutoTokenizer | Interactive tokenizer inspector | Token IDs, token strings, decode round-trip, and model-provided chat-template rendering |
| AutoModel / task-specific AutoModel classes | Model/API map and config inspector | Direct forward passes require model-specific preprocessing and postprocessing |
| AutoProcessor / AutoImageProcessor / AutoFeatureExtractor | Interactive processor lab + direct-model demo + code patterns | CLIP AutoProcessor is run live; AutoImageProcessor is used for direct classification; AutoFeatureExtractor is included in the copyable audio preprocessing pattern |
| AutoConfig / PretrainedConfig | Interactive config inspector | Loads model metadata/config without running inference; exposes revision and local_files_only options |
| GenerationConfig and generation parameters | Text-generation demo + API guide | Sampling, beam search, max tokens, cache and stopping controls |
| Chat templates and tool schemas | Tokenizer lab + generation API map | Model-specific chat formatting, generation prompts, and tool schema hand-off |
| TextStreamer / WhisperTextStreamer | API guide and streaming-generation recipe | Incremental output for chat and transcription UIs |
| Tool calling and RAG | Generation API map + code reference | Text-generation options can pass tools and documents to compatible model chat templates; actual tool execution remains application code |
| Logits processors and stopping criteria | API guide | Advanced generation control |
| Tensor | Interactive tensor operation lab | Typed N-dimensional data, reductions, normalization, reshape/transpose, elementwise math, concatenation, stacking, pooling, layer norm, tensor factories, random tensors, and embedding quantization |
| RawImage / load_image | Interactive image I/O lab | Read, resize, convert, inspect dimensions and channels |
| RawAudio / load_audio | Interactive audio I/O lab | Decode local or remote audio to mono 16 kHz Float32Array and wrap it as WAV |
| RawVideo / load_video | Interactive frame-sampling lab | Experimental; samples browser-decoded video into RawImage frames |
| ONNX Runtime backends | Runtime inspector + per-demo backend selector | WebGPU when available; WASM fallback |
| Quantized dtypes | Per-demo dtype selector | Model artifacts must actually provide the selected dtype |
| Environment and cache | Interactive runtime settings panel | Browser/WASM/filesystem/custom cache, remote/local model loading, local model path, log level, custom fetch and ONNX device discovery |
| Hub loading options | Per-demo model selector and load telemetry | Revision, dtype, progress callback, local/remote options |
| Custom cache and fetch | Runtime settings + copyable API pattern | env.useCustomCache with a Cache API-compatible match/put adapter; env.fetch for custom headers, authentication gateways, and request control |
| Maths and tensor helpers | Interactive tensor operation lab | softmax, log_softmax, dot, cosine similarity, Tensor reductions and arithmetic, tensor factories, pooling, layer normalization, embedding quantization, permutation, and interpolation |
| Random utilities | Interactive seeded PRNG lab | MT19937-compatible random, gauss, shuffle, and weighted choices |
| ModelRegistry | Interactive model preflight/cache inspector | get_files, get_pipeline_files, component file lists, get_file_metadata, available dtypes, model/pipeline cache checks, and model/pipeline cache clearing |
| Structured output | Interactive constrained-generation demo | Experimental JSON Schema / JSON / regex constrained generation through the official companion package |
| Multimodal embeddings | Interactive text-to-image/audio/video similarity demo | EmbeddingGemma 2 maps text, images, audio, and video into a shared 768-dimensional space |

## Direct model families beyond pipeline()

The public API also exposes direct AutoModel classes for sequence/token classification, seq2seq and causal language modeling, speech seq2seq, text-to-spectrogram/waveform, masked language modeling, question answering, vision-to-sequence, image classification/segmentation, semantic and universal segmentation, object detection, zero-shot object detection, mask generation, CTC, audio classification, speaker embeddings (XVector), audio-frame classification, document QA, image matting, image-to-image, depth/normal/pose estimation, image feature extraction, image-text-to-text, audio-text-to-text, LightOnOcr, LFM3, and LFM3-VL. The live direct-model example runs AutoModelForImageClassification with a separately loaded AutoImageProcessor; the remaining families are catalogued with code references rather than being falsely presented as one generic demo.

These are model-loading/forward APIs rather than one-size-fits-all pipelines. The playground provides a code/API map for them instead of pretending every architecture has a universal input/output form. In particular, pose/normal estimation and vision/audio-language models need model-specific processor and postprocessing logic.

## Explicitly not represented as first-class pipeline demos

The official task matrix includes several tasks without a supported pipeline ID in the current API list: table question answering, mask-generation as a pipeline, video classification, unconditional image generation, audio-to-audio generation, tabular classification/regression, text-to-image, and visual question answering. Some related lower-level model classes may exist (for example AutoModelForMaskGeneration), but that is not the same as a supported pipeline. Sentence similarity is included as a separate task demo but implemented through feature-extraction embeddings and cosine similarity, not a separate pipeline class.

The library task matrix also lists Reinforcement Learning as a broad machine-learning area. It is not a built-in Transformers.js inference pipeline or browser RL training environment.


## Additional platform and integration capabilities

The official documentation also covers the following capabilities. They are documented as deployment/integration patterns rather than pretending they can all be exercised inside a static browser demo:

| Capability | Coverage in this project | Notes |
|---|---|---|
| Browser execution | Live site architecture | Static Vite app; ONNX Runtime executes inference in the browser |
| Node.js execution | Documentation/reference | Transformers.js also supports Node.js; media decoding and filesystem access differ from browser APIs |
| Web Workers | Architecture note | Long-running inference can be moved off the UI thread; worker bundling and model cache strategy are app concerns |
| Local models | Runtime inspector + Hub/config guide | env local/remote model flags and local model path are shown; a real local model folder must follow the library's expected ONNX layout |
| Private / gated Hub models | Documentation link | Requires a valid Hugging Face access token; never embed a private token in a public static frontend |
| WebGPU and WASM | Runtime inspector + per-demo selector | WebGPU is preferred when available; WASM broadens compatibility but can be slower |
| Vercel AI SDK integration | Documentation link | Integration pattern, not a separate inference engine |
| Server-side inference | Documentation link | Node.js/server deployment is supported; this playground intentionally keeps inference browser-local |
| Custom models | Documentation link | Models generally need compatible ONNX exports and a Transformers.js-supported architecture |


## Important distinctions

- **Pipeline tasks vs. task matrix:** the official task matrix includes sentence similarity, but the 25 first-class pipeline IDs do not include it in the same way. This playground uses the feature-extraction pipeline to create normalized embeddings and then computes cosine similarity.
- **Direct model classes vs. pipelines:** AutoModel classes expose more architecture families than the pipeline factory. A direct forward pass is not automatically a complete end-user task; preprocessing, decoding, and visualization remain model-specific.
- **Model availability vs. task support:** a task being supported does not guarantee that every model, language, quantization, or runtime backend is compatible.
- **Browser-only limits:** video decoding depends on browser media APIs; Node.js requires a separate media-decoding path. Private/gated model credentials must never be shipped in a public browser bundle.
- **Timing interpretation:** initialization time includes model files, parsing, runtime/session setup, and possibly shader compilation. Progress callback bytes are observed telemetry, not a guaranteed exact network accounting.


### Adjacent packages and ecosystem notes

- **Standalone `@huggingface/tokenizers`:** Transformers.js v4 separated tokenization into a small standalone package for applications that need tokenization without ONNX model inference. This is a companion package, not a required dependency of this playground.
- **Structured output companion:** `@huggingface/transformers-structured-output` provides experimental JSON Schema / JSON / regex-constrained generation. The playground demonstrates JSON Schema constrained generation.
- **Progress events:** v4 exposes `progress_total` events in addition to per-file progress events. The runtime progress UI uses the total progress signal where available and retains per-file byte telemetry separately.
- **Model preflight:** `ModelRegistry.get_file_metadata` can estimate the combined model artifact size before loading. A reported size is a Hub metadata estimate; actual transfer bytes can differ because of cache hits, redirects, compression, and range requests.
- **Translation task variants:** the pipeline factory supports language-pair task identifiers such as `translation_en_to_fr` in addition to the generic `translation` task.


### Large multimodal model reference

The Direct model APIs panel includes an optional, copyable Transformers.js example for `onnx-community/LFM3-VL-2B-ONNX` using `AutoModelForImageTextToText` and `AutoProcessor`. It is deliberately not auto-run: the model is approximately 2B parameters, requires WebGPU, and can require multi-gigabyte artifact downloads. The example follows the model repository's Transformers.js v4.3.2+ usage pattern.
