# Feature coverage

This inventory is pinned to Transformers.js 4.3.2 and separates first-class pipeline tasks from lower-level APIs. The pipeline list is the primary interactive demo surface; advanced APIs are exposed in the Developer APIs area.

## First-class pipeline tasks

| Area | Pipeline ID | Demo | Example use | Input |
|---|---|---|---|---|
| Text | audio-classification | Yes | Identify speech, music, animals, or environmental sounds | Audio |
| Audio | automatic-speech-recognition | Yes | Transcribe recorded speech | Audio |
| Vision | background-removal | Yes | Create transparent product/portrait cut-outs | Image |
| Vision | depth-estimation | Yes | Estimate relative depth for image effects and scene understanding | Image |
| Multimodal | document-question-answering | Yes | Ask questions about invoices, forms, and scanned documents | Document image + question |
| Text | feature-extraction | Yes | Create sentence embeddings for semantic search and similarity | Text |
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
| AutoTokenizer | Interactive tokenizer inspector | Token IDs, token strings, decode round-trip, chat-template concepts |
| AutoModel / task-specific AutoModel classes | Model/API map and config inspector | Direct forward passes require model-specific preprocessing and postprocessing |
| AutoProcessor / AutoImageProcessor / AutoFeatureExtractor | Documented and linked from the API map | Pipelines select processors automatically; direct use is for custom preprocessing |
| AutoConfig / PretrainedConfig | Interactive config inspector | Loads model metadata/config without running inference |
| GenerationConfig and generation parameters | Text-generation demo + API guide | Sampling, beam search, max tokens, cache and stopping controls |
| TextStreamer / WhisperTextStreamer | API guide and streaming-generation recipe | Incremental output for chat and transcription UIs |
| Logits processors and stopping criteria | API guide | Advanced generation control |
| Tensor | Interactive tensor-shape and math lab | Typed N-dimensional data and basic operations |
| RawImage / load_image | Interactive image I/O lab | Read, resize, convert, inspect dimensions and channels |
| RawAudio / load_audio | API guide | Audio samples and WAV conversion; file decoding uses browser media APIs |
| RawVideo / load_video | Interactive frame-sampling lab | Experimental; samples browser-decoded video into RawImage frames |
| ONNX Runtime backends | Runtime inspector + per-demo backend selector | WebGPU when available; WASM fallback |
| Quantized dtypes | Per-demo dtype selector | Model artifacts must actually provide the selected dtype |
| Environment and cache | Runtime inspector | Browser Cache API, WASM cache, remote/local model configuration |
| Hub loading options | Per-demo model selector and load telemetry | Revision, dtype, progress callback, local/remote options |
| Math helpers and random utilities | API map | cos_sim, softmax, tensor operations, seedable PRNG |

## Direct model families beyond pipeline()

The public API also exposes direct AutoModel classes for sequence/token classification, seq2seq and causal language modeling, speech seq2seq, text-to-spectrogram/waveform, masked language modeling, question answering, vision-to-sequence, image classification/segmentation, semantic and universal segmentation, object detection, zero-shot object detection, mask generation, CTC, audio classification, speaker embeddings (XVector), audio-frame classification, document QA, image matting, image-to-image, depth/normal/pose estimation, image feature extraction, image-text-to-text, and audio-text-to-text.

These are model-loading/forward APIs rather than one-size-fits-all pipelines. The playground provides a code/API map for them instead of pretending every architecture has a universal input/output form. In particular, pose/normal estimation and vision/audio-language models need model-specific processor and postprocessing logic.

## Explicitly not represented as first-class pipeline demos

The official task matrix includes several tasks without a supported pipeline ID in the current API list: table question answering, mask-generation as a pipeline, video classification, unconditional image generation, audio-to-audio generation, tabular classification/regression, text-to-image, and visual question answering. Some related lower-level model classes may exist (for example AutoModelForMaskGeneration), but that is not the same as a supported pipeline. Sentence similarity is demonstrated through feature-extraction embeddings and cosine similarity rather than a dedicated pipeline ID.

The library documentation also discusses Reinforcement Learning as a machine-learning area, but it is not a Transformers.js browser inference pipeline in this project.
