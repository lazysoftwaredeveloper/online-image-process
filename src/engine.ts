import {
  AutoConfig,
  AutoFeatureExtractor,
  AutoImageProcessor,
  AutoModel,
  AutoModelForImageClassification,
  AutoProcessor,
  AutoTokenizer,
  ModelRegistry,
  RawAudio,
  RawImage,
  Tensor,
  cat,
  stack,
  mean_pooling,
  layer_norm,
  std_mean,
  mean,
  full,
  full_like,
  ones,
  ones_like,
  zeros,
  zeros_like,
  rand,
  randn,
  quantize_embeddings,
  cos_sim,
  dot,
  log_softmax,
  interpolate,
  permute,
  env,
  load_audio,
  load_image,
  load_video,
  LogLevel,
  pipeline,
  random,
  softmax,
} from '@huggingface/transformers';
import type { TaskDefinition } from './catalog';

export type BackendPreference = 'auto' | 'webgpu' | 'wasm';
export type DTypePreference = 'auto' | 'q1' | 'q1f16' | 'q2' | 'q2f16' | 'q4' | 'q4f16' | 'q8' | 'fp16' | 'fp32' | 'int8' | 'uint8';

export type ProgressSnapshot = {
  status: string;
  file?: string;
  progress?: number;
  loaded?: number;
  total?: number;
  events: number;
  files: number;
  transferBytes: number;
  transferWindowMs: number | null;
};

export type LoadedTask = {
  instance: any;
  backend: 'webgpu' | 'wasm';
  cacheHit: boolean;
  loadMs: number;
  progress: ProgressSnapshot;
};

const pipelineCache = new Map<string, { instance: any; backend: 'webgpu' | 'wasm' }>();
const tokenizerCache = new Map<string, any>();
const processorCache = new Map<string, any>();
const featureExtractorCache = new Map<string, any>();
const directModelCache = new Map<string, { model: any; processor: any; backend: 'webgpu' | 'wasm' }>();
let embeddingGemmaCache: { processor: any; model: any; backend: 'webgpu' | 'wasm' } | null = null;

function createProgressTracker(onProgress?: (snapshot: ProgressSnapshot) => void) {
  const fileBytes = new Map<string, number>();
  const startedAt = performance.now();
  let firstProgressAt: number | null = null;
  let lastProgressAt: number | null = null;
  let eventCount = 0;
  let lastStatus = 'waiting';
  let lastFile: string | undefined;
  let lastProgress: number | undefined;

  const snapshot = (): ProgressSnapshot => ({
    status: lastStatus,
    file: lastFile,
    progress: lastProgress,
    events: eventCount,
    files: fileBytes.size,
    transferBytes: Array.from(fileBytes.values()).reduce((sum, value) => sum + value, 0),
    transferWindowMs:
      firstProgressAt !== null && lastProgressAt !== null
        ? Math.max(0, lastProgressAt - firstProgressAt)
        : null,
  });

  return {
    callback(event: any) {
      eventCount += 1;
      lastStatus = String(event?.status ?? 'loading');
      lastFile = event?.file ? String(event.file) : lastFile;
      if (typeof event?.progress === 'number') lastProgress = event.progress;
      const now = performance.now();
      if (event?.status === 'progress' || event?.status === 'done' || event?.status === 'progress_total') {
        if (firstProgressAt === null) firstProgressAt = now;
        lastProgressAt = now;
      }
      if (event?.file && typeof event?.loaded === 'number') {
        const file = String(event.file);
        fileBytes.set(file, Math.max(fileBytes.get(file) ?? 0, event.loaded));
      }
      onProgress?.(snapshot());
    },
    finish(): ProgressSnapshot {
      const result = snapshot();
      if (eventCount === 0) {
        return { ...result, status: 'no progress events', transferWindowMs: null };
      }
      return result;
    },
    startedAt,
  };
}

async function detectWebGPU(): Promise<boolean> {
  try {
    const gpu = (navigator as any).gpu;
    if (!gpu?.requestAdapter) return false;
    const adapter = await gpu.requestAdapter();
    return Boolean(adapter);
  } catch {
    return false;
  }
}

async function createPipelineInstance(
  definition: TaskDefinition,
  device: 'webgpu' | 'wasm',
  dtype: DTypePreference,
  progressCallback: (event: any) => void,
) {
  const options: any = {
    device,
    progress_callback: progressCallback,
  };
  if (dtype !== 'auto') options.dtype = dtype;
  return await (pipeline as any)(definition.task, definition.model, options);
}

export async function loadTaskPipeline(
  definition: TaskDefinition,
  backendPreference: BackendPreference,
  dtype: DTypePreference,
  onProgress?: (snapshot: ProgressSnapshot) => void,
): Promise<LoadedTask> {
  const canUseWebGPU = backendPreference === 'webgpu'
    ? await detectWebGPU()
    : backendPreference === 'auto'
      ? await detectWebGPU()
      : false;

  if (backendPreference === 'webgpu' && !canUseWebGPU) {
    throw new Error('WebGPU is not available in this browser/device. Choose Auto or WASM.');
  }

  const preferredDevice: 'webgpu' | 'wasm' =
    backendPreference === 'wasm' ? 'wasm' : canUseWebGPU ? 'webgpu' : 'wasm';
  // Keep the requested backend in the key. An `auto` request may fall back to WASM;
  // that fallback must not make an explicit later `webgpu` request return a WASM model.
  const requestKey = [definition.task, definition.model, backendPreference, dtype].join('|');
  const cached = pipelineCache.get(requestKey);
  if (cached) {
    return {
      instance: cached.instance,
      backend: cached.backend,
      cacheHit: true,
      loadMs: 0,
      progress: {
        status: 'already loaded in this tab',
        events: 0,
        files: 0,
        transferBytes: 0,
        transferWindowMs: null,
      },
    };
  }

  const tracker = createProgressTracker(onProgress);
  const start = performance.now();
  let instance: any;
  let actualDevice = preferredDevice;

  try {
    instance = await createPipelineInstance(definition, preferredDevice, dtype, tracker.callback);
  } catch (error) {
    if (backendPreference !== 'auto' || preferredDevice !== 'webgpu') throw error;
    actualDevice = 'wasm';
    instance = await createPipelineInstance(definition, 'wasm', dtype, tracker.callback);
  }

  const loadMs = performance.now() - start;
  const cacheEntry = { instance, backend: actualDevice };
  const actualDeviceKey = [definition.task, definition.model, actualDevice, dtype].join('|');
  pipelineCache.set(requestKey, cacheEntry);
  pipelineCache.set(actualDeviceKey, cacheEntry);

  return {
    instance,
    backend: actualDevice,
    cacheHit: false,
    loadMs,
    progress: tracker.finish(),
  };
}

export async function runTaskInference(
  definition: TaskDefinition,
  instance: any,
  values: {
    text: string;
    question: string;
    labels: string[];
    image?: string;
    audio?: string;
  },
) {
  const text = values.text.trim();
  const image = values.image;
  const audio = values.audio;
  const labels = values.labels;

  switch (definition.id) {
    case 'question-answering':
      return await instance(values.question, text);
    case 'document-question-answering':
      return await instance(image, values.question);
    case 'zero-shot-classification':
      return await instance(text, labels);
    case 'zero-shot-image-classification':
      return await instance(image, labels);
    case 'zero-shot-object-detection':
      return await instance(image, labels);
    case 'zero-shot-audio-classification':
      return await instance(audio, labels);
    case 'sentence-similarity': {
      const [left, right] = await Promise.all([
        instance(text, { pooling: 'mean', normalize: true }),
        instance(values.question, { pooling: 'mean', normalize: true }),
      ]);
      const leftVector = left.tolist()[0] as number[];
      const rightVector = right.tolist()[0] as number[];
      return {
        sentenceA: text,
        sentenceB: values.question,
        similarity: cos_sim(leftVector, rightVector),
        embeddingDimensions: leftVector.length,
        interpretation: 'A relative semantic similarity score, not a calibrated probability.',
      };
    }
    case 'feature-extraction':
      return await instance(text, { pooling: 'mean', normalize: true });
    case 'text-generation':
      return await instance(text, { max_new_tokens: 64 });
    case 'summarization':
      return await instance(text, { max_new_tokens: 96 });
    case 'text2text-generation':
      return await instance(text, { max_new_tokens: 96 });
    case 'token-classification':
      return await instance(text, { aggregation_strategy: 'simple' });
    case 'text-to-audio':
      return await instance(text);
    case 'audio-classification':
    case 'automatic-speech-recognition':
      return await instance(audio);
    case 'background-removal':
    case 'depth-estimation':
    case 'image-classification':
    case 'image-feature-extraction':
    case 'image-segmentation':
    case 'image-to-image':
    case 'image-to-text':
    case 'object-detection':
      return await instance(image);
    default:
      return await instance(text);
  }
}

export async function runTokenizerLab(text: string, onProgress?: (snapshot: ProgressSnapshot) => void) {
  const modelId = 'onnx-community/SmolLM2-135M-Instruct-ONNX-MHA';
  const tracker = createProgressTracker(onProgress);
  const loadStartedAt = performance.now();
  const cacheHit = tokenizerCache.has(modelId);
  let tokenizer: any = tokenizerCache.get(modelId);
  if (!tokenizer) {
    tokenizer = await AutoTokenizer.from_pretrained(modelId, {
      progress_callback: tracker.callback,
    } as any);
    tokenizerCache.set(modelId, tokenizer);
  }
  const loadMs = performance.now() - loadStartedAt;
  const inferenceStartedAt = performance.now();
  const encoding: any = await tokenizer(text);
  const ids = encoding.input_ids.tolist()[0].map((value: bigint | number) => Number(value));
  const tokens = tokenizer.tokenize(text);
  const decoded = tokenizer.decode(ids, { skip_special_tokens: false });
  let chatTemplate: string | null = null;
  let chatTemplateError: string | null = null;
  try {
    chatTemplate = tokenizer.apply_chat_template(
      [{ role: 'user', content: text }],
      { tokenize: false, add_generation_prompt: true },
    );
  } catch (error) {
    chatTemplateError = error instanceof Error ? error.message : String(error);
  }
  return {
    modelId,
    cacheHit,
    loadMs,
    inferenceMs: performance.now() - inferenceStartedAt,
    elapsedMs: performance.now() - loadStartedAt,
    progress: tracker.finish(),
    tokenCount: ids.length,
    tokens,
    ids,
    decoded,
    chatTemplate,
    chatTemplateError,
  };
}

export async function runConfigInspector(
  modelId: string,
  onProgress?: (snapshot: ProgressSnapshot) => void,
  revision = 'main',
  localFilesOnly = false,
) {
  const tracker = createProgressTracker(onProgress);
  const startedAt = performance.now();
  const config: any = await AutoConfig.from_pretrained(modelId, {
    revision,
    local_files_only: localFilesOnly,
    progress_callback: tracker.callback,
  } as any);
  return {
    elapsedMs: performance.now() - startedAt,
    revision,
    localFilesOnly,
    progress: tracker.finish(),
    config: {
      model_type: config.model_type,
      architectures: config.architectures,
      hidden_size: config.hidden_size,
      num_hidden_layers: config.num_hidden_layers,
      vocab_size: config.vocab_size,
      image_size: config.image_size,
      id2label: config.id2label,
      keys: Object.keys(config).slice(0, 40),
    },
  };
}

export async function runProcessorLab(
  blob: Blob,
  onProgress?: (snapshot: ProgressSnapshot) => void,
) {
  const tracker = createProgressTracker(onProgress);
  const rawImage = await RawImage.read(blob);
  const modelId = 'Xenova/clip-vit-base-patch32';
  const loadStartedAt = performance.now();
  const cacheHit = processorCache.has(modelId);
  let processor: any = processorCache.get(modelId);
  if (!processor) {
    processor = await AutoProcessor.from_pretrained(modelId, {
      progress_callback: tracker.callback,
    } as any);
    processorCache.set(modelId, processor);
  }
  const loadMs = performance.now() - loadStartedAt;
  const preprocessStartedAt = performance.now();
  const processed: any = await processor(rawImage);
  const preprocessMs = performance.now() - preprocessStartedAt;
  const pixelValues = processed?.pixel_values;
  return {
    image: rawImage,
    width: rawImage.width,
    height: rawImage.height,
    channels: rawImage.channels,
    cacheHit,
    loadMs,
    preprocessMs,
    progress: tracker.finish(),
    tensor: pixelValues
      ? {
          dims: pixelValues.dims,
          type: pixelValues.type,
          size: pixelValues.size,
          preview: Array.from(pixelValues.data).slice(0, 12),
        }
      : Object.keys(processed ?? {}),
  };
}

export async function runImageIOLab(blob: Blob) {
  const startedAt = performance.now();
  const image = await RawImage.read(blob);
  const preview = await image.clone().resize(320, -1);
  const grayscale = await image.clone().resize(320, -1);
  grayscale.grayscale();
  const rgba = image.clone().rgba();
  return {
    image: preview,
    grayscale,
    width: image.width,
    height: image.height,
    channels: image.channels,
    rgbaChannels: rgba.channels,
    bytes: image.data.byteLength,
    elapsedMs: performance.now() - startedAt,
  };
}

export async function runAudioIOLab(url: string) {
  const startedAt = performance.now();
  const samples = await load_audio(url, 16000);
  const audio = new RawAudio(samples, 16000);
  let peakAmplitude = 0;
  for (let index = 0; index < samples.length; index += 1) {
    peakAmplitude = Math.max(peakAmplitude, Math.abs(samples[index]));
  }
  return {
    audio,
    sampleRate: 16000,
    sampleCount: samples.length,
    durationSeconds: samples.length / 16000,
    peakAmplitude,
    previewSamples: Array.from(samples.slice(0, 16)),
    elapsedMs: performance.now() - startedAt,
  };
}

export async function runAudioFeatureExtractorLab(
  url: string,
  onProgress?: (snapshot: ProgressSnapshot) => void,
) {
  const modelId = 'onnx-community/whisper-tiny.en';
  const tracker = createProgressTracker(onProgress);
  const loadStartedAt = performance.now();
  const cacheHit = featureExtractorCache.has(modelId);
  let extractor: any = featureExtractorCache.get(modelId);
  if (!extractor) {
    extractor = await AutoFeatureExtractor.from_pretrained(modelId, {
      progress_callback: tracker.callback,
    } as any);
    featureExtractorCache.set(modelId, extractor);
  }
  const loadMs = performance.now() - loadStartedAt;
  const samples = await load_audio(url, 16000);
  const preprocessStartedAt = performance.now();
  const processed: any = await extractor(samples);
  const preprocessMs = performance.now() - preprocessStartedAt;
  const inputFeatures = processed?.input_features;
  return {
    modelId,
    cacheHit,
    loadMs,
    preprocessMs,
    sampleRate: 16000,
    sampleCount: samples.length,
    durationSeconds: samples.length / 16000,
    outputKeys: Object.keys(processed ?? {}),
    inputFeatures: inputFeatures
      ? {
          dims: inputFeatures.dims,
          type: inputFeatures.type,
          size: inputFeatures.size,
          preview: Array.from(inputFeatures.data).slice(0, 16),
        }
      : null,
    progress: tracker.finish(),
  };
}

export async function runVideoIOLab(blob: Blob) {
  const startedAt = performance.now();
  const url = URL.createObjectURL(blob);
  try {
    const video: any = await load_video(url, { fps: 1, num_frames: 8 });
    return {
      width: video.width,
      height: video.height,
      fps: video.fps,
      duration: video.duration,
      frameCount: (video.frames ?? []).length,
      frames: (video.frames ?? []).map((frame: any, index: number) => ({
        timestamp: frame.timestamp ?? index,
        image: frame.image ?? frame,
      })),
      elapsedMs: performance.now() - startedAt,
    };
  } finally {
    URL.revokeObjectURL(url);
  }
}

export async function runModelRegistryLab(
  task: string,
  modelId: string,
  dtype: DTypePreference,
  device: 'webgpu' | 'wasm' | 'auto' = 'auto',
  revision = 'main',
  localFilesOnly = false,
) {
  const options: any = { revision, local_files_only: localFilesOnly };
  if (dtype !== 'auto') options.dtype = dtype;
  if (device !== 'auto') options.device = device;
  const startedAt = performance.now();
  const [allFiles, pipelineFiles, modelFiles, tokenizerFiles, processorFiles, availableDtypes, modelCache, modelCacheFiles, pipelineCache, pipelineCacheFiles] = await Promise.all([
    (ModelRegistry as any).get_files(modelId, options),
    (ModelRegistry as any).get_pipeline_files(task, modelId, options),
    (ModelRegistry as any).get_model_files(modelId, options),
    (ModelRegistry as any).get_tokenizer_files(modelId),
    (ModelRegistry as any).get_processor_files(modelId),
    (ModelRegistry as any).get_available_dtypes(modelId, options),
    (ModelRegistry as any).is_cached(modelId, options),
    (ModelRegistry as any).is_cached_files(modelId, options),
    (ModelRegistry as any).is_pipeline_cached(task, modelId, options),
    (ModelRegistry as any).is_pipeline_cached_files(task, modelId, options),
  ]);
  const fileMetadata = await Promise.all(
    (allFiles as string[]).map((file) => (ModelRegistry as any).get_file_metadata(modelId, file, options)),
  );
  const totalBytes = fileMetadata.reduce(
    (sum: number, item: any) => sum + (typeof item?.size === 'number' ? item.size : 0),
    0,
  );
  return {
    task,
    modelId,
    requestedDtype: dtype,
    requestedDevice: device,
    allFiles,
    pipelineFiles,
    modelFiles,
    tokenizerFiles,
    processorFiles,
    fileMetadata,
    fileCount: (allFiles as string[]).length,
    totalBytes,
    totalMiB: totalBytes > 0 ? totalBytes / 1024 / 1024 : null,
    availableDtypes,
    modelCache,
    modelCacheFiles,
    pipelineCache,
    pipelineCacheFiles,
    elapsedMs: performance.now() - startedAt,
  };
}

export async function clearModelRegistryCache(
  task: string,
  modelId: string,
  dtype: DTypePreference,
  scope: 'pipeline' | 'model' = 'pipeline',
  revision = 'main',
) {
  const options: any = { revision };
  if (dtype !== 'auto') options.dtype = dtype;
  const pipelineKeys = Array.from(pipelineCache.keys()).filter((key) => {
    const [cachedTask, cachedModel] = key.split('|');
    return cachedModel === modelId && (scope === 'model' || cachedTask === task);
  });
  const disposedPipelines = new Set<any>();
  for (const key of pipelineKeys) {
    const cached = pipelineCache.get(key);
    if (cached?.instance && !disposedPipelines.has(cached.instance)) {
      await cached.instance.dispose?.();
      disposedPipelines.add(cached.instance);
    }
    pipelineCache.delete(key);
  }

  let directModelKeys: string[] = [];
  if (scope === 'model') {
    directModelKeys = Array.from(directModelCache.keys()).filter((key) => key.startsWith(modelId + '|'));
    for (const key of directModelKeys) {
      const cached = directModelCache.get(key);
      await cached?.model?.dispose?.();
      directModelCache.delete(key);
    }
    tokenizerCache.delete(modelId);
    processorCache.delete(modelId);
    featureExtractorCache.delete(modelId);
    if (modelId === 'onnx-community/embeddinggemma-2-ONNX' && embeddingGemmaCache) {
      await embeddingGemmaCache.model?.dispose?.();
      embeddingGemmaCache = null;
    }
  }

  const result = scope === 'model'
    ? await (ModelRegistry as any).clear_cache(modelId, options)
    : await (ModelRegistry as any).clear_pipeline_cache(task, modelId, options);
  return {
    ...result,
    clearedScope: scope,
    inMemoryPipelinesDisposed: disposedPipelines.size,
    inMemoryPipelineKeysRemoved: pipelineKeys.length,
    inMemoryDirectModelsDisposed: directModelKeys.length,
  };
}

export function runRandomLab(seed: number) {
  const rng = new (random as any).Random(seed);
  const sequence = Array.from({ length: 6 }, () => rng.random());
  const shuffled = [1, 2, 3, 4, 5, 6];
  rng.shuffle(shuffled);
  const weightedChoice = rng.choices(['text', 'image', 'audio'], [5, 3, 2]);
  const normalSample = rng.gauss(0, 1);
  return {
    seed,
    sequence,
    shuffled,
    weightedChoice,
    normalSample,
    note: 'The same seed reproduces the sequence when no concurrent calls share this generator.',
  };
}

export async function runDirectModelLab(
  blob: Blob,
  backendPreference: BackendPreference,
  dtype: DTypePreference,
  onProgress?: (snapshot: ProgressSnapshot) => void,
) {
  const canUseWebGPU = backendPreference === 'wasm' ? false : await detectWebGPU();
  if (backendPreference === 'webgpu' && !canUseWebGPU) {
    throw new Error('This direct-model example requires WebGPU. Choose Auto or use a browser/device with WebGPU enabled.');
  }
  const preferredDevice: 'webgpu' | 'wasm' =
    backendPreference === 'wasm' ? 'wasm' : canUseWebGPU ? 'webgpu' : 'wasm';
  const modelId = 'Xenova/vit-base-patch16-224';
  const tracker = createProgressTracker(onProgress);
  const loadStartedAt = performance.now();
  let actualDevice = preferredDevice;
  let key = [modelId, actualDevice, dtype].join('|');
  let cached = directModelCache.get(key);
  let cacheHit = Boolean(cached);
  if (!cached) {
    const loadPair = async (device: 'webgpu' | 'wasm') => {
      const options: any = { device, progress_callback: tracker.callback };
      if (dtype !== 'auto') options.dtype = dtype;
      const [processor, model] = await Promise.all([
        AutoImageProcessor.from_pretrained(modelId, { progress_callback: tracker.callback } as any),
        AutoModelForImageClassification.from_pretrained(modelId, options),
      ]);
      return { processor, model, backend: device as 'webgpu' | 'wasm' };
    };
    try {
      cached = await loadPair(preferredDevice);
    } catch (error) {
      if (backendPreference !== 'auto' || preferredDevice !== 'webgpu') throw error;
      actualDevice = 'wasm';
      key = [modelId, actualDevice, dtype].join('|');
      cached = directModelCache.get(key) ?? await loadPair('wasm');
      cacheHit = directModelCache.has(key);
    }
    directModelCache.set(key, cached);
  }
  const loadMs = performance.now() - loadStartedAt;
  const image = await RawImage.read(blob);
  const preprocessStartedAt = performance.now();
  const inputs = await cached.processor(image);
  const preprocessMs = performance.now() - preprocessStartedAt;
  const inferenceStartedAt = performance.now();
  const output: any = await cached.model(inputs);
  const inferenceMs = performance.now() - inferenceStartedAt;
  const logits = output.logits.tolist()[0] as number[];
  const probabilities = Array.from(softmax(logits));
  const id2label = cached.model.config.id2label ?? {};
  const predictions = probabilities
    .map((score, index) => ({
      label: id2label[index] ?? id2label[String(index)] ?? String(index),
      score,
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);
  return {
    modelId,
    backend: cached.backend,
    dtype,
    cacheHit,
    width: image.width,
    height: image.height,
    loadMs,
    preprocessMs,
    inferenceMs,
    predictions,
    logitsShape: output.logits.dims,
    progress: tracker.finish(),
  };
}

export async function runMultimodalEmbeddingLab(
  kind: 'image' | 'audio' | 'video',
  mediaUrl: string,
  query: string,
  onProgress?: (snapshot: ProgressSnapshot) => void,
) {
  const modelId = 'onnx-community/embeddinggemma-2-ONNX';
  const tracker = createProgressTracker(onProgress);
  const loadStartedAt = performance.now();
  const cacheHit = embeddingGemmaCache !== null;
  if (!embeddingGemmaCache) {
    const processor: any = await AutoProcessor.from_pretrained(modelId, {
      progress_callback: tracker.callback,
    } as any);
    const backend = await detectWebGPU() ? 'webgpu' : 'wasm';
    const model: any = await AutoModel.from_pretrained(modelId, {
      device: backend,
      dtype: 'q4',
      progress_callback: tracker.callback,
    } as any);
    embeddingGemmaCache = { processor, model, backend };
  }
  const { processor, model, backend } = embeddingGemmaCache;
  const loadMs = performance.now() - loadStartedAt;
  const inferenceStartedAt = performance.now();
  const textInputs = await processor(['task: search result | query: ' + query]);
  const textOutput: any = await model(textInputs);
  let mediaInputs: any;
  if (kind === 'image') {
    const image = await load_image(mediaUrl);
    mediaInputs = await processor(null, image);
  } else if (kind === 'audio') {
    const audio = await load_audio(mediaUrl, 16000);
    mediaInputs = await processor(null, null, audio);
  } else {
    const video = await load_video(mediaUrl, { fps: 1, num_frames: 16 });
    mediaInputs = await processor(null, null, null, video);
  }
  const mediaOutput: any = await model(mediaInputs);
  const textVector = textOutput.sentence_embedding.tolist()[0] as number[];
  const mediaVector = mediaOutput.sentence_embedding.tolist()[0] as number[];
  return {
    modelId,
    kind,
    backend,
    dtype: 'q4',
    cacheHit,
    loadMs,
    inferenceMs: performance.now() - inferenceStartedAt,
    progress: tracker.finish(),
    similarity: cos_sim(textVector, mediaVector),
    embeddingDimensions: textVector.length,
    textEmbeddingPreview: textVector.slice(0, 12),
    mediaEmbeddingPreview: mediaVector.slice(0, 12),
    interpretation: 'The model embeds both inputs in a shared vector space; similarity is a ranking signal, not a probability.',
  };
}

export function runTensorLab() {
  const tensor = new Tensor('float32', new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
  const values = tensor.tolist() as number[][];
  const vectorA = [0.2, 0.4, 0.8];
  const vectorB = [0.1, 0.5, 0.7];
  const cosineSimilarity = cos_sim(vectorA, vectorB);
  const dotProduct = dot(vectorA, vectorB);
  const softmaxInput = [1.2, 0.5, -0.2];
  const softmaxOutput = Array.from(softmax(softmaxInput));
  const logSoftmaxOutput = Array.from(log_softmax(softmaxInput));
  const tensorOps = {
    sum: tensor.sum().item(),
    mean: tensor.mean().item(),
    l2Norm: tensor.norm(2).item(),
    min: tensor.min().item(),
    max: tensor.max().item(),
    argmin: Number(tensor.argmin().item()),
    argmax: Number(tensor.argmax().item()),
    transpose: tensor.transpose(1, 0).tolist(),
    normalized: tensor.normalize(2, 1).tolist(),
    clamped: tensor.clamp(0, 4).tolist(),
    flattened: tensor.flatten(0, 1).tolist(),
    sigmoid: tensor.sigmoid().tolist(),
    doubled: tensor.mul(2).tolist(),
  };
  const a = new Tensor('float32', new Float32Array([1, 2, 3, 4]), [2, 2]);
  const b = new Tensor('float32', new Float32Array([5, 6, 7, 8]), [2, 2]);
  const hiddenState = new Tensor('float32', new Float32Array([1, 2, 3, 4, 5, 6]), [1, 2, 3]);
  const attentionMask = new Tensor('int64', new BigInt64Array([1n, 1n]), [1, 2]);
  const [std, average] = std_mean(a, null, 1, false);
  const pooled = mean_pooling(hiddenState, attentionMask);
  const quantizable = new Tensor('float32', new Float32Array([1, -1, 0.5, -0.5, 2, -2, 3, -3]), [1, 8]);
  const sampleTensor = new Tensor('float32', new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
  const permuted = permute(sampleTensor, [1, 0]);
  const imageTensor = new Tensor('float32', new Float32Array([1, 2, 3, 4]), [1, 2, 2]);
  const interpolated = interpolate(imageTensor, [4, 4]);
  return {
    tensor: { type: tensor.type, dims: tensor.dims, size: tensor.size, values },
    tensorOps,
    tensorUtilities: {
      concatenatedShape: cat([a, b], 0).dims,
      stackedShape: stack([a, b], 0).dims,
      pooled: pooled.tolist(),
      layerNormalized: layer_norm(a, [2]).tolist(),
      standardDeviation: std.tolist(),
      mean: average.tolist(),
      meanFunction: mean(a, null, false).tolist(),
      full: full([2, 2], 3).tolist(),
      fullLike: full_like(a, 0).tolist(),
      ones: ones([2, 2]).tolist(),
      onesLike: ones_like(a).tolist(),
      zeros: zeros([2, 2]).tolist(),
      zerosLike: zeros_like(a).tolist(),
      random: rand([2, 2]).tolist(),
      randomNormal: randn([2, 2]).tolist(),
      quantizedEmbedding: quantize_embeddings(quantizable, 'binary').tolist(),
    },
    vectorA,
    vectorB,
    cosineSimilarity,
    dotProduct,
    softmaxInput,
    softmaxOutput,
    logSoftmaxOutput,
    tensorMath: {
      permutedShape: permuted.dims,
      permuted: permuted.tolist(),
      interpolatedShape: interpolated.dims,
      interpolatedPreview: interpolated.tolist()[0].slice(0, 8),
    },
  };
}

export async function applyRuntimeSettings(settings: {
  useBrowserCache: boolean;
  useWasmCache: boolean;
  allowRemoteModels: boolean;
  allowLocalModels: boolean;
  localModelPath: string;
  logLevel: number;
}) {
  (env as any).useBrowserCache = settings.useBrowserCache;
  (env as any).useWasmCache = settings.useWasmCache;
  (env as any).allowRemoteModels = settings.allowRemoteModels;
  (env as any).allowLocalModels = settings.allowLocalModels;
  if (settings.localModelPath.trim()) {
    (env as any).localModelPath = settings.localModelPath.trim();
  }
  (env as any).logLevel = settings.logLevel;
  return await inspectRuntime();
}

export async function inspectRuntime() {
  const webgpuAvailable = await detectWebGPU();
  const supportedDevices = (env as any).backends?.onnx?.getSupportedDevices?.() ?? [];
  return {
    version: env.version,
    webgpuAvailable,
    supportedDevices,
    browserCache: (env as any).useBrowserCache,
    wasmCache: (env as any).useWasmCache,
    filesystemAvailable: (env as any).useFS,
    filesystemCache: (env as any).useFSCache,
    customCacheEnabled: (env as any).useCustomCache,
    crossOriginStorageExperimental: (env as any).experimental_useCrossOriginStorage,
    cacheKey: (env as any).cacheKey,
    hasCustomFetch: typeof (env as any).fetch === 'function',
    cacheDir: (env as any).cacheDir,
    localModelPath: (env as any).localModelPath,
    remoteHost: (env as any).remoteHost,
    remotePathTemplate: (env as any).remotePathTemplate,
    allowLocalModels: (env as any).allowLocalModels,
    allowRemoteModels: (env as any).allowRemoteModels,
    logLevel: (env as any).logLevel,
    availableLogLevels: LogLevel,
    hardwareConcurrency: navigator.hardwareConcurrency ?? null,
    deviceMemory: (navigator as any).deviceMemory ?? null,
    userAgent: navigator.userAgent,
    secureContext: window.isSecureContext,
  };
}

export function toSafeResult(value: any, depth = 0): any {
  if (depth > 5) return '[nested output omitted]';
  if (value === null || value === undefined) return value;
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return value;
  if (typeof value === 'bigint') return value.toString();
  if (ArrayBuffer.isView(value)) {
    const typed = value as any;
    return {
      type: typed.constructor?.name ?? 'TypedArray',
      length: typed.length,
      preview: Array.from(typed).slice(0, 16).map((item: any) =>
        typeof item === 'bigint' ? item.toString() : item,
      ),
    };
  }
  if (Array.isArray(value)) return value.slice(0, 30).map((item) => toSafeResult(item, depth + 1));
  if (typeof value === 'object') {
    if (typeof value.width === 'number' && typeof value.height === 'number' && value.data) {
      return {
        type: 'RawImage',
        width: value.width,
        height: value.height,
        channels: value.channels,
        bytes: value.data.byteLength,
      };
    }
    if (typeof value.toBlob === 'function' && value.sampling_rate) {
      return {
        type: 'RawAudio',
        sampling_rate: value.sampling_rate,
        samples: value.data?.length ?? value.audio?.length ?? null,
      };
    }
    if (Array.isArray(value.dims) && value.data) {
      return {
        type: 'Tensor',
        dtype: value.type,
        dims: value.dims,
        size: value.size,
        preview: Array.from(value.data).slice(0, 16).map((item: any) =>
          typeof item === 'bigint' ? item.toString() : item,
        ),
      };
    }
    const result: Record<string, any> = {};
    for (const [key, item] of Object.entries(value).slice(0, 40)) {
      if (key === 'data' || key === 'mask') {
        result[key] = toSafeResult(item, depth + 1);
      } else {
        result[key] = toSafeResult(item, depth + 1);
      }
    }
    return result;
  }
  return String(value);
}

export function collectOutputMedia(value: any, path = 'output', found: { images: any[]; audios: any[] } = { images: [], audios: [] }, depth = 0) {
  if (depth > 6 || value === null || value === undefined) return found;
  if (Array.isArray(value)) {
    value.slice(0, 30).forEach((item, index) => collectOutputMedia(item, path + '[' + index + ']', found, depth + 1));
    return found;
  }
  if (typeof value === 'object') {
    if (typeof value.width === 'number' && typeof value.height === 'number' && value.data && typeof value.channels === 'number') {
      found.images.push({ label: path, image: value });
      return found;
    }
    if (typeof value.toBlob === 'function' && (value.sampling_rate || value.audio)) {
      found.audios.push({ label: path, audio: value });
      return found;
    }
    for (const [key, item] of Object.entries(value)) {
      collectOutputMedia(item, path + '.' + key, found, depth + 1);
    }
  }
  return found;
}

export function rawImageToDataUrl(image: any): string {
  const canvas = document.createElement('canvas');
  canvas.width = image.width;
  canvas.height = image.height;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Canvas is not available in this browser.');
  const imageData = context.createImageData(image.width, image.height);
  const source = image.data as Uint8Array | Uint8ClampedArray;
  const channels = image.channels ?? 3;
  if (channels === 4) {
    imageData.data.set(source as Uint8ClampedArray);
  } else {
    for (let i = 0, pixel = 0; i < source.length; i += channels, pixel += 4) {
      imageData.data[pixel] = source[i] ?? 0;
      imageData.data[pixel + 1] = channels > 1 ? source[i + 1] : source[i];
      imageData.data[pixel + 2] = channels > 2 ? source[i + 2] : source[i];
      imageData.data[pixel + 3] = channels === 2 ? source[i + 1] : 255;
    }
  }
  context.putImageData(imageData, 0, 0);
  return canvas.toDataURL('image/png');
}

export function rawAudioToUrl(audio: any): string {
  return URL.createObjectURL(audio.toBlob());
}
