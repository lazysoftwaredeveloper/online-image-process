import {
  AutoConfig,
  AutoImageProcessor,
  AutoTokenizer,
  RawImage,
  Tensor,
  env,
  load_video,
  pipeline,
} from '@huggingface/transformers';
import type { TaskDefinition } from './catalog';

export type BackendPreference = 'auto' | 'webgpu' | 'wasm';
export type DTypePreference = 'auto' | 'q8' | 'q4f16' | 'fp32';

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
      if (event?.status === 'progress' || event?.status === 'done') {
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
  const key = [definition.task, definition.model, preferredDevice, dtype].join('|');
  const cached = pipelineCache.get(key);
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
  const cacheKey = [definition.task, definition.model, actualDevice, dtype].join('|');
  pipelineCache.set(key, { instance, backend: actualDevice });
  pipelineCache.set(cacheKey, { instance, backend: actualDevice });

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
  const tracker = createProgressTracker(onProgress);
  const startedAt = performance.now();
  const tokenizer = await AutoTokenizer.from_pretrained('Xenova/bert-base-uncased', {
    progress_callback: tracker.callback,
  } as any);
  const encoding: any = await tokenizer(text);
  const ids = encoding.input_ids.tolist()[0].map((value: bigint | number) => Number(value));
  const tokens = tokenizer.convert_ids_to_tokens(ids);
  const decoded = tokenizer.decode(ids, { skip_special_tokens: false });
  return {
    elapsedMs: performance.now() - startedAt,
    progress: tracker.finish(),
    tokenCount: ids.length,
    tokens,
    ids,
    decoded,
  };
}

export async function runConfigInspector(modelId: string, onProgress?: (snapshot: ProgressSnapshot) => void) {
  const tracker = createProgressTracker(onProgress);
  const startedAt = performance.now();
  const config: any = await AutoConfig.from_pretrained(modelId, {
    progress_callback: tracker.callback,
  } as any);
  return {
    elapsedMs: performance.now() - startedAt,
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

export async function runProcessorLab(blob: Blob) {
  const rawImage = await RawImage.read(blob);
  const processor: any = await AutoImageProcessor.from_pretrained('Xenova/vit-base-patch16-224');
  const processed: any = await processor(rawImage);
  const pixelValues = processed?.pixel_values;
  return {
    image: rawImage,
    width: rawImage.width,
    height: rawImage.height,
    channels: rawImage.channels,
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
  const image = await RawImage.read(blob);
  const preview = await image.clone().resize(320, -1);
  const grayscale = await image.clone().resize(320, -1);
  grayscale.grayscale();
  return {
    image: preview,
    grayscale,
    width: image.width,
    height: image.height,
    channels: image.channels,
    bytes: image.data.byteLength,
  };
}

export async function runVideoIOLab(blob: Blob) {
  const video: any = await load_video(blob, { fps: 1, num_frames: 8 });
  return {
    width: video.width,
    height: video.height,
    fps: video.fps,
    duration: video.duration,
    frames: (video.frames ?? []).map((frame: any, index: number) => ({
      timestamp: frame.timestamp ?? index,
      image: frame.image ?? frame,
    })),
  };
}

export function runTensorLab() {
  const tensor = new Tensor('float32', new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
  const values = tensor.tolist() as number[][];
  const vectorA = [0.2, 0.4, 0.8];
  const vectorB = [0.1, 0.5, 0.7];
  const dot = vectorA.reduce((sum, value, index) => sum + value * vectorB[index], 0);
  const normA = Math.sqrt(vectorA.reduce((sum, value) => sum + value * value, 0));
  const normB = Math.sqrt(vectorB.reduce((sum, value) => sum + value * value, 0));
  const cosineSimilarity = dot / (normA * normB);
  return {
    tensor: { type: tensor.type, dims: tensor.dims, size: tensor.size, values },
    vectorA,
    vectorB,
    cosineSimilarity,
  };
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
    cacheKey: (env as any).cacheKey,
    allowRemoteModels: (env as any).allowRemoteModels,
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
