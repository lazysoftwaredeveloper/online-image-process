import { TextStreamer } from '@huggingface/transformers';
import type { TaskDefinition } from './catalog';
import type { BackendPreference, DTypePreference } from './engine';
import { loadTaskPipeline, toSafeResult } from './engine';

export async function runStreamingGeneration(
  definition: TaskDefinition,
  prompt: string,
  backend: BackendPreference,
  dtype: DTypePreference,
  onToken: (token: string) => void,
  onProgress?: (progress: any) => void,
) {
  const loaded = await loadTaskPipeline(definition, backend, dtype, onProgress);
  const streamer = new TextStreamer(loaded.instance.tokenizer, {
    skip_prompt: true,
    callback_function: onToken,
  });
  const startedAt = performance.now();
  const output = await loaded.instance(prompt, {
    max_new_tokens: 80,
    streamer,
  });
  return {
    loaded,
    inferenceMs: performance.now() - startedAt,
    output: toSafeResult(output),
  };
}
