import { TextStreamer } from '@huggingface/transformers';
import { StructuredOutputProcessor } from '@huggingface/transformers-structured-output';
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

export async function runStructuredOutputLab(
  prompt: string,
  onProgress?: (progress: any) => void,
) {
  const definition: TaskDefinition = {
    id: 'structured-output',
    task: 'text-generation',
    title: 'Structured output',
    category: 'Text',
    summary: 'Constrain generated output to a JSON schema.',
    useCases: ['Reliable JSON responses', 'Classification payloads', 'Form extraction'],
    scope: 'Experimental constrained generation with a JSON Schema processor.',
    limitations: 'This example requires WebGPU and one generated sequence. Model and tokenizer EOS settings must match.',
    model: 'onnx-community/LFM2.5-350M-ONNX',
    inputKind: 'text',
    docs: 'https://huggingface.co/docs/transformers.js/api/generation/parameters',
  };
  const loaded = await loadTaskPipeline(definition, 'webgpu', 'q4f16', onProgress);
  const processor = new StructuredOutputProcessor(loaded.instance.tokenizer, {
    type: 'json_schema',
    json_schema: {
      type: 'object',
      properties: {
        sentiment: { enum: ['positive', 'negative', 'neutral'] },
        topic: { enum: ['price', 'quality', 'delivery', 'other'] },
      },
      required: ['sentiment', 'topic'],
      additionalProperties: false,
    },
  });
  const messages = [{
    role: 'user',
    content: 'Classify this customer feedback into sentiment and topic. Feedback: ' + prompt,
  }];
  const startedAt = performance.now();
  const output = await loaded.instance(messages, {
    max_new_tokens: 256,
    do_sample: false,
    logits_processor: [processor],
  });
  return {
    loaded,
    inferenceMs: performance.now() - startedAt,
    output: toSafeResult(output),
  };
}
