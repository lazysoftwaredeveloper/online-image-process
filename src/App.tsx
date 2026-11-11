import { useEffect, useMemo, useState, type ReactNode } from 'react';
import {
  Activity,
  AlertCircle,
  ArrowDownToLine,
  ArrowUpRight,
  AudioLines,
  BookOpen,
  BrainCircuit,
  ChevronRight,
  Clock3,
  Code2,
  Copy,
  Cpu,
  Database,
  FileAudio,
  FileImage,
  FileVideo,
  Gauge,
  Image as ImageIcon,
  Layers,
  Loader2,
  Monitor,
  Play,
  Search,
  Settings2,
  ShieldCheck,
  Sparkles,
  Terminal,
  Upload,
  X,
  Zap,
} from 'lucide-react';
import {
  ADVANCED_APIS,
  CATEGORIES,
  SAMPLE_AUDIO,
  SAMPLE_DOCUMENT,
  TASKS,
  UNSUPPORTED_TASKS,
} from './catalog';
import {
  applyRuntimeSettings,
  clearModelRegistryCache,
  collectOutputMedia,
  loadTaskPipeline,
  rawAudioToUrl,
  rawImageToDataUrl,
  runAudioIOLab,
  runAudioFeatureExtractorLab,
  runConfigInspector,
  runDirectModelLab,
  runImageIOLab,
  runModelRegistryLab,
  runMultimodalEmbeddingLab,
  runProcessorLab,
  runRandomLab,
  runTaskInference,
  runTensorLab,
  runTokenizerLab,
  runVideoIOLab,
  toSafeResult,
  type BackendPreference,
  type DTypePreference,
  type LoadedTask,
  type ProgressSnapshot,
} from './engine';
import { runStreamingGeneration, runStructuredOutputLab } from './generation';
import {
  clearBenchmarks,
  exportBenchmarks,
  exportBenchmarksCsv,
  readBenchmarks,
  upsertBenchmark,
  type BenchmarkRecord,
} from './benchmark';

type Tab = 'explore' | 'benchmarks' | 'apis';
type OutputMedia = { images: { label: string; src: string }[]; audios: { label: string; src: string }[] };

const EMPTY_MEDIA: OutputMedia = { images: [], audios: [] };

function formatMs(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  if (value < 1000) return Math.round(value) + ' ms';
  return (value / 1000).toFixed(2) + ' s';
}

function formatBytes(value: number | null | undefined) {
  if (!value) return '—';
  if (value < 1024) return value + ' B';
  if (value < 1024 * 1024) return (value / 1024).toFixed(1) + ' KB';
  return (value / (1024 * 1024)).toFixed(2) + ' MB';
}

function sampleImageFor(taskId: string) {
  if (taskId === 'background-removal') {
    return 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/portrait-of-woman_small.jpg';
  }
  if (taskId === 'image-to-image') {
    return 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/butterfly.jpg';
  }
  if (taskId === 'image-classification') {
    return 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg';
  }
  return 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg';
}

function sampleAudioFor(taskId: string) {
  if (taskId === 'audio-classification') {
    return 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cat_meow.wav';
  }
  if (taskId === 'zero-shot-audio-classification') {
    return 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/dog_barking.wav';
  }
  return SAMPLE_AUDIO;
}

function downloadTextFile(name: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function ProgressBar({ progress }: { progress: ProgressSnapshot | null }) {
  const value = typeof progress?.progress === 'number' ? Math.max(0, Math.min(100, progress.progress)) : null;
  return (
    <div className="progress-wrap">
      <div className="progress-track">
        <div className={value === null ? 'progress-fill indeterminate' : 'progress-fill'} style={value === null ? undefined : { width: value + '%' }} />
      </div>
      <div className="progress-meta">
        <span>{progress?.status ?? 'Waiting for model files'}</span>
        <span>{value === null ? 'Preparing' : Math.round(value) + '%'}</span>
      </div>
    </div>
  );
}

function Pill({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'green' | 'blue' | 'amber' }) {
  return <span className={'pill pill-' + tone}>{children}</span>;
}

function App() {
  const [tab, setTab] = useState<Tab>('explore');
  const [category, setCategory] = useState<string>('All');
  const [search, setSearch] = useState('');
  const [selectedId, setSelectedId] = useState('background-removal');
  const selectedTask = TASKS.find((task) => task.id === selectedId) ?? TASKS[0];
  const [text, setText] = useState(selectedTask.defaultText ?? '');
  const [question, setQuestion] = useState(selectedTask.defaultQuestion ?? '');
  const [labels, setLabels] = useState((selectedTask.defaultLabels ?? []).join(', '));
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [imageObjectUrl, setImageObjectUrl] = useState<string | null>(null);
  const [audioObjectUrl, setAudioObjectUrl] = useState<string | null>(null);
  const [backend, setBackend] = useState<BackendPreference>('auto');
  const [dtype, setDtype] = useState<DTypePreference>('auto');
  const [runStatus, setRunStatus] = useState<'idle' | 'loading' | 'running' | 'ready' | 'error'>('idle');
  const [progress, setProgress] = useState<ProgressSnapshot | null>(null);
  const [lastLoad, setLastLoad] = useState<LoadedTask | null>(null);
  const [lastInferenceMs, setLastInferenceMs] = useState<number | null>(null);
  const [result, setResult] = useState<any>(null);
  const [resultMedia, setResultMedia] = useState<OutputMedia>(EMPTY_MEDIA);
  const [error, setError] = useState('');
  const [benchmarks, setBenchmarks] = useState<BenchmarkRecord[]>(() => readBenchmarks());

  const filteredTasks = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return TASKS.filter((task) => {
      const matchesCategory = category === 'All' || task.category === category;
      const matchesSearch = !needle || [
        task.title,
        task.task,
        task.summary,
        task.model,
        ...task.useCases,
      ].join(' ').toLowerCase().includes(needle);
      return matchesCategory && matchesSearch;
    });
  }, [category, search]);

  useEffect(() => {
    setText(selectedTask.defaultText ?? '');
    setQuestion(selectedTask.defaultQuestion ?? '');
    setLabels((selectedTask.defaultLabels ?? []).join(', '));
    setImageFile(null);
    setAudioFile(null);
    setRunStatus('idle');
    setProgress(null);
    setLastLoad(null);
    setLastInferenceMs(null);
    setResult(null);
    setResultMedia(EMPTY_MEDIA);
    setError('');
  }, [selectedId]);

  useEffect(() => {
    if (!imageFile) {
      setImageObjectUrl(null);
      return;
    }
    const url = URL.createObjectURL(imageFile);
    setImageObjectUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [imageFile]);

  useEffect(() => {
    if (!audioFile) {
      setAudioObjectUrl(null);
      return;
    }
    const url = URL.createObjectURL(audioFile);
    setAudioObjectUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [audioFile]);

  useEffect(() => {
    return () => {
      resultMedia.audios.forEach((audio) => URL.revokeObjectURL(audio.src));
    };
  }, [resultMedia]);

  const imageSource = imageObjectUrl ?? (selectedTask.inputKind === 'document' ? SAMPLE_DOCUMENT : sampleImageFor(selectedTask.id));
  const audioSource = audioObjectUrl ?? sampleAudioFor(selectedTask.id);
  const selectedRecord = benchmarks.find((record) =>
    record.taskId === selectedTask.id && record.model === selectedTask.model &&
    record.dtype === dtype && (record.backend === backend || backend === 'auto'),
  );

  function parseLabels() {
    return labels.split(/[\n,]/).map((label) => label.trim()).filter(Boolean);
  }

  function currentInputs() {
    return {
      text,
      question,
      labels: parseLabels(),
      image: imageSource,
      audio: audioSource,
    };
  }

  function saveRun(load: LoadedTask, inferenceMs: number, isNewLoad: boolean) {
    const key = [selectedTask.id, selectedTask.model, load.backend, dtype].join('|');
    const incoming = {
      key,
      taskId: selectedTask.id,
      taskName: selectedTask.title,
      model: selectedTask.model,
      backend: load.backend,
      dtype,
      loadMs: load.loadMs,
      transferBytes: load.progress.transferBytes,
      transferEvents: load.progress.events,
      transferWindowMs: load.progress.transferWindowMs,
      inferenceMs,
      newLoad: isNewLoad,
      userAgent: navigator.userAgent,
      hardwareConcurrency: navigator.hardwareConcurrency ?? null,
      deviceMemory: (navigator as any).deviceMemory ?? null,
    };
    setBenchmarks((current) => upsertBenchmark(current, incoming));
  }

  function updateResult(output: any) {
    setResult(output);
    const found = collectOutputMedia(output);
    const images = found.images.slice(0, 5).map((item) => ({
      label: item.label,
      src: rawImageToDataUrl(item.image),
    }));
    const audios = found.audios.slice(0, 3).map((item) => ({
      label: item.label,
      src: rawAudioToUrl(item.audio),
    }));
    setResultMedia({ images, audios });
  }

  async function handleRun() {
    if (runStatus === 'loading' || runStatus === 'running') return;
    if (selectedTask.inputKind === 'text' && !text.trim()) {
      setError('Add some text before running this demo.');
      return;
    }
    if ((selectedTask.inputKind === 'document' || selectedTask.id === 'sentence-similarity') && !question.trim()) {
      setError(selectedTask.id === 'sentence-similarity' ? 'Add a second sentence to compare.' : 'Add a question about the document.');
      return;
    }
    if (selectedTask.id.startsWith('zero-shot-') && parseLabels().length === 0) {
      setError('Add at least one candidate label.');
      return;
    }

    setError('');
    setRunStatus('loading');
    setProgress(null);
    setResult(null);
    setResultMedia(EMPTY_MEDIA);
    try {
      const loaded = await loadTaskPipeline(selectedTask, backend, dtype, setProgress);
      setLastLoad(loaded);
      setRunStatus('running');
      const startedAt = performance.now();
      const output = await runTaskInference(selectedTask, loaded.instance, currentInputs());
      const inferenceMs = performance.now() - startedAt;
      setLastInferenceMs(inferenceMs);
      updateResult(output);
      saveRun(loaded, inferenceMs, !loaded.cacheHit);
      setRunStatus('ready');
    } catch (cause) {
      setRunStatus('error');
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }

  async function handleBenchmark() {
    if (runStatus === 'loading' || runStatus === 'running') return;
    if (selectedTask.inputKind === 'text' && !text.trim()) {
      setError('Add some text before benchmarking.');
      return;
    }
    if ((selectedTask.inputKind === 'document' || selectedTask.id === 'sentence-similarity') && !question.trim()) {
      setError(selectedTask.id === 'sentence-similarity' ? 'Add a second sentence to compare.' : 'Add a question about the document.');
      return;
    }
    if (selectedTask.id.startsWith('zero-shot-') && parseLabels().length === 0) {
      setError('Add at least one candidate label.');
      return;
    }

    setError('');
    setRunStatus('loading');
    setProgress(null);
    setResult(null);
    setResultMedia(EMPTY_MEDIA);
    try {
      const loaded = await loadTaskPipeline(selectedTask, backend, dtype, setProgress);
      setLastLoad(loaded);
      setRunStatus('running');
      let latestOutput: any = null;
      let firstDuration = 0;
      const warmDurations: number[] = [];
      for (let index = 0; index < 6; index += 1) {
        const startedAt = performance.now();
        latestOutput = await runTaskInference(selectedTask, loaded.instance, currentInputs());
        const elapsed = performance.now() - startedAt;
        if (index === 0) {
          firstDuration = elapsed;
          saveRun(loaded, elapsed, !loaded.cacheHit);
        } else {
          warmDurations.push(elapsed);
          saveRun(loaded, elapsed, false);
        }
      }
      setLastInferenceMs(warmDurations[warmDurations.length - 1] ?? firstDuration);
      updateResult(latestOutput);
      setRunStatus('ready');
    } catch (cause) {
      setRunStatus('error');
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }

  function handleFile(file: File | undefined) {
    if (!file) return;
    const limit = selectedTask.inputKind === 'audio' ? 30 : 15;
    if (file.size > limit * 1024 * 1024) {
      setError('This demo accepts files up to ' + limit + ' MB. Choose a smaller file or compress it first.');
      return;
    }
    setError('');
    if (selectedTask.inputKind === 'audio') setAudioFile(file);
    else setImageFile(file);
  }

  const isBusy = runStatus === 'loading' || runStatus === 'running';

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" onClick={() => setTab('explore')}>
          <span className="brand-mark"><BrainCircuit size={19} /></span>
          <span>Browser ML Lab</span>
          <span className="brand-divider" />
          <span className="brand-sub">Transformers.js</span>
        </a>
        <nav className="topnav" aria-label="Main navigation">
          <button className={tab === 'explore' ? 'nav-link active' : 'nav-link'} onClick={() => setTab('explore')}>Explore</button>
          <button className={tab === 'benchmarks' ? 'nav-link active' : 'nav-link'} onClick={() => setTab('benchmarks')}>Benchmarks</button>
          <button className={tab === 'apis' ? 'nav-link active' : 'nav-link'} onClick={() => setTab('apis')}>Developer APIs</button>
        </nav>
        <div className="topbar-right">
          <Pill tone="green"><span className="live-dot" /> Runs locally</Pill>
          <a className="icon-link" href="https://huggingface.co/docs/transformers.js" target="_blank" rel="noreferrer" aria-label="Open official documentation">
            <BookOpen size={17} />
          </a>
          <a className="icon-link" href="https://github.com/lazysoftwaredeveloper/online-image-process" target="_blank" rel="noreferrer" aria-label="Open repository">
            <ArrowUpRight size={17} />
          </a>
        </div>
      </header>

      <main id="top">
        <section className="hero">
          <div className="hero-copy">
            <div className="eyebrow"><span className="eyebrow-line" /> ON-DEVICE AI · NO INFERENCE SERVER</div>
            <h1>Explore what runs<br /><span>inside your browser.</span></h1>
            <p className="hero-lede">A hands-on lab for Hugging Face Transformers.js. Load real ONNX models, try every documented pipeline, and measure the difference between first and repeat runs on your own device.</p>
            <div className="hero-actions">
              <button className="button button-primary" onClick={() => setTab('explore')}>
                <Play size={15} fill="currentColor" /> Open the playground
              </button>
              <button className="button button-quiet" onClick={() => setTab('benchmarks')}>
                <Gauge size={16} /> View performance data
              </button>
            </div>
            <div className="hero-footnote"><ShieldCheck size={14} /> Your input files stay in this browser. Model files are downloaded from Hugging Face Hub.</div>
          </div>
          <div className="hero-visual">
            <div className="visual-orbit orbit-one" />
            <div className="visual-orbit orbit-two" />
            <div className="visual-core">
              <div className="core-icon"><BrainCircuit size={36} /></div>
              <div className="core-label">TRANSFORMERS.JS</div>
              <div className="core-caption">local inference</div>
            </div>
            <div className="orbit-node node-text"><span>TXT</span><small>language</small></div>
            <div className="orbit-node node-vision"><ImageIcon size={18} /><small>vision</small></div>
            <div className="orbit-node node-audio"><AudioLines size={18} /><small>audio</small></div>
            <div className="orbit-node node-multi"><Layers size={18} /><small>multimodal</small></div>
            <div className="visual-grid" />
          </div>
        </section>

        <section className="stats-row">
          <div className="stat-card">
            <div className="stat-icon purple"><Layers size={17} /></div>
            <div><strong>26</strong><span>supported task demos</span></div>
          </div>
          <div className="stat-card">
            <div className="stat-icon blue"><BrainCircuit size={17} /></div>
            <div><strong>4</strong><span>modalities</span></div>
          </div>
          <div className="stat-card">
            <div className="stat-icon green"><Cpu size={17} /></div>
            <div><strong>WebGPU + WASM</strong><span>runtime backends</span></div>
          </div>
          <div className="stat-card">
            <div className="stat-icon amber"><Activity size={17} /></div>
            <div><strong>{benchmarks.length}</strong><span>benchmarked configs</span></div>
          </div>
        </section>

        <div className="content-wrap">
          {tab === 'explore' && (
            <section className="workspace">
              <aside className="catalog-panel">
                <div className="panel-heading">
                  <div><span className="section-kicker">CAPABILITY INDEX</span><h2>Explore tasks</h2></div>
                  <span className="count-chip">{TASKS.length}</span>
                </div>
                <div className="search-box">
                  <Search size={16} />
                  <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search tasks or models..." />
                  {search && <button onClick={() => setSearch('')} aria-label="Clear search"><X size={14} /></button>}
                </div>
                <div className="category-tabs">
                  {CATEGORIES.map((item) => (
                    <button key={item} className={category === item ? 'category-tab selected' : 'category-tab'} onClick={() => setCategory(item)}>
                      {item}
                    </button>
                  ))}
                </div>
                <div className="task-list">
                  {filteredTasks.map((task) => (
                    <button key={task.id} className={selectedId === task.id ? 'task-row selected' : 'task-row'} onClick={() => setSelectedId(task.id)}>
                      <span className={'task-icon icon-' + task.category.toLowerCase()}>{task.category === 'Text' ? <Terminal size={16} /> : task.category === 'Vision' ? <ImageIcon size={16} /> : task.category === 'Audio' ? <AudioLines size={16} /> : <Layers size={16} />}</span>
                      <span className="task-row-copy"><strong>{task.title}</strong><small>{task.task}</small></span>
                      <ChevronRight size={15} className="task-chevron" />
                    </button>
                  ))}
                  {filteredTasks.length === 0 && <div className="empty-state compact">No tasks match this search.</div>}
                </div>
                <div className="catalog-footer">
                  <span><span className="live-dot" /> 25 pipeline IDs + sentence similarity</span>
                  <span>v4.3.2</span>
                </div>
              </aside>

              <section className="demo-panel">
                <div className="demo-header">
                  <div className="demo-title-group">
                    <div className={'large-task-icon icon-' + selectedTask.category.toLowerCase()}>
                      {selectedTask.category === 'Text' ? <Terminal size={20} /> : selectedTask.category === 'Vision' ? <ImageIcon size={20} /> : selectedTask.category === 'Audio' ? <AudioLines size={20} /> : <Layers size={20} />}
                    </div>
                    <div>
                      <div className="demo-title-meta"><Pill tone="blue">{selectedTask.category}</Pill><span className="task-id">{selectedTask.task}</span></div>
                      <h2>{selectedTask.title}</h2>
                      <p>{selectedTask.summary}</p>
                    </div>
                  </div>
                  <a className="docs-link" href={selectedTask.docs} target="_blank" rel="noreferrer">API docs <ArrowUpRight size={13} /></a>
                </div>

                <div className="demo-grid">
                  <div className="demo-input-column">
                    <div className="subsection-heading"><span>01</span><h3>Input</h3></div>
                    {selectedTask.inputKind === 'text' && (
                      <div className="field-stack">
                        {selectedTask.id === 'question-answering' ? (
                          <>
                            <label className="field-label">Context passage</label>
                            <textarea className="text-area" value={text} onChange={(event) => setText(event.target.value)} rows={7} />
                            <label className="field-label">Question</label>
                            <input className="text-input" value={question} onChange={(event) => setQuestion(event.target.value)} />
                          </>
                        ) : selectedTask.id === 'sentence-similarity' ? (
                          <>
                            <label className="field-label">Sentence A</label>
                            <textarea className="text-area" value={text} onChange={(event) => setText(event.target.value)} rows={3} />
                            <label className="field-label">Sentence B</label>
                            <textarea className="text-area" value={question} onChange={(event) => setQuestion(event.target.value)} rows={3} />
                          </>
                        ) : (
                          <>
                            <label className="field-label">{selectedTask.id === 'text-generation' ? 'Prompt' : 'Text input'}</label>
                            <textarea className="text-area" value={text} onChange={(event) => setText(event.target.value)} rows={selectedTask.id === 'summarization' ? 9 : 6} />
                          </>
                        )}
                        {selectedTask.id === 'fill-mask' && <p className="field-hint">Include the model's mask token exactly: [MASK].</p>}
                        {selectedTask.id.startsWith('zero-shot-') && (
                          <>
                            <label className="field-label">Candidate labels <span>comma or line separated</span></label>
                            <textarea className="text-area small-area" value={labels} onChange={(event) => setLabels(event.target.value)} rows={3} />
                          </>
                        )}
                      </div>
                    )}
                    {(selectedTask.inputKind === 'image' || selectedTask.inputKind === 'document') && (
                      <div className="field-stack">
                        <label className="field-label">{selectedTask.inputKind === 'document' ? 'Document image' : 'Image file'}</label>
                        <label className="upload-zone">
                          <input type="file" accept="image/*" onChange={(event) => handleFile(event.target.files?.[0])} />
                          <span className="upload-icon"><Upload size={18} /></span>
                          <strong>{imageFile ? imageFile.name : 'Choose an image'}</strong>
                          <small>{imageFile ? formatBytes(imageFile.size) + ' · click to replace' : 'JPG, PNG, WebP · up to 15 MB'}</small>
                        </label>
                        <div className="input-preview">
                          <img src={imageSource} alt="Selected input preview" />
                          <span className="preview-label">{imageFile ? 'YOUR UPLOAD' : 'SAMPLE INPUT'}</span>
                        </div>
                        {selectedTask.inputKind === 'document' && (
                          <>
                            <label className="field-label">Question about this document</label>
                            <input className="text-input" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="What is the invoice number?" />
                          </>
                        )}
                        {selectedTask.id.startsWith('zero-shot-') && (
                          <>
                            <label className="field-label">Candidate labels <span>comma or line separated</span></label>
                            <textarea className="text-area small-area" value={labels} onChange={(event) => setLabels(event.target.value)} rows={3} />
                          </>
                        )}
                      </div>
                    )}
                    {selectedTask.inputKind === 'audio' && (
                      <div className="field-stack">
                        <label className="field-label">Audio file</label>
                        <label className="upload-zone">
                          <input type="file" accept="audio/*" onChange={(event) => handleFile(event.target.files?.[0])} />
                          <span className="upload-icon"><FileAudio size={18} /></span>
                          <strong>{audioFile ? audioFile.name : 'Choose an audio clip'}</strong>
                          <small>{audioFile ? formatBytes(audioFile.size) + ' · click to replace' : 'WAV, MP3, FLAC · up to 30 MB'}</small>
                        </label>
                        <div className="audio-preview">
                          <div className="audio-preview-icon"><AudioLines size={20} /></div>
                          <div className="audio-preview-copy"><strong>{audioFile ? 'Uploaded audio' : 'Sample audio'}</strong><small>{audioFile ? 'Your file is processed locally' : 'Hugging Face demo audio'}</small></div>
                          <audio controls src={audioSource} />
                        </div>
                        {selectedTask.id === 'zero-shot-audio-classification' && (
                          <>
                            <label className="field-label">Candidate labels <span>comma or line separated</span></label>
                            <textarea className="text-area small-area" value={labels} onChange={(event) => setLabels(event.target.value)} rows={3} />
                          </>
                        )}
                      </div>
                    )}

                    <div className="subsection-heading settings-heading"><span>02</span><h3>Runtime settings</h3></div>
                    <div className="settings-grid">
                      <label className="select-field"><span>Backend</span>
                        <select value={backend} onChange={(event) => setBackend(event.target.value as BackendPreference)}>
                          <option value="auto">Auto (WebGPU → WASM)</option>
                          <option value="webgpu">WebGPU</option>
                          <option value="wasm">WASM</option>
                        </select>
                      </label>
                      <label className="select-field"><span>Model dtype</span>
                        <select value={dtype} onChange={(event) => setDtype(event.target.value as DTypePreference)}>
                          <option value="auto">Model default</option>
                          <option value="q4">q4</option>
                          <option value="q8">q8</option>
                          <option value="q4f16">q4f16</option>
                          <option value="fp16">fp16</option>
                          <option value="fp32">fp32</option>
                        </select>
                      </label>
                    </div>
                    <div className="model-row">
                      <div><span className="field-label">MODEL REPOSITORY</span><a href={'https://huggingface.co/' + selectedTask.model} target="_blank" rel="noreferrer">{selectedTask.model} <ArrowUpRight size={12} /></a></div>
                      <span className="model-source"><span className="live-dot" /> Hugging Face Hub</span>
                    </div>
                    <div className="run-actions">
                      <button className="button button-primary run-button" onClick={handleRun} disabled={isBusy}>
                        {isBusy ? <Loader2 size={16} className="spin" /> : <Play size={15} fill="currentColor" />}
                        {runStatus === 'loading' ? 'Loading model...' : runStatus === 'running' ? 'Running inference...' : 'Load model & run'}
                      </button>
                      <button className="button button-outline" onClick={handleBenchmark} disabled={isBusy}>
                        <Gauge size={15} /> Benchmark · 1 + 5 runs
                      </button>
                    </div>
                    {isBusy && <ProgressBar progress={progress} />}
                    {error && <div className="error-banner"><AlertCircle size={16} /><span>{error}</span><button onClick={() => setError('')}><X size={14} /></button></div>}
                  </div>

                  <div className="demo-output-column">
                    <div className="subsection-heading"><span>03</span><h3>Output & timing</h3><Pill tone={runStatus === 'ready' ? 'green' : runStatus === 'error' ? 'amber' : 'neutral'}>{runStatus === 'ready' ? 'Complete' : runStatus === 'loading' ? 'Loading' : runStatus === 'running' ? 'Running' : runStatus === 'error' ? 'Error' : 'Ready to run'}</Pill></div>
                    <div className="timing-grid">
                      <div className="timing-card"><span>Model load + setup</span><strong>{lastLoad ? formatMs(lastLoad.loadMs) : '—'}</strong><small>{lastLoad?.cacheHit ? 'Already loaded in this tab' : lastLoad ? 'Initialization' : 'Download + runtime'}</small></div>
                      <div className="timing-card"><span>Latest inference</span><strong>{formatMs(lastInferenceMs)}</strong><small>Time to complete</small></div>
                    </div>
                    <div className="telemetry-card">
                      <div><span>Observed model transfer</span><strong>{lastLoad ? formatBytes(lastLoad.progress.transferBytes) : '—'}</strong></div>
                      <div><span>Progress events</span><strong>{lastLoad ? lastLoad.progress.events : '—'}</strong></div>
                      <div><span>Transfer event window</span><strong>{lastLoad ? formatMs(lastLoad.progress.transferWindowMs) : '—'}</strong></div>
                      <div><span>Active backend</span><strong>{lastLoad?.backend ?? '—'}</strong></div>
                    </div>
                    <div className="result-box">
                      {!result && (
                        <div className="result-placeholder">
                          <div className="placeholder-icon"><Sparkles size={22} /></div>
                          <strong>Your result will appear here</strong>
                          <p>Load the model and run an inference to see the output, generated media, and timing details.</p>
                        </div>
                      )}
                      {result && (
                        <div className="result-content">
                          <div className="result-toolbar"><strong>Inference result</strong><span>JSON preview</span></div>
                          {resultMedia.images.length > 0 && (
                            <div className="result-images">
                              {resultMedia.images.map((item, index) => (
                                <div className="result-image" key={item.label + index}>
                                  <img src={item.src} alt={item.label} />
                                  <small>{item.label}</small>
                                </div>
                              ))}
                            </div>
                          )}
                          {resultMedia.audios.length > 0 && resultMedia.audios.map((item, index) => (
                            <div className="generated-audio" key={item.label + index}>
                              <AudioLines size={17} /><span>{item.label}</span><audio controls src={item.src} />
                            </div>
                          ))}
                          <pre className="json-output">{JSON.stringify(toSafeResult(result), null, 2)}</pre>
                        </div>
                      )}
                    </div>
                    <div className="benchmark-note"><Clock3 size={14} /><span>Model load includes any uncached downloads, parsing, and runtime setup. Transfer telemetry is shown only when the library exposes progress events.</span></div>
                  </div>
                </div>

                <div className="learning-grid">
                  <div className="learning-card">
                    <div className="learning-card-title"><Sparkles size={15} /><strong>When to use it</strong></div>
                    <ul>{selectedTask.useCases.map((item) => <li key={item}>{item}</li>)}</ul>
                  </div>
                  <div className="learning-card">
                    <div className="learning-card-title"><BookOpen size={15} /><strong>Scope & suitable inputs</strong></div>
                    <p>{selectedTask.scope}</p>
                  </div>
                  <div className="learning-card warning-card">
                    <div className="learning-card-title"><AlertCircle size={15} /><strong>Limitations to know</strong></div>
                    <p>{selectedTask.limitations}</p>
                  </div>
                </div>
                {selectedRecord && (
                  <div className="history-strip">
                    <div><span className="field-label">LOCAL HISTORY</span><strong>{selectedRecord.loadCount} model initialization(s)</strong></div>
                    <div><span className="field-label">FIRST MEASURED INFERENCE</span><strong>{formatMs(selectedRecord.firstInferenceMs)}</strong></div>
                    <div><span className="field-label">WARM MEDIAN</span><strong>{formatMs(selectedRecord.medianWarmMs)}</strong></div>
                    <button className="text-button" onClick={() => setTab('benchmarks')}>Open benchmark report <ArrowUpRight size={13} /></button>
                  </div>
                )}
              </section>
            </section>
          )}

          {tab === 'benchmarks' && (
            <BenchmarkPanel records={benchmarks} onClear={() => { clearBenchmarks(); setBenchmarks([]); }} />
          )}

          {tab === 'apis' && (
            <ApiPanel />
          )}
        </div>
      </main>

      <footer className="site-footer">
        <div><span className="footer-mark"><BrainCircuit size={15} /></span><strong>Browser ML Lab</strong><span>Built with Transformers.js 4.3.2</span></div>
        <p>Inference runs in your browser. Models are downloaded from Hugging Face Hub and cached by the browser when available.</p>
        <a href="https://huggingface.co/docs/transformers.js" target="_blank" rel="noreferrer">Official documentation <ArrowUpRight size={13} /></a>
      </footer>
    </div>
  );
}

function BenchmarkPanel({ records, onClear }: { records: BenchmarkRecord[]; onClear: () => void }) {
  const tested = records.length;
  const warmValues = records.flatMap((record) => record.warmInferenceMs);
  const totalRuns = records.reduce((sum, record) => sum + record.warmInferenceMs.length + (record.firstInferenceMs === null ? 0 : 1), 0);
  return (
    <section className="report-page">
      <div className="page-heading">
        <div><span className="section-kicker">MEASURED ON YOUR DEVICE</span><h2>Performance report</h2><p>Every number below is captured in this browser session history. Results stay on this device and can be exported for release notes.</p></div>
        <div className="report-actions">
          <button className="button button-outline" onClick={() => downloadTextFile('transformersjs-benchmarks.json', exportBenchmarks(records), 'application/json')} disabled={!records.length}><ArrowDownToLine size={15} /> Export JSON</button>
          <button className="button button-outline" onClick={() => downloadTextFile('transformersjs-benchmarks.csv', exportBenchmarksCsv(records), 'text/csv')} disabled={!records.length}><ArrowDownToLine size={15} /> Export CSV</button>
        </div>
      </div>
      <div className="report-stats">
        <div className="report-stat"><span>Tested configurations</span><strong>{tested}</strong><small>Task + model + backend + dtype</small></div>
        <div className="report-stat"><span>Inference runs recorded</span><strong>{totalRuns}</strong><small>First and warm runs</small></div>
        <div className="report-stat"><span>Warm-run samples</span><strong>{warmValues.length}</strong><small>Across all tested configurations</small></div>
        <div className="report-stat"><span>Supported task demos</span><strong>26</strong><small>25 pipeline IDs + sentence similarity</small></div>
      </div>
      <div className="report-disclaimer">
        <div className="disclaimer-icon"><ShieldCheck size={18} /></div>
        <div><strong>Honest timing, not synthetic benchmarks</strong><p>There are no hard-coded “typical” timings. Model load + runtime setup includes network transfer on a cache miss, model parsing, and session initialization. Progress callbacks may expose transfer bytes and a progress window, but cannot always isolate pure download time. For a true cold test, clear the site's cached model assets before measuring.</p></div>
      </div>
      <div className="report-section-head"><div><h3>Measured configurations</h3><p>One row per task, model, backend, and dtype combination.</p></div><button className="text-button danger-text" onClick={onClear} disabled={!records.length}><X size={13} /> Clear local history</button></div>
      {records.length === 0 ? (
        <div className="empty-report">
          <div className="empty-report-icon"><Gauge size={24} /></div>
          <h3>No benchmark data yet</h3>
          <p>Open any pipeline demo, load a model, run it once, then use the benchmark button to collect one first inference and five warm runs.</p>
          <span>Benchmark records are stored locally in this browser only.</span>
        </div>
      ) : (
        <div className="table-wrap">
          <table className="benchmark-table">
            <thead><tr><th>Task / model</th><th>Backend</th><th>Model load + setup</th><th>Transfer telemetry</th><th>First inference</th><th>Warm median</th><th>Warm runs</th><th>Updated</th></tr></thead>
            <tbody>
              {records.map((record) => (
                <tr key={record.key}>
                  <td><strong>{record.taskName}</strong><small>{record.model}</small></td>
                  <td><Pill tone={record.backend === 'webgpu' ? 'green' : 'blue'}>{record.backend}</Pill><small>{record.dtype}</small></td>
                  <td><strong>{formatMs(record.loadMs)}</strong><small>{record.loadCount} init(s)</small></td>
                  <td><strong>{formatBytes(record.transferBytes)}</strong><small>{record.transferEvents} events · {formatMs(record.transferWindowMs)}</small></td>
                  <td>{formatMs(record.firstInferenceMs)}</td>
                  <td><strong>{formatMs(record.medianWarmMs)}</strong></td>
                  <td>{record.warmInferenceMs.length}</td>
                  <td><small>{new Date(record.updatedAt).toLocaleString()}</small></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="report-method-grid">
        <div className="learning-card"><div className="learning-card-title"><Clock3 size={15} /><strong>First use</strong></div><p>Load model + runtime setup, observed transfer bytes/events, and the first completed inference. A browser-cache hit is not a cold-download measurement.</p></div>
        <div className="learning-card"><div className="learning-card-title"><Activity size={15} /><strong>Repeat use</strong></div><p>Run the same input repeatedly after the model is ready. The report stores individual warm timings and calculates their median.</p></div>
        <div className="learning-card"><div className="learning-card-title"><Monitor size={15} /><strong>Device context</strong></div><p>Browser user agent, hardware concurrency, device memory when exposed, backend, dtype, model ID, and update time are included in exports.</p></div>
      </div>
    </section>
  );
}

function ApiPanel() {
  const [selectedApiId, setSelectedApiId] = useState('tokenizer');
  const selectedApi = ADVANCED_APIS.find((item) => item.id === selectedApiId) ?? ADVANCED_APIS[0];
  const [apiText, setApiText] = useState('Transformers.js turns text into tokens that a model can understand.');
  const [configModel, setConfigModel] = useState('Xenova/bert-base-uncased');
  const [configRevision, setConfigRevision] = useState('main');
  const [localFilesOnly, setLocalFilesOnly] = useState(false);
  const [registryTask, setRegistryTask] = useState('feature-extraction');
  const [registryDtype, setRegistryDtype] = useState<DTypePreference>('auto');
  const [registryRevision, setRegistryRevision] = useState('main');
  const [registryLocalOnly, setRegistryLocalOnly] = useState(false);
  const [multimodalKind, setMultimodalKind] = useState<'image' | 'audio' | 'video'>('image');
  const [processorKind, setProcessorKind] = useState<'image' | 'audio'>('image');
  const [runtimeSettings, setRuntimeSettings] = useState({
    useBrowserCache: true,
    useWasmCache: true,
    allowRemoteModels: true,
    allowLocalModels: false,
    localModelPath: '/models/',
    logLevel: 30,
  });
  const [apiFile, setApiFile] = useState<File | null>(null);
  const [apiFileUrl, setApiFileUrl] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const [apiError, setApiError] = useState('');
  const [apiProgress, setApiProgress] = useState<ProgressSnapshot | null>(null);
  const [apiResult, setApiResult] = useState<any>(null);
  const [apiMedia, setApiMedia] = useState<OutputMedia>(EMPTY_MEDIA);
  const [streamedText, setStreamedText] = useState('');

  useEffect(() => {
    if (!apiFile) {
      setApiFileUrl(null);
      return;
    }
    const url = URL.createObjectURL(apiFile);
    setApiFileUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [apiFile]);

  useEffect(() => {
    return () => apiMedia.audios.forEach((audio) => URL.revokeObjectURL(audio.src));
  }, [apiMedia]);

  const snippets: Record<string, string> = {
    "tokenizer": "import { AutoTokenizer } from '@huggingface/transformers';\n\nconst tokenizer = await AutoTokenizer.from_pretrained('onnx-community/SmolLM2-135M-Instruct-ONNX-MHA');\nconst encoded = await tokenizer('Hello from the browser');\nconst ids = encoded.input_ids.tolist()[0];\nconst tokens = tokenizer.convert_ids_to_tokens(ids);\nconst decoded = tokenizer.decode(ids);\nconst chat = tokenizer.apply_chat_template(\n  [{ role: 'user', content: 'Hello' }],\n  { tokenize: false, add_generation_prompt: true },\n);",
    "processor": "import { AutoProcessor, AutoFeatureExtractor, RawImage, load_audio } from '@huggingface/transformers';\n\nconst processor = await AutoProcessor.from_pretrained('Xenova/clip-vit-base-patch32');\nconst image = await RawImage.read(fileBlob);\nconst imageInputs = await processor(image);\nconst audioExtractor = await AutoFeatureExtractor.from_pretrained('onnx-community/whisper-tiny.en');\nconst audio = await load_audio(audioUrl, 16000);\nconst audioInputs = await audioExtractor(audio);\nconsole.log(imageInputs.pixel_values.dims, audioInputs.input_features.dims);",
    "tensor": "import { Tensor, cat, stack, mean_pooling, layer_norm, std_mean, mean, full, full_like, ones, ones_like, zeros, zeros_like, rand, randn, quantize_embeddings, permute, interpolate, cos_sim, dot, softmax, log_softmax } from '@huggingface/transformers';\n\nconst tensor = new Tensor('float32', new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);\nconst a = new Tensor('float32', new Float32Array([1, 2, 3, 4]), [2, 2]);\nconst b = new Tensor('float32', new Float32Array([5, 6, 7, 8]), [2, 2]);\nconst pooled = mean_pooling(new Tensor('float32', new Float32Array([1, 2, 3, 4, 5, 6]), [1, 2, 3]), new Tensor('int64', new BigInt64Array([1n, 1n]), [1, 2]));\nconst quantizable = new Tensor('float32', new Float32Array([1, -1, 0.5, -0.5, 2, -2, 3, -3]), [1, 8]);\nconsole.log(tensor.sum().item(), tensor.mean().item(), tensor.norm(2).item());\nconsole.log(tensor.transpose(1, 0).tolist(), tensor.normalize(2, 1).tolist(), tensor.clamp(0, 4).tolist(), tensor.flatten(0, 1).tolist());\nconsole.log(cat([a, b], 0).dims, stack([a, b], 0).dims, layer_norm(a, [2]).tolist(), std_mean(a, null, 1, false).map(x => x.tolist()));\nconsole.log(full([2, 2], 3).tolist(), full_like(a, 0).tolist(), ones([2, 2]).tolist(), ones_like(a).tolist(), zeros([2, 2]).tolist(), zeros_like(a).tolist());\nconsole.log(rand([2, 2]).tolist(), randn([2, 2]).tolist(), quantize_embeddings(quantizable, 'binary').tolist());\nconsole.log(softmax([1, 2]), log_softmax([1, 2]), dot([1, 2], [3, 4]), cos_sim([1, 2], [3, 4]));\nconsole.log(permute(tensor, [1, 0]).dims, interpolate(new Tensor('float32', new Float32Array([1, 2, 3, 4]), [1, 2, 2]), [4, 4]).dims);",
    "image-io": "import { RawImage, load_image } from '@huggingface/transformers';\n\nconst image = await load_image(fileBlob);\nconst resized = await image.clone().resize(320, -1);\nconst grayscale = image.clone().grayscale();\nconst rgba = image.clone().rgba();\nconsole.log(resized.width, resized.height, grayscale.channels, rgba.channels);",
    "audio-io": "import { load_audio, RawAudio } from '@huggingface/transformers';\n\nconst samples = await load_audio(audioUrl, 16000);\nconst audio = new RawAudio(samples, 16000);\nconst wavBlob = audio.toBlob();\nconsole.log(samples.length / 16000, audio.sampling_rate, wavBlob.size);",
    "video-io": "import { load_video } from '@huggingface/transformers';\n\nconst video = await load_video(videoFile, { fps: 1, num_frames: 8 });\nconsole.log(video.width, video.height, video.fps, video.duration);\nconsole.log(video.frames.map((frame) => frame.timestamp));",
    "runtime": "import { env, LogLevel } from '@huggingface/transformers';\n\nconsole.log(env.version);\nconsole.log(env.backends.onnx.getSupportedDevices?.());\nenv.useBrowserCache = true;\nenv.useWasmCache = true;\nenv.allowLocalModels = true;\nenv.allowRemoteModels = true;\nenv.localModelPath = '/models/';\nenv.logLevel = LogLevel.WARNING;\n\n// Optional custom fetch (e.g. headers / auth proxy). Never expose private tokens in a public browser bundle.\nenv.fetch = (url, options) => fetch(url, { ...options, headers: { ...options?.headers, 'X-Client': 'transformers-playground' } });\n\n// Optional Cache API-compatible custom cache.\nconst cache = await caches.open('transformers-models');\nenv.useCustomCache = true;\nenv.customCache = { match: (request) => cache.match(request), put: (request, response) => cache.put(request, response) };",
    "model-registry": "import { ModelRegistry } from '@huggingface/transformers';\n\nconst task = 'feature-extraction';\nconst modelId = 'onnx-community/all-MiniLM-L6-v2-ONNX';\nconst options = { dtype: 'q4', revision: 'main' };\nconst allFiles = await ModelRegistry.get_files(modelId, options);\nconst pipelineFiles = await ModelRegistry.get_pipeline_files(task, modelId, options);\nconst modelFiles = await ModelRegistry.get_model_files(modelId, options);\nconst tokenizerFiles = await ModelRegistry.get_tokenizer_files(modelId);\nconst processorFiles = await ModelRegistry.get_processor_files(modelId);\nconst metadata = await Promise.all(allFiles.map((file) => ModelRegistry.get_file_metadata(modelId, file, options)));\nconst totalBytes = metadata.reduce((sum, item) => sum + (item?.size ?? 0), 0);\nconst dtypes = await ModelRegistry.get_available_dtypes(modelId, options);\nconst modelCache = await ModelRegistry.is_cached(modelId, options);\nconst modelCacheFiles = await ModelRegistry.is_cached_files(modelId, options);\nconst pipelineCache = await ModelRegistry.is_pipeline_cached(task, modelId, options);\nconst pipelineCacheFiles = await ModelRegistry.is_pipeline_cached_files(task, modelId, options);\nconsole.log({ allFiles, pipelineFiles, modelFiles, tokenizerFiles, processorFiles, metadata, totalBytes, dtypes, modelCache, modelCacheFiles, pipelineCache, pipelineCacheFiles });\n// ModelRegistry.clear_pipeline_cache(task, modelId, options);\n// ModelRegistry.clear_cache(modelId, options);",
    "direct-models": "import { AutoImageProcessor, AutoModelForImageClassification, RawImage, softmax } from '@huggingface/transformers';\n\nconst modelId = 'Xenova/vit-base-patch16-224';\nconst processor = await AutoImageProcessor.from_pretrained(modelId);\nconst model = await AutoModelForImageClassification.from_pretrained(modelId, { device: 'webgpu' });\nconst image = await RawImage.read(fileBlob);\nconst inputs = await processor(image);\nconst { logits } = await model(inputs);\nconst probabilities = Array.from(softmax(logits.tolist()[0]));\nconsole.log(probabilities);",
    "config-inspector": "import { AutoConfig } from '@huggingface/transformers';\n\nconst config = await AutoConfig.from_pretrained('Xenova/bert-base-uncased', {\n  revision: 'main',\n  local_files_only: false,\n  progress_callback: (event) => console.log(event),\n});\nconsole.log(config.model_type, config.architectures, config.hidden_size);\nconsole.log(config.id2label, config.vocab_size);",
    "generation": "import { pipeline, TextStreamer, WhisperTextStreamer } from '@huggingface/transformers';\n\n// Text generation: stream completed words into a chat UI.\nconst generator = await pipeline('text-generation', 'onnx-community/SmolLM2-135M-ONNX');\nconst textStreamer = new TextStreamer(generator.tokenizer, {\n  skip_prompt: true,\n  skip_special_tokens: true,\n  callback_function: (text) => console.log(text),\n});\nawait generator('A browser can run models locally', { max_new_tokens: 64, streamer: textStreamer });\n\n// Speech recognition: stream transcript chunks and audio-time boundaries.\nconst transcriber = await pipeline('automatic-speech-recognition', 'onnx-community/whisper-tiny.en');\nconst whisperStreamer = new WhisperTextStreamer(transcriber.tokenizer, {\n  callback_function: (text) => console.log('transcript:', text),\n  on_chunk_start: (seconds) => console.log('chunk start:', seconds),\n  on_chunk_end: (seconds) => console.log('chunk end:', seconds),\n  on_finalize: () => console.log('transcription complete'),\n});\nawait transcriber(audioUrl, {\n  chunk_length_s: 30,\n  stride_length_s: 5,\n  streamer: whisperStreamer,\n});",
    "random": "import { random } from '@huggingface/transformers';\n\nconst rng = new random.Random(42);\nconst values = Array.from({ length: 6 }, () => rng.random());\nconst shuffled = [1, 2, 3, 4, 5, 6];\nrng.shuffle(shuffled);\nconst weightedChoice = rng.choices(['text', 'image', 'audio'], [5, 3, 2]);\nconst normalSample = rng.gauss(0, 1);\nconsole.log(values, shuffled, weightedChoice, normalSample);",
    "structured-output": "import { pipeline } from '@huggingface/transformers';\nimport { StructuredOutputProcessor } from '@huggingface/transformers-structured-output';\n\nconst generator = await pipeline('text-generation', 'onnx-community/LFM2.5-350M-ONNX', {\n  device: 'webgpu',\n  dtype: 'q4f16',\n});\nconst processor = new StructuredOutputProcessor(generator.tokenizer, {\n  type: 'json_schema',\n  json_schema: {\n    type: 'object',\n    properties: {\n      sentiment: { enum: ['positive', 'negative', 'neutral'] },\n      topic: { enum: ['price', 'quality', 'delivery', 'other'] },\n    },\n    required: ['sentiment', 'topic'],\n    additionalProperties: false,\n  },\n});\nconst output = await generator([{ role: 'user', content: 'Classify: The product arrived late.' }], {\n  max_new_tokens: 256,\n  do_sample: false,\n  logits_processor: [processor],\n});",
    "multimodal-embeddings": "import { AutoModel, AutoProcessor, load_image, load_audio, load_video, cos_sim } from '@huggingface/transformers';\n\nconst modelId = 'onnx-community/embeddinggemma-2-ONNX';\nconst processor = await AutoProcessor.from_pretrained(modelId);\nconst model = await AutoModel.from_pretrained(modelId, { device: 'webgpu', dtype: 'q4' });\nconst embed = async (...inputs) => (await model(await processor(...inputs))).sentence_embedding;\n\nconst query = await embed(['task: search result | query: cats sleeping on a couch']);\nconst image = await load_image(imageUrl);\nconst imageEmbedding = await embed(null, image);\nconst audio = await load_audio(audioUrl, 16000);\nconst audioEmbedding = await embed(null, null, audio);\nconst video = await load_video(videoUrl, { fps: 1 });\nconst videoEmbedding = await embed(null, null, null, video);\n\nconsole.log(cos_sim(query.tolist()[0], imageEmbedding.tolist()[0]));",
  };

  const lfm3VlSnippet = "import { AutoModelForImageTextToText, AutoProcessor, load_image } from '@huggingface/transformers';\n\nconst modelId = 'onnx-community/LFM3-VL-2B-ONNX';\nconst model = await AutoModelForImageTextToText.from_pretrained(modelId, {\n  device: 'webgpu',\n  dtype: {\n    vision_encoder: 'fp16',\n    embed_tokens: 'fp16',\n    decoder_model_merged: 'q4f16',\n  },\n});\nconst processor = await AutoProcessor.from_pretrained(modelId);\nconst image = await load_image(imageUrl);\nconst messages = [{\n  role: 'user',\n  content: [{ type: 'image' }, { type: 'text', text: 'What is in this image?' }],\n}];\nconst chatPrompt = processor.apply_chat_template(messages, {\n  add_generation_prompt: true,\n  enable_thinking: false,\n});\nconst inputs = await processor(image, chatPrompt, { add_special_tokens: false });\nconst outputs = await model.generate({ ...inputs, max_new_tokens: 128 });\nconst inputLength = inputs.input_ids.dims.at(-1);\nconst generated = outputs.slice(null, [inputLength, null]);\nconsole.log(processor.batch_decode(generated, { skip_special_tokens: true })[0]);\nawait model.dispose();";

  async function copyLfm3VlSnippet() {
    try {
      await navigator.clipboard.writeText(lfm3VlSnippet);
    } catch {
      setApiError('Clipboard access is unavailable in this browser. Select and copy the LFM3-VL code manually.');
    }
  }

  async function runApiDemo() {
    if (apiStatus === 'loading') return;
    if ((selectedApiId === 'structured-output' || selectedApiId === 'multimodal-embeddings') && !apiText.trim()) {
      setApiError('Enter a text prompt or query first.');
      return;
    }
    if (selectedApiId === 'model-registry' && (!configModel.trim() || !registryTask.trim())) {
      setApiError('Enter both a model repository and pipeline task ID.');
      return;
    }
    if (selectedApiId === 'multimodal-embeddings' && !apiFile) {
      setApiError('Choose a media file for the cross-modal comparison.');
      return;
    }
    setApiError('');
    setApiStatus('loading');
    setApiProgress(null);
    setApiResult(null);
    setApiMedia(EMPTY_MEDIA);
    setStreamedText('');
    const startedAt = performance.now();
    try {
      let output: any;
      if (selectedApiId === 'tokenizer') {
        output = await runTokenizerLab(apiText, setApiProgress);
      } else if (selectedApiId === 'processor') {
        if (processorKind === 'audio') {
          output = await runAudioFeatureExtractorLab(apiFileUrl ?? SAMPLE_AUDIO, setApiProgress);
        } else {
          if (!apiFile) throw new Error('Choose an image file to inspect the processor output.');
          output = await runProcessorLab(apiFile, setApiProgress);
        }
      } else if (selectedApiId === 'image-io') {
        if (!apiFile) throw new Error('Choose an image file to run the RawImage I/O demo.');
        output = await runImageIOLab(apiFile);
      } else if (selectedApiId === 'audio-io') {
        output = await runAudioIOLab(apiFileUrl ?? SAMPLE_AUDIO);
      } else if (selectedApiId === 'video-io') {
        if (!apiFile) throw new Error('Choose a short video file to sample frames.');
        output = await runVideoIOLab(apiFile);
      } else if (selectedApiId === 'tensor') {
        output = runTensorLab();
      } else if (selectedApiId === 'runtime') {
        output = await applyRuntimeSettings(runtimeSettings);
      } else if (selectedApiId === 'model-registry') {
        output = await runModelRegistryLab(registryTask, configModel, registryDtype, 'auto', registryRevision, registryLocalOnly);
      } else if (selectedApiId === 'direct-models') {
        if (!apiFile) throw new Error('Choose an image to run the direct AutoModel example.');
        output = await runDirectModelLab(apiFile, 'auto', 'auto', setApiProgress);
      } else if (selectedApiId === 'config-inspector') {
        output = await runConfigInspector(configModel, setApiProgress, configRevision, localFilesOnly);
      } else if (selectedApiId === 'random') {
        output = runRandomLab(42);
      } else if (selectedApiId === 'generation') {
        const generatorDefinition = TASKS.find((task) => task.id === 'text-generation')!;
        const generated = await runStreamingGeneration(
          generatorDefinition,
          apiText,
          'auto',
          'auto',
          (token) => setStreamedText((current) => current + token),
          setApiProgress,
        );
        output = {
          output: generated.output,
          loadMs: generated.loaded.loadMs,
          inferenceMs: generated.inferenceMs,
          backend: generated.loaded.backend,
          cacheHit: generated.loaded.cacheHit,
          transferBytes: generated.loaded.progress.transferBytes,
        };
      } else if (selectedApiId === 'structured-output') {
        const generated = await runStructuredOutputLab(apiText, setApiProgress);
        output = {
          output: generated.output,
          loadMs: generated.loaded.loadMs,
          inferenceMs: generated.inferenceMs,
          backend: generated.loaded.backend,
          cacheHit: generated.loaded.cacheHit,
          transferBytes: generated.loaded.progress.transferBytes,
        };
      } else if (selectedApiId === 'multimodal-embeddings') {
        if (!apiFileUrl) throw new Error('Choose an image, audio, or video file for the cross-modal comparison.');
        output = await runMultimodalEmbeddingLab(multimodalKind, apiFileUrl, apiText, setApiProgress);
      }
      const found = collectOutputMedia(output);
      setApiMedia({
        images: found.images.slice(0, 6).map((item) => ({ label: item.label, src: rawImageToDataUrl(item.image) })),
        audios: found.audios.slice(0, 3).map((item) => ({ label: item.label, src: rawAudioToUrl(item.audio) })),
      });
      setApiResult({ elapsedMs: performance.now() - startedAt, result: toSafeResult(output) });
      setApiStatus('ready');
    } catch (cause) {
      setApiStatus('error');
      setApiError(cause instanceof Error ? cause.message : String(cause));
    }
  }

  async function handleClearRegistryCache(scope: 'pipeline' | 'model') {
    const label = scope === 'model' ? 'all model component' : 'selected pipeline';
    if (!window.confirm('Clear the ' + label + ' cache for ' + configModel + '? The next run may need to download files again.')) return;
    setApiStatus('loading');
    setApiError('');
    setApiResult(null);
    setApiMedia(EMPTY_MEDIA);
    try {
      const output = await clearModelRegistryCache(registryTask, configModel, registryDtype, scope, registryRevision);
      setApiResult({ elapsedMs: 0, result: toSafeResult(output) });
      setApiStatus('ready');
    } catch (cause) {
      setApiStatus('error');
      setApiError(cause instanceof Error ? cause.message : String(cause));
    }
  }

  async function copySnippet() {
    try {
      await navigator.clipboard.writeText(snippets[selectedApiId] ?? '');
    } catch {
      setApiError('Clipboard access is unavailable in this browser. Select and copy the code manually.');
    }
  }

  const needsImage = (selectedApiId === 'processor' && processorKind === 'image') ||
    selectedApiId === 'image-io' || selectedApiId === 'direct-models' ||
    (selectedApiId === 'multimodal-embeddings' && multimodalKind === 'image');
  const needsVideo = selectedApiId === 'video-io' ||
    (selectedApiId === 'multimodal-embeddings' && multimodalKind === 'video');
  const needsAudio = selectedApiId === 'audio-io' ||
    (selectedApiId === 'processor' && processorKind === 'audio') ||
    (selectedApiId === 'multimodal-embeddings' && multimodalKind === 'audio');
  const needsFile = needsImage || needsVideo || needsAudio;
  const apiBusy = apiStatus === 'loading';

  return (
    <section className="api-page">
      <div className="page-heading">
        <div><span className="section-kicker">BEYOND PIPELINES</span><h2>Developer API lab</h2><p>Explore the lower-level building blocks behind Transformers.js: tokenizers, processors, direct models, tensors, media I/O, runtime/cache control, generation, and multimodal embeddings.</p></div>
        <Pill tone="blue">{ADVANCED_APIS.length} API areas</Pill>
      </div>
      <div className="api-workspace">
        <aside className="api-index">
          {ADVANCED_APIS.map((api) => (
            <button key={api.id} className={selectedApiId === api.id ? 'api-index-row selected' : 'api-index-row'} onClick={() => { setSelectedApiId(api.id); setApiStatus('idle'); setApiResult(null); setApiMedia(EMPTY_MEDIA); setApiError(''); setApiFile(null); setApiProgress(null); setStreamedText(''); }}>
              <span className="api-index-icon">
                {api.id === 'tokenizer' ? <Terminal size={16} />
                  : api.id === 'tensor' ? <Layers size={16} />
                  : api.id === 'image-io' ? <ImageIcon size={16} />
                  : api.id === 'audio-io' ? <FileAudio size={16} />
                  : api.id === 'video-io' ? <FileVideo size={16} />
                  : api.id === 'runtime' ? <Cpu size={16} />
                  : api.id === 'model-registry' ? <Database size={16} />
                  : api.id === 'direct-models' || api.id === 'config-inspector' ? <BrainCircuit size={16} />
                  : api.id === 'random' ? <Activity size={16} />
                  : api.id === 'structured-output' ? <Code2 size={16} />
                  : api.id === 'multimodal-embeddings' ? <Layers size={16} />
                  : api.id === 'generation' ? <Zap size={16} />
                  : <Settings2 size={16} />}
              </span>
              <span><strong>{api.title}</strong><small>{api.id.replaceAll('-', ' ')}</small></span>
              <ChevronRight size={14} />
            </button>
          ))}
        </aside>
        <div className="api-detail">
          <div className="api-detail-heading">
            <div><span className="section-kicker">API EXPLORER</span><h3>{selectedApi.title}</h3><p>{selectedApi.description}</p></div>
            <a href="https://huggingface.co/docs/transformers.js/api/transformers" target="_blank" rel="noreferrer" className="docs-link">API reference <ArrowUpRight size={13} /></a>
          </div>
          <div className="api-info-grid">
            <div><span>USE CASES</span><p>{selectedApi.use}</p></div>
            <div><span>SCOPE</span><p>{selectedApi.scope}</p></div>
            <div className="api-limit"><span>LIMITATIONS</span><p>{selectedApi.limitation}</p></div>
          </div>
          <div className="api-demo-card">
            <div className="api-demo-top"><div><span className="section-kicker">LIVE EXAMPLE</span><h4>Try this API</h4></div><Pill tone={apiStatus === 'ready' ? 'green' : apiStatus === 'error' ? 'amber' : 'neutral'}>{apiStatus === 'ready' ? 'Complete' : apiStatus === 'loading' ? 'Running' : apiStatus === 'error' ? 'Error' : 'Ready'}</Pill></div>
            {selectedApiId === 'tokenizer' && (
              <div className="field-stack"><label className="field-label">Text to tokenize</label><textarea className="text-area" rows={4} value={apiText} onChange={(event) => setApiText(event.target.value)} /></div>
            )}
            {selectedApiId === 'generation' && (
              <div className="field-stack"><label className="field-label">Generation prompt</label><textarea className="text-area" rows={4} value={apiText} onChange={(event) => setApiText(event.target.value)} /><div className="field-hint">The demo streams tokens into the output as they are produced.</div></div>
            )}
            {selectedApiId === 'processor' && (
              <div className="field-stack">
                <label className="field-label">Processor type</label>
                <select className="text-input" value={processorKind} onChange={(event) => { setProcessorKind(event.target.value as 'image' | 'audio'); setApiFile(null); }}>
                  <option value="image">AutoProcessor — image input (CLIP)</option>
                  <option value="audio">AutoFeatureExtractor — audio input (Whisper)</option>
                </select>
                <div className="field-hint">Compare multimodal/image processors with audio-only feature extractors. The audio mode uses the built-in JFK sample when no file is uploaded.</div>
              </div>
            )}
            {selectedApiId === 'structured-output' && (
              <div className="field-stack"><label className="field-label">Feedback to classify</label><textarea className="text-area" rows={4} value={apiText} onChange={(event) => setApiText(event.target.value)} /><div className="field-hint">Returns a constrained JSON object with sentiment and topic. Requires WebGPU.</div></div>
            )}
            {selectedApiId === 'multimodal-embeddings' && (
              <div className="field-stack">
                <label className="field-label">Text query</label>
                <textarea className="text-area" rows={3} value={apiText} onChange={(event) => setApiText(event.target.value)} />
                <label className="field-label">Media type to compare</label>
                <select className="text-input" value={multimodalKind} onChange={(event) => { setMultimodalKind(event.target.value as 'image' | 'audio' | 'video'); setApiFile(null); }}>
                  <option value="image">Image</option><option value="audio">Audio</option><option value="video">Video</option>
                </select>
                <div className="field-hint">EmbeddingGemma 2 has 740M parameters. First use can download several hundred megabytes.</div>
              </div>
            )}
            {selectedApiId === 'model-registry' && (
              <div className="field-stack">
                <label className="field-label">Model repository</label>
                <input className="text-input" value={configModel} onChange={(event) => setConfigModel(event.target.value)} />
                <label className="field-label">Pipeline task ID</label>
                <input className="text-input" value={registryTask} onChange={(event) => setRegistryTask(event.target.value)} />
                <label className="field-label">Dtype filter</label>
                <select className="text-input" value={registryDtype} onChange={(event) => setRegistryDtype(event.target.value as DTypePreference)}>
                  <option value="auto">All / model default</option><option value="q4">q4</option><option value="q8">q8</option><option value="fp16">fp16</option><option value="fp32">fp32</option>
                </select>
                <label className="field-label">Hub revision</label>
                <input className="text-input" value={registryRevision} onChange={(event) => setRegistryRevision(event.target.value)} />
                <label className="field-label"><input type="checkbox" checked={registryLocalOnly} onChange={(event) => setRegistryLocalOnly(event.target.checked)} /> Local files only</label>
              </div>
            )}
            {selectedApiId === 'config-inspector' && (
              <div className="field-stack">
                <label className="field-label">Model repository to inspect</label>
                <input className="text-input" value={configModel} onChange={(event) => setConfigModel(event.target.value)} />
                <label className="field-label">Hub revision / branch</label>
                <input className="text-input" value={configRevision} onChange={(event) => setConfigRevision(event.target.value)} />
                <label className="field-label"><input type="checkbox" checked={localFilesOnly} onChange={(event) => setLocalFilesOnly(event.target.checked)} /> Local files only (no Hub network request)</label>
                <div className="field-hint">AutoConfig fetches model metadata only; it does not download model weights. Revision and local_files_only are forwarded to from_pretrained().</div>
              </div>
            )}
            {needsFile && (
              <div className="field-stack">
                <label className="field-label">{needsVideo ? 'Short video file' : needsAudio ? 'Audio file' : 'Image file'}</label>
                <label className="upload-zone api-upload">
                  <input type="file" accept={needsVideo ? 'video/*' : needsAudio ? 'audio/*' : 'image/*'} onChange={(event) => { setApiFile(event.target.files?.[0] ?? null); setApiError(''); }} />
                  <span className="upload-icon">{needsVideo ? <FileVideo size={18} /> : needsAudio ? <FileAudio size={18} /> : <FileImage size={18} />}</span>
                  <strong>{apiFile ? apiFile.name : needsVideo ? 'Choose a short video' : needsAudio ? 'Choose an audio clip' : 'Choose an image'}</strong>
                  <small>{apiFile ? formatBytes(apiFile.size) : needsVideo ? 'MP4 or WebM · keep clips short' : needsAudio ? 'WAV, MP3, FLAC' : 'JPG, PNG, WebP'}</small>
                </label>
                {apiFileUrl && needsImage && <img className="api-file-preview" src={apiFileUrl} alt="API input preview" />}
                {apiFileUrl && needsVideo && <video className="api-file-preview" src={apiFileUrl} controls />}
                {apiFileUrl && needsAudio && <audio className="api-audio-preview" src={apiFileUrl} controls />}
                {selectedApiId === 'audio-io' && <div className="field-hint">Optional upload. If no file is selected, the demo uses the built-in speech sample.</div>}
              </div>
            )}
            {selectedApiId === 'runtime' && (
              <div className="field-stack">
                <div className="runtime-settings-grid">
                  <label><input type="checkbox" checked={runtimeSettings.useBrowserCache} onChange={(event) => setRuntimeSettings((current) => ({ ...current, useBrowserCache: event.target.checked }))} /> Browser model cache</label>
                  <label><input type="checkbox" checked={runtimeSettings.useWasmCache} onChange={(event) => setRuntimeSettings((current) => ({ ...current, useWasmCache: event.target.checked }))} /> WASM runtime cache</label>
                  <label><input type="checkbox" checked={runtimeSettings.allowRemoteModels} onChange={(event) => setRuntimeSettings((current) => ({ ...current, allowRemoteModels: event.target.checked }))} /> Allow Hub downloads</label>
                  <label><input type="checkbox" checked={runtimeSettings.allowLocalModels} onChange={(event) => setRuntimeSettings((current) => ({ ...current, allowLocalModels: event.target.checked }))} /> Allow local model files</label>
                </div>
                <label className="field-label">Local model path</label>
                <input className="text-input" value={runtimeSettings.localModelPath} onChange={(event) => setRuntimeSettings((current) => ({ ...current, localModelPath: event.target.value }))} />
                <label className="field-label">Library log level</label>
                <select className="text-input" value={runtimeSettings.logLevel} onChange={(event) => setRuntimeSettings((current) => ({ ...current, logLevel: Number(event.target.value) }))}>
                  <option value={10}>DEBUG</option><option value={20}>INFO</option><option value={30}>WARNING</option><option value={40}>ERROR</option><option value={50}>NONE</option>
                </select>
                <div className="runtime-hint"><Monitor size={17} /><span>Applying settings mutates the global Transformers.js env for this tab and affects later demos. If both remote and local models are disabled, model loading will fail.</span></div>
              </div>
            )}
            {selectedApiId === 'tensor' && <div className="runtime-hint"><Layers size={17} /><span>Covers public Tensor operations, reductions, normalization, tensor factories, pooling, layer normalization, quantized embeddings, permutation, interpolation, and vector maths.</span></div>}
            {selectedApiId === 'processor' && <div className="runtime-hint"><Settings2 size={17} /><span>{processorKind === 'image' ? 'Loads a CLIP AutoProcessor and shows the pixel_values tensor shape.' : 'Loads a Whisper AutoFeatureExtractor and shows the input_features tensor shape.'}</span></div>}
            {selectedApiId === 'random' && <div className="runtime-hint"><Activity size={17} /><span>Runs a fixed-seed MT19937 sequence, shuffle, weighted choice, and Gaussian sample. Re-run to verify reproducibility.</span></div>}
            {selectedApiId === 'direct-models' && <div className="runtime-hint"><BrainCircuit size={17} /><span>Loads AutoImageProcessor and AutoModelForImageClassification separately, then runs the model forward pass and decodes top labels.</span></div>}
            {selectedApiId === 'model-registry' && <div className="runtime-hint"><Database size={17} /><span>Lists required model files, available dtypes, and per-file cache status. You can clear cached files after inspection.</span></div>}
            {selectedApiId === 'audio-io' && <div className="runtime-hint"><AudioLines size={17} /><span>Decodes audio to mono 16 kHz samples, reports duration and peak amplitude, and returns a playable WAV preview.</span></div>}
            {selectedApiId === 'multimodal-embeddings' && <div className="runtime-hint"><Layers size={17} /><span>Compares the text query with the selected media in one shared embedding space. This is an embedding model, not a text-generation model.</span></div>}
            {selectedApiId === 'structured-output' && <div className="runtime-hint"><Code2 size={17} /><span>Constrains model output with a JSON Schema logits processor. The live example uses a 350M model and one generated sequence.</span></div>}
            {apiStatus === 'loading' && <ProgressBar progress={apiProgress} />}
            {apiError && <div className="error-banner"><AlertCircle size={16} /><span>{apiError}</span><button onClick={() => setApiError('')}><X size={14} /></button></div>}
            <div className="api-run-row">
              <button className="button button-primary" onClick={runApiDemo} disabled={apiBusy}>
                {apiBusy ? <Loader2 size={16} className="spin" /> : <Play size={15} fill="currentColor" />}
                {apiBusy ? 'Running...' : 'Run API demo'}
              </button>
              <button className="button button-outline" onClick={copySnippet}><Copy size={14} /> Copy code</button>
              {selectedApiId === 'model-registry' && (
                <>
                  <button className="button button-outline" onClick={() => handleClearRegistryCache('pipeline')} disabled={apiBusy}>Clear pipeline cache</button>
                  <button className="button button-outline" onClick={() => handleClearRegistryCache('model')} disabled={apiBusy}>Clear model cache</button>
                </>
              )}
            </div>
            {apiStatus === 'ready' && apiResult && (
              <div className="api-result">
                <div className="timing-grid">
                  <div className="timing-card"><span>Total elapsed</span><strong>{formatMs(apiResult.elapsedMs)}</strong><small>Wall-clock time for this API demo</small></div>
                  <div className="timing-card"><span>Model load / request</span><strong>{formatMs(apiResult.result?.loadMs ?? apiResult.result?.elapsedMs ?? null)}</strong><small>Includes downloads when needed</small></div>
                  <div className="timing-card"><span>Inference / processing</span><strong>{formatMs(apiResult.result?.inferenceMs ?? apiResult.result?.preprocessMs ?? null)}</strong><small>Depends on input and hardware</small></div>
                </div>
                {streamedText && <div className="stream-output"><span>STREAMED TOKENS</span><p>{streamedText}</p></div>}
                {apiMedia.images.length > 0 && <div className="result-images">{apiMedia.images.map((item, index) => <div className="result-image" key={item.label + index}><img src={item.src} alt={item.label} /><small>{item.label}</small></div>)}</div>}
                {apiMedia.audios.map((item, index) => <div className="generated-audio" key={item.label + index}><AudioLines size={17} /><span>{item.label}</span><audio controls src={item.src} /></div>)}
                <pre className="json-output">{JSON.stringify(apiResult.result, null, 2)}</pre>
              </div>
            )}
          </div>
          <div className="code-card">
            <div className="code-card-head"><span><Code2 size={14} /> TRANSFORMERS.JS PATTERN</span><button onClick={copySnippet}><Copy size={13} /> Copy</button></div>
            <pre>{snippets[selectedApiId]}</pre>
          </div>
          {selectedApiId === 'direct-models' && (
            <div className="model-family-card">
              <h4>Additional direct model families</h4>
              <div className="model-family-grid">
                <span>AutoModelForSequenceClassification</span><span>AutoModelForTokenClassification</span><span>AutoModelForSeq2SeqLM</span><span>AutoModelForSpeechSeq2Seq</span><span>AutoModelForTextToSpectrogram</span><span>AutoModelForTextToWaveform</span><span>AutoModelForCausalLM</span><span>AutoModelForMaskedLM</span><span>AutoModelForQuestionAnswering</span><span>AutoModelForVision2Seq</span><span>AutoModelForImageClassification</span><span>AutoModelForImageSegmentation</span><span>AutoModelForSemanticSegmentation</span><span>AutoModelForUniversalSegmentation</span><span>AutoModelForObjectDetection</span><span>AutoModelForZeroShotObjectDetection</span><span>AutoModelForMaskGeneration</span><span>AutoModelForCTC</span><span>AutoModelForAudioClassification</span><span>AutoModelForXVector</span><span>AutoModelForAudioFrameClassification</span><span>AutoModelForDocumentQuestionAnswering</span><span>AutoModelForImageMatting</span><span>AutoModelForImageToImage</span><span>AutoModelForDepthEstimation</span><span>AutoModelForNormalEstimation</span><span>AutoModelForPoseEstimation</span><span>AutoModelForImageFeatureExtraction</span><span>AutoModelForImageTextToText</span><span>AutoModelForAudioTextToText</span><span>LightOnOcr</span><span>LFM3 / LFM3-VL</span>
              </div>
              <p>These are direct model APIs, not generic pipeline wrappers. Each needs model-matched preprocessing, tensors, and output decoding.</p>
            </div>
          )}
          {selectedApiId === 'direct-models' && (
            <div className="code-card">
              <div className="code-card-head"><span><Code2 size={14} /> LFM3-VL · LARGE MULTIMODAL MODEL</span><button onClick={copyLfm3VlSnippet}><Copy size={13} /> Copy example</button></div>
              <pre>{lfm3VlSnippet}</pre>
              <div className="benchmark-note"><AlertCircle size={13} /><span>LFM3-VL-2B-ONNX is a large optional model, requires WebGPU, and can download multiple gigabytes of model artifacts. It is shown as a reference example and is not loaded automatically by this playground.</span></div>
            </div>
          )}
          {selectedApiId === 'generation' && (
            <div className="model-family-card">
              <h4>Generation controls covered by the library</h4>
              <div className="model-family-grid"><span>GenerationConfig</span><span>Sampling & beam search</span><span>Logits processors</span><span>Stopping criteria</span><span>TextStreamer</span><span>WhisperTextStreamer</span><span>KV cache</span><span>Chat templates</span><span>Tool calling</span></div>
              <p>Available controls depend on the model architecture and runtime. The code sample demonstrates incremental text streaming.</p>
            </div>
          )}
          {selectedApiId === 'structured-output' && (
            <div className="model-family-card">
              <h4>Constraint types</h4>
              <div className="model-family-grid"><span>JSON Schema</span><span>JSON object</span><span>Regular expression</span><span>One sequence at a time</span></div>
              <p>Experimental companion package. Some JSON Schema assertions and regex features are intentionally unsupported.</p>
            </div>
          )}
        </div>
      </div>
      <div className="unsupported-section">
        <div className="unsupported-heading"><div><span className="section-kicker">BOUNDARIES MATTER</span><h3>What is not a first-class pipeline?</h3><p>Related ideas that should not be presented as built-in pipeline tasks.</p></div><Pill tone="amber">{UNSUPPORTED_TASKS.length} scope notes</Pill></div>
        <div className="unsupported-grid">
          {UNSUPPORTED_TASKS.map(([name, note]) => <div className="unsupported-item" key={name}><AlertCircle size={14} /><div><strong>{name}</strong><p>{note}</p></div></div>)}
        </div>
      </div>
    </section>
  );
}

export default App;
