import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeftIcon,
  ArrowPathIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
  SunIcon,
  MoonIcon,
} from '@heroicons/react/24/outline';
import {
  getSettings,
  saveSettings,
  getProfiles,
  activateProfile,
  getConfigRaw,
  saveConfig,
  checkDoclingServer,
} from '../api';

// Small presentational helpers ------------------------------------------------
const Field = ({ label, hint, children }) => (
  <div className="mb-5">
    <label className="block text-xs font-bold text-[#4a154b] uppercase tracking-wider mb-1.5">{label}</label>
    {hint && <p className="text-xs text-[#696969] mb-2 leading-relaxed">{hint}</p>}
    {children}
  </div>
);

const inputCls =
  'w-full px-3.5 py-2.5 bg-white border border-[#e6e6e6] rounded-md text-sm text-[#1d1d1d] focus:outline-none focus:border-[#4a154b] focus:ring-2 focus:ring-[#4a154b]/20 shadow-sm transition-all';

const Select = ({ value, onChange, options }) => (
  <select value={value ?? ''} onChange={(e) => onChange(e.target.value)} className={inputCls}>
    {options.map((o) => (
      <option key={o} value={o}>{o}</option>
    ))}
  </select>
);

const Toggle = ({ value, onChange, on = 'On', off = 'Off' }) => (
  <button
    type="button"
    onClick={() => onChange(!value)}
    className={`px-5 py-2 rounded-full text-xs font-bold border transition-all ${
      value
        ? 'border-[#4a154b] bg-[#4a154b] text-white shadow-sm'
        : 'border-[#e6e6e6] bg-[#f9f0ff] text-[#696969] hover:bg-[#f3e2ff]'
    }`}
  >
    {value ? on : off}
  </button>
);

const SettingsPage = () => {
  const navigate = useNavigate();
  const [s, setS] = useState(null); // settings object from /config/settings
  const [profiles, setProfiles] = useState([]);
  const [active, setActive] = useState('');
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [newName, setNewName] = useState('');
  const [newPromptKey, setNewPromptKey] = useState('');
  const [addStepName, setAddStepName] = useState('');
  const [advanced, setAdvanced] = useState(false);
  const [yamlText, setYamlText] = useState('');
  const [serverStatus, setServerStatus] = useState(null); // null | 'checking' | 'online' | 'offline'

  // Theme state
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem('theme') === 'dark';
  });

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [darkMode]);

  const testDoclingServer = async () => {
    setServerStatus('checking');
    try {
      const { data } = await checkDoclingServer(s.docling_server_url, s.docling_mode || 'remote');
      setServerStatus(data.reachable ? 'online' : 'offline');
    } catch {
      setServerStatus('offline');
    }
  };

  useEffect(() => { load(); }, []);

  const load = async () => {
    setLoading(true);
    try {
      const [{ data: settings }, { data: profs }] = await Promise.all([
        getSettings(),
        getProfiles(),
      ]);
      setS(settings);
      setActive(settings._active || profs.active);
      setProfiles(profs.profiles || []);
      setStatus(null);
    } catch (e) {
      setStatus({ type: 'err', msg: e.message || 'Failed to load settings' });
    } finally {
      setLoading(false);
    }
  };

  const errMsg = (e) => e.response?.data?.detail || e.message || 'Request failed';
  const set = (k, v) => setS((prev) => ({ ...prev, [k]: v }));

  // vision prompts (object: route/industry/"default" -> focus text)
  const prompts = (s && s.vision_prompts) || {};
  const setPrompt = (key, val) =>
    setS((p) => ({ ...p, vision_prompts: { ...p.vision_prompts, [key]: val } }));
  const removePrompt = (key) =>
    setS((p) => {
      const np = { ...p.vision_prompts };
      delete np[key];
      return { ...p, vision_prompts: np };
    });
  const addPrompt = () => {
    const k = newPromptKey.trim();
    if (!k || prompts[k] !== undefined) return;
    setS((p) => ({ ...p, vision_prompts: { ...p.vision_prompts, [k]: '' } }));
    setNewPromptKey('');
  };

  // pipeline steps (ordered list of tool names)
  const steps = (s && s.ingestion_steps) || [];
  const available = (s && s._available_tools) || [];
  const setSteps = (next) => setS((p) => ({ ...p, ingestion_steps: next }));
  const moveStep = (i, dir) => {
    const j = i + dir;
    if (j < 0 || j >= steps.length) return;
    const next = [...steps];
    [next[i], next[j]] = [next[j], next[i]];
    setSteps(next);
  };
  const removeStep = (i) => setSteps(steps.filter((_, idx) => idx !== i));
  const addStep = () => {
    if (!addStepName || steps.includes(addStepName)) return;
    setSteps([...steps, addStepName]);
    setAddStepName('');
  };

  const payload = () => {
    // send only the editable keys (drop the _-prefixed helper fields)
    const out = {};
    Object.keys(s).forEach((k) => { if (!k.startsWith('_')) out[k] = s[k]; });
    return out;
  };

  const onSave = async (saveAs = null) => {
    setSaving(true);
    setStatus(null);
    try {
      const { data } = await saveSettings(payload(), saveAs);
      setNewName('');
      await load();
      setStatus({ type: 'ok', msg: `Saved & applied — active profile: ${data.active}` });
    } catch (e) {
      setStatus({ type: 'err', msg: errMsg(e) });
    } finally {
      setSaving(false);
    }
  };

  const onActivate = async (name) => {
    if (name === active) return;
    setSaving(true);
    setStatus(null);
    try {
      await activateProfile(name);
      await load();
      setStatus({ type: 'ok', msg: `Activated profile: ${name}` });
    } catch (e) {
      setStatus({ type: 'err', msg: errMsg(e) });
    } finally {
      setSaving(false);
    }
  };

  const toggleAdvanced = async () => {
    if (!advanced) {
      try {
        const { data } = await getConfigRaw();
        setYamlText(data.yaml);
        setAdvanced(true);
      } catch (e) {
        setStatus({ type: 'err', msg: errMsg(e) });
      }
    } else {
      setAdvanced(false);
    }
  };

  const onSaveYaml = async () => {
    setSaving(true);
    setStatus(null);
    try {
      await saveConfig(yamlText);
      setAdvanced(false);
      await load();
      setStatus({ type: 'ok', msg: 'Saved raw config & reloaded' });
    } catch (e) {
      setStatus({ type: 'err', msg: errMsg(e) });
    } finally {
      setSaving(false);
    }
  };

  const industries = (s && s._industries && s._industries.length
    ? s._industries
    : ['automotive', 'electronics', 'manufacturing', 'finance', 'legal', 'healthcare', 'general']);

  return (
    <div className="min-h-screen pastel-mesh-bg text-[#1d1d1d]">
      {/* Slacc Glassmorphic Top Header Bar */}
      <div className="bg-white/80 backdrop-blur-md border-b border-[#e6e6e6] sticky top-0 z-50 shadow-sm">
        <div className="max-w-5xl mx-auto px-6 py-3.5 flex justify-between items-center">
          <div className="flex items-center gap-3.5">
            <button
              onClick={() => navigate('/')}
              className="p-2.5 rounded-full bg-[#f9f0ff] hover:bg-[#f3e2ff] border border-[#e6e6e6] text-[#4a154b] transition-all shadow-sm flex items-center justify-center"
              title="Back"
            >
              <ArrowLeftIcon className="h-5 w-5 stroke-[2.5]" />
            </button>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold tracking-widest text-xs text-[#4a154b] uppercase">AI-ACCELERATOR</span>
                <span className="text-[#696969] text-xs">/</span>
                <span className="text-xs font-bold text-[#696969] uppercase">SETTINGS</span>
              </div>
              <h1 className="text-xl font-extrabold text-[#4a154b] display-title tracking-tight mt-0.5">System Configuration</h1>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={toggleAdvanced}
              className={`px-5 py-2 rounded-full text-xs font-bold border transition-all ${
                advanced
                  ? 'border-[#4a154b] bg-[#4a154b] text-white shadow-md'
                  : 'border-[#e6e6e6] bg-[#f9f0ff] text-[#1d1d1d] hover:bg-[#f3e2ff]'
              }`}
            >
              {advanced ? 'Form View' : 'Advanced (YAML)'}
            </button>
            <button
              onClick={load}
              className="flex items-center gap-2 px-5 py-2 bg-white border border-[#e6e6e6] hover:bg-[#f9f0ff] rounded-full text-xs font-bold text-[#4a154b] transition-all shadow-sm"
            >
              <ArrowPathIcon className="h-4 w-4 text-[#4a154b]" /> Reload
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-6 py-8">
        {/* Customer Profile Selector Pills */}
        <div className="mb-6 bg-white border border-[#e6e6e6] rounded-2xl p-5 shadow-sm flex flex-wrap items-center gap-3">
          <span className="text-xs font-bold uppercase tracking-wider text-[#4a154b] mr-2">Active Profile:</span>
          {profiles.map((p) => (
            <button
              key={p}
              onClick={() => onActivate(p)}
              disabled={saving}
              className={`px-5 py-2 rounded-full text-xs font-bold border transition-all ${
                p === active
                  ? 'border-[#4a154b] bg-[#4a154b] text-white shadow-md'
                  : 'border-[#e6e6e6] bg-[#f9f0ff] text-[#1d1d1d] hover:bg-[#f3e2ff]'
              }`}
            >
              {p}{p === active ? ' ✓' : ''}
            </button>
          ))}
        </div>

        {status && (
          <div className={`mb-6 flex items-center gap-3 px-5 py-3.5 rounded-2xl text-sm font-medium border ${
            status.type === 'ok'
              ? 'bg-[#007a5a]/10 text-[#007a5a] border-[#007a5a]/30'
              : 'bg-[#cc4117]/10 text-[#cc4117] border-[#cc4117]/30'
          }`}>
            {status.type === 'ok' ? <CheckCircleIcon className="h-5 w-5 flex-shrink-0" /> : <ExclamationCircleIcon className="h-5 w-5 flex-shrink-0" />}
            {status.msg}
          </div>
        )}

        {loading || !s ? (
          <div className="text-center py-16 bg-white border border-[#e6e6e6] rounded-2xl shadow-sm text-[#696969]">
            <ArrowPathIcon className="h-8 w-8 mx-auto animate-spin mb-3 text-[#4a154b]" />
            <p className="text-sm font-semibold">Loading system configuration…</p>
          </div>
        ) : advanced ? (
          <div className="bg-white border border-[#e6e6e6] rounded-2xl p-6 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xs font-bold text-[#4a154b] uppercase tracking-wider">
                Raw YAML Configuration Editor
              </h2>
            </div>
            <textarea
              value={yamlText}
              onChange={(e) => setYamlText(e.target.value)}
              spellCheck={false}
              className="w-full h-[60vh] font-mono text-xs bg-[#f4ede4]/50 border border-[#e6e6e6] rounded-xl p-4 text-[#1d1d1d] focus:outline-none focus:border-[#4a154b] focus:ring-2 focus:ring-[#4a154b]/20"
            />
            <div className="mt-5">
              <button
                onClick={onSaveYaml}
                disabled={saving}
                className="btn-primary-pill disabled:opacity-50"
              >
                {saving ? 'Saving…' : 'Save & Apply Config'}
              </button>
            </div>
          </div>
        ) : (
          <div className="space-y-8">
            <div className="grid md:grid-cols-2 gap-8">
              {/* Documents & Extraction */}
              <section className="bg-white border border-[#e6e6e6] rounded-2xl p-6 shadow-sm">
                <h2 className="text-xs font-bold text-[#4a154b] uppercase tracking-wider mb-5 pb-2 border-b border-[#e6e6e6]">
                  Documents & Extraction Engine
                </h2>
                <Field label="Default Industry" hint="Used when auto-detection is unsure; sets industry-specific vision prompt.">
                  <Select value={s.default_industry} onChange={(v) => set('default_industry', v)} options={industries} />
                </Field>
                <Field label="Digital PDF Extractor Engine" hint="pymupdf_pdf = Fast PyMuPDF Native. docling_pdf = IBM Docling + TableFormer.">
                  <Select value={s.pdf_extractor_digital} onChange={(v) => set('pdf_extractor_digital', v)} options={s._digital_pdf_options || ['pymupdf_pdf', 'docling_pdf']} />
                </Field>

                {s.pdf_extractor_digital === 'docling_pdf' && (
                  <div className="mb-5 p-4 bg-[#f9f0ff] border border-[#e6e6e6] rounded-xl">
                    <label className="block text-xs font-bold text-[#4a154b] uppercase tracking-wider mb-1">Docling Extraction Mode</label>
                    <p className="text-xs text-[#696969] mb-3">Choose whether to run IBM Docling PDF extraction on a remote GPU server or locally on CPU.</p>
                    <div className="flex items-center gap-3 mb-3">
                      <button
                        type="button"
                        onClick={() => { set('docling_mode', 'local'); setServerStatus(null); }}
                        className={`px-4 py-2 rounded-full text-xs font-bold border transition-all ${
                          (s.docling_mode || 'local') === 'local'
                            ? 'border-[#4a154b] bg-[#4a154b] text-white'
                            : 'border-[#e6e6e6] bg-white text-[#1d1d1d] hover:bg-[#f9f0ff]'
                        }`}
                      >
                        💻 Local CPU
                      </button>
                      <button
                        type="button"
                        onClick={() => { set('docling_mode', 'remote'); testDoclingServer(); }}
                        className={`px-4 py-2 rounded-full text-xs font-bold border transition-all ${
                          s.docling_mode === 'remote'
                            ? 'border-[#4a154b] bg-[#4a154b] text-white'
                            : 'border-[#e6e6e6] bg-white text-[#1d1d1d] hover:bg-[#f9f0ff]'
                        }`}
                      >
                        🚀 Remote GPU Server
                      </button>
                    </div>

                    {(s.docling_mode || 'local') === 'remote' && (
                      <div className="mt-3 pt-3 border-t border-[#e6e6e6] space-y-3">
                        <Field label="Remote Server URL" hint="Base URL of remote Docling server (e.g. http://localhost:8083)">
                          <input
                            className={inputCls}
                            placeholder="http://localhost:8083"
                            value={s.docling_server_url ?? ''}
                            onChange={(e) => set('docling_server_url', e.target.value)}
                          />
                        </Field>

                        <div className="flex items-center gap-3 pt-1">
                          <button
                            type="button"
                            onClick={testDoclingServer}
                            disabled={serverStatus === 'checking'}
                            className="px-4 py-1.5 bg-[#4a154b] text-white hover:bg-[#611f69] text-xs font-bold rounded-full flex items-center gap-1.5 transition-all shadow-sm"
                          >
                            {serverStatus === 'checking' && <ArrowPathIcon className="h-3.5 w-3.5 animate-spin" />}
                            Test Connection
                          </button>
                          {serverStatus === 'online' && (
                            <span className="text-xs text-[#007a5a] font-bold flex items-center gap-1">
                              <CheckCircleIcon className="h-4 w-4" /> Server online & reachable
                            </span>
                          )}
                          {serverStatus === 'offline' && (
                            <span className="text-xs text-[#cc4117] font-bold flex items-center gap-1">
                              <ExclamationCircleIcon className="h-4 w-4" /> Server unreachable
                            </span>
                          )}
                        </div>

                        {serverStatus === 'offline' && (
                          <div className="p-3 bg-[#cc4117]/10 border border-[#cc4117]/30 rounded-xl text-xs text-[#cc4117]">
                            ⚠️ <strong>Server unreachable:</strong> The remote Docling server cannot be reached. Extraction will automatically fall back to local CPU mode until restored.
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
                <Field label="Scanned OCR Engine" hint="surya = highest accuracy. paddle = fast CPU processing.">
                  <Select value={s.ocr_engine} onChange={(v) => set('ocr_engine', v)} options={['surya', 'paddle']} />
                </Field>
                <Field label="Chunking Strategy" hint="semantic = topic shift splitting. recursive = fixed token windows.">
                  <Select value={s.chunking_strategy} onChange={(v) => set('chunking_strategy', v)} options={['semantic', 'recursive']} />
                </Field>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Chunk Size (Tokens)">
                    <input type="number" className={inputCls} value={s.chunking_size ?? ''} onChange={(e) => set('chunking_size', e.target.value === '' ? null : Number(e.target.value))} />
                  </Field>
                  <Field label="Overlap (Tokens)">
                    <input type="number" className={inputCls} value={s.chunking_overlap ?? ''} onChange={(e) => set('chunking_overlap', e.target.value === '' ? null : Number(e.target.value))} />
                  </Field>
                </div>
              </section>

              {/* AI Models */}
              <section className="bg-white border border-[#e6e6e6] rounded-2xl p-6 shadow-sm">
                <h2 className="text-xs font-bold text-[#4a154b] uppercase tracking-wider mb-5 pb-2 border-b border-[#e6e6e6]">
                  AI & Vision Models
                </h2>
                <Field label="Default LLM Provider">
                  <Select value={s.llm_provider} onChange={(v) => set('llm_provider', v)} options={['groq', 'google', 'ollama', 'openai', 'anthropic']} />
                </Field>
                <Field label="Default LLM Model">
                  <input className={inputCls} value={s.llm_model ?? ''} onChange={(e) => set('llm_model', e.target.value)} />
                </Field>
                <Field label="Answer Model (Answering + Planning)">
                  <input className={inputCls} placeholder="inherit default LLM model" value={s.llm_answer_model ?? ''} onChange={(e) => set('llm_answer_model', e.target.value)} />
                </Field>
                <Field label="Agent Model (Tool Calling)">
                  <input className={inputCls} placeholder="inherit default LLM model" value={s.agent_model ?? ''} onChange={(e) => set('agent_model', e.target.value)} />
                </Field>
                <Field label="Vision Provider">
                  <Select value={s.vision_provider} onChange={(v) => set('vision_provider', v)} options={['google', 'ollama', 'openai']} />
                </Field>
                <Field label="Vision Model (Figure Captioning)">
                  <input className={inputCls} value={s.vision_model ?? ''} onChange={(e) => set('vision_model', e.target.value)} />
                </Field>
                <Field label="Image Captioning (Vision)">
                  <Toggle value={!!s.vision_enabled} onChange={(v) => set('vision_enabled', v)} on="Enabled" off="Disabled" />
                </Field>
                <Field label="Chunk Summaries + Keywords (LLM Enrichment)">
                  <div className="flex items-center gap-3">
                    <Toggle value={!!s.enrichment_summarize} onChange={(v) => set('enrichment_summarize', v)} />
                    <span className="text-xs font-medium text-[#696969]">Keywords count:</span>
                    <input type="number" className={`${inputCls} w-20`} value={s.enrichment_keyword_count ?? ''} onChange={(e) => set('enrichment_keyword_count', e.target.value === '' ? null : Number(e.target.value))} />
                  </div>
                </Field>

                <div className="mt-4 pt-4 border-t border-[#e6e6e6]">
                  <p className="text-xs text-[#696969] mb-3">Per-step model overrides — leave blank to inherit default LLM model.</p>
                  <div className="grid grid-cols-2 gap-3">
                    <Field label="Categorization">
                      <input className={inputCls} placeholder="inherit default" value={s.categorization_model ?? ''} onChange={(e) => set('categorization_model', e.target.value)} />
                    </Field>
                    <Field label="Enrichment">
                      <input className={inputCls} placeholder="inherit default" value={s.enrichment_model ?? ''} onChange={(e) => set('enrichment_model', e.target.value)} />
                    </Field>
                  </div>
                </div>
              </section>
            </div>

            {/* Storage & Auto-Ingestion */}
            <div className="grid md:grid-cols-2 gap-8">
              <section className="bg-white border border-[#e6e6e6] rounded-2xl p-6 shadow-sm">
                <h2 className="text-xs font-bold text-[#4a154b] uppercase tracking-wider mb-5 pb-2 border-b border-[#e6e6e6]">
                  Storage & Auto Ingestion
                </h2>
                <Field label="Storage Provider" hint="Where uploaded documents are saved (local disk or Supabase bucket).">
                  <Select
                    value={s.storage_provider || 'local'}
                    onChange={(v) => set('storage_provider', v)}
                    options={['local', 'supabase']}
                  />
                </Field>
                <Field label="Directory Watcher (Auto-Ingestion)">
                  <Toggle value={!!s.auto_ingestion_enabled} onChange={(v) => set('auto_ingestion_enabled', v)} on="Enabled" off="Disabled" />
                </Field>
                <Field label="Watch Directory Path">
                  <input className={inputCls} placeholder="e.g. auto_ingest" value={s.auto_ingestion_watch_dir ?? ''} onChange={(e) => set('auto_ingestion_watch_dir', e.target.value)} />
                </Field>
              </section>

              <section className="bg-white border border-[#e6e6e6] rounded-2xl p-6 shadow-sm">
                <h2 className="text-xs font-bold text-[#4a154b] uppercase tracking-wider mb-5 pb-2 border-b border-[#e6e6e6]">
                  Embeddings & Vector Search
                </h2>
                <Field label="Embedding Provider" hint="local = BAAI/bge-m3. jina = jina-embeddings-v3. openai = text-embedding-3-small.">
                  <Select 
                    value={s.embeddings_dense_provider || 'local'} 
                    onChange={(v) => {
                      set('embeddings_dense_provider', v);
                      if (v === 'jina') set('embeddings_dense_model', 'jina-embeddings-v3');
                      else if (v === 'openai') set('embeddings_dense_model', 'text-embedding-3-small');
                      else set('embeddings_dense_model', 'BAAI/bge-m3');
                    }} 
                    options={['local', 'jina', 'openai']} 
                  />
                </Field>
                <Field label="Embedding Model">
                  <input 
                    className={inputCls} 
                    value={s.embeddings_dense_model ?? ''} 
                    onChange={(e) => set('embeddings_dense_model', e.target.value)} 
                  />
                </Field>
                <Field label="Reranker Provider" hint="local = BAAI/bge-reranker-v2-m3. jina = jina-reranker-v2-base-multilingual.">
                  <Select 
                    value={s.embeddings_reranker_provider || 'local'} 
                    onChange={(v) => {
                      set('embeddings_reranker_provider', v);
                      if (v === 'jina') set('embeddings_reranker_model', 'jina-reranker-v2-base-multilingual');
                      else set('embeddings_reranker_model', 'BAAI/bge-reranker-v2-m3');
                    }} 
                    options={['local', 'jina']} 
                  />
                </Field>
              </section>
            </div>

            {/* Pipeline Tools Order */}
            <section className="bg-white border border-[#e6e6e6] rounded-2xl p-6 shadow-sm">
              <h2 className="text-xs font-bold text-[#4a154b] uppercase tracking-wider mb-2">Pipeline Tool Execution Sequence</h2>
              <p className="text-xs text-[#696969] mb-4">
                The tools that execute on document ingestion, in order. Re-order or attach registered backend tools.
              </p>
              <div className="space-y-2.5 mb-4">
                {steps.map((step, i) => (
                  <div key={`${step}-${i}`} className="flex items-center gap-3 bg-[#f9f0ff]/50 border border-[#e6e6e6] rounded-xl px-4 py-2.5">
                    <span className="text-xs font-extrabold text-[#4a154b] w-6">{i + 1}.</span>
                    <span className="text-sm font-semibold text-[#1d1d1d] flex-1">{step}</span>
                    <button onClick={() => moveStep(i, -1)} disabled={i === 0} className="px-2.5 py-1 bg-white border border-[#e6e6e6] rounded-md text-xs text-[#4a154b] font-bold hover:bg-[#4a154b] hover:text-white transition-all disabled:opacity-30">↑</button>
                    <button onClick={() => moveStep(i, 1)} disabled={i === steps.length - 1} className="px-2.5 py-1 bg-white border border-[#e6e6e6] rounded-md text-xs text-[#4a154b] font-bold hover:bg-[#4a154b] hover:text-white transition-all disabled:opacity-30">↓</button>
                    <button onClick={() => removeStep(i)} className="px-2.5 py-1 bg-[#cc4117]/10 text-[#cc4117] hover:bg-[#cc4117] hover:text-white rounded-md text-xs font-bold transition-all">✕</button>
                  </div>
                ))}
              </div>
              <div className="flex items-center gap-3">
                <select value={addStepName} onChange={(e) => setAddStepName(e.target.value)} className={`${inputCls} max-w-xs`}>
                  <option value="">Add pipeline tool stage…</option>
                  {available.filter((t) => !steps.includes(t)).map((t) => <option key={t} value={t}>{t}</option>)}
                </select>
                <button onClick={addStep} disabled={!addStepName} className="btn-secondary-pill disabled:opacity-50">Add Step</button>
              </div>
            </section>

            {/* Signature Aubergine Band Card */}
            <div className="aubergine-mesh-card rounded-2xl p-8 shadow-xl text-white flex flex-col md:flex-row items-center justify-between gap-6">
              <div>
                <h3 className="text-xl font-bold display-title text-white">Save Configuration Profile</h3>
                <p className="text-xs text-[#d9bdde] mt-1 max-w-xl leading-relaxed">
                  Routing, prompts, OCR & chunking apply live on save. Changing core AI model providers requires a server process restart.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <button onClick={() => onSave(null)} disabled={saving} className="px-8 py-3.5 rounded-full bg-white text-[#4a154b] font-bold text-sm hover:bg-[#f9f0ff] transition-all shadow-lg active:scale-95 disabled:opacity-50">
                  {saving ? 'Saving…' : `Save & Apply (${active})`}
                </button>
                <div className="flex items-center gap-2">
                  <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="new-profile-name" className="px-4 py-2.5 bg-white/10 border border-white/20 rounded-full text-xs text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-white/40" />
                  <button onClick={() => onSave(newName.trim())} disabled={saving || !newName.trim()} className="px-6 py-2.5 rounded-full bg-white/20 hover:bg-white/30 text-white font-bold text-xs transition-all disabled:opacity-40">
                    Save As Profile
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SettingsPage;
