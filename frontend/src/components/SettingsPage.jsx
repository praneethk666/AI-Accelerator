import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeftIcon,
  ArrowPathIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
} from '@heroicons/react/24/outline';
import {
  getSettings,
  saveSettings,
  getProfiles,
  activateProfile,
  getConfigRaw,
  saveConfig,
} from '../api';

// Small presentational helpers ------------------------------------------------
const Field = ({ label, hint, children }) => (
  <div className="mb-5">
    <label className="block text-sm font-medium text-gray-200 mb-1">{label}</label>
    {hint && <p className="text-xs text-gray-500 mb-2">{hint}</p>}
    {children}
  </div>
);

const inputCls =
  'w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-gray-200 focus:outline-none focus:border-blue-500/50';

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
    className={`px-4 py-2 rounded-lg text-sm border transition-colors ${
      value
        ? 'border-green-500/50 bg-green-500/10 text-green-300'
        : 'border-slate-700 bg-slate-800/40 text-gray-400'
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
      setStatus({ type: 'ok', msg: `Saved & applied — active: ${data.active}` });
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
      setStatus({ type: 'ok', msg: `Activated ${name}` });
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
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-gray-100">
      <div className="bg-slate-900/50 border-b border-slate-700 sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-6 py-6 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <button onClick={() => navigate('/')} className="p-2 rounded-lg hover:bg-slate-700/50" title="Back">
              <ArrowLeftIcon className="h-5 w-5" />
            </button>
            <div>
              <h1 className="text-3xl font-bold text-white">Configuration</h1>
              <p className="text-gray-400 mt-1">Tune the pipeline per customer — changes apply live on save</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={toggleAdvanced}
              className={`px-4 py-2 rounded-lg text-sm border ${
                advanced
                  ? 'border-blue-500/50 bg-blue-500/10 text-blue-300'
                  : 'border-slate-700 bg-slate-800/40 text-gray-300 hover:bg-slate-700/50'
              }`}
            >
              {advanced ? 'Form view' : 'Advanced (YAML)'}
            </button>
            <button onClick={load} className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm">
              <ArrowPathIcon className="h-4 w-4" /> Reload
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-6 py-8">
        {/* Profiles */}
        <div className="mb-6 flex flex-wrap items-center gap-3">
          <span className="text-sm text-gray-400">Customer profile:</span>
          {profiles.map((p) => (
            <button
              key={p}
              onClick={() => onActivate(p)}
              disabled={saving}
              className={`px-3 py-1.5 rounded-lg text-sm border transition-colors ${
                p === active
                  ? 'border-blue-500/50 bg-blue-500/10 text-blue-300'
                  : 'border-slate-700 bg-slate-800/40 text-gray-300 hover:bg-slate-700/50'
              }`}
            >
              {p}{p === active ? ' ✓' : ''}
            </button>
          ))}
        </div>

        {status && (
          <div className={`mb-5 flex items-center gap-2 px-4 py-3 rounded-lg text-sm ${
            status.type === 'ok'
              ? 'bg-green-500/10 text-green-300 border border-green-500/30'
              : 'bg-red-500/10 text-red-300 border border-red-500/30'
          }`}>
            {status.type === 'ok' ? <CheckCircleIcon className="h-5 w-5" /> : <ExclamationCircleIcon className="h-5 w-5" />}
            {status.msg}
          </div>
        )}

        {loading || !s ? (
          <div className="text-center py-12 text-gray-400">
            <ArrowPathIcon className="h-8 w-8 mx-auto animate-spin mb-3" /> Loading…
          </div>
        ) : advanced ? (
          <div>
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
                Raw config — full freedom (steps, gates, routes, prompts, anything)
              </h2>
            </div>
            <textarea
              value={yamlText}
              onChange={(e) => setYamlText(e.target.value)}
              spellCheck={false}
              className="w-full h-[60vh] font-mono text-xs bg-slate-900/70 border border-slate-700 rounded-lg p-4 text-gray-200 focus:outline-none focus:border-blue-500/50"
            />
            <div className="mt-4">
              <button
                onClick={onSaveYaml}
                disabled={saving}
                className="px-5 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded-lg text-sm font-medium"
              >
                {saving ? 'Saving…' : 'Save & apply'}
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className="grid md:grid-cols-2 gap-x-8">
              <section>
                <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">Documents</h2>
                <Field label="Default industry" hint="Used when auto-detection is unsure; also picks the industry vision prompt.">
                  <Select value={s.default_industry} onChange={(v) => set('default_industry', v)} options={industries} />
                </Field>
                <Field label="Digital PDF Extractor Engine" hint="pymupdf_pdf = Fast PyMuPDF Native + Bbox Table Exclusion + Row Alarm Chunker. docling_pdf = IBM Docling + TableFormer.">
                  <Select value={s.pdf_extractor_digital} onChange={(v) => set('pdf_extractor_digital', v)} options={s._digital_pdf_options || ['pymupdf_pdf', 'docling_pdf']} />
                </Field>
                <Field label="Scanned OCR engine" hint="surya = best quality (slower). paddle = fast (big scanned docs on CPU).">
                  <Select value={s.ocr_engine} onChange={(v) => set('ocr_engine', v)} options={['surya', 'paddle']} />
                </Field>
                <Field label="Chunking strategy" hint="semantic = split on topic shifts. recursive = fixed token windows.">
                  <Select value={s.chunking_strategy} onChange={(v) => set('chunking_strategy', v)} options={['semantic', 'recursive']} />
                </Field>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Chunk size (tokens)">
                    <input type="number" className={inputCls} value={s.chunking_size ?? ''} onChange={(e) => set('chunking_size', Number(e.target.value))} />
                  </Field>
                  <Field label="Overlap (tokens)">
                    <input type="number" className={inputCls} value={s.chunking_overlap ?? ''} onChange={(e) => set('chunking_overlap', Number(e.target.value))} />
                  </Field>
                </div>
              </section>

              <section>
                <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">AI models</h2>
                <Field label="Answer LLM provider">
                  <Select value={s.llm_provider} onChange={(v) => set('llm_provider', v)} options={['groq', 'google', 'ollama', 'openai']} />
                </Field>
                <Field label="Answer LLM model">
                  <input className={inputCls} value={s.llm_model ?? ''} onChange={(e) => set('llm_model', e.target.value)} />
                </Field>
                <Field label="Vision provider">
                  <Select value={s.vision_provider} onChange={(v) => set('vision_provider', v)} options={['google', 'ollama', 'openai']} />
                </Field>
                <Field label="Vision model">
                  <input className={inputCls} value={s.vision_model ?? ''} onChange={(e) => set('vision_model', e.target.value)} />
                </Field>
                <Field label="Image captioning (vision)">
                  <Toggle value={!!s.vision_enabled} onChange={(v) => set('vision_enabled', v)} on="Enabled" off="Disabled" />
                </Field>
                <Field label="Chunk summaries + keywords (LLM enrichment)">
                  <div className="flex items-center gap-3">
                    <Toggle value={!!s.enrichment_summarize} onChange={(v) => set('enrichment_summarize', v)} />
                    <span className="text-xs text-gray-500">keywords</span>
                    <input type="number" className={`${inputCls} w-20`} value={s.enrichment_keyword_count ?? ''} onChange={(e) => set('enrichment_keyword_count', Number(e.target.value))} />
                  </div>
                </Field>
              </section>
            </div>

            {/* Vision prompts — one per route/industry, fully editable */}
            <section className="mt-2">
              <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-1">Vision prompts</h2>
              <p className="text-xs text-gray-500 mb-4">
                Selected by route, then industry, then <code>default</code>. Add a key matching a
                route (e.g. <code>cad_route</code>) or industry (e.g. <code>automotive</code>).
              </p>
              {Object.keys(prompts).length === 0 && (
                <p className="text-sm text-gray-500 mb-3">No prompts defined.</p>
              )}
              {Object.entries(prompts).map(([key, val]) => (
                <div key={key} className="mb-4">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-blue-300">{key}</span>
                    <button onClick={() => removePrompt(key)} className="text-xs text-red-400 hover:text-red-300">remove</button>
                  </div>
                  <textarea rows={3} className={inputCls} value={val ?? ''} onChange={(e) => setPrompt(key, e.target.value)} />
                </div>
              ))}
              <div className="flex items-center gap-2">
                <input value={newPromptKey} onChange={(e) => setNewPromptKey(e.target.value)} placeholder="route or industry key" className={`${inputCls} w-56`} />
                <button onClick={addPrompt} disabled={!newPromptKey.trim()} className="px-4 py-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded-lg text-sm">Add prompt</button>
              </div>
            </section>

            {/* Enrichment prompt */}
            <section className="mt-8">
              <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-1">Chunk enrichment prompt</h2>
              <p className="text-xs text-gray-500 mb-3">
                Instruction for the per-chunk summary + keywords. The pipeline adds the chunks
                and asks for JSON, so describe <em>what to produce</em> — don't paste a chunk here.
              </p>
              <textarea rows={5} className={inputCls} value={s.enrichment_prompt ?? ''} onChange={(e) => set('enrichment_prompt', e.target.value)} />
            </section>

            {/* Pipeline steps */}
            <section className="mt-8">
              <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-1">Pipeline steps</h2>
              <p className="text-xs text-gray-500 mb-4">
                The nodes that run on ingestion, in order. Remove/reorder, or add any registered
                tool. (New node <em>logic</em> needs a code change; this composes existing tools.)
              </p>
              <div className="space-y-2 mb-3">
                {steps.map((step, i) => (
                  <div key={`${step}-${i}`} className="flex items-center gap-2 bg-slate-800/40 border border-slate-700 rounded-lg px-3 py-2">
                    <span className="text-xs text-gray-500 w-5">{i + 1}</span>
                    <span className="text-sm text-gray-200 flex-1">{step}</span>
                    <button onClick={() => moveStep(i, -1)} disabled={i === 0} className="px-2 text-gray-400 hover:text-gray-200 disabled:opacity-30">↑</button>
                    <button onClick={() => moveStep(i, 1)} disabled={i === steps.length - 1} className="px-2 text-gray-400 hover:text-gray-200 disabled:opacity-30">↓</button>
                    <button onClick={() => removeStep(i)} className="px-2 text-red-400 hover:text-red-300">✕</button>
                  </div>
                ))}
              </div>
              <div className="flex items-center gap-2">
                <select value={addStepName} onChange={(e) => setAddStepName(e.target.value)} className={`${inputCls} w-56`}>
                  <option value="">add a step…</option>
                  {available.filter((t) => !steps.includes(t)).map((t) => <option key={t} value={t}>{t}</option>)}
                </select>
                <button onClick={addStep} disabled={!addStepName} className="px-4 py-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded-lg text-sm">Add step</button>
              </div>
            </section>

            {/* Actions */}
            <div className="mt-6 flex flex-wrap items-center gap-3 border-t border-slate-700 pt-6">
              <button onClick={() => onSave(null)} disabled={saving} className="px-5 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded-lg text-sm font-medium">
                {saving ? 'Saving…' : `Save & apply (${active})`}
              </button>
              <div className="flex items-center gap-2">
                <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="new-customer" className={`${inputCls} w-44`} />
                <button onClick={() => onSave(newName.trim())} disabled={saving || !newName.trim()} className="px-4 py-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded-lg text-sm">
                  Save as new profile
                </button>
              </div>
            </div>
            <p className="text-xs text-gray-500 mt-3">
              Routing, prompts, OCR, chunking &amp; enrichment apply immediately. Changing AI
              models needs a server restart. API keys live in <code>.env</code>.
            </p>
          </>
        )}
      </div>
    </div>
  );
};

export default SettingsPage;
