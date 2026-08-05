import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { uploadFile, getFiles, deleteFile, healthCheck, getProgress, checkDoclingServer } from '../api';
import {
  CloudArrowUpIcon,
  DocumentIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
  XCircleIcon,
  TrashIcon,
  ChatBubbleLeftIcon,
  ArrowPathIcon,
  Cog6ToothIcon,
  ChevronDownIcon,
  ClockIcon,
  CurrencyDollarIcon,
} from '@heroicons/react/24/outline';

// The real ingestion steps (match tool `name`s in the metrics the API returns).
// Each stage maps to one or more tool names; "Extract" covers whichever extractor
// ran for this file type / pdf kind.
const PIPELINE_STAGES = [
  { label: 'Categorize', icon: '📂', match: ['categorize'] },
  { label: 'Extract', icon: '📊', match: ['pymupdf_pdf', 'docling_pdf', 'pdf_digital', 'scanned_pdf', 'mixed_pdf', 'excel_extraction', 'ppt_extraction', 'word_extraction', 'image_extraction', 'cad_extract'] },
  { label: 'Vision', icon: '👁️', match: ['vision_enrichment'] },
  { label: 'Chunk', icon: '✂️', match: ['chunk'] },
  { label: 'Enrich', icon: '🏷️', match: ['enrich_chunks'] },
  { label: 'Embed', icon: '🔗', match: ['embed'] },
  { label: 'Index', icon: '🗄️', match: ['index'] },
];

// Human-readable duration from milliseconds: ms < 1s, then s, m + s, h + m.
// Used for per-step notes and the run total so long OCR steps read as "2m 14s"
// instead of "134000ms".
const fmtDuration = (ms) => {
  if (ms == null || isNaN(ms)) return '';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const totalSec = ms / 1000;
  if (totalSec < 60) return `${totalSec.toFixed(1)}s`;
  const totalMin = Math.floor(totalSec / 60);
  const sec = Math.round(totalSec % 60);
  if (totalMin < 60) return `${totalMin}m ${sec}s`;
  const hr = Math.floor(totalMin / 60);
  const min = totalMin % 60;
  return `${hr}h ${min}m`;
};

const getStepLabel = (stepName) => {
  const stage = PIPELINE_STAGES.find((s) => s.match.includes(stepName));
  if (stage) return stage.label;
  // Fallback: capitalize words and replace underscores
  return (stepName || '')
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
};

const fmtLlmCallsSub = (tu) => {
  if (!tu) return '0 calls';
  const visionCalls = tu.by_kind?.vision?.calls || 0;
  const total = tu.calls || 0;
  const text = total - visionCalls;
  if (text > 0 && visionCalls > 0) {
    return `${text} text · ${visionCalls} vision calls`;
  }
  if (visionCalls > 0) {
    return `${visionCalls} vision calls`;
  }
  return `${total} LLM calls`;
};

const USD_TO_INR = 83.50;

const IngestionPage = () => {
  const navigate = useNavigate();
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [currency, setCurrency] = useState('USD');
  const [error, setError] = useState(null);
  const [serverConnected, setServerConnected] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({});
  // metrics from the most recent upload: { name, status, metrics: [{step, ms, status}] }
  const [lastRun, setLastRun] = useState(null);
  const [selectedDocId, setSelectedDocId] = useState(null);
  const [doclingAlert, setDoclingAlert] = useState(null); // null | { url: string }
  const [expandedFiles, setExpandedFiles] = useState({});
  const selectedDocIdRef = useRef(null);
  const activePolls = useRef({});

  const toggleExpandFile = (fileId) => {
    setExpandedFiles((prev) => ({
      ...prev,
      [fileId]: !prev[fileId],
    }));
  };

  const selectDocument = (docId, fileInfo = null) => {
    setSelectedDocId(docId);
    selectedDocIdRef.current = docId;
    if (fileInfo) {
      setLastRun({
        id: docId,
        name: fileInfo.filename || fileInfo.name,
        status: fileInfo.status,
        metrics: fileInfo.metrics || [],
        tokenUsage: fileInfo.token_usage,
        indexedTokens: fileInfo.indexed_tokens,
        chunks: fileInfo.chunk_count || fileInfo.chunks,
      });
    }
  };

  // ─ Smart adaptive file list polling ─────────────────────────────────────────────
  // Polls /files only when: (1) a file is actively processing, OR (2) this is
  // the first load. Stops when tab is hidden. This replaces the old always-on
  // 3-second interval that ran 20 req/min regardless of activity.
  useEffect(() => {
    checkServerHealth();
    loadFiles();

    let intervalId = null;

    const smartPoll = () => {
      // Skip polling when tab is in background — saves bandwidth and Supabase quota
      if (document.visibilityState === 'hidden') return;
      silentReloadFiles();
    };

    const startPolling = () => {
      if (intervalId) return;
      intervalId = setInterval(smartPoll, 10_000); // 10s instead of 3s
    };

    const stopPolling = () => {
      if (intervalId) { clearInterval(intervalId); intervalId = null; }
    };

    const onVisibility = () => {
      if (document.visibilityState === 'visible') { startPolling(); smartPoll(); }
      else stopPolling();
    };

    document.addEventListener('visibilitychange', onVisibility);
    startPolling();

    return () => {
      stopPolling();
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, []);

  const checkServerHealth = async () => {
    try {
      await healthCheck();
      setServerConnected(true);
      setError(null);
    } catch (err) {
      setServerConnected(false);
      setError('Cannot connect to backend server. Make sure it is running on http://localhost:8000');
    }
  };

  const silentReloadFiles = async () => {
    try {
      const res = await getFiles();
      const filesList = res.data || [];

      // Skip setFiles() if only status and ids are unchanged — prevents needless
      // re-renders when nothing has changed (the dominant source of frontend lag).
      setFiles((prev) => {
        const prevSig = prev.map((f) => `${f.document_id}:${f.status}`).join(',');
        const newSig = filesList.map((f) => `${f.document_id}:${f.status}`).join(',');
        return prevSig === newSig ? prev : filesList;
      });

      // Default select to the most recent file if nothing is selected
      if (filesList.length > 0 && !selectedDocIdRef.current) {
        const first = filesList[0];
        selectDocument(first.document_id || first.id, first);
      }

      // Auto-poll any file that is currently processing in the background
      filesList.forEach((file) => {
        const id = file.document_id || file.id;
        if (file.status === 'processing' && !activePolls.current[id]) {
          pollProgress(id, file.filename);
        }
      });
    } catch (err) {
      console.error('Failed to silently reload files:', err);
    }
  };

  const loadFiles = async () => {
    setLoading(true);
    try {
      const res = await getFiles();
      const filesList = res.data || [];
      setFiles(filesList);
      setError(null);

      // Default select to the most recent file if nothing is selected
      if (filesList.length > 0 && !selectedDocIdRef.current) {
        const first = filesList[0];
        selectDocument(first.document_id || first.id, first);
      }

      // Auto-poll any file that is currently processing in the background
      filesList.forEach((file) => {
        const id = file.document_id || file.id;
        if (file.status === 'processing') {
          pollProgress(id, file.filename);
        }
      });
    } catch (err) {
      console.error('Failed to load files:', err);
      setError(err.message || 'Failed to load files');
    } finally {
      setLoading(false);
    }
  };

  const onDrop = async (acceptedFiles) => {
    if (!serverConnected) {
      setError('Cannot upload: backend server is not connected');
      return;
    }

    // Check if any PDF is uploaded and if configured remote docling server is online
    const hasPdf = acceptedFiles.some((f) => f.name.toLowerCase().endsWith('.pdf'));
    if (hasPdf) {
      try {
        const { data: doclingHealth } = await checkDoclingServer();
        if (doclingHealth.mode === 'remote' && doclingHealth.reachable === false) {
          setDoclingAlert({ url: doclingHealth.url || 'configured server' });
        }
      } catch (err) {
        console.error('Docling server check failed:', err);
      }
    }

    setUploading(true);
    setError(null);

    for (const file of acceptedFiles) {
      try {
        setUploadProgress((prev) => ({ ...prev, [file.name]: 0 }));
        // /upload now returns immediately with status "processing"; the pipeline
        // runs in the background and we poll per-step progress below.
        const res = await uploadFile(file);
        const doc = res.data;
        const docId = doc.document_id || doc.id;
        
        setFiles((prev) => [doc, ...prev]);
        selectDocument(docId, {
          filename: file.name,
          status: doc.status,
          metrics: [],
        });

        setUploadProgress((prev) => {
          const newProgress = { ...prev };
          delete newProgress[file.name];
          return newProgress;
        });
        pollProgress(docId, file.name);
      } catch (err) {
        console.error('Upload failed:', err);
        setError(`Upload failed for ${file.name}: ${err.message}`);
        setUploadProgress((prev) => {
          const newProgress = { ...prev };
          delete newProgress[file.name];
          return newProgress;
        });
      }
    }

    setUploading(false);
  };

  // ─ Per-document ingestion progress poller ────────────────────────────────────────
  // Adaptive backoff: starts at 1.5s, slows to 3s after 10 polls, 5s after 30.
  // Hard cap: 120 attempts (~3 min max vs the old 800 = 12+ min at 900ms flat).
  const pollProgress = (docId, name) => {
    if (activePolls.current[docId]) return;
    activePolls.current[docId] = true;

    let attempts = 0;
    const tick = async () => {
      attempts += 1;
      try {
        const { data } = await getProgress(docId);

        // Only update metrics if this document is currently selected
        if (selectedDocIdRef.current === docId) {
          setLastRun({
            id: docId,
            name,
            status: data.status,
            metrics: data.metrics || [],
            tokenUsage: data.token_usage,
            indexedTokens: data.indexed_tokens,
            chunks: data.chunks,
            currentStep: data.current_step,
          });
        }

        setFiles((prev) =>
          prev.map((f) =>
            (f.document_id === docId || f.id === docId)
              ? { ...f, status: data.status, document_type: data.document_type,
                  industry: data.industry, route: data.route, confidence: data.confidence,
                  current_step: data.current_step }
              : f
          )
        );

        if (data.status === 'processing' && attempts < 120) {
          // Adaptive backoff: slow down as ingestion takes longer
          const delay = attempts > 30 ? 5000 : attempts > 10 ? 3000 : 1500;
          setTimeout(tick, delay);
        } else {
          delete activePolls.current[docId];
          loadFiles(); // final refresh from DB
        }
      } catch (e) {
        if (e.message === 'Resource not found') {
          delete activePolls.current[docId];
          return;
        }
        if (attempts < 120) {
          setTimeout(tick, 2000); // transient error — keep trying, slower
        } else {
          delete activePolls.current[docId];
        }
      }
    };
    setTimeout(tick, 600);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
    "application/pdf": [".pdf"],

    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
    "application/msword": [".doc"],

    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
    "application/vnd.ms-excel": [".xls"],

    "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"],
    "application/vnd.ms-powerpoint": [".ppt"],

    "image/*": [".png", ".jpg", ".jpeg", ".gif", ".bmp"]
  },
    disabled: uploading || !serverConnected,
  });

  const handleDelete = async (fileId, filename) => {
    if (window.confirm(`Are you sure you want to delete "${filename}"?`)) {
      try {
        await deleteFile(fileId);
        setFiles((prev) => prev.filter((f) => f.id !== fileId));
        setError(null);
        delete activePolls.current[fileId];
        setLastRun((prev) => {
          if (prev && (prev.id === fileId || prev.name === filename)) {
            return null;
          }
          return prev;
        });
      } catch (err) {
        setError(`Failed to delete file: ${err.message}`);
      }
    }
  };

  const handleOpenChat = (fileId) => {
    navigate(`/chat?fileId=${fileId}`);
  };

  // The backend emits LOWERCASE statuses: ready | processing | failed | unsupported | empty.
  // The old check compared against 'Ready' (capital R), which never matched — so every
  // successfully ingested file rendered as a yellow warning and the green state was dead
  // code. Each status now gets its own icon/colour so a hard failure, an unsupported
  // format and an empty extraction are distinguishable at a glance.
  const getStatusIcon = (status) => {
    switch ((status || '').toLowerCase()) {
      case 'ready':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
      case 'processing':
        return <ArrowPathIcon className="h-5 w-5 text-blue-400 animate-spin" />;
      case 'failed':
        return <XCircleIcon className="h-5 w-5 text-red-500" />;
      default: // unsupported | empty | unknown
        return <ExclamationCircleIcon className="h-5 w-5 text-yellow-500" />;
    }
  };

  const getStatusColor = (status) => {
    switch ((status || '').toLowerCase()) {
      case 'ready':
        return 'bg-green-500/20 text-green-400';
      case 'processing':
        return 'bg-blue-500/20 text-blue-400';
      case 'failed':
        return 'bg-red-500/20 text-red-400';
      default: // unsupported | empty | unknown
        return 'bg-yellow-500/20 text-yellow-400';
    }
  };

  return (
    <div className="min-h-screen pastel-mesh-bg text-[#1d1d1d]">
      {/* Slacc Glassmorphic Top Header Bar */}
      <div className="bg-white/80 backdrop-blur-md border-b border-[#e6e6e6] sticky top-0 z-50 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-3.5">
          <div className="flex flex-row justify-between items-center gap-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold tracking-widest text-xs text-[#4a154b] uppercase">AI-ACCELERATOR</span>
                <span className="text-[#696969] text-xs">/</span>
                <span className="text-xs font-bold text-[#696969] uppercase">INGESTION</span>
              </div>
              <h1 className="text-xl font-extrabold text-[#4a154b] display-title tracking-tight mt-0.5" style={{ fontFamily: 'Inter, sans-serif' }}>
                Document Ingestion & Pipeline
              </h1>
            </div>
            <div className="flex items-center gap-3">
              <div className={`inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-bold ${serverConnected ? 'bg-[#007a5a]/10 text-[#007a5a] border border-[#007a5a]/30' : 'bg-[#cc4117]/10 text-[#cc4117] border border-[#cc4117]/30'}`}>
                <div className={`w-2 h-2 rounded-full ${serverConnected ? 'bg-[#007a5a]' : 'bg-[#cc4117]'}`} />
                {serverConnected ? 'Backend Connected' : 'Backend Offline'}
              </div>
              <button
                onClick={() => navigate('/chat')}
                className="btn-primary-pill !py-2 !px-6 text-xs inline-flex items-center gap-2 shadow-sm"
              >
                <ChatBubbleLeftIcon className="h-4 w-4" />
                Agent Chat
              </button>
              <button
                onClick={() => navigate('/settings')}
                title="Configuration"
                className="p-2.5 rounded-full bg-[#f9f0ff] hover:bg-[#f3e2ff] border border-[#e6e6e6] text-[#4a154b] transition-all shadow-sm flex items-center justify-center"
              >
                <Cog6ToothIcon className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-[#cc4117]/10 border border-[#cc4117]/30 rounded-2xl text-[#cc4117] flex items-start gap-3 shadow-sm">
            <ExclamationCircleIcon className="h-5 w-5 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-bold">Execution Error</p>
              <p className="text-sm mt-0.5">{error}</p>
            </div>
          </div>
        )}

        {/* Upload Hero Dropzone Container — Compact */}
        <div
          {...getRootProps()}
          className={`mb-8 border-2 border-dashed rounded-2xl py-6 px-8 text-center transition-all cursor-pointer shadow-sm ${
            isDragActive
              ? 'border-[#4a154b] bg-[#f9f0ff]'
              : uploading || !serverConnected
                ? 'border-[#e6e6e6] bg-white/60 opacity-50 cursor-not-allowed'
                : 'border-[#4a154b]/30 bg-white hover:border-[#4a154b] hover:bg-[#f9f0ff]/40'
          }`}
        >
          <input {...getInputProps()} />
          <CloudArrowUpIcon className="h-9 w-9 mx-auto text-[#4a154b] mb-2" />
          <p className="text-base font-bold text-[#4a154b] display-title mb-0.5">
            {uploading ? 'Processing Uploads…' : isDragActive ? 'Drop files into canvas' : 'Drop documents or click to browse'}
          </p>
          <p className="text-xs text-[#696969] mb-4">
            Supported formats: PDF (Digital &amp; Scanned OCR), DOCX, XLSX, PPTX, Images (PNG, JPG, GIF, BMP)
          </p>
          <button
            disabled={uploading || !serverConnected}
            className={`btn-primary-pill !py-2 !px-5 text-xs inline-block ${
              uploading || !serverConnected ? 'opacity-50 cursor-not-allowed' : ''
            }`}
          >
            {uploading ? 'Uploading…' : 'Browse Local Files'}
          </button>
        </div>

        {/* Upload Progress */}
        {Object.keys(uploadProgress).length > 0 && (
          <div className="mb-8 space-y-3">
            {Object.entries(uploadProgress).map(([filename, progress]) => (
              <div key={filename} className="bg-white p-4 rounded-2xl border border-[#e6e6e6] shadow-sm">
                <p className="text-xs font-bold text-[#4a154b] mb-2">{filename}</p>
                <div className="w-full bg-[#f9f0ff] rounded-full h-2.5">
                  <div
                    className="bg-[#4a154b] h-2.5 rounded-full transition-all"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Pipeline Status — reflects the last upload's actual per-step metrics */}
        {lastRun && lastRun.status === 'processing' && (
          <div className="mb-8 bg-white border border-[#e6e6e6] rounded-2xl p-6 shadow-sm">
            <div className="flex items-center justify-between mb-5 pb-3 border-b border-[#e6e6e6]">
              <h2 className="text-xs font-bold text-[#4a154b] uppercase tracking-wider">Active Ingestion Pipeline</h2>
              <span className="text-xs font-semibold text-[#696969] truncate max-w-[60%] bg-[#f9f0ff] px-3 py-1 rounded-full border border-[#e6e6e6]">
                {lastRun.name} · Processing…
              </span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-3">
              {(() => {
                const metrics = lastRun?.metrics || [];
                const processing = lastRun?.status === 'processing';
                const lastDone = PIPELINE_STAGES.reduce(
                  (acc, s, i) => (metrics.some((x) => s.match.includes(x.step)) ? i : acc), -1);
                return PIPELINE_STAGES.map((stage, i) => {
                  const m = metrics.find((x) => stage.match.includes(x.step));
                  let state;
                  if (m) state = m.status === 'error' ? 'error' : 'done';
                  else if (!lastRun) state = 'idle';
                  else if (processing && i === lastDone + 1) state = 'running';
                  else if (processing) state = 'pending';
                  else state = 'skipped';
                  const box = {
                    idle: 'border-[#e6e6e6] bg-[#f9f0ff]/30 text-[#696969]',
                    pending: 'border-[#e6e6e6] bg-[#f9f0ff]/20 opacity-50 text-[#696969]',
                    skipped: 'border-[#e6e6e6] bg-[#f9f0ff]/10 opacity-40 text-[#696969]',
                    running: 'border-[#4a154b] bg-[#4a154b] text-white shadow-lg animate-pulse',
                    done: 'border-[#e6e6e6] bg-[#f9f0ff] text-[#4a154b] font-bold shadow-sm',
                    error: 'border-[#cc4117] bg-[#cc4117]/10 text-[#cc4117]',
                  }[state];
                  const note =
                    state === 'done' ? fmtDuration(m.ms)
                    : state === 'error' ? 'failed'
                    : state === 'running'
                      ? (lastRun && lastRun.currentStep && lastRun.currentStep.includes('page')
                         ? lastRun.currentStep.replace('docling_pdf ', '').replace('pymupdf_pdf ', '')
                         : 'running…')
                    : state === 'skipped' ? 'skipped'
                    : '';
                  const noteColor =
                    state === 'error' ? 'text-[#cc4117]'
                    : state === 'running' ? 'text-white'
                    : 'text-[#696969]';
                  return (
                    <div
                      key={stage.label}
                      className={`flex flex-col items-center p-3.5 rounded-2xl border transition-all ${box}`}
                    >
                      <span className="text-2xl mb-1.5">{stage.icon}</span>
                      <span className="text-xs text-center font-bold">{stage.label}</span>
                      <span className={`text-[10px] mt-1 font-mono h-3.5 ${noteColor}`}>{note}</span>
                    </div>
                  );
                });
              })()}
            </div>

            {/* Timing + token usage summary */}
            {lastRun && (lastRun.tokenUsage || lastRun.indexedTokens != null || (lastRun.metrics || []).length > 0) && (() => {
              const tu = lastRun.tokenUsage || {};
              const totalMs = (lastRun.metrics || []).reduce((a, m) => a + (m.ms || 0), 0);
              const Stat = ({ label, value, sub }) => (
                <div className="flex flex-col px-4 py-3 rounded-2xl bg-white border border-[#e6e6e6] min-w-[120px] shadow-sm">
                  <span className="text-[10px] uppercase font-bold tracking-wider text-[#4a154b]">{label}</span>
                  <span className="text-xl font-extrabold text-[#4a154b] display-stat mt-0.5">{value}</span>
                  {sub != null && <span className="text-[10px] text-[#696969] mt-0.5">{sub}</span>}
                </div>
              );
              return (
                <div className="mt-5 space-y-3">
                  {(lastRun.tokenUsage || lastRun.indexedTokens != null) && (
                    <div className="flex flex-wrap gap-3">
                      {lastRun.tokenUsage && (
                        <>
                          <Stat label="Total Tokens" value={(tu.total_tokens || 0).toLocaleString()} sub={fmtLlmCallsSub(tu)} />
                          <Stat label="Input Tokens" value={(tu.input_tokens || 0).toLocaleString()} />
                          <Stat label="Output Tokens" value={(tu.output_tokens || 0).toLocaleString()} />
                        </>
                      )}
                      {lastRun.indexedTokens != null && (
                        <Stat label="Indexed" value={(lastRun.indexedTokens || 0).toLocaleString()} sub={`tokens · ${lastRun.chunks ?? 0} chunks`} />
                      )}
                    </div>
                  )}
                  {totalMs > 0 && (
                    <div className="flex flex-wrap gap-3 border-t border-[#e6e6e6] pt-3">
                      <Stat label="Total Time" value={fmtDuration(totalMs)} sub={`${(lastRun.metrics || []).length} steps`} />
                      {(lastRun.metrics || []).map((m, i) => (
                        <Stat key={i} label={getStepLabel(m.step)} value={fmtDuration(m.ms)} sub={m.status === 'error' ? 'failed' : 'ok'} />
                      ))}
                    </div>
                  )}
                </div>
              );
            })()}
          </div>
        )}

        {/* Files Section */}
        <div>
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold text-[#4a154b] display-title">
              Processed Document Library {files.length > 0 && `(${files.length})`}
            </h2>
            {files.length > 0 && (() => {
              const totalCostUSD = files.reduce((sum, f) => sum + (f.token_usage?.total_cost_usd || 0), 0);
              return (
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => setCurrency(currency === 'USD' ? 'INR' : 'USD')}
                    className="btn-primary-pill flex items-center gap-2 font-bold"
                  >
                    <CurrencyDollarIcon className="h-4 w-4" />
                    {currency === 'USD' 
                      ? `Total: $${totalCostUSD.toFixed(4)}` 
                      : `Total: ₹${(totalCostUSD * USD_TO_INR).toFixed(2)}`
                    }
                  </button>
                  <button
                    onClick={loadFiles}
                    className="btn-secondary-pill flex items-center gap-2"
                  >
                    <ArrowPathIcon className="h-4 w-4" />
                    Refresh List
                  </button>
                </div>
              );
            })()}
          </div>

          {loading ? (
            <div className="grid gap-4">
              {[1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="bg-white border border-[#e6e6e6] rounded-2xl p-6 animate-pulse shadow-sm"
                  style={{ animationDelay: `${i * 150}ms`, animationDuration: '1.5s' }}
                >
                  <div className="flex items-start gap-4">
                    <div className="p-3 bg-[#f9f0ff] rounded-xl w-12 h-12 flex-shrink-0 border border-[#e6e6e6]" />
                    <div className="flex-1 space-y-3">
                      <div className="h-5 bg-[#f9f0ff] rounded w-1/3" />
                      <div className="h-3 bg-[#f9f0ff] rounded w-1/4" />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : files.length === 0 ? (
            <div className="text-center py-16 bg-white rounded-2xl border border-[#e6e6e6] shadow-sm">
              <DocumentIcon className="h-14 w-14 mx-auto text-[#4a154b] mb-3 opacity-40" />
              <p className="text-lg font-bold text-[#4a154b]">No documents uploaded yet</p>
              <p className="text-xs text-[#696969] mt-1">Upload documents above to populate the vector search index</p>
            </div>
          ) : (
            <div className="grid gap-4">
              {files.map((file) => (
                <div
                  key={file.id}
                  onClick={() => {
                    selectDocument(file.document_id || file.id, file);
                    toggleExpandFile(file.id);
                  }}
                  className={`group bg-white border rounded-2xl p-6 hover:shadow-md transition-all duration-200 cursor-pointer ${
                    selectedDocId === (file.document_id || file.id)
                      ? 'border-[#4a154b] ring-2 ring-[#4a154b]/20 shadow-md'
                      : 'border-[#e6e6e6] hover:border-[#4a154b]/40'
                  }`}
                >
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div className="flex items-center gap-4 flex-1 min-w-0">
                      <div className="p-3.5 bg-[#f9f0ff] rounded-2xl border border-[#e6e6e6] group-hover:bg-[#4a154b] group-hover:text-white transition-all duration-200 flex-shrink-0 text-[#4a154b]">
                        <DocumentIcon className="h-6 w-6" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-3 flex-wrap">
                          <h3 className="font-semibold text-[#4a154b] truncate text-sm max-w-[200px] md:max-w-[400px]">{file.filename}</h3>
                          <span className="text-[9px] font-extrabold text-[#696969] px-2 py-0.5 rounded-full bg-[#f4ede4] uppercase tracking-wider">{file.file_type || 'Document'}</span>
                          
                          {/* Total Time Badge */}
                          {(() => {
                            const totalMs = (file.metrics || []).reduce((sum, m) => sum + (m.ms || 0), 0);
                            return totalMs > 0 ? (
                              <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-[#f9f0ff] border border-[#e6e6e6] text-[10px] font-bold text-[#4a154b]">
                                <ClockIcon className="h-3 w-3 text-[#4a154b]" />
                                {fmtDuration(totalMs)}
                              </span>
                            ) : null;
                          })()}

                          {/* Total Cost Badge */}
                          {file.token_usage?.total_cost_usd !== undefined && (
                            <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-[#f9f0ff] border border-[#e6e6e6] text-[10px] font-bold text-[#4a154b]">
                              <CurrencyDollarIcon className="h-3 w-3 text-[#4a154b]" />
                              {currency === 'USD' 
                                ? `$${file.token_usage.total_cost_usd.toFixed(4)}`
                                : `₹${(file.token_usage.total_cost_usd * USD_TO_INR).toFixed(2)}`
                              }
                            </span>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-2 flex-shrink-0 self-end md:self-center">
                      <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-bold border bg-[#f9f0ff] text-[#4a154b] border-[#e6e6e6]">
                        {getStatusIcon(file.status)}
                        <span className="capitalize">
                          {file.status === 'processing' && file.current_step && file.current_step.includes('page')
                            ? file.current_step.replace('docling_pdf ', '').replace('pymupdf_pdf ', '')
                            : file.status}
                        </span>
                      </div>
                      <button
                        onClick={(e) => { e.stopPropagation(); handleOpenChat(file.id); }}
                        className="btn-primary-pill !p-2.5"
                        title="Chat about this document"
                      >
                        <ChatBubbleLeftIcon className="h-5 w-5" />
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); handleDelete(file.id, file.filename); }}
                        className="btn-secondary-pill !p-2.5 !text-[#cc4117]"
                        title="Delete file"
                      >
                        <TrashIcon className="h-5 w-5" />
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); toggleExpandFile(file.id); }}
                        className="btn-secondary-pill !p-2.5 text-[#4a154b] hover:bg-[#4a154b] hover:text-white transition-all group"
                        title={expandedFiles[file.id] ? "Hide metrics breakdown" : "Show metrics breakdown"}
                      >
                        <ChevronDownIcon className={`h-5 w-5 transition-transform duration-200 ${expandedFiles[file.id] ? 'rotate-180' : ''}`} />
                      </button>
                    </div>
                  </div>

                  {/* Expandable Detailed Statistics Breakdown */}
                  {expandedFiles[file.id] && (file.token_usage || file.indexed_tokens != null || (file.metrics && file.metrics.length > 0)) && (() => {
                    const tu = file.token_usage || {};
                    const totalMs = (file.metrics || []).reduce((sum, m) => sum + (m.ms || 0), 0);
                    
                    const StatBlock = ({ label, value, sub }) => (
                      <div className="flex flex-col px-3.5 py-2.5 rounded-xl bg-[#f9f0ff]/50 border border-[#e6e6e6] min-w-[110px]">
                        <span className="text-[9px] uppercase tracking-wider text-[#4a154b] font-bold">{label}</span>
                        <span className="text-base font-extrabold text-[#4a154b] mt-0.5 display-stat">{value}</span>
                        {sub && <span className="text-[9px] text-[#696969] mt-0.5 leading-none">{sub}</span>}
                      </div>
                    );

                    return (
                      <div className="mt-4 pt-4 border-t border-[#e6e6e6] space-y-3">
                        {/* Row 1: Tokens */}
                        {(file.token_usage || file.indexed_tokens != null) && (
                          <div className="flex flex-wrap gap-2.5">
                            {file.token_usage && (
                              <>
                                <StatBlock label="Total Tokens" value={(tu.total_tokens || 0).toLocaleString()} sub={fmtLlmCallsSub(tu)} />
                                <StatBlock label="Input Tokens" value={(tu.input_tokens || 0).toLocaleString()} />
                                <StatBlock label="Output Tokens" value={(tu.output_tokens || 0).toLocaleString()} />
                              </>
                            )}
                            {file.indexed_tokens != null && (
                              <StatBlock label="Indexed" value={(file.indexed_tokens || 0).toLocaleString()} sub={`tokens · ${file.chunk_count || 0} chunks`} />
                            )}
                          </div>
                        )}

                        {/* Row 1.5: Costs */}
                        {file.token_usage?.by_model && Object.keys(file.token_usage.by_model).length > 0 && (
                          <div className="flex flex-wrap gap-2.5 border-t border-[#e6e6e6] pt-3">
                            <StatBlock 
                              label="Total Cost" 
                              value={currency === 'USD' 
                                ? `$${(tu.total_cost_usd || 0).toFixed(4)}`
                                : `₹${((tu.total_cost_usd || 0) * USD_TO_INR).toFixed(2)}`} 
                            />
                            {Object.entries(file.token_usage.by_model)
                              .filter(([model]) => !model.toLowerCase().includes('embedding'))
                              .map(([model, metrics]) => (
                                <StatBlock 
                                  key={model} 
                                  label={model.split('/').pop()} 
                                  value={currency === 'USD'
                                    ? `$${(metrics.total_cost || 0).toFixed(4)}`
                                    : `₹${((metrics.total_cost || 0) * USD_TO_INR).toFixed(2)}`} 
                                  sub={currency === 'USD'
                                    ? `in: $${(metrics.input_cost || 0).toFixed(4)} · out: $${(metrics.output_cost || 0).toFixed(4)}`
                                    : `in: ₹${((metrics.input_cost || 0) * USD_TO_INR).toFixed(4)} · out: ₹${((metrics.output_cost || 0) * USD_TO_INR).toFixed(4)}`} 
                                />
                            ))}
                          </div>
                        )}

                        {/* Row 2: Step Timings */}
                        {(totalMs > 0 || (file.metrics && file.metrics.length > 0)) && (
                          <div className="flex flex-wrap gap-2.5 border-t border-[#e6e6e6] pt-3">
                            {totalMs > 0 && (
                              <StatBlock label="Total Time" value={fmtDuration(totalMs)} sub={`${(file.metrics || []).length} steps`} />
                            )}
                            {(file.metrics || []).map((m, i) => (
                              <StatBlock key={i} label={getStepLabel(m.step)} value={fmtDuration(m.ms)} sub={m.status === 'error' ? 'failed' : 'ok'} />
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })()}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {doclingAlert && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#1d1d1d]/60 backdrop-blur-sm p-4">
          <div className="bg-white border border-[#e6e6e6] rounded-2xl p-7 max-w-md w-full shadow-2xl space-y-4">
            <div className="flex items-center gap-3 text-[#cc4117]">
              <ExclamationCircleIcon className="h-7 w-7 flex-shrink-0" />
              <h3 className="text-lg font-bold text-[#4a154b]">Docling GPU Server Unreachable</h3>
            </div>
            <p className="text-sm text-[#1d1d1d]">
              The remote Docling server could not be reached.
            </p>
            <p className="text-sm text-[#696969]">
              This PDF upload will automatically fall back to <strong>Local CPU mode</strong> instead.
            </p>
            <div className="p-3.5 bg-[#f9f0ff] rounded-xl text-xs text-[#696969] border border-[#e6e6e6]">
              💡 To silence this warning, go to <strong>Settings</strong> and change Docling Extraction Mode to <strong>Local CPU</strong>.
            </div>
            <div className="flex justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={() => navigate('/settings')}
                className="btn-secondary-pill"
              >
                Open Settings
              </button>
              <button
                type="button"
                onClick={() => setDoclingAlert(null)}
                className="btn-primary-pill"
              >
                Dismiss & Continue
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default IngestionPage;