import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { uploadFile, getFiles, deleteFile, healthCheck, getProgress } from '../api';
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
} from '@heroicons/react/24/outline';

// The real ingestion steps (match tool `name`s in the metrics the API returns).
// Each stage maps to one or more tool names; "Extract" covers whichever extractor
// ran for this file type / pdf kind.
const PIPELINE_STAGES = [
  { label: 'Categorize', icon: '📂', match: ['categorize'] },
  { label: 'Extract', icon: '📊', match: ['docling_pdf', 'pdf_digital', 'scanned_pdf', 'mixed_pdf', 'excel_extraction', 'ppt_extraction', 'word_extraction', 'image_extraction', 'cad_extract'] },
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

const IngestionPage = () => {
  const navigate = useNavigate();
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [serverConnected, setServerConnected] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({});
  // metrics from the most recent upload: { name, status, metrics: [{step, ms, status}] }
  const [lastRun, setLastRun] = useState(null);
  const activePolls = useRef({});

  // Check server health on component mount
  useEffect(() => {
    checkServerHealth();
    loadFiles();
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

  const loadFiles = async () => {
    setLoading(true);
    try {
      const res = await getFiles();
      const filesList = res.data || [];
      setFiles(filesList);
      setError(null);

      // Auto-poll any file that is currently processing in the background
      filesList.forEach((file) => {
        if (file.status === 'processing') {
          pollProgress(file.document_id || file.id, file.filename);
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

    setUploading(true);
    setError(null);

    for (const file of acceptedFiles) {
      try {
        setUploadProgress((prev) => ({ ...prev, [file.name]: 0 }));
        // /upload now returns immediately with status "processing"; the pipeline
        // runs in the background and we poll per-step progress below.
        const res = await uploadFile(file);
        const doc = res.data;
        setFiles((prev) => [doc, ...prev]);
        setLastRun({ name: file.name, status: doc.status, metrics: [] });
        setUploadProgress((prev) => {
          const newProgress = { ...prev };
          delete newProgress[file.name];
          return newProgress;
        });
        pollProgress(doc.document_id, file.name);
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

  // Poll a document's ingestion progress and update the pipeline cards + file row
  // live, until it finishes (ready/failed).
  const pollProgress = (docId, name) => {
    if (activePolls.current[docId]) return;
    activePolls.current[docId] = true;

    let attempts = 0;
    const tick = async () => {
      attempts += 1;
      try {
        const { data } = await getProgress(docId);
        setLastRun({
          name,
          status: data.status,
          metrics: data.metrics || [],
          tokenUsage: data.token_usage,
          indexedTokens: data.indexed_tokens,
          chunks: data.chunks,
        });
        setFiles((prev) =>
          prev.map((f) =>
            (f.document_id === docId || f.id === docId)
              ? { ...f, status: data.status, document_type: data.document_type,
                  industry: data.industry, route: data.route, confidence: data.confidence }
              : f
          )
        );
        if (data.status === 'processing' && attempts < 800) {
          setTimeout(tick, 900);
        } else {
          delete activePolls.current[docId];
          loadFiles(); // final refresh from the DB
        }
      } catch (e) {
        if (attempts < 800) {
          setTimeout(tick, 1500); // transient — keep trying
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
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-gray-100">
      {/* Header */}
      <div className="bg-slate-900/50 border-b border-slate-700 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold text-white">Document Ingestion</h1>
              <p className="text-gray-400 mt-2">Upload and categorize your documents for intelligent processing</p>
            </div>
            <div className="flex items-center gap-3">
              <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm ${serverConnected ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                <div className={`w-2 h-2 rounded-full ${serverConnected ? 'bg-green-400' : 'bg-red-400'}`} />
                {serverConnected ? 'Backend Connected' : 'Backend Offline'}
              </div>
              <button
                onClick={() => navigate('/chat')}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium bg-blue-600 hover:bg-blue-500 text-white transition-colors"
              >
                <ChatBubbleLeftIcon className="h-4 w-4" />
                Agent Chat
              </button>
              <button
                onClick={() => navigate('/settings')}
                title="Configuration"
                className="p-2 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-slate-700/50 transition-colors"
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
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 flex items-start gap-3">
            <ExclamationCircleIcon className="h-5 w-5 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-medium">Error</p>
              <p className="text-sm mt-1">{error}</p>
            </div>
          </div>
        )}

        {/* Upload Area */}
        <div
          {...getRootProps()}
          className={`mb-8 border-2 border-dashed rounded-xl p-12 text-center transition-all cursor-pointer ${
            isDragActive
              ? 'border-blue-500 bg-blue-500/10'
              : uploading || !serverConnected
                ? 'border-gray-600 bg-slate-800/50 opacity-50 cursor-not-allowed'
                : 'border-slate-600 bg-slate-800/30 hover:border-blue-500 hover:bg-blue-500/5'
          }`}
        >
          <input {...getInputProps()} />
          <CloudArrowUpIcon className="h-16 w-16 mx-auto text-slate-400 mb-3" />
          <p className="text-xl font-semibold text-white mb-1">
            {uploading ? 'Uploading...' : isDragActive ? 'Drop files here' : 'Drop files or click to browse'}
          </p>
          <p className="text-sm text-gray-400 mb-4">
            Supported: PDF, Docx, Excel, PowerPoint, Images (PNG, JPG, GIF, BMP)
          </p>
          <button
            disabled={uploading || !serverConnected}
            className={`inline-block px-6 py-2 rounded-lg font-medium transition-colors ${
              uploading || !serverConnected
                ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            {uploading ? 'Uploading...' : 'Browse Files'}
          </button>
        </div>

        {/* Upload Progress */}
        {Object.keys(uploadProgress).length > 0 && (
          <div className="mb-8 space-y-3">
            {Object.entries(uploadProgress).map(([filename, progress]) => (
              <div key={filename} className="bg-slate-800/50 p-4 rounded-lg">
                <p className="text-sm text-gray-300 mb-2">{filename}</p>
                <div className="w-full bg-slate-700 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Pipeline Status — reflects the last upload's actual per-step metrics */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">Processing Pipeline</h2>
            {lastRun ? (
              <span className="text-xs text-gray-500 truncate max-w-[60%]">
                {lastRun.name} ·{' '}
                {lastRun.status === 'processing'
                  ? 'processing…'
                  : `${lastRun.status} · ${fmtDuration((lastRun.metrics || []).reduce((a, m) => a + (m.ms || 0), 0))} total`}
              </span>
            ) : (
              <span className="text-xs text-gray-500">upload a document to see step status</span>
            )}
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-2">
            {(() => {
              const metrics = lastRun?.metrics || [];
              const processing = lastRun?.status === 'processing';
              // index of the last stage that already has a metric -> the next one is "running"
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
                  idle: 'border-slate-700 bg-slate-800/30',
                  pending: 'border-slate-700 bg-slate-800/30 opacity-50',
                  skipped: 'border-slate-700 bg-slate-800/20 opacity-50',
                  running: 'border-blue-500/60 bg-blue-500/10 shadow-lg shadow-blue-500/20 animate-pulse',
                  done: 'border-green-500/50 bg-green-500/10 shadow-lg shadow-green-500/10',
                  error: 'border-red-500/50 bg-red-500/10',
                }[state];
                const note =
                  state === 'done' ? fmtDuration(m.ms)
                  : state === 'error' ? 'failed'
                  : state === 'running' ? 'running…'
                  : state === 'skipped' ? 'skipped'
                  : '';
                const noteColor =
                  state === 'error' ? 'text-red-400'
                  : state === 'running' ? 'text-blue-300'
                  : 'text-gray-500';
                return (
                  <div
                    key={stage.label}
                    className={`flex flex-col items-center p-3 rounded-lg border transition-all ${box}`}
                  >
                    <span className="text-2xl mb-1">{stage.icon}</span>
                    <span className="text-xs text-gray-300 text-center font-medium">{stage.label}</span>
                    <span className={`text-[10px] mt-1 h-3 ${noteColor}`}>{note}</span>
                  </div>
                );
              });
            })()}
          </div>

          {/* Timing + token usage summary (from the last run) — labeled stat cards */}
          {lastRun && (lastRun.tokenUsage || lastRun.indexedTokens != null || (lastRun.metrics || []).length > 0) && (() => {
            const tu = lastRun.tokenUsage || {};
            const totalMs = (lastRun.metrics || []).reduce((a, m) => a + (m.ms || 0), 0);
            const Stat = ({ label, value, sub }) => (
              <div className="flex flex-col px-3 py-2 rounded-lg bg-slate-800/50 border border-slate-700/60 min-w-[110px]">
                <span className="text-[10px] uppercase tracking-wider text-gray-500">{label}</span>
                <span className="text-sm font-semibold text-gray-100">{value}</span>
                {sub != null && <span className="text-[10px] text-gray-500 mt-0.5">{sub}</span>}
              </div>
            );
            return (
              <div className="mt-4 flex flex-wrap gap-2">
                {totalMs > 0 && (
                  <Stat label="Total time" value={fmtDuration(totalMs)}
                        sub={`${(lastRun.metrics || []).length} steps`} />
                )}
                {lastRun.tokenUsage && (
                  <>
                    <Stat label="Total tokens" value={(tu.total_tokens || 0).toLocaleString()}
                          sub={`${tu.calls || 0} LLM/vision calls`} />
                    <Stat label="Input tokens" value={(tu.input_tokens || 0).toLocaleString()} />
                    <Stat label="Output tokens" value={(tu.output_tokens || 0).toLocaleString()} />
                  </>
                )}
                {lastRun.indexedTokens != null && (
                  <Stat label="Indexed" value={(lastRun.indexedTokens || 0).toLocaleString()}
                        sub={`tokens · ${lastRun.chunks ?? 0} chunks`} />
                )}
                {tu.by_kind && Object.keys(tu.by_kind).length > 0 &&
                  Object.entries(tu.by_kind).map(([k, v]) => (
                    <Stat key={k} label={k}
                          value={((v.input_tokens || 0) + (v.output_tokens || 0)).toLocaleString()}
                          sub="tokens" />
                  ))}
              </div>
            );
          })()}
        </div>

        {/* Files Section */}
        <div>
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-white">
              Processed Files {files.length > 0 && `(${files.length})`}
            </h2>
            {files.length > 0 && (
              <button
                onClick={loadFiles}
                className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm transition-colors"
              >
                <ArrowPathIcon className="h-4 w-4" />
                Refresh
              </button>
            )}
          </div>

          {loading ? (
            <div className="text-center py-12">
              <ArrowPathIcon className="h-8 w-8 mx-auto text-blue-400 animate-spin mb-3" />
              <p className="text-gray-400">Loading files...</p>
            </div>
          ) : files.length === 0 ? (
            <div className="text-center py-12 bg-slate-800/20 rounded-lg border border-slate-700">
              <DocumentIcon className="h-12 w-12 mx-auto text-gray-500 mb-3" />
              <p className="text-gray-400">No files uploaded yet</p>
              <p className="text-sm text-gray-500 mt-1">Upload files to get started</p>
            </div>
          ) : (
            <div className="grid gap-4">
              {files.map((file) => (
                <div key={file.id} className="group bg-gradient-to-br from-slate-800/60 to-slate-800/40 border border-slate-700 rounded-lg p-6 hover:border-blue-500/30 hover:from-slate-800/80 hover:to-slate-800/60 transition-all duration-300 shadow-lg hover:shadow-blue-500/10">
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-4 flex-1">
                      <div className="p-3 bg-blue-500/10 rounded-lg border border-blue-500/20 group-hover:bg-blue-500/20 transition-colors flex-shrink-0">
                        <DocumentIcon className="h-6 w-6 text-blue-400" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className="font-semibold text-white truncate text-lg">{file.filename}</h3>
                        <p className="text-xs text-gray-500 mt-1">{file.file_type || 'Document'}</p>

                        {/* Timing & Token stats breakdown */}
                        {(file.token_usage || file.indexed_tokens != null || (file.metrics && file.metrics.length > 0)) && (() => {
                          const tu = file.token_usage || {};
                          const totalMs = (file.metrics || []).reduce((sum, m) => sum + (m.ms || 0), 0);
                          
                          const StatBlock = ({ label, value, sub }) => (
                            <div className="flex flex-col px-2.5 py-1.5 rounded bg-slate-900/40 border border-slate-700/50 min-w-[100px]">
                              <span className="text-[9px] uppercase tracking-wider text-gray-500 font-semibold">{label}</span>
                              <span className="text-xs font-semibold text-gray-200">{value}</span>
                              {sub && <span className="text-[9px] text-gray-500 mt-0.5">{sub}</span>}
                            </div>
                          );

                          return (
                            <div className="mt-4 pt-3 border-t border-slate-700/30 flex flex-wrap gap-2">
                              {totalMs > 0 && (
                                <StatBlock
                                  label="Total Time"
                                  value={fmtDuration(totalMs)}
                                  sub={`${(file.metrics || []).length} steps`}
                                />
                              )}
                              {file.token_usage && (
                                <>
                                  <StatBlock
                                    label="Total Tokens"
                                    value={(tu.total_tokens || 0).toLocaleString()}
                                    sub={`${tu.calls || 0} LLM calls`}
                                  />
                                  <StatBlock
                                    label="Input Tokens"
                                    value={(tu.input_tokens || 0).toLocaleString()}
                                  />
                                  <StatBlock
                                    label="Output Tokens"
                                    value={(tu.output_tokens || 0).toLocaleString()}
                                  />
                                </>
                              )}
                              {file.indexed_tokens != null && (
                                <StatBlock
                                  label="Indexed"
                                  value={(file.indexed_tokens || 0).toLocaleString()}
                                  sub={`tokens · ${file.chunk_count || 0} chunks`}
                                />
                              )}
                              {tu.by_kind && Object.entries(tu.by_kind).map(([kind, data]) => {
                                const kindTotal = (data.input_tokens || 0) + (data.output_tokens || 0);
                                if (kindTotal === 0) return null;
                                return (
                                  <StatBlock
                                    key={kind}
                                    label={`${kind}`}
                                    value={kindTotal.toLocaleString()}
                                    sub="tokens"
                                  />
                                );
                              })}
                            </div>
                          );
                        })()}
                      </div>
                    </div>

                    <div className="flex items-center gap-2 flex-shrink-0 mt-4 md:mt-0">
                      <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold ${getStatusColor(file.status)}`}>
                        {getStatusIcon(file.status)}
                        <span>{file.status}</span>
                      </div>
                      <button
                        onClick={() => handleOpenChat(file.id)}
                        className="p-2 hover:bg-blue-500/20 rounded-lg text-blue-400 hover:text-blue-300 transition-all border border-transparent hover:border-blue-500/30 group-hover:opacity-100"
                        title="Chat about this document"
                      >
                        <ChatBubbleLeftIcon className="h-5 w-5" />
                      </button>
                      <button
                        onClick={() => handleDelete(file.id, file.filename)}
                        className="p-2 hover:bg-red-500/20 rounded-lg text-red-400 hover:text-red-300 transition-all border border-transparent hover:border-red-500/30"
                        title="Delete file"
                      >
                        <TrashIcon className="h-5 w-5" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default IngestionPage;