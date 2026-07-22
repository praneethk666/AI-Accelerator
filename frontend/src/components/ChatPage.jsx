import React, { useState, useRef, useEffect } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  sendAgentChat, getAgentSessions, getAgentSession, deleteAgentSession,
  patchAgentSession, stageFile, getFile,
} from '../api';
import {
  PaperAirplaneIcon,
  SparklesIcon,
  DocumentIcon,
  ExclamationCircleIcon,
  ArrowPathIcon,
  PlusIcon,
  TrashIcon,
  PencilIcon,
  PaperClipIcon,
  XMarkIcon,
  CheckIcon,
  WrenchScrewdriverIcon,
  CloudArrowUpIcon,
  CircleStackIcon,
  EllipsisVerticalIcon,
  ChevronDownIcon,
  ChevronUpIcon,
} from '@heroicons/react/24/outline';

// "2m ago" / "3h ago" / "5d ago" — good enough for a sidebar, no library needed.
const relativeTime = (iso) => {
  if (!iso) return '';
  // If the string doesn't contain 'Z' or a timezone offset (+/-), assume UTC.
  const date = (iso.endsWith('Z') || iso.includes('+') || iso.includes('-'))
    ? new Date(iso)
    : new Date(iso + 'Z');   // treat as UTC
  const diffMs = Date.now() - date.getTime();
  const min = Math.floor(diffMs / 60000);
  if (min < 1) return 'just now';
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  return `${Math.floor(hr / 24)}d ago`;
};

const newSessionId = () =>
  crypto.randomUUID ? crypto.randomUUID() : `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;

const PinIcon = ({ className }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.5"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={`${className} transform rotate-45`}
  >
    <line x1="12" y1="17" x2="12" y2="22" />
    <path d="M5 17h14v-1.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V6a3 3 0 0 0-6 0v4.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24Z" />
  </svg>
);

const ChatPage = () => {
  const [searchParams] = useSearchParams();
  const [sessions, setSessions] = useState([]);
  const [sessionId, setSessionId] = useState(newSessionId);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [attachedFile, setAttachedFile] = useState(null); // {file_path, filename}
  const [attaching, setAttaching] = useState(false);
  const [contextFile, setContextFile] = useState(null); // from ?fileId= (informational only)
  const [menuOpen, setMenuOpen] = useState(null); // session_id of open dropdown
  const [renamingId, setRenamingId] = useState(null); // session_id being renamed
  const [renameValue, setRenameValue] = useState('');
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);
  // Mirrors sessionId so an in-flight reply can check whether the user switched
  // conversations before it landed (otherwise A's answer appends under B).
  const sessionIdRef = useRef(sessionId);
  useEffect(() => { sessionIdRef.current = sessionId; }, [sessionId]);

  // Close dropdown on click outside (uses document listener, no shared ref)
  useEffect(() => {
    const handler = (e) => {
      if (!e.target.closest('[data-menu]')) {
        setMenuOpen(null);
        setRenamingId(null);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const fileId = searchParams.get('fileId');

  useEffect(() => { loadSessions(); }, []);
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, loading]);

  useEffect(() => {
    if (!fileId) return;
    getFile(fileId).then((res) => setContextFile(res.data)).catch(() => {});
  }, [fileId]);

  const loadSessions = async () => {
    try {
      const res = await getAgentSessions();
      setSessions(res.data || []);
    } catch (err) {
      console.error('Failed to load sessions:', err);
    }
  };

  const handleNewChat = () => {
    setSessionId(newSessionId());
    setMessages([]);
    setAttachedFile(null);
    setError(null);
  };

  const handleSelectSession = async (id) => {
    if (id === sessionId && messages.length > 0) return;
    try {
      const res = await getAgentSession(id);
      setMessages(
        (res.data || []).map((m) => ({
          role: m.role,
          content: typeof m.content === 'string' ? m.content : (m.content ? JSON.stringify(m.content) : ''),
          toolCalls: m.tool_calls || [],
          tokenUsage: m.token_usage || null,
          traceId: m.trace_id || null,
        }))
      );
      setSessionId(id);
      setAttachedFile(null);
      setError(null);
    } catch (err) {
      console.error('Failed to load session:', err);
      setError('Could not load that conversation');
    }
  };

  const handleDeleteSession = async (id, e) => {
    e.stopPropagation();
    setMenuOpen(null);
    try {
      await deleteAgentSession(id);
      setSessions((prev) => prev.filter((s) => s.session_id !== id));
      if (id === sessionId) handleNewChat();
    } catch (err) {
      console.error('Failed to delete session:', err);
    }
  };

  const handleRenameStart = (id, currentTitle, e) => {
    e.stopPropagation();
    setRenamingId(id);
    setRenameValue(currentTitle);
    setMenuOpen(null);
  };

  const handleRenameSave = async (id) => {
    const trimmed = renameValue.trim();
    if (!trimmed) { setRenamingId(null); return; }
    try {
      await patchAgentSession(id, { title: trimmed });
      setSessions((prev) =>
        prev.map((s) => (s.session_id === id ? { ...s, title: trimmed } : s))
      );
    } catch (err) {
      console.error('Failed to rename session:', err);
    }
    setRenamingId(null);
  };

  const handleRenameKeyDown = (e, id) => {
    if (e.key === 'Enter') { e.preventDefault(); handleRenameSave(id); }
    if (e.key === 'Escape') { setRenamingId(null); }
  };

  const handleTogglePin = async (id, currentlyPinned, e) => {
    e.stopPropagation();
    setMenuOpen(null);
    try {
      await patchAgentSession(id, { pinned: !currentlyPinned });
      setSessions((prev) =>
        prev.map((s) =>
          s.session_id === id ? { ...s, pinned: !currentlyPinned } : s
        )
      );
    } catch (err) {
      console.error('Failed to toggle pin:', err);
    }
  };

  const handleAttach = async (e) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    setAttaching(true);
    setError(null);
    try {
      const res = await stageFile(file);
      setAttachedFile(res.data);
    } catch (err) {
      console.error('Attach failed:', err);
      setError(err.message || 'Could not attach that file');
    } finally {
      setAttaching(false);
    }
  };

  // What the model sees vs. what the user sees in their own bubble differ when a
  // file is attached — the model gets an explicit path to reference.
  const composeSentText = (text, attachment) => {
    if (!attachment) return text;
    const base = text.trim() || 'Please ingest this file.';
    return `${base}\n\n[Attached file: ${attachment.filename} — path: ${attachment.file_path}]`;
  };

  const agentMessageFromResponse = (data, originalText) => ({
    role: 'assistant',
    status: data.status,
    content: typeof data.answer === 'string' ? data.answer : (data.answer ? JSON.stringify(data.answer) : ''),
    toolCalls: data.tool_calls || [],
    pending: data.pending || [],
    question: data.question || null,
    options: data.options || [],
    tokenUsage: data.token_usage || null,
    traceId: data.trace_id || null,
    originalText,
  });

  // The agent asked the user to choose (needs_clarification): send their pick as the
  // next message so the agent proceeds scoped to that choice.
  const handleClarify = async (optionText, msgIdx) => {
    const reqSession = sessionId;
    // Mark the originating prompt answered so its option buttons go inert (otherwise
    // they stay clickable and a stray click re-fires the choice as a whole new turn).
    setMessages((prev) => [
      ...prev.map((m, i) => (i === msgIdx ? { ...m, clarifyAnswered: true } : m)),
      { role: 'user', content: optionText },
    ]);
    setLoading(true);
    setError(null);
    try {
      const res = await sendAgentChat(optionText, reqSession, false);
      if (sessionIdRef.current !== reqSession) { loadSessions(); return; }
      setMessages((prev) => [...prev, agentMessageFromResponse(res.data, optionText)]);
      loadSessions();
    } catch (err) {
      console.error('Clarify error:', err);
      if (sessionIdRef.current === reqSession) setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSend = async () => {
    if (!input.trim() && !attachedFile) return;

    const displayText = attachedFile
      ? `${input.trim()}${input.trim() ? '\n\n' : ''}📎 ${attachedFile.filename}`
      : input;
    const sentText = composeSentText(input, attachedFile);

    setMessages((prev) => [...prev, { role: 'user', content: displayText }]);
    setInput('');
    setAttachedFile(null);
    setLoading(true);
    setError(null);
    if (textareaRef.current) textareaRef.current.style.height = 'auto';

    const reqSession = sessionId;
    try {
      const res = await sendAgentChat(sentText, reqSession, false);
      // Drop the reply if the user switched conversations while it was in flight —
      // it's already persisted server-side under reqSession.
      if (sessionIdRef.current !== reqSession) { loadSessions(); return; }
      setMessages((prev) => [...prev, agentMessageFromResponse(res.data, sentText)]);
      loadSessions();
    } catch (err) {
      console.error('Chat error:', err);
      if (sessionIdRef.current !== reqSession) return;
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `Error: ${err.message || 'Failed to reach the agent. Make sure the backend is running.'}`,
          isError: true,
        },
      ]);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleApproval = async (msgIdx, approved) => {
    const msg = messages[msgIdx];
    if (!approved) {
      setMessages((prev) => prev.map((m, i) => (i === msgIdx ? { ...m, status: 'declined' } : m)));
      return;
    }
    setLoading(true);
    const reqSession = sessionId;
    try {
      const res = await sendAgentChat(msg.originalText, reqSession, true, msg.pending || []);
      if (sessionIdRef.current !== reqSession) { loadSessions(); return; }
      setMessages((prev) =>
        prev.map((m, i) => (i === msgIdx ? agentMessageFromResponse(res.data, msg.originalText) : m))
      );
      loadSessions();
    } catch (err) {
      console.error('Approval error:', err);
      if (sessionIdRef.current === reqSession) setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleTextareaChange = (e) => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  };

  return (
    <div className="flex h-screen bg-slate-900 text-gray-100">
      {/* Sidebar */}
      <div className="w-64 flex-shrink-0 flex flex-col bg-slate-950/60 border-r border-slate-800">
        <div className="p-3">
          <button
            onClick={handleNewChat}
            className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg border border-slate-700 hover:bg-slate-800 text-sm font-medium text-gray-200 transition-colors"
          >
            <PlusIcon className="h-4 w-4" />
            New chat
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-0.5">
          {sessions.length === 0 && (
            <p className="text-xs text-gray-600 px-3 py-2">No conversations yet</p>
          )}
          {[...sessions]
            .sort((a, b) => {
              if (a.pinned && !b.pinned) return -1;
              if (!a.pinned && b.pinned) return 1;
              return new Date(b.last_active) - new Date(a.last_active);
            })
            .map((s) => (
            <div
              key={s.session_id}
              onClick={() => { setMenuOpen(null); handleSelectSession(s.session_id); }}
              className={`group flex items-center justify-between gap-2 px-3 py-2 rounded-lg cursor-pointer text-sm transition-colors ${
                s.session_id === sessionId
                  ? 'bg-slate-800 text-white'
                  : 'text-gray-400 hover:bg-slate-800/60 hover:text-gray-200'
              }`}
              title={s.title}
            >
              <div className="min-w-0 flex-1">
                {renamingId === s.session_id ? (
                  <input
                    type="text"
                    value={renameValue}
                    onChange={(e) => setRenameValue(e.target.value)}
                    onBlur={() => handleRenameSave(s.session_id)}
                    onKeyDown={(e) => handleRenameKeyDown(e, s.session_id)}
                    className="w-full bg-slate-700 text-white text-sm px-2 py-0.5 rounded border border-blue-500 outline-none"
                    autoFocus
                    onClick={(e) => e.stopPropagation()}
                  />
                ) : (
                  <>
                    <p className="truncate flex items-center gap-1.5">
                      {s.pinned && <PinIcon className="h-3.5 w-3.5 text-slate-400 flex-shrink-0" />}
                      {s.title}
                    </p>
                    <p className="text-[10px] text-gray-600">{relativeTime(s.last_active)}</p>
                  </>
                )}
              </div>
              <div className="relative flex-shrink-0" data-menu>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setMenuOpen(menuOpen === s.session_id ? null : s.session_id);
                  }}
                  className="text-slate-500 hover:text-blue-400 hover:bg-slate-700/50 rounded p-0.5 transition-colors"
                  title="More"
                >
                  <EllipsisVerticalIcon className="h-4 w-4" />
                </button>
                {menuOpen === s.session_id && (
                  <div className="absolute right-0 top-6 z-50 w-40 bg-slate-800 border border-slate-600 rounded-lg shadow-xl py-1 text-sm" data-menu>
                    <button
                      onClick={(e) => handleRenameStart(s.session_id, s.title, e)}
                      className="w-full flex items-center gap-2 px-3 py-2 text-slate-300 hover:bg-slate-700 hover:text-white text-left group"
                    >
                      <PencilIcon className="h-4 w-4 text-slate-400 group-hover:text-slate-200" />
                      Rename
                    </button>
                    <button
                      onClick={(e) => handleTogglePin(s.session_id, s.pinned, e)}
                      className="w-full flex items-center gap-2 px-3 py-2 text-slate-300 hover:bg-slate-700 hover:text-white text-left group"
                    >
                      <PinIcon className="h-4 w-4 text-slate-400 group-hover:text-slate-200" />
                      {s.pinned ? 'Unpin' : 'Pin'}
                    </button>
                    <div className="border-t border-slate-600 my-1" />
                    <button
                      onClick={(e) => handleDeleteSession(s.session_id, e)}
                      className="w-full flex items-center gap-2 px-3 py-2 text-red-400 hover:bg-slate-700 hover:text-red-300 text-left group"
                    >
                      <TrashIcon className="h-4 w-4 text-red-400/80 group-hover:text-red-300" />
                      Delete
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        <div className="p-3 border-t border-slate-800">
          <Link
            to="/ingest"
            className="flex items-center gap-2 px-3 py-2 rounded-lg text-gray-400 hover:bg-slate-800 hover:text-gray-200 text-sm transition-colors"
          >
            <CloudArrowUpIcon className="h-4 w-4" />
            Ingestion
          </Link>
        </div>
      </div>

      {/* Main chat column */}
      <div className="flex-1 flex flex-col min-w-0">
        {contextFile && (
          <div className="border-b border-slate-800 px-4 py-2 text-xs text-gray-500 flex items-center gap-1.5 flex-shrink-0">
            <DocumentIcon className="h-3.5 w-3.5" />
            Context: {contextFile.filename}
          </div>
        )}

        <div className="flex-1 overflow-y-auto">
          <div className="max-w-3xl mx-auto px-4 py-8">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-[60vh] text-center">
                <SparklesIcon className="h-10 w-10 text-slate-600 mb-3" />
                <p className="text-gray-200 text-xl font-medium">How can I help?</p>
                <p className="text-gray-500 text-sm mt-2 max-w-sm">
                  Ask a question about your documents, or attach a file to ingest it — the
                  agent will ask before it writes anything.
                </p>
              </div>
            )}

            {messages.map((msg, idx) => (
              <MessageRow key={idx} msg={msg} onApprove={() => handleApproval(idx, true)}
                          onDecline={() => handleApproval(idx, false)}
                          onClarify={(opt) => handleClarify(opt, idx)} loading={loading} />
            ))}

            {loading && (() => {
              let text = 'Thinking...';
              if (messages.length > 0) {
                const lastMsg = messages[messages.length - 1];
                if (lastMsg.role === 'assistant' && lastMsg.pending?.some(p => p.name === 'ingest_document')) {
                  text = 'Ingesting...';
                } else if (lastMsg.role === 'user' && (lastMsg.content?.includes('📎') || lastMsg.content?.includes('[Attached file:'))) {
                  text = 'Ingesting...';
                }
              }
              return (
                <div className="flex items-center gap-2 text-gray-500 text-sm py-4">
                  <ArrowPathIcon className="h-4 w-4 animate-spin" />
                  {text}
                </div>
              );
            })()}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {error && (
          <div className="max-w-3xl mx-auto w-full px-4 flex-shrink-0">
            <div className="mb-2 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm flex items-start gap-2">
              <ExclamationCircleIcon className="h-4 w-4 flex-shrink-0 mt-0.5" />
              {error}
            </div>
          </div>
        )}

        <div className="border-t border-slate-800 bg-slate-900 p-4 flex-shrink-0">
          <div className="max-w-3xl mx-auto">
            {attachedFile && (
              <div className="mb-2 inline-flex items-center gap-2 bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-gray-300">
                <DocumentIcon className="h-3.5 w-3.5 text-blue-400" />
                {attachedFile.filename}
                <button onClick={() => setAttachedFile(null)} title="Remove attachment">
                  <XMarkIcon className="h-3.5 w-3.5 text-gray-500 hover:text-gray-300" />
                </button>
              </div>
            )}
            <div className="flex items-end gap-1 bg-slate-800 border border-slate-700 rounded-2xl px-2 py-1.5 focus-within:border-blue-500 transition-colors">
              <input type="file" ref={fileInputRef} className="hidden" onChange={handleAttach} />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={attaching || loading}
                title="Attach a file to ingest"
                className="p-2 text-gray-400 hover:text-gray-200 disabled:opacity-50 flex-shrink-0"
              >
                {attaching ? (
                  <ArrowPathIcon className="h-5 w-5 animate-spin" />
                ) : (
                  <PaperClipIcon className="h-5 w-5" />
                )}
              </button>
              <textarea
                ref={textareaRef}
                rows={1}
                className="flex-1 bg-transparent resize-none outline-none text-white placeholder-gray-500 py-2 max-h-40"
                value={input}
                onChange={handleTextareaChange}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder="Message the agent... (Shift+Enter for new line)"
                disabled={loading}
              />
              <button
                onClick={handleSend}
                disabled={loading || (!input.trim() && !attachedFile)}
                className={`p-2.5 rounded-full flex-shrink-0 transition-colors ${
                  loading || (!input.trim() && !attachedFile)
                    ? 'bg-slate-700 text-gray-500 cursor-not-allowed'
                    : 'bg-blue-600 text-white hover:bg-blue-700'
                }`}
              >
                <PaperAirplaneIcon className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Best-effort: search_documents' tool result is a JSON string of
// {answer, citations, sources}. Pull sources out for a citation strip; anything
// else (a different tool, a malformed/blocked result) is skipped, not an error.
const parseSources = (toolCalls) => {
  const sources = [];
  for (const call of toolCalls || []) {
    if (call.name !== 'search_documents' || typeof call.result !== 'string') continue;
    try {
      const parsed = JSON.parse(call.result);
      if (Array.isArray(parsed.sources)) sources.push(...parsed.sources);
    } catch {
      // not JSON (e.g. a blocked/error string) — nothing to show
    }
  }
  return sources;
};

const MessageRow = ({ msg, onApprove, onDecline, onClarify, loading }) => {
  const isUser = msg.role === 'user';
  const sources = !isUser ? parseSources(msg.toolCalls) : [];
  const [isSearchExpanded, setIsSearchExpanded] = useState(false);
  const [isListExpanded, setIsListExpanded] = useState(false);

  return (
    <div className={`py-3 flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={isUser ? 'max-w-lg' : 'max-w-full w-full'}>
        {isUser ? (
          <div className="bg-slate-800 rounded-2xl px-4 py-2.5 text-gray-100 whitespace-pre-wrap">
            {msg.content}
          </div>
        ) : (
          <div className="flex gap-3">
            <SparklesIcon className="h-5 w-5 text-blue-400 flex-shrink-0 mt-1" />
            <div className="min-w-0 flex-1">
              {msg.content && (
                <div
                  className={`prose prose-invert prose-sm max-w-none leading-relaxed ${
                    msg.isError ? 'text-red-300' : 'text-gray-100'
                  }`}
                >
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                </div>
              )}

              {/* Tool calls this turn (reads that ran, or a blocked write) */}
              {msg.toolCalls && msg.toolCalls.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {msg.toolCalls.map((call, i) => {
                    const isSearch = call.name === 'search_documents';
                    const isList = call.name === 'list_documents';
                    
                    if (isSearch && sources.length > 0) {
                      return (
                        <button
                          key={i}
                          onClick={() => {
                            setIsSearchExpanded(!isSearchExpanded);
                            setIsListExpanded(false); // collapse other
                          }}
                          className="inline-flex items-center gap-1.5 text-xs text-blue-300 bg-blue-500/10 border border-blue-500/20 rounded px-2.5 py-1.5 hover:bg-blue-500/20 transition-all cursor-pointer font-medium"
                        >
                          <WrenchScrewdriverIcon className="h-3 w-3" />
                          <span className="font-mono">{call.name}</span>
                          {isSearchExpanded ? (
                            <ChevronUpIcon className="h-3.5 w-3.5 text-blue-400" />
                          ) : (
                            <ChevronDownIcon className="h-3.5 w-3.5 text-blue-400" />
                          )}
                        </button>
                      );
                    }
                    
                    if (isList) {
                      return (
                        <button
                          key={i}
                          onClick={() => {
                            setIsListExpanded(!isListExpanded);
                            setIsSearchExpanded(false); // collapse other
                          }}
                          className="inline-flex items-center gap-1.5 text-xs text-blue-300 bg-blue-500/10 border border-blue-500/20 rounded px-2.5 py-1.5 hover:bg-blue-500/20 transition-all cursor-pointer font-medium"
                        >
                          <WrenchScrewdriverIcon className="h-3 w-3" />
                          <span className="font-mono">{call.name}</span>
                          {isListExpanded ? (
                            <ChevronUpIcon className="h-3.5 w-3.5 text-blue-400" />
                          ) : (
                            <ChevronDownIcon className="h-3.5 w-3.5 text-blue-400" />
                          )}
                        </button>
                      );
                    }

                    return (
                      <span
                        key={i}
                        className="inline-flex items-center gap-1.5 text-xs text-blue-300/60 bg-blue-500/5 border border-blue-500/10 rounded px-2 py-1 select-none"
                      >
                        <WrenchScrewdriverIcon className="h-3 w-3 opacity-60" />
                        <span className="font-mono">{call.name}</span>
                      </span>
                    );
                  })}
                </div>
              )}

              {/* Inline citation list (expanded from list_documents) */}
              {isListExpanded && (
                <div className="mt-3 space-y-1.5 max-w-md bg-slate-900/40 border border-slate-800 rounded-xl p-3">
                  <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2">Cited Documents</p>
                  {(() => {
                    const uniqueFiles = Array.from(new Set(sources.map(s => s.filename)));
                    if (uniqueFiles.length === 0) {
                      return <p className="text-xs text-gray-500 italic p-1">No documents were cited in this search query.</p>;
                    }
                    return uniqueFiles.map((filename, idx) => (
                      <div key={idx} className="flex items-center gap-2.5 text-xs text-gray-300 bg-slate-800/40 border border-slate-700/60 rounded px-3 py-2">
                        <DocumentIcon className="h-4 w-4 text-blue-400 flex-shrink-0" />
                        <span className="font-medium truncate">{filename}</span>
                      </div>
                    ));
                  })()}
                </div>
              )}

              {/* Inline chunks listing (expanded from search_documents) */}
              {isSearchExpanded && sources.length > 0 && (
                <div className="mt-3 space-y-2 max-w-2xl bg-slate-900/40 border border-slate-800 rounded-xl p-3 animate-fade-in">
                  <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2">Search Sources</p>
                  {sources.map((s, i) => (
                    <div key={i} className="text-xs text-gray-300 bg-slate-800/40 border border-slate-700/60 rounded-xl p-3.5 space-y-2">
                      <div className="flex items-center justify-between border-b border-slate-850 pb-1.5">
                        <span className="font-semibold text-blue-400">{s.filename}</span>
                        {s.page && <span className="text-gray-500 text-[10px]">Page {s.page}</span>}
                      </div>
                      {s.snippet && (
                        <p className="italic text-gray-300 leading-relaxed whitespace-pre-wrap font-mono text-[11px] bg-slate-950/30 p-2.5 rounded border border-slate-800/50">
                          "{s.snippet}"
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* Token usage for this turn.
                  "Context" = the largest single LLM call's input size this turn —
                  i.e. how much conversation/tool-result context was actually sent
                  to the model at its fullest point. Providers don't return a
                  model's max context window per-call, so this isn't a fraction of
                  one — it's the real number of tokens that went in. */}
              {msg.tokenUsage && (
                <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-gray-500">
                  {msg.tokenUsage && (
                    <>
                      <span className="inline-flex items-center gap-1 text-gray-300" title="Input + output tokens across every model call this turn">
                        <CircleStackIcon className="h-3.5 w-3.5" />
                        Tokens used: {msg.tokenUsage.total_tokens}
                      </span>
                      <span title="Tokens sent to the model (prompt, history, tool results)">
                        Input: {msg.tokenUsage.input_tokens}
                      </span>
                      <span title="Tokens the model generated">
                        Output: {msg.tokenUsage.output_tokens}
                      </span>
                      {msg.tokenUsage.reasoning_tokens > 0 && (
                        <span title="Reasoning/thinking tokens (subset of output tokens)">
                          Thinking: {msg.tokenUsage.reasoning_tokens}
                        </span>
                      )}
                      <span title="Largest single call's input size this turn — how much context (history + tool results) was actually sent to the model">
                        Context: {msg.tokenUsage.context_tokens}
                      </span>
                    </>
                  )}
                </div>
              )}

              {/* Write awaiting approval */}
              {msg.status === 'needs_approval' && (
                <div className="mt-3 pt-3 border-t border-slate-700 space-y-2">
                  {msg.pending.map((p, i) => (
                    <p key={i} className="text-sm text-amber-300">
                      Wants to run <span className="font-mono">{p.name}</span>(
                      <span className="font-mono text-amber-200">
                        {Object.entries(p.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(', ')}
                      </span>
                      )
                    </p>
                  ))}
                  <div className="flex gap-2">
                    <button
                      onClick={onApprove}
                      disabled={loading}
                      className="flex items-center gap-1 px-3 py-1.5 rounded-md text-xs font-medium bg-emerald-600 hover:bg-emerald-700 text-white disabled:opacity-50"
                    >
                      <CheckIcon className="h-3.5 w-3.5" /> Approve
                    </button>
                    <button
                      onClick={onDecline}
                      disabled={loading}
                      className="flex items-center gap-1 px-3 py-1.5 rounded-md text-xs font-medium bg-slate-700 hover:bg-slate-600 text-gray-200 disabled:opacity-50"
                    >
                      <XMarkIcon className="h-3.5 w-3.5" /> Decline
                    </button>
                  </div>
                </div>
              )}
              {/* Agent asked the user to choose (needs_clarification) */}
              {msg.status === 'needs_clarification' && (
                <div className="mt-3 pt-3 border-t border-slate-700 space-y-2">
                  {msg.options && msg.options.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                      {msg.options.map((opt, i) => (
                        <button
                          key={i}
                          onClick={() => onClarify(opt)}
                          disabled={loading || msg.clarifyAnswered}
                          className="px-3 py-1.5 rounded-md text-xs font-medium bg-sky-600 hover:bg-sky-700 text-white disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {opt}
                        </button>
                      ))}
                    </div>
                  ) : (
                    <p className="text-xs text-gray-500 italic">Type your answer below.</p>
                  )}
                </div>
              )}
              {msg.status === 'declined' && (
                <p className="mt-2 text-xs text-gray-500 italic">Declined — not run.</p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatPage;