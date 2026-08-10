import React, { useState, useRef, useEffect } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import * as XLSX from 'xlsx';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import axios from 'axios';
import {
  sendAgentChat, getAgentSessions, getAgentSession, deleteAgentSession,
  patchAgentSession, stageFile, getFile, API_BASE_URL, getFiles,
  ingestStagedFile, getProgress, initDirectIngest, cancelDirectIngest,
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
  Cog6ToothIcon,
  EllipsisVerticalIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CommandLineIcon,
  ClockIcon,
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

// Which viewer mode a citation's file should open in.
const VIEWABLE_EXT_TYPE = {
  pdf: 'pdf',
  docx: 'docx', doc: 'docx',
  pptx: 'ppt', ppt: 'ppt',
  xlsx: 'excel', xls: 'excel', xlsm: 'excel',
  png: 'image', jpg: 'image', jpeg: 'image', tif: 'image', tiff: 'image',
};
const fileTypeFromName = (filename) => {
  const ext = (filename || '').split('.').pop()?.toLowerCase();
  return VIEWABLE_EXT_TYPE[ext] || null;
};

// Sources the right-side viewer panel can actually render: PDF pages and PPT
// slides (both need a page/slide number, remapped from s.slide by
// parseSources), docx (whole document, no page concept — matched to a
// citation by snippet text at view time), excel (whole workbook, parsed
// client-side — matched to a citation's sheet), and standalone images (whole
// file, no pagination). Shared by both the auto-open-on-response path
// (ChatPage.updatePageViewer) and the per-message "View Source" button
// (MessageRow) so the two stay in sync.
const buildViewableSources = (sources) => {
  console.log("buildViewableSources input sources:", sources);
  const result = (sources || [])
    .map((s) => ({ ...s, fileType: fileTypeFromName(s.filename) }))
    .filter((s) => {
      if (!s.document_id) return false;
      if (s.fileType === 'pdf') return s.page != null;
      if (s.fileType === 'ppt') return s.page != null;
      if (s.fileType === 'docx') return true;
      if (s.fileType === 'excel') return true;
      if (s.fileType === 'image') return true;
      return false;
    })
    .filter((s, i, arr) => arr.findIndex((x) => {
      if ((x.filename || '').toLowerCase() !== (s.filename || '').toLowerCase()) return false;
      if (s.fileType === 'pdf' || s.fileType === 'ppt') {
        return x.page === s.page;
      }
      if (s.fileType === 'excel') {
        return x.sheet === s.sheet;
      }
      return true;
    }) === i);
  console.log("buildViewableSources output result:", result);
  return result;
};

/**
 * Pre-render $$..$$ (display) and $..$  (inline) math with KaTeX before the
 * markdown parser sees the text. This avoids remark-math's paragraph-boundary
 * ambiguity and ensures fonts/metrics are always correct.
 */
const renderMathInMarkdown = (text) => {
  if (!text) return text;
  // Display math first ($$...$$) — must come before inline to avoid double-parsing
  let result = text.replace(/\$\$([\s\S]*?)\$\$/g, (_, latex) => {
    try {
      return katex.renderToString(latex.trim(), {
        displayMode: true,
        throwOnError: false,
        output: 'html',
      });
    } catch {
      return `$$${latex}$$`;
    }
  });
  // Inline math ($...$) — skip if preceded/followed by another $
  result = result.replace(/(?<!\$)\$([^$\n]+?)\$(?!\$)/g, (_, latex) => {
    try {
      return katex.renderToString(latex.trim(), {
        displayMode: false,
        throwOnError: false,
        output: 'html',
      });
    } catch {
      return `$${latex}$`;
    }
  });
  return result;
};

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

const SidebarToggleIcon = ({ className }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.5"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
  >
    <rect width="18" height="18" x="3" y="3" rx="2" />
    <path d="M9 3v18" />
  </svg>
);

const NewChatIcon = ({ className }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.5"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
  >
    <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
    <line x1="12" y1="9" x2="12" y2="15" />
    <line x1="9" y1="12" x2="15" y2="12" />
  </svg>
);

const Tooltip = ({ children, content }) => {
  const [visible, setVisible] = useState(false);
  return (
    <div
      className="relative flex items-center"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
      onClick={() => setVisible(false)}
    >
      {children}
      {visible && (
        <div className="absolute top-full left-1/2 transform -translate-x-1/2 mt-2 bg-slate-950 text-white text-xs px-2.5 py-1.5 rounded-lg whitespace-nowrap z-[9999] shadow-xl border border-slate-800 font-medium pointer-events-none">
          {content}
        </div>
      )}
    </div>
  );
};

const USD_TO_INR = 95.20;

const ChatPage = () => {
  const [searchParams] = useSearchParams();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [pageViewer, setPageViewer] = useState(null);
  // pageViewer: null | { pages: [{page, documentId, filename, score}], activeIdx: number }
  const [sessions, setSessions] = useState([]);
  const [sessionId, setSessionId] = useState(newSessionId);
  const [messages, setMessages] = useState([]);
  const [currency, setCurrency] = useState('USD');
  const [input, setInput] = useState('');
  const [loadingSessions, setLoadingSessions] = useState({});
  const loading = !!loadingSessions[sessionId];
  const setSessionLoading = (sessId, val) => {
    setLoadingSessions((prev) => ({ ...prev, [sessId]: val }));
  };
  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  };
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
  const abortControllerRef = useRef(null);
  const chatContainerRef = useRef(null);
  const [showChatScrollDown, setShowChatScrollDown] = useState(false);
  const [allFiles, setAllFiles] = useState([]);

  const loadFiles = async () => {
    try {
      const res = await getFiles();
      setAllFiles(res.data || []);
    } catch (err) {
      console.error('Failed to load files:', err);
    }
  };

  const checkChatScroll = () => {
    const el = chatContainerRef.current;
    if (!el) return;
    const isScrollable = el.scrollHeight > el.clientHeight;
    // Show button if user scrolled up by more than 120px from the bottom
    const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    setShowChatScrollDown(isScrollable && !isNearBottom);
  };

  const handleChatScrollToBottom = () => {
    chatContainerRef.current?.scrollTo({
      top: chatContainerRef.current.scrollHeight,
      behavior: 'smooth',
    });
  };

  // Check scroll when messages change or loading changes
  useEffect(() => {
    const timer = setTimeout(() => {
      checkChatScroll();
    }, 100);
    return () => clearTimeout(timer);
  }, [messages, loading]);

  // Clean up direct ingestion intervals on unmount to prevent leaks and double updates
  useEffect(() => {
    return () => {
      if (window.activeDirectPolls) {
        Object.entries(window.activeDirectPolls).forEach(([msgId, poll]) => {
          clearInterval(poll);
        });
        window.activeDirectPolls = {};
      }
      if (loadSessionsTimerRef.current) {
        clearTimeout(loadSessionsTimerRef.current);
      }
    };
  }, []);

  // Mirrors sessionId so an in-flight reply can check whether the user switched
  // conversations before it landed (otherwise A's answer appends under B).
  const sessionIdRef = useRef(sessionId);
  useEffect(() => { sessionIdRef.current = sessionId; }, [sessionId]);

  // Poll database for updates if the last message is a user message but we are not loading.
  // This activates when the user reloads/reopens a session that is still being processed.
  useEffect(() => {
    if (messages.length > 0 && messages[messages.length - 1].role === 'user' && !loading) {
      const interval = setInterval(async () => {
        try {
          const res = await getAgentSession(sessionId);
          const history = res.data || [];
          if (history.length > 0) {
            const latest = history[history.length - 1];
            if (latest && latest.role === 'assistant') {
              // Detect if this conversation had a file attachment so we can show
              // "Ingesting..." instead of "Thinking..." on the assistant placeholder.
              const hadFileAttachment = history.some(
                (m) => m.role === 'user' && typeof m.content === 'string' && m.content.includes('📎')
              );
              setMessages(
                history.map((m) => {
                  let content = typeof m.content === 'string' ? m.content : (m.content ? JSON.stringify(m.content) : '');
                  if (m.role === 'user') {
                    content = content
                      // Replace path annotation with icon + filename
                      .replace(/\s*\[Attached file:\s*(.+?)\s*(?:—|-)\s*path:[^\]]*\]/g, (_, fname) => `\n\n📎 ${fname.trim()}`)
                      // Drop the now-redundant fallback sentinel (legacy stored messages)
                      .replace(/^Please ingest this file\.\s*/i, '')
                      .trim() || content;
                  }
                  return {
                    role: m.role,
                    content,
                    toolCalls: m.tool_calls || [],
                    tokenUsage: m.token_usage || null,
                    traceId: m.trace_id || null,
                    // Restore clarification / approval state from DB metadata
                    status: m.status || m.metadata?.status,
                    options: m.options || m.metadata?.options || [],
                    question: m.question || m.metadata?.question || null,
                    pending: m.pending || m.metadata?.pending || [],
                    clarifyAnswered: m.clarifyAnswered || m.metadata?.clarifyAnswered || false,
                    // Show "Ingesting..." on the empty placeholder if a file was attached
                    ...(m.role === 'assistant' && !m.content && hadFileAttachment
                      ? { isIngesting: true } : {}),
                  };
                })
              );
              clearInterval(interval);
            }
          }
        } catch (err) {
          console.error('Polling error:', err);
        }
      }, 3000);
      return () => clearInterval(interval);
    }
  }, [messages, loading, sessionId]);


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

  // Prevent the BROWSER from zooming the entire UI with Ctrl+scroll.
  // The PDF viewer handles its own internal zoom separately.
  useEffect(() => {
    const preventBrowserZoom = (e) => {
      if (e.ctrlKey) e.preventDefault();
    };
    window.addEventListener('wheel', preventBrowserZoom, { passive: false });
    return () => window.removeEventListener('wheel', preventBrowserZoom);
  }, []);

  // Automatically close sidebar when PDF Viewer opens, and open it when PDF Viewer closes
  const isFirstMount = useRef(true);
  useEffect(() => {
    if (isFirstMount.current) {
      isFirstMount.current = false;
      return;
    }
    if (pageViewer) {
      setSidebarOpen(false);
    } else {
      setSidebarOpen(true);
    }
  }, [pageViewer]);

  const fileId = searchParams.get('fileId');

  useEffect(() => { loadSessions(false, true); loadFiles(); }, []);
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, loading]);

  useEffect(() => {
    if (!fileId) return;
    getFile(fileId).then((res) => setContextFile(res.data)).catch(() => { });
  }, [fileId]);

  const loadSessionsTimerRef = useRef(null);
  const loadSessions = async (autoSelect = false, immediate = false) => {
    const fetchNow = async () => {
      try {
        const res = await getAgentSessions();
        const loaded = res.data || [];
        setSessions(loaded);
        
        if (autoSelect && loaded.length > 0) {
          const sorted = [...loaded].sort((a, b) => {
            if (a.pinned && !b.pinned) return -1;
            if (!a.pinned && b.pinned) return 1;
            return new Date(b.last_active) - new Date(a.last_active);
          });
          if (sorted[0]) {
            handleSelectSession(sorted[0].session_id);
          }
        }
      } catch (err) {
        console.error('Failed to load sessions:', err);
      }
    };

    if (immediate) {
      if (loadSessionsTimerRef.current) {
        clearTimeout(loadSessionsTimerRef.current);
        loadSessionsTimerRef.current = null;
      }
      await fetchNow();
    } else {
      if (loadSessionsTimerRef.current) clearTimeout(loadSessionsTimerRef.current);
      loadSessionsTimerRef.current = setTimeout(fetchNow, 600);
    }
  };

  const handleNewChat = () => {
    const newId = newSessionId();
    setSessionId(newId);
    setMessages([]);
    setAttachedFile(null);
    setError(null);
    setPageViewer(null);
    setSidebarOpen(true);
    setSessions((prev) => {
      // Remove any existing empty placeholders to avoid cluttering the sidebar
      const filtered = prev.filter((s) => !s.isPlaceholder);
      return [
        {
          session_id: newId,
          title: 'New chat',
          pinned: false,
          last_active: new Date().toISOString(),
          isPlaceholder: true,
        },
        ...filtered,
      ];
    });
  };

  const handleSelectSession = async (id) => {
    if (id === sessionId && messages.length > 0) return;
    try {
      const res = await getAgentSession(id);
      const rawHistory = res.data || [];
      // Deduplicate consecutive identical user messages that can accumulate if
      // the same session was submitted more than once (backend saves immediately).
      const deduped = rawHistory.filter((m, i, arr) => {
        if (m.role !== 'user') return true;
        const prev = arr[i - 1];
        return !(prev && prev.role === 'user' && prev.content === m.content);
      });
      const hadFile = deduped.some(
        (m) => m.role === 'user' && typeof m.content === 'string' && m.content.includes('📎')
      );
      setMessages(
        deduped.map((m) => {
          let content = typeof m.content === 'string' ? m.content : (m.content ? JSON.stringify(m.content) : '');
          if (m.role === 'user') {
            content = content
              .replace(/\s*\[Attached file:\s*(.+?)\s*(?:—|-)\s*path:[^\]]*\]/g, (_, fname) => `\n\n📎 ${fname.trim()}`)
              .replace(/^Please ingest this file\.\s*/i, '')
              .trim() || content;
          }
          return {
            role: m.role,
            content,
            toolCalls: m.tool_calls || [],
            tokenUsage: m.token_usage || null,
            traceId: m.trace_id || null,
            // Restore clarification / approval state from DB metadata
            status: m.status || m.metadata?.status,
            options: m.options || m.metadata?.options || [],
            question: m.question || m.metadata?.question || null,
            pending: m.pending || m.metadata?.pending || [],
            clarifyAnswered: m.clarifyAnswered || m.metadata?.clarifyAnswered || false,
            // Restore direct ingestion fields from DB metadata
            type: m.type || m.metadata?.type,
            filename: m.filename || m.metadata?.filename,
            stagedPath: m.stagedPath || m.metadata?.stagedPath,
            documentId: m.documentId || m.metadata?.documentId,
            messageId: m.message_id || m.metadata?.message_id || m.messageId,
            progressPct: m.progressPct || m.metadata?.progressPct || 0,
            metrics: m.metrics || m.metadata?.metrics || [],
            currentStep: m.currentStep || m.metadata?.currentStep || '',
            // Show "Ingesting..." on an empty placeholder when a file was attached
            ...(m.role === 'assistant' && !m.content && hadFile ? { isIngesting: true } : {}),
          };
        })
      );

      // Restart any active direct ingest progress polls
      deduped.forEach((m) => {
        const mType = m.type || m.metadata?.type;
        const docId = m.documentId || m.metadata?.documentId;
        const msgId = m.message_id || m.metadata?.message_id || m.messageId;
        const fname = m.filename || m.metadata?.filename;
        if (mType === 'ingest_progress' && docId && msgId) {
          pollDirectIngest(msgId, docId, fname);
        }
      });

      setSessionId(id);
      setAttachedFile(null);
      setError(null);
      // Clean up any empty placeholders
      setSessions((prev) => prev.filter((s) => !s.isPlaceholder || s.session_id === id));
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
  // When the user typed nothing, skip the fallback sentinel entirely — the agent
  // only needs the file path annotation; the bubble shows just "📎 filename".
  const composeSentText = (text, attachment) => {
    if (!attachment) return text;
    const base = text.trim();
    const annotation = `[Attached file: ${attachment.filename} — path: ${attachment.file_path}]`;
    return base ? `${base}\n\n${annotation}` : annotation;
  };

  const agentMessageFromResponse = (data, originalText) => ({
    role: 'assistant',
    status: data.status,
    content: typeof data.answer === 'string' ? data.answer : (data.answer ? JSON.stringify(data.answer) : ''),
    toolCalls: data.tool_calls || [],
    executionTrace: data.execution_trace || [],
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
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    const reqSession = sessionId;
    setMessages((prev) => [
      ...prev.map((m, i) => (i === msgIdx ? { ...m, clarifyAnswered: true } : m)),
      { role: 'user', content: optionText },
    ]);
    setSessionLoading(reqSession, true);
    setError(null);

    const messageId = crypto.randomUUID ? crypto.randomUUID() : `msg-${Date.now()}-${Math.random().toString(16).slice(2)}`;

    // Add empty placeholder assistant message
    setMessages((prev) => [
      ...prev,
      {
        role: 'assistant',
        content: '',
        toolCalls: [],
        executionTrace: [],
        pending: [],
        question: null,
        options: [],
        tokenUsage: null,
        traceId: null,
        messageId,
        originalText: optionText,
      },
    ]);

    const activeDocId = contextFile?.document_id || pageViewer?.docId || fileId || null;
    try {
      const res = await sendAgentChat(optionText, reqSession, false, null, controller.signal, activeDocId, messageId);
      if (sessionIdRef.current !== reqSession) {
        setSessionLoading(reqSession, false);
        loadSessions();
        return;
      }
      const data = res.data;
      setMessages((prev) =>
        prev.map((m) =>
          m.messageId === messageId
            ? {
                ...m,
                status: data.status,
                content: typeof data.answer === 'string' ? data.answer : (data.answer ? JSON.stringify(data.answer) : ''),
                toolCalls: data.tool_calls || [],
                executionTrace: data.execution_trace || [],
                pending: data.pending || [],
                question: data.question || null,
                options: data.options || [],
                tokenUsage: data.token_usage || null,
                traceId: data.trace_id || null,
              }
            : m
        )
      );

      setSessionLoading(reqSession, false);
      loadSessions();
      updatePageViewer(data);
    } catch (err) {
      if (axios.isCancel(err) || err.name === 'CanceledError' || err.name === 'AbortError') {
        if (sessionIdRef.current === reqSession) {
          setSessionLoading(reqSession, false);
          setMessages((prev) =>
            prev.map((m) =>
              m.messageId === messageId
                ? {
                    ...m,
                    content: 'Generation stopped.',
                  }
                : m
            )
          );
        }
        return;
      }
      console.error('Clarify error:', err);
      if (sessionIdRef.current !== reqSession) return;
      setSessionLoading(reqSession, false);
      setMessages((prev) =>
        prev.map((m) =>
          m.messageId === messageId
            ? {
                ...m,
                content: `Error: ${err.message || 'Failed to reach the agent. Make sure the backend is running.'}`,
                isError: true,
              }
            : m
        )
      );
      setError(err.message);
    } finally {
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
    }
  };

  // Extract page-level sources from a response and open/close the page viewer.
  // Reuses parseSources (slide/sheet -> page remap) + buildViewableSources
  // (fileType tagging, filtering, dedup, sort) so this stays in sync with the
  // per-message "View Source" button in MessageRow.
  const updatePageViewer = (data) => {
    const allSources = parseSources(data.tool_calls || [], allFiles);
    const viewableSources = buildViewableSources(allSources);

    if (viewableSources.length > 0) {
      setPageViewer({ pages: viewableSources, activeIdx: 0 });
    } else {
      setPageViewer(null);
    }
  };

  const handleSend = async () => {
    if (!input.trim() && !attachedFile) return;

    // ── Fast path: file only (no question text) ───────────────────────────────
    // Bypass the LLM agent entirely: show an inline approval card → on approve,
    // call /files/ingest-staged directly → show live progress → success message.
    if (attachedFile && !input.trim()) {
      handleDirectIngest();
      return;
    }

    // ── Normal path: text (+ optional file) → send to agent ──────────────────
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    const displayText = attachedFile
      ? `${input.trim()}${input.trim() ? '\n\n' : ''}📎 ${attachedFile.filename}`
      : input;
    const sentText = composeSentText(input, attachedFile);
    const rawInput = input.trim();

    setSessions((prev) =>
      prev.map((s) =>
        s.session_id === sessionId
          ? {
              ...s,
              title: rawInput ? rawInput.slice(0, 60) : (attachedFile ? attachedFile.filename : 'New chat'),
              last_active: new Date().toISOString(),
              isPlaceholder: false,
            }
          : s
      )
    );

    const reqSession = sessionId;
    setMessages((prev) => [...prev, { role: 'user', content: displayText }]);
    setInput('');
    setAttachedFile(null);
    setSessionLoading(reqSession, true);
    setError(null);
    if (textareaRef.current) textareaRef.current.style.height = 'auto';

    const messageId = crypto.randomUUID ? crypto.randomUUID() : `msg-${Date.now()}-${Math.random().toString(16).slice(2)}`;

    setMessages((prev) => [
      ...prev,
      {
        role: 'assistant',
        content: '',
        toolCalls: [],
        pending: [],
        question: null,
        options: [],
        tokenUsage: null,
        traceId: null,
        messageId,
        originalText: sentText,
        isIngesting: !!attachedFile,
      },
    ]);

    try {
      const activeDocId = contextFile?.document_id || pageViewer?.docId || fileId || null;
      const res = await sendAgentChat(sentText, reqSession, false, null, controller.signal, activeDocId, messageId);
      if (sessionIdRef.current !== reqSession) {
        setSessionLoading(reqSession, false);
        loadSessions();
        return;
      }
      const data = res.data;
      setMessages((prev) =>
        prev.map((m) =>
          m.messageId === messageId
            ? {
                ...m,
                status: data.status,
                content: typeof data.answer === 'string' ? data.answer : (data.answer ? JSON.stringify(data.answer) : ''),
                toolCalls: data.tool_calls || [],
                executionTrace: data.execution_trace || [],
                pending: data.pending || [],
                question: data.question || null,
                options: data.options || [],
                tokenUsage: data.token_usage || null,
                traceId: data.trace_id || null,
              }
            : m
        )
      );

      setSessionLoading(reqSession, false);
      loadSessions();
      updatePageViewer(data);
    } catch (err) {
      if (axios.isCancel(err) || err.name === 'CanceledError' || err.name === 'AbortError') {
        if (sessionIdRef.current === reqSession) {
          setSessionLoading(reqSession, false);
          setMessages((prev) =>
            prev.map((m) =>
              m.messageId === messageId
                ? {
                    ...m,
                    content: 'Generation stopped.',
                  }
                : m
            )
          );
        }
        return;
      }
      console.error('Chat error:', err);
      if (sessionIdRef.current !== reqSession) return;
      setSessionLoading(reqSession, false);
      setMessages((prev) =>
        prev.map((m) =>
          m.messageId === messageId
            ? {
                ...m,
                content: `Error: ${err.message || 'Failed to reach the agent. Make sure the backend is running.'}`,
                isError: true,
              }
            : m
        )
      );
      setError(err.message);
    } finally {
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
    }
  };

  // ── Direct (agent-free) ingestion handlers ────────────────────────────────

  /** Polls progress of an active staged ingestion. Re-used on reload. */
  function pollDirectIngest(messageId, documentId, filename) {
    // Keep a local set of active polls to prevent starting duplicate intervals
    if (!window.activeDirectPolls) window.activeDirectPolls = {};
    if (window.activeDirectPolls[messageId]) return;
    
    const poll = setInterval(async () => {
      try {
        const prog = await getProgress(documentId);
        const { status, metrics } = prog.data;

        // Compute an overall % from completed steps
        const steps = Array.isArray(metrics) ? metrics : [];
        const done = steps.filter((s) => s.status === 'done').length;
        const total = steps.length || 1;
        const pct = Math.round((done / total) * 100);
        const runningStep = steps.find((s) => s.status === 'running');
        const currentStep = runningStep ? runningStep.step : (done === total ? 'Complete' : 'Processing…');

        setMessages((prev) =>
          prev.map((m) =>
            m.messageId === messageId
              ? { ...m, progressPct: pct, currentStep, metrics: steps }
              : m
          )
        );

        const lowerStatus = (status || '').toLowerCase();
        if (lowerStatus === 'ready' || lowerStatus === 'done' || lowerStatus === 'failed') {
          clearInterval(poll);
          delete window.activeDirectPolls[messageId];
          setMessages((prev) =>
            prev.map((m) =>
              m.messageId === messageId
                ? { ...m, type: (lowerStatus === 'ready' || lowerStatus === 'done') ? 'ingest_done' : 'ingest_error', progressPct: 100 }
                : m
            )
          );
          // Refresh the ingestion page file list
          loadFiles();
          loadSessions();
        }
      } catch {
        clearInterval(poll);
        delete window.activeDirectPolls[messageId];
        setMessages((prev) =>
          prev.map((m) =>
            m.messageId === messageId ? { ...m, type: 'ingest_error' } : m
          )
        );
      }
    }, 2000);
    
    window.activeDirectPolls[messageId] = poll;
  };

  /** Called when user sends a file with no question text. Shows approval card. */
  const handleDirectIngest = async () => {
    const file = attachedFile; // { file_path, filename }
    const msgId = crypto.randomUUID ? crypto.randomUUID() : `msg-${Date.now()}-${Math.random().toString(16).slice(2)}`;

    // Update session title to the filename
    setSessions((prev) =>
      prev.map((s) =>
        s.session_id === sessionId
          ? { ...s, title: `📎 ${file.filename}`, last_active: new Date().toISOString(), isPlaceholder: false }
          : s
      )
    );

    // Add user bubble + the approval assistant card in one shot locally
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: `📎 ${file.filename}` },
      {
        role: 'assistant',
        content: '',
        type: 'ingest_approval',
        messageId: msgId,
        filename: file.filename,
        stagedPath: file.file_path,
      },
    ]);

    setInput('');
    setAttachedFile(null);
    if (textareaRef.current) textareaRef.current.style.height = 'auto';

    // Persist this initial state (approval card) to the database session history
    try {
      await initDirectIngest(sessionId, {
        filename: file.filename,
        staged_path: file.file_path,
        message_id: msgId
      });
      loadSessions();
    } catch (err) {
      console.error('Failed to persist direct-ingest initial turns:', err);
    }
  };

  /** Called when user clicks "Ingest" on the approval card. */
  const handleIngestApprove = async (messageId, stagedPath, filename) => {
    // Switch to progress state locally immediately
    setMessages((prev) =>
      prev.map((m) =>
        m.messageId === messageId
          ? { ...m, type: 'ingest_progress', documentId: null, progressPct: 0, currentStep: 'Starting…' }
          : m
      )
    );

    try {
      const res = await ingestStagedFile({
        file_path: stagedPath,
        filename,
        session_id: sessionId,
        message_id: messageId,
      });
      const documentId = res.data.document_id || res.data.id;

      // Store documentId so the poll can use it
      setMessages((prev) =>
        prev.map((m) => (m.messageId === messageId ? { ...m, documentId } : m))
      );

      // Start progress polling
      pollDirectIngest(messageId, documentId, filename);
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.messageId === messageId
            ? { ...m, type: 'ingest_error', errorMsg: err.message }
            : m
        )
      );
    }
  };

  /** Called when user clicks "Cancel" on the approval card. */
  const handleIngestCancel = async (messageId) => {
    // Retrieve the filename from message
    const msg = messages.find((m) => m.messageId === messageId);
    const filename = msg ? msg.filename : '';

    // Update state locally
    setMessages((prev) =>
      prev.map((m) =>
        m.messageId === messageId ? { ...m, type: 'ingest_cancelled' } : m
      )
    );

    // Persist cancellation to DB
    try {
      await cancelDirectIngest(sessionId, { message_id: messageId, filename });
    } catch (err) {
      console.error('Failed to persist direct ingest cancel state:', err);
    }
  };

  const handleRetryResponse = async () => {
    if (messages.length === 0) return;
    const lastUserMsg = messages[messages.length - 1];
    if (lastUserMsg.role !== 'user') return;

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    const reqSession = sessionId;
    setSessionLoading(reqSession, true);
    setError(null);

    const messageId = crypto.randomUUID ? crypto.randomUUID() : `msg-${Date.now()}-${Math.random().toString(16).slice(2)}`;

    // Add empty placeholder assistant message
    setMessages((prev) => [
      ...prev,
      {
        role: 'assistant',
        content: '',
        toolCalls: [],
        pending: [],
        question: null,
        options: [],
        tokenUsage: null,
        traceId: null,
        messageId,
        originalText: lastUserMsg.content,
      },
    ]);

    try {
      const activeDocId = contextFile?.document_id || pageViewer?.docId || fileId || null;
      const res = await sendAgentChat(lastUserMsg.content, reqSession, false, null, controller.signal, activeDocId, messageId);
      if (sessionIdRef.current !== reqSession) {
        setSessionLoading(reqSession, false);
        loadSessions();
        return;
      }
      const data = res.data;
      setMessages((prev) =>
        prev.map((m) =>
          m.messageId === messageId
            ? {
                ...m,
                status: data.status,
                content: typeof data.answer === 'string' ? data.answer : (data.answer ? JSON.stringify(data.answer) : ''),
                toolCalls: data.tool_calls || [],
                pending: data.pending || [],
                question: data.question || null,
                options: data.options || [],
                tokenUsage: data.token_usage || null,
                traceId: data.trace_id || null,
              }
            : m
        )
      );
      setSessionLoading(reqSession, false);
      loadSessions();
      updatePageViewer(data);
    } catch (err) {
      if (axios.isCancel(err) || err.name === 'CanceledError' || err.name === 'AbortError') {
        if (sessionIdRef.current === reqSession) {
          setSessionLoading(reqSession, false);
          setMessages((prev) =>
            prev.map((m) =>
              m.messageId === messageId
                ? {
                    ...m,
                    content: 'Generation stopped.',
                  }
                : m
            )
          );
        }
        return;
      }
      console.error('Retry error:', err);
      if (sessionIdRef.current !== reqSession) return;
      setSessionLoading(reqSession, false);
      setMessages((prev) =>
        prev.map((m) =>
          m.messageId === messageId
            ? {
                ...m,
                content: `Error: ${err.message || 'Failed to reach the agent. Make sure the backend is running.'}`,
                isError: true,
              }
            : m
        )
      );
      setError(err.message);
    } finally {
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
    }
  };

  const handleApproval = async (msgIdx, approved) => {
    const msg = messages[msgIdx];
    if (!approved) {
      setMessages((prev) => prev.map((m, i) => (i === msgIdx ? { ...m, status: 'declined' } : m)));
      return;
    }

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    const reqSession = sessionId;
    setSessionLoading(reqSession, true);

    // Re-use the existing messageId for LOCAL React state tracking only (which
    // bubble to update in place). This network call is functionally a NEW
    // request — approved_writes flips true and carries different args — so it
    // needs its own fresh idempotency key.
    const messageId = msg.messageId || (crypto.randomUUID ? crypto.randomUUID() : `msg-${Date.now()}-${Math.random().toString(16).slice(2)}`);
    const requestId = crypto.randomUUID ? crypto.randomUUID() : `req-${Date.now()}-${Math.random().toString(16).slice(2)}`;

    // Reset this message state to prepare for response
    setMessages((prev) =>
      prev.map((m, i) =>
        i === msgIdx
          ? {
              ...m,
              content: '',
              toolCalls: [],
              pending: [],
              question: null,
              options: [],
              tokenUsage: null,
              traceId: null,
              messageId,
            }
          : m
      )
    );

    try {
      const activeDocId = contextFile?.document_id || pageViewer?.docId || fileId || null;
      const res = await sendAgentChat(msg.originalText, reqSession, true, msg.pending || [], controller.signal, activeDocId, requestId);
      if (sessionIdRef.current !== reqSession) {
        setSessionLoading(reqSession, false);
        loadSessions();
        return;
      }
      const data = res.data;
      setMessages((prev) =>
        prev.map((m) =>
          m.messageId === messageId
            ? {
                ...m,
                status: data.status,
                content: typeof data.answer === 'string' ? data.answer : (data.answer ? JSON.stringify(data.answer) : ''),
                toolCalls: data.tool_calls || [],
                pending: data.pending || [],
                question: data.question || null,
                options: data.options || [],
                tokenUsage: data.token_usage || null,
                traceId: data.trace_id || null,
              }
            : m
        )
      );
      setSessionLoading(reqSession, false);
      loadSessions();
      updatePageViewer(data);
    } catch (err) {
      if (axios.isCancel(err) || err.name === 'CanceledError' || err.name === 'AbortError') {
        if (sessionIdRef.current === reqSession) {
          setSessionLoading(reqSession, false);
          setMessages((prev) =>
            prev.map((m) =>
              m.messageId === messageId
                ? {
                    ...m,
                    content: 'Generation stopped.',
                  }
                : m
            )
          );
        }
        return;
      }
      console.error('Approval error:', err);
      if (sessionIdRef.current !== reqSession) return;
      setSessionLoading(reqSession, false);
      setMessages((prev) =>
        prev.map((m) =>
          m.messageId === messageId
            ? {
                ...m,
                content: `Error: ${err.message || 'Failed to execute approved actions.'}`,
                isError: true,
              }
            : m
        )
      );
      setError(err.message);
    } finally {
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
    }
  };

  const handleTextareaChange = (e) => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  };

  return (
    <div className="flex h-screen pastel-mesh-bg text-[#1d1d1d] relative font-sans">
      <style>{`
        @keyframes slideInSide {
          from {
            opacity: 0;
            transform: translateX(-10px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }
      `}</style>
      {/* Slacc Sidebar */}
      <div
        className={`flex-shrink-0 flex flex-col bg-transparent border-r border-[#e6e6e6]/60 backdrop-blur-sm transition-all duration-500 ease-in-out ${sidebarOpen ? 'w-72' : 'w-0 overflow-hidden border-r-0'
          }`}
      >
        {/* Sidebar Header */}
        <div className="p-4 flex items-center justify-between h-16 border-b border-[#e6e6e6]/60">
          <div className="flex items-center gap-2">
            <span className="font-extrabold tracking-widest text-xs text-[#4a154b] uppercase">AI-ACCELERATOR</span>
          </div>
          <Tooltip content="Close sidebar">
            <button
              onClick={() => setSidebarOpen(false)}
              className="p-2 text-[#696969] hover:text-[#4a154b] hover:bg-white/60 rounded-full transition-all"
            >
              <SidebarToggleIcon className="h-4 w-4" />
            </button>
          </Tooltip>
        </div>

        <div className="px-3 py-2 flex justify-center">
          <button
            onClick={handleNewChat}
            className="btn-primary-pill w-[85%] !py-2 !px-4 !text-xs inline-flex items-center justify-center gap-1.5 shadow-sm"
          >
            <PlusIcon className="h-3.5 w-3.5" />
            New Chat
          </button>
        </div>

        {/* History Scroll Container with Top/Bottom Smooth Fade Masks */}
        <div className="relative flex-1 min-h-0 flex flex-col">
          {/* Top fade mask */}
          <div className="absolute top-0 left-0 right-0 h-4 bg-gradient-to-b from-white/30 to-transparent z-10 pointer-events-none" />

          {/* Scrollable session list */}
          <div className="flex-1 min-h-0 overflow-y-auto no-scrollbar px-3 pt-2 pb-24 space-y-1">
            {sessions.length === 0 && (
              <p className="text-xs text-[#696969] px-3 py-2 font-medium">No conversations yet</p>
            )}
            {[...sessions]
              .sort((a, b) => {
                if (a.pinned && !b.pinned) return -1;
                if (!a.pinned && b.pinned) return 1;
                return new Date(b.last_active) - new Date(a.last_active);
              })
              .map((s, index) => (
                <div
                  key={s.session_id}
                  onClick={() => { setMenuOpen(null); handleSelectSession(s.session_id); }}
                  className={`group flex items-center justify-between gap-2 px-3.5 py-3 rounded-2xl cursor-pointer text-xs font-bold transition-all ${s.session_id === sessionId
                    ? 'bg-white/90 text-[#4a154b] border border-[#e6e6e6] shadow-sm backdrop-blur-sm'
                    : 'text-[#1d1d1d] hover:bg-white/50 hover:text-[#4a154b]'
                    } ${menuOpen === s.session_id ? 'relative z-50' : ''}`}
                  style={{
                    animation: 'slideInSide 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards',
                    animationDelay: `${index * 30}ms`,
                    opacity: 0,
                  }}
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
                        className="w-full bg-white text-[#1d1d1d] text-xs px-2.5 py-1 rounded-md border border-[#4a154b] outline-none"
                        autoFocus
                        onClick={(e) => e.stopPropagation()}
                      />
                    ) : (
                      <>
                        <p className="truncate flex items-center gap-1.5 font-bold">
                          {s.pinned && <PinIcon className="h-3.5 w-3.5 flex-shrink-0 text-[#4a154b]" />}
                          {s.title}
                        </p>
                        <p className="text-[10px] text-[#696969] mt-0.5">{relativeTime(s.last_active)}</p>
                      </>
                    )}
                  </div>
                  <div className="relative flex-shrink-0" data-menu>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setMenuOpen(menuOpen === s.session_id ? null : s.session_id);
                      }}
                      className={`rounded p-1 transition-colors ${s.session_id === sessionId ? 'text-[#4a154b] hover:bg-[#4a154b]/10' : 'text-[#696969] hover:text-[#4a154b]'}`}
                      title="More"
                    >
                      <EllipsisVerticalIcon className="h-4 w-4" />
                    </button>
                    {menuOpen === s.session_id && (
                      <div className="absolute right-0 top-7 z-50 w-44 bg-white border border-[#e6e6e6] rounded-2xl shadow-xl py-1.5 text-xs font-bold text-[#1d1d1d]" data-menu>
                        <button
                          onClick={(e) => handleRenameStart(s.session_id, s.title, e)}
                          className="w-full flex items-center gap-2 px-4 py-2 hover:bg-[#f9f0ff] hover:text-[#4a154b] text-left"
                        >
                          <PencilIcon className="h-4 w-4 text-[#696969]" />
                          Rename
                        </button>
                        <button
                          onClick={(e) => handleTogglePin(s.session_id, s.pinned, e)}
                          className="w-full flex items-center gap-2 px-4 py-2 hover:bg-[#f9f0ff] hover:text-[#4a154b] text-left"
                        >
                          <PinIcon className="h-4 w-4 text-[#696969]" />
                          {s.pinned ? 'Unpin' : 'Pin'}
                        </button>
                        <div className="border-t border-[#e6e6e6] my-1" />
                        <button
                          onClick={(e) => handleDeleteSession(s.session_id, e)}
                          className="w-full flex items-center gap-2 px-4 py-2 text-[#cc4117] hover:bg-[#cc4117]/10 text-left"
                        >
                          <TrashIcon className="h-4 w-4 text-[#cc4117]" />
                          Delete
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))}
          </div>

          {/* Bottom fade mask */}
          <div className="absolute bottom-0 left-0 right-0 h-10 bg-gradient-to-t from-white/30 to-transparent z-10 pointer-events-none" />
        </div>
      </div>

      {/* Main chat column */}
      <div className="flex-1 flex flex-col min-w-0 relative">
        {/* Top Floating Actions (Absolute Overlay — No layout height gap) */}
        <div className="absolute top-4 left-6 right-6 flex items-center justify-between z-30 pointer-events-none">
          <div className="flex items-center gap-3 pointer-events-auto">
            {!sidebarOpen && (
              <div className="flex items-center gap-1.5 bg-[#f9f0ff] border border-[#e6e6e6] rounded-full px-2.5 py-1 shadow-sm">
                <Tooltip content="Open sidebar">
                  <button
                    onClick={() => setSidebarOpen(true)}
                    className="p-1 text-[#4a154b] hover:bg-white rounded-full transition-all"
                  >
                    <SidebarToggleIcon className="h-4 w-4" />
                  </button>
                </Tooltip>
                <div className="w-[1px] h-3.5 bg-[#e6e6e6] mx-1" />
                <Tooltip content="New chat">
                  <button
                    onClick={handleNewChat}
                    className="p-1 text-[#4a154b] hover:bg-white rounded-full transition-all"
                  >
                    <NewChatIcon className="h-4 w-4" />
                  </button>
                </Tooltip>
              </div>
            )}
            {contextFile && (
              <span className="text-xs font-bold text-[#4a154b] bg-[#f9f0ff] px-3 py-1 rounded-full border border-[#e6e6e6] flex items-center gap-1.5 ml-2 shadow-sm">
                <DocumentIcon className="h-3.5 w-3.5 text-[#4a154b]" />
                Context: {contextFile.filename}
              </span>
            )}
          </div>

          {!pageViewer && (
            <div className="flex items-center gap-2.5 pointer-events-auto">
              <Link to="/ingest" className="btn-secondary-pill !py-1.5 !px-4 text-xs shadow-sm">
                Ingest Docs
              </Link>
              <Link
                to="/settings"
                title="System Configuration"
                className="p-2.5 rounded-full bg-[#f9f0ff] hover:bg-[#f3e2ff] border border-[#e6e6e6] text-[#4a154b] transition-all shadow-sm flex items-center justify-center"
              >
                <Cog6ToothIcon className="h-5 w-5" />
              </Link>
            </div>
          )}
        </div>

        <div
          ref={chatContainerRef}
          onScroll={checkChatScroll}
          className="flex-1 overflow-y-auto relative"
        >
          <div className="max-w-4xl mx-auto px-6 py-8">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center min-h-[60vh] text-center max-w-2xl mx-auto py-10">
                {/* Brand Badge */}
                <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[#f9f0ff] border border-[#e6e6e6] text-[#4a154b] text-xs font-extrabold uppercase tracking-wider mb-6 shadow-sm">
                  <SparklesIcon className="h-4 w-4 text-[#4a154b]" />
                  AI Accelerator Intelligence
                </div>

                {/* Hero Title */}
                <h2 className="text-3xl md:text-5xl font-extrabold text-[#4a154b] display-hero tracking-tight mb-4">
                  How can I assist your engineering work today?
                </h2>
                <p className="text-base text-[#696969] leading-relaxed mb-8 max-w-xl">
                  Ask questions across technical specifications, CAD drawings, spreadsheets, and PDFs with 90% traceable RAG citations.
                </p>

                {/* Supported Format Pills */}
                <div className="flex flex-wrap items-center justify-center gap-2 text-xs text-[#696969]">
                  <span className="font-bold text-[#4a154b]">Supports:</span>
                  <span className="bg-white px-3.5 py-1.5 rounded-full border border-[#e6e6e6] shadow-sm font-semibold">📄 PDF</span>
                  <span className="bg-white px-3.5 py-1.5 rounded-full border border-[#e6e6e6] shadow-sm font-semibold">📊 Excel (.xlsx)</span>
                  <span className="bg-white px-3.5 py-1.5 rounded-full border border-[#e6e6e6] shadow-sm font-semibold">📐 CAD (.dwg / .dxf)</span>
                  <span className="bg-white px-3.5 py-1.5 rounded-full border border-[#e6e6e6] shadow-sm font-semibold">📝 Word (.docx)</span>
                  <span className="bg-white px-3.5 py-1.5 rounded-full border border-[#e6e6e6] shadow-sm font-semibold">🖼️ Images (.png / .jpg)</span>
                </div>
              </div>
            )}

            {messages.map((msg, idx) => (
              <MessageRow key={idx} msg={msg} onApprove={() => handleApproval(idx, true)}
                onDecline={() => handleApproval(idx, false)}
                onClarify={(opt) => handleClarify(opt, idx)} loading={loading}
                onViewPages={(pages) => setPageViewer({ pages, activeIdx: 0 })}
                onIngestApprove={handleIngestApprove}
                onIngestCancel={handleIngestCancel}
                allFiles={allFiles}
                currency={currency}
                setCurrency={setCurrency} />
            ))}

            {messages.length > 0 && messages[messages.length - 1].role === 'user' && (
              <div className="py-4 flex justify-start animate-fade-in">
                <div className="max-w-full w-full">
                  <div className="flex gap-3 items-center bg-white border border-[#e6e6e6] rounded-2xl p-4 shadow-sm">
                    <SparklesIcon className="h-5 w-5 text-[#4a154b] flex-shrink-0 animate-pulse" />
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center text-[#4a154b] text-sm font-bold">
                        <span>Generating Intelligence Response<span className="animate-dots"></span></span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {error && (
          <div className="max-w-4xl mx-auto w-full px-6 flex-shrink-0">
            <div className="mb-3 p-4 bg-[#cc4117]/10 border border-[#cc4117]/30 rounded-2xl text-[#cc4117] text-xs font-bold flex items-start gap-2.5 shadow-sm">
              <ExclamationCircleIcon className="h-4 w-4 flex-shrink-0 mt-0.5" />
              {error}
            </div>
          </div>
        )}

        {/* Input Bar — Floating Pill */}
        <div className="bg-transparent pb-6 px-4 pt-2 flex-shrink-0">
          <div className="max-w-4xl mx-auto relative">
            {(showChatScrollDown && !pageViewer) && (
              <button
                onClick={handleChatScrollToBottom}
                className="absolute bottom-full mb-4 right-4 p-2.5 rounded-full bg-[#4a154b] text-white shadow-xl hover:bg-[#611f69] active:scale-95 transition-all flex items-center justify-center border border-[#592466] z-40"
                title="Scroll to bottom"
              >
                <ChevronDownIcon className="h-4 w-4 stroke-[2.5]" />
              </button>
            )}
            {attachedFile && (
              <div className="mb-2.5 inline-flex items-center gap-2 bg-[#f9f0ff] border border-[#e6e6e6] rounded-full px-4 py-1.5 text-xs font-bold text-[#4a154b] shadow-sm">
                <DocumentIcon className="h-4 w-4 text-[#4a154b]" />
                {attachedFile.filename}
                <button onClick={() => setAttachedFile(null)} title="Remove attachment" className="ml-1 text-[#696969] hover:text-[#cc4117]">
                  <XMarkIcon className="h-4 w-4" />
                </button>
              </div>
            )}
            <div className="flex items-center gap-2 bg-white border-2 border-[#e6e6e6] focus-within:border-[#4a154b] rounded-full px-4 py-2 shadow-lg transition-all">
              <input type="file" ref={fileInputRef} className="hidden" onChange={handleAttach} />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={attaching || loading}
                title="Attach a file to ingest"
                className="p-2 text-[#4a154b] hover:bg-[#f9f0ff] rounded-full disabled:opacity-50 flex-shrink-0 transition-all"
              >
                {attaching ? (
                  <ArrowPathIcon className="h-5 w-5 animate-spin text-[#4a154b]" />
                ) : (
                  <PaperClipIcon className="h-5 w-5" />
                )}
              </button>
              <textarea
                ref={textareaRef}
                rows={1}
                className="flex-1 bg-transparent resize-none outline-none text-[#0f172a] font-medium text-sm placeholder-[#64748b] py-2 max-h-40"
                value={input}
                onChange={handleTextareaChange}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder="Ask AI Accelerator Engine… (Shift+Enter for new line)"
                disabled={loading}
              />
              {loading ? (
                <button
                  onClick={handleStop}
                  title="Stop generating"
                  className="btn-primary-pill !p-3"
                >
                  <div className="w-3.5 h-3.5 bg-white rounded-sm" />
                </button>
              ) : (
                <button
                  onClick={handleSend}
                  disabled={!input.trim() && !attachedFile}
                  className={`btn-primary-pill !p-3 flex-shrink-0 ${(!input.trim() && !attachedFile)
                    ? 'opacity-40 cursor-not-allowed'
                    : ''
                    }`}
                >
                  <PaperAirplaneIcon className="h-4 w-4" />
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Col 3: Page Viewer Panel */}
      {pageViewer && (
        <PageViewerPanel
          viewer={pageViewer}
          onClose={() => setPageViewer(null)}
          onPageChange={(idx) => setPageViewer((v) => ({ ...v, activeIdx: idx }))}
          sidebarOpen={sidebarOpen}
        />
      )}
    </div>
  );
};

// ── PageViewerPanel ───────────────────────────────────────────────────────────
// Right-side panel for a citation. Dispatches on fileType (set by
// fileTypeFromName in buildViewableSources): pdf/ppt share a paginated page/
// slide-image canvas with nav + zoom; docx renders the whole document as HTML
// and scroll/highlights the cited snippet at view time (no fixed page
// numbers); excel is parsed client-side from the raw file and shows the
// cited sheet; image is a single whole-file image, no pagination.
const PageViewerPanel = ({ viewer, onClose, onPageChange, sidebarOpen }) => {
  const { pages, activeIdx } = viewer;
  const active = pages[activeIdx];
  const fileType = active?.fileType;
  const isPaginated = fileType === 'pdf' || fileType === 'ppt';

  const normalizeSheetName = (s) => (s || '').replace(/[^a-zA-Z0-9]/g, '').toLowerCase();
  const cosmeticCleanSheetLabel = (s) => (s || '').replace(/\?{2,}/g, '…').trim();

  const getBadgeLabel = (p) => {
    if (p.fileType === 'pdf') return `P.${p.page}`;
    if (p.fileType === 'ppt') return `Slide ${p.page}`;
    if (p.fileType === 'excel') {
      if (!p.sheet) return 'Excel';
      if (workbook && p.document_id === active?.document_id) {
        if (p.sheet_index != null && p.sheet_index >= 0 && p.sheet_index < workbook.SheetNames.length) {
          return `Sheet: ${workbook.SheetNames[p.sheet_index]}`;
        }
        if (workbook.SheetNames.includes(p.sheet)) return `Sheet: ${p.sheet}`;
        const wantedNorm = normalizeSheetName(p.sheet);
        const candidates = workbook.SheetNames.filter((s) => normalizeSheetName(s) === wantedNorm);
        if (candidates.length === 1) return `Sheet: ${candidates[0]}`;
      }
      return `Sheet: ${cosmeticCleanSheetLabel(p.sheet)}`;
    }
    if (p.fileType === 'docx') return 'Word';
    if (p.fileType === 'image') return 'Image';
    return 'Doc';
  };

  const handleSelectSource = (idx) => {
    onPageChange(idx);
    const target = pages[idx];
    if (target.fileType === 'pdf' || target.fileType === 'ppt') {
      setCurrentPage(target.page || 1);
      setScale(1);
    }
  };

  const [currentPage, setCurrentPage] = useState(active?.page || 1);
  const [totalPages, setTotalPages] = useState(null);
  const [imgLoaded, setImgLoaded] = useState(false);
  const [imgError, setImgError] = useState(false);
  const [scale, setScale] = useState(1);
  const containerRef = useRef(null);

  const [docxHtml, setDocxHtml] = useState(null);
  const [docxLoading, setDocxLoading] = useState(false);
  const [docxError, setDocxError] = useState(false);
  const docxContainerRef = useRef(null);
  const lastHighlightRef = useRef(null);

  const [workbook, setWorkbook] = useState(null);
  const [excelLoading, setExcelLoading] = useState(false);
  const [excelError, setExcelError] = useState(false);
  const [activeSheet, setActiveSheet] = useState(null);
  const [citedSheet, setCitedSheet] = useState(null);

  // PDF/PPT: when active page changes from parent, sync currentPage and reset scale
  useEffect(() => {
    if (isPaginated && active?.page) {
      setCurrentPage(active.page);
      setScale(1);
    }
  }, [fileType, activeIdx, active?.page]);

  // PDF/PPT: fetch total page/slide count for active document.
  // pdf-info dispatches on the doc's own file_type server-side, so the same
  // endpoint returns a slide count for ppt and a page count for pdf.
  useEffect(() => {
    if (!isPaginated || !active?.document_id) return;
    setTotalPages(null);
    fetch(`${API_BASE_URL}/files/${active.document_id}/pdf-info`)
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((data) => {
        setTotalPages(data.total_pages);
      })
      .catch((err) => {
        console.error("Failed to load page info:", err);
      });
  }, [fileType, active?.document_id]);

  // PDF/PPT: reset image load status when current page changes
  useEffect(() => {
    if (!isPaginated) return;
    setImgLoaded(false);
    setImgError(false);
  }, [fileType, currentPage, active?.document_id]);

  // Docx: fetch the rendered HTML whenever the active document changes
  useEffect(() => {
    if (fileType !== 'docx' || !active?.document_id) return;
    setDocxHtml(null);
    setDocxError(false);
    setDocxLoading(true);
    lastHighlightRef.current = null;
    fetch(`${API_BASE_URL}/files/${active.document_id}/docx-html`)
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then((data) => setDocxHtml(data.html))
      .catch((err) => {
        console.error("Failed to load docx HTML:", err);
        setDocxError(true);
      })
      .finally(() => setDocxLoading(false));
  }, [fileType, active?.document_id]);

  // Docx: once the HTML is in the DOM, scroll to + highlight this citation's
  // snippet. Word has no fixed page numbers, so this is a view-time text
  // match against the rendered HTML rather than a jump to a persisted
  // page/paragraph number (see backend file_docx_html docstring for why).
  useEffect(() => {
    if (fileType !== 'docx' || !docxHtml || !docxContainerRef.current) return;
    const container = docxContainerRef.current;

    if (lastHighlightRef.current) {
      lastHighlightRef.current.style.backgroundColor = '';
      lastHighlightRef.current.style.transition = '';
      lastHighlightRef.current = null;
    }

    const snippet = (active?.snippet || '').trim();
    if (!snippet) return;

    // Word's smart quotes/dashes survive into the docx run text and mammoth
    // renders them verbatim, but the chunk text used for the snippet may have
    // been normalized to plain ASCII somewhere upstream (or vice versa). Fold
    // both sides to the same canonical form so a single curly quote doesn't
    // silently sink an otherwise-good match.
    const normalizeForMatch = (s) =>
      (s || '')
        .replace(/[\u2018\u2019\u201A\u201B]/g, "'")
        .replace(/[\u201C\u201D\u201E\u201F]/g, '"')
        .replace(/[\u2013\u2014]/g, '-')
        .replace(/\u00A0/g, ' ')
        .replace(/\s+/g, ' ')
        .toLowerCase();

    // Table chunks store text as markdown (df.to_markdown(): pipes and a
    // "| --- | --- |" separator row — see schemas.py), but mammoth renders
    // the same table as a real <table> with plain cell text. Strip the
    // markdown scaffolding so a table citation's needle reduces to the same
    // words the rendered cells contain, instead of characters that can never
    // appear in the HTML.
    const stripTableMarkdown = (s) =>
      (s || '')
        .split('\n')
        .filter((line) => !/^[\s|:-]+$/.test(line))
        .join(' ')
        .replace(/\|/g, ' ');

    const cleanedSnippet = normalizeForMatch(stripTableMarkdown(snippet));
    if (!cleanedSnippet) return;

    // Match against the whole document's text, not node-by-node. A needle can
    // straddle multiple text nodes — e.g. a heading mammoth renders as its own
    // <h2> immediately followed by the paragraph chunk_tool merged it with, or
    // a sentence split across a bold/italic run or a hyperlink. Concatenate
    // every text node into one normalized string (in DOM order), track which
    // node each slice of that string came from, search the combined string,
    // then map the match position back to the node(s) it falls in.
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    const nodeRanges = []; // { node, start, end } offsets into fullText
    let fullText = '';
    let node;
    while ((node = walker.nextNode())) {
      const normalized = normalizeForMatch(node.textContent);
      if (!normalized) continue;
      // Keep a boundary space between nodes so words from adjacent elements
      // (end of a heading, start of the next paragraph) don't fuse together,
      // while still letting the needle span across the boundary.
      if (fullText && !fullText.endsWith(' ') && !normalized.startsWith(' ')) {
        fullText += ' ';
      }
      const start = fullText.length;
      fullText += normalized;
      nodeRanges.push({ node, start, end: fullText.length });
    }

    // Try progressively shorter windows of the cleaned snippet. A long window
    // is more specific (less chance of matching the wrong passage) but is
    // also more likely to snag on some remaining stray character; shrinking
    // on failure trades specificity for resilience instead of giving up
    // after one attempt.
    let matchStart = -1;
    let needleLen = 0;
    for (const len of [80, 50, 30, 18]) {
      const candidate = cleanedSnippet.slice(0, len).trim();
      if (!candidate) continue;
      const idx = fullText.indexOf(candidate);
      if (idx !== -1) {
        matchStart = idx;
        needleLen = candidate.length;
        break;
      }
    }

    if (matchStart !== -1) {
      const matchEnd = matchStart + needleLen;
      const hit = nodeRanges.find((r) => r.start < matchEnd && r.end > matchStart);
      const el = hit?.node.parentElement;
      if (el) {
        el.scrollIntoView({ block: 'center', behavior: 'smooth' });
        el.style.transition = 'background-color 0.3s ease';
        el.style.backgroundColor = 'rgba(59, 130, 246, 0.25)';
        lastHighlightRef.current = el;
      }
    }
    // no match found at any window size — fall back to showing the doc from
    // the top, no highlight
  }, [fileType, docxHtml, activeIdx, active?.snippet]);

  // Excel: fetch the raw workbook and parse it client-side (SheetJS) — there's
  // no server-side HTML conversion for spreadsheets, just /raw bytes.
  useEffect(() => {
    if (fileType !== 'excel' || !active?.document_id) return;
    setWorkbook(null);
    setExcelError(false);
    setExcelLoading(true);
    fetch(`${API_BASE_URL}/files/${active.document_id}/raw`)
      .then((res) => {
        if (!res.ok) throw new Error();
        return res.arrayBuffer();
      })
      .then((buf) => setWorkbook(XLSX.read(buf, { type: 'array' })))
      .catch((err) => {
        console.error("Failed to load Excel file:", err);
        setExcelError(true);
      })
      .finally(() => setExcelLoading(false));
  }, [fileType, active?.document_id]);

  // Excel: pick the sheet this citation points at.
  useEffect(() => {
    if (fileType !== 'excel') {
      setActiveSheet(null);
      setCitedSheet(null);
      return;
    }
    if (!workbook) return;
    const wanted = active?.sheet;
    const wantedIdx = active?.sheet_index;
    let match = null;

    if (wantedIdx != null && wantedIdx >= 0 && wantedIdx < workbook.SheetNames.length) {
      match = workbook.SheetNames[wantedIdx];
    } else if (wanted) {
      if (workbook.SheetNames.includes(wanted)) {
        match = wanted;
      } else {
        const normalize = (s) => (s || '').replace(/[^a-zA-Z0-9]/g, '').toLowerCase();
        const wantedNorm = normalize(wanted);
        if (wantedNorm) {
          const candidates = workbook.SheetNames.filter((s) => normalize(s) === wantedNorm);
          if (candidates.length === 1) {
            match = candidates[0];
          } else if (candidates.length > 1) {
            console.warn(
              `Excel sheet match ambiguous for citation "${wanted}" — ` +
              `${candidates.length} sheets share the normalized name. Defaulting to first sheet.`,
              candidates
            );
          }
        }
      }
    }

    const finalMatch = match || workbook.SheetNames[0];
    setActiveSheet(finalMatch);
    setCitedSheet(finalMatch);
  }, [fileType, workbook, activeIdx, active?.sheet, active?.sheet_index]);

  // Trap Ctrl + MouseWheel / trackpad pinch zooms on the pdf/ppt/image canvas
  // to zoom the document internally and prevent the browser from zooming the
  // dashboard. Docx/excel scroll/zoom is native browser behavior. Re-runs
  // whenever the visible fileType changes, since the canvas div (and its ref)
  // only exists in the DOM for paginated/image citations.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleWheel = (e) => {
      if (e.ctrlKey) {
        e.preventDefault();
        const factor = 0.08;
        setScale((prev) => {
          const newScale = e.deltaY < 0 ? prev + factor : prev - factor;
          return Math.max(0.4, Math.min(newScale, 3.5)); // bound scale
        });
      }
    };

    container.addEventListener('wheel', handleWheel, { passive: false });
    return () => {
      container.removeEventListener('wheel', handleWheel);
    };
  }, [fileType, active?.document_id]);

  const handlePageChange = (page) => {
    // PDF/PPT only — jump by page/slide number and sync activeIdx if cited.
    setCurrentPage(page);
    setScale(1);
    const idx = pages.findIndex((p) => p.page === page && p.document_id === active?.document_id);
    if (idx !== -1) {
      onPageChange(idx);
    }
  };

  const handleToggleZoom = () => {
    setScale((prev) => (prev > 1.1 ? 1 : 1.8));
  };

  const parsedPage = parseInt(currentPage, 10);
  const imageUrl = (isPaginated && active?.document_id && !isNaN(parsedPage))
    ? `${API_BASE_URL}/files/${active.document_id}/pages/${parsedPage}/image`
    : (fileType === 'image' && active?.document_id)
      ? `${API_BASE_URL}/files/${active.document_id}/original`
      : null;

  const maxScore = Math.max(...pages.map((p) => p.score ?? 0), 0.001);
  const multiPage = pages.length > 1;

  const sheetRows = (fileType === 'excel' && workbook && activeSheet)
    ? XLSX.utils.sheet_to_json(workbook.Sheets[activeSheet], { header: 1, defval: '' })
    : null;

  return (
    <div className={`flex-shrink-0 flex flex-col bg-white border-l border-[#e6e6e6] transition-all duration-500 ease-in-out overflow-hidden shadow-2xl ${sidebarOpen
      ? 'w-[40%] min-w-[40%] max-w-[40%]'
      : 'w-[50%] min-w-[50%] max-w-[50%]'
      }`}>
      {/* Header */}
      <div className="h-16 flex items-center justify-between px-5 border-b border-[#e6e6e6] bg-white flex-shrink-0 gap-4">
        <span className="text-xs font-bold text-[#4a154b] truncate flex-1" title={active?.filename}>
          📄 {active?.filename}
        </span>

        {/* Navigation — pdf/ppt only */}
        {isPaginated && (
          <div className="flex items-center gap-2 select-none flex-shrink-0">
            <button
              onClick={() => handlePageChange(Math.max(1, currentPage - 1))}
              disabled={currentPage <= 1}
              className="p-1.5 hover:bg-[#f9f0ff] text-[#4a154b] rounded-full disabled:opacity-30 transition-all"
              title="Previous Page"
            >
              <ChevronLeftIcon className="h-4 w-4 stroke-[2.5]" />
            </button>

            <span className="text-xs font-bold text-[#4a154b] bg-[#f9f0ff] border border-[#e6e6e6] px-3 py-1 rounded-full">
              {currentPage} / {totalPages || '...'}
            </span>

            <button
              onClick={() => handlePageChange(Math.min(totalPages || 1, currentPage + 1))}
              disabled={currentPage >= (totalPages || 1)}
              className="p-1.5 hover:bg-[#f9f0ff] text-[#4a154b] rounded-full disabled:opacity-30 transition-all"
              title="Next Page"
            >
              <ChevronRightIcon className="h-4 w-4 stroke-[2.5]" />
            </button>
          </div>
        )}

        {/* Zoom Controls */}
        {(isPaginated || fileType === 'image') && (
          <div className="flex items-center gap-1 bg-[#f9f0ff] border border-[#e6e6e6] rounded-full p-1 select-none flex-shrink-0">
            <button
              onClick={() => setScale((prev) => Math.max(0.4, prev - 0.2))}
              className="px-2 py-0.5 hover:bg-white text-[#4a154b] rounded-full transition-all text-xs font-bold"
              title="Zoom Out"
            >
              －
            </button>

            <button
              onClick={() => setScale(1)}
              className="text-[10px] text-[#4a154b] font-bold w-[32px] text-center select-none hover:bg-white rounded-full py-0.5 transition-all"
              title="Reset to 100%"
            >
              {Math.round(scale * 100)}%
            </button>

            <button
              onClick={() => setScale((prev) => Math.min(3.5, prev + 0.2))}
              className="px-2 py-0.5 hover:bg-white text-[#4a154b] rounded-full transition-all text-xs font-bold"
              title="Zoom In"
            >
              ＋
            </button>

            <button
              onClick={handleToggleZoom}
              className="ml-1 text-[10px] font-bold text-[#4a154b] bg-white hover:bg-[#4a154b] hover:text-white px-2 py-0.5 rounded-full transition-all border border-[#e6e6e6]"
              title="Reset Zoom to 100%"
            >
              Reset
            </button>
          </div>
        )}

        {/* Close Button */}
        <button
          onClick={onClose}
          className="p-2 hover:bg-[#f9f0ff] text-[#4a154b] rounded-full transition-all flex-shrink-0"
          title="Close viewer"
        >
          <XMarkIcon className="h-4 w-4 stroke-[2.5]" />
        </button>
      </div>

      {/* Document/Page Badges */}
      {multiPage && (
        <div className="px-4 py-2.5 bg-[#f4ede4]/60 border-b border-[#e6e6e6] flex flex-wrap gap-2 select-none">
          {pages.map((p, i) => {
            const confidence = maxScore > 0 ? ((p.score ?? 0) / maxScore) : 0;
            const isActive = i === activeIdx;
            return (
              <button
                key={i}
                onClick={() => handleSelectSource(i)}
                className={`flex flex-col items-center rounded-full px-3.5 py-1 text-xs font-bold transition-all border ${isActive
                  ? 'bg-[#4a154b] border-[#4a154b] text-white shadow-sm'
                  : 'bg-white border-[#e6e6e6] text-[#1d1d1d] hover:bg-[#f9f0ff]'
                  }`}
              >
                <span>{getBadgeLabel(p)}</span>
              </button>
            );
          })}
        </div>
      )}

      {/* Viewer Canvas */}
      {fileType === 'docx' ? (
        <div ref={docxContainerRef} className="flex-1 overflow-auto bg-[#f4ede4]/40 p-6">
          {docxLoading && (
            <div className="flex flex-col items-center justify-center text-[#696969] text-xs py-24 gap-2">
              <ArrowPathIcon className="h-6 w-6 animate-spin text-[#4a154b]" />
              <span className="font-bold">Converting Word document…</span>
            </div>
          )}
          {docxError && (
            <div className="flex flex-col items-center justify-center py-24 text-[#696969] text-xs text-center gap-2">
              <DocumentIcon className="h-8 w-8 text-[#4a154b] opacity-40" />
              <span>Could not render Word document.</span>
            </div>
          )}
          {docxHtml && (
            <div
              className="docx-content bg-white p-8 rounded-2xl border border-[#e6e6e6] text-[#1d1d1d] text-sm leading-relaxed max-w-2xl mx-auto shadow-sm"
              dangerouslySetInnerHTML={{ __html: docxHtml }}
            />
          )}
        </div>
      ) : fileType === 'excel' ? (
        <>
          <div className="flex-1 overflow-auto bg-white p-6">
            {excelLoading && (
              <div className="flex flex-col items-center justify-center text-[#696969] text-xs py-24 gap-2">
                <ArrowPathIcon className="h-6 w-6 animate-spin text-[#4a154b]" />
                <span className="font-bold">Parsing spreadsheet…</span>
              </div>
            )}
            {excelError && (
              <div className="flex flex-col items-center justify-center py-24 text-[#696969] text-xs text-center gap-2">
                <DocumentIcon className="h-8 w-8 text-[#4a154b] opacity-40" />
                <span>Could not parse this spreadsheet.</span>
              </div>
            )}
            {sheetRows && (
              <table className="w-full text-xs border-collapse">
                <tbody>
                  {sheetRows.map((row, r) => (
                    <tr key={r} className={r === 0 ? 'bg-[#f9f0ff] font-bold text-[#4a154b]' : 'odd:bg-[#f4ede4]/30'}>
                      {row.map((cell, c) => (
                        <td key={c} className="border border-[#e6e6e6] px-3 py-1.5 text-[#1d1d1d] whitespace-nowrap">
                          {String(cell)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          {/* Excel-native sheet tab bar */}
          {workbook && workbook.SheetNames.length > 0 && (
            <div className="flex-shrink-0 flex items-end gap-0.5 px-2 pt-1.5 bg-[#f4ede4]/60 border-t border-[#e6e6e6] overflow-x-auto">
              {workbook.SheetNames.map((name) => {
                const isActive = name === activeSheet;
                const isCited = name === citedSheet;
                return (
                  <button
                    key={name}
                    onClick={() => setActiveSheet(name)}
                    title={name}
                    className={`flex-shrink-0 flex items-center gap-1.5 max-w-[160px] px-3.5 py-1.5 text-[11px] font-bold rounded-t-lg border border-b-0 transition-all ${
                      isActive
                        ? 'bg-white text-[#1d1d1d] border-[#e6e6e6] shadow-[0_-1px_4px_rgba(0,0,0,0.04)] relative z-10 -mb-px border-b-2 border-b-[#1a7a3c]'
                        : 'bg-[#f4ede4]/40 text-[#696969] border-transparent hover:bg-white/60 hover:text-[#1d1d1d]'
                    }`}
                  >
                    {isCited && (
                      <span
                        className="h-1.5 w-1.5 rounded-full bg-[#4a154b] flex-shrink-0"
                        title="Sheet referenced in this answer"
                      />
                    )}
                    <span className="truncate">{name}</span>
                  </button>
                );
              })}
            </div>
          )}
        </>
      ) : (
        <div
          ref={containerRef}
          className={`flex-1 overflow-auto bg-[#f4ede4]/60 p-6 flex items-start ${scale <= 1 ? 'justify-center' : 'justify-start'}`}
        >
          {imageUrl && (
            <div
              className="flex-shrink-0"
              style={{
                width: `${Math.round(scale * 100)}%`,
                transition: 'width 150ms ease-out',
                position: 'relative',
              }}
            >
              {!imgLoaded && !imgError && (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-[#696969] text-xs py-24 gap-2 bg-white/80 rounded-2xl border border-[#e6e6e6]">
                  <ArrowPathIcon className="h-6 w-6 animate-spin text-[#4a154b]" />
                  <span className="font-bold">{isPaginated ? `Rendering Page ${currentPage}…` : 'Loading image…'}</span>
                </div>
              )}
              {imgError && (
                <div className="flex flex-col items-center justify-center py-24 text-[#696969] text-xs text-center gap-2 bg-white rounded-2xl border border-[#e6e6e6]">
                  <DocumentIcon className="h-8 w-8 text-[#4a154b] opacity-40" />
                  <span>Page image not available.</span>
                </div>
              )}
              <img
                src={imageUrl}
                alt={isPaginated ? `Page ${currentPage}` : active?.filename}
                onLoad={() => setImgLoaded(true)}
                onError={() => { setImgError(true); setImgLoaded(true); }}
                className={`w-full rounded-2xl bg-white shadow-lg border border-[#e6e6e6] transition-opacity duration-300 ${imgLoaded && !imgError ? 'opacity-100' : 'opacity-0'
                  }`}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
};


// Best-effort: search_documents' tool result is a JSON string of
// {answer, citations, sources}. Pull sources out for a citation strip; anything
// else (a different tool, a malformed/blocked result) is skipped, not an error.
const parseSources = (toolCalls, allFiles = []) => {
  const sources = [];
  const allFilesList = Array.isArray(allFiles) ? allFiles : [];

  const resolveDoc = (nameOrId) => {
    if (!nameOrId) return null;
    const clean = String(nameOrId).trim().toLowerCase();
    let found = allFilesList.find(
      (f) => String(f.document_id || f.id).toLowerCase() === clean
    );
    if (!found) {
      found = allFilesList.find(
        (f) => String(f.filename).toLowerCase() === clean
      );
    }
    return found ? { document_id: found.document_id || found.id, filename: found.filename } : null;
  };

  for (const call of toolCalls || []) {
    if (call.name === 'search_documents' && typeof call.result === 'string') {
      try {
        const parsed = JSON.parse(call.result);
        if (Array.isArray(parsed.sources)) {
          const mapped = parsed.sources.map((s) => {
            const res = { ...s };
            if (s.page == null && s.slide != null) {
              res.page = s.slide;
            }
            if (s.page == null && s.sheet != null) {
              res.page = s.sheet;
            }
            const resolved = resolveDoc(s.filename || s.document_id);
            if (resolved) {
              res.document_id = resolved.document_id;
              res.filename = resolved.filename;
            }
            return res;
          });
          sources.push(...mapped);
        }
      } catch {
        // not JSON — nothing to show
      }
    } else if (call.name === 'excel_tool') {
      const filenameOrId = call.args?.filename_or_id;
      const sheetName = call.args?.sheet_name;
      if (filenameOrId) {
        const resolved = resolveDoc(filenameOrId);
        if (resolved) {
          sources.push({
            document_id: resolved.document_id,
            filename: resolved.filename,
            page: sheetName || 1,
            sheet: sheetName || null,
            score: 1.0,
            snippet: call.args?.code || '',
          });
        }
      }
    } else if (call.name === 'get_page_context') {
      const docId = call.args?.document_id;
      const page = call.args?.page;
      if (docId) {
        const resolved = resolveDoc(docId);
        if (resolved) {
          sources.push({
            document_id: resolved.document_id,
            filename: resolved.filename,
            page: page || 1,
            score: 1.0,
            snippet: `Fetched context of page ${page}`,
          });
        }
      }
    }
  }

  // Deduplicate
  const seen = new Set();
  return sources.filter((s) => {
    const key = `${s.filename || ''}::${s.page || ''}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
};

// Best-effort: list_documents' tool result is a JSON string of
// {count, returned, documents:[{filename, ...}], note?}. Pull out the
// returned filenames and counts so the UI can render what was actually listed.
const parseListedDocuments = (toolCalls) => {
  const documents = [];
  let note = "";
  let totalCount = 0;
  let returnedCount = 0;

  for (const call of toolCalls || []) {
    if (call.name !== 'list_documents' || typeof call.result !== 'string') continue;
    try {
      const parsed = JSON.parse(call.result);
      if (parsed && typeof parsed === 'object') {
        if (typeof parsed.count === 'number' && Number.isFinite(parsed.count)) {
          totalCount = Math.max(totalCount, parsed.count);
        }
        if (typeof parsed.returned === 'number' && Number.isFinite(parsed.returned)) {
          returnedCount = Math.max(returnedCount, parsed.returned);
        }
        if (Array.isArray(parsed.documents)) {
          documents.push(...parsed.documents);
        }
        if (!note && typeof parsed.note === 'string') {
          note = parsed.note;
        }
      }
    } catch {
      // not JSON (e.g. a blocked/error string) - nothing to show
    }
  }

  const seen = new Set();
  const uniqueDocuments = [];
  for (const doc of documents) {
    const key = `${doc?.document_id || ""}::${doc?.filename || ""}`;
    if (seen.has(key)) continue;
    seen.add(key);
    uniqueDocuments.push(doc);
  }

  return {
    documents: uniqueDocuments,
    note,
    totalCount: totalCount || uniqueDocuments.length,
    returnedCount: returnedCount || uniqueDocuments.length,
  };
};

const CustomCodeBlock = ({ language, value }) => {
  const [copied, setCopied] = useState(false);
  const [showScrollDown, setShowScrollDown] = useState(false);
  const containerRef = useRef(null);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const handleDownload = () => {
    const extensions = {
      python: 'py',
      javascript: 'js',
      typescript: 'ts',
      html: 'html',
      css: 'css',
      json: 'json',
      markdown: 'md',
      bash: 'sh',
      shell: 'sh',
      yaml: 'yaml',
      yml: 'yaml',
      sql: 'sql',
      text: 'txt',
    };
    const ext = extensions[language.toLowerCase()] || 'txt';
    const blob = new Blob([value], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `code-block.${ext}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const checkScroll = () => {
    const el = containerRef.current;
    if (!el) return;
    const isScrollable = el.scrollHeight > el.clientHeight;
    const isAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 16;
    setShowScrollDown(isScrollable && !isAtBottom);
  };

  useEffect(() => {
    checkScroll();

    if (typeof ResizeObserver !== 'undefined' && containerRef.current) {
      const observer = new ResizeObserver(() => {
        checkScroll();
      });
      observer.observe(containerRef.current);
      return () => observer.disconnect();
    }
  }, [value]);

  const handleScrollToBottom = () => {
    const el = containerRef.current;
    if (!el) return;
    el.scrollTo({
      top: el.scrollHeight,
      behavior: 'smooth',
    });
  };

  return (
    <div className="relative border border-slate-800 rounded-xl overflow-hidden my-4 bg-slate-950/90 group shadow-lg max-w-full">
      {/* Code block header */}
      <div className="flex items-center justify-between px-4 py-2 bg-slate-900/90 border-b border-slate-800/80 select-none">
        <span className="text-xs font-mono font-semibold text-slate-400 uppercase tracking-wider">
          {language}
        </span>
        <div className="flex items-center gap-3">
          {/* Copy button */}
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white transition-colors py-1 px-1.5 rounded hover:bg-slate-800"
            title="Copy to clipboard"
          >
            {copied ? (
              <>
                <CheckIcon className="h-3.5 w-3.5 text-emerald-400" />
                <span className="text-emerald-400 font-medium">Copied!</span>
              </>
            ) : (
              <>
                <svg
                  className="h-3.5 w-3.5"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                </svg>
                <span>Copy</span>
              </>
            )}
          </button>

          {/* Download button */}
          <button
            onClick={handleDownload}
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white transition-colors py-1 px-1.5 rounded hover:bg-slate-800"
            title="Download as file"
          >
            <svg
              className="h-3.5 w-3.5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            <span>Download</span>
          </button>
        </div>
      </div>

      {/* Code Container */}
      <pre
        ref={containerRef}
        onScroll={checkScroll}
        className="overflow-x-auto overflow-y-auto max-h-80 p-4 text-xs font-mono text-slate-100 leading-relaxed bg-slate-950/60 whitespace-pre scrollbar-thin"
      >
        <code>{value}</code>
      </pre>

      {/* Floating Scroll-Down Button */}
      {showScrollDown && (
        <button
          onClick={handleScrollToBottom}
          className="absolute bottom-4 right-4 p-2 rounded-full bg-slate-900/90 text-gray-400 hover:text-white hover:bg-slate-850 shadow-md transition-all hover:scale-105 active:scale-95 flex items-center justify-center border border-slate-800/80 z-10"
          title="Scroll to bottom"
        >
          <ChevronDownIcon className="h-4 w-4 stroke-[3]" />
        </button>
      )}
    </div>
  );
};

const MessageRow = ({ msg, onApprove, onDecline, onClarify, loading, onViewPages, onIngestApprove, onIngestCancel, allFiles, currency, setCurrency }) => {
  const isUser = msg.role === 'user';
  const sources = !isUser ? parseSources(msg.toolCalls, allFiles) : [];
  const rawListed = !isUser ? parseListedDocuments(msg.toolCalls) : {
    documents: [],
    note: "",
    totalCount: 0,
    returnedCount: 0,
  };

  const filteredDocs = (rawListed.documents || []).filter(doc => {
    if (sources.length === 0) return true;
    return sources.some(s =>
      String(s.document_id).toLowerCase() === String(doc.document_id).toLowerCase() ||
      String(s.filename).toLowerCase() === String(doc.filename).toLowerCase()
    );
  });

  const listedDocuments = {
    ...rawListed,
    documents: filteredDocs,
    totalCount: sources.length > 0 ? filteredDocs.length : rawListed.totalCount,
    returnedCount: sources.length > 0 ? filteredDocs.length : rawListed.returnedCount,
  };
  const [isSearchExpanded, setIsSearchExpanded] = useState(false);
  const [isListExpanded, setIsListExpanded] = useState(false);
  const [isTraceExpanded, setIsTraceExpanded] = useState(false);


  // Sources for the "View Source" button — filtered + sorted by score.
  // Covers PDF pages, docx (whole doc, snippet-matched), and images (whole file).
  const pageSources = buildViewableSources(sources);
  const hasPageSources = pageSources.length > 0;

  // ── Direct-ingestion card types ──────────────────────────────────────────
  if (!isUser && msg.type === 'ingest_approval') {
    return (
      <div className="py-3 flex justify-start">
        <div className="max-w-full w-full">
          <div className="flex gap-3">
            <SparklesIcon className="h-5 w-5 text-blue-400 flex-shrink-0 mt-1" />
            <div className="min-w-0 flex-1">
              <div className="inline-block bg-slate-800/80 border border-slate-700 rounded-2xl px-5 py-4 max-w-sm">
                <p className="text-sm font-medium text-gray-200 mb-1">Ingest this file?</p>
                <div className="flex items-center gap-2 mb-4 text-sm text-blue-300">
                  <DocumentIcon className="h-4 w-4 flex-shrink-0" />
                  <span className="truncate">{msg.filename}</span>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => onIngestApprove(msg.messageId, msg.stagedPath, msg.filename)}
                    className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white transition-all"
                  >
                    <CheckIcon className="h-3.5 w-3.5" /> Ingest
                  </button>
                  <button
                    onClick={() => onIngestCancel(msg.messageId)}
                    className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-xs font-semibold bg-slate-700 hover:bg-slate-600 text-gray-300 transition-all"
                  >
                    <XMarkIcon className="h-3.5 w-3.5" /> Cancel
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!isUser && msg.type === 'ingest_progress') {
    const steps = Array.isArray(msg.metrics) ? msg.metrics : [];
    return (
      <div className="py-3 flex justify-start">
        <div className="max-w-full w-full">
          <div className="flex gap-3">
            <SparklesIcon className="h-5 w-5 text-blue-400 flex-shrink-0 mt-1 animate-pulse" />
            <div className="min-w-0 flex-1">
              <div className="inline-block bg-slate-800/80 border border-slate-700 rounded-2xl px-5 py-4 max-w-md w-full">
                <div className="flex items-center gap-2 mb-3 text-sm text-gray-200">
                  <DocumentIcon className="h-4 w-4 text-blue-400 flex-shrink-0" />
                  <span className="truncate font-medium">{msg.filename}</span>
                </div>
                
                <div className="font-mono text-xs space-y-1.5 bg-slate-900/60 p-3 rounded-lg border border-slate-700/50">
                  {steps.length === 0 ? (
                    <div className="flex items-center gap-1.5 text-slate-400">
                      <span className="inline-block w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
                      Initializing...
                    </div>
                  ) : (() => {
                    const activeStep = steps.find((m) => m.status === 'running') || steps[steps.length - 1];
                    if (!activeStep) return null;

                    const durationStr = activeStep.ms != null ? `${(activeStep.ms / 1000).toFixed(1)}s` : '';
                    let statusStr = 'running';
                    let statusColor = 'text-blue-400';
                    if (activeStep.status === 'done') {
                      statusStr = 'ok';
                      statusColor = 'text-emerald-400';
                    } else if (activeStep.status === 'error' || activeStep.status === 'failed') {
                      statusStr = 'failed';
                      statusColor = 'text-red-400';
                    }

                    return (
                      <div className="flex justify-between items-center text-slate-300">
                        <div className="flex items-center gap-2 truncate">
                          <span className={activeStep.status === 'running' ? 'text-blue-400 animate-pulse' : 'text-slate-500'}>·</span>
                          <span className="truncate">{activeStep.step}</span>
                        </div>
                        <div className="flex items-center gap-4 flex-shrink-0">
                          <span className={`font-semibold ${statusColor}`}>{statusStr}</span>
                          {durationStr && <span className="text-slate-500 min-w-[36px] text-right">{durationStr}</span>}
                        </div>
                      </div>
                    );
                  })()}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!isUser && msg.type === 'ingest_done') {
    return (
      <div className="py-2.5 flex justify-start">
        <div className="max-w-full w-full">
          <div className="flex gap-3">
            <SparklesIcon className="h-5 w-5 text-[#007a5a] flex-shrink-0 mt-1" />
            <div className="min-w-0 flex-1">
              <div className="inline-block bg-white border border-[#007a5a]/30 rounded-2xl p-4 shadow-sm max-w-md">
                <div className="flex items-center gap-2 mb-1">
                  <CheckIcon className="h-4 w-4 text-[#007a5a] stroke-[2.5] flex-shrink-0" />
                  <p className="text-xs font-extrabold text-[#007a5a]">Ingested successfully!</p>
                </div>
                <p className="text-xs text-[#1d1d1d] mt-1 leading-relaxed">
                  <span className="font-bold text-[#4a154b]">{msg.filename}</span> is now in the knowledge base. You can ask questions about it.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!isUser && msg.type === 'ingest_cancelled') {
    return (
      <div className="py-2 flex justify-start">
        <div className="max-w-full w-full">
          <div className="flex gap-3">
            <SparklesIcon className="h-5 w-5 text-[#696969] flex-shrink-0 mt-1" />
            <div className="min-w-0 flex-1">
              <p className="text-xs text-[#696969] italic py-1 font-medium">Ingestion cancelled.</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!isUser && msg.type === 'ingest_error') {
    return (
      <div className="py-2.5 flex justify-start">
        <div className="max-w-full w-full">
          <div className="flex gap-3">
            <SparklesIcon className="h-5 w-5 text-[#cc4117] flex-shrink-0 mt-1" />
            <div className="min-w-0 flex-1">
              <div className="inline-block bg-white border border-[#cc4117]/30 rounded-2xl p-4 shadow-sm max-w-md">
                <div className="flex items-center gap-2 mb-1">
                  <ExclamationCircleIcon className="h-4 w-4 text-[#cc4117] stroke-[2.5] flex-shrink-0" />
                  <p className="text-xs font-extrabold text-[#cc4117]">Ingestion failed</p>
                </div>
                <p className="text-xs text-[#1d1d1d] mt-1 leading-relaxed">{msg.errorMsg || 'Something went wrong. Please try again.'}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }
  // ── End direct-ingestion cards ───────────────────────────────────────────

  return (
    <div className={`py-2.5 flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={isUser ? 'max-w-xl' : 'max-w-full w-full'}>
        {isUser ? (
          <div className="bg-[#4a154b] text-white rounded-2xl px-4 py-2.5 shadow-sm font-medium text-xs md:text-sm leading-relaxed whitespace-pre-wrap">
            {msg.content}
          </div>
        ) : (
          <div className="bg-white border border-[#e6e6e6] rounded-2xl p-4 md:p-5 shadow-sm text-[#1d1d1d]">
            <div className="flex gap-3">
              <SparklesIcon className="h-5 w-5 text-[#4a154b] flex-shrink-0 mt-1" />
              <div className="min-w-0 flex-1">
                {!msg.content && (
                  <div className="flex items-center text-[#4a154b] text-sm py-1 font-bold">
                    <span>
                      {msg.isIngesting
                        ? 'Ingesting Document'
                        : 'Thinking'}
                      <span className="animate-dots"></span>
                    </span>
                  </div>
                )}
                {msg.content && (
                  <div
                    className={`prose prose-sm max-w-none leading-relaxed w-full overflow-x-auto text-[#1d1d1d] ${msg.isError ? 'text-[#cc4117]' : ''
                      }`}
                  >
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      rehypePlugins={[rehypeRaw]}
                      components={{
                        code({ node, inline, className, children, ...props }) {
                          const match = /language-(\w+)/.exec(className || '');
                          const language = match ? match[1] : 'text';
                          return !inline ? (
                            <CustomCodeBlock language={language} value={String(children).replace(/\n$/, '')} />
                          ) : (
                            <code className="bg-[#f9f0ff] text-[#4a154b] px-1.5 py-0.5 rounded text-xs font-mono border border-[#e6e6e6]" {...props}>
                              {children}
                            </code>
                          );
                        },
                        a({ node, children, ...props }) {
                          return (
                            <a className="text-[#1264a3] hover:text-[#3860be] font-semibold underline" {...props}>
                              {children}
                            </a>
                          );
                        }
                      }}
                    >
                      {renderMathInMarkdown(msg.content)}
                    </ReactMarkdown>
                  </div>
                )}

                {/* Tool calls this turn */}
                {msg.toolCalls && msg.toolCalls.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {(() => {
                      const grouped = msg.toolCalls.reduce((acc, call) => {
                        const existing = acc.find(g => g.name === call.name);
                        if (existing) {
                          existing.count += 1;
                          existing.calls.push(call);
                        } else {
                          acc.push({ name: call.name, count: 1, calls: [call] });
                        }
                        return acc;
                      }, []);

                      return grouped.map((group, i) => {
                        const isSearch = group.name === 'search_documents';
                        const isList = group.name === 'list_documents';

                        if (isSearch && sources.length > 0) {
                          return (
                            <button
                              key={i}
                              onClick={() => {
                                setIsSearchExpanded(!isSearchExpanded);
                                setIsListExpanded(false);
                              }}
                              className="inline-flex items-center gap-1.5 text-xs text-[#4a154b] bg-[#f9f0ff] border border-[#e6e6e6] rounded-full px-3 py-1 hover:bg-[#4a154b] hover:text-white transition-all cursor-pointer font-bold shadow-sm"
                            >
                              <WrenchScrewdriverIcon className="h-3.5 w-3.5" />
                              <span className="font-mono">{group.name}</span>
                              {group.count > 1 && (
                                <span className="ml-1 text-[10px] bg-[#4a154b]/20 px-1.5 py-0.5 rounded-full font-sans font-bold">
                                  {group.count}
                                </span>
                              )}
                              {isSearchExpanded ? (
                                <ChevronUpIcon className="h-3.5 w-3.5" />
                              ) : (
                                <ChevronDownIcon className="h-3.5 w-3.5" />
                              )}
                            </button>
                          );
                        }

                        if (isList) {
                          const listCount = listedDocuments.totalCount || listedDocuments.returnedCount || listedDocuments.documents.length;
                          return (
                            <button
                              key={i}
                              onClick={() => {
                                setIsListExpanded(!isListExpanded);
                                setIsSearchExpanded(false);
                              }}
                              className="inline-flex items-center gap-1.5 text-xs text-[#4a154b] bg-[#f9f0ff] border border-[#e6e6e6] rounded-full px-3 py-1 hover:bg-[#4a154b] hover:text-white transition-all cursor-pointer font-bold shadow-sm"
                            >
                              <WrenchScrewdriverIcon className="h-3.5 w-3.5" />
                              <span className="font-mono">{group.name}</span>
                              {listCount > 0 && (
                                <span className="ml-1 text-[10px] bg-[#4a154b]/20 px-1.5 py-0.5 rounded-full font-sans font-bold">
                                  {listCount}
                                </span>
                              )}
                              {isListExpanded ? (
                                <ChevronUpIcon className="h-3.5 w-3.5" />
                              ) : (
                                <ChevronDownIcon className="h-3.5 w-3.5" />
                              )}
                            </button>
                          );
                        }

                        return (
                          <span
                            key={i}
                            className="inline-flex items-center gap-1.5 text-xs text-[#696969] bg-[#f9f0ff]/50 border border-[#e6e6e6] rounded-full px-3 py-1 select-none font-medium"
                          >
                            <WrenchScrewdriverIcon className="h-3 w-3 text-[#4a154b]" />
                            <span className="font-mono">{group.name}</span>
                            {group.count > 1 && (
                              <span className="text-[10px] text-[#4a154b] bg-[#f9f0ff] border border-[#e6e6e6] px-1.5 py-0.5 rounded-full font-sans font-bold">
                                x{group.count}
                              </span>
                            )}
                          </span>
                        );
                      });
                    })()}
                  </div>
                )}

                {/* Inline citation list */}
                {isListExpanded && (
                  <div className="mt-3 space-y-1.5 max-w-md bg-[#f9f0ff]/40 border border-[#e6e6e6] rounded-2xl p-4">
                    <p className="text-[10px] font-bold text-[#4a154b] uppercase tracking-wider mb-2">Listed Documents</p>
                    {(() => {
                      const docs = listedDocuments.documents || [];
                      if (docs.length === 0) {
                        return <p className="text-xs text-[#696969] italic p-1">No documents were returned by list_documents.</p>;
                      }
                      return docs.map((doc, idx) => (
                        <div key={idx} className="flex items-center gap-2.5 text-xs text-[#1d1d1d] bg-white border border-[#e6e6e6] rounded-xl px-3.5 py-2 font-medium shadow-sm">
                          <DocumentIcon className="h-4 w-4 text-[#4a154b] flex-shrink-0" />
                          <span className="font-bold truncate">{doc.filename || doc.document_id || 'Untitled document'}</span>
                        </div>
                      ));
                    })()}
                    {listedDocuments.note && (
                      <p className="text-[10px] text-[#696969] mt-2">{listedDocuments.note}</p>
                    )}
                  </div>
                )}

                {/* Inline chunks listing */}
                {isSearchExpanded && sources.length > 0 && (
                  <div className="mt-3 space-y-2.5 max-w-2xl bg-[#f9f0ff]/40 border border-[#e6e6e6] rounded-2xl p-4 animate-fade-in">
                    <p className="text-[10px] font-bold text-[#4a154b] uppercase tracking-wider mb-2">Search Sources</p>
                    {sources.map((s, i) => (
                      <div key={i} className="text-xs text-[#1d1d1d] bg-white border border-[#e6e6e6] rounded-xl p-4 space-y-2 shadow-sm">
                        <div className="flex items-center justify-between border-b border-[#e6e6e6] pb-2">
                          <span className="font-bold text-[#4a154b]">{s.filename}</span>
                          {s.page && <span className="text-[#696969] text-[10px] font-bold bg-[#f9f0ff] px-2 py-0.5 rounded-full border border-[#e6e6e6]">Page {s.page}</span>}
                        </div>
                        {s.snippet && (
                          <p className="italic text-[#1d1d1d] leading-relaxed whitespace-pre-wrap font-mono text-[11px] bg-[#f4ede4]/40 p-3 rounded-xl border border-[#e6e6e6]">
                            "{s.snippet}"
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {/* Token usage */}
                {msg.tokenUsage && (
                  <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-[#696969] border-t border-[#e6e6e6] pt-2.5">
                    <span className="inline-flex items-center gap-1 text-[#4a154b] font-bold">
                      <CircleStackIcon className="h-3.5 w-3.5" />
                      Tokens: {msg.tokenUsage.total_tokens}
                    </span>
                    {msg.tokenUsage.total_cost_usd !== undefined && (
                      <button
                        onClick={() => setCurrency(currency === 'USD' ? 'INR' : 'USD')}
                        className="btn-primary-pill !py-0.5 !px-2 !text-[10px] ml-auto font-bold inline-flex items-center gap-1"
                      >
                        {currency === 'USD'
                          ? `Cost: $${msg.tokenUsage.total_cost_usd.toFixed(4)}`
                          : `Cost: ₹${(msg.tokenUsage.total_cost_usd * USD_TO_INR).toFixed(2)}`}
                      </button>
                    )}
                    <span>Input: {msg.tokenUsage.input_tokens}</span>
                    <span>Output: {msg.tokenUsage.output_tokens}</span>
                    {msg.tokenUsage.reasoning_tokens > 0 && (
                      <span>Thinking: {msg.tokenUsage.reasoning_tokens}</span>
                    )}
                    <span>Context: {msg.tokenUsage.context_tokens}</span>
                  </div>
                )}




                {/* Write awaiting approval */}
                {msg.status === 'needs_approval' && (
                  <div className="mt-4 pt-3 border-t border-[#e6e6e6] space-y-3">
                    {msg.pending.map((p, i) => (
                      <p key={i} className="text-sm font-bold text-[#4a154b]">
                        Wants to execute tool: <span className="font-mono bg-[#f9f0ff] px-2 py-0.5 rounded border border-[#e6e6e6]">{p.name}</span>
                      </p>
                    ))}
                    <div className="flex gap-3">
                      <button
                        onClick={onApprove}
                        disabled={loading}
                        className="btn-primary-pill text-xs !px-5 !py-2"
                      >
                        <CheckIcon className="h-4 w-4" /> Approve Execution
                      </button>
                      <button
                        onClick={onDecline}
                        disabled={loading}
                        className="btn-secondary-pill text-xs !px-5 !py-2 !text-[#cc4117]"
                      >
                        <XMarkIcon className="h-4 w-4" /> Decline
                      </button>
                    </div>
                  </div>
                )}
                {/* Agent asked the user to choose (needs_clarification) */}
                {msg.status === 'needs_clarification' && (
                  <div className="mt-4 pt-3 border-t border-[#e6e6e6] space-y-3">
                    {msg.options && msg.options.length > 0 ? (
                      <div className="flex flex-wrap gap-2">
                        {msg.options.map((opt, i) => (
                          <button
                            key={i}
                            onClick={() => onClarify(opt)}
                            disabled={loading || msg.clarifyAnswered}
                            className="btn-secondary-pill text-xs !px-4 !py-2 hover:bg-[#4a154b] hover:text-white"
                          >
                            {opt}
                          </button>
                        ))}
                      </div>
                    ) : (
                      <p className="text-xs text-[#696969] italic">Type your choice in the message bar.</p>
                    )}
                  </div>
                )}
                {msg.status === 'declined' && (
                  <p className="mt-2 text-xs text-[#696969] italic">Action declined.</p>
                )}

                {/* View Source Pages button */}
                {hasPageSources && (
                  <div className="mt-4 pt-3 border-t border-[#e6e6e6]">
                    <button
                      onClick={() => onViewPages(pageSources)}
                      className="btn-secondary-pill text-xs inline-flex items-center gap-2 !py-2"
                    >
                      <DocumentIcon className="h-4 w-4 text-[#4a154b]" />
                      View Source Documents ({pageSources.length})
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatPage;
