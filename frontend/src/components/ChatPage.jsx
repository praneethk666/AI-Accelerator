import React, { useState, useRef, useEffect } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import * as XLSX from 'xlsx';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import {
  sendAgentChat, streamAgentChat, getAgentSessions, getAgentSession, deleteAgentSession,
  patchAgentSession, stageFile, getFile, API_BASE_URL, getFiles,
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
  ChevronLeftIcon,
  ChevronRightIcon,
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
    }) === i)
    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
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

const ChatPage = () => {
  const [searchParams] = useSearchParams();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [pageViewer, setPageViewer] = useState(null);
  // pageViewer: null | { pages: [{page, documentId, filename, score}], activeIdx: number }
  const [sessions, setSessions] = useState([]);
  const [sessionId, setSessionId] = useState(newSessionId);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loadingSessions, setLoadingSessions] = useState({});
  const loading = !!loadingSessions[sessionId];
  const setSessionLoading = (sessId, val) => {
    setLoadingSessions((prev) => ({ ...prev, [sessId]: val }));
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

  // Mirrors sessionId so an in-flight reply can check whether the user switched
  // conversations before it landed (otherwise A's answer appends under B).
  const sessionIdRef = useRef(sessionId);
  useEffect(() => { sessionIdRef.current = sessionId; }, [sessionId]);

  const streamControllerRef = useRef(null);
  useEffect(() => {
    return () => {
      if (streamControllerRef.current) {
        streamControllerRef.current.abort();
      }
    };
  }, []);


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
  useEffect(() => {
    if (pageViewer) {
      setSidebarOpen(false);
    } else {
      setSidebarOpen(true);
    }
  }, [pageViewer]);

  const fileId = searchParams.get('fileId');

  useEffect(() => { loadSessions(); loadFiles(); }, []);
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, loading]);

  useEffect(() => {
    if (!fileId) return;
    getFile(fileId).then((res) => setContextFile(res.data)).catch(() => { });
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
        pending: [],
        question: null,
        options: [],
        tokenUsage: null,
        traceId: null,
        messageId,
        isStreaming: true,
        originalText: optionText,
      },
    ]);

    try {
      if (streamControllerRef.current) {
        streamControllerRef.current.abort();
      }

      const controller = await streamAgentChat(
        optionText,
        reqSession,
        messageId,
        false,
        null,
        {
          onToken: (text) => {
            if (sessionIdRef.current !== reqSession) return;
            setMessages((prev) => {
              const exists = prev.some((m) => m.messageId === messageId);
              if (exists) {
                return prev.map((m) =>
                  m.messageId === messageId ? { ...m, content: m.content + text } : m
                );
              } else {
                return [
                  ...prev,
                  {
                    role: 'assistant',
                    content: text,
                    toolCalls: [],
                    pending: [],
                    messageId,
                    isStreaming: true,
                    originalText: optionText,
                  }
                ];
              }
            });
          },
          onToolStart: (name) => {
            if (sessionIdRef.current !== reqSession) return;
            setMessages((prev) => {
              const exists = prev.some((m) => m.messageId === messageId);
              if (exists) {
                return prev.map((m) =>
                  m.messageId === messageId
                    ? {
                        ...m,
                        toolCalls: [...(m.toolCalls || []), { name, args: {}, status: 'running' }],
                      }
                    : m
                );
              } else {
                return [
                  ...prev,
                  {
                    role: 'assistant',
                    content: '',
                    toolCalls: [{ name, args: {}, status: 'running' }],
                    pending: [],
                    messageId,
                    isStreaming: true,
                    originalText: optionText,
                  }
                ];
              }
            });
          },
          onToolEnd: (name) => {
            if (sessionIdRef.current !== reqSession) return;
            setMessages((prev) => {
              const exists = prev.some((m) => m.messageId === messageId);
              if (exists) {
                return prev.map((m) =>
                  m.messageId === messageId
                    ? {
                        ...m,
                        toolCalls: (m.toolCalls || []).map((t) =>
                          t.name === name ? { ...t, status: 'completed' } : t
                        ),
                      }
                    : m
                );
              } else {
                return prev;
              }
            });
          },
          onDone: (fullText) => {
            if (sessionIdRef.current !== reqSession) {
              setSessionLoading(reqSession, false);
              loadSessions();
              return;
            }
            setMessages((prev) => {
              const exists = prev.some((m) => m.messageId === messageId);
              if (exists) {
                return prev.map((m) =>
                  m.messageId === messageId ? { ...m, content: fullText, isStreaming: false } : m
                );
              } else {
                return [
                  ...prev,
                  {
                    role: 'assistant',
                    content: fullText,
                    toolCalls: [],
                    pending: [],
                    messageId,
                    isStreaming: false,
                    originalText: optionText,
                  }
                ];
              }
            });
            setSessionLoading(reqSession, false);
            loadSessions();
          },
          onMetadata: (metadata) => {
            if (sessionIdRef.current !== reqSession) {
              setSessionLoading(reqSession, false);
              return;
            }
            setMessages((prev) => {
              const exists = prev.some((m) => m.messageId === messageId);
              if (exists) {
                return prev.map((m) =>
                  m.messageId === messageId
                    ? {
                        ...m,
                        status: metadata.status,
                        content: metadata.answer,
                        toolCalls: metadata.tool_calls || [],
                        tokenUsage: metadata.token_usage || null,
                        traceId: metadata.trace_id || null,
                        isStreaming: false,
                      }
                    : m
                );
              } else {
                return [
                  ...prev,
                  {
                    role: 'assistant',
                    status: metadata.status,
                    content: metadata.answer,
                    toolCalls: metadata.tool_calls || [],
                    tokenUsage: metadata.token_usage || null,
                    traceId: metadata.trace_id || null,
                    messageId,
                    isStreaming: false,
                    originalText: optionText,
                  }
                ];
              }
            });
            setSessionLoading(reqSession, false);
            updatePageViewer(metadata);
          },
          onError: (errMsg) => {
            if (sessionIdRef.current !== reqSession) {
              setSessionLoading(reqSession, false);
              return;
            }
            setMessages((prev) => {
              const exists = prev.some((m) => m.messageId === messageId);
              if (exists) {
                return prev.map((m) =>
                  m.messageId === messageId
                    ? {
                        ...m,
                        content: `Error: ${errMsg}`,
                        isError: true,
                        isStreaming: false,
                      }
                    : m
                );
              } else {
                return [
                  ...prev,
                  {
                    role: 'assistant',
                    content: `Error: ${errMsg}`,
                    isError: true,
                    isStreaming: false,
                    messageId,
                    originalText: optionText,
                  }
                ];
              }
            });
            setSessionLoading(reqSession, false);
            setError(errMsg);
          },
        }
      );

      streamControllerRef.current = controller;
    } catch (err) {
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
                isStreaming: false,
              }
            : m
        )
      );
      setError(err.message);
    }
  };
  // Extract page-level sources from a response and open/close the page viewer.
  // Reuses parseSources (slide/sheet -> page remap) + buildViewableSources
  // (fileType tagging, filtering, dedup, sort) so this stays in sync with the
  // per-message "View Source" button in MessageRow.
  const updatePageViewer = (data) => {
    const allSources = parseSources(data.tool_calls || []);

    const toolCalls = data.tool_calls || [];
    for (const call of toolCalls) {
      if (call.name === 'excel_tst_tool') {
        const filenameOrId = call.args?.filename_or_id;
        const sheetName = call.args?.sheet_name;
        if (filenameOrId) {
          const matched = allFiles.find(
            f => f.id === filenameOrId || f.document_id === filenameOrId ||
                 f.filename === filenameOrId || f.filename?.toLowerCase() === filenameOrId.toLowerCase()
          );
          if (matched) {
            allSources.push({
              document_id: matched.document_id || matched.id,
              filename: matched.filename,
              page: sheetName || 1,
              sheet: sheetName || null,
              score: 1.0
            });
          }
        }
      }
    }

    const viewableSources = buildViewableSources(allSources);

    if (viewableSources.length > 0) {
      setPageViewer({ pages: viewableSources, activeIdx: 0 });
    } else {
      setPageViewer(null);
    }
  };

  const handleSend = async () => {
    if (!input.trim() && !attachedFile) return;

    const displayText = attachedFile
      ? `${input.trim()}${input.trim() ? '\n\n' : ''}📎 ${attachedFile.filename}`
      : input;
    const sentText = composeSentText(input, attachedFile);
    const rawInput = input.trim();

    // If we have a placeholder for the current session, update its title immediately
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

    // Generate messageId for de-duplication and state mapping
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
        isStreaming: true,
        originalText: sentText,
        isIngesting: !!attachedFile,
      },
    ]);

    try {
      if (streamControllerRef.current) {
        streamControllerRef.current.abort();
      }

      const controller = await streamAgentChat(
        sentText,
        reqSession,
        messageId,
        false,
        null,
        {
          onToken: (text) => {
            if (sessionIdRef.current !== reqSession) return;
            setMessages((prev) => {
              const exists = prev.some((m) => m.messageId === messageId);
              if (exists) {
                return prev.map((m) =>
                  m.messageId === messageId ? { ...m, content: m.content + text } : m
                );
              } else {
                return [
                  ...prev,
                  {
                    role: 'assistant',
                    content: text,
                    toolCalls: [],
                    pending: [],
                    messageId,
                    isStreaming: true,
                    originalText: sentText,
                    isIngesting: !!attachedFile,
                  }
                ];
              }
            });
          },
          onToolStart: (name) => {
            if (sessionIdRef.current !== reqSession) return;
            setMessages((prev) => {
              const exists = prev.some((m) => m.messageId === messageId);
              if (exists) {
                return prev.map((m) =>
                  m.messageId === messageId
                    ? {
                        ...m,
                        toolCalls: [...(m.toolCalls || []), { name, args: {}, status: 'running' }],
                      }
                    : m
                );
              } else {
                return [
                  ...prev,
                  {
                    role: 'assistant',
                    content: '',
                    toolCalls: [{ name, args: {}, status: 'running' }],
                    pending: [],
                    messageId,
                    isStreaming: true,
                    originalText: sentText,
                    isIngesting: !!attachedFile,
                  }
                ];
              }
            });
          },
          onToolEnd: (name) => {
            if (sessionIdRef.current !== reqSession) return;
            setMessages((prev) => {
              const exists = prev.some((m) => m.messageId === messageId);
              if (exists) {
                return prev.map((m) =>
                  m.messageId === messageId
                    ? {
                        ...m,
                        toolCalls: (m.toolCalls || []).map((t) =>
                          t.name === name ? { ...t, status: 'completed' } : t
                        ),
                      }
                    : m
                );
              } else {
                return prev;
              }
            });
          },
          onDone: (fullText) => {
            if (sessionIdRef.current !== reqSession) {
              setSessionLoading(reqSession, false);
              loadSessions();
              return;
            }
            setMessages((prev) => {
              const exists = prev.some((m) => m.messageId === messageId);
              if (exists) {
                return prev.map((m) =>
                  m.messageId === messageId ? { ...m, content: fullText, isStreaming: false } : m
                );
              } else {
                return [
                  ...prev,
                  {
                    role: 'assistant',
                    content: fullText,
                    toolCalls: [],
                    pending: [],
                    messageId,
                    isStreaming: false,
                    originalText: sentText,
                    isIngesting: !!attachedFile,
                  }
                ];
              }
            });
            setSessionLoading(reqSession, false);
            loadSessions();
          },
          onMetadata: (metadata) => {
            if (sessionIdRef.current !== reqSession) {
              setSessionLoading(reqSession, false);
              return;
            }
            setMessages((prev) => {
              const exists = prev.some((m) => m.messageId === messageId);
              if (exists) {
                return prev.map((m) =>
                  m.messageId === messageId
                    ? {
                        ...m,
                        status: metadata.status,
                        content: metadata.answer,
                        toolCalls: metadata.tool_calls || [],
                        tokenUsage: metadata.token_usage || null,
                        traceId: metadata.trace_id || null,
                        isStreaming: false,
                      }
                    : m
                );
              } else {
                return [
                  ...prev,
                  {
                    role: 'assistant',
                    status: metadata.status,
                    content: metadata.answer,
                    toolCalls: metadata.tool_calls || [],
                    tokenUsage: metadata.token_usage || null,
                    traceId: metadata.trace_id || null,
                    messageId,
                    isStreaming: false,
                    originalText: sentText,
                    isIngesting: !!attachedFile,
                  }
                ];
              }
            });
            setSessionLoading(reqSession, false);
            updatePageViewer(metadata);
          },
          onError: (errMsg) => {
            if (sessionIdRef.current !== reqSession) {
              setSessionLoading(reqSession, false);
              return;
            }
            setMessages((prev) => {
              const exists = prev.some((m) => m.messageId === messageId);
              if (exists) {
                return prev.map((m) =>
                  m.messageId === messageId
                    ? {
                        ...m,
                        content: `Error: ${errMsg}`,
                        isError: true,
                        isStreaming: false,
                      }
                    : m
                );
              } else {
                return [
                  ...prev,
                  {
                    role: 'assistant',
                    content: `Error: ${errMsg}`,
                    isError: true,
                    isStreaming: false,
                    messageId,
                    originalText: sentText,
                    isIngesting: !!attachedFile,
                  }
                ];
              }
            });
            setSessionLoading(reqSession, false);
            setError(errMsg);
          },
        }
      );

      streamControllerRef.current = controller;
    } catch (err) {
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
                isStreaming: false,
              }
            : m
        )
      );
      setError(err.message);
    }
  };

  const handleApproval = async (msgIdx, approved) => {
    const msg = messages[msgIdx];
    if (!approved) {
      setMessages((prev) => prev.map((m, i) => (i === msgIdx ? { ...m, status: 'declined' } : m)));
      return;
    }
    const reqSession = sessionId;
    setSessionLoading(reqSession, true);

    // Re-use the existing messageId or generate one if not present
    const messageId = msg.messageId || (crypto.randomUUID ? crypto.randomUUID() : `msg-${Date.now()}-${Math.random().toString(16).slice(2)}`);

    // Reset this message state to prepare for streaming approval response
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
              isStreaming: true,
            }
          : m
      )
    );

    try {
      if (streamControllerRef.current) {
        streamControllerRef.current.abort();
      }

      const controller = await streamAgentChat(
        msg.originalText,
        reqSession,
        messageId,
        true,
        msg.pending || [],
        {
          onToken: (text) => {
            if (sessionIdRef.current !== reqSession) return;
            setMessages((prev) =>
              prev.map((m) =>
                m.messageId === messageId ? { ...m, content: m.content + text } : m
              )
            );
          },
          onToolStart: (name) => {
            if (sessionIdRef.current !== reqSession) return;
            setMessages((prev) =>
              prev.map((m) =>
                m.messageId === messageId
                  ? {
                      ...m,
                      toolCalls: [...(m.toolCalls || []), { name, args: {}, status: 'running' }],
                    }
                  : m
              )
            );
          },
          onToolEnd: (name) => {
            if (sessionIdRef.current !== reqSession) return;
            setMessages((prev) =>
              prev.map((m) =>
                m.messageId === messageId
                  ? {
                      ...m,
                      toolCalls: (m.toolCalls || []).map((t) =>
                        t.name === name ? { ...t, status: 'completed' } : t
                      ),
                    }
                  : m
              )
            );
          },
          onDone: (fullText) => {
            if (sessionIdRef.current !== reqSession) {
              loadSessions();
              return;
            }
            setMessages((prev) =>
              prev.map((m) =>
                m.messageId === messageId ? { ...m, content: fullText, isStreaming: false } : m
              )
            );
            loadSessions();
          },
          onError: (errMsg) => {
            if (sessionIdRef.current !== reqSession) return;
            setMessages((prev) =>
              prev.map((m) =>
                m.messageId === messageId
                  ? {
                      ...m,
                      content: `Error: ${errMsg}`,
                      isError: true,
                      isStreaming: false,
                    }
                  : m
              )
            );
            setError(errMsg);
          },
        }
      );

      streamControllerRef.current = controller;
    } catch (err) {
      console.error('Approval error:', err);
      if (sessionIdRef.current !== reqSession) return;
      setMessages((prev) =>
        prev.map((m) =>
          m.messageId === messageId
            ? {
                ...m,
                content: `Error: ${err.message || 'Failed to execute approved actions.'}`,
                isError: true,
                isStreaming: false,
              }
            : m
        )
      );
      setError(err.message);
    } finally {
      setSessionLoading(reqSession, false);
    }
  };

  const handleTextareaChange = (e) => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  };

  return (
    <div className="flex h-screen bg-slate-900 text-gray-100 relative">
      {/* Sidebar */}
      <div
        className={`flex-shrink-0 flex flex-col bg-slate-950/60 border-r border-slate-800 transition-all duration-300 ${sidebarOpen ? 'w-64' : 'w-0 overflow-hidden border-r-0'
          }`}
      >
        {/* Sidebar Header */}
        <div className="p-3 flex items-center justify-between border-b border-slate-800/60 h-14">
          <div className="flex items-center gap-2 font-semibold text-white tracking-wide text-sm truncate">
            <span>AI Accelerator</span>
          </div>
          <Tooltip content="Close sidebar">
            <button
              onClick={() => setSidebarOpen(false)}
              className="p-1.5 text-gray-400 hover:text-white hover:bg-slate-850 rounded-lg transition-all"
            >
              <SidebarToggleIcon className="h-4 w-4" />
            </button>
          </Tooltip>
        </div>

        <div className="p-3">
          <button
            onClick={handleNewChat}
            className="w-full flex items-center justify-center gap-2 px-3 py-2.5 rounded-full border border-slate-800 hover:border-slate-700 bg-slate-900/50 hover:bg-slate-800 text-sm font-medium text-gray-200 transition-all shadow-sm"
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
                className={`group flex items-center justify-between gap-2 px-3 py-2 rounded-lg cursor-pointer text-sm transition-colors ${s.session_id === sessionId
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
      <div className="flex-1 flex flex-col min-w-0 relative">
        {/* Top Header Bar */}
        <div className="h-14 flex items-center px-4 justify-between bg-slate-900/40 backdrop-blur-sm z-30 flex-shrink-0">
          <div className="flex items-center gap-3">
            {!sidebarOpen && (
              <div className="flex items-center gap-0.5 bg-slate-950/80 border border-slate-800/80 rounded-full p-1 shadow-md">
                <Tooltip content="Open sidebar">
                  <button
                    onClick={() => setSidebarOpen(true)}
                    className="p-1.5 text-gray-400 hover:text-white hover:bg-slate-800 rounded-full transition-all"
                  >
                    <SidebarToggleIcon className="h-4 w-4" />
                  </button>
                </Tooltip>
                <div className="w-[1px] h-3.5 bg-slate-800 mx-1" />
                <Tooltip content="New chat">
                  <button
                    onClick={handleNewChat}
                    className="p-1.5 text-gray-400 hover:text-white hover:bg-slate-800 rounded-full transition-all"
                  >
                    <NewChatIcon className="h-4 w-4" />
                  </button>
                </Tooltip>
              </div>
            )}
            {contextFile && (
              <span className="text-xs text-gray-500 flex items-center gap-1.5 ml-2">
                <DocumentIcon className="h-3.5 w-3.5 text-blue-500/85" />
                Context: {contextFile.filename}
              </span>
            )}
          </div>

          <div className="flex items-center gap-2" />
        </div>

        <div
          ref={chatContainerRef}
          onScroll={checkChatScroll}
          className="flex-1 overflow-y-auto relative"
        >
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
                onClarify={(opt) => handleClarify(opt, idx)} loading={loading}
                onViewPages={(pages) => setPageViewer({ pages, activeIdx: 0 })} />
            ))}

            {loading && messages.length > 0 && messages[messages.length - 1].role === 'user' && (
              <div className="py-3 flex justify-start">
                <div className="max-w-full w-full">
                  <div className="flex gap-3">
                    <SparklesIcon className="h-5 w-5 text-blue-400 flex-shrink-0 mt-1" />
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center text-slate-400 text-sm py-1">
                        <span className="font-medium">
                          Thinking<span className="animate-dots"></span>
                        </span>
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
          <div className="max-w-3xl mx-auto w-full px-4 flex-shrink-0">
            <div className="mb-2 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm flex items-start gap-2">
              <ExclamationCircleIcon className="h-4 w-4 flex-shrink-0 mt-0.5" />
              {error}
            </div>
          </div>
        )}

        <div className="border-t border-slate-800 bg-slate-900 p-4 flex-shrink-0">
          <div className="max-w-3xl mx-auto relative">
            {showChatScrollDown && (
              <button
                onClick={handleChatScrollToBottom}
                className="absolute bottom-full mb-3 right-4 p-2.5 rounded-full bg-slate-950 hover:bg-slate-850 text-gray-400 hover:text-white shadow-xl hover:scale-105 active:scale-95 transition-all duration-200 flex items-center justify-center border border-slate-800 z-40"
                title="Scroll to bottom"
                style={{ cursor: 'pointer' }}
              >
                <ChevronDownIcon className="h-5 w-5 stroke-[2.5]" />
              </button>
            )}
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
                className={`p-2.5 rounded-full flex-shrink-0 transition-colors ${loading || (!input.trim() && !attachedFile)
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

  const getBadgeLabel = (p) => {
    if (p.fileType === 'pdf') return `P.${p.page}`;
    if (p.fileType === 'ppt') return `Slide ${p.page}`;
    if (p.fileType === 'excel') return p.sheet ? `Sheet: ${p.sheet}` : 'Excel';
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

  // Excel: pick the sheet this citation points at (falls back to the first
  // sheet if the citation didn't carry one, or named a sheet that's since
  // been renamed/removed).
  useEffect(() => {
    if (fileType !== 'excel' || !workbook) return;
    const wanted = active?.sheet;
    const match = wanted && workbook.SheetNames.includes(wanted)
      ? wanted
      : workbook.SheetNames[0];
    setActiveSheet(match);
  }, [fileType, workbook, activeIdx, active?.sheet]);

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

  const imageUrl = (isPaginated && active?.document_id)
    ? `${API_BASE_URL}/files/${active.document_id}/pages/${currentPage}/image`
    : (fileType === 'image' && active?.document_id)
      ? `${API_BASE_URL}/files/${active.document_id}/original`
      : null;

  const maxScore = Math.max(...pages.map((p) => p.score ?? 0), 0.001);
  const multiPage = pages.length > 1;

  const sheetRows = (fileType === 'excel' && workbook && activeSheet)
    ? XLSX.utils.sheet_to_json(workbook.Sheets[activeSheet], { header: 1, defval: '' })
    : null;

  return (
    <div className={`flex-shrink-0 flex flex-col bg-slate-900 border-l border-slate-800 transition-all duration-300 overflow-hidden ${sidebarOpen
      ? 'w-[40%] min-w-[40%] max-w-[40%]'
      : 'w-[50%] min-w-[50%] max-w-[50%]'
      }`}>
      {/* Header */}
      <div className="h-14 flex items-center justify-between px-4 border-b border-slate-800/80 bg-slate-950 flex-shrink-0 gap-4">
        <span className="text-xs font-bold text-gray-200 truncate flex-1" title={active?.filename}>
          {active?.filename}
        </span>

        {/* Navigation — pdf/ppt only, they're the only paginated types */}
        {isPaginated && (
          <div className="flex items-center gap-3 select-none flex-shrink-0">
            <button
              onClick={() => handlePageChange(Math.max(1, currentPage - 1))}
              disabled={currentPage <= 1}
              className="p-1 hover:bg-slate-800 text-gray-400 hover:text-white rounded disabled:opacity-20 disabled:hover:bg-transparent transition-all animate-fade-in"
              title="Previous Page"
            >
              <ChevronLeftIcon className="h-4 w-4 stroke-[2.5]" />
            </button>

            <span className="text-xs font-semibold text-gray-200 bg-slate-900 border border-slate-850 px-2.5 py-1 rounded">
              {currentPage} / {totalPages || '...'}
            </span>

            <button
              onClick={() => handlePageChange(Math.min(totalPages || 1, currentPage + 1))}
              disabled={currentPage >= (totalPages || 1)}
              className="p-1 hover:bg-slate-800 text-gray-400 hover:text-white rounded disabled:opacity-20 disabled:hover:bg-transparent transition-all animate-fade-in"
              title="Next Page"
            >
              <ChevronRightIcon className="h-4 w-4 stroke-[2.5]" />
            </button>
          </div>
        )}

        {/* Zoom Controls — pdf/ppt/image only */}
        {(isPaginated || fileType === 'image') && (
          <div className="flex items-center gap-1 bg-slate-900 border border-slate-800 rounded p-0.5 select-none flex-shrink-0">
            <button
              onClick={() => setScale((prev) => Math.max(0.4, prev - 0.2))}
              className="px-1.5 py-0.5 hover:bg-slate-800 text-gray-400 hover:text-white rounded transition-all text-[11px] font-bold"
              title="Zoom Out"
            >
              －
            </button>

            <button
              onClick={() => setScale(1)}
              className="text-[9px] text-gray-300 font-mono w-[30px] text-center select-none hover:text-white hover:bg-slate-800 rounded py-0.5 transition-all"
              title="Reset to 100%"
            >
              {Math.round(scale * 100)}%
            </button>

            <button
              onClick={() => setScale((prev) => Math.min(3.5, prev + 0.2))}
              className="px-1.5 py-0.5 hover:bg-slate-800 text-gray-400 hover:text-white rounded transition-all text-[11px] font-bold"
              title="Zoom In"
            >
              ＋
            </button>

            <button
              onClick={handleToggleZoom}
              className="ml-1 text-[9px] font-semibold text-gray-400 hover:text-white bg-slate-800 hover:bg-slate-700 px-1.5 py-0.5 rounded transition-all border border-slate-700"
              title="Reset Zoom to 100%"
            >
              Reset
            </button>
          </div>
        )}

        {/* Close Button */}
        <button
          onClick={onClose}
          className="p-1 hover:bg-slate-800 text-gray-400 hover:text-white rounded transition-all flex-shrink-0"
          title="Close viewer"
        >
          <XMarkIcon className="h-4 w-4 stroke-[2.5]" />
        </button>
      </div>

      {/* Document/Page Badges — cited pages shown below header */}
      {multiPage && (
        <div className="px-3 py-2 bg-slate-950 border-b border-slate-800 flex flex-wrap gap-1.5 select-none">
          {pages.map((p, i) => {
            const confidence = maxScore > 0 ? ((p.score ?? 0) / maxScore) : 0;
            const isActive = i === activeIdx;
            return (
              <button
                key={i}
                onClick={() => handleSelectSource(i)}
                className={`flex flex-col items-center rounded-lg px-2.5 py-1.5 text-[10px] font-semibold transition-all border ${isActive
                  ? 'bg-blue-600/20 border-blue-500 text-blue-300'
                  : 'bg-slate-900 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200'
                  }`}
                title={`${p.filename} — ${((p.score ?? 0) * 100).toFixed(0)}% match`}
              >
                <span>{getBadgeLabel(p)}</span>
                <div className="mt-1 w-8 h-0.5 rounded-full bg-slate-700 overflow-hidden">
                  <div
                    className={`h-full rounded-full ${isActive ? 'bg-blue-400' : 'bg-slate-500'}`}
                    style={{ width: `${Math.round(confidence * 100)}%` }}
                  />
                </div>
              </button>
            );
          })}
        </div>
      )}

      {/* Sheet tabs — excel only */}
      {fileType === 'excel' && workbook && workbook.SheetNames.length > 1 && (
        <div className="px-3 py-2 bg-slate-950 border-b border-slate-800 flex flex-wrap gap-1.5 select-none">
          {workbook.SheetNames.map((name) => (
            <button
              key={name}
              onClick={() => setActiveSheet(name)}
              className={`text-[11px] font-semibold px-2.5 py-1 rounded-lg border transition-all ${activeSheet === name
                ? 'bg-blue-600/20 border-blue-500 text-blue-300'
                : 'bg-slate-900 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200'
                }`}
            >
              {name}
            </button>
          ))}
        </div>
      )}

      {/* Content area */}
      {fileType === 'docx' ? (
        <div ref={docxContainerRef} className="flex-1 overflow-auto bg-white p-6">
          {docxLoading && (
            <div className="flex flex-col items-center justify-center text-slate-400 text-xs py-24 gap-2">
              <ArrowPathIcon className="h-5 w-5 animate-spin text-blue-500" />
              <span>Loading document...</span>
            </div>
          )}
          {docxError && (
            <div className="flex flex-col items-center justify-center py-24 text-slate-500 text-xs text-center gap-2">
              <DocumentIcon className="h-8 w-8 opacity-40" />
              <span>Could not load this document.</span>
            </div>
          )}
          {docxHtml && (
            <div
              className="docx-content text-slate-900 text-sm leading-relaxed max-w-2xl mx-auto"
              dangerouslySetInnerHTML={{ __html: docxHtml }}
            />
          )}
        </div>
      ) : fileType === 'excel' ? (
        <div className="flex-1 overflow-auto bg-white p-4">
          {excelLoading && (
            <div className="flex flex-col items-center justify-center text-slate-400 text-xs py-24 gap-2">
              <ArrowPathIcon className="h-5 w-5 animate-spin text-blue-500" />
              <span>Loading spreadsheet...</span>
            </div>
          )}
          {excelError && (
            <div className="flex flex-col items-center justify-center py-24 text-slate-500 text-xs text-center gap-2">
              <DocumentIcon className="h-8 w-8 opacity-40" />
              <span>Could not load this spreadsheet.</span>
            </div>
          )}
          {sheetRows && (
            <table className="w-full text-xs border-collapse">
              <tbody>
                {sheetRows.map((row, r) => (
                  <tr key={r} className={r === 0 ? 'bg-slate-100 font-semibold' : 'odd:bg-slate-50'}>
                    {row.map((cell, c) => (
                      <td key={c} className="border border-slate-200 px-2 py-1 text-slate-800 whitespace-nowrap">
                        {String(cell)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      ) : (
        <div
          ref={containerRef}
          className="flex-1 overflow-auto bg-slate-950 p-4"
        >
          {imageUrl && (
            <div
              style={{
                /* Scale < 1 → shrink & center via auto margin.
                   Scale > 1 → grow wider than panel → scrollbars appear. */
                width: `${Math.round(scale * 100)}%`,
                marginLeft: scale <= 1 ? 'auto' : '0',
                marginRight: scale <= 1 ? 'auto' : '0',
                transition: 'width 150ms ease-out',
                position: 'relative',
              }}
            >
              {!imgLoaded && !imgError && (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 text-xs py-24 gap-2 bg-slate-900/40 rounded">
                  <ArrowPathIcon className="h-5 w-5 animate-spin text-blue-500" />
                  <span>{isPaginated ? `Rendering Page ${currentPage}...` : 'Loading image...'}</span>
                </div>
              )}
              {imgError && (
                <div className="flex flex-col items-center justify-center py-24 text-slate-500 text-xs text-center gap-2 bg-slate-900/40 rounded border border-slate-800">
                  <DocumentIcon className="h-8 w-8 opacity-40 text-slate-600" />
                  <span>Page image not available.</span>
                </div>
              )}
              <img
                src={imageUrl}
                alt={isPaginated ? `Page ${currentPage}` : active?.filename}
                onLoad={() => setImgLoaded(true)}
                onError={() => { setImgError(true); setImgLoaded(true); }}
                className={`w-full rounded bg-white shadow-xl transition-opacity duration-300 ${imgLoaded && !imgError ? 'opacity-100' : 'opacity-0'
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
const parseSources = (toolCalls) => {
  const sources = [];
  for (const call of toolCalls || []) {
    if (call.name !== 'search_documents' || typeof call.result !== 'string') continue;
    try {
      const parsed = JSON.parse(call.result);
      if (Array.isArray(parsed.sources)) {
        const mapped = parsed.sources.map(s => {
          if (s.page == null && s.slide != null) {
            return { ...s, page: s.slide };
          }
          if (s.page == null && s.sheet != null) {
            return { ...s, page: s.sheet };
          }
          return s;
        });
        sources.push(...mapped);
      }
    } catch {
      // not JSON (e.g. a blocked/error string) — nothing to show
    }
  }
  return sources;
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

const MessageRow = ({ msg, onApprove, onDecline, onClarify, loading, onViewPages }) => {
  const isUser = msg.role === 'user';
  const sources = !isUser ? parseSources(msg.toolCalls) : [];
  const listedDocuments = !isUser ? parseListedDocuments(msg.toolCalls) : {
    documents: [],
    note: "",
    totalCount: 0,
    returnedCount: 0,
  };
  const [isSearchExpanded, setIsSearchExpanded] = useState(false);
  const [isListExpanded, setIsListExpanded] = useState(false);

  // Sources for the "View Source" button — filtered + sorted by score.
  // Covers PDF pages, docx (whole doc, snippet-matched), and images (whole file).
  const pageSources = buildViewableSources(sources);
  const hasPageSources = pageSources.length > 0;

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
              {!msg.content && (
                <div className="flex items-center text-slate-400 text-sm py-1">
                  <span className="font-medium">
                    {msg.isIngesting
                      ? 'Ingesting'
                      : 'Thinking'}
                    <span className="animate-dots"></span>
                  </span>
                </div>
              )}
              {msg.content && (
                <div
                  className={`prose prose-invert prose-sm max-w-none leading-relaxed w-full overflow-x-auto ${msg.isError ? 'text-red-300' : 'text-gray-100'
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
                          <code className={className} {...props}>
                            {children}
                          </code>
                        );
                      }
                    }}
                  >
                    {renderMathInMarkdown(msg.content)}
                  </ReactMarkdown>
                </div>
              )}

              {/* Tool calls this turn (reads that ran, or a blocked write) */}
              {msg.toolCalls && msg.toolCalls.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1.5">
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
                              setIsListExpanded(false); // collapse other
                            }}
                            className="inline-flex items-center gap-1.5 text-xs text-blue-300 bg-blue-500/10 border border-blue-500/20 rounded px-2.5 py-1.5 hover:bg-blue-500/20 transition-all cursor-pointer font-medium"
                          >
                            <WrenchScrewdriverIcon className="h-3 w-3" />
                            <span className="font-mono">{group.name}</span>
                            {group.count > 1 && (
                              <span className="ml-1 text-[10px] bg-blue-500/25 px-1.5 py-0.5 rounded-full font-sans font-medium">
                                {group.count}
                              </span>
                            )}
                            {isSearchExpanded ? (
                              <ChevronUpIcon className="h-3.5 w-3.5 text-blue-400" />
                            ) : (
                              <ChevronDownIcon className="h-3.5 w-3.5 text-blue-400" />
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
                              setIsSearchExpanded(false); // collapse other
                            }}
                            className="inline-flex items-center gap-1.5 text-xs text-blue-300 bg-blue-500/10 border border-blue-500/20 rounded px-2.5 py-1.5 hover:bg-blue-500/20 transition-all cursor-pointer font-medium"
                            >
                            <WrenchScrewdriverIcon className="h-3 w-3" />
                            <span className="font-mono">{group.name}</span>
                            {listCount > 0 ? (
                              <span className="ml-1 text-[10px] bg-blue-500/25 px-1.5 py-0.5 rounded-full font-sans font-medium">
                                {listCount}
                              </span>
                            ) : group.count > 1 && (
                              <span className="ml-1 text-[10px] bg-blue-500/25 px-1.5 py-0.5 rounded-full font-sans font-medium">
                                {group.count}
                              </span>
                            )}
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
                          <span className="font-mono">{group.name}</span>
                          {group.count > 1 && (
                            <span className="text-[10px] text-blue-300 bg-blue-500/10 border border-blue-500/10 px-1.5 py-0.5 rounded font-sans font-medium">
                              x{group.count}
                            </span>
                          )}
                        </span>
                      );
                    });
                  })()}
                </div>
              )}

              {/* Inline citation list (expanded from list_documents) */}
              {isListExpanded && (
                <div className="mt-3 space-y-1.5 max-w-md bg-slate-900/40 border border-slate-800 rounded-xl p-3">
                  <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2">Listed Documents</p>
                  {(() => {
                    const docs = listedDocuments.documents || [];
                    if (docs.length === 0) {
                      return <p className="text-xs text-gray-500 italic p-1">No documents were returned by list_documents.</p>;
                    }
                    return docs.map((doc, idx) => (
                      <div key={idx} className="flex items-center gap-2.5 text-xs text-gray-300 bg-slate-800/40 border border-slate-700/60 rounded px-3 py-2">
                        <DocumentIcon className="h-4 w-4 text-blue-400 flex-shrink-0" />
                        <span className="font-medium truncate">{doc.filename || doc.document_id || 'Untitled document'}</span>
                      </div>
                    ));
                  })()}
                  {listedDocuments.note && (
                    <p className="text-[10px] text-gray-500 mt-2">{listedDocuments.note}</p>
                  )}
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

              {/* View Source Pages button — always visible when page sources exist */}
              {hasPageSources && (
                <div className="mt-3 pt-2.5 border-t border-slate-800/60">
                  <button
                    onClick={() => onViewPages(pageSources)}
                    className="inline-flex items-center gap-1.5 text-xs text-blue-400 hover:text-blue-300 bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/20 hover:border-blue-500/40 rounded-lg px-3 py-1.5 transition-all font-medium"
                  >
                    <DocumentIcon className="h-3.5 w-3.5" />
                    View Source{pageSources.length > 1 ? `s (${pageSources.length})` : ''}
                  </button>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatPage;
