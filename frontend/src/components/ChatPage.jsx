import React, { useState, useRef, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { sendChat, getFile } from '../api';
import {
  PaperAirplaneIcon,
  SparklesIcon,
  DocumentIcon,
  ExclamationCircleIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';

const ChatPage = () => {
  const [searchParams] = useSearchParams();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentFile, setCurrentFile] = useState(null);
  const messagesEndRef = useRef(null);

  const fileId = searchParams.get('fileId');

  // Load file details if fileId is provided
  useEffect(() => {
    if (fileId) {
      loadFileDetails();
    }
  }, [fileId]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const loadFileDetails = async () => {
    try {
      const res = await getFile(fileId);
      setCurrentFile(res.data);
      setError(null);
    } catch (err) {
      console.error('Failed to load file:', err);
      setError('Could not load file details');
    }
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsg = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);
    setError(null);

    try {
      const res = await sendChat(input, fileId);
      const assistantMsg = {
        role: 'assistant',
        content: res.data.answer || res.data,
        sources: res.data.sources || [],
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      console.error('Chat error:', err);
      const errorMsg = {
        role: 'assistant',
        content: `Error: ${err.message || 'Failed to get response from chat service. Make sure the backend is running.'}`,
        isError: true,
      };
      setMessages((prev) => [...prev, errorMsg]);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-gray-100">
      {/* Header */}
      <div className="fixed top-0 left-0 right-0 bg-slate-900/50 border-b border-slate-700 z-20">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <SparklesIcon className="h-6 w-6 text-blue-400" />
              <div>
                <h1 className="text-xl font-bold text-white">Document Q&A</h1>
                {currentFile && (
                  <p className="text-xs text-gray-400 mt-1">
                    <DocumentIcon className="h-3 w-3 inline mr-1" />
                    {currentFile.filename}
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col mt-24 mb-28">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4 max-w-4xl mx-auto w-full">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <SparklesIcon className="h-12 w-12 text-slate-600 mb-4" />
              <p className="text-gray-400 text-lg">Ask a question about your documents</p>
              <p className="text-gray-500 text-sm mt-2">
                {currentFile ? `Asking about: ${currentFile.filename}` : 'Select a document to get started'}
              </p>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-2xl rounded-lg p-4 ${
                  msg.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : msg.isError
                      ? 'bg-red-500/20 border border-red-500/30 text-red-200'
                      : 'bg-slate-800/60 border border-slate-700 text-gray-100'
                }`}
              >
                <p className="leading-relaxed">{msg.content}</p>
                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-slate-600 space-y-2">
                    <p className="text-xs font-semibold text-gray-300">Sources:</p>
                    {msg.sources.map((source, i) => (
                      <div key={i} className="text-xs text-gray-400 bg-slate-900/50 p-2 rounded">
                        <p className="font-medium text-gray-300">
                          {source.filename} {source.page && `• Page ${source.page}`}
                        </p>
                        {source.snippet && (
                          <p className="mt-1 italic text-gray-400">"{source.snippet}"</p>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="bg-slate-800/60 border border-slate-700 rounded-lg p-4 flex items-center gap-2">
                <ArrowPathIcon className="h-4 w-4 animate-spin text-blue-400" />
                <span className="text-gray-400">Thinking...</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Error Message */}
        {error && (
          <div className="mx-4 mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 flex items-start gap-2">
            <ExclamationCircleIcon className="h-5 w-5 flex-shrink-0 mt-0.5" />
            <p className="text-sm">{error}</p>
          </div>
        )}

        {/* Input Area */}
        <div className="fixed bottom-0 left-0 right-0 bg-slate-900/90 backdrop-blur border-t border-slate-700 p-4">
          <div className="max-w-4xl mx-auto">
            {currentFile && (
              <p className="text-xs text-gray-400 mb-2 flex items-center gap-2">
                <DocumentIcon className="h-3 w-3" />
                Context: {currentFile.filename}
              </p>
            )}
            <div className="flex gap-3">
              <input
                type="text"
                className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder="Ask a question about your documents... (Shift+Enter for new line)"
                disabled={loading}
              />
              <button
                onClick={handleSend}
                disabled={loading || !input.trim()}
                className={`px-4 py-3 rounded-lg font-medium flex items-center gap-2 transition-all ${
                  loading || !input.trim()
                    ? 'bg-slate-700 text-gray-500 cursor-not-allowed'
                    : 'bg-blue-600 text-white hover:bg-blue-700 active:scale-95'
                }`}
              >
                <PaperAirplaneIcon className="h-5 w-5" />
                Send
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Source Snippet Panel (Right Sidebar) */}
      <div className="hidden lg:flex lg:flex-col w-80 bg-slate-800/30 border-l border-slate-700 mt-24">
        <div className="p-4 border-b border-slate-700">
          <h3 className="font-semibold text-white flex items-center gap-2">
            <DocumentIcon className="h-5 w-5 text-blue-400" />
            Source Snippets
          </h3>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {messages.length > 0 && messages.some((m) => m.sources && m.sources.length > 0) ? (
            messages
              .filter((m) => m.sources && m.sources.length > 0)
              .flatMap((m) => m.sources)
              .map((source, idx) => (
                <div key={idx} className="p-3 bg-slate-800/50 rounded-lg border border-slate-700 hover:bg-slate-800/70 transition-colors">
                  <p className="text-xs font-semibold text-blue-400 truncate">
                    {source.filename}
                  </p>
                  {source.page && (
                    <p className="text-xs text-gray-500 mt-1">Page {source.page}</p>
                  )}
                  {source.snippet && (
                    <p className="text-xs text-gray-300 mt-2 italic line-clamp-3">
                      "{source.snippet}"
                    </p>
                  )}
                </div>
              ))
          ) : (
            <div className="text-center text-gray-500 text-sm py-8">
              <DocumentIcon className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p>No sources yet</p>
              <p className="text-xs mt-1">Ask a question to see document sources</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatPage;
