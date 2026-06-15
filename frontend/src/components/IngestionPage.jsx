import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { uploadFile, getFiles, deleteFile, healthCheck } from '../api';
import {
  CloudArrowUpIcon,
  DocumentIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
  TrashIcon,
  ChatBubbleLeftIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';

const IngestionPage = () => {
  const navigate = useNavigate();
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [serverConnected, setServerConnected] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({});

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
      setFiles(res.data || []);
      setError(null);
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
        const res = await uploadFile(file);
        setFiles((prev) => [res.data, ...prev]);
        setUploadProgress((prev) => {
          const newProgress = { ...prev };
          delete newProgress[file.name];
          return newProgress;
        });
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

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['.pptx'],
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp'],
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

  const getStatusIcon = (status) => {
    if (status === 'Ready') {
      return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
    } else {
      return <ExclamationCircleIcon className="h-5 w-5 text-yellow-500" />;
    }
  };

  const getStatusColor = (status) => {
    if (status === 'Ready') {
      return 'bg-green-500/20 text-green-400';
    } else {
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
            <div className="text-right">
              <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm ${serverConnected ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                <div className={`w-2 h-2 rounded-full ${serverConnected ? 'bg-green-400' : 'bg-red-400'}`} />
                {serverConnected ? 'Backend Connected' : 'Backend Offline'}
              </div>
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
            Supported: PDF, Excel, PowerPoint, Images (PNG, JPG, GIF, BMP)
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

        {/* Pipeline Status */}
        <div className="mb-8">
          <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">Processing Pipeline</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2">
            {[
              { name: 'Categorize', icon: '📂' },
              { name: 'Page Profile', icon: '📄' },
              { name: 'Extraction', icon: '📊' },
              { name: 'Vision', icon: '👁️' },
              { name: 'Chunk', icon: '✂️' },
              { name: 'Embed', icon: '🔗' },
            ].map((step, idx) => (
              <div key={step.name} className="relative">
                <div className={`flex flex-col items-center p-4 rounded-lg border transition-all ${
                  idx === 0 
                    ? 'border-blue-500/50 bg-blue-500/10 shadow-lg shadow-blue-500/20' 
                    : idx === 1 
                      ? 'border-green-500/50 bg-green-500/10' 
                      : 'border-slate-700 bg-slate-800/30'
                }`}>
                  <span className="text-2xl mb-2">{step.icon}</span>
                  <span className="text-xs text-gray-300 text-center font-medium">{step.name}</span>
                  {idx < 2 && (
                    <div className={`absolute top-1/2 -right-2.5 w-4 h-4 rounded-full ${idx === 0 ? 'bg-blue-500' : 'bg-green-500'}`} />
                  )}
                </div>
              </div>
            ))}
          </div>
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
                        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mt-4 text-xs">
                          <div className="bg-slate-900/50 p-3 rounded border border-slate-600">
                            <span className="text-gray-500 text-xs uppercase tracking-wider font-semibold">Route</span>
                            <p className="text-blue-400 font-mono font-medium mt-1">{file.route || 'N/A'}</p>
                          </div>
                          <div className="bg-slate-900/50 p-3 rounded border border-slate-600">
                            <span className="text-gray-500 text-xs uppercase tracking-wider font-semibold">Type</span>
                            <p className="text-gray-300 mt-1">{file.document_type || 'Unknown'}</p>
                          </div>
                          <div className="bg-slate-900/50 p-3 rounded border border-slate-600">
                            <span className="text-gray-500 text-xs uppercase tracking-wider font-semibold">File Type</span>
                            <p className="text-gray-300 capitalize mt-1">{file.file_type || 'N/A'}</p>
                          </div>
                          <div className="bg-slate-900/50 p-3 rounded border border-slate-600">
                            <span className="text-gray-500 text-xs uppercase tracking-wider font-semibold">Industry</span>
                            <p className="text-gray-300 capitalize mt-1">{file.industry || 'N/A'}</p>
                          </div>
                          <div className="bg-slate-900/50 p-3 rounded border border-slate-600">
                            <span className="text-gray-500 text-xs uppercase tracking-wider font-semibold">Confidence</span>
                            <div className="mt-1 flex items-center gap-2">
                              <div className="flex-1 bg-slate-700 rounded-full h-1.5">
                                <div
                                  className={`h-1.5 rounded-full transition-all ${file.confidence > 0.8 ? 'bg-green-500' : file.confidence > 0.6 ? 'bg-yellow-500' : 'bg-red-500'}`}
                                  style={{ width: `${file.confidence * 100}%` }}
                                />
                              </div>
                              <span className={`font-medium ${file.confidence > 0.8 ? 'text-green-400' : file.confidence > 0.6 ? 'text-yellow-400' : 'text-red-400'}`}>
                                {(file.confidence * 100).toFixed(0)}%
                              </span>
                            </div>
                          </div>
                        </div>
                        {file.reasoning && (
                          <div className="mt-4 p-3 bg-slate-900/50 border border-slate-600 rounded-lg">
                            <span className="text-gray-500 text-xs uppercase tracking-wider font-semibold">Reasoning</span>
                            <p className="text-gray-300 text-sm mt-1 whitespace-pre-wrap">{file.reasoning}</p>
                          </div>
                        )}
                        {file.errors && file.errors.length > 0 && (
                          <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-xs space-y-1">
                            <p className="font-semibold">⚠️ Issues found:</p>
                            {file.errors.map((err, idx) => (
                              <p key={idx} className="flex items-start gap-2">
                                <span className="mt-1 text-red-500">•</span>
                                <span>{err}</span>
                              </p>
                            ))}
                          </div>
                        )}
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