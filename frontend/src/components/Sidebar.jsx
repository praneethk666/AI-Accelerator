import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  DocumentArrowUpIcon,
  ChatBubbleLeftIcon,
  Cog6ToothIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline';

const Sidebar = ({ currentPage = 'ingestion', files = [], currentFileId = null, onSelectFile = () => {} }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const navItems = [
    {
      id: 'ingestion',
      label: 'Ingestion',
      icon: DocumentArrowUpIcon,
      onClick: () => navigate('/'),
      active: currentPage === 'ingestion' || location.pathname === '/',
    },
    {
      id: 'chat',
      label: 'Chat',
      icon: ChatBubbleLeftIcon,
      onClick: () => navigate('/chat'),
      active: currentPage === 'chat' || location.pathname === '/chat',
    },
  ];

  return (
    <div className="w-64 bg-slate-800/50 border-r border-slate-700 flex flex-col h-full">
      {/* Logo/Header */}
      <div className="p-4 border-b border-slate-700">
        <div className="flex items-center gap-2 mb-1">
          <SparklesIcon className="h-6 w-6 text-blue-400" />
          <h1 className="text-lg font-bold text-white">DocAI</h1>
        </div>
        <p className="text-xs text-gray-500">Document Intelligence</p>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-6 space-y-2 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              onClick={item.onClick}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors text-left ${
                item.active
                  ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30'
                  : 'text-gray-400 hover:text-gray-300 hover:bg-slate-700/50'
              }`}
            >
              <Icon className="h-5 w-5 flex-shrink-0" />
              <span className="text-sm font-medium">{item.label}</span>
            </button>
          );
        })}

        {/* Divider */}
        <div className="my-4 h-px bg-slate-700" />

        {/* Files Section */}
        {currentPage === 'chat' && files.length > 0 && (
          <div>
            <p className="text-xs uppercase font-semibold text-gray-500 px-4 mb-3">Your Documents</p>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {files.map((file) => (
                <button
                  key={file.id}
                  onClick={() => onSelectFile(file.id)}
                  className={`w-full text-left px-4 py-2 rounded-lg transition-colors text-xs truncate ${
                    currentFileId === file.id
                      ? 'bg-blue-600/30 text-blue-300 border border-blue-500/30'
                      : 'text-gray-400 hover:text-gray-300 hover:bg-slate-700/50'
                  }`}
                  title={file.filename}
                >
                  📄 {file.filename}
                </button>
              ))}
            </div>
          </div>
        )}
      </nav>

      {/* Footer */}
      <div className="border-t border-slate-700 p-3">
        <button className="w-full flex items-center gap-2 px-4 py-2 rounded-lg text-gray-400 hover:text-gray-300 hover:bg-slate-700/50 transition-colors text-sm">
          <Cog6ToothIcon className="h-4 w-4" />
          Settings
        </button>
      </div>
    </div>
  );
};

export default Sidebar;
