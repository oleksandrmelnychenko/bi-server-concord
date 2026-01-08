import React, { useEffect, useState } from 'react';
import { fetchManagers, type ManagerItem } from '../services/api';

interface ManagerListPanelProps {
  open: boolean;
  sidebarOpen?: boolean;
  storagesPanelOpen?: boolean;
  onManagerSelect?: (manager: ManagerItem) => void;
  onClose: () => void;
}

const translations = {
  title: 'Менеджери',
  subtitle: 'Список активних менеджерів',
  loading: 'Завантаження...',
  error: 'Помилка завантаження',
  empty: 'Немає менеджерів',
  count: 'Всього:',
};

export const ManagerListPanel: React.FC<ManagerListPanelProps> = ({
  open,
  sidebarOpen = false,
  storagesPanelOpen = false,
  onManagerSelect,
  onClose,
}) => {
  const [managers, setManagers] = useState<ManagerItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);

  useEffect(() => {
    if (!open) return;

    const controller = new AbortController();
    setLoading(true);
    setError(null);

    fetchManagers(controller.signal)
      .then((data) => {
        if (data.success) {
          setManagers(data.managers);
        } else {
          setError(data.error || translations.error);
        }
      })
      .catch((err) => {
        if (err.name !== 'AbortError') {
          setError(err.message || translations.error);
        }
      })
      .finally(() => setLoading(false));

    return () => controller.abort();
  }, [open]);

  if (!open) return null;

  // Position to the right of sidebar: collapsed = 64px + 24px gap, expanded = 288px + 24px gap
  const leftPosition = sidebarOpen ? 'left-[312px]' : 'left-[88px]';
  // When storage panel is also open, position below it
  const topPosition = storagesPanelOpen ? 'top-[380px]' : 'top-6';
  const maxHeight = storagesPanelOpen ? 'max-h-[calc(100vh-420px)]' : 'max-h-[calc(100vh-100px)]';

  const handleManagerClick = (manager: ManagerItem) => {
    setSelectedId(manager.id);
    onManagerSelect?.(manager);
  };

  return (
    <div className={`fixed ${leftPosition} ${topPosition} w-[280px] ${maxHeight} z-50 flex flex-col bg-white rounded-2xl shadow-2xl border border-slate-200/60 overflow-hidden transition-all duration-300`}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100 bg-gradient-to-r from-violet-50 to-purple-50">
        <div className="flex items-center gap-2">
          <svg className="w-5 h-5 text-violet-500" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
          </svg>
          <div>
            <h2 className="text-sm font-semibold text-slate-800">{translations.title}</h2>
            <p className="text-xs text-slate-500">{translations.subtitle}</p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
          title="Закрити"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto">
        {loading && (
          <div className="flex items-center justify-center h-32">
            <div className="flex flex-col items-center gap-2">
              <div className="w-6 h-6 border-2 border-violet-500 border-t-transparent rounded-full animate-spin" />
              <span className="text-xs text-slate-500">{translations.loading}</span>
            </div>
          </div>
        )}

        {error && (
          <div className="m-3 p-3 bg-red-50 border border-red-200 rounded-xl text-red-600 text-xs">
            {error}
          </div>
        )}

        {!loading && !error && managers.length === 0 && (
          <div className="flex items-center justify-center h-32 text-slate-400 text-sm">
            {translations.empty}
          </div>
        )}

        {!loading && !error && managers.length > 0 && (
          <div className="py-1">
            {managers.map((manager, index) => (
              <button
                key={manager.id}
                onClick={() => handleManagerClick(manager)}
                className={`w-full text-left px-4 py-2.5 flex items-center gap-3 hover:bg-slate-50 transition-colors border-b border-slate-100/50 last:border-b-0 ${
                  selectedId === manager.id ? 'bg-violet-50' : ''
                }`}
              >
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium ${
                  selectedId === manager.id
                    ? 'bg-violet-500 text-white'
                    : 'bg-gradient-to-br from-violet-100 to-purple-100 text-violet-600'
                }`}>
                  {manager.name.charAt(0).toUpperCase()}
                </div>
                <div className="flex-1 min-w-0">
                  <p className={`text-sm truncate ${
                    selectedId === manager.id ? 'text-violet-700 font-medium' : 'text-slate-700'
                  }`}>
                    {manager.name}
                  </p>
                  <p className="text-xs text-slate-400">ID: {manager.id}</p>
                </div>
                {selectedId === manager.id && (
                  <svg className="w-4 h-4 text-violet-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                )}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Footer with count */}
      {!loading && managers.length > 0 && (
        <div className="px-4 py-2 border-t border-slate-100 bg-slate-50/50">
          <span className="text-xs text-slate-400">{translations.count} {managers.length}</span>
        </div>
      )}
    </div>
  );
};

export default ManagerListPanel;
