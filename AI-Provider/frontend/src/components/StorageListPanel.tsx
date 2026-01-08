import React, { useEffect, useState } from 'react';
import { fetchStorages, type StorageItem } from '../services/api';

interface StorageListPanelProps {
  open: boolean;
  sidebarOpen?: boolean;
  onStorageSelect?: (storage: StorageItem) => void;
  onClose: () => void;
}

const translations = {
  title: 'Склади',
  subtitle: 'Список всіх складів',
  loading: 'Завантаження...',
  error: 'Помилка завантаження',
  empty: 'Немає складів',
  count: 'Всього:',
};

export const StorageListPanel: React.FC<StorageListPanelProps> = ({
  open,
  sidebarOpen = false,
  onStorageSelect,
  onClose,
}) => {
  const [storages, setStorages] = useState<StorageItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);

  useEffect(() => {
    if (!open) return;

    const controller = new AbortController();
    setLoading(true);
    setError(null);

    fetchStorages(controller.signal)
      .then((data) => {
        if (data.success) {
          setStorages(data.storages);
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

  const handleStorageClick = (storage: StorageItem) => {
    setSelectedId(storage.id);
    onStorageSelect?.(storage);
  };

  return (
    <div className={`fixed ${leftPosition} top-6 w-[280px] max-h-[calc(100vh-100px)] z-50 flex flex-col bg-white rounded-2xl shadow-2xl border border-slate-200/60 overflow-hidden transition-all duration-300`}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100 bg-gradient-to-r from-indigo-50 to-blue-50">
        <div className="flex items-center gap-2">
          <svg className="w-5 h-5 text-indigo-500" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
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
              <div className="w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
              <span className="text-xs text-slate-500">{translations.loading}</span>
            </div>
          </div>
        )}

        {error && (
          <div className="m-3 p-3 bg-red-50 border border-red-200 rounded-xl text-red-600 text-xs">
            {error}
          </div>
        )}

        {!loading && !error && storages.length === 0 && (
          <div className="flex items-center justify-center h-32 text-slate-400 text-sm">
            {translations.empty}
          </div>
        )}

        {!loading && !error && storages.length > 0 && (
          <div className="py-1">
            {storages.map((storage, index) => (
              <button
                key={storage.id}
                onClick={() => handleStorageClick(storage)}
                className={`w-full text-left px-4 py-2.5 flex items-center gap-3 hover:bg-slate-50 transition-colors border-b border-slate-100/50 last:border-b-0 ${
                  selectedId === storage.id ? 'bg-indigo-50' : ''
                }`}
              >
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-xs font-medium ${
                  selectedId === storage.id
                    ? 'bg-indigo-500 text-white'
                    : 'bg-slate-100 text-slate-500'
                }`}>
                  {index + 1}
                </div>
                <div className="flex-1 min-w-0">
                  <p className={`text-sm truncate ${
                    selectedId === storage.id ? 'text-indigo-700 font-medium' : 'text-slate-700'
                  }`}>
                    {storage.name}
                  </p>
                  <p className="text-xs text-slate-400">ID: {storage.id}</p>
                </div>
                {selectedId === storage.id && (
                  <svg className="w-4 h-4 text-indigo-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                )}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Footer with count */}
      {!loading && storages.length > 0 && (
        <div className="px-4 py-2 border-t border-slate-100 bg-slate-50/50">
          <span className="text-xs text-slate-400">{translations.count} {storages.length}</span>
        </div>
      )}
    </div>
  );
};

export default StorageListPanel;
