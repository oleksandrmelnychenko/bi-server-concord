import React from 'react';
import type { ApiStatus, ChartType } from '../types';

interface SidebarProps {
  apiStatus: ApiStatus;
  onNewChat: () => void;
  onQuickQuery: (query: string) => void;
  onShowChart: (chartType: ChartType) => void;
}

const quickQueries = [
  { label: 'Продажі по роках', query: 'продажі по роках' },
  { label: 'Найкращий клієнт', query: 'найкращий клієнт хто купив найбільше' },
  { label: 'Найпопулярніший товар', query: 'який товар продали найбільше' },
  { label: 'Статистика боргів', query: 'борги' },
  { label: 'Клієнти з Києва', query: 'клієнти з Києва' },
];

const chartButtons: Array<{ label: string; type: ChartType; icon: JSX.Element }> = [
  {
    label: 'Продажі (графік)',
    type: 'sales_yearly',
    icon: (
      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M3 3v18h18" />
        <path d="M18 17V9" />
        <path d="M13 17V5" />
        <path d="M8 17v-3" />
      </svg>
    ),
  },
  {
    label: 'Топ товарів (графік)',
    type: 'top_products',
    icon: (
      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="3" y="3" width="18" height="18" rx="2" />
        <path d="M3 9h18" />
        <path d="M9 21V9" />
      </svg>
    ),
  },
  {
    label: 'Борги (графік)',
    type: 'debts',
    icon: (
      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10" />
        <path d="M12 2a10 10 0 0 1 10 10" />
      </svg>
    ),
  },
];

export const Sidebar: React.FC<SidebarProps> = ({
  apiStatus,
  onNewChat,
  onQuickQuery,
  onShowChart,
}) => {
  return (
    <aside className="w-72 bg-surface-secondary flex flex-col border-r border-surface-border">
      {/* Header */}
      <div className="p-5 border-b border-surface-border">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-content-primary flex items-center justify-center">
            <svg className="w-6 h-6 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
          </div>
          <div>
            <h1 className="text-xl font-bold text-content-primary">
              Concord AI
            </h1>
            <p className="text-xs text-content-muted">Business Intelligence</p>
          </div>
        </div>
      </div>

      {/* New Chat Button */}
      <div className="p-4">
        <button
          onClick={onNewChat}
          className="w-full bg-content-primary text-white px-4 py-3 rounded-full font-medium
                   flex items-center justify-center gap-2 hover:bg-gray-800 transition-colors"
        >
          <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          Новий чат
        </button>
      </div>

      {/* Quick Queries Section */}
      <div className="flex-1 overflow-y-auto px-4">
        <div className="mb-6">
          <h3 className="text-xs font-semibold text-content-muted uppercase tracking-wider px-3 mb-2">
            Швидкі запити
          </h3>
          <div className="space-y-1">
            {quickQueries.map((item) => (
              <button
                key={item.query}
                onClick={() => onQuickQuery(item.query)}
                className="w-full text-left px-3 py-2.5 rounded-xl text-sm text-content-secondary
                         hover:bg-surface-hover transition-colors duration-200"
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>

        {/* Reports Section */}
        <div className="mb-6">
          <h3 className="text-xs font-semibold text-content-muted uppercase tracking-wider px-3 mb-2">
            Звіти
          </h3>
          <div className="space-y-1">
            {chartButtons.map((item) => (
              <button
                key={item.type}
                onClick={() => onShowChart(item.type)}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm text-content-secondary
                         hover:bg-surface-hover transition-colors duration-200"
              >
                <span className="text-content-muted">
                  {item.icon}
                </span>
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Footer with Status */}
      <div className="p-4 border-t border-surface-border">
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${
            apiStatus.online
              ? 'bg-green-500 animate-pulse-soft'
              : 'bg-red-500'
          }`} />
          <span className="text-xs text-content-muted">
            {apiStatus.online ? 'API Online' : 'API Offline'}
          </span>
        </div>
        <div className="text-xs text-content-muted mt-1">
          Документів: {apiStatus.documents?.toLocaleString() ?? '--'}
        </div>
      </div>
    </aside>
  );
};
