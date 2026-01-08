import React from 'react';
import type { ClientScoreData } from '../services/api';

interface ClientScorePanelProps {
  open: boolean;
  loading: boolean;
  error?: string | null;
  clientId?: number;
  clientName?: string | null;
  scoreData?: ClientScoreData | null;
  sidebarOpen?: boolean;
  onClose: () => void;
}

const translations = {
  title: 'Платіжний рейтинг',
  outOf100: 'із 100 балів',
  avgDays: 'Сер. днів',
  onTime: 'Вчасно',
  unpaidCount: 'Неоплачено',
  debtAmount: 'Борг',
  trendTitle: 'Динаміка рейтингу',
  paidOrders: 'Оплачених замовлень',
  paidAmount: 'Сума оплачено',
  oldestDebt: 'Найстаріший борг',
  days: 'днів',
  loading: 'Завантаження...',
  noData: 'Немає даних',
  coldStartTitle: 'Немає даних',
  coldStartDesc: 'Клієнт не має замовлень за останні 12 місяців',
};

const gradeColors: Record<string, { bg: string; text: string; border: string }> = {
  A: { bg: 'bg-emerald-50', text: 'text-emerald-600', border: 'border-emerald-300' },
  B: { bg: 'bg-blue-50', text: 'text-blue-600', border: 'border-blue-300' },
  C: { bg: 'bg-yellow-50', text: 'text-yellow-600', border: 'border-yellow-300' },
  D: { bg: 'bg-orange-50', text: 'text-orange-600', border: 'border-orange-300' },
  F: { bg: 'bg-red-50', text: 'text-red-600', border: 'border-red-300' },
};

const formatCurrency = (value: number): string => {
  if (value >= 1000000) {
    return `${(value / 1000000).toFixed(1)}M`;
  }
  if (value >= 1000) {
    return `${(value / 1000).toFixed(0)}K`;
  }
  return value.toFixed(0);
};

export const ClientScorePanel: React.FC<ClientScorePanelProps> = ({
  open,
  loading,
  error,
  clientId,
  clientName,
  scoreData,
  sidebarOpen = false,
  onClose,
}) => {
  if (!open) return null;

  const grade = scoreData?.score_grade || 'C';
  const colors = gradeColors[grade] || gradeColors.C;

  // Detect cold start: API flag or fallback to zero order counts
  const isColdStart = scoreData?.is_cold_start ||
    (scoreData && scoreData.paid_order_count === 0 && scoreData.unpaid_order_count === 0);

  // Position to the right of sidebar: collapsed = 64px (ml-16) + 24px gap, expanded = 288px (ml-72) + 24px gap
  const leftPosition = sidebarOpen ? 'left-[312px]' : 'left-[88px]';

  return (
    <div className={`fixed ${leftPosition} top-6 w-[380px] max-h-[calc(100vh-100px)] z-50 flex flex-col bg-white rounded-2xl shadow-2xl border border-slate-200/60 overflow-hidden transition-all duration-300`}>
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-slate-100 bg-slate-50/50">
        <div>
          <h2 className="text-lg font-semibold text-slate-800">{translations.title}</h2>
          {clientName && (
            <p className="text-sm text-slate-500 truncate max-w-[280px]">{clientName}</p>
          )}
        </div>
        <button
          onClick={onClose}
          className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
          title="Закрити"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-5">
        {loading && (
          <div className="flex items-center justify-center h-40">
            <div className="flex flex-col items-center gap-3">
              <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
              <span className="text-sm text-slate-500">{translations.loading}</span>
            </div>
          </div>
        )}

        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-red-600 text-sm">
            {error}
          </div>
        )}

        {/* Cold Start - No Data State */}
        {!loading && !error && scoreData && isColdStart && (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <svg
              className="w-20 h-20 text-slate-200 mb-4"
              fill="none"
              stroke="currentColor"
              strokeWidth="1"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
              />
            </svg>
            <p className="text-lg font-medium text-slate-600 mb-1">{translations.coldStartTitle}</p>
            <p className="text-sm text-slate-400">{translations.coldStartDesc}</p>
          </div>
        )}

        {/* Normal Score Display */}
        {!loading && !error && scoreData && !isColdStart && (
          <div className="space-y-5">
            {/* Main Score Display */}
            <div className={`${colors.bg} border ${colors.border} rounded-2xl p-6 text-center`}>
              <div className={`inline-flex items-center justify-center w-16 h-16 ${colors.bg} border-2 ${colors.border} rounded-xl mb-3`}>
                <span className={`text-3xl font-bold ${colors.text}`}>{grade}</span>
              </div>
              <div className="text-4xl font-bold text-slate-800 mb-1">
                {scoreData.overall_score.toFixed(1)}
              </div>
              <div className="text-sm text-slate-500">{translations.outOf100}</div>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 gap-3">
              {/* Avg Days to Pay */}
              <div className="bg-slate-50/60 border border-slate-200 rounded-xl p-4 text-center">
                <div className="text-2xl font-bold text-slate-800">
                  {scoreData.avg_days_to_pay !== null ? scoreData.avg_days_to_pay.toFixed(1) : '--'}
                </div>
                <div className="text-xs text-slate-500 mt-1">{translations.avgDays}</div>
              </div>

              {/* On-Time % */}
              <div className="bg-emerald-50/60 border border-emerald-200 rounded-xl p-4 text-center">
                <div className="text-2xl font-bold text-emerald-600">
                  {scoreData.on_time_percentage.toFixed(0)}%
                </div>
                <div className="text-xs text-slate-500 mt-1">{translations.onTime}</div>
              </div>

              {/* Unpaid Count */}
              <div className="bg-orange-50/60 border border-orange-200 rounded-xl p-4 text-center">
                <div className="text-2xl font-bold text-orange-600">
                  {scoreData.unpaid_order_count}
                </div>
                <div className="text-xs text-slate-500 mt-1">{translations.unpaidCount}</div>
              </div>

              {/* Debt Amount */}
              <div className="bg-red-50/60 border border-red-200 rounded-xl p-4 text-center">
                <div className="text-2xl font-bold text-red-600">
                  {formatCurrency(scoreData.unpaid_amount)} UAH
                </div>
                <div className="text-xs text-slate-500 mt-1">{translations.debtAmount}</div>
              </div>
            </div>

            {/* Monthly Trend */}
            {scoreData.monthly_scores && scoreData.monthly_scores.length > 0 && (
              <div className="bg-slate-50/60 border border-slate-200 rounded-xl p-4">
                <h3 className="text-xs font-semibold text-slate-600 uppercase tracking-wide mb-3">
                  {translations.trendTitle}
                </h3>
                <div className="flex items-end justify-between gap-1 h-16">
                  {scoreData.monthly_scores.map((m, idx) => {
                    const height = Math.max(10, (m.score / 100) * 100);
                    const isLast = idx === scoreData.monthly_scores.length - 1;
                    return (
                      <div key={m.month} className="flex-1 flex flex-col items-center gap-1">
                        <div
                          className={`w-full rounded-t transition-all ${isLast ? 'bg-blue-500' : 'bg-slate-300'}`}
                          style={{ height: `${height}%` }}
                          title={`${m.month}: ${m.score.toFixed(1)}`}
                        />
                        <span className="text-[10px] text-slate-400">
                          {m.month.split('-')[1]}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Additional Stats */}
            <div className="space-y-2 text-sm">
              <div className="flex justify-between py-2 border-b border-slate-100">
                <span className="text-slate-500">{translations.paidOrders}</span>
                <span className="font-medium text-slate-800">{scoreData.paid_order_count}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-slate-100">
                <span className="text-slate-500">{translations.paidAmount}</span>
                <span className="font-medium text-slate-800">{formatCurrency(scoreData.paid_amount)} UAH</span>
              </div>
              {scoreData.oldest_unpaid_days !== null && scoreData.oldest_unpaid_days > 0 && (
                <div className="flex justify-between py-2">
                  <span className="text-slate-500">{translations.oldestDebt}</span>
                  <span className="font-medium text-red-600">
                    {scoreData.oldest_unpaid_days} {translations.days}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {!loading && !error && !scoreData && (
          <div className="flex items-center justify-center h-40 text-slate-400">
            {translations.noData}
          </div>
        )}
      </div>

      {/* Footer with client ID */}
      {clientId && (
        <div className="px-5 py-3 border-t border-slate-100 bg-slate-50/50">
          <span className="text-xs text-slate-400">Client ID: {clientId}</span>
        </div>
      )}
    </div>
  );
};

export default ClientScorePanel;
