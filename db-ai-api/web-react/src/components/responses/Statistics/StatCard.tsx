import React from 'react';
import { StatCardData } from '../../../types/responses';

const Icons = {
  users: (
    <svg className="w-5 h-5 thin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path d="M15 19a4 4 0 0 0-8 0" />
      <path d="M12 13a4 4 0 1 0-4-4 4 4 0 0 0 4 4z" />
      <path d="M20 19a3 3 0 0 0-6 0" />
      <path d="M17 13a3 3 0 1 0-3-3 3 3 0 0 0 3 3z" />
    </svg>
  ),
  money: (
    <svg className="w-5 h-5 thin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="9" />
      <path d="M9.5 9a2.5 2.5 0 0 1 5 0c0 2-5 2-5 4a2.5 2.5 0 0 0 5 0" />
      <path d="M12 6v2" />
      <path d="M12 16v2" />
    </svg>
  ),
  chart: (
    <svg className="w-5 h-5 thin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path d="M4 19h16" />
      <path d="M7 16V9" />
      <path d="M12 16V6" />
      <path d="M17 16v-5" />
    </svg>
  ),
  box: (
    <svg className="w-5 h-5 thin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path d="M4 7l8-4 8 4-8 4-8-4z" />
      <path d="M4 7v10l8 4 8-4V7" />
      <path d="M12 11v10" />
    </svg>
  ),
  truck: (
    <svg className="w-5 h-5 thin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path d="M3 7h11v8H3z" />
      <path d="M14 10h4l3 3v2h-7z" />
      <circle cx="7" cy="18" r="2" />
      <circle cx="18" cy="18" r="2" />
    </svg>
  ),
  warning: (
    <svg className="w-5 h-5 thin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path d="M12 9v4" />
      <path d="M12 17h.01" />
      <path d="M10.3 4h3.4L21 18H3z" />
    </svg>
  ),
  check: (
    <svg className="w-5 h-5 thin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="9" />
      <path d="M8 12l2.5 2.5L16 9" />
    </svg>
  ),
};

const formatCompact = (value: number): string => {
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}K`;
  }
  return value.toLocaleString('uk-UA');
};

const formatValue = (value: string | number, format?: 'number' | 'currency' | 'percent'): string => {
  if (typeof value === 'string') return value;

  switch (format) {
    case 'currency':
      return `${formatCompact(value)} UAH`;
    case 'percent':
      return `${value.toFixed(1)}%`;
    case 'number':
    default:
      return formatCompact(value);
  }
};

export const StatCard: React.FC<StatCardData> = ({
  value,
  label,
  icon,
  trend,
  highlight = false,
  format,
}) => {
  const formattedValue = formatValue(value, format);

  return (
    <div
      className={`stat-card p-4 rounded-2xl border transition-all ${
        highlight
          ? 'bg-sky-400/10 border-sky-300/40 shadow-lg'
          : 'bg-white/5 border-white/10 hover:border-white/20'
      }`}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className={`text-2xl font-semibold ${highlight ? 'text-sky-100' : 'text-slate-100'}`}>
            {formattedValue}
          </div>
          <div className="text-sm text-slate-400 mt-1">{label}</div>
        </div>

        {icon && (
          <div className={`p-2 rounded-xl border ${
            highlight ? 'border-sky-300/40 bg-sky-400/10 text-sky-200' : 'border-white/10 bg-white/5 text-slate-400'
          }`}
          >
            {Icons[icon]}
          </div>
        )}
      </div>

      {trend && (
        <div className="mt-3 flex items-center gap-2 text-xs text-slate-400">
          <span className={`flex items-center gap-1 font-medium ${
            trend.direction === 'up' ? 'text-emerald-300' : 'text-rose-300'
          }`}
          >
            <svg className="w-3.5 h-3.5 thin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              {trend.direction === 'up' ? <path d="M7 14l5-5 5 5" /> : <path d="M7 10l5 5 5-5" />}
            </svg>
            {Math.abs(trend.value)}%
          </span>
          <span>vs previous period</span>
        </div>
      )}
    </div>
  );
};

export default StatCard;
