import React from 'react';
import { StatisticsResponse } from '../../../types/responses';
import { StatCard } from './StatCard';

export const StatisticsGrid: React.FC<StatisticsResponse> = ({
  title,
  cards = [],
  layout = 'grid-4',
}) => {
  const safeCards = cards || [];
  const gridClass = {
    'grid-2': 'grid-cols-1 sm:grid-cols-2',
    'grid-3': 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3',
    'grid-4': 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4',
  }[layout];

  return (
    <div className="statistics-grid">
      {title && (
        <h3 className="text-base font-semibold text-slate-100 mb-4 flex items-center gap-2">
          <svg className="w-5 h-5 text-sky-200 thin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path d="M4 19h16" />
            <path d="M7 16V8" />
            <path d="M12 16V5" />
            <path d="M17 16v-6" />
          </svg>
          {title}
        </h3>
      )}

      <div className={`grid ${gridClass} gap-4`}>
        {safeCards.map((card, index) => (
          <StatCard key={index} {...card} />
        ))}
      </div>
    </div>
  );
};

export default StatisticsGrid;
