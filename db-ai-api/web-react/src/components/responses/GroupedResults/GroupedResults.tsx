import React, { useState } from 'react';
import { GroupedResultsResponse, ResponseSection } from '../../../types/responses';
import { DataTable } from '../DataTable/DataTable';
import { StatisticsGrid } from '../Statistics/StatisticsGrid';
import { SmartChart } from '../Charts/SmartChart';

export const GroupedResults: React.FC<GroupedResultsResponse> = ({
  title,
  groups = [],
  collapsible = true,
  defaultExpanded = true,
}) => {
  const safeGroups = groups || [];
  const [expandedGroups, setExpandedGroups] = useState<Set<number>>(
    defaultExpanded ? new Set(safeGroups.map((_, i) => i)) : new Set()
  );

  const toggleGroup = (index: number) => {
    if (!collapsible) return;

    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  const renderGroupContent = (content: ResponseSection | undefined): React.ReactNode => {
    if (!content || !content.type) return null;

    switch (content.type) {
      case 'data_table':
        return <DataTable {...content} />;
      case 'statistics':
        return <StatisticsGrid {...content} />;
      case 'chart':
        return <SmartChart {...content} />;
      default:
        return null;
    }
  };

  return (
    <div className="grouped-results space-y-4">
      {title && (
        <h3 className="text-base font-semibold text-slate-100 flex items-center gap-2">
          <svg className="w-5 h-5 text-sky-200 thin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path d="M4 7h16" />
            <path d="M4 12h16" />
            <path d="M4 17h10" />
          </svg>
          {title}
        </h3>
      )}

      <div className="space-y-3">
        {safeGroups.map((group, index) => {
          if (!group) return null;
          const isExpanded = expandedGroups.has(index);

          return (
            <div key={index} className="grok-card border-white/10 overflow-hidden">
              <button
                onClick={() => toggleGroup(index)}
                disabled={!collapsible}
                className={`w-full px-4 py-3 flex items-center justify-between text-left ${
                  collapsible ? 'hover:bg-white/5 cursor-pointer' : 'cursor-default'
                } transition-colors ${isExpanded ? 'border-b border-white/10' : ''}`}
              >
                <div className="flex items-center gap-3">
                  <span className="font-medium text-slate-100">{group.label}</span>
                  {group.badge && (
                    <span className="px-2 py-0.5 text-xs font-medium bg-sky-400/15 text-sky-200 rounded-full border border-sky-300/30">
                      {group.badge}
                    </span>
                  )}
                </div>

                {collapsible && (
                  <svg
                    className={`w-4 h-4 text-slate-400 thin-icon transition-transform ${
                      isExpanded ? 'rotate-180' : ''
                    }`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path d="M6 9l6 6 6-6" />
                  </svg>
                )}
              </button>

              {isExpanded && (
                <div className="p-4">
                  {renderGroupContent(group.content)}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {safeGroups.length === 0 && (
        <div className="p-8 text-center text-slate-400 bg-white/5 rounded-2xl border border-white/10">
          No grouped results available.
        </div>
      )}
    </div>
  );
};

export default GroupedResults;
