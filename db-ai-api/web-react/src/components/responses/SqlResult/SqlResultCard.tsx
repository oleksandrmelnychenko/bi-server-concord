import React, { useState } from 'react';
import { SqlResultResponse, ColumnDefinition } from '../../../types/responses';
import { CodeBlock } from './CodeBlock';
import { DataTable } from '../DataTable/DataTable';

export const SqlResultCard: React.FC<SqlResultResponse> = ({
  title,
  sql,
  explanation,
  result,
  executionTime,
  showSql = true,
}) => {
  const [isSqlExpanded, setIsSqlExpanded] = useState(false);

  const columns = result?.columns || [];
  const rows = result?.rows || [];

  const tableColumns: ColumnDefinition[] = columns.map((col) => ({
    key: col,
    label: col,
    sortable: true,
    type: detectColumnType(rows[0]?.[col]),
  }));

  return (
    <div className="sql-result-card space-y-4">
      {(title || explanation) && (
        <div className="grok-card border-white/10 p-4">
          {title && (
            <h3 className="font-semibold text-slate-100 flex items-center gap-2 mb-2">
              <svg className="w-5 h-5 text-sky-200 thin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <rect x="5" y="4" width="14" height="16" rx="2" />
                <path d="M9 4v4h6V4" />
              </svg>
              {title}
            </h3>
          )}
          {explanation && (
            <p className="text-slate-300 text-sm leading-relaxed">{explanation}</p>
          )}
          {executionTime && (
            <div className="mt-2 text-xs text-slate-400">
              Execution time: {executionTime}ms
            </div>
          )}
        </div>
      )}

      {showSql && sql && (
        <div className="grok-card border-white/10 overflow-hidden">
          <button
            onClick={() => setIsSqlExpanded(!isSqlExpanded)}
            className="w-full px-4 py-3 flex items-center justify-between text-left bg-white/5 hover:bg-white/10 transition-colors"
          >
            <span className="flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-slate-300">
              <svg className="w-4 h-4 thin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path d="M8 17l-4-5 4-5" />
                <path d="M16 7l4 5-4 5" />
              </svg>
              SQL query
            </span>
            <svg
              className={`w-4 h-4 text-slate-400 thin-icon transition-transform ${isSqlExpanded ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path d="M6 9l6 6 6-6" />
            </svg>
          </button>

          {isSqlExpanded && (
            <div className="p-4 border-t border-white/10">
              <CodeBlock code={sql} language="sql" />
            </div>
          )}
        </div>
      )}

      {result?.success ? (
        <DataTable
          type="data_table"
          columns={tableColumns}
          rows={rows}
          totalRows={result?.row_count}
          enableSorting={true}
          enableExport={true}
        />
      ) : (
        <div className="p-4 rounded-xl border border-rose-400/30 bg-rose-400/10">
          <h4 className="font-semibold text-rose-200 mb-2 flex items-center gap-2">
            <svg className="w-5 h-5 thin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path d="M12 9v4" />
              <path d="M12 17h.01" />
              <circle cx="12" cy="12" r="9" />
            </svg>
            Query failed
          </h4>
          <p className="text-rose-200/80">{result?.error || 'Unable to fetch results.'}</p>
        </div>
      )}
    </div>
  );
};

function detectColumnType(value: unknown): ColumnDefinition['type'] {
  if (value === null || value === undefined) return 'string';

  if (typeof value === 'number') {
    return 'number';
  }

  if (typeof value === 'boolean') {
    return 'boolean';
  }

  if (typeof value === 'string') {
    if (/^\d{4}-\d{2}-\d{2}/.test(value)) {
      return 'date';
    }
    if (/UAH|USD|EUR|GBP/i.test(value)) {
      return 'currency';
    }
    if (/^\d+\.?\d*%$/.test(value)) {
      return 'percent';
    }
  }

  return 'string';
}

export default SqlResultCard;
