import React, { useState, useMemo } from 'react';
import { DataTableResponse, ColumnDefinition, SortState } from '../../../types/responses';

const inferStringType = (value: string): ColumnDefinition['type'] => {
  if (/^\d{4}-\d{2}-\d{2}/.test(value)) return 'date';
  if (/UAH|USD|EUR|GBP/i.test(value)) return 'currency';
  if (/^\d+\.?\d*%$/.test(value)) return 'percent';
  return 'string';
};

const resolveColumnType = (value: unknown, column: ColumnDefinition): ColumnDefinition['type'] => {
  if (column.type) return column.type;
  if (value instanceof Date) return 'date';
  if (typeof value === 'number') return 'number';
  if (typeof value === 'boolean') return 'boolean';
  if (typeof value === 'string') return inferStringType(value);
  return 'string';
};

const formatNumber = (value: number, maximumFractionDigits = 3): string => {
  const isInt = Number.isInteger(value);
  return value.toLocaleString('uk-UA', {
    minimumFractionDigits: 0,
    maximumFractionDigits: isInt ? 0 : maximumFractionDigits,
  });
};

// Format cell value based on column type
const formatValue = (value: unknown, column: ColumnDefinition): string => {
  if (value === null || value === undefined) return '--';

  const columnType = resolveColumnType(value, column);

  switch (columnType) {
    case 'number':
      return typeof value === 'number' ? formatNumber(value) : String(value);
    case 'currency':
      return typeof value === 'number'
        ? `${value.toLocaleString('uk-UA', { minimumFractionDigits: 2, maximumFractionDigits: 3 })} UAH`
        : String(value);
    case 'percent':
      return typeof value === 'number' ? `${value.toFixed(3)}%` : String(value);
    case 'date':
      if (value instanceof Date) {
        return value.toLocaleDateString('uk-UA');
      }
      if (typeof value === 'string') {
        const date = new Date(value);
        return isNaN(date.getTime()) ? value : date.toLocaleDateString('uk-UA');
      }
      return String(value);
    case 'boolean':
      return value ? 'Yes' : 'No';
    default: {
      const str = String(value);
      return str.length > 100 ? `${str.substring(0, 100)}...` : str;
    }
  }
};

const formatCsvNumber = (value: number, maximumFractionDigits = 3): string => {
  const isInt = Number.isInteger(value);
  return isInt ? String(value) : value.toFixed(maximumFractionDigits);
};

const formatCsvValue = (value: unknown, column: ColumnDefinition): string => {
  if (value === null || value === undefined) return '';

  const columnType = resolveColumnType(value, column);

  switch (columnType) {
    case 'number':
      return typeof value === 'number' ? formatCsvNumber(value) : String(value);
    case 'currency':
      return typeof value === 'number' ? value.toFixed(3) : String(value);
    case 'percent':
      return typeof value === 'number' ? value.toFixed(3) : String(value);
    case 'date':
      if (value instanceof Date) {
        return value.toISOString();
      }
      return String(value);
    case 'boolean':
      return value ? 'true' : 'false';
    default:
      return String(value);
  }
};

// Export to CSV
const exportToCsv = (columns: ColumnDefinition[], rows: Record<string, unknown>[], filename: string) => {
  const headers = columns.map((col) => col.label).join(',');
  const csvRows = rows.map((row) =>
    columns
      .map((col) => {
        const value = row[col.key];
        const formatted = formatCsvValue(value, col);
        return formatted.includes(',') || formatted.includes('"')
          ? `"${formatted.replace(/"/g, '""')}"`
          : formatted;
      })
      .join(',')
  );

  const csv = [headers, ...csvRows].join('\n');
  const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${filename}.csv`;
  link.click();
  URL.revokeObjectURL(url);
};

const SortIcon: React.FC<{ direction?: 'asc' | 'desc' | null }> = ({ direction }) => {
  if (!direction) {
    return (
      <svg className="w-3.5 h-3.5 text-slate-500 thin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <path d="M8 10l4-4 4 4" />
        <path d="M16 14l-4 4-4-4" />
      </svg>
    );
  }

  return direction === 'asc' ? (
    <svg className="w-3.5 h-3.5 text-sky-200 thin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <path d="M8 14l4-4 4 4" />
    </svg>
  ) : (
    <svg className="w-3.5 h-3.5 text-sky-200 thin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <path d="M16 10l-4 4-4-4" />
    </svg>
  );
};

export const DataTable: React.FC<DataTableResponse> = ({
  title,
  columns = [],
  rows = [],
  totalRows,
  enableSorting = true,
  enableExport = true,
  maxHeight = '400px',
}) => {
  const [sort, setSort] = useState<SortState | null>(null);
  const [copied, setCopied] = useState(false);

  const safeColumns = columns || [];
  const safeRows = rows || [];

  const sortedRows = useMemo(() => {
    if (!sort) return safeRows;

    return [...safeRows].sort((a, b) => {
      const aVal = a[sort.column];
      const bVal = b[sort.column];

      if (aVal === null || aVal === undefined) return 1;
      if (bVal === null || bVal === undefined) return -1;

      let comparison = 0;
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        comparison = aVal - bVal;
      } else {
        comparison = String(aVal).localeCompare(String(bVal), 'uk-UA');
      }

      return sort.direction === 'asc' ? comparison : -comparison;
    });
  }, [safeRows, sort]);

  const handleSort = (column: ColumnDefinition) => {
    if (!enableSorting || !column.sortable) return;

    setSort((prev) => {
      if (prev?.column === column.key) {
        return prev.direction === 'asc' ? { column: column.key, direction: 'desc' } : null;
      }
      return { column: column.key, direction: 'asc' };
    });
  };

  const handleCopy = async () => {
    const text = [
      safeColumns.map((c) => c.label).join('\t'),
      ...sortedRows.map((row) => safeColumns.map((c) => formatValue(row[c.key], c)).join('\t')),
    ].join('\n');

    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="data-table-container grok-card overflow-hidden border-white/10">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-white/5 border-b border-white/10">
        <div className="flex items-center gap-3">
          {title && <h3 className="font-semibold text-slate-100 text-sm">{title}</h3>}
          <span className="text-xs text-slate-400">
            {totalRows ?? safeRows.length} rows
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleCopy}
            className="p-2 text-slate-400 hover:text-slate-100 hover:bg-white/10 rounded transition-colors"
            title="Copy table"
          >
            {copied ? (
              <svg className="w-4 h-4 text-emerald-300 thin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path d="M5 13l4 4L19 7" />
              </svg>
            ) : (
              <svg className="w-4 h-4 thin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <rect x="9" y="9" width="11" height="11" rx="2" />
                <rect x="4" y="4" width="11" height="11" rx="2" />
              </svg>
            )}
          </button>

          {enableExport && (
            <button
              onClick={() => exportToCsv(safeColumns, sortedRows, title || 'export')}
              className="flex items-center gap-1 px-3 py-1.5 text-xs uppercase tracking-[0.2em] text-slate-300 hover:text-white hover:bg-white/10 rounded transition-colors"
            >
              <svg className="w-4 h-4 thin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path d="M12 4v10" />
                <path d="M8 10l4 4 4-4" />
                <path d="M4 20h16" />
              </svg>
              CSV
            </button>
          )}
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto" style={{ maxHeight }}>
        <table className="w-full data-table">
          <thead className="bg-white/5 sticky top-0">
            <tr>
              {safeColumns.map((column) => (
                <th
                  key={column.key}
                  onClick={() => handleSort(column)}
                  className={`px-4 py-3 text-left border-b border-white/10 ${
                    enableSorting && column.sortable !== false ? 'cursor-pointer hover:bg-white/5 select-none' : ''
                  } ${column.align === 'right' ? 'text-right' : column.align === 'center' ? 'text-center' : ''}`}
                  style={{ width: column.width }}
                >
                  <div className="flex items-center gap-2">
                    <span>{column.label}</span>
                    {enableSorting && column.sortable !== false && (
                      <SortIcon direction={sort?.column === column.key ? sort.direction : null} />
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5">
            {sortedRows.map((row, rowIndex) => (
              <tr key={rowIndex} className="hover:bg-white/5 transition-colors">
                {safeColumns.map((column) => (
                  <td
                    key={column.key}
                    className={`px-4 py-3 text-sm text-slate-200 ${
                      column.align === 'right' ? 'text-right' : column.align === 'center' ? 'text-center' : ''
                    } ${column.type === 'number' || column.type === 'currency' ? 'font-mono' : ''}`}
                  >
                    {formatValue(row[column.key], column)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>

        {sortedRows.length === 0 && (
          <div className="p-8 text-center text-slate-400">
            No results to display.
          </div>
        )}
      </div>
    </div>
  );
};

export default DataTable;
