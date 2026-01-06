import React, { useState, useMemo } from 'react';
import { DataTableResponse, ColumnDefinition, SortState } from '../../../types/responses';
import type { Language } from '../../WelcomeMessage';

const tableTranslations = {
  uk: {
    rows: 'рядків',
    copyTable: 'Копіювати таблицю',
    noResults: 'Немає результатів для відображення.',
    yes: 'Так',
    no: 'Ні',
  },
  en: {
    rows: 'rows',
    copyTable: 'Copy table',
    noResults: 'No results to display.',
    yes: 'Yes',
    no: 'No',
  },
};

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
      <svg className="w-3 h-3 text-slate-300" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M8 10l4-4 4 4" />
        <path d="M16 14l-4 4-4-4" />
      </svg>
    );
  }

  return direction === 'asc' ? (
    <svg className="w-3 h-3 text-slate-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
      <path d="M8 14l4-4 4 4" />
    </svg>
  ) : (
    <svg className="w-3 h-3 text-slate-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
      <path d="M16 10l-4 4-4-4" />
    </svg>
  );
};

export const DataTable: React.FC<DataTableResponse & { language?: Language; onRowClick?: (row: Record<string, unknown>) => void }> = ({
  title,
  columns = [],
  rows = [],
  totalRows,
  enableSorting = true,
  enableExport = true,
  maxHeight = '400px',
  language = 'uk',
  onRowClick,
}) => {
  const t = tableTranslations[language];
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
    <div className="data-table-container bg-white rounded-xl overflow-hidden shadow-sm border border-slate-200/60">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-slate-100">
        <div className="flex items-center gap-3">
          {title && <h3 className="font-semibold text-slate-800 text-[15px]">{title}</h3>}
          <span className="px-2 py-0.5 text-[11px] font-medium text-slate-500 bg-slate-100 rounded-full">
            {totalRows ?? safeRows.length} {t.rows}
          </span>
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={handleCopy}
            className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-50 rounded-lg transition-all"
            title={t.copyTable}
          >
            {copied ? (
              <svg className="w-4 h-4 text-emerald-500" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path d="M5 13l4 4L19 7" />
              </svg>
            ) : (
              <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
                <rect x="9" y="9" width="11" height="11" rx="2" />
                <path d="M5 15V5a2 2 0 012-2h10" />
              </svg>
            )}
          </button>

          {enableExport && (
            <button
              onClick={() => exportToCsv(safeColumns, sortedRows, title || 'export')}
              className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium text-slate-500 hover:text-slate-700 hover:bg-slate-50 rounded-lg transition-all"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                <path d="M7 10l5 5 5-5" />
                <path d="M12 15V3" />
              </svg>
              Export
            </button>
          )}
        </div>
      </div>

      {/* Table */}
      <div className="overflow-auto" style={{ maxHeight }}>
        <table className="w-full border-collapse">
          <thead className="sticky top-0 z-10 bg-white before:absolute before:left-0 before:right-0 before:top-0 before:h-[2px] before:bg-black after:absolute after:left-0 after:right-0 after:bottom-0 after:h-[2px] after:bg-black relative">
            <tr className="bg-white">
              {safeColumns.map((column) => (
                <th
                  key={column.key}
                  onClick={() => handleSort(column)}
                  className={`px-4 py-2.5 text-left text-[11px] font-bold text-black uppercase tracking-wider ${
                    enableSorting && column.sortable !== false ? 'cursor-pointer hover:bg-slate-50 select-none transition-colors' : ''
                  } ${column.align === 'right' ? 'text-right' : column.align === 'center' ? 'text-center' : ''}`}
                  style={{ width: column.width }}
                >
                  <div className={`flex items-center gap-1.5 ${column.align === 'right' ? 'justify-end' : column.align === 'center' ? 'justify-center' : ''}`}>
                    <span>{column.label}</span>
                    {enableSorting && column.sortable !== false && (
                      <SortIcon direction={sort?.column === column.key ? sort.direction : null} />
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedRows.map((row, rowIndex) => {
              const clickable = typeof onRowClick === 'function';
              return (
                <tr
                  key={rowIndex}
                  className={`group transition-colors ${rowIndex % 2 === 0 ? 'bg-white' : 'bg-slate-50/30'} hover:bg-slate-100/50 ${clickable ? 'cursor-pointer' : ''}`}
                  onClick={() => clickable && onRowClick(row)}
                >
                  {safeColumns.map((column, colIndex) => (
                    <td
                      key={column.key}
                      className={`px-4 py-2 text-[13px] text-slate-900 border-b border-slate-100/60 ${
                        column.align === 'right' ? 'text-right' : column.align === 'center' ? 'text-center' : ''
                      } ${column.type === 'number' || column.type === 'currency' ? 'font-mono tabular-nums' : ''} ${
                        colIndex === 0 ? 'font-medium' : ''
                      }`}
                    >
                      {formatValue(row[column.key], column)}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>

        {sortedRows.length === 0 && (
          <div className="py-12 text-center">
            <p className="text-sm text-slate-400">{t.noResults}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default DataTable;
