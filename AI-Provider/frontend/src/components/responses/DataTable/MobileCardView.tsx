import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { listItem, fastStaggerContainer } from '../../../utils/animations';

interface Column {
  key: string;
  header: string;
  type?: 'text' | 'number' | 'currency' | 'date' | 'percentage';
  format?: (value: any, row: Record<string, any>) => React.ReactNode;
  isPrimary?: boolean;
  isSecondary?: boolean;
  hideOnMobile?: boolean;
}

interface MobileCardViewProps {
  data: Record<string, any>[];
  columns: Column[];
  onRowClick?: (row: Record<string, any>, index: number) => void;
  selectedRows?: Set<number>;
  onSelectRow?: (index: number) => void;
  selectable?: boolean;
  expandable?: boolean;
  className?: string;
}

export const MobileCardView: React.FC<MobileCardViewProps> = ({
  data,
  columns,
  onRowClick,
  selectedRows = new Set(),
  onSelectRow,
  selectable = false,
  expandable = true,
  className = '',
}) => {
  const [expandedRows, setExpandedRows] = React.useState<Set<number>>(new Set());

  const toggleExpand = (index: number) => {
    const newExpanded = new Set(expandedRows);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedRows(newExpanded);
  };

  // Find primary and secondary columns
  const primaryColumn = columns.find(c => c.isPrimary) || columns[0];
  const secondaryColumn = columns.find(c => c.isSecondary) || columns[1];
  const visibleColumns = columns.filter(c => !c.hideOnMobile);
  const detailColumns = visibleColumns.filter(c => c !== primaryColumn && c !== secondaryColumn);

  const formatValue = (value: any, column: Column, row: Record<string, any>) => {
    if (column.format) {
      return column.format(value, row);
    }

    if (value === null || value === undefined) {
      return <span className="text-white/30">—</span>;
    }

    switch (column.type) {
      case 'number':
        return typeof value === 'number' ? value.toLocaleString('uk-UA') : value;
      case 'currency':
        return typeof value === 'number'
          ? value.toLocaleString('uk-UA', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + ' ₴'
          : value;
      case 'date':
        if (value instanceof Date) {
          return value.toLocaleDateString('uk-UA');
        }
        if (typeof value === 'string') {
          return new Date(value).toLocaleDateString('uk-UA');
        }
        return value;
      case 'percentage':
        return typeof value === 'number' ? `${value.toFixed(1)}%` : value;
      default:
        return String(value);
    }
  };

  const getValueColorClass = (value: any, column: Column) => {
    if (column.type === 'currency' || column.type === 'number') {
      if (typeof value === 'number') {
        if (value > 0) return 'text-emerald-400';
        if (value < 0) return 'text-red-400';
      }
    }
    if (column.type === 'percentage') {
      if (typeof value === 'number') {
        if (value >= 75) return 'text-emerald-400';
        if (value >= 50) return 'text-yellow-400';
        if (value >= 25) return 'text-orange-400';
        return 'text-red-400';
      }
    }
    return 'text-white/80';
  };

  return (
    <motion.div
      className={`space-y-3 ${className}`}
      variants={fastStaggerContainer}
      initial="initial"
      animate="animate"
    >
      <AnimatePresence mode="popLayout">
        {data.map((row, index) => {
          const isExpanded = expandedRows.has(index);
          const isSelected = selectedRows.has(index);
          const primaryValue = row[primaryColumn.key];
          const secondaryValue = secondaryColumn ? row[secondaryColumn.key] : null;

          return (
            <motion.div
              key={index}
              variants={listItem}
              layout
              className={`bg-white/[0.03] border rounded-xl overflow-hidden transition-colors ${
                isSelected
                  ? 'border-sky-500/40 bg-sky-500/5'
                  : 'border-white/10 hover:border-white/20'
              }`}
            >
              {/* Card Header */}
              <div
                className="p-4 cursor-pointer"
                onClick={() => {
                  if (onRowClick) {
                    onRowClick(row, index);
                  } else if (expandable) {
                    toggleExpand(index);
                  }
                }}
              >
                <div className="flex items-start gap-3">
                  {/* Selection Checkbox */}
                  {selectable && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectRow?.(index);
                      }}
                      className={`flex-shrink-0 w-5 h-5 mt-0.5 rounded border transition-colors flex items-center justify-center ${
                        isSelected
                          ? 'bg-sky-500 border-sky-500'
                          : 'border-white/30 hover:border-white/50'
                      }`}
                    >
                      {isSelected && (
                        <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                        </svg>
                      )}
                    </button>
                  )}

                  {/* Main Content */}
                  <div className="flex-1 min-w-0">
                    {/* Primary Value */}
                    <div className="font-medium text-white/90 truncate">
                      {formatValue(primaryValue, primaryColumn, row)}
                    </div>

                    {/* Secondary Value */}
                    {secondaryColumn && secondaryValue !== undefined && (
                      <div className={`text-sm mt-0.5 ${getValueColorClass(secondaryValue, secondaryColumn)}`}>
                        <span className="text-white/40">{secondaryColumn.header}: </span>
                        {formatValue(secondaryValue, secondaryColumn, row)}
                      </div>
                    )}

                    {/* Quick Stats - Show first 2-3 important values */}
                    {!isExpanded && detailColumns.length > 0 && (
                      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2">
                        {detailColumns.slice(0, 3).map((column) => {
                          const value = row[column.key];
                          if (value === null || value === undefined) return null;
                          return (
                            <span key={column.key} className="text-xs text-white/50">
                              {column.header}: <span className={getValueColorClass(value, column)}>{formatValue(value, column, row)}</span>
                            </span>
                          );
                        })}
                      </div>
                    )}
                  </div>

                  {/* Expand Button */}
                  {expandable && detailColumns.length > 0 && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleExpand(index);
                      }}
                      className="flex-shrink-0 w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center text-white/40 hover:text-white/70 hover:bg-white/10 transition-colors"
                    >
                      <motion.svg
                        className="w-4 h-4"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        animate={{ rotate: isExpanded ? 180 : 0 }}
                        transition={{ duration: 0.2 }}
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </motion.svg>
                    </button>
                  )}
                </div>
              </div>

              {/* Expanded Details */}
              <AnimatePresence>
                {isExpanded && detailColumns.length > 0 && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <div className="px-4 pb-4 pt-2 border-t border-white/10">
                      <div className="grid grid-cols-2 gap-x-4 gap-y-3">
                        {detailColumns.map((column) => {
                          const value = row[column.key];
                          return (
                            <div key={column.key}>
                              <div className="text-xs text-white/40 mb-0.5">{column.header}</div>
                              <div className={`text-sm ${getValueColorClass(value, column)}`}>
                                {formatValue(value, column, row)}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </AnimatePresence>

      {/* Empty State */}
      {data.length === 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center py-12"
        >
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-white/5 flex items-center justify-center">
            <svg className="w-8 h-8 text-white/30" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
            </svg>
          </div>
          <h3 className="text-white/60 font-medium">Немає даних</h3>
          <p className="text-white/40 text-sm mt-1">Спробуйте змінити фільтри</p>
        </motion.div>
      )}
    </motion.div>
  );
};

export default MobileCardView;
