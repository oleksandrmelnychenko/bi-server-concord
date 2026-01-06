import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TextFilter } from './TextFilter';
import { NumberRangeFilter } from './NumberRangeFilter';
import { DateRangeFilter } from './DateRangeFilter';
import { SelectFilter } from './SelectFilter';

// Filter configuration types
export type FilterType = 'text' | 'number' | 'date' | 'select';

export interface FilterConfig {
  key: string;
  type: FilterType;
  label: string;
  placeholder?: string;
  // For number filters
  min?: number;
  max?: number;
  step?: number;
  // For select filters
  options?: Array<{ value: string; label: string; count?: number }>;
  multiple?: boolean;
  searchable?: boolean;
  // For date filters
  minDate?: string;
  maxDate?: string;
  // General
  width?: string;
}

export interface FilterValues {
  [key: string]: string | number | { min?: number; max?: number } | { from?: string; to?: string } | string[];
}

interface FilterBarProps {
  filters: FilterConfig[];
  values: FilterValues;
  onChange: (values: FilterValues) => void;
  onClearAll?: () => void;
  collapsible?: boolean;
  defaultExpanded?: boolean;
  showActiveCount?: boolean;
  className?: string;
}

export const FilterBar: React.FC<FilterBarProps> = ({
  filters,
  values,
  onChange,
  onClearAll,
  collapsible = true,
  defaultExpanded = true,
  showActiveCount = true,
  className = '',
}) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  const handleFilterChange = useCallback((key: string, value: any) => {
    onChange({ ...values, [key]: value });
  }, [values, onChange]);

  const handleClearAll = useCallback(() => {
    const clearedValues: FilterValues = {};
    filters.forEach(filter => {
      if (filter.type === 'text' || (filter.type === 'select' && !filter.multiple)) {
        clearedValues[filter.key] = '';
      } else if (filter.type === 'number') {
        clearedValues[filter.key] = { min: undefined, max: undefined };
      } else if (filter.type === 'date') {
        clearedValues[filter.key] = { from: undefined, to: undefined };
      } else if (filter.type === 'select' && filter.multiple) {
        clearedValues[filter.key] = [];
      }
    });
    onChange(clearedValues);
    onClearAll?.();
  }, [filters, onChange, onClearAll]);

  // Count active filters
  const activeFilterCount = filters.reduce((count, filter) => {
    const value = values[filter.key];
    if (!value) return count;

    if (typeof value === 'string') {
      return value ? count + 1 : count;
    }
    if (Array.isArray(value)) {
      return value.length > 0 ? count + 1 : count;
    }
    if (typeof value === 'object') {
      const hasValue = Object.values(value).some(v => v !== undefined && v !== '');
      return hasValue ? count + 1 : count;
    }
    return count;
  }, 0);

  const renderFilter = (filter: FilterConfig) => {
    const value = values[filter.key];
    const width = filter.width || 'flex-1 min-w-[180px]';

    switch (filter.type) {
      case 'text':
        return (
          <div key={filter.key} className={width}>
            <TextFilter
              value={(value as string) || ''}
              onChange={(v) => handleFilterChange(filter.key, v)}
              label={filter.label}
              placeholder={filter.placeholder}
            />
          </div>
        );

      case 'number':
        return (
          <div key={filter.key} className={width}>
            <NumberRangeFilter
              value={(value as { min?: number; max?: number }) || {}}
              onChange={(v) => handleFilterChange(filter.key, v)}
              label={filter.label}
              min={filter.min}
              max={filter.max}
              step={filter.step}
            />
          </div>
        );

      case 'date':
        return (
          <div key={filter.key} className={width}>
            <DateRangeFilter
              value={(value as { from?: string; to?: string }) || {}}
              onChange={(v) => handleFilterChange(filter.key, v)}
              label={filter.label}
              minDate={filter.minDate}
              maxDate={filter.maxDate}
            />
          </div>
        );

      case 'select':
        return (
          <div key={filter.key} className={width}>
            <SelectFilter
              value={(value as string | string[]) || (filter.multiple ? [] : '')}
              onChange={(v) => handleFilterChange(filter.key, v)}
              options={filter.options || []}
              label={filter.label}
              placeholder={filter.placeholder}
              multiple={filter.multiple}
              searchable={filter.searchable}
            />
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className={`bg-white/[0.02] border border-white/10 rounded-xl overflow-hidden ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/10">
        <div className="flex items-center gap-3">
          {collapsible && (
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="text-white/60 hover:text-white transition-colors"
            >
              <motion.svg
                className="w-5 h-5"
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

          <div className="flex items-center gap-2">
            <svg className="w-5 h-5 text-white/50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
            </svg>
            <span className="text-sm font-medium text-white/80">Фільтри</span>
            {showActiveCount && activeFilterCount > 0 && (
              <motion.span
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="bg-sky-500/20 text-sky-400 text-xs px-2 py-0.5 rounded-full"
              >
                {activeFilterCount}
              </motion.span>
            )}
          </div>
        </div>

        {activeFilterCount > 0 && (
          <motion.button
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            onClick={handleClearAll}
            className="text-sm text-white/40 hover:text-white/70 transition-colors flex items-center gap-1.5"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
            Скинути все
          </motion.button>
        )}
      </div>

      {/* Filter Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="p-4">
              <div className="flex flex-wrap gap-4">
                {filters.map(renderFilter)}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Collapsed Summary */}
      <AnimatePresence>
        {!isExpanded && activeFilterCount > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="px-4 py-2 flex flex-wrap gap-2"
          >
            {filters.map(filter => {
              const value = values[filter.key];
              if (!value) return null;

              let displayValue = '';
              let isActive = false;

              if (typeof value === 'string' && value) {
                displayValue = value;
                isActive = true;
              } else if (Array.isArray(value) && value.length > 0) {
                displayValue = `${value.length} обрано`;
                isActive = true;
              } else if (typeof value === 'object' && !Array.isArray(value)) {
                const objValue = value as Record<string, any>;
                const hasValue = Object.values(objValue).some(v => v !== undefined && v !== '');
                if (hasValue) {
                  if ('min' in objValue || 'max' in objValue) {
                    const parts = [];
                    if (objValue.min !== undefined) parts.push(`від ${objValue.min}`);
                    if (objValue.max !== undefined) parts.push(`до ${objValue.max}`);
                    displayValue = parts.join(' ');
                  } else if ('from' in objValue || 'to' in objValue) {
                    const parts = [];
                    if (objValue.from) parts.push(objValue.from);
                    if (objValue.to) parts.push(objValue.to);
                    displayValue = parts.join(' - ');
                  }
                  isActive = true;
                }
              }

              if (!isActive) return null;

              return (
                <span
                  key={filter.key}
                  className="inline-flex items-center gap-1.5 bg-sky-500/10 text-sky-400 text-xs px-2.5 py-1 rounded-full"
                >
                  <span className="text-white/50">{filter.label}:</span>
                  <span className="max-w-[120px] truncate">{displayValue}</span>
                  <button
                    onClick={() => {
                      if (filter.type === 'text' || (filter.type === 'select' && !filter.multiple)) {
                        handleFilterChange(filter.key, '');
                      } else if (filter.type === 'number') {
                        handleFilterChange(filter.key, { min: undefined, max: undefined });
                      } else if (filter.type === 'date') {
                        handleFilterChange(filter.key, { from: undefined, to: undefined });
                      } else if (filter.type === 'select' && filter.multiple) {
                        handleFilterChange(filter.key, []);
                      }
                    }}
                    className="text-white/40 hover:text-white/80 transition-colors"
                  >
                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </span>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default FilterBar;
