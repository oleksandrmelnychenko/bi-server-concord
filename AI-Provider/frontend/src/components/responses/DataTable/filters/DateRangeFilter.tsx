import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface DateRangeFilterProps {
  value: { from?: string; to?: string };
  onChange: (value: { from?: string; to?: string }) => void;
  label?: string;
  minDate?: string;
  maxDate?: string;
  presets?: Array<{
    label: string;
    from: string;
    to: string;
  }>;
  className?: string;
}

// Default presets for quick date selection
const defaultPresets = [
  {
    label: 'Сьогодні',
    from: new Date().toISOString().split('T')[0],
    to: new Date().toISOString().split('T')[0]
  },
  {
    label: 'Вчора',
    from: new Date(Date.now() - 86400000).toISOString().split('T')[0],
    to: new Date(Date.now() - 86400000).toISOString().split('T')[0]
  },
  {
    label: 'Останні 7 днів',
    from: new Date(Date.now() - 7 * 86400000).toISOString().split('T')[0],
    to: new Date().toISOString().split('T')[0]
  },
  {
    label: 'Останні 30 днів',
    from: new Date(Date.now() - 30 * 86400000).toISOString().split('T')[0],
    to: new Date().toISOString().split('T')[0]
  },
  {
    label: 'Цей місяць',
    from: new Date(new Date().getFullYear(), new Date().getMonth(), 1).toISOString().split('T')[0],
    to: new Date().toISOString().split('T')[0]
  },
  {
    label: 'Минулий місяць',
    from: new Date(new Date().getFullYear(), new Date().getMonth() - 1, 1).toISOString().split('T')[0],
    to: new Date(new Date().getFullYear(), new Date().getMonth(), 0).toISOString().split('T')[0]
  },
];

export const DateRangeFilter: React.FC<DateRangeFilterProps> = ({
  value,
  onChange,
  label,
  minDate,
  maxDate,
  presets = defaultPresets,
  className = '',
}) => {
  const [showPresets, setShowPresets] = useState(false);

  const handleFromChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    onChange({ from: e.target.value || undefined, to: value.to });
  }, [onChange, value.to]);

  const handleToChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    onChange({ from: value.from, to: e.target.value || undefined });
  }, [onChange, value.from]);

  const handlePresetClick = useCallback((preset: { from: string; to: string }) => {
    onChange({ from: preset.from, to: preset.to });
    setShowPresets(false);
  }, [onChange]);

  const handleClear = useCallback(() => {
    onChange({ from: undefined, to: undefined });
  }, [onChange]);

  const hasValue = value.from || value.to;

  const formatDateForDisplay = (dateStr?: string) => {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    return date.toLocaleDateString('uk-UA', { day: '2-digit', month: '2-digit', year: 'numeric' });
  };

  return (
    <div className={`relative ${className}`}>
      {label && (
        <div className="flex items-center justify-between mb-1.5">
          <label className="text-xs text-white/50">{label}</label>
          {hasValue && (
            <button
              onClick={handleClear}
              className="text-xs text-white/30 hover:text-white/60 transition-colors"
            >
              Скинути
            </button>
          )}
        </div>
      )}

      <div className="flex items-center gap-2">
        {/* From Date */}
        <div className="relative flex-1">
          <div className="absolute left-3 top-1/2 -translate-y-1/2 text-white/30 pointer-events-none">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
          <input
            type="date"
            value={value.from || ''}
            onChange={handleFromChange}
            min={minDate}
            max={value.to || maxDate}
            className="w-full bg-white/5 border border-white/10 rounded-lg pl-9 pr-3 py-2 text-sm text-white/90 focus:outline-none focus:border-sky-500/50 transition-colors [color-scheme:dark]"
          />
        </div>

        <span className="text-white/30 text-sm">—</span>

        {/* To Date */}
        <div className="relative flex-1">
          <div className="absolute left-3 top-1/2 -translate-y-1/2 text-white/30 pointer-events-none">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
          <input
            type="date"
            value={value.to || ''}
            onChange={handleToChange}
            min={value.from || minDate}
            max={maxDate}
            className="w-full bg-white/5 border border-white/10 rounded-lg pl-9 pr-3 py-2 text-sm text-white/90 focus:outline-none focus:border-sky-500/50 transition-colors [color-scheme:dark]"
          />
        </div>

        {/* Presets Button */}
        <button
          onClick={() => setShowPresets(!showPresets)}
          className={`p-2 rounded-lg border transition-colors ${
            showPresets
              ? 'bg-sky-500/20 border-sky-500/30 text-sky-400'
              : 'bg-white/5 border-white/10 text-white/50 hover:text-white/80'
          }`}
          title="Швидкий вибір"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </button>
      </div>

      {/* Presets Dropdown */}
      <AnimatePresence>
        {showPresets && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="absolute z-20 mt-2 right-0 bg-gray-900 border border-white/10 rounded-lg shadow-xl overflow-hidden min-w-[180px]"
          >
            {presets.map((preset, index) => (
              <button
                key={index}
                onClick={() => handlePresetClick(preset)}
                className="w-full px-4 py-2.5 text-left text-sm text-white/80 hover:bg-white/5 transition-colors flex items-center justify-between group"
              >
                <span>{preset.label}</span>
                <span className="text-xs text-white/30 group-hover:text-white/50">
                  {formatDateForDisplay(preset.from)} - {formatDateForDisplay(preset.to)}
                </span>
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Click outside to close */}
      {showPresets && (
        <div
          className="fixed inset-0 z-10"
          onClick={() => setShowPresets(false)}
        />
      )}
    </div>
  );
};

export default DateRangeFilter;
