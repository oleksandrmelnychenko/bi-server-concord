import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface TextFilterProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  label?: string;
  debounceMs?: number;
  showClearButton?: boolean;
  className?: string;
}

export const TextFilter: React.FC<TextFilterProps> = ({
  value,
  onChange,
  placeholder = 'Пошук...',
  label,
  debounceMs = 300,
  showClearButton = true,
  className = '',
}) => {
  const [localValue, setLocalValue] = useState(value);
  const [timeoutId, setTimeoutId] = useState<ReturnType<typeof setTimeout> | null>(null);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setLocalValue(newValue);

    if (timeoutId) {
      clearTimeout(timeoutId);
    }

    const newTimeoutId = setTimeout(() => {
      onChange(newValue);
    }, debounceMs);

    setTimeoutId(newTimeoutId);
  }, [onChange, debounceMs, timeoutId]);

  const handleClear = useCallback(() => {
    setLocalValue('');
    onChange('');
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
  }, [onChange, timeoutId]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      handleClear();
    }
  }, [handleClear]);

  return (
    <div className={`relative ${className}`}>
      {label && (
        <label className="block text-xs text-white/50 mb-1.5">{label}</label>
      )}
      <div className="relative">
        {/* Search Icon */}
        <div className="absolute left-3 top-1/2 -translate-y-1/2 text-white/30">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>

        <input
          type="text"
          value={localValue}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className="w-full bg-white/5 border border-white/10 rounded-lg pl-9 pr-8 py-2 text-sm text-white/90 placeholder-white/30 focus:outline-none focus:border-sky-500/50 focus:ring-1 focus:ring-sky-500/20 transition-colors"
        />

        {/* Clear Button */}
        <AnimatePresence>
          {showClearButton && localValue && (
            <motion.button
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              onClick={handleClear}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-white/30 hover:text-white/60 transition-colors"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </motion.button>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default TextFilter;
