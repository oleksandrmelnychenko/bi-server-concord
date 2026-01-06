import React, { useState, useCallback, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface SelectOption {
  value: string;
  label: string;
  count?: number;
}

interface SelectFilterProps {
  options: SelectOption[];
  value: string | string[];
  onChange: (value: string | string[]) => void;
  label?: string;
  placeholder?: string;
  multiple?: boolean;
  searchable?: boolean;
  showCount?: boolean;
  maxHeight?: number;
  className?: string;
}

export const SelectFilter: React.FC<SelectFilterProps> = ({
  options,
  value,
  onChange,
  label,
  placeholder = 'Оберіть...',
  multiple = false,
  searchable = false,
  showCount = true,
  maxHeight = 240,
  className = '',
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Handle click outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
        setSearchTerm('');
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Focus search input when opening
  useEffect(() => {
    if (isOpen && searchable && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen, searchable]);

  const selectedValues = Array.isArray(value) ? value : (value ? [value] : []);

  const filteredOptions = options.filter(option =>
    option.label.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleToggle = useCallback(() => {
    setIsOpen(!isOpen);
    if (isOpen) {
      setSearchTerm('');
    }
  }, [isOpen]);

  const handleSelect = useCallback((optionValue: string) => {
    if (multiple) {
      const newValue = selectedValues.includes(optionValue)
        ? selectedValues.filter(v => v !== optionValue)
        : [...selectedValues, optionValue];
      onChange(newValue);
    } else {
      onChange(optionValue);
      setIsOpen(false);
      setSearchTerm('');
    }
  }, [multiple, selectedValues, onChange]);

  const handleClear = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onChange(multiple ? [] : '');
  }, [multiple, onChange]);

  const handleSelectAll = useCallback(() => {
    if (multiple) {
      onChange(options.map(o => o.value));
    }
  }, [multiple, options, onChange]);

  const handleClearAll = useCallback(() => {
    onChange(multiple ? [] : '');
  }, [multiple, onChange]);

  const getDisplayValue = () => {
    if (selectedValues.length === 0) {
      return placeholder;
    }
    if (selectedValues.length === 1) {
      const option = options.find(o => o.value === selectedValues[0]);
      return option?.label || selectedValues[0];
    }
    return `Обрано: ${selectedValues.length}`;
  };

  const hasValue = selectedValues.length > 0;

  return (
    <div ref={containerRef} className={`relative ${className}`}>
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

      {/* Trigger Button */}
      <button
        onClick={handleToggle}
        className={`w-full flex items-center justify-between gap-2 bg-white/5 border rounded-lg px-3 py-2 text-sm transition-colors ${
          isOpen
            ? 'border-sky-500/50 ring-1 ring-sky-500/20'
            : 'border-white/10 hover:border-white/20'
        }`}
      >
        <span className={hasValue ? 'text-white/90' : 'text-white/30'}>
          {getDisplayValue()}
        </span>
        <div className="flex items-center gap-1">
          {hasValue && (
            <motion.span
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="bg-sky-500/20 text-sky-400 text-xs px-1.5 py-0.5 rounded"
            >
              {selectedValues.length}
            </motion.span>
          )}
          <svg
            className={`w-4 h-4 text-white/40 transition-transform ${isOpen ? 'rotate-180' : ''}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {/* Dropdown */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.15 }}
            className="absolute z-30 mt-1 w-full bg-gray-900 border border-white/10 rounded-lg shadow-xl overflow-hidden"
          >
            {/* Search Input */}
            {searchable && (
              <div className="p-2 border-b border-white/10">
                <div className="relative">
                  <svg
                    className="absolute left-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-white/30"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                  <input
                    ref={inputRef}
                    type="text"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    placeholder="Шукати..."
                    className="w-full bg-white/5 border border-white/10 rounded pl-8 pr-3 py-1.5 text-sm text-white/90 placeholder-white/30 focus:outline-none focus:border-sky-500/50"
                  />
                </div>
              </div>
            )}

            {/* Multiple Selection Actions */}
            {multiple && options.length > 0 && (
              <div className="flex items-center justify-between px-3 py-2 border-b border-white/10 text-xs">
                <button
                  onClick={handleSelectAll}
                  className="text-sky-400 hover:text-sky-300 transition-colors"
                >
                  Обрати все
                </button>
                <button
                  onClick={handleClearAll}
                  className="text-white/40 hover:text-white/60 transition-colors"
                >
                  Очистити
                </button>
              </div>
            )}

            {/* Options List */}
            <div className="overflow-y-auto" style={{ maxHeight }}>
              {filteredOptions.length === 0 ? (
                <div className="px-3 py-4 text-center text-sm text-white/40">
                  {searchTerm ? 'Нічого не знайдено' : 'Немає опцій'}
                </div>
              ) : (
                filteredOptions.map((option) => {
                  const isSelected = selectedValues.includes(option.value);
                  return (
                    <button
                      key={option.value}
                      onClick={() => handleSelect(option.value)}
                      className={`w-full flex items-center justify-between px-3 py-2.5 text-sm transition-colors ${
                        isSelected
                          ? 'bg-sky-500/10 text-sky-400'
                          : 'text-white/80 hover:bg-white/5'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        {multiple && (
                          <div className={`w-4 h-4 rounded border flex items-center justify-center transition-colors ${
                            isSelected
                              ? 'bg-sky-500 border-sky-500'
                              : 'border-white/30'
                          }`}>
                            {isSelected && (
                              <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                              </svg>
                            )}
                          </div>
                        )}
                        <span>{option.label}</span>
                      </div>
                      {showCount && option.count !== undefined && (
                        <span className="text-xs text-white/30">
                          {option.count.toLocaleString('uk-UA')}
                        </span>
                      )}
                    </button>
                  );
                })
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default SelectFilter;
