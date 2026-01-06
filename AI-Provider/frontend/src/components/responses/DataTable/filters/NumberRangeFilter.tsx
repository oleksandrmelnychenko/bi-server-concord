import React, { useState, useCallback, useEffect } from 'react';
import { motion } from 'framer-motion';

interface NumberRangeFilterProps {
  min?: number;
  max?: number;
  value: { min?: number; max?: number };
  onChange: (value: { min?: number; max?: number }) => void;
  label?: string;
  step?: number;
  formatValue?: (value: number) => string;
  className?: string;
}

export const NumberRangeFilter: React.FC<NumberRangeFilterProps> = ({
  min,
  max,
  value,
  onChange,
  label,
  step = 1,
  formatValue = (v) => v.toLocaleString('uk-UA'),
  className = '',
}) => {
  const [localMin, setLocalMin] = useState<string>(value.min?.toString() ?? '');
  const [localMax, setLocalMax] = useState<string>(value.max?.toString() ?? '');
  const [showSlider, setShowSlider] = useState(false);

  useEffect(() => {
    setLocalMin(value.min?.toString() ?? '');
    setLocalMax(value.max?.toString() ?? '');
  }, [value.min, value.max]);

  const handleMinChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setLocalMin(newValue);

    const numValue = parseFloat(newValue);
    if (!isNaN(numValue) || newValue === '') {
      onChange({
        min: newValue === '' ? undefined : numValue,
        max: value.max,
      });
    }
  }, [onChange, value.max]);

  const handleMaxChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setLocalMax(newValue);

    const numValue = parseFloat(newValue);
    if (!isNaN(numValue) || newValue === '') {
      onChange({
        min: value.min,
        max: newValue === '' ? undefined : numValue,
      });
    }
  }, [onChange, value.min]);

  const handleClear = useCallback(() => {
    setLocalMin('');
    setLocalMax('');
    onChange({ min: undefined, max: undefined });
  }, [onChange]);

  const hasValue = value.min !== undefined || value.max !== undefined;

  return (
    <div className={`${className}`}>
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
        {/* Min Input */}
        <div className="relative flex-1">
          <input
            type="number"
            value={localMin}
            onChange={handleMinChange}
            placeholder="Від"
            min={min}
            max={max}
            step={step}
            className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white/90 placeholder-white/30 focus:outline-none focus:border-sky-500/50 transition-colors"
          />
        </div>

        <span className="text-white/30 text-sm">—</span>

        {/* Max Input */}
        <div className="relative flex-1">
          <input
            type="number"
            value={localMax}
            onChange={handleMaxChange}
            placeholder="До"
            min={min}
            max={max}
            step={step}
            className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white/90 placeholder-white/30 focus:outline-none focus:border-sky-500/50 transition-colors"
          />
        </div>
      </div>

      {/* Optional Slider */}
      {min !== undefined && max !== undefined && (
        <div className="mt-2">
          <button
            onClick={() => setShowSlider(!showSlider)}
            className="text-xs text-white/40 hover:text-white/60 transition-colors flex items-center gap-1"
          >
            <svg
              className={`w-3 h-3 transition-transform ${showSlider ? 'rotate-180' : ''}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
            {showSlider ? 'Сховати слайдер' : 'Показати слайдер'}
          </button>

          {showSlider && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="mt-3 px-1"
            >
              <div className="flex justify-between text-xs text-white/40 mb-1">
                <span>{formatValue(min)}</span>
                <span>{formatValue(max)}</span>
              </div>
              <div className="relative h-2 bg-white/10 rounded-full">
                <div
                  className="absolute h-full bg-sky-500/40 rounded-full"
                  style={{
                    left: `${((value.min ?? min) - min) / (max - min) * 100}%`,
                    right: `${100 - ((value.max ?? max) - min) / (max - min) * 100}%`,
                  }}
                />
              </div>
              <div className="relative mt-1">
                <input
                  type="range"
                  min={min}
                  max={max}
                  step={step}
                  value={value.min ?? min}
                  onChange={(e) => onChange({ min: parseFloat(e.target.value), max: value.max })}
                  className="absolute w-full h-2 appearance-none bg-transparent pointer-events-auto cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-sky-400 [&::-webkit-slider-thumb]:cursor-pointer"
                />
                <input
                  type="range"
                  min={min}
                  max={max}
                  step={step}
                  value={value.max ?? max}
                  onChange={(e) => onChange({ min: value.min, max: parseFloat(e.target.value) })}
                  className="absolute w-full h-2 appearance-none bg-transparent pointer-events-auto cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-sky-400 [&::-webkit-slider-thumb]:cursor-pointer"
                />
              </div>
            </motion.div>
          )}
        </div>
      )}
    </div>
  );
};

export default NumberRangeFilter;
