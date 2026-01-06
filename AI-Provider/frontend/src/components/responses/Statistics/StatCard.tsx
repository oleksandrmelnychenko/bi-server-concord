import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import CountUp from 'react-countup';
import { StatCardData } from '../../../types/responses';
import { scaleIn } from '../../../utils/animations';
import type { Language } from '../../WelcomeMessage';

const statTranslations = {
  uk: {
    comparedToPrevious: 'порівняно з минулим періодом',
  },
  en: {
    comparedToPrevious: 'compared to previous period',
  },
};

// Enhanced icon set with more options
const Icons: Record<string, React.ReactNode> = {
  users: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M15 19.128a9.38 9.38 0 002.625.372 9.337 9.337 0 004.121-.952 4.125 4.125 0 00-7.533-2.493M15 19.128v-.003c0-1.113-.285-2.16-.786-3.07M15 19.128v.106A12.318 12.318 0 018.624 21c-2.331 0-4.512-.645-6.374-1.766l-.001-.109a6.375 6.375 0 0111.964-3.07M12 6.375a3.375 3.375 0 11-6.75 0 3.375 3.375 0 016.75 0zm8.25 2.25a2.625 2.625 0 11-5.25 0 2.625 2.625 0 015.25 0z" />
    </svg>
  ),
  money: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v12m-3-2.818l.879.659c1.171.879 3.07.879 4.242 0 1.172-.879 1.172-2.303 0-3.182C13.536 12.219 12.768 12 12 12c-.725 0-1.45-.22-2.003-.659-1.106-.879-1.106-2.303 0-3.182s2.9-.879 4.006 0l.415.33M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  chart: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
    </svg>
  ),
  box: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M20.25 7.5l-.625 10.632a2.25 2.25 0 01-2.247 2.118H6.622a2.25 2.25 0 01-2.247-2.118L3.75 7.5M10 11.25h4M3.375 7.5h17.25c.621 0 1.125-.504 1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125z" />
    </svg>
  ),
  truck: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 18.75a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m3 0h6m-9 0H3.375a1.125 1.125 0 01-1.125-1.125V14.25m17.25 4.5a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m3 0h1.125c.621 0 1.129-.504 1.09-1.124a17.902 17.902 0 00-3.213-9.193 2.056 2.056 0 00-1.58-.86H14.25M16.5 18.75h-2.25m0-11.177v-.958c0-.568-.422-1.048-.987-1.106a48.554 48.554 0 00-10.026 0 1.106 1.106 0 00-.987 1.106v7.635m12-6.677v6.677m0 4.5v-4.5m0 0h-12" />
    </svg>
  ),
  warning: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
    </svg>
  ),
  check: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  shopping: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 3h1.386c.51 0 .955.343 1.087.835l.383 1.437M7.5 14.25a3 3 0 00-3 3h15.75m-12.75-3h11.218c1.121-2.3 2.1-4.684 2.924-7.138a60.114 60.114 0 00-16.536-1.84M7.5 14.25L5.106 5.272M6 20.25a.75.75 0 11-1.5 0 .75.75 0 011.5 0zm12.75 0a.75.75 0 11-1.5 0 .75.75 0 011.5 0z" />
    </svg>
  ),
  clock: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  star: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M11.48 3.499a.562.562 0 011.04 0l2.125 5.111a.563.563 0 00.475.345l5.518.442c.499.04.701.663.321.988l-4.204 3.602a.563.563 0 00-.182.557l1.285 5.385a.562.562 0 01-.84.61l-4.725-2.885a.563.563 0 00-.586 0L6.982 20.54a.562.562 0 01-.84-.61l1.285-5.386a.562.562 0 00-.182-.557l-4.204-3.602a.563.563 0 01.321-.988l5.518-.442a.563.563 0 00.475-.345L11.48 3.5z" />
    </svg>
  ),
  percent: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M6.75 6.75a1.5 1.5 0 100 3 1.5 1.5 0 000-3zm10.5 7.5a1.5 1.5 0 100 3 1.5 1.5 0 000-3zM6 18L18 6" />
    </svg>
  ),
};

// Simple sparkline component
interface SparklineProps {
  data: number[];
  color?: string;
  height?: number;
  width?: number;
}

const Sparkline: React.FC<SparklineProps> = ({
  data,
  color = '#7c3aed',
  height = 32,
  width = 80,
}) => {
  const points = useMemo(() => {
    if (!data || data.length < 2) return '';

    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    const step = width / (data.length - 1);

    return data
      .map((value, index) => {
        const x = index * step;
        const y = height - ((value - min) / range) * (height - 4) - 2;
        return `${x},${y}`;
      })
      .join(' ');
  }, [data, height, width]);

  const areaPoints = useMemo(() => {
    if (!points) return '';
    return `0,${height} ${points} ${width},${height}`;
  }, [points, height, width]);

  if (!data || data.length < 2) return null;

  return (
    <svg width={width} height={height} className="overflow-visible">
      {/* Area fill */}
      <motion.polygon
        points={areaPoints}
        fill={`${color}15`}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      />
      {/* Line */}
      <motion.polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: 1, opacity: 1 }}
        transition={{ duration: 0.8, ease: 'easeOut' }}
      />
      {/* End dot */}
      <motion.circle
        cx={width}
        cy={parseFloat(points.split(' ').pop()?.split(',')[1] || '0')}
        r={3}
        fill={color}
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ delay: 0.6, duration: 0.2 }}
      />
    </svg>
  );
};

// Progress bar component
interface ProgressBarProps {
  value: number;
  max?: number;
  color?: string;
  showLabel?: boolean;
}

const ProgressBar: React.FC<ProgressBarProps> = ({
  value,
  max = 100,
  color = '#7c3aed',
  showLabel = true,
}) => {
  const percentage = Math.min((value / max) * 100, 100);

  return (
    <div className="mt-3">
      <div className="relative h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <motion.div
          className="absolute inset-y-0 left-0 rounded-full"
          style={{ backgroundColor: color }}
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
        />
      </div>
      {showLabel && (
        <div className="flex justify-between mt-1 text-xs text-slate-500">
          <span>{value.toLocaleString('uk-UA')}</span>
          <span>{max.toLocaleString('uk-UA')}</span>
        </div>
      )}
    </div>
  );
};

const formatCompact = (value: number): string => {
  if (Math.abs(value) >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(1)}B`;
  }
  if (Math.abs(value) >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1)}M`;
  }
  if (Math.abs(value) >= 1_000) {
    return `${(value / 1_000).toFixed(1)}K`;
  }
  return value.toLocaleString('uk-UA');
};

// Extended StatCardData interface
interface EnhancedStatCardData extends StatCardData {
  sparklineData?: number[];
  progress?: { value: number; max: number };
  subtitle?: string;
  color?: string;
  animate?: boolean;
  size?: 'sm' | 'md' | 'lg';
  language?: Language;
}

export const StatCard: React.FC<EnhancedStatCardData> = ({
  value,
  label,
  icon,
  trend,
  highlight = false,
  format,
  sparklineData,
  progress,
  subtitle,
  color,
  animate = true,
  size = 'md',
  language = 'uk',
}) => {
  const t = statTranslations[language];
  const numericValue = typeof value === 'number' ? value : parseFloat(String(value)) || 0;
  const isNumber = typeof value === 'number' || !isNaN(numericValue);

  // Determine color based on trend or custom color
  const cardColor = color || (highlight ? '#facc15' : '#7c3aed');
  const trendColor = trend?.direction === 'up' ? '#34d399' : '#f87171';

  // Size configurations
  const sizeClasses = {
    sm: 'p-3',
    md: 'p-4',
    lg: 'p-6',
  };

  const valueSizeClasses = {
    sm: 'text-xl',
    md: 'text-2xl',
    lg: 'text-3xl',
  };

  const renderValue = () => {
    if (!isNumber || !animate) {
      if (format === 'currency') {
        return `${formatCompact(numericValue)} ₴`;
      }
      if (format === 'percent') {
        return `${numericValue.toFixed(1)}%`;
      }
      return formatCompact(numericValue);
    }

    const suffix = format === 'currency' ? ' ₴' : format === 'percent' ? '%' : '';
    const decimals = format === 'percent' ? 1 : 0;

    // Determine if we need to compact the number
    let displayValue = numericValue;
    let compactSuffix = '';

    if (Math.abs(numericValue) >= 1_000_000_000) {
      displayValue = numericValue / 1_000_000_000;
      compactSuffix = 'B';
    } else if (Math.abs(numericValue) >= 1_000_000) {
      displayValue = numericValue / 1_000_000;
      compactSuffix = 'M';
    } else if (Math.abs(numericValue) >= 10_000) {
      displayValue = numericValue / 1_000;
      compactSuffix = 'K';
    }

    return (
      <CountUp
        end={displayValue}
        duration={1.5}
        decimals={compactSuffix ? 1 : decimals}
        separator=" "
        suffix={`${compactSuffix}${suffix}`}
        useEasing
      />
    );
  };

  return (
    <motion.div
      variants={scaleIn}
      initial="initial"
      animate="animate"
      whileHover={{ scale: 1.02, transition: { duration: 0.2 } }}
      className={`stat-card ${sizeClasses[size]} rounded-2xl border transition-all bg-white border-slate-200`}
      style={cardColor && !highlight ? { borderColor: `${cardColor}30`, backgroundColor: `${cardColor}08` } : undefined}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          {/* Value */}
          <div className={`${valueSizeClasses[size]} font-semibold text-slate-900`}>
            {renderValue()}
          </div>

          {/* Label */}
          <div className="text-sm text-slate-600 mt-1 truncate">{label}</div>

          {/* Subtitle */}
          {subtitle && (
            <div className="text-xs text-slate-500 mt-0.5">{subtitle}</div>
          )}
        </div>

        {/* Icon or Sparkline */}
        <div className="flex flex-col items-end gap-2">
          {icon && (
            <motion.div
              className={`p-2.5 rounded-xl border ${
                highlight
                  ? 'border-amber-200 bg-amber-50 text-amber-700'
                  : 'border-slate-200 bg-slate-50 text-slate-500'
              }`}
              style={cardColor && !highlight ? { borderColor: `${cardColor}30`, backgroundColor: `${cardColor}15`, color: cardColor } : undefined}
              whileHover={{ scale: 1.1, rotate: 5 }}
              transition={{ duration: 0.2 }}
            >
              {Icons[icon] || Icons.chart}
            </motion.div>
          )}

          {/* Sparkline */}
          {sparklineData && sparklineData.length > 1 && (
            <Sparkline
              data={sparklineData}
              color={cardColor || (highlight ? '#facc15' : '#64748b')}
              width={size === 'lg' ? 100 : size === 'sm' ? 60 : 80}
              height={size === 'lg' ? 40 : size === 'sm' ? 24 : 32}
            />
          )}
        </div>
      </div>

      {/* Trend */}
      {trend && (
        <motion.div
          className="mt-3 flex items-center gap-2 text-xs text-slate-600"
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <span
            className="flex items-center gap-1 font-medium"
            style={{ color: trendColor }}
          >
            <motion.svg
              className="w-3.5 h-3.5"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth={2}
              initial={{ y: trend.direction === 'up' ? 5 : -5 }}
              animate={{ y: 0 }}
              transition={{ delay: 0.4, type: 'spring', stiffness: 200 }}
            >
              {trend.direction === 'up' ? (
                <path strokeLinecap="round" strokeLinejoin="round" d="M7 14l5-5 5 5" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" d="M7 10l5 5 5-5" />
              )}
            </motion.svg>
            {Math.abs(trend.value)}%
          </span>
          <span className="text-slate-400">{t.comparedToPrevious}</span>
        </motion.div>
      )}

      {/* Progress Bar */}
      {progress && (
        <ProgressBar
          value={progress.value}
          max={progress.max}
          color={cardColor || (highlight ? '#facc15' : '#64748b')}
        />
      )}
    </motion.div>
  );
};

export default StatCard;
