import React, { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
  Filler,
} from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';
import { Bar, Line, Pie, Doughnut } from 'react-chartjs-2';
import { ChartResponse, CHART_COLORS } from '../../../types/responses';
import { scaleIn } from '../../../utils/animations';
import type { Language } from '../../WelcomeMessage';

const chartTranslations = {
  uk: {
    chart: 'Графік',
    exportPng: 'Експортувати як PNG',
    expand: 'Розгорнути',
    resetZoom: 'Скинути масштаб',
    export: 'Експортувати',
    zoomHint: '{t.zoomHint}',
    bar: 'Стовпці',
    horizontalBar: 'Горизонтальні',
    line: 'Лінія',
    pie: 'Кругова',
    doughnut: 'Пончик',
  },
  en: {
    chart: 'Chart',
    exportPng: 'Export as PNG',
    expand: 'Expand',
    resetZoom: 'Reset zoom',
    export: 'Export',
    zoomHint: 'Use mouse wheel to zoom • Drag to pan',
    bar: 'Bar',
    horizontalBar: 'Horizontal',
    line: 'Line',
    pie: 'Pie',
    doughnut: 'Doughnut',
  },
};

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  zoomPlugin
);

const axisColor = '#1e293b';
const gridColor = 'rgba(148, 163, 184, 0.12)';
const tooltipBg = 'rgba(15, 23, 42, 0.95)';

// Chart type icons
const chartTypeIcons: Record<string, React.ReactNode> = {
  bar: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 19h16M7 16V9M12 16V6M17 16v-5" />
    </svg>
  ),
  'horizontal-bar': (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 4v16M8 7h7M8 12h10M8 17h5" />
    </svg>
  ),
  line: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 17l6-6 4 4 8-8" />
    </svg>
  ),
  pie: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 2a10 10 0 0 1 10 10H12V2z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 12v10a10 10 0 0 1-10-10h10z" />
    </svg>
  ),
  doughnut: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <circle cx="12" cy="12" r="10" strokeWidth={1.5} />
      <circle cx="12" cy="12" r="5" strokeWidth={1.5} />
    </svg>
  ),
};

const applyDefaultColors = (data: ChartResponse['data'], chartType: ChartResponse['chartType']) => {
  if (!data) {
    return { labels: [], datasets: [] };
  }

  const safeLabels = Array.isArray(data.labels) ? data.labels : [];
  const safeDatasets = Array.isArray(data.datasets) ? data.datasets.filter(Boolean) : [];

  if (safeLabels.length === 0 || safeDatasets.length === 0) {
    return { labels: [], datasets: [] };
  }

  const newData = { labels: safeLabels, datasets: [] as typeof safeDatasets };

  newData.datasets = safeDatasets.map((dataset, index) => {
    if (!dataset) return { data: [], label: '' };

    const colors = CHART_COLORS.palette;
    const labelCount = safeLabels.length;
    const safeData = Array.isArray(dataset.data) ? dataset.data : [];

    if (chartType === 'pie' || chartType === 'doughnut') {
      return {
        ...dataset,
        data: safeData,
        backgroundColor: dataset.backgroundColor || colors.slice(0, labelCount),
        borderColor: dataset.borderColor || '#0b0d12',
        borderWidth: dataset.borderWidth ?? 1,
        hoverOffset: 6,
      };
    }

    if (chartType === 'line') {
      const color = colors[index % colors.length];
      return {
        ...dataset,
        data: safeData,
        backgroundColor: `${color}20`,
        borderColor: color,
        borderWidth: 2,
        pointBackgroundColor: color,
        pointBorderColor: '#0b0d12',
        pointBorderWidth: 2,
        pointRadius: 4,
        pointHoverRadius: 6,
        fill: true,
        tension: 0.3,
      };
    }

    return {
      ...dataset,
      data: safeData,
      backgroundColor: dataset.backgroundColor || colors[index % colors.length],
      borderColor: dataset.borderColor || colors[index % colors.length],
      borderWidth: dataset.borderWidth ?? 0,
      borderRadius: 6,
      hoverBackgroundColor: `${colors[index % colors.length]}cc`,
    };
  });

  return newData;
};

const getBarLineOptions = (chartType: string, enableZoom = false): ChartOptions<'bar'> | ChartOptions<'line'> => ({
  responsive: true,
  maintainAspectRatio: false,
  indexAxis: chartType === 'horizontal-bar' ? 'y' : 'x',
  animation: {
    duration: 800,
    easing: 'easeOutQuart',
  },
  plugins: {
    legend: {
      display: false,
    },
    title: {
      display: false,
    },
    tooltip: {
      backgroundColor: tooltipBg,
      titleColor: '#e2e8f0',
      bodyColor: '#cbd5e1',
      borderColor: 'rgba(148, 163, 184, 0.2)',
      borderWidth: 1,
      padding: 12,
      cornerRadius: 10,
      displayColors: true,
      boxPadding: 4,
    },
    zoom: enableZoom ? {
      pan: {
        enabled: true,
        mode: 'x',
      },
      zoom: {
        wheel: {
          enabled: true,
        },
        pinch: {
          enabled: true,
        },
        mode: 'x',
      },
    } : undefined,
  },
  scales: {
    x: {
      grid: { color: gridColor },
      ticks: {
        color: axisColor,
        font: { size: 11 },
        maxRotation: 45,
      },
    },
    y: {
      grid: { color: gridColor },
      ticks: {
        color: axisColor,
        font: { size: 11 },
      },
    },
  },
  interaction: {
    intersect: false,
    mode: 'index',
  },
});

const getPieOptions = (): ChartOptions<'pie'> | ChartOptions<'doughnut'> => ({
  responsive: true,
  maintainAspectRatio: false,
  animation: {
    animateRotate: true,
    animateScale: true,
    duration: 800,
    easing: 'easeOutQuart',
  },
  plugins: {
    legend: {
      display: true,
      position: 'bottom',
      labels: {
        padding: 20,
        usePointStyle: true,
        color: '#1e293b',
        font: { size: 12, weight: 500 },
      },
    },
    title: {
      display: false,
    },
    tooltip: {
      backgroundColor: tooltipBg,
      titleColor: '#f8fafc',
      bodyColor: '#e2e8f0',
      borderColor: 'rgba(148, 163, 184, 0.2)',
      borderWidth: 1,
      padding: 12,
      cornerRadius: 10,
    },
  },
});

interface EnhancedChartResponse extends ChartResponse {
  showLegend?: boolean;
  enableZoom?: boolean;
  onExport?: (imageData: string) => void;
  language?: Language;
}

export const SmartChart: React.FC<EnhancedChartResponse> = ({
  title,
  chartType,
  data,
  height = 300,
  expandable = true,
  showLegend = true,
  enableZoom = false,
  language = 'uk',
}) => {
  const t = chartTranslations[language];
  const [isExpanded, setIsExpanded] = useState(false);
  const [activeChartType, setActiveChartType] = useState(chartType);
  const [isExporting, setIsExporting] = useState(false);
  const chartRef = useRef<any>(null);
  const coloredData = applyDefaultColors(data, activeChartType);

  // Available chart type switches
  const chartTypeSwitches: Array<{ type: ChartResponse['chartType']; label: string }> = [
    { type: 'bar', label: t.bar },
    { type: 'horizontal-bar', label: t.horizontalBar },
    { type: 'line', label: t.line },
    { type: 'pie', label: t.pie },
    { type: 'doughnut', label: t.doughnut },
  ];

  // Export chart as image
  const handleExport = useCallback(async () => {
    if (!chartRef.current) return;

    setIsExporting(true);
    try {
      const canvas = chartRef.current.canvas;
      const imageData = canvas.toDataURL('image/png');

      // Create download link
      const link = document.createElement('a');
      link.download = `${title || 'chart'}_${new Date().toISOString().split('T')[0]}.png`;
      link.href = imageData;
      link.click();
    } catch (error) {
      console.error('Failed to export chart:', error);
    } finally {
      setIsExporting(false);
    }
  }, [title]);

  // Reset zoom
  const handleResetZoom = useCallback(() => {
    if (chartRef.current) {
      chartRef.current.resetZoom();
    }
  }, []);

  if (!coloredData.labels.length || !coloredData.datasets.length) {
    return (
      <motion.div
        variants={scaleIn}
        initial="initial"
        animate="animate"
        className="smart-chart bg-white/[0.03] border border-white/10 rounded-xl p-8 text-center"
      >
        <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-white/5 flex items-center justify-center">
          <svg className="w-6 h-6 text-white/30" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 19h16M7 16V9M12 16V6M17 16v-5" />
          </svg>
        </div>
        <h3 className="text-white/60 font-medium">Немає даних для графіку</h3>
        <p className="text-white/40 text-sm mt-1">Спробуйте інший запит</p>
      </motion.div>
    );
  }

  const renderChart = (zoomEnabled = false) => {
    const options = activeChartType === 'pie' || activeChartType === 'doughnut'
      ? getPieOptions()
      : getBarLineOptions(activeChartType, zoomEnabled);

    switch (activeChartType) {
      case 'bar':
      case 'horizontal-bar':
        return <Bar ref={chartRef} data={coloredData} options={options as ChartOptions<'bar'>} />;
      case 'line':
        return <Line ref={chartRef} data={coloredData} options={options as ChartOptions<'line'>} />;
      case 'pie':
        return <Pie ref={chartRef} data={coloredData} options={options as ChartOptions<'pie'>} />;
      case 'doughnut':
        return <Doughnut ref={chartRef} data={coloredData} options={options as ChartOptions<'doughnut'>} />;
      default:
        return <Bar ref={chartRef} data={coloredData} options={options as ChartOptions<'bar'>} />;
    }
  };

  return (
    <>
      <motion.div
        variants={scaleIn}
        initial="initial"
        animate="animate"
        className="smart-chart bg-white border border-slate-200/60 rounded-xl overflow-hidden shadow-sm"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100">
          <h3 className="font-semibold text-slate-800 text-sm flex items-center gap-2">
            <span className="text-slate-500">
              {chartTypeIcons[activeChartType] || chartTypeIcons.bar}
            </span>
            {title || t.chart}
          </h3>

          <div className="flex items-center gap-1">
            {/* Export Button */}
            <motion.button
              onClick={handleExport}
              disabled={isExporting}
              className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-50 rounded-lg transition-colors disabled:opacity-50"
              title={t.exportPng}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {isExporting ? (
                <motion.svg
                  className="w-4 h-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </motion.svg>
              ) : (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
              )}
            </motion.button>

            {/* Expand Button */}
            {expandable && (
              <motion.button
                onClick={() => setIsExpanded(true)}
                className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-50 rounded-lg transition-colors"
                title={t.expand}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                </svg>
              </motion.button>
            )}
          </div>
        </div>

        {/* Chart Type Switcher (hidden on small charts) */}
        {showLegend && coloredData.datasets.length === 1 && (
          <div className="flex items-center gap-1 px-4 py-2 border-b border-slate-200 overflow-x-auto bg-white">
            {chartTypeSwitches.map(({ type, label }) => (
              <motion.button
                key={type}
                onClick={() => setActiveChartType(type)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-colors ${
                  activeChartType === type
                    ? 'bg-violet-50 text-violet-800 border border-violet-200'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                {chartTypeIcons[type]}
                {label}
              </motion.button>
            ))}
          </div>
        )}

        {/* Chart */}
        <div className="p-4" style={{ height }}>
          {renderChart(enableZoom)}
        </div>
      </motion.div>

      {/* Expanded Modal */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4"
            onClick={() => setIsExpanded(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-gray-900 border border-white/10 rounded-2xl w-full max-w-5xl max-h-[90vh] overflow-hidden shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Modal Header */}
              <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 bg-white/[0.02]">
              <div className="flex items-center gap-3">
                <span className="text-violet-700">
                  {chartTypeIcons[activeChartType] || chartTypeIcons.bar}
                </span>
                <h3 className="text-lg font-semibold text-white/90">{title || t.chart}</h3>
              </div>

                <div className="flex items-center gap-2">
                  {/* Chart Type Buttons */}
                  <div className="flex items-center gap-1 mr-4">
                    {chartTypeSwitches.map(({ type, label }) => (
                      <button
                        key={type}
                        onClick={() => setActiveChartType(type)}
                className={`p-2 rounded-lg transition-colors ${
                  activeChartType === type
                    ? 'bg-violet-50 text-violet-800'
                    : 'text-slate-400 hover:text-slate-700 hover:bg-slate-100'
                }`}
                title={label}
              >
                        {chartTypeIcons[type]}
                      </button>
                    ))}
                  </div>

                  {/* Reset Zoom Button */}
                  <button
                    onClick={handleResetZoom}
                    className="p-2 text-white/40 hover:text-white/80 hover:bg-white/5 rounded-lg transition-colors"
                    title={t.resetZoom}
                  >
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10H7" />
                    </svg>
                  </button>

                  {/* Export Button */}
                  <button
                    onClick={handleExport}
                    className="p-2 text-white/40 hover:text-white/80 hover:bg-white/5 rounded-lg transition-colors"
                    title={t.export}
                  >
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                  </button>

                  {/* Close Button */}
                  <button
                    onClick={() => setIsExpanded(false)}
                    className="p-2 text-white/40 hover:text-white/80 hover:bg-white/5 rounded-lg transition-colors ml-2"
                  >
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </div>

              {/* Modal Chart */}
              <div className="p-6" style={{ height: '70vh' }}>
                {renderChart(true)}
              </div>

              {/* Zoom hint */}
              <div className="px-6 pb-4 text-center">
                <span className="text-xs text-white/30">
                  {t.zoomHint}
                </span>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default SmartChart;
