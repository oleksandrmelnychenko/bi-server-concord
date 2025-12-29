import React, { useState } from 'react';
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
} from 'chart.js';
import { Bar, Line, Pie, Doughnut } from 'react-chartjs-2';
import { ChartResponse, CHART_COLORS } from '../../../types/responses';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

const axisColor = '#94a3b8';
const gridColor = 'rgba(148, 163, 184, 0.15)';
const tooltipBg = 'rgba(15, 18, 24, 0.95)';

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
      };
    }

    return {
      ...dataset,
      data: safeData,
      backgroundColor: dataset.backgroundColor || colors[index % colors.length],
      borderColor: dataset.borderColor || colors[index % colors.length],
      borderWidth: dataset.borderWidth ?? 1,
    };
  });

  return newData;
};

const getBarLineOptions = (chartType: string): ChartOptions<'bar'> | ChartOptions<'line'> => ({
  responsive: true,
  maintainAspectRatio: false,
  indexAxis: chartType === 'horizontal-bar' ? 'y' : 'x',
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
    },
  },
  scales: {
    x: {
      grid: { color: gridColor },
      ticks: { color: axisColor },
    },
    y: {
      grid: { color: gridColor },
      ticks: { color: axisColor },
    },
  },
});

const getPieOptions = (): ChartOptions<'pie'> | ChartOptions<'doughnut'> => ({
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      display: true,
      position: 'bottom',
      labels: {
        padding: 16,
        usePointStyle: true,
        color: axisColor,
      },
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
    },
  },
});

export const SmartChart: React.FC<ChartResponse> = ({
  title,
  chartType,
  data,
  height = 300,
  expandable = true,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const coloredData = applyDefaultColors(data, chartType);

  if (!coloredData.labels.length || !coloredData.datasets.length) {
    return (
      <div className="smart-chart grok-card border-white/10 p-4 text-center text-slate-400">
        No chart data available.
      </div>
    );
  }

  const renderChart = () => {
    switch (chartType) {
      case 'bar':
      case 'horizontal-bar':
        return <Bar data={coloredData} options={getBarLineOptions(chartType) as ChartOptions<'bar'>} />;
      case 'line':
        return <Line data={coloredData} options={getBarLineOptions(chartType) as ChartOptions<'line'>} />;
      case 'pie':
        return <Pie data={coloredData} options={getPieOptions() as ChartOptions<'pie'>} />;
      case 'doughnut':
        return <Doughnut data={coloredData} options={getPieOptions() as ChartOptions<'doughnut'>} />;
      default:
        return <Bar data={coloredData} options={getBarLineOptions('bar') as ChartOptions<'bar'>} />;
    }
  };

  return (
    <>
      <div className="smart-chart grok-card border-white/10 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 bg-white/5 border-b border-white/10">
          <h3 className="font-semibold text-slate-100 text-sm flex items-center gap-2">
            <svg className="w-4 h-4 text-sky-200 thin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path d="M4 19h16" />
              <path d="M7 16V9" />
              <path d="M12 16V6" />
              <path d="M17 16v-5" />
            </svg>
            {title || 'Chart'}
          </h3>

          {expandable && (
            <button
              onClick={() => setIsExpanded(true)}
              className="p-2 text-slate-400 hover:text-slate-100 hover:bg-white/10 rounded transition-colors"
              title="Expand chart"
            >
              <svg className="w-4 h-4 thin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path d="M8 4H4v4" />
                <path d="M4 4l6 6" />
                <path d="M16 20h4v-4" />
                <path d="M20 20l-6-6" />
                <path d="M20 4h-4" />
                <path d="M20 4l-6 6" />
                <path d="M4 20h4" />
                <path d="M4 20l6-6" />
              </svg>
            </button>
          )}
        </div>

        {/* Chart */}
        <div className="p-4" style={{ height }}>
          {renderChart()}
        </div>
      </div>

      {/* Expanded Modal */}
      {isExpanded && (
        <div
          className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4"
          onClick={() => setIsExpanded(false)}
        >
          <div
            className="grok-panel rounded-2xl w-full max-w-4xl max-h-[90vh] overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
              <h3 className="text-base font-semibold text-slate-100">{title || 'Chart'}</h3>
              <button
                onClick={() => setIsExpanded(false)}
                className="p-2 text-slate-400 hover:text-slate-100 hover:bg-white/10 rounded-full transition-colors"
              >
                <svg className="w-5 h-5 thin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path d="M6 18L18 6" />
                  <path d="M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="p-6" style={{ height: '60vh' }}>
              {renderChart()}
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default SmartChart;
