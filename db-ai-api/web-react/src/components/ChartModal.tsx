import React, { useEffect, useRef, useCallback } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar, Doughnut } from 'react-chartjs-2';
import type { ChartType } from '../types';
import { fetchYearlySales, fetchTopProducts, fetchDebtSummary } from '../services/api';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, BarElement, ArcElement, Title, Tooltip, Legend);

interface ChartModalProps {
  chartType: ChartType | null;
  onClose: () => void;
}

const chartTitles: Record<ChartType, string> = {
  sales_yearly: 'Продажі по роках',
  top_products: 'Топ-10 товарів',
  debts: 'Борги по роках',
};

export const ChartModal: React.FC<ChartModalProps> = ({ chartType, onClose }) => {
  const [chartData, setChartData] = React.useState<{
    type: 'bar' | 'doughnut';
    data: unknown;
    options: unknown;
  } | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const modalRef = useRef<HTMLDivElement>(null);

  const loadChartData = useCallback(async () => {
    if (!chartType) return;

    setLoading(true);
    setError(null);

    try {
      if (chartType === 'sales_yearly') {
        const data = await fetchYearlySales();
        setChartData({
          type: 'bar',
          data: {
            labels: data.map((d) => d.year).reverse(),
            datasets: [
              {
                label: 'Продажів',
                data: data.map((d) => d.total_sales).reverse(),
                backgroundColor: 'rgba(249, 115, 22, 0.8)',
                borderRadius: 8,
              },
            ],
          },
          options: getBarOptions(),
        });
      } else if (chartType === 'top_products') {
        const data = await fetchTopProducts(10);
        setChartData({
          type: 'bar',
          data: {
            labels: data.map((d) => d.product_name?.substring(0, 20) || ''),
            datasets: [
              {
                label: 'Продано (шт)',
                data: data.map((d) => d.total_qty),
                backgroundColor: 'rgba(234, 88, 12, 0.8)',
                borderRadius: 6,
              },
            ],
          },
          options: getBarOptions('y'),
        });
      } else if (chartType === 'debts') {
        const data = await fetchDebtSummary();
        setChartData({
          type: 'doughnut',
          data: {
            labels: data.by_year?.map((d) => d.year) || [],
            datasets: [
              {
                data: data.by_year?.map((d) => d.total_amount) || [],
                backgroundColor: [
                  'rgba(249, 115, 22, 0.9)',
                  'rgba(234, 88, 12, 0.8)',
                  'rgba(194, 65, 12, 0.8)',
                  'rgba(251, 146, 60, 0.8)',
                  'rgba(253, 186, 116, 0.8)',
                ],
              },
            ],
          },
          options: getDoughnutOptions(),
        });
      }
    } catch (err) {
      setError('Помилка завантаження даних');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [chartType]);

  useEffect(() => {
    if (chartType) {
      loadChartData();
    }
  }, [chartType, loadChartData]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === modalRef.current) {
      onClose();
    }
  };

  if (!chartType) return null;

  return (
    <div
      className="fixed inset-0 bg-black/30 backdrop-blur-sm flex items-center justify-center z-50 p-4
               animate-fade-in"
      ref={modalRef}
      onClick={handleBackdropClick}
    >
      <div className="bg-white rounded-2xl w-full max-w-4xl max-h-[90vh] overflow-hidden
                    shadow-2xl animate-slide-up border border-surface-border">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-surface-border">
          <h3 className="text-xl font-semibold text-content-primary">
            {chartTitles[chartType]}
          </h3>
          <button
            onClick={onClose}
            className="w-10 h-10 rounded-full bg-surface-secondary flex items-center justify-center
                     text-content-muted hover:bg-surface-hover
                     transition-all duration-200"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Chart Content */}
        <div className="p-6 h-[500px]">
          {loading && (
            <div className="h-full flex items-center justify-center">
              <div className="flex gap-2">
                <span className="w-3 h-3 bg-gray-400 rounded-full animate-bounce-dot" style={{ animationDelay: '-0.32s' }} />
                <span className="w-3 h-3 bg-gray-400 rounded-full animate-bounce-dot" style={{ animationDelay: '-0.16s' }} />
                <span className="w-3 h-3 bg-gray-400 rounded-full animate-bounce-dot" />
              </div>
            </div>
          )}
          {error && (
            <div className="h-full flex items-center justify-center">
              <p className="text-red-500">{error}</p>
            </div>
          )}
          {chartData && !loading && !error && (
            <>
              {chartData.type === 'bar' && (
                <Bar data={chartData.data as never} options={chartData.options as never} />
              )}
              {chartData.type === 'doughnut' && (
                <Doughnut data={chartData.data as never} options={chartData.options as never} />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

function getBarOptions(indexAxis: 'x' | 'y' = 'x') {
  return {
    responsive: true,
    maintainAspectRatio: false,
    indexAxis,
    plugins: {
      legend: {
        display: false,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: { color: 'rgba(0, 0, 0, 0.05)' },
        ticks: { color: '#6b7280' },
      },
      x: {
        grid: { color: 'rgba(0, 0, 0, 0.05)' },
        ticks: { color: '#6b7280' },
      },
    },
  };
}

function getDoughnutOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    cutout: '60%',
    plugins: {
      legend: {
        display: true,
        position: 'bottom' as const,
        labels: { color: '#6b7280', padding: 20 },
      },
    },
  };
}
