import React, { useMemo } from 'react';
import { Bar, Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { useDashboardSocket, HistoryPoint, ManagerData, StorageData } from '../hooks/useDashboardSocket';
import logoSvg from '../assets/logo.svg';

// Ukrainian translations
const translations = {
  title: 'Панель моніторингу',
  subtitle: 'Симуляція бізнес-метрик в реальному часі',
  connected: 'Підключено',
  disconnected: 'Відключено',
  reconnect: 'Перепідключити',
  connecting: 'Підключення до сервера...',
  inventory: 'Запаси',
  revenueToday: 'Дохід сьогодні',
  outstandingDebt: 'Заборгованість',
  changeToday: 'Зміна за сьогодні',
  ordersToday: 'замовлень сьогодні',
  footer: 'Дані оновлюються кожні 3 секунди. Значення симульовані на основі реалістичних патернів.',
  // New translations
  managerSales: 'Продажі менеджерів сьогодні',
  topManagerMonth: 'TOP менеджер місяця',
  worstManager: 'Найгірший менеджер',
  storageInventory: 'Запаси по складах',
  orders: 'замовлень',
  ecommerce: 'E-commerce',
  defective: 'Брак',
  units: 'од.',
};

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface MetricCardProps {
  title: string;
  value: number;
  change: number;
  changeLabel: string;
  history: HistoryPoint[];
  color: 'blue' | 'green' | 'orange' | 'violet' | 'cyan';
  format?: 'number' | 'currency';
}

const formatNumber = (value: number, format: 'number' | 'currency' = 'number'): string => {
  if (format === 'currency') {
    if (value >= 1000000) {
      return `${(value / 1000000).toFixed(2)}M UAH`;
    }
    if (value >= 1000) {
      return `${(value / 1000).toFixed(1)}K UAH`;
    }
    return `${value.toLocaleString('uk-UA')} UAH`;
  }
  return value.toLocaleString('uk-UA');
};

const colorConfig = {
  blue: {
    chart: 'rgba(59, 130, 246, 1)',
    accent: 'bg-blue-400',
  },
  green: {
    chart: 'rgba(16, 185, 129, 1)',
    accent: 'bg-emerald-400',
  },
  orange: {
    chart: 'rgba(249, 115, 22, 1)',
    accent: 'bg-orange-400',
  },
  violet: {
    chart: 'rgba(139, 92, 246, 1)',
    accent: 'bg-violet-400',
  },
  cyan: {
    chart: 'rgba(6, 182, 212, 1)',
    accent: 'bg-cyan-400',
  },
};

// Business hours labels (9:00 - 18:00)
const businessHours = ['9', '10', '11', '12', '13', '14', '15', '16', '17', '18'];

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  change,
  changeLabel,
  history,
  color,
  format = 'number',
}) => {
  const colors = colorConfig[color];
  const isPositive = change >= 0;

  // Create gradient for area fill
  const createGradient = (ctx: CanvasRenderingContext2D, chartArea: any) => {
    const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
    gradient.addColorStop(0, colors.chart.replace('1)', '0.3)'));
    gradient.addColorStop(1, colors.chart.replace('1)', '0.02)'));
    return gradient;
  };

  const chartData = useMemo(() => ({
    labels: history.length > 0 ? history.map((p) => p.time) : businessHours,
    datasets: [
      {
        data: history.map((p) => p.value),
        borderColor: colors.chart,
        backgroundColor: (context: any) => {
          const { ctx, chartArea } = context.chart;
          if (!chartArea) return colors.chart.replace('1)', '0.1)');
          return createGradient(ctx, chartArea);
        },
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 4,
        pointHoverBackgroundColor: colors.chart,
        pointHoverBorderColor: '#fff',
        pointHoverBorderWidth: 2,
      },
    ],
  }), [history, colors]);

  const chartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        enabled: true,
        backgroundColor: '#1e293b',
        titleColor: '#94a3b8',
        titleFont: { size: 10 },
        bodyColor: '#fff',
        bodyFont: { size: 12, weight: '600' },
        padding: 8,
        displayColors: false,
        callbacks: {
          title: (items: any) => items[0]?.label || '',
          label: (context: any) => formatNumber(context.raw, format),
        },
      },
    },
    scales: {
      x: {
        display: true,
        grid: { display: false },
        border: { display: false },
        ticks: {
          display: true,
          color: '#cbd5e1',
          font: { size: 9 },
          maxRotation: 0,
          autoSkip: true,
          maxTicksLimit: 6,
        },
      },
      y: {
        display: false,
        beginAtZero: false,
      },
    },
    interaction: {
      intersect: false,
      mode: 'index' as const,
    },
  }), [format]);

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 flex flex-col hover:border-slate-300 transition-colors">
      {/* Header */}
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-medium text-slate-400 uppercase tracking-wide">{title}</span>
        <div
          className={`text-xs font-medium ${
            isPositive ? 'text-emerald-600' : 'text-red-500'
          }`}
        >
          {isPositive ? '+' : ''}{formatNumber(change, format)}
        </div>
      </div>

      {/* Value */}
      <div className="text-2xl font-semibold text-slate-900 mb-0.5">
        {formatNumber(value, format)}
      </div>
      <div className="text-[11px] text-slate-400 mb-2">{changeLabel}</div>

      {/* Chart - Area */}
      <div className="h-20 mt-auto">
        {history.length > 1 && <Line data={chartData} options={chartOptions} />}
      </div>
    </div>
  );
};

// Manager card with comparison bar chart
interface ManagerCardProps {
  manager: ManagerData;
  rank: number;
  isTopManagerMonth: boolean;
  allManagers: ManagerData[];
}

const ManagerCard: React.FC<ManagerCardProps> = ({ manager, rank, isTopManagerMonth, allManagers }) => {
  const isPositive = manager.change >= 0;

  // Create bar chart data - all managers, highlight current one
  const chartData = useMemo(() => ({
    labels: allManagers.map((_, i) => `${i + 1}`),
    datasets: [
      {
        data: allManagers.map((m) => m.orders_today),
        backgroundColor: allManagers.map((m) =>
          m.id === manager.id
            ? (isTopManagerMonth ? '#f59e0b' : '#8b5cf6')  // Highlighted - current manager
            : '#e2e8f0'  // Gray - other managers
        ),
        borderRadius: 2,
        barPercentage: 0.7,
      },
    ],
  }), [allManagers, manager.id, isTopManagerMonth]);

  const chartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false }, tooltip: { enabled: false } },
    scales: {
      x: { display: false },
      y: { display: false, beginAtZero: true },
    },
  }), []);

  return (
    <div className={`relative py-3 border-b transition-all ${
      isTopManagerMonth ? 'border-amber-200 bg-amber-50/20' : 'border-slate-100 hover:bg-slate-50/50'
    }`}>
      <div className="flex items-center gap-2.5">
        {/* Rank: dot + number */}
        <div className="flex items-center gap-1.5 flex-shrink-0 min-w-[28px]">
          <span className={`w-1 h-1 rounded-full ${isTopManagerMonth ? 'bg-amber-400' : 'bg-violet-400'}`}></span>
          <span className={`text-xs font-medium ${isTopManagerMonth ? 'text-amber-600' : 'text-slate-500'}`}>{rank}</span>
        </div>

        {/* Small crown for TOP manager */}
        {isTopManagerMonth && (
          <svg className="w-3.5 h-3.5 text-amber-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
            <path d="M10 2l2.5 5 5.5.75-4 3.75 1 5.5L10 14.25 4.75 17l1-5.5-4-3.75L7.5 7z" />
          </svg>
        )}

        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-slate-700 truncate" title={manager.name}>
              {manager.name}
            </span>
            {isTopManagerMonth && (
              <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-amber-100 text-amber-700 font-medium">TOP</span>
            )}
          </div>
        </div>

        {/* Orders count with change as superscript */}
        <div className="flex flex-col items-end flex-shrink-0 min-w-[45px]">
          <span className={`text-[10px] font-medium leading-none h-3 ${
            manager.change === 0 ? 'invisible' : isPositive ? 'text-emerald-500' : 'text-red-400'
          }`}>
            {manager.change === 0 ? '0' : `${isPositive ? '+' : ''}${manager.change}`}
          </span>
          <span className="text-lg font-semibold text-slate-900 leading-tight">{manager.orders_today}</span>
        </div>

        {/* Comparison bar chart - all managers, current highlighted */}
        <div className="w-24 h-8 flex-shrink-0">
          {allManagers.length > 0 && <Bar data={chartData} options={chartOptions} />}
        </div>
      </div>
    </div>
  );
};

// Storage card with comparison bar chart
interface StorageCardProps {
  storage: StorageData;
  allStorages: StorageData[];
  rank: number;
}

const StorageCard: React.FC<StorageCardProps> = ({ storage, allStorages, rank }) => {
  const isPositive = storage.change >= 0;

  // Create bar chart data - all storages, highlight current one
  const chartData = useMemo(() => ({
    labels: allStorages.map((_, i) => `${i + 1}`),
    datasets: [
      {
        data: allStorages.map((s) => s.total_stock),
        backgroundColor: allStorages.map((s) =>
          s.id === storage.id
            ? (storage.is_defective ? '#ef4444' : storage.is_ecommerce ? '#10b981' : '#06b6d4')  // Highlighted
            : '#e2e8f0'  // Gray - other storages
        ),
        borderRadius: 2,
        barPercentage: 0.7,
      },
    ],
  }), [allStorages, storage.id, storage.is_defective, storage.is_ecommerce]);

  const chartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false }, tooltip: { enabled: false } },
    scales: {
      x: { display: false },
      y: { display: false, beginAtZero: true },
    },
  }), []);

  // Small icon based on storage type
  const getIcon = () => {
    if (storage.is_defective) {
      return (
        <svg className="w-3 h-3 text-red-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
      );
    }
    if (storage.is_ecommerce) {
      return (
        <svg className="w-3 h-3 text-emerald-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z" />
        </svg>
      );
    }
    return (
      <svg className="w-3 h-3 text-cyan-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
      </svg>
    );
  };

  // Get dot color based on storage type
  const getDotColor = () => {
    if (storage.is_defective) return 'bg-red-400';
    if (storage.is_ecommerce) return 'bg-emerald-400';
    return 'bg-cyan-400';
  };

  return (
    <div className="py-3 border-b border-slate-100 hover:bg-slate-50/50 transition-all">
      <div className="flex items-center gap-2.5">
        {/* Rank: dot + number */}
        <div className="flex items-center gap-1.5 flex-shrink-0 min-w-[28px]">
          <span className={`w-1 h-1 rounded-full ${getDotColor()}`}></span>
          <span className="text-xs font-medium text-slate-500">{rank}</span>
        </div>

        {/* Small icon */}
        {getIcon()}

        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-slate-700 truncate" title={storage.name}>
              {storage.name}
            </span>
            {storage.is_ecommerce && (
              <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-emerald-50 text-emerald-600 font-medium">E-COM</span>
            )}
            {storage.is_defective && (
              <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-red-50 text-red-500 font-medium">БРАК</span>
            )}
          </div>
        </div>

        {/* Stock count with change as superscript */}
        <div className="flex flex-col items-end flex-shrink-0 min-w-[70px]">
          <span className={`text-[10px] font-medium leading-none h-3 ${
            storage.change === 0 ? 'invisible' : isPositive ? 'text-emerald-500' : 'text-red-400'
          }`}>
            {storage.change === 0 ? '0' : `${isPositive ? '+' : ''}${formatNumber(storage.change)}`}
          </span>
          <span className="text-lg font-semibold text-slate-900 leading-tight">{formatNumber(storage.total_stock)}</span>
        </div>

        {/* Comparison bar chart - all storages, current highlighted */}
        <div className="w-24 h-8 flex-shrink-0">
          {allStorages.length > 0 && <Bar data={chartData} options={chartOptions} />}
        </div>
      </div>
    </div>
  );
};

interface LiveDashboardProps {
  onClose?: () => void;
}

export const LiveDashboard: React.FC<LiveDashboardProps> = ({ onClose }) => {
  const { data, connected, error, reconnect } = useDashboardSocket();

  // Calculate TOP manager - from backend or fallback to manager with most orders today
  const topManagerId = useMemo(() => {
    if (data.top_manager_month_id) {
      return data.top_manager_month_id;
    }
    // Fallback: find manager with most orders today
    if (data.managers.length > 0) {
      const topManager = data.managers.reduce((best, current) =>
        current.orders_today > best.orders_today ? current : best
      );
      return topManager.id;
    }
    return null;
  }, [data.top_manager_month_id, data.managers]);

  // Calculate WORST manager - manager with least orders today
  const worstManagerId = useMemo(() => {
    if (data.managers.length > 1) {
      const worstManager = data.managers.reduce((worst, current) =>
        current.orders_today < worst.orders_today ? current : worst
      );
      // Don't show if same as top manager (only 1 manager or all equal)
      if (worstManager.id !== topManagerId) {
        return worstManager.id;
      }
    }
    return null;
  }, [data.managers, topManagerId]);

  return (
    <div className="fixed inset-0 z-50 bg-slate-900/50 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-white rounded-3xl shadow-2xl w-full max-w-7xl h-[95vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100">
          <div className="flex items-center gap-3">
            <img src={logoSvg} alt="BI logo" className="h-10" />
            <div>
              <h2 className="text-lg font-semibold text-slate-800">{translations.title}</h2>
              <p className="text-xs text-slate-500">{translations.subtitle}</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Connection status */}
            <div className="flex items-center gap-2">
              <span
                className={`w-2 h-2 rounded-full ${
                  connected ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'
                }`}
              />
              <span className="text-xs text-slate-500">
                {connected ? translations.connected : error || translations.disconnected}
              </span>
              {!connected && (
                <button
                  onClick={reconnect}
                  className="text-xs text-blue-600 hover:text-blue-800 font-medium"
                >
                  {translations.reconnect}
                </button>
              )}
            </div>

            {/* Close button */}
            {onClose && (
              <button
                onClick={onClose}
                className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6">
          {!connected && !data.inventory ? (
            <div className="flex flex-col items-center justify-center h-64 text-slate-400">
              <svg className="w-12 h-12 mb-4 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              <p className="text-sm">{translations.connecting}</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Inventory Card */}
              <MetricCard
                title={translations.inventory}
                value={data.inventory?.total || 0}
                change={data.inventory?.change_today || 0}
                changeLabel={translations.changeToday}
                history={data.inventory?.history || []}
                color="blue"
                format="number"
              />

              {/* Revenue Card */}
              <MetricCard
                title={translations.revenueToday}
                value={data.revenue?.total || 0}
                change={data.revenue?.change || 0}
                changeLabel={`${data.revenue?.orders_today || 0} ${translations.ordersToday}`}
                history={data.revenue?.history || []}
                color="green"
                format="currency"
              />

              {/* Debt Card */}
              <MetricCard
                title={translations.outstandingDebt}
                value={data.debt?.total || 0}
                change={data.debt?.change_today || 0}
                changeLabel={translations.changeToday}
                history={data.debt?.history || []}
                color="orange"
                format="currency"
              />
            </div>
          )}

          {/* Managers & Storages - Two columns side by side */}
          {(data.managers.length > 0 || data.storages.length > 0) && (
            <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6 flex-1 min-h-0">
              {/* Managers Section - Vertical */}
              {data.managers.length > 0 && (
                <div className="flex flex-col min-h-0">
                  <div className="flex items-center gap-2 mb-4">
                    <h3 className="text-lg font-semibold text-slate-800">{translations.managerSales}</h3>
                    <span className="text-xs text-slate-400 bg-slate-100 px-2 py-0.5 rounded-full">{data.managers.length}</span>
                  </div>

                  {/* TOP Manager of the Month Block */}
                  {topManagerId && (
                    <div className="mb-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                      <div className="flex items-center gap-2">
                        <svg className="w-5 h-5 text-amber-500" fill="currentColor" viewBox="0 0 20 20">
                          <path d="M10 2l2.5 5 5.5.75-4 3.75 1 5.5L10 14.25 4.75 17l1-5.5-4-3.75L7.5 7z" />
                        </svg>
                        <span className="text-sm font-semibold text-amber-700">{translations.topManagerMonth}</span>
                        <span className="text-sm text-amber-600">
                          {data.managers.find(m => m.id === topManagerId)?.name || '—'}
                        </span>
                      </div>
                    </div>
                  )}

                  <div className="flex flex-col flex-1 overflow-y-auto pr-2">
                    {data.managers.map((manager, index) => (
                      <ManagerCard
                        key={manager.id}
                        manager={manager}
                        rank={index + 1}
                        isTopManagerMonth={manager.id === topManagerId}
                        allManagers={data.managers}
                      />
                    ))}
                  </div>

                  {/* WORST Manager Block */}
                  {worstManagerId && (
                    <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                      <div className="flex items-center gap-2">
                        <svg className="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                        </svg>
                        <span className="text-sm font-semibold text-red-700">{translations.worstManager}</span>
                        <span className="text-sm text-red-600">
                          {data.managers.find(m => m.id === worstManagerId)?.name || '—'}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Storages Section - Vertical */}
              {data.storages.length > 0 && (
                <div className="flex flex-col min-h-0">
                  <div className="flex items-center gap-2 mb-4">
                    <h3 className="text-lg font-semibold text-slate-800">{translations.storageInventory}</h3>
                    <span className="text-xs text-slate-400 bg-slate-100 px-2 py-0.5 rounded-full">{data.storages.length}</span>
                  </div>
                  <div className="flex flex-col flex-1 overflow-y-auto pr-2">
                    {data.storages.map((storage, index) => (
                      <StorageCard key={storage.id} storage={storage} allStorages={data.storages} rank={index + 1} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Info footer */}
          <div className="mt-6 text-center text-xs text-slate-400">
            {translations.footer}
          </div>
        </div>
      </div>
    </div>
  );
};

export default LiveDashboard;
