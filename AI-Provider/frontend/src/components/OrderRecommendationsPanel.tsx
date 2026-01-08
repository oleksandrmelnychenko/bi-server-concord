import React, { useState, useCallback, useRef } from 'react';
import {
  fetchOrderRecommendationsV2,
  type OrderRecommendationRequestV2,
  type OrderRecommendationResponseV2,
  type SupplierRecommendationV2,
  type OrderRecommendationItemV2,
} from '../services/api';

interface OrderRecommendationsPanelProps {
  open: boolean;
  onClose: () => void;
  language?: 'uk' | 'en';
}

const translations = {
  uk: {
    title: 'Рекомендації замовлень',
    subtitle: 'Планування поставок на основі попиту (v2)',
    calculate: 'Розрахувати',
    loading: 'Завантаження...',
    noResults: 'Немає рекомендацій для поточних параметрів',
    error: 'Помилка',
    // Form labels
    manufacturingDays: 'Виробництво (днів)',
    logisticsDays: 'Логістика (днів)',
    warehouseDays: 'Склад (днів)',
    serviceLevel: 'Рівень сервісу',
    historyWeeks: 'Історія (тижнів)',
    maxProducts: 'Макс. товарів',
    // V2 toggles
    useTrend: 'Тренд',
    useSeasonality: 'Сезонність',
    useChurn: 'Відтік',
    adjustments: 'Коригування',
    // Results
    totalLeadTime: 'Загальний час поставки',
    days: 'днів',
    supplier: 'Постачальник',
    unknownSupplier: 'Невідомий постачальник',
    products: 'товарів',
    totalQty: 'Всього',
    units: 'од.',
    // V2 stats
    withTrend: 'з трендом',
    withSeason: 'з сезонністю',
    withChurnRisk: 'з ризиком відтоку',
    // Table headers
    product: 'Товар',
    vendorCode: 'Артикул',
    onHand: 'Дефіцит',
    reorderPoint: 'Точка замовлення',
    recommendedQty: 'Рекомендовано',
    arrivalDate: 'Очікувана дата',
    avgDemand: 'Сер. попит/тиж',
    safetyStock: 'Страховий запас',
    latency: 'Час розрахунку',
    ms: 'мс',
    resultsCount: 'Знайдено товарів',
    // V2 columns
    trend: 'Тренд',
    season: 'Сезон',
    churn: 'Відтік',
    method: 'Метод',
    confidence: 'Точність',
    growing: 'зростає',
    declining: 'падає',
    stable: 'стабільно',
    // Tooltips
    tooltipProduct: 'Назва товару з каталогу',
    tooltipVendorCode: 'Унікальний артикул товару (SKU)',
    tooltipOnHand: 'Дефіцит товару на складі. Від\'ємне = перепроданий, позитивне = є залишок',
    tooltipAvgDemand: 'Середній тижневий попит за історичний період',
    tooltipTrend: 'Напрямок тренду продажів: зростає ↑ або падає ↓',
    tooltipSeason: 'Сезонний коефіцієнт: >0% = високий сезон, <0% = низький',
    tooltipChurn: 'Коригування на відтік клієнтів: зменшує прогноз для ризикових',
    tooltipReorderPoint: 'Точка замовлення = попит за час поставки + страховий запас',
    tooltipRecommendedQty: 'Рекомендована кількість для замовлення у постачальника',
    tooltipConfidence: 'Точність прогнозу на основі якості історичних даних',
  },
  en: {
    title: 'Order Recommendations',
    subtitle: 'Supply planning based on demand (v2)',
    calculate: 'Calculate',
    loading: 'Loading...',
    noResults: 'No recommendations for current parameters',
    error: 'Error',
    // Form labels
    manufacturingDays: 'Manufacturing (days)',
    logisticsDays: 'Logistics (days)',
    warehouseDays: 'Warehouse (days)',
    serviceLevel: 'Service Level',
    historyWeeks: 'History (weeks)',
    maxProducts: 'Max products',
    // V2 toggles
    useTrend: 'Trend',
    useSeasonality: 'Seasonality',
    useChurn: 'Churn',
    adjustments: 'Adjustments',
    // Results
    totalLeadTime: 'Total lead time',
    days: 'days',
    supplier: 'Supplier',
    unknownSupplier: 'Unknown supplier',
    products: 'products',
    totalQty: 'Total',
    units: 'units',
    // V2 stats
    withTrend: 'with trend',
    withSeason: 'with seasonality',
    withChurnRisk: 'with churn risk',
    // Table headers
    product: 'Product',
    vendorCode: 'SKU',
    onHand: 'Deficit',
    reorderPoint: 'Reorder Point',
    recommendedQty: 'Recommended',
    arrivalDate: 'Arrival Date',
    avgDemand: 'Avg Demand/wk',
    safetyStock: 'Safety Stock',
    latency: 'Calculation time',
    ms: 'ms',
    resultsCount: 'Products found',
    // V2 columns
    trend: 'Trend',
    season: 'Season',
    churn: 'Churn',
    method: 'Method',
    confidence: 'Confidence',
    growing: 'growing',
    declining: 'declining',
    stable: 'stable',
    // Tooltips
    tooltipProduct: 'Product name from catalog',
    tooltipVendorCode: 'Unique product identifier (SKU)',
    tooltipOnHand: 'Stock deficit. Negative = oversold, positive = in stock',
    tooltipAvgDemand: 'Average weekly demand over history period',
    tooltipTrend: 'Sales trend direction: growing ↑ or declining ↓',
    tooltipSeason: 'Seasonal factor: >0% = high season, <0% = low season',
    tooltipChurn: 'Churn risk adjustment: reduces forecast for at-risk customers',
    tooltipReorderPoint: 'Reorder point = demand during lead time + safety stock',
    tooltipRecommendedQty: 'Recommended quantity to order from supplier',
    tooltipConfidence: 'Forecast accuracy based on historical data quality',
  },
};

// Toggle switch component
const Toggle: React.FC<{
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  disabled?: boolean;
}> = ({ checked, onChange, label, disabled }) => (
  <label className={`flex items-center gap-2 cursor-pointer ${disabled ? 'opacity-50' : ''}`}>
    <div className="relative">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        disabled={disabled}
        className="sr-only"
      />
      <div className={`w-9 h-5 rounded-full transition-colors ${checked ? 'bg-violet-500' : 'bg-slate-300'}`}>
        <div
          className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${
            checked ? 'translate-x-4' : ''
          }`}
        />
      </div>
    </div>
    <span className="text-xs font-medium text-slate-600">{label}</span>
  </label>
);

// Trend direction badge
const TrendBadge: React.FC<{ direction: string | null; factor: number | null; t: typeof translations['uk'] }> = ({
  direction,
  factor,
  t,
}) => {
  if (!direction || direction === 'stable') return <span className="text-slate-400">-</span>;

  const isGrowing = direction === 'growing';
  const pct = factor ? Math.round((factor - 1) * 100) : 0;

  return (
    <span
      className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-xs font-medium ${
        isGrowing ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
      }`}
    >
      {isGrowing ? (
        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
        </svg>
      ) : (
        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
        </svg>
      )}
      {pct > 0 ? '+' : ''}{pct}%
    </span>
  );
};

// Seasonal index badge
const SeasonBadge: React.FC<{ index: number | null; period: number | null }> = ({ index, period }) => {
  if (!index || !period) return <span className="text-slate-400">-</span>;

  const pct = Math.round((index - 1) * 100);
  const isHigh = index > 1.05;
  const isLow = index < 0.95;

  return (
    <span
      className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-xs font-medium ${
        isHigh ? 'bg-amber-100 text-amber-700' : isLow ? 'bg-violet-100 text-violet-700' : 'bg-slate-100 text-slate-600'
      }`}
      title={`Period: ${period} weeks`}
    >
      {pct > 0 ? '+' : ''}{pct}%
    </span>
  );
};

// Churn adjustment badge
const ChurnBadge: React.FC<{ adjustment: number | null; atRiskPct: number | null }> = ({ adjustment, atRiskPct }) => {
  if (!adjustment || adjustment >= 1) return <span className="text-slate-400">-</span>;

  const reduction = Math.round((1 - adjustment) * 100);

  return (
    <span
      className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-xs font-medium bg-orange-100 text-orange-700"
      title={atRiskPct ? `At-risk demand: ${Math.round(atRiskPct * 100)}%` : ''}
    >
      -{reduction}%
    </span>
  );
};

// Confidence indicator
const ConfidenceBar: React.FC<{ value: number | null }> = ({ value }) => {
  if (value === null) return <span className="text-slate-400">-</span>;

  const pct = Math.round(value * 100);
  const color = pct >= 70 ? 'bg-green-500' : pct >= 40 ? 'bg-amber-500' : 'bg-red-500';

  return (
    <div className="flex items-center gap-1">
      <div className="w-12 h-1.5 bg-slate-200 rounded-full overflow-hidden">
        <div className={`h-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-slate-500">{pct}%</span>
    </div>
  );
};

// Column header with icon and tooltip
const ColumnHeader: React.FC<{
  label: string;
  tooltip: string;
  icon: 'product' | 'sku' | 'stock' | 'demand' | 'trend' | 'season' | 'churn' | 'reorder' | 'recommend' | 'confidence';
  align?: 'left' | 'center' | 'right';
  highlight?: boolean;
}> = ({ label, tooltip, icon, align = 'left', highlight = false }) => {
  const [showTooltip, setShowTooltip] = React.useState(false);

  const icons: Record<string, JSX.Element> = {
    product: (
      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
      </svg>
    ),
    sku: (
      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
      </svg>
    ),
    stock: (
      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
      </svg>
    ),
    demand: (
      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
    trend: (
      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
      </svg>
    ),
    season: (
      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
      </svg>
    ),
    churn: (
      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    ),
    reorder: (
      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
    recommend: (
      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
      </svg>
    ),
    confidence: (
      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
      </svg>
    ),
  };

  const alignClass = align === 'right' ? 'justify-end' : align === 'center' ? 'justify-center' : 'justify-start';

  return (
    <th className={`px-3 py-2 text-xs font-medium text-slate-500 ${highlight ? 'bg-violet-50' : ''} relative`}>
      <div
        className={`flex items-center gap-1.5 ${alignClass} cursor-help`}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
      >
        <span className="text-slate-400">{icons[icon]}</span>
        <span>{label}</span>
      </div>
      {showTooltip && (
        <div className="absolute left-1/2 -translate-x-1/2 top-full mt-1 px-3 py-2 bg-white text-slate-700 text-xs rounded-lg shadow-lg border border-slate-200 z-[9999] whitespace-nowrap">
          {tooltip}
          <div className="absolute bottom-full left-1/2 -translate-x-1/2 border-4 border-transparent border-b-white" />
          <div className="absolute bottom-full left-1/2 -translate-x-1/2 border-4 border-transparent border-b-slate-200 -mb-px" />
        </div>
      )}
    </th>
  );
};

export const OrderRecommendationsPanel: React.FC<OrderRecommendationsPanelProps> = ({
  open,
  onClose,
  language = 'uk',
}) => {
  const t = translations[language];
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<OrderRecommendationResponseV2 | null>(null);
  const [expandedSuppliers, setExpandedSuppliers] = useState<Set<number | null>>(new Set());
  const controllerRef = useRef<AbortController | null>(null);

  // Form state with v2 fields
  const [formData, setFormData] = useState<OrderRecommendationRequestV2>({
    manufacturing_days: 14,
    logistics_days: 21,
    warehouse_days: 3,
    service_level: 0.95,
    history_weeks: 26,
    max_products: 100,
    // V2 fields
    use_trend_adjustment: true,
    use_seasonality: true,
    use_churn_adjustment: true,
    min_history_weeks: 8,
  });

  const handleInputChange = useCallback((field: keyof OrderRecommendationRequestV2, value: number | boolean) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  }, []);

  const handleCalculate = useCallback(async () => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetchOrderRecommendationsV2(formData, controller.signal);
      if (!controller.signal.aborted) {
        setResult(response);
        // Expand first supplier by default if has results
        if (response.recommendations.length > 0) {
          setExpandedSuppliers(new Set([response.recommendations[0].supplier_id]));
        }
      }
    } catch (err) {
      if (!controller.signal.aborted) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      }
    } finally {
      if (!controller.signal.aborted) {
        setLoading(false);
      }
    }
  }, [formData]);

  const toggleSupplier = useCallback((supplierId: number | null) => {
    setExpandedSuppliers((prev) => {
      const next = new Set(prev);
      if (next.has(supplierId)) {
        next.delete(supplierId);
      } else {
        next.add(supplierId);
      }
      return next;
    });
  }, []);

  if (!open) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-slate-900/50 backdrop-blur-sm z-50"
        onClick={onClose}
      />

      {/* Panel */}
      <div className="fixed top-4 bottom-4 left-1/2 -translate-x-1/2 w-[95vw] max-w-7xl bg-white rounded-2xl shadow-2xl z-50 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200">
          <div>
            <h2 className="text-lg font-semibold text-slate-800">{t.title}</h2>
            <p className="text-xs text-slate-500">{t.subtitle}</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Form */}
        <div className="px-6 py-4 border-b border-slate-100 bg-slate-50">
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">{t.manufacturingDays}</label>
              <input
                type="number"
                value={formData.manufacturing_days}
                onChange={(e) => handleInputChange('manufacturing_days', parseInt(e.target.value) || 0)}
                className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500"
                min={0}
                max={365}
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">{t.logisticsDays}</label>
              <input
                type="number"
                value={formData.logistics_days}
                onChange={(e) => handleInputChange('logistics_days', parseInt(e.target.value) || 0)}
                className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500"
                min={0}
                max={365}
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">{t.warehouseDays}</label>
              <input
                type="number"
                value={formData.warehouse_days}
                onChange={(e) => handleInputChange('warehouse_days', parseInt(e.target.value) || 0)}
                className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500"
                min={0}
                max={365}
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">{t.serviceLevel}</label>
              <input
                type="number"
                value={formData.service_level}
                onChange={(e) => handleInputChange('service_level', parseFloat(e.target.value) || 0.95)}
                className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500"
                min={0.5}
                max={0.999}
                step={0.01}
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">{t.historyWeeks}</label>
              <input
                type="number"
                value={formData.history_weeks}
                onChange={(e) => handleInputChange('history_weeks', parseInt(e.target.value) || 26)}
                className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500"
                min={4}
                max={104}
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">{t.maxProducts}</label>
              <input
                type="number"
                value={formData.max_products}
                onChange={(e) => handleInputChange('max_products', parseInt(e.target.value) || 100)}
                className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500"
                min={1}
                max={5000}
              />
            </div>
          </div>

          {/* V2 Toggles */}
          <div className="mt-4 flex items-center gap-6 pb-4 border-b border-slate-200">
            <span className="text-xs font-medium text-slate-500">{t.adjustments}:</span>
            <Toggle
              checked={formData.use_trend_adjustment ?? true}
              onChange={(v) => handleInputChange('use_trend_adjustment', v)}
              label={t.useTrend}
            />
            <Toggle
              checked={formData.use_seasonality ?? true}
              onChange={(v) => handleInputChange('use_seasonality', v)}
              label={t.useSeasonality}
            />
            <Toggle
              checked={formData.use_churn_adjustment ?? true}
              onChange={(v) => handleInputChange('use_churn_adjustment', v)}
              label={t.useChurn}
            />
          </div>

          <div className="mt-4 flex items-center justify-between">
            <button
              onClick={handleCalculate}
              disabled={loading}
              className="flex items-center gap-2 rounded-full bg-violet-600 px-3 py-1.5 text-xs font-medium hover:bg-violet-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle opacity="0.25" cx="12" cy="12" r="10" stroke="#fff" strokeWidth="4" />
                    <path opacity="0.75" fill="#fff" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  <span style={{ color: 'white' }}>{t.loading}</span>
                </>
              ) : (
                <>
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24">
                    <path stroke="#fff" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                  </svg>
                  <span style={{ color: 'white' }}>{t.calculate}</span>
                </>
              )}
            </button>

            {result && (
              <div className="flex items-center gap-4 text-xs text-slate-500">
                <span>{t.totalLeadTime}: <strong className="text-slate-700">{result.lead_time_days} {t.days}</strong></span>
                <span>{t.resultsCount}: <strong className="text-slate-700">{result.count}</strong></span>
                {result.products_with_trend > 0 && (
                  <span className="text-green-600">{result.products_with_trend} {t.withTrend}</span>
                )}
                {result.products_with_seasonality > 0 && (
                  <span className="text-amber-600">{result.products_with_seasonality} {t.withSeason}</span>
                )}
                {result.products_with_churn_risk > 0 && (
                  <span className="text-orange-600">{result.products_with_churn_risk} {t.withChurnRisk}</span>
                )}
                <span>{t.latency}: <strong className="text-slate-700">{result.latency_ms.toFixed(0)} {t.ms}</strong></span>
              </div>
            )}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6">
          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
              <strong>{t.error}:</strong> {error}
            </div>
          )}

          {!loading && !error && !result && (
            <div className="flex flex-col items-center justify-center h-64 text-slate-400">
              <svg className="w-16 h-16 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
              </svg>
              <p className="text-sm">{t.subtitle}</p>
            </div>
          )}

          {loading && (
            <div className="flex flex-col items-center justify-center h-64 text-slate-400">
              <svg className="w-12 h-12 mb-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              <p className="text-sm">{t.loading}</p>
            </div>
          )}

          {result && result.recommendations.length === 0 && (
            <div className="flex flex-col items-center justify-center h-64 text-slate-400">
              <svg className="w-16 h-16 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
              </svg>
              <p className="text-sm">{t.noResults}</p>
            </div>
          )}

          {result && result.recommendations.length > 0 && (
            <div className="space-y-4">
              {result.recommendations.map((supplier: SupplierRecommendationV2, idx) => {
                const isExpanded = expandedSuppliers.has(supplier.supplier_id);
                return (
                  <div key={supplier.supplier_id ?? idx} className="border border-slate-200 rounded-xl overflow-hidden">
                    {/* Supplier Header */}
                    <button
                      onClick={() => toggleSupplier(supplier.supplier_id)}
                      className="w-full px-4 py-3 bg-slate-50 hover:bg-slate-100 transition-colors flex items-center justify-between"
                    >
                      <div className="flex items-center gap-3">
                        <svg className={`w-4 h-4 text-slate-400 transition-transform ${isExpanded ? 'rotate-90' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                        <div className="w-8 h-8 bg-violet-600 rounded-lg flex items-center justify-center">
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24">
                            <path stroke="#fff" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                          </svg>
                        </div>
                        <div className="text-left">
                          <div className="font-medium text-slate-800">
                            {supplier.supplier_name || t.unknownSupplier}
                          </div>
                          <div className="text-xs text-slate-500">
                            {supplier.products.length} {t.products}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-lg font-semibold text-violet-600">
                          {supplier.total_recommended_qty.toLocaleString()}
                        </div>
                        <div className="text-xs text-slate-500">{t.totalQty} {t.units}</div>
                      </div>
                    </button>

                    {/* Products Table */}
                    {isExpanded && (
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                          <thead className="bg-slate-50 border-t border-slate-200">
                            <tr>
                              <ColumnHeader label={t.product} tooltip={t.tooltipProduct} icon="product" align="left" />
                              <ColumnHeader label={t.vendorCode} tooltip={t.tooltipVendorCode} icon="sku" align="left" />
                              <ColumnHeader label={t.onHand} tooltip={t.tooltipOnHand} icon="stock" align="right" />
                              <ColumnHeader label={t.avgDemand} tooltip={t.tooltipAvgDemand} icon="demand" align="right" />
                              <ColumnHeader label={t.trend} tooltip={t.tooltipTrend} icon="trend" align="center" />
                              <ColumnHeader label={t.season} tooltip={t.tooltipSeason} icon="season" align="center" />
                              <ColumnHeader label={t.churn} tooltip={t.tooltipChurn} icon="churn" align="center" />
                              <ColumnHeader label={t.reorderPoint} tooltip={t.tooltipReorderPoint} icon="reorder" align="right" />
                              <ColumnHeader label={t.recommendedQty} tooltip={t.tooltipRecommendedQty} icon="recommend" align="right" highlight />
                              <ColumnHeader label={t.confidence} tooltip={t.tooltipConfidence} icon="confidence" align="center" />
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-100">
                            {supplier.products.map((item: OrderRecommendationItemV2) => (
                              <tr key={item.product_id} className="hover:bg-slate-50">
                                <td className="px-3 py-2 text-slate-800 max-w-[180px] truncate" title={item.product_name || ''}>
                                  {item.product_name || `#${item.product_id}`}
                                </td>
                                <td className="px-3 py-2 text-slate-500 font-mono text-xs">
                                  {item.vendor_code || '-'}
                                </td>
                                <td className="px-3 py-2 text-right text-slate-700">
                                  {item.on_hand.toLocaleString()}
                                </td>
                                <td className="px-3 py-2 text-right text-slate-500">
                                  {item.avg_weekly_demand.toFixed(1)}
                                </td>
                                <td className="px-3 py-2 text-center">
                                  <TrendBadge direction={item.trend_direction} factor={item.trend_factor} t={t} />
                                </td>
                                <td className="px-3 py-2 text-center">
                                  <SeasonBadge index={item.seasonal_index} period={item.seasonal_period_weeks} />
                                </td>
                                <td className="px-3 py-2 text-center">
                                  <ChurnBadge adjustment={item.churn_adjustment} atRiskPct={item.at_risk_demand_pct} />
                                </td>
                                <td className="px-3 py-2 text-right text-amber-600 font-medium">
                                  {item.reorder_point.toFixed(0)}
                                </td>
                                <td className="px-3 py-2 text-right font-semibold text-violet-600 bg-violet-50">
                                  {item.recommended_qty.toLocaleString()}
                                </td>
                                <td className="px-3 py-2">
                                  <ConfidenceBar value={item.forecast_confidence} />
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default OrderRecommendationsPanel;
