import { useState } from 'react';
import type { ProductForecastResponse, ProductCharts, ProductProof } from '../services/api';

type Language = 'uk' | 'en';

interface ProductForecastScreenProps {
  isOpen: boolean;
  onClose: () => void;
  language: Language;
  loading: boolean;
  error?: string | null;
  productId?: number;
  productName?: string | null;
  vendorCode?: string | null;
  category?: string | null;
  forecast?: ProductForecastResponse | null;
  charts?: ProductCharts | null;
  proof?: ProductProof | null;
}

export function ProductForecastScreen({
  isOpen,
  onClose,
  language,
  loading,
  error,
  productId,
  productName,
  vendorCode,
  category,
  forecast,
  charts,
  proof,
}: ProductForecastScreenProps) {
  const [chartTab, setChartTab] = useState<'6m' | '1y'>('6m');
  const [forecastTab, setForecastTab] = useState<'chart' | 'table'>('chart');

  if (!isOpen) return null;

  const t = (uk: string, en: string) => language === 'uk' ? uk : en;

  // Prepare forecast chart data
  const weeklyData = forecast?.weekly_data || [];
  const predictedData = weeklyData.filter(w => w.data_type === 'predicted');
  const historicalData = weeklyData.filter(w => w.data_type === 'actual');

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40 transition-opacity"
        onClick={onClose}
      />

      {/* Side Panel */}
      <aside
        className="fixed right-0 top-0 h-full z-50 w-[680px] max-w-[90vw] bg-white border-l border-slate-200 shadow-2xl flex flex-col overflow-hidden transition-transform duration-300"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-200 bg-gradient-to-r from-emerald-50 to-slate-50 flex-shrink-0">
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-10 h-10 rounded-lg bg-emerald-500 flex items-center justify-center shadow-md flex-shrink-0">
              <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
              </svg>
            </div>
            <div className="min-w-0">
              <h2 className="text-base font-bold text-slate-800 truncate">
                {productName || `${t('Товар', 'Product')} #${productId}`}
              </h2>
              <div className="flex items-center gap-2 text-xs text-slate-500">
                {vendorCode && <span className="font-mono bg-slate-100 px-1.5 py-0.5 rounded">{vendorCode}</span>}
                {category && <span className="truncate">{category}</span>}
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-lg flex items-center justify-center text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors flex-shrink-0"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          {loading ? (
            <div className="flex flex-col items-center justify-center py-16">
              <div className="w-10 h-10 border-4 border-emerald-200 border-t-emerald-500 rounded-full animate-spin mb-3" />
              <p className="text-sm text-slate-500">{t('Генерація прогнозування...', 'Generating forecast...')}</p>
            </div>
          ) : error ? (
            <div className="flex flex-col items-center justify-center py-16 text-red-500">
              <svg className="w-10 h-10 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-sm">{error}</p>
            </div>
          ) : (
            <>
              {/* Forecast Summary Cards */}
              {forecast?.summary && (
                <div className="grid grid-cols-4 gap-3">
                  <div className="bg-gradient-to-br from-emerald-500 to-emerald-600 rounded-lg p-3 text-white shadow">
                    <div className="text-[10px] opacity-80 uppercase">{t('Прогноз', 'Forecast')}</div>
                    <div className="text-xl font-bold">{forecast.summary.total_predicted_quantity?.toLocaleString()}</div>
                    <div className="text-[9px] opacity-60">{forecast.forecast_period_weeks} {t('тижнів', 'weeks')}</div>
                  </div>
                  <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg p-3 text-white shadow">
                    <div className="text-[10px] opacity-80 uppercase">{t('Дохід', 'Revenue')}</div>
                    <div className="text-xl font-bold">{forecast.summary.total_predicted_revenue?.toLocaleString() || '—'}</div>
                    <div className="text-[9px] opacity-60">UAH</div>
                  </div>
                  <div className="bg-gradient-to-br from-violet-500 to-violet-600 rounded-lg p-3 text-white shadow">
                    <div className="text-[10px] opacity-80 uppercase">{t('Активні', 'Active')}</div>
                    <div className="text-xl font-bold">{forecast.summary.active_customers}</div>
                    <div className="text-[9px] opacity-60">{t('клієнтів', 'customers')}</div>
                  </div>
                  <div className="bg-gradient-to-br from-amber-500 to-amber-600 rounded-lg p-3 text-white shadow">
                    <div className="text-[10px] opacity-80 uppercase">{t('Ризик', 'At Risk')}</div>
                    <div className="text-xl font-bold">{forecast.summary.at_risk_customers}</div>
                    <div className="text-[9px] opacity-60">{t('клієнтів', 'customers')}</div>
                  </div>
                </div>
              )}

              {/* Forecast Chart */}
              {weeklyData.length > 0 && (
                <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                      {t('Прогноз попиту', 'Demand Forecast')}
                    </h3>
                    <div className="flex gap-1 bg-slate-100 rounded-lg p-0.5">
                      <button
                        onClick={() => setForecastTab('chart')}
                        className={`px-2 py-1 text-[10px] font-medium rounded-md transition-all ${
                          forecastTab === 'chart' ? 'bg-white text-slate-800 shadow-sm' : 'text-slate-500 hover:text-slate-700'
                        }`}
                      >
                        {t('Графік', 'Chart')}
                      </button>
                      <button
                        onClick={() => setForecastTab('table')}
                        className={`px-2 py-1 text-[10px] font-medium rounded-md transition-all ${
                          forecastTab === 'table' ? 'bg-white text-slate-800 shadow-sm' : 'text-slate-500 hover:text-slate-700'
                        }`}
                      >
                        {t('Таблиця', 'Table')}
                      </button>
                    </div>
                  </div>

                  {forecastTab === 'chart' ? (
                    <div className="h-48 relative">
                      {(() => {
                        const allQty = weeklyData.map(w => w.quantity || 0);
                        const maxQty = Math.max(...allQty, 1);

                        return (
                          <>
                            <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                              {/* Grid lines */}
                              <line x1="0" y1="25" x2="100" y2="25" stroke="#e2e8f0" strokeWidth="0.3" vectorEffect="non-scaling-stroke" />
                              <line x1="0" y1="50" x2="100" y2="50" stroke="#e2e8f0" strokeWidth="0.3" vectorEffect="non-scaling-stroke" />
                              <line x1="0" y1="75" x2="100" y2="75" stroke="#e2e8f0" strokeWidth="0.3" vectorEffect="non-scaling-stroke" />

                              {/* Historical line (dashed) */}
                              {historicalData.length > 0 && (() => {
                                const pts = historicalData.map((w, i) => {
                                  const x = (i / Math.max(weeklyData.length - 1, 1)) * 100;
                                  const y = 100 - ((w.quantity || 0) / maxQty) * 100;
                                  return `${x},${y}`;
                                }).join(' ');
                                return <polyline points={pts} fill="none" stroke="#94a3b8" strokeWidth="2" strokeDasharray="4,2" vectorEffect="non-scaling-stroke" />;
                              })()}

                              {/* Predicted area with gradient */}
                              {predictedData.length > 0 && (() => {
                                const startIdx = historicalData.length;
                                const pts = predictedData.map((w, i) => {
                                  const x = ((startIdx + i) / Math.max(weeklyData.length - 1, 1)) * 100;
                                  const y = 100 - ((w.quantity || 0) / maxQty) * 100;
                                  return `${x},${y}`;
                                });
                                const ptsStr = pts.join(' ');
                                const firstX = pts[0]?.split(',')[0] || '0';
                                const lastX = pts[pts.length - 1]?.split(',')[0] || '100';
                                const areaPoints = `${firstX},100 ${ptsStr} ${lastX},100`;

                                return (
                                  <>
                                    <polygon points={areaPoints} fill="url(#forecastGradient)" />
                                    <polyline points={ptsStr} fill="none" stroke="#10b981" strokeWidth="2.5" vectorEffect="non-scaling-stroke" />
                                  </>
                                );
                              })()}

                              <defs>
                                <linearGradient id="forecastGradient" x1="0" y1="0" x2="0" y2="1">
                                  <stop offset="0%" stopColor="#10b981" stopOpacity="0.4" />
                                  <stop offset="100%" stopColor="#10b981" stopOpacity="0.05" />
                                </linearGradient>
                              </defs>
                            </svg>

                            {/* Data points */}
                            {weeklyData.map((w, idx) => {
                              const xPct = (idx / Math.max(weeklyData.length - 1, 1)) * 100;
                              const qty = w.quantity || 0;
                              const yPct = (qty / maxQty) * 100;
                              const isPredicted = w.data_type === 'predicted';

                              return (
                                <div
                                  key={idx}
                                  className={`absolute w-2 h-2 rounded-full border-2 border-white shadow-sm hover:scale-150 transition-transform cursor-pointer ${
                                    isPredicted ? 'bg-emerald-500' : 'bg-slate-400'
                                  }`}
                                  style={{ left: `${xPct}%`, bottom: `${yPct}%`, transform: 'translate(-50%, 50%)' }}
                                  title={`${w.week_start}: ${qty.toLocaleString()} ${t('шт', 'qty')} (${isPredicted ? t('прогноз', 'forecast') : t('факт', 'actual')})`}
                                />
                              );
                            })}

                            {/* X-axis labels */}
                            <div className="absolute -bottom-5 left-0 right-0 flex justify-between text-[9px] text-slate-400">
                              {weeklyData.filter((_, i) => i === 0 || i === Math.floor(weeklyData.length / 2) || i === weeklyData.length - 1).map((w, idx) => (
                                <span key={idx}>{w.week_start?.slice(5) || ''}</span>
                              ))}
                            </div>
                          </>
                        );
                      })()}
                    </div>
                  ) : (
                    <div className="max-h-48 overflow-y-auto">
                      <table className="w-full text-xs">
                        <thead className="bg-slate-50 sticky top-0">
                          <tr>
                            <th className="text-left p-1.5 font-medium text-slate-600">{t('Тиждень', 'Week')}</th>
                            <th className="text-right p-1.5 font-medium text-slate-600">{t('К-сть', 'Qty')}</th>
                            <th className="text-right p-1.5 font-medium text-slate-600">{t('Зам.', 'Ord.')}</th>
                            <th className="text-center p-1.5 font-medium text-slate-600">{t('Тип', 'Type')}</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                          {weeklyData.map((w, idx) => (
                            <tr key={idx} className={w.data_type === 'predicted' ? 'bg-emerald-50/50' : ''}>
                              <td className="p-1.5 text-slate-700">{w.week_start}</td>
                              <td className="p-1.5 text-right font-medium text-slate-800">{(w.quantity || 0).toLocaleString()}</td>
                              <td className="p-1.5 text-right text-slate-600">{(w.orders || 0).toFixed(1)}</td>
                              <td className="p-1.5 text-center">
                                <span className={`px-1.5 py-0.5 rounded-full text-[9px] font-medium ${
                                  w.data_type === 'predicted' ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-600'
                                }`}>
                                  {w.data_type === 'predicted' ? t('Прогноз', 'Fcst') : t('Факт', 'Fact')}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}

                  {/* Legend */}
                  <div className="flex items-center gap-4 mt-3 pt-3 border-t border-slate-100">
                    <div className="flex items-center gap-1.5 text-[10px] text-slate-500">
                      <div className="w-2.5 h-2.5 rounded-full bg-slate-400" />
                      {t('Історія', 'Historical')}
                    </div>
                    <div className="flex items-center gap-1.5 text-[10px] text-slate-500">
                      <div className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                      {t('Прогноз', 'Forecast')}
                    </div>
                  </div>
                </div>
              )}

              {/* Sales History Chart */}
              {charts?.sales_history && charts.sales_history.length > 0 && (
                <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                      {t('Історія продажів', 'Sales History')}
                    </h3>
                    <div className="flex gap-1 bg-slate-100 rounded-lg p-0.5">
                      <button
                        onClick={() => setChartTab('6m')}
                        className={`px-2 py-1 text-[10px] font-medium rounded-md transition-all ${
                          chartTab === '6m' ? 'bg-white text-slate-800 shadow-sm' : 'text-slate-500 hover:text-slate-700'
                        }`}
                      >
                        {t('6 міс', '6 mo')}
                      </button>
                      <button
                        onClick={() => setChartTab('1y')}
                        className={`px-2 py-1 text-[10px] font-medium rounded-md transition-all ${
                          chartTab === '1y' ? 'bg-white text-slate-800 shadow-sm' : 'text-slate-500 hover:text-slate-700'
                        }`}
                      >
                        {t('1 рік', '1 year')}
                      </button>
                    </div>
                  </div>

                  {(() => {
                    const data = chartTab === '6m' ? charts.sales_history.slice(-6) : charts.sales_history;
                    const maxQty = Math.max(...data.map(d => d.qty || 0), 1);

                    return (
                      <div className="h-36 flex items-end gap-1.5">
                        {data.map((item, idx) => {
                          const height = ((item.qty || 0) / maxQty) * 100;
                          return (
                            <div key={idx} className="flex-1 flex flex-col items-center gap-0.5">
                              <div className="text-[9px] text-slate-500 font-medium">{item.qty?.toLocaleString()}</div>
                              <div
                                className="w-full bg-gradient-to-t from-blue-500 to-blue-400 rounded-t transition-all hover:from-blue-600 hover:to-blue-500 cursor-pointer"
                                style={{ height: `${Math.max(height, 4)}%` }}
                                title={`${item.month}: ${item.orders} ${t('замовлень', 'orders')}, ${item.amount?.toLocaleString()} UAH`}
                              />
                              <div className="text-[9px] text-slate-400">{item.month?.slice(5)}</div>
                            </div>
                          );
                        })}
                      </div>
                    );
                  })()}
                </div>
              )}

              {/* Product Proof Metrics */}
              {proof && (
                <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm">
                  <h3 className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-3">
                    {t('Аналітика товару', 'Product Analytics')}
                  </h3>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                    <div className="flex justify-between items-center py-1.5 border-b border-slate-100">
                      <span className="text-slate-600">{t('Всього замовлень', 'Total Orders')}</span>
                      <span className="font-bold text-slate-800">{proof.total_orders?.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between items-center py-1.5 border-b border-slate-100">
                      <span className="text-slate-600">{t('Продано', 'Units Sold')}</span>
                      <span className="font-bold text-slate-800">{proof.total_qty_sold?.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between items-center py-1.5 border-b border-slate-100">
                      <span className="text-slate-600">{t('Дохід', 'Revenue')}</span>
                      <span className="font-bold text-emerald-600">{proof.total_revenue?.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between items-center py-1.5 border-b border-slate-100">
                      <span className="text-slate-600">{t('Клієнтів', 'Customers')}</span>
                      <span className="font-bold text-slate-800">{proof.unique_customers}</span>
                    </div>
                    {proof.last_sale_date && (
                      <div className="flex justify-between items-center py-1.5 border-b border-slate-100">
                        <span className="text-slate-600">{t('Останній', 'Last Sale')}</span>
                        <span className="font-medium text-slate-700">{proof.last_sale_date}</span>
                      </div>
                    )}
                    {proof.days_since_last_sale !== null && (
                      <div className="flex justify-between items-center py-1.5 border-b border-slate-100">
                        <span className="text-slate-600">{t('Днів без', 'Days ago')}</span>
                        <span className={`font-bold ${(proof.days_since_last_sale || 0) > 30 ? 'text-amber-600' : 'text-slate-800'}`}>
                          {proof.days_since_last_sale}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Top Customers */}
              {forecast?.top_customers_by_volume && forecast.top_customers_by_volume.length > 0 && (
                <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm">
                  <h3 className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-3">
                    {t('Топ клієнти (прогноз)', 'Top Customers (Forecast)')}
                  </h3>
                  <div className="space-y-1.5">
                    {forecast.top_customers_by_volume.slice(0, 6).map((customer, idx) => (
                      <div key={customer.customer_id} className="flex items-center gap-2 p-2 rounded-lg hover:bg-slate-50 transition-colors">
                        <div className="w-5 h-5 rounded-full bg-emerald-100 flex items-center justify-center text-[10px] font-bold text-emerald-600 flex-shrink-0">
                          {idx + 1}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="text-xs font-medium text-slate-800 truncate">{customer.customer_name}</div>
                          <div className="text-[10px] text-slate-500">{customer.contribution_pct?.toFixed(1)}% {t('від обсягу', 'of volume')}</div>
                        </div>
                        <div className="text-xs font-bold text-emerald-600">{customer.predicted_quantity?.toLocaleString()}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* At Risk Customers */}
              {forecast?.at_risk_customers && forecast.at_risk_customers.length > 0 && (
                <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm">
                  <h3 className="text-xs font-semibold text-amber-700 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                    {t('Клієнти в ризику', 'At Risk Customers')}
                  </h3>
                  <div className="space-y-1.5 max-h-48 overflow-y-auto">
                    {forecast.at_risk_customers.slice(0, 8).map((customer) => (
                      <div key={customer.customer_id} className="p-2 rounded-lg bg-amber-50 border border-amber-200">
                        <div className="flex items-center justify-between mb-0.5">
                          <span className="text-xs font-medium text-slate-800 truncate">{customer.customer_name}</span>
                          <span className={`text-[9px] px-1.5 py-0.5 rounded-full font-medium ${
                            customer.churn_probability > 0.6 ? 'bg-red-100 text-red-700' :
                            customer.churn_probability > 0.4 ? 'bg-amber-100 text-amber-700' :
                            'bg-yellow-100 text-yellow-700'
                          }`}>
                            {(customer.churn_probability * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="text-[10px] text-slate-600">
                          {t('Остання', 'Last')}: {customer.last_order} • {customer.days_overdue} {t('днів', 'days')}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Model Info */}
              {forecast?.model_metadata && (
                <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
                  <div className="text-[10px] text-slate-500 mb-1.5">{t('Інформація про модель', 'Model Info')}</div>
                  <div className="flex flex-wrap gap-1.5">
                    <span className="px-1.5 py-0.5 bg-white rounded text-[10px] text-slate-600 border">
                      {t('Точність', 'Acc')}: {(forecast.model_metadata.forecast_accuracy_estimate * 100).toFixed(0)}%
                    </span>
                    <span className="px-1.5 py-0.5 bg-white rounded text-[10px] text-slate-600 border">
                      {forecast.model_metadata.training_customers} {t('клієнтів', 'cust.')}
                    </span>
                    {forecast.model_metadata.seasonality_detected && (
                      <span className="px-1.5 py-0.5 bg-violet-100 rounded text-[10px] text-violet-700 border border-violet-200">
                        {t('Сезонність', 'Season')}
                      </span>
                    )}
                  </div>
                </div>
              )}

              {/* No data message */}
              {!loading && !forecast && !charts && !proof && (
                <div className="text-center py-12 text-slate-500">
                  <svg className="w-10 h-10 mx-auto mb-3 text-slate-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
                  </svg>
                  <p className="text-sm">{t('Немає даних для цього товару', 'No data for this product')}</p>
                </div>
              )}
            </>
          )}
        </div>
      </aside>
    </>
  );
}
