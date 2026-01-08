import type { ApiStatus, OllamaQueryResult } from '../types';

// API Configuration - Port Reference:
// - 8000: AI-Provider Backend (Text-to-SQL)
// - 8001: Main API (Recommendations, Forecasts, Analytics)
// - 8200: Dashboard WebSocket
// - 3000: Frontend
const SQL_API = import.meta.env.VITE_SQL_API || 'http://localhost:8000';
const ANALYTICS_API = import.meta.env.VITE_ANALYTICS_API || 'http://localhost:8001';
const RECO_API = import.meta.env.VITE_RECO_API || 'http://localhost:8001';

export interface YearlySalesRow {
  year: number;
  total_sales: number;
  total_orders: number;
  total_items: number;
}

export interface TopProductRow {
  product_id: number;
  product_name: string;
  total_qty: number;
  order_count: number;
}

export interface TopClientRow {
  client_id: number;
  client_name: string;
  region_id: number | null;
  total_sales: number;
  total_orders: number;
}

export interface DebtSummaryRow {
  total_debts: number;
  total_amount: number;
  avg_amount: number;
  min_amount: number;
  max_amount: number;
}

export interface DebtByYearRow {
  year: number;
  debt_count: number;
  total_amount: number;
}

export interface DebtSummaryResponse {
  summary: DebtSummaryRow | null;
  by_year: DebtByYearRow[];
}


// Recommendation response with charts and proof
export interface RecommendationCharts {
  purchase_history: { month: string; orders: number; amount: number }[];
  top_categories: { category: string; percentage: number }[];
  recommendation_sources: { source: string; count: number }[];
}

export interface RecommendationProof {
  total_orders: number;
  avg_order_value: number;
  last_order_date: string | null;
  days_since_last_order: number | null;
  loyalty_score: number;
  total_products_purchased: number;
  total_spent: number;
  model_confidence: number;
}

export interface FullRecommendationResponse {
  customer_id: number;
  client_name: string | null;
  segment: string;
  date: string;
  recommendations: { product_id: number; score: number; rank: number; segment: string; source: string }[];
  count: number;
  discovery_count: number;
  charts: RecommendationCharts;
  proof: RecommendationProof;
  cached: boolean;
  latency_ms: number;
}


// Product forecast response with charts and proof
export interface ProductSalesHistory {
  month: string;
  orders: number;
  qty: number;
  amount: number;
}

export interface ProductTopCustomer {
  customer_id: number;
  customer_name: string;
  total_qty: number;
  order_count: number;
  total_amount: number;
}

export interface ProductCharts {
  sales_history: ProductSalesHistory[];
  top_customers: ProductTopCustomer[];
}

export interface ProductProof {
  total_orders: number;
  total_qty_sold: number;
  total_revenue: number;
  unique_customers: number;
  avg_order_qty: number;
  last_sale_date: string | null;
  first_sale_date: string | null;
  days_since_last_sale: number | null;
  product_age_days: number | null;
}

export interface ProductForecastResponse {
  product_id: number;
  product_name: string | null;
  vendor_code: string | null;
  category: string | null;
  forecast_period_weeks: number;
  historical_weeks: number;
  summary: {
    total_predicted_quantity: number;
    total_predicted_revenue?: number | null;
    total_predicted_orders?: number | null;
    average_weekly_quantity: number | null;
    historical_average?: number | null;
    active_customers: number;
    at_risk_customers: number;
  };
  weekly_data: {
    week_start: string;
    week_end: string;
    data_type: 'actual' | 'predicted';
    quantity: number;
    revenue?: number | null;
    orders: number;
    predicted_quantity?: number;
    predicted_revenue?: number | null;
    predicted_orders?: number;
    confidence_lower?: number | null;
    confidence_upper?: number | null;
  }[];
  top_customers_by_volume: { customer_id: number; customer_name: string | null; predicted_quantity: number; contribution_pct: number }[];
  at_risk_customers: { customer_id: number; customer_name: string | null; last_order: string; expected_reorder?: string; days_overdue: number; churn_probability: number; action: string }[];
  model_metadata?: {
    model_type: string;
    training_customers: number;
    forecast_accuracy_estimate: number;
    seasonality_detected: boolean;
    model_version?: string;
    statistical_methods?: string[];
  };
  charts?: ProductCharts | null;
  proof?: ProductProof | null;
}

const fetchJson = async <T>(url: string, errorMessage: string): Promise<T> => {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(errorMessage);
  }
  return response.json() as Promise<T>;
};

// Check API status
export async function checkApiStatus(): Promise<ApiStatus> {
  try {
    const response = await fetch(`${SQL_API}/health`);
    return {
      online: response.ok,
      documents: null,
    };
  } catch {
    return {
      online: false,
      documents: null,
    };
  }
}

export async function fetchYearlySales(): Promise<YearlySalesRow[]> {
  const data = await fetchJson<{ data?: YearlySalesRow[] }>(
    `${ANALYTICS_API}/sales/yearly`,
    'Failed to load yearly sales.'
  );
  return Array.isArray(data?.data) ? data.data : [];
}

export async function fetchTopProducts(limit = 10): Promise<TopProductRow[]> {
  const data = await fetchJson<{ data?: TopProductRow[] }>(
    `${ANALYTICS_API}/products/top?limit=${limit}`,
    'Failed to load top products.'
  );
  return Array.isArray(data?.data) ? data.data : [];
}

export async function fetchTopClients(limit = 10): Promise<TopClientRow[]> {
  const data = await fetchJson<{ data?: TopClientRow[] }>(
    `${ANALYTICS_API}/clients/top?limit=${limit}`,
    'Failed to load top clients.'
  );
  return Array.isArray(data?.data) ? data.data : [];
}

export async function fetchDebtSummary(): Promise<DebtSummaryResponse> {
  const data = await fetchJson<DebtSummaryResponse>(
    `${ANALYTICS_API}/debts/summary`,
    'Failed to load debt summary.'
  );
  return {
    summary: data?.summary ?? null,
    by_year: Array.isArray(data?.by_year) ? data.by_year : [],
  };
}

// Region statistics for map visualization
export interface RegionStatsRow {
  region_code: string;
  region_name: string;
  client_count: number;
  total_sales: number;
  total_qty: number;
}

export interface RegionStatsResponse {
  success: boolean;
  regions: RegionStatsRow[];
  total_regions: number;
}

export async function fetchRegionStats(): Promise<RegionStatsRow[]> {
  const data = await fetchJson<RegionStatsResponse>(
    `${SQL_API}/regions/stats`,
    'Failed to load region statistics.'
  );
  return Array.isArray(data?.regions) ? data.regions : [];
}

// Ollama Text-to-SQL query - main entry point
export async function ollamaQuery(question: string, signal?: AbortSignal): Promise<OllamaQueryResult> {
  const response = await fetch(`${SQL_API}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      execute: true,
      include_explanation: true,
    }),
    signal,
  });

  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(payload.detail || payload.error || 'Query failed');
  }

  // API returns flat structure: { success, data, columns, row_count, error }
  // Frontend expects nested: { execution: { success, rows, columns, row_count, error } }
  return {
    ...payload,
    question: payload.question ?? question,
    mode: payload.mode ?? 'sql',
    success: payload.success ?? false,
    execution: {
      success: payload.success ?? false,
      rows: payload.data ?? [],
      columns: payload.columns ?? [],
      row_count: payload.row_count ?? 0,
      error: payload.error ?? null,
    },
  };
}

// Recommendations by client ID
export async function fetchRecommendationsForClient(clientId: string | number, signal?: AbortSignal): Promise<number[]> {
  const response = await fetch(`${RECO_API}/recommendations/${clientId}`, { signal });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || 'Failed to load recommendations.');
  }
  const items = Array.isArray(payload) ? payload : Array.isArray(payload.recommendations) ? payload.recommendations : [];
  return items.map((item: any) => Number(item?.product_id ?? item)).filter((n: number) => Number.isFinite(n));
}


// Full recommendations with charts and proof
export async function fetchFullRecommendations(clientId: string | number, signal?: AbortSignal): Promise<FullRecommendationResponse | null> {
  const response = await fetch(`${RECO_API}/recommendations/${clientId}`, { signal });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || 'Failed to load recommendations.');
  }
  return payload as FullRecommendationResponse;
}

// Forecast by product ID with charts and proof
export async function fetchForecastForProduct(productId: string | number, signal?: AbortSignal): Promise<ProductForecastResponse | null> {
  const response = await fetch(`${RECO_API}/forecast/${productId}`, { signal });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || 'Failed to load forecast.');
  }
  return payload as ProductForecastResponse;
}

// Fetch client details by ID (tries ClientAgreement first, then Client table)
export async function fetchClientById(clientId: string | number, signal?: AbortSignal): Promise<Record<string, unknown> | null> {
  const numericId = Number(clientId);
  if (!Number.isFinite(numericId)) return null;

  // Try ClientAgreement -> Client join first
  const sqlAgreement = `SELECT TOP 1 ca.ID, c.Name, c.FullName, c.FirstName, c.LastName FROM dbo.ClientAgreement ca INNER JOIN dbo.Client c ON ca.ClientID = c.ID WHERE ca.Deleted = 0 AND ca.ID = ${numericId};`;

  const response1 = await fetch(`${SQL_API}/execute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sql: sqlAgreement }),
    signal,
  });

  const payload1 = await response1.json().catch(() => ({}));
  const rows1 = payload1?.data ?? payload1?.rows ?? [];
  if (Array.isArray(rows1) && rows1.length > 0) {
    return rows1[0];
  }

  // Fallback: try Client table directly
  const sqlClient = `SELECT TOP 1 ID, Name, FullName, FirstName, LastName FROM dbo.Client WHERE Deleted = 0 AND ID = ${numericId};`;

  const response2 = await fetch(`${SQL_API}/execute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sql: sqlClient }),
    signal,
  });

  const payload2 = await response2.json().catch(() => ({}));
  const rows2 = payload2?.data ?? payload2?.rows ?? [];
  return Array.isArray(rows2) && rows2.length > 0 ? rows2[0] : null;
}

// Fetch product details directly via SQL API (simple lookup by IDs)
export async function fetchProductsByIds(ids: Array<string | number>, signal?: AbortSignal): Promise<Record<string, unknown>[]> {
  if (!ids || ids.length === 0) return [];
  const numericIds = Array.from(new Set(ids
    .map((id) => Number(id))
    .filter((id) => Number.isFinite(id))))
    .slice(0, 20);

  if (numericIds.length === 0) return [];

  const idList = numericIds.join(',');
  const sql = `SELECT TOP 50 ID, Name, VendorCode FROM dbo.Product WHERE Deleted = 0 AND ID IN (${idList});`;

  const response = await fetch(`${SQL_API}/execute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sql }),
    signal,
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload?.success === false) {
    throw new Error(payload.detail || payload.error || 'Failed to load product details.');
  }

  if (payload?.data && Array.isArray(payload.data)) {
    return payload.data as Record<string, unknown>[];
  }
  if (payload?.rows && Array.isArray(payload.rows)) {
    return payload.rows as Record<string, unknown>[];
  }
  return [];
}


// Client Payment Score types and API
export interface MonthlyScoreData {
  month: string;
  score: number;
}

export interface ClientScoreData {
  is_cold_start?: boolean;
  overall_score: number;
  score_grade: string;
  paid_order_count: number;
  avg_days_to_pay: number | null;
  on_time_percentage: number;
  paid_amount: number;
  unpaid_order_count: number;
  unpaid_amount: number;
  oldest_unpaid_days: number | null;
  paid_score_component: number;
  unpaid_score_component: number;
  monthly_scores: MonthlyScoreData[];
}

export interface ClientScoreResponse {
  client_id: number;
  client_name: string | null;
  score: ClientScoreData;
  latency_ms: number;
  timestamp: string;
}

// Fetch client payment score
export async function fetchClientScore(
  clientId: string | number,
  signal?: AbortSignal
): Promise<ClientScoreResponse | null> {
  const response = await fetch(`${RECO_API}/client-score/${clientId}`, { signal });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || 'Failed to load client payment score.');
  }
  return payload as ClientScoreResponse;
}

// Storage types and API
export interface StorageItem {
  id: number;
  name: string;
}

export interface StoragesResponse {
  success: boolean;
  storages: StorageItem[];
  count: number;
  error?: string;
}

// Fetch all storages for the storage list panel
export async function fetchStorages(
  signal?: AbortSignal
): Promise<StoragesResponse> {
  const response = await fetch(`${SQL_API}/storages`, { signal });
  const payload = await response.json().catch(() => ({ success: false, storages: [], count: 0 }));
  return payload as StoragesResponse;
}

// Manager types and API
export interface ManagerItem {
  id: number;
  name: string;
}

export interface ManagersResponse {
  success: boolean;
  managers: ManagerItem[];
  count: number;
  error?: string;
}

// Fetch all managers for the manager list panel
export async function fetchManagers(
  signal?: AbortSignal
): Promise<ManagersResponse> {
  const response = await fetch(`${SQL_API}/managers`, { signal });
  const payload = await response.json().catch(() => ({ success: false, managers: [], count: 0 }));
  return payload as ManagersResponse;
}

// Order Recommendations types and API
export interface OrderRecommendationRequest {
  as_of_date?: string;
  manufacturing_days?: number;
  logistics_days?: number;
  warehouse_days?: number;
  service_level?: number;
  history_weeks?: number;
  min_recommend_qty?: number;
  product_ids?: number[];
  supplier_id?: number;
  max_products?: number;
}

export interface OrderRecommendationItem {
  product_id: number;
  product_name: string | null;
  vendor_code: string | null;
  on_hand: number;
  inbound_open: number;
  inventory_position: number;
  avg_weekly_demand: number;
  std_weekly_demand: number;
  lead_time_weeks: number;
  demand_during_lead_time: number;
  safety_stock: number;
  reorder_point: number;
  recommended_qty: number;
  expected_arrival_date: string;
}

export interface SupplierRecommendation {
  supplier_id: number | null;
  supplier_name: string | null;
  total_recommended_qty: number;
  products: OrderRecommendationItem[];
}

export interface OrderRecommendationResponse {
  as_of_date: string;
  manufacturing_days: number;
  logistics_days: number;
  warehouse_days: number;
  lead_time_days: number;
  service_level: number;
  history_weeks: number;
  recommendations: SupplierRecommendation[];
  count: number;
  latency_ms: number;
  timestamp: string;
}

// Fetch order recommendations
export async function fetchOrderRecommendations(
  request: OrderRecommendationRequest,
  signal?: AbortSignal
): Promise<OrderRecommendationResponse> {
  const response = await fetch(`${RECO_API}/order-recommendations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || 'Failed to fetch order recommendations.');
  }
  return payload as OrderRecommendationResponse;
}

// Order Recommendations V2 types and API (with trend, seasonality, churn)
export interface OrderRecommendationRequestV2 extends OrderRecommendationRequest {
  use_trend_adjustment?: boolean;
  use_seasonality?: boolean;
  use_churn_adjustment?: boolean;
  min_history_weeks?: number;
}

export interface OrderRecommendationItemV2 extends OrderRecommendationItem {
  // Trend fields
  trend_factor: number | null;
  trend_direction: string | null; // 'growing' | 'declining' | 'stable'
  // Seasonality fields
  seasonal_index: number | null;
  seasonal_period_weeks: number | null;
  // Churn fields
  churn_adjustment: number | null;
  at_risk_demand_pct: number | null;
  // Forecast metadata
  forecast_method: string;
  forecast_confidence: number | null;
  data_weeks: number | null;
}

export interface SupplierRecommendationV2 {
  supplier_id: number | null;
  supplier_name: string | null;
  total_recommended_qty: number;
  products: OrderRecommendationItemV2[];
}

export interface OrderRecommendationResponseV2 {
  as_of_date: string;
  manufacturing_days: number;
  logistics_days: number;
  warehouse_days: number;
  lead_time_days: number;
  service_level: number;
  history_weeks: number;
  // V2 metadata
  use_trend_adjustment: boolean;
  use_seasonality: boolean;
  use_churn_adjustment: boolean;
  products_with_trend: number;
  products_with_seasonality: number;
  products_with_churn_risk: number;
  // Results
  recommendations: SupplierRecommendationV2[];
  count: number;
  latency_ms: number;
  timestamp: string;
}

// Fetch order recommendations v2 (enhanced)
export async function fetchOrderRecommendationsV2(
  request: OrderRecommendationRequestV2,
  signal?: AbortSignal
): Promise<OrderRecommendationResponseV2> {
  const response = await fetch(`${RECO_API}/order-recommendations/v2`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || 'Failed to fetch order recommendations (v2).');
  }
  return payload as OrderRecommendationResponseV2;
}
