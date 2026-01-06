import type { ApiStatus, OllamaQueryResult } from '../types';

// API Configuration
const SQL_API = import.meta.env.VITE_SQL_API || 'http://localhost:8000';
const ANALYTICS_API = import.meta.env.VITE_ANALYTICS_API || SQL_API;
const RECO_API = import.meta.env.VITE_RECO_API || 'http://localhost:8100';

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
