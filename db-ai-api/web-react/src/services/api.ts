import type { ApiStatus, OllamaQueryResult } from '../types';

// API Configuration
const ANALYTICS_API = import.meta.env.VITE_ANALYTICS_API || 'http://localhost:8001';
const SQL_API = import.meta.env.VITE_SQL_API || 'http://localhost:8002';

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
  const response = await fetch(`${SQL_API}/web/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      mode: 'auto',
    }),
    signal,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || 'Query failed');
  }

  return response.json();
}
