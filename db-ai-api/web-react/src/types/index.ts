// API Response Types
export interface ApiStatus {
  online: boolean;
  documents: number | null;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export interface YearlySalesData {
  year: number;
  total_sales: number;
  total_orders: number;
  total_items: number;
}

export interface YearlyItemsData {
  year: number;
  total_quantity: number;
  unique_products: number;
}

export interface TopProduct {
  product_name: string;
  total_qty: number;
  order_count: number;
}

export interface TopClient {
  client_name: string;
  total_sales: number;
  total_orders: number;
}

export interface DebtSummary {
  summary: {
    total_amount: number;
    total_debts: number;
    avg_amount: number;
  };
  by_year: Array<{
    year: number;
    debt_count: number;
    total_amount: number;
  }>;
}

export interface ProductSearchResult {
  product_name: string;
  vendor_code: string;
  total_sold?: number;
  order_count?: number;
}

export interface SearchResult {
  table: string;
  name: string;
  similarity: number;
}

export interface SearchResponse {
  results: SearchResult[];
  n_results: number;
  detected_regions?: string[];
}

export interface OllamaQueryResult {
  sql: string;
  explanation?: string;
  execution?: {
    success: boolean;
    results?: Record<string, unknown>[];
    columns?: string[];
    error?: string;
  };
}

export type QueryType =
  | 'sales'
  | 'top_products'
  | 'top_clients'
  | 'debts'
  | 'product_keyword_search'
  | 'region'
  | 'client_search'
  | 'search';

export type ChartType = 'sales_yearly' | 'top_products' | 'debts';
