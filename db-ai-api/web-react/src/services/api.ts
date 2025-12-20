import type {
  ApiStatus,
  YearlySalesData,
  YearlyItemsData,
  TopProduct,
  TopClient,
  DebtSummary,
  ProductSearchResult,
  SearchResponse,
  OllamaQueryResult,
} from '../types';

// API Configuration - use environment variables or defaults
const RAG_API = import.meta.env.VITE_RAG_API || 'http://localhost:8000';
const ANALYTICS_API = import.meta.env.VITE_ANALYTICS_API || 'http://localhost:8001';
const SQL_API = import.meta.env.VITE_SQL_API || 'http://localhost:8002';

// Check API status
export async function checkApiStatus(): Promise<ApiStatus> {
  try {
    const response = await fetch(`${RAG_API}/`);
    const data = await response.json();
    return {
      online: true,
      documents: data.documents || null,
    };
  } catch {
    return {
      online: false,
      documents: null,
    };
  }
}

// Fetch yearly sales data
export async function fetchYearlySales(): Promise<YearlySalesData[]> {
  const response = await fetch(`${ANALYTICS_API}/sales/yearly`);
  const data = await response.json();
  return data.data;
}

// Fetch yearly items data
export async function fetchYearlyItems(): Promise<YearlyItemsData[]> {
  const response = await fetch(`${ANALYTICS_API}/orderitems/yearly`);
  const data = await response.json();
  return data.data;
}

// Fetch top products
export async function fetchTopProducts(limit = 10): Promise<TopProduct[]> {
  const response = await fetch(`${ANALYTICS_API}/products/top?limit=${limit}`);
  const data = await response.json();
  return data.data;
}

// Fetch top clients
export async function fetchTopClients(limit = 10): Promise<TopClient[]> {
  const response = await fetch(`${ANALYTICS_API}/clients/top?limit=${limit}`);
  const data = await response.json();
  return data.data;
}

// Fetch debt summary
export async function fetchDebtSummary(): Promise<DebtSummary> {
  const response = await fetch(`${ANALYTICS_API}/debts/summary`);
  return response.json();
}

// Search products by keyword
export async function searchProducts(
  keyword: string,
  limit = 30,
  sortBySales = false
): Promise<{ products: ProductSearchResult[]; count: number }> {
  const response = await fetch(
    `${ANALYTICS_API}/products/search?q=${encodeURIComponent(keyword)}&limit=${limit}&sort_by_sales=${sortBySales}`
  );
  return response.json();
}

// Smart RAG search
export async function smartSearch(query: string, n = 20): Promise<SearchResponse> {
  const response = await fetch(`${RAG_API}/search?q=${encodeURIComponent(query)}&n=${n}`);
  return response.json();
}

// Ollama Text-to-SQL query
export async function ollamaQuery(question: string): Promise<OllamaQueryResult> {
  const response = await fetch(`${SQL_API}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      execute: true,
      max_rows: 100,
      include_explanation: true,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'SQL generation failed');
  }

  return response.json();
}
