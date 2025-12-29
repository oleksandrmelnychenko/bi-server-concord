// Response Types for Structured Chat Responses

// Base response type
export type ResponseType =
  | 'data_table'
  | 'statistics'
  | 'chart'
  | 'map'
  | 'sql_result'
  | 'grouped_results'
  | 'text'
  | 'error';

// Column definition for tables
export interface ColumnDefinition {
  key: string;
  label: string;
  type?: 'string' | 'number' | 'currency' | 'date' | 'boolean' | 'percent';
  sortable?: boolean;
  width?: string;
  align?: 'left' | 'center' | 'right';
}

// Data Table Response
export interface DataTableResponse {
  type: 'data_table';
  title?: string;
  columns: ColumnDefinition[];
  rows: Record<string, unknown>[];
  totalRows?: number;
  enableSorting?: boolean;
  enableExport?: boolean;
  maxHeight?: string;
}

// Stat Card definition
export interface StatCardData {
  value: string | number;
  label: string;
  icon?: 'users' | 'money' | 'chart' | 'box' | 'truck' | 'warning' | 'check';
  trend?: {
    value: number;
    direction: 'up' | 'down';
  };
  highlight?: boolean;
  format?: 'number' | 'currency' | 'percent';
}

// Statistics Response
export interface StatisticsResponse {
  type: 'statistics';
  title?: string;
  cards: StatCardData[];
  layout?: 'grid-2' | 'grid-3' | 'grid-4';
}

// Chart data for Chart.js
export interface ChartDataset {
  label?: string;
  data: number[];
  backgroundColor?: string | string[];
  borderColor?: string | string[];
  borderWidth?: number;
}

export interface ChartData {
  labels: string[];
  datasets: ChartDataset[];
}

// Chart Response
export interface ChartResponse {
  type: 'chart';
  title?: string;
  chartType: 'bar' | 'line' | 'pie' | 'doughnut' | 'horizontal-bar';
  data: ChartData;
  height?: number;
  expandable?: boolean;
}

// Map Marker for region visualization
export interface MapMarker {
  regionCode: string;
  value: number;
  label: string;
  secondaryValue?: number;
  secondaryLabel?: string;
}

// Map Response for region visualization
export interface MapResponse {
  type: 'map';
  title?: string;
  markers: MapMarker[];
  mapType: 'markers' | 'heatmap' | 'choropleth';
  valueFormat?: 'number' | 'currency' | 'percent';
  interactive?: boolean;
  height?: number;
}

// SQL Result Response
export interface SqlResultResponse {
  type: 'sql_result';
  title?: string;
  sql: string;
  explanation?: string;
  result: {
    success: boolean;
    columns: string[];
    rows: Record<string, unknown>[];
    row_count: number;
    error?: string;
  };
  executionTime?: number;
  showSql?: boolean;
}

// Result Group for grouped results
export interface ResultGroup {
  label: string;
  badge?: string;
  content: DataTableResponse | StatisticsResponse | ChartResponse;
}

// Grouped Results Response
export interface GroupedResultsResponse {
  type: 'grouped_results';
  title?: string;
  groups: ResultGroup[];
  collapsible?: boolean;
  defaultExpanded?: boolean;
}

// Text Response (simple markdown-like text)
export interface TextResponse {
  type: 'text';
  content: string;
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info';
}

// Error Response
export interface ErrorResponse {
  type: 'error';
  title?: string;
  message: string;
  details?: string;
  retryable?: boolean;
}

// Union type for all response sections
export type ResponseSection =
  | DataTableResponse
  | StatisticsResponse
  | ChartResponse
  | MapResponse
  | SqlResultResponse
  | GroupedResultsResponse
  | TextResponse
  | ErrorResponse;

// Main Structured Response (can contain multiple sections)
export interface StructuredResponse {
  sections: ResponseSection[];
}

// Sort state for tables
export interface SortState {
  column: string;
  direction: 'asc' | 'desc';
}

// Export format options
export type ExportFormat = 'csv' | 'json';

// Color palette for charts (Ukrainian business theme)
export const CHART_COLORS = {
  primary: '#38bdf8',    // Sky
  secondary: '#22d3ee',  // Cyan
  success: '#34d399',    // Green
  warning: '#fbbf24',    // Amber
  danger: '#fb7185',     // Rose
  info: '#60a5fa',       // Blue
  palette: [
    '#38bdf8', '#22d3ee', '#34d399', '#f59e0b',
    '#f97316', '#60a5fa', '#f43f5e', '#14b8a6',
    '#0ea5e9', '#fbbf24', '#2dd4bf', '#93c5fd'
  ]
};

// Ukrainian number formatting helper type
export interface FormatOptions {
  locale?: string;
  currency?: string;
  decimals?: number;
}
