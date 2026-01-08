import React, { useState } from 'react';
import { SqlResultResponse, ColumnDefinition } from '../../../types/responses';
import { CodeBlock } from './CodeBlock';
import { DataTable } from '../DataTable/DataTable';
import type { Language } from '../../WelcomeMessage';

const sqlTranslations = {
  uk: {
    sqlQuery: 'SQL запит',
    executionTime: 'Час виконання',
    queryFailed: 'Запит не вдався',
    unableToFetch: 'Не вдалося отримати результати.',
  },
  en: {
    sqlQuery: 'SQL query',
    executionTime: 'Execution time',
    queryFailed: 'Query failed',
    unableToFetch: 'Unable to fetch results.',
  },
};

// Column name translations (SQL column names → human-readable labels)
const columnLabels: Record<string, { uk: string; en: string; type?: 'number' | 'currency' | 'qty' }> = {
  // Common names
  name: { uk: 'Назва', en: 'Name' },
  id: { uk: 'ID', en: 'ID' },

  // Warehouse/Stock
  warehousename: { uk: 'Склад', en: 'Warehouse' },
  warehouse_name: { uk: 'Склад', en: 'Warehouse' },
  warehouse: { uk: 'Склад', en: 'Warehouse' },

  // Quantities
  qty: { uk: 'Кількість', en: 'Quantity', type: 'qty' },
  quantity: { uk: 'Кількість', en: 'Quantity', type: 'qty' },
  total_qty: { uk: 'Загальна к-сть', en: 'Total Qty', type: 'qty' },
  totalqty: { uk: 'Загальна к-сть', en: 'Total Qty', type: 'qty' },
  stock_qty: { uk: 'Залишок', en: 'Stock Qty', type: 'qty' },
  stockqty: { uk: 'Залишок', en: 'Stock Qty', type: 'qty' },
  product_count: { uk: 'К-сть товарів', en: 'Product Count', type: 'qty' },
  productcount: { uk: 'К-сть товарів', en: 'Product Count', type: 'qty' },
  count: { uk: 'Кількість', en: 'Count', type: 'qty' },

  // Amounts/Money
  amount: { uk: 'Сума', en: 'Amount', type: 'currency' },
  total_amount: { uk: 'Загальна сума', en: 'Total Amount', type: 'currency' },
  totalamount: { uk: 'Загальна сума', en: 'Total Amount', type: 'currency' },
  sum: { uk: 'Сума', en: 'Sum', type: 'currency' },
  total: { uk: 'Всього', en: 'Total', type: 'currency' },
  price: { uk: 'Ціна', en: 'Price', type: 'currency' },
  cost: { uk: 'Вартість', en: 'Cost', type: 'currency' },
  revenue: { uk: 'Дохід', en: 'Revenue', type: 'currency' },

  // Products
  product: { uk: 'Товар', en: 'Product' },
  productname: { uk: 'Товар', en: 'Product' },
  product_name: { uk: 'Товар', en: 'Product' },
  vendorcode: { uk: 'Артикул', en: 'SKU' },
  vendor_code: { uk: 'Артикул', en: 'SKU' },
  sku: { uk: 'Артикул', en: 'SKU' },

  // Clients
  client: { uk: 'Клієнт', en: 'Client' },
  clientname: { uk: 'Клієнт', en: 'Client' },
  client_name: { uk: 'Клієнт', en: 'Client' },
  customer: { uk: 'Клієнт', en: 'Customer' },

  // Orders
  orders: { uk: 'Замовлення', en: 'Orders', type: 'qty' },
  order_count: { uk: 'К-сть замовлень', en: 'Order Count', type: 'qty' },
  ordercount: { uk: 'К-сть замовлень', en: 'Order Count', type: 'qty' },

  // Dates
  date: { uk: 'Дата', en: 'Date' },
  created: { uk: 'Створено', en: 'Created' },
  updated: { uk: 'Оновлено', en: 'Updated' },

  // Regions
  region: { uk: 'Регіон', en: 'Region' },
  regionname: { uk: 'Регіон', en: 'Region' },
  region_name: { uk: 'Регіон', en: 'Region' },
};

// Get translated column label
const getColumnLabel = (col: string, lang: Language): string => {
  const key = col.toLowerCase().replace(/[^a-z0-9_]/g, '');
  const translation = columnLabels[key];
  if (translation) {
    return translation[lang];
  }
  // Fallback: capitalize and replace underscores
  return col.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
};

// Get column type hint from translations
const getColumnTypeHint = (col: string): 'currency' | 'qty' | undefined => {
  const key = col.toLowerCase().replace(/[^a-z0-9_]/g, '');
  return columnLabels[key]?.type as 'currency' | 'qty' | undefined;
};

// Detect aggregate functions in SQL and generate explanation
const detectAggregates = (sql: string | undefined, lang: Language): string | null => {
  if (!sql) return null;

  const sqlUpper = sql.toUpperCase();
  const aggregates: string[] = [];

  if (sqlUpper.includes('SUM(')) {
    aggregates.push(lang === 'uk' ? 'сума значень' : 'sum of values');
  }
  if (sqlUpper.includes('COUNT(')) {
    aggregates.push(lang === 'uk' ? 'кількість записів' : 'count of records');
  }
  if (sqlUpper.includes('AVG(')) {
    aggregates.push(lang === 'uk' ? 'середнє значення' : 'average value');
  }
  if (sqlUpper.includes('MAX(')) {
    aggregates.push(lang === 'uk' ? 'максимальне значення' : 'maximum value');
  }
  if (sqlUpper.includes('MIN(')) {
    aggregates.push(lang === 'uk' ? 'мінімальне значення' : 'minimum value');
  }

  if (aggregates.length === 0) return null;

  const prefix = lang === 'uk' ? 'Розрахунки: ' : 'Calculations: ';
  return prefix + aggregates.join(', ');
};

export const SqlResultCard: React.FC<SqlResultResponse & { language?: Language }> = ({
  title,
  sql,
  explanation,
  result,
  executionTime,
  showSql = true,
  language = 'uk',
}) => {
  const t = sqlTranslations[language];
  const [isSqlExpanded, setIsSqlExpanded] = useState(false);

  // Auto-detect aggregates for explanation
  const aggregateExplanation = detectAggregates(sql, language);

  const columns = result?.columns || [];
  const rows = result?.rows || [];

  const tableColumns: ColumnDefinition[] = columns.map((col) => {
    const typeHint = getColumnTypeHint(col);
    let colType = detectColumnType(rows[0]?.[col]);

    // Override with type hint if available
    if (typeHint === 'currency') colType = 'currency';

    return {
      key: col,
      label: getColumnLabel(col, language),
      sortable: true,
      type: colType,
      align: (colType === 'number' || colType === 'currency') ? 'right' as const : undefined,
    };
  });

  return (
    <div className="sql-result-card space-y-4">
      {(title || explanation) && (
        <div className="grok-card border-white/10 p-4">
          {title && (
            <h3 className="font-semibold text-slate-100 flex items-center gap-2 mb-2">
              <svg className="w-5 h-5 text-sky-200 thin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <rect x="5" y="4" width="14" height="16" rx="2" />
                <path d="M9 4v4h6V4" />
              </svg>
              {title}
            </h3>
          )}
          {explanation && (
            <p className="text-slate-300 text-sm leading-relaxed">{explanation}</p>
          )}
          {executionTime && (
            <div className="mt-2 text-xs text-slate-400">
              {t.executionTime}: {executionTime}ms
            </div>
          )}
        </div>
      )}

      {showSql && sql && (
        <div className="grok-card border-white/10 overflow-hidden">
          <button
            onClick={() => setIsSqlExpanded(!isSqlExpanded)}
            className="w-full px-4 py-3 flex items-center justify-between text-left bg-white/5 hover:bg-white/10 transition-colors"
          >
            <span className="flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-slate-300">
              <svg className="w-4 h-4 thin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path d="M8 17l-4-5 4-5" />
                <path d="M16 7l4 5-4 5" />
              </svg>
              {t.sqlQuery}
            </span>
            <svg
              className={`w-4 h-4 text-slate-400 thin-icon transition-transform ${isSqlExpanded ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path d="M6 9l6 6 6-6" />
            </svg>
          </button>

          {isSqlExpanded && (
            <div className="p-4 border-t border-white/10">
              <CodeBlock code={sql} language="sql" />
            </div>
          )}
        </div>
      )}

      {/* Aggregate/Calculation Explanation */}
      {aggregateExplanation && result?.success && (
        <div className="flex items-center gap-2 px-4 py-2.5 bg-blue-50 border border-blue-200 rounded-lg text-sm">
          <svg className="w-4 h-4 text-blue-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span className="text-blue-700">{aggregateExplanation}</span>
        </div>
      )}

      {result?.success ? (
        <DataTable
          type="data_table"
          columns={tableColumns}
          rows={rows}
          totalRows={result?.row_count}
          enableSorting={true}
          enableExport={true}
          language={language}
        />
      ) : (
        <div className="p-4 rounded-xl border border-rose-400/30 bg-rose-400/10">
          <h4 className="font-semibold text-rose-200 mb-2 flex items-center gap-2">
            <svg className="w-5 h-5 thin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path d="M12 9v4" />
              <path d="M12 17h.01" />
              <circle cx="12" cy="12" r="9" />
            </svg>
            {t.queryFailed}
          </h4>
          <p className="text-rose-200/80">{result?.error || t.unableToFetch}</p>
        </div>
      )}
    </div>
  );
};

function detectColumnType(value: unknown): ColumnDefinition['type'] {
  if (value === null || value === undefined) return 'string';

  if (typeof value === 'number') {
    return 'number';
  }

  if (typeof value === 'boolean') {
    return 'boolean';
  }

  if (typeof value === 'string') {
    if (/^\d{4}-\d{2}-\d{2}/.test(value)) {
      return 'date';
    }
    if (/UAH|USD|EUR|GBP/i.test(value)) {
      return 'currency';
    }
    if (/^\d+\.?\d*%$/.test(value)) {
      return 'percent';
    }
  }

  return 'string';
}

export default SqlResultCard;
