import { useCallback } from 'react';
import { ollamaQuery } from '../services/api';
import { isRegionMapQuery, hasRegionData } from '../services/queryDetector';
import { UKRAINE_REGIONS, getRegionByCode } from '../constants/regions';
import type {
  StructuredResponse,
  DataTableResponse,
  ChartResponse,
  TextResponse,
  StatisticsResponse,
  MapResponse,
  MapMarker,
  StatCardData,
  ColumnDefinition,
} from '../types/responses';

const wantsChart = (query: string): boolean => {
  const keywords = ['chart', 'graph', 'plot', 'bar', 'line', 'pie'];
  const lower = query.toLowerCase();
  return keywords.some((kw) => lower.includes(kw));
};

const rankingKeywords = [
  'top',
  'most',
  'highest',
  'largest',
  'biggest',
  'best',
  'ranking',
  'rank',
  'leader',
  'leaders',
  'топ',
  'лідер',
  'лідери',
  'найбільш',
  'найбільше',
  'найкращ',
  'максим',
];

const isRankingQuery = (query: string): boolean => {
  const lower = query.toLowerCase();
  return rankingKeywords.some((kw) => lower.includes(kw));
};

const hasCyrillic = (text: string): boolean => /[\u0400-\u04FF]/.test(text);

const pluralizeUk = (count: number, one: string, few: string, many: string): string => {
  const mod10 = count % 10;
  const mod100 = count % 100;
  if (mod10 === 1 && mod100 !== 11) return one;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return few;
  return many;
};

const inferEntityIcon = (columnName: string): StatCardData['icon'] => {
  const lower = columnName.toLowerCase();
  if (['client', 'customer', 'buyer', 'company', 'partner', 'name'].some((kw) => lower.includes(kw))) {
    return 'users';
  }
  if (['product', 'item', 'sku', 'brand'].some((kw) => lower.includes(kw))) {
    return 'box';
  }
  if (['order', 'shipment', 'delivery'].some((kw) => lower.includes(kw))) {
    return 'truck';
  }
  return 'chart';
};

const detectMonthLabel = (query: string, isUkrainian: boolean): string | null => {
  const lower = query.toLowerCase();
  const months = [
    { en: ['january', 'jan'], uk: ['січ'], labelEn: 'January', labelUk: 'Січень' },
    { en: ['february', 'feb'], uk: ['лют'], labelEn: 'February', labelUk: 'Лютий' },
    { en: ['march', 'mar'], uk: ['берез'], labelEn: 'March', labelUk: 'Березень' },
    { en: ['april', 'apr'], uk: ['квіт'], labelEn: 'April', labelUk: 'Квітень' },
    { en: ['may'], uk: ['трав'], labelEn: 'May', labelUk: 'Травень' },
    { en: ['june', 'jun'], uk: ['черв'], labelEn: 'June', labelUk: 'Червень' },
    { en: ['july', 'jul'], uk: ['лип'], labelEn: 'July', labelUk: 'Липень' },
    { en: ['august', 'aug'], uk: ['серп'], labelEn: 'August', labelUk: 'Серпень' },
    { en: ['september', 'sep'], uk: ['верес'], labelEn: 'September', labelUk: 'Вересень' },
    { en: ['october', 'oct'], uk: ['жовт'], labelEn: 'October', labelUk: 'Жовтень' },
    { en: ['november', 'nov'], uk: ['листопад'], labelEn: 'November', labelUk: 'Листопад' },
    { en: ['december', 'dec'], uk: ['груд'], labelEn: 'December', labelUk: 'Грудень' },
  ];

  for (const month of months) {
    if (month.en.some((token) => lower.includes(token)) || month.uk.some((token) => lower.includes(token))) {
      return isUkrainian ? month.labelUk : month.labelEn;
    }
  }
  return null;
};

const humanizeColumn = (value: string): string =>
  value
    .replace(/_/g, ' ')
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/\s+/g, ' ')
    .trim();

const isNumericValue = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

const toNumber = (value: unknown): number => {
  if (isNumericValue(value)) return value;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
};

const pickLabelColumn = (columns: string[], rows: Record<string, unknown>[]): string | null => {
  const stringColumns = columns.filter((col) => rows.some((row) => typeof row[col] === 'string'));
  if (stringColumns.length === 0) return null;

  const preferred = stringColumns.find((col) =>
    ['name', 'client', 'customer', 'product', 'vendor', 'brand'].some((kw) => col.toLowerCase().includes(kw))
  );

  return preferred || stringColumns[0];
};

const pickValueColumn = (columns: string[], rows: Record<string, unknown>[]): string | null => {
  const numericColumns = columns.filter((col) =>
    rows.every((row) => row[col] === null || row[col] === undefined || isNumericValue(row[col]))
  );
  if (numericColumns.length === 0) return null;

  const priority = [
    'total',
    'amount',
    'sum',
    'revenue',
    'sales',
    'qty',
    'quantity',
    'count',
    'orders',
    'value',
    'payment',
    'debt',
    'spent',
    'volume',
  ];

  const preferred = numericColumns.find((col) =>
    priority.some((kw) => col.toLowerCase().includes(kw))
  );

  return preferred || numericColumns[0];
};

const inferValueFormat = (columnName: string): 'currency' | 'number' | 'percent' => {
  const lower = columnName.toLowerCase();
  if (lower.includes('percent') || lower.includes('rate')) return 'percent';
  if (['amount', 'total', 'sum', 'revenue', 'sales', 'payment', 'debt', 'spent'].some((kw) => lower.includes(kw))) {
    return 'currency';
  }
  return 'number';
};

export type QueryResult = {
  structured: StructuredResponse;
};

const isAbortError = (error: unknown): boolean => {
  if (error instanceof DOMException) {
    return error.name === 'AbortError';
  }
  return (error as Error | undefined)?.name === 'AbortError';
};

// Find region code column in results
const findRegionColumn = (columns: string[]): string | null => {
  const regionPatterns = ['regioncode', 'region_code', 'region', 'regionname', 'region_name'];
  for (const col of columns) {
    if (regionPatterns.some((p) => col.toLowerCase().includes(p))) {
      return col;
    }
  }
  return null;
};

// Build map markers from query results
const buildRegionMarkers = (
  rows: Record<string, unknown>[],
  columns: string[],
  valueColumn: string | null
): MapMarker[] => {
  const regionColumn = findRegionColumn(columns);
  if (!regionColumn) return [];

  const markers: MapMarker[] = [];
  const regionValues: Record<string, { total: number; count: number }> = {};

  // Aggregate values by region
  for (const row of rows) {
    let regionCode = String(row[regionColumn] || '').toUpperCase();

    // Normalize region code (handle full names or variations)
    if (regionCode.length > 2) {
      // Try to extract 2-letter code or map from name
      const region = Object.values(UKRAINE_REGIONS).find(
        (r) => r.nameUk.toLowerCase().includes(regionCode.toLowerCase()) || r.name.toLowerCase().includes(regionCode.toLowerCase())
      );
      if (region) {
        regionCode = region.code;
      }
    }

    if (!getRegionByCode(regionCode)) continue;

    const value = valueColumn ? Number(row[valueColumn]) || 0 : 1;

    if (!regionValues[regionCode]) {
      regionValues[regionCode] = { total: 0, count: 0 };
    }
    regionValues[regionCode].total += value;
    regionValues[regionCode].count += 1;
  }

  // Convert to markers
  for (const [code, data] of Object.entries(regionValues)) {
    const region = getRegionByCode(code);
    if (!region) continue;

    markers.push({
      regionCode: code,
      value: data.total,
      label: valueColumn ? humanizeColumn(valueColumn) : 'Кількість',
      secondaryValue: data.count,
      secondaryLabel: 'записів',
    });
  }

  return markers.sort((a, b) => b.value - a.value);
};

export function useQueryHandler() {
  const processQuery = useCallback(async (query: string, signal?: AbortSignal): Promise<QueryResult> => {
    try {
      const data = await ollamaQuery(query, signal);
      const isUkrainian = hasCyrillic(query);
      const showChart = wantsChart(query);

      if (data.mode === 'rag' || (!data.execution && data.answer)) {
        const content =
          data.success === false
            ? `Query failed: ${data.error || 'RAG request failed.'}`
            : data.answer || 'No answer available for that request.';
        const variant = data.success === false ? 'error' : 'info';

        return {
          structured: {
            sections: [
              {
                type: 'text',
                content,
                variant,
              } as TextResponse,
            ],
          },
        };
      }

      if (data.execution?.success && Array.isArray(data.execution.rows) && data.execution.rows.length > 0) {
        const results = data.execution.rows;
        const columns = Array.isArray(data.execution.columns) ? data.execution.columns : [];
        const rowCount = typeof data.execution.row_count === 'number' ? data.execution.row_count : results.length;
        const shownCount = results.length;
        const isRanking = isRankingQuery(query);
        const monthLabel = detectMonthLabel(query, isUkrainian);
        const canRenderChart = showChart || isRanking;
        const isSingleRow = results.length === 1;

        const summary = isRanking
          ? (() => {
              const topLabel = (() => {
                if (shownCount === 1) {
                  if (monthLabel) {
                    return isUkrainian ? `Лідер за ${monthLabel}.` : `Leader for ${monthLabel}.`;
                  }
                  return isUkrainian ? 'Лідер.' : 'Leader.';
                }
                if (monthLabel) {
                  return isUkrainian
                    ? `Топ ${shownCount} за ${monthLabel}.`
                    : `Top ${shownCount} for ${monthLabel}.`;
                }
                return isUkrainian ? `Топ ${shownCount}.` : `Top ${shownCount}.`;
              })();
              if (rowCount > shownCount) {
                return isUkrainian
                  ? `${topLabel} Показано ${shownCount} з ${rowCount}.`
                  : `${topLabel} Showing ${shownCount} of ${rowCount}.`;
              }
              return topLabel;
            })()
          : rowCount === 1
            ? isUkrainian
              ? 'Знайдено 1 результат.'
              : 'Found 1 result.'
            : isUkrainian
              ? `Знайдено ${rowCount} ${pluralizeUk(rowCount, 'рядок', 'рядки', 'рядків')}.`
              : `Found ${rowCount} rows.`;

        const sections: StructuredResponse['sections'] = [
          {
            type: 'text',
            content: summary,
            variant: 'info',
          } as TextResponse,
        ];

        const labelColumn = (isRanking || isSingleRow) ? pickLabelColumn(columns, results) : null;
        const valueColumn = (isRanking || isSingleRow) ? pickValueColumn(columns, results) : null;
        const rankingReady = Boolean(isRanking && labelColumn && valueColumn);

        let tableRows = results.slice(0, 100);
        let tableColumns: ColumnDefinition[] = columns.map((col) => ({ key: col, label: col, sortable: true }));

        if (rankingReady && labelColumn && valueColumn) {
          const sortedResults = [...results].sort((a, b) => toNumber(b[valueColumn]) - toNumber(a[valueColumn]));
          tableRows = sortedResults.slice(0, 100).map((row, index) => ({
            Rank: index + 1,
            ...row,
          }));
          tableColumns = [
            { key: 'Rank', label: '#', sortable: true, type: 'number', align: 'right', width: '70px' },
            ...tableColumns,
          ];

          const valueLabel = humanizeColumn(valueColumn);
          const valueFormat = inferValueFormat(valueColumn);
          const topRow = sortedResults[0];
          const topName = String(topRow[labelColumn] ?? '');
          const topValue = toNumber(topRow[valueColumn]);
          const totalValue = sortedResults.reduce((sum, row) => sum + toNumber(row[valueColumn]), 0);
          const averageValue = totalValue / sortedResults.length;
          const shortTopName = topName.length > 34 ? `${topName.slice(0, 34)}...` : topName;

          sections.push({
            type: 'statistics',
            title: isUkrainian ? 'Підсумок' : 'Highlights',
            layout: 'grid-4',
            cards: [
              {
                value: shortTopName || (isUkrainian ? 'Невідомо' : 'Unknown'),
                label: isUkrainian ? 'Лідер' : 'Leader',
                icon: 'users',
              },
              {
                value: topValue,
                label: isUkrainian ? `Найвище ${valueLabel}` : `Top ${valueLabel}`,
                icon: 'money',
                format: valueFormat,
                highlight: true,
              },
              {
                value: totalValue,
                label: isUkrainian ? `Разом ${valueLabel}` : `Total ${valueLabel}`,
                icon: 'chart',
                format: valueFormat,
              },
              {
                value: averageValue,
                label: isUkrainian ? `Середнє ${valueLabel}` : `Average ${valueLabel}`,
                icon: 'check',
                format: valueFormat,
              },
            ],
          } as StatisticsResponse);

          if (canRenderChart && sortedResults.length > 1) {
            const chartRows = sortedResults.slice(0, Math.min(sortedResults.length, 10));
            sections.push({
              type: 'chart',
              title: isUkrainian ? `Рейтинг за ${valueLabel}` : `Ranking by ${valueLabel}`,
              chartType: 'horizontal-bar',
              data: {
                labels: chartRows.map((row) => String(row[labelColumn] || '').substring(0, 30)),
                datasets: [
                  {
                    label: valueLabel,
                    data: chartRows.map((row) => toNumber(row[valueColumn])),
                  },
                ],
              },
              height: 340,
            } as ChartResponse);
          }
        } else if (isSingleRow && labelColumn && valueColumn) {
          const row = results[0];
          const labelValue = String(row[labelColumn] ?? '');
          const valueLabel = humanizeColumn(valueColumn);
          const valueFormat = inferValueFormat(valueColumn);
          const entityLabel = humanizeColumn(labelColumn);
          const shortLabel = labelValue.length > 40 ? `${labelValue.slice(0, 40)}...` : labelValue;
          const entityIcon = inferEntityIcon(labelColumn);

          sections.push({
            type: 'statistics',
            title: isUkrainian ? 'Результат' : 'Result',
            layout: 'grid-2',
            cards: [
              {
                value: shortLabel || (isUkrainian ? 'Невідомо' : 'Unknown'),
                label: entityLabel || (isUkrainian ? 'Назва' : 'Name'),
                icon: entityIcon,
              },
              {
                value: toNumber(row[valueColumn]),
                label: valueLabel,
                icon: 'money',
                format: valueFormat,
                highlight: true,
              },
            ],
          } as StatisticsResponse);
        }

        sections.push({
          type: 'data_table',
          title: isUkrainian ? (isSingleRow ? 'Деталі' : 'Повний список') : (isSingleRow ? 'Details' : 'Results'),
          columns: tableColumns,
          rows: tableRows,
          totalRows: rowCount,
          enableSorting: true,
          enableExport: true,
        } as DataTableResponse);

        // Add map section for region queries with region data
        if (isRegionMapQuery(query) || hasRegionData(results)) {
          const regionColumn = findRegionColumn(columns);
          if (regionColumn) {
            const mapMarkers = buildRegionMarkers(results, columns, valueColumn);
            if (mapMarkers.length > 0) {
              // Insert map at the beginning (after summary text)
              sections.splice(1, 0, {
                type: 'map',
                title: isUkrainian ? 'Розподіл по регіонах' : 'Distribution by Region',
                markers: mapMarkers,
                mapType: 'markers',
                valueFormat: valueColumn ? inferValueFormat(valueColumn) : 'number',
                interactive: true,
                height: 400,
              } as MapResponse);
            }
          }
        }

        if (!rankingReady && showChart && results.length > 1 && results.length <= 20) {
          const numericCols = columns.filter((col) =>
            results.every((row) => row[col] === null || row[col] === undefined || isNumericValue(row[col]))
          );
          const labelCol = columns.find((col) => results.some((row) => typeof row[col] === 'string'));

          if (numericCols.length > 0 && labelCol) {
            sections.push({
              type: 'chart',
              title: 'Chart',
              chartType: results.length <= 6 ? 'pie' : 'bar',
              data: {
                labels: results.map((r) => String(r[labelCol] || '').substring(0, 20)),
                datasets: numericCols.slice(0, 3).map((col) => ({
                  label: col,
                  data: results.map((r) => r[col] as number),
                })),
              },
              height: 300,
            } as ChartResponse);
          }
        }

        return { structured: { sections } };
      }

      if (data.execution?.success && data.execution.row_count === 0) {
        return {
          structured: {
            sections: [
              {
                type: 'text',
                content: 'No results found for that query.',
                variant: 'warning',
              } as TextResponse,
            ],
          },
        };
      }

      const errorMsg = data.execution?.error || data.error || 'SQL execution failed.';
      return {
        structured: {
          sections: [
            {
              type: 'text',
              content: `Query failed: ${errorMsg}\n\nTry rephrasing the request.`,
              variant: 'error',
            } as TextResponse,
          ],
        },
      };
    } catch (error) {
      if (isAbortError(error)) {
        throw error;
      }
      console.error('Query failed:', error);
      return {
        structured: {
          sections: [
            {
              type: 'text',
              content: 'Query failed. Check the SQL API connection and try again.',
              variant: 'error',
            } as TextResponse,
          ],
        },
      };
    }
  }, []);

  return { processQuery };
}
