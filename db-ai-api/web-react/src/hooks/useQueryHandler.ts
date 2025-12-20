import { useCallback, useRef } from 'react';
import {
  fetchYearlySales,
  fetchYearlyItems,
  fetchTopProducts,
  fetchTopClients,
  fetchDebtSummary,
  searchProducts,
  smartSearch,
  ollamaQuery,
} from '../services/api';
import {
  detectQueryType,
  extractNumber,
  extractProductKeyword,
} from '../services/queryDetector';

// Escape HTML utility
function escapeHtml(text: string): string {
  if (!text) return '';
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Format cell value for tables
function formatCellValue(value: unknown): string {
  if (value === null || value === undefined) return '-';
  if (typeof value === 'number') return value.toLocaleString();
  if (typeof value === 'boolean') return value ? 'Так' : 'Ні';
  const str = String(value);
  return escapeHtml(str.length > 100 ? str.substring(0, 100) + '...' : str);
}

export function useQueryHandler() {
  const chartCounterRef = useRef(0);
  const chartDataRef = useRef<Record<string, unknown>>({});

  const handleSalesQuery = useCallback(async (): Promise<string> => {
    const [yearlyData, itemsData] = await Promise.all([
      fetchYearlySales(),
      fetchYearlyItems(),
    ]);

    const chartId = `chart-${++chartCounterRef.current}`;

    const html = `
      <div class="analytics-response">
        <h3>📊 Аналітика продажів</h3>

        <div class="chart-inline">
          <canvas id="${chartId}" height="300"></canvas>
        </div>

        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-value">${yearlyData[0]?.total_sales?.toLocaleString() || 0}</div>
            <div class="stat-label">Продажів у ${yearlyData[0]?.year || 2025}</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">${itemsData[0]?.total_quantity?.toLocaleString() || 0}</div>
            <div class="stat-label">Товарів продано</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">${itemsData[0]?.unique_products?.toLocaleString() || 0}</div>
            <div class="stat-label">Унікальних товарів</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">${yearlyData[0]?.total_orders?.toLocaleString() || 0}</div>
            <div class="stat-label">Замовлень</div>
          </div>
        </div>

        <table class="data-table">
          <tr><th>Рік</th><th>Продажів</th><th>Замовлень</th><th>Позицій</th></tr>
          ${yearlyData
            .map(
              (row) => `
            <tr>
              <td><strong>${row.year}</strong></td>
              <td>${row.total_sales?.toLocaleString()}</td>
              <td>${row.total_orders?.toLocaleString()}</td>
              <td>${row.total_items?.toLocaleString()}</td>
            </tr>
          `
            )
            .join('')}
        </table>
      </div>
    `;

    chartDataRef.current[chartId] = {
      type: 'bar',
      data: {
        labels: yearlyData.map((d) => d.year).reverse(),
        datasets: [
          {
            label: 'Продажів',
            data: yearlyData.map((d) => d.total_sales).reverse(),
            backgroundColor: 'rgba(168, 85, 247, 0.8)',
            borderRadius: 6,
          },
        ],
      },
    };

    return html;
  }, []);

  const handleTopProductsQuery = useCallback(async (query: string): Promise<string> => {
    const limit = extractNumber(query) || 10;
    const data = await fetchTopProducts(limit);

    const chartId = `chart-${++chartCounterRef.current}`;

    const html = `
      <div class="analytics-response">
        <h3>🏆 Топ-${limit} товарів за продажами</h3>

        <div class="chart-inline">
          <canvas id="${chartId}" height="400"></canvas>
        </div>

        <table class="data-table">
          <tr><th>#</th><th>Товар</th><th>Продано (шт)</th><th>Замовлень</th></tr>
          ${data
            .map(
              (row, i) => `
            <tr>
              <td>${i + 1}</td>
              <td>${escapeHtml(row.product_name?.substring(0, 40) || '-')}</td>
              <td><strong>${row.total_qty?.toLocaleString()}</strong></td>
              <td>${row.order_count?.toLocaleString()}</td>
            </tr>
          `
            )
            .join('')}
        </table>
      </div>
    `;

    chartDataRef.current[chartId] = {
      type: 'bar',
      indexAxis: 'y',
      data: {
        labels: data.map((d) => d.product_name?.substring(0, 20) || ''),
        datasets: [
          {
            label: 'Продано (шт)',
            data: data.map((d) => d.total_qty),
            backgroundColor: 'rgba(124, 58, 237, 0.8)',
            borderRadius: 4,
          },
        ],
      },
    };

    return html;
  }, []);

  const handleTopClientsQuery = useCallback(async (query: string): Promise<string> => {
    const limit = extractNumber(query) || 10;
    const data = await fetchTopClients(limit);

    const chartId = `chart-${++chartCounterRef.current}`;

    const html = `
      <div class="analytics-response">
        <h3>👥 Топ-${limit} клієнтів</h3>

        <div class="chart-inline">
          <canvas id="${chartId}" height="400"></canvas>
        </div>

        <table class="data-table">
          <tr><th>#</th><th>Клієнт</th><th>Продажів</th><th>Замовлень</th></tr>
          ${data
            .map(
              (row, i) => `
            <tr>
              <td>${i + 1}</td>
              <td>${escapeHtml(row.client_name?.substring(0, 35) || '-')}</td>
              <td><strong>${row.total_sales?.toLocaleString()}</strong></td>
              <td>${row.total_orders?.toLocaleString()}</td>
            </tr>
          `
            )
            .join('')}
        </table>
      </div>
    `;

    chartDataRef.current[chartId] = {
      type: 'bar',
      indexAxis: 'y',
      data: {
        labels: data.map((d) => d.client_name?.substring(0, 15) || ''),
        datasets: [
          {
            label: 'Продажів',
            data: data.map((d) => d.total_sales),
            backgroundColor: 'rgba(168, 85, 247, 0.8)',
            borderRadius: 4,
          },
        ],
      },
    };

    return html;
  }, []);

  const handleDebtsQuery = useCallback(async (): Promise<string> => {
    const data = await fetchDebtSummary();
    const chartId = `chart-${++chartCounterRef.current}`;
    const s = data.summary;

    const html = `
      <div class="analytics-response">
        <h3>💰 Статистика заборгованості</h3>

        <div class="stats-grid big">
          <div class="stat-card highlight">
            <div class="stat-value">${(s.total_amount / 1000000).toFixed(1)} млн</div>
            <div class="stat-label">Загальна сума боргів (грн)</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">${s.total_debts?.toLocaleString()}</div>
            <div class="stat-label">Кількість боргів</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">${s.avg_amount?.toLocaleString('uk-UA', { maximumFractionDigits: 0 })} грн</div>
            <div class="stat-label">Середній борг</div>
          </div>
        </div>

        <div class="chart-inline" style="max-width: 400px; margin: 20px auto;">
          <canvas id="${chartId}" height="300"></canvas>
        </div>

        <table class="data-table">
          <tr><th>Рік</th><th>Кількість</th><th>Сума (грн)</th></tr>
          ${
            data.by_year
              ?.map(
                (row) => `
            <tr>
              <td><strong>${row.year}</strong></td>
              <td>${row.debt_count?.toLocaleString()}</td>
              <td>${row.total_amount?.toLocaleString('uk-UA', { maximumFractionDigits: 0 })}</td>
            </tr>
          `
              )
              .join('') || ''
          }
        </table>
      </div>
    `;

    chartDataRef.current[chartId] = {
      type: 'doughnut',
      data: {
        labels: data.by_year?.map((d) => d.year) || [],
        datasets: [
          {
            data: data.by_year?.map((d) => d.total_amount) || [],
            backgroundColor: [
              'rgba(168, 85, 247, 0.8)',
              'rgba(139, 92, 246, 0.8)',
              'rgba(124, 58, 237, 0.8)',
              'rgba(192, 132, 252, 0.8)',
              'rgba(216, 180, 254, 0.8)',
            ],
          },
        ],
      },
    };

    return html;
  }, []);

  const handleProductKeywordSearch = useCallback(async (query: string): Promise<string> => {
    const keyword = extractProductKeyword(query);
    if (!keyword) {
      return handleSmartSearch(query);
    }

    const lowerQuery = query.toLowerCase();
    const sortBySales =
      lowerQuery.includes('топ') ||
      lowerQuery.includes('продаж') ||
      lowerQuery.includes('кількіст') ||
      lowerQuery.includes('рейтинг');

    const limit = extractNumber(query) || 30;
    const data = await searchProducts(keyword, limit, sortBySales);

    if (data.count === 0) {
      return `<div class="search-response"><p>Товарів з "${keyword}" не знайдено.</p></div>`;
    }

    const chartId = `chart-${++chartCounterRef.current}`;

    let html = `
      <div class="analytics-response">
        <h3>📦 Товари з "${keyword}" ${sortBySales ? '(по продажах)' : ''}</h3>
        <p>Знайдено: <strong>${data.count}</strong> товарів</p>
    `;

    if (sortBySales && data.products.length > 0) {
      html += `
        <div class="chart-inline">
          <canvas id="${chartId}" height="400"></canvas>
        </div>
      `;

      const top10 = data.products.slice(0, 10);
      chartDataRef.current[chartId] = {
        type: 'bar',
        indexAxis: 'y',
        data: {
          labels: top10.map((d) => (d.product_name || d.vendor_code || '').substring(0, 25)),
          datasets: [
            {
              label: 'Продано (шт)',
              data: top10.map((d) => d.total_sold || 0),
              backgroundColor: 'rgba(139, 92, 246, 0.8)',
              borderRadius: 4,
            },
          ],
        },
      };
    }

    html += `
      <table class="data-table">
        <tr>
          <th>#</th>
          <th>Назва товару</th>
          <th>Артикул</th>
          ${sortBySales ? '<th>Продано</th><th>Замовлень</th>' : ''}
        </tr>
        ${data.products
          .map(
            (p, i) => `
          <tr>
            <td>${i + 1}</td>
            <td>${escapeHtml(p.product_name?.substring(0, 50) || '-')}</td>
            <td><code>${escapeHtml(p.vendor_code || '-')}</code></td>
            ${
              sortBySales
                ? `
              <td><strong>${(p.total_sold || 0).toLocaleString()}</strong></td>
              <td>${(p.order_count || 0).toLocaleString()}</td>
            `
                : ''
            }
          </tr>
        `
          )
          .join('')}
      </table>
    </div>
    `;

    return html;
  }, []);

  const handleSmartSearch = useCallback(async (query: string): Promise<string> => {
    const data = await smartSearch(query, 20);

    if (data.results.length === 0) {
      return `
        <div class="no-results">
          <p>🔍 Нічого не знайдено за запитом "<em>${escapeHtml(query)}</em>"</p>
          <p>Спробуйте інші ключові слова або перегляньте швидкі запити зліва.</p>
        </div>
      `;
    }

    // Group results by table
    const grouped: Record<string, typeof data.results> = {};
    data.results.forEach((r) => {
      if (!grouped[r.table]) {
        grouped[r.table] = [];
      }
      grouped[r.table].push(r);
    });

    let html = `
      <div class="search-response">
        <p>🔍 Знайдено <strong>${data.n_results}</strong> результатів за "<em>${escapeHtml(query)}</em>"</p>
    `;

    if (data.detected_regions && data.detected_regions.length > 0) {
      html += `<p class="detected-regions">📍 Регіони: ${data.detected_regions.join(', ')}</p>`;
    }

    html += `<div class="results-summary">`;

    for (const [table, results] of Object.entries(grouped)) {
      const tableName = table.replace('dbo.', '');
      const avgSimilarity = (
        (results.reduce((sum, r) => sum + r.similarity, 0) / results.length) *
        100
      ).toFixed(0);

      html += `
        <div class="result-group">
          <div class="result-group-header">
            <span class="table-badge">${tableName}</span>
            <span class="result-count">${results.length} записів (${avgSimilarity}% схожість)</span>
          </div>
          <div class="result-items">
      `;

      results.slice(0, 3).forEach((r) => {
        if (r.name) {
          html += `<div class="result-item">• ${escapeHtml(r.name.substring(0, 60))}</div>`;
        }
      });

      if (results.length > 3) {
        html += `<div class="result-more">... та ще ${results.length - 3} записів</div>`;
      }

      html += `</div></div>`;
    }

    html += `</div></div>`;
    return html;
  }, []);

  const handleOllamaQuery = useCallback(async (query: string): Promise<string> => {
    try {
      const data = await ollamaQuery(query);

      let html = `
        <div class="analytics-response">
          <h3>🤖 AI Query Result</h3>
      `;

      if (data.explanation) {
        html += `<p><em>${escapeHtml(data.explanation)}</em></p>`;
      }

      html += `
        <details style="margin: 10px 0;">
          <summary style="cursor: pointer; color: var(--text-muted);">📝 SQL Query</summary>
          <pre style="background: var(--bg-tertiary); padding: 10px; border-radius: 8px; overflow-x: auto; font-size: 0.85em;">${escapeHtml(data.sql)}</pre>
        </details>
      `;

      if (data.execution?.success && data.execution.results) {
        const results = data.execution.results;
        const columns = data.execution.columns || [];

        if (results.length === 0) {
          html += `<p>Результатів не знайдено.</p>`;
        } else {
          html += `
            <p>Знайдено: <strong>${results.length}</strong> записів</p>
            <table class="data-table">
              <tr>${columns.map((c) => `<th>${escapeHtml(c)}</th>`).join('')}</tr>
              ${results
                .slice(0, 50)
                .map(
                  (row) => `
                <tr>${columns.map((c) => `<td>${formatCellValue(row[c])}</td>`).join('')}</tr>
              `
                )
                .join('')}
            </table>
          `;

          if (results.length > 50) {
            html += `<p class="result-more">... та ще ${results.length - 50} записів</p>`;
          }
        }
      } else if (data.execution && !data.execution.success) {
        html += `<p style="color: var(--error);">Помилка виконання: ${escapeHtml(data.execution.error || 'Unknown error')}</p>`;
      }

      html += `</div>`;
      return html;
    } catch {
      // Fallback to semantic search if Ollama fails
      return handleSmartSearch(query);
    }
  }, [handleSmartSearch]);

  const processQuery = useCallback(
    async (query: string): Promise<string> => {
      const queryType = detectQueryType(query);

      switch (queryType) {
        case 'sales':
          return handleSalesQuery();
        case 'top_products':
          return handleTopProductsQuery(query);
        case 'top_clients':
          return handleTopClientsQuery(query);
        case 'debts':
          return handleDebtsQuery();
        case 'product_keyword_search':
          return handleProductKeywordSearch(query);
        case 'region':
        case 'client_search':
          return handleSmartSearch(query);
        default:
          return handleOllamaQuery(query);
      }
    },
    [
      handleSalesQuery,
      handleTopProductsQuery,
      handleTopClientsQuery,
      handleDebtsQuery,
      handleProductKeywordSearch,
      handleSmartSearch,
      handleOllamaQuery,
    ]
  );

  const getChartData = useCallback(() => chartDataRef.current, []);

  return { processQuery, getChartData };
}
