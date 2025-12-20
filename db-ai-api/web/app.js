// API Configuration - Pure Ollama + Embeddings
const RAG_API = 'http://localhost:8000';   // Semantic search fallback
const SQL_API = 'http://localhost:8002';   // Ollama Text-to-SQL (main)

// State
let currentChart = null;
let inlineChartCounter = 0;
let isLoading = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkApiStatus();
    setInterval(checkApiStatus, 30000);
});

// Check API Status
async function checkApiStatus() {
    const statusDot = document.getElementById('apiStatus');
    const statusText = document.getElementById('apiStatusText');
    const docCount = document.getElementById('docCount');

    try {
        const response = await fetch(`${RAG_API}/`);
        const data = await response.json();

        statusDot.className = 'status-dot online';
        statusText.textContent = 'API Online';
        docCount.textContent = `Документів: ${data.documents?.toLocaleString() || '--'}`;
    } catch (error) {
        statusDot.className = 'status-dot offline';
        statusText.textContent = 'API Offline';
        docCount.textContent = 'Документів: --';
    }
}

// Handle keyboard events
function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Auto-resize textarea
function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
}

// New chat
function newChat() {
    document.getElementById('messages').innerHTML = '';
    document.getElementById('welcomeMessage').style.display = 'block';
    document.getElementById('userInput').value = '';
    document.getElementById('userInput').focus();
}

// Send quick query
function sendQuickQuery(query) {
    document.getElementById('userInput').value = query;
    sendMessage();
}

// NOTE: All query routing removed - everything goes through Ollama AI

// Send message - ALL queries go to Ollama AI
async function sendMessage() {
    const input = document.getElementById('userInput');
    const query = input.value.trim();

    if (!query || isLoading) return;

    document.getElementById('welcomeMessage').style.display = 'none';
    addMessage(query, 'user');
    input.value = '';
    input.style.height = 'auto';

    isLoading = true;
    const loadingId = showLoading();

    try {
        // ALL queries go through Ollama Text-to-SQL
        const response = await handleOllamaQuery(query);

        hideLoading(loadingId);
        addMessage(response, 'assistant');

        // Initialize any inline charts
        initializeInlineCharts();

    } catch (error) {
        hideLoading(loadingId);
        addMessage(`Помилка: ${error.message}`, 'assistant');
    }

    isLoading = false;
}

// NOTE: Removed hardcoded handlers - handleSalesQuery, handleTopProductsQuery,
// handleTopClientsQuery, handleProductKeywordSearch, handleVendorCodesQuery, handleDebtsQuery
// All queries now go through Ollama AI

// Smart search - aggregate similar results
async function handleSmartSearch(query) {
    try {
        const response = await fetch(`${RAG_API}/search?q=${encodeURIComponent(query)}&n=20`);
        const data = await response.json();

        if (data.results.length === 0) {
            return `
                <div class="no-results">
                    <p>🔍 Нічого не знайдено за запитом "<em>${escapeHtml(query)}</em>"</p>
                    <p>Спробуйте інші ключові слова або перегляньте швидкі запити зліва.</p>
                </div>
            `;
        }

        // Group results by table
        const grouped = {};
        data.results.forEach(r => {
            if (!grouped[r.table]) {
                grouped[r.table] = [];
            }
            grouped[r.table].push(r);
        });

        // Build smart response
        let html = `
            <div class="search-response">
                <p>🔍 Знайдено <strong>${data.n_results}</strong> результатів за "<em>${escapeHtml(query)}</em>"</p>
        `;

        if (data.detected_regions?.length > 0) {
            html += `<p class="detected-regions">📍 Регіони: ${data.detected_regions.join(', ')}</p>`;
        }

        // Show summary by table
        html += `<div class="results-summary">`;

        for (const [table, results] of Object.entries(grouped)) {
            const tableName = table.replace('dbo.', '');
            const avgSimilarity = (results.reduce((sum, r) => sum + r.similarity, 0) / results.length * 100).toFixed(0);

            html += `
                <div class="result-group">
                    <div class="result-group-header">
                        <span class="table-badge">${tableName}</span>
                        <span class="result-count">${results.length} записів (${avgSimilarity}% схожість)</span>
                    </div>
                    <div class="result-items">
            `;

            // Show top 3 from each table with names
            results.slice(0, 3).forEach(r => {
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

    } catch (error) {
        return `Помилка пошуку: ${error.message}`;
    }
}

// Handle natural language query with Ollama Text-to-SQL
async function handleOllamaQuery(query) {
    try {
        const response = await fetch(`${SQL_API}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: query,
                execute: true,
                max_rows: 100,
                include_explanation: true
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'SQL generation failed');
        }

        const data = await response.json();

        let html = `
            <div class="analytics-response">
                <h3>🤖 AI Query Result</h3>
        `;

        // Show explanation if available
        if (data.explanation) {
            html += `<p><em>${escapeHtml(data.explanation)}</em></p>`;
        }

        // Show generated SQL
        html += `
            <details style="margin: 10px 0;">
                <summary style="cursor: pointer; color: var(--text-muted);">📝 SQL Query</summary>
                <pre style="background: var(--bg-tertiary); padding: 10px; border-radius: 8px; overflow-x: auto; font-size: 0.85em;">${escapeHtml(data.sql)}</pre>
            </details>
        `;

        // Show results if execution was successful
        // API returns .rows not .results
        if (data.execution && data.execution.success && data.execution.rows) {
            const results = data.execution.rows;
            const columns = data.execution.columns || [];

            if (results.length === 0) {
                html += `<p>Результатів не знайдено.</p>`;
            } else {
                // Build table
                html += `
                    <p>Знайдено: <strong>${results.length}</strong> записів</p>
                    <table class="data-table">
                        <tr>${columns.map(c => `<th>${escapeHtml(c)}</th>`).join('')}</tr>
                        ${results.slice(0, 50).map(row => `
                            <tr>${columns.map(c => `<td>${formatCellValue(row[c])}</td>`).join('')}</tr>
                        `).join('')}
                    </table>
                `;

                if (results.length > 50) {
                    html += `<p class="result-more">... та ще ${results.length - 50} записів</p>`;
                }
            }
        } else if (data.execution && !data.execution.success) {
            // Execution failed - show error and fallback to semantic search
            html += `<p style="color: var(--text-muted);">SQL запит не вдалося виконати. Пошук у базі знань...</p>`;
            html += `</div>`;
            // Add semantic search results as fallback
            const searchResults = await handleSmartSearch(query);
            return html + searchResults;
        }

        html += `</div>`;
        return html;

    } catch (error) {
        // Fallback to semantic search if Ollama fails
        console.error('Ollama query failed:', error);
        return await handleSmartSearch(query);
    }
}

// Format cell value for display
function formatCellValue(value) {
    if (value === null || value === undefined) return '-';
    if (typeof value === 'number') return value.toLocaleString();
    if (typeof value === 'boolean') return value ? 'Так' : 'Ні';
    const str = String(value);
    return escapeHtml(str.length > 100 ? str.substring(0, 100) + '...' : str);
}

// Initialize inline charts after message is added
function initializeInlineCharts() {
    setTimeout(() => {
        for (let i = 1; i <= inlineChartCounter; i++) {
            const chartId = `chart-${i}`;
            const canvas = document.getElementById(chartId);
            const chartData = window[`chartData_${chartId}`];

            if (canvas && chartData && !canvas.chart) {
                const ctx = canvas.getContext('2d');
                canvas.chart = new Chart(ctx, {
                    type: chartData.type,
                    data: chartData.data,
                    options: getChartOptions(chartData.type, chartData.indexAxis)
                });
            }
        }
    }, 100);
}

function getChartOptions(type, indexAxis) {
    const baseOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: type === 'doughnut',
                position: 'bottom',
                labels: { color: '#4a4a4a', padding: 20 }
            }
        }
    };

    if (type === 'doughnut') {
        return {
            ...baseOptions,
            cutout: '60%'
        };
    }

    return {
        ...baseOptions,
        indexAxis: indexAxis || 'x',
        scales: {
            y: {
                beginAtZero: true,
                grid: { color: 'rgba(0, 0, 0, 0.08)' },
                ticks: { color: '#4a4a4a' }
            },
            x: {
                grid: { color: 'rgba(0, 0, 0, 0.08)' },
                ticks: { color: '#4a4a4a' }
            }
        }
    };
}

// NOTE: Removed showChart function - all visualization through Ollama AI results

function closeChart() {
    document.getElementById('chartModal').style.display = 'none';
    if (currentChart) {
        currentChart.destroy();
        currentChart = null;
    }
}

// Add message to chat
function addMessage(content, role) {
    const messages = document.getElementById('messages');
    const avatar = role === 'user' ? 'Ви' : 'AI';

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="message-avatar">${avatar}</div>
            <div class="message-text">${role === 'user' ? escapeHtml(content) : content}</div>
        </div>
    `;

    messages.appendChild(messageDiv);
    scrollToBottom();
}

// Show loading indicator
function showLoading() {
    const messages = document.getElementById('messages');
    const id = 'loading-' + Date.now();

    const loadingDiv = document.createElement('div');
    loadingDiv.id = id;
    loadingDiv.className = 'message assistant';
    loadingDiv.innerHTML = `
        <div class="message-content">
            <div class="message-avatar">AI</div>
            <div class="message-text">
                <div class="loading">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        </div>
    `;

    messages.appendChild(loadingDiv);
    scrollToBottom();

    return id;
}

function hideLoading(id) {
    const loading = document.getElementById(id);
    if (loading) loading.remove();
}

// Utilities
function scrollToBottom() {
    const container = document.getElementById('chatContainer');
    container.scrollTop = container.scrollHeight;
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// NOTE: Removed extractNumber - not needed for AI queries

// Event listeners
document.getElementById('chartModal')?.addEventListener('click', (e) => {
    if (e.target.id === 'chartModal') closeChart();
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeChart();
});
