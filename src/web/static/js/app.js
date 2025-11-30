// --- Configuration ---
const FETCH_INTERVAL_MS = 30000; // 30 seconds (Updated from 15s)

// --- Element References ---
let statusIndicator, statusDot, statusText;
let totalValueEl, totalReturnEl, availableCashEl, tradeCountEl, winRateEl, riskLevelEl, sharpeRatioEl;
let positionsGridEl, historyTableBodyEl, aiHistoryContentEl, alertsContainerEl;
let botControlBtn, botControlText, botStatusText;
let tradeSearchInput; // NEW
let allTrades = []; // NEW

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    // Initialize element references
    statusIndicator = document.getElementById('statusIndicator');
    statusDot = statusIndicator.querySelector('.status-dot');
    statusText = statusIndicator.querySelector('span:last-child');

    totalValueEl = document.getElementById('totalValue');
    totalReturnEl = document.getElementById('totalReturn');
    availableCashEl = document.getElementById('availableCash');
    tradeCountEl = document.getElementById('tradeCount');
    winRateEl = document.getElementById('winRate');
    riskLevelEl = document.getElementById('riskLevel');
    sharpeRatioEl = document.getElementById('sharpeRatio');

    positionsGridEl = document.getElementById('positionsGrid');
    historyTableBodyEl = document.getElementById('historyTableBody');
    aiHistoryContentEl = document.getElementById('content-ai-history');
    alertsContainerEl = document.getElementById('alertsContainer');

    botControlBtn = document.getElementById('botControlBtn');
    botControlText = document.getElementById('botControlText');
    botStatusText = document.getElementById('botStatusText');
    tradeSearchInput = document.getElementById('tradeSearchInput'); // NEW

    if (tradeSearchInput) {
        tradeSearchInput.addEventListener('input', () => renderTradeHistory());
    }

    // Initial fetch
    initChart(); // Initialize chart
    fetchData();
    updateBotStatus();

    // Set interval for periodic updates
    setInterval(() => {
        fetchData();
        updateBotStatus();
    }, FETCH_INTERVAL_MS);
});

// --- Helper Functions ---
function apiUrl(path) {
    return path;
}

function formatCurrency(value, defaultVal = 'N/A') {
    if (value === null || value === undefined || isNaN(value)) return defaultVal;
    try {
        return Number(value).toLocaleString('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2 });
    } catch { return defaultVal; }
}

function formatPercent(value, defaultVal = 'N/A') {
    if (value === null || value === undefined || isNaN(value)) return defaultVal;
    try { return Number(value).toFixed(2) + '%'; } catch { return defaultVal; }
}

function formatNumber(value, decimals = 4, defaultVal = 'N/A') {
    if (value === null || value === undefined || isNaN(value)) return defaultVal;
    const num = Number(value);
    if (isNaN(num)) return defaultVal;
    try { return num.toFixed(decimals); } catch { return defaultVal; }
}

function formatDateTime(isoString, type = 'time', defaultVal = 'N/A') {
    if (!isoString) return defaultVal;
    try {
        const date = new Date(isoString);
        if (isNaN(date.getTime())) return defaultVal;
        if (type === 'date') {
            return date.toLocaleDateString([], { month: '2-digit', day: '2-digit', year: 'numeric' });
        } else {
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        }
    } catch (e) {
        console.error("Error formatting date:", isoString, e);
        return defaultVal;
    }
}

function escapeHtml(data) {
    if (typeof data === 'object') {
        try { return JSON.stringify(data, null, 2).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
        catch { return 'Invalid Data'; }
    }
    if (typeof data !== 'string') return String(data);
    return data.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

// --- Tab Switching Logic ---
const tabs = ['ai-history', 'trade-history', 'performance'];
function switchTab(activeTabId) {
    tabs.forEach(tabId => {
        const tabButton = document.getElementById(`tab-${tabId}`);
        const contentDiv = document.getElementById(`content-${tabId}`);
        if (!tabButton || !contentDiv) return;

        if (tabId === activeTabId) {
            tabButton.classList.remove('border-transparent', 'text-gray-400', 'hover:text-gray-200', 'hover:border-gray-500');
            tabButton.classList.add('border-indigo-500', 'text-indigo-400');
            tabButton.setAttribute('aria-current', 'page');
            contentDiv.classList.remove('hidden');
        } else {
            tabButton.classList.remove('border-indigo-500', 'text-indigo-400');
            tabButton.classList.add('border-transparent', 'text-gray-400', 'hover:text-gray-200', 'hover:border-gray-500');
            tabButton.removeAttribute('aria-current');
            contentDiv.classList.add('hidden');
        }
    });
}

// --- Data Fetching ---
async function fetchData() {
    try {
        const stateResponse = await fetch(apiUrl('/api/portfolio') + '?t=' + Date.now());
        if (!stateResponse.ok) throw new Error('portfolio_state.json fetch failed.');
        const stateData = await stateResponse.json();
        if (typeof stateData !== 'object' || stateData === null) throw new Error('Invalid state data.');
        updateDashboard(stateData);

        const tradeHistoryResponse = await fetch(apiUrl('/api/trades') + '?t=' + Date.now());
        let tradeHistoryData = [];
        if (tradeHistoryResponse.ok) {
            tradeHistoryData = await tradeHistoryResponse.json();
            if (!Array.isArray(tradeHistoryData)) throw new Error('Invalid trade history data.');
            updateTradeHistory(tradeHistoryData);
        } else {
            console.warn('Could not fetch trade_history.json');
            updateTradeHistory([]);
        }

        const cycleHistoryResponse = await fetch(apiUrl('/api/cycles') + '?t=' + Date.now());
        if (cycleHistoryResponse.ok) {
            const cycleHistoryData = await cycleHistoryResponse.json();
            if (!Array.isArray(cycleHistoryData)) throw new Error('Invalid cycle history data.');
            updateCycleHistory(cycleHistoryData);
        } else {
            console.warn('Could not fetch cycle_history.json');
            updateCycleHistory([]);
        }

        // Fetch alerts
        try {
            const alertsResponse = await fetch(apiUrl('/api/alerts') + '?t=' + Date.now());
            if (alertsResponse.ok) {
                const alertsData = await alertsResponse.json();
                if (Array.isArray(alertsData)) {
                    updateAlerts(alertsData);
                }
            }
        } catch (alertError) {
            console.warn('Could not fetch alerts.json:', alertError.message);
        }

        // Update performance metrics
        const metrics = calculatePerformanceMetrics(tradeHistoryData);
        if (winRateEl) winRateEl.textContent = formatPercent(metrics.winRate, '0%');
        if (riskLevelEl) {
            riskLevelEl.textContent = metrics.riskLevel;
            riskLevelEl.className = `text-xl sm:text-2xl font-bold mt-1 ${metrics.riskLevel === 'High' ? 'text-negative' :
                metrics.riskLevel === 'Medium' ? 'text-yellow-500' : 'text-positive'
                }`;
        }

        if (statusDot) {
            statusDot.classList.remove('status-down');
            statusDot.classList.add('status-live');
        }
        if (statusText) {
            const lastUpdatedTime = stateData.last_updated ? new Date(stateData.last_updated).toLocaleTimeString() : 'N/A';
            statusText.textContent = `LIVE (Updated: ${lastUpdatedTime})`;
            statusText.classList.remove('text-negative', 'text-secondary');
            statusText.classList.add('text-positive');
        }

    } catch (error) {
        console.error("Fetch data error:", error.message);
        if (statusDot) {
            statusDot.classList.remove('status-live');
            statusDot.classList.add('status-down');
        }
        if (statusText) {
            statusText.textContent = 'DISCONNECTED';
            statusText.classList.remove('text-positive');
            statusText.classList.add('text-negative', 'text-secondary');
        }
    }
}

function updateDashboard(data) {
    const defaultData = { total_value: null, current_balance: null, total_return: null, sharpe_ratio: null };
    const currentData = data || defaultData;
    if (totalValueEl) totalValueEl.textContent = formatCurrency(currentData.total_value, '$...');
    if (availableCashEl) availableCashEl.textContent = formatCurrency(currentData.current_balance, '$...');

    const totalReturn = currentData.total_return;
    if (totalReturnEl) {
        totalReturnEl.textContent = (totalReturn !== null && totalReturn > 0 ? '+' : '') + formatPercent(totalReturn, '...');
        totalReturnEl.classList.toggle('text-positive', totalReturn !== null && totalReturn >= 0);
        totalReturnEl.classList.toggle('text-negative', totalReturn !== null && totalReturn < 0);
    }

    // Update Sharpe ratio
    const sharpeRatio = currentData.sharpe_ratio;
    if (sharpeRatioEl) {
        sharpeRatioEl.textContent = sharpeRatio !== null && !isNaN(sharpeRatio) ? formatNumber(sharpeRatio, 3) : '...';
        sharpeRatioEl.className = `text-xl sm:text-2xl font-bold mt-1 ${sharpeRatio !== null && sharpeRatio > 1 ? 'text-positive' :
            sharpeRatio !== null && sharpeRatio > 0 ? 'text-yellow-500' : 'text-negative'
            }`;
    }

    updateActivePositions(data ? data.positions || {} : {});
}

function updateActivePositions(positions) {
    if (!positionsGridEl) return;
    positionsGridEl.innerHTML = '';
    const coinKeys = Object.keys(positions);

    if (coinKeys.length === 0) {
        positionsGridEl.innerHTML = '<p class="text-secondary text-center p-4">No active positions.</p>';
        return;
    }

    coinKeys.forEach(coin => {
        const pos = positions[coin];
        if (!pos || typeof pos !== 'object') return;

        const pnl = pos.unrealized_pnl;
        const pnlClass = (pnl !== null && !isNaN(pnl) && pnl >= 0) ? 'text-positive' : 'text-negative';
        const direction = (pos.direction || 'long').toUpperCase();
        const directionClass = direction === 'LONG' ? 'text-positive' : 'text-negative';
        const exitPlan = pos.exit_plan || {};
        const invalidationText = exitPlan.invalidation_condition || 'N/A';

        const card = `
            <div class="card p-4 border-l-4 ${direction === 'LONG' ? 'border-green-500' : 'border-red-500'}">
                <div class="flex justify-between items-center mb-2">
                    <span class="text-lg font-bold text-main">${pos.symbol || coin}</span>
                    <span class="font-semibold ${directionClass}">${direction} ${pos.leverage || 1}x</span>
                </div>
                <div class="flex justify-between items-center mb-3">
                    <span class="text-xl font-bold ${pnlClass}">${formatCurrency(pnl)}</span>
                    <span class="text-sm text-secondary">Notional: ${formatCurrency(pos.notional_usd)}</span>
                </div>
                <div class="text-xs text-secondary space-y-1 mb-4">
                    <div class="flex justify-between"><span class="font-medium">Qty:</span><span>${formatNumber(pos.quantity, 6)}</span></div>
                    <div class="flex justify-between"><span class="font-medium">Entry:</span><span>${formatCurrency(pos.entry_price)}</span></div>
                    <div class="flex justify-between"><span class="font-medium">Liq (Est):</span><span>${formatCurrency(pos.liquidation_price)}</span></div>
                    <div class="flex justify-between"><span class="font-medium">Confidence:</span><span>${formatPercent((pos.confidence || 0) * 100)}</span></div>
                    <div class="flex justify-between"><span class="font-medium">TP:</span><span>${exitPlan.profit_target || 'N/A'}</span></div>
                    <div class="flex justify-between"><span class="font-medium">SL:</span><span>${exitPlan.stop_loss || 'N/A'}</span></div>
                    <div class="flex justify-between items-center">
                        <span class="font-medium">Invalidation:</span>
                        <span class="truncate-text" title="${escapeHtml(invalidationText)}">${escapeHtml(invalidationText)}</span>
                    </div>
                </div>
                <div class="text-right">
                    <button class="btn-danger" onclick="forceClosePosition('${coin}')">
                        FORCE CLOSE
                    </button>
                </div>
            </div>
        `;
        positionsGridEl.innerHTML += card;
    });
}

function updateCycleHistory(history) {
    if (!aiHistoryContentEl) return;
    aiHistoryContentEl.innerHTML = '';

    if (!history || history.length === 0) {
        aiHistoryContentEl.innerHTML = '<p class="text-secondary text-center p-4">No AI cycle history yet.</p>';
        return;
    }

    const expandedAccordions = new Set(
        Array.from(document.querySelectorAll('.accordion-content.expanded'))
            .map(el => el.id.replace('content-', ''))
    );

    history.slice().reverse().forEach((cycle, index) => {
        if (!cycle || typeof cycle !== 'object') return;
        const accordionId = `cycle-${cycle.cycle || index}`;

        const cycleDiv = document.createElement('div');
        cycleDiv.className = 'card p-4 space-y-2';

        const headerDiv = document.createElement('div');
        headerDiv.className = 'flex justify-between items-center accordion-button';
        headerDiv.onclick = () => toggleAccordion(accordionId);
        headerDiv.innerHTML = `
            <h3 class="text-md font-semibold text-main">Cycle ${cycle.cycle || 'N/A'} - ${formatDateTime(cycle.timestamp, 'time')}</h3>
            <svg id="arrow-${accordionId}" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4 text-secondary transform transition-transform ${expandedAccordions.has(accordionId) ? 'rotate-180' : ''}">
                <path stroke-linecap="round" stroke-linejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
            </svg>
        `;
        cycleDiv.appendChild(headerDiv);

        const contentDiv = document.createElement('div');
        contentDiv.id = `content-${accordionId}`;
        const isExpanded = expandedAccordions.has(accordionId);
        contentDiv.className = `accordion-content space-y-3 pt-2 border-t border-divider mt-2 ${isExpanded ? 'expanded' : ''}`;


        contentDiv.innerHTML += `
            <div>
                <h4 class="text-xs font-medium text-secondary uppercase mb-1">Chain of Thoughts:</h4>
                <pre class="max-h-48 overflow-y-auto">${escapeHtml(cycle.chain_of_thoughts || 'N/A')}</pre>
            </div>
        `;
        contentDiv.innerHTML += `
            <div>
                <h4 class="text-xs font-medium text-secondary uppercase mb-1">Decisions:</h4>
                <pre class="max-h-48 overflow-y-auto">${escapeHtml(cycle.decisions || {})}</pre>
            </div>
        `;
        contentDiv.innerHTML += `
            <div>
                <h4 class="text-xs font-medium text-secondary uppercase mb-1">User Prompt (Summary):</h4>
                <pre class="max-h-24 overflow-y-auto">${escapeHtml(cycle.user_prompt_summary || 'N/A')}</pre>
            </div>
        `;

        cycleDiv.appendChild(contentDiv);
        aiHistoryContentEl.appendChild(cycleDiv);
    });
}

function updateTradeHistory(history) {
    allTrades = history || [];
    renderTradeHistory();
    updateChart(allTrades);
}

function renderTradeHistory() {
    if (!historyTableBodyEl) return;
    historyTableBodyEl.innerHTML = '';

    let filteredTrades = allTrades;
    if (tradeSearchInput && tradeSearchInput.value) {
        const term = tradeSearchInput.value.toLowerCase();
        filteredTrades = allTrades.filter(t =>
            (t.symbol && t.symbol.toLowerCase().includes(term)) ||
            (t.close_reason && t.close_reason.toLowerCase().includes(term))
        );
    }

    if (filteredTrades.length === 0) {
        historyTableBodyEl.innerHTML = '<tr><td colspan="8" class="px-4 py-4 text-center text-secondary">No trades found.</td></tr>';
        return;
    }

    // Sort by exit time (newest first)
    filteredTrades.sort((a, b) => {
        const timeA = new Date(a.exit_time || a.entry_time).getTime();
        const timeB = new Date(b.exit_time || b.entry_time).getTime();
        return timeB - timeA;
    });

    filteredTrades.forEach(trade => {
        const pnl = trade.pnl;
        const pnlClass = (pnl !== null && pnl >= 0) ? 'text-positive' : 'text-negative';
        const direction = (trade.direction || 'long').toUpperCase();
        const directionClass = direction === 'LONG' ? 'text-positive' : 'text-negative';

        const row = `
            <tr class="hover:bg-gray-700/50 transition-colors">
                <td class="px-4 py-3 whitespace-nowrap text-sm font-medium text-white">${trade.symbol || 'N/A'}</td>
                <td class="px-4 py-3 whitespace-nowrap text-sm font-bold ${directionClass}">${direction} ${trade.leverage || 1}x</td>
                <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-300">${formatCurrency(trade.notional_usd)}</td>
                <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-300">
                    <div>Entry: ${formatNumber(trade.entry_price)}</div>
                    <div>Exit: ${formatNumber(trade.exit_price)}</div>
                </td>
                <td class="px-4 py-3 whitespace-nowrap text-sm font-bold ${pnlClass}">${formatCurrency(pnl)}</td>
                <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-300">${formatDateTime(trade.exit_time || trade.entry_time, 'date')}</td>
                <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-300">${formatDateTime(trade.exit_time || trade.entry_time, 'time')}</td>
                <td class="px-4 py-3 text-sm text-gray-400 max-w-xs truncate" title="${escapeHtml(trade.close_reason)}">${escapeHtml(trade.close_reason || 'N/A')}</td>
            </tr>
        `;
        historyTableBodyEl.innerHTML += row;
    });
}

function toggleAccordion(accordionId) {
    const content = document.getElementById(`content-${accordionId}`);
    const arrow = document.getElementById(`arrow-${accordionId}`);
    if (content && arrow) {
        const isOpening = !content.classList.contains('expanded');

        if (isOpening) {
            document.querySelectorAll('.accordion-content.expanded').forEach(el => {
                if (el.id !== `content-${accordionId}`) {
                    el.classList.remove('expanded');
                    const otherArrowId = el.id.replace('content-', 'arrow-');
                    document.getElementById(otherArrowId)?.classList.remove('rotate-180');
                }
            });
        }

        content.classList.toggle('expanded');
        arrow.classList.toggle('rotate-180');
    }
}

async function forceClosePosition(coin) {
    if (!confirm(`Are you sure you want to manually force close the ${coin} position? This will override the AI's plan.`)) {
        return;
    }
    try {
        const response = await fetch(apiUrl('/api/force-close'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ coin: coin }),
        });
        const result = await response.json();
        if (response.ok && result.status === 'success') {
            alert(`Close command sent successfully for ${coin}. The bot will close the position in the next cycle (max 2 mins).`);
        } else {
            throw new Error(result.message || 'Unknown error.');
        }
    } catch (error) {
        alert(`Error: Could not send close command. Is the admin server running?\n\nDetails: ${error.message}`);
    }
}

function updateAlerts(alerts) {
    if (!alertsContainerEl) return;
    alertsContainerEl.innerHTML = '';

    if (!alerts || alerts.length === 0) {
        alertsContainerEl.innerHTML = '<p class="text-secondary text-center p-4">No recent alerts.</p>';
        return;
    }

    alerts.slice().reverse().forEach(alert => {
        if (!alert || typeof alert !== 'object') return;

        const levelColors = {
            'INFO': 'border-blue-500 bg-blue-500/10',
            'WARNING': 'border-yellow-500 bg-yellow-500/10',
            'CRITICAL': 'border-red-500 bg-red-500/10'
        };

        const levelTextColors = {
            'INFO': 'text-blue-400',
            'WARNING': 'text-yellow-400',
            'CRITICAL': 'text-red-400'
        };

        const level = alert.level || 'INFO';
        const borderColor = levelColors[level] || levelColors['INFO'];
        const textColor = levelTextColors[level] || levelTextColors['INFO'];

        const alertDiv = document.createElement('div');
        alertDiv.className = `card p-3 border-l-4 ${borderColor}`;

        alertDiv.innerHTML = `
            <div class="flex justify-between items-start mb-1">
                <span class="text-sm font-semibold ${textColor}">${alert.title || 'Alert'}</span>
                <span class="text-xs text-secondary">${formatDateTime(alert.timestamp, 'time')}</span>
            </div>
            <p class="text-sm text-main">${escapeHtml(alert.message || 'No message')}</p>
            ${alert.symbol ? `<span class="text-xs text-secondary mt-1">Symbol: ${alert.symbol}</span>` : ''}
        `;

        alertsContainerEl.appendChild(alertDiv);
    });
}

function calculatePerformanceMetrics(tradeHistory) {
    if (!tradeHistory || tradeHistory.length === 0) {
        return { winRate: 0, riskLevel: 'Low' };
    }

    const totalProfit = tradeHistory
        .filter(trade => trade.pnl > 0)
        .reduce((sum, trade) => sum + trade.pnl, 0);
    const totalLoss = Math.abs(tradeHistory
        .filter(trade => trade.pnl < 0)
        .reduce((sum, trade) => sum + trade.pnl, 0));

    let winRate = 0;
    if (totalProfit + totalLoss > 0) {
        winRate = (totalProfit / (totalProfit + totalLoss)) * 100;
    }

    const recentTrades = tradeHistory.slice(-10);
    const pnlVolatility = recentTrades.length > 1 ?
        Math.sqrt(recentTrades.reduce((sum, trade) => sum + Math.pow(trade.pnl, 2), 0) / recentTrades.length) : 0;

    let riskLevel = 'Low';
    if (pnlVolatility > 50) riskLevel = 'High';
    else if (pnlVolatility > 20) riskLevel = 'Medium';

    return { winRate, riskLevel };
}

// --- Bot Control Functions ---
async function updateBotStatus() {
    try {
        const response = await fetch(apiUrl('/api/bot-control'));
        if (response.ok) {
            const data = await response.json();
            const status = data.status || 'unknown';

            if (botStatusText) botStatusText.textContent = status.toUpperCase();

            if (status === 'paused') {
                if (botControlText) botControlText.textContent = '▶️ RESUME';
                if (botControlBtn) {
                    botControlBtn.classList.remove('bg-indigo-600', 'hover:bg-indigo-700');
                    botControlBtn.classList.add('bg-green-600', 'hover:bg-green-700');
                }
            } else {
                if (botControlText) botControlText.textContent = '⏸️ PAUSE';
                if (botControlBtn) {
                    botControlBtn.classList.remove('bg-green-600', 'hover:bg-green-700');
                    botControlBtn.classList.add('bg-indigo-600', 'hover:bg-indigo-700');
                }
            }
        }
    } catch (error) {
        console.error('Error fetching bot status:', error);
    }
}

async function toggleBotControl() {
    if (!botControlText) return;
    const currentAction = botControlText.textContent.includes('PAUSE') ? 'pause' : 'resume';

    try {
        const response = await fetch(apiUrl('/api/bot-control'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: currentAction })
        });

        if (response.ok) {
            const result = await response.json();
            updateBotStatus();
            alert(result.message);
        } else {
            throw new Error('Failed to update bot status');
        }
    } catch (error) {
        alert('Error updating bot status: ' + error.message);
    }
}

// --- Performance Dashboard Functions ---
async function refreshPerformance() {
    const btn = document.getElementById('refreshPerformanceBtn');
    const originalText = btn.textContent;

    try {
        btn.textContent = '🔄 Analyzing...';
        btn.disabled = true;

        const response = await fetch(apiUrl('/api/performance/refresh'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (response.ok) {
            const result = await response.json();
            if (result.status === 'success') {
                await loadPerformanceData();
                alert('✅ Performance analysis completed successfully!');
            } else {
                throw new Error(result.message || 'Unknown error');
            }
        } else {
            throw new Error('Failed to refresh performance data');
        }
    } catch (error) {
        console.error('Performance refresh error:', error);
        alert(`❌ Failed to refresh performance: ${error.message}`);
    } finally {
        btn.textContent = originalText;
        btn.disabled = false;
    }
}

async function loadPerformanceData() {
    try {
        const response = await fetch(apiUrl('/api/performance') + '?t=' + Date.now());
        if (response.ok) {
            const performanceData = await response.json();
            updatePerformanceDashboard(performanceData);
        } else {
            console.warn('Could not fetch performance data');
            updatePerformanceDashboard({});
        }
    } catch (error) {
        console.error('Error loading performance data:', error);
        updatePerformanceDashboard({});
    }
}

function updatePerformanceDashboard(data) {
    const tradePerf = data.trade_performance || {};
    const portfolioPerf = data.portfolio_performance || {};
    const tradingActivity = data.trading_activity || {};
    const coinPerf = data.coin_performance || {};
    const recommendations = data.recommendations || [];

    const winRateEl = document.getElementById('performanceWinRate');
    const profitFactorEl = document.getElementById('performanceProfitFactor');
    const totalPnlEl = document.getElementById('performanceTotalPnl');
    const avgPnlEl = document.getElementById('performanceAvgPnl');

    if (winRateEl) {
        winRateEl.textContent = formatPercent(tradePerf.win_rate, '0%');
        winRateEl.className = `text-xl font-bold mt-1 ${tradePerf.win_rate > 50 ? 'text-positive' : 'text-negative'}`;
    }
    if (profitFactorEl) {
        profitFactorEl.textContent = formatNumber(tradePerf.profit_factor, 2);
        profitFactorEl.className = `text-xl font-bold mt-1 ${tradePerf.profit_factor > 1 ? 'text-positive' : 'text-negative'}`;
    }
    if (totalPnlEl) {
        totalPnlEl.textContent = formatCurrency(tradePerf.total_pnl);
        totalPnlEl.className = `text-xl font-bold mt-1 ${(tradePerf.total_pnl || 0) >= 0 ? 'text-positive' : 'text-negative'}`;
    }
    if (avgPnlEl) {
        avgPnlEl.textContent = formatCurrency(tradePerf.average_pnl);
        avgPnlEl.className = `text-xl font-bold mt-1 ${(tradePerf.average_pnl || 0) >= 0 ? 'text-positive' : 'text-negative'}`;
    }

    // Placeholder functions for now, as they were not fully implemented in the original HTML either
    // updateTradingActivity(tradingActivity);
    // updateCoinPerformance(coinPerf);
}

// --- Chart Logic ---
let chart;
let areaSeries;

function initChart() {
    const chartContainer = document.getElementById('pnlChartContainer');
    if (!chartContainer) return;

    chart = LightweightCharts.createChart(chartContainer, {
        width: chartContainer.clientWidth,
        height: chartContainer.clientHeight,
        layout: {
            background: { type: 'solid', color: '#1f2937' },
            textColor: '#9ca3af',
        },
        grid: {
            vertLines: { color: '#374151' },
            horzLines: { color: '#374151' },
        },
        rightPriceScale: {
            borderColor: '#374151',
        },
        timeScale: {
            borderColor: '#374151',
            timeVisible: true,
        },
    });

    areaSeries = chart.addAreaSeries({
        topColor: 'rgba(34, 197, 94, 0.56)',
        bottomColor: 'rgba(34, 197, 94, 0.04)',
        lineColor: 'rgba(34, 197, 94, 1)',
        lineWidth: 2,
    });

    // Resize chart on window resize
    window.addEventListener('resize', () => {
        chart.resize(chartContainer.clientWidth, chartContainer.clientHeight);
    });
}

function updateChart(tradeHistory) {
    if (!areaSeries || !tradeHistory || tradeHistory.length === 0) return;

    // Sort trades by exit time (oldest to newest)
    const sortedTrades = [...tradeHistory].sort((a, b) => {
        const timeA = new Date(a.exit_time || a.entry_time).getTime();
        const timeB = new Date(b.exit_time || b.entry_time).getTime();
        return timeA - timeB;
    });

    let cumulativePnL = 0;
    const data = [];

    // Add initial point (optional, starts from 0)
    if (sortedTrades.length > 0) {
        const firstTime = new Date(sortedTrades[0].exit_time || sortedTrades[0].entry_time).getTime() / 1000;
        data.push({ time: firstTime - 3600, value: 0 });
    }

    sortedTrades.forEach(trade => {
        cumulativePnL += trade.pnl;
        const time = new Date(trade.exit_time || trade.entry_time).getTime() / 1000;

        // Ensure unique timestamps for Lightweight Charts
        if (data.length > 0 && data[data.length - 1].time >= time) {
            let newTime = time;
            while (newTime <= data[data.length - 1].time) {
                newTime += 1;
            }
            data.push({ time: newTime, value: cumulativePnL });
        } else {
            data.push({ time: time, value: cumulativePnL });
        }
    });

    // Update color based on current PnL
    if (cumulativePnL >= 0) {
        areaSeries.applyOptions({
            topColor: 'rgba(34, 197, 94, 0.56)',
            bottomColor: 'rgba(34, 197, 94, 0.04)',
            lineColor: 'rgba(34, 197, 94, 1)',
        });
    } else {
        areaSeries.applyOptions({
            topColor: 'rgba(239, 68, 68, 0.56)',
            bottomColor: 'rgba(239, 68, 68, 0.04)',
            lineColor: 'rgba(239, 68, 68, 1)',
        });
    }

    areaSeries.setData(data);
    chart.timeScale().fitContent();
}
