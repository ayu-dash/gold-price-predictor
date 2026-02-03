/**
 * Dashboard Logic for Gold Price Predictor
 * Handles Realtime Updates, Chart Rendering, and Polling.
 */

document.addEventListener('DOMContentLoaded', function () {
    // 1. Initial Data Fetch
    fetchAnalysis();
    renderChart();

    // 2. Setup Event Listeners
    setupEventListeners();

    // 3. Start Realtime Polling (Every 30 seconds)
    startRealtimeUpdates();

    // 4. Set Date
    const now = new Date();
    document.getElementById('current_date').textContent = now.toDateString();

    // 5. Start Dashboard Refresh Loop (Every 60s)
    setInterval(updateDashboardData, 60000);
});

// ----------------------------------------
// Realtime Price Updates (High Frequency)
// ----------------------------------------
function startRealtimeUpdates() {
    // Initial fetch
    fetchRealtimePrice();
    // Poll every 30 seconds
    setInterval(fetchRealtimePrice, 30000);
}

function fetchRealtimePrice() {
    fetch('/api/live_price')
        .then(response => {
            if (!response.ok) return; // Silent fail
            return response.json();
        })
        .then(data => {
            if (data) updatePriceDisplay(data);
        })
        .catch(err => console.debug('Live price skipped:', err));
}

function updatePriceDisplay(data) {
    const priceEl = document.getElementById('price_idr');
    const changeEl = document.getElementById('price_change');
    const usdEl = document.getElementById('price_usd');

    if (!priceEl || !data) return;

    // Animate Update (Flash Color)
    const oldPrice = parseFloat(priceEl.dataset.value || 0);
    const newPrice = data.price_idr_gram;

    if (oldPrice !== 0 && oldPrice !== newPrice) {
        priceEl.style.transition = 'color 0.5s ease';
        priceEl.style.color = newPrice > oldPrice ? '#4caf50' : '#f44336';
        setTimeout(() => { priceEl.style.color = ''; }, 1000);
    }

    // Update Text
    priceEl.dataset.value = newPrice;
    priceEl.innerText = `Rp ${newPrice.toLocaleString('id-ID')}`;

    // Update USD/Oz (and Antam if we had it separately, but API combines for now)
    // We keep existing Antam text if present, or just update USD
    // For now, simpler is better:
    usdEl.innerText = `$ ${data.price_usd.toLocaleString()} / Oz`;

    // Change Tag
    changeEl.innerText = `${data.change_pct > 0 ? '+' : ''}${data.change_pct}%`;
    changeEl.className = `change-tag ${data.change_pct >= 0 ? 'bullish' : 'bearish'}`;
}

// ----------------------------------------
// Core Analytics & Signal (Low Frequency)
// ----------------------------------------
async function fetchAnalysis() {
    await updateDashboardData();
}

async function updateDashboardData() {
    showLoader(true);
    try {
        const response = await fetch('/api/prediction');
        const data = await response.json();

        if (data.error) {
            console.error(data.error);
            return;
        }

        // Update Signal & Target
        updateSignalCard(data);

        // Update Sentiment
        updateSentimentCard(data);

        // Note: We don't overwrite price_idr here to avoid conflict with realtime updates,
        // unless realtime updates are failing. 

    } catch (error) {
        console.error('Error fetching prediction:', error);
    } finally {
        showLoader(false);
    }
}

function updateSignalCard(data) {
    const signalText = document.getElementById('signal_value');
    const forecastHtml = `<div style="font-size: 0.8rem; margin-top: 5px; opacity: 0.7;">Forecast 24h: ${data.change_pct > 0 ? '+' : ''}${data.change_pct}%</div>`;

    signalText.innerHTML = `${data.recommendation} ${forecastHtml}`;
    signalText.className = data.recommendation === 'BUY' ? 'text-gold' : (data.recommendation === 'SELL' ? 'text-red' : 'text-grey');

    if (document.getElementById('action_date')) {
        document.getElementById('action_date').textContent = data.action_date;
    }

    const targetEl = document.getElementById('target_price');
    if (targetEl) {
        if (data.recommendation === 'BUY') {
            targetEl.textContent = `Sell Target: Rp ${data.predicted_price_idr_gram.toLocaleString()}`;
        } else if (data.recommendation === 'SELL') {
            targetEl.textContent = `Exit Price: Rp ${data.current_price_idr_gram.toLocaleString()}`;
        } else {
            targetEl.textContent = `Predicted: Rp ${data.predicted_price_idr_gram.toLocaleString()}`;
        }
    }
}

function updateSentimentCard(data) {
    const nPos = data.sentiment_breakdown.positive || 0;
    const nNeg = data.sentiment_breakdown.negative || 0;
    const total = nPos + nNeg;

    if (total > 0) {
        const bullPct = (nPos / total) * 100;
        const bearPct = (nNeg / total) * 100;
        document.getElementById('sentiment_bar').style.width = bullPct + '%';
        document.getElementById('bull_ratio').textContent = bullPct.toFixed(1) + '%';
        document.getElementById('bear_ratio').textContent = bearPct.toFixed(1) + '%';
    }
    document.getElementById('sentiment_avg').textContent = `Avg Score: ${data.sentiment_score}`;
}

// ----------------------------------------
// Charts
// ----------------------------------------
let historyData = null;
let mainChart = null;

async function renderChart() {
    try {
        const response = await fetch('/api/history');
        historyData = await response.json();
        drawChart(historyData.dates, historyData.prices);
    } catch (error) {
        console.error('Error rendering chart:', error);
    }
}

function drawChart(labels, values) {
    const ctx = document.getElementById('historyChart').getContext('2d');
    if (mainChart) mainChart.destroy();

    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(212, 175, 55, 0.2)');
    gradient.addColorStop(1, 'rgba(212, 175, 55, 0)');

    mainChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Gold Price (USD/Oz)',
                data: values,
                borderColor: '#D4AF37',
                borderWidth: 2,
                pointRadius: 0,
                fill: true,
                backgroundColor: gradient,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(0,0,0,0.8)',
                    borderColor: '#D4AF37',
                    borderWidth: 1
                }
            },
            scales: {
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#888' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#888', maxRotation: 0 }
                }
            }
        }
    });
}

// ----------------------------------------
// Interactions
// ----------------------------------------
function setupEventListeners() {
    // Forecast Button
    const runBtn = document.getElementById('run_forecast');
    if (runBtn) runBtn.addEventListener('click', runForecast);

    // Timeframe Buttons
    document.querySelectorAll('.tf-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            if (!historyData) return;

            const days = parseInt(e.target.dataset.days);
            document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');

            const filteredDates = historyData.dates.slice(-days);
            const filteredPrices = historyData.prices.slice(-days);
            drawChart(filteredDates, filteredPrices);
        });
    });
}

async function runForecast() {
    const days = document.getElementById('forecast_days').value;
    const body = document.getElementById('forecast_body');
    body.innerHTML = '<tr><td colspan="4" class="text-center">Calculating future trends...</td></tr>';

    try {
        const response = await fetch(`/api/forecast?days=${days}`);
        const data = await response.json();

        body.innerHTML = '';
        if (data.length === 0) {
            body.innerHTML = '<tr><td colspan="4" class="text-center">No forecast data.</td></tr>';
            return;
        }

        data.forEach((row, index) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${index + 1}</td>
                <td>${row.date}</td>
                <td>Rp ${row.price_idr.toLocaleString('id-ID')}</td>
                <td class="${row.change_pct >= 0 ? 'text-gold' : 'text-red'}">
                    ${row.change_pct > 0 ? '+' : ''}${row.change_pct}%
                </td>
            `;
            body.appendChild(tr);
        });
    } catch (error) {
        console.error('Error fetching forecast:', error);
        body.innerHTML = '<tr><td colspan="4" class="text-center text-red">Forecast failed.</td></tr>';
    }
}

function showLoader(show) {
    const overlay = document.getElementById('loading_overlay');
    if (!overlay) return;
    if (show) overlay.classList.add('active');
    else overlay.classList.remove('active');
}
