/**
 * Dashboard Logic for Gold Price Predictor
 * Handles Realtime Updates, Chart Rendering, and Polling.
 */

document.addEventListener('DOMContentLoaded', function () {
    // 1. Initial Data Fetch
    fetchAnalysis();
    renderChart();
    runForecast(); // Auto-run on load

    // 2. Setup Event Listeners
    setupEventListeners();
    setupPortfolioLogic();

    // 3. Start Realtime Polling (Every 30 seconds)
    startRealtimeUpdates();

    // 4. Set Date
    const now = new Date();
    document.getElementById('current_date').textContent = now.toDateString();

    // 5. Start Dashboard Refresh Loop (Every 60s) - SILENT
    setInterval(() => updateDashboardData(true), 60000);

    // 6. Signal Countdown
    startSignalCountdown();
});

// ----------------------------------------
// Realtime Price Updates (High Frequency)
// ----------------------------------------
let countdownSeconds = 0;

function startSignalCountdown() {
    // Calculate seconds until midnight
    function getSecondsUntilMidnight() {
        const now = new Date();
        const midnight = new Date(now);
        midnight.setHours(24, 0, 0, 0); // Next midnight
        return Math.floor((midnight - now) / 1000);
    }

    // Initial calculation
    countdownSeconds = getSecondsUntilMidnight();
    updateCountdownUI();

    // Tick every second
    setInterval(() => {
        if (countdownSeconds > 0) {
            countdownSeconds--;
            updateCountdownUI();
        } else {
            // Reset at midnight
            countdownSeconds = getSecondsUntilMidnight();
            // Trigger data refresh at midnight
            updateDashboardData(true);
        }
    }, 1000);
}

function updateCountdownUI() {
    const el = document.getElementById('signal_countdown');
    if (!el) return;

    const h = Math.floor(countdownSeconds / 3600);
    const m = Math.floor((countdownSeconds % 3600) / 60);
    const s = countdownSeconds % 60;

    // Display as HH:MM:SS
    el.innerText = `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    el.title = 'Signal Valid Until 00:00';

    // Highlight when close to midnight
    if (countdownSeconds < 300) { // Last 5 minutes
        el.style.color = 'var(--accent-gold)';
        el.style.opacity = '1';
    } else {
        el.style.color = '';
        el.style.opacity = '0.6';
    }
}

function startRealtimeUpdates() {
    // Initial fetch
    fetchRealtimePrice();
    // Poll every 30 seconds
    setInterval(fetchRealtimePrice, 30000);

    // Terminal Heartbeat (Monitor & Ticker)
    updateMarketMonitor();
    setInterval(updateMarketMonitor, 60000); // 1-minute refresh
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
    changeEl.innerText = `${data.change_pct > 0 ? '+' : ''}${data.change_pct.toFixed(2)}%`;
    changeEl.className = `change-tag ${data.change_pct >= 0 ? 'bullish' : 'bearish'}`;

    // Update Spread Radar & Portfolio
    updateSpreadAndPortfolio(data.price_idr_gram);

    // --- CHART REALTIME PUSH ---
    // Instantly update the last point of the chart if it exists
    if (mainChart && mainChart.data.datasets.length === 1) {
        // Assuming the last point is "Today/Now" (which our backend ensures)
        const lastIdx = mainChart.data.datasets[0].data.length - 1;
        mainChart.data.datasets[0].data[lastIdx] = data.price_usd;
        mainChart.update();
    }
    // ---------------------------

    // --- REALTIME TARGET SYNC ---
    // If we have a stored forecast return, update the target price and signal dynamically
    // so they stay consistent with the moving real-time price.
    if (window.lastForecastPct !== undefined) {
        reassessSignal(newPrice, window.lastForecastPct, window.lastConfDirection, window.lastConfScore, window.lastIsBullish);
    }
}

function reassessSignal(currentPriceIdr, forecastPct, confDirection = null, confScore = 0, isBullish = false) {
    // ForecastPct is in percent (e.g. 1.2 for 1.2%)
    const predictedPrice = currentPriceIdr * (1 + (forecastPct / 100));
    const targetEl = document.getElementById('target_price');
    const signalEl = document.getElementById('signal_value');

    // Update Target Text
    if (targetEl) {
        targetEl.innerText = `Next-Day Target: Rp ${Math.round(predictedPrice).toLocaleString('id-ID')}`;
    }

    // Recalculate Signal with Momentum Filtering (Sync with predictor.py)
    const REDUCE_THRESHOLD = 0.4;
    const SELL_THRESHOLD = 0.75;
    const BULL_BOOST = 0.15; // Extra buffer if bullish

    let recommendation = 'HOLD';

    // 1. Momentum-Aware Filtering
    if (isBullish && forecastPct < 0) {
        if (Math.abs(forecastPct) < BEAR_PROTECTION) {
            recommendation = 'HOLD';
        } else if (Math.abs(forecastPct) > SELL_THRESHOLD) {
            recommendation = 'REDUCE'; // Downgrade Sell to Reduce if Bullish momentum is strong
        }
    } else {
        // 2. Standard Logic
        if (forecastPct > SELL_THRESHOLD) recommendation = 'BUY';
        else if (forecastPct > REDUCE_THRESHOLD) recommendation = 'ACCUMULATE';
        else if (forecastPct < -SELL_THRESHOLD) recommendation = 'SELL';
        else if (forecastPct < -REDUCE_THRESHOLD) recommendation = 'REDUCE';
        else if (confScore > 75.0) {
            // High confidence for small moves
            if (confDirection === 'UP' && forecastPct > 0.15) recommendation = 'ACCUMULATE';
            else if (confDirection === 'DOWN' && forecastPct < -0.15) recommendation = 'REDUCE';
        }
    }

    if (signalEl) {
        signalEl.innerText = recommendation;
        // Apply coloring based on signal
        signalEl.className = 'signal-value'; // Reset
        if (recommendation === 'BUY' || recommendation === 'ACCUMULATE') signalEl.classList.add('text-gold');
        else if (recommendation === 'SELL' || recommendation === 'REDUCE') signalEl.classList.add('text-red');
        else signalEl.classList.add('text-white');
    }

    // Generate Logic Summary
    let logicText = '';
    if (recommendation === 'BUY') {
        logicText = `AI mendeteksi potensi rally signifikan (>0.5%). Waktu yang tepat untuk <strong>Enter Market</strong> atau menambah posisi.`;
    } else if (recommendation === 'SELL') {
        logicText = `Harga diperkirakan turun tajam (>0.5%). Disarankan untuk <strong>Exit Market</strong> atau mengamankan keuntungan (Take Profit).`;
    } else if (recommendation === 'ACCUMULATE') {
        logicText = `Sinyal kenaikan stabil dengan keyakinan tinggi (${confScore}%). Disarankan <strong>Cicil Beli</strong> secara bertahap.`;
    } else if (recommendation === 'REDUCE') {
        logicText = `Sinyal penurunan stabil dengan keyakinan tinggi (${confScore}%). Disarankan <strong>Kurangi Posisi</strong> untuk meminimalkan risiko.`;
    } else {
        logicText = `Pasar cenderung netral atau dalam fase konsolidasi. Strategi terbaik saat ini adalah <strong>Wait & See</strong>.`;
    }

    const logicEl = document.getElementById('signal_logic_text');
    if (logicEl) logicEl.innerHTML = logicText;
}

function updateSpreadAndPortfolio(currentPrice) {
    // 1. Update Spread Radar
    const physEl = document.getElementById('physical_price');
    const spotEl = document.getElementById('spot_price');
    const diffEl = document.getElementById('spread_diff');

    const physicalPrice = parseInt(window.lastPhysicalPrice || 0);
    if (physicalPrice > 0) {
        physEl.innerText = `Rp ${physicalPrice.toLocaleString('id-ID')}`;
        spotEl.innerText = `Rp ${currentPrice.toLocaleString('id-ID')}`;

        const spread = ((physicalPrice - currentPrice) / currentPrice) * 100;
        diffEl.innerText = `Spread: ${spread > 0 ? '+' : ''}${spread.toFixed(2)}%`;
        diffEl.className = `spread-diff ${spread < 5 ? 'text-green' : 'text-gold'}`;
    }

    // 2. Update Portfolio Value
    const portfolio = JSON.parse(localStorage.getItem('gold_portfolio') || '{"amount": 0, "avgPrice": 0}');
    if (portfolio.amount > 0) {
        const totalValue = portfolio.amount * currentPrice;
        const totalCost = portfolio.amount * portfolio.avgPrice;
        const pl = totalValue - totalCost;
        const plPct = (pl / totalCost) * 100;

        document.getElementById('portfolio_value').innerText = `Rp ${Math.round(totalValue).toLocaleString('id-ID')}`;
        const plEl = document.getElementById('portfolio_pl');
        plEl.innerText = `${pl >= 0 ? '+' : ''}Rp ${Math.round(pl).toLocaleString('id-ID')} (${plPct.toFixed(2)}%)`;
        plEl.className = `value ${pl >= 0 ? 'text-green' : 'text-red'}`;

        document.getElementById('portfolio_holdings').innerText = portfolio.amount;
        document.getElementById('portfolio_avg_price').innerText = `Rp ${portfolio.avgPrice.toLocaleString('id-ID')}`;
    }
}

// ----------------------------------------
// Portfolio Logic
// ----------------------------------------
function setupPortfolioLogic() {
    const editBtn = document.getElementById('edit_portfolio');
    const modal = document.getElementById('portfolio_modal');
    const cancelBtn = document.getElementById('cancel_portfolio');
    const saveBtn = document.getElementById('save_portfolio');

    if (!editBtn) return;

    editBtn.onclick = () => {
        const portfolio = JSON.parse(localStorage.getItem('gold_portfolio') || '{"amount": 0, "avgPrice": 0}');
        document.getElementById('input_gold_amount').value = portfolio.amount;
        document.getElementById('input_purchase_price').value = portfolio.avgPrice;
        modal.classList.add('active');
    };

    cancelBtn.onclick = () => modal.classList.remove('active');

    saveBtn.onclick = () => {
        const amount = parseFloat(document.getElementById('input_gold_amount').value) || 0;
        const avgPrice = parseInt(document.getElementById('input_purchase_price').value) || 0;

        localStorage.setItem('gold_portfolio', JSON.stringify({ amount, avgPrice }));
        modal.classList.remove('active');

        // Trigger UI update if we have current price
        const currentPrice = parseFloat(document.getElementById('price_idr').dataset.value || 0);
        if (currentPrice > 0) updateSpreadAndPortfolio(currentPrice);
    };
}

// ----------------------------------------
// Core Analytics & Signal (Low Frequency)
// ----------------------------------------
async function fetchAnalysis(silent = false) {
    // Slider Value Updates removed
    await updateDashboardData(silent);
}

async function updateDashboardData(silent = false) {
    if (!silent) showLoader(true);
    try {
        const response = await fetch('/api/prediction');
        const data = await response.json();

        if (data.error) {
            console.error(data.error);
            return;
        }

        // Update Signal & Target
        try { updateSignalCard(data); } catch (e) { console.error("Signal update failed", e); }

        // Update Sentiment
        try { updateSentimentCard(data); } catch (e) { console.error("Sentiment update failed", e); }

        // Update Risk Watchlist
        try { updateRiskWatchlist(data); } catch (e) { console.error("Risk watchlist update failed", e); }

        // Store physical price for spread calculation
        window.lastPhysicalPrice = data.physical_price_idr;
        // Store forecast return for realtime sync
        window.lastForecastPct = data.change_pct;
        window.lastConfDirection = data.confidence_direction;
        window.lastConfScore = data.confidence_score;
        window.lastIsBullish = data.is_bullish; // Store for realtime updates

        // --- SIGNAL DISCREPANCY FIX ---
        // Explicitly update the main price display to match the "Current Price" 
        // returned by the prediction API. This ensures the Signal (which is based on this price)
        // makes sense visually to the user.
        if (data.current_price_idr_gram) {
            const priceEl = document.getElementById('price_idr');
            if (priceEl) {
                // Manually trigger spread update here to guarantee sync
                if (data.physical_price_idr) {
                    updateSpreadAndPortfolio(data.current_price_idr_gram);
                }
                priceEl.dataset.rate = data.current_idr_rate;
                // Force update the visual display
                updatePriceDisplay({
                    price_idr_gram: data.current_price_idr_gram,
                    price_usd: data.current_price_usd,
                    change_pct: data.daily_change_pct // Use actual daily change
                });
            }
        }
        // -----------------------------

    } finally {
        if (!silent) showLoader(false);
    }

    // Auto-refresh forecast significantly less often or on data change?
    // For now, let's sync it with dashboard updates to be "realtime"
    if (!silent) runForecast();
}

function updateSignalCard(data) {
    const signalText = document.getElementById('signal_value');
    const forecastHtml = `<div style="font-size: 0.8rem; margin-top: 5px; opacity: 0.7;">Forecast 24h: ${data.change_pct > 0 ? '+' : ''}${data.change_pct}%</div>`;
    const signalEl = document.getElementById('signal_value');
    const targetEl = document.getElementById('target_price');
    const lightEl = document.getElementById('signal_light');

    // Standardize recommendation logic across UI (respecting backend is_bullish)
    const REDUCE_THRESHOLD = 0.4;
    const SELL_THRESHOLD = 0.75;
    const BEAR_PROTECTION = 0.5;
    let rec = 'HOLD';

    const isBullContext = data.is_bullish || false;
    const forecastVal = parseFloat(data.change_pct);

    if (isBullContext && forecastVal < 0 && Math.abs(forecastVal) < BEAR_PROTECTION) {
        rec = 'HOLD';
    } else {
        if (forecastVal > SELL_THRESHOLD) rec = 'BUY';
        else if (forecastVal > REDUCE_THRESHOLD) rec = 'ACCUMULATE';
        else if (forecastVal < -SELL_THRESHOLD) rec = 'SELL';
        else if (forecastVal < -REDUCE_THRESHOLD) rec = 'REDUCE';
        else if (data.confidence_score > 75) {
            if (data.confidence_direction === 'UP' && forecastVal > 0.15) rec = 'ACCUMULATE';
            else if (data.confidence_direction === 'DOWN' && forecastVal < -0.15) rec = 'REDUCE';
        }
    }

    signalEl.innerText = rec;
    // ... rest handled by CSS classes later

    // Display Target + Confidence
    // Display Target
    targetEl.innerText = `Next-Day Target: Rp ${data.predicted_price_idr_gram.toLocaleString('id-ID')}`;

    // Display Confidence (Create element if missing)
    let confEl = document.getElementById('signal_conf');
    if (!confEl) {
        confEl = document.createElement('div');
        confEl.id = 'signal_conf';
        confEl.style.fontSize = '0.8rem';
        confEl.style.color = '#888';
        confEl.style.marginTop = '4px';
        // Append after targetEl
        targetEl.parentNode.insertBefore(confEl, targetEl.nextSibling);
    }

    if (data.confidence_score > 0) {
        confEl.innerText = `Prob. ${data.confidence_direction}: ${data.confidence_score}%`;
        // Color code logic
        confEl.style.color = data.confidence_direction === 'UP' ? '#4caf50' : '#f44336';
    } else {
        confEl.innerText = '';
    }

    // AI Prediction Data
    const predDate = document.getElementById('action_date');
    const lastDbEl = document.getElementById('last_db_update');
    if (predDate) predDate.innerText = data.action_date;
    if (lastDbEl) lastDbEl.innerText = data.last_db_date;

    // Signal Light Logic
    if (lightEl) {
        lightEl.className = 'status-light'; // Reset
        const change = data.change_pct;
        if (data.recommendation === 'BUY' && change > 1.0) lightEl.classList.add('green');
        else if (data.recommendation === 'SELL' && change < -1.0) lightEl.classList.add('red');
        else lightEl.classList.add('yellow');
    }

    // Update Conclusion
    reassessSignal(data.current_price_idr_gram, data.change_pct, data.confidence_direction, data.confidence_score, data.is_bullish);
}

let commentaryInterval = null;

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

    // Live Commentary Update (AI Insights + Rolling News)
    const commBox = document.getElementById('live_commentary_content');
    if (commBox) {
        let items = [];

        // 1. Add AI-aligned market insights first (priority)
        if (data.market_insights && data.market_insights.length > 0) {
            data.market_insights.forEach(insight => {
                items.push({
                    type: 'insight',
                    title: insight.title,
                    desc: insight.desc
                });
            });
        }

        // 2. Add news headlines after insights
        if (data.top_headlines && data.top_headlines.length > 0) {
            data.top_headlines.slice(0, 5).forEach(headline => {
                items.push({
                    type: 'news',
                    text: headline
                });
            });
        }

        // Fallback if no items
        if (items.length === 0) {
            items.push({ type: 'news', text: 'Monitoring global market news...' });
        }

        // 3. Check if data changed significantly
        const currentHash = JSON.stringify(items);
        if (commBox.dataset.lastHash !== currentHash) {
            commBox.dataset.lastHash = currentHash;

            // Stop existing interval if any
            if (window.newsInterval) clearInterval(window.newsInterval);

            let currentIndex = 0;
            const updateHeadline = () => {
                // Fade out
                commBox.style.opacity = '0';

                setTimeout(() => {
                    // Change text
                    const item = items[currentIndex];
                    if (item.type === 'insight') {
                        // AI Insight styling - more prominent
                        commBox.innerHTML = `
                            <div class="insight-item" style="border-left: 3px solid var(--terminal-gold); padding-left: 10px;">
                                <strong style="color: var(--terminal-gold);">${item.title}</strong><br>
                                <span style="opacity: 0.85;">${item.desc}</span>
                            </div>`;
                    } else {
                        // News headline styling
                        commBox.innerHTML = `<div class="news-item">> ${item.text}</div>`;
                    }

                    // Fade in
                    commBox.style.opacity = '1';

                    // Next index
                    currentIndex = (currentIndex + 1) % items.length;
                }, 500); // 0.5s fade out
            };

            // Initial render
            commBox.style.transition = 'opacity 0.5s ease';
            updateHeadline();

            // Start rotation (every 5 seconds for insights, 4 for news)
            if (items.length > 1) {
                window.newsInterval = setInterval(updateHeadline, 5000);
            }
        }
    }
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

function drawChart(labels, values, lowValues = null, highValues = null) {
    const ctx = document.getElementById('historyChart').getContext('2d');
    if (mainChart) mainChart.destroy();

    const datasets = [{
        label: 'Gold Price (USD/Oz)',
        data: values,
        borderColor: '#D4AF37',
        borderWidth: 2,
        pointRadius: 0,
        fill: false,
        tension: 0.4,
        zIndex: 10
    }];

    // Add Confidence Interval band if data exists
    if (lowValues && highValues) {
        datasets.unshift({
            label: 'Confidence Interval (95%)',
            data: highValues,
            borderColor: 'rgba(212, 175, 55, 0)',
            backgroundColor: 'rgba(212, 175, 55, 0.1)',
            fill: '+1', // Fill to the next dataset (lowValues)
            pointRadius: 0,
            tension: 0.4
        }, {
            label: 'Lower Bound',
            data: lowValues,
            borderColor: 'rgba(212, 175, 55, 0)',
            backgroundColor: 'transparent',
            pointRadius: 0,
            fill: false,
            tension: 0.4
        });
    }

    mainChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
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
    // Forecast Input Change
    const daysInput = document.getElementById('forecast_days');
    if (daysInput) {
        daysInput.addEventListener('change', runForecast);
        daysInput.addEventListener('keyup', (e) => {
            if (e.key === 'Enter') runForecast();
        });
    }

    // Force DB Sync
    const syncBtn = document.getElementById('force_sync_btn');
    if (syncBtn) {
        syncBtn.onclick = async () => {
            syncBtn.classList.add('spinning');
            try {
                const res = await fetch('/api/force_db_update');
                const result = await res.json();
                if (result.status === 'success') {
                    alert('Database update completed successfully.');
                    location.reload(); // Reload to refresh all data
                } else {
                    alert('Error: ' + result.message);
                }
            } catch (err) {
                alert('Connection error.');
            } finally {
                syncBtn.classList.remove('spinning');
            }
        };
    }

    // Reset Simulation button removed

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
    console.log("[Auto-Run] Starting forecast update...");
    const days = document.getElementById('forecast_days').value;
    const body = document.getElementById('forecast_body');
    body.innerHTML = '<tr><td colspan="6" class="text-center">Updating...</td></tr>';

    try {
        const url = `/api/forecast?days=${days}`;
        const response = await fetch(url);
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
                <td>Rp ${row.price_min.toLocaleString('id-ID')}</td>
                <td>Rp ${row.price_max.toLocaleString('id-ID')}</td>
                <td style="font-weight: 600; color: var(--gold);">Rp ${row.price_idr.toLocaleString('id-ID')}</td>
                <td class="${row.change_pct >= 0 ? 'text-gold' : 'text-red'}">
                    ${row.change_pct > 0 ? '+' : ''}${row.change_pct}%
                </td>
            `;
            body.appendChild(tr);
        });

        // --- CHART ENHANCEMENT ---
        if (historyData) {
            const gramsPerOz = 31.1035;
            const priceEl = document.getElementById('price_idr');
            const rate = parseFloat(priceEl ? priceEl.dataset.rate : 15500) || 15500;

            const forecastDates = data.map(d => d.date);
            const forecastValues = data.map(d => (d.price_idr * gramsPerOz) / rate);
            const lowValues = data.map(d => (d.price_min * gramsPerOz) / rate);
            const highValues = data.map(d => (d.price_max * gramsPerOz) / rate);

            // Combine last 20 days of history + forecast
            const combinedLabels = [...historyData.dates.slice(-20), ...forecastDates];
            const historyPrices = historyData.prices.slice(-20);
            const combinedValues = [...historyPrices, ...forecastValues];

            // CI bands need padding (nulls) for the historical portion
            const pad = new Array(20).fill(null);
            // We want to connect the last historical point to the first forecast point in the CI
            pad[19] = historyPrices[19];

            const paddedLow = [...pad, ...lowValues];
            const paddedHigh = [...pad, ...highValues];

            drawChart(combinedLabels, combinedValues, paddedLow, paddedHigh);
        }

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
function updateRiskWatchlist(data) {
    const listEl = document.getElementById('risk_watchlist');
    // Ensure we have data
    if (!listEl || !data || !data.dynamic_risks) return;

    // Force clear and update (removed hash check to debug "static" issue)
    listEl.innerHTML = '';

    if (data.dynamic_risks.length === 0) {
        listEl.innerHTML = '<li><span class="dot green"></span><div>Market Conditions Stable</div></li>';
        return;
    }

    data.dynamic_risks.forEach((risk) => {
        const li = document.createElement('li');

        let dotColor = 'gold';
        if (risk.severity && risk.severity === 'high') dotColor = 'red';
        else if (risk.severity && risk.severity === 'low') dotColor = 'green';

        li.innerHTML = `
            <span class="dot ${dotColor}"></span>
            <div><strong>${risk.title}</strong>: ${risk.desc}</div>
        `;
        listEl.appendChild(li);
    });
}

/**
 * Bloomberg Terminal Data Heartbeat
 */
async function updateMarketMonitor() {
    const monitorBody = document.getElementById('monitor_body');
    const tickerContent = document.getElementById('ticker_content');
    if (!monitorBody || !tickerContent) return;

    try {
        const response = await fetch('/api/market_monitor');
        const data = await response.json();

        // 1. Update Market Monitor Table (Table View)
        let monitorHtml = '';
        let tickerHtml = '';

        Object.entries(data).forEach(([name, info]) => {
            const isUp = info.change_pct >= 0;
            const sign = isUp ? '+' : '';
            const colorClass = isUp ? 'trend-up' : 'trend-down';
            const icon = isUp ? '▲' : '▼';

            // Table Row
            monitorHtml += `
                <tr>
                    <td style="text-align: left; font-weight: 700;">${name}</td>
                    <td>${info.price.toLocaleString()}</td>
                    <td class="${colorClass}">${sign}${info.change_pct.toFixed(2)}%</td>
                    <td class="${colorClass}">${icon}</td>
                </tr>
            `;

            // Ticker Item for Scrolling
            tickerHtml += `
                <span class="ticker-item">
                    ${name} ${info.price.toLocaleString()} 
                    <span class="${isUp ? 'change-up' : 'change-down'}">${sign}${info.change_pct.toFixed(2)}%</span>
                </span>
            `;
        });

        monitorBody.innerHTML = monitorHtml;

        // Double the ticker content to ensure seamless loop
        tickerContent.innerHTML = tickerHtml + tickerHtml;

    } catch (err) {
        console.error('Terminal feed error:', err);
    }
}
