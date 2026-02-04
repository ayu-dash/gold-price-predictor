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
    setupPortfolioLogic();

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
        reassessSignal(newPrice, window.lastForecastPct, window.lastConfDirection, window.lastConfScore);
    }
}

function reassessSignal(currentPriceIdr, forecastPct, confDirection = null, confScore = 0) {
    // ForecastPct is in percent (e.g. 1.2 for 1.2%)
    const predictedPrice = currentPriceIdr * (1 + (forecastPct / 100));
    const targetEl = document.getElementById('target_price');
    const signalEl = document.getElementById('signal_value');
    const lightEl = document.getElementById('signal_light');

    // Update Target Text
    if (targetEl) {
        targetEl.innerText = `Next-Day Target: Rp ${Math.round(predictedPrice).toLocaleString('id-ID')}`;
    }

    // Recalculate Signal (Threshold 0.5% = 0.5)
    let recommendation = 'HOLD';
    if (forecastPct > 0.5) recommendation = 'BUY';
    else if (forecastPct < -0.5) recommendation = 'SELL';
    else if (confScore > 65.0) {
        if (confDirection === 'UP' && forecastPct > 0.01) recommendation = 'ACCUMULATE';
        else if (confDirection === 'DOWN' && forecastPct < -0.01) recommendation = 'REDUCE';
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
async function fetchAnalysis() {
    // Slider Value Updates
    ['dxy', 'oil', 'idr'].forEach(key => {
        const slider = document.getElementById(`shift_${key}`);
        const label = document.getElementById(`v_${key}`);
        if (slider && label) {
            // Initialize label with current slider value
            label.innerText = `${slider.value > 0 ? '+' : ''}${slider.value}%`;
            slider.oninput = () => label.innerText = `${slider.value > 0 ? '+' : ''}${slider.value}%`;
        }
    });
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

        // Store physical price for spread calculation
        window.lastPhysicalPrice = data.physical_price_idr;
        // Store forecast return for realtime sync
        window.lastForecastPct = data.change_pct;
        window.lastConfDirection = data.confidence_direction;
        window.lastConfScore = data.confidence_score;

        // --- SIGNAL DISCREPANCY FIX ---
        // Explicitly update the main price display to match the "Current Price" 
        // returned by the prediction API. This ensures the Signal (which is based on this price)
        // makes sense visually to the user.
        if (data.current_price_idr_gram) {
            const priceEl = document.getElementById('price_idr');
            if (priceEl) {
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

    } catch (error) {
        console.error('Error fetching prediction:', error);
    } finally {
        showLoader(false);
    }
}

function updateSignalCard(data) {
    const signalText = document.getElementById('signal_value');
    const forecastHtml = `<div style="font-size: 0.8rem; margin-top: 5px; opacity: 0.7;">Forecast 24h: ${data.change_pct > 0 ? '+' : ''}${data.change_pct}%</div>`;
    const signalEl = document.getElementById('signal_value');
    const targetEl = document.getElementById('target_price');
    const lightEl = document.getElementById('signal_light');

    signalEl.innerText = data.recommendation;

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

    document.getElementById('action_date').innerText = data.action_date;

    // Signal Light Logic
    if (lightEl) {
        lightEl.className = 'status-light'; // Reset
        const change = data.change_pct;
        if (data.recommendation === 'BUY' && change > 1.0) lightEl.classList.add('green');
        else if (data.recommendation === 'SELL' && change < -1.0) lightEl.classList.add('red');
        else lightEl.classList.add('yellow');
    }

    // Update Conclusion
    reassessSignal(data.current_price_idr_gram, data.change_pct, data.confidence_direction, data.confidence_score);
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
    const shiftDxy = document.getElementById('shift_dxy').value;
    const shiftOil = document.getElementById('shift_oil').value;
    const shiftIdr = document.getElementById('shift_idr').value;

    const body = document.getElementById('forecast_body');
    body.innerHTML = '<tr><td colspan="6" class="text-center">Calculating future scenarios...</td></tr>';

    try {
        const url = `/api/forecast?days=${days}&dxy_shift=${shiftDxy}&oil_shift=${shiftOil}&idr_shift=${shiftIdr}`;
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
                <td class="text-muted">Rp ${row.price_min.toLocaleString('id-ID')}</td>
                <td class="text-muted">Rp ${row.price_max.toLocaleString('id-ID')}</td>
                <td style="font-weight: 600;">Rp ${row.price_idr.toLocaleString('id-ID')}</td>
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
    } finally {
        showLoader(false);
    }
}

function showLoader(show) {
    const overlay = document.getElementById('loading_overlay');
    if (!overlay) return;
    if (show) overlay.classList.add('active');
    else overlay.classList.remove('active');
}
