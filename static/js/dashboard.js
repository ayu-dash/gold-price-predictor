document.addEventListener('DOMContentLoaded', () => {
    // 1. Initial Data Fetch
    updateDashboard();
    renderChart();

    // 2. Event Listeners
    document.getElementById('run_forecast').addEventListener('click', runForecast);

    // Set current date
    const now = new Date();
    document.getElementById('current_date').textContent = now.toDateString();
    // 3. Auto-Refresh (Real-time Monitor)
    setInterval(updateDashboard, 60000); // Check for updates every 60s
});

async function updateDashboard() {
    showLoader(true);
    try {
        const response = await fetch('/api/prediction');
        const data = await response.json();

        if (data.error) {
            console.error(data.error);
            return;
        }

        // Update Price Card (Reverted to Spot Price)
        document.getElementById('price_idr').textContent = `Rp ${data.current_price_idr_gram.toLocaleString()}`;

        // Show Real Antam Price below (if available)
        let subText = `$ ${data.current_price_usd.toLocaleString()} / Oz`;
        if (data.physical_price_idr) {
            subText += ` | Antam: Rp ${data.physical_price_idr.toLocaleString()}`;
        }
        document.getElementById('price_usd').textContent = subText;

        // Update Price Tag with ACTUAL Daily Change
        const changeTag = document.getElementById('price_change');
        if (data.daily_change_pct) {
            changeTag.textContent = `${data.daily_change_pct > 0 ? '+' : ''}${data.daily_change_pct}%`;
            changeTag.className = data.daily_change_pct >= 0 ? 'change-positive' : 'change-negative';
        } else {
            // Fallback for first load
            changeTag.textContent = "0.00%";
            changeTag.className = 'change-neutral';
        }

        // Add Forecast info to Signal Box
        const signalText = document.getElementById('signal_value');
        const forecastHtml = `<div style="font-size: 0.8rem; margin-top: 5px; opacity: 0.7;">Forecast 24h: ${data.change_pct > 0 ? '+' : ''}${data.change_pct}%</div>`;

        signalText.innerHTML = `${data.recommendation} ${forecastHtml}`;
        signalText.className = data.recommendation === 'BUY' ? 'text-gold' : (data.recommendation === 'SELL' ? 'text-red' : 'text-grey');
        document.getElementById('action_date').textContent = data.action_date;

        // Target Price
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

        // Update Sentiment
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

    } catch (error) {
        console.error('Error fetching prediction:', error);
    } finally {
        showLoader(false);
    }
}

let historyData = null;
let mainChart = null;

async function renderChart() {
    try {
        const response = await fetch('/api/history');
        historyData = await response.json();

        // Initial Draw
        drawChart(historyData.dates, historyData.prices);

        // Timeframe listeners
        document.querySelectorAll('.tf-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const days = parseInt(e.target.dataset.days);

                // Update active state
                document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');

                // Filter Data
                const filteredDates = historyData.dates.slice(-days);
                const filteredPrices = historyData.prices.slice(-days);

                drawChart(filteredDates, filteredPrices);
            });
        });

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

async function runForecast() {
    const days = document.getElementById('forecast_days').value;
    const body = document.getElementById('forecast_body');
    body.innerHTML = '<tr><td colspan="4" class="text-center">Calculating future trends...</td></tr>';

    try {
        const response = await fetch(`/api/forecast?days=${days}`);
        const data = await response.json();

        body.innerHTML = '';
        data.forEach((row, index) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${index + 1}</td>
                <td>${row.date}</td>
                <td>Rp ${row.price_idr.toLocaleString()}</td>
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
    if (show) overlay.classList.add('active');
    else overlay.classList.remove('active');
}
