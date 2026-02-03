let technicalData = null;
let charts = {
    price: null,
    rsi: null,
    macd: null
};

document.addEventListener('DOMContentLoaded', () => {
    fetchTechnicalData();

    const now = new Date();
    document.getElementById('current_date').textContent = now.toDateString();
});

async function fetchTechnicalData() {
    showLoader(true);
    try {
        const response = await fetch('/api/technical');
        technicalData = await response.json();

        // Initial Draw (Default logic: last data available)
        updateAllCharts(90); // Default 3 months

        // Timeframe listeners
        document.querySelectorAll('.tf-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const days = parseInt(e.target.dataset.days);

                document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');

                updateAllCharts(days);
            });
        });

    } catch (error) {
        console.error('Error fetching technical data:', error);
    } finally {
        showLoader(false);
    }
}

function updateAllCharts(days) {
    const dates = technicalData.dates.slice(-days);

    renderPriceChart(dates, technicalData.prices.slice(-days),
        technicalData.sma_50.slice(-days), technicalData.sma_200.slice(-days));

    renderRSIChart(dates, technicalData.rsi.slice(-days));

    renderMACDChart(dates, technicalData.macd.slice(-days), technicalData.macd_signal.slice(-days));
}

function renderPriceChart(labels, prices, sma50, sma200) {
    const ctx = document.getElementById('priceChart').getContext('2d');
    if (charts.price) charts.price.destroy();

    charts.price = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                { label: 'Gold Price', data: prices, borderColor: '#D4AF37', borderWidth: 2, pointRadius: 0, tension: 0.1 },
                { label: 'SMA 50', data: sma50, borderColor: 'rgba(75, 255, 75, 0.5)', borderWidth: 1, pointRadius: 0, borderDash: [5, 5] },
                { label: 'SMA 200', data: sma200, borderColor: 'rgba(255, 75, 75, 0.5)', borderWidth: 1, pointRadius: 0, borderDash: [5, 5] }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#888' } },
                x: { grid: { display: false }, ticks: { color: '#888' } }
            },
            plugins: {
                legend: { labels: { color: '#888' } }
            }
        }
    });
}

function renderRSIChart(labels, rsiData) {
    const ctx = document.getElementById('rsiChart').getContext('2d');
    if (charts.rsi) charts.rsi.destroy();

    charts.rsi = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'RSI (14)',
                data: rsiData,
                borderColor: '#4BFF4B',
                borderWidth: 2,
                pointRadius: 0,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                y: { min: 0, max: 100, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#888' } },
                x: { grid: { display: false }, ticks: { display: false } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function renderMACDChart(labels, macd, signal) {
    const ctx = document.getElementById('macdChart').getContext('2d');
    if (charts.macd) charts.macd.destroy();

    charts.macd = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                { label: 'MACD', data: macd, borderColor: '#D4AF37', borderWidth: 2, pointRadius: 0 },
                { label: 'Signal', data: signal, borderColor: 'rgba(255, 255, 255, 0.3)', borderWidth: 1, pointRadius: 0 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#888' } },
                x: { grid: { display: false }, ticks: { color: '#888' } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function showLoader(show) {
    const overlay = document.getElementById('loading_overlay');
    if (show) overlay.classList.add('active');
    else overlay.classList.remove('active');
}
