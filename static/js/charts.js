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
        // ApexCharts handles large datasets better, can load more.
        updateAllCharts(180);

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

    // Convert dates to timestamps for X-axis sync
    // Not strictly necessary if categories match exactly, but cleaner for timeseries

    renderPriceChart(dates,
        technicalData.prices.slice(-days),
        technicalData.sma_50.slice(-days),
        technicalData.sma_200.slice(-days)
    );

    renderRSIChart(dates, technicalData.rsi.slice(-days));

    renderMACDChart(dates,
        technicalData.macd.slice(-days),
        technicalData.macd_signal.slice(-days)
    );
}

// Common Options for Global Style
const commonOptions = {
    chart: {
        background: 'transparent',
        foreColor: '#888',
        toolbar: { show: true, tools: { download: false } },
        zoom: { enabled: true, autoScaleYaxis: true }
    },
    theme: { mode: 'dark' },
    stroke: { curve: 'smooth', width: 2 },
    grid: { borderColor: 'rgba(255,255,255,0.05)' },
    dataLabels: { enabled: false },
    xaxis: {
        tooltip: { enabled: false },
        axisBorder: { show: false },
        axisTicks: { show: false }
    },
    tooltip: { theme: 'dark' }
};

function renderPriceChart(labels, prices, sma50, sma200) {
    if (charts.price) charts.price.destroy();

    const options = {
        ...commonOptions,
        chart: {
            ...commonOptions.chart,
            id: 'price-chart',
            group: 'social', // Syncs zoom with others
            type: 'area', // Area for price looks better
            height: 400
        },
        series: [
            { name: 'Gold Price', data: prices },
            { name: 'SMA 50', data: sma50 },
            { name: 'SMA 200', data: sma200 }
        ],
        colors: ['#D4AF37', '#4BFF4B', '#FF4B4B'],
        fill: {
            type: ['gradient', 'solid', 'solid'],
            gradient: {
                shadeIntensity: 1,
                opacityFrom: 0.4,
                opacityTo: 0.05,
                stops: [0, 100]
            }
        },
        xaxis: { categories: labels, labels: { show: false } }, // Hide X labels on top chart to save space?
        yaxis: {
            labels: {
                formatter: (val) => { return val.toFixed(0) }
            }
        },
        stroke: {
            width: [2, 1, 1],
            dashArray: [0, 5, 5] // Dashed lines for SMAs
        }
    };

    charts.price = new ApexCharts(document.querySelector("#priceChart"), options);
    charts.price.render();
}

function renderRSIChart(labels, rsiData) {
    if (charts.rsi) charts.rsi.destroy();

    const options = {
        ...commonOptions,
        chart: {
            ...commonOptions.chart,
            id: 'rsi-chart',
            group: 'social',
            type: 'line',
            height: 200
        },
        series: [{ name: 'RSI (14)', data: rsiData }],
        colors: ['#4BFF4B'],
        yaxis: { min: 0, max: 100, tickAmount: 4 },
        xaxis: { categories: labels, labels: { show: false } },
        annotations: {
            yaxis: [
                { y: 70, borderColor: '#FF4B4B', label: { style: { color: '#fff', background: '#FF4B4B' }, text: 'Overbought' } },
                { y: 30, borderColor: '#4BFF4B', label: { style: { color: '#fff', background: '#4BFF4B' }, text: 'Oversold' } }
            ]
        }
    };

    charts.rsi = new ApexCharts(document.querySelector("#rsiChart"), options);
    charts.rsi.render();
}

function renderMACDChart(labels, macd, signal) {
    if (charts.macd) charts.macd.destroy();

    const options = {
        ...commonOptions,
        chart: {
            ...commonOptions.chart,
            id: 'macd-chart',
            group: 'social',
            type: 'line',
            height: 250
        },
        series: [
            { name: 'MACD', data: macd },
            { name: 'Signal Line', data: signal }
        ],
        colors: ['#D4AF37', '#ffffff'],
        xaxis: { categories: labels }, // Show X labels here
        yaxis: {
            labels: { formatter: (val) => val.toFixed(2) }
        },
        stroke: { width: [2, 1] }
    };

    charts.macd = new ApexCharts(document.querySelector("#macdChart"), options);
    charts.macd.render();
}

function showLoader(show) {
    const overlay = document.getElementById('loading_overlay');
    if (show) overlay.classList.add('active');
    else overlay.classList.remove('active');
}
