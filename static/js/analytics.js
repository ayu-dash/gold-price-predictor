document.addEventListener('DOMContentLoaded', () => {
    // 1. Initial Data Fetch
    fetchYearlyStats();
    fetchModelMetrics();

    // Set current date
    const now = new Date();
    const dateEl = document.getElementById('current_date');
    if (dateEl) dateEl.textContent = now.toDateString();
});

async function fetchModelMetrics() {
    try {
        const response = await fetch('/api/model_metrics');
        const data = await response.json();

        if (data.error) {
            console.warn('Model metrics not found');
            return;
        }

        // Display Data
        document.getElementById('metric_mae').innerText = data.mae.toFixed(4);
        document.getElementById('metric_coverage').innerText = `${data.train_samples + data.test_samples} total`;
        document.getElementById('metric_timestamp').innerText = data.timestamp;

        // "Accuracy" can be ambiguous. We'll show "AVG ERROR MARGIN".
        // MAE is e.g. 0.008 (0.8%).
        const errorMargin = (data.mae * 100);
        document.getElementById('metric_accuracy').innerText = `±${errorMargin.toFixed(2)}%`;
        document.querySelector('.score-label').innerText = "Avg. Prediction Error";

        // Reliability Logic (Lower error is better)
        const reliabilityEl = document.getElementById('metric_reliability');
        let badgeText = 'Low';
        let badgeColor = 'text-red';

        if (errorMargin < 1.0) {
            badgeText = 'High';
            badgeColor = 'text-green';
        } else if (errorMargin < 2.0) {
            badgeText = 'Medium';
            badgeColor = 'text-gold';
        }

        reliabilityEl.innerText = badgeText;
        reliabilityEl.className = badgeColor;

    } catch (error) {
        console.error('Error fetching model metrics:', error);
    }
}

async function fetchYearlyStats() {
    showLoader(true);
    try {
        const response = await fetch('/api/yearly_stats');
        const data = await response.json();

        renderYearlyTable(data);
        renderROIChart(data);

    } catch (error) {
        console.error('Error fetching yearly stats:', error);
    } finally {
        showLoader(false);
    }
}

function renderYearlyTable(data) {
    const body = document.getElementById('yearly_body');
    body.innerHTML = '';

    data.reverse().forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${row.year}</strong></td>
            <td>Rp ${row.open.toLocaleString()}</td>
            <td>Rp ${row.close.toLocaleString()}</td>
            <td>Rp ${row.high.toLocaleString()}</td>
            <td>Rp ${row.low.toLocaleString()}</td>
            <td class="${row.roi_pct >= 0 ? 'text-gold' : 'text-red'}">
                ${row.roi_pct > 0 ? '+' : ''}${row.roi_pct}%
            </td>
        `;
        body.appendChild(tr);
    });
}

function renderROIChart(data) {
    // Return to original order for chart
    const sortedData = [...data].sort((a, b) => a.year - b.year);

    const ctx = document.getElementById('roiChart').getContext('2d');

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sortedData.map(d => d.year),
            datasets: [{
                label: 'Annual ROI (%)',
                data: sortedData.map(d => d.roi_pct),
                backgroundColor: sortedData.map(d => d.roi_pct >= 0 ? 'rgba(212, 175, 55, 0.6)' : 'rgba(255, 75, 75, 0.6)'),
                borderColor: sortedData.map(d => d.roi_pct >= 0 ? '#D4AF37' : '#FF4B4B'),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#888' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#888' }
                }
            }
        }
    });
}

function showLoader(show) {
    const overlay = document.getElementById('loading_overlay');
    if (!overlay) return;
    if (show) overlay.classList.add('active');
    else overlay.classList.remove('active');
}
