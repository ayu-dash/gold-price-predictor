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

        // "Accuracy" is a relative business metric here (1 - MAE/AveragePrice)
        // Simplified estimate for UI:
        const accuracy = Math.max(0, (1 - (data.mae / 0.02)) * 100);
        document.getElementById('metric_accuracy').innerText = `${accuracy.toFixed(1)}%`;

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
