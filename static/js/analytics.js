document.addEventListener('DOMContentLoaded', () => {
    // 1. Initial Data Fetch
    fetchYearlyStats();
    fetchModelMetrics();

    // 2. Button Listeners
    const retrainBtn = document.getElementById('retrain_btn');
    if (retrainBtn) {
        retrainBtn.addEventListener('click', async () => {
            if (!confirm("Start model training? This process runs in the background and takes about 2-3 minutes.")) return;

            retrainBtn.disabled = true;
            retrainBtn.innerHTML = '<span class="spinner" style="width:12px; height:12px; border-width:2px; display:inline-block; margin-right:5px;"></span> Training...';

            try {
                const res = await fetch('/api/retrain', { method: 'POST' });
                const data = await res.json();
                alert(data.message);
            } catch (err) {
                alert("Failed to start training: " + err);
            } finally {
                // Re-enable after 60 seconds (prevents spamming while background process runs)
                setTimeout(() => {
                    retrainBtn.disabled = false;
                    retrainBtn.innerHTML = '<span class="material-symbols-outlined" style="font-size: 1rem;">sync</span> Update Model';
                }, 60000);
            }
        });
    }

    // Set current date
    const now = new Date();
    const dateEl = document.getElementById('current_date');
    if (dateEl) dateEl.textContent = now.toDateString();
});

async function fetchModelMetrics() {
    try {
        // Fetch Model Metrics (Training Data)
        const metricsRes = await fetch('/api/model_metrics');
        const metricsData = await metricsRes.json();

        // Fetch Real-World Signal Performance (Production Data)
        const signalsRes = await fetch('/api/signals_history');
        const signalsData = await signalsRes.json();

        // 1. Display Training Metrics
        if (metricsData && !metricsData.error) {
            document.getElementById('metric_coverage').innerText = `${metricsData.train_samples + metricsData.test_samples} total`;
            document.getElementById('metric_timestamp').innerText = metricsData.timestamp;

            // Handle new nested structure vs old flat structure
            if (metricsData.models) {
                const m = metricsData.models;
                document.getElementById('metric_mae_med').innerText = m.median.mae.toFixed(4);
                document.getElementById('metric_mae_low').innerText = m.low.mae.toFixed(4);
                document.getElementById('metric_mae_high').innerText = m.high.mae.toFixed(4);
                document.getElementById('metric_clf_acc').innerText = `${(m.classifier.accuracy * 100).toFixed(1)}%`;

                // Big Accuracy Score (Use Median MAE)
                const errorMargin = (m.median.mae * 100);
                document.getElementById('metric_accuracy').innerText = `±${errorMargin.toFixed(2)}%`;
            } else {
                // Fallback for old flat structure
                document.getElementById('metric_mae_med').innerText = metricsData.mae.toFixed(4);
                const errorMargin = (metricsData.mae * 100);
                document.getElementById('metric_accuracy').innerText = `±${errorMargin.toFixed(2)}%`;
            }

            document.querySelector('.score-label').innerText = "Model Training Error (Margin)";
        }

        // 2. Calculate Reliability based on REAL WIN RATE
        // "Reliability" should reflect accurate signals, not just training error.
        let correctCount = 0;
        let completedCount = 0;

        signalsData.forEach(sig => {
            const out = sig.Outcome;
            if (out === 'Correct') {
                correctCount++;
                completedCount++;
            } else if (out === 'Wrong') {
                completedCount++;
            }
        });

        const winRate = completedCount > 0 ? (correctCount / completedCount) * 100 : 0;

        const reliabilityEl = document.getElementById('metric_reliability');
        let badgeText = 'Low';
        let badgeColor = 'text-red';

        // Stricter Real-World Standards
        if (winRate >= 65) {
            badgeText = 'High';
            badgeColor = 'text-green';
        } else if (winRate >= 50) {
            badgeText = 'Medium';
            badgeColor = 'text-gold';
        }

        // Override text if no data
        if (completedCount === 0 && metricsData) {
            // Fallback to training reliability if no real signals yet
            reliabilityEl.innerText = "No Signals Yet";
            reliabilityEl.className = "text-muted";
        } else {
            reliabilityEl.innerText = `${badgeText} (${Math.round(winRate)}% WR)`;
            reliabilityEl.className = badgeColor;
        }

    } catch (error) {
        console.error('Error fetching metrics:', error);
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
