document.addEventListener('DOMContentLoaded', () => {
    fetchSignalDetails();

    const now = new Date();
    document.getElementById('current_date').textContent = now.toDateString();
});

async function fetchSignalDetails() {
    showLoader(true);
    try {
        const response = await fetch('/api/prediction');
        const data = await response.json();

        // Update Signal
        const mainSignal = document.getElementById('signal_main');
        mainSignal.textContent = data.recommendation;
        mainSignal.classList.remove('text-gold', 'text-red');
        mainSignal.classList.add(data.recommendation === 'SELL' ? 'text-red' : 'text-gold');

        // Target Price
        const targetEl = document.getElementById('signal_target');
        if (targetEl) {
            if (data.recommendation === 'BUY') {
                targetEl.textContent = `Entry: Rp ${data.current_price_idr_gram.toLocaleString()} | Sell Target: Rp ${data.predicted_price_idr_gram.toLocaleString()}`;
            } else if (data.recommendation === 'SELL') {
                targetEl.textContent = `Exit Price: Rp ${data.current_price_idr_gram.toLocaleString()} | Stop Loss: Rp ${(data.current_price_idr_gram * 0.98).toLocaleString()}`;
            } else {
                targetEl.textContent = `Neutral Target: Rp ${data.predicted_price_idr_gram.toLocaleString()}`;
            }
        }

        // Mock signal logic based on data
        const logicEl = document.getElementById('signal_logic');
        if (data.recommendation === 'BUY') {
            logicEl.textContent = "AI detects upward momentum with positive news sentiment. Lower RSI suggests a potential entry point.";
        } else if (data.recommendation === 'SELL') {
            logicEl.textContent = "Overbought conditions detected or negative sentiment prevails. Protect gains by considering exit.";
        } else {
            logicEl.textContent = "Market is in sideways consolidation. Volatility is low, recommending caution.";
        }

        // Factors
        // Factors
        document.getElementById('momentum_status').textContent = data.change_pct > 0 ? "BULLISH" : "BEARISH";
        // 'geopol_risk' is updated in calculateRiskMatrix() with better logic

        // Risk Matrix Calculation
        calculateRiskMatrix(data);

        // Render History (Mocked for UI placeholder)
        renderHistory(data);

    } catch (error) {
        console.error('Error fetching signals:', error);
    } finally {
        showLoader(false);
    }
}

function calculateRiskMatrix(data) {
    // USD Strength Risk (Mock logic based on recommendation)
    const usdEl = document.getElementById('risk_usd');
    if (data.recommendation === 'SELL') {
        usdEl.textContent = 'High';
        usdEl.className = 'risk-indicator risk-high';
    } else {
        usdEl.textContent = 'Low';
        usdEl.className = 'risk-indicator risk-low';
    }

    // Geopolitical Risk (Based on sentiment score)
    const geoEl = document.getElementById('risk_geo');
    const geoCard = document.getElementById('geopol_risk');

    // Unified Logic
    // Sentiment < -0.05 (Negative News) -> HIGH Risk
    // Sentiment < 0.05 (Neutral) -> MED Risk
    // Sentiment >= 0.05 (Positive News) -> LOW Risk

    if (data.sentiment_score < -0.05) {
        geoEl.textContent = 'High';
        geoEl.className = 'risk-indicator risk-high';

        geoCard.textContent = "ELEVATED";
        geoCard.className = "signal-value text-red";

    } else if (data.sentiment_score < 0.05) {
        geoEl.textContent = 'Med';
        geoEl.className = 'risk-indicator risk-med';

        geoCard.textContent = "MODERATE";
        geoCard.className = "signal-value text-gold";

    } else {
        geoEl.textContent = 'Low';
        geoEl.className = 'risk-indicator risk-low';

        geoCard.textContent = "LOW";
        geoCard.className = "signal-value text-green";
    }

    // Inflation Hedge (Dynamic based on price trend)
    const infEl = document.getElementById('risk_inf');
    if (data.change_pct > 2) {
        infEl.textContent = 'Strong';
        infEl.className = 'risk-indicator risk-low'; // Lower risk for gold holders
    } else {
        infEl.textContent = 'Neutral';
        infEl.className = 'risk-indicator risk-med';
    }
}

async function renderHistory() {
    const body = document.getElementById('signal_history_body');
    body.innerHTML = '<tr><td colspan="4" class="text-center">Loading historical performance...</td></tr>';

    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (!data.prices || data.prices.length < 5) return;

        body.innerHTML = '';

        // Take last 5 days excluding today
        // We calculate "Outcome" based on price change to the NEXT day
        const days = data.dates.slice(-6, -1).reverse();
        const prices = data.prices.slice(-6, -1).reverse();
        const nextPrices = data.prices.slice(-5).reverse(); // Shifted by 1 for outcome calc

        // Exchange rate (approximate for history or fetch if available, using latest for simplicity of display IDR)
        // Ideally we need historical IDR but the API returns lists. 
        // data.idr_rate exists!
        const rates = data.idr_rate.slice(-6, -1).reverse();

        days.forEach((date, i) => {
            const priceUSD = prices[i];
            const nextPriceUSD = nextPrices[i]; // The price on the FOLLOWING day
            const rate = rates[i];

            const priceIDR = (priceUSD * rate) / 31.1035;

            // Calculate hypothetical outcome (Did price go up or down?)
            const change = ((nextPriceUSD - priceUSD) / priceUSD) * 100;
            const signal = change > 0 ? "BUY" : "SELL"; // Hindsight signal
            const outcomeColor = change > 0 ? "text-green" : "text-red";
            const outcomeText = change > 0 ? `Profit +${change.toFixed(2)}%` : `Loss ${change.toFixed(2)}%`;
            const status = "CLOSED";

            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${date}</td>
                <td>Rp ${priceIDR.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                <td class="${signal === 'BUY' ? 'text-gold' : 'text-red'}">${signal} (Sim)</td>
                <td class="${outcomeColor}">${status}: ${outcomeText}</td>
            `;
            body.appendChild(tr);
        });

    } catch (e) {
        console.error(e);
        body.innerHTML = '<tr><td colspan="4" class="text-center">Error loading history.</td></tr>';
    }
}


function showLoader(show) {
    const overlay = document.getElementById('loading_overlay');
    if (show) overlay.classList.add('active');
    else overlay.classList.remove('active');
}
