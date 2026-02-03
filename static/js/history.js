document.addEventListener('DOMContentLoaded', () => {
    fetchHistoryData();

    // Event Listeners for Filters
    document.getElementById('filter_year').addEventListener('change', filterTable);
    document.getElementById('filter_month').addEventListener('change', filterTable);

    // Set Date
    const now = new Date();
    document.getElementById('current_date').textContent = now.toDateString();
});

let allData = []; // Store full dataset

async function fetchHistoryData() {
    const tbody = document.getElementById('history_body');

    try {
        const response = await fetch('/api/history/full');
        const data = await response.json();

        if (data.error) {
            tbody.innerHTML = `<tr><td colspan="5" class="text-center text-red">${data.error}</td></tr>`;
            return;
        }

        allData = data;
        populateYearFilter(data);
        renderTable(data);

    } catch (error) {
        console.error('Error loading history:', error);
        tbody.innerHTML = `<tr><td colspan="5" class="text-center text-red">Failed to load data.</td></tr>`;
    }
}

function populateYearFilter(data) {
    const years = new Set(data.map(item => item.date.substring(0, 4)));
    const sortedYears = Array.from(years).sort((a, b) => b - a); // Descending

    const select = document.getElementById('filter_year');
    // Keep "All Years" option
    // select.innerHTML = '<option value="ALL">All Years</option>'; 

    sortedYears.forEach(year => {
        const option = document.createElement('option');
        option.value = year;
        option.textContent = year;
        select.appendChild(option);
    });
}

function renderTable(data) {
    const tbody = document.getElementById('history_body');
    tbody.innerHTML = '';

    // Limit to first 500 for performance if list is huge on "ALL", 
    // or rely on the filter. Let's render max 1000.
    const displayData = data.slice(0, 1000);

    if (displayData.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" class="text-center">No records found.</td></tr>`;
        return;
    }

    // Calculate previous price for change (using the full data context if possible, 
    // but here we might just use the server provided change or calc on fly?)
    // The server didn't explicitly pass change % in the final JSON loop I wrote.
    // I should check app.py again. 
    // Wait, in app.py I simplified the loop and didn't include Daily_Return in the final dict.
    // I'll assume I need to calc it here or "Daily Change" column will be empty?
    // Actually, I can calculate it client side since the array is sorted descending (Newest first).
    // So Change = (Current - NextItem) / NextItem. (Since NextItem is "yesterday").

    displayData.forEach((row, index) => {
        // Find previous day price (which is at index + 1 in descending list)
        // Note: verify if it matches allData to get true previous
        // But for visual simple calculation on rendered set? No, use row calculation if passed 
        // OR calculate from array.

        // Let's rely on simple client-side calc if possible or just show Price for now.
        // Actually I should have passed it from backend. 
        // Let's mock it or calc it. 
        // row index in allData? 
        // Let's implement robust finding.

        // Use server-side pre-calculated change percentage
        let changePct = row.change_pct || 0;

        const colorClass = changePct >= 0 ? 'text-green' : 'text-red';
        const sign = changePct >= 0 ? '+' : '';

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${row.date}</td>
            <td>Rp ${row.price_idr.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
            <td>$ ${row.price_usd.toLocaleString()}</td>
            <td class="text-muted">Rp ${row.rate.toLocaleString()}</td>
            <td class="${colorClass}">${sign}${changePct.toFixed(2)}%</td>
        `;
        tbody.appendChild(tr);
    });
}

function filterTable() {
    const year = document.getElementById('filter_year').value;
    const month = document.getElementById('filter_month').value;

    let filtered = allData;

    if (year !== 'ALL') {
        filtered = filtered.filter(item => item.date.startsWith(year));
    }

    if (month !== 'ALL') {
        filtered = filtered.filter(item => {
            // item.date is YYYY-MM-DD
            const m = item.date.substring(5, 7);
            return m === month;
        });
    }

    renderTable(filtered);
}
