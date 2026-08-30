async function loadDashboard() {
    const container = document.getElementById('metrics-container');
    try {
        const res = await fetchJSON('/api/metrics');
        if (!res.success) {
            container.innerHTML = `<p class="error">Error: ${res.error}</p>`;
            return;
        }
        const { classifier_metrics: m, tier_counts: t, audit_summary: a } = res.data;

        container.innerHTML = `
            <div class="metrics-grid">
                <div class="metric-card"><span class="metric-value">${(m.precision * 100).toFixed(1)}%</span><span class="metric-label">Precision</span></div>
                <div class="metric-card"><span class="metric-value">${(m.recall * 100).toFixed(1)}%</span><span class="metric-label">Recall</span></div>
                <div class="metric-card"><span class="metric-value">${(m.f1 * 100).toFixed(1)}%</span><span class="metric-label">F1 Score</span></div>
                <div class="metric-card"><span class="metric-value">${m.total_chargebacks}</span><span class="metric-label">Chargebacks (of ${m.total_transactions})</span></div>
            </div>
            <div class="chart-row">
                <div class="chart-box"><canvas id="tierChart"></canvas></div>
                <div class="chart-box"><canvas id="auditChart"></canvas></div>
            </div>
        `;

        new Chart(document.getElementById('tierChart'), {
            type: 'doughnut',
            data: {
                labels: Object.keys(t).map(k => k.replace(/_/g, ' ')),
                datasets: [{ data: Object.values(t), backgroundColor: ['#e05252', '#e0a852', '#52a8e0'] }]
            },
            options: { plugins: { title: { display: true, text: 'Risk Tier Distribution' } } }
        });

        const statusCounts = {};
        a.by_status.forEach(s => statusCounts[s.action_status] = s.count);
        new Chart(document.getElementById('auditChart'), {
            type: 'bar',
            data: {
                labels: Object.keys(statusCounts),
                datasets: [{ label: 'Actions Logged', data: Object.values(statusCounts), backgroundColor: '#5271e0' }]
            },
            options: { plugins: { title: { display: true, text: 'Audit Log — Actions Taken' } } }
        });
    } catch (e) {
        container.innerHTML = `<p class="error">Failed to load: ${e.message}</p>`;
    }
}
loadDashboard();