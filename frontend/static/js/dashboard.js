let rawScores = [];
let tierChartInstance = null;
let auditChartInstance = null;
let avgReviewCost = 50;

function computeMetricsAtThreshold(threshold) {
    let tp = 0, fp = 0, fn = 0, tn = 0;
    let moneyProtected = 0, moneyMissed = 0;

    rawScores.forEach(row => {
        const predicted = row.proba >= threshold ? 1 : 0;
        const actual = row.true_label;

        if (predicted === 1 && actual === 1) { tp++; moneyProtected += row.Amount; }
        else if (predicted === 1 && actual === 0) { fp++; }
        else if (predicted === 0 && actual === 1) { fn++; moneyMissed += row.Amount; }
        else { tn++; }
    });

    const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
    const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
    const f1 = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    const reviewCost = fp * avgReviewCost;
    const netImpact = moneyProtected - reviewCost;

    return { precision, recall, f1, tp, fp, fn, tn, moneyProtected, moneyMissed, reviewCost, netImpact };
}

function renderMetrics(threshold) {
    const stats = computeMetricsAtThreshold(threshold);
    const impactClass = stats.netImpact >= 0 ? 'impact-positive' : 'impact-negative';

    document.getElementById('m-precision').textContent = (stats.precision * 100).toFixed(1) + '%';
    document.getElementById('m-recall').textContent = (stats.recall * 100).toFixed(1) + '%';
    document.getElementById('m-f1').textContent = (stats.f1 * 100).toFixed(1) + '%';

    document.getElementById('m-protected').textContent = 'Rs ' + stats.moneyProtected.toLocaleString(undefined, {maximumFractionDigits: 0});
    document.getElementById('m-review-cost').textContent = 'Rs ' + stats.reviewCost.toLocaleString(undefined, {maximumFractionDigits: 0});
    document.getElementById('m-review-label').textContent = `Review cost (${stats.fp} false alarms)`;
    document.getElementById('m-missed').textContent = 'Rs ' + stats.moneyMissed.toLocaleString(undefined, {maximumFractionDigits: 0});
    document.getElementById('m-net').textContent = 'Rs ' + stats.netImpact.toLocaleString(undefined, {maximumFractionDigits: 0});
    document.getElementById('m-net').className = 'metric-value ' + impactClass;

    document.getElementById('threshold-value').textContent = threshold.toFixed(2);
}

async function loadDashboard() {
    const container = document.getElementById('metrics-container');
    try {
        const [metricsRes, scoresRes] = await Promise.all([
            fetchJSON('/api/metrics'),
            fetchJSON('/api/raw-scores'),
        ]);

        if (!metricsRes.success || !scoresRes.success) {
            container.innerHTML = `<p class="error">Error loading dashboard data.</p>`;
            return;
        }

        rawScores = scoresRes.data;
        avgReviewCost = metricsRes.data.business_impact.avg_review_cost_assumption;
        const { tier_counts: t, audit_summary: a } = metricsRes.data;

        container.innerHTML = `
            <div class="threshold-panel">
                <div class="threshold-header">
                    <label for="threshold-slider">Decision threshold</label>
                    <span id="threshold-value" class="threshold-value">0.50</span>
                </div>
                <input type="range" id="threshold-slider" min="0.1" max="0.9" step="0.01" value="0.5">
                <p class="threshold-hint">
                    Drag to see how precision, recall, and business impact trade off at
                    different decision thresholds.
                </p>
            </div>

            <div class="metrics-grid">
                <div class="metric-card"><span class="metric-value" id="m-precision">—</span><span class="metric-label">Precision</span></div>
                <div class="metric-card"><span class="metric-value" id="m-recall">—</span><span class="metric-label">Recall</span></div>
                <div class="metric-card"><span class="metric-value" id="m-f1">—</span><span class="metric-label">F1 Score</span></div>
                <div class="metric-card"><span class="metric-value">${metricsRes.data.classifier_metrics.total_chargebacks}</span><span class="metric-label">Chargebacks (of ${metricsRes.data.classifier_metrics.total_transactions})</span></div>
            </div>

            <h2 class="section-title">Business impact</h2>
            <p class="impact-note">
                Assumes an illustrative Rs ${avgReviewCost} cost per unnecessary human review.
                Real deployments should calibrate this to actual analyst time.
            </p>
            <div class="metrics-grid">
                <div class="metric-card">
                    <span class="metric-value impact-positive" id="m-protected">—</span>
                    <span class="metric-label" id="m-protected-label">Chargeback value caught</span>
                </div>
                <div class="metric-card">
                    <span class="metric-value" id="m-review-cost">—</span>
                    <span class="metric-label" id="m-review-label">Review cost</span>
                </div>
                <div class="metric-card">
                    <span class="metric-value impact-negative" id="m-missed">—</span>
                    <span class="metric-label">Chargeback value missed</span>
                </div>
                <div class="metric-card">
                    <span class="metric-value" id="m-net">—</span>
                    <span class="metric-label">Net impact vs no-model baseline</span>
                </div>
            </div>

            <div class="chart-row">
                <div class="chart-box"><canvas id="tierChart"></canvas></div>
                <div class="chart-box"><canvas id="auditChart"></canvas></div>
            </div>
        `;

        renderMetrics(0.5);

        document.getElementById('threshold-slider').addEventListener('input', (e) => {
            renderMetrics(parseFloat(e.target.value));
        });

        const tierColorMap = {
            'high_risk_auto_flag': '#C93A52',
            'medium_risk_human_review': '#B87A1C',
            'low_risk_log_only': '#1B8F72',
        };
        const tierKeys = Object.keys(t);

        tierChartInstance = new Chart(document.getElementById('tierChart'), {
            type: 'doughnut',
            data: {
                labels: tierKeys.map(k => k.replace(/_/g, ' ')),
                datasets: [{
                    data: tierKeys.map(k => t[k]),
                    backgroundColor: tierKeys.map(k => tierColorMap[k] || '#9C96A8')
                }]
            },
            options: { plugins: { title: { display: true, text: 'Risk Tier Distribution' } } }
        });

        const statusCounts = {};
        a.by_status.forEach(s => statusCounts[s.action_status] = s.count);
        auditChartInstance = new Chart(document.getElementById('auditChart'), {
            type: 'bar',
            data: {
                labels: Object.keys(statusCounts),
                datasets: [{ label: 'Actions Logged', data: Object.values(statusCounts), backgroundColor: '#5B4FE8' }]
            },
            options: { plugins: { title: { display: true, text: 'Audit Log — Actions Taken' } } }
        });
    } catch (e) {
        container.innerHTML = `<p class="error">Failed to load: ${e.message}</p>`;
    }
}
loadDashboard();