// Gruvbox Chart.js defaults
Chart.defaults.color = '#ebdbb2';
Chart.defaults.font.family = "'Space Grotesk', sans-serif";
Chart.defaults.plugins.tooltip.backgroundColor = '#1d2021';
Chart.defaults.plugins.tooltip.titleColor = '#fabd2f';
Chart.defaults.plugins.tooltip.bodyColor = '#ebdbb2';
Chart.defaults.plugins.tooltip.borderColor = '#000';
Chart.defaults.plugins.tooltip.borderWidth = 2;

// Fallback data in case JSON loading is blocked by local CORS policy
const fallbackData = {
    total_records: 150000,
    narrative_completion_rate: 34.2,
    avg_word_count: 147,
    model_accuracy: 82.5,
    trend_labels: ['2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06', '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12'],
    trend_values: [12040, 11800, 13450, 14200, 15100, 14900, 15800, 16100, 15900, 16500, 17200, 18000],
    top_product: "Credit reporting",
    top_products_labels: ["Money transfer", "Vehicle loan", "Payday loan", "Student loan", "Mortgage", "Credit card", "Debt collection", "Credit reporting"],
    top_products_values: [5000, 7200, 8900, 12000, 22000, 31000, 45000, 87000],
    top_topics: ["identity theft", "credit report", "late fee", "account closed", "customer service"]
};

// Function formatting
const formatNumber = (num) => num.toLocaleString();

async function initDashboard() {
    let data = fallbackData;
    
    // Attempt to load from JSON
    try {
        const response = await fetch('../dashboard_data.json');
        if (response.ok) {
            data = await response.json();
            console.log("Loaded active model payload!");
        }
    } catch (e) {
        console.warn("Using fallback static payload (CORS or file missing).", e);
    }
    
    // Inject Metrics
    document.getElementById('total-records').innerText = formatNumber(data.total_records);
    document.getElementById('completion-rate').innerText = data.narrative_completion_rate.toFixed(1) + '%';
    document.getElementById('avg-words').innerText = data.avg_word_count;
    document.getElementById('model-accuracy').innerText = data.model_accuracy.toFixed(1) + '%';

    // Inject Topics
    const topicList = document.getElementById('topic-list');
    data.top_topics.forEach(topic => {
        const li = document.createElement('li');
        li.innerText = topic.toUpperCase();
        topicList.appendChild(li);
    });

    // Trend Chart Builder
    const ctxTrend = document.getElementById('trendChart').getContext('2d');
    new Chart(ctxTrend, {
        type: 'line',
        data: {
            labels: data.trend_labels,
            datasets: [{
                label: 'Complaint Volumes',
                data: data.trend_values,
                borderColor: '#fabd2f', // yellow
                backgroundColor: 'rgba(250, 189, 47, 0.2)',
                borderWidth: 4,
                pointBackgroundColor: '#fb4934', // red points
                pointBorderColor: '#000',
                pointBorderWidth: 2,
                pointRadius: 6,
                fill: true,
                tension: 0.3 // smooth curves
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { grid: { color: '#3c3836' } },
                x: { grid: { color: '#3c3836' } }
            },
            plugins: { legend: { display: false } }
        }
    });

    // Product Bar Chart Builder
    const ctxProduct = document.getElementById('productChart').getContext('2d');
    new Chart(ctxProduct, {
        type: 'bar',
        data: {
            labels: data.top_products_labels,
            datasets: [{
                label: 'Volume',
                data: data.top_products_values,
                backgroundColor: '#d3869b', // purple
                borderColor: '#000',
                borderWidth: 3,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y', // horizontal bar
            scales: {
                y: { grid: { display: false } },
                x: { grid: { color: '#3c3836' } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

document.addEventListener('DOMContentLoaded', initDashboard);
