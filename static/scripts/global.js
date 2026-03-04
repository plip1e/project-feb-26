let accuracyLossChart = null;
let learningRateChart = null;

function toggleMobileNav() {
    const button = document.querySelector(".nav-toggle");
    const navList = document.getElementById("site-nav-list");

    if (!button || !navList) {
        return;
    }

    button.addEventListener("click", () => {
        const isExpanded = button.getAttribute("aria-expanded") === "true";
        button.setAttribute("aria-expanded", String(!isExpanded));
        navList.classList.toggle("is-open");
    });
}

function formatMetricName(metricKey) {
    return metricKey
        .replace(/_/g, " ")
        .replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatMetricValue(value) {
    if (typeof value === "number") {
        return Number.isInteger(value) ? String(value) : value.toFixed(4);
    }
    return String(value);
}

function getErrorObject(payload) {
    if (!payload) {
        return null;
    }

    if (payload.error && typeof payload.error === "object") {
        return payload.error;
    }

    if (typeof payload.error === "string") {
        return { type: "unknown_error", message: payload.error };
    }

    if (payload.type && payload.message) {
        return payload;
    }

    return null;
}

function getErrorMessage(payload, fallbackMessage) {
    const errorObject = getErrorObject(payload);
    if (!errorObject) {
        return fallbackMessage;
    }
    return errorObject.message || fallbackMessage;
}

function renderDiagnostic(contextLabel, payload) {
    const panel = document.getElementById("diagnostic-panel");
    const content = document.getElementById("diagnostic-content");
    if (!panel || !content) {
        return;
    }

    const error = getErrorObject(payload);
    if (!error) {
        panel.classList.add("is-hidden");
        content.textContent = "";
        return;
    }

    const lines = [
        `Context: ${contextLabel}`,
        `Type: ${error.type || "unknown_error"}`,
        `Message: ${error.message || "No message provided."}`,
    ];

    if (error.path) {
        lines.push(`Path: ${error.path}`);
    }
    if (error.details) {
        lines.push(`Details: ${error.details}`);
    }

    content.textContent = lines.join("\n");
    panel.classList.remove("is-hidden");
}

function setStatusChip(elementId, text, level) {
    const chip = document.getElementById(elementId);
    if (!chip) {
        return;
    }

    chip.textContent = text;
    chip.classList.remove("ok", "warn", "danger");
    if (level) {
        chip.classList.add(level);
    }
}

async function fetchJson(url, options = {}) {
    const response = await fetch(url, {
        headers: { Accept: "application/json" },
        ...options,
    });

    let payload = null;
    try {
        payload = await response.json();
    } catch (error) {
        payload = null;
    }

    return { response, payload };
}

function renderStatsErrorCard(containerId, title, message) {
    const container = document.getElementById(containerId);
    if (!container) {
        return;
    }

    container.innerHTML = `
        <article class="stat-card">
            <h3 class="status-danger">${title}</h3>
            <p>${message}</p>
        </article>
    `;
}

function renderHomeHighlights(stats) {
    const container = document.getElementById("home-highlights-grid");
    if (!container) {
        return;
    }

    const highlightKeys = [
        "accuracy",
        "val_accuracy",
        "loss",
        "val_loss",
        "best_val_accuracy",
        "best_val_loss",
        "epochs",
        "model_file_status",
    ];

    const cards = highlightKeys
        .filter((key) => key in stats)
        .map(
            (key) => `
                <article class="stat-card">
                    <h3>${formatMetricName(key)}</h3>
                    <p>${formatMetricValue(stats[key])}</p>
                </article>
            `
        )
        .join("");

    container.innerHTML = cards || `
        <article class="stat-card">
            <h3>No Highlights</h3>
            <p>No highlight-ready metrics found.</p>
        </article>
    `;
}

function renderAboutStatsCards(stats) {
    const container = document.getElementById("about-stats-cards");
    if (!container) {
        return;
    }

    const cardKeys = [
        "accuracy",
        "val_accuracy",
        "loss",
        "val_loss",
        "best_val_accuracy",
        "best_val_loss",
        "epochs",
        "model_file_status",
    ];

    const cards = cardKeys
        .filter((key) => key in stats)
        .map(
            (key) => `
                <article class="stat-card">
                    <h3>${formatMetricName(key)}</h3>
                    <p>${formatMetricValue(stats[key])}</p>
                </article>
            `
        )
        .join("");

    container.innerHTML = cards || `
        <article class="stat-card">
            <h3>No Stats</h3>
            <p>Stats payload is empty.</p>
        </article>
    `;
}

function renderAboutStatsTable(stats) {
    const tableBody = document.getElementById("about-stats-table-body");
    if (!tableBody) {
        return;
    }

    const rows = Object.entries(stats)
        .sort(([leftKey], [rightKey]) => leftKey.localeCompare(rightKey))
        .map(
            ([key, value]) => `
                <tr>
                    <td>${formatMetricName(key)}</td>
                    <td>${formatMetricValue(value)}</td>
                </tr>
            `
        )
        .join("");

    tableBody.innerHTML = rows || `
        <tr>
            <td colspan="2">No metrics found.</td>
        </tr>
    `;
}

function updateAboutSourceMessage(stats) {
    const messageElement = document.getElementById("about-stats-message");
    if (!messageElement) {
        return;
    }

    const source = stats.stats_source || "unknown source";
    messageElement.textContent = `Loaded from ${source} with model file metadata.`;
}

async function fetchAndRenderAboutStats() {
    const { response, payload } = await fetchJson("/api/model-stats");
    if (!response.ok) {
        renderStatsErrorCard("about-stats-cards", "Stats Unavailable", getErrorMessage(payload, "Could not load model stats."));
        renderAboutStatsTable({});
        const messageElement = document.getElementById("about-stats-message");
        if (messageElement) {
            messageElement.textContent = getErrorMessage(payload, "Stats request failed.");
        }
        return;
    }

    renderAboutStatsCards(payload);
    renderAboutStatsTable(payload);
    updateAboutSourceMessage(payload);
}

function setChartNote(elementId, text) {
    const note = document.getElementById(elementId);
    if (!note) {
        return;
    }
    note.textContent = text;
}

function buildEpochLabels(length) {
    return Array.from({ length }, (_, index) => `Epoch ${index + 1}`);
}

function normalizeSeriesLength(series, targetLength) {
    const normalized = new Array(targetLength).fill(null);
    for (let index = 0; index < series.length && index < targetLength; index += 1) {
        normalized[index] = series[index];
    }
    return normalized;
}

function renderAccuracyLossChart(history) {
    const accuracySeries = Array.isArray(history.accuracy) ? history.accuracy : [];
    const valAccuracySeries = Array.isArray(history.val_accuracy) ? history.val_accuracy : [];
    const lossSeries = Array.isArray(history.loss) ? history.loss : [];
    const valLossSeries = Array.isArray(history.val_loss) ? history.val_loss : [];
    const maxLength = Math.max(
        accuracySeries.length,
        valAccuracySeries.length,
        lossSeries.length,
        valLossSeries.length
    );

    if (accuracyLossChart) {
        accuracyLossChart.destroy();
        accuracyLossChart = null;
    }

    if (maxLength === 0) {
        setChartNote("accuracy-loss-note", "No accuracy/loss series available in history.");
        return;
    }

    const chartCanvas = document.getElementById("accuracy-loss-chart");
    if (!chartCanvas || typeof Chart === "undefined") {
        setChartNote("accuracy-loss-note", "Chart.js is unavailable.");
        return;
    }

    const labels = buildEpochLabels(maxLength);
    const datasets = [];

    if (accuracySeries.length) {
        datasets.push({
            label: "Accuracy",
            data: normalizeSeriesLength(accuracySeries, maxLength),
            borderColor: "#5de26b",
            backgroundColor: "transparent",
            yAxisID: "yAccuracy",
            tension: 0.2,
        });
    }
    if (valAccuracySeries.length) {
        datasets.push({
            label: "Val Accuracy",
            data: normalizeSeriesLength(valAccuracySeries, maxLength),
            borderColor: "#2aa8ff",
            backgroundColor: "transparent",
            yAxisID: "yAccuracy",
            tension: 0.2,
        });
    }
    if (lossSeries.length) {
        datasets.push({
            label: "Loss",
            data: normalizeSeriesLength(lossSeries, maxLength),
            borderColor: "#ffc857",
            backgroundColor: "transparent",
            yAxisID: "yLoss",
            tension: 0.2,
        });
    }
    if (valLossSeries.length) {
        datasets.push({
            label: "Val Loss",
            data: normalizeSeriesLength(valLossSeries, maxLength),
            borderColor: "#ff6b86",
            backgroundColor: "transparent",
            yAxisID: "yLoss",
            tension: 0.2,
        });
    }

    accuracyLossChart = new Chart(chartCanvas, {
        type: "line",
        data: {
            labels,
            datasets,
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: "nearest",
                intersect: false,
            },
            plugins: {
                legend: {
                    labels: {
                        color: "#f4f7ff",
                    },
                },
            },
            scales: {
                x: {
                    ticks: { color: "#adb7c9" },
                    grid: { color: "rgba(173, 183, 201, 0.15)" },
                },
                yAccuracy: {
                    type: "linear",
                    position: "left",
                    ticks: { color: "#5de26b" },
                    grid: { color: "rgba(93, 226, 107, 0.15)" },
                },
                yLoss: {
                    type: "linear",
                    position: "right",
                    ticks: { color: "#ffc857" },
                    grid: { drawOnChartArea: false },
                },
            },
        },
    });

    setChartNote("accuracy-loss-note", "Loaded from /api/model-history.");
}

function renderLearningRateChart(history) {
    const learningRateSeries = Array.isArray(history.learning_rate) ? history.learning_rate : [];

    if (learningRateChart) {
        learningRateChart.destroy();
        learningRateChart = null;
    }

    if (!learningRateSeries.length) {
        setChartNote("learning-rate-note", "No learning_rate series found in history.");
        return;
    }

    const chartCanvas = document.getElementById("learning-rate-chart");
    if (!chartCanvas || typeof Chart === "undefined") {
        setChartNote("learning-rate-note", "Chart.js is unavailable.");
        return;
    }

    learningRateChart = new Chart(chartCanvas, {
        type: "line",
        data: {
            labels: buildEpochLabels(learningRateSeries.length),
            datasets: [
                {
                    label: "Learning Rate",
                    data: learningRateSeries,
                    borderColor: "#76c7ff",
                    backgroundColor: "transparent",
                    tension: 0.2,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: "#f4f7ff",
                    },
                },
            },
            scales: {
                x: {
                    ticks: { color: "#adb7c9" },
                    grid: { color: "rgba(173, 183, 201, 0.15)" },
                },
                y: {
                    ticks: { color: "#76c7ff" },
                    grid: { color: "rgba(118, 199, 255, 0.15)" },
                },
            },
        },
    });

    setChartNote("learning-rate-note", "Learning rate series loaded.");
}

async function fetchAndRenderHistoryCharts() {
    if (typeof Chart === "undefined") {
        const chartError = {
            error: {
                type: "chart_library_missing",
                message: "Chart.js did not load. Check internet/CDN access.",
            },
        };
        setChartNote("accuracy-loss-note", chartError.error.message);
        setChartNote("learning-rate-note", chartError.error.message);
        setStatusChip("history-status-badge", "History status: chart library missing", "warn");
        renderDiagnostic("History Charts", chartError);
        return;
    }

    const { response, payload } = await fetchJson("/api/model-history");
    if (!response.ok) {
        const message = getErrorMessage(payload, "Could not load model history.");
        setChartNote("accuracy-loss-note", message);
        setChartNote("learning-rate-note", message);
        setStatusChip("history-status-badge", "History status: unavailable", "danger");
        renderDiagnostic("Model History API", payload);
        return;
    }

    renderAccuracyLossChart(payload);
    renderLearningRateChart(payload);
    setStatusChip("history-status-badge", "History status: loaded", "ok");
}

async function fetchAndRenderHomeHighlights() {
    const { response, payload } = await fetchJson("/api/model-stats");
    if (!response.ok) {
        renderStatsErrorCard("home-highlights-grid", "Highlights Unavailable", getErrorMessage(payload, "Could not load model stats."));
        setStatusChip("model-status-badge", "Model status: unavailable", "danger");
        renderDiagnostic("Model Stats API", payload);
        return;
    }

    renderHomeHighlights(payload);

    const modelStatus = String(payload.model_file_status || "unknown").toLowerCase();
    if (modelStatus === "available") {
        setStatusChip("model-status-badge", "Model status: available", "ok");
    } else if (modelStatus === "missing") {
        setStatusChip("model-status-badge", "Model status: missing", "danger");
    } else {
        setStatusChip("model-status-badge", `Model status: ${modelStatus}`, "warn");
    }
}

function createSampleCard(sample) {
    const actualLabel = sample.actual_label ? `<p><strong>Actual:</strong> ${sample.actual_label}</p>` : "";
    const notes = sample.notes ? `<p class="status-warn">${sample.notes}</p>` : "";
    const confidenceValue = typeof sample.confidence === "number"
        ? `${(sample.confidence * 100).toFixed(1)}%`
        : String(sample.confidence || "N/A");

    return `
        <article class="sample-card">
            <img
                src="${sample.image_url}"
                alt="Prediction sample: ${sample.predicted_label || "Unknown"}"
                loading="lazy"
                onerror="this.outerHTML='<div class=&quot;img-fallback&quot;>Image unavailable</div>';"
            >
            <div class="sample-meta">
                <p><strong>Predicted:</strong> ${sample.predicted_label || "Unknown"}</p>
                <p><strong>Confidence:</strong> ${confidenceValue}</p>
                ${actualLabel}
                ${notes}
            </div>
        </article>
    `;
}

function renderSampleResults(samples) {
    const grid = document.getElementById("sample-results-grid");
    if (!grid) {
        return;
    }

    if (!samples.length) {
        grid.innerHTML = `
            <article class="sample-card">
                <div class="img-fallback">No data</div>
                <div class="sample-meta">
                    <p>Sample results are currently empty.</p>
                </div>
            </article>
        `;
        return;
    }

    grid.innerHTML = samples.map(createSampleCard).join("");
}

function renderSampleError(message) {
    const grid = document.getElementById("sample-results-grid");
    if (!grid) {
        return;
    }

    grid.innerHTML = `
        <article class="sample-card">
            <div class="img-fallback">Data unavailable</div>
            <div class="sample-meta">
                <p class="status-danger">${message}</p>
            </div>
        </article>
    `;
}

async function fetchAndRenderSamples() {
    const { response, payload } = await fetchJson("/api/sample-results");
    if (!response.ok) {
        renderSampleError(getErrorMessage(payload, "Could not load sample results."));
        renderDiagnostic("Sample Predictions API", payload);
        return;
    }

    renderSampleResults(Array.isArray(payload) ? payload : []);
}

function renderPredictResult(prediction) {
    const card = document.getElementById("predict-result-card");
    if (!card) {
        return;
    }

    const confidenceValue = typeof prediction.confidence === "number"
        ? `${(prediction.confidence * 100).toFixed(1)}%`
        : String(prediction.confidence || "N/A");

    card.innerHTML = `
        <h3>Prediction Result</h3>
        <p><strong>Label:</strong> ${prediction.predicted_label || "Unknown"}</p>
        <p><strong>Class Index:</strong> ${prediction.class_index ?? "N/A"}</p>
        <p><strong>Confidence:</strong> ${confidenceValue}</p>
        <p><strong>File:</strong> ${prediction.uploaded_filename || "Uploaded image"}</p>
    `;
}

function renderPredictError(message) {
    const card = document.getElementById("predict-result-card");
    if (!card) {
        return;
    }

    card.innerHTML = `
        <h3 class="status-danger">Prediction Failed</h3>
        <p>${message}</p>
    `;
}

function setupPredictUploadForm() {
    const form = document.getElementById("predict-upload-form");
    const button = document.getElementById("predict-submit-btn");
    const fileInput = document.getElementById("predict-image-input");
    if (!form || !button || !fileInput) {
        return;
    }

    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        if (!fileInput.files || !fileInput.files.length) {
            const localError = {
                error: {
                    type: "missing_upload_field",
                    message: "Please select a JPEG file before submitting.",
                },
            };
            renderPredictError(localError.error.message);
            renderDiagnostic("Custom JPEG Prediction", localError);
            return;
        }

        button.disabled = true;
        button.textContent = "Predicting...";

        try {
            const formData = new FormData(form);
            const { response, payload } = await fetchJson("/api/predict-image", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                renderPredictError(getErrorMessage(payload, "Prediction request failed."));
                renderDiagnostic("Custom JPEG Prediction", payload);
                return;
            }

            renderPredictResult(payload);
        } catch (error) {
            renderPredictError("Network error while uploading the image.");
            renderDiagnostic("Custom JPEG Prediction", {
                error: {
                    type: "network_error",
                    message: "Network error while uploading the image.",
                    details: String(error),
                },
            });
        } finally {
            button.disabled = false;
            button.textContent = "Run Prediction";
        }
    });
}

async function fetchAndRenderHomePageData() {
    await Promise.all([
        fetchAndRenderHomeHighlights(),
        fetchAndRenderHistoryCharts(),
        fetchAndRenderSamples(),
    ]);
}

function setupHomeRefreshButton() {
    const button = document.getElementById("refresh-home-data-btn");
    if (!button) {
        return;
    }

    button.addEventListener("click", async () => {
        button.disabled = true;
        button.textContent = "Refreshing...";
        await fetchAndRenderHomePageData();
        button.disabled = false;
        button.textContent = "Refresh Home Data";
    });
}

function setupAboutRefreshButton() {
    const button = document.getElementById("refresh-about-stats-btn");
    if (!button) {
        return;
    }

    button.addEventListener("click", async () => {
        button.disabled = true;
        button.textContent = "Refreshing...";
        await fetchAndRenderAboutStats();
        button.disabled = false;
        button.textContent = "Refresh Stats";
    });
}

function initHomePage() {
    if (!document.getElementById("home-presentation")) {
        return;
    }

    setupHomeRefreshButton();
    setupPredictUploadForm();
    fetchAndRenderHomePageData();
}

function initAboutPage() {
    if (!document.getElementById("about-stats-section")) {
        return;
    }

    setupAboutRefreshButton();
    fetchAndRenderAboutStats();
}

function initApp() {
    toggleMobileNav();
    initHomePage();
    initAboutPage();
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initApp, { once: true });
} else {
    initApp();
}
