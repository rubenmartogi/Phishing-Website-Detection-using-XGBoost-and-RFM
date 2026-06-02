// Convert markdown bold (**text**) to HTML bold (<b>text</b>)
function markdownToHtml(str) {
    if (!str) return "";
    // Replace **bold** with <b>bold</b>
    return String(str).replace(/\*\*(.*?)\*\*/g, '<b>$1</b>');
}

function clamp01(v) {
    const n = Number(v);
    if (Number.isNaN(n)) return 0;
    return Math.min(1, Math.max(0, n));
}

function escapeHtml(str) {
    return String(str ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function backendToEnhancedResult(data, url) {
    const pPhish = clamp01(data.final_phishing_prob);
    const isPhishing = Number(data.final_label) === 1;

    const category = isPhishing ? "phishing" : "safe";
    const mode = data.decision_mode || "hybrid_prefilter";
    const source = data.decision_source || "-";
    let displayModelName = data.model_name || "Model Prediction";

    if (mode === "rf_only") {
        displayModelName = "Random Forest Only";
    } else if (mode === "xgb_only") {
        displayModelName = "XGBoost Only";
    } else if (mode === "ml_stacking_only") {
        displayModelName = "RF + XGB + Stacking";
    } else if (mode === "hybrid_prefilter") {
        if (source.includes("rule")) {
            displayModelName = "Rule-Based Prefilter";
        } else if (source.includes("stacking")) {
            displayModelName = "RF + XGB + Stacking";
        } else {
            displayModelName = "Rule-Based + ML + Stacking";
        }
    }

    const threshold = clamp01(
        Number.isFinite(Number(data.final_threshold)) ? Number(data.final_threshold) : 0.6
    );

    // Extract top influential features from explanation_detailed
    const topFeatures = (data.explanation_detailed && data.explanation_detailed.top_influential_features) 
        ? data.explanation_detailed.top_influential_features.slice(0, 5) 
        : [];

    return {
        result: isPhishing ? "PHISHING" : "BENIGN",
        algorithm: displayModelName,
        decisionMode: mode,
        decisionSource: source,
        category,
        decisionScore: pPhish,
        final_threshold: threshold,
        url,
        // Convert markdown to HTML for explanation
        explanation: markdownToHtml(data.llm_reasoning || "-"),
        topFeatures: topFeatures
    };
}

function displayEnhancedResult(result, url) {
    const resultDiv = document.getElementById("result");
    if (!resultDiv) return;

    if (result.category === "error") {
        resultDiv.innerHTML = `
            <div class="result error">
                <div class="result-title">ERROR</div>
                <div class="result-meta">${escapeHtml(result.result || "Unknown error")}</div>
            </div>
        `;
        return;
    }

    const resultClass = result.category === "phishing" ? "phishing" : "safe";
    const threshold = clamp01(result.final_threshold);
    const score = clamp01(result.decisionScore);

    const benignRangeText = score < threshold ? score.toFixed(2) : "-";
    const phishingRangeText = score >= threshold ? score.toFixed(2) : "-";

    // Build top features HTML
    let topFeaturesHtml = "";
    if (result.topFeatures && result.topFeatures.length > 0) {
        topFeaturesHtml = `
            <div class="mini-card features-card">
                <span>Fitur </span>
                <div class="features-list">
        `;
        result.topFeatures.forEach((feat, idx) => {
            const impactEmoji = feat.impact === "PHISHING" ? "🔴" : (feat.impact === "BENIGN" ? "🟢" : "⚪");
            topFeaturesHtml += `
                    <div class="feature-item">
                        <div class="feature-name">${idx + 1}. ${escapeHtml(feat.name)}</div>
                        <div class="feature-details">${impactEmoji} ${feat.impact} | ${escapeHtml(feat.reason)}</div>
                    </div>
            `;
        });
        topFeaturesHtml += `
                </div>
            </div>
        `;
    }

    const phishingProb = score; // 0-1

    const riskBarHtml = `
    <div class="risk-bar-container">
        <div class="risk-bar-labels">
        <span class="risk-label-left">Low Risk</span>
        <span class="risk-label-right">Hard Risk</span>
        </div>
        <div class="risk-bar-track">
        <div class="risk-bar-gradient"></div>
        <div class="risk-bar-marker" style="left: ${(phishingProb * 100).toFixed(1)}%"></div>
        </div>
        <div class="risk-bar-score">
        Phishing Score: <b>${phishingProb.toFixed(2)}</b>
        </div>
    </div>
    `;

    resultDiv.innerHTML = `
        <div class="result ${resultClass}">
            <div class="result-title">${escapeHtml(result.result)}</div>
            <div class="mini-grid">
                <div class="mini-card"><span>Model</span><b>${escapeHtml(result.algorithm)}</b></div>
                <div class="mini-card"><span>Rule</span><b>score &lt; ${threshold.toFixed(2)} = BENIGN, score ≥ ${threshold.toFixed(2)} = PHISHING</b></div>
                ${topFeaturesHtml}
            </div>
            <div class="mini-card ai-reasoning-card" style="margin:16px 0 8px 0;">
                <span>AI Reasoning</span>
                <div style="white-space:pre-line" class="ai-reasoning-html">${result.explanation}</div>
            </div>
            ${riskBarHtml}
        </div>
    `;
}

function fillURL(url) {
    const input = document.getElementById("urlInput");
    if (!input) return;
    input.value = url;
    input.focus();
}

async function analyzeURL() {
    const urlInput = document.getElementById("urlInput");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const loading = document.getElementById("loading");
    const resultDiv = document.getElementById("result");
    const modeSelect = document.getElementById("modeSelect");

    if (!urlInput || !analyzeBtn || !loading || !resultDiv) return;

    const url = urlInput.value.trim();
    const decision_mode = modeSelect ? modeSelect.value : "hybrid_prefilter";

    if (!url) {
        alert("Please enter a URL to analyze");
        return;
    }

    analyzeBtn.disabled = true;
    loading.style.display = "block";
    resultDiv.innerHTML = "";

    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url, decision_mode, debug: true })
        });

        const data = await res.json();

        if (!res.ok) {
            displayEnhancedResult(
                {
                    result: data.error || "Gagal memproses URL",
                    category: "error"
                },
                url
            );
        } else {
            const mapped = backendToEnhancedResult(data, url);
            displayEnhancedResult(mapped, url);
        }
    } catch (e) {
        displayEnhancedResult(
            {
                result: e.message || "Unable to analyze URL",
                category: "error"
            },
            url
        );
    } finally {
        analyzeBtn.disabled = false;
        loading.style.display = "none";
    }
}

function initDetectorUI() {
    const urlInput = document.getElementById("urlInput");
    if (urlInput) {
        urlInput.addEventListener("keypress", function (e) {
            if (e.key === "Enter") analyzeURL();
        });
        urlInput.focus();
    }
}

window.analyzeURL = analyzeURL;
window.fillURL = fillURL;

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initDetectorUI);
} else {
    initDetectorUI();
}