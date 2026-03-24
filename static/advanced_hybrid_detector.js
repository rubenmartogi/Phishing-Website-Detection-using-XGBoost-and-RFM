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
    let phishingProb = clamp01(data.final_phishing_prob);
    let safeProb = clamp01(data.final_safe_prob);

    if (!Number.isFinite(Number(data.final_safe_prob))) {
        safeProb = clamp01(1 - phishingProb);
    }
    if (!Number.isFinite(Number(data.final_phishing_prob))) {
        phishingProb = clamp01(1 - safeProb);
    }

    const isPhishing = Number(data.final_label) === 1;
    const category = isPhishing ? "phishing" : "safe";
    const confidence = isPhishing ? phishingProb : safeProb;

    let displayModelName = data.model_name || "Model Prediction";
    const mode = data.decision_mode || "hybrid_prefilter";
    const source = data.decision_source || "-";

    if (mode === "rf_only") {
        displayModelName = "🌳 Random Forest Only";
    } else if (mode === "xgb_only") {
        displayModelName = "⚡ XGBoost Only";
    } else if (mode === "ml_stacking_only") {
        displayModelName = "🔗 RF + XGB + Stacking";
    } else if (mode === "hybrid_prefilter") {
        if (source.includes("rule")) {
            displayModelName = "🛡️ Rule-Based Prefilter";
        } else if (source.includes("stacking")) {
            displayModelName = "🔗 RF + XGB + Stacking";
        } else {
            displayModelName = "🛡️ Rule-Based + ML + Stacking";
        }
    }

    return {
        result: isPhishing
            ? `🚨 PHISHING: Final probability ${(phishingProb * 100).toFixed(1)}%`
            : `✅ SAFE: Final probability ${(safeProb * 100).toFixed(1)}%`,
        algorithm: displayModelName,
        decisionMode: mode,
        decisionSource: source,
        confidence,
        category,
        final_phishing_prob: phishingProb,
        final_safe_prob: safeProb,
        risk_score: data.risk_score,
        risk_category: data.risk_category,
        url
    };
}

function displayEnhancedResult(result, url) {
    const resultDiv = document.getElementById("result");
    if (!resultDiv) return;

    if (result.category === "error") {
        resultDiv.innerHTML = `
            <div class="result error">
                <div class="result-title">❌ ERROR</div>
                <div class="result-meta">${escapeHtml(result.result || "Unknown error")}</div>
            </div>
        `;
        return;
    }

    const resultClass = result.category === "phishing" ? "phishing" : "safe";
    const confidencePercent = (clamp01(result.confidence) * 100).toFixed(1);
    const pPhish = (clamp01(result.final_phishing_prob) * 100).toFixed(1);
    const pSafe = (clamp01(result.final_safe_prob) * 100).toFixed(1);

    resultDiv.innerHTML = `
        <div class="result ${resultClass}">
            <div class="result-title">${escapeHtml(result.result)}</div>

            <div class="mini-grid">
                <div class="mini-card"><span>Model</span><b>${escapeHtml(result.algorithm)}</b></div>
                <div class="mini-card"><span>Decision Mode</span><b>${escapeHtml(result.decisionMode)}</b></div>
                <div class="mini-card"><span>Decision Source</span><b>${escapeHtml(result.decisionSource)}</b></div>
                <div class="mini-card"><span>Risk</span><b>${escapeHtml(String(result.risk_category ?? "-"))} (score: ${escapeHtml(String(result.risk_score ?? "-"))})</b></div>

                <!-- TANPA GARIS -->
                <div class="mini-card"><span>Phishing Probability</span><b>${pPhish}%</b></div>
                <div class="mini-card"><span>Safe Probability</span><b>${pSafe}%</b></div>
            </div>


            <div class="meter-box">
                <div class="meter-label">
                    <span>Confidence</span><b>${confidencePercent}%</b>
                </div>
                <div class="meter-scale"><span>0</span><span>100</span></div>
                <div class="meter-track">
                    <div class="meter-fill confidence" style="width:${confidencePercent}%"></div>
                </div>
            </div>

            <div class="url-box">
                <span>URL</span>
                <code>${escapeHtml(url || result.url || "")}</code>
            </div>
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
            displayEnhancedResult({
                result: data.error || "Gagal memproses URL",
                category: "error"
            }, url);
        } else {
            const mapped = backendToEnhancedResult(data, url);
            displayEnhancedResult(mapped, url);
        }
    } catch (e) {
        displayEnhancedResult({
            result: e.message || "Unable to analyze URL",
            category: "error"
        }, url);
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