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

// UI MAPPER: backend response -> UI display
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

    return {
        result: isPhishing
            ? `🚨 PHISHING: Final probability ${(phishingProb * 100).toFixed(1)}%`
            : `✅ SAFE: Final probability ${(safeProb * 100).toFixed(1)}%`,
        algorithm: "Final Decision Probability (Rule + RF + XGB + Stacking)",
        confidence,
        category,
        details: [
            `Phishing: ${(phishingProb * 100).toFixed(1)}%`,
            `Safe: ${(safeProb * 100).toFixed(1)}%`
        ],
        final_phishing_prob: phishingProb,
        final_safe_prob: safeProb
    };
}

// Display function
function displayEnhancedResult(result, url) {
    const resultDiv = document.getElementById("result");
    if (!resultDiv) return;

    if (
        result.category === "error" ||
        (result.result && String(result.result).includes("URL tidak valid"))
    ) {
        resultDiv.innerHTML = `
            <div class="result error">
                <div class="result-header">
                    <span class="result-icon">❌</span>
                    <div class="result-text">
                        URL tidak valid.<br>
                        Masukkan URL yang benar.<br>
                        Contoh: <b>https://example.com</b>
                    </div>
                </div>
            </div>
        `;
        return;
    }

    const phishingProb = clamp01(result.final_phishing_prob);
    const safeProb = clamp01(result.final_safe_prob);
    const confidencePercent = Math.max(0, Math.min(100, Number(result.confidence || 0) * 100)).toFixed(1);

    const resultClass = result.category === "phishing" ? "phishing" : "safe";
    const resultIcon = result.category === "phishing" ? "🚨" : "✅";

    resultDiv.innerHTML = `
        <div class="result ${resultClass}">
            <div class="result-header">
                <span class="result-icon">${resultIcon}</span>
                <div class="result-text">${escapeHtml(result.result)}</div>
            </div>
            <div class="confidence-section">
                <div class="confidence-header">
                    <strong style="font-size: 1.2em;">🎯 AI Confidence</strong>
                    <span style="font-size: 1.3em; font-weight: 700;">${confidencePercent}%</span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                </div>
            </div>

            <div class="details-grid">
                <div class="detail-card">
                    <div class="detail-title">🤖 AI Algorithms Used</div>
                    <div class="detail-content">${escapeHtml(result.algorithm || "Final Decision Probability")}</div>
                </div>

                <div class="detail-card">
                    <div class="detail-title">🔗 Analyzed URL</div>
                    <div class="detail-content" style="font-family: monospace; word-break: break-all; font-size: 0.9em;">${escapeHtml(url)}</div>
                </div>

                <div class="detail-card">
                    <div class="detail-title">📋 Final Probability</div>
                    <div class="detail-content">
                        Phishing: <strong>${(phishingProb * 100).toFixed(1)}%</strong> •
                        Safe: <strong>${(safeProb * 100).toFixed(1)}%</strong>
                    </div>
                </div>
            </div>
        </div>
    `;

    const explanation = generateExplanation({
        category: result.category,
        confidence: result.confidence,
        final_phishing_prob: phishingProb,
        final_safe_prob: safeProb
    });

    resultDiv.innerHTML += `
        <div class="detail-card penjelasan-otomatis">
            <div class="detail-title">📝 Penjelasan Otomatis</div>
            <div class="detail-content">${explanation}</div>
        </div>
    `;
}

function generateExplanation(result) {
    return `
        Probabilitas akhir phishing: <b>${(clamp01(result.final_phishing_prob) * 100).toFixed(1)}%</b>.<br>
        Probabilitas akhir safe: <b>${(clamp01(result.final_safe_prob) * 100).toFixed(1)}%</b>.<br>
        Kesimpulan: <b>${String(result.category || "").toUpperCase()}</b> dengan confidence
        <b>${(clamp01(result.confidence) * 100).toFixed(1)}%</b>.
    `;
}

function fillURL(url) {
    const input = document.getElementById("urlInput");
    if (!input) return;
    input.value = url;
    input.focus();
}

// Main analyze function (Backend Flask)
async function analyzeURL() {
    const urlInput = document.getElementById("urlInput");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const loading = document.getElementById("loading");
    const resultDiv = document.getElementById("result");

    if (!urlInput || !analyzeBtn || !loading || !resultDiv) return;

    const url = urlInput.value.trim();
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
            body: JSON.stringify({ url }) // single pipeline
        });

        const data = await res.json();

        if (!res.ok) {
            displayEnhancedResult({
                result: `❌ ERROR: ${data.error || "Gagal memproses URL"}`,
                algorithm: "Backend",
                confidence: 0,
                category: "error",
                final_phishing_prob: 0,
                final_safe_prob: 0
            }, url);
        } else {
            const uiResult = backendToEnhancedResult(data, url);
            uiResult.algorithm = `Final Decision Probability (${data.decision_source || "Rule + RF + XGB + Stacking"})`;
            displayEnhancedResult(uiResult, url);
        }
    } catch (e) {
        displayEnhancedResult({
            result: `❌ ERROR: ${e.message || "Unable to analyze URL"}`,
            algorithm: "Backend",
            confidence: 0,
            category: "error",
            final_phishing_prob: 0,
            final_safe_prob: 0
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