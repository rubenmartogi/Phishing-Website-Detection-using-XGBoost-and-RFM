// UI MAPPER: backend response -> UI display
function backendToEnhancedResult(data, url) {
    // Penentuan kategori dan confidence
    const isPhishing = data.final_label === 1;
    let category = "safe";
    let resultText = "✅ SAFE: No significant threats detected";
    let confidence = 0.9;

    if (data.category === "Phishing" || isPhishing) {
        category = "phishing";
        resultText = "🚨 PHISHING: Detected by ML/Rule-based";
        confidence = 0.85;
    } else if (data.category === "Suspicious") {
        category = "suspicious";
        resultText = "⚠️ SUSPICIOUS: Multiple risk indicators";
        confidence = 0.65;
    } else if (data.category === "Caution") {
        category = "caution";
        resultText = "🟡 CAUTION: Some risk factors detected";
        confidence = 0.55;
    }

    // Voting weights: rule-based tetap jadi penyaring awal, tapi ML lebih dominan
    // const weights = { rule: 0.2, rf: 0.4, xgb: 0.4 };

    // // Voting calculation (contoh sederhana)
    // let phishingScore = 0;
    // phishingScore += (data.category === "Phishing" ? weights.rule : 0);
    // phishingScore += (data.rf_pred === 1 ? weights.rf : 0);
    // phishingScore += (data.xgb_pred === 1 ? weights.xgb : 0);

    // let legitimateScore = 1 - phishingScore;

    return {
        result: `${resultText} (Risk: ${data.risk_score})`,
        algorithm: "Rule-based + Random Forest + XGBoost + Stacking (LR)",
        confidence: confidence,
        category: category,
        details: [
            `Rule Flag: ${data.rule_flag}`,
            `RF: ${data.rf_pred}, XGB: ${data.xgb_pred}`,
            `Stacking: ${data.stack_pred === null ? "N/A" : data.stack_pred}`
        ],
        votingDetails: null,
        // {
        //     phishingVotes: phishingScore,
        //     legitimateVotes: legitimateScore,
        //     totalVotes: 1.0,
        //     ruleResult: { result: data.category, confidence: 0.8 },
        //     rfResult: { prediction: data.rf_pred === 1 ? "phishing" : "legitimate", confidence: 0.7 },
        //     xgbResult: { prediction: data.xgb_pred === 1 ? "phishing" : "legitimate", confidence: 0.75 },
        //     dynamicWeights: weights
        // },
        // Tambahan: mapping probabilitas
        rule_flag: data.rule_flag,
        rule_prob: data.rule_prob,
        rf_pred: data.rf_pred,
        rf_prob: data.rf_prob,
        xgb_pred: data.xgb_pred,
        xgb_prob: data.xgb_prob,
        stack_pred: data.stack_pred,
        stack_prob: data.stack_prob
    };
}


// Display function
function displayEnhancedResult(result, url) {
    const resultDiv = document.getElementById('result');
    const confidencePercent = (result.confidence * 100).toFixed(1);
    console.log("result.details:", result.details);
    console.log("result.votingDetails:", result.votingDetails);
    
    let resultClass = result.category || 'safe';
    let resultIcon = '❓'; 
    let threatLevel = 10;
    
    if (result.category === 'phishing') {
        resultIcon = '🚨'; 
        resultClass = 'phishing';
        threatLevel = 85 + (result.confidence * 15);
    } else if (result.category === 'suspicious') {
        resultIcon = '⚠️'; 
        resultClass = 'suspicious';
        threatLevel = 40 + (result.confidence * 25);
    } else {
        resultIcon = '✅'; 
        resultClass = 'safe';
        threatLevel = Math.max(5, 20 - (result.confidence * 15));
    }
    // Tambahan: jika error validasi URL, tampilkan pesan sederhana
    if (
        result.category === 'error' ||
        (result.result && result.result.includes('URL tidak valid'))
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

    let algorithmBreakdown = '';
    let mlModelStatus = '';
    
    if (result.votingDetails) {
    const vd = result.votingDetails;
    const dw = vd.dynamicWeights;

    algorithmBreakdown = `
        <div class="algorithm-breakdown">
            <div class="algorithm-card">
                <div class="algorithm-name">Rule-based</div>
                <div class="algorithm-result">${result.category.charAt(0).toUpperCase() + result.category.slice(1)}</div>
                <div class="algorithm-confidence">
                    ${(result.rule_prob * 100).toFixed(1)}% (${(dw.rule * 100).toFixed(0)}% weight)
                </div>
            </div>
            <div class="algorithm-card">
                <div class="algorithm-name">Random Forest</div>
                <div class="algorithm-result">${result.rf_pred === 1 ? "Phishing" : "Legitimate"}</div>
                <div class="algorithm-confidence">
                    ${(result.rf_prob * 100).toFixed(1)}% (${(dw.rf * 100).toFixed(0)}% weight)
                </div>
            </div>
            <div class="algorithm-card">
                <div class="algorithm-name">XGBoost</div>
                <div class="algorithm-result">${result.xgb_pred === 1 ? "Phishing" : "Legitimate"}</div>
                <div class="algorithm-confidence">
                    ${(result.xgb_prob * 100).toFixed(1)}% (${(dw.xgb * 100).toFixed(0)}% weight)
                </div>
            </div>
            <div class="algorithm-card">
                <div class="algorithm-name">Logistic Regression Stacking</div>
                <div class="algorithm-result">${result.stack_pred === 1 ? "Phishing" : (result.stack_pred === 0 ? "Legitimate" : "N/A")}</div>
                <div class="algorithm-confidence">
                    ${result.stack_prob !== null && result.stack_prob !== undefined ? (result.stack_prob * 100).toFixed(1) + "%" : "N/A"} (Dynamic Ensemble)
                </div>
            </div>
        </div>
    `;
        
        mlModelStatus = `
            <div class="ml-model-status">
                <div class="detail-title">🤖 ML Model Status</div>
                <div class="detail-content">
                    ✅ Random Forest - ACTIVE<br>
                    ✅ XGBoost - ACTIVE<br>
                    ✅ Stacking (LR) - ACTIVE<br>
                    Ensemble Method: <strong>RF + XGB + LR</strong>
                </div>
            </div>
        `;
    }
    
    resultDiv.innerHTML = `
        <div class="result ${resultClass}">
            <div class="result-header">
                <span class="result-icon">${resultIcon}</span>
                <div class="result-text">${result.result}</div>
            </div>
            
            <div class="threat-level">
                <strong>🎯 Threat Level:</strong>
                <div class="threat-meter">
                    <div class="threat-pointer" style="left: ${threatLevel}%"></div>
                </div>
                <strong>${threatLevel.toFixed(0)}/100</strong>
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
            
            ${mlModelStatus}
            
            <div class="details-grid">
                <div class="detail-card">
                    <div class="detail-title">🤖 AI Algorithms Used</div>
                    <div class="detail-content">${result.algorithm}</div>
                </div>
                
                <div class="detail-card">
                    <div class="detail-title">🔗 Analyzed URL</div>
                    <div class="detail-content" style="font-family: monospace; word-break: break-all; font-size: 0.9em;">${url}</div>
                </div>
                
                ${result.details ? `
                <div class="detail-card">
                    <div class="detail-title">📋 Detection Details</div>
                    <div class="detail-content">${result.details.slice(0, 3).join(' • ')}</div>
                </div>
                ` : ''}
                
                ${result.votingDetails ? `
                <div class="detail-card">
                    <div class="detail-title">🗳️ Voting Results</div>
                    <div class="detail-content">
                        Phishing: <strong>${(result.votingDetails.phishingVotes * 100).toFixed(1)}%</strong><br>
                        Legitimate: <strong>${(result.votingDetails.legitimateVotes * 100).toFixed(1)}%</strong>
                    </div>
                </div>
                ` : ''}
            </div>
            
            ${algorithmBreakdown}
        </div>
    `;
    const explanation = generateExplanation(result);
    resultDiv.innerHTML += `
        <div class="detail-card penjelasan-otomatis">
            <div class="detail-title">📝 Penjelasan Otomatis</div>
            <div class="detail-content">${explanation}</div>
        </div>
    `;
}

function generateExplanation(result) {
    let explanation = "";

    // Rule-based hanya sebagai filter awal
    explanation += `Rule-based system memberikan flag: <b>${result.rule_flag}</b> (probabilitas: ${(result.rule_prob * 100).toFixed(1)}%). `;
    explanation += `Jika hasil rule-based adalah <b>Phishing</b>, maka langsung output PHISHING. Jika bukan, hasil akhir diambil dari Machine Learning (Random Forest, XGBoost, Stacking).<br>`;
    // Random Forest
    explanation += `Random Forest memprediksi: <b>${result.rf_pred}</b> (probabilitas: ${(result.rf_prob * 100).toFixed(1)}%). `;
    // XGBoost
    explanation += `XGBoost memprediksi: <b>${result.xgb_pred}</b> (probabilitas: ${(result.xgb_prob * 100).toFixed(1)}%). `;
    // Stacking LR
    explanation += `Stacking Logistic Regression memprediksi: <b>${result.stack_pred}</b> (probabilitas: ${(result.stack_prob * 100).toFixed(1)}%). `;

    // Tidak ada voting ensemble
    explanation += `<br>Kesimpulan: URL ini <b>${result.category.toUpperCase()}</b> dengan confidence <b>${(result.confidence * 100).toFixed(1)}%</b>.`;

    return explanation;
}

function fillURL(url) {
    document.getElementById('urlInput').value = url;
    document.getElementById('urlInput').focus();
}

// Main analyze function (Backend Flask)
async function analyzeURL() {
    const urlInput = document.getElementById('urlInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loading = document.getElementById('loading');
    const resultDiv = document.getElementById('result');
    
    const url = urlInput.value.trim();
    
    if (!url) {
        alert('Please enter a URL to analyze');
        return;
    }
    
    analyzeBtn.disabled = true;
    loading.style.display = 'block';
    resultDiv.innerHTML = '';
    
    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });

        const data = await res.json();
        if (!res.ok) {
            displayEnhancedResult({
                result: `❌ ERROR: ${data.error || "Gagal memproses URL"}`,
                algorithm: 'Backend',
                confidence: 0.0,
                details: [],
                category: 'error'
            }, url);
        } else {
            const uiResult = backendToEnhancedResult(data, url);
            displayEnhancedResult(uiResult, url);
        }
    } catch (e) {
        displayEnhancedResult({
            result: '❌ ERROR: Unable to analyze URL',
            algorithm: 'Backend',
            confidence: 0.0,
            details: [e.message],
            category: 'error'
        }, url);
    } finally {
        analyzeBtn.disabled = false;
        loading.style.display = 'none';
    }
}

document.getElementById('urlInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        analyzeURL();
    }
});

window.onload = function() {
    document.getElementById('urlInput').focus();
};