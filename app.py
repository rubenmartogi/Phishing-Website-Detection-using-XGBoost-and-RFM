"""
app.py — Entry point Flask
Semua logika bisnis ada di folder fungsi/:
  fungsi/config.py       → Konstanta & konfigurasi global
  fungsi/utils.py        → Fungsi utilitas umum
  fungsi/features.py     → Ekstraksi fitur URL & web content
  fungsi/rule_based.py   → Rule-based prefilter (24 rule)
  fungsi/model.py        → Load model, seleksi fitur, prediksi ML
  fungsi/shap_explain.py → SHAP explainability
  fungsi/llm_explain.py  → Prompt LLM & generate explanation
  fungsi/logger.py       → Logging ke CSV

Artifact hasil training tersimpan di folder models/:
  models/random_forest_model{_37|_81}.pkl
  models/xgboost_model{_37|_81}.pkl
  models/rule_lr{_37|_81}.pkl
  models/feature_columns{_37|_81}.txt
  models/ga_tuning_report{_37|_81}.json
  (di-generate oleh train_model.py — jangan edit manual)
"""

import os

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, request, send_from_directory

from fungsi.config import (
    DEFAULT_DECISION_MODE, DEFAULT_USE_PREFILTER,
    FINAL_THRESHOLD, PREFILTER_PHISHING_MIN_CONF,
)
from fungsi.llm_explain import generate_explanation
from fungsi.logger import check_log_locked, log_feature_extraction
from fungsi.model import (
    GLOBAL_MODELS, init_startup_cache, load_feature_columns_81,
    get_model_and_features, predict_models,
)
from fungsi.rule_based import rule_based_eval
from fungsi.utils import clamp01, is_valid_url, safe_avg, to_bool

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH  = os.path.join(BASE_DIR, "log_feature_extraction.csv")

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "static"))

# Muat model ke memori saat server start
init_startup_cache()
FEATURE_COLUMNS_81 = load_feature_columns_81()


# ── HELPERS ───────────────────────────────────────────────────────────

def normalize_decision_mode(mode):
    if not mode:
        return DEFAULT_DECISION_MODE
    m = str(mode).strip().lower()
    if m in ("rf", "rf_only"):
        return "rf_only"
    if m in ("xgb", "xgb_only"):
        return "xgb_only"
    if m in ("stack", "stacking", "ml_stacking_only"):
        return "ml_stacking_only"
    return "hybrid_prefilter"


# ── ROUTES ────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return send_from_directory("Phishing_detection_app", "advanced_hybrid_detector.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    data          = request.get_json(silent=True) or {}
    url           = (data.get("url") or "").strip()
    debug         = to_bool(data.get("debug", False))
    use_prefilter = to_bool(data.get("use_prefilter", DEFAULT_USE_PREFILTER), DEFAULT_USE_PREFILTER)
    decision_mode = normalize_decision_mode(data.get("decision_mode") or data.get("mode"))

    if not url:
        return jsonify({"error": "URL kosong"}), 400
    if not is_valid_url(url):
        return jsonify({"error": "URL tidak valid"}), 400

    log_locked, lock_msg = check_log_locked(LOG_PATH)
    if log_locked:
        return jsonify({"error": lock_msg}), 503

    rf, xgb, meta, cols, web_content_ok, feats_full = get_model_and_features(url)
    include_web_content = web_content_ok
    mode = "81" if web_content_ok else "37"
    risk_score, risk_category, rule_flag, rule_detail = rule_based_eval(url, return_detail=True)

    def build_response(final_phishing_prob, decision_source, model_name,
                       rf_prob=None, xgb_prob=None, stack_prob=None,
                       web_used=None, rf_model=None, xgb_model=None):
        p_phish     = clamp01(final_phishing_prob)
        p_safe      = clamp01(1.0 - p_phish)
        final_label = 1 if p_phish >= FINAL_THRESHOLD else 0
        category    = "phishing" if final_label == 1 else "benign"
        confidence  = round(p_phish if final_label == 1 else p_safe, 4)

        exp = generate_explanation(
            url, feats_full, cols, rf_prob, xgb_prob, stack_prob,
            decision_source, rule_detail, risk_score, rule_flag,
            rf_model, xgb_model, final_label,
        )

        resp = {
            "final_label":          final_label,
            "final_phishing_prob":  round(p_phish, 4),
            "final_safe_prob":      round(p_safe, 4),
            "final_threshold":      FINAL_THRESHOLD,
            "decision_mode":        decision_mode,
            "decision_source":      decision_source,
            "model_name":           model_name,
            "rule_flag":            rule_flag,
            "model_feature_count":  len(cols),
            "web_content_used":     include_web_content if web_used is None else bool(web_used),
            "label":                category,
            "category":             category,
            "confidence":           confidence,
            "rf_prob":              round(rf_prob, 4)    if rf_prob    is not None else None,
            "xgb_prob":             round(xgb_prob, 4)   if xgb_prob   is not None else None,
            "stack_prob":           round(stack_prob, 4) if stack_prob is not None else None,
            "llm_reasoning":        exp["llm_reasoning"],
            "explanation_detailed": {
                "final_prediction":             "PHISHING" if final_label == 1 else "BENIGN",
                "confidence_score":             confidence,
                "phishing_probability":         round(p_phish, 4),
                "benign_probability":           round(p_safe, 4),
                "main_contributing_model":      exp["main_contributing_model"],
                "top_influential_features":     exp["top_influential_features"],
                "model_contribution_probability": {
                    k: round(v, 4) for k, v in {
                        "Random Forest": rf_prob,
                        "XGBoost": xgb_prob,
                        "Logistic Regression (Stacking)": stack_prob,
                    }.items() if v is not None
                },
            },
        }

        if debug:
            resp["debug"] = {
                "prefilter_enabled": use_prefilter,
                "decision_mode":     decision_mode,
                "risk_score":        risk_score,
                "risk_category":     risk_category,
                "rule_detail":       rule_detail,
                "features_analyzed": len(cols),
            }

        # Susun log SHAP lines
        log_top_lines = []
        all_shap = exp.get("shap_items", [])
        if all_shap:
            for f in all_shap:
                shap_score = f["shap_signed"]
                feat_val   = feats_full.get(f["name"], None)
                if abs(shap_score) < 1e-6 or feat_val is None:
                    continue
                feat_val_str = f"{round(feat_val, 4)}" if isinstance(feat_val, float) else str(feat_val)
                arah = "PHISHING" if shap_score > 0 else ("BENIGN" if shap_score < 0 else "NETRAL")
                log_top_lines.append(
                    f"Fitur: {f['name']} | Nilai: {feat_val_str} | SHAP: {shap_score:+.4f} | Dampak: [{arah}]"
                )
        else:
            for f in exp.get("top_influential_features", []):
                shap_val = f.get("shap_value_signed", f.get("shap_value", 0))
                arah     = f.get("shap_direction", f.get("impact", "?"))
                raw_val  = f.get("value", 0)
                val_str  = f"{round(raw_val, 4)}" if isinstance(raw_val, float) else str(raw_val)
                log_top_lines.append(
                    f"Fitur: {f['name']} | Nilai: {val_str} | SHAP: {shap_val:+.4f} | Dampak: [{arah}]"
                )

        feats_for_log = dict(feats_full)
        feats_for_log["_phishing_prob"] = round(p_phish, 4)
        feats_for_log["_top_features"]  = "\n".join(log_top_lines)
        feats_for_log["_llm_reasoning"] = exp.get("llm_reasoning", "")

        log_err = log_feature_extraction(url, mode, feats_for_log, FEATURE_COLUMNS_81, category, decision_source, LOG_PATH)
        if log_err:
            resp["log_warning"] = log_err

        return jsonify(resp)

    # ── Routing decision mode ─────────────────────────────────────────

    if decision_mode == "rf_only":
        preds = predict_models(feats_full, cols, risk_score, rule_flag, rf, xgb, meta)
        return build_response(clamp01(preds["rf_prob"]), "rf_only", "Random Forest",
                              rf_prob=preds["rf_prob"], rf_model=rf)

    if decision_mode == "xgb_only":
        preds = predict_models(feats_full, cols, risk_score, rule_flag, rf, xgb, meta)
        return build_response(clamp01(preds["xgb_prob"]), "xgb_only", "XGBoost",
                              xgb_prob=preds["xgb_prob"], xgb_model=xgb)

    if decision_mode == "ml_stacking_only":
        preds = predict_models(feats_full, cols, risk_score, rule_flag, rf, xgb, meta)
        if preds["stack_prob"] is not None:
            return build_response(
                clamp01(preds["stack_prob"]), "ml_stacking_only",
                "RF + XGB + Logistic Regression Stacking",
                preds["rf_prob"], preds["xgb_prob"], preds["stack_prob"],
                rf_model=rf, xgb_model=xgb,
            )
        p = safe_avg(preds["rf_prob"], preds["xgb_prob"])
        return build_response(p, "ml_rf_xgb_average_only", "RF + XGB Average",
                              preds["rf_prob"], preds["xgb_prob"], rf_model=rf, xgb_model=xgb)

    # Jika rule_flag=1 → bypass ML, langsung PHISHING dengan conf >= 0.95
    if use_prefilter and rule_flag == 1:
        p = clamp01(max(PREFILTER_PHISHING_MIN_CONF, risk_score / 10.0))
        return build_response(p, "rule_based_prefilter_phishing", "Rule-Based Prefilter", web_used=False)

    preds = predict_models(feats_full, cols, risk_score, rule_flag, rf, xgb, meta)
    if preds["stack_prob"] is not None:
        return build_response(
            clamp01(preds["stack_prob"]), "ml_stacking_only",
            "RF + XGB + Logistic Regression Stacking",
            preds["rf_prob"], preds["xgb_prob"], preds["stack_prob"],
            rf_model=rf, xgb_model=xgb,
        )
    p = safe_avg(preds["rf_prob"], preds["xgb_prob"])
    return build_response(p, "ml_rf_xgb_average_only", "RF + XGB Average",
                          preds["rf_prob"], preds["xgb_prob"], rf_model=rf, xgb_model=xgb)


@app.post("/predict_url")
@app.post("/analyze")
def predict_alias():
    return predict()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
