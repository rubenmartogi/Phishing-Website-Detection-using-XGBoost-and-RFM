# ══════════════════════════════════════════════════════════════════════
# ██  BAGIAN 10 — SHAP (EXPLAINABILITY MODEL)
# ══════════════════════════════════════════════════════════════════════
# Hitung kontribusi tiap fitur via SHAP.
# TreeExplainer (cepat) → fallback KernelExplainer jika gagal.
# exp.expected_value = baseline (rata-rata prob phishing data training).
# prob_akhir = baseline + Σ(shap_positif) + Σ(shap_negatif)

import numpy as np
import shap


def _shap_vector(shap_values, class_index=1):
    if isinstance(shap_values, list):
        vals = shap_values[class_index] if len(shap_values) > class_index else shap_values[0]
        return np.array(vals).flatten()
    v = np.array(shap_values)
    if v.ndim == 2:
        return v[0]
    if v.ndim == 3:
        return v[0, :, class_index] if v.shape[-1] > class_index else v[0, :, 0]
    return v.flatten()


def get_shap_top(model, X_arr, feature_names, top_n=5):
    fn = list(feature_names)
    try:
        exp  = shap.TreeExplainer(model)
        sv   = exp.shap_values(X_arr)
        vals = _shap_vector(sv, 1)
    except Exception:
        try:
            bg   = np.zeros((5, X_arr.shape[1]))
            exp  = shap.KernelExplainer(model.predict_proba, bg)
            sv   = exp.shap_values(X_arr, nsamples=100)
            vals = _shap_vector(sv, 1)
        except Exception:
            return []

    vals    = np.array(vals, dtype=float).flatten()[:len(fn)]
    top_idx = np.argsort(np.abs(vals))[::-1][:top_n]

    result = []
    for i in top_idx:
        s = float(vals[i])
        if abs(s) < 1e-4:
            continue
        result.append({
            "name":       fn[i],
            "shap_signed": s,
            "abs_shap":   abs(s),
            "direction":  "PHISHING" if s > 0 else ("BENIGN" if s < 0 else "NEUTRAL"),
        })
    return result
