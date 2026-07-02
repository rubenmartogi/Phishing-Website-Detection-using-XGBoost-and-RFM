# ══════════════════════════════════════════════════════════════════════
# ██  BAGIAN 3, 8, 9 — MODEL ML (LOAD, SELEKSI FITUR, PREDIKSI)
# ══════════════════════════════════════════════════════════════════════

import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from fungsi.config import PHISHING_CLASS_VALUE, WEB_CONTENT_KEYS
from fungsi.features import (
    _compute_url_extended_features,
    extract_url_features,
    extract_web_content_features,
)
from fungsi.utils import clamp01, to_float

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

executor = ThreadPoolExecutor(max_workers=4)
GLOBAL_MODELS: dict = {}


# ── BAGIAN 3 — STARTUP CACHE ──────────────────────────────────────────

def init_startup_cache():
    print("[STARTUP] Inisialisasi Cache Model...")
    for suffix in ["_37", "_81"]:
        try:
            GLOBAL_MODELS[f"rf{suffix}"]  = pickle.load(open(os.path.join(MODELS_DIR, f"random_forest_model{suffix}.pkl"), "rb"))
            GLOBAL_MODELS[f"xgb{suffix}"] = pickle.load(open(os.path.join(MODELS_DIR, f"xgboost_model{suffix}.pkl"), "rb"))
            meta_path = os.path.join(MODELS_DIR, f"rule_lr{suffix}.pkl")
            GLOBAL_MODELS[f"meta{suffix}"] = pickle.load(open(meta_path, "rb")) if os.path.exists(meta_path) else None

            col_path = os.path.join(MODELS_DIR, f"feature_columns{suffix}.txt")
            if os.path.exists(col_path):
                GLOBAL_MODELS[f"cols{suffix}"] = [l.strip() for l in open(col_path, encoding="utf-8") if l.strip()]
            else:
                GLOBAL_MODELS[f"cols{suffix}"] = []
            print(f"[STARTUP] Cache model {suffix} berhasil dimuat.")
        except Exception as e:
            print(f"[STARTUP WARNING] Gagal memuat cache model {suffix}: {e}")


def load_feature_columns_81():
    try:
        return [
            l.strip()
            for l in open(os.path.join(MODELS_DIR, "feature_columns_81.txt"), encoding="utf-8")
            if l.strip()
        ]
    except Exception:
        return []


# ── BAGIAN 8 — SELEKSI MODEL & PEMBUATAN VEKTOR FITUR ────────────────

def get_model_and_features(url):
    """
    Tentukan mode 37 vs 81 fitur berdasarkan hasil web fetch.
    Return: (rf, xgb, meta, cols, web_content_ok, feats_full)
    """
    web_feats     = extract_web_content_features(url)
    web_content_ok = any(web_feats.get(k) not in (None, 0) for k in WEB_CONTENT_KEYS)

    if web_content_ok:
        # Mode 81: 37 URL + 18 URL extended + 26 web content
        feats = extract_url_features(url)
        feats.update(_compute_url_extended_features(url))
        feats.update(web_feats)
    else:
        # Mode 37: 37 URL saja
        feats = extract_url_features(url)

    suffix = "_81" if web_content_ok else "_37"
    rf     = GLOBAL_MODELS.get(f"rf{suffix}")
    xgb    = GLOBAL_MODELS.get(f"xgb{suffix}")
    meta   = GLOBAL_MODELS.get(f"meta{suffix}")
    cols   = GLOBAL_MODELS.get(f"cols{suffix}", [])

    if not cols:
        raise RuntimeError(f"Fitur kolom kosong untuk konfigurasi suffix {suffix}")
    return rf, xgb, meta, cols, web_content_ok, feats


def build_ml_vector(feats, cols):
    """Susun nilai fitur sesuai urutan kolom training — nilai tidak ada → default 0.0."""
    vector = [to_float(feats.get(c, 0.0)) for c in cols]
    return np.array([vector], dtype=float), cols


# ── BAGIAN 9 — PREDIKSI RF + XGB + STACKING ──────────────────────────

def _class_index(model, class_value, fallback_idx=1):
    classes = list(getattr(model, "classes_", []))
    if class_value in classes:
        return classes.index(class_value)
    return 1 if len(classes) == 2 and fallback_idx >= 1 else 0


def _proba_for_class(model, X, class_value):
    try:
        probs = model.predict_proba(X)[0]
        idx   = _class_index(model, class_value, 1)
        return float(probs[min(idx, len(probs) - 1)])
    except Exception:
        return 1.0 if int(model.predict(X)[0]) == class_value else 0.0


def _label_to_phishing_flag(raw):
    return 1 if int(raw) == PHISHING_CLASS_VALUE else 0


def predict_models(feats, cols, precomputed_rule_score=None, precomputed_rule_flag=None,
                   rf=None, xgb=None, meta=None):
    """
    Jalankan prediksi 3 model:
      RF → rf_prob | XGB → xgb_prob
      Meta-Learner (LR Stacking) → input: [rf_prob, xgb_prob, rule_flag, risk_score]
    """
    X, used_cols = build_ml_vector(feats, cols)
    df           = pd.DataFrame(X, columns=used_cols)
    rf_prob      = _proba_for_class(rf, df, PHISHING_CLASS_VALUE)
    xgb_prob     = _proba_for_class(xgb, df, PHISHING_CLASS_VALUE)
    stack_pred = stack_prob = meta_n_in = None

    if meta is not None:
        try:
            meta_n_in = int(getattr(meta, "n_features_in_", 2))
        except Exception:
            meta_n_in = 2

        rs  = precomputed_rule_score if precomputed_rule_score is not None else 0.0
        rf_ = precomputed_rule_flag  if precomputed_rule_flag  is not None else 0

        meta_feats = [rf_prob, xgb_prob]
        if meta_n_in >= 3:
            meta_feats = [rf_prob, xgb_prob, rf_]
        if meta_n_in >= 4:
            meta_feats = [rf_prob, xgb_prob, rf_, rs]

        meta_X = np.array([meta_feats[:meta_n_in]], dtype=float)
        try:
            stack_pred = _label_to_phishing_flag(int(meta.predict(meta_X)[0]))
            stack_prob = _proba_for_class(meta, meta_X, PHISHING_CLASS_VALUE)
        except Exception as e:
            print(f"[STACK WARNING] Meta-learner gagal: {e}")
            stack_pred = None
            stack_prob = None

    return {
        "rf_prob":    rf_prob,
        "xgb_prob":   xgb_prob,
        "rf_pred":    _label_to_phishing_flag(int(rf.predict(df)[0])),
        "xgb_pred":   _label_to_phishing_flag(int(xgb.predict(df)[0])),
        "stack_pred": stack_pred,
        "stack_prob": stack_prob,
        "meta_n_in":  meta_n_in,
    }
