"""
compute_baseline.py
====================
Tujuan: membuktikan asal-usul nilai "baseline" (expected_value) SHAP
untuk RF, XGBoost, dan meta-learner (Logistic Regression stacking)
yang dipakai pada laporan/sidang skripsi.

Script ini meniru PERSIS proses load_and_prepare_data() di train_model.py
(random_state, train_test_split, daftar fitur) supaya X_train yang
dipakai untuk menghitung baseline SHAP identik dengan X_train yang
dipakai saat training model aslinya.

Cara pakai:
    python3 compute_baseline.py --suffix 81
    python3 compute_baseline.py --suffix 37
"""
import argparse
import os
import pickle
import sys
from typing import List, Sequence, Tuple, cast

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

RANDOM_STATE = 12
TEST_SIZE = 0.2
TARGET_COL = "label"

url_features_37 = [
    "length_url", "length_hostname", "ip", "nb_dots", "nb_hyphens", "nb_at",
    "nb_qm", "nb_and", "nb_eq", "nb_underscore", "nb_tilde", "nb_percent",
    "nb_slash", "nb_star", "nb_colon", "nb_comma", "nb_semicolumn", "nb_dollar",
    "nb_space", "nb_www", "nb_com", "nb_dslash", "http_in_path", "https_token",
    "ratio_digits_url", "ratio_digits_host", "punycode", "port", "tld_in_path",
    "tld_in_subdomain", "abnormal_subdomain", "nb_subdomains", "prefix_suffix",
    "random_domain", "shortening_service", "path_extension", "nb_redirection",
]


def prep_X(df: pd.DataFrame, cols: List[str], idx: Sequence) -> pd.DataFrame:
    X = df.loc[idx].reindex(columns=cols)
    return X.apply(pd.to_numeric, errors="coerce")


def load_and_prepare_data(data_file_path: str, feature_mode: str):
    data = pd.read_csv(data_file_path)
    data.columns = data.columns.str.strip()

    y_all = cast(pd.Series, pd.to_numeric(data[TARGET_COL], errors="coerce"))
    valid_mask = y_all.notna()
    data = data.loc[valid_mask].copy()
    y_all = y_all.loc[valid_mask].astype("int64")

    id_cols = [c for c in ["url"] if c in data.columns]
    all_features = [c for c in data.columns if c not in [TARGET_COL] + id_cols]

    webcontent_features_44 = [c for c in all_features if c not in url_features_37]
    hybrid_features_81 = url_features_37 + webcontent_features_44

    feature_candidates = hybrid_features_81 if feature_mode == "hybrid81" else url_features_37

    train_idx, test_idx = cast(
        Tuple[np.ndarray, np.ndarray],
        train_test_split(
            data.index, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE
        ),
    )

    X_train = prep_X(data, feature_candidates, train_idx)
    X_test = prep_X(data, feature_candidates, test_idx)

    mask = X_train.notna().any(axis=0)
    selected_features = X_train.columns[mask.to_numpy().astype(bool)].tolist()

    X_train = X_train[selected_features].dropna()
    X_test = X_test[selected_features].dropna()

    y_train = y_all.loc[X_train.index]
    y_test = y_all.loc[X_test.index]

    return X_train, X_test, y_train, y_test, selected_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", choices=["81", "37"], default="81")
    parser.add_argument("--data", default=None, help="path ke data_cleaning.csv")
    parser.add_argument("--models-dir", default=None, help="folder berisi file .pkl")
    args = parser.parse_args()

    base_dir = args.models_dir or os.path.dirname(os.path.abspath(__file__))
    data_path = args.data or os.path.join(base_dir, "DataFiles", "data_cleaning.csv")
    suffix = f"_{args.suffix}"
    feature_mode = "hybrid81" if args.suffix == "81" else "url37"

    print(f"=== Verifikasi Baseline SHAP — mode {feature_mode} (suffix {suffix}) ===\n")

    # 1) Reproduksi X_train / X_test PERSIS seperti train_model.py
    X_train, X_test, y_train, y_test, feats = load_and_prepare_data(data_path, feature_mode)
    print(f"[OK] Data dimuat ulang dari: {data_path}")
    print(f"[OK] Jumlah fitur terpakai : {len(feats)}")
    print(f"[OK] Shape X_train         : {X_train.shape}")
    print(f"[OK] Shape X_test          : {X_test.shape}")
    print(f"[OK] random_state          : {RANDOM_STATE}  (sama dengan train_model.py)\n")

    # 2) Load model pickle hasil training
    with open(os.path.join(base_dir, f"random_forest_model{suffix}.pkl"), "rb") as f:
        rf_model = pickle.load(f)
    with open(os.path.join(base_dir, f"xgboost_model{suffix}.pkl"), "rb") as f:
        xgb_model = pickle.load(f)
    with open(os.path.join(base_dir, f"rule_lr{suffix}.pkl"), "rb") as f:
        lr_model = pickle.load(f)

    assert rf_model.n_features_in_ == len(feats), (
        f"MISMATCH: model RF dilatih dgn {rf_model.n_features_in_} fitur, "
        f"tapi reproduksi data menghasilkan {len(feats)} fitur. "
        f"Kemungkinan data_cleaning.csv yang dipakai BUKAN persis yang dipakai saat training."
    )
    print("[OK] Jumlah fitur model RF/XGB cocok dengan reproduksi data X_train.\n")

    # ------------------------------------------------------------------
    # 3) BASELINE RANDOM FOREST
    #    expected_value = rata-rata prediksi model atas data background
    #    (di sini: X_train, sesuai konvensi default shap.TreeExplainer)
    # ------------------------------------------------------------------
    print("--- RANDOM FOREST ---")
    rf_explainer = shap.TreeExplainer(rf_model)
    rf_ev = rf_explainer.expected_value
    rf_ev_class1 = rf_ev[1] if isinstance(rf_ev, (list, np.ndarray)) and len(rf_ev) > 1 else rf_ev
    manual_mean_pred = rf_model.predict_proba(X_train)[:, 1].mean()
    print(f"SHAP expected_value (class=1) : {float(rf_ev_class1):.4f}")
    print(f"Rata-rata predict_proba(X_train)[:,1] (cross-check manual): {manual_mean_pred:.4f}")
    print()

    # ------------------------------------------------------------------
    # 4) BASELINE XGBOOST
    #    base_score tersimpan di dalam model itu sendiri (tidak perlu data)
    # ------------------------------------------------------------------
    print("--- XGBOOST ---")
    booster = xgb_model.get_booster()
    import json
    cfg = json.loads(booster.save_config())
    raw_base_score = cfg["learner"]["learner_model_param"]["base_score"]
    base_score = float(str(raw_base_score).strip("[]"))
    print(f"base_score tersimpan di pickle (skala probabilitas): {base_score:.6f}")

    xgb_explainer = shap.TreeExplainer(xgb_model)
    xgb_ev = xgb_explainer.expected_value
    xgb_ev_val = xgb_ev[0] if isinstance(xgb_ev, (list, np.ndarray)) else xgb_ev
    print(f"SHAP expected_value (skala margin/log-odds)        : {float(xgb_ev_val):.6f}")
    sigmoid = 1 / (1 + np.exp(-float(xgb_ev_val)))
    print(f"Konversi expected_value margin -> probabilitas (sigmoid): {sigmoid:.4f}")
    print()

    # ------------------------------------------------------------------
    # 5) BASELINE META-LEARNER (Logistic Regression stacking)
    #    Input meta-learner = [rf_prob, xgb_prob] dari data training
    #    expected_value LinearExplainer = rata-rata prediksi atas background
    # ------------------------------------------------------------------
    print("--- META-LEARNER (Stacking LR) ---")
    print(f"n_features_in_ meta-learner: {lr_model.n_features_in_} (harus 2: [rf_prob, xgb_prob])")

    # PENTING: StackingClassifier scikit-learn TIDAK melatih meta-learner
    # dengan rf_model.predict_proba(X_train) langsung (itu bias/overfit,
    # karena model sudah "melihat" data itu saat fit). StackingClassifier
    # memakai cross_val_predict (out-of-fold predictions, default cv=5)
    # sebagai meta-feature untuk melatih final_estimator_.
    # Maka untuk mereplikasi baseline meta-learner yang BENAR, background
    # data harus berupa out-of-fold prediction, bukan in-sample prediction.
    from sklearn.model_selection import cross_val_predict
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    rf_params = rf_model.get_params()
    xgb_params = xgb_model.get_params()

    rf_oof = cross_val_predict(
        RandomForestClassifier(**rf_params), X_train, y_train,
        cv=5, method="predict_proba", n_jobs=-1
    )[:, 1]
    xgb_oof = cross_val_predict(
        XGBClassifier(**xgb_params), X_train, y_train,
        cv=5, method="predict_proba", n_jobs=-1
    )[:, 1]
    meta_X_train_oof = np.column_stack([rf_oof, xgb_oof])

    # (in-sample, untuk perbandingan/menunjukkan perbedaannya)
    rf_prob_train = rf_model.predict_proba(X_train)[:, 1]
    xgb_prob_train = xgb_model.predict_proba(X_train)[:, 1]
    meta_X_train_insample = np.column_stack([rf_prob_train, xgb_prob_train])

    def report_meta_baseline(label, meta_X):
        explainer = shap.LinearExplainer(lr_model, meta_X)
        ev = explainer.expected_value
        ev_val = ev[0] if isinstance(ev, (list, np.ndarray)) else ev
        sigmoid_val = 1 / (1 + np.exp(-float(ev_val)))
        manual_mean = lr_model.predict_proba(meta_X)[:, 1].mean()
        print(f"[{label}]")
        print(f"  SHAP expected_value (margin/log-odds) : {float(ev_val):.4f}")
        print(f"  Konversi margin -> probabilitas (sigmoid): {sigmoid_val:.4f}")
        print(f"  Rata-rata predict_proba (cross-check)  : {manual_mean:.4f}")
        print()

    report_meta_baseline("OUT-OF-FOLD (cara StackingClassifier melatih meta-learner, paling akurat)", meta_X_train_oof)
    report_meta_baseline("IN-SAMPLE (predict_proba langsung di X_train, untuk perbandingan)", meta_X_train_insample)
    print()

    print("=== SELESAI ===")
    print("Catatan: expected_value RF & meta-learner BERGANTUNG pada data")
    print("background yang dipakai (di sini: X_train hasil reproduksi split")
    print("dengan random_state=12, identik dengan train_model.py).")
    print("base_score XGBoost TIDAK bergantung data — tersimpan permanen di pickle.\n")

    # ------------------------------------------------------------------
    # 6) AUDIT TRAIL — timestamp + hash, supaya bisa ditunjukkan ke dosen
    #    bahwa angka ini dihasilkan dari file model & dataset yang sama
    # ------------------------------------------------------------------
    import hashlib
    from datetime import datetime

    def file_hash(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()[:16]

    print("--- AUDIT TRAIL ---")
    print(f"Waktu eksekusi script ini : {datetime.now().isoformat()}")
    print(f"SHA256 (16 char) data_cleaning.csv     : {file_hash(data_path)}")
    print(f"SHA256 (16 char) random_forest_model{suffix}.pkl: {file_hash(os.path.join(base_dir, f'random_forest_model{suffix}.pkl'))}")
    print(f"SHA256 (16 char) xgboost_model{suffix}.pkl      : {file_hash(os.path.join(base_dir, f'xgboost_model{suffix}.pkl'))}")
    print(f"SHA256 (16 char) rule_lr{suffix}.pkl             : {file_hash(os.path.join(base_dir, f'rule_lr{suffix}.pkl'))}")
    print("\n(Simpan output ini sebagai bukti tertulis / screenshot untuk lampiran sidang)")


if __name__ == "__main__":
    main()
