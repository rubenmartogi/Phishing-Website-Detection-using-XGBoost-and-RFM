"""
evaluate_performance.py
=======================
Script evaluasi perbandingan kinerja model:
  - Random Forest (37 fitur URL)
  - XGBoost      (37 fitur URL)
  - Stacking RF+XGB (37 fitur URL)
  - Random Forest (81 fitur Hybrid)
  - XGBoost      (81 fitur Hybrid)
  - Stacking RF+XGB (81 fitur Hybrid)

Output:
  - Tabel metrik di terminal
  - results/01_metric_comparison.png   → Bar chart Acc/Prec/Rec/F1/AUC
  - results/02_roc_curves.png          → ROC Curve semua model
  - results/03_pr_curves.png           → Precision-Recall Curve
  - results/04_confusion_matrices.png  → Confusion Matrix 6 model
  - results/05_radar_chart.png         → Radar Chart perbandingan
  - results/performance_summary.csv    → Tabel ringkasan CSV
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, confusion_matrix, average_precision_score
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ── Konfigurasi ───────────────────────────────────────────────────────────────
RANDOM_STATE = 12
TEST_SIZE    = 0.2
TARGET_COL   = "label"
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE_DIR, "DataFiles", "data_cleaning.csv")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Warna per model — 2 keluarga warna bergradasi (terang → gelap)
# Biru  : Mode 37-fitur (URL-based)   | Teal : Mode 81-fitur (Hybrid)
COLORS = {
    "RF-37":       "#90CAF9",   # Biru muda      (Blue 200)
    "XGB-37":      "#42A5F5",   # Biru sedang    (Blue 400)
    "Stack-37":    "#1565C0",   # Biru tua       (Blue 800)
    "Hybrid-37":   "#0D47A1",   # Biru paling tua(Blue 900)
    "RF-81":       "#80CBC4",   # Teal muda      (Teal 200)
    "XGB-81":      "#26A69A",   # Teal sedang    (Teal 400)
    "Stack-81":    "#00796B",   # Teal tua       (Teal 700)
    "Hybrid-81":   "#004D40",   # Teal paling tua(Teal 900)
}
MODEL_ORDER = ["RF-37", "XGB-37", "Stack-37", "Hybrid-37",
               "RF-81", "XGB-81", "Stack-81", "Hybrid-81"]

URL_FEATURES_37 = [
    "length_url","length_hostname","ip","nb_dots","nb_hyphens","nb_at","nb_qm",
    "nb_and","nb_eq","nb_underscore","nb_tilde","nb_percent","nb_slash","nb_star",
    "nb_colon","nb_comma","nb_semicolumn","nb_dollar","nb_space","nb_www","nb_com",
    "nb_dslash","http_in_path","https_token","ratio_digits_url","ratio_digits_host",
    "punycode","port","tld_in_path","tld_in_subdomain","abnormal_subdomain",
    "nb_subdomains","prefix_suffix","random_domain","shortening_service",
    "path_extension","nb_redirection",
]

# ── Helper ────────────────────────────────────────────────────────────────────
def load_model(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model tidak ditemukan: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def load_cols(filename):
    path = os.path.join(BASE_DIR, filename)
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def prep_data(feature_mode):
    print(f"  Loading data (mode={feature_mode})...")
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()
    data = data.drop_duplicates().copy()

    y_all = pd.to_numeric(data[TARGET_COL], errors="coerce")
    data  = data.loc[y_all.notna()].copy()
    y_all = y_all.loc[y_all.notna()].astype("int64")

    id_cols = [c for c in ["url"] if c in data.columns]
    all_feats = [c for c in data.columns if c not in [TARGET_COL] + id_cols]

    if feature_mode == "37":
        cols = [c for c in URL_FEATURES_37 if c in all_feats]
    else:
        web44  = [c for c in all_feats if c not in URL_FEATURES_37]
        cols   = URL_FEATURES_37 + web44
        cols   = [c for c in cols if c in all_feats]

    _, test_idx = train_test_split(
        data.index, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE
    )
    X_test = data.loc[test_idx, cols].apply(pd.to_numeric, errors="coerce").dropna()
    y_test = y_all.loc[X_test.index]
    return X_test, y_test

def evaluate_model(model, X, y, name):
    """Kembalikan dict metrik lengkap."""
    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception:
        proba = None

    pred = model.predict(X)

    acc  = accuracy_score(y, pred)
    prec = precision_score(y, pred, zero_division=0)
    rec  = recall_score(y, pred, zero_division=0)
    f1   = f1_score(y, pred, zero_division=0)
    auc  = roc_auc_score(y, proba) if proba is not None else np.nan
    ap   = average_precision_score(y, proba) if proba is not None else np.nan
    cm   = confusion_matrix(y, pred)

    fpr = tpr = thresh = pr_prec = pr_rec = None
    if proba is not None:
        fpr, tpr, thresh = roc_curve(y, proba)
        pr_prec, pr_rec, _ = precision_recall_curve(y, proba)

    print(f"  [{name}]  Acc={acc*100:.2f}%  Prec={prec*100:.2f}%  "
          f"Rec={rec*100:.2f}%  F1={f1*100:.2f}%  AUC={auc*100:.2f}%")

    return {
        "name": name, "acc": acc, "prec": prec, "rec": rec,
        "f1": f1, "auc": auc, "ap": ap,
        "cm": cm, "fpr": fpr, "tpr": tpr,
        "pr_prec": pr_prec, "pr_rec": pr_rec,
    }

# ── Rule-Based prefilter (vectorized dari fitur dataset) ─────────────────────
def rule_flag_from_df(X_df: pd.DataFrame) -> np.ndarray:
    """
    Terapkan aturan rule-based (sama dengan app.py rule_based_eval) secara
    vectorized pada DataFrame fitur. Kembalikan array 0/1 per baris.
      1 = rule mendeteksi phishing
      0 = rule tidak mendeteksi (lolos ke ML)
    """
    def col(name, default=0):
        return X_df[name].fillna(default).values if name in X_df.columns else np.full(len(X_df), default)

    # Very important (salah satu saja → Phishing)
    vi = (
        (col("suspicious_tld")   == 1) |
        (col("nb_at")            >= 1) |
        (col("ip")               == 1) |
        (col("nb_underscore")    >  3)
    )

    # Important (skor ×2)
    imp = (
        (col("ratio_digits_url") >  0.3).astype(int) +
        (col("nb_subdomains")    >  3  ).astype(int) +
        (col("nb_percent")       >  5  ).astype(int) +
        (col("nb_tilde")         >= 1  ).astype(int) +
        (col("nb_semicolumn")    >= 1  ).astype(int) +
        (col("nb_star")          >= 1  ).astype(int) +
        (col("nb_comma")         >= 1  ).astype(int) +
        (col("random_domain")    == 1  ).astype(int)
    )

    # Less important (skor ×1)
    less = (
        (col("length_hostname")  > 30).astype(int) +
        (col("nb_dollar")        >= 1).astype(int) +
        (col("nb_qm")            >  2).astype(int) +
        (col("nb_colon")         >  1).astype(int) +
        (col("nb_eq")            >  8).astype(int) +
        (col("nb_dots")          >  4).astype(int) +
        (col("nb_slash")         >  7).astype(int) +
        (col("nb_and")           >  3).astype(int) +
        (col("nb_hyphens")       >  3).astype(int) +
        (col("http_in_path")     == 1).astype(int) +
        (col("https_token")      == 1).astype(int) +
        (col("port")             == 1).astype(int) +
        (col("shortening_service") == 1).astype(int)
    )

    risk_score = 2 * imp + less
    rule_phishing = vi | (risk_score >= 5)
    return rule_phishing.astype(int)


def evaluate_from_arrays(pred: np.ndarray, proba: np.ndarray,
                         y: pd.Series, name: str) -> dict:
    """Evaluasi dari array prediksi & probabilitas yang sudah dihitung."""
    acc  = accuracy_score(y, pred)
    prec = precision_score(y, pred, zero_division=0)
    rec  = recall_score(y, pred, zero_division=0)
    f1   = f1_score(y, pred, zero_division=0)
    auc  = roc_auc_score(y, proba)
    ap   = average_precision_score(y, proba)
    cm   = confusion_matrix(y, pred)
    fpr, tpr, _ = roc_curve(y, proba)
    pr_prec, pr_rec, _ = precision_recall_curve(y, proba)

    print(f"  [{name}]  Acc={acc*100:.2f}%  Prec={prec*100:.2f}%  "
          f"Rec={rec*100:.2f}%  F1={f1*100:.2f}%  AUC={auc*100:.2f}%")

    return {
        "name": name, "acc": acc, "prec": prec, "rec": rec,
        "f1": f1, "auc": auc, "ap": ap,
        "cm": cm, "fpr": fpr, "tpr": tpr,
        "pr_prec": pr_prec, "pr_rec": pr_rec,
    }


# ── Evaluasi semua model ───────────────────────────────────────────────────────
def run_evaluation():
    results = {}

    # ── Mode 37 fitur ──
    print("\n=== Mode 37 Fitur (URL-based) ===")
    X37, y37 = prep_data("37")
    cols37   = load_cols("feature_columns_37.txt")
    cols37   = [c for c in cols37 if c in X37.columns]
    X37s     = X37[cols37]

    rf37  = load_model("random_forest_model_37.pkl")
    xgb37 = load_model("xgboost_model_37.pkl")
    lr37  = load_model("rule_lr_37.pkl")          # meta LR

    results["RF-37"]    = evaluate_model(rf37,  X37s, y37, "RF-37")
    results["XGB-37"]   = evaluate_model(xgb37, X37s, y37, "XGB-37")

    # Stacking: meta LR menerima [rf_prob, xgb_prob]
    rf_p37  = rf37.predict_proba(X37s)[:, 1]
    xgb_p37 = xgb37.predict_proba(X37s)[:, 1]
    n_in37  = int(getattr(lr37, "n_features_in_", 2))
    meta_X37 = np.column_stack([rf_p37, xgb_p37])[:, :n_in37]
    stack_p37 = lr37.predict_proba(meta_X37)[:, 1]
    results["Stack-37"] = evaluate_model(lr37, meta_X37, y37, "Stack-37")

    # Hybrid Rule + Stacking (37)
    rule37      = rule_flag_from_df(X37s)          # 1 = rule tangkap phishing
    hybrid_p37  = np.where(rule37 == 1, np.maximum(stack_p37, 0.95), stack_p37)
    hybrid_pred37 = (hybrid_p37 >= 0.6).astype(int)
    results["Hybrid-37"] = evaluate_from_arrays(hybrid_pred37, hybrid_p37, y37, "Hybrid-37")

    # ── Mode 81 fitur ──
    print("\n=== Mode 81 Fitur (Hybrid) ===")
    X81, y81 = prep_data("81")
    cols81   = load_cols("feature_columns_81.txt")
    cols81   = [c for c in cols81 if c in X81.columns]
    X81s     = X81[cols81]

    rf81  = load_model("random_forest_model_81.pkl")
    xgb81 = load_model("xgboost_model_81.pkl")
    lr81  = load_model("rule_lr_81.pkl")

    results["RF-81"]    = evaluate_model(rf81,  X81s, y81, "RF-81")
    results["XGB-81"]   = evaluate_model(xgb81, X81s, y81, "XGB-81")

    rf_p81  = rf81.predict_proba(X81s)[:, 1]
    xgb_p81 = xgb81.predict_proba(X81s)[:, 1]
    n_in81  = int(getattr(lr81, "n_features_in_", 2))
    meta_X81 = np.column_stack([rf_p81, xgb_p81])[:, :n_in81]
    stack_p81 = lr81.predict_proba(meta_X81)[:, 1]
    results["Stack-81"] = evaluate_model(lr81, meta_X81, y81, "Stack-81")

    # Hybrid Rule + Stacking (81)
    rule81      = rule_flag_from_df(X81s)
    hybrid_p81  = np.where(rule81 == 1, np.maximum(stack_p81, 0.95), stack_p81)
    hybrid_pred81 = (hybrid_p81 >= 0.6).astype(int)
    results["Hybrid-81"] = evaluate_from_arrays(hybrid_pred81, hybrid_p81, y81, "Hybrid-81")

    return results

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1: Bar Chart Metrik Perbandingan
# ══════════════════════════════════════════════════════════════════════════════
def plot_metric_comparison(results):
    # AUC tidak ditampilkan di bar chart — hanya di ROC curve
    metrics  = ["acc", "prec", "rec", "f1"]
    mlabels  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    n_metrics = len(metrics)
    n_models  = len(MODEL_ORDER)
    x = np.arange(n_metrics)
    width = 0.10

    fig, ax = plt.subplots(figsize=(15, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")

    for i, mkey in enumerate(MODEL_ORDER):
        r = results[mkey]
        vals = [r[m] * 100 for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=mkey,
                      color=COLORS[mkey], alpha=0.9, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=6.5, color="#222", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(mlabels, fontsize=13, color="#222")
    ax.set_ylabel("Score (%)", fontsize=13, color="#222")
    ax.set_title("Perbandingan Kinerja Model", fontsize=16,
                 fontweight="bold", color="#111", pad=15)
    ax.set_ylim(75, 103)
    ax.tick_params(colors="#222")
    ax.spines[:].set_color("#ccc")
    ax.yaxis.grid(True, color="#ddd", linestyle="--", alpha=0.8)
    ax.set_axisbelow(True)
    # Legend di luar grafik sisi kanan agar tidak menutupi bar
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                       fontsize=10, facecolor="white", edgecolor="#ccc",
                       labelcolor="#222", borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    out = os.path.join(RESULTS_DIR, "01_metric_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ Saved: {out}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2: ROC Curves
# ══════════════════════════════════════════════════════════════════════════════
def plot_roc_curves(results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("white")

    groups = [
        ("37 Fitur URL", ["RF-37", "XGB-37", "Stack-37", "Hybrid-37"]),
        ("81 Fitur Hybrid", ["RF-81", "XGB-81", "Stack-81", "Hybrid-81"]),
    ]

    for ax, (title, keys) in zip(axes, groups):
        ax.set_facecolor("#f8f9fa")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1, label="Random (AUC=0.50)")
        for k in keys:
            r = results[k]
            if r["fpr"] is not None:
                ax.plot(r["fpr"], r["tpr"], color=COLORS[k], lw=2.5,
                        label=f"{k}  (AUC={r['auc']:.4f})")
        ax.set_xlabel("False Positive Rate", color="#222", fontsize=11)
        ax.set_ylabel("True Positive Rate", color="#222", fontsize=11)
        ax.set_title(f"ROC Curve – {title}", color="#111", fontsize=13, fontweight="bold")
        ax.tick_params(colors="#222")
        ax.spines[:].set_color("#ccc")
        ax.grid(color="#ddd", linestyle="--", alpha=0.8)
        ax.legend(facecolor="white", edgecolor="#ccc", labelcolor="#222", fontsize=10)

    plt.suptitle("ROC Curve Perbandingan Model", fontsize=15,
                 color="#111", fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "02_roc_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ Saved: {out}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3: Precision-Recall Curves
# ══════════════════════════════════════════════════════════════════════════════
def plot_pr_curves(results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("white")

    groups = [
        ("37 Fitur URL", ["RF-37", "XGB-37", "Stack-37", "Hybrid-37"]),
        ("81 Fitur Hybrid", ["RF-81", "XGB-81", "Stack-81", "Hybrid-81"]),
    ]

    for ax, (title, keys) in zip(axes, groups):
        ax.set_facecolor("#f8f9fa")
        for k in keys:
            r = results[k]
            if r["pr_prec"] is not None:
                ax.plot(r["pr_rec"], r["pr_prec"], color=COLORS[k], lw=2.5,
                        label=f"{k}  (AP={r['ap']:.4f})")
        ax.set_xlabel("Recall", color="#222", fontsize=11)
        ax.set_ylabel("Precision", color="#222", fontsize=11)
        ax.set_title(f"Precision-Recall Curve – {title}",
                     color="#111", fontsize=13, fontweight="bold")
        ax.tick_params(colors="#222")
        ax.spines[:].set_color("#ccc")
        ax.grid(color="#ddd", linestyle="--", alpha=0.8)
        ax.legend(facecolor="white", edgecolor="#ccc", labelcolor="#222", fontsize=10)

    plt.suptitle("Precision-Recall Curve Perbandingan Model", fontsize=15,
                 color="#111", fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "03_pr_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ Saved: {out}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4: Confusion Matrices (6 model)
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrices(results):
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.patch.set_facecolor("white")
    fig.suptitle("Confusion Matrix", fontsize=17,
                 color="#111", fontweight="bold", y=1.01)

    for ax, mkey in zip(axes.flat, MODEL_ORDER):
        cm = results[mkey]["cm"]
        total = cm.sum()
        ax.set_facecolor("white")

        # Heatmap manual
        cmap = plt.cm.Blues
        im = ax.imshow(cm, cmap=cmap, aspect="auto")

        # Annotasi tiap sel
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = val / total * 100
                ax.text(j, i, f"{val}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=13, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() * 0.5 else "#222")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Prediksi\nBenign", "Prediksi\nPhishing"],
                           color="#222", fontsize=10)
        ax.set_yticklabels(["Aktual\nBenign", "Aktual\nPhishing"],
                           color="#222", fontsize=10)
        ax.tick_params(colors="#222")
        ax.spines[:].set_color("#ccc")
        ax.set_title(mkey, color="#111", fontsize=13, fontweight="bold")

        # Label TN/FP/FN/TP
        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i - 0.35, labels[i][j],
                        ha="center", va="center", fontsize=8,
                        color="#555")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "04_confusion_matrices.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ Saved: {out}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5: Radar Chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_radar_chart(results):
    metrics  = ["acc", "prec", "rec", "f1", "auc"]
    mlabels  = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # tutup lingkaran

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")

    groups = [
        ("37 Fitur URL", ["RF-37", "XGB-37", "Stack-37", "Hybrid-37"]),
        ("81 Fitur Hybrid", ["RF-81", "XGB-81", "Stack-81", "Hybrid-81"]),
    ]

    for ax, (title, keys) in zip(axes, groups):
        ax.set_facecolor("#f8f9fa")
        ax.spines["polar"].set_color("#ccc")
        ax.grid(color="#ddd", linestyle="--", alpha=0.8)

        for k in keys:
            r  = results[k]
            vals = [r[m] for m in metrics]
            vals += vals[:1]
            ax.plot(angles, vals, color=COLORS[k], lw=2.5, label=k)
            ax.fill(angles, vals, color=COLORS[k], alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(mlabels, color="#222", fontsize=11)
        ax.set_ylim(0.75, 1.0)
        ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
        ax.set_yticklabels(["80%", "85%", "90%", "95%", "100%"],
                           color="#666", fontsize=8)
        ax.tick_params(colors="#222")
        ax.set_title(f"Radar Chart – {title}", color="#111",
                     fontsize=13, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
                  facecolor="white", edgecolor="#ccc",
                  labelcolor="#222", fontsize=10)

    plt.suptitle("Radar Chart Perbandingan Model", fontsize=16,
                 color="#111", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "05_radar_chart.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ Saved: {out}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6: Summary Heatmap Table
# ══════════════════════════════════════════════════════════════════════════════
def plot_summary_table(results):
    metrics = ["acc", "prec", "rec", "f1", "auc"]
    mlabels = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]

    data = []
    for k in MODEL_ORDER:
        r = results[k]
        data.append([r[m] * 100 for m in metrics])

    arr = np.array(data)
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Normalize per kolom (metrik) untuk warna
    arr_norm = (arr - arr.min(axis=0)) / (arr.max(axis=0) - arr.min(axis=0) + 1e-9)
    im = ax.imshow(arr_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Teks nilai
    for i in range(len(MODEL_ORDER)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{arr[i, j]:.2f}%", ha="center", va="center",
                    fontsize=13, fontweight="bold", color="#111")

    ax.set_xticks(range(len(mlabels)))
    ax.set_xticklabels(mlabels, color="#222", fontsize=12)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER, color="#222", fontsize=12)
    ax.tick_params(colors="#222")
    ax.spines[:].set_color("#ccc")
    ax.set_title("Ringkasan Metrik Kinerja Model (Heatmap)", color="#111",
                 fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "06_summary_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ Saved: {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Simpan CSV ringkasan
# ══════════════════════════════════════════════════════════════════════════════
def save_csv(results):
    rows = []
    for k in MODEL_ORDER:
        r = results[k]
        tn, fp, fn, tp = r["cm"].ravel()
        rows.append({
            "Model": k,
            "Accuracy (%)":  round(r["acc"]  * 100, 4),
            "Precision (%)": round(r["prec"] * 100, 4),
            "Recall (%)":    round(r["rec"]  * 100, 4),
            "F1-Score (%)":  round(r["f1"]   * 100, 4),
            "AUC-ROC (%)":   round(r["auc"]  * 100, 4),
            "Avg Precision": round(float(r["ap"]), 4),
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        })
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "performance_summary.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n  ✅ CSV saved: {out}")
    print("\n" + "="*80)
    print(df.to_string(index=False))
    print("="*80)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 7: AUC Bar Chart Tersendiri
# ══════════════════════════════════════════════════════════════════════════════
def plot_auc_comparison(results):
    """
    Grafik batang horizontal khusus AUC-ROC, diurutkan dari tertinggi,
    dengan garis referensi dan anotasi nilai di setiap bar.
    """
    # Urutkan model berdasarkan AUC dari tertinggi
    model_auc = [(k, results[k]["auc"] * 100) for k in MODEL_ORDER]
    model_auc_sorted = sorted(model_auc, key=lambda x: x[1], reverse=True)
    names = [m[0] for m in model_auc_sorted]
    aucs  = [m[1] for m in model_auc_sorted]
    colors = [COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")

    y_pos = np.arange(len(names))
    bars  = ax.barh(y_pos, aucs, color=colors, alpha=0.88,
                    edgecolor="white", linewidth=0.6, height=0.6)

    # Anotasi nilai di ujung bar
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_width() - 0.15, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", ha="right", va="center",
                fontsize=11, fontweight="bold", color="white")

    # Garis referensi
    ax.axvline(x=95, color="#e74c3c", linestyle="--", lw=1.5, alpha=0.7,
               label="Threshold Sangat Baik (95%)")
    ax.axvline(x=99, color="#27ae60", linestyle="--", lw=1.5, alpha=0.7,
               label="Threshold Hampir Sempurna (99%)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=12, color="#222")
    ax.set_xlabel("AUC-ROC Score (%)", fontsize=13, color="#222")
    ax.set_title("Perbandingan AUC-ROC Antar Model", fontsize=16,
                 fontweight="bold", color="#111", pad=15)
    ax.set_xlim(90, 101)
    ax.tick_params(colors="#222")
    ax.spines[:].set_color("#ccc")
    ax.xaxis.grid(True, color="#ddd", linestyle="--", alpha=0.8)
    ax.set_axisbelow(True)
    ax.invert_yaxis()   # Model terbaik di atas
    ax.legend(loc="lower right", fontsize=10,
              facecolor="white", edgecolor="#ccc", labelcolor="#333")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "07_auc_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out}")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  EVALUASI PERBANDINGAN KINERJA MODEL PHISHING DETECTION")
    print("="*60)

    print("\n[1/8] Loading & evaluating models...")
    results = run_evaluation()

    print("\n[2/8] Plotting bar chart metrik (Acc/Prec/Rec/F1)...")
    plot_metric_comparison(results)

    print("\n[3/8] Plotting ROC curves...")
    plot_roc_curves(results)

    print("\n[4/8] Plotting Precision-Recall curves...")
    plot_pr_curves(results)

    print("\n[5/8] Plotting confusion matrices...")
    plot_confusion_matrices(results)

    print("\n[6/8] Plotting radar chart...")
    plot_radar_chart(results)

    print("\n[7/8] Plotting summary heatmap & saving CSV...")
    plot_summary_table(results)
    save_csv(results)

    print("\n[8/8] Plotting AUC bar chart tersendiri...")
    plot_auc_comparison(results)

    print(f"\nSemua grafik tersimpan di folder: {RESULTS_DIR}")
    print("   Buka file PNG satu per satu atau buka folder results/\n")
