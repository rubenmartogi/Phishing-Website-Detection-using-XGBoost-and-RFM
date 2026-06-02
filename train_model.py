import os
import json
import pickle
import random
import numpy as np
import pandas as pd

from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

RANDOM_STATE = 12
TEST_SIZE = 0.2
TARGET_COL = "label"

# GA settings
GA_POP_SIZE = 16
GA_N_GEN = 8
GA_CXPB = 0.7
GA_MUTPB = 0.3
GA_CV_SPLITS = 5

current_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(current_dir, "DataFiles", "data_cleaning.csv")

url_features_37 = [
    "length_url", "length_hostname", "ip", "nb_dots", "nb_hyphens", "nb_at", "nb_qm", "nb_and",
    "nb_eq", "nb_underscore", "nb_tilde", "nb_percent", "nb_slash", "nb_star", "nb_colon",
    "nb_comma", "nb_semicolumn", "nb_dollar", "nb_space", "nb_www", "nb_com", "nb_dslash",
    "http_in_path", "https_token", "ratio_digits_url", "ratio_digits_host", "punycode", "port",
    "tld_in_path", "tld_in_subdomain", "abnormal_subdomain", "nb_subdomains", "prefix_suffix",
    "random_domain", "shortening_service", "path_extension", "nb_redirection"
]

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def ensure_deap_creator(name, base_cls, **kwargs):
    if hasattr(creator, name):
        return getattr(creator, name)
    creator.create(name, base_cls, **kwargs)
    return getattr(creator, name)

def print_metrics(name, y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan

    print(f"\n{name}")
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall   : {rec * 100:.2f}%")
    print(f"F1       : {f1 * 100:.2f}%")
    print(f"AUC      : {auc * 100:.2f}%" if not np.isnan(auc) else "AUC      : N/A")

def prep_X(df: pd.DataFrame, cols, idx) -> pd.DataFrame:
    X = df.loc[idx].reindex(columns=cols)
    return X.apply(pd.to_numeric, errors="coerce")

def load_and_prepare_data(feature_mode):
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_file_path}")

    data = pd.read_csv(data_file_path)
    data.columns = data.columns.str.strip()

    # PERBAIKAN 1: Bersihkan duplikasi data untuk mencegah Kebocoran Data (Data Leakage)
    if "url" in data.columns:
        data = data.drop_duplicates(subset=["url"]).copy()
    else:
        data = data.drop_duplicates().copy()

    if TARGET_COL not in data.columns:
        raise ValueError(f"Kolom target '{TARGET_COL}' tidak ditemukan di dataset.")

    y_all = pd.to_numeric(data[TARGET_COL], errors="coerce")
    valid_mask = y_all.notna()
    data = data.loc[valid_mask].copy()
    y_all = y_all.loc[valid_mask].astype("int64")

    id_cols = [c for c in ["url"] if c in data.columns]
    all_features = [c for c in data.columns if c not in [TARGET_COL] + id_cols]

    webcontent_features_44 = [c for c in all_features if c not in url_features_37]
    hybrid_features_81 = url_features_37 + webcontent_features_44

    if feature_mode == "hybrid81":
        feature_candidates = hybrid_features_81
    else:
        feature_candidates = url_features_37

    train_idx, test_idx = train_test_split(
        data.index,
        test_size=TEST_SIZE,
        stratify=y_all,
        random_state=RANDOM_STATE
    )

    X_train = prep_X(data, feature_candidates, train_idx)
    X_test = prep_X(data, feature_candidates, test_idx)

    selected_features = X_train.columns[X_train.notna().any()].tolist()
    if not selected_features:
        raise ValueError("Tidak ada fitur valid untuk training setelah cleaning.")

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    X_train = X_train.dropna()
    X_test = X_test.dropna()

    y_train = y_all.loc[X_train.index]
    y_test = y_all.loc[X_test.index]

    print(f"Mode: {feature_mode}")
    print(f"URL features target: {len(url_features_37)}")
    print(f"Web-content features: {len(webcontent_features_44)}")
    print(f"Selected actual features: {len(selected_features)}")
    print(f"Shape train/test: {X_train.shape} / {X_test.shape}")

    return X_train, X_test, y_train, y_test, selected_features

def tune_rf_ga(X_train, y_train):
    print("\n[GA] Tuning Random Forest...")

    ensure_deap_creator("FitnessMaxRF", base.Fitness, weights=(1.0,))
    rf_ind_cls = ensure_deap_creator("IndividualRF", list, fitness=creator.FitnessMaxRF)

    cv = StratifiedKFold(n_splits=GA_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    def init_ind():
        return rf_ind_cls([
            random.randint(200, 600),  # n_estimators: Jumlah pohon ideal (200 - 600)
            random.randint(10, 25),   # max_depth: Jangan terlalu dangkal. 10 - 25 bagus untuk 81 fitur
            random.randint(2, 10),    # min_samples_split: Diperkecil (2 - 10) agar pohon lebih sensitif terhadap pola
            random.randint(1, 4),     # min_samples_leaf: (1 - 4) mencegah overfitting di ujung daun
            random.randint(0, 1),     # max_features: Batasi ke 0: "sqrt" atau 1: "log2"
        ])

    def decode(ind):
        max_feat = {0: "sqrt", 1: "log2"}[int(clamp(round(ind[4]), 0, 1))]
        return {
            "n_estimators": int(clamp(round(ind[0]), 200, 600)),
            "max_depth": int(clamp(round(ind[1]), 10, 25)),
            "min_samples_split": int(clamp(round(ind[2]), 2, 10)),
            "min_samples_leaf": int(clamp(round(ind[3]), 1, 4)),
            "max_features": max_feat,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }

    def evaluate(ind):
        params = decode(ind)
        model = RandomForestClassifier(**params)
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1).mean()
        return (float(score),)

    # REVISI: Mengumpulkan inisialisasi toolbox di satu tempat terbawah agar scope fungsi batiniah lengkap
    toolbox = base.Toolbox()
    toolbox.register("individual", init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register(
        "mutate",
        tools.mutUniformInt,
        low=[200, 10, 2, 1, 0],
        up=[600, 25, 10, 4, 1],
        indpb=0.2
    )
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=GA_POP_SIZE)
    hof = tools.HallOfFame(1)

    algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=GA_CXPB,
        mutpb=GA_MUTPB,
        ngen=GA_N_GEN,
        halloffame=hof,
        verbose=False
    )

    best_params = decode(hof[0])
    best_score = float(hof[0].fitness.values[0])
    print("[GA] RF best CV F1:", round(best_score, 4))
    print("[GA] RF best params:", best_params)
    return best_params, best_score

def tune_xgb_ga(X_train, y_train):
    print("\n[GA] Tuning XGBoost...")

    ensure_deap_creator("FitnessMaxXGB", base.Fitness, weights=(1.0,))
    xgb_ind_cls = ensure_deap_creator("IndividualXGB", list, fitness=creator.FitnessMaxXGB)

    cv = StratifiedKFold(n_splits=GA_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    def init_ind():
        return xgb_ind_cls([
            random.randint(200, 600),       # n_estimators
            random.uniform(0.03, 0.15),     # learning_rate
            random.randint(5, 12),          # max_depth
            random.uniform(0.70, 1.00),     # subsample
            random.uniform(0.70, 1.00),     # colsample_bytree
            random.uniform(1.0, 5.0),       # min_child_weight
            random.uniform(0.0, 2.0),       # gamma
            random.uniform(0.0, 2.0),       # reg_alpha
            random.uniform(1.0, 5.0),       # reg_lambda
        ])

    def decode(ind):
        return {
            "n_estimators": int(clamp(round(ind[0]), 200, 600)),
            "learning_rate": float(clamp(ind[1], 0.03, 0.15)),
            "max_depth": int(clamp(round(ind[2]), 5, 12)),
            "subsample": float(clamp(ind[3], 0.70, 1.00)),
            "colsample_bytree": float(clamp(ind[4], 0.70, 1.00)),
            "min_child_weight": float(clamp(ind[5], 1.0, 5.0)),
            "gamma": float(clamp(ind[6], 0.0, 2.0)),
            "reg_alpha": float(clamp(ind[7], 0.0, 2.0)),
            "reg_lambda": float(clamp(ind[8], 1.0, 5.0)),
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }

    def evaluate(ind):
        params = decode(ind)
        model = XGBClassifier(**params)
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1).mean()
        return (float(score),)

    def mutate_xgb(ind, indpb=0.2):
        for i in range(len(ind)):
            if random.random() < indpb:
                if i in (0, 2):
                    ind[i] += random.randint(-30, 30)
                else:
                    ind[i] += random.uniform(-0.05, 0.05)
        return (ind,)

    # REVISI: Mengumpulkan inisialisasi toolbox di satu tempat terbawah agar scope fungsi batiniah lengkap
    toolbox = base.Toolbox()
    toolbox.register("individual", init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate_xgb, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=GA_POP_SIZE)
    hof = tools.HallOfFame(1)

    algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=GA_CXPB,
        mutpb=GA_MUTPB,
        ngen=GA_N_GEN,
        halloffame=hof,
        verbose=False
    )

    best_params = decode(hof[0])
    best_score = float(hof[0].fitness.values[0])
    print("[GA] XGB best CV F1:", round(best_score, 4))
    print("[GA] XGB best params:", best_params)
    return best_params, best_score

def train_and_save(feature_mode, suffix):
    X_train, X_test, y_train, y_test, selected_features = load_and_prepare_data(feature_mode)

    best_rf_params, best_rf_cv_f1 = tune_rf_ga(X_train, y_train)
    best_xgb_params, best_xgb_cv_f1 = tune_xgb_ga(X_train, y_train)

    rf_model = RandomForestClassifier(**best_rf_params)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    print_metrics(f"Random Forest (GA Tuned) {suffix}", y_test, rf_pred, rf_prob)

    xgb_model = XGBClassifier(**best_xgb_params)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    print_metrics(f"XGBoost (GA Tuned) {suffix}", y_test, xgb_pred, xgb_prob)

    # PERBAIKAN 2: Hapus class_weight="balanced" agar threshold Logistic Regression tidak bergeser ekstrim/paranoid
    stack_model = StackingClassifier(
        estimators=[("rf", rf_model), ("xgb", xgb_model)],
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        n_jobs=-1
    )
    stack_model.fit(X_train, y_train)
    stack_pred = stack_model.predict(X_test)
    stack_prob = stack_model.predict_proba(X_test)[:, 1]
    print_metrics(f"Stacking (RF+XGB+LR) {suffix}", y_test, stack_pred, stack_prob)

    # Save model artifacts
    with open(os.path.join(current_dir, f"random_forest_model{suffix}.pkl"), "wb") as f:
        pickle.dump(rf_model, f)
    with open(os.path.join(current_dir, f"xgboost_model{suffix}.pkl"), "wb") as f:
        pickle.dump(xgb_model, f)
    with open(os.path.join(current_dir, f"rule_lr{suffix}.pkl"), "wb") as f:
        pickle.dump(stack_model.final_estimator_, f)
    with open(os.path.join(current_dir, f"feature_columns{suffix}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(selected_features))

    tuning_report = {
        "random_state": RANDOM_STATE,
        "feature_mode": feature_mode,
        "ga": {
            "population": GA_POP_SIZE,
            "generations": GA_N_GEN,
            "cv_splits": GA_CV_SPLITS,
            "rf_best_cv_f1": best_rf_cv_f1,
            "xgb_best_cv_f1": best_xgb_cv_f1,
            "rf_best_params": best_rf_params,
            "xgb_best_params": best_xgb_params,
        }
    }
    with open(os.path.join(current_dir, f"ga_tuning_report{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(tuning_report, f, indent=2)

    print(f"\n✅ Model & metadata {suffix} tersimpan dengan aman.")

def main():
    seed_everything(RANDOM_STATE)
    
    # PERBAIKAN 3: Gunakan suffix "_hybrid81" dan "_url37" agar 100% klop dengan pencarian di app.py
    # Train hybrid81
    print("="*50 + "\nTRAINING MODE: HYBRID 81\n" + "="*50)
    train_and_save("hybrid81", "_hybrid81")
    
    # Train url37
    print("\n" + "="*50 + "\nTRAINING MODE: URL 37\n" + "="*50)
    train_and_save("url37", "_url37")

if __name__ == "__main__":
    main()