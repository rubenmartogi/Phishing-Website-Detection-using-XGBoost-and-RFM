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

ACTIVE_TRAIN_FEATURES = "hybrid81"  # "url37" atau "hybrid81"

# GA settings
GA_POP_SIZE = 16
GA_N_GEN = 8
GA_CXPB = 0.7
GA_MUTPB = 0.3
GA_CV_SPLITS = 3


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


def load_and_prepare_data():
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_file_path}")

    data = pd.read_csv(data_file_path)
    data.columns = data.columns.str.strip()

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

    feature_candidates = hybrid_features_81 if ACTIVE_TRAIN_FEATURES == "hybrid81" else url_features_37

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

    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    y_train = y_all.loc[train_idx]
    y_test = y_all.loc[test_idx]

    print(f"Mode: {ACTIVE_TRAIN_FEATURES}")
    print(f"URL features target: {len(url_features_37)}")
    print(f"Web-content features: {len(webcontent_features_44)}")
    print(f"Selected actual features: {len(selected_features)}")
    print(f"Shape train/test: {X_train.shape} / {X_test.shape}")

    return X_train, X_test, y_train, y_test, selected_features, train_medians


def tune_rf_ga(X_train, y_train):
    print("\n[GA] Tuning Random Forest...")

    ensure_deap_creator("FitnessMaxRF", base.Fitness, weights=(1.0,))
    rf_ind_cls = ensure_deap_creator("IndividualRF", list, fitness=creator.FitnessMaxRF)

    cv = StratifiedKFold(n_splits=GA_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    def init_ind():
        # [n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features_idx]
        return rf_ind_cls([
            random.randint(100, 500),
            random.randint(3, 16),
            random.randint(2, 20),
            random.randint(1, 10),
            random.randint(0, 2),
        ])

    def decode(ind):
        max_feat = {0: "sqrt", 1: "log2", 2: None}[int(clamp(round(ind[4]), 0, 2))]
        return {
            "n_estimators": int(clamp(round(ind[0]), 100, 500)),
            "max_depth": int(clamp(round(ind[1]), 3, 16)),
            "min_samples_split": int(clamp(round(ind[2]), 2, 20)),
            "min_samples_leaf": int(clamp(round(ind[3]), 1, 10)),
            "max_features": max_feat,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }

    def evaluate(ind):
        params = decode(ind)
        model = RandomForestClassifier(**params)
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1).mean()
        return (float(score),)

    toolbox = base.Toolbox()
    toolbox.register("individual", init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register(
        "mutate",
        tools.mutUniformInt,
        low=[100, 3, 2, 1, 0],
        up=[500, 16, 20, 10, 2],
        indpb=0.25
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
        # [n_estimators, learning_rate, max_depth, subsample, colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda]
        return xgb_ind_cls([
            random.randint(150, 500),
            random.uniform(0.01, 0.30),
            random.randint(3, 10),
            random.uniform(0.60, 1.00),
            random.uniform(0.60, 1.00),
            random.uniform(1.0, 10.0),
            random.uniform(0.0, 5.0),
            random.uniform(0.0, 5.0),
            random.uniform(0.1, 10.0),
        ])

    def decode(ind):
        return {
            "n_estimators": int(clamp(round(ind[0]), 150, 500)),
            "learning_rate": float(clamp(ind[1], 0.01, 0.30)),
            "max_depth": int(clamp(round(ind[2]), 3, 10)),
            "subsample": float(clamp(ind[3], 0.60, 1.00)),
            "colsample_bytree": float(clamp(ind[4], 0.60, 1.00)),
            "min_child_weight": float(clamp(ind[5], 1.0, 10.0)),
            "gamma": float(clamp(ind[6], 0.0, 5.0)),
            "reg_alpha": float(clamp(ind[7], 0.0, 5.0)),
            "reg_lambda": float(clamp(ind[8], 0.1, 10.0)),
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }

    def evaluate(ind):
        params = decode(ind)
        model = XGBClassifier(**params)
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1).mean()
        return (float(score),)

    def mutate(ind, indpb=0.25):
        # Mutasi ringan per gen, lalu clamp di decode
        for i in range(len(ind)):
            if random.random() < indpb:
                if i in (0, 2):
                    ind[i] += random.randint(-30, 30)
                else:
                    ind[i] += random.uniform(-0.2, 0.2)
        return (ind,)

    toolbox = base.Toolbox()
    toolbox.register("individual", init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate, indpb=0.3)
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


def main():
    seed_everything(RANDOM_STATE)

    X_train, X_test, y_train, y_test, selected_features, train_medians = load_and_prepare_data()

    # GA tune
    best_rf_params, best_rf_cv_f1 = tune_rf_ga(X_train, y_train)
    best_xgb_params, best_xgb_cv_f1 = tune_xgb_ga(X_train, y_train)

    # Train final models with best params
    rf_model = RandomForestClassifier(**best_rf_params)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    print_metrics("Random Forest (GA Tuned)", y_test, rf_pred, rf_prob)

    xgb_model = XGBClassifier(**best_xgb_params)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    print_metrics("XGBoost (GA Tuned)", y_test, xgb_pred, xgb_prob)

    # Stacking
    stack_model = StackingClassifier(
        estimators=[("rf", rf_model), ("xgb", xgb_model)],
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        stack_method="predict_proba",
        n_jobs=-1
    )
    stack_model.fit(X_train, y_train)
    stack_pred = stack_model.predict(X_test)
    stack_prob = stack_model.predict_proba(X_test)[:, 1]
    print_metrics("Stacking (RF+XGB+LR)", y_test, stack_pred, stack_prob)

    # Save model artifacts
    with open(os.path.join(current_dir, "random_forest_model.pkl"), "wb") as f:
        pickle.dump(rf_model, f)

    with open(os.path.join(current_dir, "xgboost_model.pkl"), "wb") as f:
        pickle.dump(xgb_model, f)

    with open(os.path.join(current_dir, "rule_lr.pkl"), "wb") as f:
        pickle.dump(stack_model.final_estimator_, f)

    with open(os.path.join(current_dir, "feature_columns.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(selected_features))

    with open(os.path.join(current_dir, "feature_medians.pkl"), "wb") as f:
        pickle.dump(train_medians.to_dict(), f)

    tuning_report = {
        "random_state": RANDOM_STATE,
        "active_train_features": ACTIVE_TRAIN_FEATURES,
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

    with open(os.path.join(current_dir, "ga_tuning_report.json"), "w", encoding="utf-8") as f:
        json.dump(tuning_report, f, indent=2)

    print("\n✅ Model & metadata standar tersimpan.")
    print("✅ Tuning report: ga_tuning_report.json")


if __name__ == "__main__":
    main()