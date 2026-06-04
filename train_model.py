import os
import json
import pickle
import random
from typing import Any, Dict, List, Sequence, Tuple, cast

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

RANDOM_STATE = 12
TEST_SIZE = 0.2
TARGET_COL = "label"

# GA settings - Sesuai saran naik ke 5 agar estimasi F1 stabil
GA_POP_SIZE = 16
GA_N_GEN = 8
GA_CXPB = 0.7
GA_MUTPB = 0.3
GA_CV_SPLITS = 5

current_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(current_dir, "DataFiles", "data_cleaning.csv")

url_features_37 = [
    "length_url", "length_hostname", "ip", "nb_dots", "nb_hyphens", "nb_at", "nb_qm",
    "nb_and", "nb_eq", "nb_underscore", "nb_tilde", "nb_percent", "nb_slash", "nb_star",
    "nb_colon", "nb_comma", "nb_semicolumn", "nb_dollar", "nb_space", "nb_www", "nb_com",
    "nb_dslash", "http_in_path", "https_token", "ratio_digits_url", "ratio_digits_host",
    "punycode", "port", "tld_in_path", "tld_in_subdomain", "abnormal_subdomain",
    "nb_subdomains", "prefix_suffix", "random_domain", "shortening_service",
    "path_extension", "nb_redirection",
]

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def ensure_deap_creator(name: str, base_cls: Any, **kwargs: Any) -> Any:
    existing = getattr(creator, name, None)
    if existing is not None:
        return existing
    creator.create(name, base_cls, **kwargs)
    return getattr(creator, name)

def print_metrics(
    name: str, y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray | None = None,
) -> None:
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

def prep_X(df: pd.DataFrame, cols: List[str], idx: Sequence[Any] | NDArray[Any]) -> pd.DataFrame:
    X = df.loc[idx].reindex(columns=cols)
    return cast(pd.DataFrame, X.apply(pd.to_numeric, errors="coerce"))

def load_and_prepare_data(
    feature_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_file_path}")

    data = pd.read_csv(data_file_path)
    data.columns = data.columns.str.strip()

    # [FIX INI] Drop duplicates di awal untuk menghindari data leakage antar train/test
    data = data.drop_duplicates().copy()

    if TARGET_COL not in data.columns:
        raise ValueError(f"Kolom target '{TARGET_COL}' tidak ditemukan di dataset.")

    y_all = cast(pd.Series, pd.to_numeric(data[TARGET_COL], errors="coerce"))
    valid_mask = y_all.notna()
    data = data.loc[valid_mask].copy()
    y_all = y_all.loc[valid_mask].astype("int64")

    id_cols = [c for c in ["url"] if c in data.columns]
    all_features = [c for c in data.columns if c not in [TARGET_COL] + id_cols]

    webcontent_features_44 = [c for c in all_features if c not in url_features_37]
    hybrid_features_81 = url_features_37 + webcontent_features_44

    feature_candidates: List[str] = (
        hybrid_features_81 if feature_mode == "hybrid81" else url_features_37
    )

    train_idx, test_idx = train_test_split(
        data.index, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE,
    )

    X_train = cast(pd.DataFrame, prep_X(data, feature_candidates, train_idx))
    X_test = cast(pd.DataFrame, prep_X(data, feature_candidates, test_idx))

    mask = X_train.notna().any(axis=0)
    selected_features = X_train.columns[mask.to_numpy().astype(bool)].tolist()
    if not selected_features:
        raise ValueError("Tidak ada fitur valid untuk training setelah cleaning.")

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    X_train = X_train.dropna()
    X_test = X_test.dropna()

    y_train = cast(pd.Series, y_all.loc[X_train.index])
    y_test = cast(pd.Series, y_all.loc[X_test.index])

    print(f"\nMode: {feature_mode}")
    print(f"Selected actual features: {len(selected_features)}")
    print(f"Shape train/test: {X_train.shape} / {X_test.shape}")

    return X_train, X_test, y_train, y_test, selected_features

def tune_rf_ga(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict[str, Any], float]:
    print("\n[GA] Tuning Random Forest...")
    fitness_cls = getattr(creator, "FitnessMaxRF", None)
    if fitness_cls is None:
        ensure_deap_creator("FitnessMaxRF", base.Fitness, weights=(1.0,))
        fitness_cls = getattr(creator, "FitnessMaxRF")
    rf_ind_cls = cast(Any, ensure_deap_creator("IndividualRF", list, fitness=fitness_cls))

    cv = StratifiedKFold(n_splits=GA_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    def init_ind() -> Any:
        return rf_ind_cls([
            random.randint(100, 500), random.randint(3, 16),
            random.randint(2, 20), random.randint(1, 10), random.randint(0, 2),
        ])

    def decode(ind: Any) -> Dict[str, Any]:
        max_feat = {0: "sqrt", 1: "log2", 2: None}[int(clamp(round(ind[4]), 0, 2))]
        return {
            "n_estimators": int(clamp(round(ind[0]), 100, 500)),
            "max_depth": int(clamp(round(ind[1]), 3, 16)),
            "min_samples_split": int(clamp(round(ind[2]), 2, 20)),
            "min_samples_leaf": int(clamp(round(ind[3]), 1, 10)),
            "max_features": max_feat, "random_state": RANDOM_STATE, "n_jobs": -1,
        }

    def evaluate(ind: Any) -> Tuple[float]:
        params = decode(ind)
        model = RandomForestClassifier(**params)
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1).mean()
        return (float(score),)

    toolbox = base.Toolbox()
    toolbox.register("individual", init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=[100, 3, 2, 1, 0], up=[500, 16, 20, 10, 2], indpb=0.25)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=GA_POP_SIZE)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=GA_CXPB, mutpb=GA_MUTPB, ngen=GA_N_GEN, halloffame=hof, verbose=False)

    best_params = decode(hof[0])
    best_score = float(hof[0].fitness.values[0])
    return best_params, best_score

def tune_xgb_ga(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict[str, Any], float]:
    print("\n[GA] Tuning XGBoost...")
    fitness_cls = getattr(creator, "FitnessMaxXGB", None)
    if fitness_cls is None:
        ensure_deap_creator("FitnessMaxXGB", base.Fitness, weights=(1.0,))
        fitness_cls = getattr(creator, "FitnessMaxXGB")
    xgb_ind_cls = cast(Any, ensure_deap_creator("IndividualXGB", list, fitness=fitness_cls))

    cv = StratifiedKFold(n_splits=GA_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    def init_ind() -> Any:
        return xgb_ind_cls([
            random.randint(150, 500), random.uniform(0.01, 0.30), random.randint(3, 10),
            random.uniform(0.60, 1.00), random.uniform(0.60, 1.00), random.uniform(1.0, 10.0),
            random.uniform(0.0, 5.0), random.uniform(0.0, 5.0), random.uniform(0.1, 10.0),
        ])

    def decode(ind: Any) -> Dict[str, Any]:
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
            "eval_metric": "logloss", "random_state": RANDOM_STATE, "n_jobs": -1,
        }

    def evaluate(ind: Any) -> Tuple[float]:
        params = decode(ind)
        model = XGBClassifier(**params)
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1).mean()
        return (float(score),)

    def mutate(ind: Any, indpb: float = 0.25) -> Tuple[Any]:
        for i in range(len(ind)):
            if random.random() < indpb:
                if i in (0, 2): ind[i] += random.randint(-30, 30)
                else: ind[i] += random.uniform(-0.2, 0.2)
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
    algorithms.eaSimple(pop, toolbox, cxpb=GA_CXPB, mutpb=GA_MUTPB, ngen=GA_N_GEN, halloffame=hof, verbose=False)

    best_params = decode(hof[0])
    best_score = float(hof[0].fitness.values[0])
    return best_params, best_score

def train_and_save(feature_mode: str, suffix: str) -> None:
    X_train, X_test, y_train, y_test, selected_features = load_and_prepare_data(feature_mode)

    best_rf_params, best_rf_cv_f1 = tune_rf_ga(X_train, y_train)
    best_xgb_params, best_xgb_cv_f1 = tune_xgb_ga(X_train, y_train)

    rf_model = RandomForestClassifier(**best_rf_params)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_prob = np.asarray(rf_model.predict_proba(X_test))[:, 1]
    print_metrics(f"Random Forest (GA Tuned) {suffix}", y_test, rf_pred, rf_prob)

    xgb_model = XGBClassifier(**best_xgb_params)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = np.asarray(xgb_model.predict_proba(X_test))[:, 1]
    print_metrics(f"XGBoost (GA Tuned) {suffix}", y_test, xgb_pred, xgb_prob)

    # [FIX INI] Hapus class_weight="balanced" agar threshold tidak bergeser secara liar
    stack_model = StackingClassifier(
        estimators=[("rf", rf_model), ("xgb", xgb_model)],
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        n_jobs=-1,
    )
    stack_model.fit(X_train, y_train)
    stack_pred = stack_model.predict(X_test)
    stack_prob = np.asarray(stack_model.predict_proba(X_test))[:, 1]
    print_metrics(f"Stacking (RF+XGB+LR) {suffix}", y_test, stack_pred, stack_prob)

    # Ekspor model pkl
    with open(os.path.join(current_dir, f"random_forest_model{suffix}.pkl"), "wb") as f:
        pickle.dump(rf_model, f)
    with open(os.path.join(current_dir, f"xgboost_model{suffix}.pkl"), "wb") as f:
        pickle.dump(xgb_model, f)
    with open(os.path.join(current_dir, f"rule_lr{suffix}.pkl"), "wb") as f:
        pickle.dump(stack_model.final_estimator_, f)
        
    # Ekspor nama kolom txt
    with open(os.path.join(current_dir, f"feature_columns{suffix}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(selected_features))

    # [FIX INI] Menghasilkan berkas feature_columns_81.txt secara eksplisit untuk log Excel di app.py
    if suffix == "_81":
        with open(os.path.join(current_dir, "feature_columns_81.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(selected_features))

    tuning_report = {
        "random_state": RANDOM_STATE, "feature_mode": feature_mode,
        "ga": {
            "population": GA_POP_SIZE, "generations": GA_N_GEN, "cv_splits": GA_CV_SPLITS,
            "rf_best_cv_f1": best_rf_cv_f1, "xgb_best_cv_f1": best_xgb_cv_f1,
            "rf_best_params": best_rf_params, "xgb_best_params": best_xgb_params,
        },
    }
    with open(os.path.join(current_dir, f"ga_tuning_report{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(tuning_report, f, indent=2)

    print(f"✅ Model & metadata {suffix} tersimpan dengan sukses.")

def main() -> None:
    seed_everything(RANDOM_STATE)
    train_and_save("hybrid81", "_81")
    train_and_save("url37", "_37")

if __name__ == "__main__":
    main()