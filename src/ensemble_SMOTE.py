from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split


RESULTS_DIR = Path("cv_grid_results_SMOTE")
N_FOLDS = 5
SEED = 42
EARLY_STOPPING_ROUNDS = 200

PARAM_GRID = {
    "depth": [6, 8, 10],
    "learning_rate": [0.025, 0.05, 0.075, 0.1],
    "l2_leaf_reg": [1, 4, 7, 10],
}

BASE_PARAMS = {
    "loss_function": "MultiClass",
    "eval_metric": "TotalF1:average=Macro",
    "iterations": 1200,
    "random_seed": SEED,
    "verbose": False,
    "task_type": "GPU",
    "devices": "0",
}

SMOTE_TARGETS = {"2": 6987, "5": 8031, "7": 522, "8": 4533, "3": 25000}
SMOTE_K_NEIGHBORS = 3


def make_smote(y, categorical_feature_names, n_full_train):
    """
    SMOTENC with the targets scaled to the size of y, so every CV fold gets the same
    class balance as the final refit. A target never drops below a class's current count.
    """

    scale = len(y) / n_full_train
    counts = y.value_counts()
    strategy = {c: max(int(n * scale), int(counts.get(c, 0))) for c, n in SMOTE_TARGETS.items()}

    missing = [c for c in strategy if c not in counts.index]
    if missing:
        raise ValueError(f"SMOTE_TARGETS has classes not present in y: {missing}")
    too_small = [c for c in strategy if counts[c] <= SMOTE_K_NEIGHBORS]
    if too_small:
        raise ValueError(f"Classes {too_small} have <= {SMOTE_K_NEIGHBORS} rows; lower SMOTE_K_NEIGHBORS.")

    # SMOTENC interpolates only the continuous columns; categorical values are taken from neighbours.
    return SMOTENC(
        categorical_features=categorical_feature_names,
        sampling_strategy=strategy,
        k_neighbors=SMOTE_K_NEIGHBORS,
        random_state=SEED,
    )


def make_folds(X, y, categorical_feature_names):
    """
    Split once, oversample only the training part of each fold, and reuse it for every grid combo.
    """

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        smote = make_smote(y.iloc[train_idx], categorical_feature_names, len(y))
        X_res, y_res = smote.fit_resample(X.iloc[train_idx], y.iloc[train_idx])
        train_pool = Pool(X_res, y_res, cat_features=categorical_feature_names)
        val_pool = Pool(X.iloc[val_idx], y.iloc[val_idx], cat_features=categorical_feature_names)

        folds.append((train_pool, val_pool))
        print(f"fold {fold}: {len(train_idx)} -> {len(y_res)} training rows after SMOTENC")

    return folds


def cross_validate(X, y, categorical_feature_names):
    folds = make_folds(X, y, categorical_feature_names)
    summary = []

    for values in product(*PARAM_GRID.values()):
        combo = dict(zip(PARAM_GRID, values))
        name = "_".join(f"{k}={v}" for k, v in combo.items())
        run_dir = RESULTS_DIR / name
        run_dir.mkdir(parents=True, exist_ok=True)

        curves = []
        for fold, (train_pool, val_pool) in enumerate(folds):
            model = CatBoostClassifier(**{**BASE_PARAMS, **combo})
            model.fit(
                train_pool,
                eval_set=val_pool,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                use_best_model=False,
            )
            model.save_model(str(run_dir / f"fold_{fold}.cbm"))

            val_metrics = model.get_evals_result()["validation"]
            metric_key = next(k for k in val_metrics if k.startswith("TotalF1"))
            curves.append(np.asarray(val_metrics[metric_key]))

        # Folds may stop early at different points; average over the iterations all of them completed.
        n_iter = min(len(c) for c in curves)
        stacked = np.vstack([c[:n_iter] for c in curves])
        cv_results = pd.DataFrame({
            "iterations": np.arange(n_iter),
            "test-TotalF1-mean": stacked.mean(axis=0),
            "test-TotalF1-std": stacked.std(axis=0),
            **{f"test-TotalF1-fold{i}": row for i, row in enumerate(stacked)},
        })
        cv_results.to_csv(run_dir / "cv_results.csv", index=False)

        best_iter = int(cv_results["test-TotalF1-mean"].idxmax())
        summary.append({**combo, "best_iteration": best_iter, "test_f1": cv_results.loc[best_iter, "test-TotalF1-mean"]})
        print(f"{name}: macro F1 {summary[-1]['test_f1']:.4f} at iteration {best_iter}")

    summary = pd.DataFrame(summary).sort_values("test_f1", ascending=False)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

    return summary


def refit_and_evaluate(X_train, y_train, X_test, y_test, categorical_feature_names, summary):
    best = summary.iloc[0]
    params = {
        **BASE_PARAMS,
        "depth": int(best["depth"]),
        "learning_rate": float(best["learning_rate"]),
        "l2_leaf_reg": float(best["l2_leaf_reg"]),
        "iterations": int(best["best_iteration"]) + 1,
    }

    # Oversample the full training set once; the test set is never resampled.
    smote = make_smote(y_train, categorical_feature_names, len(y_train))

    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"final refit: {len(y_train)} -> {len(y_res)} training rows after SMOTENC")

    model = CatBoostClassifier(**params)
    model.fit(Pool(X_res, y_res, cat_features=categorical_feature_names))
    model.save_model(str(RESULTS_DIR / "final_model.cbm"))

    test_pool = Pool(X_test, cat_features=categorical_feature_names)
    report = classification_report(y_test, model.predict(test_pool).ravel(), digits=4)
    (RESULTS_DIR / "test_report.txt").write_text(report)
    print(report)

    return model


def main():
    continuous_features = pd.read_csv("../dataset/pre-processed data/continuous_features.csv", index_col=False)
    categorical_features = pd.read_csv("../dataset/pre-processed data/categorical_features.csv", index_col=False)

    X = pd.concat([continuous_features, categorical_features], axis=1)
    y = pd.read_csv("../dataset/pre-processed data/target_feature.csv", index_col=False).iloc[:, 0].astype(str)
    # Make all categorical features 'str'.
    X['ct_state_ttl'] = X['ct_state_ttl'].astype(str)
    X['is_ftp_login'] = X['is_ftp_login'].astype(str)
    X['is_sm_ips_ports'] = X['is_sm_ips_ports'].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    categorical_feature_names = list(categorical_features.columns)
    summary = cross_validate(X_train, y_train, categorical_feature_names)
    print(summary.head())

    refit_and_evaluate(X_train, y_train, X_test, y_test, categorical_feature_names, summary)


if __name__ == "__main__":
    main()
