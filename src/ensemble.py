from itertools import product
from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


RESULTS_DIR = Path("cv_grid_results")

PARAM_GRID = {
    "depth": [6, 8, 10],
    "learning_rate": [0.025, 0.05, 0.075, 0.1],
    "l2_leaf_reg": [1, 4, 7, 10],
}

BASE_PARAMS = {
    "loss_function": "MultiClass",
    "eval_metric": "TotalF1:average=Macro;hints=skip_train~false",
    "iterations": 1200,
    "auto_class_weights": "SqrtBalanced",
    "random_seed": 42,
    "verbose": False,
    "task_type": "GPU",
    "devices": "0",
}


def cross_validate(X, y, categorical_feature_names):
    pool = Pool(X, y, cat_features=categorical_feature_names)
    summary = []

    for values in product(*PARAM_GRID.values()):
        combo = dict(zip(PARAM_GRID, values))
        name = "_".join(f"{k}={v}" for k, v in combo.items())
        run_dir = RESULTS_DIR / name
        run_dir.mkdir(parents=True, exist_ok=True)

        cv_results, models = cv(
            pool=pool,
            params={**BASE_PARAMS, **combo},
            fold_count=5,
            stratified=True,
            shuffle=True,
            partition_random_seed=42,
            early_stopping_rounds=200,
            return_models=True,
            verbose=False,
        )

        cv_results.to_csv(run_dir / "cv_results.csv", index=False)
        for fold, model in enumerate(models):
            model.save_model(str(run_dir / f"fold_{fold}.cbm"))

        test_col = [c for c in cv_results.columns if c.startswith("test-TotalF1") and c.endswith("-mean")][0]
        best_iter = cv_results[test_col].idxmax()
        summary.append({**combo, "best_iteration": best_iter, "test_f1": cv_results.loc[best_iter, test_col]})
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

    model = CatBoostClassifier(**params, cat_features=categorical_feature_names)
    model.fit(X_train, y_train)
    model.save_model(str(RESULTS_DIR / "final_model.cbm"))

    report = classification_report(y_test, model.predict(X_test).ravel(), digits=4)
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    categorical_feature_names = list(categorical_features.columns)
    summary = cross_validate(X_train, y_train, categorical_feature_names)
    print(summary.head())

    refit_and_evaluate(X_train, y_train, X_test, y_test, categorical_feature_names, summary)


if __name__ == "__main__":
    main()
