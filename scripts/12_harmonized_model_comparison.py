"""Harmonized model comparison using the reference validation protocol.

Purpose: compare XGBoost, ExtraTrees, Random Forest, and CatBoost using
identical reference data, predictors, split selection, preprocessing,
normalization, and performance definitions.

Usage:
  python scripts/12_harmonized_model_comparison.py --smoke
  python scripts/12_harmonized_model_comparison.py

Output: CSV comparison table, split-level details, split audit, JSON manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("reference_validation", HERE / "04_validation_ladder.py")
assert spec and spec.loader
step04 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = step04
spec.loader.exec_module(step04)

MODELS = (
    "XGBoost_base",
    "ExtraTrees_deep",
    "RandomForest_deep",
    "CatBoost_base",
)
LABELS = {
    "XGBoost_base": "XGBoost",
    "ExtraTrees_deep": "ExtraTrees",
    "RandomForest_deep": "Random Forest",
    "CatBoost_base": "CatBoost",
}
MODE = "direct_forced_closed"
EXPECTED_R2 = {"Random-row": 0.871, "Grouped": 0.385, "LOSO pooled": 0.274}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_specs(df: pd.DataFrame, *, smoke: bool):
    n = len(df)
    stratify = None
    if "feedstock_class" in df.columns and df["feedstock_class"].notna().all():
        if df["feedstock_class"].value_counts().min() >= 2:
            stratify = df["feedstock_class"]

    for seed in step04.RANDOM_SEEDS[:1 if smoke else None]:
        tr, te = train_test_split(
            np.arange(n), test_size=step04.TEST_SIZE,
            random_state=seed, stratify=stratify,
        )
        yield "Random-row", f"RANDOM_seed_{seed:03d}", np.asarray(tr), np.asarray(te), None

    audit = step04.select_similarity_aware_group_splits(df)
    audit.to_csv(step04.output_file(
        "12_harmonized_group_split_audit_smoke.csv" if smoke else "12_harmonized_group_split_audit.csv"
    ), index=False)
    for _, row in audit.head(1 if smoke else len(audit)).iterrows():
        test_groups = set(str(row["test_groups_string"]).split(" ;; "))
        test_mask = df["validation_group_id"].astype(str).isin(test_groups).to_numpy()
        tr, te = np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
        if np.intersect1d(df.iloc[tr]["validation_group_id"], df.iloc[te]["validation_group_id"]).size:
            raise AssertionError("Group leakage in Step-04 partition")
        yield "Grouped", str(row["selected_split_id"]), tr, te, float(row["split_score"])

    studies = sorted(df["study_id"].astype(str).unique())
    for study in studies[:1 if smoke else None]:
        test_mask = df["study_id"].astype(str).eq(study).to_numpy()
        tr, te = np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
        if np.intersect1d(df.iloc[tr]["study_id"], df.iloc[te]["study_id"]).size:
            raise AssertionError("Study leakage in Step-04 LOSO")
        yield "LOSO fold", f"LOSO_{study}", tr, te, None


def predict_matched(name: str, Xtr: pd.DataFrame, ytr: np.ndarray, Xte: pd.DataFrame) -> np.ndarray:
    """Reference fitting pathway with a CatBoost compatibility fallback.

    CatBoost uses the reference preprocessed feature matrix,
    separately for each target, with the reference hyperparameters and seed.
    This avoids a CatBoost/scikit-learn multi-output estimator-tags conflict.
    """
    if name != "CatBoost_base":
        return step04.fit_predict(step04.MODEL_SETTINGS[name], MODE, Xtr, ytr, Xte)

    from catboost import CatBoostRegressor
    pre = step04.make_preprocessor()
    train_array = pre.fit_transform(Xtr)
    test_array = pre.transform(Xte)
    params = step04.MODEL_SETTINGS[name].get_params()
    params["allow_writing_files"] = False  # no CatBoost training-log side effects
    predictions = []
    for product_index in range(ytr.shape[1]):
        model = CatBoostRegressor(**params)
        model.fit(train_array, ytr[:, product_index])
        predictions.append(model.predict(test_array))
    return step04.close_to_100(np.column_stack(predictions))


def run(*, smoke: bool) -> None:
    missing = [name for name in MODELS if name not in step04.MODEL_SETTINGS]
    if missing:
        raise RuntimeError(
            f"Required model(s) unavailable: {missing}. Install catboost as well as core requirements."
        )
    import catboost
    import xgboost

    df = step04.load_model_matrix()
    assert len(df) == 442 and df["study_id"].nunique() == 42, "Unexpected dataset identity"
    X = df[step04.NUMERIC_FEATURES + step04.CATEGORICAL_FEATURES]
    y = df[step04.TARGET_COLS].to_numpy(dtype=float)
    assert np.isfinite(y).all()

    prefix = "12_harmonized_model_comparison_smoke" if smoke else "12_harmonized_model_comparison"
    details: list[dict] = []
    loso = {name: {"true": [], "pred": [], "indices": []} for name in MODELS}
    specs = list(split_specs(df, smoke=smoke))
    for i, (validation, split_id, train, test, split_score) in enumerate(specs, 1):
        Xtr, Xte, ytr, yte = X.iloc[train], X.iloc[test], y[train], y[test]
        for name in MODELS:
            pred = predict_matched(name, Xtr, ytr, Xte)
            row = step04.metric_block(
                y_true_raw=yte, y_pred=pred, validation=validation,
                split_id=split_id, model_name=name, target_mode=MODE,
                n_train=len(train), n_test=len(test),
                extra={"split_score": split_score},
            )
            details.append(row)
            if validation == "LOSO fold":
                loso[name]["true"].append(yte)
                loso[name]["pred"].append(pred)
                loso[name]["indices"].extend(test.tolist())
        if i == 1 or i % 5 == 0 or i == len(specs):
            print(f"Completed {i}/{len(specs)} partitions: {validation} {split_id}", flush=True)

    detail_df = pd.DataFrame(details)
    summaries = []
    for validation in ("Random-row", "Grouped"):
        for name in MODELS:
            sub = detail_df.loc[(detail_df["validation"] == validation) & (detail_df["model"] == name)]
            summaries.append({
                "Validation": validation,
                "Model": LABELS[name],
                "R2": float(sub["closed_true_multi_R2"].mean()),
                "RMSE_wt_pct": float(sub["closed_true_RMSE"].mean()),
                "MAE_wt_pct": float(sub["closed_true_MAE"].mean()),
                "R2_SD": float(sub["closed_true_multi_R2"].std(ddof=1)) if len(sub) > 1 else np.nan,
                "RMSE_SD": float(sub["closed_true_RMSE"].std(ddof=1)) if len(sub) > 1 else np.nan,
                "MAE_SD": float(sub["closed_true_MAE"].std(ddof=1)) if len(sub) > 1 else np.nan,
                "Partitions": len(sub),
                "Aggregation": "mean of partition metrics",
            })
    for name in MODELS:
        pack = loso[name]
        ytrue = np.vstack(pack["true"])
        pred = np.vstack(pack["pred"])
        indices = pack["indices"]
        if not smoke:
            assert len(indices) == len(df) and len(set(indices)) == len(df), "Incomplete LOSO pool"
        metric = step04.metric_block(
            y_true_raw=ytrue, y_pred=pred, validation="LOSO pooled",
            split_id="LOSO_all_studies", model_name=name, target_mode=MODE,
            n_train=np.nan, n_test=len(ytrue),
        )
        summaries.append({
            "Validation": "LOSO pooled",
            "Model": LABELS[name],
            "R2": float(metric["closed_true_multi_R2"]),
            "RMSE_wt_pct": float(metric["closed_true_RMSE"]),
            "MAE_wt_pct": float(metric["closed_true_MAE"]),
            "R2_SD": np.nan, "RMSE_SD": np.nan, "MAE_SD": np.nan,
            "Partitions": len(pack["true"]),
            "Aggregation": "pooled out-of-study predictions",
        })
    summary_df = pd.DataFrame(summaries)

    # Only a complete 20+20+42 evaluation is comparable to Fig. 3.
    checks = []
    if not smoke:
        for validation, expected in EXPECTED_R2.items():
            measured = float(summary_df.loc[
                (summary_df["Validation"] == validation) & (summary_df["Model"] == "XGBoost"), "R2"
            ].iloc[0])
            difference = abs(measured - expected)
            checks.append({"Validation": validation, "Expected": expected, "Measured": measured,
                           "Absolute_difference": difference, "Pass_within_0.005": difference < 0.005})
    checks_df = pd.DataFrame(checks, columns=[
        "Validation", "Expected", "Measured", "Absolute_difference", "Pass_within_0.005"
    ])

    step04.output_file(prefix + "_details.csv").parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(step04.output_file(prefix + "_details.csv"), index=False)
    summary_df.to_csv(step04.output_file(prefix + "_summary.csv"), index=False)
    checks_df.to_csv(step04.output_file(prefix + "_reference_check.csv"), index=False)
    manifest = {
        "status": "SMOKE TEST — NOT PUBLISHABLE" if smoke else "FULL COMPARISON",
        "data_file": str(step04.INPUT_WORKBOOK),
        "data_sha256": sha256(step04.INPUT_WORKBOOK),
        "step04_sha256": sha256(HERE / "04_validation_ladder.py"),
        "rows": len(df), "studies": int(df["study_id"].nunique()),
        "feature_matrix": {"numeric": step04.NUMERIC_FEATURES, "categorical": step04.CATEGORICAL_FEATURES},
        "mode": MODE, "models": list(MODELS),
        "splits": {"random": 20 if not smoke else 1, "grouped": 20 if not smoke else 1,
                   "loso": 42 if not smoke else 1},
        "metric_definitions": "Step-04 metric_block; random/grouped mean of per-split metrics; LOSO pooled across held-out rows",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "catboost": catboost.__version__},
    }
    step04.output_file(prefix + "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nSummary (NOT publication results if smoke):\n", summary_df.to_string(index=False))
    if smoke:
        print("\nFigure-3 consistency check skipped: only one partition per validation was run.")
    else:
        print("\nXGBoost figure-3 consistency check:\n", checks_df.to_string(index=False))
    if not smoke and not bool(checks_df["Pass_within_0.005"].all()):
        raise RuntimeError("Reference XGBoost does not reproduce Fig. 3 within 0.005 R²; investigate before reporting")
    print("\nSaved:", prefix, "files in", step04.output_file(prefix + "_summary.csv").parent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="1 partition per validation (not publication results)")
    args = parser.parse_args()
    run(smoke=args.smoke)


if __name__ == "__main__":
    main()
