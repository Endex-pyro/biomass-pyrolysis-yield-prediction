"""Study-disjoint sensitivity of product-yield models to reporting descriptors.

Runs three predictor configurations with identical leave-one-study-out folds,
XGBoost settings, preprocessing and target treatment, over three fixed seeds.
The categorical-reporting contrast is conditional on omitting numerical
composition sums. Outputs are written to the repository's ``outputs`` folder.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone

from project_config import output_file


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("yield_validation", ROOT / "04_validation_ladder.py")
if spec is None or spec.loader is None:
    raise ImportError("Cannot import reference validation pipeline")
ladder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ladder)

SEEDS = (42, 7, 123)
SUM_COLUMNS = {"proximate_sum_VM_FC_Ash", "ultimate_CHNO_sum"}
PROCESS_CATEGORY = "pyrolysis_type_v3"
VARIANTS = ("full", "no_sum", "no_sum_no_reporting")


def save_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows produced for {name}")
    path = output_file(name)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,7,123", help="Comma-separated model random seeds")
    parser.add_argument("--variants", default=",".join(VARIANTS), help="Comma-separated feature variants")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Fit one study holdout using one seed and feature set; do not write summaries",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    selected = [x.strip() for x in args.variants.split(",") if x.strip()]
    if not seeds or not selected or any(v not in VARIANTS for v in selected):
        raise ValueError(f"Select at least one seed and variants from {VARIANTS}")
    if args.smoke:
        seeds = seeds[:1]
        selected = selected[:1]

    # Prepare the reference feature matrix before changing variant-specific inputs.
    base_num = list(ladder.NUMERIC_FEATURES)
    base_cat = list(ladder.CATEGORICAL_FEATURES)
    if len(base_num) != 11 or len(base_cat) != 8:
        raise RuntimeError("Reference feature definitions changed")
    if not SUM_COLUMNS.issubset(base_num) or PROCESS_CATEGORY not in base_cat:
        raise RuntimeError("Required reporting-feature definitions are missing")
    if len([c for c in base_cat if c != PROCESS_CATEGORY]) != 7:
        raise RuntimeError("Expected seven categorical reporting descriptors")

    feature_sets = {
        "full": (base_num, base_cat),
        "no_sum": ([c for c in base_num if c not in SUM_COLUMNS], base_cat),
        "no_sum_no_reporting": (
            [c for c in base_num if c not in SUM_COLUMNS], [PROCESS_CATEGORY]
        ),
    }
    df = ladder.load_model_matrix()
    if len(df) != 442 or df["study_id"].nunique() != 42:
        raise RuntimeError("Expected 442 rows from 42 independent studies")

    truth = df[ladder.TARGET_COLS].to_numpy(dtype=float)
    truth_closed = ladder.close_to_100(truth)
    study_ids = df["study_id"].astype(str).to_numpy()
    studies = sorted(set(study_ids))
    import xgboost

    results: list[dict] = []
    fold_rows: list[dict] = []
    for variant in selected:
        numeric, categorical = feature_sets[variant]
        x = df[numeric + categorical]
        ladder.NUMERIC_FEATURES = numeric
        ladder.CATEGORICAL_FEATURES = categorical

        for seed in seeds:
            model = clone(ladder.MODEL_SETTINGS["XGBoost_base"])
            model.set_params(random_state=seed, n_jobs=2)
            predictions = np.full_like(truth, np.nan, dtype=float)
            for idx, study in enumerate(studies, 1):
                holdout = np.flatnonzero(study_ids == study)
                train = np.flatnonzero(study_ids != study)
                p = ladder.fit_predict(
                    model, "direct_forced_closed", x.iloc[train], truth[train], x.iloc[holdout]
                )
                predictions[holdout] = p
                err = p - truth_closed[holdout]
                fold_rows.append({
                    "feature_variant": variant,
                    "model_seed": seed,
                    "study_id": study,
                    "n_test": len(holdout),
                    "fold_MAE_wt_pct": float(np.abs(err).mean()),
                    "fold_RMSE_wt_pct": float(np.sqrt(np.mean(err ** 2))),
                })
                if args.smoke:
                    print(f"Smoke test: {variant}, seed {seed}, study {study}, {len(holdout)} rows", flush=True)
                    return
                if idx % 10 == 0 or idx == len(studies):
                    print(f"{variant}, seed {seed}: {idx}/{len(studies)} study folds", flush=True)

            if not np.isfinite(predictions).all():
                raise RuntimeError("LOSO predictions were incomplete")
            if not np.allclose(predictions.sum(axis=1), 100.0, atol=1e-6):
                raise RuntimeError("Predicted yields do not sum to 100 wt.%")
            metrics = ladder.metric_block(
                y_true_raw=truth, y_pred=predictions,
                validation="LOSO_study_disjoint", split_id="42_fixed_study_folds",
                model_name="XGBoost_base", target_mode="direct_forced_closed",
                n_train=np.nan, n_test=len(df),
            )
            row = {
                "feature_variant": variant,
                "model_seed": seed,
                "n_rows": len(df),
                "n_studies": len(studies),
                "closed_true_multi_R2": metrics["closed_true_multi_R2"],
                "closed_true_RMSE": metrics["closed_true_RMSE"],
                "closed_true_MAE": metrics["closed_true_MAE"],
                "char_R2": metrics["closed_true_char_R2"],
                "liquid_R2": metrics["closed_true_oil_R2"],
                "gas_R2": metrics["closed_true_gas_R2"],
                "xgboost_version": xgboost.__version__,
            }
            results.append(row)
            if variant == "full" and seed == 42 and abs(row["closed_true_multi_R2"] - 0.274208) > 0.005:
                raise RuntimeError("Reference LOSO score differs from manuscript; review dependencies")
            print(f"{variant}, seed {seed}: R²={row['closed_true_multi_R2']:.3f}", flush=True)

    save_csv("13_reporting_ablation_seed_summary.csv", results)
    save_csv("13_reporting_ablation_fold_diagnostics.csv", fold_rows)

    by_key = {(r["feature_variant"], r["model_seed"]): r for r in results}
    if set(selected) == set(VARIANTS):
        comparisons = []
        for seed in seeds:
            full = by_key[("full", seed)]
            no_sum = by_key[("no_sum", seed)]
            reduced = by_key[("no_sum_no_reporting", seed)]
            comparisons.append({
                "model_seed": seed,
                "full_R2": full["closed_true_multi_R2"],
                "no_sum_R2": no_sum["closed_true_multi_R2"],
                "no_sum_no_reporting_R2": reduced["closed_true_multi_R2"],
                "reporting_increment_R2": no_sum["closed_true_multi_R2"] - reduced["closed_true_multi_R2"],
                "reporting_increment_MAE_wt_pct": reduced["closed_true_MAE"] - no_sum["closed_true_MAE"],
            })
        save_csv("13_reporting_ablation_conditional_comparisons.csv", comparisons)


if __name__ == "__main__":
    main()
