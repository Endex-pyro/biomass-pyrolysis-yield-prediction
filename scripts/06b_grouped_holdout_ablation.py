"""Grouped-holdout feature-group ablation reported in Table 3.

The script reuses the 20 target-free grouped partitions selected by the
validation-ladder analysis and evaluates the Renormalized XGBoost workflow.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from project_config import DATA_FILE, DATA_SHEET, output_file

try:
    from xgboost import XGBRegressor
except Exception as exc:
    raise ImportError(
        "xgboost is not installed. Install it with: python -m pip install xgboost"
    ) from exc

warnings.filterwarnings("ignore")


# Analysis settings
INPUT_WORKBOOK = DATA_FILE
INPUT_SHEET = DATA_SHEET
OUTPUT_WORKBOOK = output_file("06b_grouped_holdout_ablation_results.xlsx")

EXPECTED_ROWS = 442
EXPECTED_STUDIES = 42

# Reuse the exact 20 selected study-feedstock-composition splits from the
# main validation-ladder workbook. Merely changing GroupShuffleSplit from
# 10 to 20 seeds would create a different grouped-validation experiment.
SPLIT_SOURCE_WORKBOOK = output_file("04_validation_ladder_results.xlsx")
SPLIT_SOURCE_SHEET_CANDIDATES = [
    "32_S1_Group_SplitAudit",
    "32_S1_SimAware_SplitAudit",
]
EXPECTED_GROUP_SPLITS = 20
GROUP_DELIMITER = " ;; "
MODEL_SEEDS = list(range(EXPECTED_GROUP_SPLITS))

# For this step, keep only the relevant mode.
TARGET_MODE = "direct_forced_closed"
TARGET_MODE_LABEL = "Renormalized"


# ============================================================
# 2. FEATURE DEFINITIONS
# ============================================================

PROCESS_FEATURES = [
    "temperature_c",
    "heating_rate_c_min",
]

PROXIMATE_FEATURES = [
    "volatile_matter_wt_pct",
    "fixed_carbon_wt_pct",
    "ash_wt_pct",
]

ULTIMATE_FEATURES = [
    "carbon_wt_pct",
    "hydrogen_wt_pct",
    "nitrogen_wt_pct",
    "oxygen_wt_pct",
]

COMPOSITION_SUM_DIAGNOSTICS = [
    "proximate_sum_raw_wt_pct",
    "proximate_sum_deviation_from_100",
    "ultimate_CHNO_sum_raw_wt_pct",
    "ultimate_CHNO_sum_deviation_from_100",
]

SEVERITY_FEATURES = [
    "inv_temp_K",
    "log_heating_rate",
    "temp_x_log_heating_rate",
]

REPORTING_FEATURES = [
    "yield_basis",
    "oil_definition",
    "gas_yield_method",
    "pyrolysis_type",
    "proximate_basis",
    "ultimate_basis",
]

REPORTING_COMBINED_FEATURES = [
    "proximate_ultimate_basis_pair",
    "yield_reporting_profile",
]

TARGET_COLS = [
    "char_yield_wt_pct",
    "oil_yield_wt_pct",
    "gas_yield_wt_pct",
]


FEATURE_VARIANTS: Dict[str, Dict[str, List[str]]] = {
    "Full_reporting_aware_F2": {
        "numeric": (
            PROCESS_FEATURES
            + PROXIMATE_FEATURES
            + ULTIMATE_FEATURES
            + COMPOSITION_SUM_DIAGNOSTICS
            + SEVERITY_FEATURES
        ),
        "categorical": REPORTING_FEATURES + REPORTING_COMBINED_FEATURES,
    },

    "Reporting_aware_without_sum_diagnostics": {
        "numeric": (
            PROCESS_FEATURES
            + PROXIMATE_FEATURES
            + ULTIMATE_FEATURES
            + SEVERITY_FEATURES
        ),
        "categorical": REPORTING_FEATURES + REPORTING_COMBINED_FEATURES,
    },

    "Thermochemical_only_raw": {
        "numeric": PROCESS_FEATURES + PROXIMATE_FEATURES + ULTIMATE_FEATURES,
        "categorical": [],
    },

    "Thermochemical_only_severity": {
        "numeric": (
            PROCESS_FEATURES
            + PROXIMATE_FEATURES
            + ULTIMATE_FEATURES
            + SEVERITY_FEATURES
        ),
        "categorical": [],
    },
}


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def resolve_input_workbook() -> Path:
    if INPUT_WORKBOOK.exists():
        return INPUT_WORKBOOK

    raise FileNotFoundError(
        f"Could not find the model-ready dataset at {INPUT_WORKBOOK}. "
        "Set PYROLYSIS_DATA_FILE to override the default repository path."
    )


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def clean_string_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
    )


def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if np.nanstd(y_true) < 1e-10:
        return np.nan

    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return np.nan


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def close_to_100(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = np.maximum(y, 1e-10)
    return y / y.sum(axis=1, keepdims=True) * 100.0


def get_xgb_model(random_state: int = 42):
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        reg_alpha=0.0,
        random_state=random_state,
        n_jobs=-1,
    )


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    transformers = []

    if numeric_features:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])
        transformers.append(("num", numeric_pipe, numeric_features))

    if categorical_features:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_ohe()),
        ])
        transformers.append(("cat", categorical_pipe, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def fit_renormalized_pipeline(
    X_train: pd.DataFrame,
    y_train_raw: np.ndarray,
    numeric_features: List[str],
    categorical_features: List[str],
    seed: int,
) -> Pipeline:
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model = get_xgb_model(random_state=seed)

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", MultiOutputRegressor(clone(model))),
    ])

    # Same as your Renormalized workflow:
    # train direct on raw yields, then close predictions after inference.
    pipe.fit(X_train, y_train_raw)
    return pipe


def predict_renormalized(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    pred_raw = pipe.predict(X)
    return close_to_100(pred_raw)


def evaluate_prediction(y_true_raw: np.ndarray, y_pred_closed: np.ndarray) -> Dict[str, float]:
    y_true_closed = close_to_100(y_true_raw)
    pred_sum = y_pred_closed.sum(axis=1)

    denom = np.sum(np.abs(y_true_closed))
    wape_pct = (
        float(np.sum(np.abs(y_true_closed - y_pred_closed)) / denom * 100.0)
        if denom > 0
        else np.nan
    )

    return {
        "multi_R2": float(np.nanmean([
            safe_r2(y_true_closed[:, 0], y_pred_closed[:, 0]),
            safe_r2(y_true_closed[:, 1], y_pred_closed[:, 1]),
            safe_r2(y_true_closed[:, 2], y_pred_closed[:, 2]),
        ])),
        "char_R2": safe_r2(y_true_closed[:, 0], y_pred_closed[:, 0]),
        "oil_R2": safe_r2(y_true_closed[:, 1], y_pred_closed[:, 1]),
        "gas_R2": safe_r2(y_true_closed[:, 2], y_pred_closed[:, 2]),
        "RMSE": rmse(y_true_closed, y_pred_closed),
        "MAE": float(mean_absolute_error(y_true_closed, y_pred_closed)),
        "char_MAE": float(mean_absolute_error(y_true_closed[:, 0], y_pred_closed[:, 0])),
        "oil_MAE": float(mean_absolute_error(y_true_closed[:, 1], y_pred_closed[:, 1])),
        "gas_MAE": float(mean_absolute_error(y_true_closed[:, 2], y_pred_closed[:, 2])),
        "WAPE_pct": wape_pct,
        "pred_sum_mean": float(np.mean(pred_sum)),
        "pred_sum_sd": float(np.std(pred_sum, ddof=1)) if len(pred_sum) > 1 else 0.0,
        "pct_pred_sum_outside_95_105": float(
            np.mean((pred_sum < 95) | (pred_sum > 105)) * 100.0
        ),
    }


# ============================================================
# 4. DATA PREPARATION
# ============================================================

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in PROCESS_FEATURES + PROXIMATE_FEATURES + ULTIMATE_FEATURES + TARGET_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in REPORTING_FEATURES:
        if c in df.columns:
            df[c] = clean_string_series(df[c])

    df["proximate_sum_raw_wt_pct"] = (
        df["volatile_matter_wt_pct"]
        + df["fixed_carbon_wt_pct"]
        + df["ash_wt_pct"]
    )
    df["proximate_sum_deviation_from_100"] = df["proximate_sum_raw_wt_pct"] - 100.0

    df["ultimate_CHNO_sum_raw_wt_pct"] = (
        df["carbon_wt_pct"]
        + df["hydrogen_wt_pct"]
        + df["nitrogen_wt_pct"]
        + df["oxygen_wt_pct"]
    )
    df["ultimate_CHNO_sum_deviation_from_100"] = df["ultimate_CHNO_sum_raw_wt_pct"] - 100.0

    df["proximate_ultimate_basis_pair"] = (
        clean_string_series(df["proximate_basis"])
        + "__"
        + clean_string_series(df["ultimate_basis"])
    )

    df["yield_reporting_profile"] = (
        clean_string_series(df["yield_basis"])
        + "__"
        + clean_string_series(df["oil_definition"])
        + "__"
        + clean_string_series(df["gas_yield_method"])
    )

    temp_c = pd.to_numeric(df["temperature_c"], errors="coerce")
    temp_k = temp_c + 273.15
    heating_rate = pd.to_numeric(df["heating_rate_c_min"], errors="coerce")

    df["inv_temp_K"] = np.where(temp_k > 0, 1.0 / temp_k, np.nan)
    df["log_heating_rate"] = np.log1p(np.maximum(heating_rate, 0))
    df["temp_x_log_heating_rate"] = temp_c * df["log_heating_rate"]

    return df


def build_group_column(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the canonical group keys stored in the split audit."""

    df = df.copy()

    feedstock_candidates = [
        "feedstock_name",
        "feedstock_name_std",
        "feedstock",
        "biomass_name",
        "biomass",
    ]
    feedstock_col = next((c for c in feedstock_candidates if c in df.columns), None)
    if feedstock_col is None:
        raise KeyError(
            "No usable feedstock-name column was found for group reconstruction."
        )

    study_numeric = pd.to_numeric(df["study_id"], errors="coerce")
    if study_numeric.isna().any():
        raise ValueError("study_id contains missing/non-numeric values.")
    study_text = study_numeric.astype(int).astype(str)
    feedstock_text = clean_string_series(df[feedstock_col])

    composition_cols = PROXIMATE_FEATURES + ULTIMATE_FEATURES
    formatted_parts = []
    for col in composition_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        formatted_parts.append(values.map(lambda x: f"{x:.3f}" if pd.notna(x) else np.nan))

    group = study_text + " | " + feedstock_text
    for part in formatted_parts:
        group = group + " | " + part

    if "validation_group_id" in df.columns:
        provided = clean_string_series(df["validation_group_id"])
        mapping = pd.DataFrame({"provided": provided, "canonical": group})
        if (
            mapping.groupby("provided")["canonical"].nunique().max() != 1
            or mapping.groupby("canonical")["provided"].nunique().max() != 1
        ):
            raise ValueError(
                "Provided validation_group_id is not one-to-one with the "
                "canonical study-feedstock-composition groups."
            )
        df["provided_validation_group_id"] = provided

    df["grouped_holdout_id"] = group
    return df

def load_data() -> pd.DataFrame:
    workbook = resolve_input_workbook()

    print(f"Input workbook: {workbook.resolve()}")
    print(f"Input sheet: {INPUT_SHEET}")

    df = pd.read_excel(workbook, sheet_name=INPUT_SHEET)
    df = add_engineered_features(df)
    df = build_group_column(df)

    all_needed = set(TARGET_COLS + ["study_id", "grouped_holdout_id"])

    for cfg in FEATURE_VARIANTS.values():
        all_needed.update(cfg["numeric"])
        all_needed.update(cfg["categorical"])

    missing = [c for c in all_needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    before = len(df)
    df = df.dropna(subset=list(all_needed)).copy().reset_index(drop=True)
    after = len(df)

    if before != after:
        print(f"WARNING: dropped {before - after} rows after final required-column check.")

    if after != EXPECTED_ROWS:
        print(f"WARNING: expected {EXPECTED_ROWS} rows, got {after} rows.")

    if "study_id" in df.columns:
        n_studies = df["study_id"].nunique()
        if n_studies != EXPECTED_STUDIES:
            print(f"WARNING: expected {EXPECTED_STUDIES} studies, got {n_studies}.")

    df["row_uid"] = np.arange(len(df))
    return df


# ============================================================
# 5. GROUPED ABLATION
# ============================================================

def resolve_split_source_workbook() -> Path:
    """Locate the main validation-ladder workbook containing selected splits."""

    if SPLIT_SOURCE_WORKBOOK.exists():
        return SPLIT_SOURCE_WORKBOOK

    raise FileNotFoundError(
        f"Could not find {SPLIT_SOURCE_WORKBOOK}. Run 04_validation_ladder.py "
        "before the grouped-holdout ablation."
    )


def load_selected_group_splits(df: pd.DataFrame):
    """Load and reconstruct the 20 grouped partitions used in Table 3."""

    workbook = resolve_split_source_workbook()
    sheet_names = pd.ExcelFile(workbook).sheet_names
    split_sheet = next(
        (s for s in SPLIT_SOURCE_SHEET_CANDIDATES if s in sheet_names),
        None,
    )
    if split_sheet is None:
        raise KeyError(
            f"None of the expected split-audit sheets were found in {workbook}: "
            f"{SPLIT_SOURCE_SHEET_CANDIDATES}"
        )

    audit = pd.read_excel(workbook, sheet_name=split_sheet)
    required = {
        "selected_split_id",
        "test_groups_string",
        "test_rows",
        "test_groups",
    }
    missing = sorted(required.difference(audit.columns))
    if missing:
        raise KeyError(f"Split-audit sheet is missing columns: {missing}")

    if "selected_rank" in audit.columns:
        audit = audit.sort_values("selected_rank").reset_index(drop=True)
    else:
        audit = audit.reset_index(drop=True)
        audit["selected_rank"] = np.arange(1, len(audit) + 1)

    if len(audit) != EXPECTED_GROUP_SPLITS:
        raise ValueError(
            f"Expected {EXPECTED_GROUP_SPLITS} selected grouped splits, "
            f"but found {len(audit)} in {split_sheet}."
        )

    groups = df["grouped_holdout_id"].astype(str)
    available_groups = set(groups.unique())
    all_idx = np.arange(len(df))
    splits = []

    for position, row in audit.iterrows():
        split_id = str(row["selected_split_id"]).strip()
        test_groups = [
            g.strip()
            for g in str(row["test_groups_string"]).split(GROUP_DELIMITER)
            if g.strip()
        ]
        test_group_set = set(test_groups)

        missing_groups = sorted(test_group_set.difference(available_groups))
        if missing_groups:
            preview = "\n  - ".join(missing_groups[:5])
            raise ValueError(
                f"{split_id}: {len(missing_groups)} test groups from the main "
                "split audit were not found in the current dataset. First examples:\n"
                f"  - {preview}\n"
                "Check that the same dataset sheet and validation_group_id are used."
            )

        test_mask = groups.isin(test_group_set).to_numpy()
        test_idx = all_idx[test_mask]
        train_idx = all_idx[~test_mask]

        train_group_set = set(groups.iloc[train_idx])
        overlap = train_group_set.intersection(test_group_set)
        if overlap:
            raise RuntimeError(
                f"Group leakage detected for {split_id}: {len(overlap)} overlapping groups."
            )

        expected_test_rows = int(row["test_rows"])
        expected_test_groups = int(row["test_groups"])
        if len(test_idx) != expected_test_rows:
            raise ValueError(
                f"{split_id}: reconstructed {len(test_idx)} test rows, "
                f"expected {expected_test_rows}."
            )
        if len(test_group_set) != expected_test_groups:
            raise ValueError(
                f"{split_id}: reconstructed {len(test_group_set)} test groups, "
                f"expected {expected_test_groups}."
            )

        model_seed = MODEL_SEEDS[position]
        splits.append({
            "split_id": split_id,
            "selected_rank": int(row["selected_rank"]),
            "candidate_id": row.get("candidate_id", np.nan),
            "split_score": row.get("split_score", np.nan),
            "model_seed": model_seed,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "n_train_rows": len(train_idx),
            "n_test_rows": len(test_idx),
            "n_train_groups": len(train_group_set),
            "n_test_groups": len(test_group_set),
            "n_overlap_groups": len(overlap),
        })

    selected_audit = pd.DataFrame([
        {
            "split_id": s["split_id"],
            "selected_rank": s["selected_rank"],
            "candidate_id": s["candidate_id"],
            "split_score": s["split_score"],
            "model_seed": s["model_seed"],
            "n_train_rows": s["n_train_rows"],
            "n_test_rows": s["n_test_rows"],
            "n_train_groups": s["n_train_groups"],
            "n_test_groups": s["n_test_groups"],
            "n_overlap_groups": s["n_overlap_groups"],
        }
        for s in splits
    ])

    return splits, selected_audit, workbook, split_sheet

def run_grouped_ablation(df: pd.DataFrame):
    rows = []

    y = df[TARGET_COLS].to_numpy(dtype=float)
    splits, selected_audit, split_workbook, split_sheet = load_selected_group_splits(df)

    for split_info in splits:
        split_id = split_info["split_id"]
        model_seed = split_info["model_seed"]
        train_idx = split_info["train_idx"]
        test_idx = split_info["test_idx"]

        y_train = y[train_idx]
        y_test = y[test_idx]

        for feature_variant, cfg in FEATURE_VARIANTS.items():
            numeric_features = cfg["numeric"]
            categorical_features = cfg["categorical"]
            features = numeric_features + categorical_features

            X = df[features].copy()
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            pipe = fit_renormalized_pipeline(
                X_train=X_train,
                y_train_raw=y_train,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                seed=model_seed,
            )

            pred = predict_renormalized(pipe, X_test)
            metrics = evaluate_prediction(y_test, pred)

            rows.append({
                "split_name": "grouped_holdout",
                "split_id": split_id,
                "seed": model_seed,
                "selected_rank": split_info["selected_rank"],
                "candidate_id": split_info["candidate_id"],
                "feature_variant": feature_variant,
                "target_mode": TARGET_MODE,
                "target_mode_label": TARGET_MODE_LABEL,
                "n_rows_total": len(df),
                "n_train": split_info["n_train_rows"],
                "n_test": split_info["n_test_rows"],
                "n_train_groups": split_info["n_train_groups"],
                "n_test_groups": split_info["n_test_groups"],
                "n_overlap_groups": split_info["n_overlap_groups"],
                "n_numeric_features": len(numeric_features),
                "n_categorical_features": len(categorical_features),
                "n_total_original_features": len(features),
                "numeric_features": "; ".join(numeric_features),
                "categorical_features": "; ".join(categorical_features),
                **metrics,
            })

    return pd.DataFrame(rows), selected_audit, split_workbook, split_sheet


def summarize_performance(perf_long: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "split_name",
        "feature_variant",
        "target_mode",
        "target_mode_label",
        "n_numeric_features",
        "n_categorical_features",
        "n_total_original_features",
    ]

    summary = (
        perf_long
        .groupby(group_cols, as_index=False)
        .agg(
            multi_R2_mean=("multi_R2", "mean"),
            multi_R2_sd=("multi_R2", "std"),
            char_R2_mean=("char_R2", "mean"),
            char_R2_sd=("char_R2", "std"),
            oil_R2_mean=("oil_R2", "mean"),
            oil_R2_sd=("oil_R2", "std"),
            gas_R2_mean=("gas_R2", "mean"),
            gas_R2_sd=("gas_R2", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_sd=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_sd=("MAE", "std"),
            char_MAE_mean=("char_MAE", "mean"),
            oil_MAE_mean=("oil_MAE", "mean"),
            gas_MAE_mean=("gas_MAE", "mean"),
            WAPE_pct_mean=("WAPE_pct", "mean"),
            WAPE_pct_sd=("WAPE_pct", "std"),
            pred_sum_mean=("pred_sum_mean", "mean"),
            pred_sum_sd_mean=("pred_sum_sd", "mean"),
            pct_pred_sum_outside_95_105_mean=("pct_pred_sum_outside_95_105", "mean"),
            n_splits=("split_id", "nunique"),
            n_train_mean=("n_train", "mean"),
            n_test_mean=("n_test", "mean"),
            n_train_groups_mean=("n_train_groups", "mean"),
            n_test_groups_mean=("n_test_groups", "mean"),
        )
    )

    summary["rank_within_grouped_holdout"] = (
        summary["multi_R2_mean"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    return summary.sort_values("rank_within_grouped_holdout").reset_index(drop=True)


def build_delta_vs_full(summary: pd.DataFrame) -> pd.DataFrame:
    full = summary[
        summary["feature_variant"].eq("Full_reporting_aware_F2")
    ].copy()

    full_cols = [
        "split_name",
        "target_mode",
        "multi_R2_mean",
        "RMSE_mean",
        "MAE_mean",
        "WAPE_pct_mean",
    ]

    full = full[full_cols].rename(columns={
        "multi_R2_mean": "full_multi_R2_mean",
        "RMSE_mean": "full_RMSE_mean",
        "MAE_mean": "full_MAE_mean",
        "WAPE_pct_mean": "full_WAPE_pct_mean",
    })

    out = summary.merge(full, on=["split_name", "target_mode"], how="left")

    out["delta_multi_R2_vs_full"] = out["multi_R2_mean"] - out["full_multi_R2_mean"]
    out["delta_RMSE_vs_full"] = out["RMSE_mean"] - out["full_RMSE_mean"]
    out["delta_MAE_vs_full"] = out["MAE_mean"] - out["full_MAE_mean"]
    out["delta_WAPE_pct_vs_full"] = out["WAPE_pct_mean"] - out["full_WAPE_pct_mean"]

    return out.sort_values("rank_within_grouped_holdout").reset_index(drop=True)


def build_decision_table(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in delta.iterrows():
        variant = r["feature_variant"]
        delta_r2 = r["delta_multi_R2_vs_full"]

        if variant == "Full_reporting_aware_F2":
            interpretation = (
                "Reference grouped-holdout model containing thermochemical variables, "
                "composition-sum diagnostics, and reporting-context descriptors."
            )
            comparison = "Reference configuration."
        elif variant == "Reporting_aware_without_sum_diagnostics":
            interpretation = (
                "Tests the contribution of proximate/ultimate sum and deviation diagnostics while retaining the broader reporting context."
            )
        elif variant == "Thermochemical_only_raw":
            interpretation = (
                "Strict clean grouped-holdout baseline using only process, proximate, and ultimate composition."
            )
        elif variant == "Thermochemical_only_severity":
            interpretation = (
                "Clean grouped-holdout baseline plus physical severity descriptors from temperature and heating rate."
            )
        else:
            interpretation = ""

        if variant != "Full_reporting_aware_F2":
            if delta_r2 >= -0.05:
                comparison = "Small reduction relative to the full model."
            elif delta_r2 >= -0.15:
                comparison = "Moderate reduction relative to the full model."
            else:
                comparison = "Large reduction relative to the full model."

        rows.append({
            "focus_setting": "grouped_holdout + Renormalized",
            "feature_variant": variant,
            "multi_R2_mean": r["multi_R2_mean"],
            "multi_R2_sd": r["multi_R2_sd"],
            "delta_multi_R2_vs_full": r["delta_multi_R2_vs_full"],
            "RMSE_mean": r["RMSE_mean"],
            "MAE_mean": r["MAE_mean"],
            "delta_MAE_vs_full": r["delta_MAE_vs_full"],
            "WAPE_pct_mean": r["WAPE_pct_mean"],
            "interpretation": interpretation,
            "comparison_to_full": comparison,
        })

    return pd.DataFrame(rows)


def build_split_audit(df: pd.DataFrame) -> pd.DataFrame:
    groups = df["grouped_holdout_id"].astype(str)

    audit_rows = [{
        "audit_item": "rows",
        "value": len(df),
    }, {
        "audit_item": "studies",
        "value": df["study_id"].nunique() if "study_id" in df.columns else np.nan,
    }, {
        "audit_item": "grouped_holdout_groups",
        "value": groups.nunique(),
    }, {
        "audit_item": "min_rows_per_group",
        "value": int(groups.value_counts().min()),
    }, {
        "audit_item": "median_rows_per_group",
        "value": float(groups.value_counts().median()),
    }, {
        "audit_item": "max_rows_per_group",
        "value": int(groups.value_counts().max()),
    }]

    return pd.DataFrame(audit_rows)


def build_group_size_table(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("grouped_holdout_id", as_index=False)
        .agg(
            n_rows=("grouped_holdout_id", "size"),
            n_studies=("study_id", "nunique"),
        )
        .sort_values("n_rows", ascending=False)
        .reset_index(drop=True)
    )

    return out


def build_config_table(
    df: pd.DataFrame,
    split_workbook: Path,
    split_sheet: str,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "input_workbook": str(resolve_input_workbook()),
        "input_sheet": INPUT_SHEET,
        "output_workbook": str(OUTPUT_WORKBOOK),
        "rows_after_check": len(df),
        "studies": df["study_id"].nunique() if "study_id" in df.columns else np.nan,
        "grouped_holdout_groups": df["grouped_holdout_id"].nunique(),
        "expected_rows": EXPECTED_ROWS,
        "expected_studies": EXPECTED_STUDIES,
        "split_source_workbook": str(split_workbook),
        "split_source_sheet": split_sheet,
        "selected_group_splits": EXPECTED_GROUP_SPLITS,
        "model_seeds": ", ".join(map(str, MODEL_SEEDS)),
        "target_mode": TARGET_MODE,
        "target_mode_label": TARGET_MODE_LABEL,
        "main_question": (
            "How does grouped performance change when composition-sum diagnostics or the broader reporting context are removed using the exact main validation splits?"
        ),
    }])


def build_feature_variant_table() -> pd.DataFrame:
    rows = []

    for variant, cfg in FEATURE_VARIANTS.items():
        rows.append({
            "feature_variant": variant,
            "n_numeric_features": len(cfg["numeric"]),
            "n_categorical_features": len(cfg["categorical"]),
            "n_total_features_before_onehot": len(cfg["numeric"]) + len(cfg["categorical"]),
            "numeric_features": "; ".join(cfg["numeric"]),
            "categorical_features": "; ".join(cfg["categorical"]),
        })

    return pd.DataFrame(rows)


# ============================================================
# 6. MAIN
# ============================================================

def main():
    print("\n====================================================")
    print("GROUPED-HOLDOUT FEATURE-GROUP ABLATION")
    print("====================================================")

    df = load_data()

    print(f"\nRows after check: {len(df)}")
    print(f"Studies: {df['study_id'].nunique() if 'study_id' in df.columns else 'NA'}")
    print(f"Grouped-holdout groups: {df['grouped_holdout_id'].nunique()}")
    print(f"Feature variants: {len(FEATURE_VARIANTS)}")
    print(f"Target mode: {TARGET_MODE_LABEL}")
    print(f"Selected grouped splits expected: {EXPECTED_GROUP_SPLITS}")
    print(f"Model seeds: {MODEL_SEEDS}")

    print("\nRunning grouped-holdout ablation...")
    perf_long, selected_split_audit, split_workbook, split_sheet = run_grouped_ablation(df)
    print(f"Split source: {split_workbook} [{split_sheet}]")

    print("Summarizing results...")
    perf_summary = summarize_performance(perf_long)
    delta_vs_full = build_delta_vs_full(perf_summary)
    decision = build_decision_table(delta_vs_full)

    config = build_config_table(df, split_workbook, split_sheet)
    feature_variants = build_feature_variant_table()
    split_audit = build_split_audit(df)
    group_sizes = build_group_size_table(df)

    print("\nSaving workbook...")
    with pd.ExcelWriter(OUTPUT_WORKBOOK, engine="openpyxl") as writer:
        config.to_excel(writer, sheet_name="00_Config", index=False)
        feature_variants.to_excel(writer, sheet_name="01_Feature_Variants", index=False)
        selected_split_audit.to_excel(writer, sheet_name="02_Selected_Split_Audit", index=False)
        split_audit.to_excel(writer, sheet_name="03_Group_Dataset_Audit", index=False)
        group_sizes.to_excel(writer, sheet_name="04_Group_Sizes", index=False)
        perf_long.to_excel(writer, sheet_name="10_Grouped_Performance_Long", index=False)
        perf_summary.to_excel(writer, sheet_name="11_Grouped_Summary", index=False)
        delta_vs_full.to_excel(writer, sheet_name="12_Grouped_Delta_vs_Full", index=False)
        decision.to_excel(writer, sheet_name="13_Ablation_Decision", index=False)

    print("\n====================================================")
    print("DONE")
    print("====================================================")
    print(f"Saved to: {OUTPUT_WORKBOOK.resolve()}")

    print("\nMain decision table: grouped-holdout + Renormalized")
    cols = [
        "feature_variant",
        "multi_R2_mean",
        "multi_R2_sd",
        "delta_multi_R2_vs_full",
        "MAE_mean",
        "delta_MAE_vs_full",
        "comparison_to_full",
    ]
    print(decision[cols].to_string(index=False))

if __name__ == "__main__":
    main()
