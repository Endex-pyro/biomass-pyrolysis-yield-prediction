"""Process-detail sensitivity analysis using optional reactor variables.

Holding time, particle size, and sample mass are evaluated on the same
322-observation complete-detail subset under random-row, grouped-holdout, and
leave-one-study-out validation.
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, train_test_split
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
OUTPUT_WORKBOOK = output_file("08_process_detail_sensitivity_results.xlsx")

EXPECTED_FULL_ROWS = 442
EXPECTED_FULL_STUDIES = 42

EXPECTED_COMPLETE_DETAIL_ROWS = 322
EXPECTED_COMPLETE_DETAIL_STUDIES = 29

RANDOM_SEEDS = list(range(20))
GROUP_SPLIT_SEEDS = list(range(20))
TEST_SIZE = 0.20

TARGET_MODE = "direct_forced_closed"
TARGET_LABEL = "Renormalized"


# ============================================================
# 2. FEATURE DEFINITIONS — PROCESS, COMPOSITION AND SEVERITY FEATURES
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

REPORTING_FEATURES = [
    "yield_basis",
    "oil_definition",
    "gas_yield_method",
    "pyrolysis_type",
    "proximate_basis",
    "ultimate_basis",
]

REPORTING_FALLBACKS = {
    "yield_basis": "yield_basis_std",
    "oil_definition": "oil_definition_v3",
    "gas_yield_method": "gas_yield_method_v3",
    "pyrolysis_type": "pyrolysis_type_v3",
    "proximate_basis": "proximate_basis_std",
    "ultimate_basis": "ultimate_basis_std",
}

BASIS_PROXY_NUMERIC = [
    "proximate_sum_raw_wt_pct",
    "proximate_sum_deviation_from_100",
    "ultimate_CHNO_sum_raw_wt_pct",
    "ultimate_CHNO_sum_deviation_from_100",
]

BASIS_PROXY_CATEGORICAL = [
    "proximate_ultimate_basis_pair",
    "yield_reporting_profile",
]

SEVERITY_FEATURES = [
    "inv_temp_K",
    "log_heating_rate",
    "temp_x_log_heating_rate",
]

OPTIONAL_PROCESS_FEATURES = [
    "holding_time_min_num",
    "particle_size_mm_mid",
    "sample_mass_g_num",
]

TARGET_COLS = [
    "char_yield_wt_pct",
    "oil_yield_wt_pct",
    "gas_yield_wt_pct",
]

BASE_NUMERIC_FEATURES = (
    PROCESS_FEATURES
    + PROXIMATE_FEATURES
    + ULTIMATE_FEATURES
    + BASIS_PROXY_NUMERIC
    + SEVERITY_FEATURES
)

BASE_CATEGORICAL_FEATURES = (
    REPORTING_FEATURES
    + BASIS_PROXY_CATEGORICAL
)

BASE_FEATURES = BASE_NUMERIC_FEATURES + BASE_CATEGORICAL_FEATURES


SCENARIOS: Dict[str, Dict[str, List[str] | str]] = {
    "complete_detail_baseline_no_optional": {
        "label": "n=322 baseline without holding time, particle size, or sample mass",
        "optional_features_added": [],
    },
    "complete_detail_plus_holding_time": {
        "label": "n=322 baseline + holding time",
        "optional_features_added": ["holding_time_min_num"],
    },
    "complete_detail_plus_particle_size": {
        "label": "n=322 baseline + particle size",
        "optional_features_added": ["particle_size_mm_mid"],
    },
    "complete_detail_plus_sample_mass": {
        "label": "n=322 baseline + sample mass",
        "optional_features_added": ["sample_mass_g_num"],
    },
    "complete_detail_plus_all_optional": {
        "label": "n=322 baseline + holding time + particle size + sample mass",
        "optional_features_added": [
            "holding_time_min_num",
            "particle_size_mm_mid",
            "sample_mass_g_num",
        ],
    },
}


# ============================================================
# 3. HELPERS
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


def parse_numeric_or_midrange(value) -> float:
    """
    Parse numeric values and simple ranges into one representative number.

    Examples:
      20 -> 20
      "20" -> 20
      "10-30" -> 20
      "10–30" -> 20
      "<1" -> 1
      "NS" / "NR" / blank -> NaN
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    s = str(value).strip()
    if s == "" or s.upper() in {
        "NS", "NR", "NA", "N/A", "NONE", "NAN", "-", "NOT REPORTED"
    }:
        return np.nan

    s = (
        s.replace("–", "-")
         .replace("—", "-")
         .replace("−", "-")
         .replace(",", ".")
         .replace("minutes", "")
         .replace("minute", "")
         .replace("mins", "")
         .replace("min", "")
         .replace("mm", "")
         .replace("g", "")
         .replace("<", "")
         .replace(">", "")
         .replace("≤", "")
         .replace("≥", "")
         .replace("~", "")
         .replace("≈", "")
    )

    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not nums:
        return np.nan

    vals = [float(x) for x in nums]
    if len(vals) == 1:
        return vals[0]

    return float(np.mean(vals[:2]))


def safe_r2(y_true, y_pred, multioutput="uniform_average"):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.ndim == 1:
        if np.nanstd(y_true) < 1e-10:
            return np.nan
        try:
            return float(r2_score(y_true, y_pred))
        except Exception:
            return np.nan

    scores = []
    for j in range(y_true.shape[1]):
        if np.nanstd(y_true[:, j]) < 1e-10:
            continue
        try:
            scores.append(float(r2_score(y_true[:, j], y_pred[:, j])))
        except Exception:
            continue

    if not scores:
        return np.nan

    if multioutput == "raw_values":
        return np.asarray(scores)

    return float(np.nanmean(scores))


def close_to_100(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = np.maximum(y, 1e-10)
    return y / y.sum(axis=1, keepdims=True) * 100.0


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


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
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_ohe()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )


def fit_predict_renormalized(
    X_train: pd.DataFrame,
    y_train_raw: np.ndarray,
    X_test: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    seed: int,
) -> np.ndarray:
    pipe = Pipeline([
        ("preprocess", build_preprocessor(numeric_features, categorical_features)),
        ("model", MultiOutputRegressor(clone(get_xgb_model(seed)))),
    ])

    pipe.fit(X_train, y_train_raw)
    pred_raw = pipe.predict(X_test)
    return close_to_100(pred_raw)


def evaluate_prediction(y_true_raw: np.ndarray, y_pred_closed: np.ndarray) -> Dict[str, float]:
    y_true_closed = close_to_100(y_true_raw)
    pred_sum = y_pred_closed.sum(axis=1)

    abs_error = np.abs(y_true_closed - y_pred_closed)

    target_means = np.nanmean(np.abs(y_true_closed), axis=0)
    target_ranges = np.nanmax(y_true_closed, axis=0) - np.nanmin(y_true_closed, axis=0)

    mae_by_target = np.nanmean(abs_error, axis=0)
    rmse_by_target = np.sqrt(np.nanmean((y_true_closed - y_pred_closed) ** 2, axis=0))

    # WAPE is safer than MAPE for low-yield cases.
    denom = np.sum(np.abs(y_true_closed))
    wape_pct = float(np.sum(abs_error) / denom * 100.0) if denom > 0 else np.nan

    nmae_mean_pct = np.nanmean(mae_by_target / np.maximum(target_means, 1e-12) * 100.0)
    nrmse_range_pct = np.nanmean(rmse_by_target / np.maximum(target_ranges, 1e-12) * 100.0)

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
        "char_MAE": float(mae_by_target[0]),
        "oil_MAE": float(mae_by_target[1]),
        "gas_MAE": float(mae_by_target[2]),

        "WAPE_pct": wape_pct,
        "nMAE_by_target_mean_pct": float(nmae_mean_pct),
        "nRMSE_by_target_range_pct": float(nrmse_range_pct),

        "pred_sum_mean": float(np.mean(pred_sum)),
        "pred_sum_sd": float(np.std(pred_sum, ddof=1)) if len(pred_sum) > 1 else 0.0,
        "pct_pred_sum_outside_95_105": float(
            np.mean((pred_sum < 95) | (pred_sum > 105)) * 100.0
        ),
    }


# ============================================================
# 4. DATA PREPARATION
# ============================================================

def standardize_reporting_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for readable, fallback in REPORTING_FALLBACKS.items():
        if readable not in df.columns and fallback in df.columns:
            df[readable] = df[fallback]
    return df


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

    if "holding_time_min_parsed" in df.columns:
        df["holding_time_min_num"] = pd.to_numeric(
            df["holding_time_min_parsed"], errors="coerce"
        )
    else:
        df["holding_time_min_num"] = (
            df["holding_time_min"].apply(parse_numeric_or_midrange)
            if "holding_time_min" in df.columns
            else np.nan
        )

    if "particle_size_mm_parsed" in df.columns:
        df["particle_size_mm_mid"] = pd.to_numeric(
            df["particle_size_mm_parsed"], errors="coerce"
        )
    else:
        df["particle_size_mm_mid"] = (
            df["particle_size_mm"].apply(parse_numeric_or_midrange)
            if "particle_size_mm" in df.columns
            else np.nan
        )

    if "sample_mass_g_parsed" in df.columns:
        df["sample_mass_g_num"] = pd.to_numeric(
            df["sample_mass_g_parsed"], errors="coerce"
        )
    else:
        df["sample_mass_g_num"] = (
            df["sample_mass_g"].apply(parse_numeric_or_midrange)
            if "sample_mass_g" in df.columns
            else np.nan
        )

    return df


def add_validation_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    candidate_cols = [
        "study_feedstock_composition_group_id",
        "study_feedstock_composition_group",
        "study_feedstock_composition_id",
        "study_feedstock_group",
        "group_id",
        "group_key",
    ]

    for col in candidate_cols:
        if col in df.columns:
            df["validation_group_id_final"] = clean_string_series(df[col]).fillna("Unknown_group")
            print(f"Using existing validation group column: {col}")
            return df

    print("No existing validation group column detected.")
    print("Building fallback group from study_id + feedstock + rounded composition.")

    if "study_id" in df.columns:
        group = clean_string_series(df["study_id"]).fillna("Unknown_study")
    else:
        group = pd.Series(["Unknown_study"] * len(df), index=df.index)

    feedstock_cols = [
        "feedstock_name_std",
        "feedstock_name",
        "feedstock",
        "biomass_name",
        "biomass",
        "feedstock_class",
    ]

    available_feedstock_cols = [c for c in feedstock_cols if c in df.columns]

    if available_feedstock_cols:
        for c in available_feedstock_cols:
            group = group + "__" + c + "_" + clean_string_series(df[c]).fillna("Unknown")
    else:
        group = group + "__Unknown_feedstock"

    for c in PROXIMATE_FEATURES + ULTIMATE_FEATURES:
        rounded = pd.to_numeric(df[c], errors="coerce").round(2).astype(str)
        group = group + "__" + c + "_" + rounded

    if "validation_group_id" in df.columns:
        provided = clean_string_series(df["validation_group_id"])
        mapping = pd.DataFrame({"provided": provided, "canonical": group})
        if (
            mapping.groupby("provided")["canonical"].nunique().max() != 1
            or mapping.groupby("canonical")["provided"].nunique().max() != 1
        ):
            raise ValueError(
                "Provided validation_group_id is not one-to-one with the "
                "reconstructed study-feedstock-composition groups."
            )
        df["provided_validation_group_id"] = provided

    df["validation_group_id_final"] = group.astype(str)
    return df


def load_full_data() -> pd.DataFrame:
    workbook = resolve_input_workbook()

    print(f"Input workbook: {workbook.resolve()}")
    print(f"Input sheet: {INPUT_SHEET}")

    df = pd.read_excel(workbook, sheet_name=INPUT_SHEET)
    print(f"Rows loaded: {len(df)}")

    if len(df) != EXPECTED_FULL_ROWS:
        print(f"WARNING: expected {EXPECTED_FULL_ROWS} rows, got {len(df)}.")

    if "study_id" in df.columns:
        studies = df["study_id"].nunique()
        print(f"Studies loaded: {studies}")
        if studies != EXPECTED_FULL_STUDIES:
            print(f"WARNING: expected {EXPECTED_FULL_STUDIES} studies, got {studies}.")

    df = standardize_reporting_columns(df)
    df = add_engineered_features(df)
    df = add_validation_group(df)

    df["row_uid"] = np.arange(len(df))

    return df


def build_complete_detail_subset(full_df: pd.DataFrame) -> pd.DataFrame:
    required_full = (
        BASE_FEATURES
        + OPTIONAL_PROCESS_FEATURES
        + TARGET_COLS
        + ["study_id", "validation_group_id_final"]
    )

    missing = [c for c in required_full if c not in full_df.columns]
    if missing:
        raise KeyError(f"Missing required columns after feature engineering: {missing}")

    complete_detail = full_df.dropna(subset=required_full).copy().reset_index(drop=True)
    complete_detail["complete_detail_row_uid"] = np.arange(len(complete_detail))

    print("\nFixed all-optional subset:")
    print(f"Rows: {len(complete_detail)}")
    print(f"Studies: {complete_detail['study_id'].nunique()}")
    print(f"Validation groups: {complete_detail['validation_group_id_final'].nunique()}")

    if len(complete_detail) != EXPECTED_COMPLETE_DETAIL_ROWS:
        print(f"WARNING: expected n={EXPECTED_COMPLETE_DETAIL_ROWS}, got n={len(complete_detail)}.")

    if complete_detail["study_id"].nunique() != EXPECTED_COMPLETE_DETAIL_STUDIES:
        print(
            f"WARNING: expected {EXPECTED_COMPLETE_DETAIL_STUDIES} studies, "
            f"got {complete_detail['study_id'].nunique()}."
        )

    return complete_detail


def build_availability_table(full_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    rows.append({
        "case": "full_all_rows_reference",
        "required_optional_variables": "None",
        "rows": len(full_df),
        "studies": full_df["study_id"].nunique() if "study_id" in full_df.columns else np.nan,
        "validation_groups": full_df["validation_group_id_final"].nunique(),
        "row_pct_of_full_dataset": 100.0,
    })

    for label, col in [
        ("sample_mass", "sample_mass_g_num"),
        ("holding_time", "holding_time_min_num"),
        ("particle_size", "particle_size_mm_mid"),
    ]:
        mask = full_df[col].notna()
        rows.append({
            "case": label,
            "required_optional_variables": col,
            "rows": int(mask.sum()),
            "studies": full_df.loc[mask, "study_id"].nunique(),
            "validation_groups": full_df.loc[mask, "validation_group_id_final"].nunique(),
            "row_pct_of_full_dataset": float(mask.mean() * 100.0),
        })

    all_mask = full_df[OPTIONAL_PROCESS_FEATURES].notna().all(axis=1)
    rows.append({
        "case": "all_optional_complete_detail",
        "required_optional_variables": "; ".join(OPTIONAL_PROCESS_FEATURES),
        "rows": int(all_mask.sum()),
        "studies": full_df.loc[all_mask, "study_id"].nunique(),
        "validation_groups": full_df.loc[all_mask, "validation_group_id_final"].nunique(),
        "row_pct_of_full_dataset": float(all_mask.mean() * 100.0),
    })

    return pd.DataFrame(rows)


# ============================================================
# 5. VALIDATION RUNNERS
# ============================================================

def get_stratify_or_none(df: pd.DataFrame):
    if "feedstock_class" not in df.columns:
        return None

    s = clean_string_series(df["feedstock_class"]).fillna("Unknown")
    counts = s.value_counts()

    if counts.min() >= 2:
        return s.to_numpy()

    return None


def run_random_row(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    y = df[TARGET_COLS].to_numpy(dtype=float)
    idx = np.arange(len(df))
    stratify = get_stratify_or_none(df)

    for seed in RANDOM_SEEDS:
        train_idx, test_idx = train_test_split(
            idx,
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=stratify,
        )

        y_train = y[train_idx]
        y_test = y[test_idx]

        for scenario, spec in SCENARIOS.items():
            optional = spec["optional_features_added"]
            numeric_features = BASE_NUMERIC_FEATURES + optional
            categorical_features = BASE_CATEGORICAL_FEATURES
            features = numeric_features + categorical_features

            X = df[features].copy()
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            pred = fit_predict_renormalized(
                X_train=X_train,
                y_train_raw=y_train,
                X_test=X_test,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                seed=seed,
            )

            metrics = evaluate_prediction(y_test, pred)

            rows.append({
                "validation": "random_row",
                "scenario": scenario,
                "scenario_label": spec["label"],
                "target_mode": TARGET_MODE,
                "target_label": TARGET_LABEL,
                "seed_or_fold": f"seed_{seed}",
                "n_rows_total": len(df),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_studies_total": df["study_id"].nunique(),
                "n_groups_total": df["validation_group_id_final"].nunique(),
                "n_numeric_features": len(numeric_features),
                "n_categorical_features": len(categorical_features),
                "optional_features_added": "; ".join(optional) if optional else "None",
                **metrics,
            })

    return pd.DataFrame(rows)


def run_grouped_holdout(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    y = df[TARGET_COLS].to_numpy(dtype=float)
    idx = np.arange(len(df))
    groups = df["validation_group_id_final"].astype(str).to_numpy()

    for seed in GROUP_SPLIT_SEEDS:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=seed,
        )

        train_idx, test_idx = next(splitter.split(idx, y, groups=groups))

        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups.intersection(test_groups)

        if overlap:
            raise RuntimeError(
                f"Group leakage detected for seed {seed}: {len(overlap)} overlapping groups."
            )

        y_train = y[train_idx]
        y_test = y[test_idx]

        for scenario, spec in SCENARIOS.items():
            optional = spec["optional_features_added"]
            numeric_features = BASE_NUMERIC_FEATURES + optional
            categorical_features = BASE_CATEGORICAL_FEATURES
            features = numeric_features + categorical_features

            X = df[features].copy()
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            pred = fit_predict_renormalized(
                X_train=X_train,
                y_train_raw=y_train,
                X_test=X_test,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                seed=seed + 1000,
            )

            metrics = evaluate_prediction(y_test, pred)

            rows.append({
                "validation": "grouped_holdout",
                "scenario": scenario,
                "scenario_label": spec["label"],
                "target_mode": TARGET_MODE,
                "target_label": TARGET_LABEL,
                "seed_or_fold": f"group_seed_{seed}",
                "n_rows_total": len(df),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_studies_total": df["study_id"].nunique(),
                "n_groups_total": df["validation_group_id_final"].nunique(),
                "n_train_groups": len(train_groups),
                "n_test_groups": len(test_groups),
                "n_group_overlap": len(overlap),
                "n_numeric_features": len(numeric_features),
                "n_categorical_features": len(categorical_features),
                "optional_features_added": "; ".join(optional) if optional else "None",
                **metrics,
            })

    return pd.DataFrame(rows)


def run_loso_by_study(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rows = []
    fold_rows = []

    X_base = df.copy()
    y = df[TARGET_COLS].to_numpy(dtype=float)
    groups = df["study_id"].astype(str).to_numpy()

    logo = LeaveOneGroupOut()
    unique_studies = pd.Series(groups).nunique()

    for scenario, spec in SCENARIOS.items():
        optional = spec["optional_features_added"]
        numeric_features = BASE_NUMERIC_FEATURES + optional
        categorical_features = BASE_CATEGORICAL_FEATURES
        features = numeric_features + categorical_features

        all_true = []
        all_pred = []

        for fold_no, (train_idx, test_idx) in enumerate(
            logo.split(X_base, y, groups=groups), start=1
        ):
            held_out_study = str(groups[test_idx][0])

            X = df[features].copy()
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            pred = fit_predict_renormalized(
                X_train=X_train,
                y_train_raw=y_train,
                X_test=X_test,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                seed=2000 + fold_no,
            )

            all_true.append(y_test)
            all_pred.append(pred)

            fold_metrics = evaluate_prediction(y_test, pred)

            fold_rows.append({
                "validation": "LOSO_by_study",
                "scenario": scenario,
                "scenario_label": spec["label"],
                "target_mode": TARGET_MODE,
                "target_label": TARGET_LABEL,
                "fold_no": fold_no,
                "held_out_study": held_out_study,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_numeric_features": len(numeric_features),
                "n_categorical_features": len(categorical_features),
                "optional_features_added": "; ".join(optional) if optional else "None",
                **fold_metrics,
            })

            print(
                f"  LOSO {scenario:<30s} fold {fold_no}/{unique_studies}",
                end="\r",
            )

        y_true_pooled = np.vstack(all_true)
        y_pred_pooled = np.vstack(all_pred)

        pooled_metrics = evaluate_prediction(y_true_pooled, y_pred_pooled)

        pooled_rows.append({
            "validation": "LOSO_by_study",
            "scenario": scenario,
            "scenario_label": spec["label"],
            "target_mode": TARGET_MODE,
            "target_label": TARGET_LABEL,
            "seed_or_fold": "pooled_LOSO",
            "n_rows_total": len(df),
            "n_train": np.nan,
            "n_test": len(df),
            "n_studies_total": df["study_id"].nunique(),
            "n_groups_total": df["validation_group_id_final"].nunique(),
            "n_folds": unique_studies,
            "n_numeric_features": len(numeric_features),
            "n_categorical_features": len(categorical_features),
            "optional_features_added": "; ".join(optional) if optional else "None",
            **pooled_metrics,
        })

        print()

    return pd.DataFrame(pooled_rows), pd.DataFrame(fold_rows)


# ============================================================
# 6. SUMMARIES
# ============================================================

def summarize_performance(perf_long: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "validation",
        "scenario",
        "scenario_label",
        "target_mode",
        "target_label",
        "n_numeric_features",
        "n_categorical_features",
        "optional_features_added",
    ]

    metric_cols = [
        "multi_R2",
        "char_R2",
        "oil_R2",
        "gas_R2",
        "RMSE",
        "MAE",
        "char_MAE",
        "oil_MAE",
        "gas_MAE",
        "WAPE_pct",
        "nMAE_by_target_mean_pct",
        "nRMSE_by_target_range_pct",
        "pred_sum_mean",
        "pred_sum_sd",
        "pct_pred_sum_outside_95_105",
    ]

    summary_mean = perf_long.groupby(group_cols, as_index=False)[metric_cols].mean()
    summary_sd = perf_long.groupby(group_cols, as_index=False)[metric_cols].std()
    summary_sd = summary_sd.rename(columns={c: f"{c}_sd" for c in metric_cols})

    summary = summary_mean.merge(summary_sd, on=group_cols, how="left")

    summary["rank_within_validation"] = (
        summary.groupby("validation")["multi_R2"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    summary = summary.sort_values(
        ["validation", "rank_within_validation"]
    ).reset_index(drop=True)

    return summary


def build_delta_vs_baseline(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary[
        summary["scenario"].eq("complete_detail_baseline_no_optional")
    ][[
        "validation",
        "multi_R2",
        "RMSE",
        "MAE",
        "WAPE_pct",
        "nMAE_by_target_mean_pct",
        "nRMSE_by_target_range_pct",
    ]].copy()

    baseline = baseline.rename(columns={
        "multi_R2": "baseline_multi_R2",
        "RMSE": "baseline_RMSE",
        "MAE": "baseline_MAE",
        "WAPE_pct": "baseline_WAPE_pct",
        "nMAE_by_target_mean_pct": "baseline_nMAE_by_target_mean_pct",
        "nRMSE_by_target_range_pct": "baseline_nRMSE_by_target_range_pct",
    })

    out = summary.merge(baseline, on="validation", how="left")

    out["delta_multi_R2_vs_complete_detail_baseline"] = out["multi_R2"] - out["baseline_multi_R2"]
    out["delta_RMSE_vs_complete_detail_baseline"] = out["RMSE"] - out["baseline_RMSE"]
    out["delta_MAE_vs_complete_detail_baseline"] = out["MAE"] - out["baseline_MAE"]
    out["delta_WAPE_pct_vs_complete_detail_baseline"] = out["WAPE_pct"] - out["baseline_WAPE_pct"]
    out["delta_nMAE_pct_vs_complete_detail_baseline"] = (
        out["nMAE_by_target_mean_pct"] - out["baseline_nMAE_by_target_mean_pct"]
    )
    out["delta_nRMSE_pct_vs_complete_detail_baseline"] = (
        out["nRMSE_by_target_range_pct"] - out["baseline_nRMSE_by_target_range_pct"]
    )

    return out.sort_values(
        ["validation", "rank_within_validation"]
    ).reset_index(drop=True)


def build_decision_table(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in delta.iterrows():
        scenario = r["scenario"]
        validation = r["validation"]
        delta_r2 = r["delta_multi_R2_vs_complete_detail_baseline"]
        delta_mae = r["delta_MAE_vs_complete_detail_baseline"]

        if scenario == "complete_detail_baseline_no_optional":
            comparison = "Reference baseline on the fixed n=322 subset."
        elif delta_r2 >= 0.05 and delta_mae < 0:
            comparison = "Clear improvement over the matched-subset baseline."
        elif delta_r2 >= 0.02 and delta_mae <= 0:
            comparison = "Moderate improvement over the matched-subset baseline."
        elif delta_r2 > 0 and delta_mae < 0:
            comparison = "Small improvement over the matched-subset baseline."
        elif delta_r2 < -0.02:
            comparison = "Performance is lower than the matched-subset baseline."
        else:
            comparison = "No material change from the matched-subset baseline."

        rows.append({
            "validation": validation,
            "scenario": scenario,
            "scenario_label": r["scenario_label"],
            "multi_R2": r["multi_R2"],
            "delta_multi_R2_vs_complete_detail_baseline": delta_r2,
            "MAE": r["MAE"],
            "delta_MAE_vs_complete_detail_baseline": delta_mae,
            "WAPE_pct": r["WAPE_pct"],
            "delta_WAPE_pct_vs_complete_detail_baseline": r["delta_WAPE_pct_vs_complete_detail_baseline"],
            "comparison_to_baseline": comparison,
        })

    return pd.DataFrame(rows)


def build_scenario_table() -> pd.DataFrame:
    rows = []
    for scenario, spec in SCENARIOS.items():
        optional = spec["optional_features_added"]
        rows.append({
            "scenario": scenario,
            "scenario_label": spec["label"],
            "base_numeric_features": "; ".join(BASE_NUMERIC_FEATURES),
            "base_categorical_features": "; ".join(BASE_CATEGORICAL_FEATURES),
            "optional_features_added": "; ".join(optional) if optional else "None",
            "n_numeric_features": len(BASE_NUMERIC_FEATURES + optional),
            "n_categorical_features": len(BASE_CATEGORICAL_FEATURES),
        })
    return pd.DataFrame(rows)


def build_config_table(full_df: pd.DataFrame, complete_detail: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "input_workbook": str(resolve_input_workbook()),
        "input_sheet": INPUT_SHEET,
        "output_workbook": str(OUTPUT_WORKBOOK),
        "full_rows_loaded": len(full_df),
        "full_studies": full_df["study_id"].nunique(),
        "complete_detail_rows": len(complete_detail),
        "complete_detail_studies": complete_detail["study_id"].nunique(),
        "complete_detail_validation_groups": complete_detail["validation_group_id_final"].nunique(),
        "target_mode": TARGET_LABEL,
        "random_seeds": ", ".join(map(str, RANDOM_SEEDS)),
        "group_split_seeds": ", ".join(map(str, GROUP_SPLIT_SEEDS)),
        "test_size": TEST_SIZE,
        "main_question": (
            "Does adding holding time, particle size, and sample mass on the fixed n=322 subset "
            "improve performance enough to make the reduced subset the primary model architecture?"
        ),
    }])


# ============================================================
# 7. MAIN
# ============================================================

def main():
    print("\n====================================================")
    print("n=322 PROCESS-DETAIL SENSITIVITY")
    print("====================================================")

    full_df = load_full_data()
    availability = build_availability_table(full_df)
    complete_detail = build_complete_detail_subset(full_df)

    print("\nAvailability:")
    print(availability.to_string(index=False))

    print("\nRunning random-row validation...")
    perf_random = run_random_row(complete_detail)

    print("\nRunning grouped-holdout validation...")
    perf_grouped = run_grouped_holdout(complete_detail)

    print("\nRunning LOSO-by-study validation...")
    perf_loso, loso_folds = run_loso_by_study(complete_detail)

    perf_long = pd.concat([perf_random, perf_grouped, perf_loso], ignore_index=True)

    print("\nSummarizing...")
    perf_summary = summarize_performance(perf_long)
    delta = build_delta_vs_baseline(perf_summary)
    decision = build_decision_table(delta)

    config = build_config_table(full_df, complete_detail)
    scenarios = build_scenario_table()

    context_cols = [
        "row_uid",
        "complete_detail_row_uid",
        "study_id",
        "publication_year",
        "doi",
        "feedstock_name",
        "feedstock_class",
        "temperature_c",
        "heating_rate_c_min",
        "holding_time_min",
        "holding_time_min_parsed",
        "holding_time_min_num",
        "particle_size_mm",
        "particle_size_mm_parsed",
        "particle_size_mm_mid",
        "sample_mass_g",
        "sample_mass_g_parsed",
        "sample_mass_g_num",
        "validation_group_id_final",
    ]
    context_cols = [c for c in context_cols if c in complete_detail.columns]

    print("\nSaving workbook...")
    with pd.ExcelWriter(OUTPUT_WORKBOOK, engine="openpyxl") as writer:
        config.to_excel(writer, sheet_name="00_Config", index=False)
        availability.to_excel(writer, sheet_name="01_Availability", index=False)
        scenarios.to_excel(writer, sheet_name="02_Scenarios", index=False)
        complete_detail[context_cols].to_excel(writer, sheet_name="03_complete_detail_Row_Context", index=False)

        perf_long.to_excel(writer, sheet_name="10_Performance_Long", index=False)
        perf_summary.to_excel(writer, sheet_name="11_Performance_Summary", index=False)
        delta.to_excel(writer, sheet_name="12_Delta_vs_CompleteBase", index=False)
        decision.to_excel(writer, sheet_name="13_Sensitivity_Decision", index=False)
        loso_folds.to_excel(writer, sheet_name="20_LOSO_Fold_Metrics", index=False)

    print("\n====================================================")
    print("DONE")
    print("====================================================")
    print(f"Saved to: {OUTPUT_WORKBOOK.resolve()}")

    print("\nMain decision table:")
    show_cols = [
        "validation",
        "scenario",
        "multi_R2",
        "delta_multi_R2_vs_complete_detail_baseline",
        "MAE",
        "delta_MAE_vs_complete_detail_baseline",
        "comparison_to_baseline",
    ]
    print(decision[show_cols].to_string(index=False))

if __name__ == "__main__":
    main()
