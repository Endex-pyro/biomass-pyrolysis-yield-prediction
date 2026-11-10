"""Sensitivity to stricter product-yield closure windows.

The 85–110 wt.% model-ready dataset is re-filtered at 90–110 and
95–105 wt.% and evaluated with matched random-row and grouped XGBoost analyses.
This reproduces the stricter-window checks described in Methodology Section 2.1.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from project_config import DATA_FILE, DATA_SHEET, output_file

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

warnings.filterwarnings("ignore")


# Analysis settings
INPUT_WORKBOOK = DATA_FILE
INPUT_SHEET = DATA_SHEET
OUTPUT_WORKBOOK = output_file("11_closure_window_sensitivity_results.xlsx")

EXPECTED_SOURCE_ROWS = 442
EXPECTED_MAIN_ROWS = 442
TEST_SIZE = 0.20

RANDOM_SEEDS = list(range(20))
GROUP_SPLIT_SEEDS = list(range(20))

TARGET_MODES = ["direct_raw", "direct_forced_closed", "ILR_closed"]
TARGET_LABELS = {
    "direct_raw": "Unconstrained",
    "direct_forced_closed": "Renormalized",
    "ILR_closed": "ILR-transformed",
}

FILTER_WINDOWS = {
    "W85_110_MAIN": {
        "label": "Primary closure window: 85-110 wt.%",
        "low": 85.0,
        "high": 110.0,
        "role": "main_filter",
    },
    "W90_110": {
        "label": "Stricter lower-bound filter: 90-110 wt.%",
        "low": 90.0,
        "high": 110.0,
        "role": "stricter_filter",
    },
    "W95_105": {
        "label": "Strict diagnostic subset: 95-105 wt.%",
        "low": 95.0,
        "high": 105.0,
        "role": "strict_diagnostic_subset",
    },
}

MIN_ROWS_FOR_MODELLING = 80
MIN_GROUPS_FOR_S1 = 8


# =============================================================================
# 2. FEATURE DEFINITIONS — CURRENT F2 CORE + SEVERITY SET
# =============================================================================

PROCESS_FEATURES = ["temperature_c", "heating_rate_c_min"]
PROXIMATE_FEATURES = ["volatile_matter_wt_pct", "fixed_carbon_wt_pct", "ash_wt_pct"]
ULTIMATE_FEATURES = ["carbon_wt_pct", "hydrogen_wt_pct", "nitrogen_wt_pct", "oxygen_wt_pct"]

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

BASIS_PROXY_CATEGORICAL = ["proximate_ultimate_basis_pair", "yield_reporting_profile"]

SEVERITY_FEATURES = ["inv_temp_K", "log_heating_rate", "temp_x_log_heating_rate"]

BASE_NUMERIC_FEATURES = (
    PROCESS_FEATURES + PROXIMATE_FEATURES + ULTIMATE_FEATURES + BASIS_PROXY_NUMERIC + SEVERITY_FEATURES
)
BASE_CATEGORICAL_FEATURES = REPORTING_FEATURES + BASIS_PROXY_CATEGORICAL

TARGET_COLS = ["char_yield_wt_pct", "oil_yield_wt_pct", "gas_yield_wt_pct"]

CONTEXT_COLS = [
    "source_excel_row", "study_id", "publication_year", "doi", "journal", "run_id",
    "feedstock_name", "feedstock_class", "temperature_c", "heating_rate_c_min",
    "yield_sum_wt_pct", "yield_basis", "oil_definition", "gas_yield_method",
    "pyrolysis_type", "proximate_basis", "ultimate_basis", "reactor_type",
    "reactor_variant", "operating_pressure",
]

CATEGORY_AUDIT_COLS = [
    "feedstock_class", "gas_yield_method", "oil_definition", "yield_basis",
    "proximate_basis", "ultimate_basis", "pyrolysis_type", "operating_pressure",
]

NUMERIC_AUDIT_COLS = [
    "temperature_c", "heating_rate_c_min", "char_yield_wt_pct", "oil_yield_wt_pct",
    "gas_yield_wt_pct", "yield_sum_wt_pct", "volatile_matter_wt_pct",
    "fixed_carbon_wt_pct", "ash_wt_pct", "carbon_wt_pct", "hydrogen_wt_pct",
    "nitrogen_wt_pct", "oxygen_wt_pct",
]


# =============================================================================
# 3. HELPERS
# =============================================================================

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
    return s.astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})


def safe_r2(y_true, y_pred, multioutput="uniform_average") -> float:
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


def to_ilr(comp_percent: np.ndarray) -> np.ndarray:
    comp = close_to_100(comp_percent) / 100.0
    comp = np.maximum(comp, 1e-12)
    char = comp[:, 0]
    oil = comp[:, 1]
    gas = comp[:, 2]
    z1 = math.sqrt(2.0 / 3.0) * np.log(char / np.sqrt(oil * gas))
    z2 = math.sqrt(1.0 / 2.0) * np.log(oil / gas)
    return np.column_stack([z1, z2])


def from_ilr(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    x_char = np.exp(math.sqrt(2.0 / 3.0) * z[:, 0])
    x_oil = np.exp(-math.sqrt(1.0 / 6.0) * z[:, 0] + math.sqrt(1.0 / 2.0) * z[:, 1])
    x_gas = np.exp(-math.sqrt(1.0 / 6.0) * z[:, 0] - math.sqrt(1.0 / 2.0) * z[:, 1])
    comp = np.column_stack([x_char, x_oil, x_gas])
    return comp / comp.sum(axis=1, keepdims=True) * 100.0


def standardize_reporting_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for readable, fallback in REPORTING_FALLBACKS.items():
        if readable not in df.columns and fallback in df.columns:
            df[readable] = df[fallback]
    return df


def add_basis_and_severity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in PROCESS_FEATURES + PROXIMATE_FEATURES + ULTIMATE_FEATURES + TARGET_COLS + ["yield_sum_wt_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in REPORTING_FEATURES:
        if c in df.columns:
            df[c] = clean_string_series(df[c])

    df["proximate_sum_raw_wt_pct"] = (
        df["volatile_matter_wt_pct"] + df["fixed_carbon_wt_pct"] + df["ash_wt_pct"]
    )
    df["proximate_sum_deviation_from_100"] = df["proximate_sum_raw_wt_pct"] - 100.0

    df["ultimate_CHNO_sum_raw_wt_pct"] = (
        df["carbon_wt_pct"] + df["hydrogen_wt_pct"] + df["nitrogen_wt_pct"] + df["oxygen_wt_pct"]
    )
    df["ultimate_CHNO_sum_deviation_from_100"] = df["ultimate_CHNO_sum_raw_wt_pct"] - 100.0

    df["proximate_ultimate_basis_pair"] = (
        clean_string_series(df["proximate_basis"]) + "__" + clean_string_series(df["ultimate_basis"])
    )
    df["yield_reporting_profile"] = (
        clean_string_series(df["yield_basis"]) + "__" + clean_string_series(df["oil_definition"])
        + "__" + clean_string_series(df["gas_yield_method"])
    )

    temp_c = pd.to_numeric(df["temperature_c"], errors="coerce")
    temp_k = temp_c + 273.15
    heating_rate = pd.to_numeric(df["heating_rate_c_min"], errors="coerce")

    df["inv_temp_K"] = np.where(temp_k > 0, 1.0 / temp_k, np.nan)
    df["log_heating_rate"] = np.log1p(np.maximum(heating_rate, 0))
    df["temp_x_log_heating_rate"] = temp_c * df["log_heating_rate"]

    return df


def add_validation_group_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "study_feedstock_composition_group_id" in df.columns:
        df["validation_group_id"] = df["study_feedstock_composition_group_id"].astype(str)
        return df

    submitted_group_id = (
        clean_string_series(df["validation_group_id"])
        if "validation_group_id" in df.columns
        else None
    )

    needed = [
        "study_id", "feedstock_name", "volatile_matter_wt_pct", "fixed_carbon_wt_pct",
        "ash_wt_pct", "carbon_wt_pct", "hydrogen_wt_pct", "nitrogen_wt_pct", "oxygen_wt_pct",
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = "MISSING"

    group_parts = []
    for _, r in df.iterrows():
        vals = [str(r.get("study_id", "")), str(r.get("feedstock_name", ""))]
        for col in needed[2:]:
            val = r.get(col, np.nan)
            try:
                vals.append(f"{float(val):.3f}")
            except Exception:
                vals.append(str(val))
        group_parts.append(" | ".join(vals))

    canonical_group_id = pd.Series(group_parts, index=df.index)
    if submitted_group_id is not None:
        mapping = pd.DataFrame(
            {
                "submitted": submitted_group_id,
                "canonical": canonical_group_id,
            }
        )
        if (
            mapping.groupby("submitted")["canonical"].nunique().max() != 1
            or mapping.groupby("canonical")["submitted"].nunique().max() != 1
        ):
            raise ValueError(
                "Submitted validation_group_id is not one-to-one with the "
                "canonical study-feedstock-composition groups."
            )
        df["submitted_validation_group_id"] = submitted_group_id

    df["validation_group_id"] = canonical_group_id
    return df


def load_source_data() -> pd.DataFrame:
    input_workbook = resolve_input_workbook()
    df = pd.read_excel(input_workbook, sheet_name=INPUT_SHEET)
    df = standardize_reporting_columns(df)

    if len(df) != EXPECTED_SOURCE_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_SOURCE_ROWS} rows, but loaded {len(df)}."
        )

    df = add_basis_and_severity_features(df)
    df = add_validation_group_id(df)

    required = BASE_NUMERIC_FEATURES + BASE_CATEGORICAL_FEATURES + TARGET_COLS + [
        "yield_sum_wt_pct", "study_id", "validation_group_id",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in source sheet: {missing}")

    for c in BASE_NUMERIC_FEATURES + TARGET_COLS + ["yield_sum_wt_pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in BASE_CATEGORICAL_FEATURES:
        df[c] = clean_string_series(df[c])

    # Keep only rows that are model-ready before closure-window filtering.
    # This makes the closure sensitivity fair because row changes reflect the
    # closure window rather than missing predictors.
    df = df.dropna(subset=required).copy().reset_index(drop=True)
    df["row_uid"] = np.arange(len(df))
    return df


def filter_by_window(df: pd.DataFrame, filter_key: str, spec: Dict) -> pd.DataFrame:
    out = df.copy()
    low, high = spec["low"], spec["high"]
    if low is not None:
        out = out[out["yield_sum_wt_pct"] >= low]
    if high is not None:
        out = out[out["yield_sum_wt_pct"] <= high]
    out = out.copy().reset_index(drop=True)
    out["filter_key"] = filter_key
    out["filter_label"] = spec["label"]
    out["filter_low"] = low if low is not None else np.nan
    out["filter_high"] = high if high is not None else np.nan
    out["filter_role"] = spec["role"]
    return out


# =============================================================================
# 4. MODEL / PIPELINE
# =============================================================================

def get_xgb_model(random_state=42):
    if not XGB_AVAILABLE:
        raise ImportError("xgboost is not installed. Install xgboost first.")
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


def make_pipeline(base_model) -> Pipeline:
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_ohe()),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, BASE_NUMERIC_FEATURES),
            ("cat", categorical_pipe, BASE_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    model = MultiOutputRegressor(clone(base_model), n_jobs=None)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def fit_predict(*, base_model, X_train: pd.DataFrame, y_train_raw: np.ndarray, X_test: pd.DataFrame, target_mode: str) -> np.ndarray:
    pipe = make_pipeline(base_model)
    if target_mode == "direct_raw":
        pipe.fit(X_train, y_train_raw)
        return pipe.predict(X_test)
    if target_mode == "direct_forced_closed":
        pipe.fit(X_train, y_train_raw)
        return close_to_100(pipe.predict(X_test))
    if target_mode == "ILR_closed":
        pipe.fit(X_train, to_ilr(y_train_raw))
        return from_ilr(pipe.predict(X_test))
    raise ValueError(f"Unknown target mode: {target_mode}")


def metric_block(
    y_true_raw: np.ndarray,
    y_pred: np.ndarray,
    *,
    filter_key: str,
    filter_spec: Dict,
    validation: str,
    target_mode: str,
    seed_or_split: str,
    n_train: int,
    n_test: int,
    extra: Dict | None = None,
) -> Dict:
    y_true_raw = np.asarray(y_true_raw, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true_closed = close_to_100(y_true_raw)
    pred_sum = y_pred.sum(axis=1)

    row = {
        "filter_key": filter_key,
        "filter_label": filter_spec["label"],
        "filter_low": filter_spec["low"],
        "filter_high": filter_spec["high"],
        "filter_role": filter_spec["role"],
        "validation": validation,
        "model": "XGBoost_base",
        "target_mode": target_mode,
        "target_label": TARGET_LABELS[target_mode],
        "seed_or_split": seed_or_split,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "ranking_R2": safe_r2(y_true_closed, y_pred, "uniform_average"),
        "orig_multi_R2": safe_r2(y_true_raw, y_pred, "uniform_average"),
        "orig_char_R2": safe_r2(y_true_raw[:, 0], y_pred[:, 0]),
        "orig_oil_R2": safe_r2(y_true_raw[:, 1], y_pred[:, 1]),
        "orig_gas_R2": safe_r2(y_true_raw[:, 2], y_pred[:, 2]),
        "orig_RMSE": float(np.sqrt(mean_squared_error(y_true_raw, y_pred))),
        "orig_MAE": float(mean_absolute_error(y_true_raw, y_pred)),
        "closed_true_multi_R2": safe_r2(y_true_closed, y_pred, "uniform_average"),
        "closed_true_char_R2": safe_r2(y_true_closed[:, 0], y_pred[:, 0]),
        "closed_true_oil_R2": safe_r2(y_true_closed[:, 1], y_pred[:, 1]),
        "closed_true_gas_R2": safe_r2(y_true_closed[:, 2], y_pred[:, 2]),
        "closed_true_RMSE": float(np.sqrt(mean_squared_error(y_true_closed, y_pred))),
        "closed_true_MAE": float(mean_absolute_error(y_true_closed, y_pred)),
        "pred_sum_mean": float(np.mean(pred_sum)),
        "pred_sum_sd": float(np.std(pred_sum, ddof=1)) if len(pred_sum) > 1 else 0.0,
        "pct_pred_sum_outside_95_105": float(np.mean((pred_sum < 95) | (pred_sum > 105)) * 100.0),
    }
    if extra:
        row.update(extra)
    return row


# =============================================================================
# 5. VALIDATION
# =============================================================================

def stratify_labels_or_none(df: pd.DataFrame):
    if "feedstock_class" not in df.columns:
        return None
    s = clean_string_series(df["feedstock_class"])
    if s.isna().any():
        return None
    counts = s.value_counts()
    if counts.min() < 2:
        return None
    return s.to_numpy()


def run_s0_random(df: pd.DataFrame, filter_key: str, filter_spec: Dict) -> pd.DataFrame:
    rows = []
    X = df[BASE_NUMERIC_FEATURES + BASE_CATEGORICAL_FEATURES].copy()
    y = df[TARGET_COLS].to_numpy(dtype=float)
    idx = np.arange(len(df))
    stratify = stratify_labels_or_none(df)

    for seed in RANDOM_SEEDS:
        train_idx, test_idx = train_test_split(
            idx, test_size=TEST_SIZE, random_state=seed, stratify=stratify,
        )
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for target_mode in TARGET_MODES:
            y_pred = fit_predict(
                base_model=get_xgb_model(random_state=seed),
                X_train=X_train,
                y_train_raw=y_train,
                X_test=X_test,
                target_mode=target_mode,
            )
            rows.append(metric_block(
                y_true_raw=y_test,
                y_pred=y_pred,
                filter_key=filter_key,
                filter_spec=filter_spec,
                validation="S0_random_row_repeated",
                target_mode=target_mode,
                seed_or_split=f"seed_{seed}",
                n_train=len(train_idx),
                n_test=len(test_idx),
                extra={
                    "n_rows_filter": len(df),
                    "n_studies_filter": df["study_id"].nunique(),
                    "n_groups_filter": df["validation_group_id"].nunique(),
                    "yield_sum_min": df["yield_sum_wt_pct"].min(),
                    "yield_sum_median": df["yield_sum_wt_pct"].median(),
                    "yield_sum_max": df["yield_sum_wt_pct"].max(),
                },
            ))
    return pd.DataFrame(rows)


def run_s1_group_shuffle(df: pd.DataFrame, filter_key: str, filter_spec: Dict) -> pd.DataFrame:
    rows = []
    X = df[BASE_NUMERIC_FEATURES + BASE_CATEGORICAL_FEATURES].copy()
    y = df[TARGET_COLS].to_numpy(dtype=float)
    groups = df["validation_group_id"].astype(str).to_numpy()
    n_groups = len(np.unique(groups))

    if n_groups < MIN_GROUPS_FOR_S1:
        return pd.DataFrame([{
            "filter_key": filter_key,
            "filter_label": filter_spec["label"],
            "validation": "S1_group_shuffle",
            "model": "XGBoost_base",
            "target_mode": "SKIPPED",
            "target_label": "SKIPPED",
            "seed_or_split": "SKIPPED",
            "note": f"Skipped because only {n_groups} validation groups were available.",
        }])

    for seed in GROUP_SPLIT_SEEDS:
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])

        for target_mode in TARGET_MODES:
            y_pred = fit_predict(
                base_model=get_xgb_model(random_state=seed + 1000),
                X_train=X_train,
                y_train_raw=y_train,
                X_test=X_test,
                target_mode=target_mode,
            )
            rows.append(metric_block(
                y_true_raw=y_test,
                y_pred=y_pred,
                filter_key=filter_key,
                filter_spec=filter_spec,
                validation="S1_group_shuffle",
                target_mode=target_mode,
                seed_or_split=f"group_seed_{seed}",
                n_train=len(train_idx),
                n_test=len(test_idx),
                extra={
                    "n_rows_filter": len(df),
                    "n_studies_filter": df["study_id"].nunique(),
                    "n_groups_filter": n_groups,
                    "n_train_groups": len(train_groups),
                    "n_test_groups": len(test_groups),
                    "group_overlap": len(train_groups.intersection(test_groups)),
                    "yield_sum_min": df["yield_sum_wt_pct"].min(),
                    "yield_sum_median": df["yield_sum_wt_pct"].median(),
                    "yield_sum_max": df["yield_sum_wt_pct"].max(),
                },
            ))
    return pd.DataFrame(rows)


# =============================================================================
# 6. AUDITS AND SUMMARIES
# =============================================================================

def build_filter_retention_audit(source_df: pd.DataFrame, filtered: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    total_rows = len(source_df)
    total_studies = source_df["study_id"].nunique()
    total_groups = source_df["validation_group_id"].nunique()

    for key, spec in FILTER_WINDOWS.items():
        fdf = filtered[key]
        excluded = source_df[~source_df["row_uid"].isin(set(fdf["row_uid"]))]
        below = source_df[source_df["yield_sum_wt_pct"] < spec["low"]] if spec["low"] is not None else source_df.iloc[0:0]
        above = source_df[source_df["yield_sum_wt_pct"] > spec["high"]] if spec["high"] is not None else source_df.iloc[0:0]

        rows.append({
            "filter_key": key,
            "filter_label": spec["label"],
            "filter_low": spec["low"],
            "filter_high": spec["high"],
            "filter_role": spec["role"],
            "source_rows": total_rows,
            "retained_rows": len(fdf),
            "excluded_rows": len(excluded),
            "retained_pct_of_source": 100.0 * len(fdf) / total_rows,
            "source_studies": total_studies,
            "retained_studies": fdf["study_id"].nunique(),
            "excluded_studies": excluded["study_id"].nunique(),
            "source_groups": total_groups,
            "retained_groups": fdf["validation_group_id"].nunique(),
            "excluded_below_low_rows": len(below),
            "excluded_above_high_rows": len(above),
            "yield_sum_min_retained": fdf["yield_sum_wt_pct"].min() if len(fdf) else np.nan,
            "yield_sum_median_retained": fdf["yield_sum_wt_pct"].median() if len(fdf) else np.nan,
            "yield_sum_max_retained": fdf["yield_sum_wt_pct"].max() if len(fdf) else np.nan,
        })
    return pd.DataFrame(rows)


def build_retention_by_category(source_df: pd.DataFrame, filtered: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key, spec in FILTER_WINDOWS.items():
        retained_uids = set(filtered[key]["row_uid"])
        tmp = source_df.copy()
        tmp["retained"] = tmp["row_uid"].isin(retained_uids)
        for col in CATEGORY_AUDIT_COLS:
            if col not in tmp.columns:
                continue
            for level, sdf in tmp.groupby(col, dropna=False):
                total = len(sdf)
                retained = int(sdf["retained"].sum())
                rows.append({
                    "filter_key": key,
                    "filter_label": spec["label"],
                    "category_variable": col,
                    "category_level": level,
                    "source_n": total,
                    "retained_n": retained,
                    "excluded_n": total - retained,
                    "retained_pct_within_category": 100.0 * retained / total if total else np.nan,
                    "category_pct_among_retained": 100.0 * retained / len(filtered[key]) if len(filtered[key]) else np.nan,
                })
    return pd.DataFrame(rows)


def build_retained_excluded_stats(source_df: pd.DataFrame, filtered: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key, spec in FILTER_WINDOWS.items():
        retained_uids = set(filtered[key]["row_uid"])
        for status, sdf in [
            ("retained", source_df[source_df["row_uid"].isin(retained_uids)]),
            ("excluded", source_df[~source_df["row_uid"].isin(retained_uids)]),
        ]:
            for col in NUMERIC_AUDIT_COLS:
                if col not in sdf.columns or len(sdf) == 0:
                    rows.append({
                        "filter_key": key, "filter_label": spec["label"], "record_status": status,
                        "variable": col, "n": len(sdf), "median": np.nan, "min": np.nan, "max": np.nan,
                    })
                    continue
                vals = pd.to_numeric(sdf[col], errors="coerce").dropna()
                rows.append({
                    "filter_key": key,
                    "filter_label": spec["label"],
                    "record_status": status,
                    "variable": col,
                    "n": int(vals.shape[0]),
                    "median": float(vals.median()) if len(vals) else np.nan,
                    "min": float(vals.min()) if len(vals) else np.nan,
                    "max": float(vals.max()) if len(vals) else np.nan,
                })
    return pd.DataFrame(rows)


def summarize_results(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    detail = detail[detail["target_mode"].isin(TARGET_MODES)].copy()
    if detail.empty:
        return pd.DataFrame()

    group_cols = [
        "filter_key", "filter_label", "filter_low", "filter_high", "filter_role",
        "validation", "model", "target_mode", "target_label",
    ]
    metric_cols = [
        "ranking_R2", "orig_multi_R2", "orig_char_R2", "orig_oil_R2", "orig_gas_R2",
        "orig_RMSE", "orig_MAE", "closed_true_multi_R2", "closed_true_char_R2",
        "closed_true_oil_R2", "closed_true_gas_R2", "closed_true_RMSE",
        "closed_true_MAE", "pred_sum_mean", "pred_sum_sd",
        "pct_pred_sum_outside_95_105", "n_rows_filter", "n_studies_filter",
        "n_groups_filter", "yield_sum_min", "yield_sum_median", "yield_sum_max",
    ]
    existing = [c for c in metric_cols if c in detail.columns]
    mean = detail.groupby(group_cols, as_index=False)[existing].mean()
    sd = detail.groupby(group_cols, as_index=False)[existing].std().rename(columns={c: f"{c}_sd" for c in existing})
    out = mean.merge(sd, on=group_cols, how="left")
    return out.sort_values(["validation", "target_mode", "filter_key"]).reset_index(drop=True)


def build_compact_summary(retention: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Return one Renormalized summary row per closure window."""
    if summary.empty:
        return retention.copy()

    renorm = summary[summary["target_mode"].eq("direct_forced_closed")].copy()
    renorm = renorm[
        renorm["validation"].isin(["S0_random_row_repeated", "S1_group_shuffle"])
    ]

    compact_metrics = renorm.pivot_table(
        index="filter_key",
        columns="validation",
        values=["ranking_R2", "closed_true_RMSE", "closed_true_MAE"],
        aggfunc="first",
    )
    compact_metrics.columns = [f"{metric}_{validation}" for metric, validation in compact_metrics.columns]
    compact_metrics = compact_metrics.reset_index()

    keep = [
        "filter_key", "filter_label", "filter_low", "filter_high", "filter_role",
        "source_rows", "retained_rows", "excluded_rows", "retained_pct_of_source",
        "retained_studies", "retained_groups", "excluded_below_low_rows",
        "excluded_above_high_rows", "yield_sum_min_retained", "yield_sum_median_retained",
        "yield_sum_max_retained",
    ]
    out = retention[keep].merge(compact_metrics, on="filter_key", how="left")
    return out


def build_model_matrices_compact(filtered: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for key, fdf in filtered.items():
        spec = FILTER_WINDOWS[key]
        keep_cols = [c for c in CONTEXT_COLS if c in fdf.columns]
        keep_cols += ["validation_group_id", "row_uid"] + BASE_NUMERIC_FEATURES + BASE_CATEGORICAL_FEATURES + TARGET_COLS
        keep_cols = list(dict.fromkeys([c for c in keep_cols if c in fdf.columns]))
        tmp = fdf[keep_cols].copy()
        tmp.insert(0, "filter_key", key)
        tmp.insert(1, "filter_label", spec["label"])
        tmp.insert(2, "filter_role", spec["role"])
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    print("\n==============================================================")
    print("SUPPLEMENTARY SENSITIVITY — MASS-CLOSURE FILTER BOUNDARY")
    print("==============================================================")

    source_df = load_source_data()
    print(f"\nSource rows after model-ready pre-check: {len(source_df)}")
    print(f"Studies: {source_df['study_id'].nunique()}")
    print(f"Validation groups: {source_df['validation_group_id'].nunique()}")
    print(f"Yield-sum range: {source_df['yield_sum_wt_pct'].min():.2f} to {source_df['yield_sum_wt_pct'].max():.2f}")

    filtered = {key: filter_by_window(source_df, key, spec) for key, spec in FILTER_WINDOWS.items()}
    if len(filtered["W85_110_MAIN"]) != EXPECTED_MAIN_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_MAIN_ROWS} rows in the primary window, but "
            f"retained {len(filtered['W85_110_MAIN'])}."
        )

    retention = build_filter_retention_audit(source_df, filtered)
    retention_by_category = build_retention_by_category(source_df, filtered)
    retained_excluded_stats = build_retained_excluded_stats(source_df, filtered)

    print("\nFilter retention audit:")
    print(retention[[
        "filter_key", "retained_rows", "excluded_rows", "retained_studies",
        "retained_groups", "yield_sum_min_retained", "yield_sum_max_retained",
    ]].to_string(index=False))

    s0_list = []
    s1_list = []

    print("\nRunning S0 repeated random-row and S1 grouped closure-filter sensitivity...")
    for key, spec in FILTER_WINDOWS.items():
        fdf = filtered[key]
        if len(fdf) < MIN_ROWS_FOR_MODELLING:
            print(f"  Skipping {key}: only {len(fdf)} rows (<{MIN_ROWS_FOR_MODELLING}).")
            continue
        print(f"  Running {key}: rows={len(fdf)}, studies={fdf['study_id'].nunique()}, groups={fdf['validation_group_id'].nunique()}")
        s0_list.append(run_s0_random(fdf, key, spec))
        s1_list.append(run_s1_group_shuffle(fdf, key, spec))

    s0_detail = pd.concat(s0_list, ignore_index=True) if s0_list else pd.DataFrame()
    s1_detail = pd.concat(s1_list, ignore_index=True) if s1_list else pd.DataFrame()
    all_detail = pd.concat([s0_detail, s1_detail], ignore_index=True)
    summary = summarize_results(all_detail)
    compact_summary = build_compact_summary(retention, summary)
    model_matrices = build_model_matrices_compact(filtered)

    print("\nSaving workbook...")
    with pd.ExcelWriter(OUTPUT_WORKBOOK, engine="openpyxl") as writer:
        retention.to_excel(writer, sheet_name="50_Filter_Retention_Audit", index=False)
        retention_by_category.to_excel(writer, sheet_name="51_Retention_By_Category", index=False)
        retained_excluded_stats.to_excel(writer, sheet_name="52_Retained_Excluded_Stats", index=False)
        s0_detail.to_excel(writer, sheet_name="53_S0_Random_Detailed", index=False)
        s1_detail.to_excel(writer, sheet_name="54_S1_GroupShuffle_Detailed", index=False)
        summary.to_excel(writer, sheet_name="55_Filter_Model_Summary", index=False)
        compact_summary.to_excel(writer, sheet_name="56_Compact_Summary", index=False)
        model_matrices.to_excel(writer, sheet_name="57_Model_Matrices_By_Filter", index=False)

    print("\n==============================================================")
    print("MASS-CLOSURE FILTER SENSITIVITY COMPLETE")
    print("==============================================================")
    print(f"Saved to: {OUTPUT_WORKBOOK.resolve()}")

    if not compact_summary.empty:
        print("\nCompact summary preview:")
        preview_cols = [
            "filter_key", "retained_rows", "retained_studies",
            "ranking_R2_S0_random_row_repeated", "closed_true_RMSE_S0_random_row_repeated",
            "ranking_R2_S1_group_shuffle", "closed_true_RMSE_S1_group_shuffle",
        ]
        preview_cols = [c for c in preview_cols if c in compact_summary.columns]
        print(compact_summary[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
