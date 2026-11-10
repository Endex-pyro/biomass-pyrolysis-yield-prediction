"""Matched model-selection robustness analysis reported in Table 2.

XGBoost, ExtraTrees, and Random Forest are compared with a common feature set
under random-row, grouped-holdout, and leave-one-study-out validation for
Renormalized and ILR-transformed targets.
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
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
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
OUTPUT_WORKBOOK = output_file("07_model_selection_robustness_results.xlsx")

EXPECTED_ROWS = 442
EXPECTED_STUDIES = 42

RANDOM_SEEDS = list(range(10))
TEST_SIZE = 0.20

RUN_RANDOM_ROW = True
RUN_GROUPED_HOLDOUT = True
RUN_LOSO = True

TARGET_MODES = [
    "direct_forced_closed",
    "ILR_closed",
]

TARGET_MODE_LABELS = {
    "direct_forced_closed": "Renormalized",
    "ILR_closed": "ILR-transformed",
}

TARGET_COLS = [
    "char_yield_wt_pct",
    "oil_yield_wt_pct",
    "gas_yield_wt_pct",
]

TARGET_NAMES = ["char", "oil", "gas"]


# ============================================================
# 2. FEATURE DEFINITIONS — F2 CORE + SEVERITY
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

BASIS_PROXY_NUMERIC = [
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

BASIS_PROXY_CATEGORICAL = [
    "proximate_ultimate_basis_pair",
    "yield_reporting_profile",
]

NUMERIC_FEATURES = (
    PROCESS_FEATURES
    + PROXIMATE_FEATURES
    + ULTIMATE_FEATURES
    + BASIS_PROXY_NUMERIC
    + SEVERITY_FEATURES
)

CATEGORICAL_FEATURES = (
    REPORTING_FEATURES
    + BASIS_PROXY_CATEGORICAL
)

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# 3. BASIC HELPERS
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
        return np.asarray(scores, dtype=float)

    return float(np.nanmean(scores))


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


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
    x_oil = np.exp(
        -math.sqrt(1.0 / 6.0) * z[:, 0]
        + math.sqrt(1.0 / 2.0) * z[:, 1]
    )
    x_gas = np.exp(
        -math.sqrt(1.0 / 6.0) * z[:, 0]
        - math.sqrt(1.0 / 2.0) * z[:, 1]
    )

    comp = np.column_stack([x_char, x_oil, x_gas])
    return comp / comp.sum(axis=1, keepdims=True) * 100.0


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_ohe()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


# ============================================================
# 4. MODEL DEFINITIONS
# ============================================================

def get_candidate_models(seed: int) -> Dict[str, object]:
    """Return the three tree-ensemble models used in the matched comparison."""

    return {
        "XGBoost_base": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            max_depth=4,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            reg_alpha=0.0,
            random_state=seed,
            n_jobs=-1,
        ),

        "ExtraTrees_deep": ExtraTreesRegressor(
            n_estimators=700,
            min_samples_leaf=1,
            max_features=1.00,
            random_state=seed,
            n_jobs=-1,
        ),

        "RandomForest_deep": RandomForestRegressor(
            n_estimators=700,
            min_samples_leaf=1,
            max_features=1.00,
            random_state=seed,
            n_jobs=-1,
        ),
    }


def fit_pipeline(
    model,
    X_train: pd.DataFrame,
    y_train_raw: np.ndarray,
    target_mode: str,
) -> Pipeline:
    pipe = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", MultiOutputRegressor(clone(model))),
    ])

    if target_mode == "direct_forced_closed":
        pipe.fit(X_train, y_train_raw)
        return pipe

    if target_mode == "ILR_closed":
        y_train_ilr = to_ilr(y_train_raw)
        pipe.fit(X_train, y_train_ilr)
        return pipe

    raise ValueError(f"Unknown target mode: {target_mode}")


def predict_pipeline(pipe: Pipeline, X: pd.DataFrame, target_mode: str) -> np.ndarray:
    pred = pipe.predict(X)

    if target_mode == "direct_forced_closed":
        return close_to_100(pred)

    if target_mode == "ILR_closed":
        return from_ilr(pred)

    raise ValueError(f"Unknown target mode: {target_mode}")


# ============================================================
# 5. METRICS AND OUTPUT FRAMES
# ============================================================

def evaluate_prediction(y_true_raw: np.ndarray, y_pred_closed: np.ndarray) -> Dict[str, float]:
    y_true_closed = close_to_100(y_true_raw)
    pred_sum = y_pred_closed.sum(axis=1)

    denom = np.sum(np.abs(y_true_closed))
    wape_pct = (
        float(np.sum(np.abs(y_true_closed - y_pred_closed)) / denom * 100.0)
        if denom > 0
        else np.nan
    )

    residuals = y_true_closed - y_pred_closed
    residual_sum = residuals.sum(axis=1)

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

        "residual_sum_mean": float(np.mean(residual_sum)),
        "residual_sum_sd": float(np.std(residual_sum, ddof=1)) if len(residual_sum) > 1 else 0.0,
        "residual_sum_max_abs": float(np.max(np.abs(residual_sum))),
    }


def make_prediction_frame(
    df: pd.DataFrame,
    test_idx: np.ndarray,
    y_true_raw: np.ndarray,
    y_pred_closed: np.ndarray,
    validation: str,
    model_name: str,
    target_mode: str,
    split_id: str,
) -> pd.DataFrame:
    y_true_closed = close_to_100(y_true_raw)
    residuals = y_true_closed - y_pred_closed

    base = pd.DataFrame({
        "validation": validation,
        "model": model_name,
        "target_mode": target_mode,
        "target_label": TARGET_MODE_LABELS[target_mode],
        "split_id": split_id,
        "row_uid": df.iloc[test_idx]["row_uid"].to_numpy(),

        "study_id": df.iloc[test_idx]["study_id"].astype(str).to_numpy()
        if "study_id" in df.columns else "NA",

        "grouped_holdout_id": df.iloc[test_idx]["grouped_holdout_id"].astype(str).to_numpy()
        if "grouped_holdout_id" in df.columns else "NA",

        "true_raw_char": y_true_raw[:, 0],
        "true_raw_oil": y_true_raw[:, 1],
        "true_raw_gas": y_true_raw[:, 2],

        "true_closed_char": y_true_closed[:, 0],
        "true_closed_oil": y_true_closed[:, 1],
        "true_closed_gas": y_true_closed[:, 2],

        "pred_char": y_pred_closed[:, 0],
        "pred_oil": y_pred_closed[:, 1],
        "pred_gas": y_pred_closed[:, 2],

        "resid_char_true_minus_pred": residuals[:, 0],
        "resid_oil_true_minus_pred": residuals[:, 1],
        "resid_gas_true_minus_pred": residuals[:, 2],
        "resid_sum": residuals.sum(axis=1),
        "pred_sum": y_pred_closed.sum(axis=1),
    })

    optional_cols = [
        "feedstock_name",
        "feedstock_name_std",
        "feedstock_class",
        "temperature_c",
        "heating_rate_c_min",
    ]

    for col in optional_cols:
        if col in df.columns:
            base[col] = df.iloc[test_idx][col].to_numpy()

    return base


# ============================================================
# 6. DATA PREPARATION
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
    df = df.copy()

    candidate_group_cols = [
        "study_feedstock_composition_group",
        "study_feedstock_composition_id",
        "study_feedstock_comp_group",
        "study_feedstock_group",
        "group_id",
        "group_key",
        "validation_group",
    ]

    for col in candidate_group_cols:
        if col in df.columns:
            df["grouped_holdout_id"] = clean_string_series(df[col]).fillna("Unknown_group")
            print(f"Using existing grouped-holdout column: {col}")
            return df

    print("No existing group column detected.")
    print("Building fallback group from study_id + feedstock identity + rounded composition.")

    feedstock_candidate_cols = [
        "feedstock_name",
        "feedstock_name_std",
        "feedstock",
        "biomass_name",
        "biomass",
        "feedstock_type",
        "feedstock_class",
    ]

    available_feedstock_cols = [c for c in feedstock_candidate_cols if c in df.columns]

    if "study_id" in df.columns:
        group = clean_string_series(df["study_id"]).fillna("Unknown_study")
    else:
        group = pd.Series(["Unknown_study"] * len(df), index=df.index)

    if available_feedstock_cols:
        for c in available_feedstock_cols:
            group = group + "__" + c + "_" + clean_string_series(df[c]).fillna("Unknown")
    else:
        group = group + "__Unknown_feedstock"

    for c in PROXIMATE_FEATURES + ULTIMATE_FEATURES:
        rounded = pd.to_numeric(df[c], errors="coerce").round(2).astype(str)
        group = group + "__" + c + "_" + rounded

    df["grouped_holdout_id"] = group.astype(str)
    return df


def load_data() -> pd.DataFrame:
    workbook = resolve_input_workbook()

    print(f"Input workbook: {workbook.resolve()}")
    print(f"Input sheet: {INPUT_SHEET}")

    df = pd.read_excel(workbook, sheet_name=INPUT_SHEET)

    if len(df) != EXPECTED_ROWS:
        print(f"WARNING: expected {EXPECTED_ROWS} rows from sheet, got {len(df)} rows.")

    df = add_engineered_features(df)
    df = build_group_column(df)

    required = FEATURES + TARGET_COLS + ["study_id", "grouped_holdout_id"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns after feature engineering: {missing}")

    before = len(df)
    df = df.dropna(subset=required).copy().reset_index(drop=True)
    after = len(df)

    if before != after:
        print(f"WARNING: dropped {before - after} rows after final complete-case check.")

    if after != EXPECTED_ROWS:
        print(f"WARNING: expected {EXPECTED_ROWS} rows after check, got {after} rows.")

    n_studies = df["study_id"].nunique()
    if n_studies != EXPECTED_STUDIES:
        print(f"WARNING: expected {EXPECTED_STUDIES} studies, got {n_studies}.")

    df["row_uid"] = np.arange(len(df))

    return df


# ============================================================
# 7. VALIDATION RUNNERS
# ============================================================

def get_stratify_series(df: pd.DataFrame):
    if "feedstock_class" not in df.columns:
        return None

    fc = df["feedstock_class"].fillna("Unknown")
    counts = fc.value_counts()

    if counts.min() >= 2:
        return fc

    return None


def run_repeated_split_validation(
    df: pd.DataFrame,
    validation: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pred_frames = []

    X = df[FEATURES].copy()
    y = df[TARGET_COLS].to_numpy(dtype=float)
    idx = np.arange(len(df))

    stratify_series = get_stratify_series(df)

    for seed in RANDOM_SEEDS:
        if validation == "random_row":
            train_idx, test_idx = train_test_split(
                idx,
                test_size=TEST_SIZE,
                random_state=seed,
                stratify=stratify_series.to_numpy() if stratify_series is not None else None,
            )

        elif validation == "grouped_holdout":
            groups = df["grouped_holdout_id"].astype(str).to_numpy()
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=TEST_SIZE,
                random_state=seed,
            )
            train_idx, test_idx = next(splitter.split(idx, groups=groups))

            overlap = set(groups[train_idx]).intersection(set(groups[test_idx]))
            if overlap:
                raise RuntimeError(
                    f"Group leakage detected for seed {seed}: {len(overlap)} overlapping groups."
                )
        else:
            raise ValueError(f"Unknown repeated validation: {validation}")

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        models = get_candidate_models(seed)

        for model_name, model in models.items():
            for target_mode in TARGET_MODES:
                pipe = fit_pipeline(
                    model=model,
                    X_train=X_train,
                    y_train_raw=y_train,
                    target_mode=target_mode,
                )

                pred = predict_pipeline(pipe, X_test, target_mode)
                metrics = evaluate_prediction(y_test, pred)

                rows.append({
                    "validation": validation,
                    "model": model_name,
                    "target_mode": target_mode,
                    "target_label": TARGET_MODE_LABELS[target_mode],
                    "split_id": str(seed),
                    "seed": seed,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_folds_or_repeats": 1,
                    **metrics,
                })

                pred_frames.append(
                    make_prediction_frame(
                        df=df,
                        test_idx=test_idx,
                        y_true_raw=y_test,
                        y_pred_closed=pred,
                        validation=validation,
                        model_name=model_name,
                        target_mode=target_mode,
                        split_id=str(seed),
                    )
                )

        print(f"  {validation} seed {seed} complete")

    return pd.DataFrame(rows), pd.concat(pred_frames, ignore_index=True)


def run_loso_validation(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    fold_rows = []
    pred_frames_all = []

    X = df[FEATURES].copy()
    y = df[TARGET_COLS].to_numpy(dtype=float)
    groups = df["study_id"].astype(str).to_numpy()

    logo = LeaveOneGroupOut()
    unique_studies = pd.Series(groups).nunique()

    for model_name in get_candidate_models(seed=0).keys():
        for target_mode in TARGET_MODES:
            pooled_pred_frames = []

            for fold_no, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups), start=1):
                held_out_study = str(groups[test_idx][0])
                seed = 1000 + fold_no

                model = get_candidate_models(seed=seed)[model_name]

                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train = y[train_idx]
                y_test = y[test_idx]

                pipe = fit_pipeline(
                    model=model,
                    X_train=X_train,
                    y_train_raw=y_train,
                    target_mode=target_mode,
                )

                pred = predict_pipeline(pipe, X_test, target_mode)
                metrics = evaluate_prediction(y_test, pred)

                fold_rows.append({
                    "validation": "LOSO_by_study",
                    "model": model_name,
                    "target_mode": target_mode,
                    "target_label": TARGET_MODE_LABELS[target_mode],
                    "fold_no": fold_no,
                    "held_out_study": held_out_study,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    **metrics,
                })

                pf = make_prediction_frame(
                    df=df,
                    test_idx=test_idx,
                    y_true_raw=y_test,
                    y_pred_closed=pred,
                    validation="LOSO_by_study",
                    model_name=model_name,
                    target_mode=target_mode,
                    split_id=held_out_study,
                )
                pooled_pred_frames.append(pf)

                print(
                    f"  LOSO {model_name} | {TARGET_MODE_LABELS[target_mode]} | "
                    f"fold {fold_no}/{unique_studies}",
                    end="\r",
                )

            pooled = pd.concat(pooled_pred_frames, ignore_index=True)
            pred_frames_all.append(pooled)

            y_true_raw = pooled[["true_raw_char", "true_raw_oil", "true_raw_gas"]].to_numpy(dtype=float)
            y_pred = pooled[["pred_char", "pred_oil", "pred_gas"]].to_numpy(dtype=float)

            pooled_metrics = evaluate_prediction(y_true_raw, y_pred)

            rows.append({
                "validation": "LOSO_by_study",
                "model": model_name,
                "target_mode": target_mode,
                "target_label": TARGET_MODE_LABELS[target_mode],
                "split_id": "pooled_LOSO",
                "seed": np.nan,
                "n_train": np.nan,
                "n_test": len(pooled),
                "n_folds_or_repeats": unique_studies,
                **pooled_metrics,
            })

            print()

    return (
        pd.DataFrame(rows),
        pd.DataFrame(fold_rows),
        pd.concat(pred_frames_all, ignore_index=True),
    )


# ============================================================
# 8. SUMMARIES AND DECISION TABLES
# ============================================================

def summarize_performance(perf_long: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "validation",
        "model",
        "target_mode",
        "target_label",
    ]

    summary = (
        perf_long
        .groupby(group_cols, as_index=False)
        .agg(
            multi_R2_mean=("multi_R2", "mean"),
            multi_R2_sd=("multi_R2", "std"),
            char_R2_mean=("char_R2", "mean"),
            oil_R2_mean=("oil_R2", "mean"),
            gas_R2_mean=("gas_R2", "mean"),
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
            residual_sum_max_abs_mean=("residual_sum_max_abs", "mean"),
            n_units=("split_id", "nunique"),
            n_test_mean=("n_test", "mean"),
            n_folds_or_repeats_mean=("n_folds_or_repeats", "mean"),
        )
    )

    summary["rank_within_validation_mode"] = (
        summary
        .groupby(["validation", "target_mode"])["multi_R2_mean"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    summary["best_R2_within_validation_mode"] = (
        summary
        .groupby(["validation", "target_mode"])["multi_R2_mean"]
        .transform("max")
    )

    summary["delta_R2_to_best"] = (
        summary["multi_R2_mean"] - summary["best_R2_within_validation_mode"]
    )

    summary = summary.sort_values(
        ["validation", "target_mode", "rank_within_validation_mode"]
    ).reset_index(drop=True)

    return summary


def build_ranking_decision(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()

    notes = []
    for _, r in out.iterrows():
        rank = int(r["rank_within_validation_mode"])
        delta = float(r["delta_R2_to_best"])

        if rank == 1:
            note = "Highest R² for this validation and target mode."
        elif delta >= -0.02:
            note = "Within 0.02 R² of the leading model."
        elif delta >= -0.05:
            note = "Within 0.05 R² of the leading model."
        else:
            note = "More than 0.05 R² below the leading model."

        notes.append(note)

    out["ranking_note"] = notes

    return out[
        [
            "validation",
            "target_label",
            "model",
            "rank_within_validation_mode",
            "multi_R2_mean",
            "multi_R2_sd",
            "delta_R2_to_best",
            "RMSE_mean",
            "MAE_mean",
            "WAPE_pct_mean",
            "n_units",
            "ranking_note",
        ]
    ].copy()


def build_config_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "input_workbook": str(resolve_input_workbook()),
        "input_sheet": INPUT_SHEET,
        "output_workbook": str(OUTPUT_WORKBOOK),
        "rows_after_check": len(df),
        "studies": df["study_id"].nunique(),
        "grouped_holdout_groups": df["grouped_holdout_id"].nunique(),
        "expected_rows": EXPECTED_ROWS,
        "expected_studies": EXPECTED_STUDIES,
        "random_seeds": ", ".join(map(str, RANDOM_SEEDS)),
        "test_size": TEST_SIZE,
        "models": "XGBoost_base; ExtraTrees_deep; RandomForest_deep",
        "target_modes": "; ".join([TARGET_MODE_LABELS[m] for m in TARGET_MODES]),
        "feature_set": "F2_core_severity",
        "main_question": (
            "Is XGBoost justified as the reference workflow when compared fairly with "
            "ExtraTrees and RandomForest under Renormalized and ILR target handling?"
        ),
    }])


def build_model_settings_table() -> pd.DataFrame:
    rows = []

    for model_name, model in get_candidate_models(seed=0).items():
        params = model.get_params()
        rows.append({
            "model": model_name,
            "model_class": model.__class__.__name__,
            "main_parameters": str({
                k: params.get(k)
                for k in [
                    "n_estimators",
                    "max_depth",
                    "learning_rate",
                    "subsample",
                    "colsample_bytree",
                    "reg_lambda",
                    "reg_alpha",
                    "min_samples_leaf",
                    "max_features",
                    "bootstrap",
                ]
                if k in params
            }),
        })

    return pd.DataFrame(rows)


def build_group_audit(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    group_counts = df["grouped_holdout_id"].astype(str).value_counts()

    audit = pd.DataFrame([{
        "rows": len(df),
        "studies": df["study_id"].nunique(),
        "grouped_holdout_groups": df["grouped_holdout_id"].nunique(),
        "min_rows_per_group": int(group_counts.min()),
        "median_rows_per_group": float(group_counts.median()),
        "max_rows_per_group": int(group_counts.max()),
    }])

    group_sizes = (
        df.groupby("grouped_holdout_id", as_index=False)
        .agg(
            n_rows=("grouped_holdout_id", "size"),
            n_studies=("study_id", "nunique"),
        )
        .sort_values("n_rows", ascending=False)
        .reset_index(drop=True)
    )

    return audit, group_sizes


# ============================================================
# 9. RESIDUAL DIAGNOSTICS
# ============================================================

def build_residual_diagnostics(pred_long: pd.DataFrame):
    group_cols = ["validation", "model", "target_mode", "target_label"]

    target_summary_rows = []
    cov_yield_rows = []
    corr_yield_rows = []
    eig_yield_rows = []
    ilr_cov_rows = []
    ilr_corr_rows = []
    closure_rows = []

    target_resid_cols = [
        "resid_char_true_minus_pred",
        "resid_oil_true_minus_pred",
        "resid_gas_true_minus_pred",
    ]

    target_labels = ["char", "oil", "gas"]

    for keys, sub in pred_long.groupby(group_cols):
        validation, model, target_mode, target_label = keys

        residuals = sub[target_resid_cols].to_numpy(dtype=float)
        n = residuals.shape[0]

        resid_sum = residuals.sum(axis=1)
        closure_rows.append({
            "validation": validation,
            "model": model,
            "target_mode": target_mode,
            "target_label": target_label,
            "n_predictions": n,
            "residual_sum_mean": float(np.mean(resid_sum)),
            "residual_sum_sd": float(np.std(resid_sum, ddof=1)) if n > 1 else 0.0,
            "residual_sum_max_abs": float(np.max(np.abs(resid_sum))),
        })

        for j, target in enumerate(target_labels):
            r = residuals[:, j]
            target_summary_rows.append({
                "validation": validation,
                "model": model,
                "target_mode": target_mode,
                "target_label": target_label,
                "residual_target": target,
                "n_predictions": n,
                "bias_mean_true_minus_pred": float(np.mean(r)),
                "residual_sd": float(np.std(r, ddof=1)) if n > 1 else 0.0,
                "residual_MAE": float(np.mean(np.abs(r))),
                "residual_RMSE": float(np.sqrt(np.mean(r ** 2))),
            })

        cov_yield = np.cov(residuals, rowvar=False, ddof=1)
        corr_yield = np.corrcoef(residuals, rowvar=False)

        for i, row_name in enumerate(target_labels):
            for j, col_name in enumerate(target_labels):
                cov_yield_rows.append({
                    "validation": validation,
                    "model": model,
                    "target_mode": target_mode,
                    "target_label": target_label,
                    "space": "closed_yield_space",
                    "row": row_name,
                    "column": col_name,
                    "value": float(cov_yield[i, j]),
                })
                corr_yield_rows.append({
                    "validation": validation,
                    "model": model,
                    "target_mode": target_mode,
                    "target_label": target_label,
                    "space": "closed_yield_space",
                    "row": row_name,
                    "column": col_name,
                    "value": float(corr_yield[i, j]),
                })

        eigvals = np.linalg.eigvalsh(np.nan_to_num(cov_yield))
        for k, eig in enumerate(sorted(eigvals, reverse=True), start=1):
            eig_yield_rows.append({
                "validation": validation,
                "model": model,
                "target_mode": target_mode,
                "target_label": target_label,
                "space": "closed_yield_space",
                "eigenvalue_rank_desc": k,
                "eigenvalue": float(eig),
            })

        true_closed = sub[["true_closed_char", "true_closed_oil", "true_closed_gas"]].to_numpy(dtype=float)
        pred_closed = sub[["pred_char", "pred_oil", "pred_gas"]].to_numpy(dtype=float)

        true_z = to_ilr(true_closed)
        pred_z = to_ilr(pred_closed)
        resid_z = true_z - pred_z

        cov_ilr = np.cov(resid_z, rowvar=False, ddof=1)
        corr_ilr = np.corrcoef(resid_z, rowvar=False)

        ilr_labels = ["z1_char_vs_oil_gas", "z2_oil_vs_gas"]

        for i, row_name in enumerate(ilr_labels):
            for j, col_name in enumerate(ilr_labels):
                ilr_cov_rows.append({
                    "validation": validation,
                    "model": model,
                    "target_mode": target_mode,
                    "target_label": target_label,
                    "space": "ILR_coordinate_space",
                    "row": row_name,
                    "column": col_name,
                    "value": float(cov_ilr[i, j]),
                })
                ilr_corr_rows.append({
                    "validation": validation,
                    "model": model,
                    "target_mode": target_mode,
                    "target_label": target_label,
                    "space": "ILR_coordinate_space",
                    "row": row_name,
                    "column": col_name,
                    "value": float(corr_ilr[i, j]),
                })

    return (
        pd.DataFrame(target_summary_rows),
        pd.DataFrame(cov_yield_rows),
        pd.DataFrame(corr_yield_rows),
        pd.DataFrame(eig_yield_rows),
        pd.DataFrame(ilr_cov_rows),
        pd.DataFrame(ilr_corr_rows),
        pd.DataFrame(closure_rows),
    )


# ============================================================
# 10. MAIN
# ============================================================

def main():
    print("\n====================================================")
    print("MODEL-SELECTION ROBUSTNESS: RENORMALIZED VS ILR")
    print("====================================================")

    df = load_data()

    print(f"\nRows after check: {len(df)}")
    print(f"Studies: {df['study_id'].nunique()}")
    print(f"Grouped-holdout groups: {df['grouped_holdout_id'].nunique()}")
    print(f"Features: {len(FEATURES)}")
    print(f"Models: {', '.join(get_candidate_models(seed=0).keys())}")
    print(f"Target modes: {', '.join([TARGET_MODE_LABELS[m] for m in TARGET_MODES])}")

    perf_blocks = []
    pred_blocks = []
    loso_fold_metrics = pd.DataFrame()

    if RUN_RANDOM_ROW:
        print("\nRunning random-row validation...")
        perf_random, pred_random = run_repeated_split_validation(df, validation="random_row")
        perf_blocks.append(perf_random)
        pred_blocks.append(pred_random)

    if RUN_GROUPED_HOLDOUT:
        print("\nRunning grouped-holdout validation...")
        perf_grouped, pred_grouped = run_repeated_split_validation(df, validation="grouped_holdout")
        perf_blocks.append(perf_grouped)
        pred_blocks.append(pred_grouped)

    if RUN_LOSO:
        print("\nRunning LOSO-by-study validation...")
        perf_loso, loso_fold_metrics, pred_loso = run_loso_validation(df)
        perf_blocks.append(perf_loso)
        pred_blocks.append(pred_loso)

    perf_long = pd.concat(perf_blocks, ignore_index=True)
    pred_long = pd.concat(pred_blocks, ignore_index=True)

    print("\nSummarizing performance...")
    perf_summary = summarize_performance(perf_long)
    ranking_decision = build_ranking_decision(perf_summary)

    print("\nBuilding residual diagnostics...")
    (
        residual_target_summary,
        residual_cov_yield,
        residual_corr_yield,
        residual_eig_yield,
        ilr_residual_cov,
        ilr_residual_corr,
        residual_closure,
    ) = build_residual_diagnostics(pred_long)

    config = build_config_table(df)
    model_settings = build_model_settings_table()
    group_audit, group_sizes = build_group_audit(df)

    print("\nSaving workbook...")
    with pd.ExcelWriter(OUTPUT_WORKBOOK, engine="openpyxl") as writer:
        config.to_excel(writer, sheet_name="00_Config", index=False)
        model_settings.to_excel(writer, sheet_name="01_Model_Settings", index=False)
        group_audit.to_excel(writer, sheet_name="02_Group_Audit", index=False)
        group_sizes.to_excel(writer, sheet_name="03_Group_Sizes", index=False)

        perf_long.to_excel(writer, sheet_name="10_Performance_Long", index=False)
        perf_summary.to_excel(writer, sheet_name="11_Performance_Summary", index=False)
        ranking_decision.to_excel(writer, sheet_name="12_Ranking_Decision", index=False)

        if not loso_fold_metrics.empty:
            loso_fold_metrics.to_excel(writer, sheet_name="15_LOSO_Fold_Metrics", index=False)

        residual_target_summary.to_excel(writer, sheet_name="20_Residual_Targets", index=False)
        residual_cov_yield.to_excel(writer, sheet_name="21_Residual_Cov_Yield", index=False)
        residual_corr_yield.to_excel(writer, sheet_name="22_Residual_Corr_Yield", index=False)
        residual_eig_yield.to_excel(writer, sheet_name="23_Residual_Eig_Yield", index=False)
        ilr_residual_cov.to_excel(writer, sheet_name="24_ILR_Residual_Cov", index=False)
        ilr_residual_corr.to_excel(writer, sheet_name="25_ILR_Residual_Corr", index=False)
        residual_closure.to_excel(writer, sheet_name="26_Residual_Closure", index=False)

        pred_long.to_excel(writer, sheet_name="30_Predictions_Pooled", index=False)

    print("\n====================================================")
    print("DONE")
    print("====================================================")
    print(f"Saved to: {OUTPUT_WORKBOOK.resolve()}")

    print("\nMain ranking decision table:")
    show_cols = [
        "validation",
        "target_label",
        "model",
        "rank_within_validation_mode",
        "multi_R2_mean",
        "delta_R2_to_best",
        "RMSE_mean",
        "MAE_mean",
        "ranking_note",
    ]
    print(ranking_decision[show_cols].to_string(index=False))

if __name__ == "__main__":
    main()
