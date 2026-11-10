"""Random-row, grouped, and leave-one-study-out validation ladder.

This analysis supplies the selected XGBoost metrics in Figure 3 and the fold
diagnostics for held-out study evaluation.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from project_config import DATA_FILE, DATA_SHEET, output_file

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

# Analysis settings
INPUT_WORKBOOK = DATA_FILE
INPUT_SHEET = DATA_SHEET
OUTPUT_WORKBOOK = output_file("04_validation_ladder_results.xlsx")

EXPECTED_ROWS_LOADED = 442
EXPECTED_STUDIES = 42

RANDOM_SEEDS = list(range(20))  # repeated random-row benchmark
TEST_SIZE = 0.20

# Grouped split search settings
SIM_GROUP_COL_PREFERRED = "study_feedstock_composition_group_id"  # created if missing
SIM_N_CANDIDATE_SPLITS = 300
SIM_KEEP_TOP_SPLITS = 20
SIM_RANDOM_STATE = 42

# LOSO settings
MIN_TEST_ROWS_PER_STUDY_FOR_FOLD_METRICS = 3

# =============================================================================
# MODEL SETTINGS
# =============================================================================

MODEL_SETTINGS = {}

if XGB_AVAILABLE:
    MODEL_SETTINGS["XGBoost_base"] = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        reg_alpha=0.0,
        random_state=42,
        n_jobs=-1,
    )
else:
    print("WARNING: xgboost is not installed. XGBoost will be skipped.")

MODEL_SETTINGS["GBR_deep"] = GradientBoostingRegressor(
    n_estimators=250,
    learning_rate=0.035,
    max_depth=4,
    random_state=42,
)

MODEL_SETTINGS["ExtraTrees_deep"] = ExtraTreesRegressor(
    n_estimators=700,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=1,
    max_features=1.00,
)

MODEL_SETTINGS["RandomForest_deep"] = RandomForestRegressor(
    n_estimators=700,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=1,
    max_features=1.00,
)

if CATBOOST_AVAILABLE:
    MODEL_SETTINGS["CatBoost_base"] = CatBoostRegressor(
        iterations=500,
        depth=4,
        learning_rate=0.035,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
    )
else:
    print("WARNING: catboost is not installed. CatBoost will be skipped.")

TARGET_MODES = ["direct_raw", "direct_forced_closed", "ILR_closed"]

TARGET_LABELS = {
    "direct_raw": "Unconstrained",
    "direct_forced_closed": "Renormalized",
    "ILR_closed": "ILR-transformed",
}

# =============================================================================
# FEATURE/TARGET DEFINITIONS
# =============================================================================

NUMERIC_FEATURES = [
    "temperature_c",
    "heating_rate_c_min",
    "volatile_matter_wt_pct",
    "fixed_carbon_wt_pct",
    "ash_wt_pct",
    "carbon_wt_pct",
    "hydrogen_wt_pct",
    "nitrogen_wt_pct",
    "oxygen_wt_pct",
    # reporting/basis diagnostic proxies
    "proximate_sum_VM_FC_Ash",
    "ultimate_CHNO_sum",
]

CATEGORICAL_FEATURES = [
    "yield_basis_std",
    "oil_definition_v3",
    "gas_yield_method_v3",
    "pyrolysis_type_v3",
    "proximate_basis_std",
    "ultimate_basis_std",
    "composition_basis_pair",
    "yield_reporting_profile",
]

TARGET_COLS = ["char_yield_wt_pct", "oil_yield_wt_pct", "gas_yield_wt_pct"]

CONTEXT_COLS = [
    "source_excel_row",
    "study_id",
    "publication_year",
    "doi",
    "journal",
    "run_id",
    "feedstock_name",
    "feedstock_class",
    "yield_sum_wt_pct",
    "sample_mass_g",
    "particle_size_mm",
    "holding_time_min",
    "reactor_type",
    "reactor_variant",
    "reactor_mode",
    "reactor_scale",
    "operating_pressure",
]

# =============================================================================
# TRANSFORMS AND METRICS
# =============================================================================

def close_to_100(y: np.ndarray) -> np.ndarray:
    """Normalise row-wise to sum to 100."""
    y = np.asarray(y, dtype=float)
    y = np.maximum(y, 1e-9)
    return y / y.sum(axis=1, keepdims=True) * 100.0


def to_ilr(comp_percent: np.ndarray) -> np.ndarray:
    """3-part composition in percent -> 2 ILR coordinates."""
    comp = close_to_100(comp_percent) / 100.0
    comp = np.maximum(comp, 1e-12)
    g23 = np.sqrt(comp[:, 1] * comp[:, 2])
    z1 = math.sqrt(2.0 / 3.0) * np.log(comp[:, 0] / g23)
    z2 = math.sqrt(1.0 / 2.0) * np.log(comp[:, 1] / comp[:, 2])
    return np.column_stack([z1, z2])


def from_ilr(z: np.ndarray) -> np.ndarray:
    """2 ILR coordinates -> 3-part composition in percent."""
    z = np.asarray(z, dtype=float)
    x1 = np.exp(math.sqrt(2.0 / 3.0) * z[:, 0])
    x2 = np.exp(-math.sqrt(1.0 / 6.0) * z[:, 0] + math.sqrt(1.0 / 2.0) * z[:, 1])
    x3 = np.exp(-math.sqrt(1.0 / 6.0) * z[:, 0] - math.sqrt(1.0 / 2.0) * z[:, 1])
    comp = np.column_stack([x1, x2, x3])
    return comp / comp.sum(axis=1, keepdims=True) * 100.0


def safe_r2(y_true, y_pred, multioutput="uniform_average") -> float:
    """
    R² with protection against undefined/unstable constant-target folds.

    For 1D targets, return NaN if the held-out target is effectively constant.
    For multi-output targets, compute per-output R² only for non-constant outputs
    and average the valid outputs. This avoids treating constant tiny LOSO folds
    as meaningful R² values.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.ndim == 1:
        if np.nanstd(y_true) < 1e-10:
            return np.nan
        try:
            return float(r2_score(y_true, y_pred))
        except Exception:
            return np.nan

    valid_scores = []
    for j in range(y_true.shape[1]):
        if np.nanstd(y_true[:, j]) < 1e-10:
            continue
        try:
            valid_scores.append(float(r2_score(y_true[:, j], y_pred[:, j])))
        except Exception:
            continue

    if not valid_scores:
        return np.nan

    if multioutput == "raw_values":
        return np.array(valid_scores)

    return float(np.nanmean(valid_scores))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metric_block(
    *,
    y_true_raw: np.ndarray,
    y_pred: np.ndarray,
    validation: str,
    split_id: str,
    model_name: str,
    target_mode: str,
    n_train: int,
    n_test: int,
    extra: dict | None = None,
) -> dict:
    """Return metrics against raw and closed-normalised true yields."""
    y_true_raw = np.asarray(y_true_raw, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true_closed = close_to_100(y_true_raw)
    pred_sum = y_pred.sum(axis=1)

    # Single cross-mode ranking metric for fair comparison.
    # All target-handling modes are ranked against the same closed-normalized truth.
    ranking_R2 = safe_r2(y_true_closed, y_pred, "uniform_average")

    row = {
        "validation": validation,
        "split_id": split_id,
        "model": model_name,
        "target_mode": target_mode,
        "target_label": TARGET_LABELS.get(target_mode, target_mode),
        "n_train": int(n_train) if pd.notna(n_train) else np.nan,
        "n_test":  int(n_test)  if pd.notna(n_test)  else np.nan,
        "ranking_R2": ranking_R2,

        # Metrics against original reported yields
        "orig_multi_R2": safe_r2(y_true_raw, y_pred, "uniform_average"),
        "orig_char_R2": safe_r2(y_true_raw[:, 0], y_pred[:, 0]),
        "orig_oil_R2": safe_r2(y_true_raw[:, 1], y_pred[:, 1]),
        "orig_gas_R2": safe_r2(y_true_raw[:, 2], y_pred[:, 2]),
        "orig_RMSE": rmse(y_true_raw, y_pred),
        "orig_MAE": float(mean_absolute_error(y_true_raw, y_pred)),

        # Metrics against closed/normalised true yields
        "closed_true_multi_R2": safe_r2(y_true_closed, y_pred, "uniform_average"),
        "closed_true_char_R2": safe_r2(y_true_closed[:, 0], y_pred[:, 0]),
        "closed_true_oil_R2": safe_r2(y_true_closed[:, 1], y_pred[:, 1]),
        "closed_true_gas_R2": safe_r2(y_true_closed[:, 2], y_pred[:, 2]),
        "closed_true_RMSE": rmse(y_true_closed, y_pred),
        "closed_true_MAE": float(mean_absolute_error(y_true_closed, y_pred)),

        # Closure diagnostics
        "pred_sum_mean": float(np.mean(pred_sum)),
        "pred_sum_sd": float(np.std(pred_sum, ddof=1)) if len(pred_sum) > 1 else 0.0,
        "pct_pred_sum_outside_95_105": float(np.mean((pred_sum < 95.0) | (pred_sum > 105.0)) * 100.0),
    }
    if extra:
        row.update(extra)
    return row

# =============================================================================
# PIPELINE / PREDICTION
# =============================================================================

def make_preprocessor() -> ColumnTransformer:
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe,      NUMERIC_FEATURES),
            ("cat", categorical_pipe,  CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def fit_predict(
    base_model,
    target_mode: str,
    X_train: pd.DataFrame,
    y_train_raw: np.ndarray,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Fit model under one target mode and return char/oil/gas predictions."""
    preprocessor = make_preprocessor()
    model = MultiOutputRegressor(clone(base_model))
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])

    if target_mode == "direct_raw":
        pipe.fit(X_train, y_train_raw)
        y_pred = pipe.predict(X_test)
        return y_pred

    if target_mode == "direct_forced_closed":
        pipe.fit(X_train, y_train_raw)
        y_pred = pipe.predict(X_test)
        return close_to_100(y_pred)

    if target_mode == "ILR_closed":
        y_train_ilr = to_ilr(y_train_raw)
        pipe.fit(X_train, y_train_ilr)
        z_pred = pipe.predict(X_test)
        return from_ilr(z_pred)

    raise ValueError(f"Unknown target mode: {target_mode}")

# =============================================================================
# DATA PREPARATION
# =============================================================================

def clean_string_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})


def ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize supported descriptor column names for the validation-ladder feature set.
    """
    df = df.copy()

    # Numeric diagnostic proxies
    if "proximate_sum_VM_FC_Ash" not in df.columns:
        if "proximate_sum_raw_wt_pct" in df.columns:
            df["proximate_sum_VM_FC_Ash"] = pd.to_numeric(df["proximate_sum_raw_wt_pct"], errors="coerce")
        else:
            df["proximate_sum_VM_FC_Ash"] = (
                pd.to_numeric(df.get("volatile_matter_wt_pct"), errors="coerce")
                + pd.to_numeric(df.get("fixed_carbon_wt_pct"), errors="coerce")
                + pd.to_numeric(df.get("ash_wt_pct"), errors="coerce")
            )

    if "ultimate_CHNO_sum" not in df.columns:
        if "ultimate_CHNO_sum_raw_wt_pct" in df.columns:
            df["ultimate_CHNO_sum"] = pd.to_numeric(df["ultimate_CHNO_sum_raw_wt_pct"], errors="coerce")
        else:
            df["ultimate_CHNO_sum"] = (
                pd.to_numeric(df.get("carbon_wt_pct"), errors="coerce")
                + pd.to_numeric(df.get("hydrogen_wt_pct"), errors="coerce")
                + pd.to_numeric(df.get("nitrogen_wt_pct"), errors="coerce")
                + pd.to_numeric(df.get("oxygen_wt_pct"), errors="coerce")
            )

    # Descriptor name fallbacks
    fallback_map = {
        "yield_basis_std": "yield_basis",
        "oil_definition_v3": "oil_definition",
        "gas_yield_method_v3": "gas_yield_method",
        "pyrolysis_type_v3": "pyrolysis_type",
        "proximate_basis_std": "proximate_basis",
        "ultimate_basis_std": "ultimate_basis",
    }
    for new_col, old_col in fallback_map.items():
        if new_col not in df.columns and old_col in df.columns:
            df[new_col] = df[old_col]

    if "composition_basis_pair" not in df.columns:
        if "proximate_ultimate_basis_pair" in df.columns:
            df["composition_basis_pair"] = df["proximate_ultimate_basis_pair"]
        else:
            df["composition_basis_pair"] = (
                clean_string_series(df.get("proximate_basis_std", pd.Series(["NS"] * len(df), index=df.index)))
                + "__"
                + clean_string_series(df.get("ultimate_basis_std", pd.Series(["NS"] * len(df), index=df.index)))
            )

    if "yield_reporting_profile" not in df.columns:
        df["yield_reporting_profile"] = (
            clean_string_series(df.get("yield_basis_std",       pd.Series(["NS"] * len(df), index=df.index)))
            + "__"
            + clean_string_series(df.get("oil_definition_v3",   pd.Series(["NS"] * len(df), index=df.index)))
            + "__"
            + clean_string_series(df.get("gas_yield_method_v3", pd.Series(["NS"] * len(df), index=df.index)))
        )

    return df


def add_group_id_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """Create the canonical target-free group key used for split selection."""
    df = df.copy()
    if SIM_GROUP_COL_PREFERRED in df.columns:
        df["validation_group_id"] = df[SIM_GROUP_COL_PREFERRED].astype(str)
        return df

    provided_group_id = (
        df["validation_group_id"].astype(str)
        if "validation_group_id" in df.columns
        else None
    )

    num_cols = [
        "volatile_matter_wt_pct", "fixed_carbon_wt_pct", "ash_wt_pct",
        "carbon_wt_pct", "hydrogen_wt_pct", "nitrogen_wt_pct", "oxygen_wt_pct",
    ]
    for col in ["study_id", "feedstock_name"] + num_cols:
        if col not in df.columns:
            df[col] = "MISSING"

    parts = df["study_id"].astype(str) + " | " + df["feedstock_name"].astype(str)
    for col in num_cols:
        formatted = pd.to_numeric(df[col], errors="coerce").map(
            lambda v: f"{v:.3f}" if pd.notna(v) else "nan"
        )
        parts = parts + " | " + formatted

    if provided_group_id is not None:
        mapping = pd.DataFrame(
            {"provided": provided_group_id, "canonical": parts}
        )
        if (
            mapping.groupby("provided")["canonical"].nunique().max() != 1
            or mapping.groupby("canonical")["provided"].nunique().max() != 1
        ):
            raise ValueError(
                "Provided validation_group_id is not one-to-one with the "
                "canonical study-feedstock-composition groups."
            )
        df["provided_validation_group_id"] = provided_group_id

    df["validation_group_id"] = parts
    return df


def load_model_matrix() -> pd.DataFrame:
    if not INPUT_WORKBOOK.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_WORKBOOK}. Place this script in the same folder as the workbook."
        )

    df = pd.read_excel(INPUT_WORKBOOK, sheet_name=INPUT_SHEET)
    rows_loaded = len(df)

    if rows_loaded != EXPECTED_ROWS_LOADED:
        raise ValueError(
            f"Expected {EXPECTED_ROWS_LOADED} rows from {INPUT_SHEET}, but loaded {rows_loaded}. "
            "Check that the intended dataset sheet is present and selected."
        )

    df = ensure_feature_columns(df)

    # Convert numeric feature/target columns robustly.
    for c in NUMERIC_FEATURES + TARGET_COLS + ["yield_sum_wt_pct", "sample_mass_g", "particle_size_mm", "holding_time_min"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ensure categorical columns are strings.
    for c in CATEGORICAL_FEATURES:
        if c not in df.columns:
            raise KeyError(f"Required categorical feature missing after compatibility mapping: {c}")
        df[c] = df[c].astype(str)

    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + TARGET_COLS + ["study_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    before = len(df)
    df = df.dropna(subset=required).copy()
    after = len(df)

    if after != EXPECTED_ROWS_LOADED:
        print(
            f"WARNING: rows after final feature/target check = {after}, expected {EXPECTED_ROWS_LOADED}. "
            "Continue only if the difference is intentional."
        )

    n_studies = df["study_id"].nunique()
    if n_studies != EXPECTED_STUDIES:
        print(
            f"WARNING: study count = {n_studies}, expected {EXPECTED_STUDIES}. "
            "Continue only if your model-ready sheet changed this."
        )

    df = add_group_id_if_needed(df)
    df["row_uid"] = np.arange(len(df))

    print("\n==============================")
    print("VALIDATION LADDER")
    print("==============================")
    print(f"Input:  {INPUT_WORKBOOK.resolve()}")
    print(f"Sheet:  {INPUT_SHEET}")
    print(f"Rows loaded: {before}")
    print(f"Rows after final feature/target check: {after}")
    print(f"Studies: {df['study_id'].nunique()}")
    print(f"Validation groups: {df['validation_group_id'].nunique()}")
    print(f"Models: {list(MODEL_SETTINGS.keys())}")
    print(f"Target modes: {TARGET_MODES}")

    return df

# =============================================================================
# VALIDATION STRATEGIES
# =============================================================================

def run_random_row(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLS].to_numpy(dtype=float)

    total = len(RANDOM_SEEDS) * len(MODEL_SETTINGS) * len(TARGET_MODES)
    done  = 0
    print(f"\nRunning S0 random-row repeated validation ({total} fits)...")
    for seed in RANDOM_SEEDS:
        stratify_labels = None
        if "feedstock_class" in df.columns and df["feedstock_class"].notna().all():
            counts = df["feedstock_class"].value_counts()
            if counts.min() >= 2:
                stratify_labels = df["feedstock_class"]

        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=stratify_labels,
        )
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for model_name, base_model in MODEL_SETTINGS.items():
            for mode in TARGET_MODES:
                y_pred = fit_predict(base_model, mode, X_train, y_train, X_test)
                rows.append(metric_block(
                    y_true_raw=y_test,
                    y_pred=y_pred,
                    validation="S0_random_row_repeated",
                    split_id=f"RANDOM_seed_{seed:03d}",
                    model_name=model_name,
                    target_mode=mode,
                    n_train=len(train_idx),
                    n_test=len(test_idx),
                    extra={"seed": seed},
                ))
                done += 1
            print(f"  S0 | seed {seed:02d} | {model_name:<25s} | {done}/{total}", end="\r")
    print()
    return pd.DataFrame(rows)


def prepare_similarity_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Create target-free features for split balancing only."""
    split_num = NUMERIC_FEATURES.copy()
    split_cat = CATEGORICAL_FEATURES + ["feedstock_class"] if "feedstock_class" in df.columns else CATEGORICAL_FEATURES.copy()

    X_split = df[split_num + split_cat].copy()
    for c in split_num:
        X_split[c] = pd.to_numeric(X_split[c], errors="coerce")
    for c in split_cat:
        X_split[c] = X_split[c].astype(str)

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), split_num),
        ("cat", ohe, split_cat),
    ])
    arr = pre.fit_transform(X_split)
    names = list(pre.get_feature_names_out())
    return pd.DataFrame(arr, columns=names, index=df.index), names


def score_group_split(
    df: pd.DataFrame,
    Xbal: pd.DataFrame,
    train_groups: set,
    test_groups: set,
) -> dict:
    train_mask = df["validation_group_id"].isin(train_groups).to_numpy()
    test_mask = df["validation_group_id"].isin(test_groups).to_numpy()

    train_rows = int(train_mask.sum())
    test_rows = int(test_mask.sum())
    total_rows = len(df)
    test_frac = test_rows / total_rows

    train_mean = Xbal.loc[train_mask].mean(axis=0)
    test_mean = Xbal.loc[test_mask].mean(axis=0)
    mean_diff = float(np.nanmean(np.abs(train_mean - test_mean)))

    train_sd = Xbal.loc[train_mask].std(axis=0).replace(0, np.nan)
    test_sd = Xbal.loc[test_mask].std(axis=0).replace(0, np.nan)
    sd_diff = float(np.nanmean(np.abs(train_sd - test_sd)))

    size_penalty = float(abs(test_frac - TEST_SIZE) * 5.0)

    n_test_studies = df.loc[test_mask, "study_id"].nunique()
    few_study_penalty = 0.0 if n_test_studies >= 5 else (5 - n_test_studies) * 0.2

    split_score = mean_diff + 0.5 * sd_diff + size_penalty + few_study_penalty

    return {
        "split_score": split_score,
        "mean_diff_score": mean_diff,
        "sd_diff_score": sd_diff,
        "size_penalty": size_penalty,
        "few_study_penalty": few_study_penalty,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "test_fraction_rows": test_frac,
        "train_groups": len(train_groups),
        "test_groups": len(test_groups),
        "train_studies": df.loc[train_mask, "study_id"].nunique(),
        "test_studies": n_test_studies,
    }


def select_similarity_aware_group_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Search random group splits and keep the best target-free balanced splits.

    Group strings are stored during the first RNG pass to avoid fragile two-pass
    reconstruction of the same pseudo-random sequence.
    """
    rng = np.random.default_rng(SIM_RANDOM_STATE)
    groups = np.array(sorted(df["validation_group_id"].unique()))
    Xbal, _ = prepare_similarity_features(df)

    candidate_records = []
    n_groups = len(groups)
    target_test_groups = max(1, int(round(TEST_SIZE * n_groups)))

    for i in range(SIM_N_CANDIDATE_SPLITS):
        shuffled = groups.copy()
        rng.shuffle(shuffled)

        test_groups_list = list(shuffled[:target_test_groups])
        train_groups_list = list(shuffled[target_test_groups:])

        test_groups = set(test_groups_list)
        train_groups = set(train_groups_list)

        score = score_group_split(df, Xbal, train_groups, test_groups)
        score.update({
            "candidate_id": f"SIM_CAND_{i:04d}",
            "selected_split_id": None,
            "test_groups_string": " ;; ".join(map(str, test_groups_list)),
            "train_groups_string": " ;; ".join(map(str, train_groups_list)),
        })
        candidate_records.append(score)

    audit = (
        pd.DataFrame(candidate_records)
        .sort_values("split_score", ascending=True)
        .reset_index(drop=True)
    )
    audit["selected_rank"] = np.arange(1, len(audit) + 1)

    selected = audit.head(SIM_KEEP_TOP_SPLITS).copy()
    selected["selected_split_id"] = [f"SIM_R{r:03d}" for r in range(1, len(selected) + 1)]

    return selected

def run_similarity_aware_grouped(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("\nSelecting S1 grouped validation splits...")
    split_audit = select_similarity_aware_group_splits(df)
    print(f"Selected {len(split_audit)} target-free balanced group splits.")

    rows = []
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLS].to_numpy(dtype=float)

    total = len(split_audit) * len(MODEL_SETTINGS) * len(TARGET_MODES)
    done  = 0
    print(f"Running S1 grouped validation ({total} fits)...")
    for _, s in split_audit.iterrows():
        split_id = s["selected_split_id"]
        test_groups = set(str(s["test_groups_string"]).split(" ;; "))
        test_mask = df["validation_group_id"].astype(str).isin(test_groups).to_numpy()
        train_mask = ~test_mask
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for model_name, base_model in MODEL_SETTINGS.items():
            for mode in TARGET_MODES:
                y_pred = fit_predict(base_model, mode, X_train, y_train, X_test)
                rows.append(metric_block(
                    y_true_raw=y_test,
                    y_pred=y_pred,
                    validation="S1_grouped_validation",
                    split_id=split_id,
                    model_name=model_name,
                    target_mode=mode,
                    n_train=len(train_idx),
                    n_test=len(test_idx),
                    extra={
                        "split_score": s["split_score"],
                        "train_groups": s["train_groups"],
                        "test_groups": s["test_groups"],
                        "train_studies": s["train_studies"],
                        "test_studies": s["test_studies"],
                    },
                ))
                done += 1
            print(f"  S1 | {split_id} | {model_name:<25s} | {done}/{total}", end="\r")
    print()
    return pd.DataFrame(rows), split_audit



def build_lopo_study_balanced_summary(detail_rows_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a study-balanced LOSO sensitivity summary from per-study LOSO rows.

    Pooled LOSO weights rows equally, so large studies contribute more. This
    sheet gives each held-out study equal weight by summarising fold-level
    metrics across studies. R² means are calculated after excluding folds flagged
    as unstable because they contain too few rows; medians are also reported.
    """
    if detail_rows_df.empty:
        return pd.DataFrame()

    metrics = [
        "ranking_R2",
        "orig_multi_R2", "orig_char_R2", "orig_oil_R2", "orig_gas_R2", "orig_RMSE", "orig_MAE",
        "closed_true_multi_R2", "closed_true_char_R2", "closed_true_oil_R2", "closed_true_gas_R2",
        "closed_true_RMSE", "closed_true_MAE",
        "pred_sum_mean", "pred_sum_sd", "pct_pred_sum_outside_95_105",
        "n_test",
    ]

    rows = []
    group_cols = ["model", "target_mode", "target_label"]
    for keys, sub in detail_rows_df.groupby(group_cols):
        model, target_mode, target_label = keys
        stable = sub.copy()
        if "fold_metric_note" in stable.columns:
            stable_for_r2 = stable[stable["fold_metric_note"].eq("ok")].copy()
        else:
            stable_for_r2 = stable

        row = {
            "validation": "S2_LOSO_by_study_id_study_balanced",
            "split_id": "LOSO_equal_weight_per_study",
            "model": model,
            "target_mode": target_mode,
            "target_label": target_label,
            "n_lopo_folds_total": sub["heldout_study_id"].nunique() if "heldout_study_id" in sub.columns else np.nan,
            "n_lopo_folds_used_for_r2": stable_for_r2["heldout_study_id"].nunique() if "heldout_study_id" in stable_for_r2.columns else np.nan,
        }

        for metric in metrics:
            metric_source = stable_for_r2 if "R2" in metric else sub
            row[f"{metric}_study_balanced_mean"] = float(metric_source[metric].mean()) if metric in metric_source.columns else np.nan
            row[f"{metric}_study_balanced_median"] = float(metric_source[metric].median()) if metric in metric_source.columns else np.nan
            row[f"{metric}_study_balanced_sd"] = float(metric_source[metric].std()) if metric in metric_source.columns else np.nan

        row["ranking_metric_study_balanced"] = row["ranking_R2_study_balanced_median"]
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values("ranking_metric_study_balanced", ascending=False).reset_index(drop=True)

def run_lopo_by_study(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    pooled_rows = []

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLS].to_numpy(dtype=float)
    studies = sorted(df["study_id"].astype(str).unique())

    total = len(studies) * len(MODEL_SETTINGS) * len(TARGET_MODES)
    done  = 0
    print(f"\nRunning S2 LOSO-by-study validation ({len(studies)} studies, {total} fits)...")

    pooled = {(m, mode): {"y_true": [], "y_pred": []} for m in MODEL_SETTINGS for mode in TARGET_MODES}

    for study in studies:
        test_mask = df["study_id"].astype(str).eq(study).to_numpy()
        train_mask = ~test_mask
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(test_idx) == 0 or len(train_idx) == 0:
            continue

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for model_name, base_model in MODEL_SETTINGS.items():
            for mode in TARGET_MODES:
                y_pred = fit_predict(base_model, mode, X_train, y_train, X_test)
                pooled[(model_name, mode)]["y_true"].append(y_test)
                pooled[(model_name, mode)]["y_pred"].append(y_pred)

                extra = {
                    "heldout_study_id": study,
                    "heldout_rows": len(test_idx),
                    "fold_metric_note": "unstable_if_few_rows" if len(test_idx) < MIN_TEST_ROWS_PER_STUDY_FOR_FOLD_METRICS else "ok",
                }
                detail_rows.append(metric_block(
                    y_true_raw=y_test,
                    y_pred=y_pred,
                    validation="S2_LOSO_by_study_id_fold",
                    split_id=f"LOSO_{study}",
                    model_name=model_name,
                    target_mode=mode,
                    n_train=len(train_idx),
                    n_test=len(test_idx),
                    extra=extra,
                ))
                done += 1
            print(f"  S2 | study {study:<12s} | {model_name:<25s} | {done}/{total}", end="\r")
    print()

    for (model_name, mode), pack in pooled.items():
        if not pack["y_true"]:
            continue
        y_true_all = np.vstack(pack["y_true"])
        y_pred_all = np.vstack(pack["y_pred"])
        pooled_rows.append(metric_block(
            y_true_raw=y_true_all,
            y_pred=y_pred_all,
            validation="S2_LOSO_by_study_id_pooled",
            split_id="LOSO_pooled_all_studies",
            model_name=model_name,
            target_mode=mode,
            n_train=np.nan,
            n_test=len(y_true_all),
            extra={"n_lopo_folds": len(studies)},
        ))

    return pd.DataFrame(detail_rows), pd.DataFrame(pooled_rows)

# =============================================================================
# SUMMARY
# =============================================================================

def summarise_repeated_results(detail: pd.DataFrame, validation_name: str) -> pd.DataFrame:
    metrics = [
        "ranking_R2",
        "orig_multi_R2", "orig_char_R2", "orig_oil_R2", "orig_gas_R2", "orig_RMSE", "orig_MAE",
        "closed_true_multi_R2", "closed_true_char_R2", "closed_true_oil_R2", "closed_true_gas_R2",
        "closed_true_RMSE", "closed_true_MAE",
        "pred_sum_mean", "pred_sum_sd", "pct_pred_sum_outside_95_105",
        "n_test",
    ]
    sub = detail[detail["validation"] == validation_name].copy()
    if sub.empty:
        return pd.DataFrame()
    grouped = sub.groupby(["validation", "model", "target_mode", "target_label"])[metrics]
    mean = grouped.mean().reset_index()
    sd = grouped.std().reset_index()
    mean.columns = [c if c in ["validation", "model", "target_mode", "target_label"] else f"{c}_mean" for c in mean.columns]
    sd.columns = [c if c in ["validation", "model", "target_mode", "target_label"] else f"{c}_sd" for c in sd.columns]
    return mean.merge(sd, on=["validation", "model", "target_mode", "target_label"], how="left")


def build_best_table(summary: pd.DataFrame, lopo_pooled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not summary.empty:
        for _, r in summary.iterrows():
            rows.append({**r.to_dict(), "ranking_metric": r["ranking_R2_mean"]})
    if not lopo_pooled.empty:
        for _, r in lopo_pooled.iterrows():
            d = {
                "validation": r["validation"],
                "model": r["model"],
                "target_mode": r["target_mode"],
                "target_label": r.get("target_label", TARGET_LABELS.get(r["target_mode"], r["target_mode"])),
                "ranking_metric": r["ranking_R2"],
            }
            for c in [
                "orig_multi_R2", "orig_char_R2", "orig_oil_R2", "orig_gas_R2", "orig_RMSE", "orig_MAE",
                "closed_true_multi_R2", "closed_true_RMSE", "closed_true_MAE",
                "pred_sum_mean", "pred_sum_sd", "pct_pred_sum_outside_95_105", "n_test"
            ]:
                d[f"{c}_mean"] = r.get(c, np.nan)
            rows.append(d)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["validation", "ranking_metric"], ascending=[True, False]).reset_index(drop=True)
    return out



def _q(series: pd.Series, q: float) -> float:
    """Safe quantile helper."""
    series = pd.to_numeric(series, errors="coerce").dropna()
    return float(series.quantile(q)) if len(series) else np.nan


def _mean(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    return float(series.mean()) if len(series) else np.nan


def _median(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    return float(series.median()) if len(series) else np.nan


def summarise_loso_fold_distribution(s2_detail: pd.DataFrame) -> pd.DataFrame:
    """Summarize LOSO fold medians, quantiles, and negative-R² frequencies."""
    if s2_detail.empty:
        return pd.DataFrame()

    r2_metrics = [
        "ranking_R2",
        "orig_multi_R2", "orig_char_R2", "orig_oil_R2", "orig_gas_R2",
        "closed_true_multi_R2", "closed_true_char_R2", "closed_true_oil_R2", "closed_true_gas_R2",
    ]
    err_metrics = [
        "orig_RMSE", "orig_MAE", "closed_true_RMSE", "closed_true_MAE",
        "pred_sum_mean", "pred_sum_sd", "pct_pred_sum_outside_95_105",
    ]

    rows = []
    for (model, target_mode, target_label), sub in s2_detail.groupby(["model", "target_mode", "target_label"]):
        stable = sub[sub.get("fold_metric_note", "ok").eq("ok")].copy() if "fold_metric_note" in sub.columns else sub.copy()
        row = {
            "validation": "S2_LOSO_by_study_id_fold_distribution",
            "model": model,
            "target_mode": target_mode,
            "target_label": target_label,
            "n_loso_folds_total": sub["heldout_study_id"].nunique() if "heldout_study_id" in sub.columns else len(sub),
            "n_loso_folds_used_for_r2": stable["heldout_study_id"].nunique() if "heldout_study_id" in stable.columns else len(stable),
            "heldout_rows_min": int(sub["n_test"].min()) if "n_test" in sub.columns else np.nan,
            "heldout_rows_median": float(sub["n_test"].median()) if "n_test" in sub.columns else np.nan,
            "heldout_rows_max": int(sub["n_test"].max()) if "n_test" in sub.columns else np.nan,
        }

        for metric in r2_metrics:
            if metric in stable.columns:
                vals = pd.to_numeric(stable[metric], errors="coerce").dropna()
                row[f"{metric}_fold_mean"] = float(vals.mean()) if len(vals) else np.nan
                row[f"{metric}_fold_median"] = float(vals.median()) if len(vals) else np.nan
                row[f"{metric}_fold_q10"] = float(vals.quantile(0.10)) if len(vals) else np.nan
                row[f"{metric}_fold_q90"] = float(vals.quantile(0.90)) if len(vals) else np.nan
                row[f"{metric}_negative_folds_n"] = int((vals < 0).sum()) if len(vals) else np.nan
                row[f"{metric}_negative_folds_pct"] = float((vals < 0).mean() * 100.0) if len(vals) else np.nan

        for metric in err_metrics:
            if metric in sub.columns:
                vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
                row[f"{metric}_fold_mean"] = float(vals.mean()) if len(vals) else np.nan
                row[f"{metric}_fold_median"] = float(vals.median()) if len(vals) else np.nan
                row[f"{metric}_fold_q10"] = float(vals.quantile(0.10)) if len(vals) else np.nan
                row[f"{metric}_fold_q90"] = float(vals.quantile(0.90)) if len(vals) else np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    if "ranking_R2_fold_median" in out.columns:
        out = out.sort_values("ranking_R2_fold_median", ascending=False).reset_index(drop=True)
    return out


def build_loso_compact_table(s2_pooled: pd.DataFrame, loso_dist: pd.DataFrame) -> pd.DataFrame:
    """Combine pooled LOSO performance with study-level fold distributions."""
    if s2_pooled.empty:
        return pd.DataFrame()

    pooled = s2_pooled.copy()
    pooled_cols = [
        "model", "target_mode", "target_label",
        "ranking_R2", "closed_true_multi_R2", "closed_true_RMSE", "closed_true_MAE",
        "orig_multi_R2", "orig_RMSE", "orig_MAE",
        "pred_sum_mean", "pred_sum_sd", "pct_pred_sum_outside_95_105", "n_test", "n_loso_folds",
    ]
    pooled = pooled[[c for c in pooled_cols if c in pooled.columns]].copy()
    rename_map = {
        "ranking_R2": "pooled_closed_R2",
        "closed_true_multi_R2": "pooled_closed_true_multi_R2",
        "closed_true_RMSE": "pooled_closed_RMSE",
        "closed_true_MAE": "pooled_closed_MAE",
        "orig_multi_R2": "pooled_orig_multi_R2",
        "orig_RMSE": "pooled_orig_RMSE",
        "orig_MAE": "pooled_orig_MAE",
        "n_test": "pooled_test_rows",
    }
    pooled = pooled.rename(columns=rename_map)

    dist_cols = [
        "model", "target_mode", "target_label",
        "n_loso_folds_total", "n_loso_folds_used_for_r2",
        "heldout_rows_min", "heldout_rows_median", "heldout_rows_max",
        "ranking_R2_fold_median", "ranking_R2_fold_q10", "ranking_R2_fold_q90",
        "ranking_R2_negative_folds_n", "ranking_R2_negative_folds_pct",
        "closed_true_RMSE_fold_median", "closed_true_MAE_fold_median",
    ]
    dist = loso_dist[[c for c in dist_cols if c in loso_dist.columns]].copy() if not loso_dist.empty else pd.DataFrame()
    out = pooled.merge(dist, on=["model", "target_mode", "target_label"], how="left") if not dist.empty else pooled

    # Identify selected model and relevant comparators.
    out["comparison_role"] = "Comparator"
    out.loc[(out["model"].eq("XGBoost_base")) & (out["target_mode"].eq("direct_forced_closed")), "comparison_role"] = "Selected XGBoost Renormalized pathway"
    if "target_mode" in out.columns and "pooled_closed_R2" in out.columns:
        constrained = out[out["target_mode"].isin(["direct_forced_closed", "ILR_closed"])].copy()
        if not constrained.empty:
            best_idx = constrained["pooled_closed_R2"].idxmax()
            out.loc[best_idx, "comparison_role"] = "Best constrained LOSO comparator"

    return out.sort_values("pooled_closed_R2", ascending=False).reset_index(drop=True)


def build_selected_loso_fold_table(s2_detail: pd.DataFrame) -> pd.DataFrame:
    """Per-held-out-study rows for the selected XGBoost Renormalized pathway."""
    if s2_detail.empty:
        return pd.DataFrame()
    sub = s2_detail[
        s2_detail["model"].eq("XGBoost_base")
        & s2_detail["target_mode"].eq("direct_forced_closed")
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    cols = [
        "heldout_study_id", "n_test", "fold_metric_note",
        "ranking_R2", "closed_true_multi_R2", "closed_true_char_R2", "closed_true_oil_R2", "closed_true_gas_R2",
        "closed_true_RMSE", "closed_true_MAE",
        "orig_multi_R2", "orig_RMSE", "orig_MAE",
        "pred_sum_mean", "pred_sum_sd", "pct_pred_sum_outside_95_105",
    ]
    cols = [c for c in cols if c in sub.columns]
    return sub[cols].sort_values(["fold_metric_note", "ranking_R2"], ascending=[True, False]).reset_index(drop=True)

def build_label_table() -> pd.DataFrame:
    return pd.DataFrame({
        "target_mode": list(TARGET_LABELS.keys()),
        "recommended_label": [TARGET_LABELS[k] for k in TARGET_LABELS],
        "definition": [
            "Direct prediction of reported yields without closure correction; ranked against closed-normalized true yields for cross-mode fairness.",
            "Direct prediction followed by post-hoc rescaling to a 100% yield sum; ranked against closed-normalized true yields.",
            "Prediction in ILR coordinates followed by back-transformation to a 100% yield sum; ranked against closed-normalized true yields.",
        ],
    })

# =============================================================================
# MAIN
# =============================================================================

def main():
    df = load_model_matrix()

    keep_cols = [c for c in CONTEXT_COLS if c in df.columns] + ["validation_group_id", "row_uid"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES + TARGET_COLS
    model_matrix = df[keep_cols].copy()

    s0_detail = run_random_row(df)
    s1_detail, s1_audit = run_similarity_aware_grouped(df)
    s2_detail, s2_pooled = run_lopo_by_study(df)
    s2_study_balanced = build_lopo_study_balanced_summary(s2_detail)
    s2_fold_distribution = summarise_loso_fold_distribution(s2_detail)
    s2_compact = build_loso_compact_table(s2_pooled, s2_fold_distribution)
    s2_selected_folds = build_selected_loso_fold_table(s2_detail)

    s0_summary = summarise_repeated_results(s0_detail, "S0_random_row_repeated")
    s1_summary = summarise_repeated_results(s1_detail, "S1_grouped_validation")
    all_summary = pd.concat([s0_summary, s1_summary], ignore_index=True)
    best_table = build_best_table(all_summary, s2_pooled)
    label_table = build_label_table()

    print("\n==============================")
    print("TOP VALIDATION SUMMARY")
    print("==============================")
    if not best_table.empty:
        cols_to_show = ["validation", "model", "target_label", "ranking_metric"]
        extra_cols = [c for c in ["orig_multi_R2_mean", "closed_true_multi_R2_mean", "orig_RMSE_mean", "closed_true_RMSE_mean", "pct_pred_sum_outside_95_105_mean"] if c in best_table.columns]
        print(best_table[cols_to_show + extra_cols].head(30).to_string(index=False))

    with pd.ExcelWriter(OUTPUT_WORKBOOK, engine="openpyxl") as writer:
        model_matrix.to_excel(writer, sheet_name="30_Model_Matrix_442", index=False)
        s0_detail.to_excel(writer, sheet_name="31_S0_Random_Detailed", index=False)
        s1_audit.to_excel(writer, sheet_name="32_S1_Group_SplitAudit", index=False)
        s1_detail.to_excel(writer, sheet_name="33_S1_Group_Detailed", index=False)
        s2_detail.to_excel(writer, sheet_name="34_S2_LOSO_DetailedByStudy", index=False)
        s2_pooled.to_excel(writer, sheet_name="35_S2_LOSO_PooledSummary", index=False)
        s2_study_balanced.to_excel(writer, sheet_name="36_S2_LOSO_StudyBalanced", index=False)
        all_summary.to_excel(writer, sheet_name="37_All_Validation_Summary", index=False)
        best_table.to_excel(writer, sheet_name="38_Best_By_Strategy_Mode", index=False)
        label_table.to_excel(writer, sheet_name="39_Model_Target_Labels", index=False)
        s2_fold_distribution.to_excel(writer, sheet_name="40_S2_LOSO_FoldDistribution", index=False)
        s2_compact.to_excel(writer, sheet_name="41_LOSO_Compact", index=False)
        s2_selected_folds.to_excel(writer, sheet_name="42_Selected_XGB_LOSO_Folds", index=False)

    print("\n==============================")
    print("STEP 3 COMPLETE")
    print("==============================")
    print(f"Saved to: {OUTPUT_WORKBOOK.resolve()}")


if __name__ == "__main__":
    main()
