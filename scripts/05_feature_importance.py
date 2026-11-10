"""Feature-attribution analysis reported in Figure 4.

Target-specific TreeSHAP and global permutation importance are calculated for
the severity-enhanced XGBoost workflow. Predictor-space PCA and collinearity
tables are retained as supporting, target-free diagnostics.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from project_config import DATA_FILE, DATA_SHEET, output_file

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

warnings.filterwarnings("ignore")


# Analysis settings
INPUT_WORKBOOK = DATA_FILE
INPUT_SHEET = DATA_SHEET
OUTPUT_WORKBOOK = output_file("05_feature_importance_results.xlsx")

EXPECTED_ROWS = 442
EXPECTED_STUDIES = 42

RANDOM_SEEDS = list(range(10))
TEST_SIZE = 0.20

RUN_SHAP = True
SHAP_MAX_TEST_ROWS = 200

RUN_PERMUTATION = True
N_PERMUTATION_REPEATS = 5
PERMUTATION_TARGET_MODES = ["direct_raw", "direct_forced_closed", "ILR_closed"]

RUN_PCA = True
PCA_N_COMPONENTS = 8

RUN_COLLINEARITY = True


# ============================================================
# 2. FEATURE DEFINITIONS — CORE + SEVERITY SET
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

TARGET_COLS = ["char_yield_wt_pct", "oil_yield_wt_pct", "gas_yield_wt_pct"]
TARGET_SHORT = ["char", "oil", "gas"]
TARGET_LABELS = {
    "char": "Char",
    "oil": "Oil/liquid",
    "gas": "Gas",
}

TARGET_MODE_LABELS = {
    "direct_raw": "Unconstrained",
    "direct_forced_closed": "Renormalized",
    "ILR_closed": "ILR-transformed",
}

FEATURE_GROUPS = {
    "temperature_c":                  "Process condition",
    "heating_rate_c_min":             "Process condition",
    "inv_temp_K":                     "Severity descriptor",
    "log_heating_rate":               "Severity descriptor",
    "temp_x_log_heating_rate":        "Severity descriptor",

    "volatile_matter_wt_pct":         "Proximate composition",
    "fixed_carbon_wt_pct":            "Proximate composition",
    "ash_wt_pct":                     "Proximate composition",

    "carbon_wt_pct":                  "Ultimate composition",
    "hydrogen_wt_pct":                "Ultimate composition",
    "nitrogen_wt_pct":                "Ultimate composition",
    "oxygen_wt_pct":                  "Ultimate composition",

    "proximate_sum_raw_wt_pct":             "Closure/basis diagnostic",
    "proximate_sum_deviation_from_100":     "Closure/basis diagnostic",
    "ultimate_CHNO_sum_raw_wt_pct":         "Closure/basis diagnostic",
    "ultimate_CHNO_sum_deviation_from_100": "Closure/basis diagnostic",

    "yield_basis":            "Yield/reporting descriptor",
    "oil_definition":         "Yield/reporting descriptor",
    "gas_yield_method":       "Yield/reporting descriptor",
    "yield_reporting_profile":"Yield/reporting descriptor",

    "pyrolysis_type":                    "Process descriptor",
    "proximate_basis":                   "Composition-basis descriptor",
    "ultimate_basis":                    "Composition-basis descriptor",
    "proximate_ultimate_basis_pair":     "Composition-basis descriptor",
}


# ============================================================
# 3. HELPERS
# ============================================================

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def clean_string_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})


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

    valid = []
    for j in range(y_true.shape[1]):
        if np.nanstd(y_true[:, j]) < 1e-10:
            continue
        try:
            valid.append(float(r2_score(y_true[:, j], y_pred[:, j])))
        except Exception:
            continue

    if not valid:
        return np.nan
    if multioutput == "raw_values":
        return np.array(valid)
    return float(np.nanmean(valid))


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
    x_oil = np.exp(-math.sqrt(1.0 / 6.0) * z[:, 0] + math.sqrt(1.0 / 2.0) * z[:, 1])
    x_gas = np.exp(-math.sqrt(1.0 / 6.0) * z[:, 0] - math.sqrt(1.0 / 2.0) * z[:, 1])
    comp = np.column_stack([x_char, x_oil, x_gas])
    return comp / comp.sum(axis=1, keepdims=True) * 100.0


def get_xgb_model(random_state=42):
    if not HAS_XGB:
        raise ImportError("xgboost is not installed. Install xgboost or change the model.")
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


def build_tree_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
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


def build_pca_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
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


def transformed_feature_mapping(preprocessor, numeric_features, categorical_features):
    """
    Map transformed matrix columns back to original features.
    Handles Pipeline-wrapped OneHotEncoder.
    """
    transformed = []
    original = []

    # Numeric pipeline produces features in same order.
    for f in numeric_features:
        transformed.append(f"num__{f}")
        original.append(f)

    cat_pipe = preprocessor.named_transformers_["cat"]
    if isinstance(cat_pipe, Pipeline):
        cat_encoder = cat_pipe.named_steps["onehot"]
    else:
        cat_encoder = cat_pipe

    cat_names = list(cat_encoder.get_feature_names_out(categorical_features))

    for name in cat_names:
        transformed.append(f"cat__{name}")
        matched = None
        for c in sorted(categorical_features, key=len, reverse=True):
            if name == c or name.startswith(c + "_"):
                matched = c
                break
        original.append(matched if matched is not None else name)

    return transformed, original


def score_prediction(y_true_raw, y_pred, target_mode):
    y_true_raw = np.asarray(y_true_raw, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_eval = y_true_raw if target_mode == "direct_raw" else close_to_100(y_true_raw)
    pred_sum = y_pred.sum(axis=1)

    return {
        "multi_R2": safe_r2(y_eval, y_pred, "uniform_average"),
        "char_R2": safe_r2(y_eval[:, 0], y_pred[:, 0]),
        "oil_R2": safe_r2(y_eval[:, 1], y_pred[:, 1]),
        "gas_R2": safe_r2(y_eval[:, 2], y_pred[:, 2]),
        "RMSE": rmse(y_eval, y_pred),
        "MAE": float(mean_absolute_error(y_eval, y_pred)),
        "pred_sum_mean": float(np.mean(pred_sum)),
        "pred_sum_sd": float(np.std(pred_sum, ddof=1)) if len(pred_sum) > 1 else 0.0,
        "pct_pred_sum_outside_95_105": float(np.mean((pred_sum < 95) | (pred_sum > 105)) * 100.0),
    }


def fit_multioutput_pipeline(base_model, X_train, y_train_raw, target_mode):
    preprocessor = build_tree_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", MultiOutputRegressor(clone(base_model))),
    ])

    if target_mode == "direct_raw":
        pipe.fit(X_train, y_train_raw)
        return pipe

    if target_mode == "direct_forced_closed":
        pipe.fit(X_train, y_train_raw)
        return pipe

    if target_mode == "ILR_closed":
        pipe.fit(X_train, to_ilr(y_train_raw))
        return pipe

    raise ValueError(target_mode)


def predict_multioutput(pipe, X, target_mode):
    pred = pipe.predict(X)
    if target_mode == "direct_raw":
        return pred
    if target_mode == "direct_forced_closed":
        return close_to_100(pred)
    if target_mode == "ILR_closed":
        return from_ilr(pred)
    raise ValueError(target_mode)


# ============================================================
# 4. DATA PREPARATION
# ============================================================

def add_basis_and_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
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


def load_model_matrix() -> pd.DataFrame:
    if not INPUT_WORKBOOK.exists():
        raise FileNotFoundError(f"Cannot find {INPUT_WORKBOOK}.")

    df = pd.read_excel(INPUT_WORKBOOK, sheet_name=INPUT_SHEET)
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, loaded {len(df)} from {INPUT_SHEET}.")

    df = add_basis_and_engineered_features(df)

    required = FEATURES + TARGET_COLS + ["study_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after feature engineering: {missing}")

    before = len(df)
    df = df.dropna(subset=required).copy().reset_index(drop=True)
    after = len(df)

    if after != EXPECTED_ROWS:
        print(f"WARNING: row count after final feature check = {after}; expected {EXPECTED_ROWS}.")

    studies = df["study_id"].nunique()
    if studies != EXPECTED_STUDIES:
        print(f"WARNING: study count = {studies}; expected {EXPECTED_STUDIES}.")

    df["row_uid"] = np.arange(len(df))
    return df


# ============================================================
# 5. SHAP
# ============================================================

def run_shap_for_seed(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not HAS_SHAP:
        return pd.DataFrame(), pd.DataFrame()

    X = df[FEATURES].copy()
    ydf = df[TARGET_COLS].copy()
    idx = np.arange(len(df))

    stratify = None
    if "feedstock_class" in df.columns:
        fc = df["feedstock_class"].fillna("Unknown")
        counts = fc.value_counts()
        if counts.min() >= 2:
            stratify = fc

    train_idx, test_idx = train_test_split(
        idx,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=stratify.to_numpy() if stratify is not None else None,
    )

    X_train = X.iloc[train_idx]
    X_test  = X.iloc[test_idx]
    y_train_df = ydf.iloc[train_idx]
    y_test_df  = ydf.iloc[test_idx]

    preprocessor = build_tree_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    transformed_names, original_names = transformed_feature_mapping(
        preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES
    )
    mapping = pd.DataFrame({
        "transformed_feature": transformed_names,
        "feature": original_names,
    })

    X_shap = X_test_t[:min(SHAP_MAX_TEST_ROWS, X_test_t.shape[0])]

    shap_rows = []
    baseline_rows = []

    for target_col, target_short in zip(TARGET_COLS, TARGET_SHORT):
        model = get_xgb_model(random_state=seed)
        model.fit(X_train_t, y_train_df[target_col].to_numpy(dtype=float))

        pred = model.predict(X_test_t)
        baseline_rows.append({
            "model": "XGBoost_base",
            "feature_variant": "F2_core_severity",
            "importance_method": "TreeSHAP_target_specific_direct",
            "target": target_short,
            "target_label": TARGET_LABELS[target_short],
            "seed": seed,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "R2": safe_r2(y_test_df[target_col], pred),
            "RMSE": rmse(y_test_df[target_col], pred),
            "MAE": float(mean_absolute_error(y_test_df[target_col], pred)),
        })

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_values = np.asarray(shap_values)

            abs_mean = np.abs(shap_values).mean(axis=0)
            tmp = mapping.copy()
            tmp["mean_abs_SHAP_transformed"] = abs_mean[:len(tmp)]

            agg = (
                tmp.groupby("feature", as_index=False)["mean_abs_SHAP_transformed"]
                .sum()
                .rename(columns={"mean_abs_SHAP_transformed": "mean_abs_SHAP"})
            )
            total = agg["mean_abs_SHAP"].sum()
            agg["relative_SHAP_pct"] = np.where(total > 0, agg["mean_abs_SHAP"] / total * 100.0, np.nan)
            agg["feature_group"] = agg["feature"].map(FEATURE_GROUPS).fillna("other")
            agg["model"] = "XGBoost_base"
            agg["feature_variant"] = "F2_core_severity"
            agg["importance_method"] = "TreeSHAP_target_specific_direct"
            agg["target"] = target_short
            agg["target_label"] = TARGET_LABELS[target_short]
            agg["seed"] = seed
            agg["n_train"] = len(train_idx)
            agg["n_test"] = len(test_idx)
            shap_rows.append(agg)

        except Exception as e:
            shap_rows.append(pd.DataFrame([{
                "model": "XGBoost_base",
                "feature_variant": "F2_core_severity",
                "importance_method": "TreeSHAP_target_specific_direct",
                "target": target_short,
                "target_label": TARGET_LABELS[target_short],
                "seed": seed,
                "feature": "SHAP_FAILED",
                "feature_group": "error",
                "mean_abs_SHAP": np.nan,
                "relative_SHAP_pct": np.nan,
                "error": str(e),
            }]))

    shap_long = pd.concat(shap_rows, ignore_index=True) if shap_rows else pd.DataFrame()
    shap_baseline = pd.DataFrame(baseline_rows)
    return shap_long, shap_baseline


def summarise_shap(shap_long: pd.DataFrame) -> pd.DataFrame:
    if shap_long.empty or "mean_abs_SHAP" not in shap_long.columns:
        return pd.DataFrame()

    valid = shap_long[shap_long["feature"].ne("SHAP_FAILED")].copy()
    if valid.empty:
        return pd.DataFrame()

    group_cols = ["model", "feature_variant", "target", "target_label", "feature", "feature_group"]
    out = (
        valid.groupby(group_cols, as_index=False)
        .agg(
            mean_abs_SHAP_mean=("mean_abs_SHAP", "mean"),
            mean_abs_SHAP_sd=("mean_abs_SHAP", "std"),
            relative_SHAP_pct_mean=("relative_SHAP_pct", "mean"),
            relative_SHAP_pct_sd=("relative_SHAP_pct", "std"),
            n_splits=("seed", "nunique"),
        )
    )
    out["rank_within_target"] = (
        out.groupby(["model", "feature_variant", "target"])["mean_abs_SHAP_mean"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return out.sort_values(["target", "rank_within_target"]).reset_index(drop=True)


# ============================================================
# 6. PERMUTATION IMPORTANCE
# ============================================================

def run_permutation_for_seed_mode(df: pd.DataFrame, seed: int, target_mode: str) -> Tuple[pd.DataFrame, dict]:
    X = df[FEATURES].copy()
    y = df[TARGET_COLS].to_numpy(dtype=float)
    idx = np.arange(len(df))

    stratify = None
    if "feedstock_class" in df.columns:
        fc = df["feedstock_class"].fillna("Unknown")
        counts = fc.value_counts()
        if counts.min() >= 2:
            stratify = fc

    train_idx, test_idx = train_test_split(
        idx,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=stratify.to_numpy() if stratify is not None else None,
    )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = get_xgb_model(random_state=seed)
    pipe = fit_multioutput_pipeline(model, X_train, y_train, target_mode)
    baseline_pred = predict_multioutput(pipe, X_test, target_mode)
    baseline = score_prediction(y_test, baseline_pred, target_mode)

    rng = np.random.default_rng(10000 + seed)
    rows = []

    for feature in FEATURES:
        rep_scores = []
        for rep in range(N_PERMUTATION_REPEATS):
            Xp = X_test.copy()
            values = Xp[feature].to_numpy(copy=True)
            rng.shuffle(values)
            Xp[feature] = values
            pred = predict_multioutput(pipe, Xp, target_mode)
            rep_scores.append(score_prediction(y_test, pred, target_mode))

        rep_df = pd.DataFrame(rep_scores)

        rows.append({
            "model": "XGBoost_base",
            "feature_variant": "F2_core_severity",
            "target_mode": target_mode,
            "target_label": TARGET_MODE_LABELS[target_mode],
            "seed": seed,
            "feature": feature,
            "feature_group": FEATURE_GROUPS.get(feature, "other"),
            "baseline_multi_R2": baseline["multi_R2"],
            "importance_multi_R2_drop": baseline["multi_R2"] - rep_df["multi_R2"].mean(),
            "importance_char_R2_drop": baseline["char_R2"] - rep_df["char_R2"].mean(),
            "importance_oil_R2_drop": baseline["oil_R2"] - rep_df["oil_R2"].mean(),
            "importance_gas_R2_drop": baseline["gas_R2"] - rep_df["gas_R2"].mean(),
            "importance_RMSE_increase": rep_df["RMSE"].mean() - baseline["RMSE"],
            "importance_MAE_increase": rep_df["MAE"].mean() - baseline["MAE"],
            "baseline_RMSE": baseline["RMSE"],
            "baseline_MAE": baseline["MAE"],
            "baseline_pred_sum_mean": baseline["pred_sum_mean"],
            "baseline_pct_pred_sum_outside_95_105": baseline["pct_pred_sum_outside_95_105"],
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_permutation_repeats": N_PERMUTATION_REPEATS,
        })

    baseline_row = {
        "model": "XGBoost_base",
        "feature_variant": "F2_core_severity",
        "target_mode": target_mode,
        "target_label": TARGET_MODE_LABELS[target_mode],
        "seed": seed,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        **baseline,
    }
    return pd.DataFrame(rows), baseline_row


def summarise_permutation(perm_long: pd.DataFrame) -> pd.DataFrame:
    if perm_long.empty:
        return pd.DataFrame()

    group_cols = ["model", "feature_variant", "target_mode", "target_label", "feature", "feature_group"]
    out = (
        perm_long.groupby(group_cols, as_index=False)
        .agg(
            importance_multi_R2_drop_mean=("importance_multi_R2_drop", "mean"),
            importance_multi_R2_drop_sd=("importance_multi_R2_drop", "std"),
            importance_char_R2_drop_mean=("importance_char_R2_drop", "mean"),
            importance_oil_R2_drop_mean=("importance_oil_R2_drop", "mean"),
            importance_gas_R2_drop_mean=("importance_gas_R2_drop", "mean"),
            importance_RMSE_increase_mean=("importance_RMSE_increase", "mean"),
            importance_MAE_increase_mean=("importance_MAE_increase", "mean"),
            baseline_multi_R2_mean=("baseline_multi_R2", "mean"),
            n_splits=("seed", "nunique"),
        )
    )

    out["rank_within_mode"] = (
        out.groupby(["model", "feature_variant", "target_mode"])["importance_multi_R2_drop_mean"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return out.sort_values(["target_mode", "rank_within_mode"]).reset_index(drop=True)


# ============================================================
# 7. PCA AND COLLINEARITY
# ============================================================

def run_pca(df: pd.DataFrame):
    X = df[FEATURES].copy()
    pre = build_pca_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    X_t = pre.fit_transform(X)

    transformed_names, original_names = transformed_feature_mapping(pre, NUMERIC_FEATURES, CATEGORICAL_FEATURES)

    n_comp = min(PCA_N_COMPONENTS, X_t.shape[0] - 1, X_t.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    scores = pca.fit_transform(X_t)

    explained = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(n_comp)],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
    })

    loading_rows = []
    for i in range(n_comp):
        tmp = pd.DataFrame({
            "PC": f"PC{i+1}",
            "transformed_feature": transformed_names,
            "feature": original_names,
            "loading": pca.components_[i],
            "abs_loading": np.abs(pca.components_[i]),
            "explained_variance_ratio": pca.explained_variance_ratio_[i],
        })
        tmp["weighted_abs_loading"] = tmp["abs_loading"] * tmp["explained_variance_ratio"]
        loading_rows.append(tmp)

    load_long = pd.concat(loading_rows, ignore_index=True)

    load_by_feature = (
        load_long.groupby(["PC", "feature"], as_index=False)
        .agg(
            abs_loading_sum=("abs_loading", "sum"),
            weighted_abs_loading_sum=("weighted_abs_loading", "sum"),
            explained_variance_ratio=("explained_variance_ratio", "first"),
        )
    )
    load_by_feature["feature_group"] = load_by_feature["feature"].map(FEATURE_GROUPS).fillna("other")
    load_by_feature["rank_within_PC"] = (
        load_by_feature.groupby("PC")["abs_loading_sum"].rank(method="first", ascending=False).astype(int)
    )

    overall = (
        load_by_feature.groupby(["feature", "feature_group"], as_index=False)
        .agg(
            PCA_weighted_abs_loading_total=("weighted_abs_loading_sum", "sum"),
            PCA_abs_loading_max=("abs_loading_sum", "max"),
        )
        .sort_values("PCA_weighted_abs_loading_total", ascending=False)
        .reset_index(drop=True)
    )
    overall["rank_PCA_input_variance"] = np.arange(1, len(overall) + 1)

    return explained, load_by_feature, overall


def run_collinearity(df: pd.DataFrame):
    numeric_df = df[NUMERIC_FEATURES].copy()
    numeric_df = numeric_df.apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))

    spearman = numeric_df.corr(method="spearman")
    spearman_long = (
        spearman.where(np.triu(np.ones(spearman.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    spearman_long.columns = ["feature_1", "feature_2", "spearman_rho"]
    spearman_long["abs_spearman_rho"] = spearman_long["spearman_rho"].abs()
    spearman_long = spearman_long.sort_values("abs_spearman_rho", ascending=False).reset_index(drop=True)

    vif_rows = []
    X = numeric_df.to_numpy(dtype=float)
    X = StandardScaler().fit_transform(X)

    for i, feature in enumerate(NUMERIC_FEATURES):
        y = X[:, i]
        X_other = np.delete(X, i, axis=1)

        # Add intercept
        X_design = np.column_stack([np.ones(X_other.shape[0]), X_other])
        try:
            coef, *_ = np.linalg.lstsq(X_design, y, rcond=None)
            pred = X_design @ coef
            r2 = safe_r2(y, pred)
            if pd.isna(r2) or r2 >= 0.999999:
                vif = np.inf
            else:
                vif = 1.0 / (1.0 - r2)
        except Exception:
            r2 = np.nan
            vif = np.nan

        vif_rows.append({
            "feature": feature,
            "feature_group": FEATURE_GROUPS.get(feature, "other"),
            "R2_against_other_numeric_predictors": r2,
            "VIF": vif,
        })

    vif = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False).reset_index(drop=True)
    return spearman_long, vif


# ============================================================
# 8. COMPACT FEATURE SUMMARY
# ============================================================

def build_feature_importance_summary(shap_summary, perm_summary, pca_overall):
    rows = []

    # SHAP: top 8 per target
    if not shap_summary.empty:
        for target in TARGET_SHORT:
            sub = shap_summary[
                (shap_summary["target"].eq(target))
                & (shap_summary["rank_within_target"] <= 8)
            ].copy()
            for _, r in sub.iterrows():
                rows.append({
                    "source": "SHAP",
                    "target_or_mode": TARGET_LABELS[target],
                    "feature": r["feature"],
                    "feature_group": r["feature_group"],
                    "rank": r["rank_within_target"],
                    "importance_value": r["relative_SHAP_pct_mean"],
                    "importance_metric": "relative mean absolute SHAP (%)",
                    "target_specific_or_multioutput": "target-specific",
                    "interpretation_hint": "Model-attribution feature effect for target-specific direct model.",
                })

    # Permutation: top 10 for renormalized and ILR, plus unconstrained for comparison.
    if not perm_summary.empty:
        for mode in PERMUTATION_TARGET_MODES:
            sub = perm_summary[
                (perm_summary["target_mode"].eq(mode))
                & (perm_summary["rank_within_mode"] <= 10)
            ].copy()
            for _, r in sub.iterrows():
                rows.append({
                    "source": "Permutation",
                    "target_or_mode": TARGET_MODE_LABELS[mode],
                    "feature": r["feature"],
                    "feature_group": r["feature_group"],
                    "rank": r["rank_within_mode"],
                    "importance_value": r["importance_multi_R2_drop_mean"],
                    "importance_metric": "multi-output R² drop",
                    "target_specific_or_multioutput": "multi-output",
                    "interpretation_hint": "Model-agnostic dependence of the fitted prediction pipeline on this original feature.",
                })

    # PCA: top 10 target-free input-space descriptors.
    if not pca_overall.empty:
        sub = pca_overall[pca_overall["rank_PCA_input_variance"] <= 10].copy()
        for _, r in sub.iterrows():
            rows.append({
                "source": "PCA",
                "target_or_mode": "Target-free predictor-space structure",
                "feature": r["feature"],
                "feature_group": r["feature_group"],
                "rank": r["rank_PCA_input_variance"],
                "importance_value": r["PCA_weighted_abs_loading_total"],
                "importance_metric": "weighted absolute loading total",
                "target_specific_or_multioutput": "target-free diagnostic",
                "interpretation_hint": "Input-space structure only; not predictive importance.",
            })

    return pd.DataFrame(rows)


# ============================================================
# 9. MAIN
# ============================================================

def main():
    print("\n==============================")
    print("FEATURE IMPORTANCE: XGBOOST CORE+SEVERITY")
    print("==============================")
    print(f"Input: {INPUT_WORKBOOK.resolve()}")
    print(f"Sheet: {INPUT_SHEET}")

    if RUN_SHAP and not HAS_SHAP:
        print("WARNING: SHAP is not installed. SHAP outputs will be skipped.")
        print("Install with: python -m pip install shap")

    df = load_model_matrix()

    print(f"Rows after final check: {len(df)}")
    print(f"Studies: {df['study_id'].nunique() if 'study_id' in df.columns else 'NA'}")
    print(f"Numeric features: {len(NUMERIC_FEATURES)}")
    print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")
    print(f"Total original features: {len(FEATURES)}")

    audit = pd.DataFrame([{
        "input_workbook": str(INPUT_WORKBOOK),
        "input_sheet": INPUT_SHEET,
        "rows": len(df),
        "studies": df["study_id"].nunique() if "study_id" in df.columns else np.nan,
        "model": "XGBoost_base",
        "feature_variant": "F2_core_severity",
        "target_modes_for_permutation": ", ".join(PERMUTATION_TARGET_MODES),
        "random_seeds": ", ".join(map(str, RANDOM_SEEDS)),
        "test_size": TEST_SIZE,
        "n_numeric_features": len(NUMERIC_FEATURES),
        "n_categorical_features": len(CATEGORICAL_FEATURES),
        "numeric_features": "; ".join(NUMERIC_FEATURES),
        "categorical_features": "; ".join(CATEGORICAL_FEATURES),
        "SHAP_available": HAS_SHAP,
    }])

    shap_long_all = []
    shap_base_all = []
    if RUN_SHAP and HAS_SHAP:
        n_shap = len(RANDOM_SEEDS)
        print(f"\nRunning TreeSHAP target-specific importance ({n_shap} seeds × {len(TARGET_SHORT)} targets)...")
        for i, seed in enumerate(RANDOM_SEEDS, 1):
            print(f"  SHAP seed {seed} ({i}/{n_shap})...", end="\r")
            shap_long, shap_base = run_shap_for_seed(df, seed)
            if not shap_long.empty:
                shap_long_all.append(shap_long)
            if not shap_base.empty:
                shap_base_all.append(shap_base)
        print()

    shap_long = pd.concat(shap_long_all, ignore_index=True) if shap_long_all else pd.DataFrame()
    shap_baselines = pd.concat(shap_base_all, ignore_index=True) if shap_base_all else pd.DataFrame()
    shap_summary = summarise_shap(shap_long)

    perm_long_all = []
    perm_baseline_rows = []
    if RUN_PERMUTATION:
        n_perm = len(RANDOM_SEEDS) * len(PERMUTATION_TARGET_MODES)
        print(f"\nRunning manual permutation importance ({n_perm} seed × mode combinations)...")
        done = 0
        for seed in RANDOM_SEEDS:
            for mode in PERMUTATION_TARGET_MODES:
                done += 1
                print(f"  Permutation {done}/{n_perm} | seed {seed} | {TARGET_MODE_LABELS[mode]}", end="\r")
                try:
                    perm, base = run_permutation_for_seed_mode(df, seed, mode)
                    perm_long_all.append(perm)
                    perm_baseline_rows.append(base)
                except Exception as exc:
                    print(f"\n  WARNING: permutation failed for seed={seed}, mode={mode}: {exc}")
        print()

    perm_long = pd.concat(perm_long_all, ignore_index=True) if perm_long_all else pd.DataFrame()
    perm_baselines = pd.DataFrame(perm_baseline_rows)
    perm_summary = summarise_permutation(perm_long)

    if RUN_PCA:
        print("\nRunning PCA diagnostics...")
        pca_explained, pca_loadings, pca_overall = run_pca(df)
    else:
        pca_explained, pca_loadings, pca_overall = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if RUN_COLLINEARITY:
        print("\nRunning Spearman/VIF collinearity diagnostics...")
        spearman_long, vif = run_collinearity(df)
    else:
        spearman_long, vif = pd.DataFrame(), pd.DataFrame()

    feature_summary = build_feature_importance_summary(
        shap_summary, perm_summary, pca_overall
    )

    print("\nSaving workbook...")
    with pd.ExcelWriter(OUTPUT_WORKBOOK, engine="openpyxl") as writer:
        audit.to_excel(writer, sheet_name="60_Config_Audit", index=False)

        shap_long.to_excel(writer, sheet_name="61_SHAP_Long", index=False)
        shap_summary.to_excel(writer, sheet_name="62_SHAP_Summary", index=False)
        if not shap_summary.empty:
            shap_summary[shap_summary["rank_within_target"] <= 15].to_excel(
                writer, sheet_name="63_SHAP_Top15", index=False
            )
        else:
            pd.DataFrame().to_excel(writer, sheet_name="63_SHAP_Top15", index=False)
        shap_baselines.to_excel(writer, sheet_name="64_SHAP_Target_Baselines", index=False)

        perm_long.to_excel(writer, sheet_name="65_Permutation_Long", index=False)
        perm_summary.to_excel(writer, sheet_name="66_Permutation_Summary", index=False)
        if not perm_summary.empty:
            perm_summary[perm_summary["rank_within_mode"] <= 15].to_excel(
                writer, sheet_name="67_Permutation_Top15", index=False
            )
        else:
            pd.DataFrame().to_excel(writer, sheet_name="67_Permutation_Top15", index=False)
        perm_baselines.to_excel(writer, sheet_name="67b_Permutation_Baselines", index=False)

        pca_explained.to_excel(writer, sheet_name="68_PCA_Explained", index=False)
        pca_loadings.to_excel(writer, sheet_name="69_PCA_Loadings_ByFeature", index=False)
        pca_overall.to_excel(writer, sheet_name="70_PCA_Overall_Features", index=False)

        spearman_long.to_excel(writer, sheet_name="71_Spearman_Numeric", index=False)
        vif.to_excel(writer, sheet_name="72_VIF_Numeric", index=False)
        feature_summary.to_excel(
            writer, sheet_name="73_Feature_Summary", index=False
        )

    print("\n==============================")
    print("INTERPRETABILITY STEP COMPLETE")
    print("==============================")
    print(f"Saved to: {OUTPUT_WORKBOOK.resolve()}")

    if not shap_summary.empty:
        print("\nTop SHAP features by target:")
        for target in TARGET_SHORT:
            print(f"\n{TARGET_LABELS[target]}")
            cols = ["rank_within_target", "feature", "feature_group", "relative_SHAP_pct_mean"]
            sub = shap_summary[shap_summary["target"].eq(target)][cols].head(10)
            print(sub.to_string(index=False))

    if not perm_summary.empty:
        print("\nTop permutation features for Renormalized:")
        cols = [
            "rank_within_mode", "feature", "feature_group",
            "importance_multi_R2_drop_mean",
            "importance_char_R2_drop_mean",
            "importance_oil_R2_drop_mean",
            "importance_gas_R2_drop_mean",
        ]
        sub = perm_summary[perm_summary["target_mode"].eq("direct_forced_closed")][cols].head(12)
        print(sub.to_string(index=False))

    if not pca_explained.empty:
        print("\nPCA explained variance:")
        print(pca_explained.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
