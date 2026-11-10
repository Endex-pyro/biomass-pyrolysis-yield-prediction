"""Baseline model-family and target-handling screen reported in Table 1.

Models are evaluated with repeated, feedstock-class-stratified random-row
splits. All target modes are ranked against the same closed-normalized
reference yields.
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso, BayesianRidge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from project_config import DATA_FILE, DATA_SHEET, output_file

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

# Analysis settings
INPUT_XLSX = DATA_FILE
SHEET_NAME = DATA_SHEET
OUTPUT_XLSX = output_file("01_baseline_model_screen_results.xlsx")

TEST_SIZE        = 0.20
RANDOM_SEEDS = list(range(10))
PERMUTATION_REPEATS = 10

RUN_SVR         = True
RUN_GPR         = True
RUN_NEURAL_NETS = True

EXPECTED_ROWS_LOADED = 442
EXPECTED_STUDIES     = 42

# Display labels
TARGET_LABELS = {
    "direct_raw":           "Unconstrained",
    "direct_forced_closed": "Renormalized",
    "ILR_closed":           "ILR-transformed",
}

# =========================
# COLUMN SETUP
# =========================
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

TARGET_COLS = [
    "char_yield_wt_pct",
    "oil_yield_wt_pct",
    "gas_yield_wt_pct",
]

ID_COLS_CANDIDATES = [
    "study_id",
    "doi",
    "publication_year",
    "feedstock_name",
    "feedstock_class",
    "source_excel_row",
]

# Optional sensitivity columns are NOT used in the main model.
SENSITIVITY_COLS = [
    "holding_time_min",
    "sample_mass_g",
    "particle_size_mm",
]

# =========================
# HELPERS
# =========================
def make_ohe():
    """Compatible OneHotEncoder across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def clean_string_series(s):
    return s.astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})


def safe_r2(y_true, y_pred, multioutput="uniform_average"):
    """
    R² with protection against undefined/unstable constant-target folds.

    For 1D targets, return NaN if the target is effectively constant.
    For multi-output targets, compute per-output R² only for non-constant outputs
    and average the valid outputs. This avoids letting constant char/oil/gas folds
    enter cross-mode ranking as artificial 0.0 scores.
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


def add_basis_diagnostic_proxies(df):
    """Create target-free basis/composition diagnostic proxies."""
    df = df.copy()

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
        clean_string_series(df["yield_basis"]) + "__" +
        clean_string_series(df["oil_definition"]) + "__" +
        clean_string_series(df["gas_yield_method"])
    )

    return df


def ilr_transform(comp):
    """3-part ILR transform.  Accepts raw yields in any positive scale (percent or proportion);
    normalises internally so the result is independent of scale."""
    comp = np.asarray(comp, dtype=float)
    comp = np.maximum(comp, 1e-10)
    comp = comp / comp.sum(axis=1, keepdims=True)   # normalise to proportions

    char = comp[:, 0]
    oil  = comp[:, 1]
    gas  = comp[:, 2]

    z1 = np.sqrt(2 / 3) * np.log(char / np.sqrt(oil * gas))
    z2 = np.sqrt(1 / 2) * np.log(oil / gas)
    return np.column_stack([z1, z2])


def ilr_inverse(z):
    """Inverse 3-part ILR transform; returns proportions summing to 1."""
    z    = np.asarray(z, dtype=float)
    x_char = np.exp( np.sqrt(2 / 3) * z[:, 0])
    x_oil  = np.exp(-np.sqrt(1 / 6) * z[:, 0] + np.sqrt(1 / 2) * z[:, 1])
    x_gas  = np.exp(-np.sqrt(1 / 6) * z[:, 0] - np.sqrt(1 / 2) * z[:, 1])
    comp   = np.column_stack([x_char, x_oil, x_gas])
    return comp / comp.sum(axis=1, keepdims=True)


def force_close_100(y_pred):
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.maximum(y_pred, 1e-10)
    return y_pred / y_pred.sum(axis=1, keepdims=True) * 100.0


def build_preprocessor(numeric_features, categorical_features, scale_numeric=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipe    = Pipeline(numeric_steps)
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  make_ohe()),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe,    numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ], remainder="drop")


def metric_block(y_true_raw, y_pred, validation_label, model_name, target_mode, seed, n_train, n_test):
    y_true_raw    = np.asarray(y_true_raw, dtype=float)
    y_pred        = np.asarray(y_pred,     dtype=float)
    y_true_closed = force_close_100(y_true_raw)
    pred_sum      = y_pred.sum(axis=1)

    return {
        "validation":   validation_label,
        "model":        model_name,
        "target_mode":  target_mode,
        "target_label": TARGET_LABELS.get(target_mode, target_mode),
        "seed":         seed,
        "n_train":      n_train,
        "n_test":       n_test,

        # Metrics against original reported yields
        "orig_multi_R2": safe_r2(y_true_raw, y_pred, "uniform_average"),
        "orig_char_R2":  safe_r2(y_true_raw[:, 0], y_pred[:, 0]),
        "orig_oil_R2":   safe_r2(y_true_raw[:, 1], y_pred[:, 1]),
        "orig_gas_R2":   safe_r2(y_true_raw[:, 2], y_pred[:, 2]),
        "orig_RMSE":     float(np.sqrt(mean_squared_error(y_true_raw, y_pred))),
        "orig_MAE":      float(mean_absolute_error(y_true_raw, y_pred)),

        # Metrics against closed-normalised true yields (useful for constrained modes)
        "closed_true_multi_R2": safe_r2(y_true_closed, y_pred, "uniform_average"),
        "closed_true_char_R2":  safe_r2(y_true_closed[:, 0], y_pred[:, 0]),
        "closed_true_oil_R2":   safe_r2(y_true_closed[:, 1], y_pred[:, 1]),
        "closed_true_gas_R2":   safe_r2(y_true_closed[:, 2], y_pred[:, 2]),
        "closed_true_RMSE":     float(np.sqrt(mean_squared_error(y_true_closed, y_pred))),
        "closed_true_MAE":      float(mean_absolute_error(y_true_closed, y_pred)),

        # Common closed-reference ranking metric for all target modes
        "ranking_R2": safe_r2(y_true_closed, y_pred, "uniform_average"),

        # Closure diagnostics
        "pred_sum_mean":               float(np.mean(pred_sum)),
        "pred_sum_sd":                 float(np.std(pred_sum, ddof=1)) if len(pred_sum) > 1 else 0.0,
        "pct_pred_sum_outside_95_105": float(np.mean((pred_sum < 95) | (pred_sum > 105)) * 100.0),
    }


def fit_predict_for_mode(model, preprocessor, X_train, y_train_raw, X_test, target_mode):
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model",      MultiOutputRegressor(clone(model))),
    ])

    if target_mode == "direct_raw":
        pipe.fit(X_train, y_train_raw)
        return pipe, pipe.predict(X_test)

    if target_mode == "direct_forced_closed":
        pipe.fit(X_train, y_train_raw)
        return pipe, force_close_100(pipe.predict(X_test))

    if target_mode == "ILR_closed":
        y_train_ilr = ilr_transform(y_train_raw)
        pipe.fit(X_train, y_train_ilr)
        return pipe, ilr_inverse(pipe.predict(X_test)) * 100.0

    raise ValueError(f"Unknown target_mode: {target_mode}")


def predict_for_mode(pipe, X, target_mode):
    pred = pipe.predict(X)
    if target_mode == "direct_raw":
        return pred
    if target_mode == "direct_forced_closed":
        return force_close_100(pred)
    if target_mode == "ILR_closed":
        return ilr_inverse(pred) * 100.0
    raise ValueError(target_mode)


def permutation_importance_original_features(
    candidate_models, X, y, model_name, target_mode,
    feature_cols, stratify_col=None
):
    """Compute permutation importance across all repeated splits."""
    rng = np.random.default_rng(777)

    # Accumulate per-seed importance arrays
    per_seed_drops = {feat: {"multi": [], "char": [], "oil": [], "gas": []} for feat in feature_cols}

    model_tuple = next(m for m in candidate_models if m[0] == model_name)
    _, model, preprocessor = model_tuple

    for seed in RANDOM_SEEDS:
        stratify = stratify_col.loc[X.index] if stratify_col is not None else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=stratify
        )
        pipe, _ = fit_predict_for_mode(model, preprocessor, X_train, y_train, X_test, target_mode)

        baseline_pred  = predict_for_mode(pipe, X_test, target_mode)
        baseline_multi = safe_r2(y_test, baseline_pred, "uniform_average")
        baseline_parts = {
            "char": safe_r2(y_test[:, 0], baseline_pred[:, 0]),
            "oil":  safe_r2(y_test[:, 1], baseline_pred[:, 1]),
            "gas":  safe_r2(y_test[:, 2], baseline_pred[:, 2]),
        }

        for feat in feature_cols:
            if feat not in X_test.columns:
                continue
            for _ in range(PERMUTATION_REPEATS):
                X_perm = X_test.copy()
                shuffled = X_perm[feat].to_numpy().copy()
                rng.shuffle(shuffled)
                X_perm[feat] = shuffled
                pred_perm = predict_for_mode(pipe, X_perm, target_mode)

                per_seed_drops[feat]["multi"].append(
                    baseline_multi - safe_r2(y_test, pred_perm, "uniform_average"))
                per_seed_drops[feat]["char"].append(
                    (baseline_parts["char"] or 0) - safe_r2(y_test[:, 0], pred_perm[:, 0]))
                per_seed_drops[feat]["oil"].append(
                    (baseline_parts["oil"]  or 0) - safe_r2(y_test[:, 1], pred_perm[:, 1]))
                per_seed_drops[feat]["gas"].append(
                    (baseline_parts["gas"]  or 0) - safe_r2(y_test[:, 2], pred_perm[:, 2]))

    rows = []
    for feat in feature_cols:
        d = per_seed_drops[feat]
        rows.append({
            "feature":                       feat,
            "importance_multi_R2_drop_mean": float(np.nanmean(d["multi"])),
            "importance_multi_R2_drop_sd":   float(np.nanstd(d["multi"], ddof=1)) if len(d["multi"]) > 1 else 0.0,
            "importance_char_R2_drop_mean":  float(np.nanmean(d["char"])),
            "importance_oil_R2_drop_mean":   float(np.nanmean(d["oil"])),
            "importance_gas_R2_drop_mean":   float(np.nanmean(d["gas"])),
        })

    return pd.DataFrame(rows).sort_values("importance_multi_R2_drop_mean", ascending=False).reset_index(drop=True)


# =========================
# MAIN
# =========================
def main():
    print("\n==============================")
    print("MAIN PREDICTION SCREEN")
    print("==============================")
    print(f"Input:  {INPUT_XLSX.resolve()}")
    print(f"Sheet:  {SHEET_NAME}")

    if not INPUT_XLSX.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_XLSX}. Put the workbook in the same folder as this script or edit INPUT_XLSX."
        )

    df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)
    print(f"Rows loaded from sheet: {len(df)}")
    if len(df) != EXPECTED_ROWS_LOADED:
        raise ValueError(
            f"Expected {EXPECTED_ROWS_LOADED} rows from {SHEET_NAME}, but loaded {len(df)}. "
            "Check that the intended dataset sheet is being used."
        )
    if "study_id" in df.columns:
        n_studies_loaded = df["study_id"].nunique()
        print(f"Studies loaded from sheet: {n_studies_loaded}")
        if n_studies_loaded != EXPECTED_STUDIES:
            print(
                f"WARNING: Expected {EXPECTED_STUDIES} studies, but found {n_studies_loaded}. "
                "Continue only if this matches your model-ready sheet."
            )

    # Clean likely string descriptors
    for col in REPORTING_FEATURES:
        if col in df.columns:
            df[col] = clean_string_series(df[col])

    # Convert numeric fields
    numeric_needed = PROCESS_FEATURES + PROXIMATE_FEATURES + ULTIMATE_FEATURES + TARGET_COLS + SENSITIVITY_COLS
    for col in numeric_needed:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required_cols = PROCESS_FEATURES + PROXIMATE_FEATURES + ULTIMATE_FEATURES + REPORTING_FEATURES + TARGET_COLS
    missing_cols  = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = add_basis_diagnostic_proxies(df)

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

    numeric_features     = PROCESS_FEATURES + PROXIMATE_FEATURES + ULTIMATE_FEATURES + BASIS_PROXY_NUMERIC
    categorical_features = REPORTING_FEATURES + BASIS_PROXY_CATEGORICAL
    feature_cols         = numeric_features + categorical_features

    # Complete-case filter for final main matrix
    before    = len(df)
    model_df  = df.dropna(subset=feature_cols + TARGET_COLS).copy().reset_index(drop=True)
    after     = len(model_df)

    print(f"Rows before final feature/target complete-case check: {before}")
    print(f"Rows after  final feature/target complete-case check: {after}")

    # NS audit in reporting descriptors
    ns_counts = {}
    for col in REPORTING_FEATURES:
        ns_counts[col] = int((model_df[col].astype(str).str.upper().str.strip() == "NS").sum())
    print("NS counts in reporting/basis descriptors after filtering:")
    for k, v in ns_counts.items():
        print(f"  {k}: {v}")

    # Optional sensitivity availability audit
    sens_rows = []
    for col in SENSITIVITY_COLS:
        if col in model_df.columns:
            sens_rows.append({
                "sensitivity_column":                         col,
                "numeric_non_missing_rows_within_dataset":  int(model_df[col].notna().sum()),
                "pct_available_within_dataset":              model_df[col].notna().mean() * 100.0,
            })
    sensitivity_availability = pd.DataFrame(sens_rows)

    # Model matrix sheet
    id_cols              = [c for c in ID_COLS_CANDIDATES if c in model_df.columns]
    sensitivity_present  = [c for c in SENSITIVITY_COLS if c in model_df.columns]
    model_matrix         = model_df[id_cols + feature_cols + TARGET_COLS + sensitivity_present].copy()

    X = model_df[feature_cols].copy()
    y = model_df[TARGET_COLS].to_numpy(dtype=float)

    # Stratification series for splits
    stratify_col = None
    if "feedstock_class" in model_df.columns and model_df["feedstock_class"].notna().all():
        class_counts = model_df["feedstock_class"].value_counts()
        if class_counts.min() >= 2:
            stratify_col = model_df["feedstock_class"]
        else:
            print("WARNING: feedstock_class has at least one class with <2 rows; S0 splits will not be stratified.")

    # Pre-built preprocessors
    tree_preprocessor   = build_preprocessor(numeric_features, categorical_features, scale_numeric=False)
    scaled_preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric=True)

    candidate_models = []

    # ====================================================
    # Literature-aligned model screen
    # ====================================================
    # Included: GBR, XGBoost, ANN/NN/DNN/MLP, RF, SVR, LightGBM, CatBoost,
    #           AdaBoost, DT, kNN, ET, Ridge/RR, LR/MLR, GPR, ElasticNet/Lasso.
    #
    # Not included as main scikit-learn baselines:
    # CNN/GNN: not appropriate for this small tabular dataset without custom architecture.
    # ANFIS/ELM/GAM/GMM: require specialised packages or are not direct supervised
    #   multi-output regressors in the same pipeline framework. Noted in SI.

    # Tree and boosting models
    candidate_models.append(("RF_base",   RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1), tree_preprocessor))
    candidate_models.append(("RF_smooth", RandomForestRegressor(n_estimators=700, min_samples_leaf=4, max_features=0.70, random_state=0, n_jobs=-1), tree_preprocessor))
    candidate_models.append(("RF_deep",   RandomForestRegressor(n_estimators=700, min_samples_leaf=1, max_features=1.00, random_state=0, n_jobs=-1), tree_preprocessor))

    candidate_models.append(("ET_base",   ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=0, n_jobs=-1), tree_preprocessor))
    candidate_models.append(("ET_smooth", ExtraTreesRegressor(n_estimators=700, min_samples_leaf=4, max_features=0.70, random_state=0, n_jobs=-1), tree_preprocessor))
    candidate_models.append(("ET_deep",   ExtraTreesRegressor(n_estimators=700, min_samples_leaf=1, max_features=1.00, random_state=0, n_jobs=-1), tree_preprocessor))

    candidate_models.append(("GBR_base",    GradientBoostingRegressor(n_estimators=300, learning_rate=0.04,  max_depth=3, random_state=0), tree_preprocessor))
    candidate_models.append(("GBR_shallow", GradientBoostingRegressor(n_estimators=500, learning_rate=0.025, max_depth=2, random_state=0), tree_preprocessor))
    candidate_models.append(("GBR_deep",    GradientBoostingRegressor(n_estimators=250, learning_rate=0.035, max_depth=4, random_state=0), tree_preprocessor))

    candidate_models.append(("HistGB_base",       HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04,  l2_regularization=0.01, random_state=0), tree_preprocessor))
    candidate_models.append(("HistGB_regularized", HistGradientBoostingRegressor(max_iter=500, learning_rate=0.025, l2_regularization=0.10, max_leaf_nodes=15, random_state=0), tree_preprocessor))

    candidate_models.append(("AdaBoost_base", AdaBoostRegressor(n_estimators=300, learning_rate=0.04, random_state=0), tree_preprocessor))
    candidate_models.append(("AdaBoost_slow", AdaBoostRegressor(n_estimators=500, learning_rate=0.02, random_state=0), tree_preprocessor))

    candidate_models.append(("DT_pruned", DecisionTreeRegressor(max_depth=8,    min_samples_leaf=4, random_state=0), tree_preprocessor))
    candidate_models.append(("DT_deep",   DecisionTreeRegressor(max_depth=None, min_samples_leaf=3, random_state=0), tree_preprocessor))

    # Distance and linear baselines
    candidate_models.append(("kNN_5",  KNeighborsRegressor(n_neighbors=5,  weights="distance"), scaled_preprocessor))
    candidate_models.append(("kNN_8",  KNeighborsRegressor(n_neighbors=8,  weights="distance"), scaled_preprocessor))
    candidate_models.append(("kNN_12", KNeighborsRegressor(n_neighbors=12, weights="distance"), scaled_preprocessor))

    candidate_models.append(("LR",                LinearRegression(fit_intercept=True),  scaled_preprocessor))
    candidate_models.append(("Linear_no_intercept", LinearRegression(fit_intercept=False), scaled_preprocessor))
    candidate_models.append(("RR",                Ridge(alpha=3.0),                      scaled_preprocessor))
    candidate_models.append(("Ridge_strong",       Ridge(alpha=15.0),                    scaled_preprocessor))
    candidate_models.append(("ElasticNet",         ElasticNet(alpha=0.02, l1_ratio=0.20, max_iter=30000, random_state=0), scaled_preprocessor))
    candidate_models.append(("Lasso",              Lasso(alpha=0.005,     max_iter=30000, random_state=0), scaled_preprocessor))
    candidate_models.append(("BR",                 BayesianRidge(), scaled_preprocessor))

    # Support vector regression variants
    if RUN_SVR:
        candidate_models.append(("SVR_linear", SVR(kernel="linear", C=10.0,                   epsilon=0.2),               scaled_preprocessor))
        candidate_models.append(("SVR_rbf",    SVR(kernel="rbf",    C=25.0,  gamma="scale",   epsilon=0.2),               scaled_preprocessor))
        candidate_models.append(("SVR_poly",   SVR(kernel="poly",   degree=2, C=10.0,          epsilon=0.2, gamma="scale"), scaled_preprocessor))

    # Neural-network variants: ANN/NN/DNN/MLP represented by MLPRegressor for tabular data.
    if RUN_NEURAL_NETS:
        candidate_models.append(("ANN_small",  MLPRegressor(hidden_layer_sizes=(32,),        activation="relu", alpha=0.001, learning_rate_init=0.001,  max_iter=2000, early_stopping=True, random_state=0), scaled_preprocessor))
        candidate_models.append(("ANN_medium", MLPRegressor(hidden_layer_sizes=(64, 32),     activation="relu", alpha=0.001, learning_rate_init=0.001,  max_iter=2500, early_stopping=True, random_state=0), scaled_preprocessor))
        candidate_models.append(("DNN_deep",   MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu", alpha=0.002, learning_rate_init=0.0008, max_iter=3000, early_stopping=True, random_state=0), scaled_preprocessor))
        candidate_models.append(("MLP_tanh",   MLPRegressor(hidden_layer_sizes=(64, 32),     activation="tanh", alpha=0.001, learning_rate_init=0.001,  max_iter=2500, early_stopping=True, random_state=0), scaled_preprocessor))

    # Gaussian process regression
    if RUN_GPR:
        gpr_kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        candidate_models.append(("GPR_RBF", GaussianProcessRegressor(kernel=gpr_kernel, alpha=1e-6, normalize_y=True, random_state=0), scaled_preprocessor))

    # Optional external boosting packages
    if HAS_XGB:
        candidate_models.append(("XGBoost_base",        XGBRegressor(objective="reg:squarederror", n_estimators=500, max_depth=4, learning_rate=0.035, subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0, reg_alpha=0.00, random_state=0, n_jobs=-1), tree_preprocessor))
        candidate_models.append(("XGBoost_regularized", XGBRegressor(objective="reg:squarederror", n_estimators=700, max_depth=3, learning_rate=0.025, subsample=0.80, colsample_bytree=0.80, reg_lambda=5.0, reg_alpha=0.05, random_state=0, n_jobs=-1), tree_preprocessor))
        candidate_models.append(("XGBoost_shallow",     XGBRegressor(objective="reg:squarederror", n_estimators=500, max_depth=2, learning_rate=0.035, subsample=0.85, colsample_bytree=0.85, reg_lambda=3.0, reg_alpha=0.00, random_state=0, n_jobs=-1), tree_preprocessor))
    else:
        print("WARNING: xgboost is not installed. XGBoost will be skipped.")

    if HAS_LGBM:
        candidate_models.append(("LightGBM_base",        LGBMRegressor(n_estimators=500, learning_rate=0.035, max_depth=-1, num_leaves=31, subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0, random_state=0, n_jobs=-1, verbosity=-1), tree_preprocessor))
        candidate_models.append(("LightGBM_regularized", LGBMRegressor(n_estimators=700, learning_rate=0.025, max_depth=4,  num_leaves=15, subsample=0.80, colsample_bytree=0.80, reg_lambda=5.0, reg_alpha=0.05, random_state=0, n_jobs=-1, verbosity=-1), tree_preprocessor))
    else:
        print("WARNING: lightgbm is not installed. LightGBM will be skipped.")

    if HAS_CATBOOST:
        candidate_models.append(("CatBoost_base",        CatBoostRegressor(iterations=500, depth=4, learning_rate=0.035, loss_function="RMSE", random_seed=0, verbose=False), tree_preprocessor))
        candidate_models.append(("CatBoost_regularized", CatBoostRegressor(iterations=700, depth=3, learning_rate=0.025, l2_leaf_reg=5.0, loss_function="RMSE", random_seed=0, verbose=False), tree_preprocessor))
    else:
        print("WARNING: catboost is not installed. CatBoost will be skipped.")

    print(f"Candidate model configurations: {len(candidate_models)}")

    target_modes = list(TARGET_LABELS.keys())
    total_fits   = len(RANDOM_SEEDS) * len(candidate_models) * len(target_modes)

    detail_rows = []
    fit_count   = 0
    print(f"\nRunning model screen ({total_fits} total fits)...")

    for seed in RANDOM_SEEDS:
        stratify = stratify_col if stratify_col is not None else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed,
            stratify=stratify.to_numpy() if stratify is not None else None,
        )
        for model_name, model, preprocessor in candidate_models:
            for target_mode in target_modes:
                _, pred = fit_predict_for_mode(model, preprocessor, X_train, y_train, X_test, target_mode)
                detail_rows.append(metric_block(
                    y_true_raw=y_test,
                    y_pred=pred,
                    validation_label="S0_random_row_interpolation",
                    model_name=model_name,
                    target_mode=target_mode,
                    seed=seed,
                    n_train=len(X_train),
                    n_test=len(X_test),
                ))
                fit_count += 1
            print(f"  seed {seed:02d} | {model_name:<28s} | {fit_count}/{total_fits}", end="\r")

    print()  # newline after progress line
    detailed = pd.DataFrame(detail_rows)

    # Summary: mean and sd over repeated random splits
    grp_cols    = ["validation", "model", "target_mode", "target_label"]
    metric_cols = [
        "ranking_R2",
        "orig_multi_R2", "orig_char_R2", "orig_oil_R2", "orig_gas_R2", "orig_RMSE", "orig_MAE",
        "closed_true_multi_R2", "closed_true_char_R2", "closed_true_oil_R2", "closed_true_gas_R2",
        "closed_true_RMSE", "closed_true_MAE",
        "pred_sum_mean", "pred_sum_sd", "pct_pred_sum_outside_95_105",
    ]

    summary_mean = detailed.groupby(grp_cols, as_index=False)[metric_cols].mean()
    summary_sd   = detailed.groupby(grp_cols, as_index=False)[metric_cols].std()

    summary_sd = summary_sd.rename(columns={c: f"{c}_sd" for c in metric_cols})

    summary = summary_mean.merge(summary_sd, on=grp_cols, how="left")
    summary = summary.sort_values(
        ["ranking_R2", "orig_multi_R2", "closed_true_multi_R2"], ascending=False
    ).reset_index(drop=True)
    summary.insert(0, "rank_by_ranking_R2", np.arange(1, len(summary) + 1))

    print("\nTOP 15 BY MEAN RANKING R2")
    show_cols = [
        "rank_by_ranking_R2", "model", "target_label", "ranking_R2",
        "orig_multi_R2", "orig_char_R2", "orig_oil_R2", "orig_gas_R2",
        "orig_RMSE", "orig_MAE", "pred_sum_mean", "pred_sum_sd", "pct_pred_sum_outside_95_105",
    ]
    print(summary[show_cols].head(15).to_string(index=False))

    # Competitive configurations retained for inspection
    best_ranking    = summary["ranking_R2"].max()
    comparable      = summary[summary["ranking_R2"] >= (best_ranking - 0.05)].copy()
    top15           = summary.head(15).copy()
    comparable_configurations = (
        pd.concat([comparable, top15], ignore_index=True)
        .drop_duplicates(subset=["validation", "model", "target_mode"])
        .sort_values(["ranking_R2", "orig_multi_R2", "closed_true_multi_R2"], ascending=False)
        .reset_index(drop=True)
    )

    # Literature model families not implemented in the main pipeline
    not_implemented = pd.DataFrame({
        "literature_label": ["CNN", "GNN", "ANFIS", "ELM", "GAM", "GMM"],
        "reason_not_screened_in_main_pipeline": [
            "Convolutional networks are image/sequence-oriented and require a custom architecture; not appropriate for this small tabular matrix.",
            "Graph neural networks require graph-structured inputs, which are not available in the current tabular dataset.",
            "Requires specialised neuro-fuzzy implementation and tuning outside the reproducible scikit-learn pipeline.",
            "Requires specialised extreme-learning-machine implementation; not part of the standard scikit-learn API.",
            "Generalized additive models require a separate GAM package/design; reserved for future sensitivity analysis if needed.",
            "Gaussian mixture models are primarily unsupervised density/clustering models, not a direct supervised multi-output regressor.",
        ],
    })

    top_by_mode = (
        summary.sort_values("ranking_R2", ascending=False)
        .groupby("target_mode", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    importance_sheets = {}
    print("\nComputing permutation importance for best model within each target mode (all seeds)...")
    for _, row in top_by_mode.iterrows():
        model_name  = row["model"]
        target_mode = row["target_mode"]
        print(f"  {model_name} | {TARGET_LABELS[target_mode]}")
        imp = permutation_importance_original_features(
            candidate_models=candidate_models,
            X=X,
            y=y,
            model_name=model_name,
            target_mode=target_mode,
            feature_cols=feature_cols,
            stratify_col=stratify_col,
        )
        imp.insert(0, "model",        model_name)
        imp.insert(1, "target_mode",  target_mode)
        imp.insert(2, "target_label", TARGET_LABELS[target_mode])
        importance_sheets[f"perm_{target_mode[:20]}"] = imp

    # Write outputs
    print(f"\nSaving results to: {OUTPUT_XLSX.resolve()}")
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        model_matrix.to_excel(           writer, sheet_name="30_Model_Matrix_442",        index=False)
        sensitivity_availability.to_excel(writer, sheet_name="31_Sensitivity_Availability", index=False)
        detailed.to_excel(               writer, sheet_name="32_Detailed_Seed_Results",    index=False)
        summary.to_excel(                writer, sheet_name="33_Model_Ranking_Summary",    index=False)
        top_by_mode.to_excel(            writer, sheet_name="34_Best_Model_By_TargetMode", index=False)
        comparable_configurations.to_excel(
            writer, sheet_name="35_Comparable_Configurations", index=False
        )
        not_implemented.to_excel(        writer, sheet_name="36_Not_Implemented_Notes",    index=False)
        for sheet_name_key, imp_df in importance_sheets.items():
            imp_df.to_excel(writer, sheet_name=sheet_name_key, index=False)

    print("\nDONE.")


if __name__ == "__main__":
    main()
