"""Feature-engineering sensitivity analysis reported in Supplementary Table S3.

The screen compares core, ratio, severity, reporting-interaction, and
training-fold-imputed holding-time variants under the three target strategies.
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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


# Analysis settings
INPUT_XLSX = DATA_FILE
INPUT_SHEET = DATA_SHEET
OUTPUT_XLSX = output_file("02_feature_engineering_sensitivity_results.xlsx")

EXPECTED_ROWS_LOADED = 442
EXPECTED_STUDIES = 42

TEST_SIZE = 0.20
RANDOM_SEEDS = list(range(20))

TARGET_MODES = ["direct_raw", "direct_forced_closed", "ILR_closed"]

TARGET_LABELS = {
    "direct_raw": "Unconstrained",
    "direct_forced_closed": "Renormalized",
    "ILR_closed": "ILR-transformed",
}

SELECTED_TARGET_FOR_PANEL_B = "direct_forced_closed"


# ============================================================
# 2. COLUMNS
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

SENSITIVITY_COLS = [
    "holding_time_min",
    "sample_mass_g",
    "particle_size_mm",
]


# ============================================================
# 3. HELPERS
# ============================================================

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def clean_string_series(s):
    return s.astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})


def safe_divide(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return np.where(np.abs(b) > 1e-12, a / b, np.nan)


def parse_numeric_or_midrange(value):
    """
    Parse numeric values and simple ranges into a single representative value.

    Examples:
      20 -> 20
      "20" -> 20
      "10-30" or "10–30" -> 20
      "NR", "NS", blank -> NaN

    This is used only for optional holding-time severity descriptors. Missing
    values are not filled here; they are imputed inside the modelling pipeline.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    s = str(value).strip()
    if s == "" or s.upper() in {"NS", "NR", "NA", "N/A", "NONE", "NAN", "-", "NOT REPORTED"}:
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
         .replace("<", "")
         .replace(">", "")
         .replace("≤", "")
         .replace("≥", "")
    )

    import re
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not nums:
        return np.nan

    vals = [float(x) for x in nums]
    if len(vals) == 1:
        return vals[0]
    return float(np.mean(vals[:2]))


def safe_r2(y_true, y_pred, multioutput="uniform_average"):
    """
    R² with protection against constant target columns.
    For multi-output targets, average only valid non-constant target outputs.
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


def force_close_100(y):
    y = np.asarray(y, dtype=float)
    y = np.maximum(y, 1e-10)
    return y / y.sum(axis=1, keepdims=True) * 100.0


def ilr_transform(comp):
    comp = np.asarray(comp, dtype=float)
    comp = np.maximum(comp, 1e-10)
    comp = comp / comp.sum(axis=1, keepdims=True)

    char = comp[:, 0]
    oil = comp[:, 1]
    gas = comp[:, 2]

    z1 = np.sqrt(2.0 / 3.0) * np.log(char / np.sqrt(oil * gas))
    z2 = np.sqrt(1.0 / 2.0) * np.log(oil / gas)
    return np.column_stack([z1, z2])


def ilr_inverse(z):
    z = np.asarray(z, dtype=float)
    x_char = np.exp(np.sqrt(2.0 / 3.0) * z[:, 0])
    x_oil = np.exp(-np.sqrt(1.0 / 6.0) * z[:, 0] + np.sqrt(1.0 / 2.0) * z[:, 1])
    x_gas = np.exp(-np.sqrt(1.0 / 6.0) * z[:, 0] - np.sqrt(1.0 / 2.0) * z[:, 1])

    comp = np.column_stack([x_char, x_oil, x_gas])
    return comp / comp.sum(axis=1, keepdims=True)


def build_preprocessor(numeric_features, categorical_features, scale_numeric=False):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    numeric_pipe = Pipeline(num_steps)
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


def add_basis_diagnostic_proxies(df):
    df = df.copy()

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

    return df


def add_engineered_features(df):
    """
    Add target-free engineered descriptors for feature-variant comparison.
    """
    df = df.copy()

    # Elemental ratios
    df["H_C_ratio"] = safe_divide(df["hydrogen_wt_pct"], df["carbon_wt_pct"])
    df["O_C_ratio"] = safe_divide(df["oxygen_wt_pct"], df["carbon_wt_pct"])
    df["N_C_ratio"] = safe_divide(df["nitrogen_wt_pct"], df["carbon_wt_pct"])

    # Proximate ratios/fractions
    df["VM_FC_ratio"] = safe_divide(df["volatile_matter_wt_pct"], df["fixed_carbon_wt_pct"])
    df["Ash_VM_ratio"] = safe_divide(df["ash_wt_pct"], df["volatile_matter_wt_pct"])

    prox_sum = (
        df["volatile_matter_wt_pct"]
        + df["fixed_carbon_wt_pct"]
        + df["ash_wt_pct"]
    )
    df["VM_fraction_of_proximate"] = safe_divide(df["volatile_matter_wt_pct"], prox_sum)
    df["FC_fraction_of_proximate"] = safe_divide(df["fixed_carbon_wt_pct"], prox_sum)
    df["Ash_fraction_of_proximate"] = safe_divide(df["ash_wt_pct"], prox_sum)

    # Severity descriptors
    temp_c = pd.to_numeric(df["temperature_c"], errors="coerce")
    temp_k = temp_c + 273.15
    heating_rate = pd.to_numeric(df["heating_rate_c_min"], errors="coerce")

    df["inv_temp_K"] = np.where(temp_k > 0, 1.0 / temp_k, np.nan)
    df["log_heating_rate"] = np.log1p(np.maximum(heating_rate, 0))
    df["temp_x_log_heating_rate"] = temp_c * df["log_heating_rate"]

    # Holding-time severity descriptors.
    # These are intentionally used only in the dedicated F2b imputed-holding
    # variant, because holding time is incompletely reported. Missing values
    # remain NaN here and are imputed inside the sklearn pipeline.
    if "holding_time_min" in df.columns:
        df["holding_time_min_num"] = df["holding_time_min"].apply(parse_numeric_or_midrange)
    else:
        df["holding_time_min_num"] = np.nan

    holding = pd.to_numeric(df["holding_time_min_num"], errors="coerce")
    df["log_holding_time"] = np.log1p(np.maximum(holding, 0))
    df["temp_x_log_holding_time"] = temp_c * df["log_holding_time"]
    df["thermal_severity_time_rate"] = temp_c * df["log_heating_rate"] * df["log_holding_time"]

    # Reporting-interaction descriptors are categorical, target-free, and compact.
    df["pyrolysis_yield_basis_pair"] = (
        clean_string_series(df["pyrolysis_type"])
        + "__"
        + clean_string_series(df["yield_basis"])
    )
    df["oil_gas_reporting_pair"] = (
        clean_string_series(df["oil_definition"])
        + "__"
        + clean_string_series(df["gas_yield_method"])
    )
    df["basis_oil_definition_pair"] = (
        clean_string_series(df["yield_basis"])
        + "__"
        + clean_string_series(df["oil_definition"])
    )

    return df


def get_feature_variant_definitions():
    """
    Return a dictionary of feature variants.
    Each value has:
      numeric: list of numeric predictors
      categorical: list of categorical predictors
    """

    basis_proxy_numeric = [
        "proximate_sum_raw_wt_pct",
        "proximate_sum_deviation_from_100",
        "ultimate_CHNO_sum_raw_wt_pct",
        "ultimate_CHNO_sum_deviation_from_100",
    ]

    basis_proxy_categorical = [
        "proximate_ultimate_basis_pair",
        "yield_reporting_profile",
    ]

    core_numeric = (
        PROCESS_FEATURES
        + PROXIMATE_FEATURES
        + ULTIMATE_FEATURES
        + basis_proxy_numeric
    )

    core_categorical = (
        REPORTING_FEATURES
        + basis_proxy_categorical
    )

    ratio_numeric = [
        "H_C_ratio",
        "O_C_ratio",
        "N_C_ratio",
        "VM_FC_ratio",
        "Ash_VM_ratio",
        "VM_fraction_of_proximate",
        "FC_fraction_of_proximate",
        "Ash_fraction_of_proximate",
    ]

    severity_numeric = [
        "inv_temp_K",
        "log_heating_rate",
        "temp_x_log_heating_rate",
    ]

    holding_severity_numeric = [
        "holding_time_min_num",
        "log_holding_time",
        "temp_x_log_holding_time",
        "thermal_severity_time_rate",
    ]

    reporting_interactions = [
        "pyrolysis_yield_basis_pair",
        "oil_gas_reporting_pair",
        "basis_oil_definition_pair",
    ]

    return {
        "F0_core": {
            "label": "Core",
            "numeric": core_numeric,
            "categorical": core_categorical,
        },
        "F1_core_ratios": {
            "label": "Core + ratios",
            "numeric": core_numeric + ratio_numeric,
            "categorical": core_categorical,
        },
        "F2_core_severity": {
            "label": "Core + severity",
            "numeric": core_numeric + severity_numeric,
            "categorical": core_categorical,
        },
        "F2b_core_severity_holding_imputed": {
            "label": "Core + severity\n+ imputed holding time",
            "numeric": core_numeric + severity_numeric + holding_severity_numeric,
            "categorical": core_categorical,
            "allow_missing_numeric_for_pipeline_imputation": holding_severity_numeric,
        },
        "F3_core_ratios_severity": {
            "label": "Core + ratios + severity",
            "numeric": core_numeric + ratio_numeric + severity_numeric,
            "categorical": core_categorical,
        },
        "F4_reporting_interactions": {
            "label": "Core + ratios + severity\n+ reporting interactions",
            "numeric": core_numeric + ratio_numeric + severity_numeric,
            "categorical": core_categorical + reporting_interactions,
        },
    }


def metric_block(y_true_raw, y_pred, feature_variant, feature_label, model_name, target_mode, seed, n_train, n_test):
    y_true_raw = np.asarray(y_true_raw, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true_closed = force_close_100(y_true_raw)
    pred_sum = y_pred.sum(axis=1)

    return {
        "feature_variant": feature_variant,
        "feature_label": feature_label,
        "model": model_name,
        "target_mode": target_mode,
        "target_label": TARGET_LABELS.get(target_mode, target_mode),
        "seed": seed,
        "n_train": n_train,
        "n_test": n_test,

        # Fair cross-mode ranking
        "ranking_R2": safe_r2(y_true_closed, y_pred, "uniform_average"),

        # Against originally reported yields
        "orig_multi_R2": safe_r2(y_true_raw, y_pred, "uniform_average"),
        "orig_char_R2": safe_r2(y_true_raw[:, 0], y_pred[:, 0]),
        "orig_oil_R2": safe_r2(y_true_raw[:, 1], y_pred[:, 1]),
        "orig_gas_R2": safe_r2(y_true_raw[:, 2], y_pred[:, 2]),
        "orig_RMSE": float(np.sqrt(mean_squared_error(y_true_raw, y_pred))),
        "orig_MAE": float(mean_absolute_error(y_true_raw, y_pred)),

        # Against closed-normalized true yields
        "closed_true_multi_R2": safe_r2(y_true_closed, y_pred, "uniform_average"),
        "closed_true_char_R2": safe_r2(y_true_closed[:, 0], y_pred[:, 0]),
        "closed_true_oil_R2": safe_r2(y_true_closed[:, 1], y_pred[:, 1]),
        "closed_true_gas_R2": safe_r2(y_true_closed[:, 2], y_pred[:, 2]),
        "closed_true_RMSE": float(np.sqrt(mean_squared_error(y_true_closed, y_pred))),
        "closed_true_MAE": float(mean_absolute_error(y_true_closed, y_pred)),

        # Closure diagnostics
        "pred_sum_mean": float(np.mean(pred_sum)),
        "pred_sum_sd": float(np.std(pred_sum, ddof=1)) if len(pred_sum) > 1 else 0.0,
        "pct_pred_sum_outside_95_105": float(np.mean((pred_sum < 95) | (pred_sum > 105)) * 100.0),
    }


def fit_predict_for_mode(model, preprocessor, X_train, y_train_raw, X_test, target_mode):
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", MultiOutputRegressor(clone(model))),
    ])

    if target_mode == "direct_raw":
        pipe.fit(X_train, y_train_raw)
        return pipe.predict(X_test)

    if target_mode == "direct_forced_closed":
        pipe.fit(X_train, y_train_raw)
        return force_close_100(pipe.predict(X_test))

    if target_mode == "ILR_closed":
        y_train_ilr = ilr_transform(y_train_raw)
        pipe.fit(X_train, y_train_ilr)
        return ilr_inverse(pipe.predict(X_test)) * 100.0

    raise ValueError(f"Unknown target_mode: {target_mode}")


# ============================================================
# 4. MAIN
# ============================================================

def main():
    print("\n==============================")
    print("STEP 3 — FEATURE ENGINEERING SCREEN: 442 DATASET")
    print("==============================")
    print(f"Input:  {INPUT_XLSX.resolve()}")
    print(f"Sheet:  {INPUT_SHEET}")

    if not INPUT_XLSX.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_XLSX}. Put the workbook in the same folder as this script."
        )

    if not HAS_XGB:
        raise ImportError("xgboost is not installed. Install xgboost or switch the model in this script.")

    df = pd.read_excel(INPUT_XLSX, sheet_name=INPUT_SHEET)
    print(f"Rows loaded from sheet: {len(df)}")

    if len(df) != EXPECTED_ROWS_LOADED:
        raise ValueError(
            f"Expected {EXPECTED_ROWS_LOADED} rows from {INPUT_SHEET}, but loaded {len(df)}."
        )

    if "study_id" in df.columns:
        n_studies = df["study_id"].nunique()
        print(f"Studies loaded from sheet: {n_studies}")
        if n_studies != EXPECTED_STUDIES:
            print(f"WARNING: expected {EXPECTED_STUDIES} studies, found {n_studies}.")

    # Clean descriptors
    for col in REPORTING_FEATURES:
        if col in df.columns:
            df[col] = clean_string_series(df[col])

    # Convert numeric fields
    numeric_needed = (
        PROCESS_FEATURES
        + PROXIMATE_FEATURES
        + ULTIMATE_FEATURES
        + TARGET_COLS
        + ["sample_mass_g", "particle_size_mm"]
    )
    for col in numeric_needed:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep holding_time_min in its original form until add_engineered_features(),
    # where simple ranges/strings can be parsed into holding_time_min_num.

    required_base = (
        PROCESS_FEATURES
        + PROXIMATE_FEATURES
        + ULTIMATE_FEATURES
        + REPORTING_FEATURES
        + TARGET_COLS
    )
    missing = [c for c in required_base if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = add_basis_diagnostic_proxies(df)
    df = add_engineered_features(df)

    feature_variants = get_feature_variant_definitions()

    # XGBoost configuration aligned with the baseline screen
    model_name = "XGBoost_base"
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        reg_alpha=0.0,
        random_state=0,
        n_jobs=-1,
    )

    # Prepare optional sensitivity availability audit
    sensitivity_rows = []
    for col in SENSITIVITY_COLS:
        if col in df.columns:
            if col == "holding_time_min" and "holding_time_min_num" in df.columns:
                s = df["holding_time_min_num"]
                parsed_col = "holding_time_min_num"
            else:
                s = pd.to_numeric(df[col], errors="coerce")
                parsed_col = col
            sensitivity_rows.append({
                "sensitivity_column": col,
                "parsed_or_numeric_column_used": parsed_col,
                "numeric_non_missing_rows_within_442_sheet": int(s.notna().sum()),
                "pct_available_within_442_sheet": s.notna().mean() * 100.0,
            })
    sensitivity_availability = pd.DataFrame(sensitivity_rows)

    # Stratification labels
    stratify_col = None
    if "feedstock_class" in df.columns and df["feedstock_class"].notna().all():
        counts = df["feedstock_class"].value_counts()
        if counts.min() >= 2:
            stratify_col = df["feedstock_class"]
        else:
            print("WARNING: feedstock_class has class with <2 rows; random splits will not be stratified.")

    detailed_rows = []
    matrix_audit_rows = []

    total_fits = len(feature_variants) * len(TARGET_MODES) * len(RANDOM_SEEDS)
    fit_count = 0

    for fv, spec in feature_variants.items():
        feature_label = spec["label"]
        numeric_features = spec["numeric"]
        categorical_features = spec["categorical"]
        feature_cols = numeric_features + categorical_features

        missing_fv_cols = [c for c in feature_cols if c not in df.columns]
        if missing_fv_cols:
            raise ValueError(f"{fv} missing feature columns: {missing_fv_cols}")

        # For the dedicated imputed-holding variant, do not drop rows only because
        # holding-time-derived descriptors are missing. Those values are imputed
        # inside the train/test pipeline. All other feature variants remain
        # complete-case with respect to their selected predictors.
        allow_missing_for_imputation = spec.get("allow_missing_numeric_for_pipeline_imputation", [])
        drop_subset = [
            c for c in feature_cols + TARGET_COLS
            if c not in allow_missing_for_imputation
        ]
        model_df = df.dropna(subset=drop_subset).copy().reset_index(drop=True)

        matrix_audit_rows.append({
            "feature_variant": fv,
            "feature_label": feature_label,
            "rows_before_feature_filter": len(df),
            "rows_after_feature_filter": len(model_df),
            "studies_after_feature_filter": model_df["study_id"].nunique() if "study_id" in model_df.columns else np.nan,
            "n_numeric_features": len(numeric_features),
            "n_categorical_features": len(categorical_features),
            "n_total_features_before_onehot": len(feature_cols),
            "allowed_missing_numeric_imputed_in_pipeline": "; ".join(spec.get("allow_missing_numeric_for_pipeline_imputation", [])) if spec.get("allow_missing_numeric_for_pipeline_imputation", []) else "None",
            "holding_time_available_rows_in_matrix": int(model_df["holding_time_min_num"].notna().sum()) if "holding_time_min_num" in model_df.columns else np.nan,
            "holding_time_missing_rows_imputed_in_matrix": int(model_df["holding_time_min_num"].isna().sum()) if "holding_time_min_num" in model_df.columns else np.nan,
            "numeric_features": "; ".join(numeric_features),
            "categorical_features": "; ".join(categorical_features),
        })

        if len(model_df) != EXPECTED_ROWS_LOADED:
            print(
                f"WARNING: {fv} has {len(model_df)} rows after feature filtering. "
                "This may mean an engineered feature introduced missingness."
            )

        X = model_df[feature_cols].copy()
        y = model_df[TARGET_COLS].to_numpy(dtype=float)

        local_stratify = None
        if "feedstock_class" in model_df.columns and model_df["feedstock_class"].notna().all():
            counts = model_df["feedstock_class"].value_counts()
            if counts.min() >= 2:
                local_stratify = model_df["feedstock_class"]

        preprocessor = build_preprocessor(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            scale_numeric=False,
        )

        for seed in RANDOM_SEEDS:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                random_state=seed,
                stratify=local_stratify.to_numpy() if local_stratify is not None else None,
            )

            for target_mode in TARGET_MODES:
                y_pred = fit_predict_for_mode(
                    model=model,
                    preprocessor=preprocessor,
                    X_train=X_train,
                    y_train_raw=y_train,
                    X_test=X_test,
                    target_mode=target_mode,
                )

                detailed_rows.append(metric_block(
                    y_true_raw=y_test,
                    y_pred=y_pred,
                    feature_variant=fv,
                    feature_label=feature_label,
                    model_name=model_name,
                    target_mode=target_mode,
                    seed=seed,
                    n_train=len(X_train),
                    n_test=len(X_test),
                ))

                fit_count += 1
                print(
                    f"  {fit_count}/{total_fits} | {fv:<27s} | {TARGET_LABELS[target_mode]:<16s}",
                    end="\r",
                )

    print()

    detailed = pd.DataFrame(detailed_rows)
    matrix_audit = pd.DataFrame(matrix_audit_rows)

    group_cols = ["feature_variant", "feature_label", "model", "target_mode", "target_label"]
    metric_cols = [
        "ranking_R2",
        "orig_multi_R2", "orig_char_R2", "orig_oil_R2", "orig_gas_R2",
        "orig_RMSE", "orig_MAE",
        "closed_true_multi_R2", "closed_true_char_R2", "closed_true_oil_R2", "closed_true_gas_R2",
        "closed_true_RMSE", "closed_true_MAE",
        "pred_sum_mean", "pred_sum_sd", "pct_pred_sum_outside_95_105",
        "n_test",
    ]

    summary_mean = detailed.groupby(group_cols, as_index=False)[metric_cols].mean()
    summary_sd = detailed.groupby(group_cols, as_index=False)[metric_cols].std()
    summary_sd = summary_sd.rename(columns={c: f"{c}_sd" for c in metric_cols})

    summary = summary_mean.merge(summary_sd, on=group_cols, how="left")
    summary = summary.sort_values(["ranking_R2", "closed_true_RMSE"], ascending=[False, True]).reset_index(drop=True)
    summary.insert(0, "rank_by_ranking_R2", np.arange(1, len(summary) + 1))

    best_by_feature = (
        summary.sort_values("ranking_R2", ascending=False)
        .groupby("feature_variant", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    target_label_table = pd.DataFrame({
        "target_mode": list(TARGET_LABELS.keys()),
        "target_label": [TARGET_LABELS[k] for k in TARGET_LABELS],
        "definition": [
            "Direct prediction of reported yields without closure correction; ranked against closed-normalized true yields for cross-mode fairness.",
            "Direct prediction followed by post-hoc rescaling to a 100% yield sum.",
            "Prediction in ILR coordinates followed by back-transformation to a 100% yield sum.",
        ],
    })

    print("\nTOP FEATURE-ENGINEERING RESULTS")
    show_cols = [
        "rank_by_ranking_R2", "feature_label", "model", "target_label",
        "ranking_R2", "closed_true_RMSE", "closed_true_MAE",
        "pred_sum_mean", "pct_pred_sum_outside_95_105",
    ]
    print(summary[show_cols].head(20).to_string(index=False))

    print(f"\nSaving results to: {OUTPUT_XLSX.resolve()}")
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        matrix_audit.to_excel(writer, sheet_name="30_Feature_Matrix_Audit", index=False)
        sensitivity_availability.to_excel(writer, sheet_name="31_Sensitivity_Availability", index=False)
        detailed.to_excel(writer, sheet_name="32_Detailed_Seed_Results", index=False)
        summary.to_excel(writer, sheet_name="33_Feature_Engineering_Summary", index=False)
        best_by_feature.to_excel(writer, sheet_name="34_Best_By_FeatureVariant", index=False)
        target_label_table.to_excel(writer, sheet_name="35_Target_Labels", index=False)

    print("\nDONE.")
    print("Use sheet 33_Feature_Engineering_Summary for Supplementary Table S3.")


if __name__ == "__main__":
    main()
