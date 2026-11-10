"""Random-row feature-group ablation reported in Table 3.

The analysis compares the full reporting-aware feature set with versions that
remove composition-sum diagnostics or retain thermochemical predictors only.
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
from sklearn.model_selection import train_test_split
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
OUTPUT_WORKBOOK = output_file("06a_random_row_ablation_results.xlsx")

EXPECTED_ROWS = 442
EXPECTED_STUDIES = 42

RANDOM_SEEDS = list(range(20))
TEST_SIZE = 0.20

TARGET_MODES = [
    "direct_raw",
    "direct_forced_closed",
    "ILR_closed",
]

TARGET_MODE_LABELS = {
    "direct_raw": "Unconstrained",
    "direct_forced_closed": "Renormalized",
    "ILR_closed": "ILR-transformed",
}


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

TARGET_SHORT = ["char", "oil", "gas"]


FEATURE_VARIANTS: Dict[str, Dict[str, List[str]]] = {
    # Current full F2-like model from your SHAP workflow.
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

    # Removes the numerical proximate/ultimate sum/deviation diagnostics,
    # while retaining reporting definitions such as oil_definition and gas_yield_method.
    "Reporting_aware_without_sum_diagnostics": {
        "numeric": (
            PROCESS_FEATURES
            + PROXIMATE_FEATURES
            + ULTIMATE_FEATURES
            + SEVERITY_FEATURES
        ),
        "categorical": REPORTING_FEATURES + REPORTING_COMBINED_FEATURES,
    },

    # Strict clean model: only measured process + proximate + ultimate composition.
    "Thermochemical_only_raw": {
        "numeric": PROCESS_FEATURES + PROXIMATE_FEATURES + ULTIMATE_FEATURES,
        "categorical": [],
    },

    # Clean model plus physical severity descriptors derived only from T and heating rate.
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


def fit_pipeline(
    X_train: pd.DataFrame,
    y_train_raw: np.ndarray,
    numeric_features: List[str],
    categorical_features: List[str],
    target_mode: str,
    seed: int,
) -> Pipeline:
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model = get_xgb_model(random_state=seed)

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", MultiOutputRegressor(clone(model))),
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

    raise ValueError(f"Unknown target mode: {target_mode}")


def predict_pipeline(pipe: Pipeline, X: pd.DataFrame, target_mode: str) -> np.ndarray:
    pred = pipe.predict(X)

    if target_mode == "direct_raw":
        return pred

    if target_mode == "direct_forced_closed":
        return close_to_100(pred)

    if target_mode == "ILR_closed":
        return from_ilr(pred)

    raise ValueError(f"Unknown target mode: {target_mode}")


def evaluate_prediction(y_true_raw: np.ndarray, y_pred: np.ndarray, target_mode: str) -> Dict[str, float]:
    y_true_raw = np.asarray(y_true_raw, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if target_mode == "direct_raw":
        y_eval = y_true_raw
    else:
        y_eval = close_to_100(y_true_raw)

    pred_sum = y_pred.sum(axis=1)

    char_mae = float(mean_absolute_error(y_eval[:, 0], y_pred[:, 0]))
    oil_mae = float(mean_absolute_error(y_eval[:, 1], y_pred[:, 1]))
    gas_mae = float(mean_absolute_error(y_eval[:, 2], y_pred[:, 2]))

    # WAPE is safer than MAPE when low-yield fractions are present.
    denom = np.sum(np.abs(y_eval))
    wape_pct = float(np.sum(np.abs(y_eval - y_pred)) / denom * 100.0) if denom > 0 else np.nan

    return {
        "multi_R2": float(np.nanmean([
            safe_r2(y_eval[:, 0], y_pred[:, 0]),
            safe_r2(y_eval[:, 1], y_pred[:, 1]),
            safe_r2(y_eval[:, 2], y_pred[:, 2]),
        ])),
        "char_R2": safe_r2(y_eval[:, 0], y_pred[:, 0]),
        "oil_R2": safe_r2(y_eval[:, 1], y_pred[:, 1]),
        "gas_R2": safe_r2(y_eval[:, 2], y_pred[:, 2]),
        "RMSE": rmse(y_eval, y_pred),
        "MAE": float(mean_absolute_error(y_eval, y_pred)),
        "char_MAE": char_mae,
        "oil_MAE": oil_mae,
        "gas_MAE": gas_mae,
        "WAPE_pct": wape_pct,
        "pred_sum_mean": float(np.mean(pred_sum)),
        "pred_sum_sd": float(np.std(pred_sum, ddof=1)) if len(pred_sum) > 1 else 0.0,
        "pct_pred_sum_outside_95_105": float(np.mean((pred_sum < 95) | (pred_sum > 105)) * 100.0),
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


def load_data() -> pd.DataFrame:
    workbook = resolve_input_workbook()

    print(f"Input workbook: {workbook.resolve()}")
    print(f"Input sheet: {INPUT_SHEET}")

    df = pd.read_excel(workbook, sheet_name=INPUT_SHEET)
    df = add_engineered_features(df)

    all_needed = set(TARGET_COLS + ["study_id"])
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
# 5. MAIN ABLATION RUN
# ============================================================

def run_random_row_ablation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    y = df[TARGET_COLS].to_numpy(dtype=float)
    idx = np.arange(len(df))

    stratify_base = None
    if "feedstock_class" in df.columns:
        fc = df["feedstock_class"].fillna("Unknown")
        counts = fc.value_counts()
        if counts.min() >= 2:
            stratify_base = fc

    for seed in RANDOM_SEEDS:
        train_idx, test_idx = train_test_split(
            idx,
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=stratify_base.to_numpy() if stratify_base is not None else None,
        )

        y_train = y[train_idx]
        y_test = y[test_idx]

        for feature_variant, cfg in FEATURE_VARIANTS.items():
            numeric_features = cfg["numeric"]
            categorical_features = cfg["categorical"]
            features = numeric_features + categorical_features

            X = df[features].copy()
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            for target_mode in TARGET_MODES:
                pipe = fit_pipeline(
                    X_train=X_train,
                    y_train_raw=y_train,
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    target_mode=target_mode,
                    seed=seed,
                )

                pred = predict_pipeline(pipe, X_test, target_mode)
                metrics = evaluate_prediction(y_test, pred, target_mode)

                rows.append({
                    "split_name": "random_row",
                    "split_id": f"RANDOM_seed_{seed:03d}",
                    "seed": seed,
                    "feature_variant": feature_variant,
                    "target_mode": target_mode,
                    "target_mode_label": TARGET_MODE_LABELS[target_mode],
                    "n_rows_total": len(df),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_numeric_features": len(numeric_features),
                    "n_categorical_features": len(categorical_features),
                    "n_total_original_features": len(features),
                    "numeric_features": "; ".join(numeric_features),
                    "categorical_features": "; ".join(categorical_features),
                    **metrics,
                })

    return pd.DataFrame(rows)


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
            pred_sum_mean=("pred_sum_mean", "mean"),
            pred_sum_sd_mean=("pred_sum_sd", "mean"),
            pct_pred_sum_outside_95_105_mean=("pct_pred_sum_outside_95_105", "mean"),
            n_splits=("seed", "nunique"),
        )
    )

    summary["rank_within_target_mode"] = (
        summary
        .groupby(["split_name", "target_mode"])["multi_R2_mean"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    return summary.sort_values(
        ["split_name", "target_mode", "rank_within_target_mode"]
    ).reset_index(drop=True)


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

    return out.sort_values(
        ["split_name", "target_mode", "rank_within_target_mode"]
    ).reset_index(drop=True)


def build_decision_table(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []

    focus = delta[
        (delta["split_name"].eq("random_row"))
        & (delta["target_mode"].eq("direct_forced_closed"))
    ].copy()

    for _, r in focus.iterrows():
        variant = r["feature_variant"]
        r2 = r["multi_R2_mean"]
        delta_r2 = r["delta_multi_R2_vs_full"]
        mae = r["MAE_mean"]
        delta_mae = r["delta_MAE_vs_full"]

        if variant == "Full_reporting_aware_F2":
            interpretation = (
                "Reference model containing thermochemical variables, composition-sum diagnostics, "
                "and reporting descriptors."
            )
        elif variant == "Reporting_aware_without_sum_diagnostics":
            interpretation = (
                "Tests whether proximate/ultimate sum diagnostics are carrying excessive predictive weight "
                "while other reporting descriptors remain available."
            )
        elif variant == "Thermochemical_only_raw":
            interpretation = (
                "Strict clean baseline using only process, proximate, and ultimate composition variables."
            )
        elif variant == "Thermochemical_only_severity":
            interpretation = (
                "Clean thermochemical baseline plus physical severity descriptors derived from temperature "
                "and heating rate."
            )
        else:
            interpretation = ""

        if variant == "Full_reporting_aware_F2":
            comparison = "Reference configuration."
        elif delta_r2 >= -0.05:
            comparison = "Small reduction relative to the full model."
        elif delta_r2 >= -0.15:
            comparison = "Moderate reduction relative to the full model."
        else:
            comparison = "Large reduction relative to the full model."

        rows.append({
            "focus_setting": "random_row + Renormalized",
            "feature_variant": variant,
            "multi_R2_mean": r2,
            "delta_multi_R2_vs_full": delta_r2,
            "MAE_mean": mae,
            "delta_MAE_vs_full": delta_mae,
            "interpretation": interpretation,
            "comparison_to_full": comparison,
        })

    return pd.DataFrame(rows)


def build_config_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "input_workbook": str(resolve_input_workbook()),
        "input_sheet": INPUT_SHEET,
        "output_workbook": str(OUTPUT_WORKBOOK),
        "rows_after_check": len(df),
        "studies": df["study_id"].nunique() if "study_id" in df.columns else np.nan,
        "expected_rows": EXPECTED_ROWS,
        "expected_studies": EXPECTED_STUDIES,
        "random_seeds": ", ".join(map(str, RANDOM_SEEDS)),
        "test_size": TEST_SIZE,
        "target_modes": "; ".join(TARGET_MODES),
        "main_question": (
            "Does model performance remain acceptable when reporting-context descriptors are removed?"
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
    print("RANDOM-ROW FEATURE-GROUP ABLATION")
    print("====================================================")

    df = load_data()

    print(f"\nRows after check: {len(df)}")
    print(f"Studies: {df['study_id'].nunique() if 'study_id' in df.columns else 'NA'}")
    print(f"Feature variants: {len(FEATURE_VARIANTS)}")
    print(f"Target modes: {', '.join(TARGET_MODES)}")
    print(f"Repeated random-row seeds: {RANDOM_SEEDS}")

    print("\nRunning random-row ablation...")
    perf_long = run_random_row_ablation(df)

    print("Summarizing results...")
    perf_summary = summarize_performance(perf_long)
    delta_vs_full = build_delta_vs_full(perf_summary)
    decision = build_decision_table(delta_vs_full)

    config = build_config_table(df)
    feature_variants = build_feature_variant_table()

    print("\nSaving workbook...")
    with pd.ExcelWriter(OUTPUT_WORKBOOK, engine="openpyxl") as writer:
        config.to_excel(writer, sheet_name="00_Config", index=False)
        feature_variants.to_excel(writer, sheet_name="01_Feature_Variants", index=False)
        perf_long.to_excel(writer, sheet_name="10_Performance_Long", index=False)
        perf_summary.to_excel(writer, sheet_name="11_Performance_Summary", index=False)
        delta_vs_full.to_excel(writer, sheet_name="12_Delta_vs_Full", index=False)
        decision.to_excel(writer, sheet_name="13_Ablation_Decision", index=False)

    print("\n====================================================")
    print("DONE")
    print("====================================================")
    print(f"Saved to: {OUTPUT_WORKBOOK.resolve()}")

    print("\nMain decision table: random-row + Renormalized")
    cols = [
        "feature_variant",
        "multi_R2_mean",
        "delta_multi_R2_vs_full",
        "MAE_mean",
        "delta_MAE_vs_full",
        "comparison_to_full",
    ]
    print(decision[cols].to_string(index=False))

if __name__ == "__main__":
    main()
