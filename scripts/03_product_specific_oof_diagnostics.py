"""Product-specific five-fold out-of-fold diagnostics reported in Figure 2.

Separate XGBoost regressors predict char, oil/liquid, and gas. Their predictions
are recombined and renormalized to 100 wt.% within each held-out fold.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from project_config import DATA_FILE, DATA_SHEET, output_file

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    def root_mean_squared_error(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError("xgboost is required. Install it with: pip install xgboost") from exc

warnings.filterwarnings("ignore", category=UserWarning)


# Analysis settings
INPUT_WORKBOOK = DATA_FILE
INPUT_SHEET = DATA_SHEET
OUTPUT_WORKBOOK = output_file("03_product_specific_oof_diagnostics_results.xlsx")

EXPECTED_ROWS = 442
EXPECTED_STUDIES = 42
N_SPLITS = 5
CV_RANDOM_STATE = 42
USE_FEEDSTOCK_CLASS_STRATIFICATION = True

TARGET_COLS = [
    "char_yield_wt_pct",
    "oil_yield_wt_pct",
    "gas_yield_wt_pct",
]
TARGET_LABELS = ["Char", "Oil/liquid", "Gas"]
TARGET_KEYS = ["char", "oil_liquid", "gas"]

NUMERIC_FEATURES_BASE = [
    "temperature_c",
    "heating_rate_c_min",
    "volatile_matter_wt_pct",
    "fixed_carbon_wt_pct",
    "ash_wt_pct",
    "carbon_wt_pct",
    "hydrogen_wt_pct",
    "nitrogen_wt_pct",
    "oxygen_wt_pct",
    "proximate_sum_VM_FC_Ash",
    "ultimate_CHNO_sum",
]

CATEGORICAL_FEATURES_BASE = [
    "feedstock_class",
    "proximate_basis_std",
    "ultimate_basis_std",
    "yield_basis_std",
    "oil_definition_v3",
    "gas_yield_method_v3",
    "pyrolysis_type_v3",
    "proximate_sum_95_105_flag",
    "ultimate_CHNO_sum_85_105_flag",
    "proximate_basis_known_flag",
    "ultimate_basis_known_flag",
    "composition_basis_pair",
    "yield_reporting_profile",
]

SEVERITY_FEATURES = [
    "inv_temperature_K",
    "log_heating_rate",
    "temp_x_log_heating_rate",
]

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    reg_lambda=1.0,
    n_jobs=-1,
)

PRODUCT_MODEL_SEEDS = {
    "Char": 42,
    "Oil/liquid": 43,
    "Gas": 44,
}


# -----------------------------------------------------------------------------
# 2. Helpers
# -----------------------------------------------------------------------------
def resolve_input_workbook() -> Path:
    if INPUT_WORKBOOK.exists():
        return INPUT_WORKBOOK

    raise FileNotFoundError(
        f"Could not find the model-ready dataset at {INPUT_WORKBOOK}. "
        "Set PYROLYSIS_DATA_FILE to override the default repository path."
    )


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def to_numeric_clean(series: pd.Series) -> pd.Series:
    s = series.copy()
    s = s.replace(["NS", "ND", "NaN", "nan", "", " ", "not reported"], np.nan)

    def convert_one(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float, np.number)):
            return float(value)

        text = str(value).strip()
        if "–" in text or "-" in text:
            separator = "–" if "–" in text else "-"
            numbers = []
            for part in text.split(separator):
                try:
                    numbers.append(float(part.strip()))
                except Exception:
                    pass
            if len(numbers) == 2:
                return float(np.mean(numbers))

        try:
            return float(text)
        except Exception:
            return np.nan

    return s.apply(convert_one)


def close_to_100(y: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = np.clip(y, eps, None)
    sums = y.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("At least one predicted yield vector has a non-positive sum.")
    return y / sums * 100.0


def build_single_output_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
) -> Pipeline:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot_encoder()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )

    model = XGBRegressor(random_state=random_state, **XGB_PARAMS)

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


def product_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE_wt_pct": float(root_mean_squared_error(y_true, y_pred)),
        "MAE_wt_pct": float(mean_absolute_error(y_true, y_pred)),
    }


def make_cv_splits(df: pd.DataFrame):
    indices = np.arange(len(df))

    if USE_FEEDSTOCK_CLASS_STRATIFICATION and "feedstock_class" in df.columns:
        labels = df["feedstock_class"].fillna("Unknown").astype(str)
        smallest_class = int(labels.value_counts().min())
        if smallest_class >= N_SPLITS:
            splitter = StratifiedKFold(
                n_splits=N_SPLITS,
                shuffle=True,
                random_state=CV_RANDOM_STATE,
            )
            print("CV design: shuffled StratifiedKFold using feedstock_class")
            return list(splitter.split(indices, labels.to_numpy()))

        print(
            "WARNING: a feedstock class has fewer rows than N_SPLITS; "
            "falling back to shuffled KFold."
        )

    splitter = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )
    print("CV design: shuffled KFold")
    return list(splitter.split(indices))


# -----------------------------------------------------------------------------
# 3. Load and prepare the model-ready dataset
# -----------------------------------------------------------------------------
def main() -> None:
    input_workbook = resolve_input_workbook()
    df = pd.read_excel(input_workbook, sheet_name=INPUT_SHEET)
    df.columns = df.columns.astype(str).str.strip()

    needed_columns = (
        ["source_excel_row", "study_id"]
        + TARGET_COLS
        + NUMERIC_FEATURES_BASE
        + CATEGORICAL_FEATURES_BASE
    )
    missing_columns = [column for column in needed_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns in {INPUT_SHEET}: {missing_columns}")

    for column in TARGET_COLS + NUMERIC_FEATURES_BASE:
        df[column] = to_numeric_clean(df[column])

    # Other predictors are imputed within each training fold.
    df = df.dropna(
        subset=TARGET_COLS + ["temperature_c", "heating_rate_c_min"]
    ).copy().reset_index(drop=True)

    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} modelling rows, but retained {len(df)}.")

    n_studies = int(df["study_id"].nunique())
    if n_studies != EXPECTED_STUDIES:
        raise ValueError(f"Expected {EXPECTED_STUDIES} studies, but found {n_studies}.")

    # Severity descriptors
    temperature_k = df["temperature_c"] + 273.15
    heating_rate = df["heating_rate_c_min"].clip(lower=0)
    df["inv_temperature_K"] = 1.0 / temperature_k
    df["log_heating_rate"] = np.log1p(heating_rate)
    df["temp_x_log_heating_rate"] = df["temperature_c"] * df["log_heating_rate"]

    numeric_features = NUMERIC_FEATURES_BASE + SEVERITY_FEATURES
    categorical_features = CATEGORICAL_FEATURES_BASE

    for column in categorical_features:
        df[column] = (
            df[column]
            .replace(["", " ", "nan", "NaN", "None"], np.nan)
            .astype("object")
        )

    X = df[numeric_features + categorical_features].copy()
    y_raw = df[TARGET_COLS].to_numpy(dtype=float)
    y_closed = close_to_100(y_raw)
    source_rows = df["source_excel_row"].to_numpy()

    print(f"Input workbook: {input_workbook.resolve()}")
    print(f"Rows: {len(df)}")
    print(f"Studies: {n_studies}")
    print(f"Numerical predictors: {len(numeric_features)}")
    print(f"Categorical predictors before one-hot encoding: {len(categorical_features)}")


    # -----------------------------------------------------------------------------
    # 4. Five-fold out-of-fold product-specific prediction
    # -----------------------------------------------------------------------------
    splits = make_cv_splits(df)

    n_rows = len(df)
    y_pred_raw_oof = np.full((n_rows, 3), np.nan, dtype=float)
    fold_assignment = np.full(n_rows, -1, dtype=int)

    fold_metric_rows: list[dict] = []
    split_audit_rows: list[dict] = []

    for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train_raw = y_raw[train_idx]

        fold_raw_predictions = np.zeros((len(test_idx), 3), dtype=float)

        for product_index, product_label in enumerate(TARGET_LABELS):
            pipeline = build_single_output_pipeline(
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                random_state=PRODUCT_MODEL_SEEDS[product_label],
            )
            pipeline.fit(X_train, y_train_raw[:, product_index])
            fold_raw_predictions[:, product_index] = pipeline.predict(X_test)

        fold_raw_predictions = np.clip(fold_raw_predictions, 1e-9, None)
        fold_closed_predictions = close_to_100(fold_raw_predictions)

        y_pred_raw_oof[test_idx, :] = fold_raw_predictions
        fold_assignment[test_idx] = fold_number

        for product_index, (product_label, product_key) in enumerate(
            zip(TARGET_LABELS, TARGET_KEYS)
        ):
            metrics = product_metrics(
                y_closed[test_idx, product_index],
                fold_closed_predictions[:, product_index],
            )
            fold_metric_rows.append({
                "fold": fold_number,
                "product_target": product_label,
                "product_key": product_key,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                **metrics,
            })

        train_classes = df.iloc[train_idx]["feedstock_class"].value_counts().to_dict()
        test_classes = df.iloc[test_idx]["feedstock_class"].value_counts().to_dict()
        split_audit_rows.append({
            "fold": fold_number,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_studies": int(df.iloc[train_idx]["study_id"].nunique()),
            "test_studies": int(df.iloc[test_idx]["study_id"].nunique()),
            "train_feedstock_class_counts": "; ".join(f"{k}:{v}" for k, v in sorted(train_classes.items())),
            "test_feedstock_class_counts": "; ".join(f"{k}:{v}" for k, v in sorted(test_classes.items())),
        })

        print(f"Completed fold {fold_number}/{N_SPLITS}: train={len(train_idx)}, test={len(test_idx)}")

    if np.isnan(y_pred_raw_oof).any() or np.any(fold_assignment < 1):
        raise RuntimeError("OOF prediction failed: at least one row was not predicted exactly once.")

    if len(np.unique(np.where(~np.isnan(y_pred_raw_oof[:, 0]))[0])) != n_rows:
        raise RuntimeError("OOF audit failed: not all rows received a prediction.")

    y_pred_closed_oof = close_to_100(y_pred_raw_oof)


    # -----------------------------------------------------------------------------
    # 5. Pooled all-row OOF metrics and outputs
    # -----------------------------------------------------------------------------
    pooled_metric_rows: list[dict] = []
    for product_index, (product_label, product_key) in enumerate(
        zip(TARGET_LABELS, TARGET_KEYS)
    ):
        metrics = product_metrics(
            y_closed[:, product_index],
            y_pred_closed_oof[:, product_index],
        )
        pooled_metric_rows.append({
            "product_target": product_label,
            "product_key": product_key,
            "model_type": "three independent single-output XGBoost regressors",
            "validation_design": "5-fold out-of-fold prediction",
            "prediction_basis": "post-prediction Renormalized closure",
            "evaluation_basis": "closed-normalized observed yield",
            "n_oof_rows": n_rows,
            **metrics,
        })

    pooled_metrics = pd.DataFrame(pooled_metric_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)

    fold_summary = (
        fold_metrics
        .groupby(["product_target", "product_key"], as_index=False)
        .agg(
            R2_fold_mean=("R2", "mean"),
            R2_fold_sd=("R2", "std"),
            RMSE_fold_mean=("RMSE_wt_pct", "mean"),
            RMSE_fold_sd=("RMSE_wt_pct", "std"),
            MAE_fold_mean=("MAE_wt_pct", "mean"),
            MAE_fold_sd=("MAE_wt_pct", "std"),
            n_folds=("fold", "nunique"),
        )
    )

    predictions_wide = pd.DataFrame({
        "row_uid": np.arange(n_rows),
        "source_excel_row": source_rows,
        "study_id": df["study_id"].astype(str).to_numpy(),
        "feedstock_class": df["feedstock_class"].astype(str).to_numpy(),
        "oof_fold": fold_assignment,
        "observed_char_closed": y_closed[:, 0],
        "predicted_char_closed": y_pred_closed_oof[:, 0],
        "observed_oil_liquid_closed": y_closed[:, 1],
        "predicted_oil_liquid_closed": y_pred_closed_oof[:, 1],
        "observed_gas_closed": y_closed[:, 2],
        "predicted_gas_closed": y_pred_closed_oof[:, 2],
        "raw_predicted_sum_before_closure": y_pred_raw_oof.sum(axis=1),
        "closed_predicted_sum_after_closure": y_pred_closed_oof.sum(axis=1),
    })

    prediction_long_rows: list[dict] = []
    plot_specs = [
        ("Char", "observed_char_closed", "predicted_char_closed"),
        ("Oil/liquid", "observed_oil_liquid_closed", "predicted_oil_liquid_closed"),
        ("Gas", "observed_gas_closed", "predicted_gas_closed"),
    ]
    for _, row in predictions_wide.iterrows():
        for product_label, observed_column, predicted_column in plot_specs:
            prediction_long_rows.append({
                "row_uid": int(row["row_uid"]),
                "source_excel_row": row["source_excel_row"],
                "study_id": row["study_id"],
                "feedstock_class": row["feedstock_class"],
                "oof_fold": int(row["oof_fold"]),
                "product_target": product_label,
                "observed_yield_wt_pct": row[observed_column],
                "predicted_yield_wt_pct": row[predicted_column],
            })
    predictions_long = pd.DataFrame(prediction_long_rows)

    config = pd.DataFrame([{
        "input_workbook": str(input_workbook),
        "input_sheet": INPUT_SHEET,
        "output_workbook": str(OUTPUT_WORKBOOK),
        "rows": n_rows,
        "studies": n_studies,
        "validation_design": "5-fold out-of-fold",
        "n_splits": N_SPLITS,
        "cv_random_state": CV_RANDOM_STATE,
        "stratification": "feedstock_class" if USE_FEEDSTOCK_CLASS_STRATIFICATION else "none",
        "numeric_features": "; ".join(numeric_features),
        "categorical_features": "; ".join(categorical_features),
        "xgb_parameters": str(XGB_PARAMS),
        "product_model_seeds": str(PRODUCT_MODEL_SEEDS),
    }])

    model_audit = pd.DataFrame([
        {
            "product_target": product_label,
            "training_target": TARGET_COLS[index],
            "model": "XGBoost single-output",
            "model_random_state": PRODUCT_MODEL_SEEDS[product_label],
            "closure_step": "three raw product predictions combined and renormalized to 100 wt.%",
        }
        for index, product_label in enumerate(TARGET_LABELS)
    ])

    split_audit = pd.DataFrame(split_audit_rows)

    with pd.ExcelWriter(OUTPUT_WORKBOOK, engine="openpyxl") as writer:
        config.to_excel(writer, sheet_name="00_Config", index=False)
        pooled_metrics.round(6).to_excel(writer, sheet_name="10_ProductMetrics_Pooled", index=False)
        fold_metrics.round(6).to_excel(writer, sheet_name="11_ProductMetrics_ByFold", index=False)
        fold_summary.round(6).to_excel(writer, sheet_name="12_FoldSummary_MeanSD", index=False)
        predictions_wide.round(6).to_excel(writer, sheet_name="20_OOF_Predictions_Wide", index=False)
        predictions_long.round(6).to_excel(writer, sheet_name="21_OOF_Predictions_Long", index=False)
        split_audit.to_excel(writer, sheet_name="30_SplitAudit", index=False)
        model_audit.to_excel(writer, sheet_name="31_ModelAudit", index=False)

    print("\nSaved analysis workbook:")
    print(OUTPUT_WORKBOOK.resolve())
    print("\nPooled all-row out-of-fold product metrics:")
    print(pooled_metrics[["product_target", "R2", "RMSE_wt_pct", "MAE_wt_pct", "n_oof_rows"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
