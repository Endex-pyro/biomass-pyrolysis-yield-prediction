"""Create study-level LOSO diagnostics and computational-environment summaries.

This post-processing step reads the validation-ladder workbook, derives WAPE
and relative-error summaries, and produces a compact LOSO fold audit.
"""

from __future__ import annotations

import os
import sys
import platform
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from project_config import output_file


INPUT_STEP4_CANDIDATES = [
    output_file("04_validation_ladder_results.xlsx"),
]

OUTPUT_WORKBOOK = output_file("10_loso_audit_and_environment_results.xlsx")

SELECTED_MODEL = "XGBoost_base"
SELECTED_TARGET_MODE = "direct_forced_closed"
SELECTED_TARGET_LABEL = "Renormalized"

# Worksheet names in the reference validation output.
SHEET_MODEL_MATRIX = "30_Model_Matrix_442"
SHEET_S0_S1_SUMMARY = "37_All_Validation_Summary"
SHEET_LOSO_POOLED = "35_S2_LOSO_PooledSummary"
SHEET_LOSO_STUDY_BALANCED = "36_S2_LOSO_StudyBalanced"
SHEET_LOSO_FOLD_DISTRIBUTION = "40_S2_LOSO_FoldDistribution"
SHEET_LOSO_COMPACT = "41_LOSO_Compact"
SHEET_SELECTED_XGB_FOLDS = "42_Selected_XGB_LOSO_Folds"


# ============================================================
# 2. INPUT HELPERS
# ============================================================

def resolve_step4_workbook() -> Path:
    for p in INPUT_STEP4_CANDIDATES:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Could not find {INPUT_STEP4_CANDIDATES[0]}. Run "
        "04_validation_ladder.py before this post-processing step."
    )


def read_sheet_safe(xlsx: Path, sheet: str) -> pd.DataFrame:
    try:
        return pd.read_excel(xlsx, sheet_name=sheet)
    except Exception as exc:
        print(f"WARNING: Could not read sheet '{sheet}': {exc}")
        return pd.DataFrame()


def to_numeric_safe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ============================================================
# 3. RELATIVE METRICS
# ============================================================

def add_closed_relative_metrics_to_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds WAPE-style relative metrics to summary tables.

    For closed char-oil-gas distributions:
      each row sums to 100 wt.%
      sklearn-style multioutput MAE = total_abs_error / (n * 3)
      WAPE% = total_abs_error / (n * 100) * 100
            = multioutput_MAE * 3

    Therefore:
      closed_WAPE_pct = closed_true_MAE * 3

    This is more stable than MAPE for low-yield product fractions.
    """
    if df.empty:
        return df

    out = df.copy()

    # Repeated validation summary columns usually have "_mean".
    if "closed_true_MAE_mean" in out.columns:
        out["closed_WAPE_pct_mean"] = out["closed_true_MAE_mean"] * 3.0

    if "closed_true_MAE_sd" in out.columns:
        out["closed_WAPE_pct_sd"] = out["closed_true_MAE_sd"] * 3.0

    if "closed_true_RMSE_mean" in out.columns:
        out["closed_RMSE_pct_points_mean"] = out["closed_true_RMSE_mean"]

    if "closed_true_MAE_mean" in out.columns:
        out["closed_MAE_pct_points_mean"] = out["closed_true_MAE_mean"]

    # Pooled LOSO columns usually do not have "_mean".
    if "closed_true_MAE" in out.columns:
        out["closed_WAPE_pct"] = out["closed_true_MAE"] * 3.0

    if "closed_true_RMSE" in out.columns:
        out["closed_RMSE_pct_points"] = out["closed_true_RMSE"]

    if "closed_true_MAE" in out.columns:
        out["closed_MAE_pct_points"] = out["closed_true_MAE"]

    # Compact LOSO sheet uses renamed columns.
    if "pooled_closed_MAE" in out.columns:
        out["pooled_closed_WAPE_pct"] = out["pooled_closed_MAE"] * 3.0

    if "pooled_closed_RMSE" in out.columns:
        out["pooled_closed_RMSE_pct_points"] = out["pooled_closed_RMSE"]

    if "pooled_closed_MAE" in out.columns:
        out["pooled_closed_MAE_pct_points"] = out["pooled_closed_MAE"]

    return out


def build_final_relative_metrics_table(
    s0s1_summary: pd.DataFrame,
    loso_pooled: pd.DataFrame,
    loso_compact: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one compact table for final manuscript/SI.
    """
    rows = []

    if not s0s1_summary.empty:
        s = add_closed_relative_metrics_to_summary(s0s1_summary)

        for _, r in s.iterrows():
            rows.append({
                "validation": r.get("validation"),
                "model": r.get("model"),
                "target_mode": r.get("target_mode"),
                "target_label": r.get("target_label"),
                "R2_closed": r.get("closed_true_multi_R2_mean", r.get("ranking_R2_mean")),
                "R2_closed_sd": r.get("closed_true_multi_R2_sd", r.get("ranking_R2_sd")),
                "RMSE_wt_pct": r.get("closed_true_RMSE_mean"),
                "RMSE_wt_pct_sd": r.get("closed_true_RMSE_sd"),
                "MAE_wt_pct": r.get("closed_true_MAE_mean"),
                "MAE_wt_pct_sd": r.get("closed_true_MAE_sd"),
                "WAPE_pct": r.get("closed_WAPE_pct_mean"),
                "WAPE_pct_sd": r.get("closed_WAPE_pct_sd"),
                "pred_sum_mean": r.get("pred_sum_mean_mean"),
                "pred_sum_sd": r.get("pred_sum_sd_mean"),
                "outside_95_105_pct": r.get("pct_pred_sum_outside_95_105_mean"),
                "n_test_mean": r.get("n_test_mean"),
                "source_sheet": SHEET_S0_S1_SUMMARY,
            })

    if not loso_pooled.empty:
        s = add_closed_relative_metrics_to_summary(loso_pooled)

        for _, r in s.iterrows():
            rows.append({
                "validation": r.get("validation"),
                "model": r.get("model"),
                "target_mode": r.get("target_mode"),
                "target_label": r.get("target_label"),
                "R2_closed": r.get("closed_true_multi_R2", r.get("ranking_R2")),
                "R2_closed_sd": np.nan,
                "RMSE_wt_pct": r.get("closed_true_RMSE"),
                "RMSE_wt_pct_sd": np.nan,
                "MAE_wt_pct": r.get("closed_true_MAE"),
                "MAE_wt_pct_sd": np.nan,
                "WAPE_pct": r.get("closed_WAPE_pct"),
                "WAPE_pct_sd": np.nan,
                "pred_sum_mean": r.get("pred_sum_mean"),
                "pred_sum_sd": r.get("pred_sum_sd"),
                "outside_95_105_pct": r.get("pct_pred_sum_outside_95_105"),
                "n_test_mean": r.get("n_test"),
                "source_sheet": SHEET_LOSO_POOLED,
            })

    out = pd.DataFrame(rows)

    if not out.empty:
        out["selected_reference_pathway"] = (
            out["model"].eq(SELECTED_MODEL)
            & out["target_mode"].eq(SELECTED_TARGET_MODE)
        )

        preferred_order = {
            "S0_random_row_repeated": 1,
            "S1_grouped_validation": 2,
            "S2_LOSO_by_study_id_pooled": 3,
        }
        out["validation_order"] = out["validation"].map(preferred_order).fillna(99)
        out = out.sort_values(
            ["validation_order", "selected_reference_pathway", "R2_closed"],
            ascending=[True, False, False],
        ).drop(columns=["validation_order"]).reset_index(drop=True)

    return out


def build_selected_reference_relative_table(final_relative: pd.DataFrame) -> pd.DataFrame:
    if final_relative.empty:
        return pd.DataFrame()

    out = final_relative[
        final_relative["selected_reference_pathway"].eq(True)
    ].copy()

    keep = [
        "validation",
        "model",
        "target_label",
        "R2_closed",
        "R2_closed_sd",
        "RMSE_wt_pct",
        "RMSE_wt_pct_sd",
        "MAE_wt_pct",
        "MAE_wt_pct_sd",
        "WAPE_pct",
        "WAPE_pct_sd",
        "pred_sum_mean",
        "pred_sum_sd",
        "outside_95_105_pct",
        "n_test_mean",
    ]

    keep = [c for c in keep if c in out.columns]
    return out[keep].reset_index(drop=True)


# ============================================================
# 4. LOSO AUDIT TABLES
# ============================================================

def classify_loso_fold(row: pd.Series) -> str:
    n_test = row.get("n_test", np.nan)
    r2 = row.get("ranking_R2", np.nan)
    mae = row.get("closed_true_MAE", np.nan)

    if pd.notna(n_test) and n_test < 3:
        return "unstable_few_rows"
    if pd.notna(r2) and r2 < 0:
        return "negative_R2_transfer_failure"
    if pd.notna(r2) and r2 >= 0.5:
        return "good_transfer"
    if pd.notna(r2) and r2 >= 0:
        return "limited_but_positive_transfer"
    if pd.notna(mae) and mae > 10:
        return "high_error_fold"
    return "review_manually"


def build_clean_selected_loso_audit(selected_folds: pd.DataFrame) -> pd.DataFrame:
    if selected_folds.empty:
        return pd.DataFrame()

    df = selected_folds.copy()

    numeric_cols = [
        "n_test",
        "ranking_R2",
        "closed_true_multi_R2",
        "closed_true_char_R2",
        "closed_true_oil_R2",
        "closed_true_gas_R2",
        "closed_true_RMSE",
        "closed_true_MAE",
        "orig_multi_R2",
        "orig_RMSE",
        "orig_MAE",
        "pred_sum_mean",
        "pred_sum_sd",
        "pct_pred_sum_outside_95_105",
    ]
    df = to_numeric_safe(df, numeric_cols)

    df["fold_audit_class"] = df.apply(classify_loso_fold, axis=1)

    # Directional diagnostic: which target had weakest fold-level R²?
    target_r2_cols = {
        "char": "closed_true_char_R2",
        "oil_liquid": "closed_true_oil_R2",
        "gas": "closed_true_gas_R2",
    }

    weakest = []
    for _, r in df.iterrows():
        vals = {}
        for label, col in target_r2_cols.items():
            if col in df.columns:
                vals[label] = r.get(col, np.nan)
        vals = {k: v for k, v in vals.items() if pd.notna(v)}
        weakest.append(min(vals, key=vals.get) if vals else np.nan)

    df["weakest_target_by_R2"] = weakest

    # Sort worst/highest-error first for audit.
    sort_cols = []
    ascending = []

    if "closed_true_MAE" in df.columns:
        sort_cols.append("closed_true_MAE")
        ascending.append(False)
    if "ranking_R2" in df.columns:
        sort_cols.append("ranking_R2")
        ascending.append(True)

    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    return df


def build_loso_worst_best_tables(clean_loso: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if clean_loso.empty:
        return pd.DataFrame(), pd.DataFrame()

    worst = clean_loso.copy()
    if "closed_true_MAE" in worst.columns:
        worst = worst.sort_values("closed_true_MAE", ascending=False).head(10)

    best = clean_loso.copy()
    if "ranking_R2" in best.columns:
        best = best.sort_values("ranking_R2", ascending=False).head(10)

    return worst.reset_index(drop=True), best.reset_index(drop=True)


def build_loso_audit_summary(clean_loso: pd.DataFrame) -> pd.DataFrame:
    if clean_loso.empty:
        return pd.DataFrame()

    rows = []

    rows.append({
        "audit_item": "n_loso_folds",
        "value": clean_loso["heldout_study_id"].nunique()
        if "heldout_study_id" in clean_loso.columns else len(clean_loso),
    })

    if "n_test" in clean_loso.columns:
        rows.extend([
            {"audit_item": "heldout_rows_min", "value": clean_loso["n_test"].min()},
            {"audit_item": "heldout_rows_median", "value": clean_loso["n_test"].median()},
            {"audit_item": "heldout_rows_max", "value": clean_loso["n_test"].max()},
        ])

    for col in ["ranking_R2", "closed_true_RMSE", "closed_true_MAE"]:
        if col in clean_loso.columns:
            vals = pd.to_numeric(clean_loso[col], errors="coerce").dropna()
            rows.extend([
                {"audit_item": f"{col}_mean", "value": vals.mean()},
                {"audit_item": f"{col}_median", "value": vals.median()},
                {"audit_item": f"{col}_q10", "value": vals.quantile(0.10)},
                {"audit_item": f"{col}_q90", "value": vals.quantile(0.90)},
            ])

    if "ranking_R2" in clean_loso.columns:
        vals = pd.to_numeric(clean_loso["ranking_R2"], errors="coerce").dropna()
        rows.append({
            "audit_item": "negative_R2_folds_n",
            "value": int((vals < 0).sum()),
        })
        rows.append({
            "audit_item": "negative_R2_folds_pct",
            "value": float((vals < 0).mean() * 100.0),
        })

    if "fold_audit_class" in clean_loso.columns:
        counts = clean_loso["fold_audit_class"].value_counts(dropna=False)
        for k, v in counts.items():
            rows.append({
                "audit_item": f"fold_class_count__{k}",
                "value": int(v),
            })

    return pd.DataFrame(rows)


def build_loso_response_points(
    loso_compact: pd.DataFrame,
    clean_loso_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Creates short machine-readable talking points for the response letter.
    """
    rows = []

    selected = loso_compact[
        loso_compact["model"].eq(SELECTED_MODEL)
        & loso_compact["target_mode"].eq(SELECTED_TARGET_MODE)
    ].copy() if not loso_compact.empty else pd.DataFrame()

    if not selected.empty:
        r = selected.iloc[0]
        rows.append({
            "point": "pooled_LOSO_performance",
            "text": (
                f"For the selected {SELECTED_MODEL} {SELECTED_TARGET_LABEL} pathway, "
                f"pooled LOSO performance was R2={r.get('pooled_closed_R2', np.nan):.3f}, "
                f"RMSE={r.get('pooled_closed_RMSE', np.nan):.2f} wt.%, "
                f"MAE={r.get('pooled_closed_MAE', np.nan):.2f} wt.%."
            ),
        })

        rows.append({
            "point": "fold_distribution",
            "text": (
                f"The LOSO fold distribution was heterogeneous: median fold R2="
                f"{r.get('ranking_R2_fold_median', np.nan):.3f}, with 10th–90th percentile "
                f"{r.get('ranking_R2_fold_q10', np.nan):.3f} to "
                f"{r.get('ranking_R2_fold_q90', np.nan):.3f}."
            ),
        })

        rows.append({
            "point": "negative_folds",
            "text": (
                f"Negative-R2 folds accounted for "
                f"{r.get('ranking_R2_negative_folds_pct', np.nan):.1f}% of LOSO folds, "
                f"showing that external-study transfer remained unstable for several held-out studies."
            ),
        })

    rows.append({
        "point": "recommended_framing",
        "text": (
            "Use LOSO as an external-study stress test, not as evidence that the model is ready "
            "for blind engineering deployment. The result supports cautious wording: strong "
            "within-domain interpolation, weaker cross-study transfer."
        ),
    })

    return pd.DataFrame(rows)


# ============================================================
# 5. COMPUTATIONAL ENVIRONMENT
# ============================================================

def get_package_version(package_name: str) -> str:
    try:
        module = __import__(package_name)
        return getattr(module, "__version__", "version_attribute_missing")
    except Exception:
        return "not_installed_or_not_importable"


def get_pip_freeze_subset(packages: list[str]) -> pd.DataFrame:
    rows = []
    for pkg in packages:
        rows.append({
            "package": pkg,
            "version": get_package_version(pkg),
        })
    return pd.DataFrame(rows)


def get_hardware_info() -> pd.DataFrame:
    rows = [
        {"item": "timestamp_local", "value": datetime.now().isoformat(timespec="seconds")},
        {"item": "python_version", "value": sys.version.replace("\n", " ")},
        {"item": "python_executable", "value": sys.executable},
        {"item": "platform", "value": platform.platform()},
        {"item": "processor", "value": platform.processor()},
        {"item": "machine", "value": platform.machine()},
        {"item": "system", "value": platform.system()},
        {"item": "release", "value": platform.release()},
        {"item": "cpu_count_os", "value": os.cpu_count()},
    ]

    try:
        import psutil
        vm = psutil.virtual_memory()
        rows.extend([
            {"item": "ram_total_gb", "value": round(vm.total / (1024 ** 3), 2)},
            {"item": "ram_available_gb", "value": round(vm.available / (1024 ** 3), 2)},
        ])
    except Exception:
        rows.append({"item": "ram_info", "value": "psutil_not_available"})

    return pd.DataFrame(rows)


def build_environment_tables(input_workbook: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hardware = get_hardware_info()

    packages = [
        "numpy",
        "pandas",
        "sklearn",
        "xgboost",
        "shap",
        "matplotlib",
        "scipy",
        "openpyxl",
    ]
    package_versions = get_pip_freeze_subset(packages)

    run_config = pd.DataFrame([{
        "input_step4_workbook": str(input_workbook),
        "output_workbook": str(OUTPUT_WORKBOOK),
        "selected_model": SELECTED_MODEL,
        "selected_target_mode": SELECTED_TARGET_MODE,
        "selected_target_label": SELECTED_TARGET_LABEL,
        "note": (
            "This workbook post-processes Step 4 validation results to create clean LOSO audit, "
            "relative metric, and computational environment tables."
        ),
    }])

    return hardware, package_versions, run_config


# ============================================================
# 6. MAIN
# ============================================================

def main():
    print("\n====================================================")
    print("LOSO AUDIT, RELATIVE METRICS, AND ENVIRONMENT SUMMARY")
    print("====================================================")

    input_wb = resolve_step4_workbook()
    print(f"Input validation-ladder workbook: {input_wb.resolve()}")

    model_matrix = read_sheet_safe(input_wb, SHEET_MODEL_MATRIX)
    s0s1_summary = read_sheet_safe(input_wb, SHEET_S0_S1_SUMMARY)
    loso_pooled = read_sheet_safe(input_wb, SHEET_LOSO_POOLED)
    loso_study_balanced = read_sheet_safe(input_wb, SHEET_LOSO_STUDY_BALANCED)
    loso_fold_distribution = read_sheet_safe(input_wb, SHEET_LOSO_FOLD_DISTRIBUTION)
    loso_compact = read_sheet_safe(input_wb, SHEET_LOSO_COMPACT)
    selected_xgb_folds = read_sheet_safe(input_wb, SHEET_SELECTED_XGB_FOLDS)

    print("\nBuilding relative metrics tables...")
    s0s1_relative = add_closed_relative_metrics_to_summary(s0s1_summary)
    loso_pooled_relative = add_closed_relative_metrics_to_summary(loso_pooled)
    loso_compact_relative = add_closed_relative_metrics_to_summary(loso_compact)

    final_relative = build_final_relative_metrics_table(
        s0s1_summary=s0s1_summary,
        loso_pooled=loso_pooled,
        loso_compact=loso_compact,
    )
    selected_reference_relative = build_selected_reference_relative_table(final_relative)

    print("Building LOSO audit tables...")
    clean_selected_loso = build_clean_selected_loso_audit(selected_xgb_folds)
    loso_worst10, loso_best10 = build_loso_worst_best_tables(clean_selected_loso)
    loso_audit_summary = build_loso_audit_summary(clean_selected_loso)
    loso_response_points = build_loso_response_points(
        loso_compact=loso_compact_relative,
        clean_loso_summary=loso_audit_summary,
    )

    print("Building environment tables...")
    env_hardware, env_packages, env_config = build_environment_tables(input_wb)

    print("\nSaving workbook...")
    with pd.ExcelWriter(OUTPUT_WORKBOOK, engine="openpyxl") as writer:
        env_config.to_excel(writer, sheet_name="00_Run_Config", index=False)
        env_hardware.to_excel(writer, sheet_name="01_Hardware_OS", index=False)
        env_packages.to_excel(writer, sheet_name="02_Package_Versions", index=False)

        final_relative.to_excel(writer, sheet_name="10_Final_Relative_Metrics", index=False)
        selected_reference_relative.to_excel(writer, sheet_name="11_Selected_Pathway_Metrics", index=False)
        s0s1_relative.to_excel(writer, sheet_name="12_S0S1_Relative_All", index=False)
        loso_pooled_relative.to_excel(writer, sheet_name="13_LOSO_Pooled_Relative", index=False)
        loso_compact_relative.to_excel(writer, sheet_name="14_LOSO_Compact_Relative", index=False)

        clean_selected_loso.to_excel(writer, sheet_name="20_Selected_LOSO_Audit", index=False)
        loso_audit_summary.to_excel(writer, sheet_name="21_LOSO_Audit_Summary", index=False)
        loso_worst10.to_excel(writer, sheet_name="22_LOSO_Worst10", index=False)
        loso_best10.to_excel(writer, sheet_name="23_LOSO_Best10", index=False)
        loso_response_points.to_excel(writer, sheet_name="24_LOSO_Response_Points", index=False)

        # Carry forward original key sheets for traceability.
        if not loso_compact.empty:
            loso_compact.to_excel(writer, sheet_name="30_Original_LOSO_Compact", index=False)
        if not loso_fold_distribution.empty:
            loso_fold_distribution.to_excel(writer, sheet_name="31_Original_LOSO_Distrib", index=False)
        if not loso_study_balanced.empty:
            loso_study_balanced.to_excel(writer, sheet_name="32_Original_StudyBalanced", index=False)

        # Optional: keep model-matrix dimensions only, not the full matrix.
        if not model_matrix.empty:
            matrix_audit = pd.DataFrame([{
                "model_matrix_rows": len(model_matrix),
                "model_matrix_columns": model_matrix.shape[1],
                "study_count": model_matrix["study_id"].nunique()
                if "study_id" in model_matrix.columns else np.nan,
                "target_columns_present": "; ".join(
                    [c for c in ["char_yield_wt_pct", "oil_yield_wt_pct", "gas_yield_wt_pct"]
                     if c in model_matrix.columns]
                ),
            }])
            matrix_audit.to_excel(writer, sheet_name="40_Model_Matrix_Audit", index=False)

    print("\n====================================================")
    print("DONE")
    print("====================================================")
    print(f"Saved to: {OUTPUT_WORKBOOK.resolve()}")

    print("\nKey sheets to inspect:")
    print("  10_Final_Relative_Metrics")
    print("  11_Selected_Pathway_Metrics")
    print("  20_Selected_LOSO_Audit")
    print("  21_LOSO_Audit_Summary")
    print("  24_LOSO_Response_Points")
    print("  01_Hardware_OS")
    print("  02_Package_Versions")

    if not selected_reference_relative.empty:
        print("\nSelected pathway metrics:")
        print(selected_reference_relative.to_string(index=False))

    if not loso_response_points.empty:
        print("\nLOSO response points:")
        print(loso_response_points.to_string(index=False))


if __name__ == "__main__":
    main()
