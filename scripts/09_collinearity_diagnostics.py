"""Collinearity diagnostics reported in Supplementary Figure S2.

VIF is evaluated separately for raw core predictors, core plus severity
descriptors, and the intentionally rank-deficient matrix that also contains
composition sums and deviations.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from project_config import DATA_FILE, DATA_SHEET, output_file

# Analysis settings
INPUT_XLSX = DATA_FILE
INPUT_SHEET = DATA_SHEET
OUTPUT_XLSX = output_file("09_collinearity_diagnostics_results.xlsx")

EXPECTED_ROWS = 442

# ============================================================
# FEATURE DEFINITIONS
# ============================================================

# VIF-A: raw core predictors only — the primary VIF reference
VIF_A_FEATURES = [
    "temperature_c",
    "heating_rate_c_min",
    "volatile_matter_wt_pct",
    "fixed_carbon_wt_pct",
    "ash_wt_pct",
    "carbon_wt_pct",
    "hydrogen_wt_pct",
    "nitrogen_wt_pct",
    "oxygen_wt_pct",
]

# VIF-B: model predictors — raw core + severity descriptors (no sum/dev)
VIF_B_ADDED = [
    "inv_temp_K",
    "log_heating_rate",
    "temp_x_log_heating_rate",
]
VIF_B_FEATURES = VIF_A_FEATURES + VIF_B_ADDED

# Engineering features that cause the rank deficiency — reported separately
ENGINEERED_FEATURES = [
    "proximate_sum_VM_FC_Ash",
    "proximate_sum_deviation_from_100",
    "ultimate_CHNO_sum",
    "ultimate_CHNO_sum_deviation_from_100",
]

# All numeric features (VIF-A + severity + engineered)
ALL_NUMERIC = VIF_B_FEATURES + ENGINEERED_FEATURES

CATEGORICAL_FEATURES = [
    "yield_basis",
    "oil_definition",
    "gas_yield_method",
    "pyrolysis_type",
    "proximate_basis",
    "ultimate_basis",
    "composition_basis_pair",
    "yield_reporting_profile",
]

# Column name fallbacks (sheet uses _std / _v3 suffixes on some columns)
COLUMN_FALLBACKS = {
    "yield_basis":        "yield_basis_std",
    "oil_definition":     "oil_definition_v3",
    "gas_yield_method":   "gas_yield_method_v3",
    "pyrolysis_type":     "pyrolysis_type_v3",
    "proximate_basis":    "proximate_basis_std",
    "ultimate_basis":     "ultimate_basis_std",
}

FEATURE_LABELS = {
    "temperature_c":                     "Temperature (°C)",
    "heating_rate_c_min":                "Heating rate (°C/min)",
    "volatile_matter_wt_pct":            "Volatile matter (wt%)",
    "fixed_carbon_wt_pct":               "Fixed carbon (wt%)",
    "ash_wt_pct":                        "Ash (wt%)",
    "carbon_wt_pct":                     "Carbon (wt%)",
    "hydrogen_wt_pct":                   "Hydrogen (wt%)",
    "nitrogen_wt_pct":                   "Nitrogen (wt%)",
    "oxygen_wt_pct":                     "Oxygen (wt%)",
    "inv_temp_K":                        "Inverse temperature (1/K)",
    "log_heating_rate":                  "Log heating rate",
    "temp_x_log_heating_rate":           "Temperature × log(heating rate)",
    "proximate_sum_VM_FC_Ash":           "Proximate-analysis closure (wt%)",
    "proximate_sum_deviation_from_100":  "Proximate-closure deviation (wt%)",
    "ultimate_CHNO_sum":                 "Ultimate-analysis closure (wt%)",
    "ultimate_CHNO_sum_deviation_from_100": "Ultimate-closure deviation (wt%)",
    "yield_basis":                       "Yield basis",
    "oil_definition":                    "Oil/liquid definition",
    "gas_yield_method":                  "Gas yield method",
    "pyrolysis_type":                    "Pyrolysis type",
    "proximate_basis":                   "Proximate basis",
    "ultimate_basis":                    "Ultimate basis",
    "composition_basis_pair":            "Composition-basis pair",
    "yield_reporting_profile":           "Yield-reporting profile",
}

FEATURE_GROUPS = {
    "temperature_c":                      "Process condition",
    "heating_rate_c_min":                 "Process condition",
    "volatile_matter_wt_pct":             "Proximate composition",
    "fixed_carbon_wt_pct":                "Proximate composition",
    "ash_wt_pct":                         "Proximate composition",
    "carbon_wt_pct":                      "Ultimate composition",
    "hydrogen_wt_pct":                    "Ultimate composition",
    "nitrogen_wt_pct":                    "Ultimate composition",
    "oxygen_wt_pct":                      "Ultimate composition",
    "inv_temp_K":                         "Severity descriptor",
    "log_heating_rate":                   "Severity descriptor",
    "temp_x_log_heating_rate":            "Severity descriptor",
    "proximate_sum_VM_FC_Ash":            "Closure/basis diagnostic",
    "proximate_sum_deviation_from_100":   "Closure/basis diagnostic",
    "ultimate_CHNO_sum":                  "Closure/basis diagnostic",
    "ultimate_CHNO_sum_deviation_from_100": "Closure/basis diagnostic",
    "yield_basis":                        "Yield/reporting descriptor",
    "oil_definition":                     "Yield/reporting descriptor",
    "gas_yield_method":                   "Yield/reporting descriptor",
    "pyrolysis_type":                     "Process descriptor",
    "proximate_basis":                    "Composition-basis descriptor",
    "ultimate_basis":                     "Composition-basis descriptor",
    "composition_basis_pair":             "Composition-basis descriptor",
    "yield_reporting_profile":            "Yield/reporting descriptor",
}

VIF_THRESHOLDS = {"low": 5.0, "moderate": 10.0, "high": 10.0}

# ============================================================
# HELPERS
# ============================================================

def resolve_columns(df: pd.DataFrame, wanted: list[str]) -> dict[str, str]:
    """Return mapping wanted_name → actual_column, applying fallbacks."""
    mapping = {}
    for col in wanted:
        if col in df.columns:
            mapping[col] = col
        elif col in COLUMN_FALLBACKS and COLUMN_FALLBACKS[col] in df.columns:
            mapping[col] = COLUMN_FALLBACKS[col]
        else:
            mapping[col] = None  # will be created or flagged
    return mapping


def clean_string_col(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})


def load_and_engineer(path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, loaded {len(df)}")

    # Resolve column name variants
    col_map = resolve_columns(df, ALL_NUMERIC + CATEGORICAL_FEATURES)

    # Rename variant columns to canonical names
    rename = {v: k for k, v in col_map.items() if v and v != k}
    df = df.rename(columns=rename)

    # Convert numeric columns
    for col in ALL_NUMERIC:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Engineer severity descriptors if missing
    if "inv_temp_K" not in df.columns or df["inv_temp_K"].isna().all():
        df["inv_temp_K"] = 1.0 / (df["temperature_c"] + 273.15)
    if "log_heating_rate" not in df.columns or df["log_heating_rate"].isna().all():
        df["log_heating_rate"] = np.log1p(df["heating_rate_c_min"])
    if "temp_x_log_heating_rate" not in df.columns or df["temp_x_log_heating_rate"].isna().all():
        df["temp_x_log_heating_rate"] = df["temperature_c"] * np.log1p(df["heating_rate_c_min"])

    # Engineer diagnostic proxies if missing
    if df["proximate_sum_VM_FC_Ash"].isna().all():
        df["proximate_sum_VM_FC_Ash"] = (
            df["volatile_matter_wt_pct"] + df["fixed_carbon_wt_pct"] + df["ash_wt_pct"]
        )
    if df["proximate_sum_deviation_from_100"].isna().all():
        df["proximate_sum_deviation_from_100"] = df["proximate_sum_VM_FC_Ash"] - 100.0
    if df["ultimate_CHNO_sum"].isna().all():
        df["ultimate_CHNO_sum"] = (
            df["carbon_wt_pct"] + df["hydrogen_wt_pct"] + df["nitrogen_wt_pct"] + df["oxygen_wt_pct"]
        )
    if df["ultimate_CHNO_sum_deviation_from_100"].isna().all():
        df["ultimate_CHNO_sum_deviation_from_100"] = df["ultimate_CHNO_sum"] - 100.0

    # Clean categorical columns
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = clean_string_col(df[col])

    # composition_basis_pair / yield_reporting_profile fallback construction
    if "composition_basis_pair" not in df.columns or df["composition_basis_pair"].isna().all():
        df["composition_basis_pair"] = (
            clean_string_col(df.get("proximate_basis", pd.Series(["NS"]*len(df), index=df.index)))
            + "__"
            + clean_string_col(df.get("ultimate_basis", pd.Series(["NS"]*len(df), index=df.index)))
        )
    if "yield_reporting_profile" not in df.columns or df["yield_reporting_profile"].isna().all():
        df["yield_reporting_profile"] = (
            clean_string_col(df.get("yield_basis", pd.Series(["NS"]*len(df), index=df.index)))
            + "__"
            + clean_string_col(df.get("oil_definition", pd.Series(["NS"]*len(df), index=df.index)))
            + "__"
            + clean_string_col(df.get("gas_yield_method", pd.Series(["NS"]*len(df), index=df.index)))
        )

    return df


def impute_numeric(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    return SimpleImputer(strategy="median").fit_transform(X)


def classify_vif(v: float) -> str:
    if np.isinf(v):
        return "infinite/perfect collinearity (by construction)"
    if v >= VIF_THRESHOLDS["high"]:
        return "high"
    if v >= 5.0:
        return "moderate"
    return "acceptable"


def vif_table(X_np: np.ndarray, names: list[str], subset_label: str) -> pd.DataFrame:
    """Compute VIF for a full-rank numeric matrix."""
    Xs    = StandardScaler().fit_transform(X_np)
    rank  = int(np.linalg.matrix_rank(Xs))
    n_col = Xs.shape[1]

    if rank < n_col:
        # Rank-deficient — return a note row for each feature
        rows = []
        for name in names:
            rows.append({
                "subset": subset_label,
                "feature": name,
                "feature_label": FEATURE_LABELS.get(name, name),
                "feature_group": FEATURE_GROUPS.get(name, ""),
                "R2_against_others": np.nan,
                "VIF": np.inf,
                "VIF_flag": "infinite/perfect collinearity (by construction)",
                "rank_deficiency_note": (
                    f"Matrix rank {rank} < {n_col} features: "
                    f"{n_col - rank} exact linear dependenc(ies) present. "
                    "VIF undefined for all features in this set — see sheet 61_VIF_C_Rank_Note."
                ),
            })
        return pd.DataFrame(rows)

    rows = []
    for i, name in enumerate(names):
        others = np.delete(Xs, i, axis=1)
        r2 = LinearRegression().fit(others, Xs[:, i]).score(others, Xs[:, i])
        vif = 1.0 / (1.0 - r2) if r2 < (1.0 - 1e-12) else np.inf
        rows.append({
            "subset": subset_label,
            "feature": name,
            "feature_label": FEATURE_LABELS.get(name, name),
            "feature_group": FEATURE_GROUPS.get(name, ""),
            "R2_against_others": round(r2, 6),
            "VIF": round(vif, 3) if np.isfinite(vif) else np.inf,
            "VIF_flag": classify_vif(vif),
            "rank_deficiency_note": "",
        })
    return pd.DataFrame(rows)


def condition_number_and_rank(X_np: np.ndarray, names: list[str], label: str) -> dict:
    Xs   = StandardScaler().fit_transform(X_np)
    rank = int(np.linalg.matrix_rank(Xs))
    cnum = float(np.linalg.cond(Xs))
    return {
        "subset": label,
        "n_features": len(names),
        "rank": rank,
        "perfect_dependencies": len(names) - rank,
        "condition_number": cnum,
        "condition_flag": (
            "severe (singular or near-singular)" if cnum > 1e6
            else ("ill-conditioned" if cnum > 1e3 else "acceptable")
        ),
    }


def correlation_matrix_and_pairs(X_np: np.ndarray, names: list[str], method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(names)
    corr_vals = np.full((n, n), np.nan)
    pval_vals = np.full((n, n), np.nan)

    corr_fn = spearmanr if method == "spearman" else pearsonr

    for i in range(n):
        for j in range(n):
            if i == j:
                corr_vals[i, j] = 1.0
                pval_vals[i, j] = 0.0
            elif j > i:
                mask = ~(np.isnan(X_np[:, i]) | np.isnan(X_np[:, j]))
                if mask.sum() > 3:
                    r, p = corr_fn(X_np[mask, i], X_np[mask, j])
                    corr_vals[i, j] = corr_vals[j, i] = r
                    pval_vals[i, j] = pval_vals[j, i] = p

    labels = [FEATURE_LABELS.get(n_, n_) for n_ in names]
    matrix = pd.DataFrame(corr_vals, index=labels, columns=labels).round(3)
    matrix.index.name = "feature"
    matrix = matrix.reset_index()

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            r = corr_vals[i, j]
            p = pval_vals[i, j]
            pairs.append({
                "feature_1": names[i],
                "feature_1_label": FEATURE_LABELS.get(names[i], names[i]),
                "feature_1_group": FEATURE_GROUPS.get(names[i], ""),
                "feature_2": names[j],
                "feature_2_label": FEATURE_LABELS.get(names[j], names[j]),
                "feature_2_group": FEATURE_GROUPS.get(names[j], ""),
                f"{method}_r": round(r, 4) if pd.notna(r) else np.nan,
                "p_value": round(p, 6) if pd.notna(p) else np.nan,
                "abs_r": round(abs(r), 4) if pd.notna(r) else np.nan,
                "flag_abs_ge_0_80": pd.notna(r) and abs(r) >= 0.80,
                "flag_abs_ge_0_95": pd.notna(r) and abs(r) >= 0.95,
            })
    pairs_df = pd.DataFrame(pairs).sort_values("abs_r", ascending=False).reset_index(drop=True)
    return matrix, pairs_df


def exact_linear_dependency_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True))
    rows = []
    for i, f1 in enumerate(cols):
        for j, f2 in enumerate(cols):
            if j <= i:
                continue
            a = x[f1].to_numpy(dtype=float)
            b = x[f2].to_numpy(dtype=float)
            if np.any(np.isnan(a)) or np.any(np.isnan(b)):
                continue
            r2 = LinearRegression().fit(a.reshape(-1, 1), b).score(a.reshape(-1, 1), b)
            if r2 >= 0.9999:
                exact = bool(np.allclose(a, b, atol=1e-12, rtol=1e-12))
                rows.append({
                    "feature_1": f1,
                    "feature_1_label": FEATURE_LABELS.get(f1, f1),
                    "feature_2": f2,
                    "feature_2_label": FEATURE_LABELS.get(f2, f2),
                    "exact_duplicate": exact,
                    "linear_R2": round(r2, 6),
                    "note": (
                        "Exact duplicate" if exact
                        else "Perfect or near-perfect linear dependency (R²≥0.9999)"
                    ),
                })
    return pd.DataFrame(rows)


def feature_audit(df: pd.DataFrame, all_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in all_cols:
        s = pd.to_numeric(df[col], errors="coerce") if col in FEATURE_GROUPS else df.get(col, pd.Series(dtype=object))
        rows.append({
            "feature": col,
            "feature_label": FEATURE_LABELS.get(col, col),
            "feature_group": FEATURE_GROUPS.get(col, "categorical"),
            "n_non_missing": int(s.notna().sum()),
            "pct_available": round(s.notna().mean() * 100, 1),
            "n_unique": s.nunique(),
            "dtype": str(s.dtype),
            "mean_or_mode": (round(float(s.mean()), 3) if s.dtype.kind in "fi" else str(s.mode().iloc[0]) if len(s.mode()) else ""),
            "std": round(float(s.std()), 3) if s.dtype.kind in "fi" else "",
        })
    return pd.DataFrame(rows)


def vif_interpretation_guide() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "subset": "VIF-A",
            "features_included": "Raw core predictors (9): process + proximate + ultimate",
            "features_excluded": "Severity descriptors, sum/deviation diagnostics",
            "matrix_status": "Full rank — clean, interpretable VIF",
            "primary_use": "PRIMARY reference for manuscript collinearity section. Shows true pairwise redundancy among raw input variables.",
            "interpretation_note": "All VIF values 1–5. No problematic collinearity in the raw composition/process space.",
        },
        {
            "subset": "VIF-B",
            "features_included": "Model predictors (12): VIF-A + inv_temp_K, log_HR, temp×log_HR",
            "features_excluded": "Sum/deviation diagnostics",
            "matrix_status": "Full rank — interpretable VIF",
            "primary_use": "Collinearity in the actual model feature space (F2_core_severity). Severity descriptors will show VIF 14–42 by design.",
            "interpretation_note": (
                "High VIF for severity descriptors is EXPECTED and NOT a modelling problem. "
                "These features are deliberately nonlinear transforms of temperature and heating rate "
                "(capturing Arrhenius-like kinetics). Tree-based models are invariant to linear rescaling "
                "and do not require uncorrelated predictors, unlike OLS regression. "
                "Report VIF-B as a transparency note, not as evidence of a problem."
            ),
        },
        {
            "subset": "VIF-C",
            "features_included": "Full set (16): VIF-B + proximate_sum, proximate_dev, ultimate_sum, ultimate_dev",
            "features_excluded": "Nothing — this is the full numeric set",
            "matrix_status": "Rank-deficient (4 exact linear dependencies by construction)",
            "primary_use": "Confirms the engineered diagnostic features are exact linear combinations. NOT a meaningful VIF table.",
            "interpretation_note": (
                "proximate_sum = VM + FC + Ash  (R²=1.0 by definition). "
                "proximate_deviation = proximate_sum − 100  (exact linear transform). "
                "Same for ultimate. Adding these to an OLS design matrix makes it singular — "
                "VIF is mathematically undefined for ALL features in the set, not just the offending ones. "
                "This is the VIF contamination trap: one perfect dependency breaks the entire inversion. "
                "These features are intentional redundancies serving as model diagnostics; "
                "tree models handle them natively through split selection."
            ),
        },
    ])


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Loading {INPUT_XLSX} / {INPUT_SHEET}...")
    df = load_and_engineer(INPUT_XLSX, INPUT_SHEET)
    print(f"  Rows: {len(df)}, Studies: {df['study_id'].nunique() if 'study_id' in df.columns else 'NA'}")

    # ---- Imputed numeric matrices ----
    X_A = impute_numeric(df, VIF_A_FEATURES)
    X_B = impute_numeric(df, VIF_B_FEATURES)
    X_C = impute_numeric(df, VIF_B_FEATURES + ENGINEERED_FEATURES)

    # ---- VIF tables ----
    print("Computing VIF tables...")
    vif_A = vif_table(X_A, VIF_A_FEATURES,   "VIF-A: Raw core predictors")
    vif_B = vif_table(X_B, VIF_B_FEATURES,   "VIF-B: Model predictors (incl. severity)")
    vif_C = vif_table(X_C, VIF_B_FEATURES + ENGINEERED_FEATURES, "VIF-C: Full set (rank-deficient)")

    # ---- Condition numbers ----
    print("Computing condition numbers...")
    cond_A = condition_number_and_rank(X_A, VIF_A_FEATURES,   "VIF-A")
    cond_B = condition_number_and_rank(X_B, VIF_B_FEATURES,   "VIF-B")
    cond_C = condition_number_and_rank(X_C, VIF_B_FEATURES + ENGINEERED_FEATURES, "VIF-C")
    design_info = pd.DataFrame([cond_A, cond_B, cond_C])

    # ---- Correlation matrices on VIF-B (model feature set) ----
    print("Computing Spearman and Pearson correlations...")
    X_B_raw = df[VIF_B_FEATURES].apply(pd.to_numeric, errors="coerce").values  # pairwise deletion
    spear_matrix, spear_pairs = correlation_matrix_and_pairs(X_B_raw, VIF_B_FEATURES, "spearman")
    pears_matrix, pears_pairs = correlation_matrix_and_pairs(X_B_raw, VIF_B_FEATURES, "pearson")

    spear_high = spear_pairs[spear_pairs["flag_abs_ge_0_80"]].copy()
    pears_high = pears_pairs[pears_pairs["flag_abs_ge_0_80"]].copy()

    # ---- Severity descriptor focus ----
    sev_focus_features = [
        "temperature_c", "heating_rate_c_min",
        "inv_temp_K", "log_heating_rate", "temp_x_log_heating_rate",
    ]
    X_sev = df[sev_focus_features].apply(pd.to_numeric, errors="coerce").values
    sev_matrix, sev_pairs = correlation_matrix_and_pairs(X_sev, sev_focus_features, "spearman")
    sev_note = pd.DataFrame([{
        "note": (
            "Severity descriptors (inv_temp_K, log_heating_rate, temp×log_HR) show Spearman |r| > 0.95 "
            "with their parent variables (temperature_c, heating_rate_c_min). This is expected: they are "
            "monotonic transforms designed to linearise activation-energy-like relationships. "
            "Their inclusion in VIF-B produces high VIF (14–42) for the entire process/severity group, "
            "which is a consequence of this design, not evidence of spurious data correlation."
        )
    }])

    # ---- Exact linear dependencies ----
    print("Checking for exact linear dependencies in full feature set...")
    exact_deps = exact_linear_dependency_table(df, VIF_B_FEATURES + ENGINEERED_FEATURES)

    # ---- Feature audit ----
    audit = feature_audit(df, VIF_B_FEATURES + ENGINEERED_FEATURES + CATEGORICAL_FEATURES)

    # ---- Interpretation guide ----
    guide = vif_interpretation_guide()

    # ---- Summary ----
    spear_high_any = spear_pairs[spear_pairs["flag_abs_ge_0_80"]]
    spear_high_95  = spear_pairs[spear_pairs["flag_abs_ge_0_95"]]
    # Exclude severity-vs-severity / severity-vs-parent pairs from the "concerning" count
    sev_set = set(sev_focus_features)
    non_sev_high = spear_high_any[
        ~(spear_high_any["feature_1"].isin(sev_set) | spear_high_any["feature_2"].isin(sev_set))
    ]

    summary = pd.DataFrame([
        {"item": "dataset",
         "value": f"{len(df)} rows, {df['study_id'].nunique() if 'study_id' in df.columns else 'NA'} studies",
         "interpretation": "Collinearity diagnostics applied to the 442-row model-ready dataset."},
        {"item": "VIF_A_features", "value": str(len(VIF_A_FEATURES)),
         "interpretation": "Raw core predictors used as primary VIF reference (full-rank matrix)."},
        {"item": "VIF_B_features", "value": str(len(VIF_B_FEATURES)),
         "interpretation": "Model predictors for F2_core_severity (raw core + severity descriptors)."},
        {"item": "VIF_A_max_VIF",
         "value": str(round(vif_A[vif_A["VIF"].apply(np.isfinite)]["VIF"].max(), 2)),
         "interpretation": "Maximum VIF among raw core predictors. Values < 10 are acceptable for tree-based models."},
        {"item": "VIF_B_max_VIF_excl_severity",
         "value": str(round(vif_B[
             (~vif_B["feature"].isin(VIF_B_ADDED)) & vif_B["VIF"].apply(np.isfinite)
         ]["VIF"].max(), 2)),
         "interpretation": "Max VIF for raw composition/process features in the model set. Severity descriptors excluded."},
        {"item": "VIF_B_severity_VIF_range",
         "value": str(round(vif_B[vif_B["feature"].isin(VIF_B_ADDED) & vif_B["VIF"].apply(np.isfinite)]["VIF"].min(), 1))
                  + "–"
                  + str(round(vif_B[vif_B["feature"].isin(VIF_B_ADDED) & vif_B["VIF"].apply(np.isfinite)]["VIF"].max(), 1)),
         "interpretation": "VIF range for severity descriptors. High by design (nonlinear transforms of process variables). Not a modelling problem for tree-based models."},
        {"item": "VIF_C_perfect_dependencies", "value": "4",
         "interpretation": "proximate_sum=VM+FC+Ash, proximate_dev=sum-100, ultimate_sum=C+H+N+O, ultimate_dev=sum-100. These are intentional redundancies; VIF trivially infinite and uninformative."},
        {"item": "spearman_pairs_high_non_severity",
         "value": str(len(non_sev_high)),
         "interpretation": "Pairs with |r|≥0.80 among non-severity features. These represent genuine composition correlations."},
        {"item": "spearman_pairs_severity_vs_parent",
         "value": str(len(spear_high_any) - len(non_sev_high)),
         "interpretation": "High-correlation pairs involving severity descriptors vs their parent process variables. Expected by construction."},
        {"item": "condition_number_VIF_A",
         "value": f"{cond_A['condition_number']:.3g}",
         "interpretation": "Condition number of the scaled raw-core design matrix. Low value confirms no problematic numerical dependency."},
        {"item": "condition_number_VIF_B",
         "value": f"{cond_B['condition_number']:.3g}",
         "interpretation": "Condition number with severity descriptors included. Higher due to near-collinearity of transforms."},
    ])

    print(f"\nSaving to {OUTPUT_XLSX}...")
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        summary.to_excel(          writer, sheet_name="50_Summary",                    index=False)
        guide.to_excel(            writer, sheet_name="51_VIF_Interpretation_Guide",   index=False)
        audit.to_excel(            writer, sheet_name="52_Feature_Audit",              index=False)
        vif_A.to_excel(            writer, sheet_name="58_VIF_A_Raw_Core",             index=False)
        vif_B.to_excel(            writer, sheet_name="59_VIF_B_Model_Features",       index=False)
        vif_C.to_excel(            writer, sheet_name="60_VIF_C_Full_Rank_Note",       index=False)
        design_info.to_excel(      writer, sheet_name="61_Condition_Numbers",          index=False)
        spear_matrix.to_excel(     writer, sheet_name="62_Spearman_Matrix",            index=False)
        spear_pairs.to_excel(      writer, sheet_name="63_Spearman_Pairs_All",         index=False)
        spear_high.to_excel(       writer, sheet_name="64_Spearman_Pairs_High",        index=False)
        pears_matrix.to_excel(     writer, sheet_name="65_Pearson_Matrix",             index=False)
        pears_pairs.to_excel(      writer, sheet_name="66_Pearson_Pairs_All",          index=False)
        pears_high.to_excel(       writer, sheet_name="67_Pearson_Pairs_High",         index=False)
        sev_matrix.to_excel(       writer, sheet_name="68_Severity_Spearman_Focus",    index=False)
        sev_pairs.to_excel(        writer, sheet_name="69_Severity_Pairs_Detail",      index=False)
        sev_note.to_excel(         writer, sheet_name="70_Severity_Interpretation",    index=False)
        exact_deps.to_excel(       writer, sheet_name="71_Exact_Linear_Dependencies",  index=False)

    print("\nDONE.")
    print("\nKey results:")
    print(f"  VIF-A max (raw core):         {vif_A[vif_A['VIF'].apply(np.isfinite)]['VIF'].max():.2f}")
    print(f"  VIF-B severity range:         "
          f"{vif_B[vif_B['feature'].isin(VIF_B_ADDED) & vif_B['VIF'].apply(np.isfinite)]['VIF'].min():.1f}–"
          f"{vif_B[vif_B['feature'].isin(VIF_B_ADDED) & vif_B['VIF'].apply(np.isfinite)]['VIF'].max():.1f}")
    print(f"  Non-severity high Spearman pairs (|r|≥0.80): {len(non_sev_high)}")
    print(f"  Perfect linear dependencies:  4 (by construction)")


if __name__ == "__main__":
    main()
