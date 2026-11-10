"""Validate the distributed model-ready dataset before running analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd

from project_config import DATA_FILE, DATA_SHEET, output_file


EXPECTED = {
    "rows": 442,
    "studies": 42,
    "validation_groups": 69,
    "rows_90_110": 356,
    "studies_90_110": 38,
    "rows_95_105": 303,
    "studies_95_105": 34,
    "holding_time_available": 364,
    "particle_size_available": 410,
    "sample_mass_available": 388,
    "complete_process_detail_rows": 322,
    "complete_process_detail_studies": 29,
}

REQUIRED_COLUMNS = {
    "study_id",
    "validation_group_id",
    "temperature_c",
    "heating_rate_c_min",
    "volatile_matter_wt_pct",
    "fixed_carbon_wt_pct",
    "ash_wt_pct",
    "carbon_wt_pct",
    "hydrogen_wt_pct",
    "nitrogen_wt_pct",
    "oxygen_wt_pct",
    "char_yield_wt_pct",
    "oil_yield_wt_pct",
    "gas_yield_wt_pct",
    "yield_sum_wt_pct",
    "holding_time_min_parsed",
    "particle_size_mm_parsed",
    "sample_mass_g_parsed",
}


def build_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Return the dataset inclusion and integrity checks."""
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise KeyError(f"Dataset is missing required columns: {missing}")

    yield_cols = [
        "char_yield_wt_pct",
        "oil_yield_wt_pct",
        "gas_yield_wt_pct",
    ]
    yield_sum = pd.to_numeric(df["yield_sum_wt_pct"], errors="coerce")
    optional_cols = [
        "holding_time_min_parsed",
        "particle_size_mm_parsed",
        "sample_mass_g_parsed",
    ]

    mask_90_110 = yield_sum.between(90.0, 110.0, inclusive="both")
    mask_95_105 = yield_sum.between(95.0, 105.0, inclusive="both")
    process_detail_mask = df[optional_cols].notna().all(axis=1)

    observed = {
        "rows": len(df),
        "studies": df["study_id"].nunique(),
        "validation_groups": df["validation_group_id"].nunique(),
        "rows_90_110": int(mask_90_110.sum()),
        "studies_90_110": df.loc[mask_90_110, "study_id"].nunique(),
        "rows_95_105": int(mask_95_105.sum()),
        "studies_95_105": df.loc[mask_95_105, "study_id"].nunique(),
        "holding_time_available": int(df["holding_time_min_parsed"].notna().sum()),
        "particle_size_available": int(df["particle_size_mm_parsed"].notna().sum()),
        "sample_mass_available": int(df["sample_mass_g_parsed"].notna().sum()),
        "complete_process_detail_rows": int(process_detail_mask.sum()),
        "complete_process_detail_studies": df.loc[
            process_detail_mask, "study_id"
        ].nunique(),
    }

    rows = [
        {
            "check": name,
            "expected": expected,
            "observed": observed[name],
            "status": "PASS" if observed[name] == expected else "FAIL",
        }
        for name, expected in EXPECTED.items()
    ]

    yields = df[yield_cols].apply(pd.to_numeric, errors="coerce")
    rows.extend(
        [
            {
                "check": "complete_yield_triplets",
                "expected": len(df),
                "observed": int(yields.notna().all(axis=1).sum()),
                "status": (
                    "PASS"
                    if int(yields.notna().all(axis=1).sum()) == len(df)
                    else "FAIL"
                ),
            },
            {
                "check": "rows_within_primary_85_110_window",
                "expected": len(df),
                "observed": int(
                    yield_sum.between(85.0, 110.0, inclusive="both").sum()
                ),
                "status": (
                    "PASS"
                    if bool(
                        yield_sum.between(
                            85.0, 110.0, inclusive="both"
                        ).all()
                    )
                    else "FAIL"
                ),
            },
            {
                "check": "finite_nonnegative_yields",
                "expected": len(df),
                "observed": int(
                    (np.isfinite(yields.to_numpy()) & (yields.to_numpy() >= 0))
                    .all(axis=1)
                    .sum()
                ),
                "status": (
                    "PASS"
                    if bool(
                        (
                            np.isfinite(yields.to_numpy())
                            & (yields.to_numpy() >= 0)
                        )
                        .all()
                    )
                    else "FAIL"
                ),
            },
        ]
    )

    # Dataset integrity checks.
    publication_key = df["doi"].astype(str).str.strip().str.lower().str.replace(
        r"^https?://(?:dx\.)?doi\.org/", "", regex=True
    )
    group_study_max = df.groupby("validation_group_id")["study_id"].nunique().max()
    study_doi_max = df.assign(_doi=publication_key).groupby("study_id")["_doi"].nunique().max()
    doi_study_max = df.assign(_doi=publication_key).groupby("_doi")["study_id"].nunique().max()
    actual_sums = yields.sum(axis=1)
    sum_agreement = np.isclose(actual_sums, yield_sum, atol=0.01, rtol=0).sum()
    checks = [
        ("distinct_dois", 42, publication_key.nunique()),
        ("validation_groups_nested_within_studies", 1, group_study_max),
        ("doi_per_study_max", 1, study_doi_max),
        ("studies_per_doi_max", 1, doi_study_max),
        ("yield_sum_matches_products", len(df), int(sum_agreement)),
    ]
    rows.extend(
        {"check": label, "expected": expected, "observed": observed,
         "status": "PASS" if observed == expected else "FAIL"}
        for label, expected, observed in checks
    )

    return pd.DataFrame(rows)


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find the model-ready dataset at {DATA_FILE}. "
            "Set PYROLYSIS_DATA_FILE to override the default repository path."
        )

    df = pd.read_excel(DATA_FILE, sheet_name=DATA_SHEET)
    audit = build_audit(df)
    destination = output_file("00_dataset_audit.csv")
    audit.to_csv(destination, index=False)

    print(audit.to_string(index=False))
    if audit["status"].ne("PASS").any():
        failed = audit.loc[audit["status"].ne("PASS"), "check"].tolist()
        raise RuntimeError(f"Dataset validation failed: {failed}")

    print(f"\nDataset validation passed. Saved audit to: {destination.resolve()}")


if __name__ == "__main__":
    main()
