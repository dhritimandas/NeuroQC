"""Derive M2 cortical thickness shift per (ref, cor) pair.

Joins cortical_thickness.csv (one row per scan, 70 FreeSurfer region columns
plus mean_thickness) with corruption_manifest.csv on scan_path to produce a
per-pair thickness-shift signal.

CLI:
    python code/derive_m2_thickness_shift.py \\
        --thickness-csv results/tables/cortical_thickness.csv \\
        --corruption-manifest results/tables/corruption_manifest.csv \\
        --output results/tables/m2_thickness_shift.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Columns that are metadata, not region-thickness values
_NON_REGION_COLS = {"scan_path", "seg_path", "mean_thickness"}
_MIN_REGIONS = 35  # discard pairs where fewer than this many regions are present


def _derive_m2(
    thickness_csv: Path,
    corruption_manifest: Path,
    output: Path,
) -> pd.DataFrame:
    """Compute mean absolute thickness shift and cortex fragility score.

    Args:
        thickness_csv: CSV with scan_path and per-region thickness columns.
        corruption_manifest: CSV with ref_path, cor_path, corruption_type,
            severity, dataset_tag.
        output: Path to write results CSV.

    Returns:
        DataFrame with M2 signal per (ref_path, cor_path) pair.
    """
    thick = pd.read_csv(thickness_csv)
    manifest = pd.read_csv(corruption_manifest)

    region_cols = [c for c in thick.columns if c not in _NON_REGION_COLS]
    logger.info("Thickness CSV: %d scans, %d region cols", len(thick), len(region_cols))
    logger.info("Corruption manifest: %d pairs", len(manifest))

    # Index thickness by scan_path for fast lookup
    thick_indexed = thick.set_index("scan_path")

    records = []
    missing_ref = missing_cor = skipped_degenerate = 0

    for _, row in manifest.iterrows():
        ref_path = row["ref_path"]
        cor_path = row["cor_path"]

        if ref_path not in thick_indexed.index:
            missing_ref += 1
            continue
        if cor_path not in thick_indexed.index:
            missing_cor += 1
            continue

        ref_row = thick_indexed.loc[ref_path]
        cor_row = thick_indexed.loc[cor_path]

        # Per-region delta; only where both are non-NaN
        delta = {}
        for col in region_cols:
            rv, cv = ref_row.get(col), cor_row.get(col)
            if pd.notna(rv) and pd.notna(cv):
                delta[col] = float(cv) - float(rv)

        n_present = len(delta)
        if n_present < _MIN_REGIONS:
            skipped_degenerate += 1
            continue

        abs_deltas = [abs(v) for v in delta.values()]
        mean_abs_shift = sum(abs_deltas) / n_present
        fragility = sum(1 for v in abs_deltas if v > 0.1) / n_present

        records.append({
            "ref_path": ref_path,
            "cor_path": cor_path,
            "corruption_type": row.get("corruption_type", ""),
            "severity": row.get("severity", ""),
            "dataset_tag": row.get("dataset_tag", ""),
            "mean_abs_thickness_shift_mm": round(mean_abs_shift, 6),
            "cortex_fragility_score": round(fragility, 6),
            "n_regions_present": n_present,
        })

    logger.info(
        "Pairs: %d produced, %d missing ref, %d missing cor, %d skipped degenerate",
        len(records), missing_ref, missing_cor, skipped_degenerate,
    )

    df = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.info("Wrote %d rows → %s", len(df), output)
    return df


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Derive M2 cortical thickness shift")
    p.add_argument("--thickness-csv", type=Path, required=True)
    p.add_argument("--corruption-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/tables/m2_thickness_shift.csv"))
    args = p.parse_args()
    _derive_m2(args.thickness_csv, args.corruption_manifest, args.output)
    # Sanity print for smoke check
    import pandas as pd  # noqa: F811 (already imported above; harmless re-import)
    df = pd.read_csv(args.output)
    print(df[["ref_path", "corruption_type", "severity", "mean_abs_thickness_shift_mm", "cortex_fragility_score"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
