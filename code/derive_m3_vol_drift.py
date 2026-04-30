"""Derive M3 per-region volumetric drift per (ref, cor) pair.

SynthSeg --vol writes one CSV per scan (one row, region columns in mL).
This script joins those per-scan CSVs with corruption_manifest.csv to
produce a per-pair volumetric instability signal.

Requires Block 2 SynthSeg --vol re-run to have produced *_vol.csv files.
If no _vol.csv files exist, this script exits gracefully with a warning.

CLI:
    python code/derive_m3_vol_drift.py \\
        --synthseg-dir data/derivatives/synthseg \\
        --corruption-manifest results/tables/corruption_manifest.csv \\
        --output results/tables/m3_vol_drift.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _load_vol_csv(vol_csv: Path) -> dict[str, float] | None:
    """Load a SynthSeg _vol.csv and return region → mL dict."""
    try:
        df = pd.read_csv(vol_csv)
        if df.empty:
            return None
        row = df.iloc[0]
        return {col: float(row[col]) for col in df.columns if pd.notna(row[col])}
    except Exception as exc:
        logger.warning("Could not parse %s: %s", vol_csv, exc)
        return None


def _find_vol_csv(scan_path: str, synthseg_dir: Path) -> Path | None:
    """Find the _vol.csv sidecar for a given scan path."""
    stem = Path(scan_path).stem.replace(".nii", "")
    for pattern in (f"**/{stem}_vol.csv", f"**/{stem}.vol.csv"):
        matches = list(synthseg_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _derive_m3(
    synthseg_dir: Path,
    corruption_manifest: Path,
    output: Path,
) -> pd.DataFrame:
    manifest = pd.read_csv(corruption_manifest)

    # Check if any vol CSVs exist
    vol_csvs = list(synthseg_dir.glob("**/*_vol.csv"))
    if not vol_csvs:
        logger.warning("No *_vol.csv files found under %s — M3 deferred", synthseg_dir)
        empty = pd.DataFrame(columns=["ref_path","cor_path","corruption_type","severity",
                                       "volumetric_instability_score","max_region_drift_pct","n_regions_present"])
        output.parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(output, index=False)
        return empty

    logger.info("Found %d _vol.csv files under %s", len(vol_csvs), synthseg_dir)

    records = []
    missing = 0

    for _, row in manifest.iterrows():
        ref_path, cor_path = row["ref_path"], row["cor_path"]
        ref_vol_csv = _find_vol_csv(ref_path, synthseg_dir)
        cor_vol_csv = _find_vol_csv(cor_path, synthseg_dir)

        if ref_vol_csv is None or cor_vol_csv is None:
            missing += 1
            continue

        ref_vols = _load_vol_csv(ref_vol_csv)
        cor_vols = _load_vol_csv(cor_vol_csv)
        if ref_vols is None or cor_vols is None:
            missing += 1
            continue

        common = set(ref_vols) & set(cor_vols)
        pct_changes = []
        for region in common:
            rv = ref_vols[region]
            cv = cor_vols[region]
            if rv > 0:
                pct_changes.append(abs(cv - rv) / rv * 100)

        if not pct_changes:
            missing += 1
            continue

        records.append({
            "ref_path": ref_path,
            "cor_path": cor_path,
            "corruption_type": row.get("corruption_type", ""),
            "severity": row.get("severity", ""),
            "dataset_tag": row.get("dataset_tag", ""),
            "volumetric_instability_score": round(sum(pct_changes) / len(pct_changes), 6),
            "max_region_drift_pct": round(max(pct_changes), 6),
            "n_regions_present": len(pct_changes),
        })

    logger.info("M3: %d pairs computed, %d missing vol files", len(records), missing)
    df = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.info("Wrote %d rows → %s", len(df), output)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Derive M3 volumetric drift")
    p.add_argument("--synthseg-dir", type=Path, default=Path("data/derivatives/synthseg"))
    p.add_argument("--corruption-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/tables/m3_vol_drift.csv"))
    args = p.parse_args()
    _derive_m3(args.synthseg_dir, args.corruption_manifest, args.output)


if __name__ == "__main__":
    main()
