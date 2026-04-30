"""Derive M4 SynthSeg QC score per scan.

Reads the SynthSeg --qc output CSV (one row per scan with a reliability/QC
score) and joins it with corruption_manifest.csv to produce a per-scan M4
signal. M4 is defined at the scan level (not per-pair), unlike M1/M2/M3.

Requires Block 2 (SynthSeg --qc re-run) to have completed. Exits gracefully
if the QC CSV is missing.

CLI:
    python code/derive_m4_qc_score.py \\
        --qc-csv results/tables/synthseg_proto_qc_v2.csv \\
        --corruption-manifest results/tables/corruption_manifest.csv \\
        --output results/tables/m4_qc_score.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _derive_m4(
    qc_csv: Path,
    corruption_manifest: Path,
    output: Path,
) -> pd.DataFrame:
    if not qc_csv.exists():
        logger.warning("SynthSeg QC CSV not found at %s — M4 deferred", qc_csv)
        empty = pd.DataFrame(columns=["scan_path", "is_reference", "corruption_type",
                                       "severity", "synthseg_qc_score"])
        output.parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(output, index=False)
        return empty

    qc = pd.read_csv(qc_csv)
    logger.info("QC CSV: %d rows, columns: %s", len(qc), qc.columns.tolist()[:6])

    # Detect QC score column (SynthSeg uses 'qc' or 'mean_qc' or similar)
    qc_col = None
    for candidate in ("qc", "mean_qc", "qc_score", "synthseg_qc"):
        if candidate in qc.columns:
            qc_col = candidate
            break
    if qc_col is None:
        # Fall back: use last numeric column
        numeric_cols = qc.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            qc_col = numeric_cols[-1]
            logger.warning("Could not identify QC column; using %s", qc_col)
        else:
            logger.warning("No numeric columns in QC CSV — M4 deferred")
            empty = pd.DataFrame(columns=["scan_path", "is_reference", "corruption_type",
                                           "severity", "synthseg_qc_score"])
            output.parent.mkdir(parents=True, exist_ok=True)
            empty.to_csv(output, index=False)
            return empty

    # Identify scan_path column in QC CSV
    scan_col = None
    for candidate in ("scan_path", "input", "filename", "subject"):
        if candidate in qc.columns:
            scan_col = candidate
            break
    if scan_col is None:
        logger.warning("Could not identify scan_path column in QC CSV — M4 deferred")
        empty = pd.DataFrame(columns=["scan_path", "is_reference", "corruption_type",
                                       "severity", "synthseg_qc_score"])
        output.parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(output, index=False)
        return empty

    qc_indexed = qc.set_index(scan_col)[qc_col].to_dict()
    manifest = pd.read_csv(corruption_manifest)

    records = []
    # Add ref scans
    ref_paths = manifest["ref_path"].unique()
    for rp in ref_paths:
        score = qc_indexed.get(rp)
        records.append({
            "scan_path": rp, "is_reference": True,
            "corruption_type": "none", "severity": 0,
            "synthseg_qc_score": float(score) if score is not None else float("nan"),
        })

    # Add corrupted scans
    for _, row in manifest.iterrows():
        cp = row["cor_path"]
        score = qc_indexed.get(cp)
        records.append({
            "scan_path": cp, "is_reference": False,
            "corruption_type": row.get("corruption_type", ""),
            "severity": row.get("severity", ""),
            "synthseg_qc_score": float(score) if score is not None else float("nan"),
        })

    df = pd.DataFrame(records)
    na_count = df["synthseg_qc_score"].isna().sum()
    logger.info("M4: %d scans, %d missing QC scores", len(df), na_count)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.info("Wrote %d rows → %s", len(df), output)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Derive M4 SynthSeg QC score")
    p.add_argument("--qc-csv", type=Path, required=True)
    p.add_argument("--corruption-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/tables/m4_qc_score.csv"))
    args = p.parse_args()
    _derive_m4(args.qc_csv, args.corruption_manifest, args.output)


if __name__ == "__main__":
    main()
