"""Derive composite pipeline-failure signal from M1–M4.

Joins machine_preference (M1), m2_thickness_shift (M2), m3_vol_drift (M3, if
available), and m4_qc_score (M4, if available) on (ref_path, cor_path).
Z-normalises each signal, then computes both a geometric-mean and arithmetic-
mean composite.

M4 is per-scan (not per-pair); this script uses the corrupted scan's QC score
as M4_pair, negated so that "higher = worse" like M1–M3.

If M3 or M4 CSVs are missing or empty, the composite is built from the
available signals and documented in the output metadata.

CLI:
    python code/derive_composite_signal.py \\
        --machine-preference results/tables/machine_preference.csv \\
        --m2-csv results/tables/m2_thickness_shift.csv \\
        --m3-csv results/tables/m3_vol_drift.csv \\
        --m4-csv results/tables/m4_qc_score.csv \\
        --output results/tables/composite_pipeline_failure.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

_SHIFT = 5.0  # added to z-scores before geometric mean so all values are positive


def _z_norm(series: pd.Series) -> pd.Series:
    """Z-normalise a pandas Series (mean 0, std 1)."""
    mu, sigma = series.mean(), series.std()
    if sigma < 1e-10:
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sigma


def _load_optional(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        logger.warning("%s not found at %s — signal deferred", label, path)
        return None
    df = pd.read_csv(path)
    if df.empty:
        logger.warning("%s is empty — signal deferred", label)
        return None
    logger.info("%s: %d rows", label, len(df))
    return df


def _derive_composite(
    machine_preference: Path,
    m2_csv: Path,
    m3_csv: Path | None,
    m4_csv: Path | None,
    output: Path,
) -> pd.DataFrame:
    # ── Load M1 ─────────────────────────────────────────────────────────────
    mp = pd.read_csv(machine_preference)
    # machine_preference has: ref_path, cor_path, ... mean_dice_degradation or preference_score
    # Identify the M1 column (degradation = higher → worse pipeline failure)
    if "mean_dice_degradation" in mp.columns:
        m1_col = "mean_dice_degradation"
    elif "preference_score" in mp.columns:
        # preference_score = 1 - mean_dice_degradation; flip for consistent direction
        mp["mean_dice_degradation"] = 1.0 - mp["preference_score"]
        m1_col = "mean_dice_degradation"
    elif "mean_dice" in mp.columns:
        # mean_dice is raw preference (higher = better quality);
        # invert so M1 higher = worse pipeline outcome
        mp["mean_dice_degradation"] = 1.0 - mp["mean_dice"]
        m1_col = "mean_dice_degradation"
    else:
        raise ValueError(f"No M1 column found in {machine_preference}. Columns: {mp.columns.tolist()}")

    base = mp[["ref_path", "cor_path"]].copy()
    base["M1_dice_deg"] = mp[m1_col].values

    signals_used = ["M1"]

    # ── Load M2 ─────────────────────────────────────────────────────────────
    m2 = _load_optional(m2_csv, "M2")
    if m2 is not None:
        m2_sub = m2[["ref_path", "cor_path", "mean_abs_thickness_shift_mm"]].copy()
        m2_sub = m2_sub.rename(columns={"mean_abs_thickness_shift_mm": "M2_thickness_shift"})
        base = base.merge(m2_sub, on=["ref_path", "cor_path"], how="left")
        signals_used.append("M2")
    else:
        base["M2_thickness_shift"] = float("nan")

    # ── Load M3 ─────────────────────────────────────────────────────────────
    m3 = _load_optional(m3_csv, "M3") if m3_csv else None
    if m3 is not None and not m3.empty and "volumetric_instability_score" in m3.columns:
        m3_sub = m3[["ref_path", "cor_path", "volumetric_instability_score"]].copy()
        m3_sub = m3_sub.rename(columns={"volumetric_instability_score": "M3_vol_drift"})
        base = base.merge(m3_sub, on=["ref_path", "cor_path"], how="left")
        signals_used.append("M3")
    else:
        base["M3_vol_drift"] = float("nan")

    # ── Load M4 ─────────────────────────────────────────────────────────────
    m4 = _load_optional(m4_csv, "M4") if m4_csv else None
    if m4 is not None and "synthseg_qc_score" in m4.columns:
        # M4 is per-scan; use the corrupted scan's QC score (lower = worse)
        m4_cor = m4[["scan_path", "synthseg_qc_score"]].copy()
        m4_cor = m4_cor.rename(columns={"scan_path": "cor_path", "synthseg_qc_score": "M4_qc_raw"})
        base = base.merge(m4_cor, on="cor_path", how="left")
        # Negate: higher QC score = better quality, so flip so higher = worse
        base["M4_qc_neg"] = -base["M4_qc_raw"]
        base = base.drop(columns=["M4_qc_raw"])
        signals_used.append("M4")
    else:
        base["M4_qc_neg"] = float("nan")

    # ── Z-normalise available signals ────────────────────────────────────────
    signal_cols = {
        "M1": "M1_dice_deg",
        "M2": "M2_thickness_shift",
        "M3": "M3_vol_drift",
        "M4": "M4_qc_neg",
    }
    z_cols = []
    for label, col in signal_cols.items():
        if label in signals_used and col in base.columns:
            valid = base[col].dropna()
            if len(valid) > 1:
                base[f"{col}_z"] = _z_norm(base[col])
                z_cols.append(f"{col}_z")
            else:
                logger.warning("%s has insufficient data for z-norm; skipping", label)

    if not z_cols:
        raise RuntimeError("No valid signals to build composite.")

    # ── Arithmetic composite ─────────────────────────────────────────────────
    base["composite_arith"] = base[z_cols].mean(axis=1)

    # ── Geometric composite ───────────────────────────────────────────────────
    # Shift z-scores by +_SHIFT so all values are positive before geometric mean.
    # This is a numeric choice (documented here): the +5 shift ensures positive
    # inputs for the geometric mean regardless of z-score range.
    def _geo_mean_row(row: pd.Series) -> float:
        vals = [row[c] + _SHIFT for c in z_cols if pd.notna(row[c])]
        if not vals:
            return float("nan")
        log_sum = sum(math.log(max(v, 1e-10)) for v in vals)
        return math.exp(log_sum / len(vals))

    base["composite_geo"] = base.apply(_geo_mean_row, axis=1)

    # Clean up z-score helper columns (keep only the main signal cols + composites)
    base = base.drop(columns=[c for c in z_cols])

    output.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(output, index=False)

    meta = {"signals_used": signals_used, "n_pairs": len(base), "z_shift": _SHIFT}
    meta_path = output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info("Composite built from %s (%d pairs) → %s", signals_used, len(base), output)
    return base


def main() -> None:
    p = argparse.ArgumentParser(description="Derive composite pipeline failure signal")
    p.add_argument("--machine-preference", type=Path, required=True)
    p.add_argument("--m2-csv", type=Path, required=True)
    p.add_argument("--m3-csv", type=Path, default=None)
    p.add_argument("--m4-csv", type=Path, default=None)
    p.add_argument("--output", type=Path, default=Path("results/tables/composite_pipeline_failure.csv"))
    args = p.parse_args()
    df = _derive_composite(
        args.machine_preference, args.m2_csv, args.m3_csv, args.m4_csv, args.output
    )
    print(df[["ref_path", "cor_path", "M1_dice_deg", "M2_thickness_shift", "composite_arith", "composite_geo"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
