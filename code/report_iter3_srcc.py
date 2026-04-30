"""SRCC matrix: VLM score vs each ground-truth signal with bootstrap 95% CIs.

Reads 2d_vlm_scores_seed_*.csv + composite_pipeline_failure.csv and computes
Spearman ρ for each (model, ground-truth-signal) pair. Bootstrap CIs use
1000 resamples clustered at the scan level.

CLI:
    python code/report_iter3_srcc.py \\
        --vlm-scores results/tables/2d_vlm_scores_seed_0.csv \\
        --composite-csv results/tables/composite_pipeline_failure.csv \\
        --output results/metrics/iter3_srcc_matrix.csv
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

_GROUND_TRUTH_SIGNALS = {
    "M1_Dice": "M1_dice_deg",
    "M2_Thickness": "M2_thickness_shift",
    "M3_Vol": "M3_vol_drift",
    "M4_QC": "M4_qc_neg",
    "Composite_Arith": "composite_arith",
    "Composite_Geo": "composite_geo",
}
_BOOTSTRAP_N = 1000
_SEED = 42


def _bootstrap_spearman_ci(x: list[float], y: list[float], n: int = _BOOTSTRAP_N) -> tuple[float, float]:
    """Bootstrap 95% CI for Spearman ρ via scan-level resampling."""
    rng = random.Random(_SEED)
    rhos = []
    idxs = list(range(len(x)))
    for _ in range(n):
        sample = rng.choices(idxs, k=len(idxs))
        xs = [x[i] for i in sample]
        ys = [y[i] for i in sample]
        if len(set(xs)) < 2 or len(set(ys)) < 2:
            continue
        r, _ = spearmanr(xs, ys)
        if not pd.isna(r):
            rhos.append(r)
    if len(rhos) < 10:
        return float("nan"), float("nan")
    rhos.sort()
    lo = rhos[int(0.025 * len(rhos))]
    hi = rhos[int(0.975 * len(rhos))]
    return round(lo, 3), round(hi, 3)


def compute_srcc_matrix(
    vlm_scores_csv: Path,
    composite_csv: Path,
    output: Path,
) -> pd.DataFrame:
    """Compute Spearman ρ for each (model, signal) pair with bootstrap CIs.

    Args:
        vlm_scores_csv: CSV with model, scan_path / cor_path, score.
        composite_csv: CSV with M1..M4 and composite signals.
        output: Path to write results CSV.

    Returns:
        DataFrame with rows = models × columns = signals.
    """
    vlm = pd.read_csv(vlm_scores_csv)
    comp = pd.read_csv(composite_csv)

    # Normalise VLM score to [0, 1] if it's on a 1-5 scale
    if vlm["score"].max() > 2:
        vlm["score_norm"] = (vlm["score"] - 1) / 4.0
    else:
        vlm["score_norm"] = vlm["score"]

    # VLM CSV has scan_path (the corrupted scan); join on cor_path in composite
    join_col = "cor_path" if "cor_path" in comp.columns else "ref_path"
    scan_col = "scan_path" if "scan_path" in vlm.columns else "cor_path"

    records = []
    models = vlm["model"].unique()

    available_signals = {
        k: v for k, v in _GROUND_TRUTH_SIGNALS.items()
        if v in comp.columns and comp[v].notna().sum() > 10
    }
    logger.info("Models: %s", list(models))
    logger.info("Available signals: %s", list(available_signals.keys()))

    for model in models:
        model_df = vlm[vlm["model"] == model].copy()
        model_df = model_df.dropna(subset=["score_norm"])

        # Join with ground truth on scan path
        merged = model_df.merge(
            comp[[join_col] + list(available_signals.values())],
            left_on=scan_col,
            right_on=join_col,
            how="inner",
        )

        row: dict = {"model": model, "n_valid": len(merged)}

        for sig_label, sig_col in available_signals.items():
            sub = merged.dropna(subset=[sig_col, "score_norm"])
            if len(sub) < 10:
                row[f"{sig_label}_rho"] = float("nan")
                row[f"{sig_label}_ci_lo"] = float("nan")
                row[f"{sig_label}_ci_hi"] = float("nan")
                continue
            rho, _ = spearmanr(sub["score_norm"], sub[sig_col])
            ci_lo, ci_hi = _bootstrap_spearman_ci(
                sub["score_norm"].tolist(), sub[sig_col].tolist()
            )
            row[f"{sig_label}_rho"] = round(rho, 3)
            row[f"{sig_label}_ci_lo"] = ci_lo
            row[f"{sig_label}_ci_hi"] = ci_hi
            logger.info(
                "  %s vs %s: ρ=%.3f [%.3f, %.3f] (n=%d)",
                model, sig_label, rho, ci_lo, ci_hi, len(sub),
            )

        records.append(row)

    df = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.info("SRCC matrix (%d models × %d signals) → %s", len(models), len(available_signals), output)

    # Pretty print
    print("\n=== Iter-3 SRCC Matrix (Spearman ρ vs ground-truth signals) ===")
    rho_cols = [c for c in df.columns if c.endswith("_rho")]
    print(df[["model", "n_valid"] + rho_cols].to_string(index=False))
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Compute iter-3 SRCC matrix")
    p.add_argument("--vlm-scores", type=Path, required=True)
    p.add_argument("--composite-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/metrics/iter3_srcc_matrix.csv"))
    args = p.parse_args()
    compute_srcc_matrix(args.vlm_scores, args.composite_csv, args.output)


if __name__ == "__main__":
    main()
