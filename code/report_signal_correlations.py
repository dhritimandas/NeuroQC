"""Cross-signal correlation report for iter-3 ground-truth signals.

Reads composite_pipeline_failure.csv and computes the Pearson correlation
matrix among M1, M2, M3, M4 (whichever are non-NaN). Prints to stdout and
writes results/metrics/signal_correlations.json.

CLI:
    python code/report_signal_correlations.py \\
        --composite-csv results/tables/composite_pipeline_failure.csv \\
        --output results/metrics/signal_correlations.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

_SIGNAL_COLS = {
    "M1": "M1_dice_deg",
    "M2": "M2_thickness_shift",
    "M3": "M3_vol_drift",
    "M4": "M4_qc_neg",
}


def report_correlations(composite_csv: Path, output: Path) -> dict:
    """Compute and save pairwise Pearson correlations among available signals.

    Args:
        composite_csv: Path to composite_pipeline_failure.csv.
        output: Path to write JSON output.

    Returns:
        Dict with correlation matrix and signal-pair details.
    """
    df = pd.read_csv(composite_csv)

    available = {
        label: col
        for label, col in _SIGNAL_COLS.items()
        if col in df.columns and df[col].notna().sum() > 10
    }

    if len(available) < 2:
        logger.warning("Fewer than 2 signals available — skipping correlation matrix")
        result = {"available_signals": list(available.keys()), "correlation_matrix": {}}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
        return result

    sig_df = df[[col for col in available.values()]].dropna()
    sig_df.columns = list(available.keys())
    corr = sig_df.corr(method="pearson")

    print("\n=== Signal Pearson Correlation Matrix ===")
    print(f"N pairs with all signals present: {len(sig_df)}")
    print(corr.round(3).to_string())

    # Flag potential redundancy
    for i, s1 in enumerate(available):
        for s2 in list(available)[i + 1:]:
            r = corr.loc[s1, s2]
            if abs(r) > 0.9:
                logger.warning("HIGH correlation %s--%s: r=%.3f — composite may not add value", s1, s2, r)

    result = {
        "n_pairs": len(sig_df),
        "available_signals": list(available.keys()),
        "correlation_matrix": corr.round(4).to_dict(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    logger.info("Wrote correlation matrix → %s", output)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-signal correlation report")
    p.add_argument("--composite-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/metrics/signal_correlations.json"))
    args = p.parse_args()
    report_correlations(args.composite_csv, args.output)


if __name__ == "__main__":
    main()
