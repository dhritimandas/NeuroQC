"""ABIDE rater-agreement metrics for iter-3 VLM scores.

Computes three rater consensus variants and per-model agreement metrics
(AUC, Cohen's kappa at Youden threshold, weighted kappa vs individual raters).

CLI:
    python code/report_abide_agreement.py \\
        --abide-scores results/tables/abide_proto_scores_seed_0.csv \\
        --abide-ratings data/abide/abide_ratings_iqms.csv \\
        --output results/metrics/abide_proto_agreement.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _majority_vote(row: pd.Series) -> float | None:
    """Return majority rating if present, else NaN."""
    votes = [row.get(f"rater_{i}") for i in range(1, 4)]
    valid = [v for v in votes if pd.notna(v)]
    if not valid:
        return float("nan")
    # Majority = value that appears most; ties go to NaN for variant C
    from collections import Counter
    counts = Counter(int(v) for v in valid)
    most_common = counts.most_common(1)[0]
    if len(valid) >= 2 and most_common[1] >= 2:
        return float(most_common[0])
    elif len(valid) == 1:
        return float(valid[0])
    return float("nan")


def _youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Find Youden J-optimal threshold from binary labels + continuous scores."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(thresholds[best_idx])


def _weighted_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    """Weighted (linear) Cohen's kappa between two ordinal arrays."""
    try:
        return float(cohen_kappa_score(y1, y2, weights="linear"))
    except Exception:
        return float("nan")


def compute_agreement(
    abide_scores_csv: Path,
    abide_ratings_csv: Path,
    output: Path,
) -> dict:
    """Compute agreement metrics between VLM scores and ABIDE rater ratings.

    Args:
        abide_scores_csv: CSV with FILE_ID, model, score (VLM output).
        abide_ratings_csv: CSV with FILE_ID, site, rater_1, rater_2, rater_3.
        output: Path to write JSON results.

    Returns:
        Dict with per-model per-variant AUC, kappa, and weighted kappa.
    """
    scores = pd.read_csv(abide_scores_csv)
    ratings = pd.read_csv(abide_ratings_csv)

    if "FILE_ID" not in scores.columns:
        # Try to construct from other columns
        if "scan_path" in scores.columns:
            scores["FILE_ID"] = scores["scan_path"].apply(
                lambda p: Path(p).stem.replace(".nii", "").replace(".gz", "")
            )

    if "FILE_ID" not in ratings.columns:
        if "subject_id" in ratings.columns and "site" in ratings.columns:
            ratings["FILE_ID"] = ratings["site"] + "_" + ratings["subject_id"].astype(str).str.zfill(7)

    # Normalise VLM score to [0, 1]
    if "score" in scores.columns and scores["score"].max() > 2:
        scores["score_norm"] = (scores["score"] - 1) / 4.0
    else:
        scores["score_norm"] = scores.get("score", scores.get("score_norm"))

    # Build consensus variants
    ratings["consensus_A"] = ratings["rater_3"]  # full coverage

    ratings["consensus_B"] = ratings.apply(
        lambda row: _majority_vote(row) if pd.notna(row.get("rater_1")) else row.get("rater_3"),
        axis=1,
    )

    # Variant C: strict 3-rater majority (only where all 3 present and agree 2+)
    def _strict_majority(row: pd.Series) -> float:
        votes = [row.get(f"rater_{i}") for i in range(1, 4)]
        if any(pd.isna(v) for v in votes):
            return float("nan")
        from collections import Counter
        counts = Counter(int(v) for v in votes)
        mc = counts.most_common(1)[0]
        return float(mc[0]) if mc[1] >= 2 else float("nan")

    ratings["consensus_C"] = ratings.apply(_strict_majority, axis=1)

    logger.info(
        "Consensus coverage — A: %d, B: %d, C: %d",
        ratings["consensus_A"].notna().sum(),
        ratings["consensus_B"].notna().sum(),
        ratings["consensus_C"].notna().sum(),
    )

    # Merge scores with ratings
    merged = scores.merge(
        ratings[["FILE_ID", "consensus_A", "consensus_B", "consensus_C", "rater_1", "rater_2", "rater_3"]],
        on="FILE_ID",
        how="inner",
    )
    logger.info("After merge: %d rows (models × scans)", len(merged))

    results: dict = {}
    models = merged["model"].unique()

    for model in models:
        mdf = merged[merged["model"] == model].copy()
        model_result: dict = {}

        for variant in ("A", "B", "C"):
            con_col = f"consensus_{variant}"
            sub = mdf.dropna(subset=[con_col, "score_norm"]).copy()
            if len(sub) < 10:
                model_result[variant] = {"n": len(sub), "note": "insufficient_data"}
                continue

            # Binarise: accept (rating >= 3) vs reject (rating < 3)
            # Standard mriqc-learn convention: 1 = accept, 0 = reject
            sub["accept"] = (sub[con_col] >= 3).astype(int)

            auc = float("nan")
            kappa = float("nan")
            if sub["accept"].nunique() > 1:
                try:
                    auc = round(float(roc_auc_score(sub["accept"], sub["score_norm"])), 4)
                except Exception:
                    pass
                thresh = _youden_threshold(sub["accept"].values, sub["score_norm"].values)
                pred_bin = (sub["score_norm"] >= thresh).astype(int)
                try:
                    kappa = round(float(cohen_kappa_score(sub["accept"].values, pred_bin.values)), 4)
                except Exception:
                    pass

            model_result[variant] = {
                "n": int(len(sub)),
                "auc": auc,
                "kappa_youden_insample": kappa,
                "accept_rate_rater": float(sub["accept"].mean()),
            }

        # Weighted kappa vs each individual rater
        for rater in ("rater_1", "rater_2", "rater_3"):
            sub = mdf.dropna(subset=[rater, "score_norm"]).copy()
            if len(sub) < 10:
                model_result[f"wkappa_{rater}"] = float("nan")
                continue
            # Bin VLM score to 1-5 at natural quintile boundaries
            sub["vlm_binned"] = pd.cut(sub["score_norm"], bins=5, labels=[1, 2, 3, 4, 5]).astype(float)
            sub = sub.dropna(subset=["vlm_binned"])
            wk = _weighted_kappa(sub[rater].astype(int).values, sub["vlm_binned"].astype(int).values)
            model_result[f"wkappa_{rater}"] = round(wk, 4)

        results[model] = model_result
        logger.info("Model %s: A-AUC=%.3f, B-AUC=%.3f",
                    model,
                    model_result.get("A", {}).get("auc", float("nan")) or float("nan"),
                    model_result.get("B", {}).get("auc", float("nan")) or float("nan"))

    # Print summary table
    print("\n=== ABIDE Agreement Summary ===")
    for model, res in results.items():
        auc_a = res.get("A", {}).get("auc", "NA")
        auc_c = res.get("C", {}).get("auc", "NA")
        k_a = res.get("A", {}).get("kappa_youden_insample", "NA")
        print(f"  {model:<30}  AUC-A={auc_a:<7}  AUC-C={auc_c:<7}  κ-A(in-sample)={k_a}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, default=str))
    logger.info("Wrote agreement metrics → %s", output)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="ABIDE rater agreement report")
    p.add_argument("--abide-scores", type=Path, required=True)
    p.add_argument("--abide-ratings", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/metrics/abide_proto_agreement.json"))
    args = p.parse_args()
    compute_agreement(args.abide_scores, args.abide_ratings, args.output)


if __name__ == "__main__":
    main()
