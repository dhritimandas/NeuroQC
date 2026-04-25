#!/usr/bin/env python3
"""NeuroQC Phase 11 — cross-model meta-table aggregator for ABIDE zero-shot.

Reads all per-model summary JSONs (from Phase 10) and predictions CSVs;
produces a single comparison table across the 5 zero-shot VLMs plus
pairwise DeLong tests with Benjamini-Hochberg FDR adjustment.

Inputs:
    results/metrics/abide_zeroshot_summary_*_seed_*.json   (Phase 10)
    results/tables/abide_zeroshot_predictions_*_seed_*.csv (Phase 10)

Outputs:
    results/metrics/abide_zeroshot_meta_table.csv
        Long format: one row per (model, variant, metric).
        Columns: model, variant, metric, value, ci_low, ci_high, n_seeds.

    results/metrics/abide_zeroshot_meta_table.tex
        LaTeX-ready table fragment (booktabs idiom). Rows = models, cols =
        AUC (95% CI), Cohen's κ, Krippendorff α, LOSO mean ± SD,
        parse_failure_rate. One sub-table per consensus variant (A/B/C).

    results/metrics/abide_zeroshot_pairwise_delong.csv
        Long format: one row per (model_a, model_b, variant). Columns:
        model_a, model_b, variant, auc_a, auc_b, delta_auc, delong_z,
        delong_p, bh_adjusted_p. BH FDR adjustment applied within each
        variant across the C(5,2) = 10 pairs.

    results/metrics/abide_zeroshot_meta_summary.json
        Top-line: best model per variant; ranked AUC list per variant;
        count of significant pairwise differences post-FDR.

Multi-seed aggregation: when multiple `*_seed_<N>.csv` files exist for the
same model, this script aggregates AUC and other point-estimate metrics
across seeds (mean ± SD reported in the meta table); the predictions CSV
used for pairwise DeLong picks the seed with the lowest seed integer
(deterministic; documented in summary).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.logging import RichHandler

logger = logging.getLogger(__name__)
app = typer.Typer(
    help="NeuroQC Phase 11 — cross-model meta-table aggregator.",
    add_completion=False,
)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

CONSENSUS_VARIANTS: tuple[str, ...] = ("A", "B", "C")
META_TABLE_COLUMNS: tuple[str, ...] = (
    "model", "variant", "metric", "value", "ci_low", "ci_high", "n_seeds",
)
DELONG_TABLE_COLUMNS: tuple[str, ...] = (
    "model_a", "model_b", "variant",
    "auc_a", "auc_b", "delta_auc",
    "delong_z", "delong_p", "bh_adjusted_p",
)

_SUMMARY_NAME_RE = re.compile(
    r"^abide_zeroshot_summary_(?P<model>[^_]+(?:_[^_]+)*?)_seed_(?P<seed>\d+)\.json$"
)
_PREDICTIONS_NAME_RE = re.compile(
    r"^abide_zeroshot_predictions_(?P<model>[^_]+(?:_[^_]+)*?)_seed_(?P<seed>\d+)\.csv$"
)


# ──────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────


def parse_summary_filename(filename: str) -> tuple[str, int] | None:
    """Extract (model, seed) from `abide_zeroshot_summary_{model}_seed_{N}.json`."""
    m = _SUMMARY_NAME_RE.match(filename)
    if m is None:
        return None
    return m.group("model"), int(m.group("seed"))


def parse_predictions_filename(filename: str) -> tuple[str, int] | None:
    """Extract (model, seed) from `abide_zeroshot_predictions_{model}_seed_{N}.csv`."""
    m = _PREDICTIONS_NAME_RE.match(filename)
    if m is None:
        return None
    return m.group("model"), int(m.group("seed"))


def load_summaries(summaries_dir: Path) -> list[dict[str, Any]]:
    """Glob + parse all summary JSONs under the directory.

    Returns a list of dicts with the JSON payload + parsed (model, seed).
    """
    out: list[dict[str, Any]] = []
    for path in sorted(summaries_dir.glob("abide_zeroshot_summary_*_seed_*.json")):
        meta = parse_summary_filename(path.name)
        if meta is None:
            logger.warning("Skipping unparseable name: %s", path.name)
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load %s: %s", path, exc)
            continue
        payload["_parsed_model"] = meta[0]
        payload["_parsed_seed"] = meta[1]
        payload["_path"] = str(path)
        out.append(payload)
    return out


def load_predictions(predictions_dir: Path) -> dict[tuple[str, int], pd.DataFrame]:
    """Glob predictions CSVs; key by (model, seed). DataFrame keyed by FILE_ID."""
    out: dict[tuple[str, int], pd.DataFrame] = {}
    for path in sorted(predictions_dir.glob("abide_zeroshot_predictions_*_seed_*.csv")):
        meta = parse_predictions_filename(path.name)
        if meta is None:
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load %s: %s", path, exc)
            continue
        out[meta] = df
    return out


# ──────────────────────────────────────────────
# Meta-table construction
# ──────────────────────────────────────────────


def build_meta_table(
    summaries: list[dict[str, Any]],
    variants: tuple[str, ...] = CONSENSUS_VARIANTS,
) -> pd.DataFrame:
    """Build the long-format meta table.

    Aggregates across seeds when multiple seeds per model exist:
        value = mean across seeds
        ci_low / ci_high = (value - SD, value + SD) when n_seeds > 1; else
                           the single-seed CI from the JSON.
    """
    rows: list[dict[str, Any]] = []

    # Group summaries by (model, variant, metric) for cross-seed aggregation.
    by_model: dict[str, list[dict[str, Any]]] = {}
    for s in summaries:
        by_model.setdefault(s["_parsed_model"], []).append(s)

    for model, model_summaries in by_model.items():
        n_seeds = len(model_summaries)
        for variant in variants:
            blocks = [
                s["variants"].get(variant) for s in model_summaries
                if s.get("variants", {}).get(variant) is not None
            ]
            if not blocks:
                continue
            # Per-metric aggregation across seeds.
            for metric_key, accessor, ci_extractor in [
                ("auc",
                 lambda b: b["auc"]["point"],
                 lambda b: (b["auc"]["ci_lower"], b["auc"]["ci_upper"])),
                ("accuracy",
                 lambda b: b["accuracy"]["point"],
                 lambda _b: (float("nan"), float("nan"))),
                ("sensitivity",
                 lambda b: b["sensitivity"]["point"],
                 lambda _b: (float("nan"), float("nan"))),
                ("specificity",
                 lambda b: b["specificity"]["point"],
                 lambda _b: (float("nan"), float("nan"))),
                ("cohens_kappa",
                 lambda b: b["cohens_kappa"]["point"],
                 lambda _b: (float("nan"), float("nan"))),
                ("loso_auc_mean",
                 lambda b: b["loso_auc_mean"],
                 lambda _b: (float("nan"), float("nan"))),
                ("loso_auc_std",
                 lambda b: b["loso_auc_std"],
                 lambda _b: (float("nan"), float("nan"))),
            ]:
                vals = [accessor(b) for b in blocks if accessor(b) is not None]
                vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
                if not vals:
                    continue
                if n_seeds == 1:
                    ci_low, ci_high = ci_extractor(blocks[0])
                    value = vals[0]
                else:
                    value = float(np.mean(vals))
                    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    ci_low, ci_high = value - sd, value + sd
                rows.append({
                    "model": model,
                    "variant": variant,
                    "metric": metric_key,
                    "value": value,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_seeds": n_seeds,
                })

        # Top-level (variant-independent) parse_failure_rate.
        pf_vals = [s["parse_failure_rate"] for s in model_summaries
                   if "parse_failure_rate" in s]
        if pf_vals:
            value = float(np.mean(pf_vals))
            sd = float(np.std(pf_vals, ddof=1)) if len(pf_vals) > 1 else 0.0
            rows.append({
                "model": model,
                "variant": "*",
                "metric": "parse_failure_rate",
                "value": value,
                "ci_low": value - sd,
                "ci_high": value + sd,
                "n_seeds": n_seeds,
            })

        # Krippendorff α (variant-C-only metric in 10).
        alpha_vals = [s.get("krippendorff_alpha_4way_variant_c")
                      for s in model_summaries]
        alpha_vals = [v for v in alpha_vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if alpha_vals:
            value = float(np.mean(alpha_vals))
            sd = float(np.std(alpha_vals, ddof=1)) if len(alpha_vals) > 1 else 0.0
            rows.append({
                "model": model,
                "variant": "C",
                "metric": "krippendorff_alpha_4way",
                "value": value,
                "ci_low": value - sd,
                "ci_high": value + sd,
                "n_seeds": n_seeds,
            })

    return pd.DataFrame(rows, columns=list(META_TABLE_COLUMNS))


# ──────────────────────────────────────────────
# Pairwise DeLong + BH FDR
# ──────────────────────────────────────────────


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR adjustment. Returns adjusted p-values aligned with input.

    p_adj[i] = min over k≥rank(i) of (p_sorted[k] * m / (k+1)), with monotonicity enforced.
    """
    arr = np.asarray(p_values, dtype=float)
    n = len(arr)
    if n == 0:
        return []
    finite = np.isfinite(arr)
    out = np.full(n, np.nan, dtype=float)
    valid = arr[finite]
    if valid.size == 0:
        return out.tolist()
    order = np.argsort(valid)
    ranked = valid[order]
    m = len(ranked)
    raw_adj = ranked * m / np.arange(1, m + 1)
    # Enforce monotonicity from the largest rank backward.
    monotone = np.minimum.accumulate(raw_adj[::-1])[::-1]
    monotone = np.clip(monotone, 0.0, 1.0)
    adj_sorted = np.empty(m, dtype=float)
    adj_sorted[order] = monotone
    j = 0
    for i in range(n):
        if finite[i]:
            out[i] = adj_sorted[j]
            j += 1
    return out.tolist()


def pairwise_delong(
    summaries: list[dict[str, Any]],
    predictions: dict[tuple[str, int], pd.DataFrame],
    variants: tuple[str, ...] = CONSENSUS_VARIANTS,
    delong_fn: Any = None,
) -> pd.DataFrame:
    """For every pair of models (single seed each, the lowest seed per model),
    run DeLong on the paired predictions for each variant. BH FDR within variant.

    Imports the DeLong implementation lazily from code/10 so we don't duplicate.
    """
    if delong_fn is None:
        # Lazy-import code/10's delong_test for shared implementation.
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(
            "_eval10_for_11", str(Path(__file__).resolve().parent / "10_eval_abide_zeroshot.py")
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load code/10 for DeLong implementation")
        mod10 = importlib.util.module_from_spec(spec)
        sys.modules["_eval10_for_11"] = mod10
        spec.loader.exec_module(mod10)
        delong_fn = mod10.delong_test

    # Pick one (lowest-seed) predictions df per model.
    by_model: dict[str, pd.DataFrame] = {}
    for (model, seed), df in sorted(predictions.items()):
        if model not in by_model:
            by_model[model] = df

    models = sorted(by_model.keys())
    rows: list[dict[str, Any]] = []
    for variant in variants:
        consensus_col = f"consensus_{variant.lower()}"
        rows_for_variant: list[dict[str, Any]] = []
        for i, ma in enumerate(models):
            for mb in models[i + 1:]:
                df_a, df_b = by_model[ma], by_model[mb]
                # Inner-join on FILE_ID for paired predictions.
                joined = df_a.merge(
                    df_b, on="FILE_ID", suffixes=("_a", "_b"), how="inner"
                )
                # Use either side's consensus_? column (they should agree).
                consensus_col_a = f"{consensus_col}_a"
                if consensus_col_a in joined.columns:
                    consensus = joined[consensus_col_a]
                elif consensus_col in joined.columns:
                    consensus = joined[consensus_col]
                else:
                    continue
                joined = joined[joined["vlm_score_a"].notna()
                                & joined["vlm_score_b"].notna()
                                & consensus.notna()]
                if len(joined) < 4:
                    continue
                consensus = consensus.loc[joined.index]
                result = delong_fn(
                    joined["vlm_score_a"].to_numpy(dtype=float),
                    joined["vlm_score_b"].to_numpy(dtype=float),
                    consensus.to_numpy(dtype=float),
                )
                rows_for_variant.append({
                    "model_a": ma, "model_b": mb, "variant": variant,
                    "auc_a": result["auc_a"], "auc_b": result["auc_b"],
                    "delta_auc": result["delta_auc"],
                    "delong_z": result["z"], "delong_p": result["p"],
                })
        # BH adjustment within variant.
        if rows_for_variant:
            adj = benjamini_hochberg([r["delong_p"] for r in rows_for_variant])
            for r, a in zip(rows_for_variant, adj, strict=True):
                r["bh_adjusted_p"] = a
        rows.extend(rows_for_variant)
    return pd.DataFrame(rows, columns=list(DELONG_TABLE_COLUMNS))


# ──────────────────────────────────────────────
# Top-line summary
# ──────────────────────────────────────────────


def build_meta_summary(
    meta_table: pd.DataFrame,
    delong_table: pd.DataFrame,
    alpha: float,
) -> dict[str, Any]:
    """Best model per variant + ranked AUC list + significance count post-FDR."""
    out: dict[str, Any] = {
        "alpha": alpha,
        "best_model_per_variant": {},
        "ranked_auc_per_variant": {},
        "significant_pairs_post_fdr": {},
    }
    for variant in CONSENSUS_VARIANTS:
        auc_rows = meta_table[
            (meta_table["variant"] == variant) & (meta_table["metric"] == "auc")
        ]
        if not auc_rows.empty:
            ranked = auc_rows.sort_values("value", ascending=False)
            out["best_model_per_variant"][variant] = ranked.iloc[0]["model"]
            out["ranked_auc_per_variant"][variant] = [
                {"model": row["model"], "auc": float(row["value"])}
                for _, row in ranked.iterrows()
            ]
        if not delong_table.empty:
            v_rows = delong_table[delong_table["variant"] == variant]
            sig = v_rows[v_rows["bh_adjusted_p"] < alpha]
            out["significant_pairs_post_fdr"][variant] = {
                "n_significant": int(len(sig)),
                "pairs": [
                    {"model_a": r["model_a"], "model_b": r["model_b"],
                     "delta_auc": float(r["delta_auc"]),
                     "bh_p": float(r["bh_adjusted_p"])}
                    for _, r in sig.iterrows()
                ],
            }
    return out


# ──────────────────────────────────────────────
# LaTeX writer
# ──────────────────────────────────────────────


def write_latex(meta_table: pd.DataFrame, output_path: Path) -> None:
    """Booktabs LaTeX fragment, one sub-table per consensus variant."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("% Auto-generated by code/11_compare_abide_zeroshot.py")
    lines.append("% One sub-table per consensus variant (A/B/C).")
    lines.append("")
    for variant in CONSENSUS_VARIANTS:
        lines.append(f"\\begin{{tabular}}{{lrrrrrr}}")
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\bfseries Variant {variant}}} \\\\")
        lines.append("\\toprule")
        lines.append("Model & AUC (95\\% CI) & Cohen's $\\kappa$ & Krippendorff $\\alpha$ "
                     "& LOSO mean $\\pm$ SD & Parse fail \\% & $n_{\\text{seeds}}$ \\\\")
        lines.append("\\midrule")
        for model in sorted(meta_table["model"].unique()):
            sub = meta_table[
                (meta_table["model"] == model) & (meta_table["variant"] == variant)
            ]
            auc_row = sub[sub["metric"] == "auc"]
            kappa_row = sub[sub["metric"] == "cohens_kappa"]
            loso_mean_row = sub[sub["metric"] == "loso_auc_mean"]
            loso_std_row = sub[sub["metric"] == "loso_auc_std"]
            alpha_row = meta_table[
                (meta_table["model"] == model)
                & (meta_table["metric"] == "krippendorff_alpha_4way")
                & (meta_table["variant"] == "C")
            ] if variant == "C" else pd.DataFrame()
            pf_row = meta_table[
                (meta_table["model"] == model)
                & (meta_table["metric"] == "parse_failure_rate")
            ]

            def _fmt(row, key="value", precision=3):
                if row.empty:
                    return "—"
                return f"{row.iloc[0][key]:.{precision}f}"

            def _fmt_auc(row):
                if row.empty:
                    return "—"
                v = row.iloc[0]
                return f"{v['value']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]"

            def _fmt_loso(mean_row, std_row):
                if mean_row.empty or std_row.empty:
                    return "—"
                return f"{mean_row.iloc[0]['value']:.3f} $\\pm$ {std_row.iloc[0]['value']:.3f}"

            def _fmt_pf_pct(row):
                if row.empty:
                    return "—"
                return f"{row.iloc[0]['value'] * 100:.1f}"

            n_seeds_val = int(auc_row.iloc[0]["n_seeds"]) if not auc_row.empty else 0
            lines.append(
                f"{model} & {_fmt_auc(auc_row)} & {_fmt(kappa_row)} & "
                f"{_fmt(alpha_row) if variant == 'C' else '—'} & "
                f"{_fmt_loso(loso_mean_row, loso_std_row)} & "
                f"{_fmt_pf_pct(pf_row)} & {n_seeds_val} \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("")
    output_path.write_text("\n".join(lines))


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


@app.command()
def main(
    summaries_dir: Path = typer.Option(
        Path("results/metrics"), "--summaries-dir", resolve_path=True
    ),
    predictions_dir: Path = typer.Option(
        Path("results/tables"), "--predictions-dir", resolve_path=True
    ),
    output_dir: Path = typer.Option(
        Path("results/metrics"), "--output-dir", resolve_path=True
    ),
    variants: str = typer.Option(",".join(CONSENSUS_VARIANTS), "--variants"),
    alpha: float = typer.Option(0.05, "--alpha", help="BH FDR level."),
    latex: bool = typer.Option(True, "--latex/--no-latex"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Aggregate per-model summaries + predictions into a cross-model meta table."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )

    selected_variants = tuple(v.strip() for v in variants.split(",") if v.strip())

    summaries = load_summaries(summaries_dir)
    predictions = load_predictions(predictions_dir)
    logger.info(
        "Loaded %d summary JSONs (across %d unique models) + %d predictions CSVs",
        len(summaries),
        len({s["_parsed_model"] for s in summaries}),
        len(predictions),
    )

    if dry_run:
        logger.info("--dry-run: no writes. Models found: %s",
                    sorted({s["_parsed_model"] for s in summaries}))
        return

    if not summaries:
        logger.warning("No summary JSONs found under %s; nothing to write", summaries_dir)
        return

    meta_table = build_meta_table(summaries, selected_variants)
    delong_table = pairwise_delong(summaries, predictions, selected_variants)
    meta_summary = build_meta_summary(meta_table, delong_table, alpha)

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_csv = output_dir / "abide_zeroshot_meta_table.csv"
    delong_csv = output_dir / "abide_zeroshot_pairwise_delong.csv"
    summary_json = output_dir / "abide_zeroshot_meta_summary.json"

    meta_table.to_csv(meta_csv, index=False)
    delong_table.to_csv(delong_csv, index=False)
    summary_json.write_text(json.dumps(meta_summary, indent=2, default=str))

    if latex:
        write_latex(meta_table, output_dir / "abide_zeroshot_meta_table.tex")

    console = Console()
    console.print(f"[bold]Meta table[/bold] → {meta_csv} ({len(meta_table)} rows)")
    console.print(f"[bold]Pairwise DeLong[/bold] → {delong_csv} ({len(delong_table)} rows)")
    console.print(f"[bold]Meta summary[/bold] → {summary_json}")


if __name__ == "__main__":
    app()
