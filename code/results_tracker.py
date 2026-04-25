#!/usr/bin/env python3
"""NeuroQC Results Tracker & Automation Pipeline

Run this script after each experimental phase to:
1. Aggregate results from CSVs into a unified dashboard
2. Compute all metrics (SRCC, PLCC, KRCC)
3. Generate updated tables and figures
4. Print a status report showing what's done and what's pending

Usage:
    python results_tracker.py --phase all     # full pipeline
    python results_tracker.py --phase 3       # after ground truth generation
    python results_tracker.py --phase 4       # after VLM evaluation
    python results_tracker.py --phase 5       # full analysis
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import typer
from rich.console import Console
from rich.table import Table as RichTable

app = typer.Typer(help="NeuroQC Results Tracker")
console = Console()
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

PROJECT_ROOT = Path.home() / "neuroqc"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = PROJECT_ROOT / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

EXPECTED_FILES: dict[str, list[Path]] = {
    "phase1": [TABLES_DIR / "reference_manifest.csv"],
    "phase2": [TABLES_DIR / "corruption_manifest.csv"],
    "phase3": [
        TABLES_DIR / "machine_preference.csv",
        TABLES_DIR / "iqm_features.csv",
    ],
    "phase4": [
        TABLES_DIR / "3d_vlm_scores.csv",
        TABLES_DIR / "2d_vlm_scores.csv",
    ],
    "phase5": [
        TABLES_DIR / "finetuned_scores.csv",
    ],
}


def check_phase_status() -> dict[str, dict]:
    """Check which phases have completed based on expected output files."""
    console.print("\n[bold]Phase Status Report[/bold]")
    console.print(f"Timestamp: {datetime.now().isoformat()}\n")

    status: dict[str, dict] = {}
    for phase, files in EXPECTED_FILES.items():
        completed = all(f.exists() for f in files)
        missing = [f.name for f in files if not f.exists()]
        status[phase] = {"completed": completed, "missing": missing}

        icon = "[green]DONE[/green]" if completed else "[red]PENDING[/red]"
        console.print(f"  {phase}: {icon}")
        if missing:
            for m in missing:
                console.print(f"    Missing: {m}")

    return status


def _load_dataset_tag_frame() -> pd.DataFrame | None:
    """Return a deduplicated ``[cor_path, dataset_tag]`` frame, or None.

    Source is ``results/tables/corruption_manifest.csv`` (phase 2 output).
    Returns None when the manifest is missing or predates the
    ``dataset_tag`` column; callers skip per-dataset splits in that case.
    """
    path = TABLES_DIR / "corruption_manifest.csv"
    if not path.exists():
        logger.warning(
            "corruption_manifest.csv not found at %s; skipping per-dataset splits.",
            path,
        )
        return None
    df = pd.read_csv(path)
    if "dataset_tag" not in df.columns:
        logger.warning(
            "corruption_manifest.csv has no dataset_tag column; "
            "skipping per-dataset splits. Re-run phase 2 to populate it."
        )
        return None
    return df[["cor_path", "dataset_tag"]].drop_duplicates()


def compute_correlations(
    predictions: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    """Compute SRCC, PLCC, KRCC between predictions and targets.

    Parameters
    ----------
    predictions, targets : torch.Tensor
        1-D float tensors of matched scores.

    Returns
    -------
    dict with keys: srcc, plcc, krcc, n
    """
    mask = ~(torch.isnan(predictions) | torch.isnan(targets))
    pred = predictions[mask]
    targ = targets[mask]
    n = len(pred)

    if n < 3:
        return {"srcc": float("nan"), "plcc": float("nan"),
                "krcc": float("nan"), "n": n}

    # PLCC (Pearson)
    pred_c = pred - pred.mean()
    targ_c = targ - targ.mean()
    plcc = (pred_c * targ_c).sum() / (pred_c.norm() * targ_c.norm() + 1e-8)

    # SRCC (Spearman) — rank correlation
    pred_ranks = pred.argsort().argsort().float()
    targ_ranks = targ.argsort().argsort().float()
    pred_rc = pred_ranks - pred_ranks.mean()
    targ_rc = targ_ranks - targ_ranks.mean()
    srcc = (pred_rc * targ_rc).sum() / (pred_rc.norm() * targ_rc.norm() + 1e-8)

    # KRCC (Kendall) — vectorised sign-based computation
    # For each pair (i, j) with i < j, check concordance
    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)  # (n, n)
    targ_diff = targ.unsqueeze(1) - targ.unsqueeze(0)
    # Upper triangle only (i < j)
    triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    products = torch.sign(pred_diff[triu_mask]) * torch.sign(targ_diff[triu_mask])
    concordant = (products > 0).sum().float()
    discordant = (products < 0).sum().float()
    total_pairs = n * (n - 1) / 2
    krcc = ((concordant - discordant) / total_pairs).item() if total_pairs > 0 else 0.0

    return {
        "srcc": srcc.item(),
        "plcc": plcc.item(),
        "krcc": krcc,
        "n": n,
    }


def analyze_phase3() -> pd.DataFrame | None:
    """Analyze ground truth: IQM vs machine preference correlations."""
    console.print("\n[bold cyan]Phase 3 Analysis: Signal Metrics vs Machine Preference[/bold cyan]")

    pref_path = TABLES_DIR / "machine_preference.csv"
    iqm_path = TABLES_DIR / "iqm_features.csv"

    if not pref_path.exists() or not iqm_path.exists():
        console.print("[red]Phase 3 data not available yet.[/red]")
        return None

    pref_df = pd.read_csv(pref_path)
    iqm_df = pd.read_csv(iqm_path)

    # iqm_features has "scan_path", machine_preference has "cor_path"
    merged = pref_df.merge(
        iqm_df, left_on="cor_path", right_on="scan_path", how="inner"
    )

    iqm_columns = ["snr", "cnr", "efc", "fber", "cjv"]
    target_columns = ["mean_dice"]
    if "hippocampus_dice" in merged.columns:
        target_columns.extend(["hippocampus_dice", "cortex_dice", "ventricle_dice"])

    results: list[dict] = []
    for iqm in iqm_columns:
        if iqm not in merged.columns:
            continue
        for target in target_columns:
            if target not in merged.columns:
                continue
            pred_t = torch.tensor(merged[iqm].values, dtype=torch.float32)
            targ_t = torch.tensor(merged[target].values, dtype=torch.float32)
            corr = compute_correlations(pred_t, targ_t)
            corr["predictor"] = iqm
            corr["target"] = target
            corr["dataset"] = "overall"
            results.append(corr)

    # Per-corruption correlations
    if "corruption_type" in merged.columns:
        for ctype in merged["corruption_type"].unique():
            subset = merged[merged["corruption_type"] == ctype]
            for iqm in iqm_columns:
                if iqm not in subset.columns:
                    continue
                pred_t = torch.tensor(subset[iqm].values, dtype=torch.float32)
                targ_t = torch.tensor(subset["mean_dice"].values, dtype=torch.float32)
                corr = compute_correlations(pred_t, targ_t)
                corr["predictor"] = iqm
                corr["target"] = f"mean_dice|{ctype}"
                corr["dataset"] = "overall"
                results.append(corr)

    # Per-dataset correlations (join dataset_tag from corruption_manifest on cor_path).
    tag_frame = _load_dataset_tag_frame()
    if tag_frame is not None:
        tagged = merged.merge(tag_frame, on="cor_path", how="left")
        for dtag in tagged["dataset_tag"].dropna().unique():
            subset = tagged[tagged["dataset_tag"] == dtag]
            for iqm in iqm_columns:
                if iqm not in subset.columns:
                    continue
                pred_t = torch.tensor(subset[iqm].values, dtype=torch.float32)
                targ_t = torch.tensor(subset["mean_dice"].values, dtype=torch.float32)
                corr = compute_correlations(pred_t, targ_t)
                corr["predictor"] = iqm
                corr["target"] = "mean_dice"
                corr["dataset"] = str(dtag)
                results.append(corr)

    results_df = pd.DataFrame(results)
    output_path = METRICS_DIR / "phase3_iqm_correlations.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    # Print summary
    table = RichTable(title="IQM vs Machine Preference (Overall)")
    table.add_column("IQM")
    table.add_column("SRCC", justify="right")
    table.add_column("PLCC", justify="right")
    table.add_column("KRCC", justify="right")

    overall = results_df[
        (results_df["target"] == "mean_dice") & (results_df["dataset"] == "overall")
    ]
    for _, row in overall.iterrows():
        table.add_row(
            row["predictor"],
            f"{row['srcc']:.4f}",
            f"{row['plcc']:.4f}",
            f"{row['krcc']:.4f}",
        )
    console.print(table)
    return results_df


def analyze_phase4() -> pd.DataFrame | None:
    """Analyze VLM evaluation results."""
    console.print("\n[bold cyan]Phase 4 Analysis: VLM Evaluation[/bold cyan]")

    pref_path = TABLES_DIR / "machine_preference.csv"
    if not pref_path.exists():
        console.print("[red]Machine preference data not available.[/red]")
        return None

    pref_df = pd.read_csv(pref_path)
    all_results: list[dict] = []

    for vlm_file in ["3d_vlm_scores.csv", "2d_vlm_scores.csv"]:
        vlm_path = TABLES_DIR / vlm_file
        if not vlm_path.exists():
            console.print(f"[yellow]Skipping {vlm_file} (not found)[/yellow]")
            continue

        vlm_df = pd.read_csv(vlm_path)
        merged = vlm_df.merge(pref_df, on="cor_path", how="inner")
        input_label = "3D" if "3d" in vlm_file else "2D"

        tag_frame = _load_dataset_tag_frame()
        tagged = (
            merged.merge(tag_frame, on="cor_path", how="left")
            if tag_frame is not None
            else None
        )

        for model_name in merged["model"].unique():
            model_subset = merged[merged["model"] == model_name]

            pred = torch.tensor(model_subset["predicted_score"].values, dtype=torch.float32)
            targ = torch.tensor(model_subset["mean_dice"].values, dtype=torch.float32)
            corr = compute_correlations(pred, targ)
            corr["model"] = model_name
            corr["input_type"] = input_label
            corr["target"] = "mean_dice"
            corr["dataset"] = "overall"
            all_results.append(corr)

            # Per-corruption
            if "corruption_type" in model_subset.columns:
                for ctype in model_subset["corruption_type"].unique():
                    ct_sub = model_subset[model_subset["corruption_type"] == ctype]
                    pred_ct = torch.tensor(ct_sub["predicted_score"].values, dtype=torch.float32)
                    targ_ct = torch.tensor(ct_sub["mean_dice"].values, dtype=torch.float32)
                    corr_ct = compute_correlations(pred_ct, targ_ct)
                    corr_ct["model"] = model_name
                    corr_ct["input_type"] = input_label
                    corr_ct["target"] = f"mean_dice|{ctype}"
                    corr_ct["dataset"] = "overall"
                    all_results.append(corr_ct)

            # Per-dataset
            if tagged is not None:
                model_tagged = tagged[tagged["model"] == model_name]
                for dtag in model_tagged["dataset_tag"].dropna().unique():
                    dt_sub = model_tagged[model_tagged["dataset_tag"] == dtag]
                    pred_dt = torch.tensor(
                        dt_sub["predicted_score"].values, dtype=torch.float32
                    )
                    targ_dt = torch.tensor(
                        dt_sub["mean_dice"].values, dtype=torch.float32
                    )
                    corr_dt = compute_correlations(pred_dt, targ_dt)
                    corr_dt["model"] = model_name
                    corr_dt["input_type"] = input_label
                    corr_dt["target"] = "mean_dice"
                    corr_dt["dataset"] = str(dtag)
                    all_results.append(corr_dt)

    if not all_results:
        console.print("[red]No VLM results available.[/red]")
        return None

    results_df = pd.DataFrame(all_results)
    output_path = METRICS_DIR / "phase4_vlm_correlations.csv"
    results_df.to_csv(output_path, index=False)

    # Print summary
    table = RichTable(title="VLM vs Machine Preference (Overall)")
    table.add_column("Model")
    table.add_column("Input")
    table.add_column("SRCC", justify="right")
    table.add_column("PLCC", justify="right")
    table.add_column("N", justify="right")

    overall = results_df[
        (results_df["target"] == "mean_dice") & (results_df["dataset"] == "overall")
    ].sort_values("srcc", ascending=False)
    for _, row in overall.iterrows():
        table.add_row(
            row["model"], row["input_type"],
            f"{row['srcc']:.4f}", f"{row['plcc']:.4f}", str(row["n"]),
        )
    console.print(table)
    return results_df


def analyze_phase5() -> pd.DataFrame | None:
    """Analyze fine-tuned model results and compare with baselines."""
    console.print("\n[bold cyan]Phase 5 Analysis: Fine-tuned vs Baselines[/bold cyan]")

    ft_path = TABLES_DIR / "finetuned_scores.csv"
    if not ft_path.exists():
        console.print("[red]Fine-tuned results not available yet.[/red]")
        return None

    ft_df = pd.read_csv(ft_path)
    pref_df = pd.read_csv(TABLES_DIR / "machine_preference.csv")
    merged = ft_df.merge(pref_df, on="cor_path", how="inner")

    tag_frame = _load_dataset_tag_frame()
    tagged = (
        merged.merge(tag_frame, on="cor_path", how="left")
        if tag_frame is not None
        else None
    )

    results: list[dict] = []
    for model_name in merged["model"].unique():
        model_subset = merged[merged["model"] == model_name]
        pred = torch.tensor(model_subset["predicted_score"].values, dtype=torch.float32)
        targ = torch.tensor(model_subset["mean_dice"].values, dtype=torch.float32)
        corr = compute_correlations(pred, targ)
        corr["model"] = model_name
        corr["stage"] = "fine-tuned"
        corr["target"] = "mean_dice"
        corr["dataset"] = "overall"
        results.append(corr)

        if tagged is not None:
            model_tagged = tagged[tagged["model"] == model_name]
            for dtag in model_tagged["dataset_tag"].dropna().unique():
                dt_sub = model_tagged[model_tagged["dataset_tag"] == dtag]
                pred_dt = torch.tensor(
                    dt_sub["predicted_score"].values, dtype=torch.float32
                )
                targ_dt = torch.tensor(
                    dt_sub["mean_dice"].values, dtype=torch.float32
                )
                corr_dt = compute_correlations(pred_dt, targ_dt)
                corr_dt["model"] = model_name
                corr_dt["stage"] = "fine-tuned"
                corr_dt["target"] = "mean_dice"
                corr_dt["dataset"] = str(dtag)
                results.append(corr_dt)

    results_df = pd.DataFrame(results)
    output_path = METRICS_DIR / "phase5_finetuned_correlations.csv"
    results_df.to_csv(output_path, index=False)

    # Combine all phases for comparison. Keep all mean_dice rows (overall +
    # per-dataset); downstream consumers filter on the "dataset" column.
    all_metrics: list[pd.DataFrame] = []
    for phase_file in [
        "phase3_iqm_correlations.csv",
        "phase4_vlm_correlations.csv",
        "phase5_finetuned_correlations.csv",
    ]:
        fpath = METRICS_DIR / phase_file
        if fpath.exists():
            df = pd.read_csv(fpath)
            if "target" in df.columns:
                all_metrics.append(df[df["target"] == "mean_dice"])

    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True)
        combined_path = METRICS_DIR / "all_correlations_combined.csv"
        combined.to_csv(combined_path, index=False)
        console.print(f"\n[green]Combined metrics saved to {combined_path}[/green]")

    return results_df


def generate_paper_tables() -> None:
    """Generate LaTeX-formatted tables for the paper."""
    console.print("\n[bold cyan]Generating Paper Tables[/bold cyan]")

    combined_path = METRICS_DIR / "all_correlations_combined.csv"
    if not combined_path.exists():
        console.print("[red]Run full analysis first (--phase 5)[/red]")
        return

    df = pd.read_csv(combined_path)
    if "dataset" in df.columns:
        df = df[df["dataset"] == "overall"]

    latex_lines = [
        r"\begin{table}[t]",
        r"\caption{Prediction of machine preference (SynthSeg Dice degradation) by different methods.}",
        r"\label{tab:main_results}",
        r"\centering",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Method & Input & SRCC$\uparrow$ & PLCC$\uparrow$ & KRCC$\uparrow$ \\",
        r"\midrule",
    ]

    df_sorted = df.sort_values("srcc", ascending=False)
    for _, row in df_sorted.iterrows():
        name = row.get("model", row.get("predictor", "unknown"))
        input_type = row.get("input_type", "signal")
        srcc = f"{row['srcc']:.4f}"
        plcc = f"{row['plcc']:.4f}"
        krcc = f"{row['krcc']:.4f}"
        latex_lines.append(f"{name} & {input_type} & {srcc} & {plcc} & {krcc} \\\\")

    latex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    output_path = RESULTS_DIR / "latex" / "table_main_results.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(latex_lines))
    console.print(f"[green]LaTeX table saved to {output_path}[/green]")


@app.command()
def main(
    phase: str = typer.Option(
        "status",
        help="Which phase to run: status, 3, 4, 5, all, tables",
    ),
) -> None:
    """NeuroQC Results Tracker — compute metrics and generate tables."""
    logging.basicConfig(level=logging.INFO)

    for d in [RESULTS_DIR, TABLES_DIR, FIGURES_DIR, METRICS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    check_phase_status()

    if phase in ("3", "all"):
        analyze_phase3()
    if phase in ("4", "all"):
        analyze_phase4()
    if phase in ("5", "all"):
        analyze_phase5()
    if phase in ("tables", "all"):
        generate_paper_tables()

    console.print("\n[bold green]Analysis complete.[/bold green]")
    console.print(f"Metrics saved to: {METRICS_DIR}")
    console.print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    app()
