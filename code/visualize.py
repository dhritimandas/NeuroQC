#!/usr/bin/env python3
"""NeuroQC Phase 5 — publication-quality figure generator.

Reads results CSVs from ``results/tables/`` and writes paired ``.png`` + ``.svg``
figures to ``figures/`` for the NeurIPS submission. Six figures are auto-
generated; Figs 1 (pipeline overview) and 7 (architecture) are hand-drawn
and out of scope for this script. LaTeX tables go to a separate
``results_tracker.py``.

Figures::

    Fig 2  fig_iqm_heatmap            5 IQMs × 8 corruption types, SRCC heatmap
    Fig 3  fig_corruption_sensitivity 8 corruptions × 5 severities, Dice drop bars
    Fig 4  fig_vlm_scatter            scatter grid (one per VLM model) vs Dice
    Fig 5  fig_3d_vs_2d               3D vs 2D mean SRCC per corruption + sig stars
    Fig 6  fig_finetuned              zero-shot vs fine-tuned SRCC, paired bars
    Fig 8  fig_per_structure          ≥35 structures × 8 corruptions, Dice heatmap

Style:
    IBM colorblind-safe palette; NeurIPS column widths (3.25 / 6.875 in);
    DejaVu Sans 10pt body. PNG + SVG written atomically (``.tmp`` rename).
    Multi-seed score CSVs aggregated to per-(scan, model) mean ± 95% CI;
    Spearman correlations reported with bootstrap-percentile 95% CIs.

Inputs (validated on startup; abort with actionable error if missing for the
requested figure):

    Fig 2: iqm_features.csv, machine_preference.csv
    Fig 3: machine_preference.csv, corruption_manifest.csv
    Fig 4: 3d_vlm_scores_seed_*.csv, 2d_vlm_scores_seed_*.csv, machine_preference.csv
    Fig 5: Fig 4's inputs + corruption_manifest.csv
    Fig 6: finetuned_scores_seed_*.csv + Fig 4's inputs
    Fig 8: per_structure_dice.csv  (produced by code/04_compute_preference.py
                                    --per-structure-output)

Interactive:
    ``--interactive`` is reserved for future Plotly HTML; in v1 the flag logs
    a WARNING and proceeds with PNG/SVG only. Implementing interactive
    dashboards well is a day's work; that day is better spent on writing.
"""

from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# Headless backend before pyplot import — keeps tests + CI fast on no-display
# machines and avoids GTK/Tk dependency surprises.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ──────────────────────────────────────────────
# Constants — style
# ──────────────────────────────────────────────

# IBM Design Language colorblind-safe 5-color palette.
IBM_PALETTE: tuple[str, ...] = (
    "#648FFF",
    "#785EF0",
    "#DC267F",
    "#FE6100",
    "#FFB000",
)

# NeurIPS column widths (inches).
COLUMN_WIDTH_IN: float = 3.25
DOUBLE_WIDTH_IN: float = 6.875

# Output formats per figure.
OUTPUT_FORMATS: tuple[str, ...] = ("png", "svg")

# Default rcParams. Applied once at module import; per-figure overrides are
# scoped via ``with plt.rc_context(...)`` if needed.
_DEFAULT_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
}
plt.rcParams.update(_DEFAULT_RC)

# ──────────────────────────────────────────────
# Constants — IQMs, corruption types, structures
# ──────────────────────────────────────────────

IQM_COLUMNS: tuple[str, ...] = ("snr", "cnr", "efc", "fber", "cjv")
CORRUPTION_TYPES: tuple[str, ...] = (
    "motion",
    "ghosting",
    "spike",
    "noise",
    "bias_field",
    "blur",
    "downsample",
    "gamma",
)
SEVERITIES: tuple[int, ...] = (1, 2, 3, 4, 5)

# Plot-y labels (display) vs raw IQM names (data).
_IQM_DISPLAY: dict[str, str] = {
    "snr": "SNR",
    "cnr": "CNR",
    "efc": "EFC",
    "fber": "FBER",
    "cjv": "CJV",
}

# ──────────────────────────────────────────────
# Constants — file naming
# ──────────────────────────────────────────────

FIG_FILENAMES: dict[int, str] = {
    2: "fig_iqm_heatmap",
    3: "fig_corruption_sensitivity",
    4: "fig_vlm_scatter",
    5: "fig_3d_vs_2d",
    6: "fig_finetuned",
    8: "fig_per_structure",
}
SUPPORTED_FIGURES: tuple[int, ...] = tuple(FIG_FILENAMES.keys())

# Per-figure required CSV inputs (relative to ``results-dir``). Glob patterns
# allowed; absence aborts the figure with an actionable error.
FIGURE_INPUTS: dict[int, tuple[str, ...]] = {
    2: ("iqm_features.csv", "machine_preference.csv"),
    3: ("machine_preference.csv", "corruption_manifest.csv"),
    4: ("3d_vlm_scores_seed_*.csv", "2d_vlm_scores_seed_*.csv", "machine_preference.csv"),
    5: (
        "3d_vlm_scores_seed_*.csv",
        "2d_vlm_scores_seed_*.csv",
        "machine_preference.csv",
        "corruption_manifest.csv",
    ),
    6: (
        "finetuned_scores_seed_*.csv",
        "3d_vlm_scores_seed_*.csv",
        "2d_vlm_scores_seed_*.csv",
        "machine_preference.csv",
    ),
    8: ("per_structure_dice.csv",),
}

# Where to point the user when an input is missing.
INPUT_PRODUCERS: dict[str, str] = {
    "iqm_features.csv": "code/05_extract_iqms.py",
    "machine_preference.csv": "code/04_compute_preference.py",
    "corruption_manifest.csv": "code/02_generate_corruptions.py / code/02b_corrupt_kspace_motion.py",
    "3d_vlm_scores_seed_*.csv": "code/08a_eval_3d_vlms.py",
    "2d_vlm_scores_seed_*.csv": "code/08b_eval_2d_vlms.py",
    "finetuned_scores_seed_*.csv": "code/09_finetune_lora.py",
    "per_structure_dice.csv": "code/04_compute_preference.py --per-structure-output",
}

# CSV column constants (must match what upstream phases write).
SCAN_COLUMN: str = "scan_path"
MODEL_COLUMN: str = "model"
SCORE_COLUMN: str = "score"
SEED_COLUMN: str = "seed"
PREF_REF_COLUMN: str = "ref_path"
PREF_COR_COLUMN: str = "cor_path"
PREF_DICE_COLUMN: str = "mean_dice"
TYPE_COLUMN: str = "corruption_type"
SEVERITY_COLUMN: str = "severity"
DICE_QUALITY_COLUMN: str = "dice_quality"
PS_LABEL_NAME_COLUMN: str = "label_name"
PS_DICE_COLUMN: str = "dice"

# Bootstrap defaults.
DEFAULT_N_BOOTSTRAP: int = 1000
DEFAULT_BOOTSTRAP_SEED: int = 42

logger = logging.getLogger(__name__)
app = typer.Typer(
    help="NeuroQC Phase 5 — figure generator (PNG + SVG; 6 auto figures).",
    add_completion=False,
)


# ──────────────────────────────────────────────
# Glob + multi-seed aggregation
# ──────────────────────────────────────────────


_SEED_REGEX = re.compile(r"_seed_(\d+)\.csv$")


def _glob_seed_files(results_dir: Path, pattern: str) -> list[tuple[Path, int]]:
    """Return ``[(path, seed)]`` for every ``*_seed_<N>.csv`` matching ``pattern``."""
    out: list[tuple[Path, int]] = []
    for p in sorted(results_dir.glob(pattern)):
        m = _SEED_REGEX.search(p.name)
        if m is None:
            logger.warning("Skipping %s — no seed suffix", p.name)
            continue
        out.append((p, int(m.group(1))))
    return out


def aggregate_seed_csvs(
    paths_and_seeds: list[tuple[Path, int]],
    score_column: str = SCORE_COLUMN,
    extra_keys: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Concat per-seed score CSVs, group by (scan, model, *extra_keys), aggregate.

    Returns a frame with columns
    ``[*group_keys, score_mean, score_std, n_seeds, ci_half_width]``.
    The 95 % CI half-width assumes normality:
    ``1.96 * std / sqrt(n_seeds)`` (set to NaN when ``n_seeds == 1``).
    """
    if not paths_and_seeds:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path, seed in paths_and_seeds:
        df = pd.read_csv(path)
        df[SEED_COLUMN] = seed
        frames.append(df)
    stacked = pd.concat(frames, ignore_index=True)
    group_keys = [SCAN_COLUMN, MODEL_COLUMN, *extra_keys]
    grouped = (
        stacked.groupby(group_keys, dropna=False)[score_column]
        .agg(score_mean="mean", score_std="std", n_seeds="count")
        .reset_index()
    )
    grouped["ci_half_width"] = grouped.apply(
        lambda r: (
            1.96 * r["score_std"] / math.sqrt(r["n_seeds"])
            if r["n_seeds"] > 1 and pd.notna(r["score_std"])
            else float("nan")
        ),
        axis=1,
    )
    return grouped


# ──────────────────────────────────────────────
# SRCC + bootstrap CI
# ──────────────────────────────────────────────


def srcc_with_bootstrap_ci(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Spearman rank correlation with bootstrap-percentile 95 % CI.

    scipy is required here because torch and pandas don't expose Spearman
    directly (pandas does via Series.corr but doesn't return the p-value or
    let us bootstrap cheaply). The bootstrap loop uses
    ``np.random.default_rng(seed)`` so two calls with the same seed produce
    byte-identical CI bounds — load-bearing for paper reproducibility.

    Returns:
        ``(srcc, ci_low, ci_high)``. Any of the three is NaN if there are
        fewer than 2 non-NaN paired observations.
    """
    from scipy.stats import spearmanr

    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(y, dtype=float)
    mask = np.isfinite(arr_x) & np.isfinite(arr_y)
    arr_x = arr_x[mask]
    arr_y = arr_y[mask]
    n = arr_x.size
    if n < 2:
        return math.nan, math.nan, math.nan
    point = float(spearmanr(arr_x, arr_y).statistic)

    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        samples[i] = spearmanr(arr_x[idx], arr_y[idx]).statistic
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return point, math.nan, math.nan
    lo = float(np.percentile(samples, 2.5))
    hi = float(np.percentile(samples, 97.5))
    return point, lo, hi


def format_srcc_annotation(srcc: float, lo: float, hi: float) -> str:
    """Render ``"SRCC = 0.NN [lo, hi]"`` for a figure annotation."""
    if not np.isfinite(srcc):
        return "SRCC = n/a"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"SRCC = {srcc:.2f}"
    return f"SRCC = {srcc:.2f} [{lo:.2f}, {hi:.2f}]"


# ──────────────────────────────────────────────
# Atomic write — PNG + SVG
# ──────────────────────────────────────────────


def _save_figure_atomic(fig: plt.Figure, output_path_no_suffix: Path) -> list[Path]:
    """Save ``fig`` as both PNG and SVG with atomic ``.tmp`` rename.

    On any savefig exception the partial ``.tmp`` files are cleaned up so the
    final output is either both formats present or neither — never half. Returns
    the list of final paths actually written.
    """
    output_path_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    tmp_paths: list[Path] = []
    try:
        for ext in OUTPUT_FORMATS:
            final = output_path_no_suffix.with_suffix(f".{ext}")
            tmp = final.with_suffix(f".{ext}.tmp")
            tmp_paths.append(tmp)
            fig.savefig(tmp, format=ext)
            os.replace(tmp, final)
            written.append(final)
    except Exception:
        # Cleanup any tmp leftover so the test's atomic-write invariant holds.
        for tmp in tmp_paths:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
        # Also delete any final files we wrote before the failure to avoid
        # half-finished output on disk.
        for final in written:
            if final.exists():
                try:
                    final.unlink()
                except OSError:
                    pass
        raise
    return written


# ──────────────────────────────────────────────
# Helpers for joining preference + scores
# ──────────────────────────────────────────────


def _annotate_with_dice(
    scores: pd.DataFrame, preference: pd.DataFrame
) -> pd.DataFrame:
    """Merge ``scan_path`` against preference's ``cor_path`` for the Dice ground truth.

    For clean (reference) scans there is no preference row by construction
    (``Dice(seg(ref), seg(ref)) = 1`` is structural, not measured). Those rows
    get ``mean_dice = 1.0`` — the structural rule documented in the paper's
    methods section.
    """
    merged = scores.merge(
        preference[[PREF_COR_COLUMN, PREF_DICE_COLUMN]],
        left_on=SCAN_COLUMN,
        right_on=PREF_COR_COLUMN,
        how="left",
    )
    merged.loc[merged[PREF_DICE_COLUMN].isna(), PREF_DICE_COLUMN] = 1.0
    return merged


# ──────────────────────────────────────────────
# Figure 2 — IQM × corruption SRCC heatmap
# ──────────────────────────────────────────────


def fig_iqm_heatmap(
    iqm_df: pd.DataFrame,
    pref_df: pd.DataFrame,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> plt.Figure:
    """Render the IQM × corruption SRCC heatmap (Fig 2)."""
    iqm_with_dice = iqm_df.merge(
        pref_df[[PREF_COR_COLUMN, PREF_DICE_COLUMN]],
        left_on=SCAN_COLUMN,
        right_on=PREF_COR_COLUMN,
        how="left",
    )
    iqm_with_dice.loc[
        iqm_with_dice[PREF_DICE_COLUMN].isna(), PREF_DICE_COLUMN
    ] = 1.0

    n_iqms = len(IQM_COLUMNS)
    n_corruptions = len(CORRUPTION_TYPES)
    grid = np.full((n_iqms, n_corruptions), np.nan, dtype=float)
    sig_mask = np.zeros((n_iqms, n_corruptions), dtype=bool)

    for j, ctype in enumerate(CORRUPTION_TYPES):
        sub = iqm_with_dice[iqm_with_dice[TYPE_COLUMN] == ctype]
        for i, iqm in enumerate(IQM_COLUMNS):
            if iqm not in sub.columns:
                continue
            srcc, lo, hi = srcc_with_bootstrap_ci(
                sub[iqm], sub[PREF_DICE_COLUMN],
                n_bootstrap=n_bootstrap, seed=bootstrap_seed,
            )
            grid[i, j] = srcc
            # CI brackets 0 → not significant → hatch overlay.
            sig_mask[i, j] = (
                np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)
            )

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.5))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_corruptions))
    ax.set_xticklabels(CORRUPTION_TYPES, rotation=45, ha="right")
    ax.set_yticks(range(n_iqms))
    ax.set_yticklabels([_IQM_DISPLAY.get(k, k) for k in IQM_COLUMNS])
    for i in range(n_iqms):
        for j in range(n_corruptions):
            value = grid[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
            if not sig_mask[i, j] and np.isfinite(value):
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, hatch="///", edgecolor="gray", linewidth=0,
                    )
                )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04)
    cbar.set_label("SRCC vs mean_dice")
    ax.set_title("IQM ↔ Dice rank correlation")
    return fig


# ──────────────────────────────────────────────
# Figure 3 — corruption × severity Dice drop
# ──────────────────────────────────────────────


def fig_corruption_sensitivity(
    pref_df: pd.DataFrame,
    cor_manifest: pd.DataFrame,
) -> plt.Figure:
    """Render the corruption × severity Dice-drop bars (Fig 3)."""
    # Merge severity from cor_manifest if not already on pref_df.
    if SEVERITY_COLUMN not in pref_df.columns:
        pref_df = pref_df.merge(
            cor_manifest[[PREF_COR_COLUMN, SEVERITY_COLUMN]],
            on=PREF_COR_COLUMN, how="left",
        )
    # Dice DROP from clean = 1 - mean_dice (structural baseline = 1.0).
    pref_df = pref_df.copy()
    pref_df["dice_drop"] = 1.0 - pref_df[PREF_DICE_COLUMN].astype(float)

    fig, ax = plt.subplots(figsize=(DOUBLE_WIDTH_IN, 3.0))
    bar_width = 0.15
    x_positions = np.arange(len(CORRUPTION_TYPES))

    for s_idx, sev in enumerate(SEVERITIES):
        means = []
        stds = []
        for ctype in CORRUPTION_TYPES:
            sub = pref_df[
                (pref_df[TYPE_COLUMN] == ctype) & (pref_df[SEVERITY_COLUMN] == sev)
            ]
            if sub.empty:
                means.append(0.0)
                stds.append(0.0)
            else:
                means.append(float(sub["dice_drop"].mean()))
                stds.append(float(sub["dice_drop"].std(ddof=0)))
        offset = (s_idx - (len(SEVERITIES) - 1) / 2) * bar_width
        ax.bar(
            x_positions + offset, means, bar_width,
            yerr=stds, color=IBM_PALETTE[s_idx % len(IBM_PALETTE)],
            label=f"sev {sev}", capsize=2, ecolor="dimgray",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(CORRUPTION_TYPES, rotation=45, ha="right")
    ax.set_ylabel("1 − mean_dice  (Dice drop from clean)")
    ax.set_title("Corruption sensitivity by severity")
    ax.legend(loc="upper left", ncol=len(SEVERITIES), frameon=False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    return fig


# ──────────────────────────────────────────────
# Figure 4 — VLM scatter grid
# ──────────────────────────────────────────────


def fig_vlm_scatter(
    vlm_3d: pd.DataFrame,
    vlm_2d: pd.DataFrame,
    pref_df: pd.DataFrame,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> plt.Figure:
    """Render scatter plots (one per VLM model) of predicted vs ground-truth Dice (Fig 4)."""
    combined = pd.concat([vlm_3d, vlm_2d], ignore_index=True, sort=False)
    if combined.empty:
        raise ValueError("No VLM scores available for Fig 4 (3D+2D both empty).")
    combined = _annotate_with_dice(combined, pref_df)

    models = sorted(combined[MODEL_COLUMN].dropna().unique())
    n_models = len(models)
    n_cols = min(n_models, 4) if n_models else 1
    n_rows = max(1, math.ceil(n_models / n_cols)) if n_models else 1

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(DOUBLE_WIDTH_IN, max(2.0, 2.0 * n_rows)),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax_idx, model in enumerate(models):
        ax = axes_flat[ax_idx]
        sub = combined[combined[MODEL_COLUMN] == model]
        x = sub[PREF_DICE_COLUMN].astype(float)
        y_col = "score_mean" if "score_mean" in sub.columns else SCORE_COLUMN
        y = sub[y_col].astype(float)
        ax.scatter(x, y, s=10, alpha=0.6, color=IBM_PALETTE[ax_idx % len(IBM_PALETTE)])
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        # Linear regression overlay + R²
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() >= 2:
            slope, intercept = np.polyfit(x[valid], y[valid], 1)
            xs = np.linspace(0, 1, 50)
            ax.plot(xs, slope * xs + intercept, color="black", linewidth=0.8)
            y_pred = slope * x[valid] + intercept
            ss_res = float(np.sum((y[valid] - y_pred) ** 2))
            ss_tot = float(np.sum((y[valid] - np.mean(y[valid])) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        else:
            r2 = float("nan")
        srcc, lo, hi = srcc_with_bootstrap_ci(
            x, y, n_bootstrap=n_bootstrap, seed=bootstrap_seed
        )
        annotation = format_srcc_annotation(srcc, lo, hi)
        if np.isfinite(r2):
            annotation = f"{annotation}\nR² = {r2:.2f}"
        ax.text(
            0.03, 0.97, annotation,
            transform=ax.transAxes, ha="left", va="top", fontsize=7,
            bbox={"boxstyle": "round", "fc": "white", "ec": "lightgray", "alpha": 0.85},
        )
        ax.set_title(model, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("mean_dice")
        ax.set_ylabel("VLM score")
        ax.grid(True, alpha=0.3, linestyle="--")

    # Hide any extra subplots if n_models doesn't fill the grid.
    for k in range(n_models, len(axes_flat)):
        axes_flat[k].axis("off")

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
# Figure 5 — 3D vs 2D mean SRCC per corruption
# ──────────────────────────────────────────────


def fig_3d_vs_2d(
    vlm_3d: pd.DataFrame,
    vlm_2d: pd.DataFrame,
    pref_df: pd.DataFrame,
    cor_manifest: pd.DataFrame,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> plt.Figure:
    """Render the 3D vs 2D SRCC bar comparison per corruption type (Fig 5)."""
    def _attach_corruption(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = _annotate_with_dice(df, pref_df)
        out = out.merge(
            cor_manifest[[PREF_COR_COLUMN, TYPE_COLUMN]],
            left_on=SCAN_COLUMN, right_on=PREF_COR_COLUMN, how="left",
        )
        return out

    df3 = _attach_corruption(vlm_3d)
    df2 = _attach_corruption(vlm_2d)

    fig, ax = plt.subplots(figsize=(DOUBLE_WIDTH_IN, 3.0))
    bar_width = 0.35
    x_positions = np.arange(len(CORRUPTION_TYPES))

    def _per_corruption_srcc(df: pd.DataFrame) -> tuple[list[float], list[float]]:
        means: list[float] = []
        ci_widths: list[float] = []
        for ctype in CORRUPTION_TYPES:
            sub = df[df[TYPE_COLUMN] == ctype] if not df.empty else df
            if sub.empty:
                means.append(0.0)
                ci_widths.append(0.0)
                continue
            y_col = "score_mean" if "score_mean" in sub.columns else SCORE_COLUMN
            srcc, lo, hi = srcc_with_bootstrap_ci(
                sub[PREF_DICE_COLUMN], sub[y_col],
                n_bootstrap=n_bootstrap, seed=bootstrap_seed,
            )
            means.append(srcc if np.isfinite(srcc) else 0.0)
            half = (
                (hi - lo) / 2.0
                if np.isfinite(lo) and np.isfinite(hi)
                else 0.0
            )
            ci_widths.append(half)
        return means, ci_widths

    means_3d, ci_3d = _per_corruption_srcc(df3)
    means_2d, ci_2d = _per_corruption_srcc(df2)

    ax.bar(
        x_positions - bar_width / 2, means_3d, bar_width,
        yerr=ci_3d, color=IBM_PALETTE[0], label="3D mean SRCC",
        capsize=2, ecolor="dimgray",
    )
    ax.bar(
        x_positions + bar_width / 2, means_2d, bar_width,
        yerr=ci_2d, color=IBM_PALETTE[2], label="2D mean SRCC",
        capsize=2, ecolor="dimgray",
    )

    # Significance stars: paired-bootstrap on per-scan score difference.
    for i, ctype in enumerate(CORRUPTION_TYPES):
        try:
            p = _paired_bootstrap_pvalue(
                df3, df2, ctype, n_bootstrap=n_bootstrap, seed=bootstrap_seed,
            )
        except ValueError:
            continue
        stars = ""
        if p < 0.001:
            stars = "***"
        elif p < 0.01:
            stars = "**"
        elif p < 0.05:
            stars = "*"
        if stars:
            top = max(means_3d[i] + ci_3d[i], means_2d[i] + ci_2d[i])
            ax.text(i, top + 0.02, stars, ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(CORRUPTION_TYPES, rotation=45, ha="right")
    ax.set_ylabel("Mean SRCC vs mean_dice")
    ax.set_title("3D vs 2D VLM rank correlation by corruption")
    ax.legend(frameon=False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    return fig


def _paired_bootstrap_pvalue(
    df3: pd.DataFrame,
    df2: pd.DataFrame,
    ctype: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> float:
    """Two-sided p-value for ``mean(score_3d) − mean(score_2d)`` via paired bootstrap."""
    if df3.empty or df2.empty:
        raise ValueError("empty input")
    s3 = df3[df3[TYPE_COLUMN] == ctype]
    s2 = df2[df2[TYPE_COLUMN] == ctype]
    if s3.empty or s2.empty:
        raise ValueError(f"no rows for corruption {ctype!r}")
    y_col_3 = "score_mean" if "score_mean" in s3.columns else SCORE_COLUMN
    y_col_2 = "score_mean" if "score_mean" in s2.columns else SCORE_COLUMN
    # Pair on scan_path; only keep scans present in both.
    paired = pd.merge(
        s3[[SCAN_COLUMN, y_col_3]].rename(columns={y_col_3: "y3"}),
        s2[[SCAN_COLUMN, y_col_2]].rename(columns={y_col_2: "y2"}),
        on=SCAN_COLUMN, how="inner",
    ).dropna(subset=["y3", "y2"])
    if len(paired) < 4:
        raise ValueError("too few paired observations")
    diffs = paired["y3"].to_numpy() - paired["y2"].to_numpy()
    rng = np.random.default_rng(seed)
    bootstrapped = np.empty(n_bootstrap, dtype=float)
    n = len(diffs)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        bootstrapped[i] = diffs[idx].mean()
    observed = float(diffs.mean())
    centred = bootstrapped - observed
    p = float(np.mean(np.abs(centred) >= abs(observed)))
    return p


# ──────────────────────────────────────────────
# Figure 6 — zero-shot vs fine-tuned per model
# ──────────────────────────────────────────────


def fig_finetuned(
    vlm_3d_zs: pd.DataFrame,
    vlm_2d_zs: pd.DataFrame,
    finetuned: pd.DataFrame,
    pref_df: pd.DataFrame,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> plt.Figure:
    """Render zero-shot vs fine-tuned SRCC paired bars per model (Fig 6)."""
    zs_combined = pd.concat([vlm_3d_zs, vlm_2d_zs], ignore_index=True, sort=False)
    zs_combined = _annotate_with_dice(zs_combined, pref_df)
    finetuned = _annotate_with_dice(finetuned, pref_df)

    # Aggregated zero-shot scores live in score_mean ∈ [0, 1].
    # Aggregated fine-tuned scores live in score_mean ∈ [1, 5] (Likert mean
    # over seeds). Normalise the latter to [0, 1] for the paired comparison.
    if "score_mean" in finetuned.columns:
        finetuned = finetuned.copy()
        finetuned["score_mean"] = (finetuned["score_mean"].astype(float) - 1.0) / 4.0
    elif DICE_QUALITY_COLUMN in finetuned.columns:
        finetuned = finetuned.copy()
        finetuned[SCORE_COLUMN] = (
            finetuned[DICE_QUALITY_COLUMN].astype(float) - 1.0
        ) / 4.0

    models = sorted(set(zs_combined[MODEL_COLUMN].dropna().unique())
                    | set(finetuned[MODEL_COLUMN].dropna().unique()))

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 3.0))
    bar_width = 0.35
    x_positions = np.arange(len(models))

    zs_means: list[float] = []
    zs_widths: list[float] = []
    ft_means: list[float] = []
    ft_widths: list[float] = []
    deltas: list[float] = []

    for model in models:
        zs_sub = zs_combined[zs_combined[MODEL_COLUMN] == model]
        ft_sub = finetuned[finetuned[MODEL_COLUMN] == model]
        zs_score_col = "score_mean" if "score_mean" in zs_sub.columns else SCORE_COLUMN
        ft_score_col = "score_mean" if "score_mean" in ft_sub.columns else SCORE_COLUMN

        zs_srcc, zs_lo, zs_hi = srcc_with_bootstrap_ci(
            zs_sub[PREF_DICE_COLUMN], zs_sub[zs_score_col],
            n_bootstrap=n_bootstrap, seed=bootstrap_seed,
        )
        ft_srcc, ft_lo, ft_hi = srcc_with_bootstrap_ci(
            ft_sub[PREF_DICE_COLUMN], ft_sub[ft_score_col],
            n_bootstrap=n_bootstrap, seed=bootstrap_seed,
        )
        zs_means.append(zs_srcc if np.isfinite(zs_srcc) else 0.0)
        ft_means.append(ft_srcc if np.isfinite(ft_srcc) else 0.0)
        zs_widths.append(
            (zs_hi - zs_lo) / 2.0 if np.isfinite(zs_lo) and np.isfinite(zs_hi) else 0.0
        )
        ft_widths.append(
            (ft_hi - ft_lo) / 2.0 if np.isfinite(ft_lo) and np.isfinite(ft_hi) else 0.0
        )
        deltas.append(
            (ft_srcc - zs_srcc)
            if np.isfinite(ft_srcc) and np.isfinite(zs_srcc)
            else 0.0
        )

    ax.bar(
        x_positions - bar_width / 2, zs_means, bar_width,
        yerr=zs_widths, color=IBM_PALETTE[0], label="zero-shot",
        capsize=2, ecolor="dimgray",
    )
    ax.bar(
        x_positions + bar_width / 2, ft_means, bar_width,
        yerr=ft_widths, color=IBM_PALETTE[2], label="fine-tuned",
        capsize=2, ecolor="dimgray",
    )
    for i, delta in enumerate(deltas):
        sign = "+" if delta >= 0 else "−"
        top = max(zs_means[i] + zs_widths[i], ft_means[i] + ft_widths[i])
        ax.text(
            i, top + 0.02, f"Δ {sign}{abs(delta):.2f}",
            ha="center", va="bottom", fontsize=7,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("SRCC vs mean_dice")
    ax.set_title("Zero-shot vs fine-tuned (test split)")
    ax.legend(frameon=False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    return fig


# ──────────────────────────────────────────────
# Figure 8 — per-structure Dice heatmap
# ──────────────────────────────────────────────


def fig_per_structure(per_structure_df: pd.DataFrame) -> plt.Figure:
    """Render the per-structure × corruption Dice heatmap (Fig 8)."""
    if per_structure_df.empty:
        raise ValueError("per_structure_dice.csv is empty — nothing to plot")

    pivot = per_structure_df.pivot_table(
        index=PS_LABEL_NAME_COLUMN,
        columns=TYPE_COLUMN,
        values=PS_DICE_COLUMN,
        aggfunc="mean",
    )
    # Hierarchical clustering on the row vectors to produce a sensitivity-
    # ordered presentation. Justified inline because there's no natural
    # anatomical ordering across the 35+ structures and an alphabetical
    # ordering buries the corruption-sensitivity signal.
    if pivot.shape[0] >= 3 and pivot.shape[1] >= 2:
        try:
            from scipy.cluster.hierarchy import leaves_list, linkage

            data = np.nan_to_num(pivot.values, nan=0.0)
            link = linkage(data, method="average", metric="correlation")
            order = leaves_list(link)
            pivot = pivot.iloc[order]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Hierarchical clustering failed (%s); using sorted order", exc)
            pivot = pivot.sort_index()
    else:
        pivot = pivot.sort_index()

    fig_height = max(4.0, 0.25 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(DOUBLE_WIDTH_IN, fig_height))
    im = ax.imshow(pivot.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=6)
    if pivot.shape[0] * pivot.shape[1] <= 280:
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if np.isfinite(v):
                    color = "white" if v < 0.5 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color=color, fontsize=5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025)
    cbar.set_label("mean Dice")
    ax.set_title("Per-structure Dice by corruption (clustered rows)")
    return fig


# ──────────────────────────────────────────────
# Precondition checks
# ──────────────────────────────────────────────


def _resolve_inputs(
    fig_id: int, results_dir: Path
) -> dict[str, list[Path]]:
    """Resolve every required input for ``fig_id`` against ``results_dir``.

    Returns ``{pattern_or_filename: [resolved_paths]}``. Raises ``SystemExit``
    via typer if any required input is missing — the error message names the
    missing file and points at the producing script.
    """
    required = FIGURE_INPUTS[fig_id]
    out: dict[str, list[Path]] = {}
    for pattern in required:
        if "*" in pattern:
            matches = sorted(results_dir.glob(pattern))
            if not matches:
                producer = INPUT_PRODUCERS.get(pattern, "<unknown>")
                raise typer.BadParameter(
                    f"Fig {fig_id}: no files match {pattern!r} under "
                    f"{results_dir} — produce them by running {producer}."
                )
            out[pattern] = matches
        else:
            full = results_dir / pattern
            if not full.is_file():
                producer = INPUT_PRODUCERS.get(pattern, "<unknown>")
                raise typer.BadParameter(
                    f"Fig {fig_id}: required input {full} not found — "
                    f"produce it by running {producer}."
                )
            out[pattern] = [full]
    return out


# ──────────────────────────────────────────────
# Per-figure dispatchers
# ──────────────────────────────────────────────


def _render_fig_2(
    inputs: dict[str, list[Path]], n_bootstrap: int, seed: int
) -> plt.Figure:
    iqm_df = pd.read_csv(inputs["iqm_features.csv"][0])
    pref_df = pd.read_csv(inputs["machine_preference.csv"][0])
    return fig_iqm_heatmap(
        iqm_df, pref_df, n_bootstrap=n_bootstrap, bootstrap_seed=seed
    )


def _render_fig_3(
    inputs: dict[str, list[Path]], n_bootstrap: int, seed: int
) -> plt.Figure:
    pref_df = pd.read_csv(inputs["machine_preference.csv"][0])
    cor_manifest = pd.read_csv(inputs["corruption_manifest.csv"][0])
    return fig_corruption_sensitivity(pref_df, cor_manifest)


def _aggregate_seeds_for_pattern(
    inputs: dict[str, list[Path]], pattern: str
) -> pd.DataFrame:
    paths = inputs.get(pattern, [])
    paths_and_seeds: list[tuple[Path, int]] = []
    for p in paths:
        m = _SEED_REGEX.search(p.name)
        if m is None:
            continue
        paths_and_seeds.append((p, int(m.group(1))))
    if not paths_and_seeds:
        return pd.DataFrame()
    if len(paths_and_seeds) == 1:
        logger.warning(
            "Single-seed result for %s — no error bars / CI will be plotted",
            pattern,
        )
    extras: tuple[str, ...] = ()
    score_col = SCORE_COLUMN
    if "2d_vlm_scores" in pattern:
        # 2D scores carry a slice_strategy column that's part of the unique key.
        extras = ("slice_strategy",)
    if "finetuned_scores" in pattern:
        # Fine-tuned CSVs have no `score` column — they emit Likert integers
        # under `dice_quality`. Aggregate that and let fig_finetuned normalise
        # the result to [0, 1] post-hoc.
        score_col = DICE_QUALITY_COLUMN
    return aggregate_seed_csvs(
        paths_and_seeds, score_column=score_col, extra_keys=extras
    )


def _render_fig_4(
    inputs: dict[str, list[Path]], n_bootstrap: int, seed: int
) -> plt.Figure:
    vlm_3d = _aggregate_seeds_for_pattern(inputs, "3d_vlm_scores_seed_*.csv")
    vlm_2d = _aggregate_seeds_for_pattern(inputs, "2d_vlm_scores_seed_*.csv")
    pref_df = pd.read_csv(inputs["machine_preference.csv"][0])
    return fig_vlm_scatter(
        vlm_3d, vlm_2d, pref_df,
        n_bootstrap=n_bootstrap, bootstrap_seed=seed,
    )


def _render_fig_5(
    inputs: dict[str, list[Path]], n_bootstrap: int, seed: int
) -> plt.Figure:
    vlm_3d = _aggregate_seeds_for_pattern(inputs, "3d_vlm_scores_seed_*.csv")
    vlm_2d = _aggregate_seeds_for_pattern(inputs, "2d_vlm_scores_seed_*.csv")
    pref_df = pd.read_csv(inputs["machine_preference.csv"][0])
    cor_manifest = pd.read_csv(inputs["corruption_manifest.csv"][0])
    return fig_3d_vs_2d(
        vlm_3d, vlm_2d, pref_df, cor_manifest,
        n_bootstrap=n_bootstrap, bootstrap_seed=seed,
    )


def _render_fig_6(
    inputs: dict[str, list[Path]], n_bootstrap: int, seed: int
) -> plt.Figure:
    vlm_3d = _aggregate_seeds_for_pattern(inputs, "3d_vlm_scores_seed_*.csv")
    vlm_2d = _aggregate_seeds_for_pattern(inputs, "2d_vlm_scores_seed_*.csv")
    finetuned = _aggregate_seeds_for_pattern(inputs, "finetuned_scores_seed_*.csv")
    pref_df = pd.read_csv(inputs["machine_preference.csv"][0])
    return fig_finetuned(
        vlm_3d, vlm_2d, finetuned, pref_df,
        n_bootstrap=n_bootstrap, bootstrap_seed=seed,
    )


def _render_fig_8(
    inputs: dict[str, list[Path]], n_bootstrap: int, seed: int
) -> plt.Figure:
    df = pd.read_csv(inputs["per_structure_dice.csv"][0])
    return fig_per_structure(df)


_FIG_RENDERERS: dict[
    int, Callable[[dict[str, list[Path]], int, int], plt.Figure]
] = {
    2: _render_fig_2,
    3: _render_fig_3,
    4: _render_fig_4,
    5: _render_fig_5,
    6: _render_fig_6,
    8: _render_fig_8,
}


# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────


def _print_summary(
    requested: list[int],
    written: list[Path],
    output_dir: Path,
    console: Console,
) -> None:
    table = Table(title="NeuroQC figures")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("requested figures", ", ".join(str(f) for f in requested))
    table.add_row("output files written", str(len(written)))
    table.add_row("output dir", str(output_dir))
    console.print(table)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


@app.command()
def main(
    all_figures: bool = typer.Option(
        False, "--all", help="Generate every supported figure."
    ),
    figure: list[int] = typer.Option(
        [],
        "--figure",
        help=f"Repeatable. Figure number(s) to generate. Choose from {SUPPORTED_FIGURES}.",
    ),
    output_dir: Path = typer.Option(
        Path("figures"), "--output-dir", resolve_path=True
    ),
    results_dir: Path = typer.Option(
        Path("results/tables"), "--results-dir", resolve_path=True
    ),
    n_bootstrap: int = typer.Option(DEFAULT_N_BOOTSTRAP, "--n-bootstrap"),
    bootstrap_seed: int = typer.Option(DEFAULT_BOOTSTRAP_SEED, "--bootstrap-seed"),
    dpi: int = typer.Option(300, "--dpi"),
    interactive: bool = typer.Option(
        False, "--interactive",
        help="Reserved for Plotly HTML (not implemented in v1; logs WARNING).",
    ),
) -> None:
    """Generate publication figures from results CSVs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )

    if interactive:
        logger.warning(
            "--interactive is reserved for Plotly HTML output and is not "
            "implemented in v1; proceeding with PNG/SVG only."
        )

    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi

    if all_figures:
        requested = list(SUPPORTED_FIGURES)
    elif figure:
        requested = sorted(set(figure))
    else:
        raise typer.BadParameter("Pass --all or one or more --figure N flags.")

    invalid = [f for f in requested if f not in SUPPORTED_FIGURES]
    if invalid:
        raise typer.BadParameter(
            f"Unsupported figure(s) {invalid}; choose from {SUPPORTED_FIGURES}"
        )

    if not results_dir.is_dir():
        raise typer.BadParameter(f"Results dir not found: {results_dir}")

    written: list[Path] = []
    for fig_id in requested:
        logger.info("Rendering Fig %d (%s)", fig_id, FIG_FILENAMES[fig_id])
        inputs = _resolve_inputs(fig_id, results_dir)
        fig = _FIG_RENDERERS[fig_id](inputs, n_bootstrap, bootstrap_seed)
        try:
            written.extend(_save_figure_atomic(fig, output_dir / FIG_FILENAMES[fig_id]))
        finally:
            plt.close(fig)
        logger.info("  → wrote %s.{png,svg}", FIG_FILENAMES[fig_id])

    _print_summary(requested, written, output_dir, Console())


if __name__ == "__main__":
    app()
