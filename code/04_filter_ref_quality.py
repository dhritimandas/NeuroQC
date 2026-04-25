#!/usr/bin/env python3
"""NeuroQC Phase 4 — Reference-segmentation quality gate.

Consumes the manifest emitted by ``03_run_synthseg.py``, reads each
segmentation NIfTI, and enriches the manifest with per-scan quality
metrics plus a pass/fail flag. Downstream phases (``02b`` motion
corruption, Dice-based ablation analysis) then filter to ``passed_gate
== True`` rows, so we never measure motion degradation against a
degenerate reference segmentation.

Metrics computed (from the seg NIfTI alone, no source-scan reload):
    n_unique_labels    — distinct non-zero labels in the segmentation.
    brain_fraction     — fraction of voxels with label > 0.
    seg_shape          — str repr of the segmentation array shape.

Pass criteria (configurable via CLI):
    n_unique_labels   >= --min-n-labels       (default 20)
    brain_fraction    >= --min-brain-fraction (default 0.01)

Why those defaults?
    20 labels is the floor for a non-degenerate SynthSeg output; the
    published image-spec for FastMRI flags `n_unique_labels < 20` as a
    red flag. Brain fraction = 0.01 is deliberately lenient to
    accommodate thick-slice slab acquisitions like FastMRI brain
    (8-cm axial coverage → 2-5% brain fraction, below IXI's whole-head
    baseline of 10-25%). IXI-only runs should override with
    ``--min-brain-fraction 0.05``.

Rejected scans are not deleted — their rows stay in the gated manifest
with ``passed_gate = False`` and a human-readable ``reject_reason``, so
the paper's data-availability section can tally rejections by
acquisition class, dataset, or any other groupby dimension.

Inputs:
    --synthseg-manifest  CSV path written by 03_run_synthseg.py (must
                         contain at least: input_path, seg_path).
    --gated-manifest     Output CSV path (enriched + gated).
    --min-n-labels       Integer floor on unique non-zero labels.
    --min-brain-fraction Float floor on brain_voxels / total_voxels.
    --dry-run            Compute metrics but do not write the CSV.

Outputs:
    <gated-manifest>    Same rows as the input manifest, plus columns:
        n_unique_labels, brain_fraction, seg_shape, passed_gate,
        reject_reason.

Usage:
    python code/04_filter_ref_quality.py \\
        --synthseg-manifest results/tables/synthseg_fastmri_manifest.csv \\
        --gated-manifest results/tables/ref_quality_gated_fastmri.csv \\
        --min-n-labels 20 \\
        --min-brain-fraction 0.01
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

DEFAULT_MIN_N_LABELS: int = 20
DEFAULT_MIN_BRAIN_FRACTION: float = 0.01

# Columns added on top of whatever the upstream synthseg manifest carries.
# Kept separate from MANIFEST_COLUMNS-style constants elsewhere because we
# preserve the upstream schema verbatim and only append.
ENRICHMENT_COLUMNS: list[str] = [
    "n_unique_labels",
    "brain_fraction",
    "seg_shape",
    "passed_gate",
    "reject_reason",
]

logger = logging.getLogger(__name__)
app = typer.Typer(
    help="NeuroQC Phase 4 — gate reference segmentations by quality.",
    add_completion=False,
)


# ──────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────


@dataclass(frozen=True)
class SegQuality:
    """Per-scan quality snapshot of a SynthSeg segmentation NIfTI."""

    n_unique_labels: int
    brain_fraction: float
    seg_shape: tuple[int, ...]


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────


def compute_quality(seg_path: Path) -> SegQuality:
    """Load a segmentation NIfTI and return non-zero-label count + brain fraction.

    This function is the single source of truth for the two metrics.
    Intentionally minimal — any richer quality analysis (symmetry,
    intensity coherence, connectivity) lives in
    ``code/diagnose_synthseg_on_fastmri.py`` and consumes the seg
    alongside the source scan.
    """
    img = nib.load(str(seg_path))
    data = img.get_fdata().astype(np.int32)
    unique = np.unique(data)
    n_unique_non_zero = int((unique != 0).sum())
    total = data.size
    brain_voxels = int((data > 0).sum())
    brain_fraction = brain_voxels / total if total else 0.0
    return SegQuality(
        n_unique_labels=n_unique_non_zero,
        brain_fraction=brain_fraction,
        seg_shape=tuple(int(s) for s in data.shape),
    )


def evaluate_gate(
    quality: SegQuality,
    *,
    min_n_labels: int,
    min_brain_fraction: float,
) -> tuple[bool, str]:
    """Return ``(passed, reject_reason)`` for the given quality snapshot.

    Each criterion's threshold comes from the CLI; the reject reason
    string is human-readable and enumerates every failed criterion so
    logs and manifests expose all failure modes at once, not just the
    first one hit.
    """
    reasons: list[str] = []
    if quality.n_unique_labels < min_n_labels:
        reasons.append(
            f"n_unique_labels={quality.n_unique_labels}<{min_n_labels}"
        )
    if quality.brain_fraction < min_brain_fraction:
        reasons.append(
            f"brain_fraction={quality.brain_fraction:.4f}<{min_brain_fraction}"
        )
    return (not reasons), "; ".join(reasons)


# ──────────────────────────────────────────────
# Manifest processing
# ──────────────────────────────────────────────


def gate_manifest(
    synthseg_manifest: Path,
    *,
    min_n_labels: int = DEFAULT_MIN_N_LABELS,
    min_brain_fraction: float = DEFAULT_MIN_BRAIN_FRACTION,
) -> pd.DataFrame:
    """Read a SynthSeg manifest, enrich with quality metrics + pass flag.

    Returns a DataFrame with every upstream column preserved plus the
    five ENRICHMENT_COLUMNS appended. The function does not write any
    file; the caller decides where and whether to persist (dry_run path
    just inspects the returned frame).

    Rows whose ``seg_path`` is missing or unreadable get
    ``passed_gate = False`` with an explanatory ``reject_reason`` —
    they are NOT dropped, so rejection tallies remain honest.
    """
    if not synthseg_manifest.exists():
        raise FileNotFoundError(f"--synthseg-manifest not found: {synthseg_manifest}")
    df = pd.read_csv(synthseg_manifest)
    if "seg_path" not in df.columns:
        raise ValueError(
            f"{synthseg_manifest} is missing the 'seg_path' column; "
            "was it produced by 03_run_synthseg.py?"
        )

    enriched_rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        seg_path = Path(str(row["seg_path"]))
        if not seg_path.exists():
            enriched_rows.append({
                **row_dict,
                "n_unique_labels": 0,
                "brain_fraction": 0.0,
                "seg_shape": "",
                "passed_gate": False,
                "reject_reason": f"seg_path missing: {seg_path}",
            })
            continue
        try:
            quality = compute_quality(seg_path)
        except Exception as exc:
            enriched_rows.append({
                **row_dict,
                "n_unique_labels": -1,
                "brain_fraction": -1.0,
                "seg_shape": "",
                "passed_gate": False,
                "reject_reason": f"compute_quality raised {type(exc).__name__}: {exc}",
            })
            continue
        passed, reason = evaluate_gate(
            quality,
            min_n_labels=min_n_labels,
            min_brain_fraction=min_brain_fraction,
        )
        enriched_rows.append({
            **row_dict,
            "n_unique_labels": quality.n_unique_labels,
            "brain_fraction": quality.brain_fraction,
            "seg_shape": "x".join(str(s) for s in quality.seg_shape),
            "passed_gate": passed,
            "reject_reason": reason,
        })
    return pd.DataFrame(enriched_rows)


def write_gated_manifest(df: pd.DataFrame, output_path: Path) -> None:
    """Persist the enriched DataFrame to CSV, preserving column order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────


def _print_summary(df: pd.DataFrame, output_path: Path, console: Console) -> None:
    total = len(df)
    passed = int(df["passed_gate"].sum()) if total else 0
    failed = total - passed

    table = Table(title="Phase 4 — reference quality gate")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("total scans", str(total))
    table.add_row("passed", str(passed))
    table.add_row("failed", str(failed))
    table.add_row("gated manifest", str(output_path))
    console.print(table)

    if failed and "reject_reason" in df.columns:
        reason_table = Table(title="Reject reasons")
        reason_table.add_column("reason", overflow="fold")
        reason_table.add_column("count", justify="right")
        counts = df.loc[~df["passed_gate"], "reject_reason"].value_counts()
        for reason, n in counts.items():
            reason_table.add_row(str(reason), str(int(n)))
        console.print(reason_table)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


@app.command()
def main(
    synthseg_manifest: Path = typer.Option(
        ...,
        "--synthseg-manifest",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="CSV produced by 03_run_synthseg.py.",
    ),
    gated_manifest: Path = typer.Option(
        Path("results/tables/ref_quality_gated.csv"),
        "--gated-manifest",
        help="Output CSV path with enrichment + pass/fail flag.",
    ),
    min_n_labels: int = typer.Option(
        DEFAULT_MIN_N_LABELS,
        "--min-n-labels",
        min=0,
        help="Reject scans with fewer than this many unique non-zero labels.",
    ),
    min_brain_fraction: float = typer.Option(
        DEFAULT_MIN_BRAIN_FRACTION,
        "--min-brain-fraction",
        min=0.0,
        max=1.0,
        help="Reject scans whose labeled-voxel fraction is below this value.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Compute + report; do not write the CSV."
    ),
) -> None:
    """Gate SynthSeg reference segmentations by quality; emit annotated manifest."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )
    df = gate_manifest(
        synthseg_manifest,
        min_n_labels=min_n_labels,
        min_brain_fraction=min_brain_fraction,
    )
    if not dry_run:
        write_gated_manifest(df, gated_manifest)
    _print_summary(df, gated_manifest, Console())


if __name__ == "__main__":
    app()
