#!/usr/bin/env python3
"""NeuroQC Phase 5b — aggregate SynthSeg per-scan QC (GMM log-loss) sidecars.

SynthSeg's ``mri_synthseg`` writes a per-scan ``*_qc.csv`` sidecar containing
GMM log-loss values per tissue class. Those sidecars are already produced by
Phase 03 (referenced via the ``qc_path`` column of ``synthseg_manifest.csv``)
but were never aggregated into a single table. This script walks one or more
synthseg manifests, opens each ``qc_path``, and writes one unified row per
scan to ``--output-file``.

Methodological note:
    SynthSeg's GMM log-loss measures how Gaussian the intensity distribution
    within each label is — a distribution-fit metric, NOT an anatomical-
    correctness metric. Under mild motion, smoothing can *lower* the log-loss
    (better Gaussian fit) even though the anatomy is worse. This metric is
    kept as its own CSV (separate from the mriqc-style IQMs produced by 05)
    so downstream analyses can treat the two families independently.

Inputs:
    --synthseg-manifest  synthseg_manifest.csv from Phase 03 (repeatable).
                         Must have columns {input_path, qc_path, status}.
                         Rows with status == "failed" are dropped (their
                         qc_path is typically unwritten).
    --cor-manifest       corruption_manifest.csv (optional). When provided,
                         scans whose path appears as cor_path in it are
                         annotated with corruption_type/severity/dataset_tag
                         and is_reference=False. Scans not in the manifest
                         default to is_reference=True, corruption_type="none",
                         severity=0, dataset_tag inferred from path or
                         "unknown".
    --output-file        Aggregate CSV (append-mode, per-row flush). Resume
                         skips any scan_path already present.
    --dry-run            Log the first 10 pending scans and return.

Output schema (12 columns):
    scan_path,
    general_white_matter_loss, general_grey_matter_loss, general_csf_loss,
    cerebellum_loss, brainstem_loss, thalamus_loss,
    putamen_pallidum_loss, hippocampus_amygdala_loss,
    is_reference, corruption_type, severity, dataset_tag

Usage:
    python code/05b_aggregate_synthseg_qc.py \\
        --synthseg-manifest results/tables/synthseg_fastmri_manifest.csv \\
        --synthseg-manifest results/tables/synthseg_ixi_manifest.csv \\
        --cor-manifest      results/tables/corruption_manifest.csv \\
        --output-file       results/tables/synthseg_qc_features.csv
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from tqdm import tqdm

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

SCAN_COLUMN: str = "scan_path"
IS_REFERENCE_COLUMN: str = "is_reference"
TYPE_COLUMN: str = "corruption_type"
SEVERITY_COLUMN: str = "severity"
DATASET_TAG_COLUMN: str = "dataset_tag"

# SynthSeg qc.csv schema (verified identical across IXI + FastMRI).
# First column is the subject id; the remaining 8 columns are GMM log-loss
# values per tissue class — higher = worse Gaussian fit.
_SUBJECT_COLUMN: str = "subject"

# Source-to-CSV-safe column rename map. Order is the canonical output order.
_QC_COLUMN_RENAMES: tuple[tuple[str, str], ...] = (
    ("general white matter", "general_white_matter_loss"),
    ("general grey matter", "general_grey_matter_loss"),
    ("general csf", "general_csf_loss"),
    ("cerebellum", "cerebellum_loss"),
    ("brainstem", "brainstem_loss"),
    ("thalamus", "thalamus_loss"),
    ("putamen+pallidum", "putamen_pallidum_loss"),
    ("hippocampus+amygdala", "hippocampus_amygdala_loss"),
)
SOURCE_LOSS_COLUMNS: tuple[str, ...] = tuple(src for src, _ in _QC_COLUMN_RENAMES)
LOSS_COLUMNS: tuple[str, ...] = tuple(dst for _, dst in _QC_COLUMN_RENAMES)

OUTPUT_COLUMNS: tuple[str, ...] = (
    SCAN_COLUMN,
    *LOSS_COLUMNS,
    IS_REFERENCE_COLUMN,
    TYPE_COLUMN,
    SEVERITY_COLUMN,
    DATASET_TAG_COLUMN,
)

# Synthseg-manifest columns we consume.
_SYNTHSEG_INPUT_COLUMN: str = "input_path"
_SYNTHSEG_QC_COLUMN: str = "qc_path"
_SYNTHSEG_STATUS_COLUMN: str = "status"
_STATUS_FAILED: str = "failed"

# Corruption-manifest source-column name; the other fields reuse the
# OUTPUT column names because they match by construction.
_COR_PATH_COLUMN: str = "cor_path"

_NONE_TYPE: str = "none"
_REF_SEVERITY: int = 0
_UNKNOWN_TAG: str = "unknown"

logger = logging.getLogger(__name__)
app = typer.Typer(
    help="NeuroQC Phase 5b — aggregate SynthSeg per-scan QC sidecars.",
    add_completion=False,
)


# ──────────────────────────────────────────────
# Data structure
# ──────────────────────────────────────────────


@dataclass(frozen=True)
class SynthSegQCRow:
    """One scan's GMM log-loss vector + provenance tags.

    Attributes:
        scan_path: Resolved-absolute path to the scan NIfTI (= synthseg
            manifest's ``input_path``).
        losses: ``{csv-safe-name: float}`` covering all 8 tissue classes.
            Keys match :data:`LOSS_COLUMNS` order.
        is_reference: True if the scan doesn't appear in the corruption
            manifest's ``cor_path`` column (or no cor manifest was given).
        corruption_type: ``"none"`` for references; corruption family name
            otherwise.
        severity: ``0`` for references; integer severity for corrupted scans.
        dataset_tag: From the corruption manifest (or inferred/unknown).
    """

    scan_path: str
    losses: dict[str, float]
    is_reference: bool
    corruption_type: str
    severity: int
    dataset_tag: str


# ──────────────────────────────────────────────
# Sidecar parsing
# ──────────────────────────────────────────────


def load_qc_sidecar(qc_path: Path) -> dict[str, float]:
    """Parse one ``*_qc.csv`` sidecar into a ``{csv-safe-name: float}`` dict.

    Reads the 2-row CSV (header + one data row); projects down to the 8
    known loss columns and returns them under their renamed keys. Unexpected
    extra columns in the source file are ignored (forward-compatible with
    future SynthSeg versions that may add tissue classes).

    Raises:
        FileNotFoundError: if ``qc_path`` does not exist.
        ValueError: if the sidecar is empty or missing any required source
            column (schema drift we need to fail loudly on, not silently).
    """
    if not qc_path.is_file():
        raise FileNotFoundError(f"SynthSeg QC sidecar not found: {qc_path}")
    df = pd.read_csv(qc_path)
    if len(df) == 0:
        raise ValueError(f"SynthSeg QC sidecar is empty: {qc_path}")
    missing = set(SOURCE_LOSS_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"{qc_path} missing expected columns {sorted(missing)}; "
            f"has {sorted(df.columns)}"
        )
    row = df.iloc[0]
    return {
        dst: float(row[src])
        for src, dst in _QC_COLUMN_RENAMES
    }


# ──────────────────────────────────────────────
# Manifest loading
# ──────────────────────────────────────────────


def load_synthseg_manifests(manifest_paths: list[Path]) -> pd.DataFrame:
    """Return ``[scan_path, qc_path, status]`` across all synthseg manifests.

    Drops rows with status == "failed" (their qc_path is typically missing
    on disk). Resolves both path columns to absolute strings so downstream
    joins stay consistent.

    Raises:
        typer.BadParameter: on missing file.
        ValueError: on schema mismatch (missing required columns).
    """
    required = {_SYNTHSEG_INPUT_COLUMN, _SYNTHSEG_QC_COLUMN, _SYNTHSEG_STATUS_COLUMN}
    frames: list[pd.DataFrame] = []
    for manifest_path in manifest_paths:
        if not manifest_path.is_file():
            raise typer.BadParameter(
                f"synthseg manifest not found: {manifest_path}"
            )
        df = pd.read_csv(manifest_path)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{manifest_path} missing synthseg columns {sorted(missing)}; "
                f"has {sorted(df.columns)}"
            )
        ok = df[df[_SYNTHSEG_STATUS_COLUMN].astype(str) != _STATUS_FAILED].copy()
        frames.append(ok[[_SYNTHSEG_INPUT_COLUMN, _SYNTHSEG_QC_COLUMN]])
    if not frames:
        return pd.DataFrame(columns=[SCAN_COLUMN, _SYNTHSEG_QC_COLUMN])
    out = pd.concat(frames, ignore_index=True)
    out[SCAN_COLUMN] = out[_SYNTHSEG_INPUT_COLUMN].map(
        lambda p: str(Path(p).resolve())
    )
    out[_SYNTHSEG_QC_COLUMN] = out[_SYNTHSEG_QC_COLUMN].map(
        lambda p: str(Path(p).resolve())
    )
    # Collapse duplicate entries (e.g. same scan in two manifests by accident);
    # keep the first occurrence. Different qc_path values for the same scan
    # are reported as a warning but the first wins.
    deduped = out.drop_duplicates(subset=SCAN_COLUMN, keep="first").reset_index(
        drop=True
    )
    if len(deduped) < len(out):
        logger.warning(
            "dropped %d duplicate scan_path rows across synthseg manifests",
            len(out) - len(deduped),
        )
    return deduped[[SCAN_COLUMN, _SYNTHSEG_QC_COLUMN]]


def _infer_dataset_tag(scan_path: str) -> str:
    """Crude dataset-tag inference from the scan path."""
    lowered = scan_path.lower()
    for tag in ("ixi", "fastmri", "oasis"):
        if f"/{tag}/" in lowered:
            return tag
    return _UNKNOWN_TAG


def annotate_with_corruption_manifest(
    frame: pd.DataFrame, cor_manifest_path: Path | None
) -> pd.DataFrame:
    """Left-join corruption metadata onto the synthseg frame.

    Adds columns ``is_reference, corruption_type, severity, dataset_tag``.
    Rows whose ``scan_path`` matches the corruption manifest's ``cor_path``
    column inherit its type/severity/tag and get ``is_reference=False``.
    Unmatched rows default to references.

    When ``cor_manifest_path`` is None, every row is marked as a reference
    with a path-inferred dataset tag.
    """
    frame = frame.copy()
    if cor_manifest_path is None:
        frame[IS_REFERENCE_COLUMN] = True
        frame[TYPE_COLUMN] = _NONE_TYPE
        frame[SEVERITY_COLUMN] = _REF_SEVERITY
        frame[DATASET_TAG_COLUMN] = frame[SCAN_COLUMN].map(_infer_dataset_tag)
        return frame

    cor = pd.read_csv(cor_manifest_path)
    # Use the OUTPUT column names here — they are deliberately chosen to match
    # the corruption-manifest column names (corruption_type, severity,
    # dataset_tag), so the merge lands each column directly under its final
    # name without a separate rename pass.
    required = {_COR_PATH_COLUMN, TYPE_COLUMN, SEVERITY_COLUMN}
    missing = required - set(cor.columns)
    if missing:
        raise typer.BadParameter(
            f"{cor_manifest_path} missing required columns {sorted(missing)}"
        )
    cor[SCAN_COLUMN] = cor[_COR_PATH_COLUMN].map(
        lambda p: str(Path(p).resolve())
    )
    keep = [SCAN_COLUMN, TYPE_COLUMN, SEVERITY_COLUMN]
    if DATASET_TAG_COLUMN in cor.columns:
        keep.append(DATASET_TAG_COLUMN)
    cor_slim = cor[keep].drop_duplicates(subset=SCAN_COLUMN, keep="first")

    merged = frame.merge(cor_slim, on=SCAN_COLUMN, how="left")
    # Scans that didn't match the corruption manifest are references.
    merged[IS_REFERENCE_COLUMN] = merged[TYPE_COLUMN].isna()
    merged[TYPE_COLUMN] = merged[TYPE_COLUMN].fillna(_NONE_TYPE)
    merged[SEVERITY_COLUMN] = (
        merged[SEVERITY_COLUMN]
        .map(_coerce_severity_or_zero)
        .astype(int)
    )
    if DATASET_TAG_COLUMN in merged.columns:
        merged[DATASET_TAG_COLUMN] = merged[DATASET_TAG_COLUMN].where(
            merged[DATASET_TAG_COLUMN].notna(),
            merged[SCAN_COLUMN].map(_infer_dataset_tag),
        )
    else:
        merged[DATASET_TAG_COLUMN] = merged[SCAN_COLUMN].map(_infer_dataset_tag)
    return merged.reset_index(drop=True)


def _coerce_severity_or_zero(value: object) -> int:
    """Cast severity to int; NaN or garbage → 0 (reference default)."""
    if value is None:
        return 0
    try:
        # pandas may hand us NaN as a float.
        if isinstance(value, float) and math.isnan(value):
            return 0
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


# ──────────────────────────────────────────────
# CSV I/O + resume
# ──────────────────────────────────────────────


def load_existing(output_file: Path) -> set[str]:
    """Return scan_path values already recorded in the output CSV."""
    if not output_file.exists() or output_file.stat().st_size == 0:
        return set()
    with output_file.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or SCAN_COLUMN not in reader.fieldnames:
            return set()
        return {row[SCAN_COLUMN] for row in reader if row.get(SCAN_COLUMN)}


def append_row(output_file: Path, row: SynthSegQCRow) -> None:
    """Append one row to the output CSV (crash-safe per write)."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    is_new = not output_file.exists() or output_file.stat().st_size == 0
    record: dict[str, object] = {
        SCAN_COLUMN: row.scan_path,
        IS_REFERENCE_COLUMN: row.is_reference,
        TYPE_COLUMN: row.corruption_type,
        SEVERITY_COLUMN: row.severity,
        DATASET_TAG_COLUMN: row.dataset_tag,
    }
    for col in LOSS_COLUMNS:
        record[col] = row.losses.get(col, math.nan)
    with output_file.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        if is_new:
            writer.writeheader()
        writer.writerow(record)
        handle.flush()


# ──────────────────────────────────────────────
# Per-scan scoring
# ──────────────────────────────────────────────


def score_scan(
    scan_path: str,
    qc_path: Path,
    *,
    is_reference: bool,
    corruption_type: str,
    severity: int,
    dataset_tag: str,
) -> SynthSegQCRow | None:
    """Load one qc sidecar and wrap as a :class:`SynthSegQCRow`.

    Returns None on parse error (logged warning); the per-scan loop in
    :func:`main` treats None as "skip entirely" (no NaN row written —
    missing QC is structurally different from a computed NaN).
    """
    try:
        losses = load_qc_sidecar(qc_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("QC parse failed for %s: %s", scan_path, exc)
        return None
    return SynthSegQCRow(
        scan_path=scan_path,
        losses=losses,
        is_reference=is_reference,
        corruption_type=corruption_type,
        severity=severity,
        dataset_tag=dataset_tag,
    )


# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────


def _print_summary(
    total: int,
    already_done: int,
    processed: int,
    skipped: int,
    output_file: Path,
    console: Console,
) -> None:
    table = Table(title="Phase 5b SynthSeg QC aggregation")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("total scans", str(total))
    table.add_row("already in CSV (skipped)", str(already_done))
    table.add_row("processed this run", str(processed))
    table.add_row("skipped (parse failure)", str(skipped))
    table.add_row("output file", str(output_file))
    console.print(table)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


@app.command()
def main(
    synthseg_manifest: list[Path] = typer.Option(
        ...,
        "--synthseg-manifest",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="synthseg_manifest.csv from Phase 03 (repeatable).",
    ),
    cor_manifest: Path | None = typer.Option(
        None,
        "--cor-manifest",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="corruption_manifest.csv (optional). When provided, scans "
        "matching a cor_path are annotated with corruption metadata.",
    ),
    output_file: Path = typer.Option(
        ...,
        "--output-file",
        resolve_path=True,
        help="Aggregate synthseg_qc_features.csv (one row per scan).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Log the first 10 pending scans and return."
    ),
) -> None:
    """Aggregate SynthSeg per-scan QC sidecars into a unified CSV."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )

    frame = load_synthseg_manifests(list(synthseg_manifest))
    logger.info("Loaded %d scans from %d manifest(s)", len(frame), len(synthseg_manifest))
    if frame.empty:
        logger.warning("No non-failed scans found; nothing to aggregate.")
        return

    frame = annotate_with_corruption_manifest(frame, cor_manifest)

    done = load_existing(output_file)
    pending = frame[~frame[SCAN_COLUMN].isin(done)].reset_index(drop=True)
    logger.info(
        "Plan: %d total, %d already done, %d pending",
        len(frame),
        len(done),
        len(pending),
    )

    if dry_run:
        for _, row in pending.head(10).iterrows():
            logger.info(
                "  would parse: scan=%s qc=%s ref=%s type=%s sev=%s tag=%s",
                row[SCAN_COLUMN],
                row[_SYNTHSEG_QC_COLUMN],
                row[IS_REFERENCE_COLUMN],
                row[TYPE_COLUMN],
                row[SEVERITY_COLUMN],
                row[DATASET_TAG_COLUMN],
            )
        if len(pending) > 10:
            logger.info("  ... (%d more)", len(pending) - 10)
        return

    processed = 0
    skipped = 0
    iterator = (
        tqdm(pending.itertuples(index=False), total=len(pending), desc="synthseg-qc", unit="scan")
        if len(pending) > 1
        else pending.itertuples(index=False)
    )
    for row in iterator:
        qc_row = score_scan(
            scan_path=str(row.scan_path),
            qc_path=Path(row.qc_path),
            is_reference=bool(row.is_reference),
            corruption_type=str(row.corruption_type),
            severity=int(row.severity),
            dataset_tag=str(row.dataset_tag),
        )
        if qc_row is None:
            skipped += 1
            continue
        append_row(output_file, qc_row)
        processed += 1

    _print_summary(
        total=len(frame),
        already_done=len(done),
        processed=processed,
        skipped=skipped,
        output_file=output_file,
        console=Console(),
    )


if __name__ == "__main__":
    app()
