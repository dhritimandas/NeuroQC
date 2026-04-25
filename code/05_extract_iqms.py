#!/usr/bin/env python3
"""NeuroQC Phase 5 — per-scan signal-based IQM extraction.

Thin CLI wrapper around ``nobrainer.qc.metrics.extract_iqms``. Unifies one or
more reference manifests with the corruption manifest and the Phase 03
synthseg manifest, then computes mriqc-style IQMs (SNR, CNR, EFC, FBER, CJV)
for every scan with its paired SynthSeg segmentation as the tissue-mask
source.

Inputs:
    --ref-manifest      Repeatable. Reference-scan manifest (one per dataset).
                        Accepts either a ``filepath`` column (IXI-style) or a
                        ``ref_path`` column (FastMRI-style). If ``passed_qc``
                        exists in the manifest, rows with ``passed_qc == False``
                        are dropped.
    --cor-manifest      corruption_manifest.csv written by Phase 02/02b.
    --synthseg-manifest synthseg_manifest.csv written by Phase 03. Used to
                        look up ``seg_path`` per scan via the ``input_path``
                        column. If a scan has no matching seg, its IQM row is
                        skipped (logged).
    --output-file       Aggregate CSV (one row per scan). Append-mode with
                        per-row flush.
    --dry-run           Log the first 10 pending scans and return.

Resume:
    If ``--output-file`` exists, rows whose ``scan_path`` is already present
    are skipped. Writes are crash-safe at row granularity so an interrupted
    run continues on the next invocation.

Output schema:
    scan_path, snr, cnr, efc, fber, cjv,
    is_reference, corruption_type, severity, dataset_tag

Usage:
    python code/05_extract_iqms.py \\
        --ref-manifest      results/tables/ref_quality_gated_ixi.csv \\
        --ref-manifest      results/tables/ref_quality_gated_fastmri.csv \\
        --cor-manifest      results/tables/corruption_manifest.csv \\
        --synthseg-manifest results/tables/synthseg_fastmri_manifest.csv \\
        --synthseg-manifest results/tables/synthseg_ixi_manifest.csv \\
        --output-file       results/tables/iqm_features.csv
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

from nobrainer.qc.metrics import extract_iqms

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

SCAN_COLUMN: str = "scan_path"
SEG_COLUMN: str = "seg_path"
IS_REFERENCE_COLUMN: str = "is_reference"
TYPE_COLUMN: str = "corruption_type"
SEVERITY_COLUMN: str = "severity"
DATASET_TAG_COLUMN: str = "dataset_tag"

IQM_KEYS: tuple[str, ...] = ("snr", "cnr", "efc", "fber", "cjv")

OUTPUT_COLUMNS: tuple[str, ...] = (
    SCAN_COLUMN,
    *IQM_KEYS,
    IS_REFERENCE_COLUMN,
    TYPE_COLUMN,
    SEVERITY_COLUMN,
    DATASET_TAG_COLUMN,
)

# Ref-manifest column aliases. First hit wins.
_REF_SCAN_PATH_COLUMN_ALIASES: tuple[str, ...] = ("filepath", "ref_path", "scan_path")

# Corruption-manifest columns we consume.
_COR_PATH_COLUMN: str = "cor_path"
_COR_REF_PATH_COLUMN: str = "ref_path"  # unused here but documented for clarity
_PASSED_QC_COLUMN: str = "passed_qc"

# Synthseg-manifest columns.
_SYNTHSEG_INPUT_COLUMN: str = "input_path"
_SYNTHSEG_SEG_COLUMN: str = "seg_path"

_NONE_TYPE: str = "none"
_REF_SEVERITY: int = 0

logger = logging.getLogger(__name__)
app = typer.Typer(
    help="NeuroQC Phase 5 — per-scan signal-based IQM extraction.",
    add_completion=False,
)


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────


@dataclass(frozen=True)
class IQMRow:
    """One scan's IQM vector + provenance tags.

    Attributes:
        scan_path: Resolved-absolute path to the scan NIfTI.
        iqms: ``{snr, cnr, efc, fber, cjv}`` as floats (NaN where undefined).
        is_reference: True for reference scans, False for corrupted.
        corruption_type: ``"none"`` for references; corruption family name
            otherwise.
        severity: ``0`` for references; integer severity level otherwise.
        dataset_tag: Dataset origin (``ixi``, ``fastmri``, ``oasis``, ...).
    """

    scan_path: str
    iqms: dict[str, float]
    is_reference: bool
    corruption_type: str
    severity: int
    dataset_tag: str


# ──────────────────────────────────────────────
# Manifest unification
# ──────────────────────────────────────────────


def _first_present(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate that appears in ``columns``, else None."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _infer_dataset_tag(scan_path: str, fallback: str) -> str:
    """Infer dataset_tag from the scan path when the manifest doesn't carry it.

    Path-based heuristic: first folder segment matching a known dataset tag
    wins. If nothing matches, return the caller's fallback (which should be
    conservative like ``"unknown"``).
    """
    lowered = scan_path.lower()
    for tag in ("ixi", "fastmri", "oasis"):
        if f"/{tag}/" in lowered:
            return tag
    return fallback


def load_ref_manifest(path: Path) -> pd.DataFrame:
    """Load one reference manifest and normalise it to the unified schema.

    Returns a DataFrame with columns ``[scan_path, is_reference, corruption_type,
    severity, dataset_tag]``. ``scan_path`` is resolved to an absolute path
    string for consistent joins downstream.

    Raises:
        typer.BadParameter: if the manifest doesn't have a recognised
            scan-path column (one of ``filepath``, ``ref_path``, ``scan_path``).
    """
    df = pd.read_csv(path)
    scan_col = _first_present(list(df.columns), _REF_SCAN_PATH_COLUMN_ALIASES)
    if scan_col is None:
        raise typer.BadParameter(
            f"{path} must have one of {_REF_SCAN_PATH_COLUMN_ALIASES} as the "
            f"scan-path column; has {sorted(df.columns)}"
        )

    # Drop quality-gated-out rows when the manifest carries that signal.
    if _PASSED_QC_COLUMN in df.columns:
        mask = df[_PASSED_QC_COLUMN].astype(str).str.strip().str.lower().isin(
            {"true", "1", "yes"}
        )
        dropped = int((~mask).sum())
        if dropped:
            logger.info("%s: dropped %d rows with passed_qc=False", path.name, dropped)
        df = df[mask]

    out = pd.DataFrame()
    out[SCAN_COLUMN] = df[scan_col].map(lambda p: str(Path(p).resolve()))
    out[IS_REFERENCE_COLUMN] = True
    out[TYPE_COLUMN] = _NONE_TYPE
    out[SEVERITY_COLUMN] = _REF_SEVERITY

    if DATASET_TAG_COLUMN in df.columns:
        out[DATASET_TAG_COLUMN] = df[DATASET_TAG_COLUMN].astype(str)
    else:
        # Infer from path. ``path.stem`` is a reasonable fallback label
        # (e.g. ``ref_quality_gated_ixi`` → tag ``unknown`` unless path matches).
        fallback = path.stem
        out[DATASET_TAG_COLUMN] = out[SCAN_COLUMN].map(
            lambda p: _infer_dataset_tag(p, fallback)
        )

    return out.reset_index(drop=True)


def load_cor_manifest(path: Path) -> pd.DataFrame:
    """Load the corruption manifest into the unified schema.

    Renames ``cor_path`` → ``scan_path``, marks ``is_reference=False``,
    preserves ``corruption_type``, ``severity``, ``dataset_tag``.
    """
    df = pd.read_csv(path)
    required = {_COR_PATH_COLUMN, TYPE_COLUMN, SEVERITY_COLUMN, DATASET_TAG_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise typer.BadParameter(
            f"{path} missing required columns: {sorted(missing)}"
        )

    out = pd.DataFrame()
    out[SCAN_COLUMN] = df[_COR_PATH_COLUMN].map(lambda p: str(Path(p).resolve()))
    out[IS_REFERENCE_COLUMN] = False
    out[TYPE_COLUMN] = df[TYPE_COLUMN].astype(str)
    out[SEVERITY_COLUMN] = df[SEVERITY_COLUMN].map(_coerce_severity)
    out[DATASET_TAG_COLUMN] = df[DATASET_TAG_COLUMN].astype(str)
    return out.reset_index(drop=True)


def _coerce_severity(value: object) -> int:
    """Cast severity to int, tolerating strings like ``"3"`` or ``3.0``."""
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def build_seg_map(manifest_paths: list[Path]) -> dict[str, Path]:
    """Return ``{resolved_input_path: resolved_seg_path}`` across all manifests.

    Mirrors the join primitive used by ``code/04_compute_preference.py``.
    Unlike 04, this one doesn't filter on status because even a failed
    segmentation's scan has a valid path for IQM extraction without a mask
    — but in practice Phase 03 only writes ``seg_path`` for rows that
    succeeded, so status filtering is implicit.

    Raises:
        ValueError: if two manifests disagree on the seg path for the same
            input path (signals accidental double-segmentation).
    """
    mapping: dict[str, Path] = {}
    for manifest_path in manifest_paths:
        df = pd.read_csv(manifest_path)
        required = {_SYNTHSEG_INPUT_COLUMN, _SYNTHSEG_SEG_COLUMN}
        missing = required - set(df.columns)
        if missing:
            raise typer.BadParameter(
                f"{manifest_path} missing synthseg columns {sorted(missing)}"
            )
        for _, row in df.iterrows():
            input_key = str(Path(row[_SYNTHSEG_INPUT_COLUMN]).resolve())
            seg_path = Path(row[_SYNTHSEG_SEG_COLUMN]).resolve()
            existing = mapping.get(input_key)
            if existing is not None and existing != seg_path:
                raise ValueError(
                    f"Conflicting seg paths for {input_key}: "
                    f"{existing} vs {seg_path}"
                )
            mapping[input_key] = seg_path
    return mapping


def attach_seg_paths(
    unified: pd.DataFrame, seg_map: dict[str, Path]
) -> pd.DataFrame:
    """Left-join ``seg_path`` onto the unified frame and drop rows without one.

    Rows whose scan_path is absent from seg_map OR whose seg file doesn't
    exist on disk are dropped with an INFO log (count only, not paths, to
    keep logs manageable for larger runs). The dropped rows are still in the
    input CSVs — not lost, just not scored.
    """
    unified = unified.copy()
    unified[SEG_COLUMN] = unified[SCAN_COLUMN].map(
        lambda p: seg_map.get(p)
    )

    missing_seg_mask = unified[SEG_COLUMN].isna()
    missing_seg = int(missing_seg_mask.sum())
    if missing_seg:
        logger.warning(
            "dropped %d rows (no matching seg in synthseg manifest)", missing_seg
        )
    unified = unified[~missing_seg_mask]

    nonexistent_mask = unified[SEG_COLUMN].map(
        lambda p: p is None or not Path(p).is_file()
    )
    nonexistent = int(nonexistent_mask.sum())
    if nonexistent:
        logger.warning(
            "dropped %d rows (seg file does not exist on disk)", nonexistent
        )
    unified = unified[~nonexistent_mask]

    return unified.reset_index(drop=True)


# ──────────────────────────────────────────────
# CSV I/O + resume
# ──────────────────────────────────────────────


def load_existing(output_file: Path) -> set[str]:
    """Return the set of ``scan_path`` values already in ``output_file``."""
    if not output_file.exists() or output_file.stat().st_size == 0:
        return set()
    with output_file.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or SCAN_COLUMN not in reader.fieldnames:
            return set()
        return {row[SCAN_COLUMN] for row in reader if row.get(SCAN_COLUMN)}


def append_row(output_file: Path, row: IQMRow) -> None:
    """Append one IQM row to ``output_file`` (crash-safe per write)."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    is_new = not output_file.exists() or output_file.stat().st_size == 0
    record: dict[str, object] = {
        SCAN_COLUMN: row.scan_path,
        IS_REFERENCE_COLUMN: row.is_reference,
        TYPE_COLUMN: row.corruption_type,
        SEVERITY_COLUMN: row.severity,
        DATASET_TAG_COLUMN: row.dataset_tag,
    }
    for key in IQM_KEYS:
        record[key] = row.iqms.get(key, math.nan)
    with output_file.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        if is_new:
            writer.writeheader()
        writer.writerow(record)
        handle.flush()


# ──────────────────────────────────────────────
# Per-scan scoring
# ──────────────────────────────────────────────


def _nan_iqms() -> dict[str, float]:
    return {k: math.nan for k in IQM_KEYS}


def score_scan(
    scan_path: str,
    seg_path: Path,
    *,
    is_reference: bool,
    corruption_type: str,
    severity: int,
    dataset_tag: str,
) -> IQMRow:
    """Compute IQMs for one scan; return an :class:`IQMRow`.

    Failures in ``extract_iqms`` are caught here and converted to a NaN-filled
    row so the per-scan loop in :func:`main` can continue without aborting
    the batch.
    """
    try:
        iqms = extract_iqms(Path(scan_path), seg_path=seg_path)
    except Exception as exc:  # noqa: BLE001 — catch-all is intentional
        logger.warning("IQM failure %s: %s", scan_path, exc)
        iqms = _nan_iqms()
    # Coerce all values to float and backfill missing keys with NaN.
    iqms_clean = {k: float(iqms.get(k, math.nan)) for k in IQM_KEYS}
    return IQMRow(
        scan_path=scan_path,
        iqms=iqms_clean,
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
    failed: int,
    output_file: Path,
    console: Console,
) -> None:
    table = Table(title="Phase 5 IQM extraction")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("total scans", str(total))
    table.add_row("already in CSV (skipped)", str(already_done))
    table.add_row("processed this run", str(processed))
    table.add_row("IQM failures (NaN rows)", str(failed))
    table.add_row("output file", str(output_file))
    console.print(table)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


@app.command()
def main(
    ref_manifest: list[Path] = typer.Option(
        [],
        "--ref-manifest",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Reference-scan manifest (repeatable). Accepts `filepath`, "
        "`ref_path`, or `scan_path` as the scan-path column.",
    ),
    cor_manifest: Path = typer.Option(
        ...,
        "--cor-manifest",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="corruption_manifest.csv from Phase 02/02b.",
    ),
    synthseg_manifest: list[Path] = typer.Option(
        ...,
        "--synthseg-manifest",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="synthseg_manifest.csv from Phase 03 (repeatable).",
    ),
    output_file: Path = typer.Option(
        ...,
        "--output-file",
        resolve_path=True,
        help="Aggregate iqm_features.csv (one row per scan).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Log the first 10 pending scans and return."
    ),
) -> None:
    """Compute signal-based IQMs for every reference + corrupted scan."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )

    frames: list[pd.DataFrame] = []
    for path in ref_manifest:
        frames.append(load_ref_manifest(path))
    frames.append(load_cor_manifest(cor_manifest))

    unified = pd.concat(frames, ignore_index=True)
    logger.info(
        "Unified manifest: %d ref rows, %d cor rows",
        int(unified[IS_REFERENCE_COLUMN].sum()),
        int((~unified[IS_REFERENCE_COLUMN]).sum()),
    )

    seg_map = build_seg_map(list(synthseg_manifest))
    logger.info("Loaded %d entries from %d synthseg manifest(s)", len(seg_map), len(synthseg_manifest))

    unified = attach_seg_paths(unified, seg_map)

    done = load_existing(output_file)
    pending = unified[~unified[SCAN_COLUMN].isin(done)].reset_index(drop=True)
    logger.info(
        "Plan: %d total, %d already done, %d pending",
        len(unified),
        len(done),
        len(pending),
    )

    if dry_run:
        for _, row in pending.head(10).iterrows():
            logger.info(
                "  would score: scan=%s seg=%s ref=%s type=%s sev=%s tag=%s",
                row[SCAN_COLUMN],
                row[SEG_COLUMN],
                row[IS_REFERENCE_COLUMN],
                row[TYPE_COLUMN],
                row[SEVERITY_COLUMN],
                row[DATASET_TAG_COLUMN],
            )
        if len(pending) > 10:
            logger.info("  ... (%d more)", len(pending) - 10)
        return

    processed = 0
    failed = 0
    iterator = (
        tqdm(pending.itertuples(index=False), total=len(pending), desc="iqm", unit="scan")
        if len(pending) > 1
        else pending.itertuples(index=False)
    )
    for row in iterator:
        iqm_row = score_scan(
            scan_path=row.scan_path,
            seg_path=Path(row.seg_path),
            is_reference=bool(row.is_reference),
            corruption_type=str(row.corruption_type),
            severity=int(row.severity),
            dataset_tag=str(row.dataset_tag),
        )
        append_row(output_file, iqm_row)
        processed += 1
        if all(math.isnan(v) for v in iqm_row.iqms.values()):
            failed += 1

    _print_summary(
        total=len(unified),
        already_done=len(done),
        processed=processed,
        failed=failed,
        output_file=output_file,
        console=Console(),
    )


if __name__ == "__main__":
    app()
