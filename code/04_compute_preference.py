#!/usr/bin/env python3
"""NeuroQC Phase 4 — machine preference scoring (Dice + thickness shift).

Reads ``corruption_manifest.csv`` (reference ↔ corrupted scan pairs), locates
the corresponding SynthSeg segmentations under ``--synthseg-dir`` and the
per-scan thickness vectors in ``cortical_thickness.csv``, then writes one row
per pair to ``--output-file``::

    ref_path, cor_path, corruption_type, severity,
    mean_dice, hippocampus_dice, cortex_dice, ventricle_dice, thalamus_dice,
    caudate_dice, putamen_dice, brainstem_dice, cerebellum_dice,
    ref_mean_thickness, cor_mean_thickness, thickness_shift

Dice is computed per-structure via ``nobrainer.qc.preference.compute_dice_preference``
(reference vs. corrupted SynthSeg output). The thickness shift is the mean of
``|ref_region - cor_region|`` across the 70 Desikan-Killiany columns present in
the thickness table; it collapses the per-region vector into a single scalar
sensitive to corruption-induced thickness drift.

Inputs:
    --corruption-manifest   CSV with at minimum ref_path, cor_path,
                            corruption_type, severity columns.
    --synthseg-dir          Root of ``*_synthseg.nii.gz`` outputs (Phase 3).
                            Searched recursively; match key is the scan stem
                            (filename minus ``.nii.gz`` or ``.nii``).
    --thickness-file        cortical_thickness.csv produced by Phase 3b. Rows
                            whose ``scan_path`` is missing from the table fall
                            back to NaN thickness columns (Dice is still written).
    --output-file           Aggregate machine_preference.csv.
    --dry-run               Log the first 10 pending pairs and return.

Resume:
    If ``--output-file`` already exists, (ref_path, cor_path) pairs already
    recorded are not recomputed. Writes are append-mode with a per-row flush,
    so an interrupted run continues on the next invocation.

Usage:
    python code/04_compute_preference.py \\
        --corruption-manifest results/tables/corruption_manifest.csv \\
        --synthseg-dir        data/derivatives/synthseg \\
        --thickness-file      results/tables/cortical_thickness.csv \\
        --output-file         results/tables/machine_preference.csv
"""

from __future__ import annotations

import csv
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from tqdm import tqdm

from nobrainer.qc.preference import compute_dice_preference

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

REF_COLUMN: str = "ref_path"
COR_COLUMN: str = "cor_path"
TYPE_COLUMN: str = "corruption_type"
SEVERITY_COLUMN: str = "severity"

DICE_COLUMNS: tuple[str, ...] = (
    "mean_dice",
    "hippocampus_dice",
    "cortex_dice",
    "ventricle_dice",
    "thalamus_dice",
    "caudate_dice",
    "putamen_dice",
    "brainstem_dice",
    "cerebellum_dice",
)

THICKNESS_REF_COLUMN: str = "ref_mean_thickness"
THICKNESS_COR_COLUMN: str = "cor_mean_thickness"
THICKNESS_SHIFT_COLUMN: str = "thickness_shift"

OUTPUT_COLUMNS: tuple[str, ...] = (
    REF_COLUMN,
    COR_COLUMN,
    TYPE_COLUMN,
    SEVERITY_COLUMN,
    *DICE_COLUMNS,
    THICKNESS_REF_COLUMN,
    THICKNESS_COR_COLUMN,
    THICKNESS_SHIFT_COLUMN,
)

# Columns inside cortical_thickness.csv.
THICKNESS_SCAN_COLUMN: str = "scan_path"
THICKNESS_SEG_PATH_COLUMN: str = "seg_path"
THICKNESS_MEAN_COLUMN: str = "mean_thickness"
THICKNESS_REGION_SUFFIX: str = "_thickness"

# Columns inside the synthseg manifest (03's output; matches
# code/03_run_synthseg.py::MANIFEST_COLUMNS).
SYNTHSEG_INPUT_COLUMN: str = "input_path"
SYNTHSEG_SEG_COLUMN: str = "seg_path"
SYNTHSEG_STATUS_COLUMN: str = "status"
# Both "ok" (just segmented) and "skipped" (seg already on disk from a prior
# run) have valid seg files; "failed" rows do not.
SYNTHSEG_VALID_STATUSES: frozenset[str] = frozenset({"ok", "skipped"})

_SEG_SUFFIX: str = "_synthseg.nii.gz"

# Mode flag for seg_map lookup: "manifest" keys on resolved input path,
# "dir" keys on the scan filename stem.
_MODE_MANIFEST: str = "manifest"
_MODE_DIR: str = "dir"

# Per-structure Dice CSV (long format; one row per (pair, label_id))
PS_LABEL_ID_COLUMN: str = "label_id"
PS_LABEL_NAME_COLUMN: str = "label_name"
PS_DICE_COLUMN: str = "dice"
PS_OUTPUT_COLUMNS: tuple[str, ...] = (
    REF_COLUMN,
    COR_COLUMN,
    TYPE_COLUMN,
    SEVERITY_COLUMN,
    PS_LABEL_ID_COLUMN,
    PS_LABEL_NAME_COLUMN,
    PS_DICE_COLUMN,
)
# Background label is excluded from per-structure output (no anatomical
# meaning, dominates volume on partial-FOV slabs).
_BACKGROUND_LABEL: int = 0

logger = logging.getLogger(__name__)
app = typer.Typer(
    help="NeuroQC Phase 4 — machine preference (Dice + thickness).",
    add_completion=False,
)


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────


@dataclass(frozen=True)
class PreferenceRow:
    """One reference ↔ corrupted pair aggregated into the output CSV.

    Attributes:
        ref_path: Reference scan path verbatim from the corruption manifest
            (not resolved — preserves whatever form the upstream script used
            so joins stay cheap).
        cor_path: Corrupted scan path verbatim from the corruption manifest.
        corruption_type: Corruption family label (e.g. ``motion``, ``ghosting``).
        severity: Severity level tag from the manifest (kept as the original
            string or numeric, whatever the manifest uses).
        dice: ``{column_name: value}`` for every column in DICE_COLUMNS.
        ref_mean_thickness: Whole-cortex mean thickness of the reference scan
            (from the thickness table). NaN if the scan is not in the table.
        cor_mean_thickness: Same, for the corrupted scan.
        thickness_shift: ``nanmean(|ref_regions - cor_regions|)`` over the 70
            ``ctx-*_thickness`` columns. NaN if either side is absent or if no
            region has a paired value.
    """

    ref_path: str
    cor_path: str
    corruption_type: str
    severity: str
    dice: dict[str, float] = field(default_factory=dict)
    ref_mean_thickness: float = math.nan
    cor_mean_thickness: float = math.nan
    thickness_shift: float = math.nan


# ──────────────────────────────────────────────
# Seg discovery + stem map
# ──────────────────────────────────────────────


def _scan_stem(path: Path | str) -> str:
    """Strip ``.nii.gz`` or ``.nii`` from a filename, return the bare stem.

    ``Path.stem`` only strips one suffix so ``x.nii.gz`` → ``x.nii`` — we want
    ``x``. Used as the join key between corruption_manifest paths and the
    rglob'd seg files.
    """
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return Path(name).stem


def discover_seg_map(synthseg_dir: Path) -> dict[str, Path]:
    """Return ``{scan_stem: resolved_seg_path}`` for every seg under ``synthseg_dir``.

    Fallback path used only when no ``--synthseg-manifest`` is provided. Works
    for single-tree runs where every scan has a unique basename (e.g. IXI). On
    FastMRI + corrupted outputs (which share basenames across ref and cor
    directories by design in ``nobrainer.qc.corrupt``), use the manifest-based
    :func:`load_synthseg_manifests` instead.

    Raises:
        ValueError: if two distinct seg files collapse to the same stem; this
            would silently pick the wrong seg on lookup and produce bogus Dice.
    """
    mapping: dict[str, Path] = {}
    collisions: list[tuple[str, Path, Path]] = []
    for seg in sorted(synthseg_dir.rglob(f"*{_SEG_SUFFIX}")):
        # seg filename = "<scan_stem>_synthseg.nii.gz" → key = "<scan_stem>"
        stem = seg.name[: -len(_SEG_SUFFIX)]
        resolved = seg.resolve()
        if stem in mapping and mapping[stem] != resolved:
            collisions.append((stem, mapping[stem], resolved))
        else:
            mapping[stem] = resolved
    if collisions:
        details = "; ".join(f"{s}: {a} vs {b}" for s, a, b in collisions[:3])
        raise ValueError(
            f"{len(collisions)} duplicate seg stems under {synthseg_dir}: {details}"
        )
    return mapping


def load_synthseg_manifests(manifest_paths: list[Path]) -> dict[str, Path]:
    """Build ``{resolved_input_path: resolved_seg_path}`` from synthseg manifests.

    Each input manifest must have ``input_path, seg_path, status`` columns
    (part of the schema written by ``code/03_run_synthseg.py``). Rows whose
    ``status`` is not in :data:`SYNTHSEG_VALID_STATUSES` are skipped — a
    ``failed`` row's ``seg_path`` may not exist on disk, and including it
    would cause later ``compute_dice_preference`` calls to error.

    Paths are resolved symmetrically (``Path(p).resolve()``) so downstream
    lookups with ``Path(scan).resolve()`` join cleanly regardless of whether
    03 or 02 passed absolute or relative paths.

    Raises:
        typer.BadParameter: if any manifest path does not exist.
        ValueError: on schema mismatch, or if two manifests disagree on the
            seg_path for the same input_path (signals double segmentation
            that would make Dice non-deterministic).
    """
    required_cols = {
        SYNTHSEG_INPUT_COLUMN,
        SYNTHSEG_SEG_COLUMN,
        SYNTHSEG_STATUS_COLUMN,
    }
    mapping: dict[str, Path] = {}
    for manifest_path in manifest_paths:
        if not manifest_path.is_file():
            raise typer.BadParameter(f"synthseg manifest not found: {manifest_path}")
        with manifest_path.open() as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing = required_cols - fieldnames
            if missing:
                raise ValueError(
                    f"{manifest_path} missing synthseg columns {sorted(missing)}; "
                    f"has {sorted(fieldnames)}"
                )
            for row in reader:
                if row.get(SYNTHSEG_STATUS_COLUMN) not in SYNTHSEG_VALID_STATUSES:
                    continue
                input_key = str(Path(row[SYNTHSEG_INPUT_COLUMN]).resolve())
                seg_path = Path(row[SYNTHSEG_SEG_COLUMN]).resolve()
                existing = mapping.get(input_key)
                if existing is not None and existing != seg_path:
                    raise ValueError(
                        f"Conflicting seg paths for {input_key}: "
                        f"{existing} vs {seg_path}"
                    )
                mapping[input_key] = seg_path
    return mapping


def _build_key_fn(mode: str) -> Callable[[str], str]:
    """Return the scan-path-to-seg-map-key function for the given mode.

    In manifest mode the seg_map is keyed by resolved input path, so every
    lookup resolves the incoming ref/cor path the same way. In dir mode the
    map is keyed by filename stem — cheap but breaks on ref/cor basename
    collisions, which is why manifest mode is preferred.
    """
    if mode == _MODE_MANIFEST:
        return lambda scan_path: str(Path(scan_path).resolve())
    if mode == _MODE_DIR:
        return _scan_stem
    raise ValueError(f"unknown seg-map mode: {mode!r}")


# ──────────────────────────────────────────────
# Thickness lookup
# ──────────────────────────────────────────────


def load_thickness_table(thickness_file: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load cortical_thickness.csv indexed by resolved seg-path strings.

    Phase 04 looks up thickness rows by SEG path (it has segs in hand from
    `seg_map`). Phase 3b's new schema (2026-04-25) writes both ``scan_path``
    (= input scan path when synthseg manifests are passed to 03b) and
    ``seg_path`` (= the SynthSeg output the row was computed from). Prefer
    the explicit ``seg_path`` column when present; fall back to ``scan_path``
    for legacy CSVs (where ``scan_path`` was the seg path).

    Returns:
        ``(df, region_columns)`` where ``df.index`` is the resolved seg-path
        string and ``region_columns`` is the list of ``ctx-*_thickness``
        columns (the ``mean_thickness`` column is excluded from the shift
        computation — it's reported separately).
    """
    if not thickness_file.is_file():
        raise typer.BadParameter(
            f"Thickness CSV not found: {thickness_file}. Run Phase 3b first."
        )
    df = pd.read_csv(thickness_file)
    # Use seg_path when present (new schema); else scan_path (legacy).
    if THICKNESS_SEG_PATH_COLUMN in df.columns:
        index_col = THICKNESS_SEG_PATH_COLUMN
    elif THICKNESS_SCAN_COLUMN in df.columns:
        index_col = THICKNESS_SCAN_COLUMN
        logger.info(
            "Thickness CSV has no '%s' column; falling back to '%s' "
            "(legacy 03b output where scan_path stored the seg path).",
            THICKNESS_SEG_PATH_COLUMN, THICKNESS_SCAN_COLUMN,
        )
    else:
        raise ValueError(
            f"{thickness_file} missing both '{THICKNESS_SEG_PATH_COLUMN}' and "
            f"'{THICKNESS_SCAN_COLUMN}' columns"
        )
    df[index_col] = df[index_col].map(lambda p: str(Path(p).resolve()))
    df = df.set_index(index_col)
    region_cols = [
        c
        for c in df.columns
        if c.endswith(THICKNESS_REGION_SUFFIX) and c != THICKNESS_MEAN_COLUMN
    ]
    return df, region_cols


def _thickness_for(
    seg_path: Path,
    thickness_df: pd.DataFrame,
) -> pd.Series | None:
    """Return the thickness row for ``seg_path`` or None if not in the table."""
    key = str(seg_path.resolve())
    if key not in thickness_df.index:
        return None
    return thickness_df.loc[key]


def compute_thickness_triplet(
    ref_seg: Path,
    cor_seg: Path,
    thickness_df: pd.DataFrame,
    region_cols: list[str],
) -> tuple[float, float, float]:
    """Return ``(ref_mean, cor_mean, mean(|ref - cor|) across regions)``.

    Any of the three values is NaN when the corresponding inputs are missing:
    - ``ref_mean`` NaN if ``ref_seg`` is absent from the thickness table.
    - ``cor_mean`` NaN if ``cor_seg`` is absent.
    - ``thickness_shift`` NaN if either side is absent, or if every region is
      NaN on at least one side (so no paired difference exists).
    """
    nan = math.nan
    ref_row = _thickness_for(ref_seg, thickness_df)
    cor_row = _thickness_for(cor_seg, thickness_df)

    ref_mean = (
        float(ref_row[THICKNESS_MEAN_COLUMN])
        if ref_row is not None and THICKNESS_MEAN_COLUMN in ref_row
        else nan
    )
    cor_mean = (
        float(cor_row[THICKNESS_MEAN_COLUMN])
        if cor_row is not None and THICKNESS_MEAN_COLUMN in cor_row
        else nan
    )

    if ref_row is None or cor_row is None or not region_cols:
        return ref_mean, cor_mean, nan

    diff = (
        ref_row[region_cols].astype(float) - cor_row[region_cols].astype(float)
    ).abs()
    shift = float(diff.mean(skipna=True)) if diff.notna().any() else nan
    return ref_mean, cor_mean, shift


# ──────────────────────────────────────────────
# Per-structure Dice (long-format output)
# ──────────────────────────────────────────────


def load_label_names(lut_path: Path | None) -> dict[int, str]:
    """Parse FreeSurferColorLUT.txt → ``{label_id: label_name}``.

    Lines starting with ``#`` and blank lines are skipped. Each data line
    has ``id  name  R  G  B  A`` with whitespace-separated fields; we keep
    the first two. Returns an empty mapping if ``lut_path`` is ``None`` or
    not a file (callers fall back to ``f"label_{id}"`` names).
    """
    if lut_path is None or not lut_path.is_file():
        if lut_path is not None:
            logger.info(
                "FreeSurfer LUT not found at %s; per-structure rows will "
                "use 'label_<id>' names.",
                lut_path,
            )
        return {}
    mapping: dict[int, str] = {}
    with lut_path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                lid = int(parts[0])
            except ValueError:
                continue
            mapping[lid] = parts[1]
    return mapping


def _dice_per_unique_label(
    ref: "torch.Tensor", cor: "torch.Tensor"
) -> dict[int, float]:
    """Compute Dice for every label in ``unique(ref) ∪ unique(cor)``.

    Excludes background (label 0). NaN if a label is absent from both
    sides (shouldn't happen since we built the union, but guarded anyway).
    Pure torch + no nobrainer dependency for the per-structure computation.
    """
    import torch

    label_set = set(int(x) for x in torch.unique(ref).tolist())
    label_set |= set(int(x) for x in torch.unique(cor).tolist())
    label_set.discard(_BACKGROUND_LABEL)
    out: dict[int, float] = {}
    for lid in sorted(label_set):
        ref_mask = ref == lid
        cor_mask = cor == lid
        ref_sum = ref_mask.sum()
        cor_sum = cor_mask.sum()
        if ref_sum == 0 and cor_sum == 0:
            out[lid] = math.nan
            continue
        intersection = (ref_mask & cor_mask).sum().float()
        out[lid] = (2.0 * intersection / (ref_sum + cor_sum).float()).item()
    return out


def compute_per_label_dice(
    ref_seg_path: Path, cor_seg_path: Path
) -> dict[int, float]:
    """Compute Dice for every unique non-background label in the seg pair.

    Resamples ``cor`` into ``ref`` space with nearest-neighbour interpolation
    if shapes differ — same defensive pattern as ``nobrainer.qc.metrics``
    uses for IQM extraction on anisotropic FastMRI vs 1 mm-iso SynthSeg out.
    """
    import nibabel as nib
    import torch

    ref_nii = nib.load(str(ref_seg_path))
    cor_nii = nib.load(str(cor_seg_path))
    if ref_nii.shape != cor_nii.shape:
        from nibabel.processing import resample_from_to

        cor_nii = resample_from_to(cor_nii, ref_nii, order=0)
    ref = torch.from_numpy(ref_nii.get_fdata()).long()
    cor = torch.from_numpy(cor_nii.get_fdata()).long()
    return _dice_per_unique_label(ref, cor)


def load_existing_per_structure(
    output_file: Path,
) -> set[tuple[str, str]]:
    """Return ``{(ref_path, cor_path)}`` pairs already in the long CSV.

    Resume granularity is the pair, not the (pair, label_id) triple — we
    always write all labels for a pair atomically, so any partial output
    means the pair was interrupted mid-write and should be re-emitted.
    """
    if not output_file.exists() or output_file.stat().st_size == 0:
        return set()
    with output_file.open() as handle:
        reader = csv.DictReader(handle)
        if (
            reader.fieldnames is None
            or REF_COLUMN not in reader.fieldnames
            or COR_COLUMN not in reader.fieldnames
        ):
            return set()
        return {(row[REF_COLUMN], row[COR_COLUMN]) for row in reader}


def append_per_structure_rows(
    output_file: Path,
    *,
    ref_path: str,
    cor_path: str,
    corruption_type: str,
    severity: str,
    dice_per_label: dict[int, float],
    label_names: dict[int, str],
) -> int:
    """Append one row per ``(pair, label_id)`` to the long CSV.

    Returns the number of rows written. Crash-safe per write (flush after
    the for-loop block).
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    is_new = not output_file.exists() or output_file.stat().st_size == 0
    n_written = 0
    with output_file.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PS_OUTPUT_COLUMNS))
        if is_new:
            writer.writeheader()
        for lid, dice in sorted(dice_per_label.items()):
            writer.writerow(
                {
                    REF_COLUMN: ref_path,
                    COR_COLUMN: cor_path,
                    TYPE_COLUMN: corruption_type,
                    SEVERITY_COLUMN: severity,
                    PS_LABEL_ID_COLUMN: lid,
                    PS_LABEL_NAME_COLUMN: label_names.get(lid, f"label_{lid}"),
                    PS_DICE_COLUMN: dice,
                }
            )
            n_written += 1
        handle.flush()
    return n_written


# ──────────────────────────────────────────────
# CSV I/O + resume
# ──────────────────────────────────────────────


def load_existing(output_file: Path) -> set[tuple[str, str]]:
    """Return ``{(ref_path, cor_path)}`` pairs already recorded in the output.

    Empty set if the file does not exist or is empty. Used to skip already
    computed pairs on a resumed run.
    """
    if not output_file.exists() or output_file.stat().st_size == 0:
        return set()
    with output_file.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if (
            reader.fieldnames is None
            or REF_COLUMN not in reader.fieldnames
            or COR_COLUMN not in reader.fieldnames
        ):
            return set()
        return {(row[REF_COLUMN], row[COR_COLUMN]) for row in reader}


def append_row(output_file: Path, row: PreferenceRow) -> None:
    """Append a single preference row to ``output_file`` (crash-safe per write)."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    is_new = not output_file.exists() or output_file.stat().st_size == 0
    record: dict[str, object] = {
        REF_COLUMN: row.ref_path,
        COR_COLUMN: row.cor_path,
        TYPE_COLUMN: row.corruption_type,
        SEVERITY_COLUMN: row.severity,
        THICKNESS_REF_COLUMN: row.ref_mean_thickness,
        THICKNESS_COR_COLUMN: row.cor_mean_thickness,
        THICKNESS_SHIFT_COLUMN: row.thickness_shift,
    }
    for col in DICE_COLUMNS:
        record[col] = row.dice.get(col, math.nan)
    with output_file.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        if is_new:
            writer.writeheader()
        writer.writerow(record)
        handle.flush()


# ──────────────────────────────────────────────
# Per-pair scoring
# ──────────────────────────────────────────────


def score_pair(
    ref_path: str,
    cor_path: str,
    corruption_type: str,
    severity: str,
    seg_map: dict[str, Path],
    key_fn: Callable[[str], str],
    thickness_df: pd.DataFrame,
    region_cols: list[str],
) -> PreferenceRow | None:
    """Score one (ref, cor) pair. Returns None if either seg is missing.

    Missing seg is treated as a skip (with a warning), not a NaN row — a pair
    with no seg can't produce meaningful Dice, and writing it anyway would
    mix "segmentation failed" with "segmentation ran but quality was bad".
    Thickness-only rows would be equally ambiguous, so we keep skips clean.

    ``key_fn`` computes the lookup key from the incoming scan path; its form
    depends on whether ``seg_map`` came from a synthseg manifest (resolved
    full path) or a directory rglob (filename stem). See :func:`_build_key_fn`.
    """
    ref_key = key_fn(ref_path)
    cor_key = key_fn(cor_path)
    ref_seg = seg_map.get(ref_key)
    cor_seg = seg_map.get(cor_key)
    if ref_seg is None or cor_seg is None:
        logger.warning(
            "skipping pair: seg missing for %s",
            ref_key if ref_seg is None else cor_key,
        )
        return None

    dice = compute_dice_preference(ref_seg, cor_seg)
    ref_mean, cor_mean, shift = compute_thickness_triplet(
        ref_seg, cor_seg, thickness_df, region_cols
    )
    return PreferenceRow(
        ref_path=ref_path,
        cor_path=cor_path,
        corruption_type=corruption_type,
        severity=severity,
        dice=dice,
        ref_mean_thickness=ref_mean,
        cor_mean_thickness=cor_mean,
        thickness_shift=shift,
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
    table = Table(title="Phase 4 machine preference")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("total pairs", str(total))
    table.add_row("already in CSV (skipped)", str(already_done))
    table.add_row("processed this run", str(processed))
    table.add_row("skipped (missing seg)", str(skipped))
    table.add_row("output file", str(output_file))
    console.print(table)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


@app.command()
def main(
    corruption_manifest: Path = typer.Option(
        ...,
        "--corruption-manifest",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="corruption_manifest.csv with ref_path/cor_path/corruption_type/severity.",
    ),
    synthseg_manifest: list[Path] = typer.Option(
        [],
        "--synthseg-manifest",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help=(
            "Path to a synthseg manifest CSV (repeatable). Preferred over "
            "--synthseg-dir because it keys off resolved input paths and so "
            "tolerates ref/cor scans sharing a basename."
        ),
    ),
    synthseg_dir: Path | None = typer.Option(
        None,
        "--synthseg-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help=(
            "Fallback: rglob this directory for *_synthseg.nii.gz files and "
            "key the lookup by filename stem. Use only when every scan has "
            "a unique basename (e.g. IXI). Ignored if --synthseg-manifest "
            "is passed."
        ),
    ),
    thickness_file: Path = typer.Option(
        ...,
        "--thickness-file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="cortical_thickness.csv from Phase 3b.",
    ),
    output_file: Path = typer.Option(
        ...,
        "--output-file",
        resolve_path=True,
        help="Aggregate machine_preference.csv (one row per pair).",
    ),
    per_structure_output: Path | None = typer.Option(
        None,
        "--per-structure-output",
        resolve_path=True,
        help=(
            "If set, also write a long-format per-structure Dice CSV with "
            "one row per (ref, cor, label_id). Required for visualize.py "
            "Fig 8 (per-structure heatmap)."
        ),
    ),
    label_name_source: Path | None = typer.Option(
        None,
        "--label-name-source",
        resolve_path=True,
        help=(
            "FreeSurferColorLUT.txt for label_id → label_name resolution in "
            "the per-structure output. Defaults to "
            "$FREESURFER_HOME/FreeSurferColorLUT.txt; absent → label_<id> names."
        ),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Log the first 10 pending pairs and return."
    ),
) -> None:
    """Score every ref↔cor pair from the corruption manifest."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )

    manifest_df = pd.read_csv(corruption_manifest)
    required = {REF_COLUMN, COR_COLUMN, TYPE_COLUMN, SEVERITY_COLUMN}
    missing = required - set(manifest_df.columns)
    if missing:
        raise typer.BadParameter(
            f"{corruption_manifest} missing required columns: {sorted(missing)}"
        )

    if synthseg_manifest:
        seg_map = load_synthseg_manifests(synthseg_manifest)
        mode = _MODE_MANIFEST
        source_label = f"{len(synthseg_manifest)} synthseg manifest(s)"
    elif synthseg_dir is not None:
        seg_map = discover_seg_map(synthseg_dir)
        mode = _MODE_DIR
        source_label = str(synthseg_dir)
    else:
        raise typer.BadParameter(
            "Pass at least one --synthseg-manifest or --synthseg-dir."
        )
    key_fn = _build_key_fn(mode)

    if not seg_map:
        logger.warning("No seg entries resolved from %s", source_label)
        return

    thickness_df, region_cols = load_thickness_table(thickness_file)
    logger.info(
        "Loaded %d thickness rows across %d region columns",
        len(thickness_df),
        len(region_cols),
    )

    done = load_existing(output_file)
    # Per-structure output is optional. When enabled, resume independently:
    # a pair already in machine_preference.csv but missing from per_structure
    # gets its per-label Dice computed without re-running the per-pair logic.
    write_per_structure = per_structure_output is not None
    label_names: dict[int, str] = {}
    per_structure_done: set[tuple[str, str]] = set()
    if write_per_structure:
        lut_source = label_name_source
        if lut_source is None:
            import os

            fs_home = os.environ.get("FREESURFER_HOME")
            if fs_home:
                lut_source = Path(fs_home) / "FreeSurferColorLUT.txt"
        label_names = load_label_names(lut_source)
        per_structure_done = load_existing_per_structure(per_structure_output)
        logger.info(
            "Per-structure mode: output=%s, %d labels in LUT, %d pairs already recorded",
            per_structure_output,
            len(label_names),
            len(per_structure_done),
        )

    pending_rows: list[dict[str, object]] = []
    for _, row in manifest_df.iterrows():
        key = (str(row[REF_COLUMN]), str(row[COR_COLUMN]))
        in_pref = key in done
        in_per_structure = key in per_structure_done
        # Pending if either output is missing the pair.
        if in_pref and (not write_per_structure or in_per_structure):
            continue
        pending_rows.append(row.to_dict())

    logger.info(
        "Plan: %d total, %d already done, %d pending",
        len(manifest_df),
        len(done),
        len(pending_rows),
    )

    if dry_run:
        for row in pending_rows[:10]:
            logger.info(
                "  would score: ref=%s cor=%s type=%s sev=%s",
                row[REF_COLUMN],
                row[COR_COLUMN],
                row[TYPE_COLUMN],
                row[SEVERITY_COLUMN],
            )
        if len(pending_rows) > 10:
            logger.info("  ... (%d more)", len(pending_rows) - 10)
        return

    processed = 0
    skipped = 0
    per_structure_rows_written = 0
    iterator = (
        tqdm(pending_rows, desc="preference", unit="pair")
        if len(pending_rows) > 1
        else pending_rows
    )
    for row in iterator:
        ref_path_str = str(row[REF_COLUMN])
        cor_path_str = str(row[COR_COLUMN])
        ctype = str(row[TYPE_COLUMN])
        sev = str(row[SEVERITY_COLUMN])
        key = (ref_path_str, cor_path_str)

        # Per-pair preference (skip if already recorded).
        if key not in done:
            pref = score_pair(
                ref_path=ref_path_str,
                cor_path=cor_path_str,
                corruption_type=ctype,
                severity=sev,
                seg_map=seg_map,
                key_fn=key_fn,
                thickness_df=thickness_df,
                region_cols=region_cols,
            )
            if pref is None:
                skipped += 1
                # Per-structure also skipped — can't compute Dice without segs.
                continue
            append_row(output_file, pref)
            processed += 1

        # Per-structure long CSV (optional, independent resume).
        if write_per_structure and key not in per_structure_done:
            ref_seg = seg_map.get(key_fn(ref_path_str))
            cor_seg = seg_map.get(key_fn(cor_path_str))
            if ref_seg is None or cor_seg is None:
                logger.warning(
                    "Skipping per-structure for pair: seg missing for %s",
                    ref_path_str if ref_seg is None else cor_path_str,
                )
                continue
            dice_per_label = compute_per_label_dice(ref_seg, cor_seg)
            per_structure_rows_written += append_per_structure_rows(
                per_structure_output,  # type: ignore[arg-type]
                ref_path=ref_path_str,
                cor_path=cor_path_str,
                corruption_type=ctype,
                severity=sev,
                dice_per_label=dice_per_label,
                label_names=label_names,
            )

    if write_per_structure:
        logger.info(
            "Per-structure: %d rows appended to %s",
            per_structure_rows_written,
            per_structure_output,
        )

    _print_summary(
        total=len(manifest_df),
        already_done=len(done),
        processed=processed,
        skipped=skipped,
        output_file=output_file,
        console=Console(),
    )


if __name__ == "__main__":
    app()
