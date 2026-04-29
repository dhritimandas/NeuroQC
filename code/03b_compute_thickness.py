#!/usr/bin/env python3
"""NeuroQC Phase 3b — per-region cortical thickness from SynthSeg parcellation.

Walks ``--synthseg-dir`` recursively for ``*_synthseg.nii.gz`` files produced by
``code/03_run_synthseg.py`` (with ``--parc`` on, so Desikan-Killiany labels
1001-1035 / 2001-2035 are present) and writes one row per scan to
``--output-file`` with columns::

    scan_path, seg_path, mean_thickness, <ctx-lh-...>_thickness, ..., <ctx-rh-...>_thickness

``scan_path`` stores the INPUT scan path (the NIfTI Phase 03 segmented FROM)
when ``--synthseg-manifest`` is provided. ``seg_path`` always stores the
``*_synthseg.nii.gz`` path (the file 03b actually loaded for thickness).
Without a manifest, ``scan_path`` falls back to the seg path with a deprecation
warning — downstream phases (04, 09, visualize.py) prefer the input-path
semantics, so passing a manifest is the recommended path.

Thickness is computed post-hoc from the segmentation alone — SynthSeg has no
``--thickness`` flag and no ``_thickness.csv`` sidecar, so we derive it via the
volume-based DiReCT-style approximation of Das et al. 2009:

    thickness_voxel = d(GM voxel -> nearest WM voxel)
                    + d(GM voxel -> nearest non-GM/non-WM voxel)

and mean per region across all voxels labelled with that region's integer.

Inputs:
    --synthseg-dir      Root of SynthSeg outputs (mirrors Phase 3 --output-dir).
    --output-file       Aggregate CSV written one row per scan.
    --freesurfer-home   FreeSurfer install root (for FreeSurferColorLUT.txt);
                        defaults to $FREESURFER_HOME env var.
    --dry-run           Log the first 10 pending scans and return.

Resume:
    If --output-file already exists, rows whose ``scan_path`` is already present
    are not recomputed. Writes are crash-safe at row granularity (append mode,
    flushed after each scan), so an interrupted run continues from where it
    stopped on the next invocation.

Usage:
    python code/03b_compute_thickness.py \\
        --synthseg-dir data/derivatives/synthseg \\
        --output-file  results/tables/cortical_thickness.csv
"""

from __future__ import annotations

import csv
import logging
import math
import multiprocessing as mp
import os
import re
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import typer
from monai.transforms.utils import distance_transform_edt
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from tqdm import tqdm

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

WM_LABELS: tuple[int, ...] = (2, 41)
DK_LH_LABELS: tuple[int, ...] = tuple(range(1001, 1036))
DK_RH_LABELS: tuple[int, ...] = tuple(range(2001, 2036))
CORTEX_LABELS: tuple[int, ...] = DK_LH_LABELS + DK_RH_LABELS

SCAN_COLUMN: str = "scan_path"
SEG_PATH_COLUMN: str = "seg_path"
MEAN_COLUMN: str = "mean_thickness"

# Synthseg manifest columns (must match code/03_run_synthseg.py).
_SYNTHSEG_INPUT_COLUMN: str = "input_path"
_SYNTHSEG_SEG_COLUMN: str = "seg_path"
_SYNTHSEG_STATUS_COLUMN: str = "status"
_SYNTHSEG_VALID_STATUSES: frozenset[str] = frozenset({"ok", "skipped"})

_LUT_LINE_RE = re.compile(r"^\s*(\d+)\s+(\S+)\s+")

logger = logging.getLogger(__name__)
app = typer.Typer(
    help="NeuroQC Phase 3b — per-region cortical thickness from SynthSeg.",
    add_completion=False,
)


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────


@dataclass(frozen=True)
class ThicknessRow:
    """One scan's thickness aggregation.

    Attributes:
        scan_path: Path to the INPUT scan (the NIfTI Phase 03 segmented FROM).
            When a synthseg manifest is provided to ``main()``, this is the
            ``input_path`` from the manifest. Without the manifest it falls
            back to the seg path with a deprecation warning.
        seg_path: Path to the ``*_synthseg.nii.gz`` this row was computed
            from (always populated; what 03b actually loaded).
        mean_thickness: Mean thickness across all cortical voxels, in mm. NaN
            if no cortical voxel is present in the segmentation.
        per_region: {region_name: mean_mm}, one entry per CORTEX_LABELS label.
            Value is NaN for regions absent from the scan.
    """

    scan_path: Path
    seg_path: Path
    mean_thickness: float
    per_region: dict[str, float]


# ──────────────────────────────────────────────
# LUT + seg discovery
# ──────────────────────────────────────────────


def load_region_names(lut_path: Path) -> dict[int, str]:
    """Parse ``FreeSurferColorLUT.txt`` and return ``{int: name}`` for CORTEX_LABELS.

    The LUT is a whitespace-delimited table with comment lines starting ``#``;
    each data row is ``<label_int>  <name>  <R>  <G>  <B>  <A>``. We only keep
    labels in :data:`CORTEX_LABELS` so downstream column construction can't
    accidentally pull in non-cortical labels.

    Raises:
        typer.BadParameter: if ``lut_path`` does not exist or any label in
            CORTEX_LABELS is missing from the file (signals a LUT shipped with
            an older FreeSurfer release and likely incompatible seg outputs).
    """
    if not lut_path.exists():
        raise typer.BadParameter(
            f"FreeSurferColorLUT.txt not found at {lut_path}. Pass "
            "--freesurfer-home or set $FREESURFER_HOME."
        )
    wanted = set(CORTEX_LABELS)
    names: dict[int, str] = {}
    with lut_path.open("r") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            match = _LUT_LINE_RE.match(raw)
            if match is None:
                continue
            label = int(match.group(1))
            if label in wanted:
                names[label] = match.group(2)
    missing = wanted - set(names)
    if missing:
        raise typer.BadParameter(
            f"LUT at {lut_path} is missing {len(missing)} Desikan-Killiany "
            f"labels (e.g. {sorted(missing)[:3]}). Check FreeSurfer version."
        )
    return names


def discover_seg_files(synthseg_dir: Path) -> list[Path]:
    """Return a sorted list of ``*_synthseg.nii.gz`` files under ``synthseg_dir``.

    Deterministic ordering so the aggregate CSV is reproducible run to run.
    """
    return sorted(synthseg_dir.rglob("*_synthseg.nii.gz"))


# ──────────────────────────────────────────────
# Thickness computation
# ──────────────────────────────────────────────


def compute_thickness(
    seg_path: Path,
    label_names: dict[int, str],
    scan_path: Path | None = None,
) -> ThicknessRow:
    """Compute per-region and whole-cortex mean thickness for one seg NIfTI.

    Algorithm (Das et al. 2009 volumetric approximation):
      - ``d_WM``: distance from every voxel to the nearest WM voxel.
      - ``d_outside``: distance from every voxel to the nearest voxel that is
        neither cortex nor WM (i.e. pial boundary).
      - Per-voxel thickness ≈ ``d_WM + d_outside`` (valid only inside cortex;
        we only ever read these values at cortical voxels).

    Distances are in millimetres — ``sampling`` passed to the EDT is the set of
    column norms of ``affine[:3, :3]``, which equals voxel sizes for any
    orientation (``A = R @ diag(s)`` with orthonormal ``R`` → ``||A[:,j]|| = s[j]``).
    The naive ``np.diag`` would silently under-report on oblique affines.

    Args:
        seg_path: Path to a ``*_synthseg.nii.gz`` produced with ``--parc``.
        label_names: ``{label_int: region_name}`` for labels in CORTEX_LABELS.

    Returns:
        ThicknessRow with per-region means keyed by region name and a
        whole-cortex mean. Absent regions yield NaN.
    """
    img = nib.load(str(seg_path))
    seg = np.asarray(img.dataobj).astype(np.int32, copy=False)
    affine = img.affine
    vox_mm = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))

    mask_wm = np.isin(seg, WM_LABELS)
    mask_cortex = np.isin(seg, CORTEX_LABELS)
    mask_outside = ~(mask_cortex | mask_wm)

    # MONAI's distance_transform_edt requires channel-first input. Under the
    # hood it calls scipy (or cuCIM on GPU); we route through MONAI to keep
    # the import inside the approved library hierarchy (CLAUDE.md). The
    # `~mask` is because EDT returns distance to the nearest False voxel
    # within the True region — so to measure "distance to nearest WM voxel"
    # we pass the complement and read the result at cortical voxels.
    sampling = tuple(float(v) for v in vox_mm)
    d_wm = distance_transform_edt((~mask_wm)[None, ...], sampling=sampling)[0]
    d_outside = distance_transform_edt((~mask_outside)[None, ...], sampling=sampling)[0]
    thickness = np.asarray(d_wm) + np.asarray(d_outside)

    per_region: dict[str, float] = {}
    for label_int, name in label_names.items():
        region_mask = seg == label_int
        per_region[name] = float(thickness[region_mask].mean()) if region_mask.any() else math.nan

    mean_thickness = float(thickness[mask_cortex].mean()) if mask_cortex.any() else math.nan
    return ThicknessRow(
        scan_path=scan_path if scan_path is not None else seg_path,
        seg_path=seg_path,
        mean_thickness=mean_thickness,
        per_region=per_region,
    )


# ──────────────────────────────────────────────
# CSV I/O + resume
# ──────────────────────────────────────────────


def build_header(label_names: dict[int, str]) -> list[str]:
    """Return the canonical CSV header.

    Region columns are sorted by region name so the schema is stable across
    runs and deterministic under ``DictWriter``. The ``seg_path`` column
    (added 2026-04-25) sits between ``scan_path`` and ``mean_thickness`` so
    legacy CSVs (without ``seg_path``) can be detected by checking
    ``"seg_path" in fieldnames``.
    """
    region_cols = [f"{name}_thickness" for name in sorted(label_names.values())]
    return [SCAN_COLUMN, SEG_PATH_COLUMN, MEAN_COLUMN, *region_cols]


def load_existing(output_file: Path) -> set[str]:
    """Return the set of ``seg_path`` values already recorded in ``output_file``.

    The resume check downstream compares ``str(seg_file)`` (a ``*_synthseg.nii.gz``
    path) against this set, so the comparand has to be the seg path — using
    ``scan_path`` here would never match and would silently re-process everything.

    Empty set if the file does not yet exist or is empty. Falls back to
    ``scan_path`` only on legacy CSVs (no ``seg_path`` column, pre-2026-04-25).
    """
    if not output_file.exists() or output_file.stat().st_size == 0:
        return set()
    with output_file.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return set()
        if SEG_PATH_COLUMN in reader.fieldnames:
            return {row[SEG_PATH_COLUMN] for row in reader if row.get(SEG_PATH_COLUMN)}
        if SCAN_COLUMN in reader.fieldnames:
            return {row[SCAN_COLUMN] for row in reader if row.get(SCAN_COLUMN)}
        return set()


def append_row(output_file: Path, row: ThicknessRow, header: list[str]) -> None:
    """Append a single row to ``output_file``, writing the header if new.

    Flushes after every write so a crash mid-batch does not lose the rows
    already computed — the next invocation's :func:`load_existing` will see
    them and skip.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    is_new = not output_file.exists() or output_file.stat().st_size == 0
    record: dict[str, object] = {
        SCAN_COLUMN: str(row.scan_path),
        SEG_PATH_COLUMN: str(row.seg_path),
        MEAN_COLUMN: row.mean_thickness,
    }
    for name, value in row.per_region.items():
        record[f"{name}_thickness"] = value
    with output_file.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if is_new:
            writer.writeheader()
        writer.writerow(record)
        handle.flush()


# ──────────────────────────────────────────────
# Multiprocessing worker (top-level for picklability)
# ──────────────────────────────────────────────


def _compute_thickness_worker(
    args: tuple[Path, Path | None, dict[int, str]],
) -> tuple[str, Path, ThicknessRow | str]:
    """Pool worker: run ``compute_thickness`` and tag the result.

    Returns ``("ok", seg_path, ThicknessRow)`` on success or
    ``("err", seg_path, repr(exc))`` on failure. Errors are reported back to
    the parent rather than swallowed so the parent can keep a single log.
    """
    seg_path, scan_path, label_names = args
    try:
        return ("ok", seg_path, compute_thickness(seg_path, label_names, scan_path=scan_path))
    except Exception as exc:
        return ("err", seg_path, repr(exc))


# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────


def _print_summary(
    total: int,
    already_done: int,
    processed: int,
    output_file: Path,
    console: Console,
) -> None:
    table = Table(title="Phase 3b cortical thickness")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("total seg files", str(total))
    table.add_row("already in CSV (skipped)", str(already_done))
    table.add_row("processed this run", str(processed))
    table.add_row("output file", str(output_file))
    console.print(table)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def _resolve_lut_path(freesurfer_home: Path | None) -> Path:
    """Return the path to FreeSurferColorLUT.txt or raise with a useful hint."""
    if freesurfer_home is None:
        env_home = os.environ.get("FREESURFER_HOME")
        if env_home:
            freesurfer_home = Path(env_home)
    if freesurfer_home is None:
        raise typer.BadParameter(
            "Pass --freesurfer-home or export $FREESURFER_HOME. "
            "Example: --freesurfer-home /Applications/freesurfer/8.1.0"
        )
    return freesurfer_home / "FreeSurferColorLUT.txt"


def load_seg_to_scan_map(manifests: list[Path]) -> dict[str, Path]:
    """Build ``{resolved_seg_path: input_scan_path}`` from synthseg manifests.

    Each manifest's row is ``input_path, seg_path, qc_path, vol_path, mode, status``;
    we keep ``ok`` and ``skipped`` rows (both have valid seg files on disk).
    Missing manifest paths are skipped with a warning. Conflicting seg → input
    mappings across multiple manifests raise ``ValueError`` (signals double
    segmentation that would make downstream lookups non-deterministic).
    """
    out: dict[str, Path] = {}
    for path in manifests:
        if not path.is_file():
            logger.warning("synthseg manifest not found: %s; skipping", path)
            continue
        with path.open() as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            required = {_SYNTHSEG_INPUT_COLUMN, _SYNTHSEG_SEG_COLUMN, _SYNTHSEG_STATUS_COLUMN}
            missing = required - fieldnames
            if missing:
                logger.warning(
                    "%s missing synthseg columns %s; skipping",
                    path, sorted(missing),
                )
                continue
            for row in reader:
                if row.get(_SYNTHSEG_STATUS_COLUMN) not in _SYNTHSEG_VALID_STATUSES:
                    continue
                seg_key = str(Path(row[_SYNTHSEG_SEG_COLUMN]).resolve())
                input_path = Path(row[_SYNTHSEG_INPUT_COLUMN]).resolve()
                existing = out.get(seg_key)
                if existing is not None and existing != input_path:
                    raise ValueError(
                        f"Conflicting input paths for {seg_key}: "
                        f"{existing} vs {input_path}"
                    )
                out[seg_key] = input_path
    return out


@app.command()
def main(
    synthseg_dir: Path = typer.Option(
        ...,
        "--synthseg-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Root of *_synthseg.nii.gz files (Phase 3 --output-dir).",
    ),
    output_file: Path = typer.Option(
        ...,
        "--output-file",
        resolve_path=True,
        help="Aggregate CSV path (one row per scan).",
    ),
    synthseg_manifest: list[Path] = typer.Option(
        [],
        "--synthseg-manifest",
        resolve_path=True,
        help=(
            "Synthseg manifest CSV (repeatable). When provided, the output's "
            "``scan_path`` column is set to the input scan path (the NIfTI "
            "Phase 03 segmented FROM). Without a manifest, ``scan_path`` "
            "falls back to the seg path with a deprecation warning — passing "
            "the manifest is the recommended way to keep downstream joins "
            "(04, 09, visualize.py) on the same path semantics as the "
            "corruption manifest."
        ),
    ),
    freesurfer_home: Path | None = typer.Option(
        None,
        "--freesurfer-home",
        help=(
            "FreeSurfer install root; used only to locate "
            "FreeSurferColorLUT.txt. Defaults to $FREESURFER_HOME."
        ),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Log the first 10 pending scans and return."
    ),
    num_workers: int = typer.Option(
        8,
        "--num-workers",
        help=(
            "Process pool size for per-seg thickness compute. Each scan is "
            "independent (load NIfTI + EDT + per-region mean) — embarrassingly "
            "parallel. 1 = serial loop (legacy). Default 8 fits comfortably on "
            "the 128-CPU pod and gives ~8x speedup for N>>workers."
        ),
    ),
) -> None:
    """Aggregate per-region cortical thickness across all SynthSeg outputs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )

    lut_path = _resolve_lut_path(freesurfer_home)
    label_names = load_region_names(lut_path)
    header = build_header(label_names)

    seg_files = discover_seg_files(synthseg_dir)
    if not seg_files:
        logger.warning("No *_synthseg.nii.gz files found under %s", synthseg_dir)
        return

    seg_to_scan: dict[str, Path] = {}
    if synthseg_manifest:
        seg_to_scan = load_seg_to_scan_map(list(synthseg_manifest))
        logger.info(
            "Loaded seg → scan mapping from %d manifest(s): %d entries",
            len(synthseg_manifest), len(seg_to_scan),
        )
    else:
        logger.warning(
            "No --synthseg-manifest provided; the 'scan_path' column will "
            "store the SEG path (legacy behaviour). Pass synthseg manifests "
            "to populate scan_path with the actual input scan path so "
            "downstream phases (04/09/visualize.py) can join cleanly."
        )

    done = load_existing(output_file)
    pending = [p for p in seg_files if str(p) not in done]

    logger.info(
        "Plan: %d total, %d already done, %d pending",
        len(seg_files),
        len(done),
        len(pending),
    )

    if dry_run:
        for seg_path in pending[:10]:
            logger.info("  would compute: %s", seg_path)
        if len(pending) > 10:
            logger.info("  ... (%d more)", len(pending) - 10)
        return

    processed = 0
    n_unmatched = 0
    work_items: list[tuple[Path, Path | None, dict[int, str]]] = []
    for seg_path in pending:
        scan_path = seg_to_scan.get(str(seg_path.resolve()))
        if synthseg_manifest and scan_path is None:
            n_unmatched += 1
            logger.warning(
                "Seg %s not in any synthseg manifest; falling back to seg path "
                "as scan_path for this row.", seg_path.name,
            )
        work_items.append((seg_path, scan_path, label_names))

    use_pool = num_workers > 1 and len(work_items) > 1
    if use_pool:
        logger.info("Spawning %d workers for %d pending segs", num_workers, len(work_items))
        with mp.get_context("spawn").Pool(processes=num_workers) as pool:
            iterator = tqdm(
                pool.imap_unordered(_compute_thickness_worker, work_items, chunksize=1),
                total=len(work_items), desc="thickness", unit="scan",
            )
            for status, seg_path, payload in iterator:
                if status == "err":
                    logger.warning("thickness failed for %s: %s", seg_path.name, payload)
                    continue
                append_row(output_file, payload, header)
                processed += 1
    else:
        iterator = tqdm(work_items, desc="thickness", unit="scan") if len(work_items) > 1 else work_items
        for seg_path, scan_path, _ in iterator:
            try:
                row = compute_thickness(seg_path, label_names, scan_path=scan_path)
            except Exception as exc:
                logger.warning("thickness failed for %s: %s", seg_path.name, exc)
                continue
            append_row(output_file, row, header)
            processed += 1
    if n_unmatched:
        logger.warning(
            "%d seg file(s) had no manifest match; their scan_path columns "
            "fall back to seg paths. Update the synthseg manifest list to "
            "fix.", n_unmatched,
        )

    _print_summary(
        total=len(seg_files),
        already_done=len(done),
        processed=processed,
        output_file=output_file,
        console=Console(),
    )


if __name__ == "__main__":
    app()
