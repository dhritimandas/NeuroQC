#!/usr/bin/env python3
"""NeuroQC Phase 1 — Reference Scan Curation.

Inspects every .nii.gz volume under --input-dir, applies a signal-level QC
filter, symlinks passing scans into --output-dir, and writes a manifest CSV
at --manifest-path.

Inputs:
    --input-dir      Directory containing .nii.gz reference scans (flat layout).
    --output-dir     Directory where symlinks to passing scans are created.
    --manifest-path  CSV path for the produced reference manifest.
    --dataset-tag    One of {ixi, oasis, fastmri}. Selects QC thresholds.
    --dry-run        Inspect and report, but do not write symlinks or manifest.
    --force          Overwrite an existing manifest instead of skipping.

Outputs:
    <output-dir>/<scan>.nii.gz  Symlinks (absolute) for passing scans only.
    <manifest-path>             CSV with one row per inspected scan, columns:
        filepath, subject_id, hospital, shape_x, shape_y, shape_z,
        voxel_x, voxel_y, voxel_z, min_intensity, max_intensity, passed_qc

Usage:
    python code/01_curate_references.py \\
        --input-dir data/ixi/raw \\
        --output-dir data/ixi/derivatives/references \\
        --manifest-path results/tables/reference_manifest.csv
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import nibabel as nib
import torch
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from tqdm import tqdm

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

DatasetTag = Literal["ixi", "oasis", "fastmri"]
DATASET_TAGS: tuple[DatasetTag, ...] = ("ixi", "oasis", "fastmri")

MIN_MAX_INTENSITY: float = 100.0


@dataclass(frozen=True)
class QCThresholds:
    """Per-dataset QC thresholds applied by :func:`_qc_pass`.

    Attributes:
        min_x: Minimum X (sagittal) dimension in voxels.
        min_y: Minimum Y (coronal) dimension in voxels.
        min_z: Minimum Z (axial) dimension in voxels.
        min_voxel_mm: Inclusive lower bound on voxel size (mm) along any axis.
        max_voxel_mm: Inclusive upper bound on voxel size (mm) along any axis.
    """

    min_x: int
    min_y: int
    min_z: int
    min_voxel_mm: float
    max_voxel_mm: float


# IXI / OASIS: isotropic 1mm research T1. FastMRI: anisotropic clinical,
# thick slices along Z with high in-plane resolution.
THRESHOLDS_BY_TAG: dict[DatasetTag, QCThresholds] = {
    "ixi": QCThresholds(min_x=150, min_y=150, min_z=150, min_voxel_mm=0.5, max_voxel_mm=2.0),
    "oasis": QCThresholds(min_x=150, min_y=150, min_z=150, min_voxel_mm=0.5, max_voxel_mm=2.0),
    "fastmri": QCThresholds(min_x=1, min_y=1, min_z=15, min_voxel_mm=0.5, max_voxel_mm=6.0),
}

IXI_FILENAME_RE: re.Pattern[str] = re.compile(
    r"^IXI(?P<id>\d+)-(?P<hospital>[A-Za-z]+)-(?P<session>\d+)-T1\.nii\.gz$"
)

MANIFEST_COLUMNS: list[str] = [
    "filepath",
    "subject_id",
    "hospital",
    "shape_x",
    "shape_y",
    "shape_z",
    "voxel_x",
    "voxel_y",
    "voxel_z",
    "min_intensity",
    "max_intensity",
    "passed_qc",
]

logger = logging.getLogger(__name__)
app = typer.Typer(help="NeuroQC Phase 1 — Reference Scan Curation", add_completion=False)


# ──────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────


@dataclass(frozen=True)
class ScanRecord:
    """Per-scan metadata and QC verdict.

    Attributes:
        filepath: Absolute path to the source .nii.gz file.
        subject_id: IXI subject ID (e.g. "002") or None if filename did not match.
        hospital: IXI hospital label (e.g. "Guys") or None if filename did not match.
        shape: First three (spatial) dimensions as ints.
        voxel_sizes: First three voxel sizes in mm as floats.
        min_intensity: Minimum finite intensity; NaN if volume has no finite voxels.
        max_intensity: Maximum finite intensity; NaN if volume has no finite voxels.
        passed_qc: True iff the volume satisfies every QC rule in _qc_pass.
    """

    filepath: Path
    subject_id: str | None
    hospital: str | None
    shape: tuple[int, int, int]
    voxel_sizes: tuple[float, float, float]
    min_intensity: float
    max_intensity: float
    passed_qc: bool

    def to_manifest_row(self) -> dict[str, str]:
        """Return a dict keyed by MANIFEST_COLUMNS for csv.DictWriter."""
        return {
            "filepath": str(self.filepath),
            "subject_id": self.subject_id or "",
            "hospital": self.hospital or "",
            "shape_x": str(self.shape[0]),
            "shape_y": str(self.shape[1]),
            "shape_z": str(self.shape[2]),
            "voxel_x": f"{self.voxel_sizes[0]:.6f}",
            "voxel_y": f"{self.voxel_sizes[1]:.6f}",
            "voxel_z": f"{self.voxel_sizes[2]:.6f}",
            "min_intensity": f"{self.min_intensity:.6f}",
            "max_intensity": f"{self.max_intensity:.6f}",
            "passed_qc": "true" if self.passed_qc else "false",
        }


# ──────────────────────────────────────────────
# Filename parsing
# ──────────────────────────────────────────────


def parse_ixi_filename(filename: str) -> tuple[str | None, str | None]:
    """Parse an IXI-format filename into (subject_id, hospital).

    Expected format: ``IXI{ID}-{Hospital}-{SessionID}-T1.nii.gz``.

    Args:
        filename: Basename of the scan (e.g. ``IXI002-Guys-0828-T1.nii.gz``).

    Returns:
        ``(subject_id, hospital)`` on a match; ``(None, None)`` otherwise.
    """
    match = IXI_FILENAME_RE.match(filename)
    if match is None:
        return None, None
    return match.group("id"), match.group("hospital")


# ──────────────────────────────────────────────
# Per-scan inspection and QC
# ──────────────────────────────────────────────


def _pad_shape_3d(raw_shape: tuple[int, ...]) -> tuple[int, int, int]:
    """Return the first three dims, padding with 1 if the volume has fewer."""
    padded = list(raw_shape[:3]) + [1] * max(0, 3 - len(raw_shape))
    return int(padded[0]), int(padded[1]), int(padded[2])


def _pad_zooms_3d(raw_zooms: tuple[float, ...]) -> tuple[float, float, float]:
    """Return the first three voxel sizes, padding with 1.0 if fewer."""
    padded = list(raw_zooms[:3]) + [1.0] * max(0, 3 - len(raw_zooms))
    return float(padded[0]), float(padded[1]), float(padded[2])


def _qc_pass(
    *,
    shape: tuple[int, int, int],
    voxel_sizes: tuple[float, float, float],
    has_nan_or_inf: bool,
    max_intensity: float,
    all_zeros: bool,
    thresholds: QCThresholds,
) -> bool:
    """Apply the five QC rules. Returns True iff every rule passes.

    Rules (all must hold):
      1. shape_x >= thresholds.min_x, shape_y >= min_y, shape_z >= min_z
      2. Every voxel size in [thresholds.min_voxel_mm, thresholds.max_voxel_mm]
      3. No NaN or Inf in the data
      4. Max intensity > MIN_MAX_INTENSITY
      5. Data is not all zeros
    """
    if has_nan_or_inf:
        return False
    if all_zeros:
        return False
    if shape[0] < thresholds.min_x or shape[1] < thresholds.min_y or shape[2] < thresholds.min_z:
        return False
    if any(
        not (thresholds.min_voxel_mm <= v <= thresholds.max_voxel_mm) for v in voxel_sizes
    ):
        return False
    if not max_intensity > MIN_MAX_INTENSITY:
        return False
    return True


def inspect_scan(path: Path, thresholds: QCThresholds) -> ScanRecord:
    """Load a NIfTI scan and return its metadata + QC verdict.

    Args:
        path: Path to a .nii.gz file.
        thresholds: Dataset-specific QC thresholds.

    Returns:
        A ScanRecord. Raises only if nibabel fails to open the file.
    """
    img = nib.load(str(path))
    # Touch affine and dtype to surface corruption early; values are not
    # serialized to the manifest (schema is fixed by CLAUDE.md).
    _ = img.affine
    _ = img.get_data_dtype()

    shape = _pad_shape_3d(tuple(int(s) for s in img.shape))
    voxel_sizes = _pad_zooms_3d(tuple(float(z) for z in img.header.get_zooms()))

    # nibabel returns a numpy array here; convert to torch immediately and
    # perform all subsequent computation with torch only (CLAUDE.md rule).
    data = torch.from_numpy(img.get_fdata())

    finite_mask = torch.isfinite(data)
    has_nan_or_inf = not bool(finite_mask.all().item())
    all_zeros = bool((data == 0).all().item())

    if finite_mask.any().item():
        finite_values = data[finite_mask]
        min_intensity = float(finite_values.min().item())
        max_intensity = float(finite_values.max().item())
    else:
        min_intensity = float("nan")
        max_intensity = float("nan")

    passed_qc = _qc_pass(
        shape=shape,
        voxel_sizes=voxel_sizes,
        has_nan_or_inf=has_nan_or_inf,
        max_intensity=max_intensity,
        all_zeros=all_zeros,
        thresholds=thresholds,
    )

    subject_id, hospital = parse_ixi_filename(path.name)

    return ScanRecord(
        filepath=path.resolve(),
        subject_id=subject_id,
        hospital=hospital,
        shape=shape,
        voxel_sizes=voxel_sizes,
        min_intensity=min_intensity,
        max_intensity=max_intensity,
        passed_qc=passed_qc,
    )


# ──────────────────────────────────────────────
# Symlinking
# ──────────────────────────────────────────────


def _symlink_if_new(src: Path, dst_dir: Path) -> Path:
    """Create an absolute-path symlink dst_dir/<src.name> -> src.

    Idempotent: if the symlink already points at src, does nothing. If a
    regular file is already at that path, raises rather than overwriting.
    """
    dst = dst_dir / src.name
    src_abs = src.resolve()
    if dst.is_symlink():
        if dst.readlink() == src_abs:
            return dst
        dst.unlink()
    elif dst.exists():
        raise FileExistsError(
            f"{dst} exists and is not a symlink; refusing to overwrite."
        )
    dst.symlink_to(src_abs)
    return dst


# ──────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────


def _write_manifest(records: list[ScanRecord], manifest_path: Path) -> None:
    """Write records to manifest_path as CSV with MANIFEST_COLUMNS header."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_manifest_row())


def curate(
    input_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    dataset_tag: DatasetTag = "ixi",
    dry_run: bool = False,
    force: bool = False,
) -> list[ScanRecord]:
    """Inspect every .nii.gz in input_dir, symlink passes, write manifest.

    Args:
        input_dir: Directory to scan (top-level only).
        output_dir: Directory to receive symlinks of passing scans.
        manifest_path: Output CSV path.
        dataset_tag: Selects QC thresholds from THRESHOLDS_BY_TAG.
        dry_run: If True, no symlinks or manifest are written.
        force: If True, overwrite an existing manifest instead of skipping.

    Returns:
        List of ScanRecord for every inspected file (passing and failing).
        Returns an empty list if manifest exists and force=False.
    """
    if dataset_tag not in THRESHOLDS_BY_TAG:
        raise ValueError(
            f"unknown --dataset-tag {dataset_tag!r}; expected one of {DATASET_TAGS}"
        )
    thresholds = THRESHOLDS_BY_TAG[dataset_tag]

    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input-dir does not exist: {input_dir}")

    if manifest_path.exists() and not force and not dry_run:
        logger.warning(
            "Manifest already exists at %s; pass --force to overwrite. Skipping.",
            manifest_path,
        )
        return []

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.nii.gz"))
    if not input_files:
        logger.warning("No .nii.gz files found in %s", input_dir)
        if not dry_run:
            _write_manifest([], manifest_path)
        return []

    records: list[ScanRecord] = []
    iterator = tqdm(input_files, desc="Curating", unit="scan") if len(input_files) > 10 else input_files
    for path in iterator:
        try:
            record = inspect_scan(path, thresholds)
        except Exception as exc:
            logger.exception("Failed to inspect %s: %s", path, exc)
            continue
        records.append(record)
        if record.passed_qc and not dry_run:
            _symlink_if_new(path, output_dir)

    if not dry_run:
        _write_manifest(records, manifest_path)
    return records


# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────


def _print_summary(records: list[ScanRecord], console: Console) -> None:
    """Render a small rich table summarising totals and per-hospital counts."""
    total = len(records)
    passed = sum(1 for r in records if r.passed_qc)

    table = Table(title="Reference curation summary")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("total inspected", str(total))
    table.add_row("passed", str(passed))
    table.add_row("failed", str(total - passed))
    console.print(table)

    by_hospital: dict[str, tuple[int, int]] = {}
    for record in records:
        key = record.hospital or "<unmatched>"
        total_h, passed_h = by_hospital.get(key, (0, 0))
        by_hospital[key] = (total_h + 1, passed_h + (1 if record.passed_qc else 0))

    if by_hospital:
        by_hospital_table = Table(title="By hospital")
        by_hospital_table.add_column("hospital", style="bold")
        by_hospital_table.add_column("total", justify="right")
        by_hospital_table.add_column("passed", justify="right")
        for hospital, (total_h, passed_h) in sorted(by_hospital.items()):
            by_hospital_table.add_row(hospital, str(total_h), str(passed_h))
        console.print(by_hospital_table)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


@app.command()
def main(
    input_dir: Path = typer.Option(
        ...,
        "--input-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Directory containing .nii.gz reference scans.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Directory to receive symlinks of passing scans.",
    ),
    manifest_path: Path = typer.Option(
        Path("results/tables/reference_manifest.csv"),
        "--manifest-path",
        help="Output CSV path for the reference manifest.",
    ),
    dataset_tag: str = typer.Option(
        "ixi",
        "--dataset-tag",
        case_sensitive=False,
        help="One of {ixi, oasis, fastmri}. Selects dataset-specific QC thresholds.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Inspect and report without writing anything."
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite an existing manifest."
    ),
) -> None:
    """Curate NIfTI reference scans and write the phase-1 manifest."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )
    tag = dataset_tag.lower()
    if tag not in THRESHOLDS_BY_TAG:
        raise typer.BadParameter(
            f"--dataset-tag must be one of {DATASET_TAGS}; got {dataset_tag!r}"
        )
    records = curate(
        input_dir=input_dir,
        output_dir=output_dir,
        manifest_path=manifest_path,
        dataset_tag=tag,  # type: ignore[arg-type]
        dry_run=dry_run,
        force=force,
    )
    _print_summary(records, Console())


if __name__ == "__main__":
    app()
