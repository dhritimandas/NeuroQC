#!/usr/bin/env python3
"""NeuroQC Phase 0 — FastMRI T1 NIfTI Extraction.

Extracts T1-weighted magnitude NIfTI volumes from FastMRI HDF5 files.
Uses the already-reconstructed root-sum-of-squares (RSS) magnitude image
stored under the ``reconstruction_rss`` key of each ``.h5`` file, not the
raw multi-coil k-space.

Pipeline per file:
    1. Open HDF5, read ``attrs["acquisition"]``. Keep only exact matches
       against ``--acquisitions`` (default AXT1, AXT1PRE, AXT1POST).
    2. Read ``reconstruction_rss`` of shape ``(n_slices, H, W)``.
    3. Permute to NIfTI convention ``(H, W, n_slices)``.
    4. Parse ``ismrmrd_header`` XML embedded in the .h5 to recover the
       anisotropic voxel sizes (reconSpace FOV / matrix). FastMRI brain
       AXT1 is a 2D multi-slice clinical acquisition: typical values are
       0.6875 × 0.6875 mm in-plane × 5 mm slice. Writing the NIfTI with
       this anisotropic affine is what lets SynthSeg's internal
       resampler find the real 8 cm Z extent instead of a 16 mm slab.
    5. Save as ``<output-dir>/<file_id>.nii.gz`` with a diagonal affine
       built from the parsed voxel sizes. No external resampling — the
       downstream SynthSeg preprocessor resamples to its target_res as
       needed, per Billot et al. SynthSeg 2.0.

Resume semantics:
    * If the output NIfTI already exists and ``--force`` is not passed,
      the file is still added to the manifest (via a metadata peek), but
      its volume is not re-read, re-resampled, or re-saved.
    * Files whose acquisition attribute is not in the allow-list are
      skipped entirely (not present in the manifest).

Inputs:
    --input-dir      Directory containing ``.h5`` files (searched recursively).
    --output-dir     Directory for extracted ``.nii.gz`` volumes.
    --manifest-csv   CSV listing every kept volume.
    --acquisitions   Comma-separated acquisition allow-list.
    --dry-run        Scan + report only; no NIfTI or manifest written.
    --force          Re-extract volumes even when the output NIfTI exists.

Outputs:
    <output-dir>/<file_id>.nii.gz  Per-volume NIfTI (one per passing file).
    <manifest-csv>                 Columns: file_id, acquisition_type,
                                   n_slices, H, W, field_strength.

Usage:
    python code/00_extract_fastmri_t1.py \\
        --input-dir data/fastmri/raw \\
        --output-dir data/fastmri/nifti \\
        --manifest-csv results/tables/fastmri_extraction_manifest.csv
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from tqdm import tqdm

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

RSS_KEY: str = "reconstruction_rss"
ISMRMRD_HEADER_KEY: str = "ismrmrd_header"
ACQUISITION_ATTR: str = "acquisition"
FIELD_STRENGTH_ATTR: str = "systemFieldStrength_T"

DEFAULT_ACQUISITIONS: tuple[str, ...] = ("AXT1", "AXT1PRE", "AXT1POST")

# Fallback voxel sizes when ismrmrd_header parse fails. These are the
# nominal AXT1 brain values published in Knoll et al. (2020), "fastMRI:
# A publicly available raw k-space and DICOM dataset of knee and brain
# MRI": 22 cm FOV on 320 matrix → 0.6875 mm in-plane; 5 mm slice thickness.
# Ordered (vy, vx, vz) to match the post-permute (H, W, D) NIfTI layout.
DEFAULT_VOXEL_MM: tuple[float, float, float] = (0.6875, 0.6875, 5.0)

MANIFEST_COLUMNS: list[str] = [
    "file_id",
    "acquisition_type",
    "n_slices",
    "H",
    "W",
    "field_strength",
    "voxel_y",
    "voxel_x",
    "voxel_z",
]

logger = logging.getLogger(__name__)
app = typer.Typer(
    help="NeuroQC Phase 0 — Extract T1 NIfTI volumes from FastMRI HDF5",
    add_completion=False,
)


# ──────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────


@dataclass(frozen=True)
class ExtractionRecord:
    """Per-volume manifest row.

    Attributes:
        file_id: Stem of the source ``.h5`` filename.
        acquisition_type: Raw ``attrs["acquisition"]`` value (e.g. ``AXT1POST``).
        n_slices: Slice count along the slice-select axis of ``reconstruction_rss``.
        H: In-plane height (second axis of ``reconstruction_rss``).
        W: In-plane width (third axis of ``reconstruction_rss``).
        field_strength: Scanner field strength in Tesla, or None if missing.
        voxel_mm: ``(vy, vx, vz)`` in mm, parsed from ``ismrmrd_header`` or
            the Knoll-2020 fallback. Matches the (H, W, D) NIfTI layout.
    """

    file_id: str
    acquisition_type: str
    n_slices: int
    H: int
    W: int
    field_strength: float | None
    voxel_mm: tuple[float, float, float]

    def to_manifest_row(self) -> dict[str, object]:
        """Return a dict row suitable for pandas.DataFrame construction."""
        vy, vx, vz = self.voxel_mm
        return {
            "file_id": self.file_id,
            "acquisition_type": self.acquisition_type,
            "n_slices": self.n_slices,
            "H": self.H,
            "W": self.W,
            "field_strength": (
                self.field_strength if self.field_strength is not None else ""
            ),
            "voxel_y": vy,
            "voxel_x": vx,
            "voxel_z": vz,
        }


# ──────────────────────────────────────────────
# HDF5 reading
# ──────────────────────────────────────────────


def _decode_attr(value: object) -> str:
    """Return the string form of an HDF5 attribute (bytes are utf-8 decoded)."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _coerce_field_strength(value: object) -> float | None:
    """Return a float field strength, or None if missing/unparseable."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def peek_metadata(h5_path: Path) -> tuple[str, tuple[int, int, int], float | None]:
    """Read acquisition, RSS shape, and field strength without loading RSS.

    Args:
        h5_path: Path to a FastMRI ``.h5`` file.

    Returns:
        ``(acquisition, (n_slices, H, W), field_strength)``. Acquisition is
        the empty string when the attribute is absent. Field strength is
        None when the attribute is absent or unparseable.

    Raises:
        KeyError: When the file lacks ``reconstruction_rss``.
        ValueError: When ``reconstruction_rss`` is not 3D.
    """
    with h5py.File(h5_path, "r") as handle:
        acq_raw = handle.attrs.get(ACQUISITION_ATTR)
        acquisition = _decode_attr(acq_raw) if acq_raw is not None else ""
        field_strength = _coerce_field_strength(handle.attrs.get(FIELD_STRENGTH_ATTR))
        if RSS_KEY not in handle:
            raise KeyError(
                f"{h5_path.name}: missing dataset {RSS_KEY!r}; "
                "not a reconstructed FastMRI file."
            )
        shape = tuple(int(s) for s in handle[RSS_KEY].shape)
    if len(shape) != 3:
        raise ValueError(
            f"{h5_path.name}: {RSS_KEY} has shape {shape}; expected 3D "
            "(n_slices, H, W)."
        )
    return acquisition, shape, field_strength  # type: ignore[return-value]


def load_rss_volume(h5_path: Path) -> torch.Tensor:
    """Load the RSS magnitude volume and return ``(H, W, n_slices)`` torch tensor.

    The source layout is ``(n_slices, H, W)``; we permute to the standard
    NIfTI ordering ``(H, W, D)`` and convert to float32 for torchio.
    """
    with h5py.File(h5_path, "r") as handle:
        if RSS_KEY not in handle:
            raise KeyError(
                f"{h5_path.name}: missing dataset {RSS_KEY!r}; "
                "not a reconstructed FastMRI file."
            )
        rss = handle[RSS_KEY][:]  # (n_slices, H, W)
    # nibabel/h5py give numpy; convert to torch immediately per project rules.
    tensor = torch.from_numpy(np.asarray(rss)).float()
    if tensor.ndim != 3:
        raise ValueError(
            f"{h5_path.name}: {RSS_KEY} has shape {tuple(tensor.shape)}; "
            "expected 3D (n_slices, H, W)."
        )
    return tensor.permute(1, 2, 0).contiguous()


# ──────────────────────────────────────────────
# Affine from ismrmrd_header
# ──────────────────────────────────────────────


def _strip_xml_default_namespace(xml: str) -> str:
    """Remove the default xmlns attribute so ElementTree queries don't need it.

    FastMRI ismrmrd_header starts with
    ``<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">`` — without the
    strip, ``find("encoding/reconSpace")`` returns None because every
    element's tag carries the ``{http://...}`` prefix.
    """
    import re

    return re.sub(r'\sxmlns="[^"]*"', "", xml, count=1)


def parse_voxel_sizes(h5_path: Path) -> tuple[float, float, float]:
    """Return ``(vy, vx, vz)`` voxel sizes in mm from the FastMRI ismrmrd_header.

    Reads ``<encoding><reconSpace><fieldOfView_mm/matrixSize>`` for the
    in-plane voxel sizes (FOV / matrix), and takes the z FOV directly as
    the slice thickness (FastMRI brain is 2D multi-slice, so reconSpace
    matrixSize.z is always 1 and reconSpace fieldOfView_mm.z is the slice
    thickness).

    Returned ordering matches the post-permute NIfTI layout ``(H, W, D)``:
    rss is stored ``(n_slices, H, W)``; axis 0 of the saved NIfTI is H
    (ismrmrd y), axis 1 is W (ismrmrd x), axis 2 is the slice axis.

    On any parse failure, returns :data:`DEFAULT_VOXEL_MM` and logs a
    warning — the fallback is the Knoll 2020 nominal brain AXT1 geometry.
    """
    try:
        with h5py.File(h5_path, "r") as handle:
            if ISMRMRD_HEADER_KEY not in handle:
                raise KeyError(f"missing {ISMRMRD_HEADER_KEY!r}")
            raw = handle[ISMRMRD_HEADER_KEY][()]
        xml = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        root = ET.fromstring(_strip_xml_default_namespace(xml))
        recon = root.find("encoding/reconSpace")
        if recon is None:
            raise ValueError("no encoding/reconSpace element")
        fov = recon.find("fieldOfView_mm")
        mat = recon.find("matrixSize")
        if fov is None or mat is None:
            raise ValueError("reconSpace missing fieldOfView_mm or matrixSize")
        fov_x = float(fov.findtext("x", "0"))
        fov_y = float(fov.findtext("y", "0"))
        fov_z = float(fov.findtext("z", "0"))
        mat_x = int(mat.findtext("x", "0"))
        mat_y = int(mat.findtext("y", "0"))
        if mat_x <= 0 or mat_y <= 0 or fov_x <= 0 or fov_y <= 0 or fov_z <= 0:
            raise ValueError(
                f"invalid reconSpace values (fov={fov_x,fov_y,fov_z}, mat={mat_x,mat_y})"
            )
        # Post-permute NIfTI axes ← rss (n_slices, H, W). H maps to ismrmrd y,
        # W to ismrmrd x; both are equal for square T1 matrices anyway.
        return (fov_y / mat_y, fov_x / mat_x, fov_z)
    except Exception as exc:
        logger.warning(
            "ismrmrd_header parse failed for %s (%s); using Knoll-2020 defaults %s",
            h5_path.name,
            exc,
            DEFAULT_VOXEL_MM,
        )
        return DEFAULT_VOXEL_MM


def build_affine(voxel_mm: tuple[float, float, float]) -> np.ndarray:
    """Return a diagonal 4×4 NIfTI affine from per-axis voxel sizes in mm.

    No rotation or translation — just ``diag(vy, vx, vz, 1)``. Scanner-
    accurate L/R/A/P orientation is not recoverable from FastMRI's
    header; SynthSeg's downstream pipeline works on voxel sizes
    regardless of orientation sign, and the diagnostic script's
    asymmetry metric is invariant to L↔R flip.
    """
    vy, vx, vz = voxel_mm
    return np.diag([vy, vx, vz, 1.0]).astype(np.float64)


# ──────────────────────────────────────────────
# Per-file extraction
# ──────────────────────────────────────────────


def extract_one(
    h5_path: Path,
    output_path: Path,
    allowed_acquisitions: frozenset[str],
    *,
    force: bool,
    dry_run: bool,
) -> ExtractionRecord | None:
    """Extract one FastMRI file to NIfTI, or skip it.

    Returns None when the file's acquisition attribute is not in the
    allow-list. Returns an ExtractionRecord when the file is kept — even
    if the volume write was skipped due to an existing output.

    NIfTI is saved with a diagonal affine built from voxel sizes parsed
    from ``ismrmrd_header``. No external resampling is performed;
    SynthSeg's inference pipeline resamples to its target_res internally
    when the input lies outside ``[target - 0.05, target + 0.05]``, which
    for FastMRI AXT1 ``(0.6875, 0.6875, 5.0)`` is true for every axis.
    """
    acquisition, (n_slices, height, width), field_strength = peek_metadata(h5_path)

    if acquisition not in allowed_acquisitions:
        logger.debug(
            "Skipping %s: acquisition %r not in allow-list",
            h5_path.name,
            acquisition,
        )
        return None

    voxel_mm = parse_voxel_sizes(h5_path)

    record = ExtractionRecord(
        file_id=h5_path.stem,
        acquisition_type=acquisition,
        n_slices=n_slices,
        H=height,
        W=width,
        field_strength=field_strength,
        voxel_mm=voxel_mm,
    )

    if output_path.exists() and not force:
        logger.info("Resume: %s already exists; volume not re-written", output_path.name)
        return record

    if dry_run:
        logger.info("dry-run: would extract %s -> %s", h5_path.name, output_path.name)
        return record

    volume = load_rss_volume(h5_path)  # (H, W, D) torch float32
    affine = build_affine(voxel_mm)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(volume.numpy(), affine=affine)
    nib.save(img, str(output_path))

    return record


# ──────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────


def _write_manifest(records: list[ExtractionRecord], manifest_csv: Path) -> None:
    """Write manifest CSV with the project's fixed column order."""
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [r.to_manifest_row() for r in records],
        columns=MANIFEST_COLUMNS,
    )
    df.to_csv(manifest_csv, index=False)


def extract_all(
    input_dir: Path,
    output_dir: Path,
    manifest_csv: Path,
    allowed_acquisitions: frozenset[str] = frozenset(DEFAULT_ACQUISITIONS),
    dry_run: bool = False,
    force: bool = False,
) -> list[ExtractionRecord]:
    """Extract every matching FastMRI ``.h5`` under input_dir to NIfTI.

    Args:
        input_dir: Directory containing ``.h5`` files (recursed).
        output_dir: Destination for ``.nii.gz`` volumes.
        manifest_csv: CSV path written with one row per kept volume.
        allowed_acquisitions: Exact-match set against ``attrs["acquisition"]``.
        dry_run: If True, no NIfTI or manifest is written.
        force: If True, re-extract even when the output exists.

    Returns:
        List of records for every file that passed the acquisition filter.
    """
    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input-dir does not exist: {input_dir}")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.rglob("*.h5"))
    if not input_files:
        logger.warning("No .h5 files found under %s", input_dir)
        if not dry_run:
            _write_manifest([], manifest_csv)
        return []

    records: list[ExtractionRecord] = []
    iterator: object = (
        tqdm(input_files, desc="Extracting", unit="file")
        if len(input_files) > 10
        else input_files
    )
    for h5_path in iterator:  # type: ignore[assignment]
        output_path = output_dir / f"{h5_path.stem}.nii.gz"
        try:
            record = extract_one(
                h5_path,
                output_path,
                allowed_acquisitions,
                force=force,
                dry_run=dry_run,
            )
        except Exception as exc:
            logger.exception("Failed to extract %s: %s", h5_path, exc)
            continue
        if record is not None:
            records.append(record)

    if not dry_run:
        _write_manifest(records, manifest_csv)
    return records


# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────


def _print_summary(
    records: list[ExtractionRecord],
    total_scanned: int,
    console: Console,
) -> None:
    """Render a rich summary: totals and per-acquisition counts."""
    table = Table(title="FastMRI extraction summary")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("files scanned", str(total_scanned))
    table.add_row("kept (acquisition match)", str(len(records)))
    table.add_row("skipped (other acquisition)", str(total_scanned - len(records)))
    console.print(table)

    by_acq: dict[str, int] = {}
    for record in records:
        by_acq[record.acquisition_type] = by_acq.get(record.acquisition_type, 0) + 1
    if by_acq:
        acq_table = Table(title="By acquisition")
        acq_table.add_column("acquisition", style="bold")
        acq_table.add_column("count", justify="right")
        for acquisition, count in sorted(by_acq.items()):
            acq_table.add_row(acquisition, str(count))
        console.print(acq_table)


def _parse_acquisitions(raw: str) -> frozenset[str]:
    """Parse a comma-separated acquisition list into a frozenset."""
    items = [part.strip() for part in raw.split(",") if part.strip()]
    if not items:
        raise typer.BadParameter(
            "--acquisitions must contain at least one non-empty entry"
        )
    return frozenset(items)


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
        help="Directory containing FastMRI .h5 files (searched recursively).",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Directory to receive extracted .nii.gz volumes.",
    ),
    manifest_csv: Path = typer.Option(
        Path("results/tables/fastmri_extraction_manifest.csv"),
        "--manifest-csv",
        help="Output CSV path for the extraction manifest.",
    ),
    acquisitions: str = typer.Option(
        ",".join(DEFAULT_ACQUISITIONS),
        "--acquisitions",
        help="Comma-separated allow-list for attrs['acquisition'].",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Scan and report without writing outputs."
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-extract volumes even when outputs exist."
    ),
) -> None:
    """Extract T1-weighted magnitude NIfTI volumes from FastMRI HDF5 files."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )
    allow_list = _parse_acquisitions(acquisitions)
    total_scanned = sum(1 for _ in input_dir.rglob("*.h5"))
    records = extract_all(
        input_dir=input_dir,
        output_dir=output_dir,
        manifest_csv=manifest_csv,
        allowed_acquisitions=allow_list,
        dry_run=dry_run,
        force=force,
    )
    _print_summary(records, total_scanned, Console())


if __name__ == "__main__":
    app()
