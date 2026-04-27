"""Unit tests for code/00_extract_fastmri_t1.py.

Synthetic HDF5 files are constructed in-memory with h5py to mimic the
FastMRI layout (``reconstruction_rss`` dataset + ``acquisition`` and
``systemFieldStrength_T`` attributes). No external data is downloaded.

The module under test has a digit-prefixed filename, so it is loaded
via importlib rather than a normal import statement.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import h5py
import nibabel as nib
import pandas as pd
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE_PATH = _REPO_ROOT / "code" / "00_extract_fastmri_t1.py"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("extract_fastmri_t1", _MODULE_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {_MODULE_PATH}"
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass can resolve string annotations.
    sys.modules["extract_fastmri_t1"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def extract_mod() -> types.ModuleType:
    return _load_module()


# ──────────────────────────────────────────────
# Synthesis helpers
# ──────────────────────────────────────────────


_ISMRMRD_HEADER_TEMPLATE = """<?xml version="1.0"?>
<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">
  <encoding>
    <encodedSpace>
      <matrixSize><x>{mat_x}</x><y>{mat_y}</y><z>1</z></matrixSize>
      <fieldOfView_mm><x>{fov_x}</x><y>{fov_y}</y><z>{fov_z}</z></fieldOfView_mm>
    </encodedSpace>
    <reconSpace>
      <matrixSize><x>{mat_x}</x><y>{mat_y}</y><z>1</z></matrixSize>
      <fieldOfView_mm><x>{fov_x}</x><y>{fov_y}</y><z>{fov_z}</z></fieldOfView_mm>
    </reconSpace>
    <trajectory>cartesian</trajectory>
  </encoding>
</ismrmrdHeader>"""


def _write_fastmri_h5(
    path: Path,
    *,
    acquisition: str,
    shape: tuple[int, int, int],
    field_strength: float | None = 3.0,
    fov_mm: tuple[float, float, float] = (220.0, 220.0, 5.0),
    include_ismrmrd_header: bool = True,
    rss_scale: float = 1.0,
) -> Path:
    """Write a minimal FastMRI-like .h5 file.

    Args:
        path: Destination .h5 path.
        acquisition: Value for attrs["acquisition"].
        shape: (n_slices, H, W) for the reconstruction_rss dataset.
        field_strength: Tesla value; omitted from attrs when None.
        fov_mm: (fov_x, fov_y, fov_z) mm used to build the
            reconSpace/fieldOfView_mm entry. Default matches the FastMRI
            brain AXT1 geometry: 220×220 mm FOV at 320 matrix → 0.6875 mm
            in-plane; 5 mm slice thickness.
        include_ismrmrd_header: If True, writes an ``ismrmrd_header``
            dataset with the reconSpace fields populated. Set False to
            test the fallback-voxel-size code path.
        rss_scale: Multiplier applied to the synthetic RSS magnitudes
            (default 1.0 → max ≈ 600). Use a small value (e.g. 1e-6) to
            simulate the real FastMRI normalised-magnitude regime that
            triggers the ``--rescale-intensity`` code path.

    Returns:
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rss = ((torch.rand(*shape) * 500.0 + 100.0) * rss_scale).float().numpy()
    _, height, width = shape
    fov_x, fov_y, fov_z = fov_mm
    with h5py.File(path, "w") as handle:
        handle.create_dataset("reconstruction_rss", data=rss)
        handle.attrs["acquisition"] = acquisition
        if field_strength is not None:
            handle.attrs["systemFieldStrength_T"] = float(field_strength)
        if include_ismrmrd_header:
            xml = _ISMRMRD_HEADER_TEMPLATE.format(
                mat_x=width, mat_y=height, fov_x=fov_x, fov_y=fov_y, fov_z=fov_z
            )
            handle.create_dataset("ismrmrd_header", data=xml)
    return path


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────


def test_extract_one_writes_nifti_and_permutes_axes(
    extract_mod: types.ModuleType, tmp_path: Path
) -> None:
    """An AXT1 file is extracted, saved as NIfTI, and shape is (H, W, n_slices)."""
    n_slices, height, width = 4, 12, 10
    h5_path = _write_fastmri_h5(
        tmp_path / "raw" / "file_brain_AXT1_999_0000001.h5",
        acquisition="AXT1",
        shape=(n_slices, height, width),
        field_strength=3.0,
    )
    output_path = tmp_path / "nifti" / f"{h5_path.stem}.nii.gz"

    record = extract_mod.extract_one(
        h5_path,
        output_path,
        frozenset({"AXT1", "AXT1PRE", "AXT1POST"}),
        force=False,
        dry_run=False,
    )

    assert record is not None
    assert record.file_id == h5_path.stem
    assert record.acquisition_type == "AXT1"
    assert record.n_slices == n_slices
    assert record.H == height
    assert record.W == width
    assert record.field_strength == pytest.approx(3.0)

    assert output_path.exists()
    img = nib.load(str(output_path))
    assert tuple(int(s) for s in img.shape) == (height, width, n_slices)
    # Anisotropic affine from the fixture's ismrmrd_header: FOV=220 mm at
    # matrix=(height, width) in-plane, slice thickness 5 mm.
    zooms = img.header.get_zooms()
    assert zooms[0] == pytest.approx(220.0 / height, abs=1e-6)
    assert zooms[1] == pytest.approx(220.0 / width, abs=1e-6)
    assert zooms[2] == pytest.approx(5.0, abs=1e-6)
    # Record carries the same values for the manifest CSV.
    assert record.voxel_mm == pytest.approx((220.0 / height, 220.0 / width, 5.0))


def test_parsed_voxel_sizes_drive_affine(
    extract_mod: types.ModuleType, tmp_path: Path
) -> None:
    """A realistic FastMRI brain AXT1 header (220×220×5 mm FOV on 320×320×16)
    must produce a NIfTI with (0.6875, 0.6875, 5.0) mm voxels — the single
    change that unlocks SynthSeg inference for this scan class.
    """
    n_slices, height, width = 16, 320, 320
    h5_path = _write_fastmri_h5(
        tmp_path / "brain.h5",
        acquisition="AXT1",
        shape=(n_slices, height, width),
        fov_mm=(220.0, 220.0, 5.0),
    )
    output_path = tmp_path / f"{h5_path.stem}.nii.gz"

    record = extract_mod.extract_one(
        h5_path,
        output_path,
        frozenset({"AXT1"}),
        force=False,
        dry_run=False,
    )
    assert record is not None
    img = nib.load(str(output_path))
    zooms = img.header.get_zooms()
    assert zooms == pytest.approx((0.6875, 0.6875, 5.0), abs=1e-6)
    # Check affine diagonal matches voxel sizes (no rotation/translation).
    affine = img.affine
    assert affine[0, 0] == pytest.approx(0.6875, abs=1e-6)
    assert affine[1, 1] == pytest.approx(0.6875, abs=1e-6)
    assert affine[2, 2] == pytest.approx(5.0, abs=1e-6)
    assert (affine[:3, 3] == 0).all()


def test_parse_voxel_sizes_falls_back_when_header_missing(
    extract_mod: types.ModuleType, tmp_path: Path
) -> None:
    """No ismrmrd_header in the .h5 must not crash; fall back to defaults."""
    h5_path = _write_fastmri_h5(
        tmp_path / "no_header.h5",
        acquisition="AXT1",
        shape=(4, 8, 8),
        include_ismrmrd_header=False,
    )
    vy, vx, vz = extract_mod.parse_voxel_sizes(h5_path)
    assert (vy, vx, vz) == extract_mod.DEFAULT_VOXEL_MM


def test_rescale_intensity_lifts_low_magnitudes(
    extract_mod: types.ModuleType, tmp_path: Path
) -> None:
    """FastMRI RSS magnitudes (~10^-3) must be rescaled to T1-typical (>100)
    so Phase 1 MIN_MAX_INTENSITY=100 does not reject the volume.
    """
    h5_path = _write_fastmri_h5(
        tmp_path / "low_mag.h5",
        acquisition="AXT1",
        shape=(4, 8, 8),
        rss_scale=1.0e-6,  # max ≈ 6e-4, mimics real FastMRI RSS regime
    )
    output_path = tmp_path / "rescaled.nii.gz"

    record = extract_mod.extract_one(
        h5_path,
        output_path,
        frozenset({"AXT1"}),
        force=False,
        dry_run=False,
        rescale_intensity=True,
    )

    assert record is not None
    img = nib.load(str(output_path))
    arr = img.get_fdata()
    assert float(arr.max()) > 100.0, (
        f"Rescale failed: max={arr.max():.3e}, expected > 100"
    )
    # Sanity: rescaled max should be ~RESCALE_FACTOR × pre_max ≈ 600.
    assert float(arr.max()) < 1.0e4, "Rescale overshot — check factor"


def test_no_rescale_intensity_leaves_low_magnitudes_unchanged(
    extract_mod: types.ModuleType, tmp_path: Path
) -> None:
    """Passing --no-rescale-intensity (i.e. rescale_intensity=False) must
    preserve the raw RSS magnitudes; the output's max stays in the input
    regime (well below MIN_MAX_INTENSITY).
    """
    h5_path = _write_fastmri_h5(
        tmp_path / "low_mag_keep.h5",
        acquisition="AXT1",
        shape=(4, 8, 8),
        rss_scale=1.0e-6,
    )
    output_path = tmp_path / "raw.nii.gz"

    record = extract_mod.extract_one(
        h5_path,
        output_path,
        frozenset({"AXT1"}),
        force=False,
        dry_run=False,
        rescale_intensity=False,
    )

    assert record is not None
    img = nib.load(str(output_path))
    arr = img.get_fdata()
    assert float(arr.max()) < 1.0, (
        f"Expected raw low-mag passthrough, got max={arr.max():.3e}"
    )


def test_rescale_intensity_skips_when_max_already_in_range(
    extract_mod: types.ModuleType, tmp_path: Path
) -> None:
    """Already-typical-intensity inputs (max >= RESCALE_THRESHOLD) must
    pass through unchanged even with rescale_intensity=True.
    """
    h5_path = _write_fastmri_h5(
        tmp_path / "typical.h5",
        acquisition="AXT1",
        shape=(4, 8, 8),
        # default rss_scale=1.0 → max ≈ 600 (already > RESCALE_THRESHOLD).
    )
    output_path = tmp_path / "typical.nii.gz"

    record = extract_mod.extract_one(
        h5_path,
        output_path,
        frozenset({"AXT1"}),
        force=False,
        dry_run=False,
        rescale_intensity=True,
    )

    assert record is not None
    img = nib.load(str(output_path))
    arr = img.get_fdata()
    # Should be in the original ~100-600 range, not multiplied by 1e6.
    assert 100.0 < float(arr.max()) < 1.0e4


def test_extract_all_filters_by_acquisition_and_writes_manifest(
    extract_mod: types.ModuleType, tmp_path: Path
) -> None:
    """Non-allow-listed acquisitions are skipped; manifest holds only the kept ones."""
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "nifti"
    manifest_csv = tmp_path / "manifest.csv"

    _write_fastmri_h5(
        input_dir / "file_brain_AXT1_001_0000001.h5",
        acquisition="AXT1",
        shape=(3, 8, 8),
        field_strength=1.5,
    )
    _write_fastmri_h5(
        input_dir / "file_brain_AXT1POST_002_0000002.h5",
        acquisition="AXT1POST",
        shape=(3, 8, 8),
        field_strength=3.0,
    )
    _write_fastmri_h5(
        input_dir / "file_brain_AXT2_003_0000003.h5",
        acquisition="AXT2",  # must be skipped
        shape=(3, 8, 8),
        field_strength=3.0,
    )
    _write_fastmri_h5(
        input_dir / "file_brain_AXFLAIR_004_0000004.h5",
        acquisition="AXFLAIR",  # must be skipped
        shape=(3, 8, 8),
        field_strength=3.0,
    )

    records = extract_mod.extract_all(
        input_dir=input_dir,
        output_dir=output_dir,
        manifest_csv=manifest_csv,
        allowed_acquisitions=frozenset({"AXT1", "AXT1PRE", "AXT1POST"}),
        dry_run=False,
        force=False,
    )

    assert len(records) == 2
    kept_acqs = sorted(r.acquisition_type for r in records)
    assert kept_acqs == ["AXT1", "AXT1POST"]

    # Only the kept files produce .nii.gz outputs.
    niftis = sorted(output_dir.glob("*.nii.gz"))
    assert [p.stem.replace(".nii", "") for p in niftis] == sorted(
        [
            "file_brain_AXT1_001_0000001",
            "file_brain_AXT1POST_002_0000002",
        ]
    )

    # Manifest schema and content.
    assert manifest_csv.exists()
    df = pd.read_csv(manifest_csv)
    assert list(df.columns) == extract_mod.MANIFEST_COLUMNS
    assert len(df) == 2
    assert set(df["acquisition_type"]) == {"AXT1", "AXT1POST"}
    assert set(df["field_strength"].astype(float).tolist()) == {1.5, 3.0}
    assert (df["n_slices"] == 3).all()
    assert (df["H"] == 8).all()
    assert (df["W"] == 8).all()
