"""Unit tests for code/01_curate_references.py.

All tests synthesise NIfTI volumes with nibabel.Nifti1Image; no external data
is required. The module under test has a digit-prefixed filename, so it is
loaded via importlib rather than a normal import statement.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import types
from pathlib import Path

import nibabel as nib
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE_PATH = _REPO_ROOT / "code" / "01_curate_references.py"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("curate_references", _MODULE_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {_MODULE_PATH}"
    module = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can resolve string annotations under
    # `from __future__ import annotations` (it looks up cls.__module__).
    sys.modules["curate_references"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def curate_mod() -> types.ModuleType:
    return _load_module()


# ──────────────────────────────────────────────
# Synthesis helpers
# ──────────────────────────────────────────────


def _save_volume(
    path: Path,
    shape: tuple[int, int, int],
    *,
    voxel_mm: tuple[float, float, float] | float = 1.0,
    fill: str = "rand",
) -> Path:
    """Synthesise and save a NIfTI volume. Returns the written path.

    fill:
      "rand"  — torch.rand * 500 + 500 (passes intensity rule)
      "zeros" — all zeros
      "nan"   — all NaN
    """
    if fill == "rand":
        data = torch.rand(*shape) * 500.0 + 500.0
    elif fill == "zeros":
        data = torch.zeros(*shape)
    elif fill == "nan":
        data = torch.full(shape, float("nan"))
    else:
        raise ValueError(f"unknown fill: {fill}")

    zooms: tuple[float, float, float] = (
        (float(voxel_mm),) * 3 if isinstance(voxel_mm, (int, float)) else voxel_mm
    )
    affine = torch.eye(4)
    affine[0, 0] = zooms[0]
    affine[1, 1] = zooms[1]
    affine[2, 2] = zooms[2]

    img = nib.Nifti1Image(data.numpy(), affine=affine.numpy())
    img.header.set_zooms(zooms)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))
    return path


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("IXI002-Guys-0828-T1.nii.gz", ("002", "Guys")),
        ("IXI314-HH-1234-T1.nii.gz", ("314", "HH")),
        ("IXI015-IOP-0042-T1.nii.gz", ("015", "IOP")),
        ("corrupted_001.nii.gz", (None, None)),
        ("IXI002-Guys-0828-T2.nii.gz", (None, None)),
        ("IXI002-Guys-0828-T1.nii", (None, None)),
    ],
)
def test_parse_ixi_filename(
    curate_mod: types.ModuleType,
    filename: str,
    expected: tuple[str | None, str | None],
) -> None:
    assert curate_mod.parse_ixi_filename(filename) == expected


def test_inspect_scan_passes_clean_volume(
    curate_mod: types.ModuleType, tmp_path: Path
) -> None:
    path = _save_volume(
        tmp_path / "IXI001-Guys-0001-T1.nii.gz",
        shape=(192, 192, 192),
        voxel_mm=1.0,
        fill="rand",
    )

    record = curate_mod.inspect_scan(path, curate_mod.THRESHOLDS_BY_TAG["ixi"])

    assert record.passed_qc is True
    assert record.shape == (192, 192, 192)
    assert record.voxel_sizes == pytest.approx((1.0, 1.0, 1.0))
    assert record.subject_id == "001"
    assert record.hospital == "Guys"
    assert record.max_intensity > curate_mod.MIN_MAX_INTENSITY
    assert record.min_intensity >= 0.0


def test_curate_end_to_end(
    curate_mod: types.ModuleType, tmp_path: Path
) -> None:
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    manifest_path = tmp_path / "reference_manifest.csv"

    passing = _save_volume(
        input_dir / "IXI001-Guys-0001-T1.nii.gz",
        shape=(192, 192, 192),
        fill="rand",
    )
    _save_volume(
        input_dir / "IXI002-HH-0002-T1.nii.gz",
        shape=(100, 100, 100),  # fails: too small
        fill="rand",
    )
    _save_volume(
        input_dir / "IXI003-IOP-0003-T1.nii.gz",
        shape=(192, 192, 192),
        fill="nan",  # fails: NaN
    )

    records = curate_mod.curate(
        input_dir=input_dir,
        output_dir=output_dir,
        manifest_path=manifest_path,
        dataset_tag="ixi",
        dry_run=False,
        force=False,
    )

    assert len(records) == 3
    assert sum(1 for r in records if r.passed_qc) == 1

    assert manifest_path.exists()
    with manifest_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert [r["passed_qc"] for r in rows].count("true") == 1
    assert set(rows[0].keys()) == set(curate_mod.MANIFEST_COLUMNS)

    symlinks = sorted(output_dir.iterdir())
    assert len(symlinks) == 1
    only = symlinks[0]
    assert only.is_symlink()
    assert only.readlink() == passing.resolve()
    assert only.name == "IXI001-Guys-0001-T1.nii.gz"


def test_fastmri_tag_accepts_anisotropic_thin_slab(
    curate_mod: types.ModuleType, tmp_path: Path
) -> None:
    """FastMRI tag relaxes Z to >=15 and voxel upper bound to 6.0mm.

    A 320x320x20 slab at 0.7/0.7/5.0mm should PASS under "fastmri" but FAIL
    under "ixi" (Z<150, slice voxel>2.0mm).
    """
    path = _save_volume(
        tmp_path / "fm_001.nii.gz",
        shape=(320, 320, 20),
        voxel_mm=(0.7, 0.7, 5.0),
        fill="rand",
    )

    fm_record = curate_mod.inspect_scan(
        path, curate_mod.THRESHOLDS_BY_TAG["fastmri"]
    )
    ixi_record = curate_mod.inspect_scan(
        path, curate_mod.THRESHOLDS_BY_TAG["ixi"]
    )

    assert fm_record.passed_qc is True
    assert ixi_record.passed_qc is False
    assert fm_record.shape == (320, 320, 20)
    assert fm_record.voxel_sizes == pytest.approx((0.7, 0.7, 5.0))
