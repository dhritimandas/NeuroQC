"""Unit tests for code/03b_compute_thickness.py.

Synthetic data only — no downloaded datasets, no FreeSurfer required. A minimal
colour-lookup-table file is written into tmp_path so tests exercise the real
LUT parser without depending on the installed FreeSurfer.

Expected-value notes
--------------------
The DiReCT-style EDT approximation measures center-to-center distances, so a
``k``-voxel-thick slab bounded by WM on one side and background on the other
yields ``(k + 1) * voxel_size_along_slab_axis`` millimetres, not ``k *`` the
voxel size. A 5-voxel slab at iso 1 mm therefore gives 6.0 mm exactly (not
5.0), and the same slab at 5 mm z-voxels gives 30.0 mm (not 25.0). The +1
offset is a well-known systematic bias of voxel-EDT thickness — we keep it
because the corruption ablations care about *differences* in thickness, which
the bias cancels out of.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
import types
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE_PATH = _REPO_ROOT / "code" / "03b_compute_thickness.py"


def _load_module() -> types.ModuleType:
    """Import 03b_compute_thickness.py despite its digit-prefixed filename."""
    spec = importlib.util.spec_from_file_location("compute_thickness", _MODULE_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {_MODULE_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["compute_thickness"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> types.ModuleType:
    return _load_module()


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ──────────────────────────────────────────────
# Fixtures — synthetic LUT + synthetic SynthSeg NIfTIs
# ──────────────────────────────────────────────


def _write_fake_lut(freesurfer_home: Path) -> Path:
    """Write a minimal FreeSurferColorLUT.txt covering WM + all DK labels."""
    freesurfer_home.mkdir(parents=True, exist_ok=True)
    lut_path = freesurfer_home / "FreeSurferColorLUT.txt"

    dk_lh = [
        "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "corpuscallosum",
        "cuneus", "entorhinal", "fusiform", "inferiorparietal", "inferiortemporal",
        "isthmuscingulate", "lateraloccipital", "lateralorbitofrontal", "lingual",
        "medialorbitofrontal", "middletemporal", "parahippocampal", "paracentral",
        "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine",
        "postcentral", "posteriorcingulate", "precentral", "precuneus",
        "rostralanteriorcingulate", "rostralmiddlefrontal", "superiorfrontal",
        "superiorparietal", "superiortemporal", "supramarginal", "frontalpole",
        "temporalpole", "transversetemporal", "insula",
    ]
    assert len(dk_lh) == 35, "DK schema expects 35 regions per hemisphere"

    with lut_path.open("w") as handle:
        handle.write("# minimal synthetic LUT for tests\n")
        handle.write("2\tLeft-Cerebral-White-Matter\t245\t245\t245\t0\n")
        handle.write("41\tRight-Cerebral-White-Matter\t245\t245\t245\t0\n")
        for i, name in enumerate(dk_lh):
            handle.write(f"{1001 + i}\tctx-lh-{name}\t25\t100\t40\t0\n")
            handle.write(f"{2001 + i}\tctx-rh-{name}\t25\t100\t40\t0\n")
    return lut_path


def _write_slab_seg(path: Path, affine: np.ndarray, label: int = 1001) -> None:
    """Write a 40x40x40 int16 slab: WM at z=0..19, cortex ``label`` at z=20..24."""
    seg = np.zeros((40, 40, 40), dtype=np.int16)
    seg[:, :, 0:20] = 2
    seg[:, :, 20:25] = label
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(seg, affine), str(path))


# ──────────────────────────────────────────────
# Core computation
# ──────────────────────────────────────────────


def test_slab_thickness_isotropic(mod: types.ModuleType, tmp_path: Path) -> None:
    """5-voxel iso slab → EDT center-to-center gives 6.0 mm exactly."""
    lut_path = _write_fake_lut(tmp_path / "freesurfer")
    label_names = mod.load_region_names(lut_path)

    seg_path = tmp_path / "scan_synthseg.nii.gz"
    _write_slab_seg(seg_path, affine=np.eye(4), label=1001)

    row = mod.compute_thickness(seg_path, label_names)

    assert row.per_region["ctx-lh-bankssts"] == pytest.approx(6.0, abs=0.05)
    assert row.mean_thickness == pytest.approx(6.0, abs=0.05)


def test_anisotropic_voxel_invariance(mod: types.ModuleType, tmp_path: Path) -> None:
    """5-voxel slab with 5 mm z-voxels → 30.0 mm, not 6.0 mm.

    If the implementation forgot to scale the EDT by vox_mm (computed as the
    column norms of affine[:3,:3]) the result would be 6.0 mm — the same as
    iso — which would be wrong by a factor of 5. This test guards the
    physical-mm conversion.
    """
    lut_path = _write_fake_lut(tmp_path / "freesurfer")
    label_names = mod.load_region_names(lut_path)

    seg_path = tmp_path / "scan_synthseg.nii.gz"
    aniso_affine = np.diag([0.6875, 0.6875, 5.0, 1.0])
    _write_slab_seg(seg_path, affine=aniso_affine, label=1001)

    row = mod.compute_thickness(seg_path, label_names)

    assert row.per_region["ctx-lh-bankssts"] == pytest.approx(30.0, abs=0.05)


def test_empty_region_is_nan(mod: types.ModuleType, tmp_path: Path) -> None:
    """Labels absent from the segmentation yield NaN, not 0 or an exception."""
    lut_path = _write_fake_lut(tmp_path / "freesurfer")
    label_names = mod.load_region_names(lut_path)

    seg_path = tmp_path / "scan_synthseg.nii.gz"
    # Only label 1001 present; 1035 (ctx-lh-insula) is absent.
    _write_slab_seg(seg_path, affine=np.eye(4), label=1001)

    row = mod.compute_thickness(seg_path, label_names)

    assert math.isnan(row.per_region["ctx-lh-insula"])
    assert not math.isnan(row.per_region["ctx-lh-bankssts"])


# ──────────────────────────────────────────────
# Resume / CLI
# ──────────────────────────────────────────────


def test_resume_idempotency(
    mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Scans already in the output CSV are not recomputed on a second run."""
    fs_home = tmp_path / "freesurfer"
    _write_fake_lut(fs_home)

    synthseg_dir = tmp_path / "synthseg"
    scan_a = synthseg_dir / "scanA_synthseg.nii.gz"
    scan_b = synthseg_dir / "scanB_synthseg.nii.gz"
    _write_slab_seg(scan_a, affine=np.eye(4), label=1001)
    _write_slab_seg(scan_b, affine=np.eye(4), label=1001)

    output_file = tmp_path / "cortical_thickness.csv"

    # First run: process both scans.
    result = runner.invoke(
        mod.app,
        [
            "--synthseg-dir", str(synthseg_dir),
            "--output-file", str(output_file),
            "--freesurfer-home", str(fs_home),
        ],
    )
    assert result.exit_code == 0, result.output
    with output_file.open("r", newline="") as handle:
        rows_first = list(csv.DictReader(handle))
    assert {r["scan_path"] for r in rows_first} == {str(scan_a.resolve()), str(scan_b.resolve())}
    assert len(rows_first) == 2

    # Capture mean_thickness so we can prove the row wasn't rewritten.
    first_mean_a = next(r for r in rows_first if r["scan_path"] == str(scan_a.resolve()))[
        "mean_thickness"
    ]

    # Second run with the same args: no new rows should appear.
    result = runner.invoke(
        mod.app,
        [
            "--synthseg-dir", str(synthseg_dir),
            "--output-file", str(output_file),
            "--freesurfer-home", str(fs_home),
        ],
    )
    assert result.exit_code == 0, result.output
    with output_file.open("r", newline="") as handle:
        rows_second = list(csv.DictReader(handle))
    assert len(rows_second) == 2, "resume should not duplicate existing rows"
    assert rows_second[0]["mean_thickness"] == first_mean_a

    # Add a third scan and confirm only the new one gets processed.
    scan_c = synthseg_dir / "scanC_synthseg.nii.gz"
    _write_slab_seg(scan_c, affine=np.eye(4), label=1001)
    result = runner.invoke(
        mod.app,
        [
            "--synthseg-dir", str(synthseg_dir),
            "--output-file", str(output_file),
            "--freesurfer-home", str(fs_home),
        ],
    )
    assert result.exit_code == 0, result.output
    with output_file.open("r", newline="") as handle:
        rows_third = list(csv.DictReader(handle))
    assert len(rows_third) == 3
    scan_paths_third = {r["scan_path"] for r in rows_third}
    assert scan_paths_third == {
        str(scan_a.resolve()),
        str(scan_b.resolve()),
        str(scan_c.resolve()),
    }


def test_manifest_mode_writes_input_scan_path(
    mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """When --synthseg-manifest is passed, scan_path holds the input scan path
    and seg_path holds the SynthSeg output. Both columns are written."""
    fs_home = tmp_path / "freesurfer"
    _write_fake_lut(fs_home)

    synthseg_dir = tmp_path / "synthseg"
    seg_a = synthseg_dir / "scanA_synthseg.nii.gz"
    _write_slab_seg(seg_a, affine=np.eye(4), label=1001)

    # The "input scan" the seg was supposedly derived from. 03b never reads
    # it; only its path is recorded in the output via the manifest.
    input_scan_a = tmp_path / "scans" / "scanA.nii.gz"
    input_scan_a.parent.mkdir(parents=True, exist_ok=True)
    input_scan_a.touch()

    manifest = synthseg_dir / "synthseg_manifest.csv"
    with manifest.open("w", newline="") as h:
        w = csv.DictWriter(
            h, fieldnames=["input_path", "seg_path", "qc_path", "vol_path", "mode", "status"]
        )
        w.writeheader()
        w.writerow({
            "input_path": str(input_scan_a),
            "seg_path": str(seg_a),
            "qc_path": "",
            "vol_path": "",
            "mode": "freesurfer",
            "status": "ok",
        })

    output_file = tmp_path / "cortical_thickness.csv"
    result = runner.invoke(
        mod.app,
        [
            "--synthseg-dir", str(synthseg_dir),
            "--output-file", str(output_file),
            "--freesurfer-home", str(fs_home),
            "--synthseg-manifest", str(manifest),
        ],
    )
    assert result.exit_code == 0, result.output

    with output_file.open() as h:
        rows = list(csv.DictReader(h))
    assert len(rows) == 1
    row = rows[0]
    # scan_path is the INPUT scan (post-manifest-mode); seg_path is the SEG.
    assert row["scan_path"] == str(input_scan_a.resolve())
    assert row["seg_path"] == str(seg_a.resolve())
    # mean_thickness is finite (slab seg → real DiReCT value).
    assert float(row["mean_thickness"]) > 0


def test_manifest_mode_missing_entry_falls_back_with_warning(
    mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """If --synthseg-manifest is passed but an SEG isn't in any manifest,
    that SEG's row falls back to seg_path under scan_path with a warning."""
    fs_home = tmp_path / "freesurfer"
    _write_fake_lut(fs_home)

    synthseg_dir = tmp_path / "synthseg"
    seg_a = synthseg_dir / "scanA_synthseg.nii.gz"
    seg_b = synthseg_dir / "scanB_synthseg.nii.gz"
    _write_slab_seg(seg_a, affine=np.eye(4), label=1001)
    _write_slab_seg(seg_b, affine=np.eye(4), label=1001)

    input_scan_a = tmp_path / "scans" / "scanA.nii.gz"
    input_scan_a.parent.mkdir(parents=True, exist_ok=True)
    input_scan_a.touch()

    # Manifest covers ONLY scanA; scanB is unmatched.
    manifest = synthseg_dir / "synthseg_manifest.csv"
    with manifest.open("w", newline="") as h:
        w = csv.DictWriter(
            h, fieldnames=["input_path", "seg_path", "qc_path", "vol_path", "mode", "status"]
        )
        w.writeheader()
        w.writerow({
            "input_path": str(input_scan_a),
            "seg_path": str(seg_a),
            "qc_path": "", "vol_path": "", "mode": "freesurfer", "status": "ok",
        })

    output_file = tmp_path / "cortical_thickness.csv"
    result = runner.invoke(
        mod.app,
        [
            "--synthseg-dir", str(synthseg_dir),
            "--output-file", str(output_file),
            "--freesurfer-home", str(fs_home),
            "--synthseg-manifest", str(manifest),
        ],
    )
    assert result.exit_code == 0, result.output

    with output_file.open() as h:
        rows = {r["seg_path"]: r for r in csv.DictReader(h)}
    # Two rows total. scanA: scan_path = input. scanB: scan_path falls back
    # to seg_path because no manifest entry covered it.
    assert rows[str(seg_a.resolve())]["scan_path"] == str(input_scan_a.resolve())
    assert rows[str(seg_b.resolve())]["scan_path"] == str(seg_b.resolve())
