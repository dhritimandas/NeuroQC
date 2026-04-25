"""Unit tests for code/02b_corrupt_kspace_motion.py + code/verify_02b_kspace.py.

Synthetic FastMRI-shaped .h5 fixtures are built with h5py. We construct a
random complex k-space, then compute ``reconstruction_rss`` with the SAME
centred iFFT convention the script uses (``ifft2c``) — that way the
severity-zero round-trip test is meaningful: if the reference and our
reconstruction agree, then our convention is consistent end-to-end. A
failure would mean the script's ``ifft2c`` doesn't match the fixture's,
which is the most common first-scan convention bug.

Both modules have digit-prefixed / underscore-prefixed filenames and are
loaded via importlib, consistent with the rest of the test suite.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import types
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pytest
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CORRUPT_PATH = _REPO_ROOT / "code" / "02b_corrupt_kspace_motion.py"
_VERIFY_PATH = _REPO_ROOT / "code" / "verify_02b_kspace.py"


def _load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def corrupt_mod() -> types.ModuleType:
    return _load("corrupt_kspace_motion", _CORRUPT_PATH)


@pytest.fixture(scope="module")
def verify_mod() -> types.ModuleType:
    return _load("verify_02b_kspace", _VERIFY_PATH)


# ──────────────────────────────────────────────
# Synthesis helpers
# ──────────────────────────────────────────────


def _ifft2c(k: np.ndarray) -> np.ndarray:
    """Centred orthonormal iFFT along last two axes — mirrors 02b's ifft2c.

    Uses ``norm='ortho'`` (FastMRI convention). Without this the synthetic
    ``reconstruction_rss`` would disagree with 02b's output by a factor of
    √(N·M), which is exactly the kind of bug the severity-0 test is
    supposed to catch.
    """
    return np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(k, axes=(-2, -1)), axes=(-2, -1), norm="ortho"
        ),
        axes=(-2, -1),
    )


def _rss(coil_images: np.ndarray, coil_axis: int) -> np.ndarray:
    return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=coil_axis)).astype(np.float32)


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


def _write_fastmri_like_h5(
    path: Path,
    *,
    acquisition: str,
    shape: tuple[int, int, int, int],  # (n_slices, n_coils, H, W)
    seed: int = 0,
    fov_mm: tuple[float, float, float] = (220.0, 220.0, 5.0),
) -> Path:
    """Write a minimal FastMRI-shaped .h5 file (kspace + reconstruction_rss).

    The k-space is multiplied by a Gaussian envelope centred at DC so that
    the reconstructed image has low-frequency-dominant structure (closer to
    a real MR scan than white noise). White k-space produces no visible
    ghost pattern under segmented motion, defeating the PE-axis test.

    ``fov_mm`` = (fov_x, fov_y, fov_z) populates the ismrmrd_header so 02b
    (via 00's ``parse_voxel_sizes``) derives voxel sizes. Default mirrors
    real FastMRI brain AXT1: 220 × 220 × 5 mm FOV.
    """
    rng = np.random.default_rng(seed)
    n_slices, n_coils, height, width = shape
    real = rng.standard_normal(shape).astype(np.float32)
    imag = rng.standard_normal(shape).astype(np.float32)
    kspace = (real + 1j * imag).astype(np.complex64)
    # Low-frequency-dominant envelope (sigma ~ 1/6 of matrix) — standard
    # synthetic MR k-space shaping. This is only so the ghost signatures in
    # Check 3 are measurable; the severity-zero test is unaffected because
    # we compute the stored RSS from the envelope-weighted k-space too.
    ky = np.arange(height) - height / 2.0
    kx = np.arange(width) - width / 2.0
    envelope = np.exp(
        -(ky[:, None] ** 2 / (2 * (height / 6) ** 2))
        - (kx[None, :] ** 2 / (2 * (width / 6) ** 2))
    ).astype(np.float32)
    kspace = (kspace * envelope[None, None, :, :]).astype(np.complex64)
    # reconstruction_rss computed with the same convention 02b uses.
    coil_images = _ifft2c(kspace)  # (n_slices, n_coils, H, W)
    rss = _rss(coil_images, coil_axis=1)  # (n_slices, H, W)
    fov_x, fov_y, fov_z = fov_mm
    xml = _ISMRMRD_HEADER_TEMPLATE.format(
        mat_x=width, mat_y=height, fov_x=fov_x, fov_y=fov_y, fov_z=fov_z
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("kspace", data=kspace)
        handle.create_dataset("reconstruction_rss", data=rss)
        handle.create_dataset("ismrmrd_header", data=xml)
        handle.attrs["acquisition"] = acquisition
    return path


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────


def test_severity_zero_matches_stored_rss(
    corrupt_mod: types.ModuleType, tmp_path: Path
) -> None:
    """The load-bearing check: zero-event corruption reproduces stored RSS.

    A regression here almost always means the FFT convention drifted
    (fftshift/ifftshift ordering) or the zero-event code path is no
    longer a bit-for-bit no-op.
    """
    shape = (3, 4, 16, 16)
    h5_path = _write_fastmri_like_h5(
        tmp_path / "zero.h5", acquisition="AXT1", shape=shape, seed=1
    )
    with h5py.File(h5_path, "r") as handle:
        kspace = handle["kspace"][:]
        stored_rss = handle["reconstruction_rss"][:]

    # Use the n_transforms=0 equivalent: empty events for every slice.
    zero_corrupted = np.stack(
        [
            corrupt_mod.apply_motion_to_slice(kspace[i], events=[], voxel_mm=(1.0, 1.0))
            for i in range(shape[0])
        ],
        axis=0,
    )
    assert np.allclose(zero_corrupted, kspace, atol=0.0), "zero-event path is not a no-op"

    rss_ours = corrupt_mod.reconstruct_rss_volume(zero_corrupted)
    rel_err = float(
        np.max(np.abs(rss_ours - stored_rss)) / max(float(np.max(stored_rss)), 1e-12)
    )
    assert rel_err < 1e-4, (
        f"Our reconstruction disagrees with stored RSS (rel_err={rel_err:.2e}). "
        "FFT convention is probably wrong — check ifft2c()."
    )


def test_phase_encode_axis_orientation(
    corrupt_mod: types.ModuleType, tmp_path: Path
) -> None:
    """Multi-event translation produces ghosts oriented along PE = W (axis -1).

    Motion ghosting modulates image intensity along the phase-encode axis.
    After averaging the diff image along the non-PE axis, the remaining
    variation is concentrated in the PE axis. With PE = axis -1 (W) — the
    FastMRI brain convention — the signature is
    ``var(col_mean) > 3 × var(row_mean)``. A silent swap to axis=-2 flips
    the ratio.
    """
    shape = (1, 2, 48, 36)  # asymmetric H != W to disambiguate axes
    h5_path = _write_fastmri_like_h5(
        tmp_path / "pe.h5", acquisition="AXT1", shape=shape, seed=2
    )
    with h5py.File(h5_path, "r") as handle:
        kspace_slice = handle["kspace"][0]  # (C, H, W)

    ref_img = np.sqrt(
        np.sum(np.abs(corrupt_mod.ifft2c(kspace_slice)) ** 2, axis=0)
    ).astype(np.float32)

    events = [
        corrupt_mod.MotionEvent(rotation_deg=0.0, translation_mm_x=4.0, translation_mm_y=0.0),
        corrupt_mod.MotionEvent(rotation_deg=0.0, translation_mm_x=-3.0, translation_mm_y=0.0),
        corrupt_mod.MotionEvent(rotation_deg=0.0, translation_mm_x=5.0, translation_mm_y=0.0),
        corrupt_mod.MotionEvent(rotation_deg=0.0, translation_mm_x=-6.0, translation_mm_y=0.0),
    ]
    cor = corrupt_mod.apply_motion_to_slice(
        kspace_slice, events=events, voxel_mm=(1.0, 1.0)
    )
    cor_img = np.sqrt(np.sum(np.abs(corrupt_mod.ifft2c(cor)) ** 2, axis=0)).astype(
        np.float32
    )

    diff = cor_img - ref_img
    var_row_mean = float(np.var(diff.mean(axis=1)))
    var_col_mean = float(np.var(diff.mean(axis=0)))
    assert var_col_mean > 3.0 * var_row_mean, (
        f"PE-axis ghosts not oriented along W: var(col_mean)={var_col_mean:.3e} "
        f"var(row_mean)={var_row_mean:.3e}. Partition axis is likely -2, should be -1."
    )


def test_corrupt_one_end_to_end_writes_nifti_json_and_record(
    corrupt_mod: types.ModuleType, tmp_path: Path
) -> None:
    """End-to-end pipeline on one synthetic AXT1 file at severity 3."""
    shape = (2, 3, 16, 16)
    h5_path = _write_fastmri_like_h5(
        tmp_path / "raw" / "file_brain_AXT1_999_0000001.h5",
        acquisition="AXT1",
        shape=shape,
        seed=3,
    )
    reference_dir = tmp_path / "nifti"
    reference_dir.mkdir()
    # A placeholder reference NIfTI so ref_path.resolve() is meaningful —
    # 02b doesn't actually read it, only records the path.
    (reference_dir / f"{h5_path.stem}.nii.gz").write_bytes(b"")

    output_dir = tmp_path / "cor"

    record = corrupt_mod.corrupt_one(
        h5_path,
        reference_dir=reference_dir,
        output_dir=output_dir,
        severity=3,
        base_seed=42,
        dry_run=False,
        force=False,
        voxel_mm=(1.0, 1.0, 1.0),
        n_jobs=1,
    )

    assert record is not None
    assert record.severity == 3
    assert record.cor_path.exists()
    # JSON sidecar next to the NIfTI.
    json_path = record.cor_path.with_suffix("").with_suffix(".json")
    assert json_path.exists()
    with json_path.open() as handle:
        payload = json.load(handle)
    assert payload["corruption_type"] == "motion_kspace"
    assert payload["severity"] == 3
    assert payload["n_transforms"] == corrupt_mod.SEVERITY_CONFIGS[3].n_transforms
    assert len(payload["events_per_slice"]) == shape[0]

    # NIfTI is a valid 3D volume.
    img = nib.load(str(record.cor_path))
    assert img.ndim == 3
    data = img.get_fdata()
    assert np.isfinite(data).all()

    # Manifest row shape.
    row = record.to_manifest_row()
    assert set(row.keys()) == set(corrupt_mod.MANIFEST_COLUMNS)
    assert row["corruption_type"] == "motion_kspace"
    assert row["corruption_domain"] == "kspace"
    assert row["dataset_tag"] == "fastmri"


def test_corrupt_all_filters_non_t1_and_updates_manifest_idempotently(
    corrupt_mod: types.ModuleType, tmp_path: Path
) -> None:
    """Non-T1 acquisitions are skipped; motion_kspace rows are replaced on rerun."""
    input_dir = tmp_path / "raw"
    reference_dir = tmp_path / "nifti"
    reference_dir.mkdir()
    output_dir = tmp_path / "cor"
    manifest_path = tmp_path / "corruption_manifest.csv"

    # Pre-seed the manifest with a TorchIO (image-space) motion row that
    # 02b must leave untouched.
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=corrupt_mod.MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "ref_path": "/tmp/irrelevant.nii.gz",
                "cor_path": "/tmp/irrelevant_cor.nii.gz",
                "corruption_type": "motion",
                "corruption_domain": "kspace",
                "severity": "1",
                "seed": "0",
                "transform_params": "{}",
                "dataset_tag": "ixi",
            }
        )

    for acq in ("AXT1", "AXT2", "AXFLAIR"):
        path = input_dir / f"file_brain_{acq}_{acq}_001.h5"
        _write_fastmri_like_h5(
            path, acquisition=acq, shape=(1, 2, 12, 12), seed=hash(acq) & 0xFFFF
        )
        (reference_dir / f"{path.stem}.nii.gz").write_bytes(b"")

    # First run.
    records = corrupt_mod.corrupt_all(
        input_dir=input_dir,
        reference_dir=reference_dir,
        output_dir=output_dir,
        severities=[1],
        base_seed=7,
        manifest_path=manifest_path,
        dry_run=False,
        force=False,
        voxel_mm=(1.0, 1.0, 1.0),
        n_jobs=1,
    )
    assert len(records) == 1
    assert "AXT1" in records[0].cor_path.name

    rows = list(csv.DictReader(manifest_path.open()))
    kinds = [r["corruption_type"] for r in rows]
    assert kinds.count("motion") == 1  # pre-existing row preserved
    assert kinds.count("motion_kspace") == 1

    # Second run — must be idempotent (no duplicate motion_kspace rows).
    corrupt_mod.corrupt_all(
        input_dir=input_dir,
        reference_dir=reference_dir,
        output_dir=output_dir,
        severities=[1],
        base_seed=7,
        manifest_path=manifest_path,
        dry_run=False,
        force=False,
        voxel_mm=(1.0, 1.0, 1.0),
        n_jobs=1,
    )
    rows_after = list(csv.DictReader(manifest_path.open()))
    kinds_after = [r["corruption_type"] for r in rows_after]
    assert kinds_after.count("motion") == 1
    assert kinds_after.count("motion_kspace") == 1


def test_verify_script_passes_on_valid_build(
    verify_mod: types.ModuleType, tmp_path: Path
) -> None:
    """All six diagnostic checks PASS on a fixture consistent with 02b.

    This catches regressions where a refactor to 02b silently breaks the
    convention the verify script claims to validate.
    """
    shape = (3, 4, 64, 64)
    h5_path = _write_fastmri_like_h5(
        tmp_path / "real_like.h5", acquisition="AXT1", shape=shape, seed=11
    )
    out_dir = tmp_path / "diagnostics"

    results = verify_mod.run_all_checks(
        h5_path=h5_path,
        out_dir=out_dir,
        slice_idx=None,
        strict_tol=1e-4,
        make_plot=False,  # keep the test fast; plot_grid is a pure matplotlib path.
    )
    failing = [r for r in results if r.status != "PASS"]
    assert not failing, (
        "Diagnostic check(s) failed on a fixture built with the same "
        "convention as the script:\n"
        + "\n".join(f"  - {r.name}: {r.fix_hint}" for r in failing)
    )

    # CLI entry-point smoke: does --help render without Typer errors?
    runner = CliRunner()
    result = runner.invoke(verify_mod.app, ["--help"])
    assert result.exit_code == 0, result.output
