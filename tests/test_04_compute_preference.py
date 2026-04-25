"""Unit tests for code/04_compute_preference.py.

Synthetic data only — no downloaded datasets, no SynthSeg run. Each test builds
a tiny SynthSeg-style label volume and a minimal cortical_thickness.csv in
``tmp_path`` and runs the module end-to-end.

Design notes
------------
``nobrainer.qc.preference.compute_dice_preference`` loads via nibabel and
expects integer labels matching FreeSurfer's scheme. The fixture writes
volumes with a handful of non-zero labels (one per structure we care about)
plus a WM+cortex patch so ``mean_dice`` is defined for identical-seg pairs.
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
import pandas as pd
import pytest
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE_PATH = _REPO_ROOT / "code" / "04_compute_preference.py"


def _load_module() -> types.ModuleType:
    """Import 04_compute_preference.py despite its digit-prefixed filename.

    ``sys.modules`` registration is required for ``@dataclass(frozen=True)``
    to resolve forward references under ``from __future__ import annotations``.
    """
    spec = importlib.util.spec_from_file_location("compute_preference", _MODULE_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {_MODULE_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["compute_preference"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> types.ModuleType:
    return _load_module()


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ──────────────────────────────────────────────
# Fixtures — synthetic seg volumes + thickness table
# ──────────────────────────────────────────────


# FreeSurfer label IDs used by compute_dice_preference (STRUCTURE_LABELS).
_HIPPO_LH = 17
_CORTEX_LH = 3
_VENTRICLE_LH = 4
_THALAMUS_LH = 10
_CAUDATE_LH = 11
_PUTAMEN_LH = 12
_BRAINSTEM = 16
_CEREBELLUM_LH = 8
_WM_LH = 2
_DK_BANKSSTS_LH = 1001


def _write_seg(path: Path, shuffle_z: int = 0) -> None:
    """Write a 20×20×20 int16 seg NIfTI with every scored structure present.

    ``shuffle_z`` shifts the cortex/WM slab along z to produce a controlled
    mismatch between reference and corrupted volumes. shuffle_z=0 → identical
    to the canonical reference. shuffle_z=3 → Dice < 1 for cortex/WM
    (hippocampus/thalamus etc. are left alone so those Dice remain ~1.0).
    """
    seg = np.zeros((20, 20, 20), dtype=np.int16)
    # Small cube per structure so every _dice key is defined (non-zero ∧ non-zero).
    seg[0:3, 0:3, 0:3] = _HIPPO_LH
    seg[0:3, 0:3, 4:7] = _THALAMUS_LH
    seg[0:3, 0:3, 8:11] = _CAUDATE_LH
    seg[0:3, 0:3, 12:15] = _PUTAMEN_LH
    seg[4:7, 0:3, 0:3] = _BRAINSTEM
    seg[4:7, 0:3, 4:7] = _CEREBELLUM_LH
    seg[4:7, 0:3, 8:11] = _VENTRICLE_LH
    # WM slab + cortex ribbon along z, shifted by shuffle_z to induce mismatch.
    z0 = 4 + shuffle_z
    seg[:, 10:15, z0 : z0 + 5] = _WM_LH
    seg[:, 10:15, z0 + 5 : z0 + 7] = _CORTEX_LH
    # A single DK voxel so thickness has something to reference, even though
    # these tests don't exercise the DK-label codepath directly.
    seg[10, 10, 10] = _DK_BANKSSTS_LH
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(seg, affine=np.eye(4)), str(path))


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ref_path", "cor_path", "corruption_type", "severity"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_thickness_csv(path: Path, seg_paths: list[Path], means: list[float]) -> None:
    """Write a minimal cortical_thickness.csv with mean + 2 region columns.

    The region columns are non-mean_thickness columns ending in ``_thickness``
    so the 04 module picks them up as regions for the shift computation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scan_path",
        "mean_thickness",
        "ctx-lh-bankssts_thickness",
        "ctx-rh-bankssts_thickness",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for seg_path, mean_val in zip(seg_paths, means, strict=True):
            writer.writerow(
                {
                    "scan_path": str(seg_path.resolve()),
                    "mean_thickness": mean_val,
                    "ctx-lh-bankssts_thickness": mean_val + 0.1,
                    "ctx-rh-bankssts_thickness": mean_val + 0.2,
                }
            )


def _build_env(
    tmp_path: Path,
    *,
    shuffle_z: int,
    ref_mean: float = 3.0,
    cor_mean: float = 3.0,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Lay down a full test env: synthseg dir, thickness CSV, manifest, output path.

    Returns (ref_nifti, cor_nifti, synthseg_dir, thickness_file, manifest_file,
    output_file) — ref_nifti/cor_nifti are the *input scan* paths that go into
    the corruption manifest (stems are parsed to locate the seg files).
    """
    synthseg_dir = tmp_path / "synthseg"
    ref_scan = tmp_path / "scans" / "scan_ref.nii.gz"
    cor_scan = tmp_path / "scans" / "scan_ref_cor.nii.gz"
    ref_scan.parent.mkdir(parents=True, exist_ok=True)
    # Touch input scan files (not actually read — seg discovery keys off stem).
    ref_scan.write_bytes(b"")
    cor_scan.write_bytes(b"")

    ref_seg = synthseg_dir / "scan_ref_synthseg.nii.gz"
    cor_seg = synthseg_dir / "scan_ref_cor_synthseg.nii.gz"
    _write_seg(ref_seg, shuffle_z=0)
    _write_seg(cor_seg, shuffle_z=shuffle_z)

    thickness_file = tmp_path / "cortical_thickness.csv"
    _write_thickness_csv(thickness_file, [ref_seg, cor_seg], [ref_mean, cor_mean])

    manifest_file = tmp_path / "corruption_manifest.csv"
    _write_manifest(
        manifest_file,
        [
            {
                "ref_path": str(ref_scan),
                "cor_path": str(cor_scan),
                "corruption_type": "motion",
                "severity": "1",
            }
        ],
    )

    output_file = tmp_path / "machine_preference.csv"
    return ref_scan, cor_scan, synthseg_dir, thickness_file, manifest_file, output_file


# ──────────────────────────────────────────────
# Core computation
# ──────────────────────────────────────────────


def test_identical_segs_dice_is_one(
    mod: types.ModuleType, runner: CliRunner, tmp_path: Path
) -> None:
    """Reference and corrupted are identical segs → all Dice = 1.0, shift = 0.0."""
    ref_scan, cor_scan, synthseg_dir, thickness_file, manifest_file, output_file = (
        _build_env(tmp_path, shuffle_z=0, ref_mean=3.0, cor_mean=3.0)
    )

    result = runner.invoke(
        mod.app,
        [
            "--corruption-manifest", str(manifest_file),
            "--synthseg-dir", str(synthseg_dir),
            "--thickness-file", str(thickness_file),
            "--output-file", str(output_file),
        ],
    )
    assert result.exit_code == 0, result.output

    df = pd.read_csv(output_file)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["mean_dice"] == pytest.approx(1.0, abs=1e-6)
    for col in ["hippocampus_dice", "cortex_dice", "thalamus_dice", "brainstem_dice"]:
        assert row[col] == pytest.approx(1.0, abs=1e-6), col
    assert row["ref_mean_thickness"] == pytest.approx(3.0)
    assert row["cor_mean_thickness"] == pytest.approx(3.0)
    assert row["thickness_shift"] == pytest.approx(0.0, abs=1e-9)


def test_differing_segs_dice_below_one(
    mod: types.ModuleType, runner: CliRunner, tmp_path: Path
) -> None:
    """Shifted cortex/WM slab → mean_dice < 1, thickness_shift > 0.

    ``shuffle_z=1`` keeps the cortex slab overlapping the reference by one
    slice so Dice is strictly in (0, 1); shuffle_z ≥ 2 would null the overlap
    entirely and give Dice=0, which we explicitly don't want to test here.
    """
    ref_scan, cor_scan, synthseg_dir, thickness_file, manifest_file, output_file = (
        _build_env(tmp_path, shuffle_z=1, ref_mean=3.0, cor_mean=2.5)
    )

    result = runner.invoke(
        mod.app,
        [
            "--corruption-manifest", str(manifest_file),
            "--synthseg-dir", str(synthseg_dir),
            "--thickness-file", str(thickness_file),
            "--output-file", str(output_file),
        ],
    )
    assert result.exit_code == 0, result.output

    df = pd.read_csv(output_file)
    assert len(df) == 1
    row = df.iloc[0]
    # shuffle only affects cortex (label 3) and WM (not scored). Unshuffled
    # structures remain at Dice=1.0, cortex drops, so the mean drops.
    assert 0.0 < row["cortex_dice"] < 1.0, row["cortex_dice"]
    assert row["hippocampus_dice"] == pytest.approx(1.0)
    assert row["mean_dice"] < 1.0
    # Thickness shift = mean(|ref_region - cor_region|). Per our fixture,
    # every region value differs by exactly (ref_mean - cor_mean) = 0.5.
    assert row["thickness_shift"] == pytest.approx(0.5, abs=1e-6)


def test_missing_seg_is_skipped(
    mod: types.ModuleType, runner: CliRunner, tmp_path: Path
) -> None:
    """If a scan has no seg file under --synthseg-dir, the pair is skipped (no row)."""
    _, _, synthseg_dir, thickness_file, manifest_file, output_file = _build_env(
        tmp_path, shuffle_z=0
    )

    # Add an extra pair whose scans have no corresponding *_synthseg.nii.gz.
    ghost_ref = tmp_path / "scans" / "ghost_ref.nii.gz"
    ghost_cor = tmp_path / "scans" / "ghost_ref_cor.nii.gz"
    ghost_ref.write_bytes(b"")
    ghost_cor.write_bytes(b"")
    with manifest_file.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ref_path", "cor_path", "corruption_type", "severity"],
        )
        writer.writerow(
            {
                "ref_path": str(ghost_ref),
                "cor_path": str(ghost_cor),
                "corruption_type": "ghosting",
                "severity": "2",
            }
        )

    result = runner.invoke(
        mod.app,
        [
            "--corruption-manifest", str(manifest_file),
            "--synthseg-dir", str(synthseg_dir),
            "--thickness-file", str(thickness_file),
            "--output-file", str(output_file),
        ],
    )
    assert result.exit_code == 0, result.output

    df = pd.read_csv(output_file)
    # Only the valid pair is recorded; the ghost pair is silently skipped.
    assert len(df) == 1
    assert df.iloc[0]["corruption_type"] == "motion"


def test_resume_idempotency(
    mod: types.ModuleType, runner: CliRunner, tmp_path: Path
) -> None:
    """Pairs already in the output CSV are not re-scored on a second run."""
    ref_scan, cor_scan, synthseg_dir, thickness_file, manifest_file, output_file = (
        _build_env(tmp_path, shuffle_z=0, ref_mean=3.0, cor_mean=3.0)
    )

    # First run.
    result = runner.invoke(
        mod.app,
        [
            "--corruption-manifest", str(manifest_file),
            "--synthseg-dir", str(synthseg_dir),
            "--thickness-file", str(thickness_file),
            "--output-file", str(output_file),
        ],
    )
    assert result.exit_code == 0, result.output
    first = pd.read_csv(output_file)
    assert len(first) == 1
    first_mean = first.iloc[0]["mean_dice"]

    # Second run with same args — no new rows.
    result = runner.invoke(
        mod.app,
        [
            "--corruption-manifest", str(manifest_file),
            "--synthseg-dir", str(synthseg_dir),
            "--thickness-file", str(thickness_file),
            "--output-file", str(output_file),
        ],
    )
    assert result.exit_code == 0, result.output
    second = pd.read_csv(output_file)
    assert len(second) == 1, "resume should not duplicate existing rows"
    assert second.iloc[0]["mean_dice"] == pytest.approx(first_mean)

    # Add a new pair → only that one is processed.
    new_ref = tmp_path / "scans" / "scan2.nii.gz"
    new_cor = tmp_path / "scans" / "scan2_cor.nii.gz"
    new_ref.write_bytes(b"")
    new_cor.write_bytes(b"")
    new_ref_seg = synthseg_dir / "scan2_synthseg.nii.gz"
    new_cor_seg = synthseg_dir / "scan2_cor_synthseg.nii.gz"
    _write_seg(new_ref_seg, shuffle_z=0)
    _write_seg(new_cor_seg, shuffle_z=2)
    # Re-emit thickness CSV including both old and new segs.
    _write_thickness_csv(
        thickness_file,
        [
            synthseg_dir / "scan_ref_synthseg.nii.gz",
            synthseg_dir / "scan_ref_cor_synthseg.nii.gz",
            new_ref_seg,
            new_cor_seg,
        ],
        [3.0, 3.0, 3.1, 2.9],
    )
    with manifest_file.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ref_path", "cor_path", "corruption_type", "severity"],
        )
        writer.writerow(
            {
                "ref_path": str(new_ref),
                "cor_path": str(new_cor),
                "corruption_type": "spike",
                "severity": "3",
            }
        )

    result = runner.invoke(
        mod.app,
        [
            "--corruption-manifest", str(manifest_file),
            "--synthseg-dir", str(synthseg_dir),
            "--thickness-file", str(thickness_file),
            "--output-file", str(output_file),
        ],
    )
    assert result.exit_code == 0, result.output
    third = pd.read_csv(output_file)
    assert len(third) == 2
    types_in_csv = set(third["corruption_type"])
    assert types_in_csv == {"motion", "spike"}


# ──────────────────────────────────────────────
# Manifest mode — primary code path
# ──────────────────────────────────────────────


def _write_synthseg_manifest(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write a minimal synthseg manifest CSV matching 03's schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "input_path",
        "seg_path",
        "qc_path",
        "vol_path",
        "mode",
        "status",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            full = {col: row.get(col, "") for col in fieldnames}
            writer.writerow(full)


def test_manifest_mode_basic(
    mod: types.ModuleType, runner: CliRunner, tmp_path: Path
) -> None:
    """--synthseg-manifest path keys lookups by resolved input path.

    Uses the same identical-seg fixture as test_identical_segs_dice_is_one, but
    routes seg discovery through a manifest CSV instead of a directory rglob.
    ``mean_dice`` must still be 1.0.
    """
    ref_scan, cor_scan, synthseg_dir, thickness_file, manifest_file, output_file = (
        _build_env(tmp_path, shuffle_z=0, ref_mean=3.0, cor_mean=3.0)
    )

    ref_seg = synthseg_dir / "scan_ref_synthseg.nii.gz"
    cor_seg = synthseg_dir / "scan_ref_cor_synthseg.nii.gz"
    synthseg_manifest_path = tmp_path / "synthseg_manifest.csv"
    _write_synthseg_manifest(
        synthseg_manifest_path,
        [
            {
                "input_path": str(ref_scan.resolve()),
                "seg_path": str(ref_seg.resolve()),
                "mode": "freesurfer",
                "status": "ok",
            },
            {
                "input_path": str(cor_scan.resolve()),
                "seg_path": str(cor_seg.resolve()),
                "mode": "freesurfer",
                "status": "ok",
            },
        ],
    )

    result = runner.invoke(
        mod.app,
        [
            "--corruption-manifest", str(manifest_file),
            "--synthseg-manifest", str(synthseg_manifest_path),
            "--thickness-file", str(thickness_file),
            "--output-file", str(output_file),
        ],
    )
    assert result.exit_code == 0, result.output

    df = pd.read_csv(output_file)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["mean_dice"] == pytest.approx(1.0, abs=1e-6)
    assert row["thickness_shift"] == pytest.approx(0.0, abs=1e-9)


def test_manifest_mode_handles_identical_cor_and_ref_basenames(
    mod: types.ModuleType, runner: CliRunner, tmp_path: Path
) -> None:
    """Ref and cor sharing a basename (nobrainer.qc.corrupt's convention) works.

    Regression: the directory-rglob code path collapses identical basenames to
    the same stem key and raises ``duplicate seg stems``. Manifest mode keys
    on resolved input paths so the two scans remain distinguishable, which is
    the only reason the full end-to-end pipeline (02 → 03 → 04) can succeed
    on real data where cor = ``<output>/motion/severity_1/<ref_name>``.
    """
    # Layout mimics the real pipeline:
    #   scans/scan.nii.gz
    #   corruptions/motion/severity_1/scan.nii.gz   ← same basename!
    #   synthseg_ref/scan_synthseg.nii.gz
    #   synthseg_cor/scan_synthseg.nii.gz           ← same basename too!
    scans_dir = tmp_path / "scans"
    scans_dir.mkdir()
    ref_scan = scans_dir / "scan.nii.gz"
    ref_scan.write_bytes(b"")

    cor_dir = tmp_path / "corruptions" / "motion" / "severity_1"
    cor_dir.mkdir(parents=True)
    cor_scan = cor_dir / "scan.nii.gz"
    cor_scan.write_bytes(b"")

    synthseg_ref_dir = tmp_path / "synthseg_ref"
    synthseg_cor_dir = tmp_path / "synthseg_cor"
    ref_seg = synthseg_ref_dir / "scan_synthseg.nii.gz"
    cor_seg = synthseg_cor_dir / "scan_synthseg.nii.gz"
    _write_seg(ref_seg, shuffle_z=0)
    _write_seg(cor_seg, shuffle_z=2)  # distinguishable cor → Dice < 1

    # Two separate synthseg manifests (one per 03 invocation).
    ref_manifest = tmp_path / "synthseg_ref" / "synthseg_manifest.csv"
    cor_manifest = tmp_path / "synthseg_cor" / "synthseg_manifest.csv"
    _write_synthseg_manifest(
        ref_manifest,
        [{
            "input_path": str(ref_scan.resolve()),
            "seg_path": str(ref_seg.resolve()),
            "mode": "freesurfer",
            "status": "ok",
        }],
    )
    _write_synthseg_manifest(
        cor_manifest,
        [{
            "input_path": str(cor_scan.resolve()),
            "seg_path": str(cor_seg.resolve()),
            "mode": "freesurfer",
            "status": "ok",
        }],
    )

    # Thickness table covers both segs.
    thickness_file = tmp_path / "cortical_thickness.csv"
    _write_thickness_csv(thickness_file, [ref_seg, cor_seg], [3.0, 2.7])

    # Corruption manifest pairs the identically-named ref and cor scans.
    corruption_manifest = tmp_path / "corruption_manifest.csv"
    _write_manifest(
        corruption_manifest,
        [{
            "ref_path": str(ref_scan),
            "cor_path": str(cor_scan),
            "corruption_type": "motion",
            "severity": "1",
        }],
    )

    output_file = tmp_path / "machine_preference.csv"
    result = runner.invoke(
        mod.app,
        [
            "--corruption-manifest", str(corruption_manifest),
            "--synthseg-manifest", str(ref_manifest),
            "--synthseg-manifest", str(cor_manifest),
            "--thickness-file", str(thickness_file),
            "--output-file", str(output_file),
        ],
    )
    assert result.exit_code == 0, result.output

    df = pd.read_csv(output_file)
    assert len(df) == 1
    row = df.iloc[0]
    # shuffle_z=2 makes ref and cor segs different → Dice < 1. The key
    # assertion is that we got a non-trivial number at all (i.e. 04 found
    # two distinct segs rather than colliding on basename).
    assert 0.0 < row["mean_dice"] < 1.0, row["mean_dice"]
    assert row["ref_mean_thickness"] == pytest.approx(3.0)
    assert row["cor_mean_thickness"] == pytest.approx(2.7)


# ──────────────────────────────────────────────
# Per-structure long-format CSV
# ──────────────────────────────────────────────


def test_per_structure_long_csv_schema(
    mod: types.ModuleType, runner: CliRunner, tmp_path: Path
) -> None:
    """`--per-structure-output` writes one row per (pair, label_id) with names."""
    ref_scan, cor_scan, synthseg_dir, thickness_file, manifest_file, output_file = (
        _build_env(tmp_path, shuffle_z=2, ref_mean=3.0, cor_mean=2.7)
    )
    per_structure_file = tmp_path / "per_structure_dice.csv"
    # Write a tiny LUT covering a couple of the structures we wrote in _write_seg.
    lut_file = tmp_path / "FreeSurferColorLUT.txt"
    lut_file.write_text(
        "# id name R G B A\n"
        "2 Left-Cerebral-White-Matter 245 245 245 0\n"
        "3 Left-Cerebral-Cortex 205 62 78 0\n"
        "17 Left-Hippocampus 220 216 20 0\n"
        "1001 ctx-lh-bankssts 25 100 40 0\n"
    )

    result = runner.invoke(
        mod.app,
        [
            "--corruption-manifest", str(manifest_file),
            "--synthseg-dir", str(synthseg_dir),
            "--thickness-file", str(thickness_file),
            "--output-file", str(output_file),
            "--per-structure-output", str(per_structure_file),
            "--label-name-source", str(lut_file),
        ],
    )
    if result.exit_code != 0 and result.exception is not None:
        import traceback
        traceback.print_exception(
            type(result.exception), result.exception, result.exception.__traceback__
        )
    assert result.exit_code == 0, result.output

    assert per_structure_file.is_file()
    long_df = pd.read_csv(per_structure_file)

    expected_cols = {"ref_path", "cor_path", "corruption_type", "severity",
                     "label_id", "label_name", "dice"}
    assert expected_cols.issubset(long_df.columns)
    # Background label 0 must be excluded.
    assert (long_df["label_id"] != 0).all()
    # All per-pair rows share the same (ref, cor); we wrote 1 pair → 1 group.
    assert long_df["ref_path"].nunique() == 1
    assert long_df["cor_path"].nunique() == 1
    # Every label seen in the seg fixture should be present in the long CSV.
    expected_labels = {2, 3, 4, 8, 10, 11, 12, 16, 17, 1001}
    assert expected_labels.issubset(set(long_df["label_id"].astype(int)))
    # LUT-resolved names where the LUT had them.
    assert (
        long_df.loc[long_df["label_id"] == 17, "label_name"].iloc[0]
        == "Left-Hippocampus"
    )
    # Fallback name for labels NOT in the LUT (e.g. 4, 8, 10, 11, 12, 16).
    name_for_4 = long_df.loc[long_df["label_id"] == 4, "label_name"].iloc[0]
    assert name_for_4 == "label_4"
    # Dice values are sane: all in [0, 1] or NaN, identical-mask cubes give 1.0.
    dice_vals = long_df["dice"].dropna()
    assert ((dice_vals >= 0.0) & (dice_vals <= 1.0)).all()


def test_per_structure_resume_idempotency(
    mod: types.ModuleType, runner: CliRunner, tmp_path: Path
) -> None:
    """Running 04 twice with --per-structure-output produces no duplicate rows."""
    ref_scan, cor_scan, synthseg_dir, thickness_file, manifest_file, output_file = (
        _build_env(tmp_path, shuffle_z=2, ref_mean=3.0, cor_mean=2.7)
    )
    per_structure_file = tmp_path / "per_structure_dice.csv"

    args = [
        "--corruption-manifest", str(manifest_file),
        "--synthseg-dir", str(synthseg_dir),
        "--thickness-file", str(thickness_file),
        "--output-file", str(output_file),
        "--per-structure-output", str(per_structure_file),
    ]

    first = runner.invoke(mod.app, args)
    assert first.exit_code == 0, first.output
    df_first = pd.read_csv(per_structure_file)
    n_first = len(df_first)
    assert n_first > 0

    # Second invocation: machine_preference + per_structure rows already on disk.
    second = runner.invoke(mod.app, args)
    assert second.exit_code == 0, second.output
    df_second = pd.read_csv(per_structure_file)
    # Resume must produce zero new rows for the same pair.
    assert len(df_second) == n_first

    # And the per-pair output should still be exactly 1 row.
    df_pref = pd.read_csv(output_file)
    assert len(df_pref) == 1
