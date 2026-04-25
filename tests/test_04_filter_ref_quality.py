"""Unit tests for code/04_filter_ref_quality.py.

Synthetic seg NIfTIs drive both a "pass" case (many labels, healthy
brain fraction) and two failure modes: degenerate (too few labels) and
missing-seg-file. The idempotent rerun case is covered by relying on
pandas read_csv round-trips.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import types
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE_PATH = _REPO_ROOT / "code" / "04_filter_ref_quality.py"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("filter_ref_quality", _MODULE_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {_MODULE_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["filter_ref_quality"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate_mod() -> types.ModuleType:
    return _load_module()


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_seg(path: Path, labels: list[int], shape: tuple[int, int, int] = (8, 8, 8)) -> Path:
    """Write a synthetic segmentation NIfTI with the given label set.

    The labels are tiled across the volume in equal-ish blocks so that
    every listed label appears at least N>=100 voxels (the threshold
    the diagnostic script uses).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros(shape, dtype=np.int32)
    if labels:
        # Fill 75% of voxels with a round-robin label pattern; leave the
        # rest as background so brain_fraction ≈ 0.75.
        total = arr.size
        brain_voxels = int(total * 0.75)
        per_label = max(1, brain_voxels // len(labels))
        flat = arr.flatten()
        idx = 0
        for label in labels:
            flat[idx : idx + per_label] = label
            idx += per_label
        arr = flat.reshape(shape)
    nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), str(path))
    return path


def _write_synthseg_manifest(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["input_path", "seg_path", "qc_path", "vol_path", "mode", "status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})
    return path


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────


def test_compute_quality_returns_label_count_and_brain_fraction(
    gate_mod: types.ModuleType, tmp_path: Path
) -> None:
    """A seg with 25 distinct labels reports n=25 and brain_frac ~0.75."""
    labels = list(range(1, 26))  # 25 unique non-zero labels
    seg = _write_seg(tmp_path / "seg.nii.gz", labels, shape=(20, 20, 20))

    q = gate_mod.compute_quality(seg)
    assert q.n_unique_labels == 25
    assert q.brain_fraction == pytest.approx(0.75, abs=0.01)
    assert q.seg_shape == (20, 20, 20)


def test_evaluate_gate_enumerates_all_failed_criteria(
    gate_mod: types.ModuleType,
) -> None:
    """Both n_labels AND brain_frac thresholds fail — reject_reason lists both."""
    q = gate_mod.SegQuality(n_unique_labels=5, brain_fraction=0.001, seg_shape=(8, 8, 8))
    passed, reason = gate_mod.evaluate_gate(q, min_n_labels=20, min_brain_fraction=0.01)
    assert passed is False
    assert "n_unique_labels=5<20" in reason
    assert "brain_fraction=" in reason


def test_gate_manifest_enriches_every_row_and_keeps_upstream_columns(
    gate_mod: types.ModuleType, tmp_path: Path
) -> None:
    """Healthy seg passes, degenerate seg fails, missing seg_path fails — all 3 rows kept."""
    healthy_seg = _write_seg(tmp_path / "good.nii.gz", list(range(1, 26)), shape=(16, 16, 16))
    degenerate_seg = _write_seg(tmp_path / "bad.nii.gz", [1, 2, 3], shape=(16, 16, 16))
    missing_seg = tmp_path / "does_not_exist.nii.gz"  # intentionally not written

    manifest_in = _write_synthseg_manifest(
        tmp_path / "synthseg_manifest.csv",
        [
            {"input_path": "a.nii.gz", "seg_path": str(healthy_seg), "mode": "freesurfer", "status": "ok"},
            {"input_path": "b.nii.gz", "seg_path": str(degenerate_seg), "mode": "freesurfer", "status": "ok"},
            {"input_path": "c.nii.gz", "seg_path": str(missing_seg), "mode": "freesurfer", "status": "ok"},
        ],
    )

    df = gate_mod.gate_manifest(manifest_in, min_n_labels=20, min_brain_fraction=0.01)
    assert len(df) == 3
    # Upstream columns preserved.
    for col in ("input_path", "seg_path", "mode", "status"):
        assert col in df.columns
    # Enrichment columns added.
    for col in gate_mod.ENRICHMENT_COLUMNS:
        assert col in df.columns

    rows = {row["input_path"]: row for _, row in df.iterrows()}
    assert rows["a.nii.gz"]["passed_gate"] is True or rows["a.nii.gz"]["passed_gate"] == True
    assert rows["a.nii.gz"]["n_unique_labels"] == 25
    assert rows["b.nii.gz"]["passed_gate"] is False or rows["b.nii.gz"]["passed_gate"] == False
    assert "n_unique_labels=3<20" in rows["b.nii.gz"]["reject_reason"]
    assert rows["c.nii.gz"]["passed_gate"] is False or rows["c.nii.gz"]["passed_gate"] == False
    assert "seg_path missing" in rows["c.nii.gz"]["reject_reason"]


def test_cli_writes_gated_manifest_and_preserves_extra_columns(
    gate_mod: types.ModuleType, runner: CliRunner, tmp_path: Path
) -> None:
    """End-to-end: CLI reads a manifest, writes a gated manifest that downstream can consume."""
    seg_path = _write_seg(tmp_path / "seg.nii.gz", list(range(1, 26)), shape=(16, 16, 16))

    # Extra upstream column (dataset_tag) that downstream analysis needs to keep.
    manifest_path = tmp_path / "synthseg_manifest.csv"
    manifest_path.write_text(
        "input_path,seg_path,mode,status,dataset_tag\n"
        f"x.nii.gz,{seg_path},freesurfer,ok,fastmri\n"
    )
    gated_path = tmp_path / "gated.csv"

    result = runner.invoke(
        gate_mod.app,
        [
            "--synthseg-manifest", str(manifest_path),
            "--gated-manifest", str(gated_path),
            "--min-n-labels", "20",
            "--min-brain-fraction", "0.01",
        ],
    )
    assert result.exit_code == 0, result.output
    assert gated_path.exists()
    df = pd.read_csv(gated_path)
    assert len(df) == 1
    # Upstream `dataset_tag` column is preserved.
    assert "dataset_tag" in df.columns
    assert df["dataset_tag"].iloc[0] == "fastmri"
    # Enrichment present and sensible.
    assert df["n_unique_labels"].iloc[0] == 25
    assert bool(df["passed_gate"].iloc[0]) is True
