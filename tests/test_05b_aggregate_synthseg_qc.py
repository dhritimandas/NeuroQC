"""Unit tests for code/05b_aggregate_synthseg_qc.py.

Synthetic fixtures only — zero-byte NIfTI placeholders, CSV-only qc sidecars.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE_PATH = _REPO_ROOT / "code" / "05b_aggregate_synthseg_qc.py"


def _load_module() -> types.ModuleType:
    """Import 05b despite its digit-prefixed filename.

    ``sys.modules`` registration is required for ``@dataclass(frozen=True)``
    to resolve forward references under ``from __future__ import annotations``.
    """
    spec = importlib.util.spec_from_file_location("agg5b_mod", _MODULE_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {_MODULE_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["agg5b_mod"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> types.ModuleType:
    return _load_module()


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ──────────────────────────────────────────────
# Fixture helpers
# ──────────────────────────────────────────────


_SOURCE_HEADER = (
    "subject,general white matter,general grey matter,general csf,"
    "cerebellum,brainstem,thalamus,putamen+pallidum,hippocampus+amygdala"
)


def _touch_scan(base_dir: Path, dataset_tag: str, name: str) -> Path:
    """Zero-byte scan NIfTI under ``base_dir/<dataset_tag>/name.nii.gz``."""
    scan = base_dir / dataset_tag / name
    scan.parent.mkdir(parents=True, exist_ok=True)
    scan.write_bytes(b"")
    return scan


def _write_qc_sidecar(
    path: Path, subject: str, values: tuple[float, ...]
) -> None:
    """Write a synthetic SynthSeg-format qc.csv with 2 rows.

    ``values`` must be 8 floats in the same order as the header (WM, GM, CSF,
    cerebellum, brainstem, thalamus, putamen+pallidum, hippocampus+amygdala).
    """
    assert len(values) == 8, "qc sidecar has 8 tissue classes"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        handle.write(_SOURCE_HEADER + "\n")
        handle.write(f"{subject}," + ",".join(f"{v:.4f}" for v in values) + "\n")


def _write_synthseg_manifest(
    path: Path, rows: list[dict[str, object]]
) -> None:
    """Write a minimal synthseg manifest (input_path, qc_path, status, ...)."""
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


def _write_cor_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a minimal corruption manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ref_path",
        "cor_path",
        "corruption_type",
        "corruption_domain",
        "severity",
        "seed",
        "transform_params",
        "dataset_tag",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in fieldnames})


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────


def test_qc_sidecar_parse_and_rename(mod: types.ModuleType, tmp_path: Path) -> None:
    """load_qc_sidecar returns a dict keyed by CSV-safe LOSS_COLUMNS names."""
    qc_path = tmp_path / "subj_qc.csv"
    values = (0.0475, 0.1319, 0.0291, 0.0026, 0.0327, 0.0081, 0.0032, 0.0117)
    _write_qc_sidecar(qc_path, subject="subj", values=values)

    out = mod.load_qc_sidecar(qc_path)
    # Keys are exactly the renamed LOSS_COLUMNS.
    assert set(out.keys()) == set(mod.LOSS_COLUMNS)
    # Values preserved verbatim.
    assert out["general_white_matter_loss"] == pytest.approx(0.0475)
    assert out["general_grey_matter_loss"] == pytest.approx(0.1319)
    assert out["putamen_pallidum_loss"] == pytest.approx(0.0032)
    assert out["hippocampus_amygdala_loss"] == pytest.approx(0.0117)


def test_aggregation_without_cor_manifest(
    mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Two synthseg manifests + qc sidecars → 2 rows, all is_reference=True."""
    scan_a = _touch_scan(tmp_path, "ixi", "IXI001.nii.gz")
    scan_b = _touch_scan(tmp_path, "fastmri", "fm001.nii.gz")

    qc_a = tmp_path / "qc_a.csv"
    qc_b = tmp_path / "qc_b.csv"
    _write_qc_sidecar(qc_a, subject="IXI001", values=(0.05, 0.10, 0.02, 0.01, 0.03, 0.01, 0.01, 0.01))
    _write_qc_sidecar(qc_b, subject="fm001", values=(0.08, 0.20, 0.04, 0.02, 0.05, 0.02, 0.02, 0.02))

    manifest_a = tmp_path / "synthseg_ixi.csv"
    manifest_b = tmp_path / "synthseg_fastmri.csv"
    _write_synthseg_manifest(
        manifest_a,
        [{"input_path": str(scan_a), "qc_path": str(qc_a), "mode": "freesurfer", "status": "ok"}],
    )
    _write_synthseg_manifest(
        manifest_b,
        [{"input_path": str(scan_b), "qc_path": str(qc_b), "mode": "freesurfer", "status": "ok"}],
    )

    output_file = tmp_path / "synthseg_qc_features.csv"
    result = runner.invoke(
        mod.app,
        [
            "--synthseg-manifest", str(manifest_a),
            "--synthseg-manifest", str(manifest_b),
            "--output-file", str(output_file),
        ],
    )
    assert result.exit_code == 0, result.output

    df = pd.read_csv(output_file)
    assert len(df) == 2
    assert df[mod.IS_REFERENCE_COLUMN].all()
    assert set(df[mod.TYPE_COLUMN]) == {mod._NONE_TYPE}
    assert set(df[mod.SEVERITY_COLUMN]) == {0}
    # Path-based dataset_tag inference.
    ixi_row = df[df[mod.SCAN_COLUMN] == str(scan_a.resolve())].iloc[0]
    fm_row = df[df[mod.SCAN_COLUMN] == str(scan_b.resolve())].iloc[0]
    assert ixi_row[mod.DATASET_TAG_COLUMN] == "ixi"
    assert fm_row[mod.DATASET_TAG_COLUMN] == "fastmri"
    # Loss values round-trip.
    assert ixi_row["general_white_matter_loss"] == pytest.approx(0.05)
    assert fm_row["general_grey_matter_loss"] == pytest.approx(0.20)


def test_aggregation_with_cor_manifest(
    mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Scans in cor_path of cor-manifest get is_reference=False + corruption tags."""
    ref_scan = _touch_scan(tmp_path, "fastmri", "ref.nii.gz")
    cor_a = _touch_scan(tmp_path, "fastmri", "cor_motion.nii.gz")
    cor_b = _touch_scan(tmp_path, "fastmri", "cor_spike.nii.gz")

    qc_ref = tmp_path / "qc_ref.csv"
    qc_cor_a = tmp_path / "qc_cor_a.csv"
    qc_cor_b = tmp_path / "qc_cor_b.csv"
    _write_qc_sidecar(qc_ref, "ref", (0.04, 0.10, 0.02, 0.01, 0.03, 0.01, 0.01, 0.01))
    _write_qc_sidecar(qc_cor_a, "cor_a", (0.03, 0.08, 0.03, 0.01, 0.03, 0.01, 0.01, 0.005))
    _write_qc_sidecar(qc_cor_b, "cor_b", (0.06, 0.15, 0.04, 0.02, 0.04, 0.02, 0.01, 0.01))

    synthseg_manifest = tmp_path / "synthseg.csv"
    _write_synthseg_manifest(
        synthseg_manifest,
        [
            {"input_path": str(ref_scan), "qc_path": str(qc_ref), "mode": "freesurfer", "status": "ok"},
            {"input_path": str(cor_a), "qc_path": str(qc_cor_a), "mode": "freesurfer", "status": "ok"},
            {"input_path": str(cor_b), "qc_path": str(qc_cor_b), "mode": "freesurfer", "status": "ok"},
        ],
    )

    cor_manifest = tmp_path / "corruption_manifest.csv"
    _write_cor_manifest(
        cor_manifest,
        [
            {
                "ref_path": str(ref_scan),
                "cor_path": str(cor_a),
                "corruption_type": "motion",
                "corruption_domain": "image",
                "severity": 1,
                "dataset_tag": "fastmri",
            },
            {
                "ref_path": str(ref_scan),
                "cor_path": str(cor_b),
                "corruption_type": "spike",
                "corruption_domain": "image",
                "severity": 3,
                "dataset_tag": "fastmri",
            },
        ],
    )

    output_file = tmp_path / "synthseg_qc_features.csv"
    result = runner.invoke(
        mod.app,
        [
            "--synthseg-manifest", str(synthseg_manifest),
            "--cor-manifest", str(cor_manifest),
            "--output-file", str(output_file),
        ],
    )
    assert result.exit_code == 0, result.output

    df = pd.read_csv(output_file)
    by_path = df.set_index(mod.SCAN_COLUMN)
    ref_row = by_path.loc[str(ref_scan.resolve())]
    cor_a_row = by_path.loc[str(cor_a.resolve())]
    cor_b_row = by_path.loc[str(cor_b.resolve())]

    assert bool(ref_row[mod.IS_REFERENCE_COLUMN]) is True
    assert ref_row[mod.TYPE_COLUMN] == mod._NONE_TYPE
    assert int(ref_row[mod.SEVERITY_COLUMN]) == 0

    assert bool(cor_a_row[mod.IS_REFERENCE_COLUMN]) is False
    assert cor_a_row[mod.TYPE_COLUMN] == "motion"
    assert int(cor_a_row[mod.SEVERITY_COLUMN]) == 1
    assert cor_a_row[mod.DATASET_TAG_COLUMN] == "fastmri"

    assert bool(cor_b_row[mod.IS_REFERENCE_COLUMN]) is False
    assert cor_b_row[mod.TYPE_COLUMN] == "spike"
    assert int(cor_b_row[mod.SEVERITY_COLUMN]) == 3


def test_resume_idempotency(
    mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Pre-existing output with scan A → rerun over {A, B} adds only B."""
    scan_a = _touch_scan(tmp_path, "fastmri", "a.nii.gz")
    scan_b = _touch_scan(tmp_path, "fastmri", "b.nii.gz")
    qc_a = tmp_path / "qc_a.csv"
    qc_b = tmp_path / "qc_b.csv"
    _write_qc_sidecar(qc_a, "a", (0.05, 0.10, 0.02, 0.01, 0.03, 0.01, 0.01, 0.01))
    _write_qc_sidecar(qc_b, "b", (0.06, 0.12, 0.03, 0.02, 0.04, 0.01, 0.01, 0.01))

    # First manifest sees A only.
    manifest_a = tmp_path / "synthseg_a.csv"
    _write_synthseg_manifest(
        manifest_a,
        [{"input_path": str(scan_a), "qc_path": str(qc_a), "mode": "freesurfer", "status": "ok"}],
    )

    output_file = tmp_path / "synthseg_qc_features.csv"
    result = runner.invoke(
        mod.app,
        [
            "--synthseg-manifest", str(manifest_a),
            "--output-file", str(output_file),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(pd.read_csv(output_file)) == 1

    # Second manifest sees A and B; resume should add only B.
    manifest_ab = tmp_path / "synthseg_ab.csv"
    _write_synthseg_manifest(
        manifest_ab,
        [
            {"input_path": str(scan_a), "qc_path": str(qc_a), "mode": "freesurfer", "status": "ok"},
            {"input_path": str(scan_b), "qc_path": str(qc_b), "mode": "freesurfer", "status": "ok"},
        ],
    )
    result = runner.invoke(
        mod.app,
        [
            "--synthseg-manifest", str(manifest_ab),
            "--output-file", str(output_file),
        ],
    )
    assert result.exit_code == 0, result.output
    df = pd.read_csv(output_file)
    assert len(df) == 2
    assert set(df[mod.SCAN_COLUMN]) == {
        str(scan_a.resolve()),
        str(scan_b.resolve()),
    }
