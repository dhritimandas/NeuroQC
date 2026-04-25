"""Unit tests for code/05_extract_iqms.py.

Synthetic data only. The heavy ``extract_iqms`` call is monkeypatched on the
loaded module so tests stay fast; seg files are zero-byte touches that pass
existence checks but are never actually loaded.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE_PATH = _REPO_ROOT / "code" / "05_extract_iqms.py"


def _load_module() -> types.ModuleType:
    """Import 05_extract_iqms.py despite its digit-prefixed filename.

    ``sys.modules`` registration is required for ``@dataclass(frozen=True)``
    to resolve forward references under ``from __future__ import annotations``.
    """
    spec = importlib.util.spec_from_file_location("extract_iqms_mod", _MODULE_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {_MODULE_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["extract_iqms_mod"] = module
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


def _touch_scan_and_seg(base_dir: Path, dataset_tag: str, scan_name: str) -> tuple[Path, Path]:
    """Create a zero-byte scan NIfTI + seg NIfTI under a dataset-tagged subtree.

    The path includes ``/<dataset_tag>/`` so the module's path-based tag
    inference picks it up when a ref manifest lacks a dataset_tag column.
    """
    scan_path = base_dir / dataset_tag / scan_name
    seg_path = base_dir / "synthseg" / f"{Path(scan_name).stem.replace('.nii', '')}_synthseg.nii.gz"
    scan_path.parent.mkdir(parents=True, exist_ok=True)
    seg_path.parent.mkdir(parents=True, exist_ok=True)
    scan_path.write_bytes(b"")
    seg_path.write_bytes(b"")
    return scan_path, seg_path


def _write_ixi_ref_manifest(path: Path, scan_paths: list[Path]) -> None:
    """Write an IXI-style manifest with the 'filepath' column (no dataset_tag)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filepath", "subject_id"])
        writer.writeheader()
        for i, scan in enumerate(scan_paths):
            writer.writerow({"filepath": str(scan), "subject_id": f"sub-{i:03d}"})


def _write_fastmri_ref_manifest(
    path: Path, scan_paths: list[Path], dataset_tag: str = "fastmri"
) -> None:
    """Write a FastMRI-style manifest with 'ref_path' + 'dataset_tag' columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["ref_path", "subject_id", "dataset_tag"]
        )
        writer.writeheader()
        for i, scan in enumerate(scan_paths):
            writer.writerow(
                {
                    "ref_path": str(scan),
                    "subject_id": f"sub-{i:03d}",
                    "dataset_tag": dataset_tag,
                }
            )


def _write_cor_manifest(
    path: Path, rows: list[dict[str, object]]
) -> None:
    """Write a corruption manifest with the columns consumed by load_cor_manifest."""
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


def _write_synthseg_manifest(
    path: Path, pairs: list[tuple[Path, Path]]
) -> None:
    """Write a synthseg manifest with input_path + seg_path columns."""
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
        for input_path, seg_path in pairs:
            writer.writerow(
                {
                    "input_path": str(input_path.resolve()),
                    "seg_path": str(seg_path.resolve()),
                    "qc_path": "",
                    "vol_path": "",
                    "mode": "freesurfer",
                    "status": "ok",
                }
            )


def _fake_extract_iqms(*_args: Any, **_kwargs: Any) -> dict[str, float]:
    """Deterministic fake that returns finite values for every key."""
    return {"snr": 10.0, "cnr": 2.0, "efc": 0.5, "fber": 20.0, "cjv": 0.4}


# ──────────────────────────────────────────────
# Manifest unification
# ──────────────────────────────────────────────


def test_manifest_unification_ixi(mod: types.ModuleType, tmp_path: Path) -> None:
    """IXI-style (filepath, no dataset_tag) → is_reference=True, tag inferred=ixi."""
    scan_a, _ = _touch_scan_and_seg(tmp_path, "ixi", "IXI001.nii.gz")
    scan_b, _ = _touch_scan_and_seg(tmp_path, "ixi", "IXI002.nii.gz")
    ref_manifest = tmp_path / "ref_ixi.csv"
    _write_ixi_ref_manifest(ref_manifest, [scan_a, scan_b])

    df = mod.load_ref_manifest(ref_manifest)
    assert len(df) == 2
    assert set(df.columns) == {
        mod.SCAN_COLUMN,
        mod.IS_REFERENCE_COLUMN,
        mod.TYPE_COLUMN,
        mod.SEVERITY_COLUMN,
        mod.DATASET_TAG_COLUMN,
    }
    assert df[mod.IS_REFERENCE_COLUMN].all()
    assert (df[mod.TYPE_COLUMN] == mod._NONE_TYPE).all()
    assert (df[mod.SEVERITY_COLUMN] == mod._REF_SEVERITY).all()
    # Path-inference should classify /ixi/ paths as 'ixi'.
    assert (df[mod.DATASET_TAG_COLUMN] == "ixi").all()
    # scan_path is resolved-absolute.
    for p in df[mod.SCAN_COLUMN]:
        assert Path(p).is_absolute()


def test_manifest_unification_fastmri(mod: types.ModuleType, tmp_path: Path) -> None:
    """FastMRI-style (ref_path + dataset_tag) preserves the tag verbatim."""
    scan_a, _ = _touch_scan_and_seg(tmp_path, "fastmri", "fm001.nii.gz")
    ref_manifest = tmp_path / "ref_fastmri.csv"
    _write_fastmri_ref_manifest(ref_manifest, [scan_a], dataset_tag="fastmri")

    df = mod.load_ref_manifest(ref_manifest)
    assert len(df) == 1
    assert df.iloc[0][mod.DATASET_TAG_COLUMN] == "fastmri"
    assert df.iloc[0][mod.IS_REFERENCE_COLUMN] is True or df.iloc[0][mod.IS_REFERENCE_COLUMN] == True
    assert df.iloc[0][mod.SEVERITY_COLUMN] == 0
    assert df.iloc[0][mod.TYPE_COLUMN] == mod._NONE_TYPE


# ──────────────────────────────────────────────
# Seg join
# ──────────────────────────────────────────────


def test_seg_path_join_drops_unmatched(mod: types.ModuleType, tmp_path: Path) -> None:
    """3 scans in the unified frame, synthseg manifest covers only 2 → 1 drop."""
    scan_a, seg_a = _touch_scan_and_seg(tmp_path, "fastmri", "a.nii.gz")
    scan_b, seg_b = _touch_scan_and_seg(tmp_path, "fastmri", "b.nii.gz")
    scan_c, _ = _touch_scan_and_seg(tmp_path, "fastmri", "c.nii.gz")

    # synthseg manifest only covers a and b.
    synthseg_manifest = tmp_path / "synthseg_manifest.csv"
    _write_synthseg_manifest(synthseg_manifest, [(scan_a, seg_a), (scan_b, seg_b)])
    seg_map = mod.build_seg_map([synthseg_manifest])

    unified = pd.DataFrame(
        {
            mod.SCAN_COLUMN: [str(scan_a.resolve()), str(scan_b.resolve()), str(scan_c.resolve())],
            mod.IS_REFERENCE_COLUMN: [True, False, True],
            mod.TYPE_COLUMN: [mod._NONE_TYPE, "motion", mod._NONE_TYPE],
            mod.SEVERITY_COLUMN: [0, 1, 0],
            mod.DATASET_TAG_COLUMN: ["fastmri", "fastmri", "fastmri"],
        }
    )
    attached = mod.attach_seg_paths(unified, seg_map)
    assert len(attached) == 2
    assert str(scan_c.resolve()) not in attached[mod.SCAN_COLUMN].tolist()


# ──────────────────────────────────────────────
# Resume
# ──────────────────────────────────────────────


def test_resume_idempotency(
    mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-existing output CSV with scan A → rerun on {A, B} adds only B."""
    scan_a, seg_a = _touch_scan_and_seg(tmp_path, "fastmri", "a.nii.gz")
    scan_b, seg_b = _touch_scan_and_seg(tmp_path, "fastmri", "b.nii.gz")

    ref_manifest = tmp_path / "ref.csv"
    _write_fastmri_ref_manifest(ref_manifest, [scan_a, scan_b])

    cor_manifest = tmp_path / "cor.csv"
    _write_cor_manifest(cor_manifest, [])  # header only — unified frame has only refs.

    synthseg_manifest = tmp_path / "synthseg.csv"
    _write_synthseg_manifest(synthseg_manifest, [(scan_a, seg_a), (scan_b, seg_b)])

    output_file = tmp_path / "iqm.csv"
    monkeypatch.setattr(mod, "extract_iqms", _fake_extract_iqms)

    # First run — processes both.
    result = runner.invoke(
        mod.app,
        [
            "--ref-manifest", str(ref_manifest),
            "--cor-manifest", str(cor_manifest),
            "--synthseg-manifest", str(synthseg_manifest),
            "--output-file", str(output_file),
        ],
    )
    assert result.exit_code == 0, result.output
    first = pd.read_csv(output_file)
    assert len(first) == 2

    # Second run — no new rows.
    result = runner.invoke(
        mod.app,
        [
            "--ref-manifest", str(ref_manifest),
            "--cor-manifest", str(cor_manifest),
            "--synthseg-manifest", str(synthseg_manifest),
            "--output-file", str(output_file),
        ],
    )
    assert result.exit_code == 0, result.output
    second = pd.read_csv(output_file)
    assert len(second) == 2


# ──────────────────────────────────────────────
# extract_iqms failure → NaN row, loop continues
# ──────────────────────────────────────────────


def test_extract_iqms_failure_yields_nan_row(
    mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If extract_iqms raises on one scan, a NaN row is written and the loop continues."""
    scan_a, seg_a = _touch_scan_and_seg(tmp_path, "fastmri", "a.nii.gz")
    scan_b, seg_b = _touch_scan_and_seg(tmp_path, "fastmri", "b.nii.gz")

    ref_manifest = tmp_path / "ref.csv"
    _write_fastmri_ref_manifest(ref_manifest, [scan_a, scan_b])

    cor_manifest = tmp_path / "cor.csv"
    _write_cor_manifest(cor_manifest, [])

    synthseg_manifest = tmp_path / "synthseg.csv"
    _write_synthseg_manifest(synthseg_manifest, [(scan_a, seg_a), (scan_b, seg_b)])

    output_file = tmp_path / "iqm.csv"

    # Fake that raises on scan_a, succeeds on scan_b.
    resolved_a = str(scan_a.resolve())

    def flaky(scan_path: object, seg_path: object | None = None) -> dict[str, float]:
        if str(scan_path) == resolved_a:
            raise RuntimeError("synthetic failure")
        return _fake_extract_iqms()

    monkeypatch.setattr(mod, "extract_iqms", flaky)

    result = runner.invoke(
        mod.app,
        [
            "--ref-manifest", str(ref_manifest),
            "--cor-manifest", str(cor_manifest),
            "--synthseg-manifest", str(synthseg_manifest),
            "--output-file", str(output_file),
        ],
    )
    assert result.exit_code == 0, result.output

    df = pd.read_csv(output_file)
    assert len(df) == 2
    by_path = df.set_index("scan_path")
    row_a = by_path.loc[resolved_a]
    row_b = by_path.loc[str(scan_b.resolve())]
    for key in mod.IQM_KEYS:
        assert math.isnan(row_a[key]), f"scan_a {key} should be NaN"
        assert math.isfinite(row_b[key]), f"scan_b {key} should be finite"
