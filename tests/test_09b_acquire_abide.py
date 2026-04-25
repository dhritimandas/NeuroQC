"""Unit tests for code/09b_acquire_abide.py.

Synthetic fixtures only — every test patches mriqc_learn.datasets.load_dataset
and / or boto3.client so the suite runs without touching the network or
relying on the real ABIDE data being present.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE_PATH = _REPO_ROOT / "code" / "09b_acquire_abide.py"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("acquire_abide_mod", _MODULE_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {_MODULE_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["acquire_abide_mod"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> types.ModuleType:
    return _load_module()


# ──────────────────────────────────────────────
# Synthetic mriqc-learn dataset (matches verified shape)
# ──────────────────────────────────────────────


def _build_canned_dataset(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_x, train_y) shaped (1101, 68) + (1101, 5) per real ABIDE."""
    sites = [
        "CALTECH", "CMU", "KKI", "LEUVEN", "MAX_MUN", "NYU", "OHSU",
        "OLIN", "PITT", "SBL", "SDSU", "STANFORD", "TRINITY",
        "UCLA", "UM", "USM", "YALE",
    ]
    counts = [38, 27, 54, 64, 57, 184, 28, 36, 57, 30, 36, 40, 49, 99, 145, 101, 56]
    assert sum(counts) == 1101
    rng = np.random.RandomState(seed)
    rows: list[dict[str, Any]] = []
    sid = 50000
    for site, n in zip(sites, counts, strict=True):
        for _ in range(n):
            rows.append({
                "subject_id": sid,
                "site": site,
                # rater_3 fully covered; rater_1/2 each ~half rated.
                "rater_1": float("nan") if (sid % 2 == 0) else float(rng.choice([-1, 0, 1])),
                "rater_2": float("nan") if (sid % 3 == 0) else float(rng.choice([-1, 0, 1])),
                "rater_3": float(rng.choice([-1, 0, 1])),
            })
            sid += 1
    train_y = pd.DataFrame(rows)

    iqm_names = (
        ["cjv", "cnr", "efc", "fber"]
        + ["snr_total", "snr_csf", "snr_gm", "snr_wm",
           "snrd_total", "snrd_csf", "snrd_gm", "snrd_wm"]
        + [f"misc_iqm_{i:02d}" for i in range(56)]
    )
    assert len(iqm_names) == 68
    train_x = pd.DataFrame(rng.randn(1101, 68), columns=iqm_names)
    return train_x, train_y


@pytest.fixture
def canned_loader(mod, monkeypatch):
    """Patch ``mod._import_mriqc_learn`` to return a stub returning canned data."""
    train_x, train_y = _build_canned_dataset()

    def _fake_load_dataset(dataset, split_strategy):
        assert dataset == "abide"
        assert split_strategy == "none"
        return (train_x, train_y), (None, None)

    fake_module = types.SimpleNamespace(__version__="0.0.3-test")
    monkeypatch.setattr(
        mod, "_import_mriqc_learn", lambda: (fake_module, _fake_load_dataset)
    )
    return train_x, train_y


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _write_synthetic_t1w(path: Path, shape: tuple[int, int, int] = (64, 64, 64)) -> None:
    """Write a synthetic 3D NIfTI with the integrity-check passing shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nii = nib.Nifti1Image(np.zeros(shape, dtype=np.int16), np.eye(4))
    nib.save(nii, str(path))


# ──────────────────────────────────────────────
# Phase A tests
# ──────────────────────────────────────────────


def test_phase_a_basic_smoke_with_mocked_dataset(
    mod, canned_loader, tmp_path
) -> None:
    """Phase A end-to-end on canned data: 1101 rows × 74 cols, FILE_IDs derived."""
    train_x, train_y, version = mod.load_mriqc_learn_dataset()
    assert train_x.shape == (1101, 68)
    mod.assert_phase_a_schema(train_x, train_y)
    combined = mod.build_phase_a_frame(train_x, train_y)

    assert len(combined) == 1101
    # 6 metadata + 12 priority IQMs + 56 sorted IQMs = 74
    assert len(combined.columns) == 74

    # Column ordering — 18 deterministic columns first.
    expected_first_18 = [
        "FILE_ID", "site", "subject_id",
        "rater_1", "rater_2", "rater_3",
        "mriqc_cjv", "mriqc_cnr", "mriqc_efc", "mriqc_fber",
        "mriqc_snr_total", "mriqc_snr_csf", "mriqc_snr_gm", "mriqc_snr_wm",
        "mriqc_snrd_total", "mriqc_snrd_csf", "mriqc_snrd_gm", "mriqc_snrd_wm",
    ]
    assert list(combined.columns[:18]) == expected_first_18

    # Remaining 56 are mriqc_-prefixed and alphabetically sorted.
    remainder = list(combined.columns[18:])
    assert all(c.startswith("mriqc_") for c in remainder)
    assert remainder == sorted(remainder)

    # FILE_ID format spot check.
    sample_fid = combined["FILE_ID"].iloc[0]
    assert sample_fid.startswith("CALTECH_")
    assert len(sample_fid) == len("CALTECH_") + 7  # "CALTECH" + "_" + "0050000"

    assert version == "0.0.3-test"


def test_phase_a_schema_drift_aborts(mod) -> None:
    """train_x with wrong N rows aborts via SystemExit."""
    bad_x = pd.DataFrame(np.zeros((1100, 68)))
    bad_y = pd.DataFrame({
        "subject_id": list(range(50000, 51100)),
        "site": ["NYU"] * 1100,
        "rater_1": [1.0] * 1100,
        "rater_2": [0.0] * 1100,
        "rater_3": [1.0] * 1100,
    })
    with pytest.raises(SystemExit, match=r"shape"):
        mod.assert_phase_a_schema(bad_x, bad_y)


def test_file_id_derivation(mod) -> None:
    """FILE_ID handles int / float subject_id; rejects NaN; rejects bad pattern."""
    assert mod.derive_file_id("NYU", 51012) == "NYU_0051012"
    assert mod.derive_file_id("CALTECH", 51463.0) == "CALTECH_0051463"
    # MAX_MUN has an underscore — pattern allows uppercase + underscore.
    assert mod.derive_file_id("MAX_MUN", 51333) == "MAX_MUN_0051333"

    with pytest.raises(ValueError, match="NaN"):
        mod.derive_file_id("NYU", float("nan"))
    with pytest.raises(ValueError, match="NaN"):
        mod.derive_file_id(float("nan"), 51012)

    # Lowercase / pattern-failing site rejected.
    with pytest.raises(ValueError, match="pattern"):
        mod.derive_file_id("nyu", 51012)


def test_phase_a_provenance_json_written(
    mod, canned_loader, tmp_path
) -> None:
    """Provenance JSON has version, n_rows, IQM list, per-site rater coverage."""
    train_x, train_y, version = mod.load_mriqc_learn_dataset()
    combined = mod.build_phase_a_frame(train_x, train_y)
    out_path = tmp_path / "prov.json"
    mod.write_provenance_json(combined, version, out_path)

    payload = json.loads(out_path.read_text())
    assert payload["mriqc_learn_version"] == "0.0.3-test"
    assert payload["n_rows"] == 1101
    assert len(payload["iqm_columns"]) == 68
    assert all(c.startswith("mriqc_") for c in payload["iqm_columns"])
    assert len(payload["rater_coverage_by_site"]) == 17
    # NYU has 184 scans per the canned distribution.
    assert payload["rater_coverage_by_site"]["NYU"]["n_scans"] == 184


# ──────────────────────────────────────────────
# Phase B common tests
# ──────────────────────────────────────────────


def test_volume_integrity_rejects_2d_and_small_dims(mod, tmp_path) -> None:
    """verify_volume_integrity rejects 2D / multi-frame 4D / dim<64; passes 3D 64+."""
    # 2D NIfTI → reject (ndim==2 unsupported).
    nii_2d = nib.Nifti1Image(np.zeros((64, 64), dtype=np.int16), np.eye(4))
    p2d = tmp_path / "bad_2d.nii.gz"
    nib.save(nii_2d, str(p2d))
    with pytest.raises(ValueError, match=r"ndim"):
        mod.verify_volume_integrity(p2d)

    # 32×32×32 → reject (dim < 64).
    nii_small = nib.Nifti1Image(np.zeros((32, 32, 32), dtype=np.int16), np.eye(4))
    psmall = tmp_path / "bad_small.nii.gz"
    nib.save(nii_small, str(psmall))
    with pytest.raises(ValueError, match=r"Dim < 64"):
        mod.verify_volume_integrity(psmall)

    # 64×64×64 → OK.
    _write_synthetic_t1w(tmp_path / "ok.nii.gz", shape=(64, 64, 64))
    integ = mod.verify_volume_integrity(tmp_path / "ok.nii.gz")
    assert integ["shape"] == (64, 64, 64)
    assert integ["size_bytes"] > 0


def test_resume_skips_completed_files(mod, tmp_path) -> None:
    """is_already_acquired returns True for valid existing file with matching size."""
    output_dir = tmp_path / "abide"
    output_dir.mkdir()
    fid = "NYU_0051012"
    _write_synthetic_t1w(output_dir / f"{fid}.nii.gz", shape=(64, 64, 64))

    actual_size = (output_dir / f"{fid}.nii.gz").stat().st_size
    manifest_entry = {"file_size_bytes": actual_size}
    assert mod.is_already_acquired(fid, output_dir, manifest_entry) is True

    # Wrong size → mismatch → False.
    assert mod.is_already_acquired(
        fid, output_dir, {"file_size_bytes": 999999}
    ) is False

    # No file → False.
    assert mod.is_already_acquired("NOT_THERE_0000000", output_dir, None) is False

    # File exists but no manifest entry → True (relies on integrity check).
    assert mod.is_already_acquired(fid, output_dir, None) is True


def test_acquisition_summary_set_arithmetic(mod, tmp_path) -> None:
    """Summary cardinalities + warning when |B - A| > 0."""
    phase_a = tmp_path / "phase_a.csv"
    pd.DataFrame({"FILE_ID": [f"NYU_{i:07d}" for i in range(10)]}).to_csv(
        phase_a, index=False
    )

    manifest = tmp_path / "manifest.csv"
    rows: list[dict[str, Any]] = []
    for i in range(8):
        rows.append({"FILE_ID": f"NYU_{i:07d}", "preprocessing_state": "raw"})
    rows.append({"FILE_ID": "STRAY_0000001", "preprocessing_state": "raw"})
    pd.DataFrame(rows).to_csv(manifest, index=False)

    failures = tmp_path / "failures.csv"
    pd.DataFrame({"FILE_ID": [f"NYU_{i:07d}" for i in range(8, 10)]}).to_csv(
        failures, index=False
    )

    summary = tmp_path / "summary.json"
    mod.write_acquisition_summary(phase_a, manifest, failures, summary, "fcp-indi-raw")
    payload = json.loads(summary.read_text())

    assert payload["n_phase_a"] == 10
    assert payload["n_acquired"] == 9
    assert payload["n_intersection_A_B"] == 8
    assert payload["n_phase_a_only"] == 2
    assert payload["n_acquired_only"] == 1
    assert payload["n_failed"] == 2
    assert payload["acquisition_path_used"] == "fcp-indi-raw"
    assert payload["preprocessing_state_distribution"] == {"raw": 9}
    assert any("not in Phase A" in w for w in payload["warnings"])


# ──────────────────────────────────────────────
# Phase B path tests
# ──────────────────────────────────────────────


def test_phase_b_local_pattern_resolution(mod, tmp_path) -> None:
    """resolve_local: 3 BIDS patterns + glob fallback. First match wins."""
    # Pattern 2: sub-XXXXX/anat/sub-XXXXX_T1w.nii.gz
    bids_root = tmp_path / "bids"
    sub = bids_root / "sub-0051463" / "anat"
    sub.mkdir(parents=True)
    target = sub / "sub-0051463_T1w.nii.gz"
    target.write_bytes(b"fake")
    found = mod.resolve_local("CALTECH_0051463", "CALTECH", 51463, bids_root)
    assert found == target

    # Pattern 3: {site}/sub-XXXXX/anat/...
    bids_root3 = tmp_path / "bids3"
    sub3 = bids_root3 / "CALTECH" / "sub-0051463" / "anat"
    sub3.mkdir(parents=True)
    target3 = sub3 / "sub-0051463_T1w.nii.gz"
    target3.write_bytes(b"fake")
    found3 = mod.resolve_local("CALTECH_0051463", "CALTECH", 51463, bids_root3)
    assert found3 == target3

    # Glob fallback (none of the 3 patterns match).
    bids_root_glob = tmp_path / "bids_glob"
    odd = bids_root_glob / "weird" / "deeper" / "0051463_run-01_T1w.nii.gz"
    odd.parent.mkdir(parents=True)
    odd.write_bytes(b"fake")
    found_glob = mod.resolve_local("CALTECH_0051463", "CALTECH", 51463, bids_root_glob)
    assert found_glob == odd


def test_phase_b_nitrc_id_mapping_validation(mod, tmp_path) -> None:
    """NITRC id-mapping CSV missing required columns → SystemExit."""
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"phenotype_file_id": ["NYU_0051012"]}).to_csv(bad_csv, index=False)
    with pytest.raises(SystemExit, match=r"missing required columns"):
        mod.load_id_mapping_csv(bad_csv)

    good_csv = tmp_path / "good.csv"
    pd.DataFrame({
        "phenotype_file_id": ["NYU_0051012"],
        "nitrc_session_id": ["S001"],
        "downloaded_path": ["/tmp/foo.nii.gz"],
    }).to_csv(good_csv, index=False)
    df = mod.load_id_mapping_csv(good_csv)
    assert len(df) == 1


def test_phase_b_s3_raw_mocked(mod, tmp_path, monkeypatch) -> None:
    """S3 raw BIDS path with mocked boto3 client → manifest written, ``preprocessing_state="raw"``."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from botocore.exceptions import ClientError

    # Build a valid 64×64×64 NIfTI on disk to be "downloaded" by the fake.
    nii_path = tmp_path / "fake_t1w.nii.gz"
    _write_synthetic_t1w(nii_path)

    class FakeClient:
        def head_object(self, Bucket, Key):
            # Caltech canonical key matches the verified site map.
            if "Caltech/sub-0051463/anat/sub-0051463_T1w.nii.gz" in Key:
                return {}
            raise ClientError(
                {"Error": {"Code": "404", "Message": "not found"}}, "HeadObject"
            )

        def download_file(self, Bucket, Key, Filename):
            shutil.copy(str(nii_path), Filename)

    monkeypatch.setattr(mod, "make_anonymous_s3_client", lambda: FakeClient())
    monkeypatch.setattr(
        mod, "_import_boto3", lambda: (boto3, UNSIGNED, Config, ClientError)
    )

    output_dir = tmp_path / "abide"
    manifest = tmp_path / "manifest.csv"
    failures = tmp_path / "failures.csv"

    file_ids = [("CALTECH_0051463", "CALTECH", 51463)]
    n_acquired = mod.acquire_via_s3_raw(
        file_ids, output_dir, manifest, failures, max_concurrent=1
    )
    assert n_acquired == 1
    assert (output_dir / "CALTECH_0051463.nii.gz").is_file()

    df = pd.read_csv(manifest)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["FILE_ID"] == "CALTECH_0051463"
    assert row["preprocessing_state"] == "raw"
    assert row["acquisition_path"] == "fcp-indi-raw"
    assert row["site"] == "CALTECH"
    assert row["subject_id"] == 51463
    assert row["shape_x"] == 64
    assert "Caltech/sub-0051463/anat/sub-0051463_T1w.nii.gz" in row["source_path"]
