"""Unit tests for code/results_tracker.py per-dataset correlation logic.

Focused on the dataset_tag propagation from corruption_manifest.csv through
the three analyze_phase* functions. Uses tmp_path + monkeypatch to redirect
TABLES_DIR/METRICS_DIR onto a per-test scratch tree. The module under test
has no digit prefix, so it is importable via importlib directly.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE_PATH = _REPO_ROOT / "code" / "results_tracker.py"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("results_tracker", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["results_tracker"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rt_mod() -> types.ModuleType:
    return _load_module()


@pytest.fixture
def scratch_dirs(
    rt_mod: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Redirect the module's TABLES_DIR and METRICS_DIR into tmp_path."""
    tables = tmp_path / "tables"
    metrics = tmp_path / "metrics"
    tables.mkdir()
    metrics.mkdir()
    monkeypatch.setattr(rt_mod, "TABLES_DIR", tables)
    monkeypatch.setattr(rt_mod, "METRICS_DIR", metrics)
    return tables, metrics


def _write_corruption_manifest(
    tables: Path, rows: list[tuple[str, str]]
) -> None:
    """Write a minimal corruption_manifest.csv with (cor_path, dataset_tag) rows."""
    df = pd.DataFrame(rows, columns=["cor_path", "dataset_tag"])
    # Phase 2 writes more columns; we only need these two for the join.
    df.to_csv(tables / "corruption_manifest.csv", index=False)


def test_load_dataset_tag_frame_returns_none_when_missing(
    rt_mod: types.ModuleType, scratch_dirs: tuple[Path, Path]
) -> None:
    assert rt_mod._load_dataset_tag_frame() is None


def test_load_dataset_tag_frame_returns_none_when_column_missing(
    rt_mod: types.ModuleType, scratch_dirs: tuple[Path, Path]
) -> None:
    tables, _ = scratch_dirs
    pd.DataFrame({"cor_path": ["a"], "severity": [1]}).to_csv(
        tables / "corruption_manifest.csv", index=False
    )
    assert rt_mod._load_dataset_tag_frame() is None


def test_load_dataset_tag_frame_deduplicates(
    rt_mod: types.ModuleType, scratch_dirs: tuple[Path, Path]
) -> None:
    tables, _ = scratch_dirs
    # Phase 2 emits one row per (cor_path, corruption, severity); a single
    # cor_path therefore appears exactly once, but defend against dup rows.
    _write_corruption_manifest(
        tables,
        [
            ("/cor/a.nii.gz", "ixi"),
            ("/cor/a.nii.gz", "ixi"),
            ("/cor/b.nii.gz", "fastmri"),
        ],
    )
    frame = rt_mod._load_dataset_tag_frame()
    assert frame is not None
    assert set(frame.columns) == {"cor_path", "dataset_tag"}
    assert len(frame) == 2


def test_analyze_phase3_emits_per_dataset_rows(
    rt_mod: types.ModuleType, scratch_dirs: tuple[Path, Path]
) -> None:
    """analyze_phase3 tags existing rows 'overall' and adds one row per
    dataset × IQM pair joined via corruption_manifest."""
    tables, metrics = scratch_dirs

    cor_paths = [f"/cor/s{i}.nii.gz" for i in range(10)]
    pd.DataFrame({
        "cor_path": cor_paths,
        "corruption_type": ["motion"] * 10,
        "mean_dice": [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45],
    }).to_csv(tables / "machine_preference.csv", index=False)

    pd.DataFrame({
        "scan_path": cor_paths,
        "snr": [10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5],
        "cnr": [5.0, 4.8, 4.6, 4.4, 4.2, 4.0, 3.8, 3.6, 3.4, 3.2],
        "efc": [0.1] * 10,
        "fber": [0.2] * 10,
        "cjv": [0.3] * 10,
    }).to_csv(tables / "iqm_features.csv", index=False)

    _write_corruption_manifest(
        tables,
        [(p, "ixi") for p in cor_paths[:5]] + [(p, "fastmri") for p in cor_paths[5:]],
    )

    results_df = rt_mod.analyze_phase3()
    assert results_df is not None
    assert "dataset" in results_df.columns

    overall = results_df[
        (results_df["target"] == "mean_dice") & (results_df["dataset"] == "overall")
    ]
    assert len(overall) == 5  # one per IQM column

    per_dataset = results_df[
        (results_df["target"] == "mean_dice") & (results_df["dataset"] != "overall")
    ]
    assert set(per_dataset["dataset"].unique()) == {"ixi", "fastmri"}
    # 5 IQMs x 2 datasets = 10 per-dataset rows
    assert len(per_dataset) == 10

    # Sanity: with 5 samples per dataset and monotone snr vs mean_dice, srcc == 1.0
    ixi_snr = per_dataset[
        (per_dataset["dataset"] == "ixi") & (per_dataset["predictor"] == "snr")
    ]
    assert len(ixi_snr) == 1
    assert ixi_snr.iloc[0]["srcc"] == pytest.approx(1.0, abs=1e-4)


def test_analyze_phase4_emits_per_dataset_rows(
    rt_mod: types.ModuleType, scratch_dirs: tuple[Path, Path]
) -> None:
    tables, _ = scratch_dirs

    cor_paths = [f"/cor/s{i}.nii.gz" for i in range(6)]
    pd.DataFrame({
        "cor_path": cor_paths,
        "corruption_type": ["motion"] * 6,
        "mean_dice": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
    }).to_csv(tables / "machine_preference.csv", index=False)

    pd.DataFrame({
        "cor_path": cor_paths,
        "model": ["modelA"] * 6,
        "predicted_score": [0.95, 0.85, 0.75, 0.65, 0.55, 0.45],
        "corruption_type": ["motion"] * 6,
    }).to_csv(tables / "3d_vlm_scores.csv", index=False)

    _write_corruption_manifest(
        tables,
        [(p, "ixi") for p in cor_paths[:3]] + [(p, "oasis") for p in cor_paths[3:]],
    )

    results_df = rt_mod.analyze_phase4()
    assert results_df is not None
    assert "dataset" in results_df.columns

    per_dataset = results_df[
        (results_df["model"] == "modelA")
        & (results_df["target"] == "mean_dice")
        & (results_df["dataset"] != "overall")
    ]
    assert set(per_dataset["dataset"].unique()) == {"ixi", "oasis"}
    assert len(per_dataset) == 2


def test_analyze_phase3_without_manifest_still_emits_overall(
    rt_mod: types.ModuleType, scratch_dirs: tuple[Path, Path]
) -> None:
    """Missing corruption_manifest.csv must not crash analyze_phase3; it
    simply skips per-dataset rows."""
    tables, _ = scratch_dirs

    cor_paths = [f"/cor/s{i}.nii.gz" for i in range(5)]
    pd.DataFrame({
        "cor_path": cor_paths,
        "corruption_type": ["motion"] * 5,
        "mean_dice": [0.9, 0.8, 0.7, 0.6, 0.5],
    }).to_csv(tables / "machine_preference.csv", index=False)

    pd.DataFrame({
        "scan_path": cor_paths,
        "snr": [10.0, 9.0, 8.0, 7.0, 6.0],
        "cnr": [5.0, 4.0, 3.0, 2.0, 1.0],
        "efc": [0.1] * 5,
        "fber": [0.2] * 5,
        "cjv": [0.3] * 5,
    }).to_csv(tables / "iqm_features.csv", index=False)

    results_df = rt_mod.analyze_phase3()
    assert results_df is not None
    assert set(results_df["dataset"].unique()) == {"overall"}
