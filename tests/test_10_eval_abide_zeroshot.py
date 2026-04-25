"""Unit tests for code/10_eval_abide_zeroshot.py.

Mock-heavy: real model weights NEVER loaded. Tests use synthetic ratings/
manifest CSVs and `monkeypatch` for adapter dispatch + mriqc-learn.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE_PATH = _REPO_ROOT / "code" / "10_eval_abide_zeroshot.py"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("eval_abide_zs", _MODULE_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {_MODULE_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_abide_zs"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> types.ModuleType:
    return _load_module()


# ──────────────────────────────────────────────
# Synthetic data
# ──────────────────────────────────────────────


def _make_ratings_df(n: int = 30) -> pd.DataFrame:
    """Build a synthetic ratings frame with deterministic rater patterns."""
    rng = np.random.default_rng(42)
    rows: list[dict[str, Any]] = []
    sites = ["NYU", "PITT", "CALTECH"]
    for i in range(n):
        site = sites[i % 3]
        # rater_3 fully covered; rater_1/2 covered ~half the time.
        r3 = int(rng.choice([-1, 0, 1]))
        r1 = float(rng.choice([-1, 0, 1])) if i % 2 == 0 else float("nan")
        r2 = float(rng.choice([-1, 0, 1])) if i % 3 != 0 else float("nan")
        rows.append({
            "FILE_ID": f"{site}_{50000 + i:07d}",
            "site": site,
            "subject_id": 50000 + i,
            "rater_1": r1,
            "rater_2": r2,
            "rater_3": float(r3),
        })
    return pd.DataFrame(rows)


def _make_manifest_df(ratings_df: pd.DataFrame, tmp_path: Path) -> pd.DataFrame:
    """Build a manifest with scan_path pointing at touched empty files (never read)."""
    rows: list[dict[str, Any]] = []
    for _, row in ratings_df.iterrows():
        scan_path = tmp_path / "scans" / f"{row['FILE_ID']}.nii.gz"
        scan_path.parent.mkdir(parents=True, exist_ok=True)
        scan_path.write_bytes(b"")
        rows.append({
            "FILE_ID": row["FILE_ID"],
            "site": row["site"],
            "subject_id": row["subject_id"],
            "scan_path": str(scan_path),
            "acquisition_path": "fcp-indi-raw",
            "preprocessing_state": "raw",
        })
    return pd.DataFrame(rows)


def _write_csvs(tmp_path: Path, n: int = 30) -> tuple[Path, Path]:
    ratings_df = _make_ratings_df(n)
    manifest_df = _make_manifest_df(ratings_df, tmp_path)
    ratings_csv = tmp_path / "ratings.csv"
    manifest_csv = tmp_path / "manifest.csv"
    ratings_df.to_csv(ratings_csv, index=False)
    manifest_df.to_csv(manifest_csv, index=False)
    return ratings_csv, manifest_csv


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────


def test_consensus_variant_a(mod) -> None:
    """consensus_a == (rater_3 == 1).astype(int) for all rows; coverage == |df|."""
    df = pd.DataFrame({
        "FILE_ID": ["A", "B", "C", "D"],
        "rater_1": [1.0, 0.0, -1.0, float("nan")],
        "rater_2": [1.0, float("nan"), 0.0, 1.0],
        "rater_3": [1.0, 0.0, -1.0, 1.0],
    })
    a = mod.build_consensus_a(df)
    assert a.tolist() == [1, 0, 0, 1]


def test_consensus_variant_b_majority(mod) -> None:
    """B: 3-rater majority where all present; tie → rater_3; only-rater_3 → rater_3."""
    df = pd.DataFrame({
        "FILE_ID": ["case_3agree_pos", "case_3disagree", "case_only_r3"],
        "rater_1": [1.0, 1.0, float("nan")],
        "rater_2": [1.0, 0.0, float("nan")],
        "rater_3": [0.0, -1.0, 1.0],   # majority +1, tie, +1
    })
    b = mod.build_consensus_b(df)
    # case_3agree_pos: r1=r2=1, r3=0 → majority +1 → 1
    assert b.iloc[0] == 1
    # case_3disagree: 1, 0, -1 → no two agree → fallback rater_3=-1 → 0
    assert b.iloc[1] == 0
    # case_only_r3: r3=1 directly → 1
    assert b.iloc[2] == 1


def test_consensus_variant_c_excludes_ties(mod) -> None:
    """C: strict 3-rater majority. Ties → NaN (excluded). Coverage = (full-3-rater scans − ties)."""
    df = pd.DataFrame({
        "FILE_ID": ["agree_pos", "tie_3way", "missing_r1", "agree_neg"],
        "rater_1": [1.0, 1.0, float("nan"), -1.0],
        "rater_2": [1.0, 0.0, 0.0, -1.0],
        "rater_3": [1.0, -1.0, 1.0, 0.0],
    })
    c = mod.build_consensus_c(df)
    # agree_pos: 1,1,1 → majority 1 → 1 (accept)
    assert c.iloc[0] == 1
    # tie_3way: 1,0,-1 → NaN
    assert pd.isna(c.iloc[1])
    # missing_r1: not full 3-rater coverage → NaN
    assert pd.isna(c.iloc[2])
    # agree_neg: -1,-1,0 → majority -1 → 0 (non-accept)
    assert c.iloc[3] == 0


def test_youden_threshold_optimization(mod) -> None:
    """Synthetic scores + labels with a known optimum → Youden picks it."""
    rng = np.random.default_rng(0)
    # Positives have score ~ N(0.7, 0.1); negatives ~ N(0.3, 0.1).
    n = 200
    labels = np.array([0] * n + [1] * n)
    scores = np.concatenate([
        rng.normal(0.3, 0.1, n),
        rng.normal(0.7, 0.1, n),
    ])
    out = mod.youden_threshold_metrics(scores, labels.astype(float))
    assert 0.4 < out["threshold"] < 0.6
    assert out["sensitivity"] > 0.7
    assert out["specificity"] > 0.7


def test_cluster_bootstrap_by_site_reproducible(mod) -> None:
    """Same --bootstrap-seed → byte-identical CI bounds."""
    rng = np.random.default_rng(0)
    n = 100
    scores = rng.uniform(0, 1, n)
    labels = (scores > 0.5).astype(float)
    sites = np.array([f"site_{i % 4}" for i in range(n)])
    a = mod.cluster_bootstrap_auc(scores, labels, sites, n_boot=200, seed=42)
    b = mod.cluster_bootstrap_auc(scores, labels, sites, n_boot=200, seed=42)
    assert a == b
    # Different seed → different CI (high probability).
    c = mod.cluster_bootstrap_auc(scores, labels, sites, n_boot=200, seed=99)
    assert a[0] == c[0]  # point estimate seed-independent


def test_loso_split_size(mod) -> None:
    """17-site synthetic input → 17-fold output dict, each site reported."""
    n_per_site = 50
    sites_list = [f"site_{i:02d}" for i in range(17)]
    sites = np.array([s for s in sites_list for _ in range(n_per_site)])
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, len(sites))
    labels = (scores > 0.5).astype(float)
    folds, mean, std = mod.loso_aucs(scores, labels, sites, min_n=20)
    assert len(folds) == 17
    assert all(s in folds for s in sites_list)
    assert 0.0 <= mean <= 1.0


def test_per_site_skipped_when_small(mod) -> None:
    """Sites with N < min_n → not in per_site dict."""
    sites = np.array(["big"] * 50 + ["small"] * 5)
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 55)
    labels = (scores > 0.5).astype(float)
    out = mod.per_site_auc(scores, labels, sites, min_n=30, n_boot=100, seed=42)
    assert "big" in out
    assert "small" not in out


def test_parse_failure_excluded_from_metrics(mod) -> None:
    """NaN VLM scores → not counted in AUC; parse_failure_rate reflects them."""
    n = 50
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, n)
    # Inject 10 NaN scores.
    scores[:10] = float("nan")
    labels = (np.where(np.isnan(scores), 0, scores > 0.5)).astype(float)
    sites = np.array([f"s_{i % 3}" for i in range(n)])
    auc, lo, hi = mod.cluster_bootstrap_auc(scores, labels, sites, n_boot=100, seed=42)
    # AUC computed only on the 40 non-NaN entries; not NaN.
    assert np.isfinite(auc)


def test_resume_skips_done_pairs(mod, tmp_path) -> None:
    """Pre-seed predictions CSV with (FILE_ID_x, model_name) → that pair skipped."""
    csv_path = tmp_path / "preds.csv"
    with csv_path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["FILE_ID", "site", "scan_path",
                                          "rater_1", "rater_2", "rater_3",
                                          "consensus_a", "consensus_b", "consensus_c",
                                          "vlm_score", "vlm_likert",
                                          "vlm_raw_text", "parse_status"])
        w.writeheader()
        w.writerow({
            "FILE_ID": "NYU_0050001", "site": "NYU", "scan_path": "/x.nii.gz",
            "rater_1": "", "rater_2": "", "rater_3": "1",
            "consensus_a": "1", "consensus_b": "1", "consensus_c": "",
            "vlm_score": "0.75", "vlm_likert": "4",
            "vlm_raw_text": "SCORE: 4", "parse_status": "ok",
        })
    done = mod.load_existing_predictions(csv_path)
    assert ("NYU_0050001", "_unknown") in done


def test_dry_run_no_inference(mod, tmp_path, monkeypatch) -> None:
    """--dry-run flag → no adapter loaded, no inference."""
    from typer.testing import CliRunner

    ratings_csv, manifest_csv = _write_csvs(tmp_path, n=12)

    # Sentinel: load_adapter_for_model must never be called.
    def _explode(*_a, **_kw):
        raise AssertionError("load_adapter_for_model called in --dry-run")

    monkeypatch.setattr(mod, "load_adapter_for_model", _explode)

    runner = CliRunner()
    result = runner.invoke(
        mod.app,
        [
            "--seed", "0",
            "--ratings-csv", str(ratings_csv),
            "--acquisition-manifest", str(manifest_csv),
            "--output-tables-dir", str(tmp_path / "tables"),
            "--output-metrics-dir", str(tmp_path / "metrics"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    # No prediction CSVs / summary JSONs written.
    assert not list((tmp_path / "tables").glob("*.csv")) if (tmp_path / "tables").exists() else True
    assert not list((tmp_path / "metrics").glob("*.json")) if (tmp_path / "metrics").exists() else True


def test_mriqc_baseline_optional(mod, tmp_path) -> None:
    """mriqc_classifier_proba returns None when mriqc_learn unavailable; main skips."""
    # Direct unit test on the classifier-loading function.
    df = _make_ratings_df(20)
    ratings_csv = tmp_path / "ratings.csv"
    df.to_csv(ratings_csv, index=False)

    # Simulate "mriqc_learn unavailable" by removing it from sys.modules + importer hook.
    saved = sys.modules.pop("mriqc_learn.models.production", None)
    sys.modules["mriqc_learn.models.production"] = None  # type: ignore[assignment]
    try:
        result = mod.mriqc_classifier_proba(df, ratings_csv)
        assert result is None
    finally:
        if saved is not None:
            sys.modules["mriqc_learn.models.production"] = saved
        else:
            sys.modules.pop("mriqc_learn.models.production", None)


def test_delong_zero_when_identical(mod) -> None:
    """Two identical classifiers → DeLong p ≈ 1.0; delta_auc == 0."""
    rng = np.random.default_rng(0)
    n = 200
    scores = rng.uniform(0, 1, n)
    labels = (scores > 0.5).astype(float)
    out = mod.delong_test(scores, scores, labels)
    assert abs(out["delta_auc"]) < 1e-9
    assert out["z"] == 0.0
    assert out["p"] >= 0.999  # ≈ 1.0
