"""Unit tests for code/11_compare_abide_zeroshot.py."""

from __future__ import annotations

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
_MODULE_PATH = _REPO_ROOT / "code" / "11_compare_abide_zeroshot.py"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("compare_abide_zs", _MODULE_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {_MODULE_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_abide_zs"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> types.ModuleType:
    return _load_module()


# ──────────────────────────────────────────────
# Synthetic factories
# ──────────────────────────────────────────────


def _build_summary(model: str, seed: int, auc_a: float, auc_c: float = 0.7) -> dict[str, Any]:
    """Build a synthetic per-model summary JSON with a known AUC for variant A."""
    return {
        "model": model,
        "seed": seed,
        "scans_evaluated": 100,
        "parse_failure_rate": 0.05,
        "variants": {
            "A": {
                "n": 100, "n_positive": 50, "n_negative": 50,
                "auc": {"point": auc_a, "ci_lower": auc_a - 0.05, "ci_upper": auc_a + 0.05},
                "youden_threshold": 0.5,
                "youden_threshold_is_in_sample": True,
                "accuracy": {"point": 0.75},
                "sensitivity": {"point": 0.8}, "specificity": {"point": 0.7},
                "cohens_kappa": {"point": 0.4},
                "loso_aucs_per_site": {"NYU": 0.7, "PITT": 0.65},
                "loso_auc_mean": 0.675, "loso_auc_std": 0.035,
            },
            "B": {
                "n": 100, "n_positive": 50, "n_negative": 50,
                "auc": {"point": auc_a + 0.02, "ci_lower": auc_a - 0.03, "ci_upper": auc_a + 0.07},
                "youden_threshold": 0.5,
                "youden_threshold_is_in_sample": True,
                "accuracy": {"point": 0.76},
                "sensitivity": {"point": 0.81}, "specificity": {"point": 0.71},
                "cohens_kappa": {"point": 0.42},
                "loso_aucs_per_site": {"NYU": 0.71, "PITT": 0.66},
                "loso_auc_mean": 0.685, "loso_auc_std": 0.035,
            },
            "C": {
                "n": 60, "n_positive": 30, "n_negative": 30,
                "auc": {"point": auc_c, "ci_lower": auc_c - 0.07, "ci_upper": auc_c + 0.07},
                "youden_threshold": 0.5,
                "youden_threshold_is_in_sample": True,
                "accuracy": {"point": 0.7},
                "sensitivity": {"point": 0.75}, "specificity": {"point": 0.65},
                "cohens_kappa": {"point": 0.35},
                "loso_aucs_per_site": {"NYU": 0.7, "PITT": 0.6},
                "loso_auc_mean": 0.65, "loso_auc_std": 0.07,
            },
        },
        "weighted_kappa_per_rater": {
            "rater_1": 0.3, "rater_2": 0.32, "rater_3": 0.4,
        },
        "krippendorff_alpha_4way_variant_c": 0.45,
        "per_site_variant_c": {},
    }


def _write_summaries(tmp_path: Path, models_aucs: list[tuple[str, float]]) -> Path:
    """Write summary JSONs for the given (model, auc_a) pairs."""
    d = tmp_path / "metrics"
    d.mkdir(parents=True, exist_ok=True)
    for model, auc in models_aucs:
        s = _build_summary(model, seed=0, auc_a=auc)
        (d / f"abide_zeroshot_summary_{model}_seed_0.json").write_text(json.dumps(s))
    return d


def _write_predictions(
    tmp_path: Path, model: str, file_ids: list[str], scores: np.ndarray, labels: np.ndarray
) -> Path:
    """Write a synthetic predictions CSV for a model."""
    d = tmp_path / "tables"
    d.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for fid, sc, lab in zip(file_ids, scores, labels, strict=True):
        rows.append({
            "FILE_ID": fid,
            "site": "NYU",
            "scan_path": f"/x/{fid}.nii.gz",
            "rater_1": float("nan"),
            "rater_2": float("nan"),
            "rater_3": 1 if lab > 0.5 else -1,
            "consensus_a": int(lab),
            "consensus_b": int(lab),
            "consensus_c": int(lab),
            "vlm_score": float(sc),
            "vlm_likert": 4 if sc > 0.5 else 2,
            "vlm_raw_text": "SCORE: 4",
            "parse_status": "ok",
        })
    pd.DataFrame(rows).to_csv(
        d / f"abide_zeroshot_predictions_{model}_seed_0.csv", index=False
    )
    return d


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────


def test_load_summaries_globs_correctly(mod, tmp_path) -> None:
    """5 model JSONs in summaries_dir → all loaded with parsed (model, seed)."""
    summaries_dir = _write_summaries(
        tmp_path,
        [("m3d_lamed", 0.85), ("llava_ov", 0.75), ("qwen2_vl", 0.72),
         ("medgemma", 0.70), ("gpt4o", 0.78)],
    )
    out = mod.load_summaries(summaries_dir)
    assert len(out) == 5
    models = sorted({s["_parsed_model"] for s in out})
    assert models == sorted(["m3d_lamed", "llava_ov", "qwen2_vl", "medgemma", "gpt4o"])
    assert all(s["_parsed_seed"] == 0 for s in out)


def test_meta_table_long_schema(mod, tmp_path) -> None:
    """Meta table has the documented long-format schema; one row per (model, variant, metric)."""
    summaries_dir = _write_summaries(
        tmp_path,
        [("m3d_lamed", 0.85), ("llava_ov", 0.72)],
    )
    summaries = mod.load_summaries(summaries_dir)
    meta = mod.build_meta_table(summaries, mod.CONSENSUS_VARIANTS)
    expected_cols = {"model", "variant", "metric", "value", "ci_low", "ci_high", "n_seeds"}
    assert expected_cols.issubset(set(meta.columns))
    # 2 models × 3 variants × 7 metrics + parse_failure (variant *) + α (variant C only)
    # = 2 * (3*7 + 1 + 1) = 46 rows max
    assert len(meta) > 0
    # AUC for both models present.
    auc_rows = meta[meta["metric"] == "auc"]
    assert set(auc_rows["model"].unique()) == {"m3d_lamed", "llava_ov"}


def test_pairwise_delong_uses_paired_predictions(mod, tmp_path) -> None:
    """Synthetic predictions with a known AUC difference → DeLong delta_auc reflects it."""
    summaries_dir = _write_summaries(
        tmp_path,
        [("m3d_lamed", 0.85), ("llava_ov", 0.65)],
    )
    summaries = mod.load_summaries(summaries_dir)

    # Build paired predictions: same FILE_IDs, different scores.
    file_ids = [f"NYU_{50000 + i:07d}" for i in range(40)]
    rng = np.random.default_rng(0)
    labels = (rng.uniform(0, 1, 40) > 0.5).astype(float)
    # Model A: very predictive scores.
    scores_a = labels + rng.normal(0, 0.1, 40)
    # Model B: weakly predictive scores (more noise).
    scores_b = labels * 0.3 + rng.normal(0, 0.4, 40)

    predictions_dir = _write_predictions(tmp_path, "m3d_lamed", file_ids, scores_a, labels)
    _write_predictions(tmp_path, "llava_ov", file_ids, scores_b, labels)

    predictions = mod.load_predictions(predictions_dir)
    assert ("m3d_lamed", 0) in predictions
    assert ("llava_ov", 0) in predictions

    delong = mod.pairwise_delong(summaries, predictions, mod.CONSENSUS_VARIANTS)
    assert len(delong) > 0
    # m3d_lamed should have higher AUC than llava_ov in the synthetic data.
    pair_a = delong[
        (delong["model_a"] == "llava_ov") & (delong["model_b"] == "m3d_lamed")
    ]
    pair_b = delong[
        (delong["model_a"] == "m3d_lamed") & (delong["model_b"] == "llava_ov")
    ]
    pair = pair_a if not pair_a.empty else pair_b
    assert not pair.empty
    # delta_auc points the right direction.
    a_row = pair.iloc[0]
    if a_row["model_a"] == "m3d_lamed":
        assert a_row["delta_auc"] > 0
    else:
        assert a_row["delta_auc"] < 0


def test_bh_fdr_adjustment_monotone(mod) -> None:
    """BH-adjusted p ≥ raw p; monotonicity preserved."""
    raw = [0.001, 0.01, 0.02, 0.04, 0.05, 0.5, 0.8]
    adj = mod.benjamini_hochberg(raw)
    assert len(adj) == len(raw)
    for r, a in zip(raw, adj, strict=True):
        assert a >= r - 1e-9, f"adjusted {a} < raw {r}"
    # Monotonicity in input order is NOT required (BH preserves the rank order).
    # But when we sort by raw p, the adjusted should be non-decreasing.
    sorted_raw_adj = sorted(zip(raw, adj), key=lambda x: x[0])
    sorted_adj = [a for _, a in sorted_raw_adj]
    for i in range(1, len(sorted_adj)):
        assert sorted_adj[i] >= sorted_adj[i - 1] - 1e-9


def test_latex_emits_when_flag_set(mod, tmp_path) -> None:
    """write_latex produces a .tex file with booktabs idiom."""
    summaries_dir = _write_summaries(
        tmp_path, [("m3d_lamed", 0.85), ("llava_ov", 0.72)]
    )
    summaries = mod.load_summaries(summaries_dir)
    meta = mod.build_meta_table(summaries, mod.CONSENSUS_VARIANTS)
    out_path = tmp_path / "out.tex"
    mod.write_latex(meta, out_path)
    assert out_path.is_file()
    content = out_path.read_text()
    assert "\\toprule" in content
    assert "\\bottomrule" in content
    assert "Variant A" in content
    assert "m3d_lamed" in content


def test_dry_run_no_writes(mod, tmp_path) -> None:
    """--dry-run flag → no output files written."""
    from typer.testing import CliRunner

    summaries_dir = _write_summaries(tmp_path, [("m3d_lamed", 0.85)])
    output_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        mod.app,
        [
            "--summaries-dir", str(summaries_dir),
            "--predictions-dir", str(tmp_path),  # no predictions written
            "--output-dir", str(output_dir),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    # No output files created.
    if output_dir.exists():
        assert not list(output_dir.glob("*.csv"))
        assert not list(output_dir.glob("*.tex"))
        assert not list(output_dir.glob("*.json"))
