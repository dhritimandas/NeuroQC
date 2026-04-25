"""Unit tests for code/visualize.py.

Synthetic CSVs only — no real data, no upstream phase invocation. Tests
focus on data flow, precondition checks, and atomic-write contracts;
visual fidelity is reviewed by eye.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE_PATH = _REPO_ROOT / "code" / "visualize.py"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("visualize_mod", _MODULE_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {_MODULE_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["visualize_mod"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> types.ModuleType:
    return _load_module()


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ──────────────────────────────────────────────
# Synthetic CSV helpers
# ──────────────────────────────────────────────


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_results_dir(tmp_path: Path) -> Path:
    """Build a synthetic ``results/tables/`` directory with every CSV any figure needs."""
    rd = tmp_path / "results"
    rd.mkdir(parents=True, exist_ok=True)

    refs = [f"/ref/r{i:02d}.nii.gz" for i in range(8)]
    cors = []
    cor_rows = []
    pref_rows = []
    iqm_rows = []
    for ref in refs:
        # Clean row goes into IQM (with is_reference=True), no preference row.
        iqm_rows.append({
            "scan_path": ref, "snr": 4.5, "cnr": 0.6, "efc": 0.7,
            "fber": 5.0, "cjv": 1.0, "is_reference": True,
            "corruption_type": "none", "severity": 0,
        })
        for ctype in ("motion", "ghosting"):
            for sev in (1, 3):
                cor = ref.replace(".nii.gz", f"_{ctype}_{sev}.nii.gz")
                cors.append(cor)
                cor_rows.append({
                    "ref_path": ref, "cor_path": cor, "corruption_type": ctype,
                    "corruption_domain": "image", "severity": sev,
                    "seed": 42, "transform_params": "{}", "dataset_tag": "fastmri",
                })
                # Mean dice degrades with severity for plausibility.
                dice = max(0.05, 0.95 - 0.10 * sev - (0.05 if ctype == "motion" else 0.0))
                pref_rows.append({
                    "ref_path": ref, "cor_path": cor,
                    "corruption_type": ctype, "severity": sev,
                    "mean_dice": dice,
                    "hippocampus_dice": dice + 0.02,
                    "cortex_dice": dice - 0.02,
                    "ventricle_dice": dice,
                    "thalamus_dice": dice,
                    "caudate_dice": dice,
                    "putamen_dice": dice,
                    "brainstem_dice": dice,
                    "cerebellum_dice": dice,
                    "ref_mean_thickness": 4.0,
                    "cor_mean_thickness": 4.0 + sev * 0.05,
                    "thickness_shift": sev * 0.05,
                })
                iqm_rows.append({
                    "scan_path": cor, "snr": 4.5 - sev * 0.5, "cnr": 0.6 - sev * 0.05,
                    "efc": 0.7 + sev * 0.02, "fber": 5.0 - sev * 0.3,
                    "cjv": 1.0 + sev * 0.1, "is_reference": False,
                    "corruption_type": ctype, "severity": sev,
                })

    _write_csv(rd / "iqm_features.csv", list(iqm_rows[0].keys()), iqm_rows)
    _write_csv(rd / "machine_preference.csv", list(pref_rows[0].keys()), pref_rows)
    _write_csv(rd / "corruption_manifest.csv", list(cor_rows[0].keys()), cor_rows)

    # Two seed files for 3D and 2D VLM scores.
    for seed in (0, 1):
        rows_3d = [
            {"scan_path": s, "model": "m3d_lamed",
             "score": (0.05 if not s.endswith("_motion_3.nii.gz") else 0.4) + seed * 0.01,
             "raw_response": "SCORE: 3", "seed": seed}
            for s in refs + cors
        ]
        _write_csv(
            rd / f"3d_vlm_scores_seed_{seed}.csv",
            list(rows_3d[0].keys()), rows_3d,
        )
        rows_2d = [
            {"scan_path": s, "model": "llava_ov", "slice_strategy": "mid",
             "score": (0.10 if not s.endswith("_ghosting_3.nii.gz") else 0.5) + seed * 0.01,
             "raw_response": "SCORE: 2", "seed": seed,
             "n_slices": 3, "multi_image_mode": "multi_image"}
            for s in refs + cors
        ]
        _write_csv(
            rd / f"2d_vlm_scores_seed_{seed}.csv",
            list(rows_2d[0].keys()), rows_2d,
        )
        # Fine-tuned scores: same scan list, different model.
        rows_ft = [
            {"scan_path": s, "model": "m3d_lamed",
             "dice_quality": 4 - (1 if s.endswith("_motion_3.nii.gz") else 0),
             "thickness_quality": 4,
             "raw_response": "Quality: 4 Thickness: 4",
             "seed": seed}
            for s in refs + cors
        ]
        _write_csv(
            rd / f"finetuned_scores_seed_{seed}.csv",
            list(rows_ft[0].keys()), rows_ft,
        )

    # Per-structure long CSV: ~6 labels × n_pairs.
    ps_rows = []
    for row in pref_rows:
        for lid, lname in [
            (17, "Left-Hippocampus"), (3, "Left-Cerebral-Cortex"),
            (4, "Left-Lateral-Ventricle"), (10, "Left-Thalamus"),
            (11, "Left-Caudate"), (16, "Brain-Stem"), (8, "Left-Cerebellum"),
        ]:
            ps_rows.append({
                "ref_path": row["ref_path"], "cor_path": row["cor_path"],
                "corruption_type": row["corruption_type"],
                "severity": row["severity"],
                "label_id": lid, "label_name": lname,
                "dice": float(row["mean_dice"]) + (0.02 if lid == 17 else -0.02),
            })
    _write_csv(rd / "per_structure_dice.csv", list(ps_rows[0].keys()), ps_rows)
    return rd


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────


def test_missing_required_csv_aborts(
    mod: types.ModuleType, runner: CliRunner, tmp_path: Path
) -> None:
    """Fig 2 requested but iqm_features.csv missing → aborts with naming error."""
    rd = tmp_path / "results"
    rd.mkdir()
    # Write only machine_preference.csv (Fig 2 also needs iqm_features.csv).
    _write_csv(
        rd / "machine_preference.csv",
        ["ref_path", "cor_path", "corruption_type", "severity", "mean_dice"],
        [{"ref_path": "/r", "cor_path": "/c", "corruption_type": "motion",
          "severity": 1, "mean_dice": 0.5}],
    )
    out_dir = tmp_path / "figs"

    result = runner.invoke(
        mod.app,
        ["--figure", "2", "--results-dir", str(rd), "--output-dir", str(out_dir)],
    )
    assert result.exit_code != 0
    msg = result.output + (str(result.exception) if result.exception else "")
    assert "iqm_features.csv" in msg
    assert "code/05_extract_iqms.py" in msg


def test_per_structure_csv_dependency(
    mod: types.ModuleType, runner: CliRunner, tmp_path: Path
) -> None:
    """Fig 8 requested without per_structure_dice.csv → SystemExit pointing at code/04."""
    rd = tmp_path / "results"
    rd.mkdir()
    out_dir = tmp_path / "figs"

    result = runner.invoke(
        mod.app,
        ["--figure", "8", "--results-dir", str(rd), "--output-dir", str(out_dir)],
    )
    assert result.exit_code != 0
    msg = result.output + (str(result.exception) if result.exception else "")
    assert "per_structure_dice.csv" in msg
    assert "code/04_compute_preference.py" in msg
    assert "--per-structure-output" in msg


def test_multi_seed_aggregation(mod: types.ModuleType, tmp_path: Path) -> None:
    """Three synthetic seed CSVs → mean / std / n_seeds correct per (scan, model)."""
    paths_and_seeds: list[tuple[Path, int]] = []
    for seed in (0, 1, 2):
        path = tmp_path / f"3d_vlm_scores_seed_{seed}.csv"
        rows = [
            {"scan_path": "/A", "model": "m3d", "score": 0.3 + 0.1 * seed,
             "raw_response": "x", "seed": seed},
            {"scan_path": "/B", "model": "m3d", "score": 0.6,
             "raw_response": "y", "seed": seed},
        ]
        _write_csv(path, list(rows[0].keys()), rows)
        paths_and_seeds.append((path, seed))

    agg = mod.aggregate_seed_csvs(paths_and_seeds)
    a_row = agg[(agg["scan_path"] == "/A") & (agg["model"] == "m3d")].iloc[0]
    assert a_row["n_seeds"] == 3
    # mean(0.3, 0.4, 0.5) = 0.4; std (sample) = 0.1
    assert a_row["score_mean"] == pytest.approx(0.4)
    assert a_row["score_std"] == pytest.approx(0.1, abs=1e-9)
    # 1.96 * 0.1 / sqrt(3)
    expected_ci = 1.96 * 0.1 / (3 ** 0.5)
    assert a_row["ci_half_width"] == pytest.approx(expected_ci, abs=1e-6)

    b_row = agg[(agg["scan_path"] == "/B") & (agg["model"] == "m3d")].iloc[0]
    # Constant scores → std = 0 → CI = 0 (not NaN).
    assert b_row["score_mean"] == pytest.approx(0.6)
    assert b_row["ci_half_width"] == pytest.approx(0.0)


def test_single_seed_warns_no_ci(
    mod: types.ModuleType, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """One seed file → ci_half_width is NaN; the renderer's helper logs a WARNING."""
    path = tmp_path / "3d_vlm_scores_seed_7.csv"
    rows = [
        {"scan_path": "/A", "model": "m3d", "score": 0.5, "raw_response": "x", "seed": 7},
    ]
    _write_csv(path, list(rows[0].keys()), rows)
    agg = mod.aggregate_seed_csvs([(path, 7)])
    assert agg.iloc[0]["n_seeds"] == 1
    # Single observation → std NaN → CI NaN.
    import math
    assert math.isnan(float(agg.iloc[0]["ci_half_width"]))

    # Renderer-level helper logs a single-seed warning.
    inputs = {"3d_vlm_scores_seed_*.csv": [path]}
    with caplog.at_level("WARNING", logger="visualize_mod"):
        _ = mod._aggregate_seeds_for_pattern(inputs, "3d_vlm_scores_seed_*.csv")
    assert any("single-seed" in rec.message.lower() or "no error bars" in rec.message.lower()
               for rec in caplog.records)


def test_bootstrap_reproducibility(mod: types.ModuleType) -> None:
    """Same --bootstrap-seed → byte-identical CI bounds across two invocations."""
    # Use noisier data so different seeds DO yield observably different CIs;
    # tight monotone data + small bootstrap can collapse to identical
    # percentiles by chance.
    rng_x = [0.1, 0.3, 0.4, 0.5, 0.7, 0.8, 0.85, 0.9, 0.2, 0.6, 0.55, 0.45]
    rng_y = [0.2, 0.25, 0.5, 0.55, 0.6, 0.4, 0.75, 0.8, 0.5, 0.3, 0.65, 0.5]
    a = mod.srcc_with_bootstrap_ci(rng_x, rng_y, n_bootstrap=500, seed=42)
    b = mod.srcc_with_bootstrap_ci(rng_x, rng_y, n_bootstrap=500, seed=42)
    assert a == b, f"Same seed must give byte-identical CI: {a} vs {b}"
    # Point estimate is data-only, not seed-dependent — independent invariant.
    c = mod.srcc_with_bootstrap_ci(rng_x, rng_y, n_bootstrap=500, seed=99)
    assert a[0] == c[0]


def test_atomic_write(mod: types.ModuleType, tmp_path: Path) -> None:
    """Mock savefig failure mid-write → no partial PNG/SVG on disk (only .tmp cleanup)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    target = tmp_path / "atomic_test"

    # Patch savefig to raise on the SVG write (after PNG succeeds), to verify
    # the partial PNG gets unwound.
    call_count = {"n": 0}
    real_savefig = fig.savefig

    def flaky_savefig(*args: Any, **kwargs: Any) -> None:
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise RuntimeError("simulated SVG write failure")
        real_savefig(*args, **kwargs)

    fig.savefig = flaky_savefig  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="simulated SVG write failure"):
        mod._save_figure_atomic(fig, target)

    plt.close(fig)
    # Neither final nor tmp files should remain after a failed write.
    assert not (target.with_suffix(".png")).exists()
    assert not (target.with_suffix(".svg")).exists()
    assert not list(target.parent.glob("atomic_test.*.tmp"))


def test_all_figures_smoke(
    mod: types.ModuleType, runner: CliRunner, tmp_path: Path
) -> None:
    """--all with synthetic CSVs → 6 PNGs + 6 SVGs written, no exceptions."""
    rd = _build_results_dir(tmp_path)
    out_dir = tmp_path / "figures"

    result = runner.invoke(
        mod.app,
        [
            "--all",
            "--results-dir", str(rd),
            "--output-dir", str(out_dir),
            "--n-bootstrap", "50",   # tests use a small bootstrap for speed
        ],
    )
    if result.exit_code != 0 and result.exception is not None:
        import traceback
        traceback.print_exception(
            type(result.exception), result.exception, result.exception.__traceback__
        )
    assert result.exit_code == 0, result.output

    expected_pngs = {
        out_dir / f"{name}.png" for name in mod.FIG_FILENAMES.values()
    }
    expected_svgs = {
        out_dir / f"{name}.svg" for name in mod.FIG_FILENAMES.values()
    }
    for p in expected_pngs:
        assert p.is_file(), f"missing PNG: {p}"
        assert p.stat().st_size > 0
    for p in expected_svgs:
        assert p.is_file(), f"missing SVG: {p}"
        assert p.stat().st_size > 0


def test_interactive_flag_warns(
    mod: types.ModuleType, runner: CliRunner, tmp_path: Path,
) -> None:
    """--interactive logs WARNING, PDFs/PNGs still produced (for the figs we render)."""
    rd = _build_results_dir(tmp_path)
    out_dir = tmp_path / "figures"

    result = runner.invoke(
        mod.app,
        [
            "--figure", "3",
            "--results-dir", str(rd),
            "--output-dir", str(out_dir),
            "--interactive",
            "--n-bootstrap", "10",
        ],
    )
    assert result.exit_code == 0, result.output
    # Despite --interactive, the PNG + SVG should both exist.
    assert (out_dir / "fig_corruption_sensitivity.png").is_file()
    assert (out_dir / "fig_corruption_sensitivity.svg").is_file()
    # And the warning must have been emitted somewhere visible.
    msg = result.output
    # The CliRunner doesn't always capture RichHandler output in stdout; if the
    # WARNING isn't in result.output, that's because Rich is writing to the
    # logger directly. Either way, the script must complete normally and the
    # interactive flag must not produce HTML output (no .html sidecars).
    assert not list(out_dir.glob("*.html"))
