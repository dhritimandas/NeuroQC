"""Unit tests for code/08b_eval_2d_vlms.py.

Synthetic fixtures only — no VLM downloads, no OpenAI calls. Adapters are
replaced via the module-level ``_ADAPTERS`` registry with a FakeAdapter
that captures the `slices`/`prompt` arguments so the tests can assert the
multi-image protocol without loading any model weights.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest
import torch
from PIL import Image
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE_PATH = _REPO_ROOT / "code" / "08b_eval_2d_vlms.py"


def _load_module() -> types.ModuleType:
    """Import 08b despite the digit-prefixed filename.

    sys.modules registration required for `@dataclass(frozen=True)` to
    resolve forward refs under `from __future__ import annotations`.
    """
    spec = importlib.util.spec_from_file_location("eval2d_mod", _MODULE_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {_MODULE_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval2d_mod"] = module
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


def _write_subsample(path: Path, scans: list[Path]) -> None:
    """Write a minimal benchmark_subsample.csv (only scan_path matters here)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ref_id", "scan_path", "is_clean", "corruption_type",
                "severity", "dataset_tag", "split", "preference_score",
            ],
        )
        writer.writeheader()
        for i, scan in enumerate(scans):
            writer.writerow(
                {
                    "ref_id": f"scan_{i}",
                    "scan_path": str(scan),
                    "is_clean": True,
                    "corruption_type": "none",
                    "severity": 0,
                    "dataset_tag": "fastmri",
                    "split": "train",
                    "preference_score": "",
                }
            )


def _write_cached_slices(cache_dir: Path, strategy: str, scan_stem: str) -> list[Path]:
    """Pre-write 3 PNG slices so the cache-hit path is exercised."""
    paths: list[Path] = []
    for i in range(3):
        path = cache_dir / strategy / f"{scan_stem}_slice{i}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), color=(i * 50, i * 50, i * 50)).save(path)
        paths.append(path)
    return paths


def _touch_scan(dir_: Path, name: str) -> Path:
    """Zero-byte scan NIfTI placeholder. Tests mock slice extraction, so
    the scan is never actually read."""
    path = dir_ / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


@dataclass
class FakeAdapter:
    """Injectable stand-in for all 4 real adapters.

    ``scripted_response(scan_path, strategy, slices)`` returns the raw
    string OR raises; tests drive the Phase B code paths by swapping in
    different scripted_response callables.
    """

    name: str
    multi_image_mode: str = "multi_image"
    scripted_response: Callable[[str, str, list[Image.Image]], str] = field(
        default=lambda _sp, _strat, _sl: "SCORE: 3\nREASON: ok"
    )
    loaded: bool = False
    _captured_slices: list[Image.Image] = field(default_factory=list)

    def load(self, device: torch.device, dtype: torch.dtype) -> None:
        self.loaded = True

    def run_inference(self, slices: list[Image.Image], max_new_tokens: int = 16) -> str:
        self._captured_slices = list(slices)
        # Recover scan path from the first slice's .info if present; tests
        # that care about scan-path routing set it there.
        scan = slices[0].info.get("scan_path", "UNKNOWN") if slices else "UNKNOWN"
        strat = slices[0].info.get("strategy", "UNKNOWN") if slices else "UNKNOWN"
        return self.scripted_response(scan, strat, slices)

    def parse_score(self, raw: str) -> float:
        # Use the module-under-test's normaliser. Tests import `mod` so they
        # can reach it; set here via injection before instantiation.
        return _score_helper(raw)

    def unload(self) -> None:
        self.loaded = False


# Bridge between module-under-test's normaliser and the FakeAdapter.
_score_helper: Callable[[str], float] = lambda raw: math.nan


def _install_fake_adapter(
    mod: types.ModuleType,
    name: str,
    *,
    multi_image_mode: str = "multi_image",
    scripted_response: Callable[[str, str, list[Image.Image]], str] | None = None,
) -> None:
    """Replace `mod._ADAPTERS[name]` with a factory producing FakeAdapter."""
    global _score_helper
    _score_helper = mod._normalize_qc_score

    def factory() -> FakeAdapter:
        return FakeAdapter(
            name=name,
            multi_image_mode=multi_image_mode,
            scripted_response=scripted_response
            or (lambda _sp, _strat, _sl: "SCORE: 3\nREASON: ok"),
        )

    mod._ADAPTERS[name] = factory


def _install_slice_extract_bypass(
    mod: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replace extract_and_cache_slices with a deterministic PIL-only stub.

    Returns 3 tiny RGB images and tags them with (scan_path, strategy)
    via PIL's ``.info`` dict so FakeAdapter can recover routing info
    without reading any NIfTI.
    """

    def fake(
        scan_path: Path, strategy: str, cache_dir: Path
    ) -> list[Image.Image]:
        imgs: list[Image.Image] = []
        for i in range(3):
            im = Image.new("RGB", (8, 8), color=(i * 40, i * 40, i * 40))
            im.info["scan_path"] = str(scan_path)
            im.info["strategy"] = strategy
            imgs.append(im)
        return imgs

    monkeypatch.setattr(mod, "extract_and_cache_slices", fake)


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────


def test_missing_subsample_aborts(
    mod: types.ModuleType, runner: CliRunner, tmp_path: Path
) -> None:
    """No benchmark_subsample.csv → exit non-zero with an actionable error."""
    missing = tmp_path / "does_not_exist.csv"
    output = tmp_path / "scores.csv"

    result = runner.invoke(
        mod.app,
        [
            "--seed", "7",
            "--subsample-manifest", str(missing),
            "--output-file", str(output),
        ],
    )
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "benchmark_subsample.csv not found" in combined
    # Error points at 08a Phase A so the user knows what to do.
    assert "08a" in combined


def test_slice_cache_hit(mod: types.ModuleType, tmp_path: Path) -> None:
    """Pre-written PNGs → extract_slices never called."""
    scan = _touch_scan(tmp_path, "subj001.nii.gz")
    cache_dir = tmp_path / "slices"
    _write_cached_slices(cache_dir, "mid", _scan_stem := mod._scan_stem(scan))

    # If extract_and_cache_slices reaches the cache-miss branch it will try
    # to import nobrainer.qc.slice_extractor and parse the zero-byte NIfTI
    # — which fails loudly. So a successful call here means we hit the cache.
    slices = mod.extract_and_cache_slices(scan, "mid", cache_dir)
    assert len(slices) == 3
    for s in slices:
        assert isinstance(s, Image.Image)
        assert s.mode == "RGB"


def test_resume_skips_done_triples(
    mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(scan, llava_ov, mid) pre-seeded → that triple is not re-run; but
    (scan, llava_ov, max_info) still runs."""
    scan = _touch_scan(tmp_path, "subj001.nii.gz")
    subsample = tmp_path / "bench.csv"
    _write_subsample(subsample, [scan])

    output = tmp_path / "scores.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mod.OUTPUT_COLUMNS))
        writer.writeheader()
        writer.writerow(
            {
                mod.SCAN_COLUMN: str(scan),
                mod.MODEL_COLUMN: "llava_ov",
                mod.STRATEGY_COLUMN: "mid",
                mod.SCORE_COLUMN: 0.5,
                mod.RAW_RESPONSE_COLUMN: "SCORE: 3",
                mod.SEED_COLUMN: 7,
                mod.N_SLICES_COLUMN: 3,
                mod.MULTI_IMAGE_MODE_COLUMN: "multi_image",
            }
        )

    existing = mod.load_existing(output)
    assert (str(scan), "llava_ov", "mid") in existing
    assert (str(scan), "llava_ov", "max_info") not in existing

    # Install a fake adapter and extraction bypass, then run the CLI; only
    # (scan, llava_ov, max_info) should be processed this time.
    _install_fake_adapter(mod, "llava_ov", multi_image_mode="multi_image")
    _install_slice_extract_bypass(mod, monkeypatch)

    result = runner.invoke(
        mod.app,
        [
            "--seed", "7",
            "--subsample-manifest", str(subsample),
            "--output-file", str(output),
            "--models", "llava_ov",
            "--strategies", "mid,max_info",
            "--slice-cache-dir", str(tmp_path / "cache"),
        ],
    )
    assert result.exit_code == 0, result.output
    df = pd.read_csv(output)
    assert len(df) == 2
    strategies_done = set(df[mod.STRATEGY_COLUMN])
    assert strategies_done == {"mid", "max_info"}


def test_oom_recovery(
    mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """torch.cuda.OutOfMemoryError → row with raw_response='OOM', score=NaN."""
    scan = _touch_scan(tmp_path, "subj001.nii.gz")
    subsample = tmp_path / "bench.csv"
    _write_subsample(subsample, [scan])
    output = tmp_path / "scores.csv"

    def oom(_sp: str, _strat: str, _sl: list[Image.Image]) -> str:
        raise torch.cuda.OutOfMemoryError("synthetic OOM")

    _install_fake_adapter(mod, "llava_ov", scripted_response=oom)
    _install_slice_extract_bypass(mod, monkeypatch)

    result = runner.invoke(
        mod.app,
        [
            "--seed", "7",
            "--subsample-manifest", str(subsample),
            "--output-file", str(output),
            "--models", "llava_ov",
            "--strategies", "mid",
            "--slice-cache-dir", str(tmp_path / "cache"),
        ],
    )
    assert result.exit_code == 0, result.output
    df = pd.read_csv(output)
    assert len(df) == 1
    row = df.iloc[0]
    assert row[mod.RAW_RESPONSE_COLUMN] == "OOM"
    assert math.isnan(row[mod.SCORE_COLUMN])


def test_parse_failure_is_nan(
    mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adapter returns unparseable text → score=NaN, raw_response preserved."""
    scan = _touch_scan(tmp_path, "subj001.nii.gz")
    subsample = tmp_path / "bench.csv"
    _write_subsample(subsample, [scan])
    output = tmp_path / "scores.csv"

    _install_fake_adapter(
        mod,
        "llava_ov",
        scripted_response=lambda _sp, _strat, _sl: "no digits here at all",
    )
    _install_slice_extract_bypass(mod, monkeypatch)

    result = runner.invoke(
        mod.app,
        [
            "--seed", "7",
            "--subsample-manifest", str(subsample),
            "--output-file", str(output),
            "--models", "llava_ov",
            "--strategies", "mid",
            "--slice-cache-dir", str(tmp_path / "cache"),
        ],
    )
    assert result.exit_code == 0, result.output
    df = pd.read_csv(output)
    assert len(df) == 1
    row = df.iloc[0]
    assert math.isnan(row[mod.SCORE_COLUMN])
    assert row[mod.RAW_RESPONSE_COLUMN] == "no digits here at all"


def test_gpt4o_budget_cap(
    mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--max-api-calls=2 on GPT-4o → only 2 real rows; remainder skipped."""
    scans = [_touch_scan(tmp_path, f"subj{i:03d}.nii.gz") for i in range(4)]
    subsample = tmp_path / "bench.csv"
    _write_subsample(subsample, scans)
    output = tmp_path / "scores.csv"

    # Provide a working OPENAI_API_KEY so the skip-on-absent path doesn't fire.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    # Monkeypatch GPT4oAdapter.load so it installs a stub client, and
    # patch run_inference to exercise the budget-check branch deterministically.
    class _StubClient:
        class chat:
            class completions:
                @staticmethod
                def create(**_kwargs: Any) -> Any:
                    return types.SimpleNamespace(
                        choices=[
                            types.SimpleNamespace(
                                message=types.SimpleNamespace(content="SCORE: 4")
                            )
                        ]
                    )

    def fake_load(self: Any, device: Any, dtype: Any) -> None:
        del device, dtype
        self.client = _StubClient()
        self._no_api_key = False

    monkeypatch.setattr(mod.GPT4oAdapter, "load", fake_load)
    _install_slice_extract_bypass(mod, monkeypatch)

    result = runner.invoke(
        mod.app,
        [
            "--seed", "7",
            "--subsample-manifest", str(subsample),
            "--output-file", str(output),
            "--models", "gpt4o",
            "--strategies", "mid",
            "--slice-cache-dir", str(tmp_path / "cache"),
            "--max-api-calls", "2",
        ],
    )
    assert result.exit_code == 0, result.output

    df = pd.read_csv(output)
    # Exactly 2 "real" rows (SCORE: 4 → 0.75). After that, the budget cap
    # kicks in: on the 3rd pending scan run_inference sees call_count >=
    # max_calls, flips budget_exhausted=True, and returns BUDGET_CAP; the
    # main loop writes that sentinel row then breaks. So we expect at most
    # 3 total rows (2 real + 1 BUDGET_CAP). Critically, real calls must
    # stop at the budget cap, not keep going.
    real_rows = df[df[mod.RAW_RESPONSE_COLUMN] == "SCORE: 4"]
    assert len(real_rows) == 2, (
        f"expected 2 real calls given max-api-calls=2, got {len(real_rows)}"
    )
    assert len(df) <= 3, (
        f"budget cap should halt model loop; wrote {len(df)} rows for 4 pending"
    )
    # The budget-exhausted sentinel appears among non-real rows (if any).
    non_real = df[df[mod.RAW_RESPONSE_COLUMN] != "SCORE: 4"]
    if len(non_real) > 0:
        assert (non_real[mod.RAW_RESPONSE_COLUMN] == mod._BUDGET_SENTINEL).all()
        # Budget-sentinel rows must parse to NaN.
        assert non_real[mod.SCORE_COLUMN].isna().all()


def test_gpt4o_no_api_key(
    mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPENAI_API_KEY unset → GPT-4o skipped with warning; other models run."""
    scan = _touch_scan(tmp_path, "subj001.nii.gz")
    subsample = tmp_path / "bench.csv"
    _write_subsample(subsample, [scan])
    output = tmp_path / "scores.csv"

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _install_fake_adapter(mod, "llava_ov")
    _install_slice_extract_bypass(mod, monkeypatch)

    result = runner.invoke(
        mod.app,
        [
            "--seed", "7",
            "--subsample-manifest", str(subsample),
            "--output-file", str(output),
            "--models", "gpt4o,llava_ov",
            "--strategies", "mid",
            "--slice-cache-dir", str(tmp_path / "cache"),
        ],
    )
    assert result.exit_code == 0, result.output

    df = pd.read_csv(output)
    # llava_ov row must exist (other models still run even when GPT-4o
    # skips). The gpt4o row, if written, must carry the NO_API_KEY sentinel
    # and parse to NaN — never a real score.
    llava_rows = df[df[mod.MODEL_COLUMN] == "llava_ov"]
    assert len(llava_rows) == 1
    gpt_rows = df[df[mod.MODEL_COLUMN] == "gpt4o"]
    if len(gpt_rows) > 0:
        assert (gpt_rows[mod.RAW_RESPONSE_COLUMN] == mod._NO_API_KEY_SENTINEL).all()
        assert gpt_rows[mod.SCORE_COLUMN].isna().all()


def test_multi_image_protocol_logged(
    mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each row's multi_image_mode reflects the adapter's declared protocol."""
    scan = _touch_scan(tmp_path, "subj001.nii.gz")
    subsample = tmp_path / "bench.csv"
    _write_subsample(subsample, [scan])
    output = tmp_path / "scores.csv"

    _install_fake_adapter(mod, "llava_ov", multi_image_mode="multi_image")
    _install_fake_adapter(mod, "medgemma", multi_image_mode="concat_grid")
    _install_slice_extract_bypass(mod, monkeypatch)

    result = runner.invoke(
        mod.app,
        [
            "--seed", "7",
            "--subsample-manifest", str(subsample),
            "--output-file", str(output),
            "--models", "llava_ov,medgemma",
            "--strategies", "mid",
            "--slice-cache-dir", str(tmp_path / "cache"),
        ],
    )
    assert result.exit_code == 0, result.output
    df = pd.read_csv(output)
    # Rows: 1 scan × 2 models × 1 strategy = 2 rows.
    assert len(df) == 2
    by_model = df.set_index(mod.MODEL_COLUMN)
    assert by_model.loc["llava_ov", mod.MULTI_IMAGE_MODE_COLUMN] == "multi_image"
    assert by_model.loc["medgemma", mod.MULTI_IMAGE_MODE_COLUMN] == "concat_grid"
    # n_slices always 3 in the fake pipeline (3 PIL images in, 3 reported).
    assert int(by_model.loc["llava_ov", mod.N_SLICES_COLUMN]) == 3
    assert int(by_model.loc["medgemma", mod.N_SLICES_COLUMN]) == 3
