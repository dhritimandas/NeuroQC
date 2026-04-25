"""Unit tests for code/08a_eval_3d_vlms.py.

Synthetic fixtures only — no HuggingFace model downloads, no NIfTI I/O.
The VLM adapter is a fake class that tests can inject via the module's
``_ADAPTERS`` registry.
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
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE_PATH = _REPO_ROOT / "code" / "08a_eval_3d_vlms.py"


def _load_module() -> types.ModuleType:
    """Import 08a despite the digit-prefixed filename.

    ``sys.modules`` registration is required for ``@dataclass(frozen=True)``
    to resolve forward references under ``from __future__ import annotations``.
    """
    spec = importlib.util.spec_from_file_location("eval3d_mod", _MODULE_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {_MODULE_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval3d_mod"] = module
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


def _touch_scan(base_dir: Path, dataset_tag: str, name: str) -> Path:
    """Zero-byte placeholder NIfTI. Adapter never reads it — the test
    monkey-patches the transform to return a synthetic tensor."""
    scan = base_dir / dataset_tag / name
    scan.parent.mkdir(parents=True, exist_ok=True)
    scan.write_bytes(b"")
    return scan


def _write_ref_manifest(path: Path, scan_paths: list[Path], tag: str = "fastmri") -> None:
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
                    "dataset_tag": tag,
                }
            )


def _write_cor_manifest(
    path: Path,
    pairs: list[tuple[Path, Path, str, int]],
    tag: str = "fastmri",
) -> None:
    """Write a minimal corruption manifest.

    Each tuple: (ref_path, cor_path, corruption_type, severity).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ref_path", "cor_path", "corruption_type", "corruption_domain",
        "severity", "seed", "transform_params", "dataset_tag",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for ref, cor, ctype, sev in pairs:
            writer.writerow(
                {
                    "ref_path": str(ref),
                    "cor_path": str(cor),
                    "corruption_type": ctype,
                    "corruption_domain": "image",
                    "severity": sev,
                    "seed": 0,
                    "transform_params": "{}",
                    "dataset_tag": tag,
                }
            )


# ──────────────────────────────────────────────
# Phase A — determinism, split disjointness, size
# ──────────────────────────────────────────────


def test_phase_a_deterministic(mod: types.ModuleType, tmp_path: Path) -> None:
    """Same split-seed → byte-identical subsample CSVs across invocations."""
    refs = [_touch_scan(tmp_path, "fastmri", f"r_{i:03d}.nii.gz") for i in range(20)]
    ref_manifest = tmp_path / "ref.csv"
    _write_ref_manifest(ref_manifest, refs)

    # Cor manifest: 1 corruption per ref (motion sev=1) to keep the test small.
    cor_rows: list[tuple[Path, Path, str, int]] = []
    for r in refs:
        cor = r.with_name(r.stem.replace(".nii", "") + "_cor.nii.gz")
        cor.write_bytes(b"")
        cor_rows.append((r, cor, "motion", 1))
    cor_manifest = tmp_path / "cor.csv"
    _write_cor_manifest(cor_manifest, cor_rows)

    def build() -> pd.DataFrame:
        return mod.build_subsample(
            ref_manifest_paths=[ref_manifest],
            cor_manifest_path=cor_manifest,
            preference_csv=None,
            n_refs=8,
            split_seed=42,
            severities=(1,),
            corruption_types=("motion",),
        )

    df_a = build()
    df_b = build()
    # Columns, row count, values all identical.
    assert df_a.columns.tolist() == df_b.columns.tolist()
    assert df_a.equals(df_b), "two runs with same seed must be identical"

    # Also check a different seed gives a different ref set (otherwise the
    # hash-based sample isn't actually seed-sensitive).
    df_seed_a = build()
    df_seed_b = mod.build_subsample(
        ref_manifest_paths=[ref_manifest],
        cor_manifest_path=cor_manifest,
        preference_csv=None,
        n_refs=8,
        split_seed=1337,
        severities=(1,),
        corruption_types=("motion",),
    )
    set_a = set(df_seed_a[mod.REF_ID_COLUMN])
    set_b = set(df_seed_b[mod.REF_ID_COLUMN])
    assert set_a != set_b, "different split_seed should yield a different ref set"


def test_phase_a_ref_level_split(mod: types.ModuleType, tmp_path: Path) -> None:
    """Every ref_id lives in exactly one split (no cross-split leakage)."""
    refs = [_touch_scan(tmp_path, "fastmri", f"r_{i:03d}.nii.gz") for i in range(40)]
    ref_manifest = tmp_path / "ref.csv"
    _write_ref_manifest(ref_manifest, refs)

    cor_rows: list[tuple[Path, Path, str, int]] = []
    for r in refs:
        for ctype, sev in [("motion", 1), ("motion", 3)]:
            cor = r.with_name(r.stem.replace(".nii", "") + f"_{ctype}_{sev}.nii.gz")
            cor.write_bytes(b"")
            cor_rows.append((r, cor, ctype, sev))
    cor_manifest = tmp_path / "cor.csv"
    _write_cor_manifest(cor_manifest, cor_rows)

    df = mod.build_subsample(
        ref_manifest_paths=[ref_manifest],
        cor_manifest_path=cor_manifest,
        preference_csv=None,
        n_refs=20,
        split_seed=42,
        severities=(1, 3),
        corruption_types=("motion",),
    )

    # Group by ref_id; each group must have exactly one unique split label.
    splits_per_ref = df.groupby(mod.REF_ID_COLUMN)[mod.SPLIT_COLUMN].nunique()
    assert (splits_per_ref == 1).all(), (
        "a ref_id cannot appear in multiple splits; "
        f"offenders: {splits_per_ref[splits_per_ref > 1].index.tolist()}"
    )
    # And all three splits are non-empty at 20-ref scale.
    assert set(df[mod.SPLIT_COLUMN]) == {"train", "val", "test"}


def test_phase_a_subsample_size(mod: types.ModuleType, tmp_path: Path) -> None:
    """n_refs=10, 2 severities, 8 types → 10 × (1 clean + 2×8 cor) = 170 rows."""
    refs = [_touch_scan(tmp_path, "fastmri", f"r_{i:03d}.nii.gz") for i in range(15)]
    ref_manifest = tmp_path / "ref.csv"
    _write_ref_manifest(ref_manifest, refs)

    # Full grid of 2 severities × 8 types per ref.
    corruption_types = (
        "motion", "ghosting", "spike", "noise",
        "bias_field", "blur", "downsample", "gamma",
    )
    severities = (1, 3)
    cor_rows: list[tuple[Path, Path, str, int]] = []
    for r in refs:
        for ctype in corruption_types:
            for sev in severities:
                cor = r.with_name(
                    r.stem.replace(".nii", "") + f"_{ctype}_{sev}.nii.gz"
                )
                cor.write_bytes(b"")
                cor_rows.append((r, cor, ctype, sev))
    cor_manifest = tmp_path / "cor.csv"
    _write_cor_manifest(cor_manifest, cor_rows)

    df = mod.build_subsample(
        ref_manifest_paths=[ref_manifest],
        cor_manifest_path=cor_manifest,
        preference_csv=None,
        n_refs=10,
        split_seed=42,
        severities=severities,
        corruption_types=corruption_types,
    )
    assert len(df) == 10 * (1 + 2 * 8), f"expected 170 rows, got {len(df)}"
    n_clean = int(df[mod.IS_CLEAN_COLUMN].sum())
    assert n_clean == 10
    # Every cor row has corruption_type != none and severity != 0.
    cor_df = df[~df[mod.IS_CLEAN_COLUMN]]
    assert (cor_df[mod.TYPE_COLUMN] != mod._NONE_TYPE).all()
    assert (cor_df[mod.SEVERITY_COLUMN] > 0).all()


# ──────────────────────────────────────────────
# Phase B — resume + score handling
# ──────────────────────────────────────────────


@dataclass
class FakeAdapter:
    """Deterministic fake VLM for testing the Phase B loop in isolation.

    The ``scripted_response`` callable receives the scan's path (as a
    string) and returns either a raw string (normal path) or raises an
    exception (tests the crash-handling branches).
    """

    scripted_response: Callable[[str], str] = field(
        default=lambda _: "SCORE: 3\nREASON: ok"
    )
    name: str = "m3d_lamed"
    last_vol: torch.Tensor | None = None
    # We need to retrieve the scan path inside run_inference. Pass it
    # through the tensor's ``.scan_path`` attribute monkey-patched by the
    # fake transform below.

    def load(self, device: torch.device, dtype: torch.dtype) -> None:
        pass

    def run_inference(
        self, vol_cdhw: torch.Tensor, max_new_tokens: int = 16
    ) -> str:
        self.last_vol = vol_cdhw
        scan_path = getattr(vol_cdhw, "_scan_path", "UNKNOWN")
        return self.scripted_response(scan_path)

    def parse_score(self, raw: str) -> float:
        # Import lazily from the module-under-test to stay in sync with its
        # normalisation rule.
        return _score_helper(raw)

    def unload(self) -> None:
        self.last_vol = None


# Capture the module-under-test's normalisation once; used by FakeAdapter.
_score_helper: Callable[[str], float] = lambda raw: math.nan


def _fake_transform(_: str) -> torch.Tensor:
    """Return a (1, 256, 256, 32) tensor carrying its scan_path in an attr."""
    t = torch.zeros((1, 256, 256, 32), dtype=torch.float32)
    return t


def test_resume_skips_done_pairs(
    mod: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(scan_x, m3d_lamed) pre-seeded in the output → not re-run."""
    global _score_helper
    _score_helper = mod._normalize_qc_score

    scan_a = str((tmp_path / "a.nii.gz").resolve())
    scan_b = str((tmp_path / "b.nii.gz").resolve())
    (tmp_path / "a.nii.gz").write_bytes(b"")
    (tmp_path / "b.nii.gz").write_bytes(b"")

    # Pre-seed output with scan_a already done.
    output_file = tmp_path / "scores.csv"
    with output_file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mod.OUTPUT_COLUMNS))
        writer.writeheader()
        writer.writerow(
            {
                mod.SCAN_COLUMN: scan_a,
                mod.MODEL_COLUMN: "m3d_lamed",
                mod.SCORE_COLUMN: 0.5,
                mod.RAW_RESPONSE_COLUMN: "SCORE: 3",
                mod.SEED_COLUMN: 7,
            }
        )

    existing = mod.load_existing(output_file)
    assert (scan_a, "m3d_lamed") in existing
    assert (scan_b, "m3d_lamed") not in existing

    # Simulate a run that would do both: filter out done pairs.
    candidates = [(scan_a, "m3d_lamed"), (scan_b, "m3d_lamed")]
    pending = [c for c in candidates if c not in existing]
    assert pending == [(scan_b, "m3d_lamed")]


def test_parse_failure_is_nan(mod: types.ModuleType, tmp_path: Path) -> None:
    """Model returns 'garbage' → score=NaN, raw_response preserved verbatim."""
    fake = FakeAdapter(scripted_response=lambda _: "complete garbage no digit here")
    transform = lambda _p: _fake_transform(_p)  # noqa: E731
    device = torch.device("cpu")
    dtype = torch.float32

    scan = tmp_path / "scan.nii.gz"
    scan.write_bytes(b"")

    score, raw = mod.score_one_scan(
        fake, scan, transform, device=device, dtype=dtype
    )
    assert math.isnan(score), f"expected NaN score, got {score}"
    assert raw == "complete garbage no digit here"


def test_oom_recovery(mod: types.ModuleType, tmp_path: Path) -> None:
    """OutOfMemoryError → row written with score=NaN, raw_response='OOM'."""
    def boom(_: str) -> str:
        raise torch.cuda.OutOfMemoryError("synthetic OOM")

    fake = FakeAdapter(scripted_response=boom)
    transform = lambda _p: _fake_transform(_p)  # noqa: E731
    scan = tmp_path / "scan.nii.gz"
    scan.write_bytes(b"")

    score, raw = mod.score_one_scan(
        fake, scan, transform, device=torch.device("cpu"), dtype=torch.float32
    )
    assert math.isnan(score)
    assert raw == "OOM"

    # And any other Exception is classified as ERROR:... without crashing.
    fake_err = FakeAdapter(
        scripted_response=lambda _: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    score, raw = mod.score_one_scan(
        fake_err, scan, transform, device=torch.device("cpu"), dtype=torch.float32
    )
    assert math.isnan(score)
    assert raw.startswith("ERROR:")
