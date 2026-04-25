"""Unit tests for code/09_finetune_lora.py.

Synthetic fixtures only — no HuggingFace model downloads, no NIfTI I/O,
no real ``Trainer.train`` invocation. The adapter is a fake injected via
``_FT_ADAPTERS`` and the trainer is a fake injected via ``_build_trainer``.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE_PATH = _REPO_ROOT / "code" / "09_finetune_lora.py"


def _load_module() -> types.ModuleType:
    """Import 09 despite the digit-prefixed filename.

    ``sys.modules`` registration required for ``@dataclass`` field resolution
    under ``from __future__ import annotations``.
    """
    spec = importlib.util.spec_from_file_location("ft_mod", _MODULE_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {_MODULE_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["ft_mod"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> types.ModuleType:
    return _load_module()


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ──────────────────────────────────────────────
# Synthetic CSV fixtures
# ──────────────────────────────────────────────


def _write_subsample(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write a benchmark_subsample.csv with the schema 08a Phase A produces."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ref_id", "scan_path", "is_clean",
        "corruption_type", "severity", "dataset_tag",
        "split", "preference_score",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_preference(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    """Write a machine_preference.csv with per-pair (ref_path, cor_path) rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ref_path", "cor_path", "corruption_type", "severity",
        "mean_dice", "ref_mean_thickness", "cor_mean_thickness", "thickness_shift",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_thickness(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    """Write a cortical_thickness.csv with the per-scan schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scan_path", "mean_thickness"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_synthetic_csvs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write a small set of paired CSVs for end-to-end tests.

    2 refs × (1 clean + 2 cor) = 6 subsample rows; 4 cor rows in preference
    (one per (ref, cor) pair); 6 thickness rows (one per scan).
    """
    subsample_path = tmp_path / "subsample.csv"
    preference_path = tmp_path / "preference.csv"
    thickness_path = tmp_path / "thickness.csv"

    rows_subsample = [
        # ref A (train split): 1 clean + 2 cor
        {"ref_id": "refA", "scan_path": "/ref/A.nii.gz", "is_clean": True,
         "corruption_type": "none", "severity": 0, "dataset_tag": "fastmri",
         "split": "train", "preference_score": ""},
        {"ref_id": "refA", "scan_path": "/cor/A_motion_1.nii.gz", "is_clean": False,
         "corruption_type": "motion", "severity": 1, "dataset_tag": "fastmri",
         "split": "train", "preference_score": 0.85},
        {"ref_id": "refA", "scan_path": "/cor/A_motion_3.nii.gz", "is_clean": False,
         "corruption_type": "motion", "severity": 3, "dataset_tag": "fastmri",
         "split": "train", "preference_score": 0.55},
        # ref B (test split): 1 clean + 2 cor
        {"ref_id": "refB", "scan_path": "/ref/B.nii.gz", "is_clean": True,
         "corruption_type": "none", "severity": 0, "dataset_tag": "fastmri",
         "split": "test", "preference_score": ""},
        {"ref_id": "refB", "scan_path": "/cor/B_motion_1.nii.gz", "is_clean": False,
         "corruption_type": "motion", "severity": 1, "dataset_tag": "fastmri",
         "split": "test", "preference_score": 0.92},
        {"ref_id": "refB", "scan_path": "/cor/B_motion_5.nii.gz", "is_clean": False,
         "corruption_type": "motion", "severity": 5, "dataset_tag": "fastmri",
         "split": "test", "preference_score": 0.30},
    ]
    _write_subsample(subsample_path, rows_subsample)

    rows_preference = [
        {"ref_path": "/ref/A.nii.gz", "cor_path": "/cor/A_motion_1.nii.gz",
         "corruption_type": "motion", "severity": 1, "mean_dice": 0.85,
         "ref_mean_thickness": 4.0, "cor_mean_thickness": 3.92,
         "thickness_shift": 0.08},
        {"ref_path": "/ref/A.nii.gz", "cor_path": "/cor/A_motion_3.nii.gz",
         "corruption_type": "motion", "severity": 3, "mean_dice": 0.55,
         "ref_mean_thickness": 4.0, "cor_mean_thickness": 4.20,
         "thickness_shift": 0.20},
        {"ref_path": "/ref/B.nii.gz", "cor_path": "/cor/B_motion_1.nii.gz",
         "corruption_type": "motion", "severity": 1, "mean_dice": 0.92,
         "ref_mean_thickness": 5.0, "cor_mean_thickness": 4.95,
         "thickness_shift": 0.05},
        {"ref_path": "/ref/B.nii.gz", "cor_path": "/cor/B_motion_5.nii.gz",
         "corruption_type": "motion", "severity": 5, "mean_dice": 0.30,
         "ref_mean_thickness": 5.0, "cor_mean_thickness": 5.65,
         "thickness_shift": 0.65},
    ]
    _write_preference(preference_path, rows_preference)

    rows_thickness = [
        {"scan_path": "/ref/A.nii.gz", "mean_thickness": 4.0},
        {"scan_path": "/cor/A_motion_1.nii.gz", "mean_thickness": 3.92},
        {"scan_path": "/cor/A_motion_3.nii.gz", "mean_thickness": 4.20},
        {"scan_path": "/ref/B.nii.gz", "mean_thickness": 5.0},
        {"scan_path": "/cor/B_motion_1.nii.gz", "mean_thickness": 4.95},
        {"scan_path": "/cor/B_motion_5.nii.gz", "mean_thickness": 5.65},
    ]
    _write_thickness(thickness_path, rows_thickness)

    return subsample_path, preference_path, thickness_path


# ──────────────────────────────────────────────
# Fake adapter for monkeypatching
# ──────────────────────────────────────────────


@dataclass
class _FakeAdapter:
    """Test-only FineTuneAdapter that returns dummy tensors and mock model/processor."""

    name: str = "fake"
    hf_id: str = "fake/model"
    input_type: str = "3d"
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    vision_encoder_module: str = "vision_tower"
    _loaded: bool = False
    _model: Any = None
    _processor: Any = None

    @property
    def model(self) -> Any:
        return self._model

    @property
    def processor(self) -> Any:
        return self._processor

    def load(
        self,
        device: torch.device,
        dtype: torch.dtype,
        bnb_config: Any,
        llm_lora_config: Any,
        vision_lora_config: Any | None,
    ) -> None:
        self._loaded = True
        self._model = MagicMock(name="fake_model")
        # generate() returns a tensor that decode()s to a parseable string.
        self._model.generate = MagicMock(return_value=torch.tensor([[0, 1, 2]]))
        self._model.eval = MagicMock(return_value=self._model)
        self._model.parameters = MagicMock(return_value=[])
        self._processor = MagicMock(name="fake_processor")
        self._processor.decode = MagicMock(return_value="Quality: 4 Thickness: 3")

    def prepare_inputs(
        self, scan_path: Path, target_text: str
    ) -> dict[str, torch.Tensor]:
        ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
            "pixel_values": torch.zeros((1, 32, 256, 256)),
        }

    def collate_fn(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        }

    def unload(self) -> None:
        self._loaded = False
        self._model = None
        self._processor = None


def _install_fake_adapter(mod: types.ModuleType, model_name: str = "m3d_lamed") -> None:
    """Replace one entry in ``_FT_ADAPTERS`` with a FakeAdapter factory."""
    mod._FT_ADAPTERS[model_name] = _FakeAdapter


class _FakeTrainer:
    """Test-only Trainer stub recording calls to ``train`` and returning a stub state."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.train_calls: list[Any] = []
        self.state = types.SimpleNamespace(
            log_history=[
                {"epoch": 1, "eval_srcc": 0.5, "eval_loss": 0.1},
                {"epoch": 2, "eval_srcc": 0.7, "eval_loss": 0.08},
            ]
        )

    def train(self, resume_from_checkpoint: Any = None) -> Any:
        self.train_calls.append(resume_from_checkpoint)
        return MagicMock(metrics={"train_loss": 0.1})

    def evaluate(self) -> dict[str, float]:
        return {"eval_loss": 0.1, "eval_srcc": 0.7}

    def save_model(self, *args: Any, **kwargs: Any) -> None:
        pass


def _install_fake_trainer(mod: types.ModuleType) -> list[_FakeTrainer]:
    """Replace ``_build_trainer`` with a factory returning a recorded FakeTrainer."""
    instances: list[_FakeTrainer] = []

    def factory(**kwargs: Any) -> _FakeTrainer:
        t = _FakeTrainer(**kwargs)
        instances.append(t)
        return t

    mod._build_trainer = factory
    return instances


def _bypass_preconditions(mod: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the PEFT version + parser availability hard aborts for tests."""
    monkeypatch.setattr(mod, "assert_peft_version", lambda: None)
    monkeypatch.setattr(mod, "assert_dual_parser_available", lambda: None)


def _bypass_score_test_split(mod: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid test-split scoring (which calls model.generate)."""
    monkeypatch.setattr(
        mod, "score_test_split",
        lambda **kwargs: (0, 0),
    )


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────


def test_split_validation_no_ref_overlap(mod: types.ModuleType) -> None:
    """A ref_id that appears in two splits triggers ValueError."""
    df = pd.DataFrame(
        {
            "ref_id": ["A", "A", "A", "B"],
            "scan_path": ["/a1", "/a2", "/a3", "/b1"],
            "is_clean": [True, False, False, True],
            "split": ["train", "train", "test", "val"],  # A leaks across train/test
        }
    )
    with pytest.raises(ValueError, match="ref_id leaked"):
        mod.validate_split(df)


def test_thickness_shift_derivation(mod: types.ModuleType) -> None:
    """Per-ref clean baseline → signed shifts (negative thinning, positive thickening)."""
    df = pd.DataFrame(
        {
            mod.REF_ID_COLUMN: ["A", "A", "A", "B", "B"],
            mod.SCAN_COLUMN: ["/a/c", "/a/cor1", "/a/cor2", "/b/c", "/b/cor1"],
            mod.IS_CLEAN_COLUMN: [True, False, False, True, False],
            mod._THICKNESS_MEAN: [4.0, 3.5, 4.7, 5.0, 5.0],
            mod.SPLIT_COLUMN: ["train"] * 3 + ["test"] * 2,
        }
    )
    out = mod.derive_signed_thickness_shift(df)
    # ref A clean baseline = 4.0
    a_rows = out[out[mod.REF_ID_COLUMN] == "A"].sort_values(mod.SCAN_COLUMN)
    shifts_a = a_rows[mod._DERIVED_SIGNED_SHIFT].tolist()
    # /a/c=0, /a/cor1=-0.5 (thinning), /a/cor2=+0.7 (thickening)
    expected_a = {0.0, -0.5, 0.7000000000000002}
    assert set(round(s, 4) for s in shifts_a) == set(round(s, 4) for s in expected_a)
    # ref B clean baseline = 5.0; cor at 5.0 → shift = 0
    b_rows = out[out[mod.REF_ID_COLUMN] == "B"]
    assert all(abs(s) < 1e-9 for s in b_rows[mod._DERIVED_SIGNED_SHIFT].tolist())


def test_dice_bucket_boundaries(mod: types.ModuleType) -> None:
    """Verify exact boundary handling at 0.95 / 0.90 / 0.80 / 0.60."""
    cases = [
        (1.0, 5),
        (0.95, 5),
        (0.9499999, 4),
        (0.90, 4),
        (0.8999999, 3),
        (0.80, 3),
        (0.7999999, 2),
        (0.60, 2),
        (0.5999999, 1),
        (0.0, 1),
    ]
    for value, expected in cases:
        got = mod.discretize_dice(value)
        assert got == expected, f"discretize_dice({value}) = {got}, expected {expected}"
    # NaN preserved as None
    assert mod.discretize_dice(float("nan")) is None
    assert mod.discretize_dice(None) is None


def test_target_modules_per_model(mod: types.ModuleType) -> None:
    """Registry is well-formed for all 4 models; target_modules non-empty."""
    expected_models = {"m3d_lamed", "llava_ov", "qwen2_vl", "medgemma"}
    assert set(mod._FT_REGISTRY.keys()) == expected_models
    assert set(mod._FT_ADAPTERS.keys()) == expected_models
    for name, cfg in mod._FT_REGISTRY.items():
        assert cfg["target_modules"], f"empty target_modules for {name}"
        assert isinstance(cfg["target_modules"], list)
        assert all(isinstance(t, str) for t in cfg["target_modules"])
        assert cfg["vision_encoder_module"], f"empty vision_encoder_module for {name}"
        assert cfg["input_type"] in ("2d", "3d"), f"invalid input_type for {name}"
        assert cfg["hf_id"], f"empty hf_id for {name}"


def test_provenance_clean_tree(
    mod: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty git status → JSON.git_dirty=False, no diff file written."""
    monkeypatch.setattr(mod, "_git_status_porcelain", lambda: "")
    monkeypatch.setattr(mod, "_git_rev_parse_head", lambda: "deadbeef" * 5)
    monkeypatch.setattr(mod, "_git_diff_head", lambda: "")
    monkeypatch.setattr(mod, "_library_versions", lambda: {"torch": "1.0"})

    json_path = tmp_path / "info.json"
    diff_path = tmp_path / "info.diff"
    provenance = mod.build_provenance(
        seed=0,
        model_name="m3d_lamed",
        hyperparameters={"lr": 2e-5},
        data_splits={"n_train": 10, "n_val": 2, "n_test": 2},
        bucket_distributions={"dice_train": {5: 5}},
        start_time="2026-04-25T00:00:00Z",
        end_time="2026-04-25T00:01:00Z",
        best_val_srcc=0.7,
        best_epoch=2,
        trainable_params=100,
        total_params=1000,
        diff_path=diff_path,
    )
    mod.save_provenance(json_path, provenance, diff_path=diff_path)

    assert json_path.is_file()
    payload = json.loads(json_path.read_text())
    assert payload["git_dirty"] is False
    assert payload["git_diff_path"] is None
    assert not diff_path.exists()


def test_provenance_dirty_tree(
    mod: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-empty git status → JSON.git_dirty=True, diff file written."""
    monkeypatch.setattr(
        mod, "_git_status_porcelain", lambda: " M code/09_finetune_lora.py\n"
    )
    monkeypatch.setattr(mod, "_git_rev_parse_head", lambda: "cafebabe" * 5)
    monkeypatch.setattr(
        mod, "_git_diff_head", lambda: "diff --git a/x b/x\n+++ added\n"
    )
    monkeypatch.setattr(mod, "_library_versions", lambda: {"torch": "1.0"})

    json_path = tmp_path / "info.json"
    diff_path = tmp_path / "info.diff"
    provenance = mod.build_provenance(
        seed=0,
        model_name="m3d_lamed",
        hyperparameters={"lr": 2e-5},
        data_splits={"n_train": 10, "n_val": 2, "n_test": 2},
        bucket_distributions={"dice_train": {5: 5}},
        start_time="2026-04-25T00:00:00Z",
        end_time="2026-04-25T00:01:00Z",
        best_val_srcc=0.7,
        best_epoch=2,
        trainable_params=100,
        total_params=1000,
        diff_path=diff_path,
    )
    mod.save_provenance(json_path, provenance, diff_path=diff_path)

    payload = json.loads(json_path.read_text())
    assert payload["git_dirty"] is True
    assert payload["git_diff_path"] == str(diff_path)
    assert diff_path.is_file()
    assert "diff --git" in diff_path.read_text()


def test_resume_loads_checkpoint(
    mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--resume-from-checkpoint plumbs through to Trainer.train(resume_from_checkpoint=...)."""
    sub, pref, thick = _build_synthetic_csvs(tmp_path)
    _install_fake_adapter(mod, "m3d_lamed")
    instances = _install_fake_trainer(mod)
    _bypass_preconditions(mod, monkeypatch)
    _bypass_score_test_split(mod, monkeypatch)

    ckpt_path = tmp_path / "checkpoints"
    ckpt_path.mkdir()
    resume_path = ckpt_path / "checkpoint-100"
    resume_path.mkdir()

    output_scores = tmp_path / "scores.csv"

    result = runner.invoke(
        mod.app,
        [
            "--seed", "0",
            "--model", "m3d_lamed",
            "--subsample-manifest", str(sub),
            "--preference-csv", str(pref),
            "--thickness-csv", str(thick),
            "--output-checkpoint-dir", str(ckpt_path),
            "--output-scores-file", str(output_scores),
            "--resume-from-checkpoint", str(resume_path),
            "--no-lora-vision-encoder",
        ],
    )
    if result.exit_code != 0 and result.exception is not None:
        import traceback
        traceback.print_exception(
            type(result.exception), result.exception, result.exception.__traceback__
        )
    assert result.exit_code == 0, result.output
    assert len(instances) == 1
    assert instances[0].train_calls == [str(resume_path)]


def test_dry_run_exits_before_training(
    mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--dry-run logs distributions and exits before adapter.load() / Trainer."""
    sub, pref, thick = _build_synthetic_csvs(tmp_path)

    # Adapter installation that would raise if .load() were called.
    class _ExplodingAdapter(_FakeAdapter):
        def load(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("adapter.load called in dry-run mode")

    mod._FT_ADAPTERS["m3d_lamed"] = _ExplodingAdapter
    instances = _install_fake_trainer(mod)
    _bypass_preconditions(mod, monkeypatch)

    result = runner.invoke(
        mod.app,
        [
            "--seed", "0",
            "--model", "m3d_lamed",
            "--subsample-manifest", str(sub),
            "--preference-csv", str(pref),
            "--thickness-csv", str(thick),
            "--output-checkpoint-dir", str(tmp_path / "ckpt"),
            "--output-scores-file", str(tmp_path / "scores.csv"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    # No trainer ever instantiated.
    assert instances == []


def test_clean_rows_get_structural_targets(
    mod: types.ModuleType, tmp_path: Path
) -> None:
    """Clean rows are assigned mean_dice=1.0 and shift=0.0 by structural rule."""
    sub, pref, thick = _build_synthetic_csvs(tmp_path)
    df, _ = mod.prepare_dataframe(sub, pref, thick)

    clean_rows = df[df[mod.IS_CLEAN_COLUMN].astype(bool)]
    assert len(clean_rows) == 2
    assert all(abs(v - 1.0) < 1e-9 for v in clean_rows[mod._DERIVED_DICE].tolist())
    assert all(
        abs(v) < 1e-9 for v in clean_rows[mod._DERIVED_SIGNED_SHIFT].tolist()
    )

    # Cor rows preserve real measured values.
    cor_a1 = df[df[mod.SCAN_COLUMN] == "/cor/A_motion_1.nii.gz"]
    assert abs(float(cor_a1[mod._DERIVED_DICE].iloc[0]) - 0.85) < 1e-9
    assert abs(float(cor_a1[mod._DERIVED_SIGNED_SHIFT].iloc[0]) - (-0.08)) < 1e-9


def test_join_uses_cor_path_not_scan_path(
    mod: types.ModuleType, tmp_path: Path
) -> None:
    """Join works against preference's cor_path (no scan_path column on that frame)."""
    sub, pref, thick = _build_synthetic_csvs(tmp_path)

    # Sanity check: machine_preference.csv has no scan_path column.
    pref_frame = pd.read_csv(pref)
    assert "scan_path" not in pref_frame.columns
    assert "cor_path" in pref_frame.columns

    df, _ = mod.prepare_dataframe(sub, pref, thick)
    cor_rows = df[~df[mod.IS_CLEAN_COLUMN].astype(bool)]
    # Every cor row matched a preference row.
    assert cor_rows[mod._DERIVED_DICE].notna().all()
    expected_dice = {
        "/cor/A_motion_1.nii.gz": 0.85,
        "/cor/A_motion_3.nii.gz": 0.55,
        "/cor/B_motion_1.nii.gz": 0.92,
        "/cor/B_motion_5.nii.gz": 0.30,
    }
    for path, dice in expected_dice.items():
        row = df[df[mod.SCAN_COLUMN] == path]
        assert abs(float(row[mod._DERIVED_DICE].iloc[0]) - dice) < 1e-9


def test_thickness_seg_path_remap_via_synthseg_manifest(
    mod: types.ModuleType, tmp_path: Path
) -> None:
    """Phase 03b emits seg paths in cortical_thickness.csv:scan_path; passing a
    synthseg manifest maps them back to scan paths so the join works."""
    # Build a thickness CSV with SEG paths (mimicking real Phase 03b output).
    thickness_path = tmp_path / "cortical_thickness.csv"
    seg_a = tmp_path / "synthseg_ref" / "scanA_synthseg.nii.gz"
    seg_b = tmp_path / "synthseg_cor" / "scanB_synthseg.nii.gz"
    seg_a.parent.mkdir(parents=True, exist_ok=True)
    seg_b.parent.mkdir(parents=True, exist_ok=True)
    seg_a.touch()
    seg_b.touch()
    scan_a = tmp_path / "refs" / "scanA.nii.gz"
    scan_b = tmp_path / "cors" / "scanB.nii.gz"
    scan_a.parent.mkdir(parents=True, exist_ok=True)
    scan_b.parent.mkdir(parents=True, exist_ok=True)
    scan_a.touch()
    scan_b.touch()

    _write_thickness(thickness_path, [
        {"scan_path": str(seg_a), "mean_thickness": 4.0},
        {"scan_path": str(seg_b), "mean_thickness": 4.5},
    ])

    # Synthseg manifest: input_path → seg_path mapping.
    manifest_path = tmp_path / "synthseg_manifest.csv"
    with manifest_path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["input_path", "seg_path", "status"])
        w.writeheader()
        w.writerow({"input_path": str(scan_a), "seg_path": str(seg_a), "status": "ok"})
        w.writerow({"input_path": str(scan_b), "seg_path": str(seg_b), "status": "ok"})

    # Without manifest: scan_path stays as seg path → join will fail downstream.
    no_remap = mod.load_thickness(thickness_path)
    assert str(seg_a) in no_remap[mod._THICKNESS_SCAN_PATH].astype(str).tolist()

    # With manifest: scan_path remaps to actual scan paths.
    remapped = mod.load_thickness(thickness_path, synthseg_manifests=[manifest_path])
    paths = set(remapped[mod._THICKNESS_SCAN_PATH].astype(str).tolist())
    # Both seg paths should have been replaced by their input scan counterparts.
    assert str(scan_a.resolve()) in paths
    assert str(scan_b.resolve()) in paths
    assert str(seg_a) not in paths
    assert str(seg_b) not in paths
