"""Unit tests for code/02_generate_corruptions.py.

The underlying nobrainer.qc.generate_corrupted_dataset is monkeypatched on the
module under test so these tests stay fast and hermetic — no TorchIO
transforms, no NIfTI I/O. We verify:

1. CLI argument parsing: "all" vs comma-separated lists for --corruptions /
   --severities produces the expected arguments to the underlying function.
2. The manifest CSV produced by the wrapper matches the CLAUDE.md schema
   exactly (header + row contents).
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE_PATH = _REPO_ROOT / "code" / "02_generate_corruptions.py"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("generate_corruptions", _MODULE_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {_MODULE_PATH}"
    module = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass and other tooling can resolve
    # string annotations under `from __future__ import annotations`.
    sys.modules["generate_corruptions"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen_mod() -> types.ModuleType:
    return _load_module()


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _fake_metadata(input_path: str, cor_path: str) -> dict[str, Any]:
    """Build a single well-formed metadata dict matching the schema."""
    return {
        "ref_path": input_path,
        "cor_path": cor_path,
        "corruption_type": "motion",
        "corruption_domain": "kspace",
        "severity": 1,
        "seed": 42,
        "transform_params": {"degrees": 2, "translation": 1, "num_transforms": 2},
    }


def test_all_keyword_passes_none_and_writes_schema_correct_csv(
    gen_mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--corruptions all --severities all forwards None/None to the generator,
    and the resulting manifest header matches CLAUDE.md exactly."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    manifest_path = tmp_path / "corruption_manifest.csv"

    captured: dict[str, Any] = {}

    def fake_generate(
        input_dir: Path,
        output_dir: Path,
        corruptions: list[str] | None = None,
        severities: list[int] | None = None,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> list[dict]:
        captured["corruptions"] = corruptions
        captured["severities"] = severities
        captured["dry_run"] = dry_run
        return [
            _fake_metadata(str(input_dir / "a.nii.gz"), str(output_dir / "motion/severity_1/a.nii.gz")),
            # Failure-shaped entry (missing some keys) — must be filtered out.
            {
                "ref_path": str(input_dir / "b.nii.gz"),
                "cor_path": str(output_dir / "motion/severity_1/b.nii.gz"),
                "corruption_type": "motion",
                "severity": 1,
                "error": "synthetic failure",
            },
        ]

    monkeypatch.setattr(gen_mod, "generate_corrupted_dataset", fake_generate)

    result = runner.invoke(
        gen_mod.app,
        [
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--corruptions", "all",
            "--severities", "all",
            "--dataset-tag", "ixi",
            "--manifest-path", str(manifest_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["corruptions"] is None
    assert captured["severities"] is None
    assert captured["dry_run"] is False

    with manifest_path.open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == gen_mod.MANIFEST_COLUMNS

    # Only the well-formed row is written; the failure row was dropped.
    assert len(rows) == 1
    row = rows[0]
    assert row["corruption_type"] == "motion"
    assert row["corruption_domain"] == "kspace"
    assert row["severity"] == "1"
    assert row["seed"] == "42"
    assert row["dataset_tag"] == "ixi"
    # transform_params is a JSON string — must round-trip.
    import json
    assert json.loads(row["transform_params"]) == {
        "degrees": 2,
        "translation": 1,
        "num_transforms": 2,
    }


def test_explicit_lists_parsed_and_forwarded(
    gen_mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--corruptions motion,noise --severities 1,3 forwards the parsed lists."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    manifest_path = tmp_path / "corruption_manifest.csv"

    captured: dict[str, Any] = {}

    def fake_generate(
        input_dir: Path,
        output_dir: Path,
        corruptions: list[str] | None = None,
        severities: list[int] | None = None,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> list[dict]:
        captured["corruptions"] = corruptions
        captured["severities"] = severities
        return []

    monkeypatch.setattr(gen_mod, "generate_corrupted_dataset", fake_generate)

    result = runner.invoke(
        gen_mod.app,
        [
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--corruptions", "motion,noise",
            "--severities", "1,3",
            "--dataset-tag", "ixi",
            "--manifest-path", str(manifest_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["corruptions"] == ["motion", "noise"]
    assert captured["severities"] == [1, 3]

    # Unknown corruption should be rejected by the wrapper *before* calling through.
    result_bad = runner.invoke(
        gen_mod.app,
        [
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--corruptions", "motion,not_a_real_corruption",
            "--severities", "1",
            "--dataset-tag", "ixi",
            "--manifest-path", str(manifest_path),
        ],
    )
    assert result_bad.exit_code != 0
    assert "Unknown corruption" in result_bad.output or "Unknown corruption" in (result_bad.stderr or "")


def test_dataset_tag_written_to_manifest(
    gen_mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--dataset-tag fastmri is recorded as the dataset_tag column of every row."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    manifest_path = tmp_path / "corruption_manifest.csv"

    def fake_generate(
        input_dir: Path,
        output_dir: Path,
        corruptions: list[str] | None = None,
        severities: list[int] | None = None,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> list[dict]:
        return [
            _fake_metadata(
                str(input_dir / "a.nii.gz"),
                str(output_dir / "motion/severity_1/a.nii.gz"),
            ),
            _fake_metadata(
                str(input_dir / "b.nii.gz"),
                str(output_dir / "motion/severity_1/b.nii.gz"),
            ),
        ]

    monkeypatch.setattr(gen_mod, "generate_corrupted_dataset", fake_generate)

    # Uppercase on input to also verify case-insensitive handling.
    result = runner.invoke(
        gen_mod.app,
        [
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--corruptions", "motion",
            "--severities", "1",
            "--dataset-tag", "FastMRI",
            "--manifest-path", str(manifest_path),
        ],
    )

    assert result.exit_code == 0, result.output
    with manifest_path.open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == gen_mod.MANIFEST_COLUMNS
    assert len(rows) == 2
    assert all(row["dataset_tag"] == "fastmri" for row in rows)


def test_invalid_dataset_tag_rejected(
    gen_mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--dataset-tag not in DATASET_TAGS must fail before any corruption runs."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    manifest_path = tmp_path / "corruption_manifest.csv"

    called = {"n": 0}

    def fake_generate(*args: Any, **kwargs: Any) -> list[dict]:
        called["n"] += 1
        return []

    monkeypatch.setattr(gen_mod, "generate_corrupted_dataset", fake_generate)

    result = runner.invoke(
        gen_mod.app,
        [
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--corruptions", "all",
            "--severities", "all",
            "--dataset-tag", "not_a_dataset",
            "--manifest-path", str(manifest_path),
        ],
    )

    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    for valid in gen_mod.DATASET_TAGS:
        assert valid in combined
    assert called["n"] == 0
    assert not manifest_path.exists()


def test_manifest_preserves_other_corruption_types_and_datasets(
    gen_mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """02 must not wipe 02b's k-space rows or other-dataset rows on re-run.

    Seeds a manifest with three rows:
      - ``motion_kspace`` + ``fastmri`` (written by 02b)
      - ``motion`` + ``ixi`` (written by a prior 02 run on the other dataset)
      - ``ghosting`` + ``fastmri`` (written by a prior 02 run for a different
        corruption type on the same dataset)

    Then invokes 02 with ``--corruptions motion --dataset-tag fastmri``. Only
    rows matching *both* ``motion`` and ``fastmri`` should be dropped; the
    three seeded rows must all survive because none match both conditions.
    """
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    manifest_path = tmp_path / "corruption_manifest.csv"

    # Seed the manifest with three existing rows that must all survive.
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=gen_mod.MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerow({
            "ref_path": "/data/fastmri/ref_a.nii.gz",
            "cor_path": "/data/fastmri/cor_a.nii.gz",
            "corruption_type": "motion_kspace",
            "corruption_domain": "kspace",
            "severity": "3",
            "seed": "100",
            "transform_params": '{"n_transforms": 5}',
            "dataset_tag": "fastmri",
        })
        writer.writerow({
            "ref_path": "/data/ixi/ref_b.nii.gz",
            "cor_path": "/data/ixi/cor_b.nii.gz",
            "corruption_type": "motion",
            "corruption_domain": "image",
            "severity": "1",
            "seed": "101",
            "transform_params": '{"degrees": 2}',
            "dataset_tag": "ixi",
        })
        writer.writerow({
            "ref_path": "/data/fastmri/ref_c.nii.gz",
            "cor_path": "/data/fastmri/cor_c.nii.gz",
            "corruption_type": "ghosting",
            "corruption_domain": "image",
            "severity": "2",
            "seed": "102",
            "transform_params": '{"intensity": 0.5}',
            "dataset_tag": "fastmri",
        })

    def fake_generate(*args: Any, **kwargs: Any) -> list[dict]:
        return [
            _fake_metadata(
                str(input_dir / "new.nii.gz"),
                str(output_dir / "motion/severity_1/new.nii.gz"),
            )
        ]

    monkeypatch.setattr(gen_mod, "generate_corrupted_dataset", fake_generate)

    result = runner.invoke(
        gen_mod.app,
        [
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--corruptions", "motion",
            "--severities", "1",
            "--dataset-tag", "fastmri",
            "--manifest-path", str(manifest_path),
        ],
    )
    assert result.exit_code == 0, result.output

    with manifest_path.open() as handle:
        rows = list(csv.DictReader(handle))

    # All three seeded rows survive (none match both motion AND fastmri).
    seeded_cor_paths = {
        "/data/fastmri/cor_a.nii.gz",  # motion_kspace — wrong type
        "/data/ixi/cor_b.nii.gz",      # motion but ixi — wrong dataset
        "/data/fastmri/cor_c.nii.gz",  # ghosting — wrong type
    }
    present = {r["cor_path"] for r in rows}
    assert seeded_cor_paths.issubset(present), f"lost seeded rows: {seeded_cor_paths - present}"

    # The new motion/fastmri row from this run is also present.
    assert any(
        r["corruption_type"] == "motion" and r["dataset_tag"] == "fastmri"
        for r in rows
    )
    assert len(rows) == 4


def test_manifest_replaces_own_prior_run_on_same_dataset(
    gen_mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-running 02 with the same (corruption_type, dataset_tag) replaces in place.

    Without this guarantee, repeated invocations would accumulate duplicate
    rows for the same ref/cor pair. The read-modify-write pattern scopes the
    drop by both fields so the second run wipes the first run's motion/ixi
    rows before writing new ones.
    """
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    manifest_path = tmp_path / "corruption_manifest.csv"

    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=gen_mod.MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerow({
            "ref_path": "/data/ixi/old.nii.gz",
            "cor_path": "/data/ixi/old_motion.nii.gz",
            "corruption_type": "motion",
            "corruption_domain": "image",
            "severity": "1",
            "seed": "1",
            "transform_params": "{}",
            "dataset_tag": "ixi",
        })

    def fake_generate(*args: Any, **kwargs: Any) -> list[dict]:
        return [
            _fake_metadata(
                str(input_dir / "new.nii.gz"),
                str(output_dir / "motion/severity_1/new.nii.gz"),
            )
        ]

    monkeypatch.setattr(gen_mod, "generate_corrupted_dataset", fake_generate)

    result = runner.invoke(
        gen_mod.app,
        [
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--corruptions", "motion",
            "--severities", "1",
            "--dataset-tag", "ixi",
            "--manifest-path", str(manifest_path),
        ],
    )
    assert result.exit_code == 0, result.output

    with manifest_path.open() as handle:
        rows = list(csv.DictReader(handle))

    # Old row is gone; new one is in.
    cor_paths = {r["cor_path"] for r in rows}
    assert "/data/ixi/old_motion.nii.gz" not in cor_paths
    assert len(rows) == 1
