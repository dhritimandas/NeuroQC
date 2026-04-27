"""Unit tests for code/03_run_synthseg.py.

All tests mock out SynthSeg — neither ``mri_synthseg`` nor the Python package
is assumed to be available. Tests cover:

1. FreeSurfer mode builds the correct subprocess argv per scan.
2. Python mode chunks inputs and calls SynthSeg.predict.predict with the
   expected keyword arguments.
3. Resume: scans whose segmentation output already exists are skipped
   (neither subprocess nor predict is invoked for them).
"""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE_PATH = _REPO_ROOT / "code" / "03_run_synthseg.py"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("run_synthseg", _MODULE_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {_MODULE_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_synthseg"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def synth_mod() -> types.ModuleType:
    return _load_module()


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _touch_nii_gz(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


# ──────────────────────────────────────────────
# Freesurfer mode
# ──────────────────────────────────────────────


def test_freesurfer_mode_builds_correct_subprocess_args(
    synth_mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    manifest_path = tmp_path / "synthseg_manifest.csv"

    _touch_nii_gz(input_dir / "refs" / "IXI001-Guys-0001-T1.nii.gz")
    _touch_nii_gz(input_dir / "corruptions" / "motion" / "severity_1" / "IXI001-Guys-0001-T1.nii.gz")

    # Hermetic env: drop FREESURFER_HOME so the resolver falls through to
    # shutil.which, which we patch to return a fake binary path.
    monkeypatch.delenv("FREESURFER_HOME", raising=False)
    fake_bin = "/fake/bin/mri_synthseg"
    monkeypatch.setattr(synth_mod.shutil, "which", lambda _name: fake_bin)

    captured_calls: list[list[str]] = []
    captured_envs: list[dict[str, str] | None] = []

    def fake_run(cmd: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        captured_calls.append(list(cmd))
        captured_envs.append(kwargs.get("env"))
        assert kwargs.get("check") is True
        assert kwargs.get("capture_output") is True
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(synth_mod.subprocess, "run", fake_run)

    result = runner.invoke(
        synth_mod.app,
        [
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--mode", "freesurfer",
            "--manifest-path", str(manifest_path),
        ],
    )
    assert result.exit_code == 0, result.output

    assert len(captured_calls) == 2
    call_by_input = {cmd[cmd.index("--i") + 1]: cmd for cmd in captured_calls}

    ref_cmd = call_by_input[str(input_dir / "refs" / "IXI001-Guys-0001-T1.nii.gz")]
    # Resolver returns the absolute path from shutil.which when no freesurfer_home
    # is known; subprocess is invoked with that absolute path.
    assert ref_cmd[0] == fake_bin
    assert "--parc" in ref_cmd
    assert ref_cmd[ref_cmd.index("--o") + 1] == str(
        output_dir / "refs" / "IXI001-Guys-0001-T1_synthseg.nii.gz"
    )
    assert ref_cmd[ref_cmd.index("--qc") + 1] == str(
        output_dir / "refs" / "IXI001-Guys-0001-T1_qc.csv"
    )
    assert ref_cmd[ref_cmd.index("--vol") + 1] == str(
        output_dir / "refs" / "IXI001-Guys-0001-T1_vol.csv"
    )

    # env is always passed (never None) so the bash wrapper has FREESURFER_HOME
    # inherited from our process (here, empty since we delenv'd).
    assert all(env is not None for env in captured_envs)

    with manifest_path.open() as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == synth_mod.MANIFEST_COLUMNS
        rows = list(reader)
    assert len(rows) == 2
    assert all(row["status"] == synth_mod.STATUS_OK for row in rows)
    assert all(row["mode"] == "freesurfer" for row in rows)


def test_freesurfer_home_flag_builds_absolute_binary_and_sets_env(
    synth_mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--freesurfer-home locates bin/mri_synthseg and passes FREESURFER_HOME."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    manifest_path = tmp_path / "synthseg_manifest.csv"
    fs_home = tmp_path / "freesurfer"
    bin_path = fs_home / "bin" / "mri_synthseg"
    bin_path.parent.mkdir(parents=True)
    bin_path.write_text("#!/bin/bash\nexit 0\n")
    bin_path.chmod(0o755)

    _touch_nii_gz(input_dir / "IXI001-Guys-0001-T1.nii.gz")

    # Drop FREESURFER_HOME from env so the test is hermetic; only the flag
    # should drive resolution here.
    monkeypatch.delenv("FREESURFER_HOME", raising=False)

    captured_calls: list[list[str]] = []
    captured_envs: list[dict[str, str]] = []

    def fake_run(cmd: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        captured_calls.append(list(cmd))
        captured_envs.append(kwargs.get("env") or {})
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(synth_mod.subprocess, "run", fake_run)

    result = runner.invoke(
        synth_mod.app,
        [
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--mode", "freesurfer",
            "--freesurfer-home", str(fs_home),
            "--manifest-path", str(manifest_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(captured_calls) == 1
    cmd = captured_calls[0]
    assert cmd[0] == str(bin_path)
    assert captured_envs[0].get("FREESURFER_HOME") == str(fs_home)


def test_freesurfer_home_flag_rejects_missing_binary(
    synth_mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A --freesurfer-home path without bin/mri_synthseg is a BadParameter."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    manifest_path = tmp_path / "synthseg_manifest.csv"
    fs_home = tmp_path / "freesurfer_missing"  # never created
    _touch_nii_gz(input_dir / "IXI001-Guys-0001-T1.nii.gz")
    monkeypatch.delenv("FREESURFER_HOME", raising=False)

    result = runner.invoke(
        synth_mod.app,
        [
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--mode", "freesurfer",
            "--freesurfer-home", str(fs_home),
            "--manifest-path", str(manifest_path),
        ],
    )
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "does not exist" in combined or str(fs_home) in combined


# ──────────────────────────────────────────────
# Python mode
# ──────────────────────────────────────────────


def test_python_mode_invokes_synthseg_cli_with_list_files(
    synth_mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python mode shells out to SynthSeg's CLI with list files (one
    path per line) for ``--i / --o / --qc / --vol``. This amortises
    SynthSeg's ~80 s startup cost over the whole batch instead of paying
    it per scan.
    """
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    manifest_path = tmp_path / "synthseg_manifest.csv"

    inputs = [
        _touch_nii_gz(input_dir / "a" / "IXI001-Guys-0001-T1.nii.gz"),
        _touch_nii_gz(input_dir / "a" / "IXI002-Guys-0002-T1.nii.gz"),
        _touch_nii_gz(input_dir / "b" / "IXI003-HH-0003-T1.nii.gz"),
    ]

    # Pretend SynthSeg's CLI lives somewhere — we don't actually need the
    # file because subprocess.run is monkeypatched.
    fake_cli = tmp_path / "fake_SynthSeg" / "scripts" / "commands" / "SynthSeg_predict.py"
    fake_cli.parent.mkdir(parents=True, exist_ok=True)
    fake_cli.write_text("# stub")
    monkeypatch.setattr(synth_mod, "_locate_synthseg_cli", lambda: fake_cli)

    captured_calls: list[dict[str, Any]] = []

    def fake_run(cmd: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        # Read the list-file contents NOW (before the temp dir is cleaned)
        # and stash them alongside the argv for later assertions.
        list_contents: dict[str, list[str]] = {}
        for flag in ("--i", "--o", "--qc", "--vol"):
            if flag in cmd:
                list_path = Path(cmd[cmd.index(flag) + 1])
                if list_path.exists():
                    list_contents[flag] = [
                        line.strip()
                        for line in list_path.read_text().splitlines()
                        if line.strip()
                    ]
        captured_calls.append({"cmd": list(cmd), "lists": list_contents})

        # Touch each seg path so run_python's exists() check thinks the
        # run succeeded.
        for seg in list_contents.get("--o", []):
            seg_path = Path(seg)
            seg_path.parent.mkdir(parents=True, exist_ok=True)
            seg_path.write_bytes(b"")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(synth_mod.subprocess, "run", fake_run)

    result = runner.invoke(
        synth_mod.app,
        [
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--mode", "python",
            "--batch-size", "2",
            "--manifest-path", str(manifest_path),
        ],
    )
    assert result.exit_code == 0, result.output

    # 3 inputs, batch_size=2 → two CLI invocations (2, 1).
    assert len(captured_calls) == 2

    # Each call must reference the SynthSeg CLI script and pass list files
    # to --i / --o / --qc / --vol; the lists must have matching lengths.
    seen_inputs: list[str] = []
    for entry in captured_calls:
        cmd = entry["cmd"]
        lists = entry["lists"]
        assert any("SynthSeg_predict.py" in tok for tok in cmd)
        assert set(lists.keys()) == {"--i", "--o", "--qc", "--vol"}
        n = len(lists["--i"])
        assert n in (1, 2), f"unexpected batch size {n}; expected 1 or 2"
        assert len(lists["--o"]) == n
        assert len(lists["--qc"]) == n
        assert len(lists["--vol"]) == n
        seen_inputs.extend(lists["--i"])

    assert sorted(seen_inputs) == sorted(str(p.resolve()) for p in inputs)

    with manifest_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert all(row["status"] == synth_mod.STATUS_OK for row in rows)
    assert all(row["mode"] == "python" for row in rows)
    # Manifest must reference all 3 input scans.
    manifest_inputs = {row["input_path"] for row in rows}
    assert manifest_inputs == {str(p.resolve()) for p in inputs}


# ──────────────────────────────────────────────
# Resume
# ──────────────────────────────────────────────


def test_resume_skips_when_seg_already_exists(
    synth_mod: types.ModuleType,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    manifest_path = tmp_path / "synthseg_manifest.csv"

    existing_input = _touch_nii_gz(input_dir / "IXI001-Guys-0001-T1.nii.gz")
    missing_input = _touch_nii_gz(input_dir / "IXI002-HH-0002-T1.nii.gz")

    # Pre-create the seg output for the "existing" scan.
    existing_seg = output_dir / "IXI001-Guys-0001-T1_synthseg.nii.gz"
    existing_seg.parent.mkdir(parents=True, exist_ok=True)
    existing_seg.write_bytes(b"")

    monkeypatch.delenv("FREESURFER_HOME", raising=False)
    monkeypatch.setattr(synth_mod.shutil, "which", lambda _name: "/fake/bin/mri_synthseg")

    captured_calls: list[list[str]] = []

    def fake_run(cmd: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        captured_calls.append(list(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(synth_mod.subprocess, "run", fake_run)

    result = runner.invoke(
        synth_mod.app,
        [
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--mode", "freesurfer",
            "--manifest-path", str(manifest_path),
        ],
    )
    assert result.exit_code == 0, result.output

    # Only the missing scan should have triggered subprocess.run.
    assert len(captured_calls) == 1
    only_cmd = captured_calls[0]
    assert only_cmd[only_cmd.index("--i") + 1] == str(missing_input)

    with manifest_path.open() as handle:
        rows = list(csv.DictReader(handle))
    rows_by_input = {row["input_path"]: row for row in rows}
    assert rows_by_input[str(existing_input)]["status"] == synth_mod.STATUS_SKIPPED
    assert rows_by_input[str(missing_input)]["status"] == synth_mod.STATUS_OK
