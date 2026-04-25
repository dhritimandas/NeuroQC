#!/usr/bin/env python3
"""Verify that FreeSurfer (and optionally SynthSeg Python) are set up for phase 3.

Reports:
    - FREESURFER_HOME set and pointing at a directory
    - mri_synthseg on PATH and --help runs
    - License file present
    - FreeSurfer version from the build-stamp
    - (optional) SynthSeg Python package importable
    - (optional) TensorFlow version if present

Exit 0 when everything critical is present; non-zero if FreeSurfer is missing
or misconfigured. Optional components (SynthSeg Python, TF) just emit a warn.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Verify FreeSurfer / SynthSeg install for neuroqc phase 3.", add_completion=False)


def _find_freesurfer_install() -> Path | None:
    """Return a detected FreeSurfer install path, or None.

    Search order:
      1. $FREESURFER_HOME if it contains SetUpFreeSurfer.sh.
      2. Common base dirs, either directly or in any versioned subdir.
    """
    env_value = os.environ.get("FREESURFER_HOME")
    if env_value:
        candidate = Path(env_value)
        if (candidate / "SetUpFreeSurfer.sh").is_file():
            return candidate

    candidate_bases = [
        Path("/Applications/freesurfer"),
        Path("/opt/freesurfer"),
        Path("/usr/local/freesurfer"),
        Path.home() / "freesurfer",
    ]
    for base in candidate_bases:
        if not base.is_dir():
            continue
        if (base / "SetUpFreeSurfer.sh").is_file():
            return base
        # Versioned subdir (e.g. freesurfer/8.2.0/SetUpFreeSurfer.sh)
        for sub in sorted(base.iterdir(), reverse=True):
            if sub.is_dir() and (sub / "SetUpFreeSurfer.sh").is_file():
                return sub
    return None


def _check_fs_home() -> tuple[str, Path | None]:
    env_value = os.environ.get("FREESURFER_HOME")
    detected = _find_freesurfer_install()
    if detected is None:
        if env_value:
            return f"FREESURFER_HOME={env_value} (invalid — no SetUpFreeSurfer.sh)", None
        return "not set and no install found in common paths", None
    if env_value and Path(env_value) != detected:
        return f"{detected} (env says {env_value}; using detected)", detected
    return f"{detected}", detected


def _check_mri_synthseg() -> tuple[str, Path | None]:
    located = shutil.which("mri_synthseg")
    if located is None:
        return "not on PATH", None
    return located, Path(located)


def _check_license(fs_home: Path | None) -> str:
    if fs_home is None:
        return "n/a (no FREESURFER_HOME)"
    candidate = fs_home / "license.txt"
    if candidate.is_file():
        return f"found: {candidate}"
    return f"missing at {candidate}"


def _check_version(fs_home: Path | None) -> str:
    if fs_home is None:
        return "n/a"
    stamp = fs_home / "build-stamp.txt"
    if stamp.is_file():
        return stamp.read_text().strip()
    return "build-stamp.txt missing"


def _check_mri_synthseg_help() -> str:
    try:
        result = subprocess.run(
            ["mri_synthseg", "--help"], capture_output=True, text=True, timeout=30
        )
    except FileNotFoundError:
        return "failed: binary not found"
    except subprocess.TimeoutExpired:
        return "failed: timed out"
    if result.returncode != 0:
        return f"failed: exit {result.returncode}"
    return "ok"


def _check_synthseg_py() -> str:
    try:
        spec = importlib.util.find_spec("SynthSeg")
    except (ImportError, ValueError):
        return "not importable"
    if spec is None:
        return "not importable"
    return f"{spec.origin}"


def _check_tensorflow() -> str:
    try:
        spec = importlib.util.find_spec("tensorflow")
    except (ImportError, ValueError):
        return "not installed"
    if spec is None:
        return "not installed"
    try:
        import tensorflow as tf  # noqa: WPS433  — verify import succeeds

        return f"{tf.__version__}"
    except Exception as exc:  # pragma: no cover  — diagnostic path
        return f"installed but broken ({type(exc).__name__}: {exc})"


@app.command()
def main() -> None:
    """Verify the phase-3 execution environment and print a report."""
    console = Console()
    fs_home_str, fs_home = _check_fs_home()
    mri_str, mri_path = _check_mri_synthseg()
    license_str = _check_license(fs_home)
    version_str = _check_version(fs_home)
    help_str = _check_mri_synthseg_help() if mri_path is not None else "skipped"
    synthseg_str = _check_synthseg_py()
    tf_str = _check_tensorflow()

    table = Table(title="NeuroQC phase-3 install check")
    table.add_column("component", style="bold")
    table.add_column("status")
    table.add_row("FREESURFER_HOME", fs_home_str)
    table.add_row("mri_synthseg", mri_str)
    table.add_row("mri_synthseg --help", help_str)
    table.add_row("FreeSurfer version", version_str)
    table.add_row("License file", license_str)
    table.add_row("SynthSeg Python (optional)", synthseg_str)
    table.add_row("TensorFlow (optional)", tf_str)
    console.print(table)

    critical_ok = fs_home is not None and mri_path is not None and help_str == "ok"
    if not critical_ok:
        console.print(
            "[red]✗ Critical FreeSurfer checks failed. "
            "Install FS (scripts/install_freesurfer_*.sh) and source SetUpFreeSurfer.sh.[/red]"
        )
        raise typer.Exit(code=1)

    if "missing" in license_str:
        console.print(
            "[yellow]⚠ License missing — mri_synthseg will refuse to run until you "
            "place license.txt at the path above.[/yellow]"
        )

    console.print("[green]✓ Phase 3 can run in --mode freesurfer.[/green]")


if __name__ == "__main__":
    app()
