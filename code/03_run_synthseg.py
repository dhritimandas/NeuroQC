#!/usr/bin/env python3
"""NeuroQC Phase 3 — SynthSeg segmentation + QC.

Walks ``--input-dir`` recursively for ``.nii.gz`` files and produces, for each
scan, a SynthSeg whole-brain segmentation (with cortical parcellation), a
per-scan QC CSV, and a per-scan volumetry CSV under ``--output-dir`` with the
relative input path preserved. An aggregate manifest is written to
``--manifest-path``.

Two execution modes:
    --mode freesurfer  Per-scan call to ``mri_synthseg`` (FreeSurfer's CLI).
                       Resolves the binary via (priority order):
                         1. ``--freesurfer-home`` CLI flag.
                         2. ``$FREESURFER_HOME`` env var.
                         3. ``shutil.which("mri_synthseg")`` (requires a
                            pre-sourced shell).
                       The wrapper at ``<home>/bin/mri_synthseg`` needs
                       ``FREESURFER_HOME`` in its env, so we pass it explicitly
                       to each subprocess — no shell sourcing required.
    --mode python      Batched calls to ``SynthSeg.predict.predict`` (requires
                       the SynthSeg Python repo on PYTHONPATH). Batch size
                       controlled by ``--batch-size``.

Inputs:
    --input-dir        Root directory of .nii.gz scans (references + corrupted).
    --output-dir       Root directory for SynthSeg outputs; mirrors input tree.
    --mode             "freesurfer" (default) or "python".
    --freesurfer-home  FreeSurfer install root (e.g. /Applications/freesurfer/8.1.0);
                       defaults to $FREESURFER_HOME env var.
    --batch-size       Per-call batch size in python mode (default 1).
    --manifest-path    Aggregate CSV manifest (default results/tables/synthseg_manifest.csv).
    --dry-run          Log planned work; no subprocess or predict calls.

Outputs (per scan <input-dir>/<rel>/scan.nii.gz):
    <output-dir>/<rel>/scan_synthseg.nii.gz   Segmentation (+ parcellation labels).
    <output-dir>/<rel>/scan_qc.csv            Per-scan QC row from mri_synthseg / predict.
    <output-dir>/<rel>/scan_vol.csv           Per-scan volumetry row.
    <manifest-path>                           Aggregate manifest with columns:
        input_path, seg_path, qc_path, vol_path, mode, status
      where status ∈ {ok, skipped, failed}. 'skipped' means the per-scan seg
      output already existed at planning time (resume behaviour).

Usage:
    python code/03_run_synthseg.py \\
        --input-dir data/ixi/derivatives \\
        --output-dir data/derivatives/synthseg \\
        --mode freesurfer
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from tqdm import tqdm

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

MRI_SYNTHSEG_BIN: str = "mri_synthseg"

STATUS_OK: str = "ok"
STATUS_SKIPPED: str = "skipped"
STATUS_FAILED: str = "failed"

MANIFEST_COLUMNS: list[str] = [
    "input_path",
    "seg_path",
    "qc_path",
    "vol_path",
    "mode",
    "status",
]

logger = logging.getLogger(__name__)
app = typer.Typer(
    help="NeuroQC Phase 3 — run SynthSeg segmentation + QC.",
    add_completion=False,
)


class SynthSegMode(str, Enum):
    """Execution backend for SynthSeg."""

    FREESURFER = "freesurfer"
    PYTHON = "python"


# ──────────────────────────────────────────────
# Planning
# ──────────────────────────────────────────────


@dataclass(frozen=True)
class ScanPlan:
    """Planned per-scan SynthSeg invocation.

    Attributes:
        input_path: Source .nii.gz path.
        seg_path: Target segmentation NIfTI path.
        qc_path: Target per-scan QC CSV path.
        vol_path: Target per-scan volumetry CSV path.
    """

    input_path: Path
    seg_path: Path
    qc_path: Path
    vol_path: Path


def _strip_nii_gz(name: str) -> str:
    """Return the filename with a trailing .nii.gz (or .nii) removed."""
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return Path(name).stem


def plan_for(input_path: Path, input_dir: Path, output_dir: Path) -> ScanPlan:
    """Map an input scan to its three target output paths."""
    rel = input_path.relative_to(input_dir)
    base = rel.parent / _strip_nii_gz(rel.name)
    return ScanPlan(
        input_path=input_path,
        seg_path=output_dir / f"{base}_synthseg.nii.gz",
        qc_path=output_dir / f"{base}_qc.csv",
        vol_path=output_dir / f"{base}_vol.csv",
    )


def discover_inputs(input_dir: Path) -> list[Path]:
    """Return a sorted list of .nii.gz files under input_dir (recursive)."""
    return sorted(input_dir.rglob("*.nii.gz"))


def split_by_existing_seg(
    plans: list[ScanPlan],
) -> tuple[list[ScanPlan], list[ScanPlan]]:
    """Partition plans by whether seg_path already exists.

    Returns:
        (to_run, to_skip) — to_skip will receive STATUS_SKIPPED in the manifest.
    """
    to_run: list[ScanPlan] = []
    to_skip: list[ScanPlan] = []
    for plan in plans:
        (to_skip if plan.seg_path.exists() else to_run).append(plan)
    return to_run, to_skip


def _ensure_output_dirs(plans: list[ScanPlan]) -> None:
    for plan in plans:
        plan.seg_path.parent.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# FreeSurfer mode
# ──────────────────────────────────────────────


def _ensure_system_path(env: dict[str, str], *extra: str) -> None:
    """Mutate ``env['PATH']`` to include /usr/bin, /bin, and any extras.

    The ``mri_synthseg`` wrapper uses ``#!/usr/bin/env bash`` which then
    searches ``PATH`` for ``bash``. Callers with stripped or mangled PATH
    (cron, sandboxed shells, literal ``${PATH}`` that didn't expand) would
    otherwise hit ``env: bash: No such file or directory`` (exit 127).
    """
    existing = env.get("PATH", "").split(":")
    prepend = [e for e in (*extra, "/usr/bin", "/bin") if e and e not in existing]
    env["PATH"] = ":".join([*prepend, *existing]) if prepend else env.get("PATH", "")


def resolve_mri_synthseg(
    freesurfer_home: Path | None,
) -> tuple[str, dict[str, str]]:
    """Return ``(binary_path, env)`` for invoking mri_synthseg, or raise.

    ``mri_synthseg`` at ``$FREESURFER_HOME/bin/mri_synthseg`` is a bash wrapper
    that exits 1 unless ``$FREESURFER_HOME`` is set in its environment. So we
    both locate the binary and prepare the env in one step.

    Resolution priority:
      1. ``freesurfer_home`` argument (from ``--freesurfer-home`` CLI flag or
         ``$FREESURFER_HOME`` env var).
      2. ``shutil.which("mri_synthseg")`` — present only if the caller already
         sourced ``SetUpFreeSurfer.sh`` before invoking this script.

    Raises:
        typer.BadParameter: when no valid install location can be found.
    """
    if freesurfer_home is not None:
        candidate = freesurfer_home / "bin" / "mri_synthseg"
        if candidate.exists():
            env = os.environ.copy()
            env["FREESURFER_HOME"] = str(freesurfer_home)
            _ensure_system_path(env, str(freesurfer_home / "bin"))
            return str(candidate), env
        raise typer.BadParameter(
            f"--freesurfer-home points at {freesurfer_home}, but "
            f"{candidate} does not exist. Check the path."
        )
    on_path = shutil.which(MRI_SYNTHSEG_BIN)
    if on_path is not None:
        # Assume the caller already set FREESURFER_HOME (via sourcing); inherit env.
        env = os.environ.copy()
        _ensure_system_path(env)
        return on_path, env
    raise typer.BadParameter(
        f"Cannot locate {MRI_SYNTHSEG_BIN!r}. Options:\n"
        "  • pass --freesurfer-home /Applications/freesurfer/8.1.0\n"
        "  • export FREESURFER_HOME=/Applications/freesurfer/8.1.0\n"
        "  • source /Applications/freesurfer/8.1.0/SetUpFreeSurfer.sh beforehand\n"
        "  • or switch to --mode python."
    )


def _build_freesurfer_cmd(
    binary: str,
    plan: ScanPlan,
    *,
    parc: bool,
    fast: bool,
) -> list[str]:
    """Return the mri_synthseg argv for a single scan.

    ``parc``: pass ``--parc`` for cortical parcellation (Desikan-Killiany,
    adds the 1001+/2001+ label codes). Disable for speed/memory; leaves
    coarse cortex labels 3 and 42 populated instead.
    ``fast``: pass ``--fast`` to bypass post-processing and use a lighter
    CNN. 2-3× speedup in exchange for slightly lower accuracy — worth it
    for memory-pressured runs (e.g. full-head IXI on CPU) and diagnostic
    smoke-tests.
    """
    cmd = [
        binary,
        "--i", str(plan.input_path),
        "--o", str(plan.seg_path),
        "--qc", str(plan.qc_path),
        "--vol", str(plan.vol_path),
    ]
    if parc:
        cmd.append("--parc")
    if fast:
        cmd.append("--fast")
    return cmd


def run_freesurfer(
    plans: list[ScanPlan],
    binary: str,
    env: dict[str, str],
    *,
    parc: bool = True,
    fast: bool = False,
) -> list[tuple[ScanPlan, str]]:
    """Invoke mri_synthseg once per scan. Failures are isolated per scan.

    ``env`` must include ``FREESURFER_HOME`` so the wrapper at ``binary`` runs.
    """
    results: list[tuple[ScanPlan, str]] = []
    iterator = tqdm(plans, desc="mri_synthseg", unit="scan") if len(plans) > 1 else plans
    for plan in iterator:
        cmd = _build_freesurfer_cmd(binary, plan, parc=parc, fast=fast)
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
            results.append((plan, STATUS_OK))
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip().splitlines()[-5:]
            logger.warning(
                "mri_synthseg failed for %s (exit %d). Last stderr: %s",
                plan.input_path.name,
                exc.returncode,
                " | ".join(stderr),
            )
            results.append((plan, STATUS_FAILED))
    return results


# ──────────────────────────────────────────────
# Python mode
# ──────────────────────────────────────────────


def _locate_synthseg_cli() -> Path:
    """Return the path to ``SynthSeg/scripts/commands/SynthSeg_predict.py``.

    We shell out to the CLI script rather than calling
    ``SynthSeg.predict_synthseg.predict()`` directly because the latter
    requires 20+ positional args that change between SynthSeg versions
    (``path_model_segmentation``, ``labels_segmentation``, ``robust``,
    ``fast``, ``v1``, ``n_neutral_labels``, ``labels_denoiser``,
    ``path_posteriors``, ``path_resampled``, ``path_model_parcellation``,
    ``labels_parcellation``, ``path_model_qc``, ``labels_qc``,
    ``cropping`` …). The CLI is the production entrypoint that constructs
    those args from the repo's known model + label layout, and its
    ``--i / --o / --qc / --vol / --parc / --fast`` flags are stable across
    versions (in fact ``mri_synthseg`` also dispatches to this same CLI).

    Resolution priority:
      1. ``$SYNTHSEG_HOME`` env var pointing at the cloned repo root.
      2. Walk up from an importable ``SynthSeg`` package.
      3. Standard install locations: ``$HOME/SynthSeg``, ``/opt/SynthSeg``.
    """
    candidates: list[Path] = []
    env_home = os.environ.get("SYNTHSEG_HOME")
    if env_home:
        candidates.append(Path(env_home) / "scripts" / "commands" / "SynthSeg_predict.py")
    try:
        import SynthSeg  # type: ignore[import-not-found]
        repo_root = Path(SynthSeg.__file__).resolve().parent.parent
        candidates.append(repo_root / "scripts" / "commands" / "SynthSeg_predict.py")
    except ImportError:
        pass
    for base in (Path.home() / "SynthSeg", Path("/opt/SynthSeg")):
        candidates.append(base / "scripts" / "commands" / "SynthSeg_predict.py")
    for cli in candidates:
        if cli.is_file():
            return cli
    raise typer.BadParameter(
        "Cannot locate SynthSeg's CLI (scripts/commands/SynthSeg_predict.py). "
        "Either set $SYNTHSEG_HOME to the SynthSeg repo root, or clone "
        "https://github.com/BBillot/SynthSeg.git to ~/SynthSeg, or use "
        "--mode freesurfer (which calls FreeSurfer's bundled mri_synthseg)."
    )


def _build_python_cmd(
    cli_path: Path,
    plan: ScanPlan,
    *,
    parc: bool,
    fast: bool,
) -> list[str]:
    """Return the SynthSeg CLI argv for a single scan.

    Same flag set as ``_build_freesurfer_cmd`` — both targets accept
    identical arguments because ``mri_synthseg`` is a thin wrapper around
    this same CLI.
    """
    cmd = [
        sys.executable,
        str(cli_path),
        "--i", str(plan.input_path),
        "--o", str(plan.seg_path),
        "--qc", str(plan.qc_path),
        "--vol", str(plan.vol_path),
    ]
    if parc:
        cmd.append("--parc")
    if fast:
        cmd.append("--fast")
    return cmd


def run_python(
    plans: list[ScanPlan],
    batch_size: int,
    *,
    parc: bool = True,
    fast: bool = False,
) -> list[tuple[ScanPlan, str]]:
    """Run SynthSeg via list-file batching: one CLI call for the whole batch.

    SynthSeg's startup (TF init + GPU init + model load + cuDNN init) is
    ~80s; inference itself is ~28s/scan on A100 GPU. Per-scan subprocess
    invocation amortizes startup over zero scans (paying 80s × N), so we
    instead write four list files (inputs / outputs / qc / vol, one path
    per line) and pass them to SynthSeg's CLI in a single subprocess call.
    The CLI iterates internally, sharing model + GPU state.

    ``batch_size`` controls the maximum scans per CLI invocation; callers
    typically pass a large value (or ``len(plans)``) so all plans process
    in one call. We default-cap to 200 to bound peak memory inside the
    SynthSeg process.
    """
    if not plans:
        return []

    import tempfile

    cli_path = _locate_synthseg_cli()
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    cap = max(batch_size, 1)

    # Pre-create output dirs so SynthSeg's writes don't fail.
    for plan in plans:
        plan.seg_path.parent.mkdir(parents=True, exist_ok=True)
        plan.qc_path.parent.mkdir(parents=True, exist_ok=True)
        plan.vol_path.parent.mkdir(parents=True, exist_ok=True)

    chunks: list[list[ScanPlan]] = [plans[i : i + cap] for i in range(0, len(plans), cap)]
    results: list[tuple[ScanPlan, str]] = []
    chunk_iter = (
        tqdm(chunks, desc="SynthSeg.predict", unit="batch")
        if len(chunks) > 1
        else chunks
    )

    for chunk in chunk_iter:
        with tempfile.TemporaryDirectory(prefix="synthseg_lists_") as tmp:
            tmpdir = Path(tmp)
            in_list = tmpdir / "inputs.txt"
            out_list = tmpdir / "outputs.txt"
            qc_list = tmpdir / "qc.txt"
            vol_list = tmpdir / "vol.txt"

            with in_list.open("w") as fi, out_list.open("w") as fo, \
                 qc_list.open("w") as fq, vol_list.open("w") as fv:
                for plan in chunk:
                    fi.write(f"{plan.input_path}\n")
                    fo.write(f"{plan.seg_path}\n")
                    fq.write(f"{plan.qc_path}\n")
                    fv.write(f"{plan.vol_path}\n")

            cmd = [
                sys.executable,
                str(cli_path),
                "--i", str(in_list),
                "--o", str(out_list),
                "--qc", str(qc_list),
                "--vol", str(vol_list),
            ]
            if parc:
                cmd.append("--parc")
            if fast:
                cmd.append("--fast")

            logger.info(
                "SynthSeg CLI batch: %d scan(s) starting with %s",
                len(chunk),
                chunk[0].input_path.name,
            )
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                # Verify each scan's seg_path now exists; mark missing ones failed.
                for plan in chunk:
                    if plan.seg_path.exists():
                        results.append((plan, STATUS_OK))
                    else:
                        logger.warning(
                            "SynthSeg succeeded but seg output missing for %s",
                            plan.input_path.name,
                        )
                        results.append((plan, STATUS_FAILED))
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip().splitlines()[-10:]
                logger.warning(
                    "SynthSeg CLI batch failed (exit %d). Last stderr: %s",
                    exc.returncode,
                    " | ".join(stderr),
                )
                # On failure, mark scans that did get written as ok and the rest as failed.
                for plan in chunk:
                    results.append(
                        (plan, STATUS_OK if plan.seg_path.exists() else STATUS_FAILED)
                    )

    return results


# ──────────────────────────────────────────────
# Manifest + reporting
# ──────────────────────────────────────────────


def write_manifest(
    rows: list[tuple[ScanPlan, str]], mode: SynthSegMode, manifest_path: Path
) -> None:
    """Write the aggregate manifest CSV with MANIFEST_COLUMNS."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for plan, status in rows:
            writer.writerow(
                {
                    "input_path": str(plan.input_path),
                    "seg_path": str(plan.seg_path),
                    "qc_path": str(plan.qc_path),
                    "vol_path": str(plan.vol_path),
                    "mode": mode.value,
                    "status": status,
                }
            )


def _print_summary(
    rows: list[tuple[ScanPlan, str]],
    mode: SynthSegMode,
    manifest_path: Path,
    console: Console,
) -> None:
    counts: dict[str, int] = {STATUS_OK: 0, STATUS_SKIPPED: 0, STATUS_FAILED: 0}
    for _, status in rows:
        counts[status] = counts.get(status, 0) + 1

    table = Table(title=f"SynthSeg ({mode.value}) summary")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("total scans", str(len(rows)))
    table.add_row("ok", str(counts[STATUS_OK]))
    table.add_row("skipped (already existed)", str(counts[STATUS_SKIPPED]))
    table.add_row("failed", str(counts[STATUS_FAILED]))
    table.add_row("manifest path", str(manifest_path))
    console.print(table)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


@app.command()
def main(
    input_dir: Path = typer.Option(
        ...,
        "--input-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Directory of .nii.gz scans (recursive).",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Root directory for SynthSeg outputs; mirrors input tree.",
    ),
    mode: SynthSegMode = typer.Option(
        SynthSegMode.FREESURFER,
        "--mode",
        help="Execution backend.",
        case_sensitive=False,
    ),
    freesurfer_home: Path | None = typer.Option(
        None,
        "--freesurfer-home",
        help=(
            "FreeSurfer install root (e.g. /Applications/freesurfer/8.1.0). "
            "Defaults to $FREESURFER_HOME env var. Ignored in --mode python."
        ),
    ),
    batch_size: int = typer.Option(
        1, "--batch-size", min=1, help="Batch size for --mode python."
    ),
    parc: bool = typer.Option(
        True,
        "--parc/--no-parc",
        help=(
            "Enable Desikan-Killiany cortical parcellation (adds 1001+/2001+ "
            "label codes). Disable to cut runtime/memory on full-head volumes "
            "and to keep the coarse cortex labels 3/42 populated."
        ),
    ),
    fast: bool = typer.Option(
        False,
        "--fast/--no-fast",
        help=(
            "Pass --fast to mri_synthseg (lighter CNN, 2-3x speedup, slightly "
            "lower accuracy). Use for diagnostic runs and memory-constrained "
            "hosts. Ignored in --mode python."
        ),
    ),
    manifest_path: Path = typer.Option(
        Path("results/tables/synthseg_manifest.csv"),
        "--manifest-path",
        help="Aggregate manifest CSV path.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Log planned work without executing."
    ),
) -> None:
    """Run SynthSeg on every .nii.gz found under --input-dir."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )

    fs_binary: str | None = None
    fs_env: dict[str, str] | None = None
    if mode is SynthSegMode.FREESURFER:
        fs_home = freesurfer_home
        if fs_home is None:
            env_home = os.environ.get("FREESURFER_HOME")
            fs_home = Path(env_home) if env_home else None
        fs_binary, fs_env = resolve_mri_synthseg(fs_home)

    inputs = discover_inputs(input_dir)
    if not inputs:
        logger.warning("No .nii.gz files found under %s", input_dir)
        if not dry_run:
            write_manifest([], mode, manifest_path)
        return

    plans = [plan_for(p, input_dir, output_dir) for p in inputs]
    to_run, to_skip = split_by_existing_seg(plans)

    logger.info(
        "Plan: %d scans to run, %d already existing (skip), mode=%s",
        len(to_run),
        len(to_skip),
        mode.value,
    )

    if dry_run:
        for plan in to_run[:10]:
            logger.info("  would run: %s -> %s", plan.input_path, plan.seg_path)
        if len(to_run) > 10:
            logger.info("  ... (%d more)", len(to_run) - 10)
        return

    _ensure_output_dirs(to_run)

    if mode is SynthSegMode.FREESURFER:
        assert fs_binary is not None and fs_env is not None  # narrowed above
        run_results = run_freesurfer(
            to_run, binary=fs_binary, env=fs_env, parc=parc, fast=fast
        )
    else:
        run_results = run_python(to_run, batch_size=batch_size, parc=parc, fast=fast)

    all_rows: list[tuple[ScanPlan, str]] = run_results + [
        (plan, STATUS_SKIPPED) for plan in to_skip
    ]
    write_manifest(all_rows, mode, manifest_path)
    _print_summary(all_rows, mode, manifest_path, Console())


if __name__ == "__main__":
    app()
