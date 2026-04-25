#!/usr/bin/env python3
"""Reproducibility bundle — pack results CSVs + figures + provenance into tar.gz.

Single-shot archive for a paper-submission supplementary materials drop or a
collaborator handoff. Produces ``bundles/neuroqc_bundle_<UTC>_<git_short>.tar.gz``
containing:

* ``results/tables/*.csv`` — every results CSV under ``--results-dir``
  (machine_preference, per_structure_dice, cortical_thickness, iqm_features,
  synthseg_qc_features, corruption_manifest, benchmark_subsample,
  3d_vlm_scores_seed_*, 2d_vlm_scores_seed_*, finetuned_scores_seed_*).
* ``figures/*.{png,svg}`` — every figure under ``--figures-dir``.
* ``results/tables/finetune_run_info_*.json`` — fine-tune provenance JSONs.
* ``results/tables/finetune_diff_*.diff`` — git diffs from dirty-tree fine-tunes.
* ``manifest.json`` — for every file in the bundle: SHA-256, byte size, relative
  path. Plus the bundle's own metadata (creation time, git commit, dirty bit,
  library versions, hostname).
* ``README.txt`` — auto-generated, explains structure + how to reuse the bundle
  (extract → run ``code/visualize.py --results-dir <extracted>/results/tables``).

LoRA checkpoints (``results/checkpoints/``, large) and SynthSeg outputs (``data/
derivatives/synthseg/``, very large) are EXCLUDED by default. Pass
``--include-checkpoints`` and/or ``--include-segmentations`` to include them.

Usage::

    python scripts/bundle_results.py
    python scripts/bundle_results.py --tag pre-revision-2026-04
    python scripts/bundle_results.py --include-checkpoints --dry-run
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import tarfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_RESULTS_DIR = _REPO_ROOT / "results" / "tables"
_DEFAULT_FIGURES_DIR = _REPO_ROOT / "figures"
_DEFAULT_CHECKPOINTS_DIR = _REPO_ROOT / "results" / "checkpoints"
_DEFAULT_SEGMENTATIONS_DIR = _REPO_ROOT / "data" / "derivatives" / "synthseg"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "bundles"

# File-pattern groups; each group has (subdir relative to project root, glob).
_RESULTS_PATTERNS: tuple[str, ...] = (
    "*.csv",
    "*.json",
    "*.diff",
)
_FIGURE_PATTERNS: tuple[str, ...] = ("*.png", "*.svg")


# ──────────────────────────────────────────────
# Reproducibility metadata
# ──────────────────────────────────────────────


def _git_env() -> dict[str, str]:
    """Subprocess env with system PATH prepended (for sandboxed runners)."""
    env = os.environ.copy()
    env["PATH"] = f"/usr/local/bin:/usr/bin:/bin:{env.get('PATH', '')}"
    return env


def _git_command(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], env=_git_env(), text=True, cwd=_REPO_ROOT
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def get_git_state() -> dict[str, Any]:
    """Capture the commit hash, dirty bit, and (if dirty) diff."""
    commit = _git_command("rev-parse", "HEAD") or "unknown"
    short = _git_command("rev-parse", "--short", "HEAD") or "unknown"
    status = _git_command("status", "--porcelain")
    dirty = bool(status.strip())
    diff = _git_command("diff", "HEAD") if dirty else ""
    return {
        "commit": commit,
        "short_commit": short,
        "dirty": dirty,
        "status": status,
        "diff": diff,
    }


def get_library_versions() -> dict[str, str]:
    """Versions of the libraries that materially affect reproducibility."""
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for name in (
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "monai",
        "torchio",
        "nibabel",
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
    ):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = "not-installed"
    return versions


# ──────────────────────────────────────────────
# File discovery + checksum
# ──────────────────────────────────────────────


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _glob_dir(directory: Path, patterns: Iterable[str]) -> list[Path]:
    """Sorted, deduplicated list of files under ``directory`` matching any pattern."""
    if not directory.is_dir():
        return []
    out: set[Path] = set()
    for pattern in patterns:
        out.update(directory.glob(pattern))
    return sorted(p for p in out if p.is_file())


def _walk_dir(directory: Path) -> list[Path]:
    """Recursive listing of every file under ``directory``."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.rglob("*") if p.is_file())


def collect_files(
    *,
    results_dir: Path,
    figures_dir: Path,
    checkpoints_dir: Path,
    segmentations_dir: Path,
    include_csvs: bool,
    include_figures: bool,
    include_provenance: bool,
    include_checkpoints: bool,
    include_segmentations: bool,
) -> dict[Path, str]:
    """Build ``{absolute_path: arcname}`` mapping for files going into the bundle.

    The arcname is the path relative to the project root, so the extracted
    bundle mirrors the project layout (``results/tables/foo.csv`` etc).
    """
    out: dict[Path, str] = {}

    def _add(p: Path) -> None:
        try:
            arcname = str(p.resolve().relative_to(_REPO_ROOT))
        except ValueError:
            arcname = p.name
        out[p.resolve()] = arcname

    if include_csvs or include_provenance:
        for path in _glob_dir(results_dir, _RESULTS_PATTERNS):
            is_provenance = (
                path.name.startswith("finetune_run_info_")
                or path.name.startswith("finetune_diff_")
            )
            if is_provenance and not include_provenance:
                continue
            if not is_provenance and not include_csvs:
                continue
            _add(path)

    if include_figures:
        for path in _glob_dir(figures_dir, _FIGURE_PATTERNS):
            _add(path)

    if include_checkpoints:
        for path in _walk_dir(checkpoints_dir):
            _add(path)

    if include_segmentations:
        for path in _walk_dir(segmentations_dir):
            _add(path)

    return out


# ──────────────────────────────────────────────
# Manifest + README
# ──────────────────────────────────────────────


def build_manifest(
    *,
    files: dict[Path, str],
    bundle_filename: str,
    git_state: dict[str, Any],
    library_versions: dict[str, str],
    tag: str | None,
    options: dict[str, bool],
) -> dict[str, Any]:
    """Build the bundle manifest. Pure function — easy to test."""
    file_entries: list[dict[str, Any]] = []
    total_bytes = 0
    for path, arcname in sorted(files.items(), key=lambda kv: kv[1]):
        if not path.is_file():
            continue
        size = path.stat().st_size
        file_entries.append(
            {
                "arcname": arcname,
                "size_bytes": size,
                "sha256": _sha256_file(path),
            }
        )
        total_bytes += size

    return {
        "bundle_filename": bundle_filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "git": git_state,
        "library_versions": library_versions,
        "options": options,
        "n_files": len(file_entries),
        "total_bytes": total_bytes,
        "files": file_entries,
    }


_README_TEMPLATE = """\
NeuroQC reproducibility bundle
==============================

Bundle: {bundle_filename}
Created: {created_at}
Git commit: {git_short} (dirty={git_dirty})
Tag: {tag}

This archive contains the results CSVs and figures that produced the
NeuroQC paper's claims at the time of bundling. Every file in the bundle
has a SHA-256 checksum recorded in ``manifest.json`` so a reviewer can
verify the bundle was extracted intact.

Layout
------
results/tables/        Per-phase results CSVs (machine_preference,
                       per_structure_dice, cortical_thickness, iqm_features,
                       3d/2d_vlm_scores_seed_*, finetuned_scores_seed_*, ...).
results/tables/        Provenance JSONs from fine-tune runs
finetune_run_info_*.json   (git hash, hyperparameters, library versions,
                            bucket distributions, val SRCC at best epoch).
results/tables/        Git diffs captured at fine-tune time when the working
finetune_diff_*.diff       tree was dirty.
figures/               Auto-generated figures (Figs 2/3/4/5/6/8) as PNG + SVG.
manifest.json          File index with per-file size + SHA-256, plus bundle
                       metadata (host, user, git state, library versions).
README.txt             This file.

Reusing the bundle
------------------
1. Extract the archive::

       tar xzf {bundle_filename}
       cd {bundle_dir}

2. Verify checksums against the manifest::

       python -c "import hashlib, json; m=json.load(open('manifest.json'));\\
       [print(f['arcname'], hashlib.sha256(open(f['arcname'],'rb').read()).hexdigest()==f['sha256']) for f in m['files']]"

3. Regenerate figures from the cached CSVs (no experiments re-run)::

       python code/visualize.py --all \\
           --results-dir results/tables \\
           --output-dir  figures_regenerated

   The bundle's own ``figures/`` is what the paper used; ``figures_regenerated``
   is the reviewer's reproduction. They should match modulo bootstrap CIs (which
   are deterministic given the same ``--bootstrap-seed``, default 42).

Library versions at bundle time
-------------------------------
{lib_versions_block}

Reproducibility caveats
-----------------------
* Bootstrap CI determinism is exact within a (numpy, scipy) version pair. The
  manifest pins both versions; reviewers using different versions may see CI
  bounds that differ by < 0.01 SRCC.
* Fine-tune scores in ``finetuned_scores_seed_*.csv`` come from a bf16 + 4-bit
  bnb run, which is not bit-deterministic across hardware. The provenance JSON
  records seed + git hash + lib versions; reproducing the exact scores requires
  matching all three.
* Phase 03 SynthSeg outputs (``data/derivatives/synthseg/``) and LoRA checkpoints
  (``results/checkpoints/``) are NOT included by default — the CSVs that
  summarise them suffice for figure regeneration. Pass
  ``--include-segmentations`` and ``--include-checkpoints`` to bundle them.
"""


def render_readme(manifest: dict[str, Any], bundle_dir: str) -> str:
    """Render the bundle's README.txt content from the manifest."""
    lib_block = "\n".join(
        f"  {name}: {ver}" for name, ver in sorted(manifest["library_versions"].items())
    )
    return _README_TEMPLATE.format(
        bundle_filename=manifest["bundle_filename"],
        created_at=manifest["created_at"],
        git_short=manifest["git"]["short_commit"],
        git_dirty=manifest["git"]["dirty"],
        tag=manifest["tag"] or "(none)",
        bundle_dir=bundle_dir,
        lib_versions_block=lib_block,
    )


# ──────────────────────────────────────────────
# Tar writing
# ──────────────────────────────────────────────


def write_bundle(
    *,
    files: dict[Path, str],
    manifest: dict[str, Any],
    output_path: Path,
) -> Path:
    """Write the tar.gz with files + manifest.json + README.txt under a single
    top-level directory named after the bundle.

    Returns the absolute path to the written archive.
    """
    bundle_dir = output_path.stem.removesuffix(".tar")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    readme_text = render_readme(manifest, bundle_dir)
    manifest_text = json.dumps(manifest, indent=2, default=str)

    try:
        with tarfile.open(tmp, "w:gz") as tar:
            for path, arcname in sorted(files.items(), key=lambda kv: kv[1]):
                if not path.is_file():
                    continue
                tar.add(path, arcname=f"{bundle_dir}/{arcname}")
            # README.txt
            readme_bytes = readme_text.encode("utf-8")
            info = tarfile.TarInfo(name=f"{bundle_dir}/README.txt")
            info.size = len(readme_bytes)
            info.mtime = int(datetime.now(timezone.utc).timestamp())
            from io import BytesIO

            tar.addfile(info, BytesIO(readme_bytes))
            # manifest.json
            manifest_bytes = manifest_text.encode("utf-8")
            info = tarfile.TarInfo(name=f"{bundle_dir}/manifest.json")
            info.size = len(manifest_bytes)
            info.mtime = int(datetime.now(timezone.utc).timestamp())
            tar.addfile(info, BytesIO(manifest_bytes))
        os.replace(tmp, output_path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
    return output_path


# ──────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────


def make_bundle(
    *,
    results_dir: Path = _DEFAULT_RESULTS_DIR,
    figures_dir: Path = _DEFAULT_FIGURES_DIR,
    checkpoints_dir: Path = _DEFAULT_CHECKPOINTS_DIR,
    segmentations_dir: Path = _DEFAULT_SEGMENTATIONS_DIR,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    include_csvs: bool = True,
    include_figures: bool = True,
    include_provenance: bool = True,
    include_checkpoints: bool = False,
    include_segmentations: bool = False,
    tag: str | None = None,
    dry_run: bool = False,
) -> Path | None:
    """Assemble + write the bundle. Returns the archive path or ``None`` on dry-run."""
    files = collect_files(
        results_dir=results_dir,
        figures_dir=figures_dir,
        checkpoints_dir=checkpoints_dir,
        segmentations_dir=segmentations_dir,
        include_csvs=include_csvs,
        include_figures=include_figures,
        include_provenance=include_provenance,
        include_checkpoints=include_checkpoints,
        include_segmentations=include_segmentations,
    )

    git_state = get_git_state()
    lib_versions = get_library_versions()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    git_short = git_state["short_commit"] or "unknown"
    suffix = f"_{tag}" if tag else ""
    bundle_filename = f"neuroqc_bundle_{timestamp}_{git_short}{suffix}.tar.gz"
    output_path = output_dir / bundle_filename

    options = {
        "include_csvs": include_csvs,
        "include_figures": include_figures,
        "include_provenance": include_provenance,
        "include_checkpoints": include_checkpoints,
        "include_segmentations": include_segmentations,
    }

    manifest = build_manifest(
        files=files,
        bundle_filename=bundle_filename,
        git_state=git_state,
        library_versions=lib_versions,
        tag=tag,
        options=options,
    )

    if dry_run:
        logger.info(
            "Dry run — would write %s with %d files (%.2f MB):",
            output_path, manifest["n_files"], manifest["total_bytes"] / 1e6,
        )
        for entry in manifest["files"]:
            logger.info("  %s (%d bytes)", entry["arcname"], entry["size_bytes"])
        return None

    archive_path = write_bundle(
        files=files, manifest=manifest, output_path=output_path
    )
    logger.info(
        "Wrote bundle: %s (%d files, %.2f MB)",
        archive_path, manifest["n_files"], manifest["total_bytes"] / 1e6,
    )
    return archive_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR,
        help=f"Default: {_DEFAULT_RESULTS_DIR.relative_to(_REPO_ROOT)}",
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=_DEFAULT_FIGURES_DIR,
        help=f"Default: {_DEFAULT_FIGURES_DIR.relative_to(_REPO_ROOT)}",
    )
    parser.add_argument(
        "--checkpoints-dir", type=Path, default=_DEFAULT_CHECKPOINTS_DIR,
    )
    parser.add_argument(
        "--segmentations-dir", type=Path, default=_DEFAULT_SEGMENTATIONS_DIR,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR,
        help=f"Default: {_DEFAULT_OUTPUT_DIR.relative_to(_REPO_ROOT)}",
    )
    parser.add_argument(
        "--no-csvs", action="store_true",
        help="Exclude results CSVs (rare; useful for figures-only bundles).",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Exclude figures (rare; useful for CSV-only bundles).",
    )
    parser.add_argument(
        "--no-provenance", action="store_true",
        help="Exclude finetune_run_info_*.json + finetune_diff_*.diff.",
    )
    parser.add_argument(
        "--include-checkpoints", action="store_true",
        help="Include LoRA adapter checkpoints (large; default off).",
    )
    parser.add_argument(
        "--include-segmentations", action="store_true",
        help="Include SynthSeg outputs (very large; default off).",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Suffix to append to the bundle filename (e.g. 'pre-revision-2026-04').",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List files that would go in the bundle and exit.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    archive = make_bundle(
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        checkpoints_dir=args.checkpoints_dir,
        segmentations_dir=args.segmentations_dir,
        output_dir=args.output_dir,
        include_csvs=not args.no_csvs,
        include_figures=not args.no_figures,
        include_provenance=not args.no_provenance,
        include_checkpoints=args.include_checkpoints,
        include_segmentations=args.include_segmentations,
        tag=args.tag,
        dry_run=args.dry_run,
    )
    if not args.dry_run and archive is not None:
        # Echo path on stdout for shell-pipeline use.
        print(archive)
    return 0


if __name__ == "__main__":
    sys.exit(main())
