#!/usr/bin/env python3
"""NeuroQC Phase 2 — Corruption Generation (thin CLI wrapper).

Delegates to ``nobrainer.qc.corrupt.generate_corrupted_dataset``. This script
only parses CLI arguments, calls the nobrainer function, and serialises the
returned metadata list to ``results/tables/corruption_manifest.csv``.

Inputs:
    --input-dir       Directory of reference .nii.gz scans (output of phase 1).
    --output-dir      Root directory for corrupted outputs (one subdir per corruption).
    --corruptions     Comma-separated subset (e.g. "motion,noise") or "all".
    --severities      Comma-separated subset (e.g. "1,3,5") or "all".
    --dataset-tag     One of {ixi, oasis, fastmri}. Recorded in every manifest row
                      so downstream analyses can group by dataset.
    --manifest-path   CSV path for the produced corruption manifest.
    --dry-run         Print the plan without writing any corrupted files.

Outputs:
    <output-dir>/<corruption>/severity_<n>/<scan>.nii.gz   Corrupted NIfTIs.
    <output-dir>/<corruption>/severity_<n>/<scan>.json     Per-scan provenance sidecars.
    <manifest-path>                                         CSV aggregate manifest.

Note on resume semantics:
    The underlying generator skips per-scan outputs that already exist
    (resume=True), so the returned metadata — and therefore the manifest —
    reflects only what was generated in *this* run. Delete the relevant
    outputs to force regeneration.

Usage:
    python code/02_generate_corruptions.py \\
        --input-dir data/ixi/derivatives/references \\
        --output-dir data/ixi/derivatives/corruptions \\
        --corruptions motion,ghosting,spike \\
        --severities 1,3,5
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from nobrainer.qc.corrupt import generate_corrupted_dataset
from nobrainer.qc.corruption_configs import get_corruption_configs

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

ALL_KEYWORD: str = "all"
VALID_SEVERITIES: tuple[int, ...] = (1, 2, 3, 4, 5)

# Keep in sync with code/01_curate_references.py::DATASET_TAGS.
DATASET_TAGS: tuple[str, ...] = ("ixi", "oasis", "fastmri")

MANIFEST_COLUMNS: list[str] = [
    "ref_path",
    "cor_path",
    "corruption_type",
    "corruption_domain",
    "severity",
    "seed",
    "transform_params",
    "dataset_tag",
]

logger = logging.getLogger(__name__)
app = typer.Typer(
    help="NeuroQC Phase 2 — generate corrupted scans via nobrainer.qc.",
    add_completion=False,
)


# ──────────────────────────────────────────────
# CLI argument parsing
# ──────────────────────────────────────────────


def _parse_corruptions(spec: str) -> list[str] | None:
    """Parse the --corruptions CLI value.

    Returns None for "all" (so the underlying function uses its default),
    or a list of corruption names to apply. Validates names against the
    nobrainer corruption registry.
    """
    if spec.strip().lower() == ALL_KEYWORD:
        return None
    items = [x.strip() for x in spec.split(",") if x.strip()]
    if not items:
        raise typer.BadParameter("--corruptions must be 'all' or a non-empty list.")
    known = set(get_corruption_configs().keys())
    unknown = [x for x in items if x not in known]
    if unknown:
        raise typer.BadParameter(
            f"Unknown corruption(s): {unknown}. Valid: {sorted(known)}"
        )
    return items


def _parse_severities(spec: str) -> list[int] | None:
    """Parse the --severities CLI value.

    Returns None for "all", or a list of ints in VALID_SEVERITIES.
    """
    if spec.strip().lower() == ALL_KEYWORD:
        return None
    try:
        items = [int(x.strip()) for x in spec.split(",") if x.strip()]
    except ValueError as exc:
        raise typer.BadParameter(
            f"--severities must be integers or 'all'; got {spec!r}"
        ) from exc
    if not items:
        raise typer.BadParameter("--severities must be 'all' or a non-empty list.")
    invalid = [x for x in items if x not in VALID_SEVERITIES]
    if invalid:
        raise typer.BadParameter(
            f"Severities must be in {VALID_SEVERITIES}; got invalid: {invalid}"
        )
    return items


# ──────────────────────────────────────────────
# Manifest serialisation
# ──────────────────────────────────────────────


def _read_existing_manifest(path: Path) -> list[dict[str, str]]:
    """Return existing manifest rows, or [] when the file doesn't exist.

    Validates that the header matches MANIFEST_COLUMNS exactly; a mismatch
    signals schema drift between 02, 02b, or an older 02 run and must be
    reconciled by hand rather than silently discarded.
    """
    if not path.exists():
        return []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if set(fieldnames) != set(MANIFEST_COLUMNS):
            raise ValueError(
                f"Existing manifest schema mismatch at {path}: "
                f"expected {MANIFEST_COLUMNS}, got {fieldnames}"
            )
        return list(reader)


def write_manifest(
    metadata: list[dict], manifest_path: Path, dataset_tag: str
) -> int:
    """Idempotently update the shared corruption manifest for this run.

    Read-modify-write against any pre-existing file so the manifest stays
    compatible with ``code/02b_corrupt_kspace_motion.py`` (which writes
    ``motion_kspace`` rows) and with earlier ``02`` runs for other datasets.

    Rows are dropped only when **both** conditions hold:
        * ``row.corruption_type`` is one of the types produced in this run
          (derived from ``metadata``), AND
        * ``row.dataset_tag`` equals the current ``dataset_tag``.

    Everything else is preserved verbatim — so re-running 02 on one dataset
    never touches another dataset's rows, and never wipes 02b's k-space
    motion rows.

    Only ``metadata`` entries that carry every required key are written;
    partial rows (per-scan failures returned with an ``"error"`` key) are
    dropped because the CLAUDE.md manifest schema has no error column.

    Args:
        metadata: The list returned by generate_corrupted_dataset.
        manifest_path: Output CSV path (shared with 02b).
        dataset_tag: Dataset origin tag written to every new row.

    Returns:
        Number of new rows written (not total rows in the file).
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    required = set(MANIFEST_COLUMNS) - {"dataset_tag"}

    valid_entries = [e for e in metadata if required.issubset(e.keys())]
    types_written = {str(e["corruption_type"]) for e in valid_entries}

    existing = _read_existing_manifest(manifest_path)
    kept = [
        row
        for row in existing
        if not (
            row.get("corruption_type") in types_written
            and row.get("dataset_tag") == dataset_tag
        )
    ]

    rows_written = 0
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in kept:
            writer.writerow(row)
        for entry in valid_entries:
            params = entry["transform_params"]
            params_serialised = (
                json.dumps(params, default=str, sort_keys=True)
                if not isinstance(params, str)
                else params
            )
            writer.writerow(
                {
                    "ref_path": str(entry["ref_path"]),
                    "cor_path": str(entry["cor_path"]),
                    "corruption_type": str(entry["corruption_type"]),
                    "corruption_domain": str(entry["corruption_domain"]),
                    "severity": str(entry["severity"]),
                    "seed": str(entry["seed"]),
                    "transform_params": params_serialised,
                    "dataset_tag": dataset_tag,
                }
            )
            rows_written += 1
    return rows_written


# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────


def _print_summary(
    metadata: list[dict],
    rows_written: int,
    manifest_path: Path,
    console: Console,
) -> None:
    success = sum(1 for m in metadata if "error" not in m)
    failed = len(metadata) - success

    table = Table(title="Corruption generation summary")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("metadata entries returned", str(len(metadata)))
    table.add_row("successful", str(success))
    table.add_row("failed", str(failed))
    table.add_row("manifest rows written", str(rows_written))
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
        help="Directory of reference .nii.gz scans (phase-1 output).",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Root directory for corrupted outputs.",
    ),
    corruptions: str = typer.Option(
        ALL_KEYWORD,
        "--corruptions",
        help="Comma-separated corruption names or 'all'.",
    ),
    severities: str = typer.Option(
        ALL_KEYWORD,
        "--severities",
        help="Comma-separated severity levels (1-5) or 'all'.",
    ),
    dataset_tag: str = typer.Option(
        ...,
        "--dataset-tag",
        case_sensitive=False,
        help=f"Dataset origin for every row in the manifest. One of {DATASET_TAGS}.",
    ),
    manifest_path: Path = typer.Option(
        Path("results/tables/corruption_manifest.csv"),
        "--manifest-path",
        help="Output CSV path for the corruption manifest.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print plan without writing corrupted files."
    ),
) -> None:
    """Thin CLI wrapper around nobrainer.qc.generate_corrupted_dataset."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )

    tag = dataset_tag.lower()
    if tag not in DATASET_TAGS:
        raise typer.BadParameter(
            f"--dataset-tag must be one of {DATASET_TAGS}; got {dataset_tag!r}"
        )

    corruption_list = _parse_corruptions(corruptions)
    severity_list = _parse_severities(severities)

    metadata = generate_corrupted_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        corruptions=corruption_list,
        severities=severity_list,
        dry_run=dry_run,
    )

    rows_written = 0
    if not dry_run:
        rows_written = write_manifest(metadata, manifest_path, dataset_tag=tag)

    _print_summary(metadata, rows_written, manifest_path, Console())


if __name__ == "__main__":
    app()
