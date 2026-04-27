#!/usr/bin/env python3
"""Reconstruct ``results/tables/corruption_manifest.csv`` from the per-scan
JSON sidecars that ``code/02_generate_corruptions.py`` writes alongside each
corrupted NIfTI.

Why this exists:
    The Phase 02 wrapper writes the aggregate manifest CSV at the **end** of
    its run, after all NIfTIs (and per-scan ``.json`` sidecars) are on disk.
    If Phase 02 is interrupted before that final write — or if the file gets
    truncated to 0 bytes by a prior failed write — the manifest is lost but
    every successful corruption still has its sidecar.

    This script walks ``data/<ds>/corrupted_proto/**/*.json`` for each known
    dataset, parses the sidecar JSON, infers ``dataset_tag`` from the path,
    and writes a fresh manifest matching the project's documented schema:
    ``ref_path, cor_path, corruption_type, corruption_domain, severity, seed,
    transform_params, dataset_tag``.

Usage:
    python scripts/rebuild_corruption_manifest.py
    python scripts/rebuild_corruption_manifest.py \\
        --data-root /root/NeuroQC/data \\
        --output    /root/NeuroQC/results/tables/corruption_manifest.csv
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

logger = logging.getLogger(__name__)
app = typer.Typer(
    help="Rebuild corruption_manifest.csv from per-scan JSON sidecars.",
    add_completion=False,
)

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

DATASET_TAGS: tuple[str, ...] = ("ixi", "fastmri", "abide")


def _infer_dataset_tag(cor_path: str) -> str:
    """Path-based heuristic mirroring code/05_extract_iqms.py's logic."""
    lowered = cor_path.lower()
    for tag in DATASET_TAGS:
        if f"/{tag}/" in lowered:
            return tag
    return "unknown"


def _normalise_transform_params(value: object) -> str:
    """Coerce transform_params to a JSON string (CSV-safe)."""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def collect_rows(data_root: Path) -> list[dict[str, str]]:
    """Walk ``<data_root>/<ds>/corrupted_proto/**/*.json`` and parse rows.

    Args:
        data_root: Project's ``data/`` directory; expected to contain
            ``ixi``, ``fastmri``, ``abide`` subdirs each with their own
            ``corrupted_proto/`` tree (Phase 02 output).

    Returns:
        List of dicts keyed by ``MANIFEST_COLUMNS``. Sidecars whose JSON
        contains an ``"error"`` key are skipped (they represent failed
        attempts, not successful corruptions).
    """
    rows: list[dict[str, str]] = []
    sidecars = sorted(data_root.glob("*/corrupted_proto/**/*.json"))
    for sidecar in sidecars:
        try:
            meta = json.loads(sidecar.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable sidecar %s: %s", sidecar, exc)
            continue
        if "error" in meta:
            logger.debug("Skipping failed entry %s", sidecar)
            continue
        required = {"ref_path", "cor_path", "corruption_type", "severity"}
        if not required.issubset(meta.keys()):
            logger.warning(
                "Sidecar %s missing required keys (have %s); skipping",
                sidecar,
                sorted(meta.keys()),
            )
            continue

        meta.setdefault("corruption_domain", "image")
        meta.setdefault("seed", 0)
        meta.setdefault("transform_params", "")
        meta["transform_params"] = _normalise_transform_params(
            meta["transform_params"]
        )
        meta["dataset_tag"] = _infer_dataset_tag(str(meta["cor_path"]))
        rows.append({k: str(meta.get(k, "")) for k in MANIFEST_COLUMNS})
    return rows


def write_manifest(rows: list[dict[str, str]], output: Path) -> None:
    """Write ``rows`` to ``output`` as CSV with ``MANIFEST_COLUMNS`` header."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict[str, str]], output: Path, console: Console) -> None:
    """Show row total + per-(dataset, corruption_type) counts."""
    table = Table(title="Manifest rebuild summary")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("rows reconstructed", str(len(rows)))
    table.add_row("output", str(output))
    console.print(table)

    breakdown = Counter((r["dataset_tag"], r["corruption_type"]) for r in rows)
    if breakdown:
        ds_table = Table(title="By (dataset, corruption_type)")
        ds_table.add_column("dataset_tag", style="bold")
        ds_table.add_column("corruption_type")
        ds_table.add_column("count", justify="right")
        for (ds, ctype), count in sorted(breakdown.items()):
            ds_table.add_row(ds, ctype, str(count))
        console.print(ds_table)


@app.command()
def main(
    data_root: Path = typer.Option(
        Path("data"),
        "--data-root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Project data/ directory containing per-dataset corrupted_proto trees.",
    ),
    output: Path = typer.Option(
        Path("results/tables/corruption_manifest.csv"),
        "--output",
        help="Manifest CSV path to write.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Walk sidecars and report counts without writing."
    ),
) -> None:
    """Reconstruct corruption_manifest.csv from per-scan JSON sidecars."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )
    rows = collect_rows(data_root)
    if not rows:
        raise typer.BadParameter(
            f"No valid sidecars found under {data_root}/*/corrupted_proto/. "
            f"Did Phase 02 run? Expected pattern: <ds>/corrupted_proto/**/*.json"
        )
    if not dry_run:
        write_manifest(rows, output)
    _print_summary(rows, output, Console())


if __name__ == "__main__":
    app()
