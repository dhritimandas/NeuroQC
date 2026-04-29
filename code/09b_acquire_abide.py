#!/usr/bin/env python3
"""NeuroQC Phase 09b — ABIDE-I external multi-rater validation acquisition.

Acquires the ABIDE-I dataset as an external human-rated validation set keyed
on FILE_ID. Two phases:

PHASE A (always, no network):
    Loads mriqc-learn's pre-computed ABIDE reference IQMs (1101 scans × 68
    columns) and three-rater accept/doubtful/exclude labels (-1/0/1) from
    rater_1, rater_2, rater_3. Aligned 1:1 with the IQM table by row index.
    Output: ``data/abide/abide_ratings_iqms.csv`` (74 cols: FILE_ID + 5
    metadata + 68 mriqc_-prefixed IQMs) + ``abide_mriqc_learn_provenance.json``
    (mriqc-learn version, IQM column list, rater coverage by site).

    Schema is fact-pinned (verified 2026-04-25 against mriqc-learn 0.0.3):
        train_x.shape == (1101, 68)
        train_y.columns ⊇ {subject_id, site, rater_1, rater_2, rater_3}
        17 sites: CALTECH, CMU, KKI, LEUVEN, MAX_MUN, NYU, OHSU, OLIN, PITT,
                  SBL, SDSU, STANFORD, TRINITY, UCLA, UM, USM, YALE
        rater_3 fully covers all 1101 scans; rater_1/rater_2 each cover 600
        (501 NaNs per rater).
        FILE_ID := f"{site}_{int(subject_id):07d}"

    IQM column overlap with project pipeline (Phase 05: snr/cnr/efc/fber/cjv):
        Direct match: cjv, cnr, efc, fber.
        snr proxy: project's snr ≈ mriqc's snr_total (mriqc-learn ships
            snr_total / snr_csf / snr_gm / snr_wm + four snrd_* variants;
            no plain "snr" column — document the proxy mapping in the V1
            cross-check caption).

PHASE B (optional, three documented paths):
    Acquires the actual ABIDE-I T1w volumes for downstream IQM cross-check
    (V1) and VLM-vs-humans agreement analyses (V2-V4). User must opt in via
    ``--acquisition-path``.

    Path "fcp-indi-raw" (CANONICAL, recommended):
        s3://fcp-indi/data/Projects/ABIDE_Initiative/RawDataBIDS/
        Anonymous read; BIDS-canonical layout
            {S3site}/sub-{sid:07d}/anat/sub-{sid:07d}_T1w.nii.gz
        17 mriqc-learn UPPERCASE sites map onto 24 S3 directories (5 sites
        are split into batches: CMU_a/b, Leuven_1/2, MaxMun_a/b/c/d, UCLA_1/2,
        UM_1/2). The site map below is verified empirically; ``find_s3_raw_key``
        searches each batch in order, first hit wins. All 1101 phenotype
        FILE_IDs verified to have a matching S3 key on 2026-04-25.

    Path "nitrc-ir" (registration-gated, manual):
        Validates a user-pre-downloaded set passed via
        ``--id-mapping-csv`` with columns
            phenotype_file_id, nitrc_session_id, downloaded_path
        (the script does NOT automate the NITRC-IR XNAT login or download;
        users must register at nitrc.org and follow XNAT REST API or
        oasis-scripts download workflows).

    Path "fmriprep-derivatives" (DOCUMENTED ALTERNATIVE; STALE):
        s3://fcp-indi/data/Projects/ABIDE/Outputs/fmriprep/
        Preprocessed (bias-corrected, T1w native space) outputs from
        sensein.group, ~7+ years old. Not recommended for V1 (IQM
        cross-check breaks because mriqc-learn references were computed
        on raw scans). Exposed as an option for users who specifically
        want preprocessed T1w; raises NotImplementedError in v1 of this
        script — user must implement on demand.

    Path "local":
        ``--local-bids-root`` points at a directory; per-FILE_ID resolution
        tries 3 BIDS patterns + a glob fallback. preprocessing_state="unknown"
        on output (caller can post-process the manifest if they know
        whether their copy is raw or preproc).

OUTPUTS:
    data/abide/abide_ratings_iqms.csv         (Phase A; always)
    data/abide/abide_mriqc_learn_provenance.json
    data/abide/{FILE_ID}.nii.gz               (Phase B; when acquired)
    data/abide/abide_acquisition_manifest.csv (Phase B; one row per acquired)
    results/tables/abide_acquisition_failures.csv
    results/tables/abide_acquisition_summary.json

RESUME: every Phase B path checks {output_dir}/{FILE_ID}.nii.gz exists,
loads via nibabel, and matches the manifest's recorded file_size_bytes
before skipping. Crashes resume from where they left off.

DEFAULT: Phase A only (no network). Pass ``--acquisition-path`` to opt
into Phase B.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import pandas as pd
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from tqdm import tqdm

# ──────────────────────────────────────────────
# Constants — schema (fact-pinned to mriqc-learn 0.0.3, verified 2026-04-25)
# ──────────────────────────────────────────────

ABIDE_SITES: tuple[str, ...] = (
    "CALTECH", "CMU", "KKI", "LEUVEN", "MAX_MUN", "NYU", "OHSU",
    "OLIN", "PITT", "SBL", "SDSU", "STANFORD", "TRINITY",
    "UCLA", "UM", "USM", "YALE",
)
ABIDE_TOTAL_SCANS: int = 1101
ABIDE_IQM_COLUMNS_EXPECTED_COUNT: int = 68
RATER_COLUMNS: tuple[str, ...] = ("rater_1", "rater_2", "rater_3")
MRIQC_PREFIX: str = "mriqc_"

# Priority IQM ordering: 4 directly comparable + 8 snr variants. Remaining 56
# sorted alphabetically post-prefix.
PRIORITY_IQM_BASE: tuple[str, ...] = (
    "cjv", "cnr", "efc", "fber",
    "snr_total", "snr_csf", "snr_gm", "snr_wm",
    "snrd_total", "snrd_csf", "snrd_gm", "snrd_wm",
)

# Output column constants.
FID_COLUMN: str = "FILE_ID"
SITE_COLUMN: str = "site"
SUBJECT_ID_COLUMN: str = "subject_id"
_FILE_ID_PATTERN = re.compile(r"^[A-Z_]+_\d{7}$")

# ──────────────────────────────────────────────
# Constants — Phase B
# ──────────────────────────────────────────────

INTEGRITY_MIN_DIM: int = 64

ACQUISITION_PATH_NITRC: str = "nitrc-ir"
ACQUISITION_PATH_FCP_RAW: str = "fcp-indi-raw"
ACQUISITION_PATH_FMRIPREP: str = "fmriprep-derivatives"
ACQUISITION_PATH_LOCAL: str = "local"
ACQUISITION_PATHS: tuple[str, ...] = (
    ACQUISITION_PATH_NITRC,
    ACQUISITION_PATH_FCP_RAW,
    ACQUISITION_PATH_FMRIPREP,
    ACQUISITION_PATH_LOCAL,
)

S3_BUCKET: str = "fcp-indi"
S3_RAW_BIDS_PREFIX: str = "data/Projects/ABIDE_Initiative/RawDataBIDS/"
S3_FMRIPREP_PREFIX: str = "data/Projects/ABIDE/Outputs/fmriprep/"
S3_RETRY_BACKOFFS: tuple[int, ...] = (1, 2, 4, 8, 16)

# Verified 2026-04-25: 17 mriqc-learn UPPERCASE sites → 24 S3 RawDataBIDS
# subdirs. Multi-batch sites (CMU/LEUVEN/MAX_MUN/UCLA/UM) are split into
# session batches with site-suffix names. The acquisition logic searches
# each batch in declaration order; first hit wins. All 1101 mriqc-learn
# FILE_IDs were confirmed to have a unique match across these batches.
SITE_TO_S3_RAW_BIDS_DIRS: dict[str, tuple[str, ...]] = {
    "CALTECH":  ("Caltech",),
    "CMU":      ("CMU_a", "CMU_b"),
    "KKI":      ("KKI",),
    "LEUVEN":   ("Leuven_1", "Leuven_2"),
    "MAX_MUN":  ("MaxMun_a", "MaxMun_b", "MaxMun_c", "MaxMun_d"),
    "NYU":      ("NYU",),
    "OHSU":     ("OHSU",),
    "OLIN":     ("Olin",),
    "PITT":     ("Pitt",),
    "SBL":      ("SBL",),
    "SDSU":     ("SDSU",),
    "STANFORD": ("Stanford",),
    "TRINITY":  ("Trinity",),
    "UCLA":     ("UCLA_1", "UCLA_2"),
    "UM":       ("UM_1", "UM_2"),
    "USM":      ("USM",),
    "YALE":     ("Yale",),
}

# Local BIDS patterns (path 3) — first hit wins, glob fallback last.
LOCAL_PATTERNS: tuple[str, ...] = (
    "{site}_{sid:07d}/anat/{site}_{sid:07d}_T1w.nii.gz",
    "sub-{sid:07d}/anat/sub-{sid:07d}_T1w.nii.gz",
    "{site}/sub-{sid:07d}/anat/sub-{sid:07d}_T1w.nii.gz",
)
LOCAL_GLOB_FALLBACK_PATTERN: str = "**/{sid:07d}*T1w.nii*"

# Manifest schema (Phase B output).
MANIFEST_COLUMNS: tuple[str, ...] = (
    "FILE_ID", "site", "subject_id",
    "scan_path", "source_path",
    "acquisition_path", "preprocessing_state",
    "voxel_x_mm", "voxel_y_mm", "voxel_z_mm",
    "shape_x", "shape_y", "shape_z",
    "file_size_bytes", "acquisition_timestamp_iso",
)
FAILURE_COLUMNS: tuple[str, ...] = (
    "FILE_ID", "source_path", "reason", "timestamp_iso",
)

logger = logging.getLogger(__name__)
app = typer.Typer(
    help="ABIDE-I acquisition (ratings + IQMs + optional T1w volumes).",
    add_completion=False,
)


# ──────────────────────────────────────────────
# Phase A — ratings + reference IQMs
# ──────────────────────────────────────────────


def _import_mriqc_learn():
    """Defer mriqc-learn import; abort with install hint on miss."""
    try:
        import mriqc_learn
        from mriqc_learn.datasets import load_dataset

        return mriqc_learn, load_dataset
    except ImportError as exc:
        raise typer.BadParameter(
            "mriqc-learn not installed. Install with: pip install mriqc-learn (>=0.0.3)"
        ) from exc


def load_mriqc_learn_dataset() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load mriqc-learn's ABIDE dataset; return (train_x, train_y, version)."""
    mriqc_learn_module, load_dataset = _import_mriqc_learn()
    (train_x, train_y), _ = load_dataset(dataset="abide", split_strategy="none")
    return train_x, train_y, mriqc_learn_module.__version__


def assert_phase_a_schema(train_x: pd.DataFrame, train_y: pd.DataFrame) -> None:
    """Strict assertion that the loaded mriqc-learn data matches our hardcoded shape.

    Aborts with diagnostic on drift. Future mriqc-learn versions that ship a
    different shape need explicit code update + version pin bump.
    """
    if train_x.shape != (ABIDE_TOTAL_SCANS, ABIDE_IQM_COLUMNS_EXPECTED_COUNT):
        raise SystemExit(
            f"mriqc-learn schema drift: train_x.shape={tuple(train_x.shape)}, "
            f"expected ({ABIDE_TOTAL_SCANS}, {ABIDE_IQM_COLUMNS_EXPECTED_COUNT}). "
            "Pin a known-good version with `pip install mriqc-learn==0.0.3` or "
            "update this script's hardcoded constants."
        )
    expected_y = {SUBJECT_ID_COLUMN, SITE_COLUMN, *RATER_COLUMNS}
    missing = expected_y - set(train_y.columns)
    if missing:
        raise SystemExit(
            f"mriqc-learn schema drift: train_y missing columns {sorted(missing)}; "
            f"got {sorted(train_y.columns)}"
        )
    sites_in_data = set(train_y[SITE_COLUMN].unique())
    extra = sites_in_data - set(ABIDE_SITES)
    missing_sites = set(ABIDE_SITES) - sites_in_data
    if extra or missing_sites:
        raise SystemExit(
            f"Site set drift: extra={sorted(extra)}, missing={sorted(missing_sites)}"
        )


def derive_file_id(site: Any, subject_id: Any) -> str:
    """Build ``f'{site}_{int(subject_id):07d}'`` with strict validation.

    Raises:
        ValueError: site or subject_id is NaN; or the constructed FILE_ID
            fails the ``^[A-Z_]+_\\d{7}$`` pattern (catches lowercase sites
            or non-numeric subject_ids).
    """
    if pd.isna(site):
        raise ValueError("site is NaN")
    if pd.isna(subject_id):
        raise ValueError(f"subject_id is NaN for site {site!r}")
    sid = int(subject_id)
    fid = f"{site}_{sid:07d}"
    if not _FILE_ID_PATTERN.match(fid):
        raise ValueError(
            f"Constructed FILE_ID {fid!r} fails pattern check (^[A-Z_]+_\\d{{7}}$)"
        )
    return fid


def build_phase_a_frame(
    train_x: pd.DataFrame, train_y: pd.DataFrame
) -> pd.DataFrame:
    """Join train_y + train_x, derive FILE_IDs, prefix IQM columns, reorder."""
    file_ids = [
        derive_file_id(s, sid)
        for s, sid in zip(
            train_y[SITE_COLUMN], train_y[SUBJECT_ID_COLUMN], strict=True
        )
    ]
    if len(set(file_ids)) != len(file_ids):
        # Find duplicates.
        seen: set[str] = set()
        dups: list[str] = []
        for f in file_ids:
            if f in seen:
                dups.append(f)
            seen.add(f)
        raise ValueError(
            f"Duplicate FILE_IDs in mriqc-learn data ({len(dups)}): {dups[:5]}"
        )

    iqm_cols = list(train_x.columns)
    prefixed = [f"{MRIQC_PREFIX}{c}" for c in iqm_cols]
    train_x_renamed = train_x.copy()
    train_x_renamed.columns = prefixed

    out = train_y.reset_index(drop=True).copy()
    out[FID_COLUMN] = file_ids
    iqm_data = train_x_renamed.reset_index(drop=True)
    combined = pd.concat([out, iqm_data], axis=1)

    priority_iqm = [f"{MRIQC_PREFIX}{c}" for c in PRIORITY_IQM_BASE]
    # All priority columns must be present — else schema drift.
    missing_priority = set(priority_iqm) - set(prefixed)
    if missing_priority:
        raise SystemExit(
            f"mriqc-learn schema drift: missing priority IQM columns "
            f"{sorted(missing_priority)}; got {sorted(prefixed)[:8]}..."
        )
    remaining_iqm = sorted(set(prefixed) - set(priority_iqm))

    final_order = [
        FID_COLUMN, SITE_COLUMN, SUBJECT_ID_COLUMN,
        *RATER_COLUMNS,
        *priority_iqm,
        *remaining_iqm,
    ]
    return combined[final_order]


def write_provenance_json(
    combined: pd.DataFrame,
    mriqc_learn_version: str,
    output_path: Path,
) -> None:
    """Persist Phase A provenance: version, IQM list, per-site rater coverage."""
    iqm_cols = [c for c in combined.columns if c.startswith(MRIQC_PREFIX)]
    rater_coverage: dict[str, dict[str, int]] = {}
    for site in ABIDE_SITES:
        sub = combined[combined[SITE_COLUMN] == site]
        rater_coverage[site] = {
            "n_scans": int(len(sub)),
            "rater_1_n_rated": int(sub["rater_1"].notna().sum()),
            "rater_2_n_rated": int(sub["rater_2"].notna().sum()),
            "rater_3_n_rated": int(sub["rater_3"].notna().sum()),
        }
    provenance = {
        "mriqc_learn_version": mriqc_learn_version,
        "n_rows": int(len(combined)),
        "iqm_columns": iqm_cols,
        "rater_coverage_by_site": rater_coverage,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(provenance, indent=2))


# ──────────────────────────────────────────────
# Phase B — common (VolumeRecord, integrity, manifest I/O, resume)
# ──────────────────────────────────────────────


@dataclass(frozen=True)
class VolumeRecord:
    """One acquired volume; serialised as one row of the acquisition manifest."""

    file_id: str
    site: str
    subject_id: int
    scan_path: str
    source_path: str
    acquisition_path: str
    preprocessing_state: str
    voxel_x_mm: float
    voxel_y_mm: float
    voxel_z_mm: float
    shape_x: int
    shape_y: int
    shape_z: int
    file_size_bytes: int
    acquisition_timestamp_iso: str


def verify_volume_integrity(scan_path: Path) -> dict[str, Any]:
    """Load via nibabel; assert 3D-T1w-ish shape; return integrity metadata.

    Raises ValueError on bad ndim, on 4D with N > 1 frames, or on any spatial
    dim < INTEGRITY_MIN_DIM (paranoid lower bound to catch truncated downloads
    or wrong-modality images).
    """
    nii = nib.load(str(scan_path))
    if nii.ndim == 4:
        if nii.shape[3] != 1:
            raise ValueError(
                f"4D NIfTI with N > 1 frames at {scan_path}: shape={nii.shape}"
            )
        shape = nii.shape[:3]
    elif nii.ndim == 3:
        shape = nii.shape
    else:
        raise ValueError(
            f"Unsupported ndim={nii.ndim} at {scan_path}: shape={nii.shape}"
        )
    if any(d < INTEGRITY_MIN_DIM for d in shape):
        raise ValueError(
            f"Dim < {INTEGRITY_MIN_DIM} at {scan_path}: shape={shape}"
        )
    voxels = nii.header.get_zooms()[:3]
    return {
        "shape": tuple(int(s) for s in shape),
        "voxels": tuple(float(v) for v in voxels),
        "size_bytes": int(scan_path.stat().st_size),
    }


def load_existing_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """``{file_id: row_dict}`` for resume; empty if manifest absent."""
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        return {}
    df = pd.read_csv(manifest_path)
    return {str(row["FILE_ID"]): row.to_dict() for _, row in df.iterrows()}


def is_already_acquired(
    file_id: str,
    output_dir: Path,
    manifest_entry: dict[str, Any] | None,
) -> bool:
    """Resume condition: file exists, loads, and (when manifest known) size matches."""
    expected = output_dir / f"{file_id}.nii.gz"
    if not expected.is_file():
        return False
    try:
        verify_volume_integrity(expected)
    except Exception:
        return False
    if manifest_entry is not None and "file_size_bytes" in manifest_entry:
        try:
            expected_size = int(manifest_entry["file_size_bytes"])
        except (TypeError, ValueError):
            return True  # manifest size un-parseable; trust the integrity check
        if expected.stat().st_size != expected_size:
            return False
    return True


def append_manifest_row(manifest_path: Path, record: VolumeRecord) -> None:
    """Append one VolumeRecord to manifest CSV (header on first write).

    The dataclass uses snake_case field names; the manifest schema uses
    ``FILE_ID`` (uppercase) per the spec. Build the row dict explicitly
    so the rename happens at one place.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not manifest_path.is_file() or manifest_path.stat().st_size == 0
    row = {
        "FILE_ID": record.file_id,
        "site": record.site,
        "subject_id": record.subject_id,
        "scan_path": record.scan_path,
        "source_path": record.source_path,
        "acquisition_path": record.acquisition_path,
        "preprocessing_state": record.preprocessing_state,
        "voxel_x_mm": record.voxel_x_mm,
        "voxel_y_mm": record.voxel_y_mm,
        "voxel_z_mm": record.voxel_z_mm,
        "shape_x": record.shape_x,
        "shape_y": record.shape_y,
        "shape_z": record.shape_z,
        "file_size_bytes": record.file_size_bytes,
        "acquisition_timestamp_iso": record.acquisition_timestamp_iso,
    }
    with manifest_path.open("a", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(MANIFEST_COLUMNS))
        if is_new:
            w.writeheader()
        w.writerow(row)
        h.flush()


def append_failure_row(
    failure_path: Path, file_id: str, source_path: str, reason: str
) -> None:
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not failure_path.is_file() or failure_path.stat().st_size == 0
    with failure_path.open("a", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(FAILURE_COLUMNS))
        if is_new:
            w.writeheader()
        w.writerow(
            {
                "FILE_ID": file_id,
                "source_path": source_path,
                "reason": reason,
                "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            }
        )
        h.flush()


def build_volume_record(
    file_id: str,
    site: str,
    subject_id: int,
    scan_path: Path,
    source_path: str,
    acquisition_path: str,
    preprocessing_state: str,
    integ: dict[str, Any],
) -> VolumeRecord:
    sx, sy, sz = integ["shape"]
    vx, vy, vz = integ["voxels"]
    return VolumeRecord(
        file_id=file_id,
        site=site,
        subject_id=int(subject_id),
        scan_path=str(scan_path),
        source_path=source_path,
        acquisition_path=acquisition_path,
        preprocessing_state=preprocessing_state,
        voxel_x_mm=float(vx),
        voxel_y_mm=float(vy),
        voxel_z_mm=float(vz),
        shape_x=int(sx),
        shape_y=int(sy),
        shape_z=int(sz),
        file_size_bytes=int(integ["size_bytes"]),
        acquisition_timestamp_iso=datetime.now(timezone.utc).isoformat(),
    )


# ──────────────────────────────────────────────
# Phase B path 1 — NITRC-IR (manual, registration-gated)
# ──────────────────────────────────────────────


NITRC_INSTRUCTIONS: str = """\
NITRC-IR registration steps:
  1. Register at https://nitrc.org (free).
  2. Request membership in the '1000 Functional Connectomes Project'
     resource. Approval is manual and may take days/weeks.
  3. After approval, log into NITRC-IR's XNAT portal and download
     ABIDE-I scans using either:
       - the NrgXnat/oasis-scripts download pattern, OR
       - the NITRC-IR XNAT REST API (see XNAT documentation).
  4. NOTE the documented subject-ID mismatch: phenotype IDs (e.g. 51456)
     differ from COINS image IDs (e.g. A00032016). Construct an
     id-mapping CSV with columns:
       phenotype_file_id  (FILE_ID from this script's Phase A output),
       nitrc_session_id   (the XNAT session you downloaded),
       downloaded_path    (absolute local path to the downloaded NIfTI).
  5. Pass the mapping CSV via --id-mapping-csv.

This script does NOT automate NITRC login or download. It validates
your already-downloaded scans and writes a manifest for the verified
subset.
"""


def load_id_mapping_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"phenotype_file_id", "nitrc_session_id", "downloaded_path"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"--id-mapping-csv {path} missing required columns: {sorted(missing)}"
        )
    return df


def acquire_via_nitrc(
    mapping: pd.DataFrame,
    site_lookup: dict[str, tuple[str, int]],
    output_dir: Path,
    manifest_path: Path,
    failure_path: Path,
) -> int:
    """Validate user-pre-downloaded NITRC-IR scans + populate manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_done = load_existing_manifest(manifest_path)
    n_acquired = 0
    iterator = (
        tqdm(mapping.iterrows(), total=len(mapping), desc="nitrc")
        if len(mapping) > 1
        else mapping.iterrows()
    )
    for _, row in iterator:
        file_id = str(row["phenotype_file_id"])
        downloaded = Path(str(row["downloaded_path"]))
        if file_id not in site_lookup:
            append_failure_row(
                failure_path, file_id, str(downloaded),
                "FILE_ID not in Phase A output",
            )
            continue
        if is_already_acquired(file_id, output_dir, manifest_done.get(file_id)):
            continue
        if not downloaded.is_file():
            append_failure_row(
                failure_path, file_id, str(downloaded),
                "downloaded_path does not exist",
            )
            continue
        try:
            integ = verify_volume_integrity(downloaded)
        except Exception as exc:  # noqa: BLE001
            append_failure_row(
                failure_path, file_id, str(downloaded), f"integrity: {exc}"
            )
            continue
        site, subject_id = site_lookup[file_id]
        dest = output_dir / f"{file_id}.nii.gz"
        if downloaded.resolve() != dest.resolve():
            shutil.copy(str(downloaded), str(dest))
        record = build_volume_record(
            file_id, site, subject_id, dest, str(downloaded),
            ACQUISITION_PATH_NITRC, "raw", integ,
        )
        append_manifest_row(manifest_path, record)
        n_acquired += 1
    return n_acquired


# ──────────────────────────────────────────────
# Phase B path 2 — fcp-indi raw BIDS (CANONICAL)
# ──────────────────────────────────────────────


def _import_boto3():
    """Defer boto3 import; abort with install hint on miss."""
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        from botocore.exceptions import ClientError

        return boto3, UNSIGNED, Config, ClientError
    except ImportError as exc:
        raise SystemExit(
            "boto3 not installed. Install with: pip install boto3"
        ) from exc


def make_anonymous_s3_client():
    """Anonymous (UNSIGNED) S3 client for public buckets like fcp-indi."""
    boto3_module, UNSIGNED, Config, _ = _import_boto3()
    return boto3_module.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")


def find_s3_raw_key(client, site: str, subject_id: int) -> str | None:
    """Search every batch of ``site`` for a ``sub-{sid:07d}_T1w.nii.gz`` key.

    Returns the first key whose ``head_object`` succeeds, or None.
    """
    _, _, _, ClientError = _import_boto3()
    sid_str = f"{int(subject_id):07d}"
    candidates = SITE_TO_S3_RAW_BIDS_DIRS.get(site, ())
    for s3_dir in candidates:
        key = (
            f"{S3_RAW_BIDS_PREFIX}{s3_dir}/sub-{sid_str}/anat/"
            f"sub-{sid_str}_T1w.nii.gz"
        )
        try:
            client.head_object(Bucket=S3_BUCKET, Key=key)
            return key
        except ClientError:
            continue
    return None


def download_with_retry(client, key: str, dest: Path) -> None:
    """Exponential backoff download (up to S3_RETRY_BACKOFFS attempts)."""
    last_exc: Exception | None = None
    for backoff in S3_RETRY_BACKOFFS:
        try:
            client.download_file(S3_BUCKET, key, str(dest))
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(backoff)
    raise RuntimeError(
        f"Download failed after {len(S3_RETRY_BACKOFFS)} attempts: {last_exc}"
    )


def acquire_via_s3_raw(
    file_ids_with_meta: list[tuple[str, str, int]],
    output_dir: Path,
    manifest_path: Path,
    failure_path: Path,
    max_concurrent: int,
) -> int:
    """Download every ``(file_id, site, subject_id)`` from the raw BIDS S3 prefix."""
    client = make_anonymous_s3_client()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_done = load_existing_manifest(manifest_path)

    pending = [
        (fid, site, sid)
        for fid, site, sid in file_ids_with_meta
        if not is_already_acquired(fid, output_dir, manifest_done.get(fid))
    ]
    n_skipped = len(file_ids_with_meta) - len(pending)
    logger.info(
        "fcp-indi-raw plan: %d total, %d already done, %d to acquire",
        len(file_ids_with_meta), n_skipped, len(pending),
    )
    if not pending:
        return 0

    n_acquired = 0
    n_failed = 0

    def _process(arg: tuple[str, str, int]) -> tuple[VolumeRecord | None, str]:
        fid, site, sid = arg
        key = find_s3_raw_key(client, site, sid)
        if key is None:
            return None, (
                f"no T1w key found in {SITE_TO_S3_RAW_BIDS_DIRS.get(site, ())}"
                f" for {fid}"
            )
        dest = output_dir / f"{fid}.nii.gz"
        try:
            download_with_retry(client, key, dest)
            integ = verify_volume_integrity(dest)
        except Exception as exc:  # noqa: BLE001
            return None, f"download/integrity for {fid}: {exc}"
        return (
            build_volume_record(
                fid, site, sid, dest, f"s3://{S3_BUCKET}/{key}",
                ACQUISITION_PATH_FCP_RAW, "raw", integ,
            ),
            key,
        )

    with ThreadPoolExecutor(max_workers=max_concurrent) as ex:
        futures = {ex.submit(_process, x): x for x in pending}
        with tqdm(total=len(futures), desc="s3-raw") as pbar:
            for fut in as_completed(futures):
                fid, site, sid = futures[fut]
                record_or_none, info = fut.result()
                if record_or_none is not None:
                    append_manifest_row(manifest_path, record_or_none)
                    n_acquired += 1
                else:
                    append_failure_row(
                        failure_path, fid, f"s3://{S3_BUCKET}/{S3_RAW_BIDS_PREFIX}", info
                    )
                    n_failed += 1
                pbar.update(1)

    logger.info(
        "fcp-indi-raw complete: %d acquired, %d failed", n_acquired, n_failed
    )
    return n_acquired


# ──────────────────────────────────────────────
# Phase B path 3 — local BIDS
# ──────────────────────────────────────────────


def resolve_local(file_id: str, site: str, subject_id: int, bids_root: Path) -> Path | None:
    """Try the 3 BIDS patterns + a glob fallback. First match wins."""
    sid = int(subject_id)
    for pattern in LOCAL_PATTERNS:
        candidate = bids_root / pattern.format(site=site, sid=sid)
        if candidate.is_file():
            return candidate
    glob_pat = LOCAL_GLOB_FALLBACK_PATTERN.format(sid=sid)
    matches = sorted(bids_root.glob(glob_pat))
    return matches[0] if matches else None


def acquire_via_local(
    file_ids_with_meta: list[tuple[str, str, int]],
    bids_root: Path,
    output_dir: Path,
    manifest_path: Path,
    failure_path: Path,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_done = load_existing_manifest(manifest_path)
    n_acquired = 0
    iterator = (
        tqdm(file_ids_with_meta, desc="local")
        if len(file_ids_with_meta) > 1
        else file_ids_with_meta
    )
    for fid, site, sid in iterator:
        if is_already_acquired(fid, output_dir, manifest_done.get(fid)):
            continue
        resolved = resolve_local(fid, site, sid, bids_root)
        if resolved is None:
            append_failure_row(
                failure_path, fid, str(bids_root),
                "no match in BIDS tree (3 patterns + glob)",
            )
            continue
        try:
            integ = verify_volume_integrity(resolved)
        except Exception as exc:  # noqa: BLE001
            append_failure_row(failure_path, fid, str(resolved), f"integrity: {exc}")
            continue
        dest = output_dir / f"{fid}.nii.gz"
        if resolved.resolve() != dest.resolve():
            shutil.copy(str(resolved), str(dest))
        record = build_volume_record(
            fid, site, sid, dest, str(resolved),
            ACQUISITION_PATH_LOCAL, "unknown", integ,
        )
        append_manifest_row(manifest_path, record)
        n_acquired += 1
    return n_acquired


# ──────────────────────────────────────────────
# Acquisition summary
# ──────────────────────────────────────────────


def write_acquisition_summary(
    phase_a_csv: Path,
    manifest_csv: Path,
    failure_csv: Path,
    summary_json: Path,
    acquisition_path_used: str | None,
) -> None:
    """Set arithmetic A/B/F summary + preprocessing_state distribution."""
    A: set[str] = (
        set(pd.read_csv(phase_a_csv)["FILE_ID"].astype(str))
        if phase_a_csv.is_file()
        else set()
    )
    B: set[str] = set()
    pp_states: dict[str, int] = {}
    if manifest_csv.is_file() and manifest_csv.stat().st_size > 0:
        m = pd.read_csv(manifest_csv)
        B = set(m["FILE_ID"].astype(str))
        if "preprocessing_state" in m.columns:
            counts = m["preprocessing_state"].value_counts().to_dict()
            pp_states = {str(k): int(v) for k, v in counts.items()}
    F: set[str] = set()
    if failure_csv.is_file() and failure_csv.stat().st_size > 0:
        F = set(pd.read_csv(failure_csv)["FILE_ID"].astype(str))

    summary = {
        "n_phase_a": len(A),
        "n_acquired": len(B),
        "n_intersection_A_B": len(A & B),
        "n_phase_a_only": len(A - B),
        "n_acquired_only": len(B - A),
        "n_failed": len(F),
        "acquisition_path_used": acquisition_path_used,
        "preprocessing_state_distribution": pp_states,
        "warnings": [],
    }
    if B - A:
        summary["warnings"].append(
            f"{len(B - A)} FILE_IDs in manifest but not in Phase A — possible "
            "data drift or stale manifest"
        )
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2))


# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────


def _print_summary(
    acquisition_path_used: str | None,
    n_phase_a: int,
    n_acquired: int,
    n_failed: int,
    output_paths: dict[str, Path],
    console: Console,
) -> None:
    table = Table(title="Phase 09b — ABIDE-I acquisition")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("Phase A scans (mriqc-learn)", str(n_phase_a))
    table.add_row("Phase B path", str(acquisition_path_used or "skipped"))
    table.add_row("Phase B acquired", str(n_acquired))
    table.add_row("Phase B failed", str(n_failed))
    for label, path in output_paths.items():
        table.add_row(label, str(path))
    console.print(table)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


@app.command()
def main(
    skip_download: bool = typer.Option(
        False, "--skip-download", help="Force Phase A only; ignore --acquisition-path."
    ),
    acquisition_path: str | None = typer.Option(
        None,
        "--acquisition-path",
        help=f"One of {ACQUISITION_PATHS}. If unset, only Phase A runs.",
    ),
    local_bids_root: Path | None = typer.Option(
        None, "--local-bids-root", resolve_path=True,
        help="Required when --acquisition-path local.",
    ),
    id_mapping_csv: Path | None = typer.Option(
        None, "--id-mapping-csv", resolve_path=True,
        help="Required when --acquisition-path nitrc-ir.",
    ),
    output_dir: Path = typer.Option(
        Path("data/abide"), "--output-dir", resolve_path=True,
        help="Where to drop {FILE_ID}.nii.gz files (Phase B).",
    ),
    max_concurrent: int = typer.Option(
        4, "--max-concurrent",
        help="Threadpool size for Phase B path 2 (S3) downloads.",
    ),
    max_scans: int = typer.Option(
        0, "--max-scans",
        help="Cap on number of FILE_IDs to acquire (0 = unlimited). Useful for smoke tests.",
    ),
    yes: bool = typer.Option(
        False, "--yes",
        help="Skip pre-flight size confirmation prompts (currently unused; reserved).",
    ),
    ratings_output: Path = typer.Option(
        Path("data/abide/abide_ratings_iqms.csv"), "--ratings-output", resolve_path=True
    ),
    acquisition_manifest: Path = typer.Option(
        Path("data/abide/abide_acquisition_manifest.csv"),
        "--acquisition-manifest", resolve_path=True,
    ),
    failure_log: Path = typer.Option(
        Path("results/tables/abide_acquisition_failures.csv"),
        "--failure-log", resolve_path=True,
    ),
    provenance_log: Path = typer.Option(
        Path("data/abide/abide_mriqc_learn_provenance.json"),
        "--provenance-log", resolve_path=True,
    ),
    acquisition_summary: Path = typer.Option(
        Path("results/tables/abide_acquisition_summary.json"),
        "--acquisition-summary", resolve_path=True,
    ),
) -> None:
    """ABIDE-I acquisition: ratings + IQMs (Phase A) + optional T1w volumes (Phase B)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )
    console = Console()

    # ── Phase A ────────────────────────────────
    logger.info("Phase A — loading mriqc-learn ABIDE dataset…")
    train_x, train_y, mriqc_version = load_mriqc_learn_dataset()
    assert_phase_a_schema(train_x, train_y)
    combined = build_phase_a_frame(train_x, train_y)
    ratings_output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(ratings_output, index=False)
    write_provenance_json(combined, mriqc_version, provenance_log)
    logger.info(
        "Phase A complete: %d rows × %d cols → %s",
        len(combined), len(combined.columns), ratings_output,
    )
    logger.info("Provenance → %s", provenance_log)

    # ── Decide whether to run Phase B ──────────
    run_phase_b = (not skip_download) and acquisition_path is not None
    if not run_phase_b:
        logger.info(
            "Phase B skipped (skip_download=%s, acquisition_path=%s).",
            skip_download, acquisition_path,
        )
        write_acquisition_summary(
            ratings_output, acquisition_manifest, failure_log,
            acquisition_summary, None,
        )
        _print_summary(
            None, len(combined), 0, 0,
            {"ratings": ratings_output, "provenance": provenance_log,
             "summary": acquisition_summary},
            console,
        )
        return

    if acquisition_path not in ACQUISITION_PATHS:
        raise typer.BadParameter(
            f"--acquisition-path must be one of {ACQUISITION_PATHS}, "
            f"got {acquisition_path!r}"
        )

    # Build (file_id, site, subject_id) tuples in deterministic FILE_ID order.
    file_ids_with_meta: list[tuple[str, str, int]] = [
        (str(row[FID_COLUMN]), str(row[SITE_COLUMN]), int(row[SUBJECT_ID_COLUMN]))
        for _, row in combined.iterrows()
    ]
    file_ids_with_meta.sort(key=lambda x: x[0])
    if max_scans > 0:
        file_ids_with_meta = file_ids_with_meta[:max_scans]
    site_lookup = {fid: (site, sid) for fid, site, sid in file_ids_with_meta}

    # ── Dispatch ───────────────────────────────
    logger.info(
        "Phase B — path=%s, %d FILE_IDs, output=%s",
        acquisition_path, len(file_ids_with_meta), output_dir,
    )

    if acquisition_path == ACQUISITION_PATH_NITRC:
        if id_mapping_csv is None:
            raise typer.BadParameter("--id-mapping-csv required for nitrc-ir")
        for line in NITRC_INSTRUCTIONS.splitlines():
            logger.info(line)
        mapping = load_id_mapping_csv(id_mapping_csv)
        n_acquired = acquire_via_nitrc(
            mapping, site_lookup, output_dir, acquisition_manifest, failure_log,
        )
    elif acquisition_path == ACQUISITION_PATH_FCP_RAW:
        n_acquired = acquire_via_s3_raw(
            file_ids_with_meta, output_dir, acquisition_manifest, failure_log,
            max_concurrent=max_concurrent,
        )
    elif acquisition_path == ACQUISITION_PATH_FMRIPREP:
        raise typer.BadParameter(
            "--acquisition-path fmriprep-derivatives is documented as an "
            "alternative source but is not implemented in this script. The "
            "fmriprep S3 outputs are 7+ years old (sensein.group warning) and "
            "preprocessed; using them breaks the V1 IQM cross-check because "
            "mriqc-learn references were computed on raw scans. Use "
            "--acquisition-path fcp-indi-raw instead, OR implement this path "
            "if you specifically need the preprocessed copy."
        )
    elif acquisition_path == ACQUISITION_PATH_LOCAL:
        if local_bids_root is None:
            raise typer.BadParameter("--local-bids-root required for local")
        n_acquired = acquire_via_local(
            file_ids_with_meta, local_bids_root, output_dir,
            acquisition_manifest, failure_log,
        )
    else:  # pragma: no cover — guarded above
        raise typer.BadParameter(f"Unhandled --acquisition-path: {acquisition_path}")

    # ── Summary ────────────────────────────────
    write_acquisition_summary(
        ratings_output, acquisition_manifest, failure_log,
        acquisition_summary, acquisition_path,
    )

    n_failed = 0
    if failure_log.is_file() and failure_log.stat().st_size > 0:
        n_failed = len(pd.read_csv(failure_log))

    _print_summary(
        acquisition_path, len(combined), n_acquired, n_failed,
        {
            "ratings": ratings_output,
            "provenance": provenance_log,
            "manifest": acquisition_manifest,
            "failures": failure_log,
            "summary": acquisition_summary,
        },
        console,
    )


if __name__ == "__main__":
    app()
