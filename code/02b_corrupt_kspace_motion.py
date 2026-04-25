#!/usr/bin/env python3
"""NeuroQC Phase 2b — Physical k-space motion corruption for FastMRI.

Applies rigid-motion phase shifts directly to multi-coil k-space and
reconstructs corrupted magnitude volumes via centred iFFT + root-sum-of-
squares. This is the ground-truth counterpart to the TorchIO image-space
RandomMotion approximation used by ``code/02_generate_corruptions.py``.

Algorithm (Shaw et al. 2019, operating on real k-space instead of a
magnitude iFFT/FFT round-trip):

    For each slice of shape (n_coils, H, W):
      1. Sample ``severity.n_transforms`` motion events (rotation degrees,
         translation mm along x and y), independently per slice.
      2. Prepend an identity event so the subject starts still.
      3. Partition columns (phase-encode axis = -1 for FastMRI brain)
         into ``n_transforms+1`` equal segments. FastMRI brain stores
         k-space as ``(C, 640, 320)`` with oversampled frequency-encode
         at axis -2 and phase-encode at axis -1. (The task prompt's
         "second-to-last" assertion is the knee convention, not brain;
         the first real-scan verify run empirically confirmed axis=-1.)
      4. For each segment, rotate the complex k-space by ``theta`` and
         multiply by the translation phase ramp, then keep only the
         columns of this segment.
      5. Sum (non-overlapping) segments back into a full corrupted slice.

    Coil-combined iFFT:
      6. ``coil_image = ifft2c(k_corrupted)`` using the centred FastMRI
         convention ``fftshift(ifft2(ifftshift(k, axes=(-2,-1))), axes=(-2,-1))``.
         At severity 0 (zero events) this matches the stored
         ``reconstruction_rss`` within numerical precision — the
         load-bearing correctness check.
      7. ``rss = sqrt(sum(|coil_image|^2 over coils))``.

    Output:
      8. Stack slices to (H, W, n_slices), resample to 1 mm isotropic via
         torchio b-spline (consistent with 00_extract_fastmri_t1.py), save
         NIfTI plus JSON sidecar.

Inputs:
    --input-dir       Directory of FastMRI .h5 files (searched recursively).
    --reference-dir   Directory of reference magnitude NIfTIs from
                      00_extract_fastmri_t1.py (for manifest ref_path).
    --output-dir      Root of the corrupted output tree.
    --severities      Comma list of severities in 1..5, or "all".
    --seed            Base RNG seed; per-(file_id, severity) seed is derived.
    --manifest-path   Shared corruption manifest CSV (see note below).
    --n-jobs          joblib Parallel workers for slice-level parallelism.
    --dry-run         Scan and report without writing any corrupted files.
    --force           Re-extract volumes even when the output NIfTI+JSON exist.

Outputs:
    <output-dir>/severity_<sev>/<file_id>.nii.gz   Corrupted magnitude volume.
    <output-dir>/severity_<sev>/<file_id>.json     Per-scan provenance sidecar.
    <manifest-path>                                Shared corruption manifest.

Operational ordering:
    02 and 02b share ``results/tables/corruption_manifest.csv`` and are
    order-independent. Both use the same read-modify-write pattern, scoped
    to the ``corruption_type`` values each run produces:
      - 02 drops existing rows whose ``(corruption_type, dataset_tag)`` pair
        matches this run, then appends its own rows.
      - 02b drops existing rows with ``corruption_type == "motion_kspace"``
        (implicitly scoped to ``dataset_tag == "fastmri"``), then appends.
    Either script can be re-run any number of times without touching rows
    written by the other.

Usage:
    python code/02b_corrupt_kspace_motion.py \\
        --input-dir data/fastmri/raw \\
        --reference-dir data/fastmri/nifti \\
        --output-dir data/fastmri/corrupted_kspace \\
        --severities 1,2,3,4,5
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import logging
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import typer
from joblib import Parallel, delayed
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from scipy import ndimage
from tqdm import tqdm

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

KSPACE_KEY: str = "kspace"
RSS_KEY: str = "reconstruction_rss"
ACQUISITION_ATTR: str = "acquisition"

T1_ACQUISITIONS: frozenset[str] = frozenset({"AXT1", "AXT1PRE", "AXT1POST"})

# Fallback voxel sizes when ismrmrd_header parse fails (Knoll 2020 brain AXT1
# nominal: 0.6875 × 0.6875 mm in-plane × 5 mm slice). Kept symmetric with
# 00_extract_fastmri_t1.py::DEFAULT_VOXEL_MM so ref and cor volumes from 00
# and 02b are guaranteed to share affines.
DEFAULT_VOXEL_MM: tuple[float, float, float] = (0.6875, 0.6875, 5.0)
# In-plane subset used by the Fourier translation ramp. Kept as a separate
# 2-tuple so function defaults type-check cleanly (tuple[float, float]).
_DEFAULT_VOXEL_MM_XY: tuple[float, float] = (DEFAULT_VOXEL_MM[0], DEFAULT_VOXEL_MM[1])

# Path to the extractor module we reuse via importlib. 00's digit-prefixed
# filename blocks normal imports; same importlib pattern the verify script
# and tests use.
_EXTRACTOR_MODULE_PATH: Path = Path(__file__).resolve().parent / "00_extract_fastmri_t1.py"

CORRUPTION_TYPE: str = "motion_kspace"
CORRUPTION_DOMAIN: str = "kspace"
DATASET_TAG: str = "fastmri"

ALL_KEYWORD: str = "all"
VALID_SEVERITIES: tuple[int, ...] = (1, 2, 3, 4, 5)

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
    help="NeuroQC Phase 2b — physical k-space motion corruption (FastMRI).",
    add_completion=False,
)


# ──────────────────────────────────────────────
# 00-extractor interop
# ──────────────────────────────────────────────


def _load_extractor_module() -> types.ModuleType:
    """Load ``code/00_extract_fastmri_t1.py`` via importlib.

    We reuse its ``parse_voxel_sizes`` and ``build_affine`` helpers so the
    ref (produced by 00) and cor (produced by 02b) NIfTIs share the same
    affine construction path. Without this, drift between the two scripts
    would silently break downstream Dice overlap computations.
    """
    already_loaded = sys.modules.get("_neuroqc_extract_fastmri_t1")
    if already_loaded is not None:
        return already_loaded
    spec = importlib.util.spec_from_file_location(
        "_neuroqc_extract_fastmri_t1", _EXTRACTOR_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extractor at {_EXTRACTOR_MODULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_neuroqc_extract_fastmri_t1"] = mod
    spec.loader.exec_module(mod)
    return mod


# ──────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────


@dataclass(frozen=True)
class MotionSeverity:
    """Per-severity motion sampling configuration."""

    n_transforms: int
    rotation_deg: float
    translation_mm: float


# Linearly interpolated between the task's anchors {1,3,5}. Matches
# VALID_SEVERITIES from code/02_generate_corruptions.py:57.
SEVERITY_CONFIGS: dict[int, MotionSeverity] = {
    1: MotionSeverity(n_transforms=2, rotation_deg=2.0, translation_mm=2.0),
    2: MotionSeverity(n_transforms=3, rotation_deg=4.0, translation_mm=4.0),
    3: MotionSeverity(n_transforms=5, rotation_deg=6.0, translation_mm=6.0),
    4: MotionSeverity(n_transforms=7, rotation_deg=8.0, translation_mm=8.0),
    5: MotionSeverity(n_transforms=8, rotation_deg=10.0, translation_mm=10.0),
}


@dataclass(frozen=True)
class MotionEvent:
    """A single rigid-motion event: rotation in degrees, translation in mm.

    Translation is expressed in physical millimetres (not pixels) so the
    phase ramp respects the image FOV via ``voxel_mm``.
    """

    rotation_deg: float
    translation_mm_x: float
    translation_mm_y: float


@dataclass(frozen=True)
class CorruptionRecord:
    """One manifest row. Matches the project-wide 8-column schema."""

    ref_path: Path
    cor_path: Path
    severity: int
    seed: int
    severity_config: MotionSeverity
    events_per_slice: list[list[MotionEvent]] = field(default_factory=list)

    def to_manifest_row(self) -> dict[str, str]:
        return {
            "ref_path": str(self.ref_path),
            "cor_path": str(self.cor_path),
            "corruption_type": CORRUPTION_TYPE,
            "corruption_domain": CORRUPTION_DOMAIN,
            "severity": str(self.severity),
            "seed": str(self.seed),
            "transform_params": json.dumps(
                {
                    "n_transforms": self.severity_config.n_transforms,
                    "rotation_deg": self.severity_config.rotation_deg,
                    "translation_mm": self.severity_config.translation_mm,
                },
                sort_keys=True,
            ),
            "dataset_tag": DATASET_TAG,
        }


# ──────────────────────────────────────────────
# Core algorithm
# ──────────────────────────────────────────────


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    """Centred, orthonormal 2D iFFT along the last two axes (FastMRI).

    Matches the ``reconstruction_rss`` dataset stored in FastMRI .h5 files
    exactly when applied slice-wise per coil:
    ``rss = sqrt(sum(|ifft2c(k)|^2, axis=coil))`` — after FE-oversampling
    crop (see ``center_crop_to_square``).

    Three conventions must all match FastMRI's pipeline:
      * shift sequence: ``fftshift(ifft2(ifftshift(k)))`` (centred DC).
      * normalization: ``norm='ortho'`` — FastMRI uses the symmetric
        ``1/√(N·M)`` scaling, not numpy's default ``1/(N·M)``. Without
        this the reconstruction is off by a factor of √(N·M) ≈ 452 for
        640×320 k-space — exactly the 1/452 ≈ 0.22% we saw on our first
        real-scan verify run before this fix was in place.
      * FE oversampling crop is applied downstream in
        ``reconstruct_rss_volume``.
    """
    shifted = np.fft.ifftshift(kspace, axes=(-2, -1))
    image = np.fft.ifft2(shifted, axes=(-2, -1), norm="ortho")
    return np.fft.fftshift(image, axes=(-2, -1))


def rotate_kspace_2d(kspace_2d: np.ndarray, theta_deg: float) -> np.ndarray:
    """Rotate a 2D complex k-space slice by ``theta_deg`` (CCW positive).

    Fourier rotation theorem: rotating the image by θ is equivalent to
    rotating k-space by the same θ. We apply the rotation as a grid
    resample on real and imaginary channels separately (scipy.ndimage is
    allow-listed per task prompt; we stay narrow-scope).

    ``theta_deg == 0.0`` is treated as an exact identity (bit-for-bit)
    so the zero-event path through ``apply_motion_to_slice`` is a no-op
    — required for the severity-0 test.
    """
    if theta_deg == 0.0:
        return kspace_2d
    real = ndimage.rotate(
        kspace_2d.real,
        angle=theta_deg,
        reshape=False,
        order=3,
        mode="constant",
        cval=0.0,
    )
    imag = ndimage.rotate(
        kspace_2d.imag,
        angle=theta_deg,
        reshape=False,
        order=3,
        mode="constant",
        cval=0.0,
    )
    return (real + 1j * imag).astype(kspace_2d.dtype)


def translation_phase_ramp(
    height: int,
    width: int,
    tx_mm: float,
    ty_mm: float,
    voxel_mm: tuple[float, float] = _DEFAULT_VOXEL_MM_XY,
) -> np.ndarray:
    """Return a (H, W) complex phase ramp for a Fourier image-space shift.

    Conventions:
      * The k-space array is stored with DC at the array centre
        (FastMRI). Frequency at row i is ``(i - H//2)/(H*vy)`` cycles/mm.
      * ``tx_mm``, ``ty_mm`` are image-space translations in millimetres.
        Applying this ramp to k-space (element-wise multiply) shifts the
        reconstructed image by ``(+ty_mm, +tx_mm)``.
      * ``voxel_mm = (vy, vx)``.

    Both translations equal to zero returns an all-ones array, so the
    zero-event path is an exact no-op.
    """
    if tx_mm == 0.0 and ty_mm == 0.0:
        return np.ones((height, width), dtype=np.complex64)
    vy, vx = voxel_mm
    ky = (np.arange(height) - height // 2) / (height * vy)  # (H,) cycles/mm
    kx = (np.arange(width) - width // 2) / (width * vx)     # (W,) cycles/mm
    phase = np.exp(
        -2j * np.pi * (ky[:, None] * ty_mm + kx[None, :] * tx_mm)
    ).astype(np.complex64)
    return phase


def sample_events(rng: np.random.Generator, config: MotionSeverity) -> list[MotionEvent]:
    """Sample ``config.n_transforms`` motion events from uniform ranges.

    rotation ~ U(-rot, +rot), tx ~ U(-t, +t), ty ~ U(-t, +t).
    Returns a list; may be empty when ``n_transforms == 0``.
    """
    events: list[MotionEvent] = []
    for _ in range(config.n_transforms):
        events.append(
            MotionEvent(
                rotation_deg=float(rng.uniform(-config.rotation_deg, config.rotation_deg)),
                translation_mm_x=float(rng.uniform(-config.translation_mm, config.translation_mm)),
                translation_mm_y=float(rng.uniform(-config.translation_mm, config.translation_mm)),
            )
        )
    return events


def apply_motion_to_slice(
    kspace_slice: np.ndarray,
    events: list[MotionEvent],
    voxel_mm: tuple[float, float] = _DEFAULT_VOXEL_MM_XY,
) -> np.ndarray:
    """Apply segmented rigid motion to one multi-coil k-space slice.

    Partitions the phase-encode axis (W, axis=-1) into ``len(events)+1``
    segments. Each segment receives a distinct motion state: the first
    segment is identity (subject still), then one state per event.

    PE axis convention (empirically verified on FastMRI brain
    ``file_brain_AXT1_201_6002725.h5``): k-space shape is
    ``(C, 640, 320)`` with the 2×-oversampled FE axis at -2 and the
    phase-encode axis at -1. Motion during acquisition therefore
    affects lines along axis -1. The task prompt's "second-to-last"
    assertion describes the FastMRI knee convention, not brain.

    Args:
        kspace_slice: Complex array of shape ``(n_coils, H, W)``.
        events: Motion events sampled for this slice; may be empty.
        voxel_mm: In-plane voxel size ``(vy, vx)`` in mm.

    Returns:
        Corrupted complex k-space, same shape and dtype as the input.

    Empty ``events`` returns the input unchanged (bit-for-bit), which is
    what the severity-0 round-trip test relies on.
    """
    if kspace_slice.ndim != 3:
        raise ValueError(
            f"kspace_slice must be (n_coils, H, W); got {kspace_slice.shape}"
        )
    if not events:
        return kspace_slice

    coils, height, width = kspace_slice.shape

    # Prepend an identity segment so the subject starts still.
    segment_events = [MotionEvent(0.0, 0.0, 0.0), *events]
    n_segments = len(segment_events)

    # Partition columns (PE axis=-1). np.array_split handles non-divisible sizes.
    col_groups = np.array_split(np.arange(width), n_segments)

    result = np.zeros_like(kspace_slice)
    for event, cols in zip(segment_events, col_groups):
        if cols.size == 0:
            continue
        # Rotation is identical across coils (rigid-body subject motion).
        if event.rotation_deg == 0.0:
            rotated = kspace_slice
        else:
            rotated = np.empty_like(kspace_slice)
            for c in range(coils):
                rotated[c] = rotate_kspace_2d(kspace_slice[c], event.rotation_deg)
        ramp = translation_phase_ramp(
            height, width, event.translation_mm_x, event.translation_mm_y, voxel_mm
        )
        transformed = rotated * ramp[None, :, :]
        # Only this segment's columns keep this motion state; others are
        # zero here and will be filled by other segments. Columns are
        # disjoint across segments, so nothing is overwritten.
        result[:, :, cols] = transformed[:, :, cols]
    return result


def center_crop_to_square(images: np.ndarray) -> np.ndarray:
    """Center-crop the last two axes to ``min(H, W) × min(H, W)``.

    FastMRI stores brain k-space with a 2×-oversampled frequency-encode
    axis (H=640 vs W=320 for AXT1). The stored ``reconstruction_rss``
    dataset is cropped to square (320×320) before RSS. We mirror that
    step so our reconstructions align with both the stored reference
    and the reference NIfTI emitted by ``00_extract_fastmri_t1.py``.

    No-op on already-square inputs (synthetic test fixtures, typically).
    """
    *_, height, width = images.shape
    target = min(height, width)
    h0 = (height - target) // 2
    w0 = (width - target) // 2
    return images[..., h0 : h0 + target, w0 : w0 + target]


def reconstruct_rss_volume(kspace_vol: np.ndarray) -> np.ndarray:
    """Reconstruct per-slice RSS magnitude from multi-coil k-space.

    Pipeline:
      1. Centred iFFT per slice per coil (see ``ifft2c``).
      2. Center-crop to square to undo FE oversampling (see
         ``center_crop_to_square``).
      3. RSS over the coil axis.

    Args:
        kspace_vol: Complex array of shape ``(n_slices, n_coils, H, W)``.

    Returns:
        Float32 array of shape ``(n_slices, min(H,W), min(H,W))`` — the
        post-crop square magnitude RSS per slice.
    """
    coil_images = ifft2c(kspace_vol)
    coil_images = center_crop_to_square(coil_images)
    rss = np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=1))
    return rss.astype(np.float32)


def corrupt_kspace_volume(
    kspace_vol: np.ndarray,
    severity_config: MotionSeverity,
    base_seed: int,
    voxel_mm: tuple[float, float] = _DEFAULT_VOXEL_MM_XY,
    n_jobs: int = 1,
) -> tuple[np.ndarray, list[list[MotionEvent]]]:
    """Corrupt an entire multi-slice k-space volume.

    Each slice gets its own event list sampled from a deterministic
    per-slice seed derived from ``base_seed``, so reruns are reproducible
    and independent slices cannot accidentally share identical sampling.
    """
    n_slices = kspace_vol.shape[0]
    slice_seeds = [base_seed + i for i in range(n_slices)]
    rngs = [np.random.default_rng(s) for s in slice_seeds]
    events_per_slice = [sample_events(rng, severity_config) for rng in rngs]

    if n_jobs == 1 or n_slices == 1:
        corrupted_slices = [
            apply_motion_to_slice(kspace_vol[i], events_per_slice[i], voxel_mm)
            for i in range(n_slices)
        ]
    else:
        corrupted_slices = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(apply_motion_to_slice)(kspace_vol[i], events_per_slice[i], voxel_mm)
            for i in range(n_slices)
        )
    return np.stack(corrupted_slices, axis=0), events_per_slice


# ──────────────────────────────────────────────
# Seeding
# ──────────────────────────────────────────────


def derive_seed(file_id: str, severity: int, base_seed: int) -> int:
    """Deterministic per-(file, severity) seed via SHA-256 of the tuple.

    Returns an int in ``[0, 2**31)`` suitable for ``np.random.default_rng``.
    Uses hashlib, not Python's ``hash`` — Python's hash is salted across
    process invocations and therefore non-reproducible.
    """
    digest = hashlib.sha256(f"{file_id}|{severity}|{base_seed}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


# ──────────────────────────────────────────────
# Per-file orchestration
# ──────────────────────────────────────────────


def _read_acquisition(h5_path: Path) -> str:
    """Return attrs['acquisition'] decoded to str, or empty if missing."""
    with h5py.File(h5_path, "r") as handle:
        raw = handle.attrs.get(ACQUISITION_ATTR)
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def _write_sidecar(
    path: Path,
    h5_path: Path,
    severity: int,
    seed: int,
    events_per_slice: list[list[MotionEvent]],
    severity_config: MotionSeverity,
) -> None:
    """Write the per-scan JSON provenance sidecar."""
    payload = {
        "source_h5": str(h5_path),
        "corruption_type": CORRUPTION_TYPE,
        "corruption_domain": CORRUPTION_DOMAIN,
        "severity": severity,
        "seed": seed,
        "n_transforms": severity_config.n_transforms,
        "rotation_deg": severity_config.rotation_deg,
        "translation_mm": severity_config.translation_mm,
        "fft_convention": "ifft2c_centered",
        "phase_encode_axis": "W (last axis) — FastMRI brain convention",
        "events_per_slice": [
            [
                {
                    "rotation_deg": e.rotation_deg,
                    "tx_mm": e.translation_mm_x,
                    "ty_mm": e.translation_mm_y,
                }
                for e in slice_events
            ]
            for slice_events in events_per_slice
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def corrupt_one(
    h5_path: Path,
    reference_dir: Path,
    output_dir: Path,
    severity: int,
    base_seed: int,
    *,
    dry_run: bool = False,
    force: bool = False,
    voxel_mm: tuple[float, float, float] | None = None,
    n_jobs: int = 1,
) -> CorruptionRecord | None:
    """Corrupt one FastMRI .h5 at one severity; return a manifest record.

    Returns None when the file's acquisition attribute is not a T1 variant.
    When an output already exists and ``force`` is False, a record is still
    produced (for the manifest) but the volume is not re-written.

    NIfTI is saved with the anisotropic affine parsed from the .h5's
    ``ismrmrd_header`` — same path as ``00_extract_fastmri_t1.py``, so
    ref and cor NIfTIs share the same voxel geometry. The in-plane
    subset ``(vy, vx)`` is threaded into the translation phase ramp so
    motion in mm is physically correct relative to the pixel grid.
    Pass ``voxel_mm`` explicitly to override the parsed value (used in
    the synthetic tests that don't ship a header).
    """
    acquisition = _read_acquisition(h5_path)
    if acquisition not in T1_ACQUISITIONS:
        logger.debug("Skipping %s: acquisition %r not a T1 variant", h5_path.name, acquisition)
        return None

    extractor = _load_extractor_module()
    if voxel_mm is None:
        voxel_mm = extractor.parse_voxel_sizes(h5_path)
    voxel_mm_xy: tuple[float, float] = (voxel_mm[0], voxel_mm[1])

    file_id = h5_path.stem
    severity_dir = output_dir / f"severity_{severity}"
    out_nifti = severity_dir / f"{file_id}.nii.gz"
    out_json = severity_dir / f"{file_id}.json"
    ref_path = (reference_dir / f"{file_id}.nii.gz").resolve()

    config = SEVERITY_CONFIGS[severity]
    seed = derive_seed(file_id, severity, base_seed)

    # Resume: both NIfTI and JSON must exist to count as "already done".
    if out_nifti.exists() and out_json.exists() and not force:
        logger.info("Resume: %s (sev=%d) already extracted; skipping volume write", file_id, severity)
        return CorruptionRecord(
            ref_path=ref_path,
            cor_path=out_nifti.resolve(),
            severity=severity,
            seed=seed,
            severity_config=config,
        )

    if dry_run:
        logger.info("dry-run: would corrupt %s @ sev=%d -> %s", file_id, severity, out_nifti)
        return CorruptionRecord(
            ref_path=ref_path,
            cor_path=out_nifti.resolve(),
            severity=severity,
            seed=seed,
            severity_config=config,
        )

    with h5py.File(h5_path, "r") as handle:
        if KSPACE_KEY not in handle:
            raise KeyError(f"{h5_path.name}: missing dataset {KSPACE_KEY!r}")
        kspace_vol = handle[KSPACE_KEY][:]  # (n_slices, n_coils, H, W), complex
    if kspace_vol.ndim != 4:
        raise ValueError(
            f"{h5_path.name}: kspace has shape {kspace_vol.shape}; expected 4D"
        )

    kspace_corrupted, events_per_slice = corrupt_kspace_volume(
        kspace_vol,
        severity_config=config,
        base_seed=seed,
        voxel_mm=voxel_mm_xy,
        n_jobs=n_jobs,
    )
    rss = reconstruct_rss_volume(kspace_corrupted)  # (n_slices, H_c, W_c) float32
    volume = np.ascontiguousarray(np.transpose(rss, (1, 2, 0)))  # (H, W, D)
    affine = extractor.build_affine(voxel_mm)

    severity_dir.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(volume, affine=affine)
    nib.save(img, str(out_nifti))
    _write_sidecar(out_json, h5_path, severity, seed, events_per_slice, config)

    return CorruptionRecord(
        ref_path=ref_path,
        cor_path=out_nifti.resolve(),
        severity=severity,
        seed=seed,
        severity_config=config,
        events_per_slice=events_per_slice,
    )


# ──────────────────────────────────────────────
# Manifest I/O (shared with code/02_generate_corruptions.py)
# ──────────────────────────────────────────────


def _read_existing_manifest(path: Path) -> list[dict[str, str]]:
    """Return existing manifest rows, or [] when the file doesn't exist.

    Validates that the header matches MANIFEST_COLUMNS exactly; a mismatch
    indicates a schema drift between 02 and 02b that must be reconciled
    by hand rather than silently discarded.
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


def update_manifest(
    manifest_path: Path, new_records: list[CorruptionRecord]
) -> int:
    """Idempotently replace motion_kspace rows in the shared manifest.

    - If ``manifest_path`` doesn't exist, write header + new rows.
    - Otherwise, read existing rows, drop those with
      ``corruption_type == CORRUPTION_TYPE``, append new rows, rewrite.

    This leaves rows written by code/02_generate_corruptions.py untouched.
    Returns the number of new rows written (not total rows in the file).
    """
    existing = _read_existing_manifest(manifest_path)
    kept = [r for r in existing if r.get("corruption_type") != CORRUPTION_TYPE]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in kept:
            writer.writerow(row)
        for record in new_records:
            writer.writerow(record.to_manifest_row())
    return len(new_records)


# ──────────────────────────────────────────────
# Batch orchestration
# ──────────────────────────────────────────────


def corrupt_all(
    input_dir: Path,
    reference_dir: Path,
    output_dir: Path,
    severities: list[int],
    base_seed: int,
    manifest_path: Path,
    *,
    dry_run: bool = False,
    force: bool = False,
    voxel_mm: tuple[float, float, float] | None = None,
    n_jobs: int = 1,
) -> list[CorruptionRecord]:
    """Corrupt every T1 .h5 under ``input_dir`` at each severity.

    ``voxel_mm``: optional per-call override of the per-file parsed voxel
    sizes (used by tests to pin the fixture's geometry). When None, each
    file's ismrmrd_header is parsed to derive its own voxel sizes.
    """
    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input-dir does not exist: {input_dir}")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(input_dir.rglob("*.h5"))
    if not h5_files:
        logger.warning("No .h5 files found under %s", input_dir)
        if not dry_run:
            update_manifest(manifest_path, [])
        return []

    iterator: object = (
        tqdm(h5_files, desc="Corrupting", unit="file")
        if len(h5_files) > 10
        else h5_files
    )
    records: list[CorruptionRecord] = []
    for h5_path in iterator:  # type: ignore[assignment]
        for severity in severities:
            try:
                record = corrupt_one(
                    h5_path,
                    reference_dir=reference_dir,
                    output_dir=output_dir,
                    severity=severity,
                    base_seed=base_seed,
                    dry_run=dry_run,
                    force=force,
                    voxel_mm=voxel_mm,
                    n_jobs=n_jobs,
                )
            except Exception as exc:
                logger.exception(
                    "Failed to corrupt %s @ severity=%d: %s", h5_path, severity, exc
                )
                continue
            if record is not None:
                records.append(record)

    if not dry_run:
        update_manifest(manifest_path, records)
    return records


# ──────────────────────────────────────────────
# CLI parsing helpers
# ──────────────────────────────────────────────


def _parse_severities(spec: str) -> list[int]:
    """Parse the --severities CLI value into a list of ints in VALID_SEVERITIES.

    Mirrors code/02_generate_corruptions.py:_parse_severities but returns a
    concrete list (never None) because 02b does not delegate to a generator
    with its own default; we own the loop here.
    """
    if spec.strip().lower() == ALL_KEYWORD:
        return list(VALID_SEVERITIES)
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
            f"Severities must be in {VALID_SEVERITIES}; invalid entries: {invalid}"
        )
    return items


# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────


def _print_summary(
    records: list[CorruptionRecord],
    manifest_path: Path,
    console: Console,
) -> None:
    table = Table(title="k-space motion corruption summary")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("records produced", str(len(records)))
    table.add_row("manifest path", str(manifest_path))
    console.print(table)

    if not records:
        return
    by_sev: dict[int, int] = {}
    for record in records:
        by_sev[record.severity] = by_sev.get(record.severity, 0) + 1
    sev_table = Table(title="By severity")
    sev_table.add_column("severity", style="bold")
    sev_table.add_column("count", justify="right")
    for severity, count in sorted(by_sev.items()):
        sev_table.add_row(str(severity), str(count))
    console.print(sev_table)


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
        help="Directory containing FastMRI .h5 files (recursed).",
    ),
    reference_dir: Path = typer.Option(
        Path("data/fastmri/nifti"),
        "--reference-dir",
        resolve_path=True,
        help="Directory of reference NIfTIs from 00_extract_fastmri_t1.py.",
    ),
    output_dir: Path = typer.Option(
        Path("data/fastmri/corrupted_kspace"),
        "--output-dir",
        resolve_path=True,
        help="Output root for corrupted volumes (severity_<n>/ subtree).",
    ),
    severities: str = typer.Option(
        ",".join(str(s) for s in VALID_SEVERITIES),
        "--severities",
        help="Comma-separated severity levels in 1..5, or 'all'.",
    ),
    seed: int = typer.Option(
        42, "--seed", help="Base RNG seed for deterministic per-file sampling."
    ),
    manifest_path: Path = typer.Option(
        Path("results/tables/corruption_manifest.csv"),
        "--manifest-path",
        help="Shared corruption manifest CSV (idempotent motion_kspace rows).",
    ),
    n_jobs: int = typer.Option(
        -1,
        "--n-jobs",
        help="joblib workers for slice-level parallelism (-1 = all cores).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the plan without writing any files."
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-extract even when outputs already exist."
    ),
) -> None:
    """Apply physical k-space motion corruption to FastMRI .h5 files."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )
    severity_list = _parse_severities(severities)
    # Resolve joblib -1 now so tests (which pass n_jobs=1) don't need to fight
    # multiprocessing quirks. joblib itself handles -1 -> all cores; nothing
    # else to do.
    records = corrupt_all(
        input_dir=input_dir,
        reference_dir=reference_dir,
        output_dir=output_dir,
        severities=severity_list,
        base_seed=seed,
        manifest_path=manifest_path,
        dry_run=dry_run,
        force=force,
        n_jobs=n_jobs,
    )
    _print_summary(records, manifest_path, Console())


if __name__ == "__main__":
    app()
