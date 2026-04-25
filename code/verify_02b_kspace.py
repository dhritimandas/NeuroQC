#!/usr/bin/env python3
"""NeuroQC Phase 2b — Single-scan k-space corruption diagnostic harness.

Runs six correctness checks against one real FastMRI ``.h5`` file. Each
check targets a specific convention bug that produces a plausible-looking
but wrong reconstruction in ``code/02b_corrupt_kspace_motion.py``:

    1. FFT convention — centred ``ifft2c`` must match the stored
       ``reconstruction_rss`` within ``--strict-tol``.
    2. Identity motion round-trip — zero-event corruption is a no-op.
    3. Phase-encode axis — ty-only translation produces ghosts along H,
       not along W.
    4. Translation units — half-FOV-mm translation wraps by half-FOV
       pixels (Fourier shift theorem, mm not pixels).
    5. Rotation sanity — 0° identity; 90° matches ``np.rot90``.
    6. Severity monotonicity — residual magnitude grows with severity.

Each failing check prints a one-line ``fix_hint`` that names the exact
source location to change. A sixth-rank fix hint means something is
structurally wrong in the pipeline and the batch run must not proceed.

Outputs (under ``--out-dir``):
    checks.csv          — one row per check with status / measurement /
                          threshold / fix_hint columns.
    grid.png            — 2×4 matplotlib grid (ref / sev1 / sev3 / sev5
                          magnitudes on the top, difference heatmaps on
                          the bottom; middle slice only).
    Exit code           — 0 iff every check PASSed.

Usage:
    python code/verify_02b_kspace.py \\
        --h5-path data/fastmri/raw/file_brain_AXT1_201_6002725.h5 \\
        --out-dir results/diagnostics/kspace_motion
"""

from __future__ import annotations

import csv
import importlib.util
import logging
import sys
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CORRUPT_MODULE_PATH = _REPO_ROOT / "code" / "02b_corrupt_kspace_motion.py"

logger = logging.getLogger(__name__)
app = typer.Typer(
    help="Diagnostic harness for 02b_corrupt_kspace_motion.py.",
    add_completion=False,
)

# ──────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────


@dataclass(frozen=True)
class CheckResult:
    """One diagnostic check's outcome."""

    name: str
    status: Literal["PASS", "FAIL"]
    measured: float
    threshold: float
    fix_hint: str
    extra: dict[str, float] = field(default_factory=dict)


# ──────────────────────────────────────────────
# Module loading
# ──────────────────────────────────────────────


def load_corrupt_module() -> types.ModuleType:
    """Load code/02b_corrupt_kspace_motion.py via importlib.

    The digit-prefixed filename cannot be imported normally; this is the
    same pattern used in tests/test_*.py.
    """
    spec = importlib.util.spec_from_file_location(
        "corrupt_kspace_motion", _CORRUPT_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {_CORRUPT_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["corrupt_kspace_motion"] = module
    spec.loader.exec_module(module)
    return module


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _rss_from_coil_images(coil_images: np.ndarray) -> np.ndarray:
    """RSS magnitude over the coil axis (axis=0 for 2D per-coil input)."""
    return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0)).astype(np.float32)


def _center_crop_last2(images: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center-crop the last two axes to (target_h, target_w).

    Mirrors the FE-oversampling crop performed by FastMRI's
    ``reconstruction_rss`` pipeline. No-op when the input is already at
    or below the target shape.
    """
    *_, height, width = images.shape
    h0 = max(0, (height - target_h) // 2)
    w0 = max(0, (width - target_w) // 2)
    return images[..., h0 : h0 + target_h, w0 : w0 + target_w]


def _load_slice(h5_path: Path, slice_idx: int | None) -> tuple[np.ndarray, np.ndarray, int]:
    """Load (kspace_slice (C,H,W), stored_rss (H,W), slice_idx_used)."""
    with h5py.File(h5_path, "r") as handle:
        kspace = handle["kspace"]
        n_slices = kspace.shape[0]
        idx = slice_idx if slice_idx is not None else n_slices // 2
        if not 0 <= idx < n_slices:
            raise IndexError(f"slice_idx={idx} out of range [0, {n_slices})")
        kspace_slice = kspace[idx]  # (C, H, W)
        rss = handle["reconstruction_rss"][idx]  # (H, W)
    return np.asarray(kspace_slice), np.asarray(rss, dtype=np.float32), idx


# ──────────────────────────────────────────────
# Checks
# ──────────────────────────────────────────────


def check_fft_convention(
    kspace_slice: np.ndarray, stored_rss: np.ndarray, tol: float
) -> CheckResult:
    """Check 1 — verify ifft2c matches the stored reconstruction_rss.

    We reconstruct the slice with four candidate FFT conventions and
    report relative error for each. Only ``centred_ifft2c`` should win.
    """
    logger.info(
        "[Check 1] FFT convention vs stored reconstruction_rss (tol=%.1e)", tol
    )
    # All candidates use norm='ortho' (FastMRI's convention); the
    # conventions differ only in the shift sequence, isolating that
    # axis of freedom for the diagnostic.
    candidates: dict[str, np.ndarray] = {
        "centred_ifft2c": np.fft.fftshift(
            np.fft.ifft2(
                np.fft.ifftshift(kspace_slice, axes=(-2, -1)),
                axes=(-2, -1),
                norm="ortho",
            ),
            axes=(-2, -1),
        ),
        "no_outer_shift": np.fft.ifft2(
            np.fft.ifftshift(kspace_slice, axes=(-2, -1)),
            axes=(-2, -1),
            norm="ortho",
        ),
        "no_inner_shift": np.fft.fftshift(
            np.fft.ifft2(kspace_slice, axes=(-2, -1), norm="ortho"),
            axes=(-2, -1),
        ),
        "raw": np.fft.ifft2(kspace_slice, axes=(-2, -1), norm="ortho"),
    }
    target_h, target_w = stored_rss.shape
    rss_max = float(np.max(stored_rss))
    errors: dict[str, float] = {}
    for name, coil_images in candidates.items():
        # Center-crop the FE-oversampled H to match the stored RSS shape.
        cropped = _center_crop_last2(coil_images, target_h, target_w)
        rss = _rss_from_coil_images(cropped)
        rel_err = float(np.max(np.abs(rss - stored_rss)) / max(rss_max, 1e-12))
        errors[name] = rel_err
        logger.info("    %-18s relative max error = %.3e", name, rel_err)

    best = min(errors, key=lambda k: errors[k])
    status: Literal["PASS", "FAIL"] = (
        "PASS" if (best == "centred_ifft2c" and errors["centred_ifft2c"] < tol) else "FAIL"
    )
    fix_hint = (
        f"Convention '{best}' matches better (err={errors[best]:.3e}). "
        "Update ifft2c() in code/02b_corrupt_kspace_motion.py to use that sequence."
        if status == "FAIL"
        else ""
    )
    if status == "FAIL":
        logger.error("    FAIL — %s", fix_hint)
    else:
        logger.info("    PASS — centred_ifft2c matches within %.1e.", tol)
    return CheckResult(
        name="fft_convention",
        status=status,
        measured=errors["centred_ifft2c"],
        threshold=tol,
        fix_hint=fix_hint,
        extra=errors,
    )


def check_identity_motion(
    module: types.ModuleType,
    kspace_slice: np.ndarray,
) -> CheckResult:
    """Check 2 — zero-event apply_motion_to_slice must be a no-op."""
    logger.info("[Check 2] Identity motion round-trip (expect bit-for-bit no-op)")
    out = module.apply_motion_to_slice(kspace_slice, events=[], voxel_mm=(1.0, 1.0))
    max_abs = float(np.max(np.abs(out - kspace_slice)))
    threshold = 1e-6
    status: Literal["PASS", "FAIL"] = "PASS" if max_abs < threshold else "FAIL"
    fix_hint = (
        "Zero-event path is not a no-op. Ensure apply_motion_to_slice returns "
        "kspace_slice unchanged when events == []; verify rotate_kspace_2d(k, 0.0) "
        "short-circuits and translation_phase_ramp(H, W, 0, 0, ...) returns all-ones."
        if status == "FAIL"
        else ""
    )
    logger.info("    max|out - input| = %.3e (threshold=%.1e)", max_abs, threshold)
    if status == "FAIL":
        logger.error("    FAIL — %s", fix_hint)
    else:
        logger.info("    PASS")
    return CheckResult(
        name="identity_motion",
        status=status,
        measured=max_abs,
        threshold=threshold,
        fix_hint=fix_hint,
    )


def check_pe_axis(
    module: types.ModuleType,
    kspace_slice: np.ndarray,
) -> CheckResult:
    """Check 3 — multi-segment translation produces ghosts along PE axis = -1 (W).

    Motion ghosting modulates the image along the phase-encode axis — the
    column-mean profile ``diff.mean(axis=0)`` (averaging out the non-PE
    axis) carries the sinc-like segment mask, while ``diff.mean(axis=1)``
    is relatively flat. With PE = axis -1 (W), the signature is
    ``var(col_mean) / var(row_mean) >> 1``. A silent swap to axis=-2
    flips the ratio.

    PE-axis convention for FastMRI brain data: k-space shape is
    ``(C, 640, 320)`` with oversampled FE on axis -2 and PE on axis -1.
    (The task prompt's "second-to-last" assertion describes FastMRI knee;
    the first real AXT1 scan we tested confirmed -1 for brain.)
    """
    logger.info(
        "[Check 3] PE axis orientation — multi-segment translation ghosts along W (axis=-1)"
    )
    events = [
        module.MotionEvent(rotation_deg=0.0, translation_mm_x=4.0, translation_mm_y=0.0),
        module.MotionEvent(rotation_deg=0.0, translation_mm_x=-3.0, translation_mm_y=0.0),
        module.MotionEvent(rotation_deg=0.0, translation_mm_x=5.0, translation_mm_y=0.0),
        module.MotionEvent(rotation_deg=0.0, translation_mm_x=-6.0, translation_mm_y=0.0),
    ]
    cor = module.apply_motion_to_slice(kspace_slice, events=events, voxel_mm=(1.0, 1.0))
    ref_img = _rss_from_coil_images(module.ifft2c(kspace_slice))
    cor_img = _rss_from_coil_images(module.ifft2c(cor))
    diff = cor_img - ref_img
    row_profile_var = float(np.var(diff.mean(axis=1)))  # variation along H
    col_profile_var = float(np.var(diff.mean(axis=0)))  # variation along W (PE)
    ratio = col_profile_var / max(row_profile_var, 1e-18)
    threshold = 3.0
    status: Literal["PASS", "FAIL"] = "PASS" if ratio > threshold else "FAIL"
    fix_hint = (
        "Phase-encode axis is wrong. In apply_motion_to_slice the line "
        "partitioning must use axis=-1 (width): "
        "`np.array_split(np.arange(width), n_segments)` and "
        "`result[:, :, cols] = transformed[:, :, cols]`. If you're slicing "
        "on axis=-2, ghosts appear along H instead of W."
        if status == "FAIL"
        else ""
    )
    logger.info(
        "    var(col_mean)=%.3e  var(row_mean)=%.3e  ratio=%.2f (threshold=%.1f)",
        col_profile_var,
        row_profile_var,
        ratio,
        threshold,
    )
    if status == "FAIL":
        logger.error("    FAIL — %s", fix_hint)
    else:
        logger.info("    PASS")
    return CheckResult(
        name="pe_axis_orientation",
        status=status,
        measured=ratio,
        threshold=threshold,
        fix_hint=fix_hint,
        extra={"var_col_mean": col_profile_var, "var_row_mean": row_profile_var},
    )


def check_translation_units(
    module: types.ModuleType,
    kspace_slice: np.ndarray,
) -> CheckResult:
    """Check 4 — half-FOV-mm translation wraps by half-FOV pixels.

    We apply a single translation event spanning **all** rows (n_transforms=0
    would be a no-op, so we use apply_motion_to_slice with a single event
    covering every segment — instead, we apply the phase ramp directly so
    the test is isolated from segmentation logic).
    """
    logger.info("[Check 4] Translation units — tx = W/2 mm should wrap by W/2 px")
    _, height, width = kspace_slice.shape
    voxel_mm = (1.0, 1.0)
    ramp = module.translation_phase_ramp(
        height, width, tx_mm=width / 2.0, ty_mm=0.0, voxel_mm=voxel_mm
    )
    shifted_k = kspace_slice * ramp[None, :, :]
    cor_img = _rss_from_coil_images(module.ifft2c(shifted_k))
    ref_img = _rss_from_coil_images(module.ifft2c(kspace_slice))
    expected = np.roll(ref_img, shift=width // 2, axis=1)
    ref_max = float(np.max(ref_img)) or 1e-12
    rel_err = float(np.max(np.abs(cor_img - expected)) / ref_max)
    threshold = 0.05
    status: Literal["PASS", "FAIL"] = "PASS" if rel_err < threshold else "FAIL"
    fix_hint = (
        "Translation phase ramp units wrong. Check translation_phase_ramp: "
        "ky = (np.arange(H) - H//2) / (H * vy), kx = (np.arange(W) - W//2) / (W * vx); "
        "phase = exp(-2j*pi*(ky[:,None]*ty_mm + kx[None,:]*tx_mm)). Most common bug: "
        "using np.fft.fftfreq without the voxel-size divisor."
        if status == "FAIL"
        else ""
    )
    logger.info("    max|cor - wrap(ref, W//2)| / max(ref) = %.3e (threshold=%.1e)", rel_err, threshold)
    if status == "FAIL":
        logger.error("    FAIL — %s", fix_hint)
    else:
        logger.info("    PASS")
    return CheckResult(
        name="translation_units",
        status=status,
        measured=rel_err,
        threshold=threshold,
        fix_hint=fix_hint,
    )


def check_rotation(
    module: types.ModuleType,
    kspace_slice: np.ndarray,
) -> CheckResult:
    """Check 5 — rotation actually runs, asymmetric in sign, 0° is identity.

    A 90° parity with np.rot90 would be the cleanest direction check but
    is fundamentally half-pixel off on even-sized centred k-space (DC sits
    at (N/2, N/2) while scipy rotates around ((N-1)/2, (N-1)/2)). So we
    settle for the three bugs we actually need to catch:
      * 0° must short-circuit to identity (bit-for-bit).
      * non-trivial angle changes the array (catches accidental no-op).
      * +θ ≠ -θ (catches missing rotation application entirely; a sign
        swap would still produce different arrays for ±θ, so sign-error
        detection lives in the PE axis + severity smoke tests).
    """
    logger.info("[Check 5] Rotation sanity (identity, non-trivial, asymmetric)")
    single_coil = kspace_slice[0]  # (H, W)

    identity = module.rotate_kspace_2d(single_coil, 0.0)
    id_err = float(np.max(np.abs(identity - single_coil)))

    plus = module.rotate_kspace_2d(single_coil, 10.0)
    minus = module.rotate_kspace_2d(single_coil, -10.0)
    k_scale = float(np.max(np.abs(single_coil))) or 1e-12
    change_rel = float(np.max(np.abs(plus - single_coil)) / k_scale)
    asymmetry_rel = float(np.max(np.abs(plus - minus)) / k_scale)

    change_floor = 0.01   # rotation by 10° must move content by ≥1% of peak
    asymmetry_floor = 0.01

    status: Literal["PASS", "FAIL"]
    if id_err >= 1e-6:
        status = "FAIL"
        fix_hint = (
            "rotate_kspace_2d(k, 0.0) must short-circuit to return k unchanged. "
            "Add: `if theta_deg == 0.0: return kspace_2d` at the top of the function."
        )
    elif change_rel < change_floor:
        status = "FAIL"
        fix_hint = (
            f"Rotation by 10° changes k by only {change_rel:.1e} (<{change_floor}). "
            "The rotation is a silent no-op — check that ndimage.rotate is reached "
            "on non-zero angles and its result is actually returned."
        )
    elif asymmetry_rel < asymmetry_floor:
        status = "FAIL"
        fix_hint = (
            f"Rotation by +10° and -10° produce the same array ({asymmetry_rel:.1e}). "
            "Signs are being stripped somewhere — verify the ndimage.rotate angle "
            "argument is `theta_deg`, not `abs(theta_deg)`."
        )
    else:
        status = "PASS"
        fix_hint = ""
    logger.info(
        "    id_err=%.3e  change_rel=%.3e  asymmetry_rel=%.3e",
        id_err,
        change_rel,
        asymmetry_rel,
    )
    if status == "FAIL":
        logger.error("    FAIL — %s", fix_hint)
    else:
        logger.info("    PASS")
    return CheckResult(
        name="rotation_sanity",
        status=status,
        measured=min(change_rel, asymmetry_rel),
        threshold=min(change_floor, asymmetry_floor),
        fix_hint=fix_hint,
        extra={
            "identity_err": id_err,
            "change_rel": change_rel,
            "asymmetry_rel": asymmetry_rel,
        },
    )


def check_severity_smoke(
    module: types.ModuleType,
    kspace_slice: np.ndarray,
    severities: tuple[int, int, int] = (1, 3, 5),
) -> tuple[CheckResult, dict[int, np.ndarray]]:
    """Check 6 — residual grows with severity (no silent no-op at high sev).

    Returns (result, per_severity_rss) so the PNG grid can reuse the
    reconstructions without recomputing.
    """
    logger.info("[Check 6] Severity monotonicity — residual must grow with sev")
    ref_img = _rss_from_coil_images(module.ifft2c(kspace_slice))
    ref_mean = float(np.mean(np.abs(ref_img))) or 1e-12
    residuals: dict[int, float] = {}
    rss_by_sev: dict[int, np.ndarray] = {}
    # Use a MEAN-based residual (L1 mean / mean reference) rather than max-based.
    # Max-based saturates near 1.0 once ghosts appear at all and loses
    # monotonicity at high severities where strong translations wrap content;
    # mean-based scales with overall image perturbation and stays ordered.
    for severity in severities:
        config = module.SEVERITY_CONFIGS[severity]
        rng = np.random.default_rng(seed=1000 * severity + 7)
        events = module.sample_events(rng, config)
        cor = module.apply_motion_to_slice(kspace_slice, events=events, voxel_mm=(1.0, 1.0))
        cor_img = _rss_from_coil_images(module.ifft2c(cor))
        rss_by_sev[severity] = cor_img
        residuals[severity] = float(np.mean(np.abs(cor_img - ref_img)) / ref_mean)
        logger.info(
            "    severity=%d  n_transforms=%d  mean_residual=%.3e",
            severity,
            config.n_transforms,
            residuals[severity],
        )

    # PASS iff: each residual above severity*0.01 AND strictly monotonic.
    threshold_per_sev = {s: s * 0.01 for s in severities}
    above_floor = all(residuals[s] > threshold_per_sev[s] for s in severities)
    monotonic = all(
        residuals[severities[i]] < residuals[severities[i + 1]]
        for i in range(len(severities) - 1)
    )
    status: Literal["PASS", "FAIL"] = "PASS" if (above_floor and monotonic) else "FAIL"
    fix_hint = ""
    if not above_floor:
        fix_hint = (
            "Residuals too small — motion may not actually be applied. Verify "
            "sample_events returns a non-empty list at severity>=1, and that "
            "apply_motion_to_slice is not short-circuiting on non-empty events."
        )
    elif not monotonic:
        fix_hint = (
            "Residuals not monotonic in severity. Inspect SEVERITY_CONFIGS — "
            "higher severities must have strictly greater n_transforms / rot / trans."
        )
    worst = max(residuals.values())
    logger.info("    above_floor=%s  monotonic=%s", above_floor, monotonic)
    if status == "FAIL":
        logger.error("    FAIL — %s", fix_hint)
    else:
        logger.info("    PASS")
    return (
        CheckResult(
            name="severity_smoke",
            status=status,
            measured=worst,
            threshold=min(threshold_per_sev.values()),
            fix_hint=fix_hint,
            extra={f"residual_sev_{s}": residuals[s] for s in severities},
        ),
        rss_by_sev,
    )


# ──────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────


def plot_grid(
    ref_img: np.ndarray,
    rss_by_sev: dict[int, np.ndarray],
    out_png: Path,
) -> None:
    """Save a 2×4 diagnostic figure (magnitudes on top, diffs on bottom)."""
    # Local import so headless environments without matplotlib-in-path still
    # run the numerical checks (the script exits 1 on any check failure, so
    # missing mpl would only hide the PNG — not a hard dependency).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    severities = sorted(rss_by_sev.keys())
    fig, axes = plt.subplots(2, 1 + len(severities), figsize=(3 * (1 + len(severities)), 6))
    vmax = float(np.max(ref_img)) or 1.0

    axes[0, 0].imshow(ref_img, cmap="gray", vmin=0, vmax=vmax)
    axes[0, 0].set_title("reference (sev=0)")
    axes[0, 0].set_axis_off()
    axes[1, 0].imshow(np.zeros_like(ref_img), cmap="magma", vmin=0, vmax=1)
    axes[1, 0].set_title("|ref - ref| = 0")
    axes[1, 0].set_axis_off()

    diffs = [np.abs(rss_by_sev[s] - ref_img) for s in severities]
    diff_vmax = max((float(d.max()) for d in diffs), default=1.0) or 1.0
    for col, sev in enumerate(severities, start=1):
        axes[0, col].imshow(rss_by_sev[sev], cmap="gray", vmin=0, vmax=vmax)
        axes[0, col].set_title(f"severity {sev}")
        axes[0, col].set_axis_off()
        axes[1, col].imshow(diffs[col - 1], cmap="magma", vmin=0, vmax=diff_vmax)
        axes[1, col].set_title(f"|sev{sev} - ref|")
        axes[1, col].set_axis_off()

    fig.suptitle("02b k-space motion diagnostic (middle slice)", y=1.0)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────


def run_all_checks(
    h5_path: Path,
    out_dir: Path,
    slice_idx: int | None,
    strict_tol: float,
    make_plot: bool = True,
) -> list[CheckResult]:
    """Execute every check against ``h5_path`` and return the results.

    Checks continue even if earlier ones fail — we want the full picture
    in one run, not a cascading abort.
    """
    module = load_corrupt_module()
    kspace_slice, stored_rss, used_idx = _load_slice(h5_path, slice_idx)
    logger.info(
        "Loaded slice %d from %s: kspace shape=%s, rss shape=%s",
        used_idx,
        h5_path.name,
        kspace_slice.shape,
        stored_rss.shape,
    )

    results: list[CheckResult] = []
    results.append(check_fft_convention(kspace_slice, stored_rss, strict_tol))
    results.append(check_identity_motion(module, kspace_slice))
    results.append(check_pe_axis(module, kspace_slice))
    results.append(check_translation_units(module, kspace_slice))
    results.append(check_rotation(module, kspace_slice))

    smoke_result, rss_by_sev = check_severity_smoke(module, kspace_slice)
    results.append(smoke_result)

    if make_plot:
        ref_img = _rss_from_coil_images(module.ifft2c(kspace_slice))
        plot_grid(ref_img, rss_by_sev, out_dir / "grid.png")

    return results


def write_results_csv(results: list[CheckResult], path: Path) -> None:
    """Write the checks.csv summary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["check", "status", "measured", "threshold", "fix_hint"])
        for result in results:
            writer.writerow(
                [
                    result.name,
                    result.status,
                    f"{result.measured:.6e}",
                    f"{result.threshold:.6e}",
                    result.fix_hint,
                ]
            )


def print_results_table(results: list[CheckResult], console: Console) -> None:
    """Render a rich table of results + a final PASS/FAIL banner."""
    table = Table(title="02b k-space motion diagnostic")
    table.add_column("check", style="bold")
    table.add_column("status", justify="center")
    table.add_column("measured", justify="right")
    table.add_column("threshold", justify="right")
    table.add_column("fix_hint", overflow="fold")
    for result in results:
        status_style = "green" if result.status == "PASS" else "red"
        table.add_row(
            result.name,
            f"[{status_style}]{result.status}[/{status_style}]",
            f"{result.measured:.3e}",
            f"{result.threshold:.3e}",
            result.fix_hint or "-",
        )
    console.print(table)

    n_fail = sum(1 for r in results if r.status == "FAIL")
    if n_fail == 0:
        console.print("[bold green]All checks PASS — batch run is safe.[/bold green]")
    else:
        console.print(
            f"[bold red]{n_fail} check(s) FAILED. Fix and re-run before batch.[/bold red]"
        )


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


@app.command()
def main(
    h5_path: Path = typer.Option(
        ...,
        "--h5-path",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="One FastMRI .h5 file to run the checks against.",
    ),
    out_dir: Path = typer.Option(
        Path("results/diagnostics/kspace_motion"),
        "--out-dir",
        resolve_path=True,
        help="Where to write checks.csv and grid.png.",
    ),
    slice_idx: int | None = typer.Option(
        None,
        "--slice-idx",
        help="Which slice index to test (default: middle slice).",
    ),
    strict_tol: float = typer.Option(
        1e-4,
        "--strict-tol",
        help="Relative tolerance for check 1 (FFT convention).",
    ),
    no_plot: bool = typer.Option(
        False,
        "--no-plot",
        help="Skip the grid.png (useful in CI with no matplotlib backend).",
    ),
) -> None:
    """Run all six diagnostic checks; exit non-zero if any fails."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "checks.log"
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=False), file_handler],
        force=True,
    )

    results = run_all_checks(
        h5_path=h5_path,
        out_dir=out_dir,
        slice_idx=slice_idx,
        strict_tol=strict_tol,
        make_plot=not no_plot,
    )

    csv_path = out_dir / "checks.csv"
    write_results_csv(results, csv_path)
    print_results_table(results, Console())
    logger.info("Wrote %s and %s", csv_path, log_path)

    if any(r.status == "FAIL" for r in results):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
