#!/usr/bin/env bash
#
# scripts/runpod_stage_data.sh — copy data from /workspace (MooseFS) to /root
# (local NVMe) and rescale FastMRI normalised magnitudes in place.
#
# Why this script exists:
#   /workspace is MooseFS (~50-200 MB/s, ms latency).  /root is the container
#   overlay on local NVMe (~3-7 GB/s, µs latency).  SynthSeg --parc does many
#   small reads + writes per scan, so running off /root is ~10× faster.  This
#   script copies the prototype-relevant subset of data once and reapplies the
#   FastMRI intensity rescale (Phase 0 might have been skipped if NIfTIs were
#   pre-extracted on the dev box without `--rescale-intensity`).
#
# Trade-off: /root is wiped on pod *Terminate* (not on Stop).  Re-run this
# script after a fresh pod launch.
#
# Usage:
#   bash scripts/runpod_stage_data.sh                # default subsets
#   STAGE_FULL=1 bash scripts/runpod_stage_data.sh   # copy entire datasets

set -euo pipefail

WORK_DIR="${WORK_DIR:-/workspace}"
NEUROQC_DIR="${NEUROQC_DIR:-${WORK_DIR}/NeuroQC}"
SRC_DATA="${NEUROQC_DIR}/data"
DST_DATA="${DST_DATA:-/root/data}"
STAGE_FULL="${STAGE_FULL:-0}"
RESCALE_FASTMRI="${RESCALE_FASTMRI:-1}"

log() { printf '[runpod_stage_data %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

if [ ! -d "${SRC_DATA}" ]; then
    log "ERROR: ${SRC_DATA} not found. Did gpu_server_setup.sh run first?"
    exit 1
fi

mkdir -p "${DST_DATA}/ixi/raw" "${DST_DATA}/fastmri/nifti" "${DST_DATA}/abide"

log "Staging IXI raw -> ${DST_DATA}/ixi/raw"
if [ "${STAGE_FULL}" = "1" ]; then
    cp -n "${SRC_DATA}"/ixi/raw/*.nii.gz "${DST_DATA}/ixi/raw/" 2>/dev/null || true
else
    # Default: first 30 scans alphabetically — gives ~22 passing post-QC,
    # comfortable headroom over the 15-ref target.
    ls "${SRC_DATA}"/ixi/raw/*.nii.gz 2>/dev/null | sort | head -30 | \
        xargs -I{} cp -n {} "${DST_DATA}/ixi/raw/" 2>/dev/null || true
fi
log "  $(ls "${DST_DATA}/ixi/raw"/*.nii.gz 2>/dev/null | wc -l | tr -d ' ') IXI NIfTI files staged"

log "Staging FastMRI NIfTI -> ${DST_DATA}/fastmri/nifti"
if [ "${STAGE_FULL}" = "1" ]; then
    cp -n "${SRC_DATA}"/fastmri/nifti/*.nii.gz "${DST_DATA}/fastmri/nifti/" 2>/dev/null || true
else
    ls "${SRC_DATA}"/fastmri/nifti/*.nii.gz 2>/dev/null | sort | head -25 | \
        xargs -I{} cp -n {} "${DST_DATA}/fastmri/nifti/" 2>/dev/null || true
fi
log "  $(ls "${DST_DATA}/fastmri/nifti"/*.nii.gz 2>/dev/null | wc -l | tr -d ' ') FastMRI NIfTI files staged"

log "Staging ABIDE -> ${DST_DATA}/abide"
if [ "${STAGE_FULL}" = "1" ]; then
    cp -n "${SRC_DATA}"/abide/*.nii.gz "${DST_DATA}/abide/" 2>/dev/null || true
else
    ls "${SRC_DATA}"/abide/*.nii.gz 2>/dev/null | sort | head -30 | \
        xargs -I{} cp -n {} "${DST_DATA}/abide/" 2>/dev/null || true
fi
log "  $(ls "${DST_DATA}/abide"/*.nii.gz 2>/dev/null | wc -l | tr -d ' ') ABIDE NIfTI files staged"

# FastMRI intensity rescale — idempotent (skips files whose max already > 1.0).
# Same logic as code/00_extract_fastmri_t1.py's --rescale-intensity, applied
# in-place to NIfTIs that were extracted before that flag was added.
if [ "${RESCALE_FASTMRI}" = "1" ]; then
    log "Rescaling FastMRI intensities (idempotent)"
    python - <<'PY'
import nibabel as nib, numpy as np
from pathlib import Path
ROOT = Path("/root/data/fastmri/nifti")
for fp in sorted(ROOT.glob("*.nii.gz")):
    img = nib.load(str(fp))
    arr = np.asanyarray(img.dataobj).astype(np.float32)
    if arr.max() < 1.0:
        scaled = arr * 1e6
        nib.save(nib.Nifti1Image(scaled, img.affine, img.header), str(fp))
        print(f"rescaled {fp.name}: {arr.max():.3e} -> {scaled.max():.1f}")
    else:
        print(f"skipped {fp.name} (max={arr.max():.1f})")
PY
fi

log "Done. Inputs ready at ${DST_DATA}/{ixi,fastmri,abide}/"
log "Re-run with STAGE_FULL=1 to copy the entire dataset (slow but reproducible)."
