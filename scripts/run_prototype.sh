#!/usr/bin/env bash
#
# scripts/run_prototype.sh — DANDI Hub calibration prototype (CPU-only).
#
# Runs Phases 00 → 08a/A on a small subsample (default 5 IXI + 5 FastMRI
# references, motion-only corruptions at severities 1/3/5) so the calibration
# CSVs (machine_preference, cortical_thickness, per_structure_dice,
# iqm_features, benchmark_subsample) are produced on real data before the
# Jarvis A100 full run. The bucket boundaries in
# code/09_finetune_lora.py:DICE_BUCKET_THRESHOLDS / THICKNESS_BUCKET_THRESHOLDS
# are calibrated from the resulting distributions.
#
# Wall-clock estimate at NUM_REFS=5: ~16-21 hr on a single CPU core
# (Phase 03 SynthSeg --parc dominates at ~18 min/scan).
#
# Usage on DANDI Hub (after scripts/setup_dandi.sh has succeeded):
#     bash scripts/run_prototype.sh
#
# Override the scope via env vars (small-smoke first run is recommended):
#     NUM_REFS=1 SEVERITIES=3 bash scripts/run_prototype.sh   # ~30 min smoke
#     NUM_REFS=5 SEVERITIES=1,3,5 bash scripts/run_prototype.sh  # full calibration
#
# Resumes automatically on rerun: every phase script honours append-mode and
# skips inputs whose outputs already exist. To force a full rerun, pass
# FORCE=1 (deletes phase-marker touch-files only — does not delete CSVs).
#
# Outputs land in:
#     results/tables/   — CSVs
#     data/<ds>/references_proto/, data/<ds>/corrupted_proto/
#     data/derivatives/synthseg/
#     results/logs/dandi_prototype_<timestamp>.log

set -euo pipefail

# ── config (override via env vars) ──
NUM_REFS="${NUM_REFS:-5}"
SEVERITIES="${SEVERITIES:-1,3,5}"
CORRUPTIONS="${CORRUPTIONS:-motion}"
FORCE="${FORCE:-0}"
SKIP_FREESURFER_SETUP="${SKIP_FREESURFER_SETUP:-0}"
# Phase 02b operates on raw FastMRI H5 (~1.5 GB/file). On DANDI Hub the H5
# corpus is usually absent — keep this off unless the H5s were copied in.
SKIP_PHASE2B="${SKIP_PHASE2B:-1}"

PROJECT_ROOT="$(cd "$(/usr/bin/dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

TS="$(/bin/date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/results/logs"
LOG_FILE="${LOG_DIR}/dandi_prototype_${TS}.log"
PHASES_DIR="${PROJECT_ROOT}/.phases_proto"
/bin/mkdir -p "${LOG_DIR}" "${PHASES_DIR}" "${PROJECT_ROOT}/results/tables"

# Tee everything to the log file from this point on.
exec > >(tee -a "${LOG_FILE}") 2>&1

log() { printf '\n[run_prototype %(%H:%M:%S)T] %s\n' -1 "$*"; }
section() { printf '\n========================================\n%s\n========================================\n' "$*"; }

section "NeuroQC DANDI Hub prototype"
log "PROJECT_ROOT=${PROJECT_ROOT}"
log "NUM_REFS=${NUM_REFS} SEVERITIES=${SEVERITIES} CORRUPTIONS=${CORRUPTIONS} FORCE=${FORCE}"
log "Logging to: ${LOG_FILE}"

if [ "${FORCE}" = "1" ]; then
    log "FORCE=1 — clearing phase markers (CSV outputs are kept; phase scripts will resume from append state)."
    /bin/rm -f "${PHASES_DIR}"/.phase*
fi

# ── 0. Environment: FreeSurfer + Python ──
section "0. Environment setup"
if [ "${SKIP_FREESURFER_SETUP}" != "1" ]; then
    if ! command -v mri_synthseg >/dev/null 2>&1; then
        log "FreeSurfer not on PATH; running scripts/setup_dandi.sh"
        bash "${PROJECT_ROOT}/scripts/setup_dandi.sh"
    else
        log "FreeSurfer already on PATH at $(command -v mri_synthseg)"
    fi
fi

# Use whatever python is on PATH (DANDI Hub conda env, or local venv).
PYBIN="${PYBIN:-$(command -v python || true)}"
if [ -z "${PYBIN}" ]; then
    log "ERROR: no python on PATH. Activate your DANDI Hub conda env first."
    exit 2
fi
log "Python: ${PYBIN} ($(${PYBIN} --version 2>&1))"

# Sanity-check the venv has nobrainer + monai + torchio etc.
${PYBIN} -c "import nobrainer, monai, torchio, torch, nibabel; print('deps ok')" \
    || { log "ERROR: missing Python deps. Install via 'pip install -e .' and 'pip install -e /path/to/nobrainer'"; exit 3; }

# ── 1. Phase 00 — FastMRI HDF5 → NIfTI (skip if already extracted) ──
section "Phase 00 — FastMRI extraction (T1w only)"
FASTMRI_NIFTI_DIR="${PROJECT_ROOT}/data/fastmri/nifti"
FASTMRI_NIFTI_COUNT="$(/usr/bin/find "${FASTMRI_NIFTI_DIR}" -maxdepth 1 -name '*.nii.gz' 2>/dev/null | /usr/bin/wc -l | /usr/bin/tr -d ' ')"
if [ "${FASTMRI_NIFTI_COUNT}" -lt "${NUM_REFS}" ]; then
    log "Only ${FASTMRI_NIFTI_COUNT} FastMRI NIfTIs present; need ${NUM_REFS}. Running Phase 00."
    ${PYBIN} code/00_extract_fastmri_t1.py \
        --input-dir data/fastmri/raw \
        --output-dir data/fastmri/nifti \
        --manifest-csv results/tables/fastmri_extraction_manifest.csv
else
    log "Phase 00 skipped: ${FASTMRI_NIFTI_COUNT} NIfTI(s) already extracted."
fi

# ── 2. Phase 01 — curate references (per-dataset, full sweep, fast) ──
section "Phase 01 — curate IXI + FastMRI references"
if [ ! -f "${PHASES_DIR}/.phase1_ixi" ]; then
    ${PYBIN} code/01_curate_references.py \
        --input-dir data/ixi/raw \
        --output-dir data/ixi/references \
        --manifest-path results/tables/reference_manifest_ixi.csv \
        --dataset-tag ixi \
        --force
    /usr/bin/touch "${PHASES_DIR}/.phase1_ixi"
fi
if [ ! -f "${PHASES_DIR}/.phase1_fastmri" ]; then
    ${PYBIN} code/01_curate_references.py \
        --input-dir data/fastmri/nifti \
        --output-dir data/fastmri/references \
        --manifest-path results/tables/reference_manifest_fastmri.csv \
        --dataset-tag fastmri \
        --force
    /usr/bin/touch "${PHASES_DIR}/.phase1_fastmri"
fi

# ── 3. Subsample to NUM_REFS per dataset (deterministic, sorted) ──
section "Subsample → ${NUM_REFS} refs per dataset"
${PYBIN} - <<PYSUBSAMPLE
import shutil
from pathlib import Path
import pandas as pd

NUM_REFS = ${NUM_REFS}
ROOT = Path("${PROJECT_ROOT}")
TABLES = ROOT / "results" / "tables"

for ds in ("ixi", "fastmri"):
    manifest = TABLES / f"reference_manifest_{ds}.csv"
    df = pd.read_csv(manifest)
    df = df[df["passed_qc"] == True].sort_values("subject_id").head(NUM_REFS)
    if len(df) < NUM_REFS:
        raise SystemExit(
            f"[subsample] {ds}: only {len(df)} refs passed QC, need {NUM_REFS}. "
            "Either lower NUM_REFS or extract more raw scans."
        )
    out_csv = TABLES / f"reference_manifest_{ds}_proto.csv"
    df.to_csv(out_csv, index=False)
    print(f"[subsample] {ds}: kept {len(df)} → {out_csv}")

    proto_dir = ROOT / "data" / ds / "references_proto"
    proto_dir.mkdir(parents=True, exist_ok=True)
    for fp in df["filepath"]:
        src = Path(fp)
        if not src.is_absolute():
            src = ROOT / src
        dst = proto_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
    print(f"[subsample] {ds}: symlinked {len(df)} refs into {proto_dir}")
PYSUBSAMPLE

# ── 4. Phase 02 — image-space motion corruption on the prototype refs ──
section "Phase 02 — image-space motion corruption (sev=${SEVERITIES})"
if [ ! -f "${PHASES_DIR}/.phase2_ixi" ]; then
    ${PYBIN} code/02_generate_corruptions.py \
        --input-dir data/ixi/references_proto \
        --output-dir data/ixi/corrupted_proto \
        --corruptions "${CORRUPTIONS}" \
        --severities "${SEVERITIES}" \
        --dataset-tag ixi \
        --manifest-path results/tables/corruption_manifest.csv
    /usr/bin/touch "${PHASES_DIR}/.phase2_ixi"
fi
if [ ! -f "${PHASES_DIR}/.phase2_fastmri" ]; then
    ${PYBIN} code/02_generate_corruptions.py \
        --input-dir data/fastmri/references_proto \
        --output-dir data/fastmri/corrupted_proto \
        --corruptions "${CORRUPTIONS}" \
        --severities "${SEVERITIES}" \
        --dataset-tag fastmri \
        --manifest-path results/tables/corruption_manifest.csv
    /usr/bin/touch "${PHASES_DIR}/.phase2_fastmri"
fi

# ── 5. Phase 02b — k-space motion corruption (FastMRI only, matched H5s) ──
section "Phase 02b — k-space motion corruption (FastMRI, sev=${SEVERITIES})"
if [ "${SKIP_PHASE2B}" = "1" ]; then
    log "SKIP_PHASE2B=1 — skipping k-space corruption (set SKIP_PHASE2B=0 to enable)."
elif [ ! -f "${PHASES_DIR}/.phase2b_fastmri" ]; then
    # Stage the H5 files matching the prototype refs into a small subdir.
    ${PYBIN} - <<PYH5
from pathlib import Path
import pandas as pd

ROOT = Path("${PROJECT_ROOT}")
TABLES = ROOT / "results" / "tables"
df = pd.read_csv(TABLES / "reference_manifest_fastmri_proto.csv")

raw_dir = ROOT / "data" / "fastmri" / "raw"
proto_h5 = ROOT / "data" / "fastmri" / "raw_proto"
proto_h5.mkdir(parents=True, exist_ok=True)

picked = 0
for fp in df["filepath"]:
    stem = Path(fp).name.replace(".nii.gz", "")
    src = raw_dir / f"{stem}.h5"
    if not src.exists():
        print(f"[h5-stage] WARNING: H5 missing for {stem} (looked at {src})")
        continue
    dst = proto_h5 / src.name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())
    picked += 1
print(f"[h5-stage] symlinked {picked} H5 files into {proto_h5}")
PYH5

    ${PYBIN} code/02b_corrupt_kspace_motion.py \
        --input-dir data/fastmri/raw_proto \
        --reference-dir data/fastmri/references_proto \
        --output-dir data/fastmri/corrupted_kspace_proto \
        --severities "${SEVERITIES}" \
        --manifest-path results/tables/corruption_manifest.csv
    /usr/bin/touch "${PHASES_DIR}/.phase2b_fastmri"
fi

# ── 6. Phase 03 — SynthSeg --parc on refs + corrupted ──
section "Phase 03 — SynthSeg (--parc) on refs + corrupted"
SYNTHSEG_OUT="${PROJECT_ROOT}/data/derivatives/synthseg"
SYNTHSEG_MODE="${SYNTHSEG_MODE:-freesurfer}"
for DS in ixi fastmri; do
    if [ ! -f "${PHASES_DIR}/.phase3_${DS}_refs" ]; then
        ${PYBIN} code/03_run_synthseg.py \
            --input-dir "data/${DS}/references_proto" \
            --output-dir "${SYNTHSEG_OUT}/${DS}" \
            --mode "${SYNTHSEG_MODE}" \
            --parc \
            --manifest-path "results/tables/synthseg_${DS}_manifest.csv"
        /usr/bin/touch "${PHASES_DIR}/.phase3_${DS}_refs"
    fi
    if [ ! -f "${PHASES_DIR}/.phase3_${DS}_cor" ]; then
        ${PYBIN} code/03_run_synthseg.py \
            --input-dir "data/${DS}/corrupted_proto" \
            --output-dir "${SYNTHSEG_OUT}/${DS}" \
            --mode "${SYNTHSEG_MODE}" \
            --parc \
            --manifest-path "results/tables/synthseg_${DS}_manifest.csv"
        /usr/bin/touch "${PHASES_DIR}/.phase3_${DS}_cor"
    fi
done
if [ -d "${PROJECT_ROOT}/data/fastmri/corrupted_kspace_proto" ] && [ ! -f "${PHASES_DIR}/.phase3_fastmri_kspace" ]; then
    ${PYBIN} code/03_run_synthseg.py \
        --input-dir data/fastmri/corrupted_kspace_proto \
        --output-dir "${SYNTHSEG_OUT}/fastmri" \
        --mode "${SYNTHSEG_MODE}" \
        --parc \
        --manifest-path "results/tables/synthseg_fastmri_manifest.csv"
    /usr/bin/touch "${PHASES_DIR}/.phase3_fastmri_kspace"
fi

# ── 7. Phase 03b — cortical thickness from --parc segs ──
section "Phase 03b — cortical thickness"
if [ ! -f "${PHASES_DIR}/.phase3b" ]; then
    ${PYBIN} code/03b_compute_thickness.py \
        --synthseg-dir "${SYNTHSEG_OUT}" \
        --output-file results/tables/cortical_thickness.csv
    /usr/bin/touch "${PHASES_DIR}/.phase3b"
fi

# ── 8. Phase 04 — machine preference (Dice + thickness) ──
section "Phase 04 — machine preference + per-structure Dice"
if [ ! -f "${PHASES_DIR}/.phase4" ]; then
    ${PYBIN} code/04_compute_preference.py \
        --corruption-manifest results/tables/corruption_manifest.csv \
        --synthseg-dir "${SYNTHSEG_OUT}" \
        --thickness-file results/tables/cortical_thickness.csv \
        --output-file results/tables/machine_preference.csv \
        --per-structure-output results/tables/per_structure_dice.csv
    /usr/bin/touch "${PHASES_DIR}/.phase4"
fi

# ── 9. Phase 05 — IQM extraction ──
section "Phase 05 — extract IQMs"
if [ ! -f "${PHASES_DIR}/.phase5" ]; then
    ${PYBIN} code/05_extract_iqms.py \
        --ref-manifest results/tables/reference_manifest_ixi_proto.csv \
        --ref-manifest results/tables/reference_manifest_fastmri_proto.csv \
        --cor-manifest results/tables/corruption_manifest.csv \
        --synthseg-manifest results/tables/synthseg_ixi_manifest.csv \
        --synthseg-manifest results/tables/synthseg_fastmri_manifest.csv \
        --output-file results/tables/iqm_features.csv
    /usr/bin/touch "${PHASES_DIR}/.phase5"
fi

# ── 10. Phase 05b — SynthSeg internal-QC aggregation ──
section "Phase 05b — aggregate SynthSeg QC"
if [ ! -f "${PHASES_DIR}/.phase5b" ]; then
    ${PYBIN} code/05b_aggregate_synthseg_qc.py \
        --synthseg-manifest results/tables/synthseg_ixi_manifest.csv \
        --synthseg-manifest results/tables/synthseg_fastmri_manifest.csv \
        --cor-manifest results/tables/corruption_manifest.csv \
        --output-file results/tables/synthseg_qc_features.csv
    /usr/bin/touch "${PHASES_DIR}/.phase5b"
fi

# ── 11. Phase 08a/A — build benchmark_subsample.csv (no VLM inference) ──
section "Phase 08a/A — build benchmark_subsample.csv (--dry-run, no GPU)"
if [ ! -f "${PHASES_DIR}/.phase8a_A" ]; then
    ${PYBIN} code/08a_eval_3d_vlms.py \
        --seed 42 \
        --dry-run \
        --ref-manifest results/tables/reference_manifest_ixi_proto.csv \
        --ref-manifest results/tables/reference_manifest_fastmri_proto.csv \
        --cor-manifest results/tables/corruption_manifest.csv \
        --preference-csv results/tables/machine_preference.csv \
        --n-refs "${NUM_REFS}" \
        --severities "${SEVERITIES}" \
        --corruption-types "${CORRUPTIONS}" \
        --subsample-manifest results/tables/benchmark_subsample.csv
    /usr/bin/touch "${PHASES_DIR}/.phase8a_A"
fi

# ── 12. Wrap up: results tracker + datalad save ──
section "Wrap up"
${PYBIN} code/results_tracker.py --phase 3 || log "results_tracker phase 3 failed (non-fatal)"

if command -v datalad >/dev/null 2>&1; then
    log "datalad save (best-effort, non-fatal if no .datalad)"
    PATH=/usr/bin:/bin:/usr/local/bin datalad save \
        -m "DANDI Hub prototype run ${TS} (NUM_REFS=${NUM_REFS} CORRUPTIONS=${CORRUPTIONS} SEVERITIES=${SEVERITIES})" \
        results/ data/ 2>&1 || log "datalad save reported issues (often expected on first run)"
fi

# Bundle the result CSVs for download from DANDI Hub.
BUNDLE="${PROJECT_ROOT}/results/dandi_prototype_${TS}.tar.gz"
/usr/bin/tar -czf "${BUNDLE}" \
    -C "${PROJECT_ROOT}" \
    results/tables/ results/logs/ 2>/dev/null || true
log "Bundle: ${BUNDLE}"

section "Prototype complete: $(date)"
log "Key outputs:"
log "  results/tables/machine_preference.csv"
log "  results/tables/per_structure_dice.csv"
log "  results/tables/cortical_thickness.csv"
log "  results/tables/iqm_features.csv"
log "  results/tables/benchmark_subsample.csv"
log "Next: download the bundle, inspect distributions, calibrate code/09 bucket boundaries."
