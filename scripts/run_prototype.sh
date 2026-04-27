#!/usr/bin/env bash
#
# scripts/run_prototype.sh — DANDI Hub combined calibration + ABIDE smoke run.
#
# Single orchestrator that runs both purposes end-to-end:
#
#   * IXI / FastMRI calibration: 5 refs/dataset × motion × sev 1/3/5
#     → 30 corrupted scans for bucket-boundary calibration of
#       code/09_finetune_lora.py:DICE_BUCKET_THRESHOLDS / THICKNESS_BUCKET_THRESHOLDS.
#   * ABIDE end-to-end smoke: 3 refs × {motion, ghosting, spike} × sev 1/5
#     → 18 corrupted scans for pipeline E2E verification before Jarvis dispatch.
#
# Total: 13 refs, 61 SynthSeg --parc runs.
# Wall-clock target: ~30 min on DANDI Hub T4 GPU image (default
# SYNTHSEG_MODE=python); ~18 hr on CPU fallback (SYNTHSEG_MODE=freesurfer).
#
# Usage on DANDI Hub T4 image:
#     bash scripts/run_prototype.sh
#
# Fastest E2E smoke (skip calibration; one ABIDE scan, one corruption, one severity):
#     INCLUDE_IXI=0 INCLUDE_FASTMRI=0 \
#     NUM_REFS_SMOKE=1 SMOKE_CORRUPTIONS=motion SMOKE_SEVERITIES=1 \
#         bash scripts/run_prototype.sh
#
# Local Mac smoke (CPU FreeSurfer, no datalad provenance):
#     INCLUDE_IXI=0 INCLUDE_FASTMRI=0 \
#     NUM_REFS_SMOKE=1 SMOKE_CORRUPTIONS=motion SMOKE_SEVERITIES=1 \
#     SYNTHSEG_MODE=freesurfer USE_DATALAD_RUN=0 \
#         bash scripts/run_prototype.sh
#
# Resumes automatically: each phase has a `.phases_proto/.phaseN_*` marker;
# rerunning the script skips already-done phases. Pass FORCE=1 to clear.

set -euo pipefail

# ── config (override via env vars) ─────────────────────────────────────────
INCLUDE_IXI="${INCLUDE_IXI:-1}"
INCLUDE_FASTMRI="${INCLUDE_FASTMRI:-1}"
INCLUDE_ABIDE="${INCLUDE_ABIDE:-1}"

NUM_REFS_CALIB="${NUM_REFS_CALIB:-5}"
NUM_REFS_SMOKE="${NUM_REFS_SMOKE:-3}"

CALIB_CORRUPTIONS="${CALIB_CORRUPTIONS:-motion}"
CALIB_SEVERITIES="${CALIB_SEVERITIES:-1,3,5}"
SMOKE_CORRUPTIONS="${SMOKE_CORRUPTIONS:-motion,ghosting,spike}"
SMOKE_SEVERITIES="${SMOKE_SEVERITIES:-1,5}"

SYNTHSEG_MODE="${SYNTHSEG_MODE:-python}"   # "python" (T4 GPU) | "freesurfer" (CPU)
SYNTHSEG_FAST="${SYNTHSEG_FAST:-0}"        # pass --fast to mri_synthseg (~3x speedup, slightly lower accuracy)
SKIP_PHASE2B="${SKIP_PHASE2B:-1}"          # FastMRI k-space corruption (needs H5 corpus)
USE_DATALAD_RUN="${USE_DATALAD_RUN:-1}"    # wrap each phase in datalad run
FORCE="${FORCE:-0}"
SKIP_FREESURFER_SETUP="${SKIP_FREESURFER_SETUP:-0}"

PROJECT_ROOT="$(cd "$(/usr/bin/dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

TS="$(/bin/date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/results/logs"
LOG_FILE="${LOG_DIR}/dandi_combined_${TS}.log"
PHASES_DIR="${PROJECT_ROOT}/.phases_proto"
/bin/mkdir -p "${LOG_DIR}" "${PHASES_DIR}" "${PROJECT_ROOT}/results/tables"

exec > >(/usr/bin/tee -a "${LOG_FILE}") 2>&1

log()     { printf '\n[run_prototype %s] %s\n' "$(/bin/date +%H:%M:%S)" "$*"; }
section() { printf '\n========================================\n%s\n========================================\n' "$*"; }

section "NeuroQC DANDI Hub combined run"
log "PROJECT_ROOT=${PROJECT_ROOT}"
log "INCLUDE_IXI=${INCLUDE_IXI} INCLUDE_FASTMRI=${INCLUDE_FASTMRI} INCLUDE_ABIDE=${INCLUDE_ABIDE}"
log "calibration: ${NUM_REFS_CALIB} refs/ds × ${CALIB_CORRUPTIONS} × sev ${CALIB_SEVERITIES}"
log "smoke (ABIDE): ${NUM_REFS_SMOKE} refs × ${SMOKE_CORRUPTIONS} × sev ${SMOKE_SEVERITIES}"
log "SYNTHSEG_MODE=${SYNTHSEG_MODE}  USE_DATALAD_RUN=${USE_DATALAD_RUN}  FORCE=${FORCE}"
log "log: ${LOG_FILE}"

if [ "${FORCE}" = "1" ]; then
    log "FORCE=1 — clearing phase markers."
    /bin/rm -f "${PHASES_DIR}"/.phase*
fi

# ── dl_run helper ──────────────────────────────────────────────────────────
# Usage: dl_run "<commit message>" -i <pat> [-i <pat> ...] -o <pat> [-o <pat> ...] -- <command-string>
# When USE_DATALAD_RUN=1 wraps the command in `datalad run`; otherwise eval's it.
dl_run() {
    local msg="$1"; shift
    local args=()
    while [ $# -gt 0 ] && [ "${1:-}" != "--" ]; do
        args+=("$1")
        shift
    done
    if [ "${1:-}" != "--" ]; then
        echo "[dl_run] ERROR: missing '--' separator before command in '${msg}'" >&2
        return 2
    fi
    shift  # consume --
    local cmd="$*"
    if [ "${USE_DATALAD_RUN}" = "1" ]; then
        # Ensure git is on PATH (datalad shells out to it).
        PATH="/usr/bin:/bin:${PATH:-}" datalad run -m "${msg}" "${args[@]}" "${cmd}"
    else
        eval "${cmd}"
    fi
}

# ── Phase 0 — environment ──────────────────────────────────────────────────
section "Phase 0 — environment"
PYBIN="${PYBIN:-$(command -v python || true)}"
if [ -z "${PYBIN}" ]; then
    log "ERROR: no python on PATH. Activate your DANDI Hub conda env first."
    exit 2
fi
log "python: ${PYBIN} ($(${PYBIN} --version 2>&1))"

# Verify project deps.
${PYBIN} -c "import nobrainer, monai, torchio, torch, nibabel, pandas; print('deps ok')" \
    || { log "ERROR: missing Python deps. Install via 'pip install -e ../nobrainer' + project deps."; exit 3; }

if [ "${SYNTHSEG_MODE}" = "freesurfer" ]; then
    if [ "${SKIP_FREESURFER_SETUP}" != "1" ] && ! command -v mri_synthseg >/dev/null 2>&1; then
        # Auto-detect FREESURFER_HOME so we can `source` it in *this* shell
        # (running setup_dandi.sh as a subshell would not propagate the exports).
        if [ -z "${FREESURFER_HOME:-}" ] || ! [ -f "${FREESURFER_HOME}/SetUpFreeSurfer.sh" ]; then
            for base in /Applications/freesurfer /opt/freesurfer /usr/local/freesurfer "${HOME}/freesurfer"; do
                if [ -f "${base}/SetUpFreeSurfer.sh" ]; then
                    FREESURFER_HOME="${base}"
                    break
                fi
                for ver in "${base}"/*/SetUpFreeSurfer.sh; do
                    [ -f "${ver}" ] && FREESURFER_HOME="$(/usr/bin/dirname "${ver}")" && break 2
                done
            done
        fi
        if [ -z "${FREESURFER_HOME:-}" ]; then
            log "ERROR: FreeSurfer not found. Run scripts/install_freesurfer_linux.sh or set FREESURFER_HOME."
            exit 4
        fi
        export FREESURFER_HOME
        # FreeSurfer's setup references unbound vars and runs internal tests
        # whose non-zero returns trip `set -e`; toggle errexit + nounset around
        # the source.
        set +eu
        # shellcheck disable=SC1091
        source "${FREESURFER_HOME}/SetUpFreeSurfer.sh" > /dev/null
        set -eu
    fi
    command -v mri_synthseg >/dev/null 2>&1 \
        || { log "ERROR: mri_synthseg still not on PATH after FreeSurfer setup"; exit 4; }
    log "FREESURFER_HOME=${FREESURFER_HOME:-<unset>}; mri_synthseg=$(command -v mri_synthseg)"
elif [ "${SYNTHSEG_MODE}" = "python" ]; then
    ${PYBIN} -c "import tensorflow as tf; gpus=tf.config.list_physical_devices('GPU'); print(f'tf={tf.__version__} GPUs={gpus}'); assert gpus, 'no GPU detected — re-launch DANDI Hub with the T4 image, or set SYNTHSEG_MODE=freesurfer for CPU fallback'" \
        || { log "ERROR: TF GPU unavailable; either re-launch on T4 image or set SYNTHSEG_MODE=freesurfer."; exit 5; }
else
    log "ERROR: SYNTHSEG_MODE='${SYNTHSEG_MODE}' must be 'python' or 'freesurfer'"
    exit 6
fi

# ── Phase 00 — FastMRI HDF5 → NIfTI extraction ─────────────────────────────
section "Phase 00 — FastMRI extraction"
if [ "${INCLUDE_FASTMRI}" = "1" ]; then
    # Honour FASTMRI_INPUT_DIR for the count check too — operators who
    # pre-extract NIfTIs to a custom path (e.g. /root/data/fastmri/nifti
    # on RunPod) need Phase 00 to detect those and skip H5 extraction.
    FASTMRI_NIFTI_DIR="${FASTMRI_INPUT_DIR:-${PROJECT_ROOT}/data/fastmri/nifti}"
    if [ "${FASTMRI_NIFTI_DIR#/}" = "${FASTMRI_NIFTI_DIR}" ]; then
        # Relative path → resolve under PROJECT_ROOT.
        FASTMRI_NIFTI_DIR="${PROJECT_ROOT}/${FASTMRI_NIFTI_DIR}"
    fi
    FASTMRI_NIFTI_COUNT="$(/usr/bin/find "${FASTMRI_NIFTI_DIR}" -maxdepth 1 -name '*.nii.gz' 2>/dev/null | /usr/bin/wc -l | /usr/bin/tr -d ' ')"
    if [ "${FASTMRI_NIFTI_COUNT}" -lt "${NUM_REFS_CALIB}" ] && [ ! -f "${PHASES_DIR}/.phase00" ]; then
        dl_run "Phase 00: Extract FastMRI T1w" \
            -i "data/fastmri/raw/*.h5" \
            -o "data/fastmri/nifti/" \
            -o "results/tables/fastmri_extraction_manifest.csv" \
            -- "${PYBIN} code/00_extract_fastmri_t1.py \
                --input-dir data/fastmri/raw \
                --output-dir data/fastmri/nifti \
                --manifest-csv results/tables/fastmri_extraction_manifest.csv"
        /usr/bin/touch "${PHASES_DIR}/.phase00"
    else
        log "Phase 00 skipped (${FASTMRI_NIFTI_COUNT} NIfTIs at ${FASTMRI_NIFTI_DIR}, need ${NUM_REFS_CALIB})."
    fi
else
    log "Phase 00 skipped (INCLUDE_FASTMRI=0)."
fi

# ── Phase 01 — curate references per dataset ───────────────────────────────
section "Phase 01 — curate references"

curate_ds() {
    local ds="$1" input_dir="$2" tag="$3"
    local marker="${PHASES_DIR}/.phase01_${ds}"
    [ -f "${marker}" ] && { log "Phase 01 ${ds}: already done."; return 0; }
    dl_run "Phase 01: Curate ${ds} references" \
        -i "${input_dir}/*.nii.gz" \
        -o "data/${ds}/references/" \
        -o "results/tables/reference_manifest_${ds}.csv" \
        -- "${PYBIN} code/01_curate_references.py \
            --input-dir ${input_dir} \
            --output-dir data/${ds}/references \
            --manifest-path results/tables/reference_manifest_${ds}.csv \
            --dataset-tag ${tag} --force"
    /usr/bin/touch "${marker}"
}

[ "${INCLUDE_IXI}" = "1" ]     && curate_ds ixi     "${IXI_INPUT_DIR:-data/ixi/raw}"           ixi
[ "${INCLUDE_FASTMRI}" = "1" ] && curate_ds fastmri "${FASTMRI_INPUT_DIR:-data/fastmri/nifti}" fastmri
[ "${INCLUDE_ABIDE}" = "1" ]   && curate_ds abide   "${ABIDE_INPUT_DIR:-data/abide}"           abide

# ── Subsample to NUM_REFS per dataset ──────────────────────────────────────
section "Subsample"
${PYBIN} - <<PYSUBSAMPLE
import shutil
from pathlib import Path
import pandas as pd

ROOT = Path("${PROJECT_ROOT}")
TABLES = ROOT / "results" / "tables"

specs = []
if "${INCLUDE_IXI}" == "1":     specs.append(("ixi",     ${NUM_REFS_CALIB}))
if "${INCLUDE_FASTMRI}" == "1": specs.append(("fastmri", ${NUM_REFS_CALIB}))
if "${INCLUDE_ABIDE}" == "1":   specs.append(("abide",   ${NUM_REFS_SMOKE}))

for ds, n_refs in specs:
    manifest = TABLES / f"reference_manifest_{ds}.csv"
    if not manifest.exists():
        raise SystemExit(f"[subsample] missing manifest: {manifest}")
    df = pd.read_csv(manifest)
    # passed_qc is "true"/"false" string after Phase 01 csv.DictWriter.
    passed = df[df["passed_qc"].astype(str).str.lower().eq("true")].copy()
    if len(passed) < n_refs:
        raise SystemExit(
            f"[subsample] {ds}: {len(passed)} refs passed QC, need {n_refs}. "
            f"Lower NUM_REFS or extract more raws."
        )
    passed = passed.sort_values("subject_id", kind="stable").head(n_refs)
    out_csv = TABLES / f"reference_manifest_{ds}_proto.csv"
    passed.to_csv(out_csv, index=False)
    proto_dir = ROOT / "data" / ds / "references_proto"
    proto_dir.mkdir(parents=True, exist_ok=True)
    for fp in passed["filepath"]:
        src = Path(fp)
        if not src.is_absolute():
            src = ROOT / src
        dst = proto_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
    print(f"[subsample] {ds}: kept {len(passed)} → {out_csv} ; symlinks → {proto_dir}")
PYSUBSAMPLE

# ── Phase 02 — image-space corruption per dataset ──────────────────────────
# Phase 02 device: defaults to "auto" (GPU when available), CPU otherwise.
# FFT-based transforms (motion / ghosting / spike) are ~100-300x faster on
# CUDA. Override via CORRUPT_DEVICE env var (e.g. CORRUPT_DEVICE=cpu to force).
CORRUPT_DEVICE="${CORRUPT_DEVICE:-auto}"
section "Phase 02 — image-space corruption (device=${CORRUPT_DEVICE})"

corrupt_ds() {
    local ds="$1" tag="$2" corruptions="$3" severities="$4"
    local marker="${PHASES_DIR}/.phase02_${ds}"
    [ -f "${marker}" ] && { log "Phase 02 ${ds}: already done."; return 0; }
    dl_run "Phase 02: ${ds} corruption (${corruptions} sev=${severities} dev=${CORRUPT_DEVICE})" \
        -i "data/${ds}/references_proto/*.nii.gz" \
        -o "data/${ds}/corrupted_proto/" \
        -o "results/tables/corruption_manifest.csv" \
        -- "${PYBIN} code/02_generate_corruptions.py \
            --input-dir data/${ds}/references_proto \
            --output-dir data/${ds}/corrupted_proto \
            --corruptions ${corruptions} \
            --severities ${severities} \
            --dataset-tag ${tag} \
            --device ${CORRUPT_DEVICE} \
            --manifest-path results/tables/corruption_manifest.csv"
    /usr/bin/touch "${marker}"
}

[ "${INCLUDE_IXI}" = "1" ]     && corrupt_ds ixi     ixi     "${CALIB_CORRUPTIONS}" "${CALIB_SEVERITIES}"
[ "${INCLUDE_FASTMRI}" = "1" ] && corrupt_ds fastmri fastmri "${CALIB_CORRUPTIONS}" "${CALIB_SEVERITIES}"
[ "${INCLUDE_ABIDE}" = "1" ]   && corrupt_ds abide   abide   "${SMOKE_CORRUPTIONS}" "${SMOKE_SEVERITIES}"

# ── Phase 02b — k-space corruption (FastMRI) ───────────────────────────────
section "Phase 02b — k-space corruption (FastMRI)"
if [ "${SKIP_PHASE2B}" = "1" ]; then
    log "SKIP_PHASE2B=1 — skipping (set SKIP_PHASE2B=0 to enable; needs FastMRI H5 corpus)."
elif [ "${INCLUDE_FASTMRI}" != "1" ]; then
    log "Phase 02b skipped (INCLUDE_FASTMRI=0)."
elif [ ! -f "${PHASES_DIR}/.phase02b_fastmri" ]; then
    ${PYBIN} - <<PYH5
from pathlib import Path
import pandas as pd
ROOT = Path("${PROJECT_ROOT}")
TABLES = ROOT / "results" / "tables"
df = pd.read_csv(TABLES / "reference_manifest_fastmri_proto.csv")
raw_dir = ROOT / "data" / "fastmri" / "raw"
proto = ROOT / "data" / "fastmri" / "raw_proto"
proto.mkdir(parents=True, exist_ok=True)
picked = 0
for fp in df["filepath"]:
    stem = Path(fp).name.replace(".nii.gz", "")
    src = raw_dir / f"{stem}.h5"
    if not src.exists():
        print(f"[h5-stage] WARNING: H5 missing for {stem}")
        continue
    dst = proto / src.name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())
    picked += 1
print(f"[h5-stage] symlinked {picked} H5 files into {proto}")
PYH5
    dl_run "Phase 02b: FastMRI k-space motion (sev=${CALIB_SEVERITIES})" \
        -i "data/fastmri/raw_proto/*.h5" \
        -o "data/fastmri/corrupted_kspace_proto/" \
        -o "results/tables/corruption_manifest.csv" \
        -- "${PYBIN} code/02b_corrupt_kspace_motion.py \
            --input-dir data/fastmri/raw_proto \
            --reference-dir data/fastmri/references_proto \
            --output-dir data/fastmri/corrupted_kspace_proto \
            --severities ${CALIB_SEVERITIES} \
            --manifest-path results/tables/corruption_manifest.csv"
    /usr/bin/touch "${PHASES_DIR}/.phase02b_fastmri"
fi

# ── Phase 03 — SynthSeg --parc per dataset (refs + corrupted) ──────────────
section "Phase 03 — SynthSeg --parc (mode=${SYNTHSEG_MODE})"

synthseg_ds() {
    local ds="$1" sub="$2" input_dir="$3"
    local marker="${PHASES_DIR}/.phase03_${ds}_${sub}"
    # Per-(ds, sub) manifest because 03_run_synthseg.py opens manifest_path
    # with "w" — a second call (cor) would otherwise overwrite the first
    # call's (refs) entries. 04/05/05b take --synthseg-manifest repeatable.
    local manifest="results/tables/synthseg_${ds}_${sub}_manifest.csv"
    [ -f "${marker}" ] && { log "Phase 03 ${ds} ${sub}: already done."; return 0; }
    if [ ! -d "${input_dir}" ] || [ -z "$(/bin/ls -A "${input_dir}" 2>/dev/null)" ]; then
        log "Phase 03 ${ds} ${sub}: input dir empty — skipping."
        /usr/bin/touch "${marker}"
        return 0
    fi
    local fast_flag=""
    [ "${SYNTHSEG_FAST}" = "1" ] && fast_flag="--fast"
    dl_run "Phase 03: SynthSeg ${ds} ${sub} (--parc${fast_flag:+ --fast}, mode=${SYNTHSEG_MODE})" \
        -i "${input_dir}/" \
        -o "data/derivatives/synthseg/${ds}/" \
        -o "${manifest}" \
        -- "${PYBIN} code/03_run_synthseg.py \
            --input-dir ${input_dir} \
            --output-dir data/derivatives/synthseg/${ds} \
            --mode ${SYNTHSEG_MODE} \
            --parc \
            ${fast_flag} \
            --manifest-path ${manifest}"
    /usr/bin/touch "${marker}"
}

# Build the repeatable --synthseg-manifest argument list that 03b/04/05/05b consume.
synthseg_manifest_args() {
    local args=""
    if [ "${INCLUDE_IXI}" = "1" ]; then
        args="${args} --synthseg-manifest results/tables/synthseg_ixi_refs_manifest.csv"
        args="${args} --synthseg-manifest results/tables/synthseg_ixi_cor_manifest.csv"
    fi
    if [ "${INCLUDE_FASTMRI}" = "1" ]; then
        args="${args} --synthseg-manifest results/tables/synthseg_fastmri_refs_manifest.csv"
        args="${args} --synthseg-manifest results/tables/synthseg_fastmri_cor_manifest.csv"
        if [ "${SKIP_PHASE2B}" != "1" ]; then
            args="${args} --synthseg-manifest results/tables/synthseg_fastmri_cor_kspace_manifest.csv"
        fi
    fi
    if [ "${INCLUDE_ABIDE}" = "1" ]; then
        args="${args} --synthseg-manifest results/tables/synthseg_abide_refs_manifest.csv"
        args="${args} --synthseg-manifest results/tables/synthseg_abide_cor_manifest.csv"
    fi
    # Filter out manifests that don't exist yet (early phases may skip).
    local kept=""
    local prev=""
    for tok in $args; do
        if [ "$prev" = "--synthseg-manifest" ]; then
            if [ -f "${tok}" ]; then
                kept="${kept} --synthseg-manifest ${tok}"
            fi
            prev=""
        else
            prev="${tok}"
        fi
    done
    echo "${kept}"
}

if [ "${INCLUDE_IXI}" = "1" ]; then
    synthseg_ds ixi refs "data/ixi/references_proto"
    synthseg_ds ixi cor  "data/ixi/corrupted_proto"
fi
if [ "${INCLUDE_FASTMRI}" = "1" ]; then
    synthseg_ds fastmri refs "data/fastmri/references_proto"
    synthseg_ds fastmri cor  "data/fastmri/corrupted_proto"
    if [ "${SKIP_PHASE2B}" != "1" ] && [ -d "data/fastmri/corrupted_kspace_proto" ]; then
        synthseg_ds fastmri cor_kspace "data/fastmri/corrupted_kspace_proto"
    fi
fi
if [ "${INCLUDE_ABIDE}" = "1" ]; then
    synthseg_ds abide refs "data/abide/references_proto"
    synthseg_ds abide cor  "data/abide/corrupted_proto"
fi

# ── Phase 03b — cortical thickness ────────────────────────────────────────
section "Phase 03b — cortical thickness"
if [ ! -f "${PHASES_DIR}/.phase03b" ]; then
    SS_ARGS="$(synthseg_manifest_args)"
    dl_run "Phase 03b: Compute cortical thickness" \
        -i "data/derivatives/synthseg/" \
        -o "results/tables/cortical_thickness.csv" \
        -- "${PYBIN} code/03b_compute_thickness.py \
            --synthseg-dir data/derivatives/synthseg \
            ${SS_ARGS} \
            --output-file results/tables/cortical_thickness.csv"
    /usr/bin/touch "${PHASES_DIR}/.phase03b"
fi

# ── Phase 04 — machine preference + per-structure Dice ─────────────────────
# Use --synthseg-manifest (one per ds × {refs, cor}) instead of --synthseg-dir.
# Reasons: (a) Phase 03 overwrites manifest on each call so per-(ds, sub)
# files are necessary, (b) discover_seg_map keys by filename stem and would
# collide on same-stem ref/cor pairs that 03 places in different subdirs.
section "Phase 04 — machine preference"
if [ ! -f "${PHASES_DIR}/.phase04" ]; then
    SS_ARGS="$(synthseg_manifest_args)"
    dl_run "Phase 04: Machine preference + per-structure Dice" \
        -i "results/tables/corruption_manifest.csv" \
        -i "results/tables/cortical_thickness.csv" \
        -o "results/tables/machine_preference.csv" \
        -o "results/tables/per_structure_dice.csv" \
        -- "${PYBIN} code/04_compute_preference.py \
            --corruption-manifest results/tables/corruption_manifest.csv \
            ${SS_ARGS} \
            --thickness-file results/tables/cortical_thickness.csv \
            --output-file results/tables/machine_preference.csv \
            --per-structure-output results/tables/per_structure_dice.csv"
    /usr/bin/touch "${PHASES_DIR}/.phase04"
fi

# ── Phase 05 — IQM extraction ──────────────────────────────────────────────
section "Phase 05 — IQM extraction"
if [ ! -f "${PHASES_DIR}/.phase05" ]; then
    REF_ARGS=""
    [ "${INCLUDE_IXI}" = "1" ]     && REF_ARGS="${REF_ARGS} --ref-manifest results/tables/reference_manifest_ixi_proto.csv"
    [ "${INCLUDE_FASTMRI}" = "1" ] && REF_ARGS="${REF_ARGS} --ref-manifest results/tables/reference_manifest_fastmri_proto.csv"
    [ "${INCLUDE_ABIDE}" = "1" ]   && REF_ARGS="${REF_ARGS} --ref-manifest results/tables/reference_manifest_abide_proto.csv"
    SS_ARGS="$(synthseg_manifest_args)"
    dl_run "Phase 05: Extract IQMs" \
        -i "results/tables/corruption_manifest.csv" \
        -o "results/tables/iqm_features.csv" \
        -- "${PYBIN} code/05_extract_iqms.py \
            ${REF_ARGS} \
            --cor-manifest results/tables/corruption_manifest.csv \
            ${SS_ARGS} \
            --output-file results/tables/iqm_features.csv"
    /usr/bin/touch "${PHASES_DIR}/.phase05"
fi

# ── Phase 05b — SynthSeg internal-QC aggregation ───────────────────────────
section "Phase 05b — SynthSeg QC aggregation"
if [ ! -f "${PHASES_DIR}/.phase05b" ]; then
    SS_ARGS="$(synthseg_manifest_args)"
    dl_run "Phase 05b: Aggregate SynthSeg QC" \
        -i "results/tables/corruption_manifest.csv" \
        -o "results/tables/synthseg_qc_features.csv" \
        -- "${PYBIN} code/05b_aggregate_synthseg_qc.py \
            ${SS_ARGS} \
            --cor-manifest results/tables/corruption_manifest.csv \
            --output-file results/tables/synthseg_qc_features.csv"
    /usr/bin/touch "${PHASES_DIR}/.phase05b"
fi

# ── Phase 08a/A — benchmark subsample (no VLM inference) ───────────────────
section "Phase 08a/A — benchmark_subsample.csv"
if [ ! -f "${PHASES_DIR}/.phase08a_A" ]; then
    REF_ARGS=""
    [ "${INCLUDE_IXI}" = "1" ]     && REF_ARGS="${REF_ARGS} --ref-manifest results/tables/reference_manifest_ixi_proto.csv"
    [ "${INCLUDE_FASTMRI}" = "1" ] && REF_ARGS="${REF_ARGS} --ref-manifest results/tables/reference_manifest_fastmri_proto.csv"
    [ "${INCLUDE_ABIDE}" = "1" ]   && REF_ARGS="${REF_ARGS} --ref-manifest results/tables/reference_manifest_abide_proto.csv"
    # n-refs is the total target subsample size; cap to what we have.
    N_REFS_TOTAL=0
    [ "${INCLUDE_IXI}" = "1" ]     && N_REFS_TOTAL=$((N_REFS_TOTAL + NUM_REFS_CALIB))
    [ "${INCLUDE_FASTMRI}" = "1" ] && N_REFS_TOTAL=$((N_REFS_TOTAL + NUM_REFS_CALIB))
    [ "${INCLUDE_ABIDE}" = "1" ]   && N_REFS_TOTAL=$((N_REFS_TOTAL + NUM_REFS_SMOKE))
    # 08a Phase A intersects calib + smoke severity sets: pass the union.
    UNION_SEV="${CALIB_SEVERITIES},${SMOKE_SEVERITIES}"
    UNION_COR="${CALIB_CORRUPTIONS},${SMOKE_CORRUPTIONS}"
    if [ "${N_REFS_TOTAL}" -lt 4 ]; then
        log "Phase 08a/A: only ${N_REFS_TOTAL} ref(s) — sklearn train_test_split needs ≥ 4. Skipping (best-effort)."
        /usr/bin/touch "${PHASES_DIR}/.phase08a_A"
    else
        # 08a Phase A is dry-run; failures here are not fatal for the smoke,
        # so trap and log rather than propagate.
        if dl_run "Phase 08a Phase A: build benchmark_subsample.csv (no VLM)" \
            -i "results/tables/machine_preference.csv" \
            -i "results/tables/corruption_manifest.csv" \
            -o "results/tables/benchmark_subsample.csv" \
            -- "${PYBIN} code/08a_eval_3d_vlms.py \
                --seed 42 --dry-run \
                ${REF_ARGS} \
                --cor-manifest results/tables/corruption_manifest.csv \
                --preference-csv results/tables/machine_preference.csv \
                --n-refs ${N_REFS_TOTAL} \
                --severities ${UNION_SEV} \
                --corruption-types ${UNION_COR} \
                --subsample-manifest results/tables/benchmark_subsample.csv"; then
            /usr/bin/touch "${PHASES_DIR}/.phase08a_A"
        else
            log "Phase 08a/A failed (best-effort — see traceback above). Continuing wrap-up."
        fi
    fi
fi

# ── Wrap-up ────────────────────────────────────────────────────────────────
section "Wrap-up"
${PYBIN} code/results_tracker.py --phase 3 || log "results_tracker phase 3 reported issues (non-fatal)"

BUNDLE="${PROJECT_ROOT}/results/dandi_combined_${TS}.tar.gz"
/usr/bin/tar -czf "${BUNDLE}" \
    -C "${PROJECT_ROOT}" \
    results/tables/ results/logs/ 2>/dev/null || true
log "results bundle: ${BUNDLE}"

section "Combined run complete: $(date)"
log "Key outputs:"
log "  results/tables/machine_preference.csv"
log "  results/tables/per_structure_dice.csv"
log "  results/tables/cortical_thickness.csv"
log "  results/tables/iqm_features.csv"
log "  results/tables/synthseg_qc_features.csv"
log "  results/tables/benchmark_subsample.csv"
if [ "${USE_DATALAD_RUN}" = "1" ]; then
    log "Provenance: every phase is one git commit. Inspect with:"
    log "  git log --oneline -20"
fi
