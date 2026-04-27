#!/usr/bin/env bash
#
# scripts/gpu_server_launch_smoke.sh — resilient launcher for run_prototype.sh
# on a cloud GPU server (RunPod, Jarvis, Lambda, ...).
#
# Why this wrapper exists:
#   * SSH disconnects + container PID-1 cleanup kill `nohup` smokes silently;
#     run this under tmux instead so the smoke survives drops:
#         tmux new-session -d -s smoke 'bash scripts/gpu_server_launch_smoke.sh'
#         tmux attach -t smoke           # detach with Ctrl-B D
#   * Transient pod-level events (brief OOM, network blip during a `pip`
#     subcall, temporary GPU disappearance) can fail one phase. A 5-attempt
#     outer loop turns those into self-healing retries.
#   * tqdm `\r` writes don't flush through `tee`; wrapping the run in
#     `stdbuf -oL -eL` gives a live log so progress is visible in real time.
#   * Phase 02 / 03 already skip non-empty resume targets; defensively wipe
#     0-byte files between attempts so a truncated CSV doesn't poison resume.
#
# Environment overrides (defaults match RunPod-style hosts):
#   WORK_DIR            /workspace        — env sentinel + log location
#   NEUROQC_DIR         ${WORK_DIR}/NeuroQC or /root/NeuroQC if present
#   SMOKE_LOG           ${WORK_DIR}/smoke.log
#   MAX_ATTEMPTS        5
#   SYNTHSEG_MODE       python
#   NUM_REFS_CALIB      15
#   NUM_REFS_SMOKE      15
#   CORRUPT_DEVICE      cuda
#   IXI_INPUT_DIR       /root/data/ixi/raw
#   FASTMRI_INPUT_DIR   /root/data/fastmri/nifti
#   ABIDE_INPUT_DIR     /root/data/abide
#
# Exit code:
#   0 if any attempt completes; non-zero if all attempts fail (so the parent
#   tmux pane / shell sees a real failure rather than silent looping).

set -uo pipefail

WORK_DIR="${WORK_DIR:-/workspace}"
NEUROQC_DIR="${NEUROQC_DIR:-${WORK_DIR}/NeuroQC}"
if [ ! -d "${NEUROQC_DIR}" ] && [ -d "/root/NeuroQC" ]; then
    NEUROQC_DIR="/root/NeuroQC"
fi
SMOKE_LOG="${SMOKE_LOG:-${WORK_DIR}/smoke.log}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"

# Run-prototype env defaults (kept in sync with the cookbook in
# scripts/gpu_server_setup.sh's final-message block).
export SYNTHSEG_MODE="${SYNTHSEG_MODE:-python}"
export NUM_REFS_CALIB="${NUM_REFS_CALIB:-15}"
export NUM_REFS_SMOKE="${NUM_REFS_SMOKE:-15}"
export CORRUPT_DEVICE="${CORRUPT_DEVICE:-cuda}"
export IXI_INPUT_DIR="${IXI_INPUT_DIR:-/root/data/ixi/raw}"
export FASTMRI_INPUT_DIR="${FASTMRI_INPUT_DIR:-/root/data/fastmri/nifti}"
export ABIDE_INPUT_DIR="${ABIDE_INPUT_DIR:-/root/data/abide}"

# Calmer Python + TF + glibc behaviour for long-running multi-process runs.
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

log() { printf '[launch_smoke %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

if [ ! -d "${NEUROQC_DIR}" ]; then
    log "ERROR: NEUROQC_DIR=${NEUROQC_DIR} not found."
    exit 2
fi
if [ ! -f "${WORK_DIR}/neuroqc_env.sh" ]; then
    log "ERROR: env sentinel ${WORK_DIR}/neuroqc_env.sh missing — run scripts/gpu_server_setup.sh first."
    exit 2
fi

# Defensive cleanup of 0-byte outputs that would confuse resume logic.
# Scoped narrowly to known output trees so we never touch user data or
# .phases_proto/ markers.
cleanup_zero_byte_outputs() {
    local pruned=0
    local roots=(
        "${NEUROQC_DIR}/results/tables"
        "${NEUROQC_DIR}/data"
    )
    for root in "${roots[@]}"; do
        [ -d "${root}" ] || continue
        # find -delete with -size 0 + name filters — covers .csv manifests
        # and corrupted_proto / synthseg .nii.gz outputs without touching
        # legitimate empty marker files under .phases_proto/.
        while IFS= read -r f; do
            rm -f "$f" && pruned=$((pruned + 1))
        done < <(find "${root}" \
            \( -path '*/.phases_proto*' -o -path '*/.git*' \) -prune -o \
            -type f -size 0 \
            \( -name '*.csv' -o -name '*.nii.gz' \) -print 2>/dev/null)
    done
    log "Pruned ${pruned} zero-byte output(s)"
}

run_attempt() {
    local attempt="$1"
    log "Attempt ${attempt}/${MAX_ATTEMPTS} — launching scripts/run_prototype.sh"
    # shellcheck disable=SC1091
    source "${WORK_DIR}/neuroqc_env.sh"
    cd "${NEUROQC_DIR}"
    # stdbuf -oL -eL forces line buffering so tqdm's \r writes flush through
    # tee instead of pooling for >10 minutes.
    set -o pipefail
    bash scripts/run_prototype.sh 2>&1 \
        | stdbuf -oL -eL tee -a "${SMOKE_LOG}"
    local rc=${PIPESTATUS[0]}
    set +o pipefail
    return "$rc"
}

print_summary() {
    log "──── Phase markers ────"
    if [ -d "${NEUROQC_DIR}/.phases_proto" ]; then
        ls -1 "${NEUROQC_DIR}/.phases_proto" 2>/dev/null | sort | sed 's/^/  /'
    else
        log "  (no .phases_proto directory)"
    fi
    log "──── Manifest row counts ────"
    for f in \
        "${NEUROQC_DIR}/results/tables/reference_manifest.csv" \
        "${NEUROQC_DIR}/results/tables/corruption_manifest.csv" \
        "${NEUROQC_DIR}/results/tables/synthseg_manifest.csv" \
        "${NEUROQC_DIR}/results/tables/iqm_features.csv" \
        "${NEUROQC_DIR}/results/tables/machine_preference.csv"
    do
        if [ -f "$f" ]; then
            # row count = lines - 1 (header)
            local rows
            rows=$(($(wc -l < "$f") - 1))
            log "  $(basename "$f"): ${rows} rows"
        fi
    done
}

: > "${SMOKE_LOG}"  # truncate at start so attempts share one growing log
log "Smoke log: ${SMOKE_LOG}"
log "NEUROQC_DIR=${NEUROQC_DIR}  WORK_DIR=${WORK_DIR}  MAX_ATTEMPTS=${MAX_ATTEMPTS}"

attempt=1
while [ "${attempt}" -le "${MAX_ATTEMPTS}" ]; do
    cleanup_zero_byte_outputs
    if run_attempt "${attempt}"; then
        log "Attempt ${attempt} succeeded."
        print_summary
        exit 0
    fi
    log "Attempt ${attempt} failed (rc=$?). Sleeping 30 s before retry."
    sleep 30
    attempt=$((attempt + 1))
done

log "All ${MAX_ATTEMPTS} attempts failed. See ${SMOKE_LOG} for details."
print_summary
exit 1
