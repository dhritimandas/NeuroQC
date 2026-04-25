#!/usr/bin/env bash
#
# setup_dandi.sh — prepare a DANDI Hub session for phase 3.
#
# DANDI Hub images typically ship FreeSurfer preinstalled; this script assumes
# that and just sources SetUpFreeSurfer.sh, then runs the verifier. If FS is
# missing, it falls back to install_freesurfer_linux.sh (which requires an
# FS_LICENSE_PATH or FS_LICENSE_BASE64 env var).

set -euo pipefail

log() { printf '[setup_dandi] %s\n' "$*"; }

# Search version-agnostic common FS install locations.
FS_HOME=""
if [ -n "${FREESURFER_HOME:-}" ] && [ -f "${FREESURFER_HOME}/SetUpFreeSurfer.sh" ]; then
    FS_HOME="${FREESURFER_HOME}"
else
    for base in "/opt/freesurfer" "/usr/local/freesurfer" "$HOME/freesurfer"; do
        # Accept a plain-base install or any versioned subdir.
        if [ -f "${base}/SetUpFreeSurfer.sh" ]; then
            FS_HOME="${base}"
            break
        fi
        for versioned in "${base}"/*/SetUpFreeSurfer.sh; do
            [ -f "${versioned}" ] && FS_HOME="$(/usr/bin/dirname "${versioned}")" && break 2
        done
    done
fi

if [ -z "${FS_HOME}" ]; then
    log "FreeSurfer not found in common DANDI locations; falling back to install_freesurfer_linux.sh"
    SCRIPT_DIR="$(/usr/bin/dirname "$(/usr/bin/readlink -f "$0")")"
    bash "${SCRIPT_DIR}/install_freesurfer_linux.sh"
    FS_HOME="${HOME}/freesurfer/${FS_VERSION:-8.2.0}"
fi

export FREESURFER_HOME="${FS_HOME}"
# shellcheck disable=SC1091
source "${FREESURFER_HOME}/SetUpFreeSurfer.sh" > /dev/null
log "FREESURFER_HOME=${FREESURFER_HOME}"
log "mri_synthseg: $(command -v mri_synthseg || echo MISSING)"

SCRIPT_DIR="$(/usr/bin/dirname "$(/usr/bin/readlink -f "$0")")"
REPO_ROOT="$(/usr/bin/dirname "${SCRIPT_DIR}")"

# Prefer project-local venv if present (matches the macOS dev layout); otherwise
# fall back to whichever `python` is on PATH (DANDI Hub conda env).
if [ -x "${REPO_ROOT}/bin/python" ]; then
    PYBIN="${REPO_ROOT}/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYBIN="$(command -v python)"
else
    log "ERROR: no python on PATH and no ${REPO_ROOT}/bin/python venv. Activate your conda env first."
    exit 4
fi
log "Using python: ${PYBIN}"
"${PYBIN}" "${SCRIPT_DIR}/verify_install.py"
