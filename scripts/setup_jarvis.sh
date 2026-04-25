#!/usr/bin/env bash
#
# setup_jarvis.sh — prepare a Jarvis Labs GPU instance for phase 3.
#
# Jarvis Labs images do not ship FreeSurfer. This script installs it via
# install_freesurfer_linux.sh then runs the verifier. A license must be
# provided via FS_LICENSE_PATH or FS_LICENSE_BASE64.

set -euo pipefail

log() { printf '[setup_jarvis] %s\n' "$*"; }

if [ -z "${FS_LICENSE_PATH:-}" ] && [ -z "${FS_LICENSE_BASE64:-}" ]; then
    log "ERROR: supply your FreeSurfer license via FS_LICENSE_PATH=/path/to/license.txt"
    log "or FS_LICENSE_BASE64=\$(base64 < license.txt)"
    log "Register for a free license at: https://surfer.nmr.mgh.harvard.edu/registration.html"
    exit 2
fi

SCRIPT_DIR="$(/usr/bin/dirname "$(/usr/bin/readlink -f "$0")")"
REPO_ROOT="$(/usr/bin/dirname "${SCRIPT_DIR}")"

bash "${SCRIPT_DIR}/install_freesurfer_linux.sh"

# Re-source so the PATH is current in this shell.
export FREESURFER_HOME="${FREESURFER_HOME:-$HOME/freesurfer/${FS_VERSION:-8.2.0}}"
# shellcheck disable=SC1091
source "${FREESURFER_HOME}/SetUpFreeSurfer.sh" > /dev/null

"${REPO_ROOT}/bin/python" "${SCRIPT_DIR}/verify_install.py"
