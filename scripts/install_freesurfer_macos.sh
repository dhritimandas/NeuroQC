#!/usr/bin/env bash
#
# install_freesurfer_macos.sh — guided instructions for installing FreeSurfer
# on macOS. This script does NOT download the installer: FreeSurfer URLs rotate
# between releases and you may want a specific version. Download it yourself
# from the official page and run this script to verify your setup afterwards.
#
# Steps you'll perform manually:
#   1. Download the macOS .pkg from:
#        https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
#      (Choose the arm64 / Apple Silicon build if you're on M-series.)
#
#   2. Run the .pkg (Apple admin password required).
#      Default install path:  /Applications/freesurfer/<VERSION>
#
#   3. Register for a license:
#        https://surfer.nmr.mgh.harvard.edu/registration.html
#      Save the emailed license.txt to <install_path>/license.txt
#
#   4. Add to ~/.zshrc (or ~/.bashrc):
#        export FREESURFER_HOME=/Applications/freesurfer/<VERSION>
#        source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
#      Open a new terminal to pick this up.
#
# This script then verifies the result by calling scripts/verify_install.py.

set -euo pipefail

log() { printf '[install_freesurfer_macos] %s\n' "$*"; }

if [ "$(uname)" != "Darwin" ]; then
    log "ERROR: run install_freesurfer_linux.sh on Linux (detected $(uname))."
    exit 1
fi

cat <<'EOS'

──────────────────────────────────────────────────────────
FreeSurfer macOS install — manual steps
──────────────────────────────────────────────────────────

1. Download the .pkg from:
     https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
   Pick the arm64 / Apple Silicon build for M-series Macs.

2. Run the .pkg; default path: /Applications/freesurfer/<VERSION>

3. License: register at
     https://surfer.nmr.mgh.harvard.edu/registration.html
   then save the emailed license.txt to:
     /Applications/freesurfer/<VERSION>/license.txt

4. Add to ~/.zshrc (replace <VERSION> with your install, e.g. 8.0.0):
     export FREESURFER_HOME=/Applications/freesurfer/<VERSION>
     source "$FREESURFER_HOME/SetUpFreeSurfer.sh"

5. Open a new terminal window; then run this to verify:
     python scripts/verify_install.py

──────────────────────────────────────────────────────────

EOS

SCRIPT_DIR="$(/usr/bin/dirname "$0")"
REPO_ROOT="$(/usr/bin/dirname "${SCRIPT_DIR}")"
if [ -x "${REPO_ROOT}/bin/python" ]; then
    log "Running verify_install.py now (will report missing FS if not installed yet)..."
    "${REPO_ROOT}/bin/python" "${SCRIPT_DIR}/verify_install.py" || true
fi
