#!/usr/bin/env bash
#
# install_freesurfer_linux.sh — fully automated FreeSurfer install on Linux
# (Ubuntu 22 / CentOS 7). Defaults to version 7.4.1 (last release with a
# Linux tarball at the canonical surfer.nmr.mgh.harvard.edu mirror; FS 8.x
# ships as a Debian .deb for Ubuntu, see docs/runpod_setup.md for the apt
# install path). Override via FS_VERSION.
# Intended for DANDI Hub (if FS is not already present) and Jarvis Labs.
# Idempotent: skips already-downloaded/extracted artefacts.
#
# License file:
#   Provide your license file contents via FS_LICENSE_PATH (path to a local
#   license.txt) or FS_LICENSE_BASE64 (base64-encoded contents, handy for
#   piping through env vars). Exactly one must be set.
#
# Environment overrides:
#   FS_VERSION      — default 7.4.1 (override for other releases)
#   FS_INSTALL_DIR  — default $HOME/freesurfer (version dir appended)
#   FS_DISTRO       — "ubuntu22" (default) or "centos7"
#   FS_TARBALL_URL  — override the download URL if defaults 404

set -euo pipefail

FS_VERSION="${FS_VERSION:-7.4.1}"
FS_INSTALL_DIR="${FS_INSTALL_DIR:-$HOME/freesurfer}"
FS_DISTRO="${FS_DISTRO:-ubuntu22}"

case "${FS_DISTRO}" in
    ubuntu22) FS_TARBALL_URL_DEFAULT="https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/${FS_VERSION}/freesurfer-linux-ubuntu22_amd64-${FS_VERSION}.tar.gz" ;;
    centos7)  FS_TARBALL_URL_DEFAULT="https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/${FS_VERSION}/freesurfer-linux-centos7_x86_64-${FS_VERSION}.tar.gz" ;;
    *) echo "[install_freesurfer_linux] ERROR: unknown FS_DISTRO='${FS_DISTRO}' (expected ubuntu22 or centos7)" >&2; exit 2 ;;
esac

FS_TARBALL_URL="${FS_TARBALL_URL:-$FS_TARBALL_URL_DEFAULT}"
FS_TARBALL_PATH="/tmp/freesurfer-${FS_VERSION}-${FS_DISTRO}.tar.gz"
FREESURFER_HOME="${FS_INSTALL_DIR}/${FS_VERSION}"

log() { printf '[install_freesurfer_linux] %s\n' "$*"; }

if [ "$(uname)" != "Linux" ]; then
    log "ERROR: this script is for Linux only (detected $(uname))."
    exit 1
fi

mkdir -p "${FS_INSTALL_DIR}"

# ── download ──
if [ -d "${FREESURFER_HOME}" ] && [ -f "${FREESURFER_HOME}/SetUpFreeSurfer.sh" ]; then
    log "FreeSurfer already extracted at ${FREESURFER_HOME}; skipping download."
else
    if [ -f "${FS_TARBALL_PATH}" ]; then
        log "Tarball already downloaded at ${FS_TARBALL_PATH}"
    else
        log "Downloading ${FS_TARBALL_URL} (~9 GB) to ${FS_TARBALL_PATH}"
        /usr/bin/curl -L -C - --fail --output "${FS_TARBALL_PATH}" "${FS_TARBALL_URL}"
    fi

    log "Extracting tarball into ${FS_INSTALL_DIR}"
    /usr/bin/tar -xzf "${FS_TARBALL_PATH}" -C "${FS_INSTALL_DIR}"
    # The tarball extracts to a directory named `freesurfer/` — rename to include version.
    if [ -d "${FS_INSTALL_DIR}/freesurfer" ] && [ ! -d "${FREESURFER_HOME}" ]; then
        mv "${FS_INSTALL_DIR}/freesurfer" "${FREESURFER_HOME}"
    fi
fi

# ── license ──
LICENSE_DEST="${FREESURFER_HOME}/license.txt"
if [ -f "${LICENSE_DEST}" ]; then
    log "License already present at ${LICENSE_DEST}"
elif [ -n "${FS_LICENSE_PATH:-}" ]; then
    log "Copying license from FS_LICENSE_PATH=${FS_LICENSE_PATH}"
    cp "${FS_LICENSE_PATH}" "${LICENSE_DEST}"
elif [ -n "${FS_LICENSE_BASE64:-}" ]; then
    log "Writing license from FS_LICENSE_BASE64"
    printf '%s' "${FS_LICENSE_BASE64}" | /usr/bin/base64 -d > "${LICENSE_DEST}"
else
    log "WARNING: no license provided. Register at"
    log "  https://surfer.nmr.mgh.harvard.edu/registration.html"
    log "and copy license.txt to ${LICENSE_DEST} before running mri_synthseg."
fi

# ── setup + verify ──
export FREESURFER_HOME
# FreeSurfer's setup references unbound vars (e.g. SUBJECTS_DIR) and runs
# internal tests with non-zero returns; toggle errexit + nounset around
# the source so it doesn't trip our `set -euo pipefail`.
set +eu
# shellcheck disable=SC1091
source "${FREESURFER_HOME}/SetUpFreeSurfer.sh" > /dev/null
set -eu

if command -v mri_synthseg >/dev/null 2>&1; then
    log "OK: mri_synthseg available at $(command -v mri_synthseg)"
else
    log "ERROR: mri_synthseg not on PATH after sourcing SetUpFreeSurfer.sh" >&2
    exit 3
fi

cat <<EOS

──────────────────────────────────────────────────────────
FreeSurfer ${FS_VERSION} installed at ${FREESURFER_HOME}

Add to your shell rc (~/.bashrc or ~/.zshrc):
  export FREESURFER_HOME=${FREESURFER_HOME}
  source "\$FREESURFER_HOME/SetUpFreeSurfer.sh"

Verify full install:
  python scripts/verify_install.py
──────────────────────────────────────────────────────────
EOS
