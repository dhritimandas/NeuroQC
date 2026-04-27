#!/usr/bin/env bash
#
# scripts/gpu_server_setup.sh — one-shot bootstrap for any A100 / Hopper-class
# cloud GPU server (RunPod, Jarvis, Lambda, Coreweave, vast.ai, ...). The
# script is provider-agnostic; the only host-specific assumption is the
# /workspace + /root layout described below, which is the de-facto standard
# on these hosts.
#
# What this script delivers (idempotent end-to-end):
#   1. NeuroQC + nobrainer (feat/qc-update) cloned and editable-installed
#   2. Python 3.12 venv on local NVMe (default /root/neuroqc-venv, not the
#      quota'd network FS that backs /workspace on most cloud GPU hosts)
#   3. Project deps + GPU PyTorch (cu118) + TensorFlow GPU + tf-keras shim,
#      with a hardened Phase 6 (cu13 cleanup, cu12 force-reinstall, numpy<2)
#   4. FreeSurfer 8.2.0 via apt .deb (Ubuntu 22) — fallback 7.4.1 tarball
#   5. BBillot/SynthSeg standalone clone with NumPy-2 / Keras-3 / predict.py
#      sed patches + symlink to FS-bundled model weights
#   6. HuggingFace cache pinned to local NVMe (avoids /workspace quota)
#   7. Env sentinel ${WORK_DIR}/neuroqc_env.sh that primes a fresh shell with
#      everything (venv + FS + SynthSeg + GPU + TF + HF + DataLad opt-out)
#   8. Conv3D smoke (Phase 12.5) that exercises cuDNN end-to-end in ~5 s,
#      so any cu13/cuDNN ABI mismatch surfaces before the first real phase
#
# Why /root over /workspace for hot data:
#   /workspace on most cloud GPU hosts is a quota'd network filesystem
#   (e.g. MooseFS, EFS) at ~50-200 MB/s with ms-scale latency.
#   /root is the container's local NVMe overlay at ~3-7 GB/s, µs latency.
#   For SynthSeg's metadata-heavy I/O this is a ~10× speedup.
#   Trade-off: /root is wiped when the pod is *Terminated* (not Stopped).
#
# Assumptions:
#   * Container image: any Python 3.11+, CUDA 12.x, NVIDIA driver >= 525,
#     root access (e.g. runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04).
#   * Container disk size >= 150 GB.
#   * ${WORK_DIR} (default /workspace) contains:
#         ${WORK_DIR}/NeuroQC/             — full repo source
#         ${WORK_DIR}/freesurfer_license.txt
#   * Pod is on-demand (not spot) so we won't be reclaimed mid-install.
#
# Time: ~20-30 min total — most of it is FS install (8-13 GB) + nobrainer
# pip install (12 GB of CUDA wheels under tensorflow[and-cuda]).

set -euo pipefail

WORK_DIR="${WORK_DIR:-/workspace}"
NEUROQC_DIR="${NEUROQC_DIR:-${WORK_DIR}/NeuroQC}"
NOBRAINER_DIR="${NOBRAINER_DIR:-${WORK_DIR}/nobrainer}"

# Local NVMe overlays — much faster than /workspace's MooseFS.
VENV_DIR="${VENV_DIR:-/root/neuroqc-venv}"
SYNTHSEG_DIR="${SYNTHSEG_DIR:-/root/SynthSeg}"
HF_CACHE="${HF_CACHE:-/root/hf_cache}"

FS_VERSION="${FS_VERSION:-8.2.0}"        # primary path: apt .deb on Ubuntu 22
FS_HOME_APT="/usr/local/freesurfer/${FS_VERSION}"
FS_FALLBACK_VERSION="${FS_FALLBACK_VERSION:-7.4.1}"  # tarball path
FS_HOME_TAR="${WORK_DIR}/freesurfer/${FS_FALLBACK_VERSION}"
FS_LICENSE_SRC="${FS_LICENSE_SRC:-${WORK_DIR}/freesurfer_license.txt}"

# PyTorch wheel index — cu118 has the lowest driver floor (NVIDIA 525.x), so
# it works across the widest range of RunPod hosts. Override with
# TORCH_INDEX=https://download.pytorch.org/whl/cu121 if your host has driver
# 530+.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu118}"

log() { printf '[gpu_server_setup %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
section() { printf '\n========================================\n%s\n========================================\n' "$*"; }

# ── Phase 0: GPU + driver check ────────────────────────────────────────────
section "Phase 0 — GPU + driver check"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "ERROR: nvidia-smi missing. Pod was likely launched without a GPU."
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | awk -F. '{print $1}')
log "Detected NVIDIA driver major: ${DRIVER_MAJOR}"
if [ "${DRIVER_MAJOR}" -lt 525 ]; then
    log "WARNING: driver < 525.x — even cu118 PyTorch may fail. Consider another host."
fi

# ── Phase 1: NeuroQC source layout ─────────────────────────────────────────
section "Phase 1 — verify NeuroQC source"
if [ ! -d "${NEUROQC_DIR}" ]; then
    log "ERROR: ${NEUROQC_DIR} not found."
    log "Did you extract neuroqc_runpod.tar.gz under ${WORK_DIR}? Layout expected:"
    log "  ${NEUROQC_DIR}/code/"
    log "  ${NEUROQC_DIR}/scripts/"
    log "  ${NEUROQC_DIR}/data/{ixi,fastmri,abide}/"
    exit 2
fi
log "NeuroQC at ${NEUROQC_DIR}"

# ── Phase 2: nobrainer fork (feat/qc-update branch) ────────────────────────
section "Phase 2 — nobrainer (feat/qc-update)"
if [ ! -d "${NOBRAINER_DIR}" ]; then
    log "Cloning dhritimandas/nobrainer feat/qc-update branch"
    cd "${WORK_DIR}"
    git clone --depth=1 --branch=feat/qc-update https://github.com/dhritimandas/nobrainer.git
else
    log "nobrainer already present at ${NOBRAINER_DIR}"
fi

# ── Phase 3: Python 3.12 + venv on local NVMe ─────────────────────────────
section "Phase 3 — Python 3.12 + venv (${VENV_DIR})"
PY_BIN="$(command -v python3.12 || true)"
if [ -z "${PY_BIN}" ]; then
    log "python3.12 not found — installing via deadsnakes PPA"
    apt-get update -qq
    apt-get install -y software-properties-common >/dev/null
    add-apt-repository -y ppa:deadsnakes/ppa >/dev/null
    apt-get update -qq
    apt-get install -y python3.12 python3.12-venv python3.12-dev >/dev/null
    PY_BIN="$(command -v python3.12)"
fi
log "Using ${PY_BIN} ($(${PY_BIN} --version))"

if [ ! -d "${VENV_DIR}" ]; then
    log "Creating venv at ${VENV_DIR}"
    "${PY_BIN}" -m venv "${VENV_DIR}"
fi
# Symlink convenience: most docs reference $NEUROQC_DIR/.venv.
if [ ! -e "${NEUROQC_DIR}/.venv" ] && [ ! -L "${NEUROQC_DIR}/.venv" ]; then
    ln -s "${VENV_DIR}" "${NEUROQC_DIR}/.venv"
    log "Symlinked ${NEUROQC_DIR}/.venv -> ${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
log "Upgrading pip"
pip install --upgrade pip --quiet

# ── Phase 4: project + nobrainer + base deps ───────────────────────────────
section "Phase 4 — project + nobrainer + base deps"
cd "${NEUROQC_DIR}"
log "Installing nobrainer (editable)"
pip install -e "${NOBRAINER_DIR}" --quiet
log "Installing NeuroQC + transitive deps from pyproject.toml"
pip install -e . --quiet
log "Installing datalad"
pip install datalad --quiet

# ── Phase 5: GPU PyTorch ───────────────────────────────────────────────────
section "Phase 5 — GPU PyTorch (cu118)"
log "Installing torch from ${TORCH_INDEX}"
pip install --upgrade torch torchvision --index-url "${TORCH_INDEX}" --quiet
python -c "
import torch
print('torch', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'PyTorch cannot see CUDA — driver may be too old; try TORCH_INDEX=https://download.pytorch.org/whl/cu121 or upgrade host.'
"

# ── Phase 6: TensorFlow GPU + tf-keras shim ────────────────────────────────
section "Phase 6 — TensorFlow GPU + tf-keras"
log "Installing tensorflow[and-cuda] and tf-keras (Keras-2 shim)"
pip install 'tensorflow[and-cuda]' tf-keras --quiet

# tensorflow[and-cuda]>=2.21 transitively pulls nvidia-*-cu13 wheels whose
# cu13/lib lands FIRST in LD_LIBRARY_PATH (alphabetical) and serves a cuDNN
# ABI that TF (built against CUDA 12.5.1) cannot initialise. Symptoms:
# "No DNN in stream executor" / CUDNN_STATUS_INTERNAL_ERROR. Strip them and
# force-reinstall the cu12 cuDNN (the uninstall removes shared .so files
# that other cu12 packages still need).
log "Removing cu13 NVIDIA wheels and force-reinstalling cu12 cuDNN"
pip uninstall -y nvidia-cudnn-cu13 nvidia-cusparselt-cu13 nvidia-nccl-cu13 \
    nvidia-nvshmem-cu13 2>/dev/null || true
rm -rf "${VENV_DIR}/lib/python"*"/site-packages/nvidia/cu13" 2>/dev/null || true
pip install --force-reinstall nvidia-cudnn-cu12 --quiet

# SynthSeg's predict.py / get_flip_indices does `array_2d[i,j] = np.where(..)[0]`
# which NumPy 2.x rejects ("setting an array element with a sequence" — the
# implicit length-1-array → scalar coercion was removed in 2.0). SynthSeg's
# upstream environment.yml pins numpy=1.24; we accept anything <2.0. TF 2.21
# supports the full 1.x line.
log "Pinning numpy<2.0 (SynthSeg incompatible with NumPy 2.x scalar rules)"
pip install 'numpy<2.0' --quiet

log "TF version + GPU check (in venv, before LD_LIBRARY_PATH patch)"
python -c "
import tensorflow as tf
print('tf', tf.__version__, 'GPUs:', tf.config.list_physical_devices('GPU'))
" || log "WARNING: TF GPU may not be visible yet — Phase 7 patches LD_LIBRARY_PATH."

# ── Phase 7: CUDA library discovery patch ──────────────────────────────────
section "Phase 7 — CUDA library discovery"
ACTIVATE="${VENV_DIR}/bin/activate"
if ! grep -q "CUDA wheel discovery" "${ACTIVATE}"; then
    log "Appending LD_LIBRARY_PATH override to ${ACTIVATE}"
    cat >> "${ACTIVATE}" <<'EOF'

# CUDA wheel discovery: prepend the venv's nvidia/*/lib so tensorflow[and-cuda]
# finds its bundled CUDA 12 user-space libs ahead of the host's CUDA stack.
NVIDIA_LIB_PATHS=$(find "$VIRTUAL_ENV/lib/python"*"/site-packages/nvidia" -type d -name lib 2>/dev/null | sort -u | tr '\n' ':' | sed 's/:$//')
export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}:${LD_LIBRARY_PATH:-}"
EOF
    deactivate
    # shellcheck disable=SC1091
    source "${ACTIVATE}"
else
    log "CUDA discovery already wired in ${ACTIVATE}"
fi
log "Verifying TF GPU detection from venv"
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('TF', tf.__version__, 'GPUs:', gpus)
assert gpus, 'TF cannot see GPU — check LD_LIBRARY_PATH and pod GPU exposure'
"

# ── Phase 8: FreeSurfer (apt .deb preferred, tarball fallback) ─────────────
section "Phase 8 — FreeSurfer ${FS_VERSION}"
FS_HOME=""
if [ -d "${FS_HOME_APT}" ] && [ -f "${FS_HOME_APT}/SetUpFreeSurfer.sh" ]; then
    log "FreeSurfer 8.x already installed at ${FS_HOME_APT}"
    FS_HOME="${FS_HOME_APT}"
elif command -v apt-get >/dev/null 2>&1 && [ "${FS_VERSION%%.*}" -ge 8 ]; then
    log "Installing FreeSurfer ${FS_VERSION} via apt (.deb)"
    apt-get update -qq
    # surfer.nmr.mgh.harvard.edu hosts a .deb under
    # /pub/dist/freesurfer/<v>/freesurfer_ubuntu22-<v>_amd64.deb.
    DEB_URL="https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/${FS_VERSION}/freesurfer_ubuntu22-${FS_VERSION}_amd64.deb"
    DEB_PATH="/tmp/freesurfer_${FS_VERSION}.deb"
    if [ ! -f "${DEB_PATH}" ]; then
        log "Downloading ${DEB_URL}"
        curl -fL --output "${DEB_PATH}" "${DEB_URL}"
    fi
    log "apt-get install -y ${DEB_PATH}"
    apt-get install -y "${DEB_PATH}" >/dev/null \
        || log "apt install reported errors; will check ${FS_HOME_APT} below"
    if [ -f "${FS_HOME_APT}/SetUpFreeSurfer.sh" ]; then
        FS_HOME="${FS_HOME_APT}"
        log "FreeSurfer ${FS_VERSION} installed at ${FS_HOME}"
    fi
fi
if [ -z "${FS_HOME}" ]; then
    log "Falling back to FS ${FS_FALLBACK_VERSION} tarball install"
    if [ -f "${FS_HOME_TAR}/SetUpFreeSurfer.sh" ]; then
        FS_HOME="${FS_HOME_TAR}"
    else
        if [ -z "${FS_LICENSE_PATH:-}" ] && [ -f "${FS_LICENSE_SRC}" ]; then
            export FS_LICENSE_PATH="${FS_LICENSE_SRC}"
        fi
        FS_INSTALL_DIR="${WORK_DIR}/freesurfer" \
        FS_VERSION="${FS_FALLBACK_VERSION}" \
            bash "${NEUROQC_DIR}/scripts/install_freesurfer_linux.sh"
        FS_HOME="${FS_HOME_TAR}"
    fi
fi
# Place the license if not already present.
if [ -f "${FS_LICENSE_SRC}" ] && [ ! -f "${FS_HOME}/license.txt" ]; then
    cp "${FS_LICENSE_SRC}" "${FS_HOME}/license.txt"
    log "Copied license to ${FS_HOME}/license.txt"
fi

# ── Phase 9: BBillot/SynthSeg standalone + sed patches ─────────────────────
section "Phase 9 — BBillot/SynthSeg standalone"
if [ ! -d "${SYNTHSEG_DIR}" ]; then
    log "Cloning BBillot/SynthSeg to ${SYNTHSEG_DIR}"
    git clone --depth=1 https://github.com/BBillot/SynthSeg.git "${SYNTHSEG_DIR}"
else
    log "SynthSeg already at ${SYNTHSEG_DIR}"
fi

log "Patching NumPy-2 deprecations (np.int → int, np.float → float, np.bool → bool)"
# Limit scope to the SynthSeg/ext python files; idempotent.
find "${SYNTHSEG_DIR}/SynthSeg" "${SYNTHSEG_DIR}/ext" -name "*.py" -print0 2>/dev/null | xargs -0 -I{} sed -i \
    -e 's/np\.int\b/int/g' \
    -e 's/np\.float\b/float/g' \
    -e 's/np\.bool\b/bool/g' \
    -e 's/np\.long\b/int/g' \
    {} 2>/dev/null || true

log "Patching Keras-3 import drift (import keras → import tf_keras as keras)"
# tf-keras provides Keras-2 API surface SynthSeg expects; importing the new
# Keras 3 instead would crash on legacy ops. Idempotent grep guard.
find "${SYNTHSEG_DIR}/SynthSeg" "${SYNTHSEG_DIR}/ext" -name "*.py" -print0 2>/dev/null | xargs -0 grep -l "^import keras" 2>/dev/null | while read -r f; do
    if ! grep -q "import tf_keras as keras" "$f"; then
        sed -i 's/^import keras$/import tf_keras as keras/' "$f"
    fi
done
find "${SYNTHSEG_DIR}/SynthSeg" "${SYNTHSEG_DIR}/ext" -name "*.py" -print0 2>/dev/null | xargs -0 grep -l "^from keras" 2>/dev/null | while read -r f; do
    if ! grep -q "from tf_keras" "$f"; then
        sed -i 's/^from keras\b/from tf_keras/' "$f"
    fi
done

# Catch the dotted-import variant — `import keras.layers as KL` mixes a
# Keras-3 module into code that expects tf-keras' Keras-2 Model class,
# triggering "AttributeError: 'KerasTensor' object has no attribute 'node'"
# at graph build time. The two preceding loops only catch `^import keras$`
# and `^from keras\b`, so this dotted form slips through.
find "${SYNTHSEG_DIR}/SynthSeg" "${SYNTHSEG_DIR}/ext" -name "*.py" -print0 2>/dev/null | xargs -0 sed -i \
    -e 's|^import keras\.\([a-zA-Z0-9_]*\)|import tf_keras.\1|' \
    -e 's|^from keras\.\([a-zA-Z0-9_]*\)|from tf_keras.\1|' 2>/dev/null || true

# SynthSeg/predict.py:579 does `lr_indices[i, j] = np.where(labels == lab)[0]`
# which assigns a length-1 array into a scalar slot. Worked under NumPy 1.x's
# implicit coercion; NumPy 2.x raises ValueError. The numpy<2.0 pin in Phase 6
# is the primary fix; this sed is defence-in-depth (safe under both lines and
# survives an accidental numpy upgrade).
log "Patching SynthSeg/predict.py:579 explicit scalar coercion"
if [ -f "${SYNTHSEG_DIR}/SynthSeg/predict.py" ]; then
    sed -i 's|lr_indices\[i, j\] = np\.where(labels_segmentation == lab)\[0\]$|lr_indices[i, j] = np.where(labels_segmentation == lab)[0][0]|' \
        "${SYNTHSEG_DIR}/SynthSeg/predict.py"
fi

log "Symlinking SynthSeg model weights from ${FS_HOME}/models"
SS_MODELS="${SYNTHSEG_DIR}/models"
if [ -d "${FS_HOME}/models" ] && [ ! -L "${SS_MODELS}" ]; then
    if [ -d "${SS_MODELS}" ] && [ ! "$(ls -A "${SS_MODELS}" 2>/dev/null)" ]; then
        rmdir "${SS_MODELS}"
    fi
    if [ ! -e "${SS_MODELS}" ]; then
        ln -s "${FS_HOME}/models" "${SS_MODELS}"
        log "Symlinked ${SS_MODELS} -> ${FS_HOME}/models"
    fi
fi

# ── Phase 10: HuggingFace cache on local NVMe ──────────────────────────────
section "Phase 10 — HuggingFace cache"
mkdir -p "${HF_CACHE}"
log "HF_HOME=${HF_CACHE}"

# ── Phase 11: env-export sentinel ──────────────────────────────────────────
section "Phase 11 — env-export sentinel"
cat > "${WORK_DIR}/neuroqc_env.sh" <<EOF
# Source this at the start of any new shell on this pod to get the full env:
#   source ${WORK_DIR}/neuroqc_env.sh
cd ${NEUROQC_DIR}
source ${VENV_DIR}/bin/activate

# FreeSurfer (used by --mode freesurfer fallback; harmless when mode=python).
export FREESURFER_HOME=${FS_HOME}
set +eu
source \$FREESURFER_HOME/SetUpFreeSurfer.sh > /dev/null
set -eu

# Standalone SynthSeg (the production path; --mode python in run_prototype).
export SYNTHSEG_HOME=${SYNTHSEG_DIR}
export SYNTHSEG_MODE=python

# Keras-2 compat for SynthSeg's TF stack.
export TF_USE_LEGACY_KERAS=1

# Don't pre-allocate the full GPU; saves 5+ min of CUDA ioctls per scan.
export TF_FORCE_GPU_ALLOW_GROWTH=true

# HuggingFace cache on local NVMe (avoids /workspace MooseFS quota).
export HF_HOME=${HF_CACHE}
export HUGGINGFACE_HUB_CACHE=${HF_CACHE}/hub
export TRANSFORMERS_CACHE=${HF_CACHE}/transformers

# DataLad off by default for cloud smokes (working tree is dirty after
# setup; datalad run would refuse).
export USE_DATALAD_RUN=0
EOF
log "Wrote ${WORK_DIR}/neuroqc_env.sh"

# ── Phase 12: smoke-readiness check ────────────────────────────────────────
section "Phase 12 — smoke-readiness"
log "Sourcing FS"
export FREESURFER_HOME="${FS_HOME}"
set +eu
# shellcheck disable=SC1091
source "${FREESURFER_HOME}/SetUpFreeSurfer.sh" > /dev/null
set -eu

log "mri_synthseg:    $(command -v mri_synthseg 2>&1)"
log "Python:          $(command -v python)"
log "nobrainer:       $(python -c 'import nobrainer; print(nobrainer.__file__)')"
log "torch GPU:       $(python -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)')"
log "TF GPU:          $(python -c 'import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))')"
log "SynthSeg CLI:    $(ls -la ${SYNTHSEG_DIR}/scripts/commands/SynthSeg_predict.py 2>&1 | head -1)"
log "SynthSeg models: $(ls -la ${SYNTHSEG_DIR}/models 2>&1 | head -1)"

# ── Phase 12.5: Conv3D smoke ───────────────────────────────────────────────
# Phase 12's TF GPU check only verifies the device is *visible*; it doesn't
# exercise cuDNN. A cu13 ABI mismatch shows up as a successful tf.config call
# followed by CUDNN_STATUS_INTERNAL_ERROR the first time a real op runs. Doing
# a 5-second Conv3D here turns "5+ minutes into Phase 03 before it crashes"
# into "fails immediately with an actionable assert".
section "Phase 12.5 — Conv3D smoke (catches cuDNN ABI issues fast)"
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
assert gpus, 'TF cannot see GPU; check Phase 6 cu13 cleanup + Phase 7 LD_LIBRARY_PATH'
x = tf.random.uniform((1, 16, 16, 16, 1))
y = tf.keras.layers.Conv3D(8, 3, padding='same')(x)
expected = [1, 16, 16, 16, 8]
assert y.shape.as_list() == expected, f'Conv3D returned wrong shape: {y.shape}'
print('Conv3D OK on', gpus[0].name)
"

cat <<EOF

──────────────────────────────────────────────────────────
GPU server setup complete.

Next step (1-scan GPU detection test):

    cd ${NEUROQC_DIR}
    source ${WORK_DIR}/neuroqc_env.sh
    mkdir -p /tmp/synthseg_test
    time python -m SynthSeg.scripts.commands.SynthSeg_predict \\
        --i \$(ls data/abide/*.nii.gz | head -1) \\
        --o /tmp/synthseg_test/out.nii.gz --parc --fast \\
        --vol /tmp/synthseg_test/vol.csv --qc /tmp/synthseg_test/qc.csv

  Watch GPU in another terminal:
    watch -n 1 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv && nvidia-smi --query-gpu=utilization.gpu --format=csv'

  Decision:
    * < 1 min wall-clock + GPU memory in use → GPU works, run full smoke.
    * > 5 min + GPU memory empty            → CPU only; verify Phase 7 LD_LIBRARY_PATH.

If GPU works, stage data to /root and launch the smoke (~50-60 min on A100):

    bash ${NEUROQC_DIR}/scripts/runpod_stage_data.sh

    # Launch under tmux + auto-restart so SSH disconnects don't kill the run.
    # The helper sources neuroqc_env.sh, retries up to 5x on transient failure,
    # and prints final phase markers on completion.
    tmux new-session -d -s smoke 'bash ${NEUROQC_DIR}/scripts/gpu_server_launch_smoke.sh'
    tmux attach -t smoke   # detach with Ctrl-B D; reconnect any time

  If you don't have tmux available and accept the SSH-drop risk, the legacy
  nohup invocation still works:
    nohup bash -c '
        source ${WORK_DIR}/neuroqc_env.sh
        SYNTHSEG_MODE=python NUM_REFS_CALIB=15 NUM_REFS_SMOKE=15 \\
        IXI_INPUT_DIR=/root/data/ixi/raw \\
        FASTMRI_INPUT_DIR=/root/data/fastmri/nifti \\
        ABIDE_INPUT_DIR=/root/data/abide \\
            bash scripts/run_prototype.sh
    ' > ${WORK_DIR}/smoke.log 2>&1 &
    echo \$! > ${WORK_DIR}/smoke.pid
    disown
    tail -f ${WORK_DIR}/smoke.log
──────────────────────────────────────────────────────────
EOF
