# NeuroQC

**Can multimodal large language models predict when neuroimaging pipelines
fail?** NeuroQC is a research codebase for evaluating vision-language models
(VLMs) as quality-control oracles on brain MRI, against (a) machine-derived
ground truth (SynthSeg-Dice degradation under controlled corruption) and
(b) human expert ratings (ABIDE-I, three raters per scan).

This repository contains the **code** for the full pipeline — from raw
NIfTI extraction through corruption, segmentation, IQM extraction,
zero-shot VLM scoring, LoRA fine-tuning, and figure / table generation.
The data and derived artifacts are excluded by design (see
[Data not included](#data-not-included) below).

## Pipeline overview

The pipeline is a sequence of numbered phases under `code/`. Each phase
consumes outputs of upstream phases and writes to `results/tables/` (or
`data/derivatives/` for segmentations). Every phase is resume-aware:
re-running picks up where the last run stopped.

```
00 extract NIfTI ──┐
                   │
02 corrupt ────────┤            (TorchIO + k-space motion)
                   ▼
03 SynthSeg ──► 03b thickness ──► 04 preference (Dice + thickness shift)
                                  │       │
                                  ▼       ▼
                             05 IQMs    05b SynthSeg-QC aggregation
                                  │
                                  ▼
                  08a Phase A: build benchmark subsample
                                  │
                  ┌───────────────┼───────────────┐
                  ▼               ▼               ▼
            08a Phase B       08b              09 LoRA fine-tune
            (3D VLMs)         (2D VLMs)
                  │               │               │
                  └───────┬───────┴───────────────┘
                          ▼
                       visualize.py  ──►  figures/*.{png,svg}

External validation:
09b acquire ABIDE ──► 10 zero-shot eval ──► 11 cross-model meta-table
```

## Repository structure

```
neuroqc/
├── code/                          # Pipeline scripts (phase-prefixed)
│   ├── 00_extract_fastmri_t1.py   # FastMRI HDF5 → NIfTI
│   ├── 01_curate_references.py    # Reference manifest builder
│   ├── 02_generate_corruptions.py # Image-space corruption (TorchIO)
│   ├── 02b_corrupt_kspace_motion.py # K-space motion (Shaw 2019)
│   ├── 03_run_synthseg.py         # SynthSeg whole-brain seg
│   ├── 03b_compute_thickness.py   # DiReCT cortical thickness
│   ├── 04_compute_preference.py   # Per-pair Dice + thickness shift
│   ├── 04_filter_ref_quality.py   # Reference-scan QC gate
│   ├── 05_extract_iqms.py         # mriqc-style IQMs (SNR/CNR/EFC/FBER/CJV)
│   ├── 05b_aggregate_synthseg_qc.py # SynthSeg log-loss aggregation
│   ├── 08a_eval_3d_vlms.py        # 3D VLM zero-shot eval (M3D-LaMed)
│   ├── 08b_eval_2d_vlms.py        # 2D VLM zero-shot (LLaVA-OV/Qwen2-VL/MedGemma/GPT-4o)
│   ├── 09_finetune_lora.py        # QLoRA fine-tune (4-bit + LoRA)
│   ├── 09b_acquire_abide.py       # ABIDE-I acquisition (mriqc-learn + S3)
│   ├── 10_eval_abide_zeroshot.py  # ABIDE zero-shot eval (5 models, 3 variants)
│   ├── 11_compare_abide_zeroshot.py # Cross-model meta-table + DeLong
│   ├── visualize.py               # Publication figures (PNG + SVG)
│   ├── results_tracker.py         # Metrics + status dashboard
│   ├── diagnose_synthseg_on_fastmri.py # Per-scan QC diagnostic
│   └── verify_02b_kspace.py       # 02b sanity-check harness
├── tests/                         # pytest unit tests (one per code/ script)
├── scripts/                       # Setup + orchestration shell scripts
│   ├── bundle_results.py          # Reproducibility tar.gz packer
│   ├── install_freesurfer_*.sh    # FreeSurfer install helpers
│   ├── run_full.sh                # Full-batch driver (Jarvis A100)
│   ├── run_prototype.sh           # DANDI Hub CPU calibration prototype
│   ├── run_analysis_only.sh       # Analysis-only re-run (no inference)
│   ├── setup_dandi.sh             # DANDI Hub env setup
│   ├── setup_jarvis.sh            # Jarvis Labs A100 env setup
│   ├── verify_install.py          # Post-install sanity check
│   └── watch_results.sh           # Continuous status monitor
├── pyproject.toml                 # Dependencies + project metadata
├── Makefile                       # Phase orchestration targets
├── CLAUDE.md                      # Project conventions + library hierarchy
└── README.md                      # This file
```

## Installation

NeuroQC requires **Python 3.11+** and (for the GPU phases) a CUDA-enabled
PyTorch build. The project uses standard Python packaging via
`pyproject.toml`.

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/dhritimandas/NeuroQC.git
cd NeuroQC
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -e .                # runtime deps (production)
pip install -e ".[dev]"         # + pytest, ruff, black (development)
```

This installs everything from `pyproject.toml`'s `[project.dependencies]`
section: torch, transformers, peft, monai, torchio, nibabel, mriqc-learn,
boto3, krippendorff, scipy, scikit-learn, matplotlib, seaborn, typer, rich,
tqdm, openai (and on Linux, bitsandbytes for 4-bit quantization).

### 3. External tools (required for some phases)

| Tool                  | Required for             | Install                                         |
| --------------------- | ------------------------ | ----------------------------------------------- |
| **FreeSurfer**        | Phase 03 SynthSeg        | macOS: 8.1.0+ via `bash scripts/install_freesurfer_macos.sh`. Linux: 8.2.0 via apt `.deb` (Ubuntu 22 — preferred) or 7.4.1 tarball via `bash scripts/install_freesurfer_linux.sh`. All paths require a free license at <https://surfer.nmr.mgh.harvard.edu/registration.html>. Note: FreeSurfer's bundled SynthSeg is CPU-only on modern hosts (FS team has deprecated GPU support); for GPU runs, use the standalone `BBillot/SynthSeg` clone — see `docs/runpod_setup.md`. |
| **AWS CLI**           | Phase 09b S3 acquisition | `brew install awscli` (macOS) / `pip install awscli` |
| **DataLad** + git-annex | Canonical orchestration via `Makefile` and `scripts/run_*.sh` (per-phase `datalad save`, `datalad run` provenance). Optional if you invoke phase scripts directly. | `brew install datalad git-annex` (macOS) / `pip install datalad && conda install -c conda-forge git-annex` (Linux) |
| **NVIDIA GPU + CUDA** | Phase 08a/08b/09 (VLM inference + fine-tuning) | Set `CUDA_VISIBLE_DEVICES`; bf16 + bitsandbytes 4-bit not bit-deterministic across hardware |
| **OpenAI API key**    | Phase 08b/10 GPT-4o head | `export OPENAI_API_KEY=...` |

### 4. Verify install

```bash
python scripts/verify_install.py
pytest -q                        # 131 tests; ~25 sec
```

If `pytest` is green, the codebase is in working order. Heavy GPU paths
are mocked in tests (real model weights are never loaded), so this works
on CPU-only laptops.

## Cloud GPU deployment

The full pipeline (Phases 03 SynthSeg, 08 VLM eval, 09 LoRA fine-tune)
benefits from a GPU. We run cloud bootstraps on two platforms; setup
guides for each are kept alongside this repo (gitignored — request from
the maintainer for now):

- `docs/runpod_setup.md` — A100 SXM 80 GB on RunPod, the production
  target. `scripts/runpod_setup.sh` is a one-shot bootstrap that clones
  the standalone `BBillot/SynthSeg` repo, applies portability patches
  (NumPy-2 / Keras-3), installs cu118 PyTorch + tensorflow[and-cuda] +
  tf-keras, sources FreeSurfer 8.2.0 (apt `.deb`), and writes an env
  sentinel that re-establishes the full state on any new shell.
  `scripts/runpod_stage_data.sh` then copies hot data to local NVMe
  (avoiding RunPod's MooseFS-backed `/workspace`, which is ~10× slower
  for SynthSeg's metadata-heavy I/O).
- `docs/dandi_hub_setup.md` — DANDI Hub T4 GPU image, free for
  academic use. Different friction (idle culler + EFS), same end-state.

The smoke run (`scripts/run_prototype.sh`) is platform-agnostic — set
`SYNTHSEG_MODE=python` for GPU, `SYNTHSEG_MODE=freesurfer` for CPU
fallback. Default is `python`. See env vars in the script header.

## Running the pipeline

### Single-scan example

To run the full ground-truth pipeline (Phases 00 → 04) on a single FastMRI
scan, then score it with one VLM:

```bash
# Phase 00 — extract a single FastMRI HDF5 to NIfTI
python code/00_extract_fastmri_t1.py \
    --input-h5      data/fastmri/raw/file_brain_AXT1_201_6002725.h5 \
    --output-dir    data/fastmri/nifti/

# Phase 02 — generate one corruption (motion sev 1)
python code/02_generate_corruptions.py \
    --input-dir     data/fastmri/nifti/ \
    --output-dir    data/derivatives/corruptions/ \
    --corruption-type motion --severity 1 \
    --manifest      results/tables/corruption_manifest.csv

# Phase 03 — SynthSeg --parc on the ref + corrupted scan (~36 min on CPU)
python code/03_run_synthseg.py \
    --input-dir     data/fastmri/nifti/ \
    --output-dir    data/derivatives/synthseg/ \
    --freesurfer-home /Applications/freesurfer/8.1.0

# Phase 03b — DiReCT-style cortical thickness
python code/03b_compute_thickness.py \
    --synthseg-dir  data/derivatives/synthseg/ \
    --output-file   results/tables/cortical_thickness.csv \
    --synthseg-manifest data/derivatives/synthseg/synthseg_manifest.csv \
    --freesurfer-home /Applications/freesurfer/8.1.0

# Phase 04 — Per-pair Dice + thickness shift
python code/04_compute_preference.py \
    --corruption-manifest results/tables/corruption_manifest.csv \
    --synthseg-manifest   data/derivatives/synthseg/synthseg_manifest.csv \
    --thickness-file      results/tables/cortical_thickness.csv \
    --output-file         results/tables/machine_preference.csv \
    --per-structure-output results/tables/per_structure_dice.csv

# Phase 05 — mriqc-style IQMs (SNR/CNR/EFC/FBER/CJV)
python code/05_extract_iqms.py \
    --ref-manifest        results/tables/reference_manifest.csv \
    --cor-manifest        results/tables/corruption_manifest.csv \
    --synthseg-manifest   data/derivatives/synthseg/synthseg_manifest.csv \
    --output-file         results/tables/iqm_features.csv

# Phase 08a — Score with a 3D VLM (M3D-LaMed, requires GPU)
python code/08a_eval_3d_vlms.py \
    --seed 0 --models m3d_lamed \
    --ref-manifest        results/tables/reference_manifest.csv \
    --cor-manifest        results/tables/corruption_manifest.csv \
    --preference-csv      results/tables/machine_preference.csv \
    --output-file         results/tables/3d_vlm_scores_seed_0.csv
```

### Batch example (full pipeline, all corruptions, all severities)

For a real run on a corpus of scans, use the orchestration shell:

```bash
# Full pipeline on Jarvis Labs A100:
bash scripts/run_full.sh

# Or, smaller calibration prototype on DANDI Hub (CPU-only, ~16-21 hr):
bash scripts/run_prototype.sh

# Custom batch:
NUM_REFS=10 SEVERITIES="1,3,5" CORRUPTIONS="motion,ghosting" \
  bash scripts/run_prototype.sh
```

The `run_*.sh` scripts invoke each phase in sequence with the right flags
and resume-aware behavior. They all converge on the same canonical CSV
schema in `results/tables/`.

### External validation on ABIDE-I

```bash
# Phase A: load mriqc-learn ratings + IQMs (1101 scans, no network — always
# runs)
python code/09b_acquire_abide.py --skip-download

# Phase B: download raw T1w volumes from canonical S3 (~6.8 GB, ~10 min,
# anonymous read)
python code/09b_acquire_abide.py --acquisition-path fcp-indi-raw

# Zero-shot evaluation against 3-rater consensus (5 VLMs in parallel)
python code/10_eval_abide_zeroshot.py --seed 0

# Cross-model comparison + LaTeX table for the paper
python code/11_compare_abide_zeroshot.py
```

### Generating figures

```bash
# All 6 figures (PNG + SVG, atomic write):
python code/visualize.py --all

# A single figure with custom output dir:
python code/visualize.py --figure 4 --output-dir paper/figures/
```

## CLI commands at a glance

Every phase script is a Typer CLI; `--help` always works:

```bash
python code/<script>.py --help
```

Common flags shared across phases:

| Flag                  | Purpose                                                |
| --------------------- | ------------------------------------------------------ |
| `--seed INT`          | Torch / Python RNG seed; recorded in output rows.      |
| `--dry-run`           | Load inputs, print plan, exit before any inference.    |
| `--max-scans INT`     | Cap N for smoke / dev runs (0 = unlimited).            |
| `--output-file PATH`  | Aggregate output CSV.                                  |
| `--max-budget-usd FLOAT` | (GPT-4o paths) hard ceiling on API spend.           |

## Reproducibility

NeuroQC treats reproducibility as a first-class engineering concern. The
recipe has five layers:

**1. Pinned dependency versions.** `pyproject.toml` declares minor-version
floors for the entire scientific stack (numpy 2.x, torch ≥ 2.4,
transformers ≥ 4.45, monai ≥ 1.4, …). bf16 + bitsandbytes 4-bit
quantization is best-effort bit-deterministic across hardware and is noted
as such in the dependency block.

**2. Seed pinning + per-phase provenance manifests.** Every phase script
takes a `--seed` flag (default 0) that pins Python / NumPy / Torch RNGs,
then records the seed alongside the input git hash and the resolved
library versions in the output CSV / JSON. Phase 09 (LoRA fine-tune)
emits a `finetune_run_info_seed_*.json` with full hyperparameters,
bucket distributions, and resume points. Phase 10 (ABIDE zero-shot)
emits an `abide_zeroshot_summary_{model}_seed_*.json` with bootstrap
seed, n_bootstrap, threshold-in-sample disclosure, and per-site /
per-rater coverage.

**3. DataLad-versioned data + derivatives** (used by the canonical
orchestration in `Makefile` and `scripts/run_*.sh`). Each phase target
wraps execution with `datalad run -i <inputs> -o <outputs> --` so the
derived files (`data/derivatives/synthseg/`, `results/tables/*.csv`,
`figures/*.png`) carry git-annex-tracked provenance: the exact input
hashes, the command that produced them, and the env. After each phase
the runner calls `datalad save -m "Phase N complete"`, producing one
versioned commit per phase. **DataLad is optional for users who invoke
the phase scripts directly** — the scripts are pure Python and run
without DataLad — but the canonical re-run uses it.

**4. Reproducibility archive.** `scripts/bundle_results.py` packages all
results CSVs + figures + provenance JSONs into a `bundles/<tag>.tar.gz`
with a per-file SHA-256 manifest, suitable for Zenodo-style deposits or
collaborator handoffs without shipping the raw data.

**5. Behavior-locked test suite.** `pytest -q` runs the full 131-test
suite on CPU in ~25 seconds. Heavy GPU paths (VLM inference, LoRA
fine-tuning) are mocked via the module-level `_ADAPTERS` registry — real
model weights are never loaded. Bootstrap-CI tests assert byte-identical
output for the same seed; consensus-variant tests assert the documented
tie-breaking. The suite functions as a behavioral lock against silent
metric drift.

## Data not included

This repository ships **code only**. The following are excluded by design:

* `data/` — raw and corrupted scan volumes (~7 GB ABIDE alone).
* `data/derivatives/` — SynthSeg outputs, slice caches.
* `results/` — analytical CSVs / JSONs (regenerable by re-running the pipeline).
* `figures/` — generated figures.
* `paper/`, `docs/` — manuscript sources and design docs.

To reproduce the data side: run `code/00_extract_fastmri_t1.py` on FastMRI
HDF5 inputs (subscription required) and `code/09b_acquire_abide.py` on the
public ABIDE-I S3 mirror. Both scripts are anonymous-read; no credentials
needed for ABIDE.

## Troubleshooting common issues

Operational issues we've hit and their fixes. Pipeline-specific issues
are in the per-phase script docstrings.

| Symptom | Fix |
| --- | --- |
| Phase 01 reports `0 passed` for FastMRI | The H5→NIfTI extractor writes RSS magnitudes verbatim (max ≈ 10⁻³), which fails `MIN_MAX_INTENSITY=100`. Re-extract with `python code/00_extract_fastmri_t1.py --rescale-intensity --input-dir … --output-dir …` (default ON). For already-extracted NIfTIs, use the in-place rescale snippet in `scripts/runpod_stage_data.sh`. |
| Phase 03 SynthSeg `--mode python` fails with `model path does not exist` | Standalone SynthSeg doesn't bundle weights. Symlink them from FreeSurfer: `ln -s $FREESURFER_HOME/models <SynthSeg-clone>/models`. `scripts/runpod_setup.sh:Phase 9` automates this. |
| `RuntimeError: NVIDIA driver too old` (PyTorch) | The default `pip install torch` pulls cu121 wheels (need driver 530+). Reinstall with the cu118 index (works on driver 525+): `pip install torch torchvision --upgrade --index-url https://download.pytorch.org/whl/cu118`. |
| Phase 03 wall-clock unexpectedly multi-hour | Confirm `--mode python` (not `--mode freesurfer`) — FreeSurfer's bundled SynthSeg is CPU-only on modern hosts. Also verify TF sees the GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`. |
| RunPod SSH `Permission denied (publickey)` | Add your SSH pubkey via RunPod **Settings → SSH Keys** (account-level — injected into every new pod). Or use the Web Terminal once: `cat <pubkey> >> ~/.ssh/authorized_keys`. |
| TensorFlow allocates all GPU memory at start | `export TF_FORCE_GPU_ALLOW_GROWTH=true` — included in `scripts/runpod_setup.sh`'s env sentinel. |
| FreeSurfer setup script trips `set -euo pipefail` | `SetUpFreeSurfer.sh` uses unbound vars and runs internal tests with non-zero returns. Wrap with `set +eu; source $FREESURFER_HOME/SetUpFreeSurfer.sh; set -eu`. |
| Pipeline I/O slow on cloud GPU instance | If working dir is a network filesystem (RunPod `/workspace`, DANDI Hub `/home/<user>`), stage hot data to a local-disk path. Per-scan SynthSeg I/O can be ~10× faster on local NVMe than network FS. |
| `pytest` fails with `command 'git-annex' not found` | DataLad-managed repos require `git-annex` to commit. Install via Homebrew (macOS): `brew install git-annex`. Linux: `conda install -c conda-forge git-annex` or `apt install git-annex`. |

## Citation

If you use this codebase, please cite the corresponding NeuroQC paper
(NeurIPS 2026 submission; bibtex pending camera-ready).

## License

MIT — see `pyproject.toml` for canonical declaration.
