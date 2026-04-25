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
| **FreeSurfer 8.1.0+** | Phase 03 SynthSeg        | `bash scripts/install_freesurfer_macos.sh` (or `_linux.sh`); requires a free license at <https://surfer.nmr.mgh.harvard.edu/registration.html> |
| **AWS CLI**           | Phase 09b S3 acquisition | `brew install awscli` (macOS) / `pip install awscli` |
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

* Every phase script writes a manifest with seed + git hash + library
  versions where applicable. Phase 09 produces a `finetune_run_info_*.json`
  with full hyperparameters + bucket distributions.
* `scripts/bundle_results.py` packages results CSVs + figures + provenance
  JSONs into a tar.gz with per-file SHA-256 manifest, suitable for
  Zenodo-style reproducibility deposits or collaborator handoffs.
* The full test suite runs on CPU in ~25 seconds; heavy GPU paths are
  mocked. Adapter classes for the 5 VLMs are testable via the module-level
  `_ADAPTERS` registry.

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

## Citation

If you use this codebase, please cite the corresponding NeuroQC paper
(NeurIPS 2026 submission; bibtex pending camera-ready).

## License

MIT — see `pyproject.toml` for canonical declaration.
