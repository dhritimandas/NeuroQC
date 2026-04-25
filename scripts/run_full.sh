#!/bin/bash
# scripts/run_full.sh
# Run on Jarvis Labs A100. Full pipeline, all phases.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== NeuroQC Full Pipeline (Jarvis A100) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no GPU detected')"
echo "Start: $(date)"

# Start watcher in background (updates figures as CSVs land)
bash scripts/watch_results.sh &
WATCHER_PID=$!
trap "kill $WATCHER_PID 2>/dev/null || true" EXIT

# Phases 1-2 already done locally before push to Jarvis
# Phase 3: SynthSeg + preferences + IQMs
make phase3

# Phase 4: VLM evaluation (3 seeds for main results)
for seed in 42 1337 2024; do
    python code/08a_eval_3d_vlms.py --seed "$seed" --subsample-severities 1,5
    python code/08b_eval_2d_vlms.py --seed "$seed" --subsample-severities 1,5
done
python code/results_tracker.py --phase 4

# Phase 5: LoRA fine-tuning (single seed for compute budget)
python code/09_finetune_lora.py --seed 42 --model llava-ov --input-type 2d
python code/09_finetune_lora.py --seed 42 --model m3d-lamed --input-type 3d
python code/results_tracker.py --phase 5

# ABIDE zero-shot holdout
python code/09b_download_abide.py
python code/10_eval_abide_zeroshot.py --model-name best_vlm --seed 42

# Figures + tables
python code/visualize.py --all
python code/results_tracker.py --phase tables

# DataLad commit
datalad save -m "Full pipeline complete" results/ figures/

echo "=== Pipeline done: $(date) ==="