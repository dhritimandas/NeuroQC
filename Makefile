PYTHON := python
DATALAD := datalad run

# Directories
DATA_DIR := data
RESULTS_DIR := results
FIGURES_DIR := figures

# Phase markers (touch files to track completion)
PHASES_DIR := .phases
$(shell mkdir -p $(PHASES_DIR))

.PHONY: all clean status prototype full phase1 phase2 phase3 phase4 phase5 figures tables help

# ─── Top-level targets ───

all: phase1 phase2 phase3 phase4 phase5 figures tables

prototype: ## Run pipeline on 10 scans (DANDI Hub)
	bash scripts/run_prototype.sh

full: ## Run full pipeline (Jarvis Labs A100)
	bash scripts/run_full.sh

status: ## Show pipeline status
	$(PYTHON) code/results_tracker.py --phase status

# ─── Individual phases ───

phase1: $(PHASES_DIR)/.phase1 ## Curate reference scans
$(PHASES_DIR)/.phase1:
	$(DATALAD) -m "Phase 1: Curate references" \
		-i "data/ixi/raw/*.nii.gz" \
		-o "data/ixi/references/" \
		-o "results/tables/reference_manifest.csv" \
		"$(PYTHON) code/01_curate_references.py \
			--input-dir data/ixi/raw \
			--output-dir data/ixi/references"
	touch $@

phase2: phase1 $(PHASES_DIR)/.phase2 ## Generate corruptions
$(PHASES_DIR)/.phase2:
	$(DATALAD) -m "Phase 2: Generate corruptions" \
		-i "data/ixi/references/*.nii.gz" \
		-o "data/ixi/corrupted/" \
		-o "results/tables/corruption_manifest.csv" \
		"$(PYTHON) code/02_generate_corruptions.py \
			--input-dir data/ixi/references \
			--output-dir data/ixi/corrupted"
	touch $@

phase3: phase2 $(PHASES_DIR)/.phase3 ## SynthSeg + preferences + IQMs
$(PHASES_DIR)/.phase3:
	$(DATALAD) -m "Phase 3a: SynthSeg segmentation" \
		-i "data/ixi/references/*.nii.gz" \
		-i "data/ixi/corrupted/" \
		-o "data/derivatives/synthseg/" \
		"$(PYTHON) code/03_run_synthseg.py \
			--input-dir data/ixi/references \
			--output-dir data/derivatives/synthseg && \
		 $(PYTHON) code/03_run_synthseg.py \
			--input-dir data/ixi/corrupted \
			--output-dir data/derivatives/synthseg"
	$(DATALAD) -m "Phase 3b: Compute machine preference" \
		-i "data/derivatives/synthseg/" \
		-o "results/tables/machine_preference.csv" \
		"$(PYTHON) code/04_compute_preference.py"
	$(DATALAD) -m "Phase 3c: Extract IQMs" \
		-o "results/tables/iqm_features.csv" \
		"$(PYTHON) code/05_extract_iqms.py"
	$(PYTHON) code/results_tracker.py --phase 3
	$(PYTHON) code/visualize.py --figure 2 --figure 3 --figure 8
	datalad save -m "Phase 3 complete: ground truth + IQMs + initial figures"
	touch $@

phase4: phase3 $(PHASES_DIR)/.phase4 ## VLM evaluation
$(PHASES_DIR)/.phase4:
	$(DATALAD) -m "Phase 4a: 3D VLM evaluation" \
		-o "results/tables/3d_vlm_scores.csv" \
		"$(PYTHON) code/08a_eval_3d_vlms.py --subsample-severities 1,5"
	$(DATALAD) -m "Phase 4b: 2D VLM evaluation" \
		-o "results/tables/2d_vlm_scores.csv" \
		"$(PYTHON) code/08b_eval_2d_vlms.py --subsample-severities 1,5"
	$(PYTHON) code/results_tracker.py --phase 4
	$(PYTHON) code/visualize.py --figure 4 --figure 5
	datalad save -m "Phase 4 complete: VLM evaluation + figures"
	touch $@

phase5: phase4 $(PHASES_DIR)/.phase5 ## Fine-tuning
$(PHASES_DIR)/.phase5:
	$(DATALAD) -m "Phase 5: LoRA fine-tuning (2D)" \
		-o "results/checkpoints/llava_ov_lora/" \
		-o "results/tables/finetuned_scores.csv" \
		"$(PYTHON) code/09_finetune_lora.py --model llava-ov --input-type 2d"
	$(DATALAD) -m "Phase 5: LoRA fine-tuning (3D)" \
		-o "results/checkpoints/m3d_lamed_lora/" \
		"$(PYTHON) code/09_finetune_lora.py --model m3d-lamed --input-type 3d"
	$(PYTHON) code/results_tracker.py --phase 5
	$(PYTHON) code/visualize.py --figure 6
	datalad save -m "Phase 5 complete: fine-tuned models + figures"
	touch $@

phase6: phase5 $(PHASES_DIR)/.phase6 ## ABIDE zero-shot clinical holdout
$(PHASES_DIR)/.phase6:
	$(DATALAD) -m "Phase 6: ABIDE download + zero-shot eval" \
		-i "data/abide/ratings/*.csv" \
		-o "data/abide/raw/" \
		-o "results/tables/abide_zeroshot_predictions.csv" \
		"$(PYTHON) code/09b_download_abide.py && \
		 $(PYTHON) code/10_eval_abide_zeroshot.py \
			--abide-dir data/abide/raw \
			--ratings-csv data/abide/ratings/y_abide.csv"
	touch $@

figures: phase5 ## Generate all figures
	$(PYTHON) code/visualize.py --all
	datalad save -m "All figures generated"

tables: phase5 ## Generate LaTeX tables
	$(PYTHON) code/results_tracker.py --phase tables
	datalad save -m "LaTeX tables generated"

# ─── Utilities ───

clean-phases: ## Reset phase markers (rerun everything)
	rm -f $(PHASES_DIR)/.phase*

clean-results: ## Remove computed results (keep data)
	rm -rf results/tables/*.csv results/metrics/*.csv results/latex/
	rm -rf figures/*.pdf figures/*.png figures/*.html
	rm -f $(PHASES_DIR)/.phase*

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
