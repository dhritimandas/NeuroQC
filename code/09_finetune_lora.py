#!/usr/bin/env python3
"""NeuroQC Phase 09 — LoRA fine-tune VLMs on machine preference labels (RQ3).

Same 4 model architectures as 08a/08b (M3D-LaMed for 3D; LLaVA-OV, Qwen2-VL,
MedGemma for 2D) so the fine-tuned-vs-zero-shot comparison stays paired
on the shared ``benchmark_subsample.csv``. QLoRA recipe (4-bit base + LoRA
on LLM attention projections + optional smaller LoRA on the vision encoder)
with a two-head joint output string ``"Quality: N Thickness: M"``.

Inputs (all required; abort with clear error on any missing):
    --subsample-manifest   results/tables/benchmark_subsample.csv (08a Phase A).
    --preference-csv       results/tables/machine_preference.csv (Phase 04).
    --thickness-csv        results/tables/cortical_thickness.csv (Phase 03b).

Join semantics:
    benchmark.merge(thickness, on="scan_path", how="left")
    benchmark.merge(preference, left_on="scan_path", right_on="cor_path", how="left")
    NOT ``on="scan_path"`` for preference — that frame is per-pair, no scan_path.

Clean-row handling:
    is_clean=True rows (no preference match by construction) get the
    structural targets ``mean_dice = 1.0`` (Sørensen-Dice idempotence) and
    ``thickness_shift = 0.0`` (definitional — shift relative to clean baseline).
    Defended in docs/plan.md §"Reference (clean-row) handling rule".

Signed thickness shift:
    Computed FRESH from cortical_thickness.csv:mean_thickness — NOT reused
    from machine_preference.csv:thickness_shift (which is unsigned per-region
    nanmean). For each ref_id, looks up the clean row's mean_thickness and
    subtracts: ``mean_thickness - clean_thickness`` (negative = thinning,
    positive = thickening). Bucketed on absolute value.

Bucket discretization (defaults; calibrate post-data-landing):
    Dice → 1-5: 5 if ≥0.95; 4 if [0.90,0.95); 3 if [0.80,0.90); 2 if [0.60,0.80); 1 otherwise.
    |thickness_shift| (mm) → 1-5: 5 if <0.05; 4 if <0.15; 3 if <0.30; 2 if <0.50; 1 if ≥0.50.
    Per-bucket train counts are logged at startup; a warning fires if any
    bucket has < 20 examples. Calibration is operational, post-data.

Output target string format:
    "Quality: <N> Thickness: <M>"   where N, M ∈ {1, 2, 3, 4, 5}.
    Standard cross-entropy on the joint output tokens (no per-head
    reweighting). The eval-time parser is
    ``nobrainer.qc.evaluate.parse_dual_qc_response``; absence of that
    function aborts the script with an actionable error.

Adapter implementations:
    M3DLamedAdapter is the primary path (mirrors the 08a inference adapter
    plus a training-side ``prepare_inputs`` that emits an HF-compatible
    ``{input_ids, attention_mask, pixel_values, labels}`` dict). The 2D
    adapters (LLaVA-OV, Qwen2-VL, MedGemma) ship as scaffolds that raise
    ``NotImplementedError`` from ``prepare_inputs`` so the structural code
    is testable end-to-end before each model's HF processor surface gets
    GPU-verified. Same scaffold pattern as 08a's RadFM/Med-2E3 adapters.

PEFT-version caveat:
    The ``--lora-vision-encoder`` flag relies on ``peft_model.add_adapter``
    (PEFT ≥ 0.7). The script asserts the version at module load and aborts
    with an upgrade message otherwise.

Determinism caveat:
    bf16 + bitsandbytes 4-bit quantisation are not bit-deterministic across
    hardware. Reproducibility is best-effort; the per-run provenance JSON
    captures library versions, git hash, and seeds for replication.

Outputs:
    results/checkpoints/{model}_lora_seed_{seed}/                (HF Trainer)
    results/tables/finetuned_scores_seed_{seed}.csv              (per scan)
    results/tables/finetune_run_info_{model}_seed_{seed}.json    (provenance)
    results/tables/finetune_diff_{timestamp}.diff                (if git dirty)
"""

from __future__ import annotations

import csv
import gc
import json
import logging
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

import pandas as pd
import torch
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from tqdm import tqdm

# ──────────────────────────────────────────────
# Constants — input schemas
# ──────────────────────────────────────────────

# benchmark_subsample.csv (per-scan, written by 08a Phase A)
SCAN_COLUMN: str = "scan_path"
REF_ID_COLUMN: str = "ref_id"
IS_CLEAN_COLUMN: str = "is_clean"
SPLIT_COLUMN: str = "split"

# machine_preference.csv (per-pair, written by Phase 04)
_PREF_REF_PATH: str = "ref_path"
_PREF_COR_PATH: str = "cor_path"
_PREF_MEAN_DICE: str = "mean_dice"

# cortical_thickness.csv (per-scan, written by Phase 03b)
_THICKNESS_SCAN_PATH: str = "scan_path"
_THICKNESS_MEAN: str = "mean_thickness"

# Derived columns (post-join)
_DERIVED_DICE: str = "mean_dice"
_DERIVED_SIGNED_SHIFT: str = "thickness_shift_signed"
_DERIVED_DICE_BUCKET: str = "dice_quality"
_DERIVED_THICKNESS_BUCKET: str = "thickness_quality"

# Split labels (must match what 08a writes)
_TRAIN: str = "train"
_VAL: str = "val"
_TEST: str = "test"

# Output CSV (Phase 09 eval) schema
MODEL_COLUMN: str = "model"
DICE_Q_COLUMN: str = "dice_quality"
THICK_Q_COLUMN: str = "thickness_quality"
RAW_RESPONSE_COLUMN: str = "raw_response"
SEED_COLUMN: str = "seed"
OUTPUT_COLUMNS: tuple[str, ...] = (
    SCAN_COLUMN,
    MODEL_COLUMN,
    DICE_Q_COLUMN,
    THICK_Q_COLUMN,
    RAW_RESPONSE_COLUMN,
    SEED_COLUMN,
)

# ──────────────────────────────────────────────
# Constants — bucket boundaries
# ──────────────────────────────────────────────

# Dice → 1-5; descending sort lets the first-True branch win.
# 5 if mean_dice ≥ 0.95; 4 if ≥ 0.90; 3 if ≥ 0.80; 2 if ≥ 0.60; 1 otherwise.
DICE_BUCKET_THRESHOLDS: tuple[tuple[int, float], ...] = (
    (5, 0.95),
    (4, 0.90),
    (3, 0.80),
    (2, 0.60),
)
# Absolute thickness shift (mm) → 1-5. Smaller |shift| = better preserved.
THICKNESS_BUCKET_THRESHOLDS: tuple[tuple[int, float], ...] = (
    (5, 0.05),
    (4, 0.15),
    (3, 0.30),
    (2, 0.50),
)
MIN_BUCKET_COUNT_WARNING: int = 20

# ──────────────────────────────────────────────
# Constants — model registry
# ──────────────────────────────────────────────

# Per-model fine-tune config. target_modules are LLM-side (verified against
# named_modules() at load time); vision_encoder_module names the submodule
# whose attention projections get the optional smaller LoRA when
# --lora-vision-encoder is enabled.
_FT_REGISTRY: dict[str, dict[str, Any]] = {
    "m3d_lamed": {
        "hf_id": "GoodBaiBai88/M3D-LaMed-Phi-3-4B",
        "target_modules": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        "vision_encoder_module": "vision_tower",
        "input_type": "3d",
    },
    "llava_ov": {
        "hf_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "vision_encoder_module": "vision_tower",
        "input_type": "2d",
    },
    "qwen2_vl": {
        "hf_id": "Qwen/Qwen2-VL-7B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "vision_encoder_module": "visual",
        "input_type": "2d",
    },
    "medgemma": {
        "hf_id": "google/medgemma-4b-it",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "vision_encoder_module": "vision_tower",
        "input_type": "2d",
    },
}
_MODEL_CHOICES: tuple[str, ...] = tuple(_FT_REGISTRY.keys())

# ──────────────────────────────────────────────
# Constants — defaults
# ──────────────────────────────────────────────

DEFAULT_LORA_RANK: int = 16
DEFAULT_LORA_ALPHA: int = 32
DEFAULT_LORA_DROPOUT: float = 0.05
DEFAULT_VISION_LORA_RANK: int = 8
DEFAULT_VISION_LORA_ALPHA: int = 16
DEFAULT_LR: float = 2e-5
DEFAULT_WD: float = 0.01
DEFAULT_NUM_EPOCHS: int = 5
DEFAULT_GRAD_ACCUM: int = 4
DEFAULT_WARMUP_FRACTION: float = 0.05
DEFAULT_PATIENCE: int = 2
DEFAULT_MAX_NEW_TOKENS: int = 24
DEFAULT_MAX_GRAD_NORM: float = 1.0

# Auto-resolved batch size by input_type when --batch-size None.
_AUTO_BATCH_SIZE: dict[str, int] = {"3d": 4, "2d": 8}

# (C, D, H, W) target for 3D input tensors (matches 08a).
_VLM_3D_TARGET_DHW: tuple[int, int, int] = (32, 256, 256)

# 2D slice extraction strategy.
_SLICE_STRATEGY_2D: str = "mid"
_ORIENTATIONS_2D: tuple[str, ...] = ("axial", "coronal", "sagittal")
_N_SLICES_2D: int = 3

logger = logging.getLogger(__name__)
app = typer.Typer(
    help="NeuroQC Phase 09 — LoRA fine-tune VLMs on machine preference (RQ3).",
    add_completion=False,
)


# ──────────────────────────────────────────────
# Determinism (mirrors 08a:191-204)
# ──────────────────────────────────────────────


def set_seeds(seed: int) -> None:
    """Seed torch / CUDA / Python RNGs and enable deterministic cudnn.

    bf16 + bitsandbytes 4-bit are not bit-deterministic across hardware;
    reproducibility is best-effort but the seed is recorded in the output
    CSV and the provenance JSON.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


# ──────────────────────────────────────────────
# Helpers — paths, parsing
# ──────────────────────────────────────────────


def _scan_stem(path: Path | str) -> str:
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return Path(name).stem


def _format_target_string(dice_q: int, thick_q: int) -> str:
    """Build the joint two-head target string."""
    return f"Quality: {dice_q} Thickness: {thick_q}"


# ──────────────────────────────────────────────
# Data prep
# ──────────────────────────────────────────────


def load_subsample(path: Path) -> pd.DataFrame:
    """Load benchmark_subsample.csv. Required columns enforced."""
    if not path.is_file():
        raise FileNotFoundError(
            f"benchmark subsample not found at {path} — "
            f"run code/08a_eval_3d_vlms.py Phase A first."
        )
    df = pd.read_csv(path)
    required = {SCAN_COLUMN, REF_ID_COLUMN, IS_CLEAN_COLUMN, SPLIT_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"benchmark_subsample.csv missing required columns: {sorted(missing)}; "
            f"got {sorted(df.columns)}"
        )
    return df


def load_preference(path: Path) -> pd.DataFrame:
    """Load machine_preference.csv. Returns frame keyed on cor_path."""
    if not path.is_file():
        raise FileNotFoundError(f"machine_preference.csv not found at {path}")
    df = pd.read_csv(path)
    required = {_PREF_COR_PATH, _PREF_MEAN_DICE}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"machine_preference.csv missing required columns: {sorted(missing)}; "
            f"got {sorted(df.columns)}"
        )
    return df[[_PREF_COR_PATH, _PREF_MEAN_DICE]].copy()


def _build_seg_to_scan_map(synthseg_manifests: list[Path]) -> dict[str, str]:
    """Build ``{resolved_seg_path: resolved_scan_path}`` from synthseg manifests.

    Phase 03b's ``cortical_thickness.csv`` writes the SEG path under the
    ``scan_path`` column (it loads from the seg file directly). Phase 09
    needs to join thickness against the benchmark subsample's actual scan
    paths, so we remap via the synthseg manifests' ``input_path → seg_path``
    mapping (inverted to seg → input). Empty mapping if no manifests are
    provided; the load step then leaves values unchanged.
    """
    out: dict[str, str] = {}
    for manifest_path in synthseg_manifests:
        if not manifest_path.is_file():
            logger.warning("synthseg manifest not found: %s; skipping", manifest_path)
            continue
        try:
            mdf = pd.read_csv(manifest_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read %s: %s; skipping", manifest_path, exc)
            continue
        required = {"input_path", "seg_path"}
        if not required.issubset(mdf.columns):
            logger.warning(
                "synthseg manifest %s missing %s; skipping",
                manifest_path, sorted(required - set(mdf.columns)),
            )
            continue
        for _, row in mdf.iterrows():
            seg = str(Path(row["seg_path"]).resolve())
            scan = str(Path(row["input_path"]).resolve())
            out[seg] = scan
    return out


def load_thickness(
    path: Path,
    synthseg_manifests: list[Path] | None = None,
) -> pd.DataFrame:
    """Load cortical_thickness.csv; remap seg-path keys to scan-path keys if needed.

    Phase 03b writes ``scan_path`` valued at the SEG path (the file it
    actually loaded). When ``synthseg_manifests`` is provided, every value
    that resolves to a known seg path is remapped to its source scan path
    so the downstream join with ``benchmark_subsample.csv`` succeeds.
    Without manifests the column is taken at face value.
    """
    if not path.is_file():
        raise FileNotFoundError(f"cortical_thickness.csv not found at {path}")
    df = pd.read_csv(path)
    required = {_THICKNESS_SCAN_PATH, _THICKNESS_MEAN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"cortical_thickness.csv missing required columns: {sorted(missing)}; "
            f"got {sorted(df.columns)}"
        )
    out = df[[_THICKNESS_SCAN_PATH, _THICKNESS_MEAN]].copy()
    if synthseg_manifests:
        seg_to_scan = _build_seg_to_scan_map(synthseg_manifests)
        if seg_to_scan:
            n_remapped = 0

            def _maybe_remap(p: str) -> str:
                nonlocal n_remapped
                resolved = str(Path(p).resolve())
                target = seg_to_scan.get(resolved)
                if target is not None and target != resolved:
                    n_remapped += 1
                    return target
                return p

            out[_THICKNESS_SCAN_PATH] = out[_THICKNESS_SCAN_PATH].map(_maybe_remap)
            logger.info(
                "thickness remap via synthseg manifest: %d/%d rows remapped seg → scan",
                n_remapped, len(out),
            )
    return out


def join_data(
    subsample: pd.DataFrame,
    preference: pd.DataFrame,
    thickness: pd.DataFrame,
) -> pd.DataFrame:
    """Three-way join with clean-row structural-target rule.

    Steps:
      1. Left-join thickness on ``scan_path`` (per-scan, direct join).
      2. Left-join preference on ``scan_path == cor_path`` (preference is
         per-pair; clean rows match nothing by construction).
      3. For ``is_clean=True`` rows, set ``mean_dice = 1.0`` (Sørensen-Dice
         idempotence) and ``thickness_shift = 0.0`` (handled in
         :func:`derive_signed_thickness_shift`).
    """
    out = subsample.merge(thickness, on=SCAN_COLUMN, how="left")
    out = out.merge(
        preference,
        left_on=SCAN_COLUMN,
        right_on=_PREF_COR_PATH,
        how="left",
        suffixes=("", "_pref"),
    )
    if _PREF_COR_PATH in out.columns:
        out = out.drop(columns=[_PREF_COR_PATH])
    # Clean-row structural Dice = 1.0 (idempotence; not measured because tautological).
    clean_mask = out[IS_CLEAN_COLUMN].astype(bool)
    out.loc[clean_mask, _DERIVED_DICE] = 1.0
    return out


def derive_signed_thickness_shift(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SIGNED ``mean_thickness - clean_thickness`` per ref_id.

    For each ref_id, looks up the clean baseline (the row where
    ``is_clean=True``) and subtracts. Negative = thinning, positive =
    thickening. Clean rows necessarily get 0.0 (baseline minus baseline).

    Refs without a clean baseline (e.g. clean row missing from thickness CSV)
    are dropped; the dropped count is logged.
    """
    out = df.copy()
    out[_DERIVED_SIGNED_SHIFT] = float("nan")

    n_dropped_refs = 0
    keep_mask = pd.Series(True, index=out.index)

    for ref_id, group in out.groupby(REF_ID_COLUMN):
        clean_rows = group[group[IS_CLEAN_COLUMN].astype(bool)]
        if len(clean_rows) == 0:
            n_dropped_refs += 1
            keep_mask.loc[group.index] = False
            continue
        clean_thickness = clean_rows.iloc[0][_THICKNESS_MEAN]
        if pd.isna(clean_thickness):
            n_dropped_refs += 1
            keep_mask.loc[group.index] = False
            continue
        out.loc[group.index, _DERIVED_SIGNED_SHIFT] = (
            out.loc[group.index, _THICKNESS_MEAN] - clean_thickness
        )

    if n_dropped_refs > 0:
        logger.warning(
            "Dropped %d refs (and all their cor rows) lacking a clean thickness "
            "baseline. These cannot contribute thickness-quality training signal.",
            n_dropped_refs,
        )

    return out[keep_mask].reset_index(drop=True)


def discretize_dice(value: float) -> int | None:
    """Map continuous mean_dice ∈ [0, 1] to bucket 1-5; None on NaN."""
    if value is None or pd.isna(value):
        return None
    for bucket, threshold in DICE_BUCKET_THRESHOLDS:
        if value >= threshold:
            return bucket
    return 1


def discretize_thickness_abs(signed_shift: float) -> int | None:
    """Map SIGNED thickness_shift to bucket 1-5 via absolute value; None on NaN."""
    if signed_shift is None or pd.isna(signed_shift):
        return None
    abs_v = abs(signed_shift)
    for bucket, threshold in THICKNESS_BUCKET_THRESHOLDS:
        if abs_v < threshold:
            return bucket
    return 1


def add_bucket_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Append integer dice/thickness bucket columns to the joined frame."""
    out = df.copy()
    out[_DERIVED_DICE_BUCKET] = out[_DERIVED_DICE].map(discretize_dice)
    out[_DERIVED_THICKNESS_BUCKET] = out[_DERIVED_SIGNED_SHIFT].map(
        discretize_thickness_abs
    )
    return out


def validate_split(df: pd.DataFrame) -> None:
    """Assert no ref_id appears in more than one split (ref-level disjointness)."""
    counts = df.groupby(REF_ID_COLUMN)[SPLIT_COLUMN].nunique()
    leaked = counts[counts > 1]
    if len(leaked) > 0:
        sample = leaked.head(5).to_dict()
        raise ValueError(
            f"ref_id leaked across splits ({len(leaked)} affected): {sample}; "
            f"the benchmark subsample's split column is corrupt."
        )


def bucket_distribution(df: pd.DataFrame, column: str) -> dict[int, int]:
    """Count rows per integer bucket value (drops NaN)."""
    counts = df[column].dropna().astype(int).value_counts().to_dict()
    return {int(k): int(v) for k, v in sorted(counts.items())}


def log_bucket_distributions(df: pd.DataFrame) -> dict[str, dict[int, int]]:
    """Log per-split, per-axis bucket counts; warn on any low-count train bucket."""
    summary: dict[str, dict[int, int]] = {}
    for split in (_TRAIN, _VAL, _TEST):
        sub = df[df[SPLIT_COLUMN] == split]
        dice_dist = bucket_distribution(sub, _DERIVED_DICE_BUCKET)
        thick_dist = bucket_distribution(sub, _DERIVED_THICKNESS_BUCKET)
        summary[f"dice_{split}"] = dice_dist
        summary[f"thickness_{split}"] = thick_dist
        logger.info("Split=%s n=%d dice=%s thickness=%s", split, len(sub), dice_dist, thick_dist)

    train_dice = summary.get(f"dice_{_TRAIN}", {})
    train_thickness = summary.get(f"thickness_{_TRAIN}", {})
    for bucket in range(1, 6):
        for label, dist in (("dice", train_dice), ("thickness", train_thickness)):
            n = dist.get(bucket, 0)
            if n < MIN_BUCKET_COUNT_WARNING:
                logger.warning(
                    "Train bucket %s=%d has only %d examples (< %d) — "
                    "calibrate bucket boundaries against your data distribution.",
                    label, bucket, n, MIN_BUCKET_COUNT_WARNING,
                )
    return summary


def prepare_dataframe(
    subsample_path: Path,
    preference_path: Path,
    thickness_path: Path,
    synthseg_manifests: list[Path] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[int, int]]]:
    """End-to-end data prep: load → join → derive shift → bucket → validate.

    Returns ``(df, bucket_summary)`` where ``df`` has the bucket columns
    populated and split validation has passed. Refs with NaN in either bucket
    are not dropped here — the per-row training loop skips NaN-target rows
    so they don't contribute gradient signal but still surface in counts.

    ``synthseg_manifests``: optional list of synthseg manifest CSVs. When
    provided, ``cortical_thickness.csv``'s ``scan_path`` column is remapped
    from seg-path values (Phase 03b's convention) to actual scan paths
    (what the benchmark subsample uses). Without it, the thickness join
    silently produces NaN for every cor scan when 03b emitted seg paths.
    """
    subsample = load_subsample(subsample_path)
    preference = load_preference(preference_path)
    thickness = load_thickness(thickness_path, synthseg_manifests=synthseg_manifests)

    joined = join_data(subsample, preference, thickness)

    # Clean-row structural shift = 0.0 (set BEFORE derive_signed_thickness_shift
    # so the per-ref baseline lookup sees the structural value).
    clean_mask = joined[IS_CLEAN_COLUMN].astype(bool)
    joined.loc[clean_mask, _DERIVED_SIGNED_SHIFT] = 0.0

    df = derive_signed_thickness_shift(joined)
    df = add_bucket_columns(df)

    validate_split(df)
    summary = log_bucket_distributions(df)
    return df, summary


# ──────────────────────────────────────────────
# Adapter Protocol + scaffolds
# ──────────────────────────────────────────────


class FineTuneAdapter(Protocol):
    """Per-model interface for QLoRA fine-tuning.

    Each adapter encapsulates HF model loading (with ``BitsAndBytesConfig``),
    PEFT wrapping (LLM LoRA + optional vision LoRA), input preparation
    (returns model-ready ``{input_ids, attention_mask, pixel_values, labels}``
    dict per scan), and the matching collator. Tests inject a fake adapter
    via ``_FT_ADAPTERS[name] = FakeAdapter`` to avoid HF downloads.
    """

    name: str
    hf_id: str
    input_type: Literal["2d", "3d"]
    target_modules: list[str]
    vision_encoder_module: str

    def load(
        self,
        device: torch.device,
        dtype: torch.dtype,
        bnb_config: Any,
        llm_lora_config: Any,
        vision_lora_config: Any | None,
    ) -> None: ...

    def prepare_inputs(
        self, scan_path: Path, target_text: str
    ) -> dict[str, torch.Tensor]: ...

    def collate_fn(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]: ...

    @property
    def model(self) -> Any: ...

    @property
    def processor(self) -> Any: ...

    def unload(self) -> None: ...


def _walk_vision_target_modules(
    model: Any, vision_encoder_module: str
) -> list[str]:
    """Heuristically gather attention-projection module names under the vision encoder.

    Walks ``model.<vision_encoder_module>.named_modules()`` and collects names
    whose final segment matches one of the canonical projection names. The
    returned names are relative to the parent ``model`` (so PEFT's
    ``LoraConfig(target_modules=[...])`` resolves them against the same root).
    """
    sub = getattr(model, vision_encoder_module, None)
    if sub is None:
        logger.warning(
            "Vision encoder submodule %r not found on model; vision-LoRA disabled",
            vision_encoder_module,
        )
        return []
    proj_keywords = ("q_proj", "k_proj", "v_proj", "o_proj", "qkv", "out_proj")
    targets: list[str] = []
    for name, _ in sub.named_modules():
        leaf = name.split(".")[-1]
        if leaf in proj_keywords:
            targets.append(f"{vision_encoder_module}.{name}")
    return targets


@dataclass
class M3DLamedAdapter:
    """Primary 3D-VLM adapter for QLoRA fine-tuning.

    Mirrors 08a's M3DLamedAdapter plus a training-side ``prepare_inputs`` that
    builds an ``{input_ids, attention_mask, pixel_values, labels}`` dict from
    one ``(scan_path, target_text)`` pair.

    Tensor pipeline matches 08a — MONAI chain to (1, 256, 256, 32) then
    permute to (1, 32, 256, 256) for the model. Target text is tokenised
    by the model's tokenizer; labels = input_ids with the prompt-prefix
    masked to -100 so loss is only on the answer span.
    """

    name: str = "m3d_lamed"
    hf_id: str = field(default_factory=lambda: _FT_REGISTRY["m3d_lamed"]["hf_id"])
    input_type: Literal["2d", "3d"] = "3d"
    target_modules: list[str] = field(
        default_factory=lambda: list(_FT_REGISTRY["m3d_lamed"]["target_modules"])
    )
    vision_encoder_module: str = field(
        default_factory=lambda: _FT_REGISTRY["m3d_lamed"]["vision_encoder_module"]
    )

    _model: Any = None
    _processor: Any = None
    _tokenizer: Any = None
    _transform: Any = None
    _device: torch.device | None = None
    _dtype: torch.dtype | None = None

    @property
    def model(self) -> Any:
        return self._model

    @property
    def processor(self) -> Any:
        return self._processor

    def load(
        self,
        device: torch.device,
        dtype: torch.dtype,
        bnb_config: Any,
        llm_lora_config: Any,
        vision_lora_config: Any | None,
    ) -> None:
        from peft import get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from monai.transforms import (
            Compose,
            EnsureChannelFirst,
            LoadImage,
            ResizeWithPadOrCrop,
            ScaleIntensityRangePercentiles,
            Spacing,
        )

        self._device = device
        self._dtype = dtype
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.hf_id, trust_remote_code=True
        )
        base = AutoModelForCausalLM.from_pretrained(
            self.hf_id,
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        peft_model = get_peft_model(base, llm_lora_config)
        if vision_lora_config is not None:
            try:
                peft_model.add_adapter("vision", vision_lora_config)
            except Exception as exc:  # noqa: BLE001 — surface the API caveat
                logger.warning(
                    "vision LoRA adapter add failed (%s); falling back to "
                    "LLM-only LoRA for this run.", exc,
                )
        peft_model.print_trainable_parameters()
        self._model = peft_model.to(device)
        self._processor = self._tokenizer
        self._transform = Compose(
            [
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
                ResizeWithPadOrCrop(spatial_size=(256, 256, 32)),
                ScaleIntensityRangePercentiles(
                    lower=2, upper=98, b_min=0.0, b_max=1.0, clip=True
                ),
            ]
        )

    def prepare_inputs(
        self, scan_path: Path, target_text: str
    ) -> dict[str, torch.Tensor]:
        from nobrainer.qc.evaluate import QC_DUAL_PROMPT

        if self._transform is None or self._tokenizer is None:
            raise RuntimeError("M3DLamedAdapter.load() must be called first")

        vol = self._transform(str(scan_path))
        vol = vol.permute(0, 3, 1, 2).contiguous()  # (1, 32, 256, 256)

        prompt = QC_DUAL_PROMPT
        full_text = f"{prompt}\n{target_text}"
        prompt_ids = self._tokenizer(
            prompt + "\n", add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]
        full_ids = self._tokenizer(
            full_text, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]
        labels = full_ids.clone()
        labels[: prompt_ids.shape[0]] = -100  # mask prompt tokens from loss
        attention_mask = torch.ones_like(full_ids)
        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": vol,
        }

    def collate_fn(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        pad_id = (
            self._tokenizer.pad_token_id
            if self._tokenizer is not None and self._tokenizer.pad_token_id is not None
            else 0
        )
        max_len = max(item["input_ids"].shape[0] for item in batch)
        input_ids = torch.full(
            (len(batch), max_len), pad_id, dtype=torch.long
        )
        attention = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
        for i, item in enumerate(batch):
            n = item["input_ids"].shape[0]
            input_ids[i, :n] = item["input_ids"]
            attention[i, :n] = item["attention_mask"]
            labels[i, :n] = item["labels"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
            "pixel_values": pixel_values,
        }

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._transform = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


@dataclass
class _Scaffold2DAdapter:
    """Base scaffold for 2D adapters (LLaVA-OV, Qwen2-VL, MedGemma).

    Each subclass overrides ``name`` / ``hf_id`` / ``vision_encoder_module``
    via ``__post_init__``. ``prepare_inputs`` raises ``NotImplementedError``
    so the structural code path is testable end-to-end before each model's
    HF processor surface gets GPU-verified — same pattern as 08a's
    RadFM/Med-2E3 scaffolds.
    """

    name: str = ""
    hf_id: str = ""
    input_type: Literal["2d", "3d"] = "2d"
    target_modules: list[str] = field(default_factory=list)
    vision_encoder_module: str = ""

    @property
    def model(self) -> Any:
        return None

    @property
    def processor(self) -> Any:
        return None

    def load(
        self,
        device: torch.device,
        dtype: torch.dtype,
        bnb_config: Any,
        llm_lora_config: Any,
        vision_lora_config: Any | None,
    ) -> None:
        logger.warning(
            "%s scaffold adapter loaded — prepare_inputs() will raise "
            "NotImplementedError. Replace with a real implementation against "
            "the model's HF processor before training.",
            self.name,
        )

    def prepare_inputs(
        self, scan_path: Path, target_text: str
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            f"{self.name} fine-tune adapter is a scaffold; provide a real "
            f"implementation against {self.hf_id}'s HF processor."
        )

    def collate_fn(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            f"{self.name} fine-tune adapter is a scaffold; provide a real collator."
        )

    def unload(self) -> None:
        pass


@dataclass
class LlavaOVAdapter(_Scaffold2DAdapter):
    name: str = "llava_ov"
    hf_id: str = field(default_factory=lambda: _FT_REGISTRY["llava_ov"]["hf_id"])
    target_modules: list[str] = field(
        default_factory=lambda: list(_FT_REGISTRY["llava_ov"]["target_modules"])
    )
    vision_encoder_module: str = field(
        default_factory=lambda: _FT_REGISTRY["llava_ov"]["vision_encoder_module"]
    )


@dataclass
class Qwen2VLAdapter(_Scaffold2DAdapter):
    name: str = "qwen2_vl"
    hf_id: str = field(default_factory=lambda: _FT_REGISTRY["qwen2_vl"]["hf_id"])
    target_modules: list[str] = field(
        default_factory=lambda: list(_FT_REGISTRY["qwen2_vl"]["target_modules"])
    )
    vision_encoder_module: str = field(
        default_factory=lambda: _FT_REGISTRY["qwen2_vl"]["vision_encoder_module"]
    )


@dataclass
class MedGemmaAdapter(_Scaffold2DAdapter):
    name: str = "medgemma"
    hf_id: str = field(default_factory=lambda: _FT_REGISTRY["medgemma"]["hf_id"])
    target_modules: list[str] = field(
        default_factory=lambda: list(_FT_REGISTRY["medgemma"]["target_modules"])
    )
    vision_encoder_module: str = field(
        default_factory=lambda: _FT_REGISTRY["medgemma"]["vision_encoder_module"]
    )


_FT_ADAPTERS: dict[str, Callable[[], FineTuneAdapter]] = {
    "m3d_lamed": M3DLamedAdapter,
    "llava_ov": LlavaOVAdapter,
    "qwen2_vl": Qwen2VLAdapter,
    "medgemma": MedGemmaAdapter,
}


# ──────────────────────────────────────────────
# QLoRA configuration
# ──────────────────────────────────────────────


def build_bnb_config(vision_encoder_module: str) -> Any:
    """4-bit nf4 with bf16 compute, vision encoder kept in compute dtype."""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=[vision_encoder_module],
    )


def build_lora_configs(
    target_modules: list[str],
    rank: int,
    alpha: int,
    dropout: float,
    enable_vision: bool,
    vision_target_modules: list[str] | None,
    vision_rank: int = DEFAULT_VISION_LORA_RANK,
    vision_alpha: int = DEFAULT_VISION_LORA_ALPHA,
) -> tuple[Any, Any | None]:
    """Build the LLM LoRA config and (optionally) the vision LoRA config."""
    from peft import LoraConfig, TaskType

    llm = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    if not enable_vision:
        return llm, None
    if not vision_target_modules:
        logger.warning(
            "Vision LoRA requested but no vision target_modules resolved; "
            "falling back to LLM-only LoRA."
        )
        return llm, None
    vision = LoraConfig(
        r=vision_rank,
        lora_alpha=vision_alpha,
        lora_dropout=dropout,
        target_modules=vision_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    return llm, vision


def assert_peft_version() -> None:
    """Hard-abort if PEFT < 0.7 (where add_adapter is supported)."""
    try:
        import peft
    except ImportError as exc:
        raise RuntimeError(
            "peft is required for code/09_finetune_lora.py — install via "
            "`pip install 'peft>=0.7'`."
        ) from exc
    version = getattr(peft, "__version__", "0.0")
    parts = version.split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        major, minor = 0, 0
    if (major, minor) < (0, 7):
        raise RuntimeError(
            f"peft >= 0.7 required for vision-LoRA add_adapter support; "
            f"got {version}. Upgrade via `pip install -U 'peft>=0.7'`."
        )


def assert_dual_parser_available() -> None:
    """Hard-abort if nobrainer.qc.evaluate.parse_dual_qc_response is missing."""
    try:
        from nobrainer.qc.evaluate import parse_dual_qc_response  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "nobrainer.qc.evaluate.parse_dual_qc_response is required for "
            "code/09_finetune_lora.py. Install/update local nobrainer with "
            "the dual-target parser PR (feat/qc-dual-parser)."
        ) from exc


# ──────────────────────────────────────────────
# Dataset + Trainer
# ──────────────────────────────────────────────


class PreferenceDataset(torch.utils.data.Dataset):
    """Wraps a pre-bucketed DataFrame slice for HF Trainer.

    Each ``__getitem__`` returns the adapter's ``prepare_inputs`` dict for one
    row (scan + target_text). Rows with NaN dice/thickness buckets are
    filtered upstream by ``filter_trainable_rows``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        adapter: FineTuneAdapter,
        dice_col: str = _DERIVED_DICE_BUCKET,
        thick_col: str = _DERIVED_THICKNESS_BUCKET,
    ) -> None:
        self.rows: list[dict[str, Any]] = df.to_dict("records")
        self.adapter = adapter
        self.dice_col = dice_col
        self.thick_col = thick_col

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        target = _format_target_string(
            int(row[self.dice_col]), int(row[self.thick_col])
        )
        return self.adapter.prepare_inputs(Path(row[SCAN_COLUMN]), target_text=target)


def filter_trainable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaN in either bucket column (cannot form a target)."""
    mask = df[_DERIVED_DICE_BUCKET].notna() & df[_DERIVED_THICKNESS_BUCKET].notna()
    return df[mask].reset_index(drop=True)


def compute_val_srcc(
    adapter: FineTuneAdapter,
    val_df: pd.DataFrame,
    device: torch.device,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> float:
    """Run model.generate on val rows, parse, return Spearman vs ground-truth Dice.

    NaN if val is empty or the parsed-quality list has fewer than 2 valid
    entries (Spearman undefined). Pandas handles the rank correlation
    natively (`Series.corr(method="spearman")`), avoiding scipy.
    """
    from nobrainer.qc.evaluate import parse_dual_qc_response

    pred_quality: list[float] = []
    truth_dice: list[float] = []
    model = adapter.model
    if model is None:
        return float("nan")
    model.eval()
    with torch.no_grad():
        for _, row in val_df.iterrows():
            try:
                inputs = adapter.prepare_inputs(
                    Path(row[SCAN_COLUMN]), target_text=""
                )
            except Exception as exc:  # noqa: BLE001 — robust eval-loop
                logger.warning("Val prep failure on %s: %s", row[SCAN_COLUMN], exc)
                continue
            input_ids = inputs["input_ids"].unsqueeze(0).to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(0).to(device)
            pixel_values = inputs.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.unsqueeze(0).to(device)
            try:
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Val generate failure on %s: %s", row[SCAN_COLUMN], exc)
                continue
            text = adapter.processor.decode(output_ids[0], skip_special_tokens=True)
            parsed = parse_dual_qc_response(text)
            quality = parsed.get("quality")
            if quality is None:
                continue
            pred_quality.append(float(quality))
            truth_dice.append(float(row[_DERIVED_DICE]))
    if len(pred_quality) < 2:
        return float("nan")
    pred_series = pd.Series(pred_quality)
    truth_series = pd.Series(truth_dice)
    srcc = pred_series.corr(truth_series, method="spearman")
    return float(srcc) if not pd.isna(srcc) else float("nan")


def _build_trainer(
    *,
    adapter: FineTuneAdapter,
    train_dataset: torch.utils.data.Dataset,
    val_df: pd.DataFrame,
    output_dir: Path,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
    grad_accum: int,
    warmup_fraction: float,
    early_stop_patience: int,
    max_grad_norm: float,
    device: torch.device,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> Any:
    """Construct the HF Trainer subclass. Tests monkeypatch this."""
    from transformers import (
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    total_steps = max(
        1, math.ceil(len(train_dataset) / max(batch_size, 1) / max(grad_accum, 1))
    ) * max(num_epochs, 1)
    warmup_steps = max(20, int(warmup_fraction * total_steps))

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=DEFAULT_WD,
        adam_beta1=0.9,
        adam_beta2=0.999,
        max_grad_norm=max_grad_norm,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        bf16=device.type == "cuda",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="srcc",
        greater_is_better=True,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=[],
    )

    class GenerateForEvalTrainer(Trainer):
        """Trainer that augments evaluate() with a generation-based SRCC metric.

        ``compute_loss`` continues to use cross-entropy on the joint-string
        labels (default Trainer behaviour). ``evaluate()`` adds an
        ``eval_srcc`` key by running ``model.generate`` over ``val_df`` and
        computing Spearman against the per-row ``mean_dice``.
        """

        def evaluate(self, *eval_args: Any, **eval_kwargs: Any) -> dict[str, float]:
            metrics = super().evaluate(*eval_args, **eval_kwargs)
            srcc = compute_val_srcc(
                adapter, val_df, device=device, max_new_tokens=max_new_tokens
            )
            metrics["eval_srcc"] = srcc
            return metrics

    return GenerateForEvalTrainer(
        model=adapter.model,
        args=args,
        train_dataset=train_dataset,
        data_collator=adapter.collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stop_patience)],
    )


# ──────────────────────────────────────────────
# Provenance JSON
# ──────────────────────────────────────────────


def _git_env() -> dict[str, str]:
    """Subprocess env with system PATH prepended (per anti-pattern in plan.md)."""
    env = os.environ.copy()
    env["PATH"] = f"/usr/local/bin:/usr/bin:/bin:{env.get('PATH', '')}"
    return env


def _git_status_porcelain() -> str:
    """Return ``git status --porcelain`` output (empty string ⇒ clean tree)."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], env=_git_env(), text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("git status failed: %s", exc)
        return ""
    return out


def _git_rev_parse_head() -> str:
    """Return ``git rev-parse HEAD`` (or 'unknown' on failure)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], env=_git_env(), text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("git rev-parse failed: %s", exc)
        return "unknown"
    return out.strip()


def _git_diff_head() -> str:
    """Return ``git diff HEAD`` for dirty-tree provenance."""
    try:
        return subprocess.check_output(
            ["git", "diff", "HEAD"], env=_git_env(), text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("git diff failed: %s", exc)
        return ""


def _library_versions() -> dict[str, str]:
    """Capture installed versions of the libraries that affect reproducibility."""
    versions: dict[str, str] = {"torch": torch.__version__}
    for name in ("transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = "not-installed"
    return versions


def build_provenance(
    *,
    seed: int,
    model_name: str,
    hyperparameters: dict[str, Any],
    data_splits: dict[str, int],
    bucket_distributions: dict[str, dict[int, int]],
    start_time: str,
    end_time: str,
    best_val_srcc: float | None,
    best_epoch: int | None,
    trainable_params: int | None,
    total_params: int | None,
    diff_path: Path | None,
) -> dict[str, Any]:
    """Build the provenance dict (pure function — easy to test)."""
    git_status = _git_status_porcelain()
    git_dirty = bool(git_status.strip())
    return {
        "seed": seed,
        "model": model_name,
        "hf_id": _FT_REGISTRY[model_name]["hf_id"],
        "input_type": _FT_REGISTRY[model_name]["input_type"],
        "git_commit": _git_rev_parse_head(),
        "git_dirty": git_dirty,
        "git_diff_path": str(diff_path) if (git_dirty and diff_path is not None) else None,
        "hyperparameters": hyperparameters,
        "data_splits": data_splits,
        "bucket_distributions": bucket_distributions,
        "start_time": start_time,
        "end_time": end_time,
        "best_val_srcc": best_val_srcc,
        "best_epoch": best_epoch,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "library_versions": _library_versions(),
    }


def save_provenance(
    json_path: Path,
    provenance: dict[str, Any],
    diff_path: Path | None,
) -> None:
    """Write provenance JSON; if dirty, also write the diff to ``diff_path``."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as handle:
        json.dump(provenance, handle, indent=2, default=str)
    if provenance.get("git_dirty") and diff_path is not None:
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff_path.write_text(_git_diff_head())


# ──────────────────────────────────────────────
# Eval — score test split
# ──────────────────────────────────────────────


def load_existing(output_file: Path) -> set[tuple[str, str]]:
    """Return ``{(scan_path, model)}`` pairs already recorded in the output."""
    if not output_file.exists() or output_file.stat().st_size == 0:
        return set()
    with output_file.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {SCAN_COLUMN, MODEL_COLUMN}.issubset(
            reader.fieldnames
        ):
            return set()
        return {(row[SCAN_COLUMN], row[MODEL_COLUMN]) for row in reader}


def append_row(
    output_file: Path,
    *,
    scan_path: str,
    model_name: str,
    dice_q: float,
    thick_q: float,
    raw_response: str,
    seed: int,
) -> None:
    """Append one (scan, model) row to the eval CSV (crash-safe per write)."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    is_new = not output_file.exists() or output_file.stat().st_size == 0
    record: dict[str, object] = {
        SCAN_COLUMN: scan_path,
        MODEL_COLUMN: model_name,
        DICE_Q_COLUMN: dice_q,
        THICK_Q_COLUMN: thick_q,
        RAW_RESPONSE_COLUMN: raw_response,
        SEED_COLUMN: seed,
    }
    with output_file.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        if is_new:
            writer.writeheader()
        writer.writerow(record)
        handle.flush()


def score_test_split(
    *,
    adapter: FineTuneAdapter,
    test_df: pd.DataFrame,
    output_file: Path,
    model_name: str,
    seed: int,
    device: torch.device,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> tuple[int, int]:
    """Iterate test_df, generate per-scan, parse, append CSV rows.

    Returns ``(processed, failed)`` counts. Resume-safe via :func:`load_existing`.
    """
    from nobrainer.qc.evaluate import parse_dual_qc_response

    done = load_existing(output_file)
    pending = [
        row for _, row in test_df.iterrows()
        if (row[SCAN_COLUMN], model_name) not in done
    ]
    processed = 0
    failed = 0
    iterator = (
        tqdm(pending, desc=f"score:{model_name}", unit="scan")
        if len(pending) > 1
        else pending
    )
    model = adapter.model
    if model is not None:
        model.eval()
    for row in iterator:
        scan_path = row[SCAN_COLUMN]
        try:
            inputs = adapter.prepare_inputs(Path(scan_path), target_text="")
            input_ids = inputs["input_ids"].unsqueeze(0).to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(0).to(device)
            pixel_values = inputs.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.unsqueeze(0).to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    max_new_tokens=max_new_tokens,
                )
            raw = adapter.processor.decode(output_ids[0], skip_special_tokens=True)
            parsed = parse_dual_qc_response(raw)
            quality = parsed.get("quality")
            thickness = parsed.get("thickness")
            dice_q = float(quality) if quality is not None else math.nan
            thick_q = float(thickness) if thickness is not None else math.nan
            if math.isnan(dice_q) or math.isnan(thick_q):
                failed += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Eval failure on %s: %s", scan_path, exc)
            dice_q = math.nan
            thick_q = math.nan
            raw = f"ERROR:{exc}"
            failed += 1
        append_row(
            output_file,
            scan_path=str(scan_path),
            model_name=model_name,
            dice_q=dice_q,
            thick_q=thick_q,
            raw_response=raw,
            seed=seed,
        )
        processed += 1
    return processed, failed


# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────


def _print_summary(
    model_name: str,
    n_train: int,
    n_val: int,
    n_test: int,
    best_srcc: float | None,
    output_file: Path,
    console: Console,
) -> None:
    table = Table(title=f"Phase 09 — LoRA fine-tune ({model_name})")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("train rows", str(n_train))
    table.add_row("val rows", str(n_val))
    table.add_row("test rows", str(n_test))
    table.add_row("best val SRCC", f"{best_srcc:.4f}" if best_srcc is not None else "n/a")
    table.add_row("scores CSV", str(output_file))
    console.print(table)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


@app.command()
def main(
    seed: int = typer.Option(..., "--seed", help="Torch/Python RNG seed."),
    model: str = typer.Option(
        ...,
        "--model",
        help=f"One of {_MODEL_CHOICES}.",
    ),
    subsample_manifest: Path = typer.Option(
        Path("results/tables/benchmark_subsample.csv"),
        "--subsample-manifest",
        resolve_path=True,
    ),
    preference_csv: Path = typer.Option(
        Path("results/tables/machine_preference.csv"),
        "--preference-csv",
        resolve_path=True,
    ),
    thickness_csv: Path = typer.Option(
        Path("results/tables/cortical_thickness.csv"),
        "--thickness-csv",
        resolve_path=True,
    ),
    synthseg_manifest: list[Path] = typer.Option(
        [],
        "--synthseg-manifest",
        resolve_path=True,
        help=(
            "Optional synthseg manifest CSV (repeatable). When provided, "
            "cortical_thickness.csv's scan_path column is remapped from "
            "seg-path values (Phase 03b's convention) to actual scan paths. "
            "Without this flag, 09 takes the column at face value — which "
            "produces NaN thickness for every cor scan if Phase 03b emitted "
            "seg paths."
        ),
    ),
    output_checkpoint_dir: Path = typer.Option(
        Path("results/checkpoints"),
        "--output-checkpoint-dir",
        resolve_path=True,
    ),
    output_scores_file: Path | None = typer.Option(
        None,
        "--output-scores-file",
        resolve_path=True,
        help="Defaults to results/tables/finetuned_scores_seed_{seed}.csv.",
    ),
    lora_rank: int = typer.Option(DEFAULT_LORA_RANK, "--lora-rank"),
    lora_alpha: int = typer.Option(DEFAULT_LORA_ALPHA, "--lora-alpha"),
    lora_dropout: float = typer.Option(DEFAULT_LORA_DROPOUT, "--lora-dropout"),
    lora_vision_encoder: bool = typer.Option(
        True, "--lora-vision-encoder/--no-lora-vision-encoder"
    ),
    learning_rate: float = typer.Option(DEFAULT_LR, "--learning-rate"),
    num_epochs: int = typer.Option(DEFAULT_NUM_EPOCHS, "--num-epochs"),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        help="Auto-resolved from input_type when None: 4 for 3d, 8 for 2d.",
    ),
    grad_accum: int = typer.Option(DEFAULT_GRAD_ACCUM, "--grad-accum"),
    warmup_fraction: float = typer.Option(DEFAULT_WARMUP_FRACTION, "--warmup-fraction"),
    early_stop_patience: int = typer.Option(
        DEFAULT_PATIENCE, "--early-stop-patience"
    ),
    resume_from_checkpoint: Path | None = typer.Option(
        None, "--resume-from-checkpoint", resolve_path=True
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Load CSVs, log split sizes + bucket distributions, exit.",
    ),
) -> None:
    """LoRA fine-tune one VLM on machine preference labels (RQ3)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )
    console = Console()

    if model not in _MODEL_CHOICES:
        raise typer.BadParameter(
            f"Unknown model {model!r}; choose from {_MODEL_CHOICES}"
        )

    set_seeds(seed)

    # ── Data prep ──
    df, bucket_summary = prepare_dataframe(
        subsample_manifest,
        preference_csv,
        thickness_csv,
        synthseg_manifests=list(synthseg_manifest) if synthseg_manifest else None,
    )

    # NaN-bucket rows can't form training targets — skip them after counts logged.
    trainable_df = filter_trainable_rows(df)
    train_df = trainable_df[trainable_df[SPLIT_COLUMN] == _TRAIN].reset_index(drop=True)
    val_df = trainable_df[trainable_df[SPLIT_COLUMN] == _VAL].reset_index(drop=True)
    # Test split is scored with whatever's available (NaN buckets included as
    # NaN rows in the output CSV so coverage is honest).
    test_df = df[df[SPLIT_COLUMN] == _TEST].reset_index(drop=True)

    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    n_refs_per_split = {
        split: int(df[df[SPLIT_COLUMN] == split][REF_ID_COLUMN].nunique())
        for split in (_TRAIN, _VAL, _TEST)
    }

    logger.info(
        "Splits — train=%d val=%d test=%d (refs per split: %s)",
        n_train, n_val, n_test, n_refs_per_split,
    )

    if dry_run:
        logger.info("--dry-run: data prep complete; exiting before model load.")
        return

    # ── Hard preconditions before any HF download ──
    assert_dual_parser_available()
    assert_peft_version()

    # ── Resolve config ──
    config = _FT_REGISTRY[model]
    if batch_size is None:
        batch_size = _AUTO_BATCH_SIZE[config["input_type"]]

    output_scores = output_scores_file or Path(
        f"results/tables/finetuned_scores_seed_{seed}.csv"
    ).resolve()
    ckpt_dir = (output_checkpoint_dir / f"{model}_lora_seed_{seed}").resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if device.type == "cpu":
        logger.warning("CUDA not available; QLoRA training is impractical on CPU.")

    # ── Adapter load + LoRA wrap ──
    adapter_cls = _FT_ADAPTERS[model]
    adapter: FineTuneAdapter = adapter_cls()  # type: ignore[call-arg]
    bnb_config = build_bnb_config(config["vision_encoder_module"])

    # Vision target_modules resolved at load time (after model exists).
    llm_lora_config, _ = build_lora_configs(
        target_modules=list(config["target_modules"]),
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        enable_vision=False,
        vision_target_modules=None,
    )
    adapter.load(
        device=device,
        dtype=dtype,
        bnb_config=bnb_config,
        llm_lora_config=llm_lora_config,
        vision_lora_config=None,
    )

    if lora_vision_encoder and adapter.model is not None:
        # Now walk the loaded model for vision proj names; build + add the
        # second adapter via PEFT's add_adapter.
        vision_targets = _walk_vision_target_modules(
            adapter.model, config["vision_encoder_module"]
        )
        _, vision_lora = build_lora_configs(
            target_modules=list(config["target_modules"]),
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            enable_vision=True,
            vision_target_modules=vision_targets,
        )
        if vision_lora is not None:
            try:
                adapter.model.add_adapter("vision", vision_lora)
                logger.info(
                    "Vision LoRA attached on %d targets under %s.",
                    len(vision_targets), config["vision_encoder_module"],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Vision LoRA attach failed: %s", exc)

    # ── Trainer ──
    train_dataset = PreferenceDataset(train_df, adapter)
    trainer = _build_trainer(
        adapter=adapter,
        train_dataset=train_dataset,
        val_df=val_df,
        output_dir=ckpt_dir,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        warmup_fraction=warmup_fraction,
        early_stop_patience=early_stop_patience,
        max_grad_norm=DEFAULT_MAX_GRAD_NORM,
        device=device,
    )

    # ── Train + provenance (always written, even on KeyboardInterrupt) ──
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    diff_path = Path(f"results/tables/finetune_diff_{timestamp}.diff").resolve()
    provenance_json = Path(
        f"results/tables/finetune_run_info_{model}_seed_{seed}.json"
    ).resolve()
    start_time = datetime.now(timezone.utc).isoformat()
    best_srcc: float | None = None
    best_epoch: int | None = None
    trainable_params: int | None = None
    total_params: int | None = None
    if adapter.model is not None and hasattr(adapter.model, "num_parameters"):
        try:
            trainable_params = sum(
                p.numel() for p in adapter.model.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in adapter.model.parameters())
        except Exception:  # noqa: BLE001
            pass

    try:
        trainer.train(resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None)
        log_history = getattr(trainer.state, "log_history", []) or []
        srcc_entries = [
            (entry.get("epoch"), entry.get("eval_srcc"))
            for entry in log_history
            if entry.get("eval_srcc") is not None
        ]
        if srcc_entries:
            best_epoch_val, best_srcc = max(srcc_entries, key=lambda x: x[1])
            best_epoch = int(best_epoch_val) if best_epoch_val is not None else None
    finally:
        end_time = datetime.now(timezone.utc).isoformat()
        provenance = build_provenance(
            seed=seed,
            model_name=model,
            hyperparameters={
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "lora_vision_encoder": lora_vision_encoder,
                "target_modules": list(config["target_modules"]),
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "warmup_fraction": warmup_fraction,
                "early_stop_patience": early_stop_patience,
            },
            data_splits={
                "n_train": n_train,
                "n_val": n_val,
                "n_test": n_test,
                **{f"n_refs_{k}": v for k, v in n_refs_per_split.items()},
            },
            bucket_distributions=bucket_summary,
            start_time=start_time,
            end_time=end_time,
            best_val_srcc=best_srcc,
            best_epoch=best_epoch,
            trainable_params=trainable_params,
            total_params=total_params,
            diff_path=diff_path,
        )
        save_provenance(provenance_json, provenance, diff_path=diff_path)

    # ── Eval test split ──
    processed, failed = score_test_split(
        adapter=adapter,
        test_df=test_df,
        output_file=output_scores,
        model_name=model,
        seed=seed,
        device=device,
    )
    logger.info(
        "Test scoring complete: processed=%d, failed=%d, output=%s",
        processed, failed, output_scores,
    )

    adapter.unload()
    _print_summary(model, n_train, n_val, n_test, best_srcc, output_scores, console)


if __name__ == "__main__":
    app()
