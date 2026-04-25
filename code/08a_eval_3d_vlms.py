#!/usr/bin/env python3
"""NeuroQC Phase 08a — evaluate 3D VLMs on a ref-level-stratified benchmark.

Two phases:

  Phase A — Build `benchmark_subsample.csv` (if not already on disk): sample
  a deterministic subset of reference scans, attach all matching corruption
  rows at the requested (type × severity) combinations plus one clean row
  per ref, assign a ref-level 70/15/15 train/val/test split, and left-join
  machine preference where available.

  Phase B — Load each requested 3D VLM in turn and write per-scan quality
  scores to `3d_vlm_scores_seed_{seed}.csv`. Primary model is M3D-LaMed
  (GoodBaiBai88/M3D-LaMed-Phi-3-4B); RadFM and Med-2E3 are scaffolded as
  optional adapters and skip with a warning if their HF repos don't expose
  a working 3D-volume inference API yet.

`code/08b_eval_2d_vlms.py` (forthcoming) will read the same
`benchmark_subsample.csv` to keep the 3D-vs-2D comparison paired (RQ2).

Inputs:
    --seed INT                Required. Controls torch/CUDA/Python RNGs AND
                              is recorded as a column in the output CSV.
    --models TEXT             Comma-separated list from {m3d_lamed, radfm,
                              med2e3}. Default: "m3d_lamed".
    --subsample-manifest PATH `benchmark_subsample.csv`. If absent, Phase A
                              builds it; otherwise it's loaded as-is.
    --output-file PATH        `3d_vlm_scores_seed_{seed}.csv`. Append-mode,
                              per-row flush. Resume skips (scan, model) pairs
                              already recorded.
    --ref-manifest PATH       Phase A input (repeatable; ignored if the
                              subsample manifest already exists).
    --cor-manifest PATH       Phase A input.
    --preference-csv PATH     Optional Phase A input. When present, used
                              only to write a `preference_score` column into
                              the subsample — never for filtering (avoids
                              selection-bias leakage).
    --n-refs INT              Subsample size (default 120).
    --severities TEXT         Comma-separated severities (default "1,3,5").
    --corruption-types TEXT   Comma-separated corruption families.
    --split-seed INT          Seed for Phase A determinism (default 42).
    --dry-run                 Print pending counts + first 10 scans, exit.

VLM contract reminders:
    * `from nobrainer.qc.evaluate import QC_PROMPT, parse_qc_response` —
      `parse_qc_response` currently returns a dict
      ``{"score": 1-5 | None, "reason": str, "parse_success": bool}``. This
      script normalises to ``(score - 1) / 4`` for a float in [0, 1],
      NaN on parse failure, so downstream RQ3 targets stay in a stable
      range.
    * M3D-LaMed expects a tensor shaped ``(1, 32, 256, 256)`` = (C, D, H, W);
      we build that via the MONAI chain defined in `build_vlm_transform`.
    * On OOM we clear the CUDA cache, write a NaN row with
      ``raw_response="OOM"``, and continue.

Output schema (one row per (scan, model) pair):
    scan_path, model, score, raw_response, seed
"""

from __future__ import annotations

import csv
import gc
import hashlib
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import pandas as pd
import torch
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

# Output CSV (Phase B) schema.
SCAN_COLUMN: str = "scan_path"
MODEL_COLUMN: str = "model"
SCORE_COLUMN: str = "score"
RAW_RESPONSE_COLUMN: str = "raw_response"
SEED_COLUMN: str = "seed"
OUTPUT_COLUMNS: tuple[str, ...] = (
    SCAN_COLUMN,
    MODEL_COLUMN,
    SCORE_COLUMN,
    RAW_RESPONSE_COLUMN,
    SEED_COLUMN,
)

# Subsample CSV (Phase A) schema.
REF_ID_COLUMN: str = "ref_id"
IS_CLEAN_COLUMN: str = "is_clean"
TYPE_COLUMN: str = "corruption_type"
SEVERITY_COLUMN: str = "severity"
DATASET_TAG_COLUMN: str = "dataset_tag"
SPLIT_COLUMN: str = "split"
PREFERENCE_COLUMN: str = "preference_score"
SUBSAMPLE_COLUMNS: tuple[str, ...] = (
    REF_ID_COLUMN,
    SCAN_COLUMN,
    IS_CLEAN_COLUMN,
    TYPE_COLUMN,
    SEVERITY_COLUMN,
    DATASET_TAG_COLUMN,
    SPLIT_COLUMN,
    PREFERENCE_COLUMN,
)

# Ref-manifest column aliases for the scan-path column. First hit wins.
_REF_SCAN_PATH_COLUMN_ALIASES: tuple[str, ...] = ("filepath", "ref_path", "scan_path")
_PASSED_QC_COLUMN: str = "passed_qc"

# Corruption-manifest column names.
_COR_REF_PATH_COLUMN: str = "ref_path"
_COR_PATH_COLUMN: str = "cor_path"
_COR_TYPE_COLUMN: str = "corruption_type"
_COR_SEVERITY_COLUMN: str = "severity"
_COR_DATASET_TAG_COLUMN: str = "dataset_tag"

# machine_preference.csv — we key on its `cor_path` column and read
# `mean_dice` as the scalar preference.
_PREF_COR_COLUMN: str = "cor_path"
_PREF_SCORE_COLUMN: str = "mean_dice"

# Split labels.
_TRAIN: str = "train"
_VAL: str = "val"
_TEST: str = "test"

# Defaults.
DEFAULT_CORRUPTION_TYPES: tuple[str, ...] = (
    "motion", "ghosting", "spike", "noise",
    "bias_field", "blur", "downsample", "gamma",
)
DEFAULT_SEVERITIES: tuple[int, ...] = (1, 3, 5)
DEFAULT_N_REFS: int = 120
DEFAULT_SPLIT_SEED: int = 42
DEFAULT_TEST_FRACTION: float = 0.15
DEFAULT_VAL_FRACTION_OF_TRAINVAL: float = 15.0 / 85.0

# M3D-LaMed tensor geometry (C, D, H, W).
_VLM_TARGET_DHW: tuple[int, int, int] = (32, 256, 256)
_MAX_NEW_TOKENS_DEFAULT: int = 16

_NONE_TYPE: str = "none"
_CLEAN_SEVERITY: int = 0
_UNKNOWN_TAG: str = "unknown"

_MODEL_CHOICES: tuple[str, ...] = ("m3d_lamed", "radfm", "med2e3")

logger = logging.getLogger(__name__)
app = typer.Typer(
    help="NeuroQC Phase 08a — 3D VLM quality evaluation.",
    add_completion=False,
)


# ──────────────────────────────────────────────
# Helpers: path utilities + seed
# ──────────────────────────────────────────────


def _scan_stem(path: Path | str) -> str:
    """Strip ``.nii.gz`` / ``.nii`` from a filename, return the bare stem."""
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return Path(name).stem


def _infer_dataset_tag(scan_path: str) -> str:
    """Crude path-based dataset-tag inference (ixi/fastmri/oasis)."""
    lowered = scan_path.lower()
    for tag in ("ixi", "fastmri", "oasis"):
        if f"/{tag}/" in lowered:
            return tag
    return _UNKNOWN_TAG


def set_seeds(seed: int) -> None:
    """Set torch / CUDA / Python RNG seeds and enable deterministic cudnn.

    Determinism is on a best-effort basis — some CUDA kernels have
    non-deterministic fallbacks, so two runs may still diverge in the final
    token bits. That's fine for the paper as long as we record the seed
    (done — `seed` is a column in the output CSV).
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


# ──────────────────────────────────────────────
# Phase A — benchmark subsample construction
# ──────────────────────────────────────────────


def _first_present(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def load_ref_manifest(path: Path) -> pd.DataFrame:
    """Load one reference manifest, normalise to [scan_path, dataset_tag].

    Accepts ``filepath`` (IXI-style) or ``ref_path`` (FastMRI-style) or
    ``scan_path`` as the scan-path column. Drops rows with
    ``passed_qc == False`` when the column is present.
    """
    df = pd.read_csv(path)
    scan_col = _first_present(list(df.columns), _REF_SCAN_PATH_COLUMN_ALIASES)
    if scan_col is None:
        raise typer.BadParameter(
            f"{path} must have one of {_REF_SCAN_PATH_COLUMN_ALIASES}"
        )
    if _PASSED_QC_COLUMN in df.columns:
        mask = (
            df[_PASSED_QC_COLUMN]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes"})
        )
        df = df[mask]
    out = pd.DataFrame()
    out[SCAN_COLUMN] = df[scan_col].map(lambda p: str(Path(p).resolve()))
    if DATASET_TAG_COLUMN in df.columns:
        out[DATASET_TAG_COLUMN] = df[DATASET_TAG_COLUMN].astype(str)
    else:
        out[DATASET_TAG_COLUMN] = out[SCAN_COLUMN].map(_infer_dataset_tag)
    return out.reset_index(drop=True)


def _hash_rank(scan_path: str, split_seed: int) -> str:
    """Deterministic pseudo-random rank for a scan_path + seed pair.

    SHA-256 gives a uniform, reproducible ordering that the standard
    ``hash()`` builtin cannot (hash is salted per-process).
    """
    return hashlib.sha256(f"{scan_path}|{split_seed}".encode()).hexdigest()


def subsample_refs(
    ref_frame: pd.DataFrame, n_refs: int, split_seed: int
) -> pd.DataFrame:
    """Take a deterministic n_refs subset by hashing (scan_path, split_seed).

    Sort ascending by SHA-256(``scan_path|split_seed``) and take the first
    ``n_refs``. Produces the same subset whenever the candidate set + seed
    are unchanged. Raises if fewer than ``n_refs`` refs are available.
    """
    if len(ref_frame) < n_refs:
        raise typer.BadParameter(
            f"Asked for {n_refs} refs but only {len(ref_frame)} are "
            "available after quality gating; increase the pool or lower --n-refs."
        )
    ranked = ref_frame.copy()
    ranked["_rank"] = ranked[SCAN_COLUMN].map(
        lambda p: _hash_rank(p, split_seed)
    )
    ranked = ranked.sort_values("_rank").head(n_refs)
    ranked = ranked.drop(columns=["_rank"]).reset_index(drop=True)
    return ranked


def assign_splits(
    refs: pd.DataFrame,
    split_seed: int,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    val_fraction_of_trainval: float = DEFAULT_VAL_FRACTION_OF_TRAINVAL,
) -> pd.DataFrame:
    """Stratify-free 70/15/15 train/val/test split at the ref level.

    Two-stage split: (1) carve off test; (2) carve off val from the
    remainder. All indices come out of ``sklearn.model_selection.
    train_test_split`` with ``random_state=split_seed`` for reproducibility.
    """
    indices = list(refs.index)
    trainval_idx, test_idx = train_test_split(
        indices, test_size=test_fraction, random_state=split_seed, shuffle=True
    )
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_fraction_of_trainval,
        random_state=split_seed,
        shuffle=True,
    )
    labels = pd.Series(index=indices, dtype="object")
    labels.loc[train_idx] = _TRAIN
    labels.loc[val_idx] = _VAL
    labels.loc[test_idx] = _TEST
    out = refs.copy()
    out[SPLIT_COLUMN] = labels.values
    return out


def collect_corruption_rows(
    cor_manifest_path: Path,
    refs: pd.DataFrame,
    corruption_types: tuple[str, ...],
    severities: tuple[int, ...],
) -> pd.DataFrame:
    """Return corruption rows whose ref is in the subsample and whose
    (type, severity) is in the requested grid.

    Inherits the parent ref's ``dataset_tag`` and ``split`` via a left-join
    on ``ref_path → scan_path``.
    """
    required = {
        _COR_REF_PATH_COLUMN,
        _COR_PATH_COLUMN,
        _COR_TYPE_COLUMN,
        _COR_SEVERITY_COLUMN,
    }
    cor = pd.read_csv(cor_manifest_path)
    missing = required - set(cor.columns)
    if missing:
        raise typer.BadParameter(
            f"{cor_manifest_path} missing required cor-manifest columns "
            f"{sorted(missing)}"
        )
    cor[_COR_REF_PATH_COLUMN] = cor[_COR_REF_PATH_COLUMN].map(
        lambda p: str(Path(p).resolve())
    )
    cor[_COR_PATH_COLUMN] = cor[_COR_PATH_COLUMN].map(
        lambda p: str(Path(p).resolve())
    )
    sev_int = cor[_COR_SEVERITY_COLUMN].map(_coerce_int_or_none)
    mask = (
        cor[_COR_REF_PATH_COLUMN].isin(refs[SCAN_COLUMN])
        & cor[_COR_TYPE_COLUMN].isin(corruption_types)
        & sev_int.isin(severities)
    )
    cor = cor.loc[mask].copy()
    cor[_COR_SEVERITY_COLUMN] = sev_int[mask]

    # Carry split + fallback dataset_tag from the parent ref. Rename the
    # ref-side dataset_tag to avoid column collision with cor's own column
    # (both frames carry `dataset_tag`; a plain join would raise).
    ref_lookup = refs.set_index(SCAN_COLUMN)[[SPLIT_COLUMN, DATASET_TAG_COLUMN]]
    ref_lookup = ref_lookup.rename(columns={DATASET_TAG_COLUMN: "_ref_dataset_tag"})
    joined = cor.join(ref_lookup, on=_COR_REF_PATH_COLUMN, how="inner")

    out = pd.DataFrame()
    out[REF_ID_COLUMN] = joined[_COR_REF_PATH_COLUMN].map(_scan_stem)
    out[SCAN_COLUMN] = joined[_COR_PATH_COLUMN]
    out[IS_CLEAN_COLUMN] = False
    out[TYPE_COLUMN] = joined[_COR_TYPE_COLUMN].astype(str)
    out[SEVERITY_COLUMN] = joined[_COR_SEVERITY_COLUMN].astype(int)
    # Prefer the cor manifest's own dataset_tag; fall back to the ref's tag
    # when the cor row doesn't have one (unlikely given Phase 02's schema,
    # but cheap to handle).
    if _COR_DATASET_TAG_COLUMN in joined.columns:
        out[DATASET_TAG_COLUMN] = (
            joined[_COR_DATASET_TAG_COLUMN]
            .where(joined[_COR_DATASET_TAG_COLUMN].notna(), joined["_ref_dataset_tag"])
            .astype(str)
        )
    else:
        out[DATASET_TAG_COLUMN] = joined["_ref_dataset_tag"].astype(str)
    out[SPLIT_COLUMN] = joined[SPLIT_COLUMN].astype(str)
    return out.reset_index(drop=True)


def _coerce_int_or_none(value: object) -> int | None:
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def build_clean_rows(refs: pd.DataFrame) -> pd.DataFrame:
    """One row per ref with is_clean=True; mirrors the corruption-row schema."""
    out = pd.DataFrame()
    out[REF_ID_COLUMN] = refs[SCAN_COLUMN].map(_scan_stem)
    out[SCAN_COLUMN] = refs[SCAN_COLUMN]
    out[IS_CLEAN_COLUMN] = True
    out[TYPE_COLUMN] = _NONE_TYPE
    out[SEVERITY_COLUMN] = _CLEAN_SEVERITY
    out[DATASET_TAG_COLUMN] = refs[DATASET_TAG_COLUMN]
    out[SPLIT_COLUMN] = refs[SPLIT_COLUMN]
    return out.reset_index(drop=True)


def attach_preference(
    frame: pd.DataFrame, preference_csv: Path | None
) -> pd.DataFrame:
    """Left-join machine preference's mean_dice on scan_path == cor_path.

    Missing values stay NaN; the caller logs the count. Clean rows never
    match (there's no ``(ref, ref)`` row in machine_preference.csv) and
    therefore always get NaN preference_score. The spec requires that
    behaviour — selection-bias leakage would follow if we filtered.
    """
    frame = frame.copy()
    if preference_csv is None or not preference_csv.is_file():
        if preference_csv is not None:
            logger.warning(
                "preference CSV not found at %s; filling preference_score=NaN",
                preference_csv,
            )
        frame[PREFERENCE_COLUMN] = math.nan
        return frame

    pref = pd.read_csv(preference_csv)
    if _PREF_COR_COLUMN not in pref.columns or _PREF_SCORE_COLUMN not in pref.columns:
        logger.warning(
            "%s missing '%s' or '%s'; filling preference_score=NaN",
            preference_csv,
            _PREF_COR_COLUMN,
            _PREF_SCORE_COLUMN,
        )
        frame[PREFERENCE_COLUMN] = math.nan
        return frame

    pref[_PREF_COR_COLUMN] = pref[_PREF_COR_COLUMN].map(
        lambda p: str(Path(p).resolve())
    )
    pref_slim = (
        pref[[_PREF_COR_COLUMN, _PREF_SCORE_COLUMN]]
        .drop_duplicates(subset=_PREF_COR_COLUMN, keep="first")
        .rename(columns={_PREF_COR_COLUMN: SCAN_COLUMN, _PREF_SCORE_COLUMN: PREFERENCE_COLUMN})
    )
    merged = frame.merge(pref_slim, on=SCAN_COLUMN, how="left")
    n_missing = int(merged[PREFERENCE_COLUMN].isna().sum())
    logger.info(
        "preference_score: %d/%d rows have a value (%d NaN)",
        len(merged) - n_missing,
        len(merged),
        n_missing,
    )
    return merged


def build_subsample(
    ref_manifest_paths: list[Path],
    cor_manifest_path: Path,
    preference_csv: Path | None,
    n_refs: int,
    split_seed: int,
    severities: tuple[int, ...],
    corruption_types: tuple[str, ...],
) -> pd.DataFrame:
    """Full Phase A pipeline → subsample DataFrame."""
    frames = [load_ref_manifest(p) for p in ref_manifest_paths]
    if not frames:
        raise typer.BadParameter("--ref-manifest is required to build Phase A")
    refs_all = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=SCAN_COLUMN, keep="first"
    )
    logger.info(
        "Ref pool: %d unique scans from %d manifest(s)", len(refs_all), len(frames)
    )
    refs = subsample_refs(refs_all, n_refs=n_refs, split_seed=split_seed)
    refs = assign_splits(refs, split_seed=split_seed)

    clean = build_clean_rows(refs)
    cors = collect_corruption_rows(
        cor_manifest_path, refs, corruption_types, severities
    )
    frame = pd.concat([clean, cors], ignore_index=True)
    frame = attach_preference(frame, preference_csv)

    frame = frame[list(SUBSAMPLE_COLUMNS)]
    frame = frame.sort_values([REF_ID_COLUMN, SCAN_COLUMN]).reset_index(drop=True)
    return frame


def write_subsample(frame: pd.DataFrame, path: Path) -> None:
    """Write the subsample CSV with a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[list(SUBSAMPLE_COLUMNS)].to_csv(path, index=False)


# ──────────────────────────────────────────────
# Phase B — transforms + adapter interface
# ──────────────────────────────────────────────


def build_vlm_transform() -> Callable[[str], torch.Tensor]:
    """Return a MONAI transform that loads a NIfTI → (1, 256, 256, 32) tensor.

    Uses ``ResizeWithPadOrCrop`` instead of the spec's ``CenterSpatialCrop``
    so the output is *guaranteed* to be 256×256×32. ``CenterSpatialCrop``
    only crops; on FastMRI (native 0.6875 mm in-plane × 5 mm slices) the
    volume after 1 mm ``Spacing`` is (220, 220, 70), and a plain crop would
    leave shape (220, 220, 32) — failing the preflight assertion in
    :func:`verify_tensor_shape`. ResizeWithPadOrCrop pads with zeros when
    any dim is smaller than the target, crops when larger, and is the
    idiomatic MONAI way to hit an exact spatial size.

    The caller still needs to ``.permute(0, 3, 1, 2)`` to reach the
    (C, D, H, W) layout M3D-LaMed expects — that step is kept explicit
    in :func:`score_one_scan` so the intent is visible at the call site.
    """
    from monai.transforms import (
        Compose,
        EnsureChannelFirst,
        LoadImage,
        ResizeWithPadOrCrop,
        ScaleIntensityRangePercentiles,
        Spacing,
    )

    return Compose(
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


class VLMAdapter(Protocol):
    """Minimal interface every 3D-VLM adapter must satisfy.

    Adapters own their own model-specific prompt, tensor munging, and
    response parsing. They must leave an empty CUDA cache on ``unload``.
    """

    name: str

    def load(self, device: torch.device, dtype: torch.dtype) -> None: ...

    def run_inference(
        self, vol_cdhw: torch.Tensor, max_new_tokens: int
    ) -> str: ...

    def parse_score(self, raw: str) -> float: ...

    def unload(self) -> None: ...


def _try_load_hf_model(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
) -> Any:
    """Load a HF causal-LM model, preferring flash-attn-2, falling back to sdpa.

    Logs the attention implementation that actually activated (the kwarg may
    be silently ignored by models with custom ``from_pretrained`` logic).
    """
    from transformers import AutoModelForCausalLM

    last_exc: Exception | None = None
    for attn_impl in ("flash_attention_2", "sdpa"):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
            )
            actual = getattr(
                getattr(model, "config", None), "_attn_implementation", None
            )
            logger.info(
                "Loaded %s with attn_implementation=%s (requested %s)",
                model_id,
                actual,
                attn_impl,
            )
            return model.to(device)
        except Exception as exc:  # noqa: BLE001 — broad by design, fallback-driven
            last_exc = exc
            logger.warning(
                "Loading %s with %s failed: %s; trying next attn impl",
                model_id,
                attn_impl,
                exc,
            )
    raise RuntimeError(f"Unable to load {model_id}: {last_exc}")


# ──────────────────────────────────────────────
# M3D-LaMed adapter
# ──────────────────────────────────────────────


@dataclass
class M3DLamedAdapter:
    """Primary 3D-VLM adapter. Uses the project's `QC_PROMPT`.

    Notes:
        * `trust_remote_code=True` — M3D-LaMed ships custom modeling code.
        * Input tensor must be (1, 32, 256, 256); the caller handles the
          permute from MONAI's (1, 256, 256, 32) output.
        * The model's `generate` expects `images` as a torch tensor on the
          same device/dtype as the model.
    """

    model_id: str = "GoodBaiBai88/M3D-LaMed-Phi-3-4B"
    name: str = "m3d_lamed"
    model: Any = None
    tokenizer: Any = None
    device: torch.device | None = None
    dtype: torch.dtype | None = None

    def load(self, device: torch.device, dtype: torch.dtype) -> None:
        from transformers import AutoTokenizer

        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = _try_load_hf_model(
            self.model_id, device=device, dtype=dtype, trust_remote_code=True
        )
        self.model.eval()

    def run_inference(
        self, vol_cdhw: torch.Tensor, max_new_tokens: int = _MAX_NEW_TOKENS_DEFAULT
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("M3DLamedAdapter.load() must be called first")
        from nobrainer.qc.evaluate import QC_PROMPT

        images = vol_cdhw.unsqueeze(0).to(self.device, dtype=self.dtype)
        # M3D-LaMed exposes a .generate(...) method with keyword args
        # (images, question, max_new_tokens) per its model card; wrap defensively
        # because the API may evolve under trust_remote_code.
        with torch.no_grad():
            output = self.model.generate(
                images=images,
                question=QC_PROMPT,
                max_new_tokens=max_new_tokens,
            )
        if isinstance(output, str):
            return output
        if isinstance(output, torch.Tensor):
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
        return str(output)

    def parse_score(self, raw: str) -> float:
        return _normalize_qc_score(raw)

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# ──────────────────────────────────────────────
# Optional adapters — scaffolding only
# ──────────────────────────────────────────────


@dataclass
class RadFMAdapter:
    """Stub adapter for RadFM (Wu et al.).

    The public HF repo for RadFM does not yet expose a stable 3D-volume
    inference API; the published code expects multi-image tokens arranged
    into an "interleaved" prompt. Implementing that inline here would
    duplicate hundreds of lines from its reference repo and drift fast.
    For now, `load` logs a warning and `run_inference` raises
    ``NotImplementedError`` so the per-scan loop catches it and writes a
    NaN row with ``raw_response="ERROR:..."``.

    Revisit when RadFM ships a HF-native 3D endpoint.
    """

    name: str = "radfm"
    _loaded: bool = False

    def load(self, device: torch.device, dtype: torch.dtype) -> None:
        logger.warning(
            "RadFM adapter is a scaffold only; no working 3D-volume HF API. "
            "Every scan will error out until this is implemented."
        )
        self._loaded = True

    def run_inference(
        self, vol_cdhw: torch.Tensor, max_new_tokens: int = _MAX_NEW_TOKENS_DEFAULT
    ) -> str:
        raise NotImplementedError(
            "RadFM 3D-volume inference is not yet wired up in this adapter."
        )

    def parse_score(self, raw: str) -> float:
        return _normalize_qc_score(raw)

    def unload(self) -> None:
        self._loaded = False


@dataclass
class Med2E3Adapter:
    """Stub adapter for Med-2E3 (Wang et al.).

    Same story as :class:`RadFMAdapter` — public repo doesn't expose a
    drop-in 3D-volume pipeline yet. Keeping the class in place so RQ2
    tables can report "Med-2E3: not yet available" rather than silently
    omitting the row.
    """

    name: str = "med2e3"
    _loaded: bool = False

    def load(self, device: torch.device, dtype: torch.dtype) -> None:
        logger.warning(
            "Med-2E3 adapter is a scaffold only; no working 3D-volume HF API."
        )
        self._loaded = True

    def run_inference(
        self, vol_cdhw: torch.Tensor, max_new_tokens: int = _MAX_NEW_TOKENS_DEFAULT
    ) -> str:
        raise NotImplementedError(
            "Med-2E3 3D-volume inference is not yet wired up in this adapter."
        )

    def parse_score(self, raw: str) -> float:
        return _normalize_qc_score(raw)

    def unload(self) -> None:
        self._loaded = False


_ADAPTERS: dict[str, Callable[[], VLMAdapter]] = {
    "m3d_lamed": M3DLamedAdapter,
    "radfm": RadFMAdapter,
    "med2e3": Med2E3Adapter,
}


# ──────────────────────────────────────────────
# Score parsing
# ──────────────────────────────────────────────


def _normalize_qc_score(raw: str) -> float:
    """Map a VLM response to a float in [0, 1] or NaN.

    Delegates the SCORE/REASON parsing to
    ``nobrainer.qc.evaluate.parse_qc_response``, then applies the 1-5 → [0, 1]
    affine normalisation documented in this script's module docstring. NaN
    is reserved for "no integer 1-5 could be recovered at all".
    """
    from nobrainer.qc.evaluate import parse_qc_response

    parsed = parse_qc_response(raw)
    score = parsed.get("score")
    if score is None:
        return math.nan
    try:
        score_int = int(score)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan
    if not 1 <= score_int <= 5:
        return math.nan
    return (score_int - 1) / 4.0


# ──────────────────────────────────────────────
# Phase B — per-scan scoring + CSV I/O
# ──────────────────────────────────────────────


def score_one_scan(
    adapter: VLMAdapter,
    scan_path: Path,
    transform: Callable[[str], torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int = _MAX_NEW_TOKENS_DEFAULT,
) -> tuple[float, str]:
    """Run one scan through one adapter; return ``(score, raw_response)``.

    Handles the three expected exception regimes:
      * ``torch.cuda.OutOfMemoryError`` → empty cache, return (NaN, "OOM").
      * Any other exception → return (NaN, "ERROR:<exc>").
    So the per-scan loop in :func:`main` doesn't need its own try/except.
    """
    try:
        vol = transform(str(scan_path))  # (1, 256, 256, 32)
        vol = vol.permute(0, 3, 1, 2).contiguous()  # (1, 32, 256, 256)
        vol = vol.to(device=device, dtype=dtype)
        raw = adapter.run_inference(vol, max_new_tokens=max_new_tokens)
        return adapter.parse_score(raw), raw
    except torch.cuda.OutOfMemoryError:
        logger.warning("OOM on %s, clearing CUDA cache", scan_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return math.nan, "OOM"
    except Exception as exc:  # noqa: BLE001 — logged + propagated as NaN row
        logger.warning("Inference failure on %s: %s", scan_path, exc)
        return math.nan, f"ERROR:{exc}"


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
    scan_path: str,
    model_name: str,
    score: float,
    raw_response: str,
    seed: int,
) -> None:
    """Append one (scan, model) row to the output CSV (crash-safe per write)."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    is_new = not output_file.exists() or output_file.stat().st_size == 0
    record: dict[str, object] = {
        SCAN_COLUMN: scan_path,
        MODEL_COLUMN: model_name,
        SCORE_COLUMN: score,
        RAW_RESPONSE_COLUMN: raw_response,
        SEED_COLUMN: seed,
    }
    with output_file.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        if is_new:
            writer.writeheader()
        writer.writerow(record)
        handle.flush()


# ──────────────────────────────────────────────
# Tensor-shape preflight check
# ──────────────────────────────────────────────


def verify_tensor_shape(
    scan_path: Path,
    transform: Callable[[str], torch.Tensor],
    png_out: Path = Path("/tmp/axis_check.png"),
) -> None:
    """Load one scan, permute to (C, D, H, W), assert shape + save a mid-slice.

    Intended to be run once before a real sweep. Aborts the script if the
    tensor shape doesn't match the M3D-LaMed target, because every
    downstream generate() call will otherwise crash with a misaligned
    attention mask.
    """
    vol = transform(str(scan_path))
    vol = vol.permute(0, 3, 1, 2).contiguous()
    expected = torch.Size([1, *_VLM_TARGET_DHW])
    logger.info(
        "Verify: shape=%s min=%.4f max=%.4f",
        tuple(vol.shape),
        float(vol.min()),
        float(vol.max()),
    )
    if vol.shape != expected:
        raise RuntimeError(
            f"Tensor shape {tuple(vol.shape)} != expected {tuple(expected)}; "
            "aborting before launching the VLM sweep."
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        middle = vol[0, _VLM_TARGET_DHW[0] // 2, :, :].float().cpu().numpy()
        png_out.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(str(png_out), middle, cmap="gray")
        logger.info("Wrote middle-slice preview to %s — eyeball before full run", png_out)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write PNG preview: %s", exc)


# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────


def _print_summary(
    total: int,
    already_done: int,
    processed: int,
    failed: int,
    output_file: Path,
    console: Console,
) -> None:
    table = Table(title="Phase 08a — 3D VLM evaluation")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("total (scan × model) pairs", str(total))
    table.add_row("already in CSV (skipped)", str(already_done))
    table.add_row("processed this run", str(processed))
    table.add_row("inference failures (NaN)", str(failed))
    table.add_row("output file", str(output_file))
    console.print(table)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def _parse_int_list(spec: str) -> tuple[int, ...]:
    items = [x.strip() for x in spec.split(",") if x.strip()]
    return tuple(int(x) for x in items)


def _parse_str_list(spec: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in spec.split(",") if x.strip())


def _validate_models(models: tuple[str, ...]) -> tuple[str, ...]:
    invalid = [m for m in models if m not in _MODEL_CHOICES]
    if invalid:
        raise typer.BadParameter(
            f"Unknown model(s) {invalid}; choose from {_MODEL_CHOICES}"
        )
    return models


@app.command()
def main(
    seed: int = typer.Option(..., "--seed", help="Torch/Python RNG seed."),
    models: str = typer.Option(
        "m3d_lamed",
        "--models",
        help=f"Comma-separated subset of {_MODEL_CHOICES}.",
    ),
    subsample_manifest: Path = typer.Option(
        Path("results/tables/benchmark_subsample.csv"),
        "--subsample-manifest",
        resolve_path=True,
        help="Benchmark subsample CSV. Built from ref/cor/preference inputs "
        "if the file is absent; loaded verbatim if present.",
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output-file",
        resolve_path=True,
        help="Output CSV. Defaults to results/tables/3d_vlm_scores_seed_{seed}.csv.",
    ),
    ref_manifest: list[Path] = typer.Option(
        [],
        "--ref-manifest",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Reference-scan manifest (repeatable). Used only when Phase A "
        "has to build the subsample manifest from scratch.",
    ),
    cor_manifest: Path | None = typer.Option(
        None,
        "--cor-manifest",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="corruption_manifest.csv — required when Phase A runs.",
    ),
    preference_csv: Path | None = typer.Option(
        None,
        "--preference-csv",
        exists=False,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Optional machine_preference.csv for preference_score annotation.",
    ),
    n_refs: int = typer.Option(DEFAULT_N_REFS, "--n-refs"),
    severities: str = typer.Option(
        ",".join(str(s) for s in DEFAULT_SEVERITIES), "--severities"
    ),
    corruption_types: str = typer.Option(
        ",".join(DEFAULT_CORRUPTION_TYPES), "--corruption-types"
    ),
    split_seed: int = typer.Option(DEFAULT_SPLIT_SEED, "--split-seed"),
    max_new_tokens: int = typer.Option(
        _MAX_NEW_TOKENS_DEFAULT, "--max-new-tokens"
    ),
    verify_shape: bool = typer.Option(
        True,
        "--verify-shape/--no-verify-shape",
        help="Load one scan and assert the expected (1, 32, 256, 256) "
        "tensor before launching the sweep.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Log pending counts + first-10 scans, exit."
    ),
) -> None:
    """Evaluate one or more 3D VLMs on the benchmark subsample."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )

    set_seeds(seed)

    selected_models = _validate_models(_parse_str_list(models))
    sev_tuple = _parse_int_list(severities)
    type_tuple = _parse_str_list(corruption_types)

    # ── Phase A ──
    if subsample_manifest.is_file():
        logger.info("Loading existing subsample manifest from %s", subsample_manifest)
        subsample = pd.read_csv(subsample_manifest)
    else:
        if cor_manifest is None:
            raise typer.BadParameter(
                "--cor-manifest is required when the subsample manifest "
                "does not already exist."
            )
        logger.info(
            "Building subsample manifest: n_refs=%d severities=%s types=%s",
            n_refs, sev_tuple, type_tuple,
        )
        subsample = build_subsample(
            ref_manifest_paths=list(ref_manifest),
            cor_manifest_path=cor_manifest,
            preference_csv=preference_csv,
            n_refs=n_refs,
            split_seed=split_seed,
            severities=sev_tuple,
            corruption_types=type_tuple,
        )
        write_subsample(subsample, subsample_manifest)
        logger.info("Wrote %d rows to %s", len(subsample), subsample_manifest)

    # ── Phase B setup ──
    resolved_output = output_file or Path(
        f"results/tables/3d_vlm_scores_seed_{seed}.csv"
    ).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if device.type == "cpu":
        logger.warning("CUDA not available; inference will be slow on CPU.")
    logger.info(
        "Device=%s dtype=%s models=%s seed=%d output=%s",
        device, dtype, selected_models, seed, resolved_output,
    )

    transform = build_vlm_transform()

    # ── Preflight shape check ──
    if verify_shape and not dry_run and len(subsample) > 0:
        first_scan = Path(subsample.iloc[0][SCAN_COLUMN])
        if first_scan.is_file():
            verify_tensor_shape(first_scan, transform)
        else:
            logger.warning(
                "First scan %s not on disk; skipping shape preflight.", first_scan
            )

    # ── Resume bookkeeping ──
    done = load_existing(resolved_output)
    pending: list[tuple[str, str]] = []
    for scan_path in subsample[SCAN_COLUMN].tolist():
        for model_name in selected_models:
            if (scan_path, model_name) in done:
                continue
            pending.append((scan_path, model_name))

    logger.info(
        "Plan: %d total pairs, %d already done, %d pending",
        len(subsample) * len(selected_models),
        len(done),
        len(pending),
    )

    if dry_run:
        for scan_path, model_name in pending[:10]:
            logger.info("  would score: %s × %s", scan_path, model_name)
        if len(pending) > 10:
            logger.info("  ... (%d more)", len(pending) - 10)
        return

    # ── Inference, one model at a time (so we can unload between models) ──
    processed = 0
    failed = 0
    for model_name in selected_models:
        per_model = [sp for sp, mn in pending if mn == model_name]
        if not per_model:
            logger.info("No pending scans for %s; skipping load.", model_name)
            continue
        adapter_cls = _ADAPTERS[model_name]
        adapter: VLMAdapter = adapter_cls()  # type: ignore[call-arg]
        try:
            adapter.load(device=device, dtype=dtype)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to load %s (%s); skipping the model entirely.",
                model_name,
                exc,
            )
            continue

        iterator = (
            tqdm(per_model, desc=f"vlm:{model_name}", unit="scan")
            if len(per_model) > 1
            else per_model
        )
        for scan_path in iterator:
            score, raw = score_one_scan(
                adapter,
                Path(scan_path),
                transform,
                device=device,
                dtype=dtype,
                max_new_tokens=max_new_tokens,
            )
            append_row(
                resolved_output,
                scan_path=scan_path,
                model_name=model_name,
                score=score,
                raw_response=raw,
                seed=seed,
            )
            processed += 1
            if math.isnan(score):
                failed += 1

        adapter.unload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    _print_summary(
        total=len(subsample) * len(selected_models),
        already_done=len(done),
        processed=processed,
        failed=failed,
        output_file=resolved_output,
        console=Console(),
    )


if __name__ == "__main__":
    app()
