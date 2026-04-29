#!/usr/bin/env python3
"""NeuroQC Phase 08b — evaluate 2D VLMs on extracted slices (RQ2 paired).

Runs on the same ref set as `code/08a_eval_3d_vlms.py` by reading the
benchmark subsample it writes — so RQ2 (native-3D vs 2D-slice) compares
like-for-like scans rather than re-subsampling. Aborts with a clear error
if the subsample manifest is missing.

Four VLMs, four code paths:

  Local (HF transformers, `model.generate`):
    * LLaVA-OneVision-7B  — native multi-image.
    * Qwen2-VL-7B         — native multi-image.
    * MedGemma-4B-IT      — gated weights; single-image-only processor, so
                            the 3 slices are concatenated horizontally into
                            one PIL image (mode ``concat_grid``). If HF
                            access is denied, the model is skipped with a
                            warning instead of failing the run.

  Remote (OpenAI chat.completions):
    * GPT-4o (via `--gpt-model` CLI argument; default
      ``gpt-4o-2024-11-20``). Honours `--max-api-calls` and
      `--max-budget-usd` budget ceilings; skipped gracefully when
      ``OPENAI_API_KEY`` is not in the environment.

Two slice strategies (via `nobrainer.qc.slice_extractor.extract_slices`):

  * ``mid``      — middle axial + coronal + sagittal slice.
  * ``max_info`` — per-orientation argmax of non-zero voxel count (Vote-MI
                   proxy).

**Multi-image protocol** (paper-critical, documented here so the reviewer
can trace the methodology):

  Each scan is scored ONCE per (model, strategy) pair. The 3 orientation
  slices travel to the VLM in a single call — either as a native
  multi-image input (`multi_image_mode="multi_image"`) or as a single
  horizontally-concatenated PIL (`multi_image_mode="concat_grid"`). This
  mirrors Vote-MI (NeurIPS 2024) and keeps RQ2 a valid paired comparison
  against the 3D evaluator's "one score per scan per model" contract.

Resume is keyed on the (scan_path, model, slice_strategy) triple so a
re-run after any transient failure picks up exactly where it left off.

Inputs:
    --seed INT                  Required. Torch/Python RNG seed + written
                                per-row into the output CSV. Also used as
                                the OpenAI `seed` parameter on GPT-4o calls
                                (best-effort determinism; not guaranteed by
                                OpenAI).
    --subsample-manifest PATH   `benchmark_subsample.csv` written by 08a.
    --output-file PATH          Default
                                ``results/tables/2d_vlm_scores_seed_{seed}.csv``.
    --models TEXT               Comma-separated from
                                {llava_ov, qwen2_vl, medgemma, gpt4o}.
    --strategies TEXT           Comma-separated from {mid, max_info}.
    --slice-cache-dir PATH      PNG cache root. PNGs saved as
                                ``{cache_dir}/{strategy}/{scan_stem}_slice{0,1,2}.png``.
    --gpt-model TEXT            Default ``gpt-4o-2024-11-20``. Pass the
                                current stable GPT-4o ID.
    --max-api-calls INT         Upper bound on GPT-4o calls this run.
    --max-budget-usd FLOAT      Upper bound on estimated GPT-4o spend.
    --per-call-usd FLOAT        Estimated cost per GPT-4o call; multiplied
                                by the running call count to track spend.
    --dry-run FLAG              Log pending triple counts + first 10, exit.

Output schema (one row per (scan, model, strategy) triple):
    scan_path, model, slice_strategy, score, raw_response, seed,
    n_slices, multi_image_mode
"""

from __future__ import annotations

import base64
import csv
import gc
import io
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import pandas as pd
import torch
import typer
from PIL import Image
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from tqdm import tqdm

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

# Output CSV schema.
SCAN_COLUMN: str = "scan_path"
MODEL_COLUMN: str = "model"
STRATEGY_COLUMN: str = "slice_strategy"
SCORE_COLUMN: str = "score"
RAW_RESPONSE_COLUMN: str = "raw_response"
SEED_COLUMN: str = "seed"
N_SLICES_COLUMN: str = "n_slices"
MULTI_IMAGE_MODE_COLUMN: str = "multi_image_mode"
OUTPUT_COLUMNS: tuple[str, ...] = (
    SCAN_COLUMN,
    MODEL_COLUMN,
    STRATEGY_COLUMN,
    SCORE_COLUMN,
    RAW_RESPONSE_COLUMN,
    SEED_COLUMN,
    N_SLICES_COLUMN,
    MULTI_IMAGE_MODE_COLUMN,
)

# Subsample-manifest column we consume.
_SUBSAMPLE_SCAN_COLUMN: str = "scan_path"

# Strategy names (match `nobrainer.qc.slice_extractor.extract_slices`
# `method` parameter). Three orientations per strategy.
STRATEGY_MID: str = "mid"
STRATEGY_MAX_INFO: str = "max_info"
DEFAULT_STRATEGIES: tuple[str, ...] = (STRATEGY_MID, STRATEGY_MAX_INFO)
_ORIENTATIONS: tuple[str, ...] = ("axial", "coronal", "sagittal")
_N_SLICES: int = 3
_MAX_NEW_TOKENS_DEFAULT: int = 16

# Multi-image protocol tags (written to the CSV).
MULTI_IMAGE: str = "multi_image"
CONCAT_GRID: str = "concat_grid"

# Model registry tags.
_MODEL_CHOICES: tuple[str, ...] = ("llava_ov", "qwen2_vl", "medgemma", "gpt4o")
DEFAULT_MODELS: tuple[str, ...] = _MODEL_CHOICES

# GPT-4o defaults. `--gpt-model` is CLI so users can track OpenAI's naming
# churn without editing source; pricing is a tunable too.
DEFAULT_GPT_MODEL: str = "gpt-4o-2024-11-20"
DEFAULT_MAX_API_CALLS: int = 8000
DEFAULT_MAX_BUDGET_USD: float = 50.0
DEFAULT_PER_CALL_USD: float = 0.005

_BUDGET_SENTINEL: str = "BUDGET_CAP"
_NO_API_KEY_SENTINEL: str = "NO_API_KEY"

logger = logging.getLogger(__name__)
app = typer.Typer(
    help="NeuroQC Phase 08b — 2D VLM quality evaluation on extracted slices.",
    add_completion=False,
)


# ──────────────────────────────────────────────
# Helpers: seeding + path utilities
# ──────────────────────────────────────────────


def _scan_stem(path: Path | str) -> str:
    """Strip ``.nii.gz`` / ``.nii`` from a filename, return the bare stem."""
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return Path(name).stem


def set_seeds(seed: int) -> None:
    """Mirror 08a's deterministic-best-effort seeding."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


# ──────────────────────────────────────────────
# Slice extraction + PNG cache
# ──────────────────────────────────────────────


def _slice_paths(cache_dir: Path, strategy: str, scan_path: Path) -> list[Path]:
    """PNG cache locations for one (scan, strategy), indices 0/1/2.

    Layout: ``{cache_dir}/{strategy}/{scan_stem}_slice{i}.png``. Indices
    match `_ORIENTATIONS` order (0=axial, 1=coronal, 2=sagittal) so a
    downstream consumer can reconstruct anatomy from the filename.
    """
    stem = _scan_stem(scan_path)
    return [
        cache_dir / strategy / f"{stem}_slice{i}.png" for i in range(_N_SLICES)
    ]


def _tensor_to_pil_rgb(slice_2d: torch.Tensor) -> Image.Image:
    """Convert a uint8 (H, W) torch tensor to an RGB PIL Image.

    VLM processors expect PIL; `nobrainer.qc.slice_extractor` returns
    tensors. `numpy` is a necessary boundary here (`Image.fromarray` has
    no pure-torch path) — CLAUDE.md permits PIL for PNG-slice export,
    which is the same conversion pattern.
    """
    import numpy as np

    arr = slice_2d.cpu().numpy().astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def extract_and_cache_slices(
    scan_path: Path, strategy: str, cache_dir: Path
) -> list[Image.Image]:
    """Return 3 PIL RGB slices for ``(scan_path, strategy)``, caching to disk.

    Cache hit: load the 3 PNGs from ``cache_dir/strategy/`` and return them.
    Cache miss: call `nobrainer.qc.slice_extractor.extract_slices` with
    ``method=strategy`` and ``orientations=("axial", "coronal", "sagittal")``,
    convert each returned tensor to PIL RGB, write the PNGs, return the list.

    ``extract_slices`` returns a ``dict[str, torch.Tensor]`` with keys like
    ``{strategy}_{orientation}``; we fetch them in orientation order.
    """
    from nobrainer.qc.slice_extractor import extract_slices

    cache_paths = _slice_paths(cache_dir, strategy, scan_path)
    if all(p.is_file() for p in cache_paths):
        return [Image.open(p).convert("RGB") for p in cache_paths]

    results = extract_slices(
        scan_path=scan_path,
        method=strategy,
        orientations=list(_ORIENTATIONS),
    )
    pil_slices: list[Image.Image] = []
    for i, orient in enumerate(_ORIENTATIONS):
        key = f"{strategy}_{orient}"
        if key not in results:
            raise RuntimeError(
                f"extract_slices missing expected key {key!r} — "
                f"has {sorted(results)}"
            )
        pil = _tensor_to_pil_rgb(results[key])
        pil_slices.append(pil)

    # Persist PNGs atomically-ish (write to .tmp, rename).
    for path, img in zip(cache_paths, pil_slices, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        img.save(tmp)
        tmp.replace(path)
    return pil_slices


def _concat_horizontal(slices: list[Image.Image]) -> Image.Image:
    """Horizontally concatenate a list of same-height PIL images.

    Used when a VLM processor doesn't accept multi-image inputs (MedGemma
    today). Produces a single (sum_W, max_H) RGB image; each constituent
    keeps its native aspect. ``multi_image_mode="concat_grid"`` is logged
    on the output row so we can trace the protocol post-hoc.
    """
    if not slices:
        raise ValueError("cannot concat an empty slice list")
    heights = [s.size[1] for s in slices]
    max_h = max(heights)
    widths = [s.size[0] for s in slices]
    total_w = sum(widths)
    canvas = Image.new("RGB", (total_w, max_h), color=(0, 0, 0))
    x = 0
    for img in slices:
        # Center-vertical if heights differ (defensive; in practice all 3
        # orientations at 256 iso give matching sizes).
        y = (max_h - img.size[1]) // 2
        canvas.paste(img, (x, y))
        x += img.size[0]
    return canvas


# ──────────────────────────────────────────────
# Score normalization (shared with 08a's idiom)
# ──────────────────────────────────────────────


def _normalize_qc_score(raw: str) -> float:
    """Map a VLM response to [0, 1] via `parse_qc_response`, NaN otherwise."""
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
# Adapter protocol + local VLMs
# ──────────────────────────────────────────────


class VLM2DAdapter(Protocol):
    """Interface every 2D VLM adapter must implement.

    Adapters own their own multi-image protocol (`multi_image_mode`), HF
    model-loading quirks, and prompt-response handling. The main loop only
    calls ``load`` → ``run_inference`` → ``parse_score`` → ``unload``.
    """

    name: str
    multi_image_mode: str  # {"multi_image", "concat_grid"}

    def load(self, device: torch.device, dtype: torch.dtype) -> None: ...

    def run_inference(
        self, slices: list[Image.Image], max_new_tokens: int
    ) -> str: ...

    def parse_score(self, raw: str) -> float: ...

    def unload(self) -> None: ...


def _try_load_hf_vlm(
    model_id: str,
    cls: Any,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool = False,
) -> Any:
    """Load a HF VLM preferring flash_attention_2, falling back to sdpa.

    Logs the ``config._attn_implementation`` that actually activated after
    load — `from_pretrained` may silently ignore the kwarg on some models.
    """
    last_exc: Exception | None = None
    for attn_impl in ("flash_attention_2", "sdpa"):
        try:
            model = cls.from_pretrained(
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
                model_id, actual, attn_impl,
            )
            return model.to(device).eval()
        except Exception as exc:  # noqa: BLE001 — fallback-driven
            last_exc = exc
            logger.warning(
                "Loading %s with %s failed: %s; trying next attn impl",
                model_id, attn_impl, exc,
            )
    raise RuntimeError(f"Unable to load {model_id}: {last_exc}")


@dataclass
class LlavaOVAdapter:
    """LLaVA-OneVision-7B. Native multi-image support via AutoProcessor."""

    model_id: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    name: str = "llava_ov"
    multi_image_mode: str = MULTI_IMAGE
    model: Any = None
    processor: Any = None
    device: torch.device | None = None
    dtype: torch.dtype | None = None

    def load(self, device: torch.device, dtype: torch.dtype) -> None:
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

        self.device, self.dtype = device, dtype
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = _try_load_hf_vlm(
            self.model_id,
            LlavaOnevisionForConditionalGeneration,
            device=device,
            dtype=dtype,
        )

    def run_inference(
        self, slices: list[Image.Image], max_new_tokens: int = _MAX_NEW_TOKENS_DEFAULT
    ) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("LlavaOVAdapter.load() must be called first")
        from nobrainer.qc.evaluate import QC_PROMPT

        inputs = self.processor(
            images=slices, text=QC_PROMPT, return_tensors="pt"
        )
        inputs = {
            k: (
                v.to(self.device, self.dtype)
                if torch.is_tensor(v) and v.is_floating_point()
                else (v.to(self.device) if torch.is_tensor(v) else v)
            )
            for k, v in inputs.items()
        }
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        return self.processor.decode(out[0], skip_special_tokens=True)

    def parse_score(self, raw: str) -> float:
        return _normalize_qc_score(raw)

    def unload(self) -> None:
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


@dataclass
class Qwen2VLAdapter:
    """Qwen2-VL-7B-Instruct. Native multi-image support via AutoProcessor."""

    model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    name: str = "qwen2_vl"
    multi_image_mode: str = MULTI_IMAGE
    model: Any = None
    processor: Any = None
    device: torch.device | None = None
    dtype: torch.dtype | None = None

    def load(self, device: torch.device, dtype: torch.dtype) -> None:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.device, self.dtype = device, dtype
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = _try_load_hf_vlm(
            self.model_id,
            Qwen2VLForConditionalGeneration,
            device=device,
            dtype=dtype,
        )

    def run_inference(
        self, slices: list[Image.Image], max_new_tokens: int = _MAX_NEW_TOKENS_DEFAULT
    ) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("Qwen2VLAdapter.load() must be called first")
        from nobrainer.qc.evaluate import QC_PROMPT

        inputs = self.processor(
            images=slices, text=QC_PROMPT, return_tensors="pt"
        )
        inputs = {
            k: (
                v.to(self.device, self.dtype)
                if torch.is_tensor(v) and v.is_floating_point()
                else (v.to(self.device) if torch.is_tensor(v) else v)
            )
            for k, v in inputs.items()
        }
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        return self.processor.decode(out[0], skip_special_tokens=True)

    def parse_score(self, raw: str) -> float:
        return _normalize_qc_score(raw)

    def unload(self) -> None:
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


@dataclass
class MedGemmaAdapter:
    """MedGemma-4B-IT. Gated weights; single-image processor → concat grid.

    Handles two failure modes gracefully:
      * HF access denied (401 / HFValidationError) → `load` raises a tagged
        exception that the main loop catches and skips the model with a
        warning.
      * Concat-grid path: 3 PIL slices → one horizontally-concatenated
        PIL image. ``multi_image_mode="concat_grid"`` is what gets logged.
    """

    model_id: str = "google/medgemma-4b-it"
    name: str = "medgemma"
    multi_image_mode: str = CONCAT_GRID
    model: Any = None
    processor: Any = None
    device: torch.device | None = None
    dtype: torch.dtype | None = None

    def load(self, device: torch.device, dtype: torch.dtype) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.device, self.dtype = device, dtype
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = _try_load_hf_vlm(
            self.model_id,
            AutoModelForCausalLM,
            device=device,
            dtype=dtype,
            trust_remote_code=True,
        )

    def run_inference(
        self, slices: list[Image.Image], max_new_tokens: int = _MAX_NEW_TOKENS_DEFAULT
    ) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("MedGemmaAdapter.load() must be called first")
        from nobrainer.qc.evaluate import QC_PROMPT

        concat = _concat_horizontal(slices)
        inputs = self.processor(
            images=concat, text=QC_PROMPT, return_tensors="pt"
        )
        inputs = {
            k: (
                v.to(self.device, self.dtype)
                if torch.is_tensor(v) and v.is_floating_point()
                else (v.to(self.device) if torch.is_tensor(v) else v)
            )
            for k, v in inputs.items()
        }
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        return self.processor.decode(out[0], skip_special_tokens=True)

    def parse_score(self, raw: str) -> float:
        return _normalize_qc_score(raw)

    def unload(self) -> None:
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# ──────────────────────────────────────────────
# GPT-4o adapter (remote, budget-aware)
# ──────────────────────────────────────────────


@dataclass
class GPT4oAdapter:
    """OpenAI chat.completions adapter with budget + retry handling.

    Budget (calls + USD) is enforced in `run_inference`: when either
    ceiling is hit, the adapter flips ``budget_exhausted=True`` and returns
    a sentinel raw response so the main loop can short-circuit cleanly.

    Retry policy:
      * RateLimitError → exponential backoff 1s → 16s, max 5 attempts.
      * APITimeoutError → single retry with 60s timeout.
      * Anything else  → caught upstream; this run returns score=NaN.
    """

    model_name: str = DEFAULT_GPT_MODEL
    name: str = "gpt4o"
    multi_image_mode: str = MULTI_IMAGE
    max_calls: int = DEFAULT_MAX_API_CALLS
    max_budget_usd: float = DEFAULT_MAX_BUDGET_USD
    per_call_usd: float = DEFAULT_PER_CALL_USD
    seed: int = 0
    client: Any = None
    call_count: int = 0
    budget_exhausted: bool = False
    _no_api_key: bool = False

    def load(self, device: torch.device, dtype: torch.dtype) -> None:
        """`device`/`dtype` are ignored; GPT-4o is remote."""
        del device, dtype
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY not set; GPT-4o adapter will skip with NO_API_KEY sentinel."
            )
            self._no_api_key = True
            return
        try:
            import openai

            self.client = openai.OpenAI(api_key=api_key)
        except ImportError as exc:
            logger.warning(
                "openai SDK unavailable (%s); GPT-4o skipped.", exc
            )
            self._no_api_key = True

    def _estimated_cost(self) -> float:
        return self.call_count * self.per_call_usd

    def _budget_hit(self) -> bool:
        return (
            self.call_count >= self.max_calls
            or self._estimated_cost() >= self.max_budget_usd
        )

    def run_inference(
        self, slices: list[Image.Image], max_new_tokens: int = _MAX_NEW_TOKENS_DEFAULT
    ) -> str:
        if self._no_api_key:
            return _NO_API_KEY_SENTINEL
        if self.client is None:
            raise RuntimeError("GPT4oAdapter.load() must be called first")
        if self._budget_hit():
            self.budget_exhausted = True
            logger.info(
                "GPT-4o budget ceiling hit (%d calls, est $%.2f); skipping.",
                self.call_count, self._estimated_cost(),
            )
            return _BUDGET_SENTINEL

        from nobrainer.qc.evaluate import QC_PROMPT
        import openai  # local — already imported in load(), keeps mypy happy

        content: list[dict[str, Any]] = [{"type": "text", "text": QC_PROMPT}]
        for img in slices:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                        "detail": "high",
                    },
                }
            )
        messages = [{"role": "user", "content": content}]

        delay = 1
        for attempt in range(5):
            try:
                self.call_count += 1
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                    seed=self.seed,
                    max_tokens=max_new_tokens,
                )
                return response.choices[0].message.content or ""
            except openai.RateLimitError as exc:
                logger.warning(
                    "GPT-4o rate-limited (attempt %d/5): %s; backing off %ds",
                    attempt + 1, exc, delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 16)
            except openai.APITimeoutError as exc:
                if attempt == 0:
                    logger.warning("GPT-4o timeout; retrying once at 60s")
                    continue
                return f"ERROR:{exc}"
            except Exception as exc:  # noqa: BLE001
                return f"ERROR:{exc}"
        return f"ERROR:GPT-4o exhausted retries"

    def parse_score(self, raw: str) -> float:
        if raw in (_BUDGET_SENTINEL, _NO_API_KEY_SENTINEL):
            return math.nan
        return _normalize_qc_score(raw)

    def unload(self) -> None:
        # Nothing to free — `client` just holds a thin HTTP wrapper.
        self.client = None


_ADAPTERS: dict[str, Callable[[], VLM2DAdapter]] = {
    "llava_ov": LlavaOVAdapter,
    "qwen2_vl": Qwen2VLAdapter,
    "medgemma": MedGemmaAdapter,
    "gpt4o": GPT4oAdapter,
}


# ──────────────────────────────────────────────
# Per-triple scoring
# ──────────────────────────────────────────────


@dataclass
class TripleResult:
    score: float
    raw_response: str
    n_slices: int
    multi_image_mode: str


def score_one_triple(
    adapter: VLM2DAdapter,
    scan_path: Path,
    strategy: str,
    cache_dir: Path,
    max_new_tokens: int = _MAX_NEW_TOKENS_DEFAULT,
) -> TripleResult:
    """Run one (scan, strategy) triple through one adapter → `TripleResult`.

    OOM / arbitrary-exception handling matches 08a's `score_one_scan`. The
    n_slices and multi_image_mode fields are logged so paper tables can
    reconstruct the per-model protocol without re-reading the source.
    """
    try:
        slices = extract_and_cache_slices(scan_path, strategy, cache_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Slice extraction failed on %s: %s", scan_path, exc)
        return TripleResult(
            score=math.nan,
            raw_response=f"ERROR:slice_extract:{exc}",
            n_slices=0,
            multi_image_mode=adapter.multi_image_mode,
        )

    try:
        raw = adapter.run_inference(slices, max_new_tokens=max_new_tokens)
        return TripleResult(
            score=adapter.parse_score(raw),
            raw_response=raw,
            n_slices=len(slices),
            multi_image_mode=adapter.multi_image_mode,
        )
    except torch.cuda.OutOfMemoryError:
        logger.warning("OOM on %s; clearing CUDA cache", scan_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return TripleResult(
            score=math.nan,
            raw_response="OOM",
            n_slices=len(slices),
            multi_image_mode=adapter.multi_image_mode,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Inference failure on %s: %s", scan_path, exc)
        return TripleResult(
            score=math.nan,
            raw_response=f"ERROR:{exc}",
            n_slices=len(slices),
            multi_image_mode=adapter.multi_image_mode,
        )


# ──────────────────────────────────────────────
# CSV I/O + resume
# ──────────────────────────────────────────────


def load_existing(output_file: Path) -> set[tuple[str, str, str]]:
    """Return ``{(scan_path, model, slice_strategy)}`` triples on disk."""
    if not output_file.exists() or output_file.stat().st_size == 0:
        return set()
    with output_file.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        needed = {SCAN_COLUMN, MODEL_COLUMN, STRATEGY_COLUMN}
        if reader.fieldnames is None or not needed.issubset(reader.fieldnames):
            return set()
        return {
            (row[SCAN_COLUMN], row[MODEL_COLUMN], row[STRATEGY_COLUMN])
            for row in reader
        }


def append_row(
    output_file: Path,
    scan_path: str,
    model_name: str,
    strategy: str,
    result: TripleResult,
    seed: int,
) -> None:
    """Append one (scan, model, strategy) row, flushing after each write."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    is_new = not output_file.exists() or output_file.stat().st_size == 0
    record: dict[str, object] = {
        SCAN_COLUMN: scan_path,
        MODEL_COLUMN: model_name,
        STRATEGY_COLUMN: strategy,
        SCORE_COLUMN: result.score,
        RAW_RESPONSE_COLUMN: result.raw_response,
        SEED_COLUMN: seed,
        N_SLICES_COLUMN: result.n_slices,
        MULTI_IMAGE_MODE_COLUMN: result.multi_image_mode,
    }
    with output_file.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        if is_new:
            writer.writeheader()
        writer.writerow(record)
        handle.flush()


# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────


def _print_summary(
    total: int,
    already_done: int,
    processed: int,
    failed: int,
    budget_hits: int,
    output_file: Path,
    console: Console,
) -> None:
    table = Table(title="Phase 08b — 2D VLM evaluation")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("total (scan × model × strategy) triples", str(total))
    table.add_row("already in CSV (skipped)", str(already_done))
    table.add_row("processed this run", str(processed))
    table.add_row("inference failures (NaN)", str(failed))
    table.add_row("GPT-4o budget/quota skips", str(budget_hits))
    table.add_row("output file", str(output_file))
    console.print(table)


# ──────────────────────────────────────────────
# CLI parsing helpers
# ──────────────────────────────────────────────


def _parse_str_list(spec: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in spec.split(",") if x.strip())


def _validate_models(models: tuple[str, ...]) -> tuple[str, ...]:
    invalid = [m for m in models if m not in _MODEL_CHOICES]
    if invalid:
        raise typer.BadParameter(
            f"Unknown model(s) {invalid}; choose from {_MODEL_CHOICES}"
        )
    return models


def _validate_strategies(strategies: tuple[str, ...]) -> tuple[str, ...]:
    invalid = [s for s in strategies if s not in (STRATEGY_MID, STRATEGY_MAX_INFO)]
    if invalid:
        raise typer.BadParameter(
            f"Unknown strategy(ies) {invalid}; choose from "
            f"{(STRATEGY_MID, STRATEGY_MAX_INFO)}"
        )
    return strategies


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


@app.command()
def main(
    seed: int = typer.Option(..., "--seed", help="Torch/Python RNG seed."),
    subsample_manifest: Path = typer.Option(
        Path("results/tables/benchmark_subsample.csv"),
        "--subsample-manifest",
        resolve_path=True,
        help="Benchmark subsample CSV from 08a Phase A.",
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output-file",
        resolve_path=True,
        help="Defaults to results/tables/2d_vlm_scores_seed_{seed}.csv.",
    ),
    models: str = typer.Option(
        ",".join(DEFAULT_MODELS),
        "--models",
        help=f"Comma-separated subset of {_MODEL_CHOICES}.",
    ),
    strategies: str = typer.Option(
        ",".join(DEFAULT_STRATEGIES), "--strategies",
        help="Comma-separated slice strategies.",
    ),
    slice_cache_dir: Path = typer.Option(
        Path("data/derivatives/slices"),
        "--slice-cache-dir",
        resolve_path=True,
    ),
    gpt_model: str = typer.Option(DEFAULT_GPT_MODEL, "--gpt-model"),
    max_api_calls: int = typer.Option(DEFAULT_MAX_API_CALLS, "--max-api-calls"),
    max_budget_usd: float = typer.Option(DEFAULT_MAX_BUDGET_USD, "--max-budget-usd"),
    per_call_usd: float = typer.Option(DEFAULT_PER_CALL_USD, "--per-call-usd"),
    max_new_tokens: int = typer.Option(
        _MAX_NEW_TOKENS_DEFAULT, "--max-new-tokens"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print pending triple counts + first 10, exit."
    ),
) -> None:
    """Evaluate one or more 2D VLMs on the benchmark subsample."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )

    if not subsample_manifest.is_file():
        raise typer.BadParameter(
            f"benchmark_subsample.csv not found at {subsample_manifest}. "
            "Run `python code/08a_eval_3d_vlms.py ...` Phase A first "
            "(or pass --ref-manifest / --cor-manifest to 08a's CLI)."
        )

    set_seeds(seed)

    selected_models = _validate_models(_parse_str_list(models))
    selected_strategies = _validate_strategies(_parse_str_list(strategies))

    subsample = pd.read_csv(subsample_manifest)
    if _SUBSAMPLE_SCAN_COLUMN not in subsample.columns:
        raise typer.BadParameter(
            f"{subsample_manifest} is missing required '{_SUBSAMPLE_SCAN_COLUMN}' "
            "column; regenerate via 08a Phase A."
        )
    scan_paths = subsample[_SUBSAMPLE_SCAN_COLUMN].tolist()

    resolved_output = output_file or Path(
        f"results/tables/2d_vlm_scores_seed_{seed}.csv"
    ).resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if device.type == "cpu":
        logger.warning("CUDA not available; local-VLM inference will be slow on CPU.")
    logger.info(
        "Device=%s dtype=%s models=%s strategies=%s seed=%d output=%s",
        device, dtype, selected_models, selected_strategies, seed, resolved_output,
    )

    done = load_existing(resolved_output)
    pending: list[tuple[str, str, str]] = []
    for scan_path in scan_paths:
        for model_name in selected_models:
            for strategy in selected_strategies:
                if (scan_path, model_name, strategy) in done:
                    continue
                pending.append((scan_path, model_name, strategy))
    total_triples = len(scan_paths) * len(selected_models) * len(selected_strategies)
    logger.info(
        "Plan: %d total triples, %d already done, %d pending",
        total_triples, len(done), len(pending),
    )

    if dry_run:
        for sp, mn, strat in pending[:10]:
            logger.info("  would score: %s × %s × %s", sp, mn, strat)
        if len(pending) > 10:
            logger.info("  ... (%d more)", len(pending) - 10)
        return

    processed = 0
    failed = 0
    budget_hits = 0

    for model_name in selected_models:
        per_model = [
            (sp, strat) for sp, mn, strat in pending if mn == model_name
        ]
        if not per_model:
            logger.info("No pending triples for %s; skipping load.", model_name)
            continue

        adapter_cls = _ADAPTERS[model_name]
        if model_name == "gpt4o":
            adapter = adapter_cls(  # type: ignore[call-arg]
                model_name=gpt_model,
                max_calls=max_api_calls,
                max_budget_usd=max_budget_usd,
                per_call_usd=per_call_usd,
                seed=seed,
            )
        else:
            adapter = adapter_cls()  # type: ignore[call-arg]

        try:
            adapter.load(device=device, dtype=dtype)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to load %s (%s); skipping model.", model_name, exc
            )
            continue

        iterator = (
            tqdm(per_model, desc=f"vlm2d:{model_name}", unit="triple")
            if len(per_model) > 1
            else per_model
        )
        for scan_path, strategy in iterator:
            result = score_one_triple(
                adapter,
                Path(scan_path),
                strategy,
                cache_dir=slice_cache_dir,
                max_new_tokens=max_new_tokens,
            )
            append_row(
                resolved_output,
                scan_path=scan_path,
                model_name=model_name,
                strategy=strategy,
                result=result,
                seed=seed,
            )
            processed += 1
            if result.raw_response in (_BUDGET_SENTINEL, _NO_API_KEY_SENTINEL):
                budget_hits += 1
            if math.isnan(result.score):
                failed += 1
            # GPT-4o: once budget is exhausted, stop scanning further rows
            # for this model (continuing would just append NaN rows).
            if (
                isinstance(adapter, GPT4oAdapter)
                and getattr(adapter, "budget_exhausted", False)
            ):
                logger.info(
                    "GPT-4o budget exhausted after %d calls; stopping model loop.",
                    adapter.call_count,
                )
                break

        adapter.unload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    _print_summary(
        total=total_triples,
        already_done=len(done),
        processed=processed,
        failed=failed,
        budget_hits=budget_hits,
        output_file=resolved_output,
        console=Console(),
    )


if __name__ == "__main__":
    app()
