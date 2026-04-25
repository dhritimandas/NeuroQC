#!/usr/bin/env python3
"""NeuroQC Phase 10 — zero-shot VLM evaluation on ABIDE-I with expert ratings.

Tests RQ3 generalization: do machine-preference-trained VLMs (zero-shot here;
fine-tuned in a future Phase 10b) agree with human raters on an unseen
external dataset?

Inputs (verified at startup; abort with clear error if missing):
    data/abide/abide_ratings_iqms.csv          (1101 rows × 74 cols)
    data/abide/abide_acquisition_manifest.csv  (1100 rows; 1 upstream miss)

For each model in {m3d_lamed, llava_ov, qwen2_vl, medgemma, gpt4o} and seed,
the script writes:
    results/tables/abide_zeroshot_predictions_{model}_seed_{s}.csv
    results/metrics/abide_zeroshot_summary_{model}_seed_{s}.json

THREE CONSENSUS VARIANTS (computed in parallel; all 3 reported per scan):
    A  full-coverage rater_3 only: consensus = (rater_3 == 1).astype(int)
       — coverage 1101 (rater_3 has zero NaNs).
    B  3-rater majority where present, fallback to rater_3 alone otherwise.
       Tied 3-way disagreements resolved by rater_3. Coverage 1101.
    C  STRICT 3-rater majority. Ties (no two raters agree) → EXCLUDED from
       Variant C. Coverage ≈ 600 (final count logged at runtime).

The paper reports primary results on Variant C (closest to MRIQC paper),
with A and B as sensitivity analyses.

PER-MODEL METRICS:
    threshold-free:   AUC + cluster-bootstrap 95% CI (resample sites with
                      replacement, then scans within each chosen site)
    threshold-tuned:  Youden-J in-sample threshold + accuracy/sensitivity/
                      specificity (flagged optimistic upper bound)
    agreement:        Cohen's κ (binary, Variant C); weighted κ (linear,
                      ordinal) per rater; Krippendorff's α (4-way ordinal,
                      Variant C only)
    cross-site:       per-site AUC + bootstrap CI; LOSO mean ± std
    parse health:     parse failure rate (warn if > 10%)
    head-to-head:     mriqc-learn pre-trained classifier AUC + DeLong test
                      vs VLM (when --include-mriqc-baseline; default True)

REUSED FROM 08a/08b (via importlib, since digit-prefixed):
    M3DLamedAdapter + build_vlm_transform     (3D path)
    LlavaOVAdapter + Qwen2VLAdapter +
    MedGemmaAdapter + GPT4oAdapter +
    extract_and_cache_slices                  (2D path)
    QC_PROMPT + parse_qc_response             (from nobrainer.qc.evaluate)

DeLong implementation: inline (~50 LoC, Sun & Xu 2014). Self-contained.
Krippendorff's α: via the `krippendorff` PyPI package (lazy-imported).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from tqdm import tqdm

logger = logging.getLogger(__name__)
app = typer.Typer(
    help="NeuroQC Phase 10 — zero-shot VLM evaluation on ABIDE-I.",
    add_completion=False,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

MODEL_CHOICES: tuple[str, ...] = (
    "m3d_lamed", "llava_ov", "qwen2_vl", "medgemma", "gpt4o",
)
CONSENSUS_VARIANTS: tuple[str, ...] = ("A", "B", "C")
MIN_SITE_N_DEFAULT: int = 30
N_BOOTSTRAP_DEFAULT: int = 1000
BOOTSTRAP_SEED_DEFAULT: int = 42
MAX_BUDGET_USD_DEFAULT: float = 50.0
MAX_API_CALLS_DEFAULT: int = 4000
MAX_NEW_TOKENS_DEFAULT: int = 16
MIN_POS_NEG_PER_SITE: int = 3  # need 3+ positives AND 3+ negatives for AUC

# Likert (1-5) → 3-class rater scale {-1, 0, +1} mapping for weighted κ.
# Documented in summary JSON under `vlm_likert_to_rater_scale_mapping`.
def likert_to_rater_3class(likert: int | float) -> int:
    """Map Likert 1-5 → 3-class {-1 exclude, 0 doubtful, +1 accept}.

    Rule: >=4 → +1; ==3 → 0; <=2 → -1. NaN → NaN (caller must handle).
    """
    if likert is None or (isinstance(likert, float) and math.isnan(likert)):
        return None  # type: ignore[return-value]
    val = int(likert)
    if val >= 4:
        return 1
    if val == 3:
        return 0
    return -1


MRIQC_PAPER_BASELINE_REFERENCE = {
    "loso_accuracy": "76% +/- 13%",
    "post_label_denoising_accuracy": "81%",
    "source": "Esteban et al. 2017, PLOS ONE",
}

# Output schemas.
PREDICTIONS_COLUMNS: tuple[str, ...] = (
    "FILE_ID", "site", "scan_path",
    "rater_1", "rater_2", "rater_3",
    "consensus_a", "consensus_b", "consensus_c",
    "vlm_score", "vlm_likert", "vlm_raw_text", "parse_status",
)


# ──────────────────────────────────────────────
# Module loading: 08a/08b digit-prefixed → importlib
# ──────────────────────────────────────────────


def _load_eval_module(script_name: str, alias: str):
    """Import a digit-prefixed code/<script>.py module by file path."""
    path = _REPO_ROOT / "code" / script_name
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _load_eval_modules() -> tuple[Any, Any]:
    """Load 08a (3D) + 08b (2D) modules; tests may monkeypatch this."""
    mod_3d = _load_eval_module("08a_eval_3d_vlms.py", "_eval3d_for_10")
    mod_2d = _load_eval_module("08b_eval_2d_vlms.py", "_eval2d_for_10")
    return mod_3d, mod_2d


# ──────────────────────────────────────────────
# Data prep + consensus variants
# ──────────────────────────────────────────────


def load_inputs(ratings_csv: Path, acquisition_manifest: Path) -> pd.DataFrame:
    """Load ratings + acquisition manifest; inner-join on FILE_ID.

    Logs |ratings|, |manifest|, |joined|; flags rated-but-unacquired and
    acquired-but-unrated counts.
    """
    if not ratings_csv.is_file():
        raise typer.BadParameter(
            f"ratings CSV not found at {ratings_csv} — run code/09b_acquire_abide.py first."
        )
    if not acquisition_manifest.is_file():
        raise typer.BadParameter(
            f"acquisition manifest not found at {acquisition_manifest} — "
            "run code/09b_acquire_abide.py with --acquisition-path fcp-indi-raw first."
        )
    ratings = pd.read_csv(ratings_csv)
    manifest = pd.read_csv(acquisition_manifest)
    joined = ratings.merge(manifest, on="FILE_ID", how="inner", suffixes=("", "_man"))
    rated_only = set(ratings["FILE_ID"]) - set(manifest["FILE_ID"])
    acquired_only = set(manifest["FILE_ID"]) - set(ratings["FILE_ID"])
    logger.info(
        "Inputs: |ratings|=%d, |manifest|=%d, |joined|=%d. "
        "Rated-but-unacquired: %d. Acquired-but-unrated: %d.",
        len(ratings), len(manifest), len(joined),
        len(rated_only), len(acquired_only),
    )
    if rated_only:
        logger.info("First 3 rated-but-unacquired: %s", sorted(rated_only)[:3])
    return joined


def build_consensus_a(df: pd.DataFrame) -> pd.Series:
    """Variant A: rater_3 == +1 → 1; else 0. Coverage 1101 (rater_3 fully covered)."""
    return (df["rater_3"] == 1).astype(int)


def _three_rater_majority(r1: float, r2: float, r3: float) -> int | float:
    """Return majority vote among r1/r2/r3 ∈ {-1, 0, +1}; NaN if 3-way tie."""
    vals = [v for v in (r1, r2, r3) if not pd.isna(v)]
    if len(vals) < 3:
        return float("nan")  # caller decides fallback
    counts: dict[int, int] = {}
    for v in vals:
        counts[int(v)] = counts.get(int(v), 0) + 1
    max_count = max(counts.values())
    winners = [k for k, c in counts.items() if c == max_count]
    if len(winners) == 1:
        return winners[0]
    return float("nan")  # tie


def build_consensus_b(df: pd.DataFrame) -> pd.Series:
    """Variant B: 3-rater majority where all present; else fallback to rater_3.

    Tied 3-way disagreements (no 2 raters agree) → fallback to rater_3 too.
    Then binarize +1 vs {-1, 0}. Coverage 1101.
    """
    out: list[int] = []
    for _, row in df.iterrows():
        r1, r2, r3 = row["rater_1"], row["rater_2"], row["rater_3"]
        if pd.notna(r1) and pd.notna(r2) and pd.notna(r3):
            mv = _three_rater_majority(r1, r2, r3)
            if pd.isna(mv):
                mv = r3  # tie → rater_3 (most-trusted single rater)
        else:
            mv = r3  # rater_3 only
        out.append(1 if int(mv) == 1 else 0)
    return pd.Series(out, index=df.index, dtype=int)


def build_consensus_c(df: pd.DataFrame) -> pd.Series:
    """Variant C: strict 3-rater majority. Ties → NaN (excluded). Coverage ≈ 600.

    Returns 1 if accept (+1), 0 if non-accept ({-1, 0}), NaN if rater coverage
    is incomplete OR if all 3 raters disagree.
    """
    out: list[int | float] = []
    for _, row in df.iterrows():
        r1, r2, r3 = row["rater_1"], row["rater_2"], row["rater_3"]
        if pd.notna(r1) and pd.notna(r2) and pd.notna(r3):
            mv = _three_rater_majority(r1, r2, r3)
            if pd.isna(mv):
                out.append(float("nan"))
            else:
                out.append(1 if int(mv) == 1 else 0)
        else:
            out.append(float("nan"))
    return pd.Series(out, index=df.index, dtype=float)


def attach_consensus_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add consensus_a/b/c columns to a copy of df."""
    out = df.copy()
    out["consensus_a"] = build_consensus_a(out)
    out["consensus_b"] = build_consensus_b(out)
    out["consensus_c"] = build_consensus_c(out)
    return out


# ──────────────────────────────────────────────
# Adapter dispatch
# ──────────────────────────────────────────────


def load_adapter_for_model(
    model_name: str,
    *,
    max_budget_usd: float,
    max_api_calls: int,
    gpt_model: str,
    slice_cache_dir: Path | None,
) -> tuple[Any, Callable[[Path], Any], str]:
    """Return (adapter_instance, prepare_input_fn, modality).

    Modality ∈ {"3d", "2d"}. The prepare_input_fn produces the per-scan input
    (3D: torch.Tensor; 2D: list[PIL.Image]) ready for adapter.run_inference.
    """
    mod_3d, mod_2d = _load_eval_modules()
    if model_name == "m3d_lamed":
        adapter = mod_3d.M3DLamedAdapter()
        transform = mod_3d.build_vlm_transform()

        def prepare_3d(scan_path: Path) -> torch.Tensor:
            vol = transform(str(scan_path))
            return vol.permute(0, 3, 1, 2).contiguous()  # (1, 32, 256, 256)

        return adapter, prepare_3d, "3d"

    if model_name in ("llava_ov", "qwen2_vl", "medgemma", "gpt4o"):
        if model_name == "gpt4o":
            adapter = mod_2d.GPT4oAdapter(
                max_calls=max_api_calls,
                max_budget_usd=max_budget_usd,
                model=gpt_model,
            )
        else:
            cls_map = {
                "llava_ov": mod_2d.LlavaOVAdapter,
                "qwen2_vl": mod_2d.Qwen2VLAdapter,
                "medgemma": mod_2d.MedGemmaAdapter,
            }
            adapter = cls_map[model_name]()

        cache_dir = slice_cache_dir or (_REPO_ROOT / "data" / "derivatives" / "slices_abide")

        def prepare_2d(scan_path: Path) -> list:
            return mod_2d.extract_and_cache_slices(scan_path, "mid", cache_dir)

        return adapter, prepare_2d, "2d"

    raise typer.BadParameter(f"Unknown model {model_name!r}; choose from {MODEL_CHOICES}")


# ──────────────────────────────────────────────
# Inference loop
# ──────────────────────────────────────────────


def load_existing_predictions(predictions_csv: Path) -> set[tuple[str, str]]:
    """Return {(FILE_ID, model)} pairs already recorded in the predictions CSV."""
    if not predictions_csv.is_file() or predictions_csv.stat().st_size == 0:
        return set()
    with predictions_csv.open("r", newline="") as h:
        reader = csv.DictReader(h)
        if reader.fieldnames is None or "FILE_ID" not in reader.fieldnames:
            return set()
        # All rows in this file are for one model; we still encode the pair for
        # parity with 08a/08b's resume idiom.
        return {(row["FILE_ID"], row.get("model", "_unknown")) for row in reader}


def append_prediction_row(
    predictions_csv: Path,
    row: dict[str, Any],
) -> None:
    """Append one prediction row (crash-safe)."""
    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    is_new = not predictions_csv.is_file() or predictions_csv.stat().st_size == 0
    with predictions_csv.open("a", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(PREDICTIONS_COLUMNS))
        if is_new:
            w.writeheader()
        # Project the row to only the schema columns.
        w.writerow({col: row.get(col) for col in PREDICTIONS_COLUMNS})
        h.flush()


def score_one_scan(
    adapter: Any,
    prepare_fn: Callable[[Path], Any],
    modality: str,
    scan_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int = MAX_NEW_TOKENS_DEFAULT,
) -> tuple[float, float, str, str]:
    """Run inference; return (vlm_score [0,1] or NaN, vlm_likert int or NaN, raw, status).

    Status ∈ {"ok", "parse_failed", "oom", "error"}.
    """
    from nobrainer.qc.evaluate import parse_qc_response

    try:
        x = prepare_fn(scan_path)
        if modality == "3d" and isinstance(x, torch.Tensor):
            x = x.to(device=device, dtype=dtype)
        raw = adapter.run_inference(x, max_new_tokens=max_new_tokens)
        parsed = parse_qc_response(raw)
        likert = parsed.get("score")
        if likert is None:
            return float("nan"), float("nan"), str(raw), "parse_failed"
        likert_int = int(likert)
        if not 1 <= likert_int <= 5:
            return float("nan"), float("nan"), str(raw), "parse_failed"
        score = (likert_int - 1) / 4.0
        return score, float(likert_int), str(raw), "ok"
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return float("nan"), float("nan"), "OOM", "oom"
    except Exception as exc:  # noqa: BLE001
        return float("nan"), float("nan"), f"ERROR: {exc}", "error"


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────


def cluster_bootstrap_auc(
    scores: np.ndarray,
    labels: np.ndarray,
    sites: np.ndarray,
    n_boot: int = N_BOOTSTRAP_DEFAULT,
    seed: int = BOOTSTRAP_SEED_DEFAULT,
) -> tuple[float, float, float]:
    """AUC + cluster-bootstrap 95% CI (resample sites, then scans within site).

    Returns (point_estimate, ci_lower, ci_upper). NaN-tuple on degenerate input.
    """
    from sklearn.metrics import roc_auc_score

    mask = np.isfinite(scores) & np.isfinite(labels)
    s, y, st = scores[mask], labels[mask], sites[mask]
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    point = float(roc_auc_score(y, s))

    rng = np.random.default_rng(seed)
    site_ids = np.unique(st)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        chosen_sites = rng.choice(site_ids, size=len(site_ids), replace=True)
        idx_list: list[int] = []
        for site_choice in chosen_sites:
            site_idx = np.where(st == site_choice)[0]
            if len(site_idx) == 0:
                continue
            picks = rng.choice(site_idx, size=len(site_idx), replace=True)
            idx_list.extend(picks.tolist())
        idx = np.array(idx_list)
        try:
            if len(np.unique(y[idx])) < 2:
                boots[i] = np.nan
            else:
                boots[i] = roc_auc_score(y[idx], s[idx])
        except ValueError:
            boots[i] = np.nan
    valid = boots[np.isfinite(boots)]
    if valid.size == 0:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))


def youden_threshold_metrics(
    scores: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Find Youden-J optimal threshold; return threshold + accuracy/sens/spec."""
    from sklearn.metrics import roc_curve

    mask = np.isfinite(scores) & np.isfinite(labels)
    s, y = scores[mask], labels[mask]
    if len(np.unique(y)) < 2 or len(s) < 4:
        return {
            "threshold": float("nan"),
            "accuracy": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
        }
    fpr, tpr, thresholds = roc_curve(y, s)
    j = tpr - fpr
    best = int(np.argmax(j))
    threshold = float(thresholds[best])
    sens = float(tpr[best])
    spec = float(1.0 - fpr[best])
    preds = (s >= threshold).astype(int)
    acc = float((preds == y).mean())
    return {
        "threshold": threshold,
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
    }


def cohen_kappa_binary(vlm_pred: np.ndarray, consensus: np.ndarray) -> float:
    """Cohen's κ between two binary classifications. NaN on undefined."""
    from sklearn.metrics import cohen_kappa_score

    mask = np.isfinite(vlm_pred) & np.isfinite(consensus)
    if mask.sum() < 4:
        return float("nan")
    return float(cohen_kappa_score(vlm_pred[mask].astype(int), consensus[mask].astype(int)))


def cohen_kappa_weighted_per_rater(
    vlm_3class: np.ndarray,
    raters: dict[str, np.ndarray],
) -> dict[str, float]:
    """Linear-weighted κ between VLM 3-class labels and each rater's 3-class labels.

    Each rater key uses its own marginal coverage (NaN-rows excluded).
    """
    from sklearn.metrics import cohen_kappa_score

    out: dict[str, float] = {}
    for rater_name, rater_arr in raters.items():
        mask = np.isfinite(vlm_3class) & np.isfinite(rater_arr)
        if mask.sum() < 4:
            out[rater_name] = float("nan")
            continue
        try:
            out[rater_name] = float(
                cohen_kappa_score(
                    vlm_3class[mask].astype(int),
                    rater_arr[mask].astype(int),
                    weights="linear",
                    labels=[-1, 0, 1],
                )
            )
        except Exception:  # noqa: BLE001
            out[rater_name] = float("nan")
    return out


def krippendorff_alpha_4way(
    rater_1: np.ndarray,
    rater_2: np.ndarray,
    rater_3: np.ndarray,
    vlm_3class: np.ndarray,
) -> float:
    """4-way ordinal Krippendorff's α. Lazy-import krippendorff."""
    try:
        import krippendorff
    except ImportError:
        return float("nan")
    data = np.vstack([
        rater_1.astype(float),
        rater_2.astype(float),
        rater_3.astype(float),
        vlm_3class.astype(float),
    ])
    # krippendorff treats nan as missing.
    try:
        return float(
            krippendorff.alpha(reliability_data=data, level_of_measurement="ordinal")
        )
    except Exception:  # noqa: BLE001
        return float("nan")


def per_site_auc(
    scores: np.ndarray,
    labels: np.ndarray,
    sites: np.ndarray,
    min_n: int = MIN_SITE_N_DEFAULT,
    n_boot: int = N_BOOTSTRAP_DEFAULT,
    seed: int = BOOTSTRAP_SEED_DEFAULT,
) -> dict[str, dict[str, float]]:
    """Per-site AUC + bootstrap CI. Skip sites with N<min_n or class imbalance.

    Returns {site: {auc, ci_lower, ci_upper, n}} — sites that fail filters omitted.
    """
    from sklearn.metrics import roc_auc_score

    out: dict[str, dict[str, float]] = {}
    for site in np.unique(sites):
        mask = sites == site
        s, y = scores[mask], labels[mask]
        # Drop NaN within site.
        finite = np.isfinite(s) & np.isfinite(y)
        s, y = s[finite], y[finite]
        if len(s) < min_n:
            continue
        if y.sum() < MIN_POS_NEG_PER_SITE or (1 - y).sum() < MIN_POS_NEG_PER_SITE:
            continue
        try:
            point = float(roc_auc_score(y, s))
        except ValueError:
            continue
        # Within-site bootstrap CI.
        rng = np.random.default_rng(seed)
        boots = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx = rng.choice(len(s), size=len(s), replace=True)
            try:
                if len(np.unique(y[idx])) < 2:
                    boots[i] = np.nan
                else:
                    boots[i] = roc_auc_score(y[idx], s[idx])
            except ValueError:
                boots[i] = np.nan
        valid = boots[np.isfinite(boots)]
        out[str(site)] = {
            "auc": point,
            "ci_lower": float(np.percentile(valid, 2.5)) if valid.size else float("nan"),
            "ci_upper": float(np.percentile(valid, 97.5)) if valid.size else float("nan"),
            "n": int(len(s)),
        }
    return out


def loso_aucs(
    scores: np.ndarray,
    labels: np.ndarray,
    sites: np.ndarray,
    min_n: int = MIN_SITE_N_DEFAULT,
) -> tuple[dict[str, float], float, float]:
    """Per-site held-out AUC + LOSO mean/SD across folds.

    Returns (per_site_dict, mean_loso, std_loso). Sites with N<min_n or
    insufficient class balance are skipped (logged). LOSO for zero-shot is
    "report per-site performance and aggregate" — there's no training, so
    "leave-out" just means measuring per-site to quantify cross-site spread.
    """
    from sklearn.metrics import roc_auc_score

    folds: dict[str, float] = {}
    for site in np.unique(sites):
        mask = sites == site
        s, y = scores[mask], labels[mask]
        finite = np.isfinite(s) & np.isfinite(y)
        s, y = s[finite], y[finite]
        if len(s) < min_n:
            continue
        if y.sum() < MIN_POS_NEG_PER_SITE or (1 - y).sum() < MIN_POS_NEG_PER_SITE:
            continue
        try:
            folds[str(site)] = float(roc_auc_score(y, s))
        except ValueError:
            continue
    vals = list(folds.values())
    if not vals:
        return folds, float("nan"), float("nan")
    return folds, float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def _placement_values(a_pos: np.ndarray, a_neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """V10[i] = E_j ψ(a_pos[i], a_neg[j]); V01[j] = E_i ψ(a_pos[i], a_neg[j])."""
    diff = a_pos[:, None] - a_neg[None, :]  # (m, n)
    psi = (diff > 0).astype(float) + 0.5 * (diff == 0).astype(float)
    return psi.mean(axis=1), psi.mean(axis=0)


def delong_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Paired DeLong test for AUC(a) − AUC(b) on identical labels.

    Sun & Xu 2014 / DeLong 1988 algorithm. Returns:
        {"auc_a", "auc_b", "delta_auc", "z", "p"}
    NaN dict on degenerate input.
    """
    from scipy.stats import norm

    mask = np.isfinite(scores_a) & np.isfinite(scores_b) & np.isfinite(labels)
    a, b, y = scores_a[mask], scores_b[mask], labels[mask]
    pos = y == 1
    neg = ~pos
    a_pos, a_neg = a[pos], a[neg]
    b_pos, b_neg = b[pos], b[neg]
    m, n = len(a_pos), len(a_neg)
    if m < 2 or n < 2:
        return {
            "auc_a": float("nan"), "auc_b": float("nan"),
            "delta_auc": float("nan"), "z": float("nan"), "p": float("nan"),
        }

    diff_a = a_pos[:, None] - a_neg[None, :]
    diff_b = b_pos[:, None] - b_neg[None, :]
    auc_a = float(((diff_a > 0).astype(float) + 0.5 * (diff_a == 0).astype(float)).mean())
    auc_b = float(((diff_b > 0).astype(float) + 0.5 * (diff_b == 0).astype(float)).mean())

    V10_a, V01_a = _placement_values(a_pos, a_neg)
    V10_b, V01_b = _placement_values(b_pos, b_neg)

    S10 = np.cov(np.vstack([V10_a, V10_b]), ddof=1) if m > 1 else np.zeros((2, 2))
    S01 = np.cov(np.vstack([V01_a, V01_b]), ddof=1) if n > 1 else np.zeros((2, 2))
    if S10.ndim == 0:
        S10 = np.array([[S10, 0], [0, 0]])
    if S01.ndim == 0:
        S01 = np.array([[S01, 0], [0, 0]])

    S = S10 / m + S01 / n
    L = np.array([1.0, -1.0])
    diff_var = float(L @ S @ L)
    delta = auc_a - auc_b
    if diff_var <= 0:
        # Identical predictions or singular variance.
        z = 0.0
        p = 1.0 if abs(delta) < 1e-12 else float("nan")
    else:
        z = float(delta / np.sqrt(diff_var))
        p = float(2.0 * (1.0 - norm.cdf(abs(z))))
    return {
        "auc_a": auc_a, "auc_b": auc_b,
        "delta_auc": float(delta),
        "z": z, "p": p,
    }


# ──────────────────────────────────────────────
# MRIQC classifier baseline
# ──────────────────────────────────────────────


def mriqc_classifier_proba(
    df: pd.DataFrame,
    ratings_iqm_csv: Path,
) -> tuple[np.ndarray, list[str]] | None:
    """Score mriqc-learn's pre-trained classifier on the same scans + IQMs.

    Returns (proba_array, file_ids) where proba_array[i] aligns with file_ids[i].
    Returns None if mriqc-learn isn't installed or the classifier API differs.
    """
    try:
        from mriqc_learn.models.production import load_model
    except Exception as exc:  # noqa: BLE001
        logger.warning("mriqc-learn classifier unavailable: %s", exc)
        return None

    iqm_df = pd.read_csv(ratings_iqm_csv)
    feature_cols = [c for c in iqm_df.columns if c.startswith("mriqc_")]
    if not feature_cols:
        logger.warning("No mriqc_-prefixed columns in %s", ratings_iqm_csv)
        return None

    sub = iqm_df[iqm_df["FILE_ID"].isin(df["FILE_ID"])].copy()
    X = sub[feature_cols].copy()
    X.columns = [c.removeprefix("mriqc_") for c in X.columns]
    try:
        clf = load_model()
        proba = clf.predict_proba(X)[:, 1]
    except Exception as exc:  # noqa: BLE001
        logger.warning("mriqc classifier predict_proba failed: %s", exc)
        return None
    return proba, sub["FILE_ID"].astype(str).tolist()


# ──────────────────────────────────────────────
# Per-variant metric assembly
# ──────────────────────────────────────────────


def compute_variant_metrics(
    predictions_df: pd.DataFrame,
    variant: str,
    n_boot: int,
    seed: int,
    min_site_n: int,
) -> dict[str, Any]:
    """Compute the full metric block for one consensus variant.

    Variant ∈ {"A", "B", "C"}; the consensus column is consensus_a/b/c.
    """
    consensus_col = f"consensus_{variant.lower()}"
    df = predictions_df.dropna(subset=[consensus_col]).copy()
    scores = df["vlm_score"].to_numpy(dtype=float)
    labels = df[consensus_col].to_numpy(dtype=float)
    sites = df["site"].astype(str).to_numpy()

    n_total = len(df)
    n_pos = int(np.nansum(labels == 1))
    n_neg = int(np.nansum(labels == 0))

    auc_pt, auc_lo, auc_hi = cluster_bootstrap_auc(scores, labels, sites, n_boot, seed)
    youden = youden_threshold_metrics(scores, labels)
    if not math.isnan(youden["threshold"]):
        vlm_pred_binary = (scores >= youden["threshold"]).astype(float)
        vlm_pred_binary[~np.isfinite(scores)] = float("nan")
        kappa = cohen_kappa_binary(vlm_pred_binary, labels)
    else:
        kappa = float("nan")

    loso_per_site, loso_mean, loso_std = loso_aucs(scores, labels, sites, min_site_n)

    return {
        "n": n_total,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "auc": {"point": auc_pt, "ci_lower": auc_lo, "ci_upper": auc_hi},
        "youden_threshold": youden["threshold"],
        "youden_threshold_is_in_sample": True,
        "accuracy": {"point": youden["accuracy"]},
        "sensitivity": {"point": youden["sensitivity"]},
        "specificity": {"point": youden["specificity"]},
        "cohens_kappa": {"point": kappa},
        "loso_aucs_per_site": loso_per_site,
        "loso_auc_mean": loso_mean,
        "loso_auc_std": loso_std,
    }


# ──────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────


def write_summary_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomic JSON write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────


def _print_summary(model: str, summary: dict[str, Any], console: Console) -> None:
    table = Table(title=f"Phase 10 — zero-shot ABIDE eval ({model})")
    table.add_column("variant", style="bold")
    table.add_column("N", justify="right")
    table.add_column("AUC [95% CI]", justify="right")
    table.add_column("κ (binary)", justify="right")
    table.add_column("LOSO mean ± SD", justify="right")
    for v in CONSENSUS_VARIANTS:
        block = summary["variants"].get(v, {})
        if not block:
            continue
        auc = block["auc"]
        loso = f"{block.get('loso_auc_mean', float('nan')):.3f} ± {block.get('loso_auc_std', float('nan')):.3f}"
        table.add_row(
            v, str(block["n"]),
            f"{auc['point']:.3f} [{auc['ci_lower']:.3f}, {auc['ci_upper']:.3f}]",
            f"{block['cohens_kappa']['point']:.3f}",
            loso,
        )
    console.print(table)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


@app.command()
def main(
    models: str = typer.Option(",".join(MODEL_CHOICES), "--models"),
    seed: int = typer.Option(..., "--seed"),
    ratings_csv: Path = typer.Option(
        Path("data/abide/abide_ratings_iqms.csv"), "--ratings-csv", resolve_path=True
    ),
    acquisition_manifest: Path = typer.Option(
        Path("data/abide/abide_acquisition_manifest.csv"),
        "--acquisition-manifest", resolve_path=True,
    ),
    output_tables_dir: Path = typer.Option(
        Path("results/tables"), "--output-tables-dir", resolve_path=True
    ),
    output_metrics_dir: Path = typer.Option(
        Path("results/metrics"), "--output-metrics-dir", resolve_path=True
    ),
    include_mriqc_baseline: bool = typer.Option(
        True, "--include-mriqc-baseline/--no-include-mriqc-baseline"
    ),
    n_bootstrap: int = typer.Option(N_BOOTSTRAP_DEFAULT, "--n-bootstrap"),
    bootstrap_seed: int = typer.Option(BOOTSTRAP_SEED_DEFAULT, "--bootstrap-seed"),
    min_site_n: int = typer.Option(MIN_SITE_N_DEFAULT, "--min-site-n"),
    max_budget_usd: float = typer.Option(MAX_BUDGET_USD_DEFAULT, "--max-budget-usd"),
    max_api_calls: int = typer.Option(MAX_API_CALLS_DEFAULT, "--max-api-calls"),
    gpt_model: str = typer.Option("gpt-4o-2024-11-20", "--gpt-model"),
    max_scans: int = typer.Option(0, "--max-scans"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Zero-shot VLM evaluation on ABIDE-I against 3 rater consensus variants."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )
    console = Console()

    # ── Inputs ──
    df = load_inputs(ratings_csv, acquisition_manifest)
    df = attach_consensus_columns(df)
    n_a = int(df["consensus_a"].notna().sum())
    n_b = int(df["consensus_b"].notna().sum())
    n_c = int(df["consensus_c"].notna().sum())
    logger.info(
        "Variant coverage: A=%d, B=%d, C=%d (Variant C excludes 3-way ties)",
        n_a, n_b, n_c,
    )

    if max_scans > 0:
        df = df.head(max_scans)
        logger.info("--max-scans applied: %d scans", max_scans)

    if dry_run:
        logger.info("Dry run — exiting before adapter load / inference.")
        return

    # ── Per-model evaluation ──
    selected_models = [m.strip() for m in models.split(",") if m.strip()]
    invalid = [m for m in selected_models if m not in MODEL_CHOICES]
    if invalid:
        raise typer.BadParameter(
            f"Unknown model(s) {invalid}; choose from {MODEL_CHOICES}"
        )

    output_tables_dir.mkdir(parents=True, exist_ok=True)
    output_metrics_dir.mkdir(parents=True, exist_ok=True)

    # Optional mriqc baseline scoring (independent of any one VLM).
    mriqc_proba: np.ndarray | None = None
    mriqc_file_ids: list[str] | None = None
    if include_mriqc_baseline:
        result = mriqc_classifier_proba(df, ratings_csv)
        if result is not None:
            mriqc_proba, mriqc_file_ids = result
            logger.info("mriqc baseline classifier scored on %d scans", len(mriqc_proba))
        else:
            logger.warning("mriqc baseline unavailable; skipping V4 head-to-head")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if device.type == "cpu":
        logger.warning("CUDA unavailable — only gpt4o is practical on CPU.")

    for model_name in selected_models:
        logger.info("=" * 70)
        logger.info("Model: %s", model_name)
        logger.info("=" * 70)

        predictions_csv = (
            output_tables_dir / f"abide_zeroshot_predictions_{model_name}_seed_{seed}.csv"
        )
        summary_json = (
            output_metrics_dir / f"abide_zeroshot_summary_{model_name}_seed_{seed}.json"
        )

        try:
            adapter, prepare_fn, modality = load_adapter_for_model(
                model_name,
                max_budget_usd=max_budget_usd,
                max_api_calls=max_api_calls,
                gpt_model=gpt_model,
                slice_cache_dir=None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to instantiate %s: %s", model_name, exc)
            continue

        try:
            adapter.load(device=device, dtype=dtype)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load %s weights: %s", model_name, exc)
            continue

        done = load_existing_predictions(predictions_csv)
        n_pending = sum(1 for fid in df["FILE_ID"] if (fid, model_name) not in done)
        logger.info(
            "Predictions plan: %d total, %d already done, %d pending",
            len(df), len(done), n_pending,
        )

        for _, row in tqdm(df.iterrows(), total=len(df), desc=model_name):
            fid = str(row["FILE_ID"])
            if (fid, model_name) in done:
                continue
            score, likert, raw, status = score_one_scan(
                adapter, prepare_fn, modality, Path(row["scan_path"]), device, dtype,
            )
            append_prediction_row(
                predictions_csv,
                {
                    "FILE_ID": fid,
                    "site": row["site"],
                    "scan_path": row["scan_path"],
                    "rater_1": row.get("rater_1"),
                    "rater_2": row.get("rater_2"),
                    "rater_3": row.get("rater_3"),
                    "consensus_a": row["consensus_a"],
                    "consensus_b": row["consensus_b"],
                    "consensus_c": row["consensus_c"],
                    "vlm_score": score,
                    "vlm_likert": likert,
                    "vlm_raw_text": raw,
                    "parse_status": status,
                },
            )

        adapter.unload()

        # ── Compute per-variant metrics ──
        preds = pd.read_csv(predictions_csv)
        # Re-attach consensus columns from the full df (in case the predictions CSV
        # came from a partial run that used a different subsample).
        preds = preds.merge(
            df[["FILE_ID", "site"]].rename(columns={"site": "_site_check"}),
            on="FILE_ID", how="inner", suffixes=("", "_dup"),
        )
        if "site" not in preds.columns:
            preds["site"] = preds["_site_check"]
        preds = preds.drop(columns=[c for c in preds.columns if c.endswith("_dup") or c == "_site_check"])

        n_parse_failed = int((preds["parse_status"] != "ok").sum())
        parse_failure_rate = n_parse_failed / max(1, len(preds))
        if parse_failure_rate > 0.10:
            logger.warning(
                "Parse failure rate for %s: %.1f%% (>10%%; model is not reliably "
                "emitting parseable scores)", model_name, parse_failure_rate * 100,
            )

        variants_block: dict[str, Any] = {}
        for v in CONSENSUS_VARIANTS:
            variants_block[v] = compute_variant_metrics(
                preds, v, n_bootstrap, bootstrap_seed, min_site_n,
            )

        # ── Per-rater weighted κ + Krippendorff α (Variant C only) ──
        likert_arr = preds["vlm_likert"].to_numpy(dtype=float)
        vlm_3class = np.array(
            [likert_to_rater_3class(v) if not pd.isna(v) else float("nan") for v in likert_arr],
            dtype=float,
        )
        weighted_kappa = cohen_kappa_weighted_per_rater(
            vlm_3class,
            {
                "rater_1": preds["rater_1"].to_numpy(dtype=float),
                "rater_2": preds["rater_2"].to_numpy(dtype=float),
                "rater_3": preds["rater_3"].to_numpy(dtype=float),
            },
        )
        # Krippendorff α restricted to Variant C scans (full 3-rater coverage AND no tie).
        c_mask = preds["consensus_c"].notna().to_numpy()
        alpha_c = krippendorff_alpha_4way(
            preds.loc[c_mask, "rater_1"].to_numpy(dtype=float),
            preds.loc[c_mask, "rater_2"].to_numpy(dtype=float),
            preds.loc[c_mask, "rater_3"].to_numpy(dtype=float),
            vlm_3class[c_mask],
        )

        # ── Per-site (Variant C as primary) ──
        c_df = preds.dropna(subset=["consensus_c"])
        per_site_block = per_site_auc(
            c_df["vlm_score"].to_numpy(dtype=float),
            c_df["consensus_c"].to_numpy(dtype=float),
            c_df["site"].astype(str).to_numpy(),
            min_site_n, n_bootstrap, bootstrap_seed,
        )

        # ── MRIQC head-to-head (Variant C) ──
        mriqc_block: dict[str, Any] | None = None
        if mriqc_proba is not None and mriqc_file_ids is not None:
            mriqc_df = pd.DataFrame({"FILE_ID": mriqc_file_ids, "mriqc_proba": mriqc_proba})
            joined = preds.merge(mriqc_df, on="FILE_ID", how="inner")
            joined = joined.dropna(subset=["consensus_c", "vlm_score"])
            if len(joined) > 4:
                vlm_arr = joined["vlm_score"].to_numpy(dtype=float)
                mriqc_arr = joined["mriqc_proba"].to_numpy(dtype=float)
                labels = joined["consensus_c"].to_numpy(dtype=float)
                mriqc_block = delong_test(vlm_arr, mriqc_arr, labels)

        # ── Assemble + write summary JSON ──
        summary = {
            "model": model_name,
            "seed": seed,
            "scans_evaluated": len(preds),
            "parse_failure_rate": parse_failure_rate,
            "vlm_likert_to_rater_scale_mapping": {
                ">=4": 1, "==3": 0, "<=2": -1,
            },
            "variants": variants_block,
            "weighted_kappa_per_rater": weighted_kappa,
            "krippendorff_alpha_4way_variant_c": alpha_c,
            "per_site_variant_c": per_site_block,
            "mriqc_classifier_comparison": mriqc_block,
            "mriqc_paper_baseline_reference": MRIQC_PAPER_BASELINE_REFERENCE,
        }
        write_summary_json(summary_json, summary)
        logger.info("Summary → %s", summary_json)
        _print_summary(model_name, summary, console)


if __name__ == "__main__":
    app()
