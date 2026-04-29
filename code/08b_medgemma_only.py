#!/usr/bin/env python3
"""NeuroQC Phase 08b standalone — MedGemma only, transformers ≥4.50 venv.

MedGemma uses Gemma3ForConditionalGeneration which only exists in
transformers ≥4.50. M3D-LaMed (Phase 08a) requires transformers <4.50,
so the two cannot share a venv. This script is meant to run inside a
separate venv (e.g. /workspace/medgemma_venv) that has transformers
≥4.50 installed.

Reads ``benchmark_subsample.csv`` and writes scores in the SAME schema as
``code/08b_eval_2d_vlms.py`` so the output can be merged with the main
2D-VLM CSV via concat. Slice extraction is inlined here (mid-slice per
orientation) to avoid the ``nobrainer`` import.

Usage:
    /workspace/medgemma_venv/bin/python code/08b_medgemma_only.py \\
        --subsample-manifest results/tables/benchmark_subsample.csv \\
        --output-file results/tables/2d_vlm_scores_medgemma_seed_0.csv \\
        --seed 0
"""

from __future__ import annotations

import csv
import gc
import logging
import math
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import typer
from PIL import Image
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm

QC_PROMPT: str = (
    "Assess the quality of this brain MRI scan. Rate it from 1 (unusable, "
    "severe artifacts making it unsuitable for analysis) to 5 (excellent "
    "quality, no visible artifacts). Describe any quality issues you "
    "observe including: motion artifacts, noise, blurring, intensity "
    "inhomogeneity, ghosting, or resolution problems.\n"
    "Output your response in this exact format:\n"
    "SCORE: [integer 1-5]\n"
    "REASON: [one paragraph description]"
)

OUTPUT_COLUMNS = (
    "scan_path",
    "model",
    "slice_strategy",
    "score",
    "raw_response",
    "seed",
    "n_slices",
    "multi_image_mode",
)

_SCORE_RE = re.compile(r"SCORE:\s*([1-5])\b")
logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)


def load_volume_uint8(path: Path) -> np.ndarray:
    """Load NIfTI as uint8 (H, W, D) — 0-255 normalized."""
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj).astype(np.float32)
    arr = arr.squeeze()
    while arr.ndim > 3:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {arr.shape}")
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = (arr - lo) / (hi - lo) * 255.0
    return arr.astype(np.uint8)


def mid_slices(volume: np.ndarray) -> list[Image.Image]:
    """Return [axial, coronal, sagittal] mid-slices as RGB PIL images."""
    h, w, d = volume.shape
    axial = volume[:, :, d // 2]
    coronal = volume[:, w // 2, :]
    sagittal = volume[h // 2, :, :]
    return [Image.fromarray(s).convert("RGB") for s in (axial, coronal, sagittal)]


def concat_horizontal(images: list[Image.Image]) -> Image.Image:
    """Concat 3 PIL images horizontally; pad to common height."""
    h_max = max(im.height for im in images)
    rescaled = [
        im.resize((int(im.width * h_max / im.height), h_max)) if im.height != h_max else im
        for im in images
    ]
    w_total = sum(im.width for im in rescaled)
    canvas = Image.new("RGB", (w_total, h_max))
    x = 0
    for im in rescaled:
        canvas.paste(im, (x, 0))
        x += im.width
    return canvas


def parse_score(raw: str) -> float:
    """Parse SCORE: N (1-5) → [0,1]; NaN if no match."""
    m = _SCORE_RE.search(raw)
    if not m:
        return math.nan
    n = int(m.group(1))
    return (n - 1) / 4.0


def load_done(output_file: Path) -> set[tuple[str, str, str]]:
    """Resume support: existing (scan_path, model, slice_strategy) triples."""
    if not output_file.exists() or output_file.stat().st_size == 0:
        return set()
    out: set[tuple[str, str, str]] = set()
    with output_file.open("r", newline="") as h:
        reader = csv.DictReader(h)
        for row in reader:
            sp = row.get("scan_path", "")
            mdl = row.get("model", "")
            strat = row.get("slice_strategy", "")
            if sp and mdl and strat:
                out.add((sp, mdl, strat))
    return out


def append_row(output_file: Path, row: dict, header: list[str]) -> None:
    is_new = not output_file.exists() or output_file.stat().st_size == 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", newline="") as h:
        w = csv.DictWriter(h, fieldnames=header)
        if is_new:
            w.writeheader()
        w.writerow(row)
        h.flush()


@app.command()
def main(
    subsample_manifest: Path = typer.Option(..., "--subsample-manifest", exists=True),
    output_file: Path = typer.Option(..., "--output-file"),
    seed: int = typer.Option(0, "--seed"),
    model_id: str = typer.Option("google/medgemma-4b-it", "--model-id"),
    max_new_tokens: int = typer.Option(48, "--max-new-tokens"),
    strategies: str = typer.Option("mid", "--strategies", help="Comma-separated"),
) -> None:
    """Run MedGemma on the benchmark subsample's scans."""
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    strats = [s.strip() for s in strategies.split(",") if s.strip()]
    if any(s != "mid" for s in strats):
        raise typer.BadParameter("Only 'mid' strategy supported in this minimal script")

    # Load model.
    logger.info("Loading %s", model_id)
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device,
    )
    model.eval()
    logger.info("Loaded MedGemma on %s (%s)", device, dtype)

    # Read subsample.
    with subsample_manifest.open("r") as h:
        scans = [row["scan_path"] for row in csv.DictReader(h) if row.get("scan_path")]
    logger.info("Subsample: %d scans", len(scans))

    done = load_done(output_file)
    work = [(s, st) for s in scans for st in strats if (s, "medgemma", st) not in done]
    logger.info("To process: %d (scan × strategy) pairs", len(work))

    header = list(OUTPUT_COLUMNS)
    processed, failed = 0, 0
    for scan_path, strat in tqdm(work, desc="medgemma", unit="pair"):
        try:
            vol = load_volume_uint8(Path(scan_path))
            slices = mid_slices(vol)
            concat = concat_horizontal(slices)
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": concat},
                    {"type": "text", "text": QC_PROMPT},
                ],
            }]
            inputs = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=True,
                return_tensors="pt", return_dict=True,
            )
            inputs = {
                k: (v.to(device, dtype) if torch.is_tensor(v) and v.is_floating_point()
                    else (v.to(device) if torch.is_tensor(v) else v))
                for k, v in inputs.items()
            }
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            text = processor.decode(out[0], skip_special_tokens=True)
            score = parse_score(text)
            row = {
                "scan_path": scan_path,
                "model": "medgemma",
                "slice_strategy": strat,
                "score": score,
                "raw_response": text,
                "seed": seed,
                "n_slices": 3,
                "multi_image_mode": "concat_grid",
            }
            append_row(output_file, row, header)
            processed += 1
        except Exception as exc:
            logger.warning("medgemma failed on %s (%s): %s", scan_path, strat, exc)
            row = {
                "scan_path": scan_path,
                "model": "medgemma",
                "slice_strategy": strat,
                "score": math.nan,
                "raw_response": f"ERROR:{exc}",
                "seed": seed,
                "n_slices": 3,
                "multi_image_mode": "concat_grid",
            }
            append_row(output_file, row, header)
            failed += 1

    logger.info("Done: processed=%d failed=%d output=%s", processed, failed, output_file)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    app()
