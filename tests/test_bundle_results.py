"""Unit tests for scripts/bundle_results.py.

Synthetic project tree only — no real CSVs or figures from the repo. The tests
build a fake project layout under ``tmp_path``, point the bundler at it via the
``make_bundle`` API (avoiding argparse), and assert the archive's contents +
manifest match expectations.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import types
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE_PATH = _REPO_ROOT / "scripts" / "bundle_results.py"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("bundle_results", _MODULE_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {_MODULE_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["bundle_results"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> types.ModuleType:
    return _load_module()


# ──────────────────────────────────────────────
# Synthetic project tree
# ──────────────────────────────────────────────


def _build_synthetic_tree(tmp_path: Path) -> dict[str, Path]:
    """Create a fake results / figures / checkpoints / segmentations layout."""
    results_dir = tmp_path / "results" / "tables"
    figures_dir = tmp_path / "figures"
    checkpoints_dir = tmp_path / "results" / "checkpoints"
    segmentations_dir = tmp_path / "data" / "derivatives" / "synthseg"

    for d in (results_dir, figures_dir, checkpoints_dir, segmentations_dir):
        d.mkdir(parents=True, exist_ok=True)

    (results_dir / "machine_preference.csv").write_text("ref_path,cor_path\n/r,/c\n")
    (results_dir / "per_structure_dice.csv").write_text(
        "ref_path,cor_path,label_id,dice\n/r,/c,17,0.85\n"
    )
    (results_dir / "iqm_features.csv").write_text("scan_path,snr\n/c,4.5\n")
    (results_dir / "3d_vlm_scores_seed_0.csv").write_text(
        "scan_path,model,score\n/c,m3d_lamed,0.6\n"
    )
    (results_dir / "finetune_run_info_m3d_lamed_seed_0.json").write_text(
        json.dumps({"seed": 0, "best_val_srcc": 0.7})
    )
    (results_dir / "finetune_diff_20260425.diff").write_text(
        "diff --git a/x b/x\n+++ added\n"
    )

    (figures_dir / "fig_iqm_heatmap.png").write_bytes(b"\x89PNG fake\n")
    (figures_dir / "fig_iqm_heatmap.svg").write_text("<svg/>")

    (checkpoints_dir / "m3d_lamed_lora_seed_0").mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / "m3d_lamed_lora_seed_0" / "adapter.bin").write_bytes(
        b"fake_lora_weights" * 100
    )

    (segmentations_dir / "scanA_synthseg.nii.gz").write_bytes(b"fake_seg" * 50)

    return {
        "results_dir": results_dir,
        "figures_dir": figures_dir,
        "checkpoints_dir": checkpoints_dir,
        "segmentations_dir": segmentations_dir,
        "output_dir": tmp_path / "bundles",
        "tmp_path": tmp_path,
    }


def _patch_repo_root_to_tmp(
    mod: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Make arcname computation think the project root is tmp_path."""
    monkeypatch.setattr(mod, "_REPO_ROOT", tmp_path)


def _read_archive_members(archive_path: Path) -> dict[str, bytes]:
    """Return ``{member_name: raw_bytes}`` for every file inside the tarball."""
    out: dict[str, bytes] = {}
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            extracted = tar.extractfile(member)
            assert extracted is not None
            out[member.name] = extracted.read()
    return out


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────


def test_default_bundle_includes_csvs_figures_provenance(
    mod: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default bundle includes results/* + figures/* + provenance JSONs/diffs."""
    paths = _build_synthetic_tree(tmp_path)
    _patch_repo_root_to_tmp(mod, tmp_path, monkeypatch)

    archive = mod.make_bundle(
        results_dir=paths["results_dir"],
        figures_dir=paths["figures_dir"],
        checkpoints_dir=paths["checkpoints_dir"],
        segmentations_dir=paths["segmentations_dir"],
        output_dir=paths["output_dir"],
    )
    assert archive is not None
    assert archive.is_file()
    assert archive.suffix == ".gz"

    members = _read_archive_members(archive)
    member_names = {m.split("/", 1)[1] if "/" in m else m for m in members}

    # Always-present.
    assert "manifest.json" in member_names
    assert "README.txt" in member_names
    # CSVs.
    assert "results/tables/machine_preference.csv" in member_names
    assert "results/tables/per_structure_dice.csv" in member_names
    assert "results/tables/iqm_features.csv" in member_names
    assert "results/tables/3d_vlm_scores_seed_0.csv" in member_names
    # Figures (PNG + SVG).
    assert "figures/fig_iqm_heatmap.png" in member_names
    assert "figures/fig_iqm_heatmap.svg" in member_names
    # Provenance.
    assert "results/tables/finetune_run_info_m3d_lamed_seed_0.json" in member_names
    assert "results/tables/finetune_diff_20260425.diff" in member_names
    # Defaults to NOT including checkpoints / segmentations.
    assert not any("checkpoints" in n for n in member_names)
    assert not any("synthseg" in n for n in member_names)


def test_excludes_checkpoints_and_segmentations_by_default(
    mod: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sanity invariant: large binaries stay out unless the flags are set."""
    paths = _build_synthetic_tree(tmp_path)
    _patch_repo_root_to_tmp(mod, tmp_path, monkeypatch)

    archive = mod.make_bundle(
        results_dir=paths["results_dir"],
        figures_dir=paths["figures_dir"],
        checkpoints_dir=paths["checkpoints_dir"],
        segmentations_dir=paths["segmentations_dir"],
        output_dir=paths["output_dir"],
    )
    members = _read_archive_members(archive)
    assert not any("adapter.bin" in n for n in members)
    assert not any("scanA_synthseg" in n for n in members)


def test_include_checkpoints_flag_adds_them(
    mod: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--include-checkpoints does what it says."""
    paths = _build_synthetic_tree(tmp_path)
    _patch_repo_root_to_tmp(mod, tmp_path, monkeypatch)

    archive = mod.make_bundle(
        results_dir=paths["results_dir"],
        figures_dir=paths["figures_dir"],
        checkpoints_dir=paths["checkpoints_dir"],
        segmentations_dir=paths["segmentations_dir"],
        output_dir=paths["output_dir"],
        include_checkpoints=True,
    )
    members = _read_archive_members(archive)
    assert any("adapter.bin" in n for n in members)


def test_manifest_records_sha256_and_sizes(
    mod: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every file in the bundle has a SHA-256 + size in manifest.json that
    matches the actual archived bytes."""
    paths = _build_synthetic_tree(tmp_path)
    _patch_repo_root_to_tmp(mod, tmp_path, monkeypatch)

    archive = mod.make_bundle(
        results_dir=paths["results_dir"],
        figures_dir=paths["figures_dir"],
        checkpoints_dir=paths["checkpoints_dir"],
        segmentations_dir=paths["segmentations_dir"],
        output_dir=paths["output_dir"],
    )
    members = _read_archive_members(archive)
    manifest_member = next(n for n in members if n.endswith("/manifest.json"))
    manifest = json.loads(members[manifest_member].decode("utf-8"))

    assert manifest["n_files"] == len(manifest["files"])
    assert manifest["n_files"] > 0
    bundle_dir = manifest_member.rsplit("/", 1)[0]
    for entry in manifest["files"]:
        archived_name = f"{bundle_dir}/{entry['arcname']}"
        assert archived_name in members, f"missing {archived_name} in archive"
        sha = hashlib.sha256(members[archived_name]).hexdigest()
        assert sha == entry["sha256"], (
            f"SHA-256 mismatch for {entry['arcname']}: "
            f"manifest={entry['sha256']} vs archived={sha}"
        )
        assert entry["size_bytes"] == len(members[archived_name])


def test_dry_run_writes_no_archive(
    mod: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """--dry-run logs the file list and returns None without creating an archive."""
    paths = _build_synthetic_tree(tmp_path)
    _patch_repo_root_to_tmp(mod, tmp_path, monkeypatch)

    with caplog.at_level("INFO", logger="bundle_results"):
        result = mod.make_bundle(
            results_dir=paths["results_dir"],
            figures_dir=paths["figures_dir"],
            checkpoints_dir=paths["checkpoints_dir"],
            segmentations_dir=paths["segmentations_dir"],
            output_dir=paths["output_dir"],
            dry_run=True,
        )
    assert result is None
    # No archive on disk.
    assert not paths["output_dir"].exists() or not list(paths["output_dir"].glob("*.tar.gz"))
    # Log mentions at least one expected file.
    log_text = "\n".join(rec.message for rec in caplog.records)
    assert "machine_preference.csv" in log_text


def test_filename_has_timestamp_and_git_hash(
    mod: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bundle filename matches ``neuroqc_bundle_<UTC>_<git_short>.tar.gz``."""
    paths = _build_synthetic_tree(tmp_path)
    _patch_repo_root_to_tmp(mod, tmp_path, monkeypatch)

    # Stub git state so the filename is deterministic.
    monkeypatch.setattr(mod, "get_git_state", lambda: {
        "commit": "abc" * 13 + "def",
        "short_commit": "abc1234",
        "dirty": False,
        "status": "",
        "diff": "",
    })

    archive = mod.make_bundle(
        results_dir=paths["results_dir"],
        figures_dir=paths["figures_dir"],
        checkpoints_dir=paths["checkpoints_dir"],
        segmentations_dir=paths["segmentations_dir"],
        output_dir=paths["output_dir"],
    )
    name = archive.name
    assert name.startswith("neuroqc_bundle_")
    assert "abc1234" in name
    assert name.endswith(".tar.gz")
    # ISO timestamp segment: 8 digits + T + 6 digits + Z
    import re
    assert re.search(r"_\d{8}T\d{6}Z_", name) is not None


def test_tag_appends_to_filename(
    mod: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--tag suffix appears at the tail of the filename before .tar.gz."""
    paths = _build_synthetic_tree(tmp_path)
    _patch_repo_root_to_tmp(mod, tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "get_git_state", lambda: {
        "commit": "x" * 40, "short_commit": "x1234567",
        "dirty": False, "status": "", "diff": "",
    })

    archive = mod.make_bundle(
        results_dir=paths["results_dir"],
        figures_dir=paths["figures_dir"],
        checkpoints_dir=paths["checkpoints_dir"],
        segmentations_dir=paths["segmentations_dir"],
        output_dir=paths["output_dir"],
        tag="pre-revision",
    )
    assert "_pre-revision.tar.gz" in archive.name


def test_readme_contains_extraction_instructions(
    mod: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """README inside the archive explains how to verify checksums + regenerate figures."""
    paths = _build_synthetic_tree(tmp_path)
    _patch_repo_root_to_tmp(mod, tmp_path, monkeypatch)

    archive = mod.make_bundle(
        results_dir=paths["results_dir"],
        figures_dir=paths["figures_dir"],
        checkpoints_dir=paths["checkpoints_dir"],
        segmentations_dir=paths["segmentations_dir"],
        output_dir=paths["output_dir"],
    )
    members = _read_archive_members(archive)
    readme_name = next(n for n in members if n.endswith("/README.txt"))
    readme = members[readme_name].decode("utf-8")
    assert "tar xzf" in readme
    assert "manifest.json" in readme
    assert "code/visualize.py" in readme
    assert "library versions" in readme.lower()
