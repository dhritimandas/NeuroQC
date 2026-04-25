import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path

FREESURFER_LABELS = {
    "left_cortex": 3,    "right_cortex": 42,
    "left_wm": 2,        "right_wm": 41,
    "left_hippo": 17,    "right_hippo": 53,
    "left_vent": 4,      "right_vent": 43,
    "left_thalamus": 10, "right_thalamus": 49,
    "brainstem": 16,
    "left_cerebellum_cortex": 8,  "right_cerebellum_cortex": 47,
    "csf": 24,
}

def diagnose_segmentation(seg_path, scan_path):
    seg_img = nib.load(seg_path)
    seg = seg_img.get_fdata().astype(int)
    scan_img = nib.load(scan_path)
    # SynthSeg (without --keepgeom) outputs in its internal 1mm-iso space,
    # which can differ from the input scan's native grid (e.g. FastMRI
    # 320x320x16 @ 0.6875x0.6875x5mm input → 220x220x80 @ 1mm iso seg).
    # Resample the scan to the seg's grid so mask-based intensity metrics
    # work. Linear interpolation since we only use this for computing
    # intensity_cv within labeled regions.
    if scan_img.shape != seg.shape:
        from nibabel.processing import resample_from_to
        scan_img = resample_from_to(scan_img, seg_img, order=1)
    scan = scan_img.get_fdata()
    total_voxels = seg.size
    brain_voxels = (seg > 0).sum()

    results = {"scan": Path(scan_path).stem}

    # Check 1: Brain coverage (should be 8-25% of volume for whole brain MRI)
    results["brain_fraction"] = brain_voxels / total_voxels

    # Check 2: Number of unique labels found (SynthSeg+ outputs ~35 labels)
    results["n_unique_labels"] = len(np.unique(seg)) - 1  # minus background

    # Check 3: Presence of expected structures (non-zero voxel count)
    for name, label in FREESURFER_LABELS.items():
        results[f"has_{name}"] = int((seg == label).sum() > 100)

    # Check 4: Symmetry — L/R paired structures should have similar volumes
    for pair in ["cortex", "wm", "hippo", "vent", "thalamus", "cerebellum_cortex"]:
        l_count = (seg == FREESURFER_LABELS[f"left_{pair}"]).sum()
        r_count = (seg == FREESURFER_LABELS[f"right_{pair}"]).sum()
        if l_count + r_count > 0:
            asymmetry = abs(l_count - r_count) / (l_count + r_count)
            results[f"asymmetry_{pair}"] = asymmetry
        else:
            results[f"asymmetry_{pair}"] = np.nan

    # Check 5: Intensity coherence — within a tissue class, intensities
    # should be tightly clustered. High CV = label noise / failed segmentation.
    for name, label in [("left_wm", 2), ("left_cortex", 3)]:
        mask = seg == label
        if mask.sum() > 100:
            vals = scan[mask]
            results[f"intensity_cv_{name}"] = vals.std() / (vals.mean() + 1e-8)
        else:
            results[f"intensity_cv_{name}"] = np.nan

    # Check 6: Spatial connectivity of major structures
    # A healthy segmentation has each large structure as ~1-2 connected components.
    # Fragmented labels = SynthSeg is confused.
    from scipy.ndimage import label as cc_label
    for name, freesurfer_label in [("left_cortex", 3), ("left_wm", 2)]:
        mask = seg == freesurfer_label
        if mask.sum() > 100:
            _, n_components = cc_label(mask)
            results[f"n_components_{name}"] = n_components
        else:
            results[f"n_components_{name}"] = np.nan

    return results

# Run on 5 FastMRI + 5 IXI scans (IXI as reference baseline)
records = []
for scan_path in Path("data/fastmri/nifti").glob("*.nii.gz"):
    seg_path = Path("data/derivatives/synthseg/fastmri_diagnostic") / scan_path.name
    if seg_path.exists():
        r = diagnose_segmentation(seg_path, scan_path)
        r["dataset"] = "fastmri"
        records.append(r)

for scan_path in sorted(Path("data/ixi/raw").glob("*.nii.gz"))[:5]:
    seg_path = Path("data/derivatives/synthseg/ixi_diagnostic") / scan_path.name
    if seg_path.exists():
        r = diagnose_segmentation(seg_path, scan_path)
        r["dataset"] = "ixi"
        records.append(r)

df = pd.DataFrame(records)
df.to_csv("results/tables/synthseg_diagnostic.csv", index=False)
print(df.groupby("dataset").mean(numeric_only=True).T)
