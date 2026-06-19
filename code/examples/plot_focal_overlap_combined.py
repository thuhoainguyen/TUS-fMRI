"""
plot_focal_overlap_combined.py

Plot combined -3dB and -6dB focal zones of the planned pressure maps
for all subjects against the sgACC ROI masks, in a single 2-row report.
Row 1: Left planned (Axial, Sagittal, Coronal)
Row 2: Right planned (Axial, Sagittal, Coronal)

@author Hoai Thu Nguyen
"""

import os
import glob
import numpy as np
import nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from nilearn import image, plotting

from ok_plan.focus import build_focus_masks_amplitude_db
from ok_plan.nii_utils import roi_centroid_mm

# Paths
BASE_DIR = "/Users/hoaithunguyen/Projects/Master-thesis/CITRUS"
INPUT_BASE = os.path.join(BASE_DIR, "data", "input")
PLAN_BASE = os.path.join(BASE_DIR, "data", "output")
OUT_DIR = os.path.join(BASE_DIR, "derivatives", "focal_overlap", "planned")

os.makedirs(OUT_DIR, exist_ok=True)

# Custom Colormaps: solid colors (transparency is handled by Nilearn's transparency argument)
cmap_6db = ListedColormap(["#D973FF"]) # purple
cmap_3db = ListedColormap(["#33B3FF"]) # cyan

# List of subjects to process
SUBJECTS = ["sub-03", "sub-04", "sub-05", "sub-06", "sub-11"]

for sub in SUBJECTS:
    print(f"\n=========================================")
    print(f" Processing subject: {sub}")
    print(f"=========================================")
    
    INPUT_DIR = os.path.join(INPUT_BASE, sub)
    PLAN_DIR = os.path.join(PLAN_BASE, sub, "planning", "exp-focused")
    
    # Common files
    t1_path = os.path.join(INPUT_DIR, f"{sub}_T1w_kplan.nii.gz")
    brain_mask_path = os.path.join(INPUT_DIR, f"{sub}_T1w_kplan_brain_mask.nii.gz")
    
    if not os.path.exists(t1_path):
        print(f"Skipping {sub}: T1w image not found at {t1_path}")
        continue
    if not os.path.exists(brain_mask_path):
        print(f"Skipping {sub}: Brain mask not found at {brain_mask_path}")
        continue
        
    print("Loading background T1w and brain mask...")
    t1_img = nib.load(t1_path)
    brain_mask_img = nib.load(brain_mask_path)
    
    # Collect data for both sides
    data = {}
    skip_subject = False
    for side in ["L", "R"]:
        # sgACC ROI
        roi_path = os.path.join(INPUT_DIR, f"sgACC_BA25_{side}_kplan.nii.gz")
        if not os.path.exists(roi_path):
            print(f"Warning: ROI path not found: {roi_path}")
            skip_subject = True
            break
        roi_img = nib.load(roi_path)
        
        # Pressure map
        press_pattern = os.path.join(PLAN_DIR, f"*_{side}_pos-*_exp-Pressure.nii.gz")
        press_files = glob.glob(press_pattern)
        if not press_files:
            print(f"Warning: No pressure files found matching pattern: {press_pattern}")
            skip_subject = True
            break
        press_path = press_files[0]
        print(f"Found pressure file ({side}): {os.path.basename(press_path)}")
        pressure_img = nib.load(press_path)
        
        # Resample ROI and T1w to the pressure voxel grid dynamically
        print(f"Resampling sgACC ROI and T1w to pressure grid ({side})...")
        roi_res = image.resample_to_img(roi_img, pressure_img, interpolation="nearest")
        t1_res = image.resample_to_img(t1_img, pressure_img, interpolation="continuous")
        
        # Compute centroid of resampled ROI for cropping reference
        centroid = roi_centroid_mm(roi_res)
        print(f"Resampled ROI centroid ({side}): {centroid}")
        
        # Resample brain mask to pressure voxel grid
        print(f"Resampling and eroding brain mask ({side})...")
        brain_res = image.resample_to_img(brain_mask_img, pressure_img, interpolation="nearest")
        
        # Erode 3x to match the report's eroded brain pressure map
        mask_data = brain_res.get_fdata() > 0.5
        eroded_data = ndimage.binary_erosion(mask_data, iterations=3).astype(np.float32)
        eroded_img = nib.Nifti1Image(eroded_data, pressure_img.affine, pressure_img.header)
        
        # Calculate -3dB and -6dB focal zones
        print(f"Calculating focal zones ({side})...")
        focus_3db, focus_6db = build_focus_masks_amplitude_db(pressure_img, eroded_img)
        
        data[side] = {
            "t1_res": t1_res,
            "roi_res": roi_res,
            "focus_3db": focus_3db,
            "focus_6db": focus_6db,
            "centroid": centroid
        }
        
    if skip_subject:
        print(f"Skipping Subject {sub} due to missing files.")
        continue

    # Create figure: 2 rows x 3 columns (Axial, Sagittal, Coronal)
    fig = plt.figure(figsize=(15, 10), facecolor="black")
    fig.suptitle(
        f"{sub} - Planned Experiment Focal Overlap on sgACC ROI\nRow 1: Left planned  |  Row 2: Right planned",
        color="white", fontsize=14, fontweight="bold", y=0.97
    )
    
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.15, wspace=0.05, left=0.03, right=0.97, top=0.88, bottom=0.08)
    
    # Layout setup per row/column
    # Columns: 0 = Sagittal, 1 = Axial, 2 = Coronal
    views = [
        {"mode": "x", "title": "Sagittal", "index": 0},
        {"mode": "z", "title": "Axial", "index": 2},
        {"mode": "y", "title": "Coronal", "index": 1}
    ]
    
    PLOT_KW = dict(annotate=False, draw_cross=False, black_bg=True, colorbar=False)
    
    for row_idx, side in enumerate(["L", "R"]):
        side_data = data[side]
        centroid = side_data["centroid"]
        t1_res = side_data["t1_res"]
        roi_res = side_data["roi_res"]
        focus_3db = side_data["focus_3db"]
        focus_6db = side_data["focus_6db"]
        
        side_label = "Left" if side == "L" else "Right"
        
        for col_idx, view in enumerate(views):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            mode = view["mode"]
            coord = centroid[view["index"]]
            
            # Plot anatomical underlay
            d = plotting.plot_anat(t1_res, display_mode=mode, cut_coords=[coord], axes=ax, figure=fig, **PLOT_KW)
            
            # Overlay -6dB focus (underlayer, purple)
            d.add_overlay(focus_6db, cmap=cmap_6db, threshold=0.5, transparency=0.40, colorbar=False)
            
            # Overlay -3dB focus (upperlayer, cyan)
            d.add_overlay(focus_3db, cmap=cmap_3db, threshold=0.5, transparency=0.60, colorbar=False)
            
            # Overlay ROI contours (white outline) - NO RED FILL!
            d.add_contours(roi_res, levels=[0.5], colors=["white"], linewidths=0.8)
            
            # Clear default Nilearn annotations and render clean custom labels
            for cut_ax in d.axes.values():
                for text_artist in list(cut_ax.ax.texts):
                    text_artist.remove()
                
                # L/R labels on Axial & Coronal
                if mode in ["z", "y"]:
                    cut_ax.ax.invert_xaxis()
                    cut_ax.ax.text(0.04, 0.90, "L", color="white", fontsize=16.0,
                                   fontweight="bold", transform=cut_ax.ax.transAxes,
                                   va="top", ha="left", zorder=100)
                    cut_ax.ax.text(0.96, 0.90, "R", color="white", fontsize=16.0,
                                   fontweight="bold", transform=cut_ax.ax.transAxes,
                                   va="top", ha="right", zorder=100)
                elif mode == "x" and side == "L":
                    cut_ax.ax.invert_xaxis()
    
            # Set column titles on the top row
            if row_idx == 0:
                ax.set_title(view["title"], color="white", fontsize=11, pad=6)
                
            # Set row labels on the leftmost column
            if col_idx == 0:
                ax.set_ylabel(f"{side_label} planned", color="white", fontsize=11, labelpad=8)
    
    # Unified Legend
    legend_elements = [
        Patch(facecolor="none", edgecolor="white", linewidth=1.0, label="sgACC"),
        Patch(facecolor="#D973FF", edgecolor="none", label="-6 dB Focal Zone"),
        Patch(facecolor="#33B3FF", edgecolor="none", label="-3 dB Focal Zone"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=3,
        frameon=False, fontsize=11, labelcolor="white"
    )
    
    # Save subject figure
    out_path = os.path.join(OUT_DIR, f"{sub}_planned_focal_overlap_combined.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"Saved combined report plot to: {out_path}")

print("\nAll subjects processed successfully!")
