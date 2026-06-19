"""
focal_overlap_planned.py

Calculate volumetric overlap and pressure statistics for the planned -3dB focal zone 
with the sgACC ROI across all subjects, format them into Typst table syntax,
and plot combined -3dB and -6dB focal zones in a single report per subject.

Combined from calculate_all_subjects_stats_typst.py and plot_focal_overlap_combined.py.

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
from ok_plan.focus_roi import focus_roi_pressure_stats
from ok_plan.nii_utils import pa_to_isppa_w_per_cm2, roi_centroid_mm

# Paths
BASE_DIR = "/Users/hoaithunguyen/Projects/Master-thesis/CITRUS"
INPUT_BASE = os.path.join(BASE_DIR, "data", "input")
PLAN_BASE = os.path.join(BASE_DIR, "data", "output")
OUT_DIR = os.path.join(BASE_DIR, "derivatives", "focal_overlap")
PLOT_DIR = os.path.join(OUT_DIR, "planned")

os.makedirs(PLOT_DIR, exist_ok=True)

# Custom Colormaps for plotting
cmap_6db = ListedColormap(["#D973FF"]) # purple
cmap_3db = ListedColormap(["#33B3FF"]) # cyan

# List of subjects to process
SUBJECTS = ["sub-03", "sub-04", "sub-05", "sub-06", "sub-11"]

table1_rows = []
table2_rows = []

for sub in SUBJECTS:
    print(f"\n=========================================")
    print(f" Processing subject: {sub}")
    print(f"=========================================")
    
    INPUT_DIR = os.path.join(INPUT_BASE, sub)
    PLAN_DIR = os.path.join(PLAN_BASE, sub, "planning", "exp-focused")
    
    # Common files
    t1_path = os.path.join(INPUT_DIR, f"{sub}_T1w_kplan.nii.gz")
    brain_mask_path = os.path.join(INPUT_DIR, f"{sub}_T1w_kplan_brain_mask.nii.gz")
    
    if not os.path.exists(t1_path) or not os.path.exists(brain_mask_path):
        print(f"Skipping {sub}: T1w or brain mask missing.")
        continue
        
    t1_img = nib.load(t1_path)
    brain_mask_img = nib.load(brain_mask_path)
    
    # Collect data for both sides for plotting and stats
    data = {}
    skip_subject = False
    for side in ["L", "R"]:
        side_name = "Left" if side == "L" else "Right"
        
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
        pressure_img = nib.load(press_files[0])
        
        # Resample ROI and T1w to the pressure voxel grid dynamically
        roi_res = image.resample_to_img(roi_img, pressure_img, interpolation="nearest")
        t1_res = image.resample_to_img(t1_img, pressure_img, interpolation="continuous")
        
        # Compute centroid of resampled ROI for cropping reference
        centroid = roi_centroid_mm(roi_res)
        
        # Resample and erode brain mask
        brain_res = image.resample_to_img(brain_mask_img, pressure_img, interpolation="nearest")
        mask_data = brain_res.get_fdata() > 0.5
        eroded_data = ndimage.binary_erosion(mask_data, iterations=3).astype(np.float32)
        eroded_img = nib.Nifti1Image(eroded_data, pressure_img.affine, pressure_img.header)
        
        # Calculate -3dB and -6dB focal zones
        focus_3db, focus_6db = build_focus_masks_amplitude_db(pressure_img, eroded_img)
        
        # Compute Stats (using -3dB focus)
        stats = focus_roi_pressure_stats(pressure_img, roi_res, focus_3db, label="-3 dB focus")
        
        # Calculate Dice coefficient
        dice = 2.0 * stats.n_overlap_voxels / (stats.n_focus_voxels + stats.n_roi_voxels + 1e-9)
        
        # Calculate pressure and intensity in overlap
        p_max_mpa = stats.p_max_in_overlap_pa / 1e6 if not np.isnan(stats.p_max_in_overlap_pa) else 0.0
        p_mean_mpa = stats.p_mean_in_overlap_pa / 1e6 if not np.isnan(stats.p_mean_in_overlap_pa) else 0.0
        i_max_w_cm2 = pa_to_isppa_w_per_cm2(stats.p_max_in_overlap_pa) if not np.isnan(stats.p_max_in_overlap_pa) else 0.0
        i_mean_w_cm2 = pa_to_isppa_w_per_cm2(stats.p_mean_in_overlap_pa) if not np.isnan(stats.p_mean_in_overlap_pa) else 0.0
        
        # Save table rows
        table1_rows.append((
            sub,
            side_name,
            f"{stats.overlap_pct_of_roi:.1f}%",
            f"{stats.on_target_pct:.1f}%",
            f"{stats.off_target_pct:.1f}%",
            f"{stats.overlap_vol_mm3:.1f}",
            f"{dice:.3f}"
        ))
        
        table2_rows.append((
            sub,
            side_name,
            f"{p_max_mpa:.4f}",
            f"{i_max_w_cm2:.3f}",
            f"{p_mean_mpa:.4f}",
            f"{i_mean_w_cm2:.3f}"
        ))

        t1_brain_data = t1_res.get_fdata() * eroded_data
        t1_brain_img = nib.Nifti1Image(t1_brain_data, pressure_img.affine, pressure_img.header)
        t1_brain_img = image.crop_img(t1_brain_img)

        data[side] = {
            "t1_res": t1_brain_img,
            "roi_res": roi_res,
            "focus_3db": focus_3db,
            "focus_6db": focus_6db,
            "centroid": centroid
        }
        
    if skip_subject:
        print(f"Skipping Subject {sub} due to missing files.")
        continue

    # Plotting Logic
    fig = plt.figure(figsize=(15, 10), facecolor="black")
    fig.suptitle(
        f"{sub} - Planned Experiment Focal Overlap on sgACC ROI\nRow 1: Left planned  |  Row 2: Right planned",
        color="white", fontsize=14, fontweight="bold", y=0.97
    )
    
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.02, wspace=0.01, left=0.03, right=0.97, top=0.88, bottom=0.08)
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
            
            d = plotting.plot_anat(t1_res, display_mode=mode, cut_coords=[coord], axes=ax, figure=fig, **PLOT_KW)
            d.add_overlay(focus_6db, cmap=cmap_6db, threshold=0.5, transparency=0.40, colorbar=False)
            d.add_overlay(focus_3db, cmap=cmap_3db, threshold=0.5, transparency=0.60, colorbar=False)
            d.add_contours(roi_res, levels=[0.5], colors=["white"], linewidths=0.8)
            
            for cut_ax in d.axes.values():
                for text_artist in list(cut_ax.ax.texts):
                    text_artist.remove()
                if mode in ["z", "y"]:
                    cut_ax.ax.invert_xaxis()
                    cut_ax.ax.text(0.04, 0.90, "L", color="white", fontsize=16.0, fontweight="bold", transform=cut_ax.ax.transAxes, va="top", ha="left")
                    cut_ax.ax.text(0.96, 0.90, "R", color="white", fontsize=16.0, fontweight="bold", transform=cut_ax.ax.transAxes, va="top", ha="right")
                elif mode == "x" and side == "L":
                    cut_ax.ax.invert_xaxis()
    
            if row_idx == 0:
                ax.set_title(view["title"], color="white", fontsize=16.0, pad=6)
            if col_idx == 0:
                ax.set_ylabel(f"{side_label} planned", color="white", fontsize=16.0, labelpad=8)
    
    legend_elements = [
        Patch(facecolor="none", edgecolor="white", linewidth=1.0, label="sgACC"),
        Patch(facecolor="#D973FF", edgecolor="none", label="-6 dB Focal Zone"),
        Patch(facecolor="#33B3FF", edgecolor="none", label="-3 dB Focal Zone"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, frameon=False, fontsize=16.0, labelcolor="white")
    
    out_path = os.path.join(PLOT_DIR, f"{sub}_planned_focal_overlap_combined.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"Saved combined report plot to: {out_path}")

# Generate Typst Output
typst_content = []
typst_content.append("// Overlap Statistics Table (planned −3 dB focus ∩ sgACC ROI)")
typst_content.append(r"""#table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  align: horizon + center,
  fill: (x, y) => if y == 0 { rgb("e0e0e0") } else if calc.even(y) { rgb("f9f9f9") } else { rgb("ffffff") },
  [*Subject*], [*Hemisphere*], [*Target coverage*], [*On-target*], [*Off-target*], [*Overlap volume (mm³)*], [*Dice Coefficient*],""")

for row in table1_rows:
    typst_content.append(f"  [{row[0]}], [{row[1]}], [{row[2]}], [{row[3]}], [{row[4]}], [{row[5]}], [{row[6]}],")
typst_content.append(")\n")

typst_content.append("// Pressure and Intensity in Overlap Table (planned −3 dB focus ∩ sgACC ROI)")
typst_content.append(r"""#table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: horizon + center,
  fill: (x, y) => if y == 0 { rgb("e0e0e0") } else if calc.even(y) { rgb("f9f9f9") } else { rgb("ffffff") },
  [*Subject*], [*Hemisphere*], [*P max (MPa)*], [*Isppa max (W/cm²)*], [*P mean (MPa)*], [*Isppa mean (W/cm²)*],""")

for row in table2_rows:
    typst_content.append(f"  [{row[0]}], [{row[1]}], [{row[2]}], [{row[3]}], [{row[4]}], [{row[5]}],")
typst_content.append(")\n")

output_file = os.path.join(OUT_DIR, "planned_overlap_stats_typst.txt")
with open(output_file, "w") as f:
    f.write("\n".join(typst_content))

print(f"\nTypst tables successfully written to: {output_file}")
print("All subjects processed successfully!")
