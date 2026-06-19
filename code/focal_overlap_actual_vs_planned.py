"""
focal_overlap_actual_vs_planned.py

Plot planned vs. actual -3dB focal zones for all subjects against the sgACC ROI masks,
calculate volumetric overlap, spatial shift, and dosimetry statistics for the actual focus,
and format them into Typst table syntax.

Layout: 2 rows x 3 columns (Axial, Sagittal, Coronal)
Row 1: Left planned vs. actual L
Row 2: Right planned vs. actual R

Color scheme:
- Planned -3dB focal zone (only): Yellow (#FFFF00), 50% transparency
- Actual -3dB focal zone (only): Blue (#33B3FF), 50% transparency
- Overlap (planned & actual): Green (#00FF00), 70% transparency
- sgACC target ROI: White contour outline only

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
OUT_DIR = os.path.join(BASE_DIR, "derivatives", "focal_overlap", "actual_vs_planned")

os.makedirs(OUT_DIR, exist_ok=True)

# Custom Colormaps for the different segments
cmap_plan_only = ListedColormap(["#FFFF00"])    # Yellow
cmap_act_only = ListedColormap(["#33B3FF"])     # Blue (cyan)
cmap_overlap = ListedColormap(["#00FF00"])      # Green

# List of subjects to process
SUBJECTS = ["sub-03", "sub-04", "sub-05", "sub-06", "sub-11"]

def mask_com_mm(mask_img):
    """Calculate Center of Mass in mm for a binary Nifti image."""
    data = mask_img.get_fdata() > 0.5
    if not np.any(data):
        return None
    vox = ndimage.center_of_mass(data)
    return nib.affines.apply_affine(mask_img.affine, vox)


def pca_dimensions_mm(binary_mask: np.ndarray, affine: np.ndarray):
    """Calculate the 3D dimensions of a binary mask along its principal components."""
    coords = np.argwhere(binary_mask)
    if coords.shape[0] < 3:
        return 0.0, 0.0, 0.0
    from nibabel.affines import apply_affine
    world = apply_affine(affine, coords)
    centered = world - world.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    proj = centered @ vecs
    dims = proj.max(axis=0) - proj.min(axis=0)
    dims = np.sort(dims)[::-1]
    return float(dims[0]), float(dims[1]), float(dims[2])


records = []

for sub in SUBJECTS:
    print(f"\n=========================================")
    print(f" Processing subject: {sub}")
    print(f"=========================================")
    
    INPUT_DIR = os.path.join(INPUT_BASE, sub)
    PLAN_DIR = os.path.join(PLAN_BASE, sub, "planning", "exp-focused")
    POST_DIR = os.path.join(PLAN_BASE, sub, "posthoc", "exp-focused")
    
    # Common files
    t1_path = os.path.join(INPUT_DIR, f"{sub}_T1w_kplan.nii.gz")
    brain_mask_path = os.path.join(INPUT_DIR, f"{sub}_T1w_kplan_brain_mask.nii.gz")
    
    if not os.path.exists(t1_path) or not os.path.exists(brain_mask_path):
        print(f"Skipping {sub}: T1w or brain mask missing.")
        continue
        
    t1_img = nib.load(t1_path)
    brain_mask_img = nib.load(brain_mask_path)
    
    # Collect data for both sides
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
        
        # Planned Pressure map
        press_pattern = os.path.join(PLAN_DIR, f"*_{side}_pos-*_exp-Pressure.nii.gz")
        press_files = glob.glob(press_pattern)
        if not press_files:
            print(f"Warning: No planned pressure files found matching pattern: {press_pattern}")
            skip_subject = True
            break
        plan_press_path = press_files[0]
        plan_press_img = nib.load(plan_press_path)
        
        # Actual Pressure map
        act_patterns = [
            os.path.join(POST_DIR, f"*_{side}_pos-medoid-*_post-hoc-exp - Pressure.nii.gz"),
            os.path.join(POST_DIR, f"*_{side}_pos-medoid-*_post-hoc-exp-Pressure.nii.gz"),
            os.path.join(POST_DIR, f"*_{side}_pos-*Pressure.nii.gz"),
        ]
        act_files = []
        for pat in act_patterns:
            act_files = glob.glob(pat)
            if act_files:
                break
        if not act_files:
            print(f"Warning: No actual pressure files found matching patterns in: {POST_DIR}")
            skip_subject = True
            break
        act_press_path = act_files[0]
        act_press_img = nib.load(act_press_path)
        
        # Resample
        roi_res = image.resample_to_img(roi_img, plan_press_img, interpolation="nearest")
        t1_res = image.resample_to_img(t1_img, plan_press_img, interpolation="continuous")
        actual_res = image.resample_to_img(act_press_img, plan_press_img, interpolation="continuous")
        
        centroid = roi_centroid_mm(roi_res)
        
        brain_res = image.resample_to_img(brain_mask_img, plan_press_img, interpolation="nearest")
        mask_data = brain_res.get_fdata() > 0.5
        eroded_data = ndimage.binary_erosion(mask_data, iterations=3).astype(np.float32)
        eroded_img = nib.Nifti1Image(eroded_data, plan_press_img.affine, plan_press_img.header)
        
        # Calculate -3dB focal zones
        plan_3db, _ = build_focus_masks_amplitude_db(plan_press_img, eroded_img)
        actual_3db, _ = build_focus_masks_amplitude_db(actual_res, eroded_img)
        
        # Boolean masks for overlap analysis
        plan_data = plan_3db.get_fdata() > 0.5
        act_data = actual_3db.get_fdata() > 0.5
        overlap_sim_data = (plan_data & act_data)
        n_plan = int(np.sum(plan_data))
        n_act = int(np.sum(act_data))
        n_overlap_pa = int(np.sum(overlap_sim_data))

        # Actual Temperature map
        act_temp_patterns = [
            os.path.join(POST_DIR, f"*_{side}_pos-medoid-*_post-hoc-exp - Temperature.nii.gz"),
            os.path.join(POST_DIR, f"*_{side}_pos-medoid-*_post-hoc-exp-Temperature.nii.gz"),
            os.path.join(POST_DIR, f"*_{side}_pos-*Temperature.nii.gz"),
        ]
        act_temp_files = []
        for pat in act_temp_patterns:
            act_temp_files = glob.glob(pat)
            if act_temp_files:
                break
        if act_temp_files:
            act_temp_img = nib.load(act_temp_files[0])
            act_temp_res = image.resample_to_img(act_temp_img, plan_press_img, interpolation="continuous")
            act_temp_data = act_temp_res.get_fdata()
            peak_t_brain = float(np.max(act_temp_data[eroded_data > 0.5]))
        else:
            peak_t_brain = 37.0

        # Peak pressure and Isppa AT the sgACC target mask
        # (matches kplan PDF: kplan reports at the planned focus/target, not beam-path max)
        act_press_data = np.abs(actual_res.get_fdata())
        roi_data_bool = roi_res.get_fdata() > 0.5
        pmax_target_pa = float(np.max(act_press_data[roi_data_bool])) if np.any(roi_data_bool) else 0.0
        peak_p_brain_mpa = pmax_target_pa / 1e6
        isppa_max = pa_to_isppa_w_per_cm2(pmax_target_pa)

        # FWHM of actual focus
        act_dims = pca_dimensions_mm(act_data, plan_press_img.affine)
        axial_fwhm = act_dims[0]
        lateral_fwhm = (act_dims[1] + act_dims[2]) / 2.0

        # Dice: actual vs planned focal zones (not vs sgACC)
        dice_coeff = 2.0 * n_overlap_pa / (n_plan + n_act + 1e-9)
        # % of actual focal volume that falls within the planned focal zone
        pct_focus_in_sgacc = 100.0 * n_overlap_pa / (n_act + 1e-9)

        # Centroid shift: closest component of each focal zone to sgACC centroid
        # (avoids centroid being pulled by disconnected proximal side-lobes along beam path)
        com_roi = ndimage.center_of_mass(roi_data_bool.astype(float))
        com_roi_mm = np.array(nib.affines.apply_affine(plan_press_img.affine, np.array(com_roi)))

        # Centroid shift: center of mass of planned vs actual -3dB focal zone
        com_plan = mask_com_mm(plan_3db)
        com_act  = mask_com_mm(actual_3db)
        centroid_shift = float(np.linalg.norm(np.array(com_plan) - np.array(com_act))) if (com_plan is not None and com_act is not None) else 0.0

        records.append({
            "subject": sub,
            "focus_hemisphere": side_name,
            "peak_pressure_mpa": peak_p_brain_mpa,
            "isppa_max": isppa_max,
            "peak_temp_c": peak_t_brain,
            "axial_fwhm": axial_fwhm,
            "lateral_fwhm": lateral_fwhm,
            "dice": dice_coeff,
            "pct_focus_in_sgacc": pct_focus_in_sgacc,
            "centroid_shift": centroid_shift
        })

        # Segment masks for plotting
        plan_only_img = nib.Nifti1Image((plan_data & ~act_data).astype(np.float32), plan_3db.affine, plan_3db.header)
        act_only_img = nib.Nifti1Image((act_data & ~plan_data).astype(np.float32), plan_3db.affine, plan_3db.header)
        overlap_img = nib.Nifti1Image(overlap_sim_data.astype(np.float32), plan_3db.affine, plan_3db.header)
        
        t1_brain_data = t1_res.get_fdata() * eroded_data
        t1_brain_img = nib.Nifti1Image(t1_brain_data, plan_press_img.affine, plan_press_img.header)
        t1_brain_img = image.crop_img(t1_brain_img)
        
        data[side] = {
            "t1_brain": t1_brain_img,
            "roi_res": roi_res,
            "plan_only": plan_only_img,
            "act_only": act_only_img,
            "overlap": overlap_img,
            "centroid": centroid
        }
        
    if skip_subject:
        continue

    # Plotting Logic
    fig = plt.figure(figsize=(15, 10), facecolor="black")
    fig.suptitle(
        f"{sub} - Planned vs. Post-hoc -3 dB Focal Zone Overlap\nRow 1: Left hemisphere  |  Row 2: Right hemisphere",
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
        t1_brain = side_data["t1_brain"]
        roi_res = side_data["roi_res"]
        plan_only = side_data["plan_only"]
        act_only = side_data["act_only"]
        overlap = side_data["overlap"]
        side_label = "Left" if side == "L" else "Right"
        
        for col_idx, view in enumerate(views):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            mode = view["mode"]
            coord = centroid[view["index"]]
            
            d = plotting.plot_anat(t1_brain, display_mode=mode, cut_coords=[coord], axes=ax, figure=fig, **PLOT_KW)
            d.add_overlay(plan_only, cmap=cmap_plan_only, threshold=0.5, transparency=0.50, colorbar=False)
            d.add_overlay(act_only, cmap=cmap_act_only, threshold=0.5, transparency=0.50, colorbar=False)
            d.add_overlay(overlap, cmap=cmap_overlap, threshold=0.5, transparency=0.70, colorbar=False)
            d.add_contours(roi_res, levels=[0.5], colors=["white"], linewidths=0.8)
            
            # kplan x-axis increases to the RIGHT — nilearn renders correctly without inversion.
            # Labels follow neurological convention: L on left, R on right.
            for cut_ax in d.axes.values():
                for text_artist in list(cut_ax.ax.texts):
                    text_artist.remove()
                if mode in ["z", "y"]:
                    cut_ax.ax.text(0.04, 0.90, "L", color="white", fontsize=16.0, fontweight="bold", transform=cut_ax.ax.transAxes, va="top", ha="left")
                    cut_ax.ax.text(0.96, 0.90, "R", color="white", fontsize=16.0, fontweight="bold", transform=cut_ax.ax.transAxes, va="top", ha="right")
    
            if row_idx == 0:
                ax.set_title(view["title"], color="white", fontsize=16.0, pad=6)
            if col_idx == 0:
                ax.set_ylabel(f"{side_label} hemisphere", color="white", fontsize=16.0, labelpad=8)
    
    legend_elements = [
        Patch(facecolor="none", edgecolor="white", linewidth=1.2, label="sgACC (BA25) ROI"),
        Patch(facecolor="#FFFF00", edgecolor="none", alpha=0.50, label="Planned Focus (−3 dB)"),
        Patch(facecolor="#33B3FF", edgecolor="none", alpha=0.50, label="Post-hoc Focus (−3 dB)"),
        Patch(facecolor="#00FF00", edgecolor="none", alpha=0.70, label="Overlap (Plan ∩ Post-hoc)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, frameon=False, fontsize=16.0, labelcolor="white")
    
    out_path = os.path.join(OUT_DIR, f"{sub}_planned_vs_actual_overlap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"Saved overlap report plot to: {out_path}")

# Generate Output Reports
    md_table_lines = [
        "# Actual vs. Planned Focal Overlap Statistics",
        "",
        "| Subject | Hemisphere | Peak Pressure (MPa) | Isppa Max (W/cm²) | Peak Temp (°C) | Focal Spot FWHM (mm) | Dice (Post-hoc∩Plan) | Post-hoc Vol in Planned (%) | Centroid Shift (mm) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    typst_table_lines = [
        "// Post-hoc vs. Planned Focal Overlap Statistics Table",
        "#table(",
        "  columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),",
        "  align: horizon + center,",
        "  fill: (x, y) => if y == 0 { rgb(\"e0e0e0\") } else if calc.even(y) { rgb(\"f9f9f9\") } else { rgb(\"ffffff\") },",
        "  [*Subject*], [*Hemisphere*], [*Peak Pressure (MPa)*], [*Isppa Max (W/cm²)*], [*Peak Temp (°C)*], [*Focal Spot FWHM (mm)*], [*Dice (Post-hoc∩Plan)*], [*Post-hoc Vol in Planned (%)*], [*Centroid Shift (mm)*],"
    ]

    # Print header to terminal
    print("\n" + "="*105)
    print(" POST-HOC VS. PLANNED FOCAL OVERLAP STATISTICS")
    print("="*105)
    print(f"{'Subject':<8} | {'Hemisphere':<10} | {'P max (MPa)':<12} | {'Isppa max':<10} | {'T max (°C)':<10} | {'FWHM (mm)':<12} | {'Dice':<6} | {'Post-hoc in Plan':<17} | {'Shift (mm)':<10}")
    print("-"*105)

    for r in records:
        fwhm_str = f"{r['axial_fwhm']:.1f} × {r['lateral_fwhm']:.1f}"
        print(f"{r['subject']:<8} | {r['focus_hemisphere']:<10} | {r['peak_pressure_mpa']:<12.4f} | {r['isppa_max']:<10.2f} | {r['peak_temp_c']:<10.2f} | {fwhm_str:<12} | {r['dice']:<6.3f} | {r['pct_focus_in_sgacc']:<12.1f}% | {r['centroid_shift']:<10.2f}")

        md_table_lines.append(
            f"| {r['subject']} | {r['focus_hemisphere']} | {r['peak_pressure_mpa']:.4f} | {r['isppa_max']:.2f} | {r['peak_temp_c']:.2f} | {fwhm_str} | {r['dice']:.3f} | {r['pct_focus_in_sgacc']:.1f}% | {r['centroid_shift']:.2f} |"
        )

        typst_table_lines.append(
            f"  [{r['subject']}], [{r['focus_hemisphere']}], [{r['peak_pressure_mpa']:.4f}], [{r['isppa_max']:.2f}], [{r['peak_temp_c']:.2f}], [{fwhm_str}], [{r['dice']:.3f}], [{r['pct_focus_in_sgacc']:.1f}%], [{r['centroid_shift']:.2f}],"
        )

    typst_table_lines.append(")")

    # Save md table
    md_path = os.path.join(OUT_DIR, "focal_overlap_actual_vs_planned_stats.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_table_lines) + "\n")
    print(f"Saved Markdown table to: {md_path}")

    # Save plain text report
    txt_lines = [
        "=========================================================================================================",
        " ACTUAL VS. PLANNED FOCAL OVERLAP STATISTICS",
        "  Peak pressure/Isppa = max inside sgACC target mask (matches kplan PDF focal-spot reporting)",
        "  Dice = 2*|post-hoc∩plan| / (|post-hoc|+|plan|)    Post-hoc in Plan = % of post-hoc focal volume within planned zone",
        "=========================================================================================================",
        f"{'Subject':<8} | {'Hemisphere':<10} | {'P max (MPa)':<12} | {'Isppa max':<10} | {'T max (°C)':<10} | {'FWHM (mm)':<12} | {'Dice':<6} | {'Act in Plan':<13} | {'Centroid Shift (mm)':<10}",
        "---------------------------------------------------------------------------------------------------------"
    ]
    for r in records:
        fwhm_str = f"{r['axial_fwhm']:.1f} x {r['lateral_fwhm']:.1f}"
        txt_lines.append(
            f"{r['subject']:<8} | {r['focus_hemisphere']:<10} | {r['peak_pressure_mpa']:<12.4f} | {r['isppa_max']:<10.2f} | {r['peak_temp_c']:<10.2f} | {fwhm_str:<12} | {r['dice']:<6.3f} | {r['pct_focus_in_sgacc']:<12.1f}% | {r['centroid_shift']:<10.2f}"
        )
    txt_path = os.path.join(OUT_DIR, "focal_overlap_actual_vs_planned_stats.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines) + "\n")
    print(f"Saved plain text report to: {txt_path}")

    # Save typst table as .typ
    typst_path = os.path.join(OUT_DIR, "focal_overlap_actual_vs_planned_stats.typ")
    with open(typst_path, "w", encoding="utf-8") as f:
        f.write("\n".join(typst_table_lines) + "\n")
    print(f"Saved Typst table to: {typst_path}")

    # Save typst table as .txt for backward compatibility
    typst_txt_path = os.path.join(OUT_DIR, "focal_overlap_actual_vs_planned_typst.txt")
    with open(typst_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(typst_table_lines) + "\n")
    print(f"Saved Typst table (txt) to: {typst_txt_path}")
