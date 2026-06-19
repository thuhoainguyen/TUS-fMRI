# -*- coding: utf-8 -*-
"""
planned_vs_mask_overlap.py
===========================
Analyze and visualize the overlap of the planned -3dB pressure focal zone
with the sgACC (BA25) mask for all subjects.

@author Hoai Thu Nguyen
"""

import os
import sys
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap  #//$NON-NLS-1$
from matplotlib.patches import Patch
from nilearn import image, plotting

# Enforce repo root in python path to resolve ok_plan imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))  #//$NON-NLS-1$

from ok_plan.focus import build_focus_masks_amplitude_db
from ok_plan.focus_roi import focus_roi_pressure_stats
from ok_plan.nii_utils import pa_to_isppa_w_per_cm2, roi_centroid_mm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",  #//$NON-NLS-1$
    datefmt="%H:%M:%S",  #//$NON-NLS-1$
)
log = logging.getLogger("planned_vs_mask_overlap")  #//$NON-NLS-1$

# Paths
BASE_DIR = "/Users/hoaithunguyen/Projects/Master-thesis/CITRUS"  #//$NON-NLS-1$
INPUT_BASE = os.path.join(BASE_DIR, "data", "input")  #//$NON-NLS-1$
PLAN_BASE = os.path.join(BASE_DIR, "data", "output")  #//$NON-NLS-1$
OUT_DIR = os.path.join(BASE_DIR, "derivatives", "focal_overlap")  #//$NON-NLS-1$
PLOT_DIR = os.path.join(OUT_DIR, "planned_vs_mask_overlap")  #//$NON-NLS-1$

# List of subjects to process
SUBJECTS = ["sub-03", "sub-04", "sub-05", "sub-06", "sub-11"]  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$


def find_planning_file(plan_dir: Path, side: str, map_type: str) -> Optional[Path]:
    """Search for planning NIfTI maps in directory using wildcard patterns.

    Args:
        plan_dir: Path to planning output directory for a subject.
        side: Hemisphere string ('L' or 'R').
        map_type: Map category string ('Pressure' or 'Temperature').

    Returns:
        Optional[Path]: Path to matching map file or None if not found.
    """
    if not plan_dir.exists():
        return None
    patterns = [
        f"*_{side}_pos-*-exp-{map_type}.nii.gz",  #//$NON-NLS-1$
        f"*_{side}_pos-* - exp-{map_type}.nii.gz",  #//$NON-NLS-1$
        f"*_{side}_pos-*_exp-{map_type}.nii.gz",  #//$NON-NLS-1$
    ]
    for pattern in patterns:
        files = list(plan_dir.glob(pattern))
        if files:
            return files[0]
    return None


def pca_dimensions_mm(binary_mask: np.ndarray, affine: np.ndarray) -> Tuple[float, float, float]:
    """Calculate the 3D dimensions of a binary mask along its principal components.

    Args:
        binary_mask: 3D boolean numpy array.
        affine: 4x4 affine mapping voxel coordinates to mm space.

    Returns:
        Tuple[float, float, float]: Sorted dimensions (major, middle, minor) in mm.
    """
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


def main() -> None:
    """Main execution function to run the overlap analysis and visualization.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)  #//$NON-NLS-1$

    records = []

    for sub in SUBJECTS:
        log.info("Processing subject: %s", sub)  #//$NON-NLS-1$

        input_dir = Path(INPUT_BASE) / sub
        plan_dir = Path(PLAN_BASE) / sub / "planning" / "exp-focused"  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$

        t1_path = input_dir / f"{sub}_T1w_kplan.nii.gz"  #//$NON-NLS-1$
        brain_mask_path = input_dir / f"{sub}_T1w_kplan_brain_mask.nii.gz"  #//$NON-NLS-1$

        if not t1_path.exists() or not brain_mask_path.exists():
            log.error("Missing structural data for %s", sub)  #//$NON-NLS-1$
            continue

        t1_img = nib.load(str(t1_path))
        brain_mask_img = nib.load(str(brain_mask_path))

        plot_data = {}
        skip_sub = False

        for side in ["L", "R"]:  #//$NON-NLS-1$  #//$NON-NLS-1$
            side_name = "Left" if side == "L" else "Right"  #//$NON-NLS-1$  #//$NON-NLS-1$

            roi_l_path = input_dir / f"sgACC_BA25_L_kplan.nii.gz"  #//$NON-NLS-1$
            roi_r_path = input_dir / f"sgACC_BA25_R_kplan.nii.gz"  #//$NON-NLS-1$
            if not roi_l_path.exists() or not roi_r_path.exists():
                log.warning("sgACC L/R ROI masks not found under: %s", input_dir)  #//$NON-NLS-1$
                skip_sub = True
                break
            roi_l_img = nib.load(str(roi_l_path))
            roi_r_img = nib.load(str(roi_r_path))

            pressure_path = find_planning_file(plan_dir, side, "Pressure")  #//$NON-NLS-1$
            temp_path = find_planning_file(plan_dir, side, "Temperature")  #//$NON-NLS-1$

            if pressure_path is None or temp_path is None:
                log.warning("Missing pressure/temperature map for %s on %s side", sub, side_name)  #//$NON-NLS-1$
                skip_sub = True
                break

            pressure_img = nib.load(str(pressure_path))
            temp_img = nib.load(str(temp_path))

            # Resample structures to pressure voxel grid
            roi_l_res = image.resample_to_img(roi_l_img, pressure_img, interpolation="nearest")  #//$NON-NLS-1$
            roi_r_res = image.resample_to_img(roi_r_img, pressure_img, interpolation="nearest")  #//$NON-NLS-1$
            t1_res = image.resample_to_img(t1_img, pressure_img, interpolation="continuous")  #//$NON-NLS-1$
            brain_res = image.resample_to_img(brain_mask_img, pressure_img, interpolation="nearest")  #//$NON-NLS-1$
            temp_res = image.resample_to_img(temp_img, pressure_img, interpolation="continuous")  #//$NON-NLS-1$

            # Erode brain mask
            mask_data = brain_res.get_fdata() > 0.5
            eroded_data = ndimage.binary_erosion(mask_data, iterations=3).astype(np.float32)
            eroded_img = nib.Nifti1Image(eroded_data, pressure_img.affine, pressure_img.header)

            # Use ipsilateral centroid for cropping cuts reference
            roi_ipsi_res = roi_l_res if side == "L" else roi_r_res  #//$NON-NLS-1$
            sgacc_centroid = roi_centroid_mm(roi_ipsi_res)

            # Get planned -3dB focal zone mask
            focus_3db_img, _ = build_focus_masks_amplitude_db(pressure_img, eroded_img)

            # PCA FWHM of the focus mask
            focus_3db_data = focus_3db_img.get_fdata() > 0.5
            dims = pca_dimensions_mm(focus_3db_data, pressure_img.affine)
            axial_fwhm = dims[0]
            lateral_fwhm = (dims[1] + dims[2]) / 2.0
            focal_centroid = roi_centroid_mm(focus_3db_img)

            # Match target mask to the focal zone hemisphere
            target_side = side
            target_name = "Left" if target_side == "L" else "Right"  #//$NON-NLS-1$  #//$NON-NLS-1$
            roi_res = roi_l_res if target_side == "L" else roi_r_res  #//$NON-NLS-1$

            # Compute stats
            stats = focus_roi_pressure_stats(pressure_img, roi_res, focus_3db_img, label="-3 dB focus")  #//$NON-NLS-1$

            # Peak pressure and Isppa AT the sgACC target mask
            # (matches kplan PDF: kplan reports Isppa at the planned focus/target point,
            #  not the maximum along the beam path which includes proximal side-lobes)
            press_data = np.abs(pressure_img.get_fdata())
            roi_data = roi_res.get_fdata() > 0.5
            pmax_target_pa = float(np.max(press_data[roi_data])) if np.any(roi_data) else 0.0
            peak_p_brain_mpa = pmax_target_pa / 1e6
            isppa_max = pa_to_isppa_w_per_cm2(pmax_target_pa)

            temp_data = temp_res.get_fdata()
            peak_t_brain = float(np.max(temp_data[roi_data])) if np.any(roi_data) else 37.0

            # Centroid shift: closest focal zone component to sgACC centroid vs sgACC centroid
            com_roi = ndimage.center_of_mass(roi_data.astype(float))
            com_roi_mm = nib.affines.apply_affine(pressure_img.affine, np.array(com_roi))
            labeled_focus, n_comp = ndimage.label(focus_3db_data)
            if n_comp == 0:
                centroid_shift = float("nan")
            elif n_comp == 1:
                com_focus = ndimage.center_of_mass(focus_3db_data.astype(float))
                com_focus_mm = nib.affines.apply_affine(pressure_img.affine, np.array(com_focus))
                centroid_shift = float(np.linalg.norm(com_focus_mm - com_roi_mm))
            else:
                best_dist = np.inf
                for ci in range(1, n_comp + 1):
                    comp = labeled_focus == ci
                    com_c = ndimage.center_of_mass(comp.astype(float))
                    com_c_mm = nib.affines.apply_affine(pressure_img.affine, np.array(com_c))
                    d = float(np.linalg.norm(com_c_mm - com_roi_mm))
                    if d < best_dist:
                        best_dist = d
                centroid_shift = best_dist

            # Dice overlap coefficient
            dice_coeff = 2.0 * stats.n_overlap_voxels / (stats.n_focus_voxels + stats.n_roi_voxels + 1e-9)

            # Percentage of focal volume within sgACC
            pct_focus_in_sgacc = stats.on_target_pct

            records.append({
                "subject": sub,  #//$NON-NLS-1$
                "focus_hemisphere": side_name,  #//$NON-NLS-1$
                "peak_pressure_mpa": peak_p_brain_mpa,  #//$NON-NLS-1$
                "isppa_max": isppa_max,  #//$NON-NLS-1$
                "peak_temp_c": peak_t_brain,  #//$NON-NLS-1$
                "axial_fwhm": axial_fwhm,  #//$NON-NLS-1$
                "lateral_fwhm": lateral_fwhm,  #//$NON-NLS-1$
                "dice": dice_coeff,  #//$NON-NLS-1$
                "pct_focus_in_sgacc": pct_focus_in_sgacc,  #//$NON-NLS-1$
                "centroid_shift": centroid_shift,  #//$NON-NLS-1$
            })

            # Save data for plotting
            t1_brain_data = t1_res.get_fdata() * eroded_data
            t1_brain_img = nib.Nifti1Image(t1_brain_data, pressure_img.affine, pressure_img.header)
            t1_brain_img_cropped = image.crop_img(t1_brain_img)

            plot_data[side] = {
                "t1_brain": t1_brain_img_cropped,
                "roi_l_res": roi_l_res,
                "roi_r_res": roi_r_res,
                "focus_3db": focus_3db_img,
                "centroid": sgacc_centroid
            }

        if skip_sub:
            log.warning("Skipping visualization for subject %s due to missing files", sub)  #//$NON-NLS-1$
            continue

        # Draw Figure
        fig = plt.figure(figsize=(15, 10), facecolor="black")  #//$NON-NLS-1$
        fig.suptitle(
            f"{sub} - Planned Focus vs. sgACC Mask Overlap\nRow 1: Left planned  |  Row 2: Right planned",  #//$NON-NLS-1$
            color="white", fontsize=14, fontweight="bold", y=0.97  #//$NON-NLS-1$  #//$NON-NLS-1$
        )

        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.02, wspace=0.01, left=0.03, right=0.97, top=0.88, bottom=0.08)
        views = [
            {"mode": "x", "title": "Sagittal", "index": 0},  #//$NON-NLS-1$  #//$NON-NLS-1$
            {"mode": "z", "title": "Axial", "index": 2},  #//$NON-NLS-1$  #//$NON-NLS-1$
            {"mode": "y", "title": "Coronal", "index": 1}  #//$NON-NLS-1$  #//$NON-NLS-1$
        ]

        PLOT_KW = dict(annotate=False, draw_cross=False, black_bg=True, colorbar=False)  #//$NON-NLS-1$

        cmap_3db = ListedColormap(["yellow"])  #//$NON-NLS-1$

        for row_idx, side in enumerate(["L", "R"]):  #//$NON-NLS-1$  #//$NON-NLS-1$
            side_data = plot_data[side]
            centroid = side_data["centroid"]
            t1_brain = side_data["t1_brain"]
            roi_l_res = side_data["roi_l_res"]
            roi_r_res = side_data["roi_r_res"]
            focus_3db = side_data["focus_3db"]
            side_label = "Left" if side == "L" else "Right"  #//$NON-NLS-1$  #//$NON-NLS-1$

            for col_idx, view in enumerate(views):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                mode = view["mode"]
                coord = centroid[view["index"]]

                # Plot T1w
                d = plotting.plot_anat(t1_brain, display_mode=mode, cut_coords=[coord], axes=ax, figure=fig, **PLOT_KW)
                # Fill in color for -3db focal zone (yellow overlay)
                d.add_overlay(focus_3db, cmap=cmap_3db, threshold=0.5, transparency=0.60, colorbar=False)
                # sgACC contours (both Left and Right)
                d.add_contours(roi_l_res, levels=[0.5], colors=["white"], linewidths=1.2)  #//$NON-NLS-1$
                d.add_contours(roi_r_res, levels=[0.5], colors=["white"], linewidths=1.2)  #//$NON-NLS-1$
                # -3dB focal zone contour (yellow)
                d.add_contours(focus_3db, levels=[0.5], colors=["yellow"], linewidths=1.2)  #//$NON-NLS-1$

                # kplan x-axis increases to the RIGHT — nilearn renders correctly without inversion.
                # Labels follow neurological convention: L on left, R on right.
                for cut_ax in d.axes.values():
                    for text_artist in list(cut_ax.ax.texts):
                        text_artist.remove()
                    if mode in ["z", "y"]:  #//$NON-NLS-1$  #//$NON-NLS-1$
                        cut_ax.ax.text(0.04, 0.90, "L", color="white", fontsize=16.0, fontweight="bold", transform=cut_ax.ax.transAxes, va="top", ha="left")  #//$NON-NLS-1$  #//$NON-NLS-1$
                        cut_ax.ax.text(0.96, 0.90, "R", color="white", fontsize=16.0, fontweight="bold", transform=cut_ax.ax.transAxes, va="top", ha="right")  #//$NON-NLS-1$  #//$NON-NLS-1$

                if row_idx == 0:
                    ax.set_title(view["title"], color="white", fontsize=16.0, pad=6)  #//$NON-NLS-1$
                if col_idx == 0:
                    ax.set_ylabel(f"{side_label} planned", color="white", fontsize=16.0, labelpad=8)  #//$NON-NLS-1$  #//$NON-NLS-1$

        # Custom legend elements
        legend_elements = [
            Patch(facecolor="none", edgecolor="white", linewidth=1.2, label="sgACC (BA25) Masks"),  #//$NON-NLS-1$  #//$NON-NLS-1$
            Patch(facecolor="yellow", edgecolor="yellow", alpha=0.6, linewidth=1.2, label="-3 dB Focal Zone"),  #//$NON-NLS-1$  #//$NON-NLS-1$
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=2, frameon=False, fontsize=16.0, labelcolor="white")  #//$NON-NLS-1$  #//$NON-NLS-1$

        out_fig_path = os.path.join(PLOT_DIR, f"{sub}_planned_vs_mask_overlap.png")  #//$NON-NLS-1$
        fig.savefig(out_fig_path, dpi=150, bbox_inches="tight", facecolor="black")  #//$NON-NLS-1$  #//$NON-NLS-1$
        plt.close(fig)
        log.info("Saved overlap figure to: %s", out_fig_path)  #//$NON-NLS-1$

    # Write report files
    md_table_lines = [
        "# Planned Focus vs. sgACC (BA25) Mask Overlap Statistics",
        "",
        "| Subject | Hemisphere | Peak Pressure (MPa) | Isppa Max (W/cm²) | Peak Temp (°C) | Focal Spot FWHM (mm) | Dice Overlap | Focal Vol in target (%) | Centroid Shift (mm) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    typst_table_lines = [
        "// Planned Focus vs. sgACC (BA25) Mask Overlap Statistics Table",
        "#table(",
        "  columns: (auto, auto, auto, auto, auto, auto),",
        "  align: horizon + center,",
        "  fill: (x, y) => if y == 0 { rgb(\"e0e0e0\") } else if calc.even(y) { rgb(\"f9f9f9\") } else { rgb(\"ffffff\") },",
        "  [*Subject*], [*Target*], [*Focal Spot FWHM (mm)*], [*Dice Coefficient*], [*Focal Vol in target (%)*], [*Centroid Shift (mm)*],"
    ]

    # Print header to terminal
    print("\n" + "="*105)
    print(" PLANNED FOCUS VS. sgACC MASK OVERLAP STATISTICS")
    print("="*105)
    print(f"{'Subject':<8} | {'Hemisphere':<10} | {'P max (MPa)':<12} | {'Isppa max':<10} | {'T max (°C)':<10} | {'FWHM (mm)':<12} | {'Dice':<6} | {'Vol in target':<14} | {'Shift (mm)':<10}")
    print("-"*105)

    for r in records:
        fwhm_str = f"{r['axial_fwhm']:.1f} × {r['lateral_fwhm']:.1f}"
        print(f"{r['subject']:<8} | {r['focus_hemisphere']:<10} | {r['peak_pressure_mpa']:<12.4f} | {r['isppa_max']:<10.2f} | {r['peak_temp_c']:<10.2f} | {fwhm_str:<12} | {r['dice']:<6.3f} | {r['pct_focus_in_sgacc']:<13.1f}% | {r['centroid_shift']:<10.2f}")

        md_table_lines.append(
            f"| {r['subject']} | {r['focus_hemisphere']} | {r['peak_pressure_mpa']:.4f} | {r['isppa_max']:.2f} | {r['peak_temp_c']:.2f} | {fwhm_str} | {r['dice']:.3f} | {r['pct_focus_in_sgacc']:.1f}% | {r['centroid_shift']:.2f} |"
        )

        typst_table_lines.append(
            f"  [{r['subject']}], [{r['focus_hemisphere']}], [{fwhm_str}], [{r['dice']:.3f}], [{r['pct_focus_in_sgacc']:.1f}%], [{r['centroid_shift']:.2f}],"
        )

    typst_table_lines.append(")")

    # Save md table inside the subfolder planned_vs_mask_overlap
    md_path = os.path.join(PLOT_DIR, "planned_vs_mask_overlap_stats.md")  #//$NON-NLS-1$
    with open(md_path, "w", encoding="utf-8") as f:  #//$NON-NLS-1$  #//$NON-NLS-1$
        f.write("\n".join(md_table_lines) + "\n")  #//$NON-NLS-1$
    log.info("Saved Markdown table to: %s", md_path)  #//$NON-NLS-1$

    # Save plain text report
    txt_lines = [
        "===============================================================================================",
        " PLANNED FOCUS VS. sgACC MASK OVERLAP STATISTICS",
        " P max / Isppa / T max = max inside sgACC target mask (matches kplan PDF focal-point values)",
        "===============================================================================================",
        f"{'Subject':<8} | {'Hemisphere':<10} | {'P max (MPa)':<12} | {'Isppa max':<10} | {'T max (°C)':<10} | {'FWHM (mm)':<12} | {'Dice':<6} | {'Vol in target':<14}",
        "-----------------------------------------------------------------------------------------------"
    ]
    for r in records:
        fwhm_str = f"{r['axial_fwhm']:.1f} x {r['lateral_fwhm']:.1f}"
        txt_lines.append(
            f"{r['subject']:<8} | {r['focus_hemisphere']:<10} | {r['peak_pressure_mpa']:<12.4f} | {r['isppa_max']:<10.2f} | {r['peak_temp_c']:<10.2f} | {fwhm_str:<12} | {r['dice']:<6.3f} | {r['pct_focus_in_sgacc']:<13.1f}%"
        )
    txt_path = os.path.join(PLOT_DIR, "planned_vs_mask_overlap_stats.txt")  #//$NON-NLS-1$
    with open(txt_path, "w", encoding="utf-8") as f:  #//$NON-NLS-1$  #//$NON-NLS-1$
        f.write("\n".join(txt_lines) + "\n")  #//$NON-NLS-1$
    log.info("Saved plain text report to: %s", txt_path)  #//$NON-NLS-1$

    # Save typst table as .typ
    typst_path = os.path.join(PLOT_DIR, "planned_vs_mask_overlap_stats.typ")  #//$NON-NLS-1$
    with open(typst_path, "w", encoding="utf-8") as f:  #//$NON-NLS-1$  #//$NON-NLS-1$
        f.write("\n".join(typst_table_lines) + "\n")  #//$NON-NLS-1$
    log.info("Saved Typst table to: %s", typst_path)  #//$NON-NLS-1$

    # Save typst table as .txt for backward compatibility
    typst_txt_path = os.path.join(PLOT_DIR, "planned_vs_mask_overlap_typst.txt")  #//$NON-NLS-1$
    with open(typst_txt_path, "w", encoding="utf-8") as f:  #//$NON-NLS-1$  #//$NON-NLS-1$
        f.write("\n".join(typst_table_lines) + "\n")  #//$NON-NLS-1$
    log.info("Saved Typst table (txt) to: %s", typst_txt_path)  #//$NON-NLS-1$


if __name__ == "__main__":
    main()
