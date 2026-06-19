"""
Generate per-subject anatomy figure reports.

This script processes subject NIfTI volumes to output two high-quality figures per subject:
1. structural T1w and Density report (2 rows x 3 columns)
2. sgACC target localization overlay report (1 row x 4 columns)

@author Hoai Thu Nguyen
"""

import os
import glob
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from nilearn import plotting, image
from scipy import ndimage

# ── paths ────────────────────────────────────────────────────────────────────

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR  = os.path.join(BASE, "data", "input")
OUTPUT_DIR = os.path.join(BASE, "derivatives", "anatomy")

SUBJECTS = ["sub-03", "sub-04", "sub-05", "sub-06", "sub-11"]


# ── helpers ──────────────────────────────────────────────────────────────────

def mask_centroid_mm(mask_path: str) -> np.ndarray:
    """Centre-of-mass of a binary mask in world (mm) coordinates."""
    img = nib.load(mask_path)
    vox = ndimage.center_of_mass(img.get_fdata() > 0)
    return nib.affines.apply_affine(img.affine, vox)


def add_mask_overlay(display, bin_img, color: str, alpha: float = 1.0) -> None:
    """Overlay a binary mask on a nilearn display with a solid colormap and solid visibility."""
    cmap = ListedColormap([color])
    display.add_overlay(bin_img, cmap=cmap, threshold=0.1, transparency=alpha)


def binarise(mask_path: str):
    """Load mask NIfTI and binarise it as float32 to prevent interpolation problems."""
    img = nib.load(mask_path)
    return nib.Nifti1Image((img.get_fdata() > 0).astype(np.float32), img.affine)


# ── main ─────────────────────────────────────────────────────────────────────

def make_report(sub: str) -> None:
    sub_in  = os.path.join(INPUT_DIR, sub)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t1w_path        = os.path.join(sub_in, f"{sub}_T1w_kplan.nii.gz")
    brain_mask_path = os.path.join(sub_in, f"{sub}_T1w_kplan_brain_mask.nii.gz")
    sgacc_l_path    = os.path.join(sub_in, "sgACC_BA25_L_kplan.nii.gz")
    sgacc_r_path    = os.path.join(sub_in, "sgACC_BA25_R_kplan.nii.gz")
    density_path    = os.path.join(sub_in, f"{sub}_density_kplan.nii.gz")

    for p in (t1w_path, brain_mask_path, sgacc_l_path, sgacc_r_path, density_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    # Calculate centroids directly from mask volumes
    cog_l = mask_centroid_mm(sgacc_l_path)
    cog_r = mask_centroid_mm(sgacc_r_path)
    cog   = (cog_l + cog_r) / 2.0
    x, y, z = cog

    # Hemispheric x-coordinates for sagittal cuts
    x_l = cog_l[0]
    x_r = cog_r[0]

    # Find a planned pressure map to use as reference grid for head cropping (keeps skull, crops neck)
    plan_dir = os.path.join(BASE, "data", "output", sub, "planning", "exp-focused")
    press_pattern = os.path.join(plan_dir, "*_pos-*-Pressure.nii.gz")
    press_files = glob.glob(press_pattern)
    if not press_files:
        raise FileNotFoundError(f"No planned pressure map found for {sub} in {plan_dir}")
    ref_grid_img = nib.load(press_files[0])

    # Load structural images resampled to the pressure simulation grid
    t1w_img = image.resample_to_img(nib.load(t1w_path), ref_grid_img, interpolation="continuous")
    density_img = image.resample_to_img(nib.load(density_path), ref_grid_img, interpolation="continuous")
    
    # Binarise and resample masks to the pressure simulation grid
    sgacc_l_bin = image.resample_to_img(binarise(sgacc_l_path), ref_grid_img, interpolation="nearest")
    sgacc_r_bin = image.resample_to_img(binarise(sgacc_r_path), ref_grid_img, interpolation="nearest")

    # Convert 3D world (mm) coordinates to 3D integer voxel coordinates (slice numbers) using inverse affine matrix
    inv_affine = np.linalg.inv(t1w_img.affine)
    vox_mid = np.round(nib.affines.apply_affine(inv_affine, cog)).astype(int)
    vox_l   = np.round(nib.affines.apply_affine(inv_affine, cog_l)).astype(int)
    vox_r   = np.round(nib.affines.apply_affine(inv_affine, cog_r)).astype(int)

    PLOT_KW = dict(annotate=False, draw_cross=False, black_bg=True, colorbar=False)

    # ── FILE 1: Structural Anatomy Report (T1w and Density, 2 rows × 3 columns) ──
    fig1 = plt.figure(figsize=(15, 7.5), facecolor="black")

    gs1 = gridspec.GridSpec(
        2, 3, figure=fig1,
        hspace=0.06, wspace=0.03,
        left=0.07, right=0.98, top=0.97, bottom=0.05,
    )

    col_defs = [
        ("z", z, "Axial", vox_mid[2]),
        ("x", x, "Sagittal", vox_mid[0]),
        ("y", y, "Coronal", vox_mid[1])
    ]

    # Row 0: T1w Only
    for col, (mode, coord, col_title, slice_num) in enumerate(col_defs):
        ax = fig1.add_subplot(gs1[0, col])
        d = plotting.plot_anat(t1w_img, display_mode=mode, cut_coords=[coord],
                           axes=ax, figure=fig1, **PLOT_KW)
        ax.set_title(col_title, color="white", fontsize=18, pad=6)
        if col == 0:
            ax.set_ylabel("T1w", color="white", fontsize=9, labelpad=6)
        
        # Coordinate slice number label and R/L labels on Nilearn's cut axis
        for cut_ax in d.axes.values():
            cut_ax.ax.text(0.03, 0.03, f"{mode} = {slice_num}", color="white", fontsize=12.0,
                           transform=cut_ax.ax.transAxes, va="bottom", ha="left", zorder=100,
                           bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=1.5))
            if mode in ("z", "y"):
                cut_ax.ax.text(0.03, 0.8, "L", color="white", fontsize=16.0, fontweight="bold",
                               transform=cut_ax.ax.transAxes, va="center", ha="left", zorder=100)
                cut_ax.ax.text(0.97, 0.8, "R", color="white", fontsize=16.0, fontweight="bold",
                               transform=cut_ax.ax.transAxes, va="center", ha="right", zorder=100)

    # Row 1: Density Image
    for col, (mode, coord, _, slice_num) in enumerate(col_defs):
        ax = fig1.add_subplot(gs1[1, col])
        d = plotting.plot_anat(density_img, display_mode=mode, cut_coords=[coord],
                           axes=ax, figure=fig1, **PLOT_KW)
        if col == 0:
            ax.set_ylabel("Density", color="white", fontsize=9, labelpad=6)
        
        # Coordinate slice number label and R/L labels on Nilearn's cut axis
        for cut_ax in d.axes.values():
            cut_ax.ax.text(0.03, 0.03, f"{mode} = {slice_num}", color="white", fontsize=12.0,
                           transform=cut_ax.ax.transAxes, va="bottom", ha="left", zorder=100,
                           bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=1.5))
            if mode in ("z", "y"):
                cut_ax.ax.text(0.03, 0.8, "L", color="white", fontsize=16.0, fontweight="bold",
                               transform=cut_ax.ax.transAxes, va="center", ha="left", zorder=100)
                cut_ax.ax.text(0.97, 0.8, "R", color="white", fontsize=16.0, fontweight="bold",
                               transform=cut_ax.ax.transAxes, va="center", ha="right", zorder=100)

    report_path1 = os.path.join(OUTPUT_DIR, f"{sub}_anatomy_report.png")
    fig1.savefig(report_path1, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig1)
    print(f"[saved]  {report_path1}")


    # ── FILE 2: sgACC Overlay Report (1 row × 4 columns) ──
    fig2 = plt.figure(figsize=(18, 4.5), facecolor="black")

    gs2 = gridspec.GridSpec(
        1, 4, figure=fig2,
        wspace=0.03,
        left=0.07, right=0.98, top=0.92, bottom=0.05,
    )

    overlay_defs = [
        # (display_mode, scanner_coord, label_title, slice_var_name, slice_num, show_L, show_R)
        ("x", x_l, "Left sagittal",  "x", vox_l[0],   True,  False),
        ("x", x_r, "Right sagittal", "x", vox_r[0],   False, True),
        ("z", z,   "Axial",          "z", vox_mid[2], True,  True),
        ("y", y,   "Coronal",        "y", vox_mid[1], True,  True),
    ]

    for col, (mode, coord, title, slice_var, slice_num, show_L, show_R) in enumerate(overlay_defs):
        ax = fig2.add_subplot(gs2[0, col])
        d = plotting.plot_anat(t1w_img, display_mode=mode, cut_coords=[coord],
                               axes=ax, figure=fig2, **PLOT_KW)
        
        # Bold yellow and red colors (alpha = 1.0)
        if show_L:
            add_mask_overlay(d, sgacc_l_bin, color="yellow")
        if show_R:
            add_mask_overlay(d, sgacc_r_bin, color="red")
            
        ax.set_title(title, color="white", fontsize=18, pad=6)
        
        if col == 0:
            ax.set_ylabel("T1w + sgACC\n(L=yellow  R=red)", color="white", fontsize=9, labelpad=6)
            
        # Coordinate slice number label and R/L labels on Nilearn's cut axis
        for cut_ax in d.axes.values():
            cut_ax.ax.text(0.03, 0.03, f"{slice_var} = {slice_num}", color="white", fontsize=12.0,
                           transform=cut_ax.ax.transAxes, va="bottom", ha="left", zorder=100,
                           bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=1.5))
            if mode in ("z", "y"):
                cut_ax.ax.text(0.03, 0.8, "L", color="white", fontsize=16.0, fontweight="bold",
                               transform=cut_ax.ax.transAxes, va="center", ha="left", zorder=100)
                cut_ax.ax.text(0.97, 0.8, "R", color="white", fontsize=16.0, fontweight="bold",
                               transform=cut_ax.ax.transAxes, va="center", ha="right", zorder=100)

    report_path2 = os.path.join(OUTPUT_DIR, f"{sub}_sgacc_overlay_report.png")
    fig2.savefig(report_path2, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig2)
    print(f"[saved]  {report_path2}")


if __name__ == "__main__":
    for sub in SUBJECTS:
        print(f"Processing {sub} …")
        make_report(sub)
    print("Done.")
