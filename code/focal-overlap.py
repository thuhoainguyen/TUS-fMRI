"""
focal-overlap.py

Visualize the overlap of planned and actual (post-hoc) -3dB focal zones
on the sgACC target region for each subject.

Layout per subject (1 row x 4 columns):
  Left sagittal | Right sagittal | Axial | Coronal

Overlays:
  - sgACC L & R: white contour outline only
  - Left planned -3dB zone:       yellow  (pressure gradient)
  - Left actual  -3dB zone:       red     (pressure gradient)
  - Left overlap (plan ∩ actual): orange  (flat)
  - Right planned -3dB zone:      pink    (pressure gradient)
  - Right actual  -3dB zone:      purple  (pressure gradient)
  - Right overlap (plan ∩ actual):magenta (flat)

Output: derivatives/focal_overlap/{sub}_focal_overlap_report.png

@author Hoai Thu Nguyen
"""

import os
import glob
import subprocess
import numpy as np
import nibabel as nib
from nilearn import plotting, image
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

# ── paths ─────────────────────────────────────────────────────────────────────

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR  = os.path.join(BASE, "data", "input")
OUTPUT_DIR = os.path.join(BASE, "data", "output")
DERIV_DIR  = os.path.join(BASE, "derivatives", "focal_overlap")

SUBJECTS = ["sub-03", "sub-04", "sub-05", "sub-06", "sub-11"]

DB3_FACTOR = 10 ** (-3 / 20)   # ≈ 0.7079  (pressure amplitude -3 dB)

# ── colormaps ─────────────────────────────────────────────────────────────────

# Single-hue gradients: dark → bright, for showing pressure intensity within zone
CMAPS = {
    "plan_L":   LinearSegmentedColormap.from_list("yellow_cm",
                    [(0.35, 0.35, 0.0, 1.0), (1.0, 1.0, 0.0, 1.0)]),
    "actual_L": LinearSegmentedColormap.from_list("red_cm",
                    [(0.40, 0.0,  0.0, 1.0), (1.0, 0.0, 0.0, 1.0)]),
    "plan_R":   LinearSegmentedColormap.from_list("pink_cm",
                    [(0.55, 0.20, 0.30, 1.0), (1.0, 0.70, 0.80, 1.0)]),
    "actual_R": LinearSegmentedColormap.from_list("purple_cm",
                    [(0.20, 0.0,  0.40, 1.0), (0.65, 0.0, 1.0,  1.0)]),
    "turbo":    plt.get_cmap("turbo"),
}

LEGEND_ENTRIES = [
    ("Pressure beam (turbo, 30%)", "#888888"),  # placeholder, shown as patch below
    ("Left planned",               "#FFFF00"),
    ("Left post-hoc",              "#FF0000"),
    ("Left overlap (blended)",     "#FF8800"),
    ("Right planned",              "#FF99BB"),
    ("Right post-hoc",             "#9900FF"),
    ("Right overlap (blended)",    "#CC44CC"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def find_pressure(folder: str, side: str) -> str | None:
    """Glob for any Pressure file for the given side in the given folder."""
    patterns = [
        os.path.join(folder, f"*_{side}_pos-*-Pressure.nii.gz"),
        os.path.join(folder, f"*_{side}_pos-* - Pressure.nii.gz"),
    ]
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return sorted(files)[0]
    return None


def ensure_brain_mask(sub: str, sub_in: str) -> str:
    """Return path to brain mask; run BET automatically if missing."""
    mask_path = os.path.join(sub_in, f"{sub}_T1w_kplan_brain_mask.nii.gz")
    if os.path.exists(mask_path):
        return mask_path

    t1w_path   = os.path.join(sub_in, f"{sub}_T1w_kplan.nii.gz")
    brain_base = os.path.join(sub_in, f"{sub}_T1w_kplan_brain")
    print(f"    [BET] Brain mask missing for {sub} — running BET...")
    subprocess.run(["bet", t1w_path, brain_base, "-m", "-f", "0.5"],
                   check=True)
    print(f"    [BET] Done → {mask_path}")
    return mask_path


def compute_focal_zone(pressure_path: str, brain_mask_path: str):
    """
    Compute the -3 dB focal zone for a pressure map.

    Steps:
        1. Resample brain mask to pressure map space (nearest-neighbour).
        2. Erode brain mask 3x to remove skull-adjacent voxels.
        3. Mask pressure to eroded brain → pressure_eroded (full beam in brain).
        4. Find peak pressure → -3 dB threshold = peak x 10^(-3/20).
        5. Keep only voxels ≥ threshold → zone_img (focal zone with pressure values).

    Returns:
        zone_img          : NIfTI1Image – pressure values ≥ -3dB threshold, 0 elsewhere
        threshold         : float       – the -3 dB threshold value (Pa)
        binary_img        : NIfTI1Image – binary mask of the zone
        pressure_eroded   : NIfTI1Image – full pressure masked to eroded brain (for Layer 1)
        None, 0, None, None if pressure map is empty.
    """
    pressure_img   = nib.load(pressure_path)
    brain_mask_img = nib.load(brain_mask_path)

    # Resample brain mask to pressure voxel grid
    brain_res = image.resample_to_img(
        brain_mask_img, pressure_img, interpolation="nearest"
    )

    # Erode 3x to pull away from skull boundary (~2.7 mm at 0.9 mm/voxel)
    mask_data = brain_res.get_fdata() > 0.5
    eroded    = ndimage.binary_erosion(mask_data, iterations=3).astype(np.float32)

    # Apply eroded mask to pressure
    pressure_data  = pressure_img.get_fdata().astype(np.float32)
    pressure_brain = pressure_data * eroded

    max_val = float(pressure_brain.max())
    if max_val == 0:
        print("    [WARN] Pressure map empty after masking — skipping.")
        return None, 0.0, None, None

    threshold  = max_val * DB3_FACTOR
    zone_data  = np.where(pressure_brain >= threshold, pressure_brain, 0.0)
    binary_data = (zone_data > 0).astype(np.float32)

    affine, hdr       = pressure_img.affine, pressure_img.header
    zone_img          = nib.Nifti1Image(zone_data,    affine, hdr)
    binary_img        = nib.Nifti1Image(binary_data,  affine, hdr)
    pressure_eroded   = nib.Nifti1Image(pressure_brain, affine, hdr)

    return zone_img, threshold, binary_img, pressure_eroded


def compute_overlap_stats(binary_a, binary_b):
    """
    Compute Dice and overlap % between two binary NIfTI images.
    Resamples binary_b to binary_a's space.
    Returns dict of stats.
    """
    b_res    = image.resample_to_img(binary_b, binary_a, interpolation="nearest").get_fdata() > 0.5
    a_bin    = binary_a.get_fdata() > 0.5
    intersect = (a_bin & b_res).sum()
    dice      = 2 * intersect / (a_bin.sum() + b_res.sum() + 1e-9)
    return dict(
        dice            = float(dice),
        pct_plan_in_act = float(intersect / (a_bin.sum()  + 1e-9) * 100),
        pct_act_in_plan = float(intersect / (b_res.sum()  + 1e-9) * 100),
    )


def mask_centroid_mm(mask_path: str) -> np.ndarray:
    """Centre-of-mass of a binary mask in world (mm) coordinates."""
    img = nib.load(mask_path)
    vox = ndimage.center_of_mass(img.get_fdata() > 0)
    return nib.affines.apply_affine(img.affine, vox)


def add_zone_overlay(display, zone_img, cmap, threshold, alpha=0.85):
    """
    Overlay a pressure zone image with gradient colormap.
    vmin=threshold so the colormap spans from -3dB threshold → max,
    showing the full pressure gradient within the zone.
    """
    if zone_img is None:
        return
    max_val = float(zone_img.get_fdata().max())
    if max_val == 0:
        return
    # vmin=50% of peak so colormap spans 50%→100% of max pressure
    # threshold stays at -3dB (~70.7%) — voxels below it remain transparent
    display.add_overlay(zone_img, cmap=cmap, threshold=threshold,
                        vmin=max_val * 0.5, vmax=max_val,
                        alpha=alpha, colorbar=False)


def add_overlap_overlay(display, overlap_img, cmap, alpha=1.0):
    """Overlay a binary overlap mask with a flat color."""
    if overlap_img is None:
        return
    display.add_overlay(overlap_img, cmap=cmap, threshold=0.5,
                        alpha=alpha, colorbar=False)


# ── main ──────────────────────────────────────────────────────────────────────

def make_report(sub: str, plot: bool = True) -> dict:
    print(f"\n{'=' * 55}")
    print(f"  {sub}")
    print(f"{'=' * 55}")

    sub_in   = os.path.join(INPUT_DIR,  sub)
    plan_dir = os.path.join(OUTPUT_DIR, sub, "planning", "exp-focused")
    post_dir = os.path.join(OUTPUT_DIR, sub, "posthoc",  "exp-focused")
    os.makedirs(DERIV_DIR, exist_ok=True)

    # ── file paths ──
    t1w_path        = os.path.join(sub_in, f"{sub}_T1w_kplan.nii.gz")
    sgacc_l_path    = os.path.join(sub_in, "sgACC_BA25_L_kplan.nii.gz")
    sgacc_r_path    = os.path.join(sub_in, "sgACC_BA25_R_kplan.nii.gz")
    brain_mask_path = ensure_brain_mask(sub, sub_in)

    plan_l_path   = find_pressure(plan_dir, "L")
    plan_r_path   = find_pressure(plan_dir, "R")
    actual_l_path = find_pressure(post_dir, "L")
    actual_r_path = find_pressure(post_dir, "R")

    for label, path in [("Planned  L", plan_l_path),  ("Planned  R", plan_r_path),
                         ("Actual   L", actual_l_path), ("Actual   R", actual_r_path)]:
        status = os.path.basename(path) if path else "NOT FOUND"
        print(f"  {label}: {status}")

    # ── compute -3 dB focal zones ──
    print("  Computing -3 dB focal zones …")

    zones      = {}
    thresholds = {}
    binaries   = {}

    pressure_eroded = {}  # full beam in eroded brain, for Layer 1

    for key, path in [("plan_L",   plan_l_path),
                      ("actual_L", actual_l_path),
                      ("plan_R",   plan_r_path),
                      ("actual_R", actual_r_path)]:
        if path:
            z, thr, b, p_ero = compute_focal_zone(path, brain_mask_path)
            zones[key]           = z
            thresholds[key]      = thr
            binaries[key]        = b
            pressure_eroded[key] = p_ero
            if thr > 0:
                print(f"    {key}: peak={thr / DB3_FACTOR:,.0f} Pa  "
                      f"→ -3 dB threshold={thr:,.0f} Pa")
        else:
            zones[key] = thresholds[key] = binaries[key] = None
            pressure_eroded[key] = None

    # ── overlap stats (for table only — no split needed for display) ──
    print("    Computing overlap stats …")

    # ── slice coordinates ──
    cog_l   = mask_centroid_mm(sgacc_l_path)
    cog_r   = mask_centroid_mm(sgacc_r_path)
    cog_mid = (cog_l + cog_r) / 2.0

    t1w_img    = nib.load(t1w_path)
    inv_aff    = np.linalg.inv(t1w_img.affine)
    vox_l      = np.round(nib.affines.apply_affine(inv_aff, cog_l)).astype(int)
    vox_r      = np.round(nib.affines.apply_affine(inv_aff, cog_r)).astype(int)
    vox_mid    = np.round(nib.affines.apply_affine(inv_aff, cog_mid)).astype(int)

    sgacc_l_img = nib.load(sgacc_l_path)
    sgacc_r_img = nib.load(sgacc_r_path)

    # ── compute stats for text box ──
    stats = {}
    for side, key_plan, key_actual in [("L", "plan_L", "actual_L"),
                                        ("R", "plan_R", "actual_R")]:
        b_plan   = binaries.get(key_plan)
        b_actual = binaries.get(key_actual)
        if b_plan is not None and b_actual is not None:
            ovlp = compute_overlap_stats(b_plan, b_actual)
            stats[side] = dict(
                plan_max        = thresholds.get(key_plan,   0) / DB3_FACTOR,
                plan_thr        = thresholds.get(key_plan,   0),
                act_max         = thresholds.get(key_actual, 0) / DB3_FACTOR,
                act_thr         = thresholds.get(key_actual, 0),
                dice            = ovlp["dice"],
                pct_plan_in_act = ovlp["pct_plan_in_act"],
                pct_act_in_plan = ovlp["pct_act_in_plan"],
            )

    if not plot:
        return stats

    # ── brain-extracted T1w for background (skull stripped, crops to brain) ──
    brain_mask_t1w = nib.load(brain_mask_path)
    # Resample brain mask to T1w space if needed
    brain_mask_t1w_res = image.resample_to_img(
        brain_mask_t1w, t1w_img, interpolation="nearest"
    )
    t1w_data    = t1w_img.get_fdata().astype(np.float32)
    mask_data_t1w = brain_mask_t1w_res.get_fdata() > 0.5
    t1w_brain   = t1w_data * mask_data_t1w
    t1w_brain_img = nib.Nifti1Image(t1w_brain, t1w_img.affine, t1w_img.header)
    t1w_brain_img = image.crop_img(t1w_brain_img)

    # ── figure: 1 row × 4 cols + stats + legend ──
    fig = plt.figure(figsize=(20, 9), facecolor="black")
    fig.suptitle(
        f"{sub}  –  -3 dB Focal Zone Overlap Report  (exp-focused)",
        color="white", fontsize=13, fontweight="bold", y=1.0,
    )

    # Layout: panels (top) | stats table (middle) | legend (bottom)
    outer_gs = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[3.8, 1.0, 0.15],
        hspace=0.05,
        left=0.03, right=0.97, top=0.96, bottom=0.01,
    )
    gs = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer_gs[0],
        wspace=0.01,
    )

    panels = [
        ("x", cog_l[0],   "Left sagittal",  "x", vox_l[0],   True,  False, False),
        ("x", cog_r[0],   "Right sagittal", "x", vox_r[0],   False, True,  False),
        ("z", cog_mid[2], "Axial",          "z", vox_mid[2], True,  True,  True),
        ("y", cog_mid[1], "Coronal",        "y", vox_mid[1], True,  True,  True),
    ]

    PLOT_KW = dict(annotate=False, draw_cross=False, black_bg=True, colorbar=False)

    for col, (mode, coord, title, sv, sn, show_L, show_R, flip) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])

        # Use brain-extracted T1w as background — no skull, focuses on brain
        d = plotting.plot_anat(t1w_brain_img, display_mode=mode, cut_coords=[coord],
                               axes=ax, figure=fig, **PLOT_KW)

        # ── Layer 1: T1w brain extracted (background anatomy only) ──
        # pressure_brain_eroded removed — keeps display clean

        # ── Layer 2: -3dB focal zones ──
        # Planned drawn first, actual on top — alpha compositing blends naturally
        # in overlap regions (yellow+red→orange, pink+purple→magenta)
        if show_L:
            add_zone_overlay(d, zones.get("plan_L"),   CMAPS["plan_L"],
                             thresholds.get("plan_L",   0), alpha=0.80)
            add_zone_overlay(d, zones.get("actual_L"), CMAPS["actual_L"],
                             thresholds.get("actual_L", 0), alpha=0.65)
        if show_R:
            add_zone_overlay(d, zones.get("plan_R"),   CMAPS["plan_R"],
                             thresholds.get("plan_R",   0), alpha=0.80)
            add_zone_overlay(d, zones.get("actual_R"), CMAPS["actual_R"],
                             thresholds.get("actual_R", 0), alpha=0.65)

        # sgACC contours
        d.add_contours(sgacc_l_img, levels=[0.5], colors=["white"], linewidths=0.9)
        d.add_contours(sgacc_r_img, levels=[0.5], colors=["white"], linewidths=0.9)

        # Column title — close to image, small pad
        ax.set_title(title, color="white", fontsize=16.0, pad=3)

        # Slice label: placed on the subplot ax (not cut_ax) for consistent height
        ax.text(0.03, 0.03, f"{sv} = {sn}",
                color="white", fontsize=12.0, transform=ax.transAxes,
                va="bottom", ha="left", zorder=100,
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=1.5))

        # L/R labels on axial & coronal only
        for cut_ax in d.axes.values():
            if flip:
                cut_ax.ax.invert_xaxis()
                cut_ax.ax.text(0.03, 0.88, "L", color="white", fontsize=16.0,
                               fontweight="bold", transform=cut_ax.ax.transAxes,
                               va="top", ha="left", zorder=100)
                cut_ax.ax.text(0.97, 0.88, "R", color="white", fontsize=16.0,
                               fontweight="bold", transform=cut_ax.ax.transAxes,
                               va="top", ha="right", zorder=100)
            elif title == "Left sagittal":
                cut_ax.ax.invert_xaxis()

    # ── stats table ──
    ax_stats = fig.add_subplot(outer_gs[1])
    ax_stats.axis("off")

    def fmt(v): return f"{v:,.0f}"
    sl = stats.get("L", {})
    sr = stats.get("R", {})

    col_labels = ["", "Left planned", "Left post-hoc", "Left overlap",
                      "Right planned", "Right post-hoc", "Right overlap"]
    rows_data  = [
        ["Max pressure (Pa)",
         fmt(sl.get("plan_max", 0)), fmt(sl.get("act_max", 0)), "—",
         fmt(sr.get("plan_max", 0)), fmt(sr.get("act_max", 0)), "—"],
        ["-3 dB threshold (Pa)",
         fmt(sl.get("plan_thr", 0)), fmt(sl.get("act_thr", 0)), "—",
         fmt(sr.get("plan_thr", 0)), fmt(sr.get("act_thr", 0)), "—"],
        ["Dice coefficient",
         "—", "—", f"{sl.get('dice', 0):.3f}",
         "—", "—", f"{sr.get('dice', 0):.3f}"],
        ["% planned in post-hoc",
         "—", "—", f"{sl.get('pct_plan_in_act', 0):.1f}%",
         "—", "—", f"{sr.get('pct_plan_in_act', 0):.1f}%"],
        ["% post-hoc in planned",
         "—", "—", f"{sl.get('pct_act_in_plan', 0):.1f}%",
         "—", "—", f"{sr.get('pct_act_in_plan', 0):.1f}%"],
    ]

    tbl = ax_stats.table(
        cellText=rows_data, colLabels=col_labels,
        cellLoc="center", loc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(14.0)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#333333")
        cell.set_facecolor("#111111")
        cell.set_text_props(color="white",
                            fontweight="bold" if r == 0 else "normal")

    # ── legend: in its own row below table, no overlap ──
    ax_leg = fig.add_subplot(outer_gs[2])
    ax_leg.axis("off")
    handles = [
        Patch(facecolor="#FFFF00", edgecolor="none", label="Left planned"),
        Patch(facecolor="#FF0000", edgecolor="none", label="Left post-hoc"),
        Patch(facecolor="#FF8800", edgecolor="none", label="Left overlap"),
        Patch(facecolor="#FF99BB", edgecolor="none", label="Right planned"),
        Patch(facecolor="#9900FF", edgecolor="none", label="Right post-hoc"),
        Patch(facecolor="#CC44CC", edgecolor="none", label="Right overlap"),
        Patch(facecolor="none", edgecolor="white", linewidth=1.2,
              label="sgACC L & R (contour)"),
    ]
    ax_leg.legend(
        handles=handles, loc="center", ncol=len(handles),
        frameon=False, fontsize=16.0, labelcolor="white",
    )

    out_path = os.path.join(DERIV_DIR, f"{sub}_focal_overlap_report.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  [saved] {out_path}")
    return stats


if __name__ == "__main__":
    all_stats = {}  #//$NON-NLS-1$
    for sub in SUBJECTS:
        stats = make_report(sub, plot=True)
        all_stats[sub] = stats

    # Save summary tables to derivatives/tables/
    tables_dir = os.path.join(BASE, "derivatives", "tables")  #//$NON-NLS-1$ #//$NON-NLS-2$
    os.makedirs(tables_dir, exist_ok=True)

    # 1. Plain text summary
    txt_path = os.path.join(tables_dir, "focal_overlap_summary.txt")  #//$NON-NLS-1$
    with open(txt_path, "w") as f:  #//$NON-NLS-1$
        f.write("Subject\tHemisphere\tP_max_plan(Pa)\tP_max_actual(Pa)\tDice_Coef\tPct_Plan_in_Act\tPct_Act_in_Plan\n")  #//$NON-NLS-1$
        for sub in SUBJECTS:
            for side in ["L", "R"]:  #//$NON-NLS-1$ #//$NON-NLS-2$
                s = all_stats[sub].get(side, {})
                if not s:
                    continue
                hemi = "Left" if side == "L" else "Right"  #//$NON-NLS-1$ #//$NON-NLS-2$
                p_plan = s.get("plan_max", 0.0)  #//$NON-NLS-1$
                p_act = s.get("act_max", 0.0)  #//$NON-NLS-1$
                dice = s.get("dice", 0.0)  #//$NON-NLS-1$
                pct_plan = s.get("pct_plan_in_act", 0.0)  #//$NON-NLS-1$
                pct_act = s.get("pct_act_in_plan", 0.0)  #//$NON-NLS-1$
                f.write(f"{sub}\t{hemi}\t{p_plan:.1f}\t{p_act:.1f}\t{dice:.4f}\t{pct_plan:.2f}\t{pct_act:.2f}\n")  #//$NON-NLS-1$
    print(f"  [saved] {txt_path}")  #//$NON-NLS-1$

    # 2. Typst summary table
    typst_path = os.path.join(tables_dir, "focal_overlap_summary_typst.txt")  #//$NON-NLS-1$
    typst_path_typ = os.path.join(tables_dir, "focal_overlap_summary.typ")  #//$NON-NLS-1$
    typst_content = [
        "// Planned vs. Actual Focal Overlap Statistics Table (-3 dB Focus)",  #//$NON-NLS-1$
        "#table(",  #//$NON-NLS-1$
        "  columns: (auto, auto, auto, auto, auto, auto, auto),",  #//$NON-NLS-1$
        "  align: horizon + center,",  #//$NON-NLS-1$
        "  fill: (x, y) => if y == 0 { rgb(\"e0e0e0\") } else if calc.even(y) { rgb(\"f9f9f9\") } else { rgb(\"ffffff\") },",  #//$NON-NLS-1$ #//$NON-NLS-2$ #//$NON-NLS-3$ #//$NON-NLS-4$
        "  [*Subject*], [*Hemisphere*], [*Planned $P_(\"max\")$ (Pa)*], [*Actual $P_(\"max\")$ (Pa)*], [*Dice Coefficient*], [*% Planned in Actual*], [*% Actual in Planned*],"  #//$NON-NLS-1$ #//$NON-NLS-2$ #//$NON-NLS-3$ #//$NON-NLS-4$ #//$NON-NLS-5$ #//$NON-NLS-6$ #//$NON-NLS-7$
    ]
    for sub in SUBJECTS:
        for side in ["L", "R"]:  #//$NON-NLS-1$ #//$NON-NLS-2$
            s = all_stats[sub].get(side, {})
            if not s:
                continue
            hemi = "Left" if side == "L" else "Right"  #//$NON-NLS-1$ #//$NON-NLS-2$
            p_plan = s.get("plan_max", 0.0)  #//$NON-NLS-1$
            p_act = s.get("act_max", 0.0)  #//$NON-NLS-1$
            dice = s.get("dice", 0.0)  #//$NON-NLS-1$
            pct_plan = s.get("pct_plan_in_act", 0.0)  #//$NON-NLS-1$
            pct_act = s.get("pct_act_in_plan", 0.0)  #//$NON-NLS-1$
            typst_content.append(f"  [{sub}], [{hemi}], [{p_plan:,.0f}], [{p_act:,.0f}], [{dice:.3f}], [{pct_plan:.1f}%], [{pct_act:.1f}%],")  #//$NON-NLS-1$
    typst_content.append(")")  #//$NON-NLS-1$

    with open(typst_path, "w") as f:  #//$NON-NLS-1$
        f.write("\n".join(typst_content) + "\n")  #//$NON-NLS-1$
    print(f"  [saved] {typst_path}")  #//$NON-NLS-1$

    with open(typst_path_typ, "w") as f:  #//$NON-NLS-1$
        f.write("\n".join(typst_content) + "\n")  #//$NON-NLS-1$
    print(f"  [saved] {typst_path_typ}")  #//$NON-NLS-1$

    print("\nAll done.")  #//$NON-NLS-1$
