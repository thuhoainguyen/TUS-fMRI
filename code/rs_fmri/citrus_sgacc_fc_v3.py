#!/usr/bin/env python3
"""
CITRUS — sgACC single-subject functional connectivity  (v3)
Converted from citrus_sgacc_fc_v2.py with extended visualisations.

New in v3:
  • CLI args: --subject / --outdir
  • ΔFC heatmaps  (Exp, Control, Exp−Control contrast)
  • Spaghetti plots (absolute FC and ΔFC)
  • Radar overlaying all Exp timepoints
  • IQR temporal-variability bar chart
  • CSV exports for all numeric tables
  • 300 DPI PNGs + PDF exports
  • Publication/thesis-quality layout (tight margins, no overlap,
    legends outside data area, readable labels)

Usage:
  python citrus_sgacc_fc_v3.py [--subject sub-05] [--outdir /path/to/out]
"""

import argparse
import os
import subprocess
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image
from nilearn.maskers import NiftiMasker

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CLI ARGS
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="CITRUS sgACC FC v3")
    p.add_argument("--subject",  default="sub-05",
                   help="Subject ID (default: sub-05)")
    p.add_argument("--outdir",
                   default="/Users/hoaithunguyen/Projects/Master-thesis/CITRUS/derivatives/rs_fmri_v3",
                   help="Output directory")
    return p.parse_args()

# ─────────────────────────────────────────────────────────────
# CONFIGURATION  (paths not subject-specific stay as constants)
# ─────────────────────────────────────────────────────────────
MEPREP_ROOT  = Path("/Volumes/Extreme SSD/THESIS-MSC/MEPrep output")
CITRUS_INPUT = Path("/Users/hoaithunguyen/Projects/Master-thesis/CITRUS/data/input")
FSL_BIN      = Path("/Users/hoaithunguyen/fsl/bin")

SESSIONS   = ["ses-exp", "ses-con"]
TIMEPOINTS = ["preTUS15", "postTUS15", "postTUS30", "postTUS45"]
POST_TPS   = ["postTUS15", "postTUS30", "postTUS45"]

TR          = 1.50
SMOOTH_FWHM = 6.0
BANDPASS    = (0.01, 0.10)

MIDLINE_NODES = {
    "Frontal_Medial": 25,
    "Paracingulate":  28,
    "ACC":            29,
    "PCC":            30,
    "Precuneus":      31,
}
LATERAL_NODES = {
    "Insula":      (2,  2),
    "Hippocampus": (9,  19),
    "Amygdala":    (10, 20),
    "Thalamus":    (4,  15),
    "Accumbens":   (11, 21),
}

BOLD_PROC  = "pmeica"
BOLD_SPACE = "MNI152NLin2009cAsym"
NET_ROIS   = list(MIDLINE_NODES.keys()) + list(LATERAL_NODES.keys())

SES_COLORS = {"ses-exp": "#e05c5c", "ses-con": "#5c7de0"}
SES_LABELS = {"ses-exp": "Experimental (focused)", "ses-con": "Control (defocused)"}
TP_LABELS  = {
    "preTUS15":  "Pre\n(−15 min)",
    "postTUS15": "+15 min",
    "postTUS30": "+30 min",
    "postTUS45": "+45 min",
}
TP_SHORT   = {
    "preTUS15":  "Pre",
    "postTUS15": "+15",
    "postTUS30": "+30",
    "postTUS45": "+45",
}

# ROIs to highlight in spaghetti plots
HIGHLIGHT_ROIS = {"PCC", "Precuneus", "ACC", "Amygdala", "Hippocampus", "Accumbens"}

FSLDIR = FSL_BIN.parent
NETWORK_MASKS = {}

DPI    = 300
FIGFMT = ["png"]   # PNG only

# Global rcParams for publication-quality figures
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "figure.dpi":        100,       # screen; DPI applied at save time
    "savefig.dpi":       DPI,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.1,
})

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def savefig(fig, path_stem: Path):
    for ext in FIGFMT:
        fig.savefig(str(path_stem) + f".{ext}", dpi=DPI,
                    bbox_inches="tight", pad_inches=0.1)


def _roi_label(name: str) -> str:
    """Shorten long ROI names so axis labels don't overlap."""
    mapping = {"Frontal_Medial": "FronMed", "Paracingulate": "Paracin"}
    return mapping.get(name, name)


# ─────────────────────────────────────────────────────────────
# FSL WRAPPERS / REGISTRATION
# ─────────────────────────────────────────────────────────────
def _flirt(args):
    fsl_env = {**os.environ,
               "FSLDIR": str(FSLDIR),
               "FSLOUTPUTTYPE": "NIFTI_GZ",
               "PATH": str(FSL_BIN) + ":" + os.environ.get("PATH", "")}
    return subprocess.run([str(FSL_BIN / "flirt")] + args,
                          capture_output=True, text=True, env=fsl_env)

def register_kplan_to_mni(sub: str, out_dir: Path):
    kplan_t1 = CITRUS_INPUT / sub / f"{sub}_T1w_kplan.nii.gz"
    mni_t1   = MEPREP_ROOT / "ses-intake" / f"{sub}_ses-intake_acq-HCP_space-{BOLD_SPACE}_desc-preproc_T1w.nii.gz"
    seeds_dir = out_dir / "seeds" / sub
    seeds_dir.mkdir(parents=True, exist_ok=True)
    mat_path = seeds_dir / f"{sub}_kplan_to_MNI.mat"

    if mat_path.exists():
        print(f"  [{sub}] flirt .mat already exists — skipping")
        return mat_path
    if not kplan_t1.exists():
        print(f"  [{sub}] MISSING kplan T1w: {kplan_t1}")
        return None
    if not mni_t1.exists():
        print(f"  [{sub}] MISSING MNI T1w: {mni_t1}")
        return None

    print(f"  [{sub}] Running flirt (kplan T1w -> MNI T1w)...")
    result = _flirt(["-in", str(kplan_t1), "-ref", str(mni_t1),
                     "-omat", str(mat_path), "-dof", "12",
                     "-cost", "corratio", "-interp", "spline"])
    if result.returncode != 0:
        print(f"  [{sub}] flirt FAILED:\n{result.stderr}")
        return None
    print(f"  [{sub}] flirt done -> {mat_path.name}")
    return mat_path

def warp_mask_to_mni(sub: str, hemi: str, mat_path, out_dir: Path) -> Path:
    mask_kplan = CITRUS_INPUT / sub / f"sgACC_BA25_{hemi}_kplan.nii.gz"
    mni_t1     = MEPREP_ROOT / "ses-intake" / f"{sub}_ses-intake_acq-HCP_space-{BOLD_SPACE}_desc-preproc_T1w.nii.gz"
    seeds_dir  = out_dir / "seeds" / sub
    out_mask   = seeds_dir / f"{sub}_sgACC_BA25_{hemi}_MNI.nii.gz"

    if out_mask.exists():
        return out_mask
    if not mask_kplan.exists():
        print(f"  [{sub}] MISSING kplan mask: {mask_kplan}")
        return None
    result = _flirt(["-in", str(mask_kplan), "-ref", str(mni_t1),
                     "-init", str(mat_path), "-applyxfm",
                     "-interp", "nearestneighbour", "-out", str(out_mask)])
    if result.returncode != 0:
        print(f"  [{sub}] mask warp FAILED:\n{result.stderr}")
        return None
    return out_mask

def make_bilateral_seed_mask(sub: str, mask_L: Path, mask_R: Path, out_dir: Path) -> Path:
    seeds_dir = out_dir / "seeds" / sub
    out_path  = seeds_dir / f"{sub}_sgACC_BA25_bilateral_MNI.nii.gz"
    if out_path.exists():
        return out_path
    img_L = nib.load(str(mask_L))
    img_R = nib.load(str(mask_R))
    bilateral = (img_L.get_fdata() + img_R.get_fdata()) > 0.5
    nib.save(nib.Nifti1Image(bilateral.astype(np.float32), img_L.affine), str(out_path))
    return out_path

# ─────────────────────────────────────────────────────────────
# DATA LOADING / PREPROCESSING
# ─────────────────────────────────────────────────────────────
def find_run(sub, ses, acq):
    func_dir = MEPREP_ROOT / ses / "func"
    stem = f"{sub}_{ses}_task-rest_acq-{acq}_proc-{BOLD_PROC}"
    bold = func_dir / f"{stem}_space-{BOLD_SPACE}_desc-preproc_bold.nii.gz"
    mask = func_dir / f"{stem}_space-{BOLD_SPACE}_desc-brain_mask.nii.gz"
    conf = func_dir / f"{stem}_desc-confounds_timeseries.tsv"
    if bold.exists() and mask.exists() and conf.exists():
        return bold, mask, conf
    print(f"  MISSING: {sub} | {ses} | {acq}")
    return None, None, None

def get_confounds(conf_path):
    df = pd.read_csv(conf_path, sep="\t")
    hmp    = ["trans_x","trans_y","trans_z","rot_x","rot_y","rot_z"]
    hmp24  = hmp + [f"{c}_derivative1" for c in hmp] + \
                   [f"{c}_power2"       for c in hmp] + \
                   [f"{c}_derivative1_power2" for c in hmp]
    acomp  = [f"a_comp_cor_{i:02d}" for i in range(5)]
    cosine = [c for c in df.columns if c.startswith("cosine")]
    outliers = [c for c in df.columns if c.startswith("motion_outlier")]
    cols = [c for c in hmp24 + acomp + cosine + outliers if c in df.columns]
    return df[cols].fillna(0).values

# ─────────────────────────────────────────────────────────────
# ATLAS / ROI SETUP
# ─────────────────────────────────────────────────────────────
def setup_atlases(ref_bold_path):
    global NETWORK_MASKS
    print("Loading Harvard-Oxford atlases...")
    ho_cort = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    ho_sub  = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")

    ref_img   = image.index_img(image.load_img(str(ref_bold_path)), 0)
    cort_res  = image.resample_to_img(ho_cort.maps, ref_img, interpolation="nearest")
    sub_res   = image.resample_to_img(ho_sub.maps,  ref_img, interpolation="nearest")
    cort_data = cort_res.get_fdata()
    sub_data  = sub_res.get_fdata()

    def roi_mask(data, idx):
        return image.new_img_like(ref_img, (data == idx).astype(np.float32))

    def bilateral_mask(data, l_idx, r_idx):
        return image.new_img_like(ref_img, ((data == l_idx) | (data == r_idx)).astype(np.float32))

    for name, idx in MIDLINE_NODES.items():
        NETWORK_MASKS[name] = roi_mask(cort_data, idx)

    NETWORK_MASKS["Insula"] = roi_mask(cort_data, LATERAL_NODES["Insula"][0])

    for name, (l_idx, r_idx) in LATERAL_NODES.items():
        if name == "Insula":
            continue
        NETWORK_MASKS[name] = bilateral_mask(sub_data, l_idx, r_idx)

# ─────────────────────────────────────────────────────────────
# FC EXTRACTION  (unchanged logic from v2)
# ─────────────────────────────────────────────────────────────
def extract_network_connectivity(bold_path, mask_path, conf_path, seed_mask=None):
    confounds = get_confounds(conf_path)
    roi_names = list(NETWORK_MASKS.keys())

    combined_mask_data = None
    for roi_name in roi_names:
        md = NETWORK_MASKS[roi_name].get_fdata() > 0
        combined_mask_data = md if combined_mask_data is None else combined_mask_data | md

    combined_mask_img = image.new_img_like(NETWORK_MASKS[roi_names[0]],
                                           combined_mask_data.astype(np.float32))

    masker = NiftiMasker(
        mask_img=combined_mask_img,
        smoothing_fwhm=SMOOTH_FWHM,
        detrend=True,
        standardize="zscore_sample",
        low_pass=BANDPASS[1],
        high_pass=BANDPASS[0],
        t_r=TR,
        verbose=0,
    )
    all_voxels_ts = masker.fit_transform(str(bold_path), confounds=confounds)

    mask_voxel_indices = np.where(masker.mask_img_.get_fdata() > 0)
    roi_ts_list = []
    for roi_name in roi_names:
        roi_mask_data = NETWORK_MASKS[roi_name].get_fdata() > 0
        in_roi = roi_mask_data[mask_voxel_indices]
        roi_ts_list.append(all_voxels_ts[:, in_roi].mean(axis=1))

    roi_ts = np.column_stack(roi_ts_list)
    r_mat  = np.corrcoef(roi_ts.T)
    z_mat  = np.arctanh(np.clip(r_mat, -0.999, 0.999))
    np.fill_diagonal(z_mat, 0)

    seed_fc = None
    if seed_mask is not None:
        bold_ref   = image.index_img(image.load_img(str(bold_path)), 0)
        seed_img   = image.resample_to_img(image.load_img(str(seed_mask)),
                                           bold_ref, interpolation="nearest")
        seed_masker = NiftiMasker(mask_img=seed_img, smoothing_fwhm=None,
                                  detrend=True, standardize="zscore_sample",
                                  low_pass=BANDPASS[1], high_pass=BANDPASS[0],
                                  t_r=TR, verbose=0)
        seed_ts = seed_masker.fit_transform(str(bold_path), confounds=confounds).mean(axis=1)
        n = len(seed_ts)
        seed_fc = {
            roi_name: float(np.arctanh(np.clip(
                np.dot(seed_ts, roi_ts[:, i]) / n, -0.999, 0.999)))
            for i, roi_name in enumerate(roi_names)
        }

    return z_mat, roi_ts, roi_names, seed_fc

# ─────────────────────────────────────────────────────────────
# HELPER: pivot FC into (roi × timepoint) arrays
# ─────────────────────────────────────────────────────────────
def fc_matrix(df_fc, ses, timepoints, rois):
    """Return (n_roi, n_tp) numpy array of mean fc_z values."""
    mat = np.full((len(rois), len(timepoints)), np.nan)
    for j, tp in enumerate(timepoints):
        for i, roi in enumerate(rois):
            sub = df_fc[(df_fc["session"] == ses) &
                        (df_fc["timepoint"] == tp) &
                        (df_fc["roi"] == roi)]["fc_z"]
            if len(sub):
                mat[i, j] = sub.mean()
    return mat

# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────

def plot_connectivity_matrices(conn_matrices, roi_names_list, sub, out_dir):
    """Connectivity matrix grid — colorbar in the gap below the title."""
    short_labels = [_roi_label(r) for r in roi_names_list]
    n_tp  = len(TIMEPOINTS)
    n_ses = len(SESSIONS)

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(4.2 * n_tp, 4.5 * n_ses + 1.0), facecolor="white")

    # Three rows: [colorbar strip, Experimental row, Control row]
    gs = GridSpec(
        n_ses + 1, n_tp,
        figure=fig,
        height_ratios=[0.08] + [1.0] * n_ses,
        hspace=0.35, wspace=0.25,
        top=0.93, bottom=0.08, left=0.07, right=0.98,
    )

    fig.suptitle(f"{sub} — sgACC network connectivity matrix (Fisher z)",
                 fontsize=13, fontweight="bold", y=0.98)

    # Colorbar axes spanning all columns in row 0
    cax = fig.add_subplot(gs[0, :])

    im_ref = None
    for r, ses in enumerate(SESSIONS):
        for c, acq in enumerate(TIMEPOINTS):
            ax  = fig.add_subplot(gs[r + 1, c])
            key = (sub, ses, acq)
            if key not in conn_matrices:
                ax.set_visible(False)
                continue
            im = ax.imshow(conn_matrices[key], cmap="RdBu_r",
                           vmin=-1, vmax=1, aspect="auto")
            im_ref = im
            n = len(short_labels)
            ax.set_xticks(range(n))
            ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(n))
            ax.set_yticklabels(short_labels, fontsize=8)
            if r == 0:
                ax.set_title(TP_LABELS[acq], fontsize=10, pad=6)
            if c == 0:
                ax.set_ylabel(SES_LABELS[ses], fontsize=9,
                              fontweight="bold", labelpad=6)

    if im_ref is not None:
        cb = fig.colorbar(im_ref, cax=cax, orientation="horizontal")
        cb.set_label("Fisher z", fontsize=10, labelpad=4)
        cb.ax.tick_params(labelsize=9)
        cax.xaxis.set_label_position("top")
        cax.xaxis.tick_top()

    savefig(fig, out_dir / f"{sub}_connectivity_matrix")
    plt.close(fig)


def plot_delta_heatmaps(df_fc, sub, out_dir):
    """ΔFC heatmaps: Exp, Control, and Exp−Control contrast."""
    rois        = NET_ROIS
    short_rois  = [_roi_label(r) for r in rois]
    post_labels = [TP_SHORT[tp] for tp in POST_TPS]

    # Build ΔFC arrays for each session
    delta = {}
    for ses in SESSIONS:
        full       = fc_matrix(df_fc, ses, TIMEPOINTS, rois)  # (n_roi, 4)
        pre        = full[:, 0:1]
        delta[ses] = full[:, 1:] - pre                         # (n_roi, 3)

    contrast = delta["ses-exp"] - delta["ses-con"]

    # Shared symmetric colour limits for Exp/Control; separate for contrast
    all_dc   = np.concatenate([delta["ses-exp"].ravel(), delta["ses-con"].ravel()])
    lim_dc   = max(abs(np.nanmax(all_dc)),   abs(np.nanmin(all_dc)))   or 0.5
    lim_cont = max(abs(np.nanmax(contrast)), abs(np.nanmin(contrast))) or 0.5

    n_rois = len(rois)
    # Height: 0.55 per ROI + room for title & colorbar, minimum 5 in
    cell_h = 0.55
    fig_h  = max(5.0, cell_h * n_rois + 2.5)

    def _heatmap(data, title, clabel, clim, stem):
        # Width: 3 columns + space for y-labels + colorbar
        fig_w = max(6.0, 1.2 * data.shape[1] + 3.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")

        im = ax.imshow(data, cmap="RdBu_r", vmin=-clim, vmax=clim,
                       aspect="auto", interpolation="nearest")

        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels(post_labels, fontsize=11)
        ax.set_yticks(range(n_rois))
        ax.set_yticklabels(short_rois, fontsize=10)
        ax.tick_params(axis="x", pad=4)
        ax.tick_params(axis="y", pad=4)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

        # Colorbar: fixed fraction of figure height, outside plot
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, aspect=20)
        cb.set_label(clabel, fontsize=10, labelpad=6)
        cb.ax.tick_params(labelsize=9)
        # Ensure colorbar ticks are symmetric and readable
        cb.locator = mticker.MaxNLocator(nbins=6, symmetric=True)
        cb.update_ticks()

        # Annotate each cell — use white text when background is saturated
        thresh = 0.55 * clim
        for i in range(n_rois):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isnan(v):
                    continue
                txt_color = "white" if abs(v) >= thresh else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8.5, color=txt_color, fontweight="bold")

        fig.tight_layout(pad=0.6)
        savefig(fig, out_dir / stem)
        plt.close(fig)

    _heatmap(delta["ses-exp"],
             f"{sub} — Experimental ΔFC (Post − Pre)",
             "ΔFC (Post − Pre)", lim_dc,
             f"{sub}_sgacc_delta_heatmap_exp")

    _heatmap(delta["ses-con"],
             f"{sub} — Control ΔFC (Post − Pre)",
             "ΔFC (Post − Pre)", lim_dc,
             f"{sub}_sgacc_delta_heatmap_ctrl")

    _heatmap(contrast,
             f"{sub} — sgACC FC: ΔExp − ΔCtrl",
             "ΔFC Exp − ΔFC Ctrl", lim_cont,
             f"{sub}_sgacc_delta_contrast_exp_minus_ctrl")

    return delta, contrast


def plot_spaghetti(df_fc, sub, out_dir):
    """Spaghetti plots: absolute FC and ΔFC."""
    rois   = NET_ROIS
    gray   = "#c0c0c0"

    highlight_colors = {
        "PCC":         "#e05c5c",
        "Precuneus":   "#e09b5c",
        "ACC":         "#5c7de0",
        "Amygdala":    "#8e5ce0",
        "Hippocampus": "#5cb8e0",
        "Accumbens":   "#5ce07a",
    }

    # ── A. Absolute FC ──────────────────────────────────────
    tp_list       = TIMEPOINTS
    tp_x_abs      = list(range(len(tp_list)))
    tp_labels_abs = [TP_SHORT[t] for t in tp_list]

    # Legend goes to the right → leave space
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True, facecolor="white")
    fig.suptitle(f"{sub} — sgACC-to-ROI absolute FC over time",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, ses in zip(axes, SESSIONS):
        # Non-highlighted first (behind)
        for roi in rois:
            if roi in HIGHLIGHT_ROIS:
                continue
            vals = [df_fc[(df_fc["session"] == ses) &
                          (df_fc["timepoint"] == tp) &
                          (df_fc["roi"] == roi)]["fc_z"].mean()
                    for tp in tp_list]
            ax.plot(tp_x_abs, vals, "-o", lw=1.0, ms=4,
                    color=gray, alpha=0.55, zorder=1)

        # Highlighted ROIs (on top)
        for roi in rois:
            if roi not in HIGHLIGHT_ROIS:
                continue
            vals = [df_fc[(df_fc["session"] == ses) &
                          (df_fc["timepoint"] == tp) &
                          (df_fc["roi"] == roi)]["fc_z"].mean()
                    for tp in tp_list]
            ax.plot(tp_x_abs, vals, "-o", lw=2.5, ms=7,
                    color=highlight_colors[roi], zorder=3, label=roi)

        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.45)
        ax.set_xticks(tp_x_abs)
        ax.set_xticklabels(tp_labels_abs, fontsize=10)
        ax.set_xlabel("Timepoint", fontsize=10, labelpad=4)
        if ax is axes[0]:
            ax.set_ylabel("sgACC FC (Fisher z)", fontsize=10, labelpad=6)
        ax.set_title(SES_LABELS[ses], fontsize=11, fontweight="bold", pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
                  framealpha=0.85, title="ROI", title_fontsize=8)

    fig.tight_layout(rect=[0, 0, 0.88, 0.97], w_pad=3)
    savefig(fig, out_dir / f"{sub}_sgacc_spaghetti_absolute_fc")
    plt.close(fig)

    # ── B. ΔFC ──────────────────────────────────────────────
    post_list       = POST_TPS
    tp_x_delta      = list(range(len(post_list)))
    tp_labels_delta = [TP_SHORT[t] for t in post_list]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, facecolor="white")
    fig.suptitle(f"{sub} — sgACC-to-ROI ΔFC (relative to Pre)",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, ses in zip(axes, SESSIONS):
        pre_vals = {
            roi: df_fc[(df_fc["session"] == ses) &
                       (df_fc["timepoint"] == "preTUS15") &
                       (df_fc["roi"] == roi)]["fc_z"].mean()
            for roi in rois
        }
        # Non-highlighted
        for roi in rois:
            if roi in HIGHLIGHT_ROIS:
                continue
            delta = [df_fc[(df_fc["session"] == ses) &
                           (df_fc["timepoint"] == tp) &
                           (df_fc["roi"] == roi)]["fc_z"].mean() - pre_vals[roi]
                     for tp in post_list]
            ax.plot(tp_x_delta, delta, "-o", lw=1.0, ms=4,
                    color=gray, alpha=0.55, zorder=1)

        # Highlighted
        for roi in rois:
            if roi not in HIGHLIGHT_ROIS:
                continue
            delta = [df_fc[(df_fc["session"] == ses) &
                           (df_fc["timepoint"] == tp) &
                           (df_fc["roi"] == roi)]["fc_z"].mean() - pre_vals[roi]
                     for tp in post_list]
            ax.plot(tp_x_delta, delta, "-o", lw=2.5, ms=7,
                    color=highlight_colors[roi], zorder=3, label=roi)

        ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.5)
        ax.set_xticks(tp_x_delta)
        ax.set_xticklabels(tp_labels_delta, fontsize=10)
        ax.set_xlabel("Timepoint", fontsize=10, labelpad=4)
        if ax is axes[0]:
            ax.set_ylabel("ΔFC (Post − Pre, Fisher z)", fontsize=10, labelpad=6)
        ax.set_title(SES_LABELS[ses], fontsize=11, fontweight="bold", pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
                  framealpha=0.85, title="ROI", title_fontsize=8)

    fig.tight_layout(rect=[0, 0, 0.88, 0.97], w_pad=3)
    savefig(fig, out_dir / f"{sub}_sgacc_spaghetti_delta_fc")
    plt.close(fig)


def plot_radar_exp_timepoints(df_fc, sub, out_dir):
    """Radar chart overlaying all four Experimental timepoints.

    Shift strategy: add a constant so every radial value is ≥ 0.
    The shift is clearly annotated; data interpretation is unaffected.
    """
    rois        = NET_ROIS
    short_rois  = [_roi_label(r) for r in rois]
    N           = len(rois)
    angles      = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_cl   = angles + angles[:1]   # closed polygon

    # Compute global min across all Exp timepoints (all ROIs)
    all_exp = np.array([
        df_fc[(df_fc["session"] == "ses-exp") &
              (df_fc["roi"] == roi)]["fc_z"].mean()
        for roi in rois
        for _ in TIMEPOINTS   # repeated per tp — kept simple; we just need min
    ])
    global_min = np.nanmin(all_exp)
    # Shift to keep radial axis non-negative with 15 % padding below
    shift = max(0.0, -global_min + 0.05)

    tp_styles = {
        "preTUS15":  {"color": "#444444", "ls": "--", "lw": 2.0, "ms": 6, "alpha": 0.85},
        "postTUS15": {"color": "#e05c5c", "ls": "-",  "lw": 2.5, "ms": 7, "alpha": 0.95},
        "postTUS30": {"color": "#e09b5c", "ls": "-",  "lw": 2.5, "ms": 7, "alpha": 0.95},
        "postTUS45": {"color": "#8e5ce0", "ls": "-",  "lw": 2.5, "ms": 7, "alpha": 0.95},
    }

    # Figure large enough that outer ROI labels + legend don't clip
    fig = plt.figure(figsize=(11, 10), facecolor="white")
    # Polar axes centred with room for labels and legend
    ax = fig.add_axes([0.08, 0.08, 0.62, 0.82], projection="polar")

    tp_vals_shifted = {}
    for tp in TIMEPOINTS:
        vals = np.array([
            df_fc[(df_fc["session"] == "ses-exp") &
                  (df_fc["timepoint"] == tp) &
                  (df_fc["roi"] == roi)]["fc_z"].mean()
            for roi in rois
        ])
        tp_vals_shifted[tp] = vals + shift

    # Global radial range with 15 % headroom
    all_shifted = np.concatenate(list(tp_vals_shifted.values()))
    r_max = np.nanmax(all_shifted) * 1.15
    r_max = max(r_max, 0.1)
    r_min = 0.0

    # Draw fills first, then lines on top.
    # zorder=2 puts fills above the polar grid (zorder=1 by default).
    for tp in TIMEPOINTS:
        vs = tp_vals_shifted[tp]
        vs_cl = vs.tolist() + [vs[0]]
        st = tp_styles[tp]
        ax.fill(angles_cl, vs_cl, color=st["color"], alpha=0.40, zorder=2)

    for tp in TIMEPOINTS:
        vs = tp_vals_shifted[tp]
        vs_cl = vs.tolist() + [vs[0]]
        st = tp_styles[tp]
        ax.plot(angles_cl, vs_cl,
                ls=st["ls"], lw=st["lw"], color=st["color"],
                marker="o", ms=st["ms"], alpha=st["alpha"],
                label=TP_SHORT[tp], zorder=4)

    ax.set_xticks(angles)
    ax.set_xticklabels(short_rois, fontsize=10)

    # Radial limits and grid lines
    ax.set_ylim(r_min, r_max)
    n_grid = 5
    grid_vals = np.linspace(r_min, r_max, n_grid + 1)[1:]   # skip 0
    ax.set_yticks(grid_vals)
    # Show grid-tick labels as original (un-shifted) values
    ax.set_yticklabels(
        [f"{v - shift:.2f}" for v in grid_vals],
        fontsize=8, color="gray"
    )
    ax.yaxis.set_tick_params(pad=3)
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    # Highlight the zero circle (un-shifted value = 0 → radius = shift)
    if shift > 0:
        zero_r = shift
        theta_full = np.linspace(0, 2 * np.pi, 300)
        ax.plot(theta_full, [zero_r] * 300,
                color="black", lw=1.2, ls=":", alpha=0.6, zorder=5)

    ax.set_title(
        f"{sub} — Experimental sgACC connectivity fingerprint over time",
        fontsize=11, fontweight="bold",
        pad=22,
    )

    # Legend outside the polar circle
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1.05),
        bbox_transform=ax.transAxes,
        fontsize=10,
        framealpha=0.8,
        title="Timepoint",
        title_fontsize=9,
    )

    # Annotation about the shift
    note_lines = [
        f"Radial axis = FC + {shift:.2f}",
        f"(shift applied so axis starts at 0).",
        "Dotted circle = FC = 0." if shift > 0 else "",
    ]
    fig.text(0.73, 0.12, "\n".join(l for l in note_lines if l),
             fontsize=8, color="gray", va="bottom", ha="left",
             style="italic")

    savefig(fig, out_dir / f"{sub}_sgacc_radar_exp_timepoints")
    plt.close(fig)


def compute_iqr(df_fc):
    """Compute IQR of FC across post-TUS timepoints — no plot, CSV only."""
    records = []
    for ses in SESSIONS:
        for roi in NET_ROIS:
            abs_vals = np.array([
                df_fc[(df_fc["session"] == ses) &
                      (df_fc["timepoint"] == tp) &
                      (df_fc["roi"] == roi)]["fc_z"].mean()
                for tp in POST_TPS
            ])
            iqr_abs = float(np.nanpercentile(abs_vals, 75) -
                            np.nanpercentile(abs_vals, 25))
            pre = df_fc[(df_fc["session"] == ses) &
                        (df_fc["timepoint"] == "preTUS15") &
                        (df_fc["roi"] == roi)]["fc_z"].mean()
            delta_vals = abs_vals - pre
            iqr_delta  = float(np.nanpercentile(delta_vals, 75) -
                               np.nanpercentile(delta_vals, 25))
            records.append({"session": ses, "roi": roi,
                            "iqr_abs": iqr_abs, "iqr_delta": iqr_delta})
    return pd.DataFrame(records)


def export_csvs(df_fc, delta, contrast, df_iqr, sub, out_dir):  # noqa: PLR0913
    """Export all numeric tables as CSV."""
    rois = NET_ROIS

    # 1. Absolute FC table  (roi × timepoint)
    rows = []
    for ses in SESSIONS:
        for tp in TIMEPOINTS:
            for roi in rois:
                val = df_fc[(df_fc["session"] == ses) &
                            (df_fc["timepoint"] == tp) &
                            (df_fc["roi"] == roi)]["fc_z"].mean()
                rows.append({"session": ses, "timepoint": tp, "roi": roi, "fc_z": val})
    pd.DataFrame(rows).to_csv(out_dir / f"{sub}_sgacc_absolute_fc.csv", index=False)

    # 2. ΔFC Experimental
    df_exp = pd.DataFrame(delta["ses-exp"], index=rois, columns=[TP_SHORT[t] for t in POST_TPS])
    df_exp.index.name = "roi"
    df_exp.to_csv(out_dir / f"{sub}_sgacc_delta_fc_exp.csv")

    # 3. ΔFC Control
    df_ctrl = pd.DataFrame(delta["ses-con"], index=rois, columns=[TP_SHORT[t] for t in POST_TPS])
    df_ctrl.index.name = "roi"
    df_ctrl.to_csv(out_dir / f"{sub}_sgacc_delta_fc_ctrl.csv")

    # 4. ΔExp − ΔCtrl contrast
    df_cont = pd.DataFrame(contrast, index=rois, columns=[TP_SHORT[t] for t in POST_TPS])
    df_cont.index.name = "roi"
    df_cont.to_csv(out_dir / f"{sub}_sgacc_delta_contrast_exp_minus_ctrl.csv")

    # 5. IQR summary
    df_iqr.to_csv(out_dir / f"{sub}_sgacc_iqr_summary.csv", index=False)

    print(f"CSV tables written to {out_dir}")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    args    = parse_args()
    sub     = args.subject
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Subject   : {sub}")
    print(f"Output dir: {out_dir}")

    # 1. Seed masks
    SUBJECT_MASKS = {}
    mat = register_kplan_to_mni(sub, out_dir)
    if mat:
        hemi_masks = {}
        for hemi in ["L", "R"]:
            p = warp_mask_to_mni(sub, hemi, mat, out_dir)
            if p:
                hemi_masks[hemi] = p
        if "L" in hemi_masks and "R" in hemi_masks:
            bil = make_bilateral_seed_mask(sub, hemi_masks["L"], hemi_masks["R"], out_dir)
            SUBJECT_MASKS[sub] = bil
        elif hemi_masks:
            SUBJECT_MASKS[sub] = list(hemi_masks.values())[0]

    # 2. Atlas setup from first available BOLD
    bold_ref = None
    for ses in SESSIONS:
        for acq in TIMEPOINTS:
            bold, _, _ = find_run(sub, ses, acq)
            if bold:
                bold_ref = bold
                break
        if bold_ref:
            break

    if not bold_ref:
        print("No BOLD data found. Exit.")
        return

    setup_atlases(bold_ref)

    # 3. Compute FC
    net_records   = []
    conn_matrices = {}
    roi_names_list = None
    seed_mask = SUBJECT_MASKS.get(sub)

    for ses in SESSIONS:
        for acq in TIMEPOINTS:
            bold, mask, conf = find_run(sub, ses, acq)
            if not bold:
                continue
            print(f"  Processing {sub} | {ses} | {acq} ...")
            z_mat, _, roi_names, seed_fc = extract_network_connectivity(
                bold, mask, conf, seed_mask)
            conn_matrices[(sub, ses, acq)] = z_mat
            roi_names_list = roi_names

            if seed_fc:
                for roi_name, fc_z in seed_fc.items():
                    net_records.append({
                        "subject": sub, "session": ses,
                        "timepoint": acq, "roi": roi_name, "fc_z": fc_z,
                    })

    df_fc = pd.DataFrame(net_records)
    if df_fc.empty:
        print("No FC data computed. Check inputs.")
        return

    # 4. Plots
    print("Generating plots...")

    # 4a. Connectivity matrices (from v2, kept intact)
    if roi_names_list:
        plot_connectivity_matrices(conn_matrices, roi_names_list, sub, out_dir)

    # 4b. ΔFC heatmaps
    delta, contrast = plot_delta_heatmaps(df_fc, sub, out_dir)

    # 4c. Spaghetti plots (replaces old temporal line plots)
    plot_spaghetti(df_fc, sub, out_dir)

    # 4d. Radar: Experimental timepoints overlaid
    plot_radar_exp_timepoints(df_fc, sub, out_dir)

    # 5. CSV exports
    df_iqr = compute_iqr(df_fc)
    export_csvs(df_fc, delta, contrast, df_iqr, sub, out_dir)

    # 6. Raw FC table (full)
    fc_path = out_dir / f"{sub}_sgacc_fc_values.tsv"
    df_fc.to_csv(fc_path, sep="\t", index=False)

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
