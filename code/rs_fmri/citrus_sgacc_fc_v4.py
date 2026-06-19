#!/usr/bin/env python3
"""
CITRUS — sgACC single-subject functional connectivity  (v4)

Changes from v3:
  • Contrast trajectory plot — data-driven auto-highlighting
  • Spaghetti plots updated to use same data-driven highlighting
  • Heatmap titles & layout cleaned up
  • PNG + SVG for trajectory figure; PNG only for everything else
  • FIGFMT = ["png"] globally; trajectory saved separately as .svg too

Usage:
  python citrus_sgacc_fc_v4.py [--subject sub-05] [--outdir /path/to/out]
"""

import argparse
import os
import subprocess
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
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
    p = argparse.ArgumentParser(description="CITRUS sgACC FC v4")
    p.add_argument("--subject", default="sub-05",
                   help="Subject ID (default: sub-05)")
    p.add_argument("--outdir",
                   default="/Users/hoaithunguyen/Projects/Master-thesis/CITRUS/derivatives/rs_fmri_v4",
                   help="Output directory")
    return p.parse_args()

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
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
TP_SHORT = {
    "preTUS15":  "Pre",
    "postTUS15": "+15",
    "postTUS30": "+30",
    "postTUS45": "+45",
}

# Color palettes for auto-highlighting
# Positive contrast (ΔExp > ΔCtrl): warm
POS_COLORS = ["#d62728", "#ff7f0e", "#e377c2"]   # red, orange, pink
# Negative contrast (ΔExp < ΔCtrl): cool
NEG_COLORS = ["#1f77b4", "#17becf", "#9467bd"]   # blue, cyan, purple

FSLDIR        = FSL_BIN.parent
NETWORK_MASKS = {}

DPI = 300

# ─────────────────────────────────────────────────────────────
# GLOBAL RC PARAMS
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    9,
    "figure.dpi":         100,
    "savefig.dpi":        DPI,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.1,
})

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def savefig(fig, path_stem: Path, extra_fmts=()):
    """Save PNG always; optionally also SVG or other formats."""
    for ext in ["png"] + list(extra_fmts):
        fig.savefig(str(path_stem) + f".{ext}", dpi=DPI,
                    bbox_inches="tight", pad_inches=0.1)


def _roi_label(name: str) -> str:
    mapping = {"Frontal_Medial": "FronMed", "Paracingulate": "Paracin"}
    return mapping.get(name, name)


def get_contrast_highlights(contrast_mat, rois, n_top=3):
    """
    Auto-select top-n positive and top-n negative ROIs by
    max absolute contrast value across post-TUS timepoints.

    Returns dict: roi_name → color string.
    """
    max_abs = {roi: np.nanmax(np.abs(contrast_mat[i, :]))
               for i, roi in enumerate(rois)}
    mean_c  = {roi: np.nanmean(contrast_mat[i, :])
               for i, roi in enumerate(rois)}

    # Sort by mean contrast
    ranked = sorted(rois, key=lambda r: mean_c[r], reverse=True)
    pos_rois = ranked[:n_top]          # most positive
    neg_rois = ranked[-(n_top):][::-1] # most negative

    colors = {}
    for i, roi in enumerate(pos_rois):
        colors[roi] = POS_COLORS[i % len(POS_COLORS)]
    for i, roi in enumerate(neg_rois):
        if roi not in colors:   # avoid double-assigning if overlap
            colors[roi] = NEG_COLORS[i % len(NEG_COLORS)]
    return colors


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


def register_kplan_to_mni(sub, out_dir):
    kplan_t1  = CITRUS_INPUT / sub / f"{sub}_T1w_kplan.nii.gz"
    mni_t1    = MEPREP_ROOT / "ses-intake" / \
                f"{sub}_ses-intake_acq-HCP_space-{BOLD_SPACE}_desc-preproc_T1w.nii.gz"
    seeds_dir = out_dir / "seeds" / sub
    seeds_dir.mkdir(parents=True, exist_ok=True)
    mat_path  = seeds_dir / f"{sub}_kplan_to_MNI.mat"

    if mat_path.exists():
        print(f"  [{sub}] flirt .mat already exists — skipping")
        return mat_path
    for label, path in [("kplan T1w", kplan_t1), ("MNI T1w", mni_t1)]:
        if not path.exists():
            print(f"  [{sub}] MISSING {label}: {path}")
            return None

    print(f"  [{sub}] Running flirt (kplan T1w → MNI T1w)...")
    r = _flirt(["-in", str(kplan_t1), "-ref", str(mni_t1),
                "-omat", str(mat_path), "-dof", "12",
                "-cost", "corratio", "-interp", "spline"])
    if r.returncode != 0:
        print(f"  [{sub}] flirt FAILED:\n{r.stderr}")
        return None
    print(f"  [{sub}] flirt done → {mat_path.name}")
    return mat_path


def warp_mask_to_mni(sub, hemi, mat_path, out_dir):
    mask_kplan = CITRUS_INPUT / sub / f"sgACC_BA25_{hemi}_kplan.nii.gz"
    mni_t1     = MEPREP_ROOT / "ses-intake" / \
                 f"{sub}_ses-intake_acq-HCP_space-{BOLD_SPACE}_desc-preproc_T1w.nii.gz"
    seeds_dir  = out_dir / "seeds" / sub
    out_mask   = seeds_dir / f"{sub}_sgACC_BA25_{hemi}_MNI.nii.gz"
    if out_mask.exists():
        return out_mask
    if not mask_kplan.exists():
        print(f"  [{sub}] MISSING kplan mask: {mask_kplan}")
        return None
    r = _flirt(["-in", str(mask_kplan), "-ref", str(mni_t1),
                "-init", str(mat_path), "-applyxfm",
                "-interp", "nearestneighbour", "-out", str(out_mask)])
    if r.returncode != 0:
        print(f"  [{sub}] mask warp FAILED:\n{r.stderr}")
        return None
    return out_mask


def make_bilateral_seed_mask(sub, mask_L, mask_R, out_dir):
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
# DATA LOADING
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
    hmp   = ["trans_x","trans_y","trans_z","rot_x","rot_y","rot_z"]
    hmp24 = hmp + [f"{c}_derivative1" for c in hmp] + \
                  [f"{c}_power2"       for c in hmp] + \
                  [f"{c}_derivative1_power2" for c in hmp]
    acomp    = [f"a_comp_cor_{i:02d}" for i in range(5)]
    cosine   = [c for c in df.columns if c.startswith("cosine")]
    outliers = [c for c in df.columns if c.startswith("motion_outlier")]
    cols = [c for c in hmp24 + acomp + cosine + outliers if c in df.columns]
    return df[cols].fillna(0).values

# ─────────────────────────────────────────────────────────────
# ATLAS SETUP
# ─────────────────────────────────────────────────────────────
def setup_atlases(ref_bold_path):
    global NETWORK_MASKS
    print("Loading Harvard-Oxford atlases...")
    ho_cort = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    ho_sub  = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    ref_img   = image.index_img(image.load_img(str(ref_bold_path)), 0)
    cort_data = image.resample_to_img(ho_cort.maps, ref_img,
                                       interpolation="nearest").get_fdata()
    sub_data  = image.resample_to_img(ho_sub.maps,  ref_img,
                                       interpolation="nearest").get_fdata()

    def roi_mask(data, idx):
        return image.new_img_like(ref_img, (data == idx).astype(np.float32))
    def bilateral_mask(data, l, r):
        return image.new_img_like(ref_img, ((data == l) | (data == r)).astype(np.float32))

    for name, idx in MIDLINE_NODES.items():
        NETWORK_MASKS[name] = roi_mask(cort_data, idx)
    NETWORK_MASKS["Insula"] = roi_mask(cort_data, LATERAL_NODES["Insula"][0])
    for name, (l_idx, r_idx) in LATERAL_NODES.items():
        if name == "Insula": continue
        NETWORK_MASKS[name] = bilateral_mask(sub_data, l_idx, r_idx)

# ─────────────────────────────────────────────────────────────
# FC EXTRACTION 
# ─────────────────────────────────────────────────────────────
def extract_network_connectivity(bold_path, mask_path, conf_path, seed_mask=None):
    confounds = get_confounds(conf_path)
    roi_names = list(NETWORK_MASKS.keys())

    combined = None
    for rn in roi_names:
        md = NETWORK_MASKS[rn].get_fdata() > 0
        combined = md if combined is None else combined | md
    combined_img = image.new_img_like(NETWORK_MASKS[roi_names[0]],
                                      combined.astype(np.float32))

    masker = NiftiMasker(mask_img=combined_img, smoothing_fwhm=SMOOTH_FWHM,
                         detrend=True, standardize="zscore_sample",
                         low_pass=BANDPASS[1], high_pass=BANDPASS[0],
                         t_r=TR, verbose=0)
    all_ts = masker.fit_transform(str(bold_path), confounds=confounds)

    vox_idx = np.where(masker.mask_img_.get_fdata() > 0)
    roi_ts_list = []
    for rn in roi_names:
        in_roi = NETWORK_MASKS[rn].get_fdata()[vox_idx] > 0
        roi_ts_list.append(all_ts[:, in_roi].mean(axis=1))

    roi_ts = np.column_stack(roi_ts_list)
    r_mat  = np.corrcoef(roi_ts.T)
    z_mat  = np.arctanh(np.clip(r_mat, -0.999, 0.999))
    np.fill_diagonal(z_mat, 0)

    seed_fc = None
    if seed_mask is not None:
        bold_ref  = image.index_img(image.load_img(str(bold_path)), 0)
        seed_img  = image.resample_to_img(image.load_img(str(seed_mask)),
                                          bold_ref, interpolation="nearest")
        sm = NiftiMasker(mask_img=seed_img, smoothing_fwhm=None,
                         detrend=True, standardize="zscore_sample",
                         low_pass=BANDPASS[1], high_pass=BANDPASS[0],
                         t_r=TR, verbose=0)
        seed_ts = sm.fit_transform(str(bold_path), confounds=confounds).mean(axis=1)
        n = len(seed_ts)
        seed_fc = {rn: float(np.arctanh(np.clip(
                       np.dot(seed_ts, roi_ts[:, i]) / n, -0.999, 0.999)))
                   for i, rn in enumerate(roi_names)}

    return z_mat, roi_ts, roi_names, seed_fc

# ─────────────────────────────────────────────────────────────
# DATA PIVOT
# ─────────────────────────────────────────────────────────────
def fc_matrix(df_fc, ses, timepoints, rois):
    """(n_roi × n_tp) array of mean Fisher-z values."""
    mat = np.full((len(rois), len(timepoints)), np.nan)
    for j, tp in enumerate(timepoints):
        for i, roi in enumerate(rois):
            v = df_fc[(df_fc["session"] == ses) &
                      (df_fc["timepoint"] == tp) &
                      (df_fc["roi"] == roi)]["fc_z"]
            if len(v):
                mat[i, j] = v.mean()
    return mat


def build_delta_contrast(df_fc, rois):
    """Return delta dict and contrast array."""
    delta = {}
    for ses in SESSIONS:
        full       = fc_matrix(df_fc, ses, TIMEPOINTS, rois)  # (n_roi, 4)
        delta[ses] = full[:, 1:] - full[:, 0:1]               # (n_roi, 3)
    contrast = delta["ses-exp"] - delta["ses-con"]             # (n_roi, 3)
    return delta, contrast

# ─────────────────────────────────────────────────────────────
# PLOT 1 — CONNECTIVITY MATRICES
# ─────────────────────────────────────────────────────────────
def plot_connectivity_matrices(conn_matrices, roi_names_list, sub, out_dir):
    short_labels = [_roi_label(r) for r in roi_names_list]
    n_tp  = len(TIMEPOINTS)
    n_ses = len(SESSIONS)

    fig = plt.figure(figsize=(4.2 * n_tp, 4.5 * n_ses + 1.0), facecolor="white")
    gs  = GridSpec(n_ses + 1, n_tp, figure=fig,
                   height_ratios=[0.08] + [1.0] * n_ses,
                   hspace=0.35, wspace=0.25,
                   top=0.93, bottom=0.08, left=0.07, right=0.98)

    fig.suptitle(f"{sub} — sgACC network connectivity matrix",
                 fontsize=15, fontweight="bold", y=0.98)
    cax    = fig.add_subplot(gs[0, :])
    im_ref = None

    for r, ses in enumerate(SESSIONS):
        for c, acq in enumerate(TIMEPOINTS):
            ax  = fig.add_subplot(gs[r + 1, c])
            key = (sub, ses, acq)
            if key not in conn_matrices:
                ax.set_visible(False); continue
            im = ax.imshow(conn_matrices[key], cmap="RdBu_r",
                           vmin=-1, vmax=1, aspect="auto")
            im_ref = im
            n = len(short_labels)
            ax.set_xticks(range(n))
            ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(n))
            ax.set_yticklabels(short_labels, fontsize=8)
            if r == 0: ax.set_title(TP_LABELS[acq], fontsize=10, pad=6)
            if c == 0: ax.set_ylabel(SES_LABELS[ses], fontsize=9,
                                     fontweight="bold", labelpad=6)

    if im_ref is not None:
        cb = fig.colorbar(im_ref, cax=cax, orientation="horizontal")
        cb.set_label("Fisher z", fontsize=10, labelpad=4)
        cb.ax.tick_params(labelsize=9)
        cax.xaxis.set_label_position("top")
        cax.xaxis.tick_top()
        
        # Shift colorbar and its label down slightly to avoid crowding the title
        pos = cax.get_position()
        cax.set_position([pos.x0, pos.y0 - 0.025, pos.width, pos.height])

    savefig(fig, out_dir / f"{sub}_connectivity_matrix")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────
# PLOT 2 — DELTA HEATMAPS
# ─────────────────────────────────────────────────────────────
def plot_delta_heatmaps(delta, contrast, sub, out_dir):
    rois       = NET_ROIS
    short_rois = [_roi_label(r) for r in rois]
    post_lbl   = [TP_SHORT[tp] for tp in POST_TPS]
    n_rois     = len(rois)

    # Symmetric colour limits — shared for Exp/Ctrl; separate for contrast
    all_dc   = np.concatenate([delta["ses-exp"].ravel(), delta["ses-con"].ravel()])
    lim_dc   = max(abs(np.nanmax(all_dc)),   abs(np.nanmin(all_dc)))   or 0.5
    lim_cont = max(abs(np.nanmax(contrast)), abs(np.nanmin(contrast))) or 0.5

    cell_h = 0.58
    fig_h  = max(6.0, cell_h * n_rois + 2.5)
    fig_w  = max(6.5, 1.4 * 3 + 4.0)    # 3 columns

    def _heatmap(data, title, clabel, clim, stem):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
        im = ax.imshow(data, cmap="RdBu_r", vmin=-clim, vmax=clim,
                       aspect="auto", interpolation="nearest")
        ax.set_xticks(range(3))
        ax.set_xticklabels(post_lbl, fontsize=11)
        ax.set_yticks(range(n_rois))
        ax.set_yticklabels(short_rois, fontsize=10)
        ax.tick_params(axis="both", pad=5)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04, aspect=22)
        cb.set_label(clabel, fontsize=10, labelpad=6)
        cb.ax.tick_params(labelsize=9)
        cb.locator = mticker.MaxNLocator(nbins=6, symmetric=True)
        cb.update_ticks()

        thresh = 0.55 * clim
        for i in range(n_rois):
            for j in range(3):
                v = data[i, j]
                if np.isnan(v): continue
                tc = "white" if abs(v) >= thresh else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color=tc, fontweight="bold")

        fig.tight_layout(pad=0.5)
        savefig(fig, out_dir / stem)
        plt.close(fig)

    _heatmap(delta["ses-exp"],
             "Experimental ΔFC (Post − Pre)",
             "ΔFC (Fisher z)", lim_dc,
             f"{sub}_sgacc_delta_heatmap_exp")

    _heatmap(delta["ses-con"],
             "Control ΔFC (Post − Pre)",
             "ΔFC (Fisher z)", lim_dc,
             f"{sub}_sgacc_delta_heatmap_ctrl")

    _heatmap(contrast,
             "Focused TUS Effect (ΔExp − ΔCtrl)",
             "ΔExp − ΔCtrl (Fisher z)", lim_cont,
             f"{sub}_sgacc_delta_contrast_exp_minus_ctrl")

# ─────────────────────────────────────────────────────────────
# PLOT 3 — CONTRAST TRAJECTORY  (primary result figure)
# ─────────────────────────────────────────────────────────────
def plot_contrast_trajectory(contrast, highlight_colors, sub, out_dir):
    """
    All-ROI trajectory of ΔExp − ΔCtrl over +15, +30, +45.
    Top-3 positive and top-3 negative ROIs auto-highlighted.
    Direct end-labels at +45 with vertical offset to avoid overlap.
    Saved as PNG + SVG.
    """
    rois      = NET_ROIS
    tp_x      = [0, 1, 2]
    tp_labels = [TP_SHORT[t] for t in POST_TPS]
    gray      = "#bbbbbb"

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.set_title(
        f"{sub} — Focused TUS-specific sgACC connectivity effect over time",
        fontsize=12, fontweight="bold", pad=10)

    # ── 1. Gray background lines (all ROIs) ──────────────────
    for i, roi in enumerate(rois):
        vals = contrast[i, :]
        ax.plot(tp_x, vals, "-o", lw=1.0, ms=4,
                color=gray, alpha=0.45, zorder=1)

    # ── 2. Highlighted ROIs ───────────────────────────────────
    end_labels = []   # (y_val_at_45, label_text, color)
    for roi, color in highlight_colors.items():
        i    = rois.index(roi)
        vals = contrast[i, :]
        ax.plot(tp_x, vals, "-o", lw=2.5, ms=7,
                color=color, zorder=3, label=roi)
        end_labels.append((float(vals[-1]), _roi_label(roi), color))

    # ── 3. Direct labels at +45 with collision avoidance ─────
    # Sort by y value, then nudge overlapping labels apart
    end_labels.sort(key=lambda x: x[0])
    min_gap = 0.025                     # minimum y gap between labels
    placed  = []
    for y_raw, lbl, col in end_labels:
        y_adj = y_raw
        for y_prev in placed:
            if abs(y_adj - y_prev) < min_gap:
                y_adj = y_prev + min_gap * np.sign(y_raw - y_prev + 1e-9)
        placed.append(y_adj)
        ax.annotate(lbl,
                    xy=(tp_x[-1], y_raw),
                    xytext=(tp_x[-1] + 0.12, y_adj),
                    fontsize=8.5, color=col, fontweight="bold",
                    va="center",
                    arrowprops=dict(arrowstyle="-", color=col,
                                   lw=0.8, alpha=0.6)
                    if abs(y_adj - y_raw) > 0.01 else None)

    ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.5, zorder=2)
    ax.set_xticks(tp_x)
    ax.set_xticklabels(tp_labels, fontsize=11)
    ax.set_xlim(-0.3, 2.85)            # right margin for end-labels
    ax.set_xlabel("Post-TUS timepoint", fontsize=11, labelpad=5)
    ax.set_ylabel("ΔExp − ΔCtrl (Fisher z)", fontsize=11, labelpad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    # Legend — only highlighted ROIs, placed outside right edge
    ax.legend(fontsize=9, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
              framealpha=0.85, title="Highlighted ROIs", title_fontsize=8)

    fig.tight_layout(rect=[0, 0, 0.85, 1.0])
    # Save PNG + SVG
    savefig(fig, out_dir / f"{sub}_sgacc_contrast_trajectory", extra_fmts=("svg",))
    plt.close(fig)

# ─────────────────────────────────────────────────────────────
# PLOT 4 — SPAGHETTI  (absolute FC + ΔFC, data-driven highlights)
# ─────────────────────────────────────────────────────────────
def plot_spaghetti(df_fc, highlight_colors, sub, out_dir):
    rois  = NET_ROIS
    gray  = "#c0c0c0"

    # ── A. Absolute FC ──────────────────────────────────────────
    tp_list   = TIMEPOINTS
    tp_x      = list(range(len(tp_list)))
    tp_labels = [TP_SHORT[t] for t in tp_list]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True, facecolor="white")
    fig.suptitle(f"{sub} — sgACC-to-ROI absolute FC over time",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, ses in zip(axes, SESSIONS):
        # Gray background lines
        for roi in rois:
            if roi in highlight_colors: continue
            vals = [df_fc[(df_fc["session"] == ses) &
                          (df_fc["timepoint"] == tp) &
                          (df_fc["roi"] == roi)]["fc_z"].mean()
                    for tp in tp_list]
            ax.plot(tp_x, vals, "-o", lw=1.0, ms=4,
                    color=gray, alpha=0.55, zorder=1)
        # Highlighted
        for roi, color in highlight_colors.items():
            vals = [df_fc[(df_fc["session"] == ses) &
                          (df_fc["timepoint"] == tp) &
                          (df_fc["roi"] == roi)]["fc_z"].mean()
                    for tp in tp_list]
            ax.plot(tp_x, vals, "-o", lw=2.5, ms=7,
                    color=color, zorder=3, label=_roi_label(roi))

        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.4)
        ax.set_xticks(tp_x)
        ax.set_xticklabels(tp_labels, fontsize=10)
        ax.set_xlabel("Timepoint", fontsize=10, labelpad=4)
        if ax is axes[0]:
            ax.set_ylabel("sgACC FC (Fisher z)", fontsize=10, labelpad=6)
        ax.set_title(SES_LABELS[ses], fontsize=11, fontweight="bold", pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
                  framealpha=0.85,
                  title_fontsize=8)

    fig.tight_layout(rect=[0, 0, 0.86, 0.97], w_pad=3)
    savefig(fig, out_dir / f"{sub}_sgacc_spaghetti_absolute_fc")
    plt.close(fig)

    # ── B. ΔFC ──────────────────────────────────────────────────
    post_list   = POST_TPS
    tp_x_d      = list(range(len(post_list)))
    tp_labels_d = [TP_SHORT[t] for t in post_list]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, facecolor="white")
    fig.suptitle(f"{sub} — sgACC-to-ROI ΔFC (relative to Pre)",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, ses in zip(axes, SESSIONS):
        pre_vals = {roi: df_fc[(df_fc["session"] == ses) &
                               (df_fc["timepoint"] == "preTUS15") &
                               (df_fc["roi"] == roi)]["fc_z"].mean()
                    for roi in rois}
        # Gray
        for roi in rois:
            if roi in highlight_colors: continue
            dv = [df_fc[(df_fc["session"] == ses) &
                        (df_fc["timepoint"] == tp) &
                        (df_fc["roi"] == roi)]["fc_z"].mean() - pre_vals[roi]
                  for tp in post_list]
            ax.plot(tp_x_d, dv, "-o", lw=1.0, ms=4,
                    color=gray, alpha=0.55, zorder=1)
        # Highlighted
        for roi, color in highlight_colors.items():
            dv = [df_fc[(df_fc["session"] == ses) &
                        (df_fc["timepoint"] == tp) &
                        (df_fc["roi"] == roi)]["fc_z"].mean() - pre_vals[roi]
                  for tp in post_list]
            ax.plot(tp_x_d, dv, "-o", lw=2.5, ms=7,
                    color=color, zorder=3, label=_roi_label(roi))

        ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.5)
        ax.set_xticks(tp_x_d)
        ax.set_xticklabels(tp_labels_d, fontsize=10)
        ax.set_xlabel("Timepoint", fontsize=10, labelpad=4)
        if ax is axes[0]:
            ax.set_ylabel("ΔFC (Post − Pre, Fisher z)", fontsize=10, labelpad=6)
        ax.set_title(SES_LABELS[ses], fontsize=11, fontweight="bold", pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
                  framealpha=0.85,
                  title_fontsize=8)

    fig.tight_layout(rect=[0, 0, 0.86, 0.97], w_pad=3)
    savefig(fig, out_dir / f"{sub}_sgacc_spaghetti_delta_fc")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────
# PLOT 5 — RADAR (Experimental, all 4 timepoints overlaid)
# ─────────────────────────────────────────────────────────────
def plot_radar_exp_timepoints(df_fc, sub, out_dir):
    """Radar with true signed radial axis — center = r_min (negative ok)."""
    rois       = NET_ROIS
    short_rois = [_roi_label(r) for r in rois]
    N          = len(rois)
    angles     = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_cl  = angles + angles[:1]

    # Collect all Exp FC values to set radial limits
    all_exp = np.array([
        df_fc[(df_fc["session"] == "ses-exp") &
              (df_fc["timepoint"] == tp) &
              (df_fc["roi"] == roi)]["fc_z"].mean()
        for roi in rois for tp in TIMEPOINTS
    ])
    data_min = np.nanmin(all_exp)
    data_max = np.nanmax(all_exp)

    # Round outward to nearest 0.1 for clean grid ticks
    tick_step = 0.1
    r_min = np.floor(data_min / tick_step) * tick_step - tick_step  # one step below min
    r_max = np.ceil (data_max / tick_step) * tick_step + tick_step  # one step above max

    tp_styles = {
        "preTUS15":  {"color": "#444444", "ls": "--", "lw": 2.0, "alpha": 0.80},
        "postTUS15": {"color": "#e05c5c", "ls": "-",  "lw": 2.5, "alpha": 0.95},
        "postTUS30": {"color": "#e09b5c", "ls": "-",  "lw": 2.5, "alpha": 0.95},
        "postTUS45": {"color": "#8e5ce0", "ls": "-",  "lw": 2.5, "alpha": 0.95},
    }

    fig = plt.figure(figsize=(11, 10), facecolor="white")
    ax  = fig.add_axes([0.08, 0.08, 0.62, 0.82], projection="polar")

    # Fills first, then lines on top (zorder keeps them above grid)
    for tp in TIMEPOINTS:
        vals  = np.array([
            df_fc[(df_fc["session"] == "ses-exp") &
                  (df_fc["timepoint"] == tp) &
                  (df_fc["roi"] == roi)]["fc_z"].mean()
            for roi in rois
        ])
        vs_cl = vals.tolist() + [vals[0]]
        ax.fill(angles_cl, vs_cl,
                color=tp_styles[tp]["color"], alpha=0.35, zorder=2)
    for tp in TIMEPOINTS:
        vals  = np.array([
            df_fc[(df_fc["session"] == "ses-exp") &
                  (df_fc["timepoint"] == tp) &
                  (df_fc["roi"] == roi)]["fc_z"].mean()
            for roi in rois
        ])
        vs_cl = vals.tolist() + [vals[0]]
        st    = tp_styles[tp]
        ax.plot(angles_cl, vs_cl,
                ls=st["ls"], lw=st["lw"], color=st["color"],
                alpha=st["alpha"], label=TP_SHORT[tp], zorder=4)

    ax.set_xticks(angles)
    ax.set_xticklabels(short_rois, fontsize=10)
    ax.set_ylim(r_min, r_max)

    # Grid circles at every tick_step
    grid_vals = np.arange(r_min, r_max + tick_step / 2, tick_step)
    ax.set_yticks(grid_vals)
    ax.set_yticklabels([f"{v:.1f}" for v in grid_vals],
                       fontsize=8, color="gray")
    ax.yaxis.set_tick_params(pad=3)
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.45)

    ax.set_title(f"{sub} — Experimental sgACC connectivity fingerprint over time",
                 fontsize=11, fontweight="bold", pad=22)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05),
              bbox_transform=ax.transAxes,
              fontsize=10, framealpha=0.8,
              title="Timepoint", title_fontsize=9)

    savefig(fig, out_dir / f"{sub}_sgacc_radar_exp_timepoints")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────
# CSV EXPORT
# ─────────────────────────────────────────────────────────────
def export_csvs(df_fc, delta, contrast, sub, out_dir):
    rois = NET_ROIS
    cols = [TP_SHORT[t] for t in POST_TPS]

    # 1. Absolute FC (long format)
    rows = []
    for ses in SESSIONS:
        for tp in TIMEPOINTS:
            for roi in rois:
                v = df_fc[(df_fc["session"] == ses) &
                          (df_fc["timepoint"] == tp) &
                          (df_fc["roi"] == roi)]["fc_z"]
                rows.append({"session": ses, "timepoint": tp,
                             "roi": roi, "fc_z": v.mean() if len(v) else np.nan})
    pd.DataFrame(rows).to_csv(out_dir / f"{sub}_sgacc_absolute_fc.csv", index=False)

    # 2. ΔFC Exp
    pd.DataFrame(delta["ses-exp"], index=rois, columns=cols)\
      .rename_axis("roi").to_csv(out_dir / f"{sub}_sgacc_delta_fc_exp.csv")

    # 3. ΔFC Ctrl
    pd.DataFrame(delta["ses-con"], index=rois, columns=cols)\
      .rename_axis("roi").to_csv(out_dir / f"{sub}_sgacc_delta_fc_ctrl.csv")

    # 4. Contrast
    pd.DataFrame(contrast, index=rois, columns=cols)\
      .rename_axis("roi").to_csv(out_dir / f"{sub}_sgacc_delta_contrast_exp_minus_ctrl.csv")

    print(f"  CSV tables written to {out_dir}")

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
            if p: hemi_masks[hemi] = p
        if "L" in hemi_masks and "R" in hemi_masks:
            SUBJECT_MASKS[sub] = make_bilateral_seed_mask(
                sub, hemi_masks["L"], hemi_masks["R"], out_dir)
        elif hemi_masks:
            SUBJECT_MASKS[sub] = list(hemi_masks.values())[0]

    # 2. Atlas setup
    bold_ref = None
    for ses in SESSIONS:
        for acq in TIMEPOINTS:
            bold, _, _ = find_run(sub, ses, acq)
            if bold: bold_ref = bold; break
        if bold_ref: break
    if not bold_ref:
        print("No BOLD data found. Exit.")
        return
    setup_atlases(bold_ref)

    # 3. Compute FC
    net_records    = []
    conn_matrices  = {}
    roi_names_list = None
    seed_mask      = SUBJECT_MASKS.get(sub)

    for ses in SESSIONS:
        for acq in TIMEPOINTS:
            bold, mask, conf = find_run(sub, ses, acq)
            if not bold: continue
            print(f"  Processing {sub} | {ses} | {acq} ...")
            z_mat, _, roi_names, seed_fc = extract_network_connectivity(
                bold, mask, conf, seed_mask)
            conn_matrices[(sub, ses, acq)] = z_mat
            roi_names_list = roi_names
            if seed_fc:
                for roi_name, fc_z in seed_fc.items():
                    net_records.append({"subject": sub, "session": ses,
                                        "timepoint": acq, "roi": roi_name,
                                        "fc_z": fc_z})

    df_fc = pd.DataFrame(net_records)
    if df_fc.empty:
        print("No FC data computed. Check inputs.")
        return

    # 4. Derive delta & contrast, auto-select highlights
    delta, contrast = build_delta_contrast(df_fc, NET_ROIS)
    highlight_colors = get_contrast_highlights(contrast, NET_ROIS, n_top=3)
    print(f"  Auto-highlighted ROIs: {list(highlight_colors.keys())}")

    # 5. Plots
    print("Generating plots...")

    if roi_names_list:
        plot_connectivity_matrices(conn_matrices, roi_names_list, sub, out_dir)

    plot_delta_heatmaps(delta, contrast, sub, out_dir)

    plot_contrast_trajectory(contrast, highlight_colors, sub, out_dir)

    plot_spaghetti(df_fc, highlight_colors, sub, out_dir)

    plot_radar_exp_timepoints(df_fc, sub, out_dir)

    # 6. CSV
    export_csvs(df_fc, delta, contrast, sub, out_dir)
    df_fc.to_csv(out_dir / f"{sub}_sgacc_fc_values.tsv", sep="\t", index=False)

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
