#!/usr/bin/env python3
"""
CITRUS — sgACC single-subject functional connectivity
Converted from citrus_sgacc_fc_v2.ipynb

Goal: Evaluate sgACC (BA25) resting-state connectivity changes before and after TUS,
comparing exp-focused vs con-defocused sessions across 4 timepoints.

Usage: python citrus_sgacc_fc_v2.py
"""

import os
import subprocess
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image
from nilearn.maskers import NiftiMasker

# Use Agg backend for non-interactive plotting (important for terminal/servers)
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
MEPREP_ROOT  = Path("/Volumes/Extreme SSD/THESIS-MSC/MEPrep output")
CITRUS_INPUT = Path("/Users/hoaithunguyen/Projects/Master-thesis/CITRUS/data/input")
OUT_DIR      = Path("/Users/hoaithunguyen/Projects/Master-thesis/CITRUS/derivatives/rs_fmri_v2")
FSL_BIN      = Path("/Users/hoaithunguyen/fsl/bin")

# Subjects to process
SUBJECTS   = ["sub-05"]   # add others when MEPrep is done
SESSIONS   = ["ses-exp", "ses-con"]
TIMEPOINTS = ["preTUS15", "postTUS15", "postTUS30", "postTUS45"]

# Acquisition parameters
TR          = 1.50          # seconds
SMOOTH_FWHM = 6.0           # mm
BANDPASS    = (0.01, 0.10)  # Hz

# Harvard-Oxford network nodes (L+R averaged for lateralised)
MIDLINE_NODES = {
    "Frontal_Medial":  25,   # Frontal Medial Cortex
    "Paracingulate":   28,   # Paracingulate Gyrus
    "ACC":             29,   # Cingulate Gyrus, anterior division
    "PCC":             30,   # Cingulate Gyrus, posterior division
    "Precuneus":       31,   # Precuneous Cortex
}

LATERAL_NODES = {
    "Insula":      (2,  2),    # Insular Cortex
    "Hippocampus": (9,  19),   # Left/Right Hippocampus
    "Amygdala":    (10, 20),   # Left/Right Amygdala
    "Thalamus":    (4,  15),   # Left/Right Thalamus
    "Accumbens":   (11, 21),   # Left/Right Accumbens
}

BOLD_PROC  = "pmeica"
BOLD_SPACE = "MNI152NLin2009cAsym"
NET_ROIS   = list(MIDLINE_NODES.keys()) + list(LATERAL_NODES.keys())

SES_COLORS = {"ses-exp": "#e05c5c", "ses-con": "#5c7de0"}
SES_LABELS = {"ses-exp": "Experimental (focused)", "ses-con": "Control (defocused)"}
TP_LABELS  = {"preTUS15": "Pre\n(−15 min)", "postTUS15": "Post\n(+15 min)",
              "postTUS30": "Post\n(+30 min)", "postTUS45": "Post\n(+45 min)"}

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# FSL WRAPPERS / REGISTRATION
# ─────────────────────────────────────────────────────────────
FSLDIR = FSL_BIN.parent

def _flirt(args):
    """Run flirt with FSL env vars set explicitly."""
    fsl_env = {**os.environ,
               "FSLDIR": str(FSLDIR),
               "FSLOUTPUTTYPE": "NIFTI_GZ",
               "PATH": str(FSL_BIN) + ":" + os.environ.get("PATH", "")}
    return subprocess.run([str(FSL_BIN / "flirt")] + args,
                          capture_output=True, text=True, env=fsl_env)

def register_kplan_to_mni(sub: str):
    kplan_t1 = CITRUS_INPUT / sub / f"{sub}_T1w_kplan.nii.gz"
    mni_t1   = MEPREP_ROOT / "ses-intake" / f"{sub}_ses-intake_acq-HCP_space-{BOLD_SPACE}_desc-preproc_T1w.nii.gz"
    out_dir  = OUT_DIR / "seeds" / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    mat_path = out_dir / f"{sub}_kplan_to_MNI.mat"

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

def warp_mask_to_mni(sub: str, hemi: str, mat_path) -> Path:
    mask_kplan = CITRUS_INPUT / sub / f"sgACC_BA25_{hemi}_kplan.nii.gz"
    mni_t1     = MEPREP_ROOT / "ses-intake" / f"{sub}_ses-intake_acq-HCP_space-{BOLD_SPACE}_desc-preproc_T1w.nii.gz"
    out_dir    = OUT_DIR / "seeds" / sub
    out_mask   = out_dir / f"{sub}_sgACC_BA25_{hemi}_MNI.nii.gz"

    if out_mask.exists():
        return out_mask
    if not mask_kplan.exists():
        print(f"  [{sub}] MISSING kplan mask: {mask_kplan}")
        return None
    if not mni_t1.exists():
        print(f"  [{sub}] MISSING MNI T1w: {mni_t1}")
        return None

    result = _flirt(["-in", str(mask_kplan), "-ref", str(mni_t1),
                     "-init", str(mat_path), "-applyxfm",
                     "-interp", "nearestneighbour", "-out", str(out_mask)])
    if result.returncode != 0:
        print(f"  [{sub}] mask warp FAILED:\n{result.stderr}")
        return None
    return out_mask

def make_bilateral_seed_mask(sub: str, mask_L: Path, mask_R: Path) -> Path:
    out_dir  = OUT_DIR / "seeds" / sub
    out_path = out_dir / f"{sub}_sgACC_BA25_bilateral_MNI.nii.gz"
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
    hmp = ["trans_x","trans_y","trans_z","rot_x","rot_y","rot_z"]
    hmp24 = hmp + [f"{c}_derivative1" for c in hmp] + \
                  [f"{c}_power2"       for c in hmp] + \
                  [f"{c}_derivative1_power2" for c in hmp]
    acomp  = [f"a_comp_cor_{i:02d}" for i in range(5)]
    cosine = [c for c in df.columns if c.startswith("cosine")]
    outliers = [c for c in df.columns if c.startswith("motion_outlier")]
    cols = [c for c in hmp24 + acomp + cosine + outliers if c in df.columns]
    return df[cols].fillna(0).values

# ─────────────────────────────────────────────────────────────
# NETWORK ROI EXTRACTION
# ─────────────────────────────────────────────────────────────
NETWORK_MASKS = {}

def setup_atlases(ref_bold_path):
    global NETWORK_MASKS
    print("Loading Harvard-Oxford atlases...")
    ho_cort = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    ho_sub  = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    
    ref_img = image.index_img(image.load_img(str(ref_bold_path)), 0)
    cort_res = image.resample_to_img(ho_cort.maps, ref_img, interpolation="nearest")
    sub_res  = image.resample_to_img(ho_sub.maps,  ref_img, interpolation="nearest")
    cort_data = cort_res.get_fdata()
    sub_data  = sub_res.get_fdata()

    def make_roi_mask(label_img_data, label_idx, ref_img):
        mask = (label_img_data == label_idx).astype(np.float32)
        return image.new_img_like(ref_img, mask)

    def make_bilateral_mask(data, l_idx, r_idx, ref_img):
        mask = ((data == l_idx) | (data == r_idx)).astype(np.float32)
        return image.new_img_like(ref_img, mask)

    for name, idx in MIDLINE_NODES.items():
        NETWORK_MASKS[name] = make_roi_mask(cort_data, idx, ref_img)

    insula_idx = LATERAL_NODES["Insula"][0]
    NETWORK_MASKS["Insula"] = make_roi_mask(cort_data, insula_idx, ref_img)

    for name, (l_idx, r_idx) in LATERAL_NODES.items():
        if name == "Insula": continue
        NETWORK_MASKS[name] = make_bilateral_mask(sub_data, l_idx, r_idx, ref_img)

def extract_network_connectivity(bold_path, mask_path, conf_path, seed_mask=None):
    """
    Optimized network extraction: Loads BOLD once and extracts all ROI signals.
    """
    confounds = get_confounds(conf_path)
    roi_names = list(NETWORK_MASKS.keys())
    
    # 1. Create a combined mask of all network ROIs to load data once
    combined_mask_data = None
    for roi_name in roi_names:
        mask_data = NETWORK_MASKS[roi_name].get_fdata() > 0
        if combined_mask_data is None:
            combined_mask_data = mask_data
        else:
            combined_mask_data = combined_mask_data | mask_data
    
    combined_mask_img = image.new_img_like(NETWORK_MASKS[roi_names[0]], combined_mask_data.astype(np.float32))
    
    # 2. Extract signals from all network voxels in one go
    masker = NiftiMasker(
        mask_img=combined_mask_img,
        smoothing_fwhm=SMOOTH_FWHM,
        detrend=True,
        standardize="zscore_sample",
        low_pass=BANDPASS[1],
        high_pass=BANDPASS[0],
        t_r=TR,
        verbose=0
    )
    
    all_voxels_ts = masker.fit_transform(str(bold_path), confounds=confounds)
    
    # 3. Average voxels belonging to each ROI
    # We need to know which voxel index in the masker output belongs to which ROI.
    roi_ts_list = []
    # Masker.mask_img_ is the actual mask used.
    mask_voxel_indices = np.where(masker.mask_img_.get_fdata() > 0)
    
    for roi_name in roi_names:
        roi_mask_data = NETWORK_MASKS[roi_name].get_fdata() > 0
        # Voxel indices in the ROI that are also in the masker
        # (This is guaranteed since combined_mask is union of all ROIs)
        in_roi = roi_mask_data[mask_voxel_indices]
        roi_ts = all_voxels_ts[:, in_roi].mean(axis=1)
        roi_ts_list.append(roi_ts)
    
    roi_ts = np.column_stack(roi_ts_list)
    r_mat = np.corrcoef(roi_ts.T)
    z_mat = np.arctanh(np.clip(r_mat, -0.999, 0.999))
    np.fill_diagonal(z_mat, 0)

    # 4. Subject-specific sgACC seed -> ROIs
    seed_fc = None
    if seed_mask is not None:
        bold_ref = image.index_img(image.load_img(str(bold_path)), 0)
        seed_img = image.resample_to_img(image.load_img(str(seed_mask)), bold_ref, interpolation="nearest")
        seed_masker = NiftiMasker(mask_img=seed_img, smoothing_fwhm=None,
                                 detrend=True, standardize="zscore_sample",
                                 low_pass=BANDPASS[1], high_pass=BANDPASS[0],
                                 t_r=TR, verbose=0)
        seed_ts = seed_masker.fit_transform(str(bold_path), confounds=confounds).mean(axis=1)
        
        n_samples = len(seed_ts)
        # Since both are z-scored, correlation is simple dot product
        seed_fc = {roi_name: float(np.arctanh(np.clip(np.dot(seed_ts, roi_ts[:, i]) / n_samples, -0.999, 0.999)))
                   for i, roi_name in enumerate(roi_names)}
    
    return z_mat, roi_ts, roi_names, seed_fc

# ─────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────
def main():
    print(f"Output directory: {OUT_DIR}")
    
    # 1. Prepare Seed Masks
    SUBJECT_MASKS = {}
    for sub in SUBJECTS:
        mat = register_kplan_to_mni(sub)
        if mat is None: continue
        hemi_masks = {}
        for hemi in ["L", "R"]:
            p = warp_mask_to_mni(sub, hemi, mat)
            if p: hemi_masks[hemi] = p
        if "L" in hemi_masks and "R" in hemi_masks:
            bil = make_bilateral_seed_mask(sub, hemi_masks["L"], hemi_masks["R"])
            SUBJECT_MASKS[sub] = bil
        elif hemi_masks:
            SUBJECT_MASKS[sub] = list(hemi_masks.values())[0]

    # 2. Setup Atlases using first available BOLD
    bold_ref = None
    for sub in SUBJECTS:
        for ses in SESSIONS:
            for acq in TIMEPOINTS:
                bold, _, _ = find_run(sub, ses, acq)
                if bold: bold_ref = bold; break
            if bold_ref: break
        if bold_ref: break
    
    if not bold_ref:
        print("No BOLD data found. Exit.")
        return
    
    setup_atlases(bold_ref)

    # 3. Compute Connectivity
    net_records = []
    conn_matrices = {}
    roi_names_list = None

    for sub in SUBJECTS:
        seed_mask = SUBJECT_MASKS.get(sub)
        for ses in SESSIONS:
            for acq in TIMEPOINTS:
                bold, mask, conf = find_run(sub, ses, acq)
                if not bold: continue

                print(f"Processing {sub} | {ses} | {acq} ...")
                
                # Network FC (ROI-to-ROI and Seed-to-ROI)
                z_mat, _, roi_names, seed_fc = extract_network_connectivity(bold, mask, conf, seed_mask)
                conn_matrices[(sub, ses, acq)] = z_mat
                roi_names_list = roi_names
                
                if seed_fc:
                    for roi_name, fc_z in seed_fc.items():
                        net_records.append({"subject": sub, "session": ses, "timepoint": acq, "roi": roi_name, "fc_z": fc_z})

    df_fc = pd.DataFrame(net_records)
    if df_fc.empty:
        print("No FC data computed. Check inputs.")
        return

    # 4. PLOTTING
    print("Generating plots...")
    sub = SUBJECTS[0]
    
    # 4a. Connectivity Matrix
    n_tp = len(TIMEPOINTS)
    fig, axes = plt.subplots(len(SESSIONS), n_tp, figsize=(5*n_tp, 9), facecolor="white")
    fig.suptitle(f"{sub} — sgACC network connectivity matrix (Fisher z)", fontsize=13, fontweight="bold")
    for r, ses in enumerate(SESSIONS):
        for c, acq in enumerate(TIMEPOINTS):
            ax = axes[r, c]
            key = (sub, ses, acq)
            if key not in conn_matrices: ax.set_visible(False); continue
            im = ax.imshow(conn_matrices[key], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(roi_names_list)))
            ax.set_xticklabels(roi_names_list, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(roi_names_list)))
            ax.set_yticklabels(roi_names_list, fontsize=7)
            if r == 0: ax.set_title(TP_LABELS[acq], fontsize=10)
            if c == 0: ax.set_ylabel(SES_LABELS[ses], fontsize=9, fontweight="bold")
    fig.colorbar(im, ax=axes, shrink=0.5, label="Fisher z", pad=0.02)
    plt.savefig(OUT_DIR / f"{sub}_connectivity_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4b. Temporal Line Plots
    n_rois = len(NET_ROIS)
    n_cols = 4
    n_rows = int(np.ceil(n_rois / n_cols))
    tp_x = list(range(len(TIMEPOINTS)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows), facecolor="white")
    fig.suptitle("Single-subject sgACC connectivity to network nodes over time", fontsize=13, fontweight="bold")
    axes = axes.flatten()
    for ax_i, roi in enumerate(NET_ROIS):
        ax = axes[ax_i]
        roi_df = df_fc[df_fc["roi"] == roi]
        for ses in SESSIONS:
            ses_df = roi_df[roi_df["session"] == ses]
            vals = [ses_df[ses_df["timepoint"] == tp]["fc_z"].mean() for tp in TIMEPOINTS]
            ax.plot(tp_x, vals, "o-", color=SES_COLORS[ses], lw=2.5, label=SES_LABELS[ses])
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.set_xticks(tp_x)
        ax.set_xticklabels([TP_LABELS[t] for t in TIMEPOINTS], fontsize=8)
        ax.set_title(roi, fontsize=11, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        if ax_i == 0: ax.legend(fontsize=8, loc="upper right")
    for ax in axes[n_rois:]: ax.set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_DIR / "sgacc_temporal_fc_single_subject.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4c. Radar Charts
    N = len(NET_ROIS)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig, axes = plt.subplots(1, len(TIMEPOINTS), figsize=(20, 5), subplot_kw={"projection": "polar"}, facecolor="white")
    r_max = df_fc["fc_z"].abs().quantile(0.95) * 1.2 if len(df_fc) > 0 else 1.0
    for ax, tp in zip(axes, TIMEPOINTS):
        for ses in SESSIONS:
            vals = [df_fc[(df_fc["session"]==ses) & (df_fc["timepoint"]==tp) & (df_fc["roi"]==roi)]["fc_z"].mean() for roi in NET_ROIS]
            vals_closed = vals + vals[:1]
            ax.plot(angles, vals_closed, "-o", color=SES_COLORS[ses], lw=2, markersize=4, label=SES_LABELS[ses])
            ax.fill(angles, vals_closed, color=SES_COLORS[ses], alpha=0.1)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(NET_ROIS, fontsize=8)
        ax.set_ylim(-r_max, r_max); ax.set_title(TP_LABELS[tp], fontsize=10, fontweight="bold", pad=14)
    plt.savefig(OUT_DIR / "sgacc_radar_single_subject.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4d. Delta FC Radar (post - pre)
    post_tps = [tp for tp in TIMEPOINTS if tp != "preTUS15"]
    fig, axes = plt.subplots(1, len(post_tps), figsize=(16, 5), subplot_kw={"projection": "polar"}, facecolor="white")
    fig.suptitle("Δ sgACC connectivity from pre-TUS baseline", fontsize=13, fontweight="bold", y=1.03)
    for ax, tp in zip(axes, post_tps):
        for ses in SESSIONS:
            pre  = [df_fc[(df_fc["session"]==ses) & (df_fc["timepoint"]=="preTUS15") & (df_fc["roi"]==roi)]["fc_z"].mean() for roi in NET_ROIS]
            post = [df_fc[(df_fc["session"]==ses) & (df_fc["timepoint"]==tp) & (df_fc["roi"]==roi)]["fc_z"].mean() for roi in NET_ROIS]
            delta = [p - pr for p, pr in zip(post, pre)]
            delta_closed = delta + delta[:1]
            ax.plot(angles, delta_closed, "-o", color=SES_COLORS[ses], lw=2, markersize=4, label=SES_LABELS[ses])
            ax.fill(angles, delta_closed, color=SES_COLORS[ses], alpha=0.1)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(NET_ROIS, fontsize=8)
        ax.set_ylim(-0.6, 0.6); ax.set_yticks([-0.3, 0, 0.3])
        ax.set_title(f"Δ {TP_LABELS[tp]}", fontsize=10, fontweight="bold", pad=14)
    plt.savefig(OUT_DIR / "sgacc_radar_delta.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5. Save Data Table
    summary_path = OUT_DIR / "sgacc_fc_single_subject_values.tsv"
    df_fc.to_csv(summary_path, sep="\t", index=False)
    print(f"Results saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
