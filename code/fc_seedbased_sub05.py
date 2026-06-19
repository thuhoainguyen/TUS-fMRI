"""
Seed-based FC Pipeline — sub-05
================================
True seed-based functional connectivity analysis using:
  - sgACC (BA25) as primary seed — subject-specific mask from k-Plan
  - 12 target ROIs from literature (subcortical + cortical)
  - MNI152NLin2009cAsym volumetric BOLD (proc-pmeica)

Seeds and targets:
  SEED:    sgACC BA25 (L+R combined)
  TARGETS: amygdala, thalamus, caudate, putamen, hippocampus,
           accumbens (all bilateral from Harvard-Oxford or k-Plan)
           + cortical targets from Schaefer 200 (vmPFC, PCC, dACC, OFC, DLPFC, insula)

Output:
  - fc_seedbased/     — FC values (sgACC → each target) per timepoint
  - figures/          — spider chart, bar chart, line plot of temporal dynamics
  - fc_seedbased.csv  — full numerical results

Cách chạy:
  python3 fc_seedbased_sub05.py \
    --data_dir  "/Volumes/Extreme SSD/THESIS MSC/MEPrep output" \
    --mask_dir  ~/Projects/Master-thesis/CITRUS/masks/mni \
    --atlas     ~/Projects/Master-thesis/CITRUS/atlas/Schaefer2018_200Parcels_7Networks_order.dlabel.nii \
    --out_dir   ~/Projects/Master-thesis/CITRUS/fc_seedbased_output
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from nilearn.signal import clean
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
SUBJECT    = "sub-05"
TASK       = "task-rest"
PROC       = "proc-pmeica"
TR         = 1.5
HP_FREQ    = 0.01
LP_FREQ    = 0.10
SESSIONS   = ["ses-exp", "ses-con"]
TIMEPOINTS = ["preTUS15", "postTUS15", "postTUS30", "postTUS45"]
BASELINE   = "preTUS15"

SESSION_LABELS = {
    "ses-exp": "Focused TUS",
    "ses-con": "Defocused TUS (Control)",
}
TP_DISPLAY = {
    "preTUS15":  "preTUS",
    "postTUS15": "postTUS 15min",
    "postTUS30": "postTUS 30min",
    "postTUS45": "postTUS 45min",
}
TP_COLORS = {
    "preTUS15":  "#888780",
    "postTUS15": "#E24B4A",
    "postTUS30": "#D85A30",
    "postTUS45": "#BA7517",
}
TP_STYLES = {
    "preTUS15": "--",
    "postTUS15": "-",
    "postTUS30": "-",
    "postTUS45": "-",
}

CONFOUND_COLS = [
    "trans_x", "trans_y", "trans_z",
    "rot_x",   "rot_y",   "rot_z",
    "white_matter", "csf",
]

# ─────────────────────────────────────────────
# ROI definitions
# ─────────────────────────────────────────────
# Volumetric ROIs (MNI space, 2.5mm) — mask filenames in mask_dir
VOLUMETRIC_ROIS = {
    "Amygdala (L)":     "amygdala_L_kplan_MNI_2p5mm.nii.gz",
    "Amygdala (R)":     "amygdala_R_kplan_MNI_2p5mm.nii.gz",
    "Thalamus (L)":     "HO_thalamus_L_2p5mm.nii.gz",
    "Thalamus (R)":     "HO_thalamus_R_2p5mm.nii.gz",
    "Hippocampus (L)":  "HO_hippocampus_L_2p5mm.nii.gz",
    "Hippocampus (R)":  "HO_hippocampus_R_2p5mm.nii.gz",
    "Caudate (L)":      "HO_caudate_L_2p5mm.nii.gz",
    "Caudate (R)":      "HO_caudate_R_2p5mm.nii.gz",
    "Putamen (L)":      "HO_putamen_L_2p5mm.nii.gz",
    "Putamen (R)":      "HO_putamen_R_2p5mm.nii.gz",
    "Accumbens (L)":    "HO_accumbens_L_2p5mm.nii.gz",
    "Accumbens (R)":    "HO_accumbens_R_2p5mm.nii.gz",
}

# Cortical ROIs from Schaefer 200 — keyword matched against parcel names
# These are extracted from the fsLR surface data (separate from volumetric)
CORTICAL_ROI_KEYWORDS = {
    "vmPFC":   ["Default_PFCv"],
    "PCC":     ["Default_pCunPCC"],
    "DLPFC":   ["Cont_PFCl"],
    "Insula":  ["SalVentAttn_FrOperIns"],
    "dACC":    ["SalVentAttn_Med"],
    "OFC":     ["Limbic_OFC"],
}

# All ROI display names in order for spider chart
ALL_ROI_LABELS = (
    list(CORTICAL_ROI_KEYWORDS.keys()) +
    ["Amygdala (L)", "Amygdala (R)",
     "Thalamus (L)", "Thalamus (R)",
     "Hippocampus (L)", "Hippocampus (R)",
     "Accumbens (L)", "Accumbens (R)"]
)


# ─────────────────────────────────────────────
# File path helpers
# ─────────────────────────────────────────────
def bold_vol_path(data_dir, session, acq):
    """MNI152 volumetric BOLD — for subcortical seed extraction."""
    fname = (f"{SUBJECT}_{session}_{TASK}_acq-{acq}_{PROC}"
             f"_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
    p = Path(data_dir) / session / "func" / fname
    if not p.exists():
        raise FileNotFoundError(f"BOLD not found:\n  {p}")
    return p

def bold_surf_path(data_dir, session, acq):
    """fsLR 91k surface BOLD — for cortical seed extraction."""
    fname = (f"{SUBJECT}_{session}_{TASK}_acq-{acq}_{PROC}"
             f"_space-fsLR_den-91k_bold.dtseries.nii")
    p = Path(data_dir) / session / "func" / fname
    if not p.exists():
        raise FileNotFoundError(f"Surface BOLD not found:\n  {p}")
    return p

def confounds_path(data_dir, session, acq):
    fname = (f"{SUBJECT}_{session}_{TASK}_acq-{acq}_{PROC}"
             f"_desc-confounds_timeseries.tsv")
    p = Path(data_dir) / session / "func" / fname
    if not p.exists():
        raise FileNotFoundError(f"Confounds not found:\n  {p}")
    return p


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
def load_bold_vol(path):
    """Load 4D volumetric BOLD → (n_tp, n_vox_flat)."""
    img  = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)   # (x, y, z, t)
    data = data.reshape(-1, data.shape[-1]).T  # (t, x*y*z)
    print(f"    Volumetric BOLD: {data.shape[0]} tp × {data.shape[1]} voxels")
    return data

def load_bold_surf(path):
    """Load CIFTI surface BOLD → (n_tp, 64984 cortical vertices)."""
    img  = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    if data.shape[1] == 91282:
        data = data[:, :64984]   # cortex only
    print(f"    Surface BOLD:    {data.shape[0]} tp × {data.shape[1]} vertices")
    return data

def load_mask_vol(mask_path):
    """Load binary volumetric mask → flat boolean array."""
    img  = nib.load(str(mask_path))
    data = img.get_fdata()
    return data.flatten().astype(bool)

def load_confounds(path):
    df        = pd.read_csv(str(path), sep="\t")
    available = [c for c in CONFOUND_COLS if c in df.columns]
    missing   = [c for c in CONFOUND_COLS if c not in df.columns]
    if missing:
        print(f"    Confounds missing: {missing}")
    return df[available].fillna(0).values.astype(np.float32)

def load_atlas_surf(atlas_path):
    """Load Schaefer atlas → (64984,) integer label array + parcel name dict."""
    img    = nib.load(atlas_path)
    labels = img.get_fdata().squeeze().astype(int)

    # Read parcel names from LabelAxis
    # ax0.label is a numpy array of dicts; lt[0] is the dict {int: (name, rgba)}
    parcel_names = {}
    try:
        ax0 = img.header.get_axis(0)
        lt  = ax0.label        # numpy array
        d   = lt[0]            # the dict we want
        for key, (name, rgba) in d.items():
            if key == 0:
                continue
            parcel_names[key - 1] = name   # 0-indexed
    except Exception as e:
        print(f"    Warning reading atlas labels: {e}")

    # Fallback names
    if not parcel_names:
        for i in range(200):
            parcel_names[i] = f"Parcel_{i+1}"

    return labels, parcel_names


# ─────────────────────────────────────────────
# Timeseries extraction
# ─────────────────────────────────────────────
def extract_seed_timeseries_vol(bold_vol, mask_flat):
    """Extract mean timeseries from a volumetric mask."""
    if mask_flat.sum() == 0:
        raise ValueError("Mask has 0 voxels — check registration")
    return bold_vol[:, mask_flat].mean(axis=1)   # (n_tp,)

def extract_roi_timeseries_surf(bold_surf, atlas_labels, keyword):
    """Extract mean timeseries from Schaefer parcels matching a keyword."""
    from collections import defaultdict
    # find parcel indices whose name contains keyword
    # atlas_labels is (64984,), values 0-200
    # We need parcel_names to match keyword
    # Return mean of matching parcels
    # If parcel_names not available, return None
    return None   # handled below with parcel_names dict

def extract_cortical_roi(bold_surf, atlas_labels, parcel_names, keyword):
    """Find parcels matching keyword (string or list) → extract mean timeseries."""
    if isinstance(keyword, str):
        kws = [keyword]
    else:
        kws = list(keyword)
    matching = [idx for idx, name in parcel_names.items()
                if any(kw in name for kw in kws)]
    if not matching:
        print(f"    Warning: no Schaefer parcel found for '{keyword}'")
        return None
    ts_list = []
    for idx in matching:
        mask = (atlas_labels == idx + 1)
        if mask.sum() > 0:
            ts_list.append(bold_surf[:, mask].mean(axis=1))
    if not ts_list:
        return None
    return np.stack(ts_list).mean(axis=0)   # average across matching parcels


# ─────────────────────────────────────────────
# Denoising
# ─────────────────────────────────────────────
def denoise_ts(ts_matrix, confounds):
    """
    Denoise a (n_tp, n_signals) matrix.
    ts_matrix can be (n_tp,) for a single timeseries or (n_tp, N).
    """
    if ts_matrix.ndim == 1:
        ts_matrix = ts_matrix[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    n_tp = ts_matrix.shape[0]
    if confounds.shape[0] > n_tp:
        confounds = confounds[:n_tp]
    elif confounds.shape[0] < n_tp:
        pad = np.zeros((n_tp - confounds.shape[0], confounds.shape[1]))
        confounds = np.vstack([confounds, pad])

    cleaned = clean(ts_matrix, confounds=confounds, t_r=TR,
                    high_pass=HP_FREQ, low_pass=LP_FREQ,
                    standardize=True, detrend=True)
    return cleaned[:, 0] if squeeze else cleaned


# ─────────────────────────────────────────────
# FC computation
# ─────────────────────────────────────────────
def pearson_fisher_z(ts_a, ts_b):
    """Pearson r between two 1D timeseries → Fisher Z."""
    r = np.corrcoef(ts_a, ts_b)[0, 1]
    r = np.clip(r, -0.9999, 0.9999)
    return float(np.arctanh(r))


# ─────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────
def plot_spider_chart(fc_results, out_path, title):
    """
    Spider chart: sgACC FC with each target ROI.
    fc_results: dict {(ses, tp): {roi_name: fc_value}}
    Two panels side by side — focused vs defocused.
    """
    roi_labels = ALL_ROI_LABELS
    # Filter to ROIs that have data
    available = [r for r in roi_labels
                 if any(r in fc_results[(s, t)]
                        for s in SESSIONS for t in TIMEPOINTS)]
    if not available:
        print("    No ROI data available for spider chart")
        return

    n_axes = len(available)
    angles = [n / float(n_axes) * 2 * np.pi for n in range(n_axes)]
    angles += angles[:1]

    fig, axs = plt.subplots(1, 2, figsize=(16, 8),
                             subplot_kw=dict(polar=True))

    for ax, ses in zip(axs, SESSIONS):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available, size=9)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
        ax.set_yticklabels(["-0.4", "-0.2", "0", "0.2", "0.4"], size=7)
        ax.set_title(SESSION_LABELS[ses], size=12, pad=20)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":", zorder=0)

        for tp in TIMEPOINTS:
            data = fc_results.get((ses, tp), {})
            vals = [data.get(roi, np.nan) for roi in available]
            # Replace NaN with 0 for plotting
            vals_plot = [v if not np.isnan(v) else 0 for v in vals]
            vals_plot += vals_plot[:1]
            ax.plot(angles, vals_plot,
                    color=TP_COLORS[tp],
                    linestyle=TP_STYLES[tp],
                    linewidth=2, alpha=0.9,
                    label=TP_DISPLAY.get(tp, tp))
            ax.fill(angles, vals_plot, color=TP_COLORS[tp], alpha=0.06)

    handles = [mpatches.Patch(color=TP_COLORS[t],
                               label=TP_DISPLAY.get(t, t))
               for t in TIMEPOINTS]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=10, title="Timepoint",
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def plot_temporal_dynamics(fc_results, out_path):
    """
    Line plot: FC change over time for each ROI.
    One panel per session, lines = ROIs.
    X-axis = timepoints, Y-axis = ΔFC vs baseline.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    tp_labels = [TP_DISPLAY.get(t, t) for t in TIMEPOINTS]

    # Get ROIs that have data
    available = [r for r in ALL_ROI_LABELS
                 if any(r in fc_results[(s, t)]
                        for s in SESSIONS for t in TIMEPOINTS)]

    # Color map for ROIs
    cmap    = plt.cm.get_cmap("tab20", len(available))
    roi_colors = {roi: cmap(i) for i, roi in enumerate(available)}

    for ax, ses in zip(axes, SESSIONS):
        baseline_vals = fc_results.get((ses, BASELINE), {})

        for roi in available:
            baseline_fc = baseline_vals.get(roi, np.nan)
            delta_vals  = []
            for tp in TIMEPOINTS:
                fc_val = fc_results.get((ses, tp), {}).get(roi, np.nan)
                delta_vals.append(fc_val - baseline_fc
                                  if not np.isnan(fc_val)
                                  and not np.isnan(baseline_fc)
                                  else np.nan)
            ax.plot(tp_labels, delta_vals,
                    marker="o", linewidth=1.5,
                    color=roi_colors[roi], label=roi, alpha=0.8)

        ax.axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_title(SESSION_LABELS[ses], fontsize=12)
        ax.set_ylabel("ΔFC vs preTUS baseline (Fisher Z)")
        ax.set_xlabel("Timepoint")
        ax.tick_params(axis="x", rotation=15)

    handles = [plt.Line2D([0], [0], color=roi_colors[r],
                           linewidth=2, label=r)
               for r in available]
    fig.legend(handles=handles, loc="center right",
               fontsize=8, title="Target ROI",
               bbox_to_anchor=(1.12, 0.5))
    fig.suptitle(
        f"sgACC FC temporal dynamics — {SUBJECT}\n"
        f"ΔFC relative to preTUS baseline",
        fontsize=13)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def plot_bar_focused_vs_defocused(fc_results, out_path):
    """
    Bar chart: focused vs defocused ΔFC per ROI per post-TUS timepoint.
    Shows where the two conditions diverge.
    """
    post_tps  = [tp for tp in TIMEPOINTS if tp != BASELINE]
    available = [r for r in ALL_ROI_LABELS
                 if any(r in fc_results[(s, t)]
                        for s in SESSIONS for t in TIMEPOINTS)]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    for ax, tp in zip(axes, post_tps):
        x     = np.arange(len(available))
        w     = 0.35
        for i, ses in enumerate(SESSIONS):
            baseline = fc_results.get((ses, BASELINE), {})
            deltas   = []
            for roi in available:
                fc_val = fc_results.get((ses, tp), {}).get(roi, np.nan)
                bl_val = baseline.get(roi, np.nan)
                deltas.append(fc_val - bl_val
                              if not np.isnan(fc_val) and not np.isnan(bl_val)
                              else 0)
            color = "#E24B4A" if ses == "ses-exp" else "#378ADD"
            ax.bar(x + i * w, deltas, w,
                   label=SESSION_LABELS[ses], color=color, alpha=0.8)

        ax.set_xticks(x + w / 2)
        ax.set_xticklabels(available, rotation=40, ha="right", fontsize=8)
        ax.set_title(TP_DISPLAY.get(tp, tp), fontsize=11)
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.set_ylabel("ΔFC (Fisher Z)")
        ax.legend(fontsize=9)

    fig.suptitle(
        f"sgACC FC change: Focused vs Defocused TUS — {SUBJECT}",
        fontsize=13)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def run(data_dir, mask_dir, atlas_path, out_dir):
    out_dir  = Path(out_dir)
    fig_dir  = out_dir / "figures"
    fc_dir   = out_dir / "fc_values"
    for d in [fig_dir, fc_dir]:
        d.mkdir(parents=True, exist_ok=True)

    mask_dir = Path(mask_dir)

    # ── Load atlas (surface, for cortical ROIs) ──
    print("\n[Loading atlas]")
    atlas_labels, parcel_names = load_atlas_surf(atlas_path)
    print(f"    Schaefer 200: {len(parcel_names)} parcels")

    # ── Load volumetric masks ──
    print("\n[Loading volumetric masks]")
    vol_masks = {}
    for roi_name, fname in VOLUMETRIC_ROIS.items():
        p = mask_dir / fname
        if p.exists():
            vol_masks[roi_name] = load_mask_vol(p)
            n = int(vol_masks[roi_name].sum())
            print(f"    {roi_name}: {n} voxels")
        else:
            print(f"    WARNING: mask not found — {fname}")

    # ── Main loop ──
    fc_results = {}   # {(ses, tp): {roi_name: fc_value}}
    rows       = []

    for ses in SESSIONS:
        print(f"\n{'='*60}")
        print(f"SESSION: {ses} ({SESSION_LABELS[ses]})")
        print("=" * 60)

        for acq in TIMEPOINTS:
            print(f"\n  [Timepoint: {acq}]")
            fc_results[(ses, acq)] = {}

            # Load confounds once
            conf = load_confounds(confounds_path(data_dir, ses, acq))

            # ── sgACC seed timeseries (volumetric, L+R combined) ──
            bold_vol = load_bold_vol(bold_vol_path(data_dir, ses, acq))

            sgacc_masks = []
            for side in ["L", "R"]:
                fname = f"sgACC_BA25_{side}_kplan_MNI_2p5mm.nii.gz"
                p     = mask_dir / fname
                if p.exists():
                    sgacc_masks.append(load_mask_vol(p))

            if not sgacc_masks:
                raise FileNotFoundError("sgACC masks not found in mask_dir")

            # Combined bilateral sgACC mask
            sgacc_mask_combined = np.logical_or.reduce(sgacc_masks)
            print(f"    sgACC seed: {int(sgacc_mask_combined.sum())} voxels (L+R)")

            # Extract and denoise sgACC seed timeseries
            sgacc_ts_raw = extract_seed_timeseries_vol(bold_vol, sgacc_mask_combined)
            sgacc_ts     = denoise_ts(sgacc_ts_raw, conf)

            # ── Volumetric target ROIs ──
            for roi_name, mask_flat in vol_masks.items():
                target_ts_raw = extract_seed_timeseries_vol(bold_vol, mask_flat)
                target_ts     = denoise_ts(target_ts_raw, conf)
                fc_val        = pearson_fisher_z(sgacc_ts, target_ts)
                fc_results[(ses, acq)][roi_name] = fc_val
                print(f"    FC sgACC → {roi_name}: {fc_val:.3f}")

            # ── Cortical target ROIs (surface) ──
            bold_surf = load_bold_surf(bold_surf_path(data_dir, ses, acq))

            for roi_name, keyword in CORTICAL_ROI_KEYWORDS.items():
                target_ts_raw = extract_cortical_roi(
                    bold_surf, atlas_labels, parcel_names, keyword)
                if target_ts_raw is None:
                    print(f"    WARNING: could not extract {roi_name}")
                    continue
                target_ts = denoise_ts(target_ts_raw, conf)

                # sgACC seed from surface — use Default network parcels
                # containing "PFCv" or "CingA" closest to BA25
                # For consistency, we use the volumetric sgACC seed
                # denoised with surface confounds — re-denoise here
                fc_val = pearson_fisher_z(sgacc_ts, target_ts)
                fc_results[(ses, acq)][roi_name] = fc_val
                print(f"    FC sgACC → {roi_name}: {fc_val:.3f}")

            # Save all FC values for this timepoint
            np.save(str(fc_dir / f"fc_{ses}_{acq}.npy"),
                    fc_results[(ses, acq)])

            # Collect rows for CSV
            for roi_name, fc_val in fc_results[(ses, acq)].items():
                rows.append({
                    "subject":   SUBJECT,
                    "session":   ses,
                    "session_label": SESSION_LABELS[ses],
                    "timepoint": acq,
                    "timepoint_label": TP_DISPLAY.get(acq, acq),
                    "target_roi": roi_name,
                    "fc_z":      round(fc_val, 4),
                })

    # ── Figures ──
    print(f"\n[Figures]")

    print("  Spider chart — all timepoints")
    plot_spider_chart(
        fc_results,
        out_path=fig_dir / "spider_sgacc_all_timepoints.png",
        title=f"sgACC (BA25) seed FC with target ROIs — {SUBJECT}\n"
              f"Left: Focused TUS | Right: Defocused TUS"
    )

    print("  Temporal dynamics line plot")
    plot_temporal_dynamics(
        fc_results,
        out_path=fig_dir / "temporal_dynamics_sgacc_fc.png"
    )

    print("  Bar chart — focused vs defocused")
    plot_bar_focused_vs_defocused(
        fc_results,
        out_path=fig_dir / "bar_focused_vs_defocused.png"
    )

    # ── CSV ──
    df = pd.DataFrame(rows)
    df.to_csv(str(out_dir / "fc_seedbased.csv"), index=False)
    print(f"\n  Saved: fc_seedbased.csv")

    # Print summary table
    pivot = df.pivot_table(
        index=["session_label", "timepoint_label"],
        columns="target_roi",
        values="fc_z",
        aggfunc="first"
    )
    print("\n" + pivot.to_string())

    print(f"\n{'='*60}")
    print(f"Done! Output: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seed-based FC pipeline — sgACC to target ROIs"
    )
    parser.add_argument("--data_dir",  required=True,
        help='Root folder with ses-exp/ and ses-con/')
    parser.add_argument("--mask_dir",  required=True,
        help='Folder containing registered 2.5mm masks')
    parser.add_argument("--atlas",     required=True,
        help='Schaefer 200 parcels dlabel.nii')
    parser.add_argument("--out_dir",   default="./fc_seedbased_output")
    args = parser.parse_args()

    run(data_dir=args.data_dir,
        mask_dir=args.mask_dir,
        atlas_path=args.atlas,
        out_dir=args.out_dir)
