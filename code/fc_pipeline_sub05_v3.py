"""
Functional Connectivity Pipeline — sub-05 — v3 (Complete Standalone)
===================================================================
Sessions  : ses-exp (focused TUS) + ses-con (defocused TUS)
Timepoints: preTUS15, postTUS15, postTUS30, postTUS45

Output figures generated here:
  - Radar chart: 8 cortical ROIs from sgACC literature (Alternative 3)

Usage:
  python3 fc_pipeline_sub05_v3.py \
    --data_dir "/Volumes/Extreme SSD/THESIS MSC/MEPrep output" \
    --atlas ~/Projects/Master-thesis/CITRUS/atlas/Schaefer2018_200Parcels_7Networks_order.dlabel.nii \
    --out_dir ~/Projects/Master-thesis/CITRUS/fc_output
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
import matplotlib.patches mpatches
from nilearn.signal import clean
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
SUBJECT    = "sub-05"
TASK       = "task-rest"
PROC       = "proc-pmeica"
SPACE      = "space-fsLR"
DEN        = "den-91k"
TR         = 1.5
HP_FREQ    = 0.01
LP_FREQ    = 0.10

SESSIONS   = ["ses-exp", "ses-con"]
TIMEPOINTS = ["preTUS15", "postTUS15", "postTUS30", "postTUS45"]
BASELINE       = "preTUS15"
BASELINE_LABEL = "preTUS"

CONFOUND_COLS = [
    "trans_x", "trans_y", "trans_z",
    "rot_x",   "rot_y",   "rot_z",
    "white_matter", "csf",
]

NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]

NETWORK_LABELS = {
    "Vis":         "Visual (Vis)",
    "SomMot":      "Somatomotor (SomMot)",
    "DorsAttn":    "Dorsal Attention (DAN)",
    "SalVentAttn": "Salience/Ventral Attn (SAL)",
    "Limbic":      "Limbic",
    "Cont":        "Frontoparietal Control (FPN)",
    "Default":     "Default Mode (DMN)",
}

TP_DISPLAY = {
    "preTUS15":  "preTUS",
    "postTUS15": "postTUS 15min",
    "postTUS30": "postTUS 30min",
    "postTUS45": "postTUS 45min",
}

SESSION_LABELS = {
    "ses-exp": "Focused TUS",
    "ses-con": "Defocused TUS (Control)",
}

# Alternative 3: 8 cortical targets connected to sgACC from literature
# Key = Target ROI display name, Value = substring search pattern in Schaefer 200 name
SGACC_ROIS = {
    "vmPFC":         "PFCv",
    "PCC":           "pCunPCC",
    "DLPFC":         "PFCl",
    "Insula":        "Ins",
    "dACC":          "CingA",
    "Temp pole":     "TempPole",
    "OFC":           "OFC",
    "Angular":       "IPL",
}

# Exact keyword identifying our specialized seed region within the Schaefer 200 mapping taxonomy
SGACC_SEED_SUBSTRING = "Default_PFCv"

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


# ─────────────────────────────────────────────
# File paths
# ─────────────────────────────────────────────
def bold_path(data_dir, session, acq):
    fname = (f"{SUBJECT}_{session}_{TASK}_acq-{acq}_{PROC}"
             f"_{SPACE}_{DEN}_bold.dtseries.nii")
    p = Path(data_dir) / session / "func" / fname
    if not p.exists():
        raise FileNotFoundError(f"Missing BOLD path:\n  {p}")
    return p

def confounds_path(data_dir, session, acq):
    fname = (f"{SUBJECT}_{session}_{TASK}_acq-{acq}_{PROC}"
             f"_desc-confounds_timeseries.tsv")
    p = Path(data_dir) / session / "func" / fname
    if not p.exists():
        raise FileNotFoundError(f"Missing confounds path:\n  {p}")
    return p


# ─────────────────────────────────────────────
# Atlas helpers - read parcel names from CIFTI
# ─────────────────────────────────────────────
def read_parcel_names(atlas_path):
    """
    Read parcel names from Schaefer dlabel.nii CIFTI file.
    """
    img = nib.load(atlas_path)
    parcel_net = {}
    parcel_name_dict = {}

    try:
        axes = [img.header.get_axis(i) for i in range(len(img.shape))]
        label_axis = None
        for ax in axes:
            if hasattr(ax, 'label'):
                label_axis = ax
                break

        if label_axis is not None and hasattr(label_axis, 'label'):
            label_table = label_axis.label
            if hasattr(label_table, 'items'):
                for key, lbl in label_table.items():
                    if key == 0:
                        continue
                    name = lbl.label if hasattr(lbl, 'label') else str(lbl)
                    idx  = key - 1   # Convert to 0-indexed reference
                    parcel_name_dict[idx] = name
                    parts = name.split("_")
                    net = parts[2] if len(parts) > 2 else "Unknown"
                    parcel_net[idx] = net
    except Exception as e:
        print(f"    Warning: Could not read label axis directly: {e}")

    # Fallback: use known Schaefer 200 7-network layout sequencing
    if len(parcel_net) == 0:
        print("    Using fallback: generating parcel-network map from known Schaefer ordering")
        schaefer_networks_lh = (
            ["Vis"] * 15 + ["SomMot"] * 20 + ["DorsAttn"] * 15 +
            ["SalVentAttn"] * 13 + ["Limbic"] * 7 + ["Cont"] * 12 + ["Default"] * 18
        )
        schaefer_networks_rh = (
            ["Vis"] * 15 + ["SomMot"] * 20 + ["DorsAttn"] * 14 +
            ["SalVentAttn"] * 12 + ["Limbic"] * 7 + ["Cont"] * 13 + ["Default"] * 19
        )
        all_networks = schaefer_networks_lh + schaefer_networks_rh
        for i, net in enumerate(all_networks[:200]):
            parcel_net[i]  = net
            parcel_name_dict[i] = f"Parcel_{i+1}_{net}"

    print(f"    Loaded map: {len(parcel_net)} parcels extracted successfully.")
    return parcel_net, parcel_name_dict


# ─────────────────────────────────────────────
# Core Processing & Extraction
# ─────────────────────────────────────────────
def load_cifti(path):
    img  = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    # Handle standard subcortical cropping if needed
    if data.shape[1] == 91282:
        data = data[:, :64984]
    return data

def run_sgacc_roi_analysis(fc_matrix, parcel_name_dict):
    """
    Finds the exact subgenual cortex seed parcel and computes average
    cross-correlation connectivity against our 8 individual target ROIs.
    """
    # 1. Identify specific localized seed indices for sgACC
    seed_indices = [idx for idx, name in parcel_name_dict.items() if SGACC_SEED_SUBSTRING in name]

    if not seed_indices:
        # Emergency secondary structural check if user configuration substring differs
        seed_indices = [idx for idx, name in parcel_name_dict.items() if "PFCv" in name]
        if not seed_indices:
            raise ValueError(f"Could not locate seed via target substring filter '{SGACC_SEED_SUBSTRING}'.")

    roi_values = {}

    # 2. Slice connectivity matrices per target ROI criteria
    for roi_label, substring in SGACC_ROIS.items():
        target_indices = [idx for idx, name in parcel_name_dict.items() if substring in name]

        if not target_indices:
            roi_values[roi_label] = np.nan
            continue

        # 3. Cross-index safely to get seed-to-target connectivity sub-blocks
        sub_matrix = fc_matrix[np.ix_(seed_indices, target_indices)]
        roi_values[roi_label] = float(np.nanmean(sub_matrix))

    return roi_values


# ─────────────────────────────────────────────
# Visualization (Spider/Radar Chart)
# ─────────────────────────────────────────────
def plot_sgacc_radar(roi_results_df, out_dir):
    """
    Generates a publication-grade circular Spider/Radar chart
    for Focused TUS vs Defocused TUS.
    """
    categories = list(SGACC_ROIS.keys())
    num_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Close the circular graphic loop

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(polar=True))
    fig.suptitle(f"sgACC Target Circuit Connectivity Profile ({SUBJECT})", fontsize=16, weight='bold', y=1.05)

    for ax_idx, session in enumerate(SESSIONS):
        ax = axes[ax_idx]
        ax.set_theta_offset(np.pi / 2) # Top positioning start
        ax.set_theta_direction(-1)     # Clockwise layout

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11, weight='bold')

        session_df = roi_results_df[roi_results_df['Session'] == session]

        for tp in TIMEPOINTS:
            tp_df = session_df[session_df['Timepoint'] == tp]

            values = [
                tp_df[tp_df['ROI'] == r]['FC_Value'].values[0]
                if r in tp_df['ROI'].values else 0 for r in categories
            ]
            values += values[:1] # Complete graphic vector

            color = TP_COLORS.get(tp, "#888888")
            style = TP_STYLES.get(tp, "-")
            label = TP_DISPLAY.get(tp, tp)

            ax.plot(angles, values, label=label, color=color, linestyle=style, linewidth=2)
            ax.fill(angles, values, color=color, alpha=0.04)

        ax.set_title(SESSION_LABELS.get(session, session), fontsize=13, weight='bold', pad=20)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_rlabel_position(180)
        ax.tick_params(colors='#555555', labelsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.06),
               ncol=4, frameon=True, facecolor='#fdfdfd', edgecolor='#cccccc', fontsize=12)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{SUBJECT}_sgacc_circuit_radar.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
