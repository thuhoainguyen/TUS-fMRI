"""
Functional Connectivity Pipeline — sub-05 — v4 (Fixed Matrix Extraction)
===================================================================
Sessions  : ses-exp (focused TUS) + ses-con (defocused TUS)
Timepoints: preTUS15, postTUS15, postTUS30, postTUS45

Usage:
  python3 ~/Projects/Master-thesis/CITRUS/code/fc_pipeline_sub05_v4.py \
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
from nilearn.signal import clean
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
SUBJECT = "sub-05"
TASK = "task-rest"
PROC = "proc-pmeica"
SPACE = "space-fsLR"
DEN = "den-91k"
TR = 1.5
HP_FREQ = 0.01
LP_FREQ = 0.10

SESSIONS = ["ses-exp", "ses-con"]
TIMEPOINTS = ["preTUS15", "postTUS15", "postTUS30", "postTUS45"]

CONFOUND_COLS = [
    "trans_x", "trans_y", "trans_z",
    "rot_x", "rot_y", "rot_z",
    "white_matter", "csf",
]

TP_DISPLAY = {
    "preTUS15": "preTUS",
    "postTUS15": "postTUS 15min",
    "postTUS30": "postTUS 30min",
    "postTUS45": "postTUS 45min",
}

SESSION_LABELS = {
    "ses-exp": "Focused TUS",
    "ses-con": "Defocused TUS (Control)",
}

# Alternative 3: 8 cortical targets connected to sgACC from literature
SGACC_ROIS = {
    "vmPFC": "PFCv",
    "PCC": "pCunPCC",
    "DLPFC": "PFCl",
    "Insula": "Ins",
    "dACC": "CingA",
    "Temp pole": "TempPole",
    "OFC": "OFC",
    "Angular": "IPL",
}

SGACC_SEED_SUBSTRING = "Default_PFCv"

TP_COLORS = {
    "preTUS15": "#888780",
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
# Helpers
# ─────────────────────────────────────────────
def bold_path(data_dir, session, acq):
    fname = f"{SUBJECT}_{session}_{TASK}_acq-{acq}_{PROC}_{SPACE}_{DEN}_bold.dtseries.nii"
    p = Path(data_dir) / session / "func" / fname
    if not p.exists():
        raise FileNotFoundError(f"Missing BOLD path: {p}")
    return p


def confounds_path(data_dir, session, acq):
    fname = f"{SUBJECT}_{session}_{TASK}_acq-{acq}_{PROC}_desc-confounds_timeseries.tsv"
    p = Path(data_dir) / session / "func" / fname
    if not p.exists():
        raise FileNotFoundError(f"Missing confounds path: {p}")
    return p


def read_atlas_and_labels(atlas_path):
    """
    Đọc đồng thời bản đồ nhãn (vùng chứa các đỉnh vertex) và danh sách tên vùng từ dlabel.nii
    """
    img = nib.load(atlas_path)
    atlas_data = np.asanyarray(img.dataobj).squeeze()  # Mảng chứa ID nhãn của từng vertex (thường là 64984 phần tử)

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
            for key, lbl in label_table.items():
                if key == 0:  # Thường 0 là Medial Wall / background
                    continue
                name = lbl.label if hasattr(lbl, 'label') else str(lbl)
                idx = key - 1  # Chuyển về 0-indexed cho ma trận FC (0 đến 199)
                parcel_name_dict[idx] = name
    except Exception as e:
        print(f"    Warning: Không đọc được trực tiếp nhãn từ header: {e}")

    # Fallback nếu header không chứa bảng nhãn rõ ràng
    if len(parcel_name_dict) == 0:
        print("    Using fallback: Tự động điền nhãn Schaefer 200 mặc định")
        for i in range(200):
            parcel_name_dict[i] = f"Schaefer200_Parcel_{i + 1}"

    return atlas_data, parcel_name_dict


def run_sgacc_roi_analysis(fc_matrix, parcel_name_dict):
    seed_indices = [idx for idx, name in parcel_name_dict.items() if SGACC_SEED_SUBSTRING in name]
    if not seed_indices:
        seed_indices = [idx for idx, name in parcel_name_dict.items() if "PFCv" in name]
        if not seed_indices:
            raise ValueError(
                f"Không tìm thấy Seed ROI nào chứa '{SGACC_SEED_SUBSTRING}' hoặc 'PFCv' trong danh sách nhãn.")

    roi_values = {}
    for roi_label, substring in SGACC_ROIS.items():
        target_indices = [idx for idx, name in parcel_name_dict.items() if substring in name]
        if not target_indices:
            roi_values[roi_label] = np.nan
            continue

        sub_matrix = fc_matrix[np.ix_(seed_indices, target_indices)]
        roi_values[roi_label] = float(np.nanmean(sub_matrix))

    return roi_values


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def plot_sgacc_radar(roi_results_df, out_dir):
    categories = list(SGACC_ROIS.keys())
    num_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(polar=True))
    fig.suptitle(f"sgACC Target Circuit Connectivity Profile ({SUBJECT})", fontsize=16, weight='bold', y=1.05)

    for ax_idx, session in enumerate(SESSIONS):
        ax = axes[ax_idx]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11, weight='bold')

        session_df = roi_results_df[roi_results_df['Session'] == session]

        for tp in TIMEPOINTS:
            tp_df = session_df[session_df['Timepoint'] == tp]
            values = [
                tp_df[tp_df['ROI'] == r]['FC_Value'].values[0]
                if r in tp_df['ROI'].values and len(tp_df[tp_df['ROI'] == r]['FC_Value'].values) > 0 else 0
                for r in categories
            ]
            values += values[:1]

            color = TP_COLORS.get(tp, "#888888")
            style = TP_STYLES.get(tp, "-")
            label = TP_DISPLAY.get(tp, tp)

            ax.plot(angles, values, label=label, color=color, linestyle=style, linewidth=2)
            ax.fill(angles, values, color=color, alpha=0.04)

        ax.set_title(SESSION_LABELS.get(session, session), fontsize=13, weight='bold', pad=20)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_rlabel_position(180)
        ax.tick_params(colors='#555555', labelsize=9)

    handles, labels = axes.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.06),
               ncol=4, frameon=True, facecolor='#fdfdfd', edgecolor='#cccccc', fontsize=12)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{SUBJECT}_sgacc_circuit_radar.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[SUCCESS] --> Đã xuất biểu đồ Radar tại:\n    {output_path}")


# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run localized functional connectivity pipeline.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--atlas", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    print("Step 1: Đang đọc Atlas và cấu trúc phân vùng...")
    atlas_map, parcel_name_dict = read_atlas_and_labels(args.atlas)

    compiled_records = []

    print("\nStep 2: Bắt đầu xử lý dữ liệu chuỗi thời gian (Timeseries)...")
    for session in SESSIONS:
        for tp in TIMEPOINTS:
            print(f"  -> Đang xử lý: {session} | Timepoint: {tp}")

            try:
                b_path = bold_path(args.data_dir, session, tp)
                c_path = confounds_path(args.data_dir, session, tp)

                # Đọc dữ liệu BOLD (Hình dạng: Timepoints x Vertices)
                img = nib.load(str(b_path))
                bold_data = img.get_fdata(dtype=np.float32)

                # Cắt bớt phần subcortex của 91k để map khớp chính xác với 64k của phần vỏ não (Cortex) Schaefer
                if bold_data.shape[1] == 91282:
                    bold_data = bold_data[:, :64984]

                # Đọc & Làm sạch nhiễu (Confounds)
                confounds_df = pd.read_csv(c_path, sep='\t')
                cleaned_confounds = confounds_df[CONFOUND_COLS].bfill().ffill().values

                # Khởi tạo ma trận chuỗi thời gian cho 200 vùng (Timepoints x 200)
                num_timepoints = bold_data.shape[0]
                parcel_ts = np.zeros((num_timepoints, 200))

                # Vòng lặp trích xuất trung bình tín hiệu CHÍNH XÁC từ các đỉnh vertex thuộc từng vùng
                for i in range(200):
                    # Schaefer nhãn trong file gốc thường lưu từ 1 -> 200. Nên key thực tế là i + 1
                    roi_mask = (atlas_map == (i + 1))

                    # Nếu vùng đó có vertex tương ứng trong file BOLD vỏ não
                    if np.any(roi_mask):
                        parcel_ts[:, i] = np.mean(bold_data[:, roi_mask], axis=1)
                    else:
                        parcel_ts[:, i] = 0.0

