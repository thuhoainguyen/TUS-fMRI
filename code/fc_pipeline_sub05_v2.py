"""
Functional Connectivity Pipeline — sub-05
==========================================
Chạy FC analysis cho cả hai sessions: ses-exp (focused TUS) và ses-con (defocused TUS)
4 timepoints mỗi session: preTUS15, postTUS15, postTUS30, postTUS45

Cách chạy:
    python3 fc_pipeline_sub05_v2.py \
        --data_dir "/Volumes/Extreme SSD/THESIS MSC/MEPrep output" \
        --atlas ~/fc_analysis/atlas/Schaefer2018_200Parcels_7Networks_order.dlabel.nii \
        --out_dir ~/fc_analysis/output

Output:
    output/
    ├── fc_matrices/         — FC matrix .npy cho từng session × timepoint
    ├── fc_change/           — ΔFC = post − preTUS15 baseline
    ├── figures/             — heatmaps, change maps, network summary
    └── fc_summary.csv       — mean within-network FC, tất cả conditions
"""

import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nilearn.signal import clean
from pathlib import Path

# ─────────────────────────────────────────────
# Cấu hình cố định
# ─────────────────────────────────────────────
SUBJECT    = "sub-05"
TASK       = "task-rest"
PROC       = "proc-pmeica"
SPACE      = "space-fsLR"
DEN        = "den-91k"
TR         = 1.5          # seconds
HP_FREQ    = 0.01         # Hz bandpass low
LP_FREQ    = 0.10         # Hz bandpass high

SESSIONS   = ["ses-exp", "ses-con"]
TIMEPOINTS = ["preTUS15", "postTUS15", "postTUS30", "postTUS45"]
BASELINE   = "preTUS15"

# Confounds cần regress out (sau khi ME-ICA đã denoised rồi)
CONFOUND_COLS = [
    "trans_x", "trans_y", "trans_z",
    "rot_x",   "rot_y",   "rot_z",
    "white_matter", "csf",
]

# 7 Yeo networks (thứ tự Schaefer)
NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
NET_COLORS = {
    "Vis": "#7B2D8B", "SomMot": "#4682B4", "DorsAttn": "#2CA02C",
    "SalVentAttn": "#D62728", "Limbic": "#F4A460", "Cont": "#E377C2",
    "Default": "#FF7F0E",
}
SESSION_LABELS = {"ses-exp": "Focused TUS", "ses-con": "Defocused TUS (Control)"}


# ─────────────────────────────────────────────
# Helpers: đường dẫn file
# ─────────────────────────────────────────────
def bold_path(data_dir, session, acq):
    fname = f"{SUBJECT}_{session}_{TASK}_acq-{acq}_{PROC}_{SPACE}_{DEN}_bold.dtseries.nii"
    p = Path(data_dir) / session / "func" / fname
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file BOLD:\n  {p}")
    return p

def confounds_path(data_dir, session, acq):
    fname = f"{SUBJECT}_{session}_{TASK}_acq-{acq}_{PROC}_desc-confounds_timeseries.tsv"
    p = Path(data_dir) / session / "func" / fname
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file confounds:\n  {p}")
    return p


# ─────────────────────────────────────────────
# Bước 1: Load CIFTI
# ─────────────────────────────────────────────
def load_cifti(path):
    img  = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"CIFTI shape không đúng: {data.shape}")
    print(f"    Loaded: {data.shape[0]} timepoints × {data.shape[1]} vertices")
    return data


# ─────────────────────────────────────────────
# Bước 2: Load atlas
# ─────────────────────────────────────────────
def load_atlas(atlas_path):
    if not os.path.exists(atlas_path):
        raise FileNotFoundError(
            f"\nKhông tìm thấy atlas: {atlas_path}\n"
            "Chạy lệnh sau để tải:\n"
            'curl -L "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/'
            'stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/'
            'HCP/fslr32k/cifti/Schaefer2018_200Parcels_7Networks_order.dlabel.nii" '
            f'-o {atlas_path}\n'
        )
    atlas  = nib.load(atlas_path)
    labels = np.array(atlas.get_fdata()).squeeze().astype(int)
    print(f"    Atlas: {labels.max()} parcels, {(labels > 0).sum()} vertices có nhãn")
    return labels


# ─────────────────────────────────────────────
# Bước 3: Load confounds
# ─────────────────────────────────────────────
def load_confounds(path):
    df        = pd.read_csv(str(path), sep="\t")
    available = [c for c in CONFOUND_COLS if c in df.columns]
    missing   = [c for c in CONFOUND_COLS if c not in df.columns]
    if missing:
        print(f"    ⚠️  Confound columns không có (bỏ qua): {missing}")
    conf = df[available].fillna(0).values.astype(np.float32)
    print(f"    Confounds: {conf.shape[1]} regressors")
    return conf


# ─────────────────────────────────────────────
# Bước 4: Extract parcel timeseries
# ─────────────────────────────────────────────
def extract_parcels(data, labels):
    n_parcels = labels.max()
    n_tp      = data.shape[0]
    parcel_ts = np.zeros((n_tp, n_parcels), dtype=np.float32)
    for p in range(1, n_parcels + 1):
        mask = (labels == p)
        if mask.sum() == 0:
            print(f"    ⚠️  Parcel {p} không có vertices")
            continue
        parcel_ts[:, p - 1] = data[:, mask].mean(axis=1)
    print(f"    Parcels: {parcel_ts.shape[1]} regions × {parcel_ts.shape[0]} timepoints")
    return parcel_ts


# ─────────────────────────────────────────────
# Bước 5: Denoise
# ─────────────────────────────────────────────
def denoise(parcel_ts, confounds):
    n_tp = parcel_ts.shape[0]
    # Align confound length với BOLD
    if confounds.shape[0] > n_tp:
        confounds = confounds[:n_tp]
    elif confounds.shape[0] < n_tp:
        pad = np.zeros((n_tp - confounds.shape[0], confounds.shape[1]))
        confounds = np.vstack([confounds, pad])

    cleaned = clean(
        parcel_ts,
        confounds=confounds,
        t_r=TR,
        high_pass=HP_FREQ,
        low_pass=LP_FREQ,
        standardize=True,
        detrend=True,
    )
    print(f"    Denoised: confound regression + bandpass {HP_FREQ}–{LP_FREQ} Hz")
    return cleaned


# ─────────────────────────────────────────────
# Bước 6: Tính FC matrix
# ─────────────────────────────────────────────
def compute_fc(parcel_ts):
    r  = np.corrcoef(parcel_ts.T)
    r  = np.clip(r, -0.9999, 0.9999)
    np.fill_diagonal(r, 0)
    fc = np.arctanh(r)   # Fisher Z transform
    print(f"    FC matrix: {fc.shape} | mean |Z| = {np.abs(fc).mean():.3f}")
    return fc


# ─────────────────────────────────────────────
# Bước 7: Network order (sort matrix by network)
# ─────────────────────────────────────────────
def get_network_order(atlas_path):
    atlas       = nib.load(atlas_path)
    label_axis  = atlas.header.get_axis(0)
    parcel_nets = []
    for key, label in label_axis.label.items():
        if key == 0:
            continue
        parts = label.label.split("_")
        net   = parts[2] if len(parts) > 2 else "Unknown"
        parcel_nets.append((key - 1, net))   # 0-indexed

    parcel_nets.sort(key=lambda x: (
        NETWORKS.index(x[1]) if x[1] in NETWORKS else 99, x[0]
    ))
    sorted_idx = np.array([p[0] for p in parcel_nets])

    ticks, prev_net, start = [], None, 0
    for i, (_, net) in enumerate(parcel_nets):
        if net != prev_net:
            if prev_net is not None:
                ticks.append(((start + i - 1) // 2, prev_net))
            start, prev_net = i, net
    ticks.append(((start + len(parcel_nets) - 1) // 2, prev_net))
    return sorted_idx, ticks


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
def plot_fc(fc, title, out_path, sorted_idx=None, ticks=None):
    mat  = fc[np.ix_(sorted_idx, sorted_idx)] if sorted_idx is not None else fc
    vmax = np.percentile(np.abs(mat), 95)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Fisher Z")
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Parcel (sorted by network)")
    ax.set_ylabel("Parcel (sorted by network)")
    if ticks:
        t, l = zip(*ticks)
        ax.set_xticks(list(t)); ax.set_xticklabels(list(l), rotation=45, ha="right", fontsize=8)
        ax.set_yticks(list(t)); ax.set_yticklabels(list(l), fontsize=8)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()

def plot_change(fc_change, title, out_path, sorted_idx=None, ticks=None):
    mat  = fc_change[np.ix_(sorted_idx, sorted_idx)] if sorted_idx is not None else fc_change
    vmax = np.percentile(np.abs(mat), 97)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="bwr", vmin=-vmax, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="ΔFisher Z")
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Parcel (sorted by network)")
    ax.set_ylabel("Parcel (sorted by network)")
    if ticks:
        t, l = zip(*ticks)
        ax.set_xticks(list(t)); ax.set_xticklabels(list(l), rotation=45, ha="right", fontsize=8)
        ax.set_yticks(list(t)); ax.set_yticklabels(list(l), fontsize=8)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()

def plot_network_summary(fc_all, atlas_path, out_path):
    """Bar chart: mean within-network FC, ses-exp vs ses-con, across timepoints."""
    atlas      = nib.load(atlas_path)
    label_axis = atlas.header.get_axis(0)
    parcel_net = {}
    for key, label in label_axis.label.items():
        if key == 0: continue
        parts = label.label.split("_")
        parcel_net[key - 1] = parts[2] if len(parts) > 2 else "Unknown"

    rows = []
    for (ses, tp), fc in fc_all.items():
        for net in NETWORKS:
            idx = [i for i, n in parcel_net.items() if n == net]
            if len(idx) < 2: continue
            sub = fc[np.ix_(idx, idx)].copy()
            np.fill_diagonal(sub, np.nan)
            rows.append({"Session": SESSION_LABELS[ses], "Timepoint": tp,
                         "Network": net, "Mean FC (Z)": float(np.nanmean(sub))})

    df  = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    for ax, ses in zip(axes, SESSIONS):
        sub_df = df[df.Session == SESSION_LABELS[ses]]
        x      = np.arange(len(NETWORKS))
        w      = 0.2
        for i, tp in enumerate(TIMEPOINTS):
            vals = [sub_df[(sub_df.Timepoint == tp) & (sub_df.Network == n)]["Mean FC (Z)"].values
                    for n in NETWORKS]
            vals = [v[0] if len(v) else 0 for v in vals]
            color = "steelblue" if tp == BASELINE else plt.cm.Oranges(0.3 + 0.22 * i)
            ax.bar(x + i * w, vals, w, label=tp, color=color)
        ax.set_xticks(x + w * 1.5)
        ax.set_xticklabels(NETWORKS, rotation=20, ha="right")
        ax.set_title(SESSION_LABELS[ses], fontsize=12)
        ax.set_ylabel("Mean within-network FC (Fisher Z)")
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.legend(fontsize=8, title="Timepoint")

    plt.suptitle(f"Within-network FC across TUS timepoints — {SUBJECT}", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  Saved: {Path(out_path).name}")

def plot_exp_vs_con(fc_all, atlas_path, out_path):
    """ΔFC comparison: ses-exp vs ses-con at each post-TUS timepoint."""
    post_tps = [tp for tp in TIMEPOINTS if tp != BASELINE]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    atlas      = nib.load(atlas_path)
    label_axis = atlas.header.get_axis(0)
    parcel_net = {}
    for key, label in label_axis.label.items():
        if key == 0: continue
        parts = label.label.split("_")
        parcel_net[key - 1] = parts[2] if len(parts) > 2 else "Unknown"

    rows = []
    for tp_idx, tp in enumerate(post_tps):
        # ΔFC = post − baseline for each session
        dfc_exp = fc_all[("ses-exp", tp)] - fc_all[("ses-exp", BASELINE)]
        dfc_con = fc_all[("ses-con", tp)] - fc_all[("ses-con", BASELINE)]
        diff    = dfc_exp - dfc_con   # focused − defocused

        vmax = np.percentile(np.abs(diff), 97)
        for row_idx, (mat, label) in enumerate([(dfc_exp, f"Focused ΔFC\n{tp}"),
                                                 (dfc_con, f"Defocused ΔFC\n{tp}"),
                                                 (diff, f"Focused − Defocused\n{tp}")]):
            ax = axes[row_idx // 3 if False else row_idx % 2][tp_idx] if False else None

        # Row 0: ses-exp ΔFC
        ax = axes[0][tp_idx]
        vmax_d = np.percentile(np.abs(dfc_exp), 97)
        im = ax.imshow(dfc_exp, cmap="bwr", vmin=-vmax_d, vmax=vmax_d,
                       aspect="auto", interpolation="nearest")
        ax.set_title(f"Focused ΔFC — {tp}", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Row 1: ses-con ΔFC
        ax = axes[1][tp_idx]
        vmax_c = np.percentile(np.abs(dfc_con), 97)
        im = ax.imshow(dfc_con, cmap="bwr", vmin=-vmax_c, vmax=vmax_c,
                       aspect="auto", interpolation="nearest")
        ax.set_title(f"Defocused ΔFC — {tp}", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(f"ΔFC (post − baseline): Focused vs Defocused — {SUBJECT}", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  Saved: {Path(out_path).name}")


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
def run(data_dir, atlas_path, out_dir):
    out_dir = Path(out_dir)
    dirs = {k: out_dir / k for k in ["fc_matrices", "fc_change", "parcel_ts", "figures"]}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print("\n[Atlas]")
    labels     = load_atlas(atlas_path)
    sorted_idx, net_ticks = get_network_order(atlas_path)

    fc_all = {}   # {(session, timepoint): fc_matrix}

    for ses in SESSIONS:
        print(f"\n{'='*60}")
        print(f"SESSION: {ses} ({SESSION_LABELS[ses]})")
        print("=" * 60)

        for acq in TIMEPOINTS:
            print(f"\n  [Timepoint: {acq}]")

            # Load BOLD
            data = load_cifti(bold_path(data_dir, ses, acq))

            # Load confounds
            conf = load_confounds(confounds_path(data_dir, ses, acq))

            # Extract parcels
            parcel_ts = extract_parcels(data, labels)

            # Denoise
            parcel_ts_clean = denoise(parcel_ts, conf)

            # Save timeseries
            np.save(str(dirs["parcel_ts"] / f"ts_{ses}_{acq}.npy"), parcel_ts_clean)

            # FC matrix
            fc = compute_fc(parcel_ts_clean)
            fc_all[(ses, acq)] = fc
            np.save(str(dirs["fc_matrices"] / f"fc_{ses}_{acq}.npy"), fc)

            # Plot FC matrix
            plot_fc(fc,
                    title=f"FC — {ses} {acq} ({SUBJECT})",
                    out_path=dirs["figures"] / f"fc_{ses}_{acq}.png",
                    sorted_idx=sorted_idx, ticks=net_ticks)
            print(f"    Saved FC figure: fc_{ses}_{acq}.png")

        # ΔFC vs baseline within session
        print(f"\n  [ΔFC vs baseline — {ses}]")
        for acq in TIMEPOINTS:
            if acq == BASELINE:
                continue
            dfc = fc_all[(ses, acq)] - fc_all[(ses, BASELINE)]
            np.save(str(dirs["fc_change"] / f"dfc_{ses}_{acq}_vs_{BASELINE}.npy"), dfc)
            plot_change(dfc,
                        title=f"ΔFC ({acq} − {BASELINE}) — {ses} ({SUBJECT})",
                        out_path=dirs["figures"] / f"dfc_{ses}_{acq}_vs_{BASELINE}.png",
                        sorted_idx=sorted_idx, ticks=net_ticks)
            print(f"    Saved ΔFC: dfc_{ses}_{acq}_vs_{BASELINE}.png")

    # ─── Cross-session comparison figures ───
    print(f"\n[Cross-session plots]")
    plot_network_summary(fc_all, atlas_path,
                         out_path=dirs["figures"] / "network_summary_both_sessions.png")
    plot_exp_vs_con(fc_all, atlas_path,
                    out_path=dirs["figures"] / "delta_fc_exp_vs_con.png")

    # ─── Radar/Spider chart ───
    print(f"\n[Radar chart]")
    plot_radar_network(fc_all, atlas_path,
                       out_path=dirs["figures"] / "radar_network_fc.png")

    # ─── Brain surface plots ───
    print(f"\n[Brain surface plots]")
    plot_brain_surface(fc_all, atlas_path,
                       out_dir=dirs["figures"])

    # ─── CSV summary ───
    atlas      = nib.load(atlas_path)
    label_axis = atlas.header.get_axis(0)
    parcel_net = {}
    for key, label in label_axis.label.items():
        if key == 0: continue
        parts = label.label.split("_")
        parcel_net[key - 1] = parts[2] if len(parts) > 2 else "Unknown"

    rows = []
    for (ses, tp), fc in fc_all.items():
        for net in NETWORKS:
            idx = [i for i, n in parcel_net.items() if n == net]
            if len(idx) < 2: continue
            sub = fc[np.ix_(idx, idx)].copy()
            np.fill_diagonal(sub, np.nan)
            rows.append({
                "subject": SUBJECT, "session": ses,
                "session_label": SESSION_LABELS[ses],
                "timepoint": tp, "network": net,
                "mean_fc_z": round(float(np.nanmean(sub)), 4),
                "std_fc_z":  round(float(np.nanstd(sub)), 4),
            })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "fc_summary.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"\n  Saved summary: fc_summary.csv")
    print("\n" + df.pivot_table(
        index=["session", "timepoint"], columns="network",
        values="mean_fc_z", aggfunc="first"
    ).to_string())

    print(f"\n{'='*60}")
    print(f"Xong! Tất cả output trong: {out_dir}")
    print("=" * 60)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FC pipeline cho sub-05 — ses-exp và ses-con"
    )
    parser.add_argument("--data_dir", required=True,
        help='Thư mục gốc chứa ses-exp/ và ses-con/ '
             '(ví dụ: "/Volumes/Extreme SSD/THESIS MSC/MEPrep output")')
    parser.add_argument("--atlas", required=True,
        help="Đường dẫn đến Schaefer 200 parcels dlabel.nii")
    parser.add_argument("--out_dir", default="./fc_output",
        help="Thư mục output (mặc định: ./fc_output)")
    args = parser.parse_args()

    run(data_dir=args.data_dir,
        atlas_path=args.atlas,
        out_dir=args.out_dir)


# ─────────────────────────────────────────────
# Thêm: Radar/Spider chart
# ─────────────────────────────────────────────
def plot_radar_network(fc_all, atlas_path, out_path):
    """
    Spider/radar chart: mean FC của sgACC seed với từng network
    Tương đương hình B của Folloni 2019
    Mỗi trục = 1 network, mỗi đường = 1 timepoint × session
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.path import Path
    import matplotlib.patheffects as pe

    atlas      = nib.load(atlas_path)
    label_axis = atlas.header.get_axis(0)
    parcel_net = {}
    for key, label in label_axis.label.items():
        if key == 0: continue
        parts = label.label.split("_")
        parcel_net[key - 1] = parts[2] if len(parts) > 2 else "Unknown"

    # Tìm parcels thuộc Limbic network (gần nhất với sgACC/Default)
    # Dùng Default Mode Network làm "seed network" vì sgACC thuộc DMN
    seed_network = "Default"
    seed_idx = [i for i, n in parcel_net.items() if n == seed_network]

    target_networks = NETWORKS  # 7 networks làm các trục

    n_axes = len(target_networks)
    angles = [n / float(n_axes) * 2 * np.pi for n in range(n_axes)]
    angles += angles[:1]  # đóng vòng tròn

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                              subplot_kw=dict(polar=True))

    colors_tp = {
        "preTUS15":  "#5F5E5A",
        "postTUS15": "#E24B4A",
        "postTUS30": "#D85A30",
        "postTUS45": "#BA7517",
    }
    linestyles_tp = {
        "preTUS15":  "--",
        "postTUS15": "-",
        "postTUS30": "-",
        "postTUS45": "-",
    }

    for ax_idx, (ax, ses) in enumerate(zip(axes, SESSIONS)):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(target_networks, size=10)
        ax.set_ylim(-0.35, 0.35)
        ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
        ax.set_yticklabels(["-0.2", "-0.1", "0", "0.1", "0.2"], size=8)
        ax.set_title(SESSION_LABELS[ses], size=12, pad=20)

        for tp in TIMEPOINTS:
            fc = fc_all[(ses, tp)]
            # Mean FC từ seed network đến mỗi target network
            vals = []
            for net in target_networks:
                tgt_idx = [i for i, n in parcel_net.items() if n == net]
                if net == seed_network:
                    # Within-network: lấy off-diagonal
                    sub = fc[np.ix_(seed_idx, tgt_idx)].copy()
                    np.fill_diagonal(sub, np.nan)
                    vals.append(float(np.nanmean(sub)))
                else:
                    sub = fc[np.ix_(seed_idx, tgt_idx)]
                    vals.append(float(np.nanmean(sub)))

            vals += vals[:1]  # đóng vòng tròn

            ax.plot(angles, vals,
                    color=colors_tp[tp],
                    linestyle=linestyles_tp[tp],
                    linewidth=2 if tp != "preTUS15" else 1.5,
                    label=tp, alpha=0.9)
            ax.fill(angles, vals,
                    color=colors_tp[tp], alpha=0.05)

        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

    # Legend chung
    handles = [mpatches.Patch(color=c, label=t)
               for t, c in colors_tp.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=10, title="Timepoint", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"FC của Default Mode Network với các mạng lưới khác\n"
        f"Focused vs Defocused TUS — {SUBJECT}",
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {Path(out_path).name}")


# ─────────────────────────────────────────────
# Thêm: Brain surface plot (ΔFC lên bề mặt não)
# ─────────────────────────────────────────────
def plot_brain_surface(fc_all, atlas_path, out_dir):
    """
    Vẽ ΔFC lên bề mặt não dạng flatmap — giống hình A Folloni 2019
    Dùng nilearn plotting với fsaverage template
    Tạo hình cho từng post-TUS timepoint, so sánh exp vs con
    """
    try:
        from nilearn import plotting, datasets, surface
    except ImportError:
        print("  ⚠️  nilearn plotting không khả dụng — bỏ qua brain surface plot")
        print("     Chạy: pip install nilearn matplotlib")
        return

    atlas      = nib.load(atlas_path)
    label_axis = atlas.header.get_axis(0)
    parcel_net = {}
    for key, label in label_axis.label.items():
        if key == 0: continue
        parts = label.label.split("_")
        parcel_net[key - 1] = parts[2] if len(parts) > 2 else "Unknown"

    # Lấy fsaverage surface template
    try:
        fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    except Exception as e:
        print(f"  ⚠️  Không tải được fsaverage: {e}")
        return

    # Atlas labels trên fsLR 91k — cần map về fsaverage
    # Lấy label array: shape (91282,) — left hem: 0:32492, right hem: 32492:64984
    atlas_data = np.array(atlas.get_fdata()).squeeze().astype(int)
    n_left  = 32492   # vertices bán cầu trái trên fsLR 32k (trong 91k CIFTI)
    n_right = 32492   # bán cầu phải

    post_tps = [tp for tp in TIMEPOINTS if tp != BASELINE]

    for tp in post_tps:
        fig, axes_grid = plt.subplots(2, 4, figsize=(20, 9))
        fig.suptitle(
            f"ΔFC tại {tp} vs baseline — Focused (top) & Defocused (bottom)\n{SUBJECT}",
            fontsize=13
        )

        for row_idx, ses in enumerate(SESSIONS):
            dfc = fc_all[(ses, tp)] - fc_all[(ses, BASELINE)]
            # dfc shape: (200, 200)

            # Map parcel ΔFC lên vertices
            # Dùng mean ΔFC của mỗi parcel (mean across all other parcels)
            parcel_mean_dfc = np.nanmean(np.where(
                np.eye(200, dtype=bool), np.nan, dfc
            ), axis=1)  # shape (200,)

            # Tạo vertex-level map
            vertex_map = np.zeros(91282)
            for p_idx in range(200):
                mask = (atlas_data == p_idx + 1)
                vertex_map[mask] = parcel_mean_dfc[p_idx]

            # Split left/right hemisphere
            lh_data = vertex_map[:n_left]
            rh_data = vertex_map[n_left:n_left + n_right]

            vmax = np.percentile(np.abs(parcel_mean_dfc), 95)
            vmax = max(vmax, 0.05)  # minimum range

            views = ["lateral", "medial"]
            hemis = [("left", lh_data, fsaverage.infl_left, fsaverage.sulc_left),
                     ("right", rh_data, fsaverage.infl_right, fsaverage.sulc_right)]

            col = 0
            for hemi_name, hemi_data, mesh, bg_map in hemis:
                for view in views:
                    ax = axes_grid[row_idx][col]
                    try:
                        # Resample: atlas fsLR → fsaverage5
                        # Karena fsLR 32k dan fsaverage5 sama jumlah verteksnya
                        # kita bisa langsung pakai hemi_data jika ukurannya cocok
                        if len(hemi_data) == 32492:
                            # Resample ke fsaverage5 (10242 vertices)
                            from nilearn.surface import load_surf_mesh
                            from scipy.interpolate import griddata

                            # Simple approach: plot langsung menggunakan texture
                            display = plotting.plot_surf_stat_map(
                                surf_mesh=mesh,
                                stat_map=hemi_data,
                                hemi=hemi_name,
                                view=view,
                                bg_map=bg_map,
                                colorbar=False,
                                cmap="RdYlBu_r",
                                vmax=vmax,
                                threshold=vmax * 0.1,
                                axes=ax,
                                figure=fig,
                            )
                    except Exception as e:
                        ax.text(0.5, 0.5, f"{hemi_name}\n{view}\n(error)",
                               ha="center", va="center", transform=ax.transAxes,
                               fontsize=9, color="gray")
                        ax.axis("off")
                    col += 1

            # Label cho mỗi row
            axes_grid[row_idx][0].set_ylabel(
                SESSION_LABELS[ses], fontsize=10, labelpad=10
            )

        # Colorbar chung
        sm = plt.cm.ScalarMappable(
            cmap="RdYlBu_r",
            norm=plt.Normalize(-vmax, vmax)
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes_grid, shrink=0.4, pad=0.02)
        cbar.set_label("Mean ΔFC (Fisher Z)", fontsize=10)

        out_path = Path(out_dir) / f"brain_surface_dfc_{tp}.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path.name}")
