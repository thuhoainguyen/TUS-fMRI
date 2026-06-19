"""
rs_fmri/04_temporal_plots.py
==============================
Step 4: Visualise sgACC connectivity change over time.

Plots:
  A. Line plot — sgACC FC to each network node across timepoints (exp vs con)
     One panel per ROI, lines = exp / con, shaded SEM across subjects
  B. Heatmap — connectivity matrix per condition × timepoint (group average)
  C. Brain overlay — group-average FC seed map at each timepoint (exp vs con)

Output: derivatives/rs_fmri/figures/
"""

import sys
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_rs import (
    OUT_ROOT, SUBJECTS, SESSIONS, TIMEPOINTS,
    TP_LABELS, SES_LABELS, SES_COLORS, TP_COLORS,
    NETWORK_ROIS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("temporal_plots")

NET_DIR = OUT_ROOT / "roi_network"
FIG_DIR = OUT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ROI_NAMES = [r for r in NETWORK_ROIS if r not in ("sgACC_L", "sgACC_R")]
TP_X      = list(range(len(TIMEPOINTS)))
TP_TICK   = ["Pre", "+15 min", "+30 min", "+45 min"]


def load_fc_table() -> pd.DataFrame:
    path = NET_DIR / "sgacc_network_fc_all.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Run 03_roi_network.py first. Not found: {path}")
    return pd.read_csv(path, sep="\t")


# ── Plot A: Line plots per ROI, exp vs con across timepoints ──────────────────

def plot_temporal_lines(df: pd.DataFrame):
    n_rois = len(ROI_NAMES)
    n_cols = 4
    n_rows = int(np.ceil(n_rois / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows),
                             facecolor="white", constrained_layout=False)
    fig.suptitle("sgACC network connectivity over time\n(exp-focused vs con-defocused)",
                 fontsize=14, fontweight="bold", y=0.99)
    axes = axes.flatten()

    for ax_i, roi in enumerate(ROI_NAMES):
        ax = axes[ax_i]
        roi_df = df[df["roi"] == roi]

        for ses in SESSIONS:
            ses_df = roi_df[roi_df["session"] == ses]
            tp_vals = []
            for tp in TIMEPOINTS:
                tp_df = ses_df[ses_df["timepoint"] == tp]
                tp_vals.append(tp_df["fc_z"].values)

            means = [v.mean() if len(v) > 0 else np.nan for v in tp_vals]
            sems  = [v.std() / np.sqrt(len(v)) if len(v) > 1 else 0 for v in tp_vals]

            color = SES_COLORS[ses]
            ax.plot(TP_X, means, "o-", color=color, lw=2,
                    label=SES_LABELS[ses], zorder=3, markersize=5)
            ax.fill_between(TP_X,
                            [m - s for m, s in zip(means, sems)],
                            [m + s for m, s in zip(means, sems)],
                            color=color, alpha=0.15, zorder=2)

        ax.axhline(0, color="gray", lw=1.0, ls="--", alpha=0.6, zorder=1)
        ax.axvline(0.5, color="black", lw=0.8, ls=":", alpha=0.4)
        ax.set_xticks(TP_X)
        ax.set_xticklabels(TP_TICK, fontsize=9)
        ax.set_title(roi, fontsize=11, fontweight="bold", pad=6)
        ax.set_ylabel("FC (Fisher z)", fontsize=9)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if ax_i == 0:
            leg = ax.legend(fontsize=8, frameon=True,
                            loc="upper left", bbox_to_anchor=(0.0, 1.0))
            leg.get_frame().set_linewidth(0.5)

    for ax in axes[n_rois:]:
        ax.set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0, w_pad=2.5)
    out = FIG_DIR / "sgacc_temporal_lines.png"
    plt.savefig(str(out), dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    log.info("Saved → %s", out.name)


# ── Plot B: Heatmap of connectivity matrix per condition × timepoint ───────────

def plot_connectivity_heatmaps(df: pd.DataFrame):
    n_rois = len(ROI_NAMES)
    tp_short = ["Pre", "+15 min", "+30 min", "+45 min"]

    # Symmetric color scale based on data
    all_means = [
        df[(df["session"] == ses) & (df["timepoint"] == tp) & (df["roi"] == roi)]["fc_z"].mean()
        for ses in SESSIONS for tp in TIMEPOINTS for roi in ROI_NAMES
    ]
    abs_max = np.nanmax(np.abs(all_means))
    vlim = np.ceil(abs_max * 10) / 10  # round up to nearest 0.1

    fig, axes = plt.subplots(len(SESSIONS), len(TIMEPOINTS),
                             figsize=(4 * len(TIMEPOINTS) + 2, 2.5 * n_rois * len(SESSIONS) / 5 + 1.5),
                             facecolor="white", sharex=False, sharey=True)
    fig.suptitle("sgACC → network node connectivity (Fisher z, group average)",
                 fontsize=13, fontweight="bold", y=1.01)

    for r, ses in enumerate(SESSIONS):
        for c, tp in enumerate(TIMEPOINTS):
            ax = axes[r, c]
            vals = np.array([
                df[(df["session"] == ses) & (df["timepoint"] == tp) & (df["roi"] == roi)]["fc_z"].mean()
                for roi in ROI_NAMES
            ]).reshape(-1, 1)

            im = ax.imshow(vals, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")

            ax.set_yticks(range(n_rois))
            ax.set_yticklabels(ROI_NAMES, fontsize=9)
            ax.set_xticks([])

            # Annotate cell values
            for i, v in enumerate(vals[:, 0]):
                if not np.isnan(v):
                    txt_color = "white" if abs(v) > 0.6 * vlim else "black"
                    ax.text(0, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=8, color=txt_color, fontweight="bold")

            col_title = tp_short[c]
            if r == 0:
                ax.set_title(col_title, fontsize=10, fontweight="bold", pad=6)
            if c == 0:
                ses_short = "EXP (focused)" if ses == "ses-exp" else "CON (defocused)"
                ax.set_ylabel(ses_short, fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=axes, shrink=0.6, label="Fisher z",
                        pad=0.03, aspect=30)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Fisher z", fontsize=10)

    fig.tight_layout(rect=[0, 0, 0.93, 0.98])
    out = FIG_DIR / "sgacc_connectivity_heatmap.png"
    plt.savefig(str(out), dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    log.info("Saved → %s", out.name)


# ── Plot C: Group-average FC seed maps on brain ───────────────────────────────

def plot_group_fc_brainmaps():
    """Average Fisher-z FC maps across subjects and display on MNI brain."""
    try:
        from nilearn import plotting, image
    except ImportError:
        log.warning("nilearn not available for brain map plotting")
        return

    seed_fc_dir = OUT_ROOT / "seed_fc"
    seed_name   = "sgACC_L"  # primary seed

    from config_rs import BOLD_SPACE
    import nibabel as nib

    fig, axes = plt.subplots(len(SESSIONS), len(TIMEPOINTS),
                             figsize=(20, 8), facecolor="black")
    fig.suptitle("Group-average sgACC_L seed FC map (Fisher z)\nAxial cut at z = −8 mm",
                 color="white", fontsize=13, fontweight="bold")

    PLOT_KW = dict(display_mode="z", cut_coords=[-8], colorbar=False,
                   black_bg=True, annotate=False, draw_cross=False,
                   vmin=-1.5, vmax=1.5, cmap="RdBu_r")

    for r, ses in enumerate(SESSIONS):
        for c, tp in enumerate(TIMEPOINTS):
            ax = axes[r, c]
            maps = []
            for sub in SUBJECTS:
                fc_path = seed_fc_dir / sub / ses / f"{sub}_{ses}_acq-{tp}_seed-{seed_name}_desc-fc_map.nii.gz"
                if fc_path.exists():
                    maps.append(image.load_img(str(fc_path)))
            if not maps:
                ax.set_facecolor("black")
                ax.set_title(f"{tp}\n(no data)", color="white", fontsize=8)
                continue
            mean_img = image.mean_img(maps)
            d = plotting.plot_stat_map(mean_img, axes=ax, figure=fig, **PLOT_KW)
            if r == 0:
                ax.set_title(TP_LABELS[tp], color="white", fontsize=9)
            if c == 0:
                ax.set_ylabel(SES_LABELS[ses], color="white", fontsize=9)

    out = FIG_DIR / "sgacc_group_fc_brainmaps.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    log.info("Saved → %s", out.name)


def main():
    df = load_fc_table()
    log.info("Loaded FC table: %d rows, subjects: %s", len(df), df["subject"].unique().tolist())
    plot_temporal_lines(df)
    plot_connectivity_heatmaps(df)
    plot_group_fc_brainmaps()


if __name__ == "__main__":
    main()
