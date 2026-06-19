"""
rs_fmri/05_radar_chart.py
==========================
Step 5: Radar (spider) chart of sgACC network connectivity profile.

Each axis = one sgACC network node.
Each line = one condition × timepoint combination.
Shows how the connectivity profile of sgACC changes post-TUS.

Variants:
  A. Per-subject radars (2×4 grid: session × timepoint)
  B. Group-average radar: exp vs con, coloured by timepoint
  C. Change-from-baseline radar: Δ FC (post − pre) for exp vs con

Output: derivatives/rs_fmri/figures/
"""

import sys
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_rs import (
    OUT_ROOT, SUBJECTS, SESSIONS, TIMEPOINTS,
    TP_LABELS, SES_LABELS, SES_COLORS, TP_COLORS,
    NETWORK_ROIS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("radar_chart")

NET_DIR = OUT_ROOT / "roi_network"
FIG_DIR = OUT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ROI_NAMES = [r for r in NETWORK_ROIS if r not in ("sgACC_L", "sgACC_R")]
N_AXES    = len(ROI_NAMES)
ANGLES    = np.linspace(0, 2 * np.pi, N_AXES, endpoint=False).tolist()
ANGLES   += ANGLES[:1]   # close the polygon


def load_fc_table() -> pd.DataFrame:
    path = NET_DIR / "sgacc_network_fc_all.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Run 03_roi_network.py first.")
    return pd.read_csv(path, sep="\t")


def radar_values(df: pd.DataFrame, ses: str, tp: str, sub: str = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, sem) radar vectors for a given ses/timepoint (and optional subject)."""
    mask = (df["session"] == ses) & (df["timepoint"] == tp)
    if sub:
        mask &= (df["subject"] == sub)
    vals = [df[mask & (df["roi"] == roi)]["fc_z"].values for roi in ROI_NAMES]
    means = np.array([v.mean() if len(v) > 0 else np.nan for v in vals])
    sems  = np.array([v.std() / np.sqrt(len(v)) if len(v) > 1 else 0 for v in vals])
    return means, sems


def draw_radar(ax, values: np.ndarray, color: str, label: str,
               alpha_fill: float = 0.12, lw: float = 2.0):
    """Draw one radar polygon on ax with markers at every ROI axis."""
    vals = np.concatenate([values, values[:1]])
    ax.plot(ANGLES, vals, "-o", color=color, lw=lw, markersize=5,
            label=label, zorder=3, clip_on=False)
    ax.fill(ANGLES, vals, color=color, alpha=alpha_fill, zorder=2)


def style_radar_ax(ax, title: str = "", r_lim: tuple = (-1.0, 1.0)):
    """Style a polar radar axis with readable labels and sufficient grid lines."""
    r_min, r_max = r_lim
    ax.set_xticks(ANGLES[:-1])
    # Wrap long ROI names at underscore
    wrapped = [r.replace("_", "\n") for r in ROI_NAMES]
    ax.set_xticklabels(wrapped, fontsize=8, ha="center")
    ax.set_ylim(r_min, r_max)

    # Generate 5–7 evenly spaced grid rings, rounded to nearest 0.1
    span = r_max - r_min
    step = max(0.1, round(span / 6, 1))
    tick_start = np.ceil(r_min / step) * step
    yticks = np.arange(tick_start, r_max + step * 0.1, step)
    yticks = np.round(yticks, 2)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:.1f}" for v in yticks], fontsize=7, color="gray")

    # Explicit zero ring so negative/positive is visually clear
    zero_ring = [0] * (N_AXES + 1)
    ax.plot(ANGLES, zero_ring, "-", color="black", lw=0.9, alpha=0.5, zorder=2)

    ax.spines["polar"].set_visible(False)
    ax.grid(color="gray", lw=0.4, alpha=0.35)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=18)


# ── Plot B: Group-average radar (session × timepoint) ─────────────────────────

def _auto_r_lim(df: pd.DataFrame, padding: float = 0.15) -> tuple:
    """Compute symmetric radial limits with padding from data extremes."""
    fc_vals = df["fc_z"].dropna().values
    if len(fc_vals) == 0:
        return (-1.0, 1.0)
    abs_max = np.abs(fc_vals).quantile(0.97) if hasattr(fc_vals, "quantile") else np.percentile(np.abs(fc_vals), 97)
    r_max = float(abs_max) * (1 + padding)
    r_max = max(r_max, 0.3)  # floor so trivially-small FC still shows a readable chart
    return (-r_max, r_max)


def plot_group_radar(df: pd.DataFrame):
    """4-panel radar: one per timepoint, exp vs con overlaid."""
    tp_short = ["Pre", "+15 min", "+30 min", "+45 min"]
    fig, axes = plt.subplots(1, len(TIMEPOINTS), figsize=(6 * len(TIMEPOINTS), 6),
                             subplot_kw={"projection": "polar"}, facecolor="white")
    fig.suptitle("sgACC network connectivity profile — group average\n(exp-focused vs con-defocused)",
                 fontsize=13, fontweight="bold", y=1.04)

    r_lim = _auto_r_lim(df)

    for ax, tp, short in zip(axes, TIMEPOINTS, tp_short):
        for ses in SESSIONS:
            ses_short = "EXP (focused)" if ses == "ses-exp" else "CON (defocused)"
            means, _ = radar_values(df, ses, tp)
            draw_radar(ax, means, SES_COLORS[ses], ses_short)
        style_radar_ax(ax, short, r_lim)

    # Legend outside the last panel to avoid covering data
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.15, 1.05),
                    fontsize=9, frameon=True, borderpad=0.8)

    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    out = FIG_DIR / "radar_group_by_timepoint.png"
    plt.savefig(str(out), dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    log.info("Saved → %s", out.name)


# ── Plot C: Change-from-baseline radar (Δ FC = postTUS − preTUS) ──────────────

def plot_delta_radar(df: pd.DataFrame):
    """Radar showing change from preTUS baseline for each postTUS timepoint."""
    post_tps  = [tp for tp in TIMEPOINTS if tp != "preTUS15"]
    tp_short  = ["+15 min", "+30 min", "+45 min"]
    fig, axes = plt.subplots(1, len(post_tps), figsize=(6 * len(post_tps), 6),
                             subplot_kw={"projection": "polar"}, facecolor="white")
    fig.suptitle("Δ sgACC network FC from pre-TUS baseline\n(exp-focused vs con-defocused)",
                 fontsize=13, fontweight="bold", y=1.04)

    # Compute delta range across all sessions/timepoints for consistent axis
    deltas_all = []
    for ses in SESSIONS:
        pre_means, _ = radar_values(df, ses, "preTUS15")
        for tp in post_tps:
            post_means, _ = radar_values(df, ses, tp)
            deltas_all.extend((post_means - pre_means).tolist())
    abs_max = np.nanmax(np.abs(deltas_all)) * 1.15
    abs_max = max(abs_max, 0.2)
    r_lim = (-abs_max, abs_max)

    for ax, tp, short in zip(axes, post_tps, tp_short):
        for ses in SESSIONS:
            ses_short = "EXP (focused)" if ses == "ses-exp" else "CON (defocused)"
            pre_means, _  = radar_values(df, ses, "preTUS15")
            post_means, _ = radar_values(df, ses, tp)
            delta = post_means - pre_means
            draw_radar(ax, delta, SES_COLORS[ses], ses_short)
        style_radar_ax(ax, f"Δ {short}", r_lim)

    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.15, 1.05),
                    fontsize=9, frameon=True, borderpad=0.8)

    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    out = FIG_DIR / "radar_delta_from_baseline.png"
    plt.savefig(str(out), dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    log.info("Saved → %s", out.name)


# ── Plot A: Per-subject radar grid ────────────────────────────────────────────

def plot_per_subject_radars(df: pd.DataFrame):
    """Grid: each subject gets a row, each column = timepoint, exp vs con overlaid."""
    n_subs  = len(SUBJECTS)
    n_tps   = len(TIMEPOINTS)
    tp_short = ["Pre", "+15 min", "+30 min", "+45 min"]

    fig, axes = plt.subplots(n_subs, n_tps, figsize=(5.5 * n_tps, 5 * n_subs),
                             subplot_kw={"projection": "polar"}, facecolor="white")
    fig.suptitle("Per-subject sgACC network connectivity radar",
                 fontsize=14, fontweight="bold", y=1.01)

    r_lim = _auto_r_lim(df)

    for r, sub in enumerate(SUBJECTS):
        for c, tp in enumerate(TIMEPOINTS):
            ax = axes[r, c]
            for ses in SESSIONS:
                ses_short = "EXP" if ses == "ses-exp" else "CON"
                means, _ = radar_values(df, ses, tp, sub=sub)
                if not np.all(np.isnan(means)):
                    draw_radar(ax, means, SES_COLORS[ses], ses_short)

            col_title = tp_short[c]
            row_label = sub if c == 0 else ""
            full_title = f"{sub}\n{col_title}" if c == 0 else col_title
            style_radar_ax(ax, full_title, r_lim)

            if r == 0 and c == n_tps - 1:
                ax.legend(loc="upper left", bbox_to_anchor=(1.15, 1.1),
                          fontsize=8, frameon=True, borderpad=0.6)

    fig.tight_layout(rect=[0, 0, 0.93, 0.98], h_pad=3.5, w_pad=2.0)
    out = FIG_DIR / "radar_per_subject.png"
    plt.savefig(str(out), dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    log.info("Saved → %s", out.name)


def main():
    df = load_fc_table()
    log.info("Loaded FC table: %d rows", len(df))
    plot_group_radar(df)
    plot_delta_radar(df)
    plot_per_subject_radars(df)


if __name__ == "__main__":
    main()
