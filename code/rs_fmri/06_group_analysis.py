"""
rs_fmri/06_group_analysis.py
==============================
Step 6: Group-level analysis — statistics and summary tables.

Tests (non-parametric given N=5):
  - Wilcoxon signed-rank test: exp vs con at each timepoint × ROI
  - Friedman test: FC change across timepoints within each session × ROI
  - Effect size: Cohen's d (with Hedges' correction for small N)

Output:
  derivatives/rs_fmri/group/
    sgacc_network_stats.tsv        — all pairwise tests
    sgacc_network_summary.tsv      — mean ± SD per condition × timepoint × ROI
    figures/sgacc_change_barplot.png
"""

import sys
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_rs import (
    OUT_ROOT, SUBJECTS, SESSIONS, TIMEPOINTS,
    TP_LABELS, SES_LABELS, SES_COLORS, TP_COLORS,
    NETWORK_ROIS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("group_analysis")

NET_DIR   = OUT_ROOT / "roi_network"
GROUP_DIR = OUT_ROOT / "group"
FIG_DIR   = OUT_ROOT / "figures"
GROUP_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

ROI_NAMES = [r for r in NETWORK_ROIS if r not in ("sgACC_L", "sgACC_R")]


def load_fc_table() -> pd.DataFrame:
    path = NET_DIR / "sgacc_network_fc_all.tsv"
    if not path.exists():
        raise FileNotFoundError("Run 03_roi_network.py first.")
    return pd.read_csv(path, sep="\t")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Hedges' g (bias-corrected Cohen's d) for paired samples."""
    n = len(a)
    diff = a - b
    sd   = diff.std(ddof=1)
    if sd == 0:
        return 0.0
    d = diff.mean() / sd
    # Hedges' correction factor
    correction = 1 - (3 / (4 * (n - 1) - 1))
    return float(d * correction)


def run_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each ROI × timepoint: Wilcoxon exp vs con.
    For each ROI × session:   Friedman across timepoints.
    """
    records = []
    for roi in ROI_NAMES:
        roi_df = df[df["roi"] == roi]

        # ── Wilcoxon: exp vs con at each timepoint ──
        for tp in TIMEPOINTS:
            exp_vals = []
            con_vals = []
            for sub in SUBJECTS:
                e = roi_df[(roi_df["session"] == "ses-exp") &
                            (roi_df["timepoint"] == tp) &
                            (roi_df["subject"] == sub)]["fc_z"].values
                c = roi_df[(roi_df["session"] == "ses-con") &
                            (roi_df["timepoint"] == tp) &
                            (roi_df["subject"] == sub)]["fc_z"].values
                if len(e) > 0 and len(c) > 0:
                    exp_vals.append(e[0])
                    con_vals.append(c[0])

            exp_arr = np.array(exp_vals)
            con_arr = np.array(con_vals)
            n = len(exp_arr)

            if n >= 3:
                try:
                    stat, p = stats.wilcoxon(exp_arr, con_arr, alternative="two-sided")
                except ValueError:
                    stat, p = np.nan, np.nan
            else:
                stat, p = np.nan, np.nan

            records.append({
                "roi":        roi,
                "test":       "wilcoxon_exp_vs_con",
                "timepoint":  tp,
                "session":    "both",
                "n":          n,
                "stat":       stat,
                "p_uncorr":   p,
                "hedges_g":   cohens_d(exp_arr, con_arr) if n >= 2 else np.nan,
                "exp_mean":   exp_arr.mean() if n > 0 else np.nan,
                "con_mean":   con_arr.mean() if n > 0 else np.nan,
            })

        # ── Friedman: FC change across timepoints per session ──
        for ses in SESSIONS:
            ses_df = roi_df[roi_df["session"] == ses]
            tp_arrays = []
            for tp in TIMEPOINTS:
                vals = [ses_df[(ses_df["subject"] == sub) &
                               (ses_df["timepoint"] == tp)]["fc_z"].values
                        for sub in SUBJECTS]
                arr = np.array([v[0] for v in vals if len(v) > 0])
                tp_arrays.append(arr)
            # Align lengths
            n = min(len(a) for a in tp_arrays)
            if n >= 3:
                try:
                    stat, p = stats.friedmanchisquare(*[a[:n] for a in tp_arrays])
                except ValueError:
                    stat, p = np.nan, np.nan
            else:
                stat, p = np.nan, np.nan

            records.append({
                "roi":        roi,
                "test":       "friedman_across_timepoints",
                "timepoint":  "all",
                "session":    ses,
                "n":          n,
                "stat":       stat,
                "p_uncorr":   p,
                "hedges_g":   np.nan,
                "exp_mean":   np.nan,
                "con_mean":   np.nan,
            })

    stats_df = pd.DataFrame(records)
    # FDR correction (Benjamini-Hochberg) across all tests
    from scipy.stats import false_discovery_control
    valid = ~stats_df["p_uncorr"].isna()
    if valid.sum() > 0:
        p_vals = stats_df.loc[valid, "p_uncorr"].values
        try:
            p_fdr = false_discovery_control(p_vals)
        except Exception:
            p_fdr = p_vals * len(p_vals)  # Bonferroni fallback
        stats_df.loc[valid, "p_fdr"] = p_fdr
    return stats_df


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± SD per session × timepoint × ROI, plus delta from baseline."""
    rows = []
    for ses in SESSIONS:
        for tp in TIMEPOINTS:
            for roi in ROI_NAMES:
                vals = df[(df["session"] == ses) & (df["timepoint"] == tp) & (df["roi"] == roi)]["fc_z"].values
                pre_vals = df[(df["session"] == ses) & (df["timepoint"] == "preTUS15") & (df["roi"] == roi)]["fc_z"].values
                rows.append({
                    "session":   ses,
                    "timepoint": tp,
                    "roi":       roi,
                    "n":         len(vals),
                    "mean_fc_z": vals.mean() if len(vals) > 0 else np.nan,
                    "sd_fc_z":   vals.std(ddof=1) if len(vals) > 1 else np.nan,
                    "delta_z":   (vals.mean() - pre_vals.mean()) if (len(vals) > 0 and len(pre_vals) > 0) else np.nan,
                })
    return pd.DataFrame(rows)


def plot_change_barplot(df: pd.DataFrame):
    """Bar plot: Δ FC (postTUS − preTUS) per ROI × session × postTUS timepoint."""
    post_tps  = [tp for tp in TIMEPOINTS if tp != "preTUS15"]
    tp_short  = ["+15 min", "+30 min", "+45 min"]
    n_rois    = len(ROI_NAMES)
    n_post    = len(post_tps)

    fig, axes = plt.subplots(1, n_post, figsize=(6 * n_post, 5.5), sharey=True, facecolor="white")
    fig.suptitle("Δ sgACC network FC from pre-TUS (group mean ± SEM)",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, tp, short in zip(axes, post_tps, tp_short):
        x = np.arange(n_rois)
        width = 0.35
        for i, ses in enumerate(SESSIONS):
            ses_short = "EXP (focused)" if ses == "ses-exp" else "CON (defocused)"
            deltas, sems = [], []
            for roi in ROI_NAMES:
                post_v = df[(df["session"] == ses) & (df["timepoint"] == tp) & (df["roi"] == roi)]["fc_z"].values
                pre_v  = df[(df["session"] == ses) & (df["timepoint"] == "preTUS15") & (df["roi"] == roi)]["fc_z"].values
                if len(post_v) > 0 and len(pre_v) > 0:
                    d = post_v.mean() - pre_v.mean()
                    s = (post_v - pre_v).std(ddof=1) / np.sqrt(len(post_v)) if len(post_v) > 1 else 0
                else:
                    d, s = np.nan, 0
                deltas.append(d)
                sems.append(s)
            ax.bar(x + i * width, deltas, width, yerr=sems,
                   label=ses_short, color=SES_COLORS[ses], alpha=0.8,
                   capsize=4, error_kw={"linewidth": 1.2}, edgecolor="none")

        ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.7)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(ROI_NAMES, rotation=40, ha="right", fontsize=9)
        ax.set_title(short, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel("Δ Fisher z", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Single shared legend outside the rightmost panel
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.0, 0.98),
               fontsize=9, frameon=True, borderpad=0.8)

    fig.tight_layout(rect=[0, 0, 0.88, 0.97])
    out = FIG_DIR / "sgacc_change_barplot.png"
    plt.savefig(str(out), dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    log.info("Saved → %s", out.name)


def main():
    df = load_fc_table()
    log.info("Running group-level statistics (N=%d subjects)...", len(df["subject"].unique()))

    stats_df   = run_statistics(df)
    summary_df = summary_table(df)

    stats_df.to_csv(str(GROUP_DIR / "sgacc_network_stats.tsv"),   sep="\t", index=False, float_format="%.4f")
    summary_df.to_csv(str(GROUP_DIR / "sgacc_network_summary.tsv"), sep="\t", index=False, float_format="%.4f")
    log.info("Saved statistics → %s", GROUP_DIR)

    # Print significant results
    sig = stats_df[(stats_df["p_fdr"] < 0.05) & (~stats_df["p_fdr"].isna())]
    if len(sig) > 0:
        log.info("Significant results (FDR p < 0.05):\n%s", sig[["roi","test","timepoint","session","p_fdr","hedges_g"]].to_string())
    else:
        log.info("No FDR-significant results (N=%d, limited power expected)", len(SUBJECTS))

    plot_change_barplot(df)


if __name__ == "__main__":
    main()
