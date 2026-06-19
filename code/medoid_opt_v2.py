# -*- coding: utf-8 -*-
"""
medoid_opt_v2.py
================
Optimized script to find the medoid frame for TUS transducer actual trajectory data.
Enhanced version (v2) with subject-level summary plots (2x2 grids) and adaptive 
axis scaling based on per-subject variation.

@author Hoai Thu Nguyen
"""

import os
import re
import csv
import math
import logging
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Run headlessly
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from lxml import etree
from contextlib import contextmanager
from matplotlib.ticker import MaxNLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import seaborn as sns
    sns.set_theme(style="white", context="notebook", font="Arial")
except ImportError:
    sns = None

# Global Plotting Defaults
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "legend.frameon": False,
})

# Styling variables
AXIS_LIMIT_MM = 1.0  # Default, will be overridden by adaptive scaling
BIN_WIDTH_MM = 0.1
POINT_SIZE = 24
POINT_ALPHA = 0.65
MEDOID_SIZE = int(POINT_SIZE * 1.25)
LINE_WIDTH = 1.5
MEDOID_LINE_WIDTH = LINE_WIDTH * 1.25
REF_LINE_WIDTH = 0.5
REF_COLOR_MINOR = "#cccccc"
HIST_SIZE = "24%"
HIST_PAD = 0.38
HIST_LABEL_SIZE = 9
HIST_TICK_SIZE = 9
GENERAL_COLOR = "#c4c4c4"
MEDOID_COLOR = "black"
XYZ_COLORS = ["#66c2a5", "#fc8d62", "#8da0cb"]
XYZ_LABELS = ["X", "Y", "Z"]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("medoid_opt_v2")

FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
POSITION_XPATH = "//Element[@type='InstrumentMarker']"
MATRIX_ORDER = "row-major"

# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_16_floats(text):
    values = [float(x) for x in FLOAT_RE.findall(text or "")]
    if len(values) == 16: return values
    return None

def matrix4d_attrib_to_text(element):
    """Parse Localite Matrix4D data00..data33 attributes into 16 floats."""
    for child in element.iterdescendants():
        tag = etree.QName(child).localname
        if tag != "Matrix4D": continue
        comp = {k: float(v) for k, v in child.attrib.items() if k.startswith("data")}
        if len(comp) < 16: continue
        values = [comp.get(f"data{r}{c}", 0.0) for r in range(4) for c in range(4)]
        return " ".join(str(v) for v in values)
    return None

def find_matrix_text_in_element(element):
    likely_attr_names = ["matrix", "affine", "transform", "transformation"]
    for attr_name, attr_value in element.attrib.items():
        if any(name in attr_name.lower() for name in likely_attr_names):
            if parse_16_floats(attr_value): return attr_value
    if element.text and parse_16_floats(element.text): return element.text
    for child in element.iterdescendants():
        if child.text and parse_16_floats(child.text): return child.text
        for k, v in child.attrib.items():
            if any(name in k.lower() for name in likely_attr_names):
                if parse_16_floats(v): return v
    return matrix4d_attrib_to_text(element)

def matrix_from_text(text, matrix_order):
    values = parse_16_floats(text)
    arr = np.array(values, dtype=float)
    return arr.reshape((4, 4)) if matrix_order == "row-major" else arr.reshape((4, 4), order="F")

def extract_translation_and_rotation(matrix):
    return matrix[:3, 3], matrix[:3, :3]

def discover_position_elements(root, position_xpath=None):
    candidates = root.xpath(position_xpath) if position_xpath else list(root.iter())
    return [el for el in candidates if find_matrix_text_in_element(el) is not None]

def element_frame_index(element):
    idx = element.get("index")
    return int(idx) if idx is not None else None

def select_elements_by_frame_range(position_elements, start_frame, end_frame):
    selected = [el for el in position_elements if (idx := element_frame_index(el)) is not None and start_frame <= idx <= end_frame]
    selected.sort(key=element_frame_index)
    return selected

def compute_medoid_index(points):
    points = np.asarray(points, dtype=float)
    diff = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    return int(np.argmin(distances.sum(axis=1))), distances.sum(axis=1)

def project_to_rotation_matrix(R):
    U, _, Vt = np.linalg.svd(R)
    R_clean = U @ Vt
    if np.linalg.det(R_clean) < 0:
        U[:, -1] *= -1
        R_clean = U @ Vt
    return R_clean

def rotation_matrix_to_euler_xyz_degrees(R):
    R = project_to_rotation_matrix(R)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-8
    if not singular:
        x, y, z = math.atan2(R[2, 1], R[2, 2]), math.atan2(-R[2, 0], sy), math.atan2(R[1, 0], R[0, 0])
    else:
        x, y, z = math.atan2(-R[1, 2], R[1, 1]), math.atan2(-R[2, 0], sy), 0.0
    return np.degrees([x, y, z])

def relative_rotations_degrees(rotations):
    if len(rotations) == 0: return np.empty((0, 3))
    R0_inv = project_to_rotation_matrix(rotations[0]).T
    rel_eulers = [rotation_matrix_to_euler_xyz_degrees(R0_inv @ project_to_rotation_matrix(R)) for R in rotations]
    return np.array(rel_eulers, dtype=float)

# ── Plotting Helpers ──────────────────────────────────────────────────────────

@contextmanager
def use_ggpubr_theme():
    if sns is not None:
        sns.set_theme(style="white", context="notebook", font="Arial")
    with plt.rc_context({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "black",
        "axes.linewidth": 0.8, "axes.labelsize": 12, "axes.titlesize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
        "legend.frameon": False, "axes.grid": False,
    }):
        yield

def _ggpubr_open_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

def _boxed_ax(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.grid(False)

def _set_square_limits(ax, limit_mm):
    ax.set_xlim(-limit_mm, limit_mm)
    ax.set_ylim(-limit_mm, limit_mm)
    ax.set_aspect("equal", adjustable="box")

def _ref_line_kwargs(color):
    return dict(color=color, linewidth=REF_LINE_WIDTH, linestyle=":", alpha=0.75, zorder=0)

def _add_y_ref_lines(ax, limit_mm, zero_color=REF_COLOR_MINOR):
    levels = [0.0]
    if limit_mm > 0.4: levels.extend([-0.5, 0.5])
    if limit_mm > 0.9: levels.extend([-1.0, 1.0])
    if limit_mm > 1.9: levels.extend([-2.0, 2.0])
    for level in levels:
        if abs(level) <= limit_mm:
            ax.axhline(level, **_ref_line_kwargs(zero_color if level == 0.0 else REF_COLOR_MINOR))

def _add_xy_ref_lines(ax, x_zero_color, y_zero_color, limit_mm):
    levels = [0.0]
    if limit_mm > 0.4: levels.extend([-0.5, 0.5])
    if limit_mm > 0.9: levels.extend([-1.0, 1.0])
    if limit_mm > 1.9: levels.extend([-2.0, 2.0])
    for level in levels:
        if abs(level) <= limit_mm:
            ax.axhline(level, **_ref_line_kwargs(y_zero_color if level == 0.0 else REF_COLOR_MINOR))
            ax.axvline(level, **_ref_line_kwargs(x_zero_color if level == 0.0 else REF_COLOR_MINOR))

def _filled_scatter(ax, x, y, *, c, s, alpha=1.0, zorder=2):
    ax.scatter(x, y, c=c, s=s, alpha=alpha, linewidths=0, edgecolors="none", zorder=zorder)

def _style_projection_ax(ax, xl, yl, title, d0, d1, limit_mm):
    _set_square_limits(ax, limit_mm)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, pad=8)
    _boxed_ax(ax)
    _add_xy_ref_lines(ax, XYZ_COLORS[d0], XYZ_COLORS[d1], limit_mm)

def _style_hist_axis_only(ax, orientation="horizontal"):
    for spine in ax.spines.values(): spine.set_visible(False)
    if orientation == "vertical": # TOP histogram
        ax.spines["bottom"].set_visible(True) # Just a line
        ax.tick_params(axis="x", labelbottom=False, length=0) # No labels/ticks on shared axis
        ax.set_ylabel("Count", fontsize=HIST_LABEL_SIZE, labelpad=3)
        ax.tick_params(axis="y", labelsize=HIST_TICK_SIZE, length=3)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    else: # RIGHT histogram
        ax.spines["left"].set_visible(True) # Just a line
        ax.tick_params(axis="y", labelleft=False, length=0) # No labels/ticks on shared axis
        ax.set_xlabel("Count", fontsize=HIST_LABEL_SIZE, labelpad=3)
        ax.tick_params(axis="x", labelsize=HIST_TICK_SIZE, length=3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))

def _add_marginal_histograms(ax, x, y, color_x, color_y, limit_mm):
    bins = np.arange(-limit_mm, limit_mm + BIN_WIDTH_MM/2, BIN_WIDTH_MM)
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", size=HIST_SIZE, pad=HIST_PAD, sharex=ax)
    ax_histy = divider.append_axes("right", size=HIST_SIZE, pad=HIST_PAD, sharey=ax)
    ax_histx.hist(x, bins=bins, color=color_x, alpha=0.8, edgecolor="none")
    ax_histy.hist(y, bins=bins, orientation="horizontal", color=color_y, alpha=0.8, edgecolor="none")
    _style_hist_axis_only(ax_histx, "vertical"); _style_hist_axis_only(ax_histy, "horizontal")
    ax.set_aspect("equal", adjustable="box")

def _scatter_points_2d(ax, x, y, medoid_idx):
    mask = np.ones(len(x), dtype=bool)
    mask[medoid_idx] = False
    _filled_scatter(ax, x[mask], y[mask], c=GENERAL_COLOR, s=POINT_SIZE, alpha=POINT_ALPHA)
    _filled_scatter(ax, [x[medoid_idx]], [y[medoid_idx]], c=MEDOID_COLOR, s=MEDOID_SIZE, alpha=1.0, zorder=5)

def _boxplot_summary(ax, data, labels, colors, ylabel, title, limit_mm):
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True, widths=0.55)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.35); patch.set_edgecolor(color)
    for med, color in zip(bp["medians"], colors):
        med.set_color(color); med.set_linewidth(1.5)
    for i, (d, color) in enumerate(zip(data, colors)):
        ax.scatter(i + 1, np.mean(d), marker="D", s=30, c=color, zorder=5)
    ax.set_ylabel(ylabel); ax.set_title(title)
    _add_y_ref_lines(ax, limit_mm)
    _ggpubr_open_ax(ax)

# ── Main Plotting Functions ───────────────────────────────────────────────────

def plot_translation_rotation_lines(trans, rot, pos_ids, med_idx, save_path, limit_mm):
    with use_ggpubr_theme():
        fig = plt.figure(figsize=(14, 7)); fig.suptitle("Transducer variation relative to starting position", fontsize=14, y=0.98)
        gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], hspace=0.35, wspace=0.25)
        panels = [
            (gs[0, 0], gs[0, 1], trans, "Δ position (mm)", "Translation vs position ID", "Translation summary", False, limit_mm),
            (gs[1, 0], gs[1, 1], rot, "Δ orientation (degree)", "Rotation vs position ID", "Rotation summary", True, 2.0)
        ]
        for ax_line_spec, ax_box_spec, vals, yl, lt, bt, show_x, lim in panels:
            ax_line, ax_box = fig.add_subplot(ax_line_spec), fig.add_subplot(ax_box_spec)
            for j, lab in enumerate(XYZ_LABELS): ax_line.plot(pos_ids, vals[:, j], label=lab, color=XYZ_COLORS[j], lw=LINE_WIDTH)
            _add_y_ref_lines(ax_line, lim)
            ax_line.axvline(pos_ids[med_idx], color=MEDOID_COLOR, lw=MEDOID_LINE_WIDTH, ls="--", label="Medoid", zorder=4)
            ax_line.set_ylim(-lim, lim); ax_line.set_ylabel(yl); ax_line.set_title(lt); _ggpubr_open_ax(ax_line); ax_line.legend(loc="best")
            _boxplot_summary(ax_box, [vals[:, 0], vals[:, 1], vals[:, 2]], ["ΔX", "ΔY", "ΔZ"], XYZ_COLORS, yl, bt, lim)
            ax_box.set_ylim(-lim, lim)
            if show_x: ax_line.set_xlabel("Position ID")
        plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()

def plot_spatial_1x4(coords, med_idx, save_path, limit_mm):
    with use_ggpubr_theme():
        fig = plt.figure(figsize=(14, 10)); fig.suptitle("Transducer variation relative to starting position", fontsize=14, y=0.98)
        gs = gridspec.GridSpec(2, 3, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)
        ax3d = fig.add_subplot(gs[0, :], projection="3d")
        mask = np.ones(len(coords), dtype=bool); mask[med_idx] = False
        ax3d.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2], c=GENERAL_COLOR, s=POINT_SIZE, alpha=POINT_ALPHA, depthshade=False)
        ax3d.scatter(coords[med_idx, 0], coords[med_idx, 1], coords[med_idx, 2], c=MEDOID_COLOR, s=MEDOID_SIZE, alpha=1.0, depthshade=False, zorder=10)
        ax3d.set_xlabel("ΔX (mm)", color=XYZ_COLORS[0]); ax3d.set_ylabel("ΔY (mm)", color=XYZ_COLORS[1]); ax3d.set_zlabel("ΔZ (mm)", color=XYZ_COLORS[2])
        ax3d.set_xlim(-limit_mm, limit_mm); ax3d.set_ylim(-limit_mm, limit_mm); ax3d.set_zlim(-limit_mm, limit_mm)
        for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
            axis.set_major_locator(MultipleLocator(0.5))
        ax3d.view_init(elev=22, azim=-58); ax3d.grid(False)
        
        projections = [
            (gs[1, 0], 0, 1, "ΔX", "ΔY", "XY projection"), 
            (gs[1, 1], 0, 2, "ΔX", "ΔZ", "XZ projection"), 
            (gs[1, 2], 1, 2, "ΔY", "ΔZ", "YZ projection")
        ]
        for spec, d0, d1, xl, yl, title in projections:
            ax = fig.add_subplot(spec)
            _scatter_points_2d(ax, coords[:, d0], coords[:, d1], med_idx)
            _style_projection_ax(ax, xl, yl, title, d0, d1, limit_mm)
            _add_marginal_histograms(ax, coords[:, d0], coords[:, d1], XYZ_COLORS[d0], XYZ_COLORS[d1], limit_mm)
        plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()

def _annotate_sd(ax, data, labels):
    """Print SD below each box as a small annotation."""
    ylim = ax.get_ylim()
    y_text = ylim[0] + 0.04 * (ylim[1] - ylim[0])
    for i, d in enumerate(data):
        sd = np.std(d)
        ax.text(i + 1, y_text, f"σ={sd:.2f}", ha="center", va="bottom", fontsize=12,
                color="#555555", zorder=6)


def plot_subject_comprehensive_summary(subject, conditions_data, limit_mm, save_path):
    """
    Summary plot: 4 rows (Exp-L, Exp-R, Con-L, Con-R)
    Each row: [Trans Line, Trans Box, Rot Line, Rot Box]
    All y-axes fixed to ±2 with 1.0 tick spacing for consistent cross-condition comparison.
    Figure-level legend (X/Y/Z only) placed directly below the title.
    SD annotated below each box column. N annotated on each line panel.
    """
    labels_order = [
        ("exp_l", "Experimental - Left"), ("exp_r", "Experimental - Right"),
        ("con_l", "Control - Left"), ("con_r", "Control - Right")
    ]
    SUMMARY_YLIM = 2.0  # Fixed ±2 for all panels — uniform cross-condition scale

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "legend.frameon": False,
        "axes.grid": False,
    }):
        fig = plt.figure(figsize=(22, 18), facecolor="white")

        # Title sits at y=0.99; legend is injected just below it via fig.text
        sub_id = subject.replace("sub-", "sub-")  # normalise if needed
        fig.suptitle(f"{sub_id} – Transducer Variations",
                     fontsize=25, y=0.995, fontweight="bold")

        # Figure-level legend: X/Y/Z lines + Medoid dashed line
        legend_handles = [
            plt.Line2D([0], [0], color=XYZ_COLORS[j], linewidth=2, label=XYZ_LABELS[j])
            for j in range(3)
        ] + [
            plt.Line2D([0], [0], color=MEDOID_COLOR, linewidth=MEDOID_LINE_WIDTH,
                       linestyle="--", label="Medoid")
        ]
        legend_labels = XYZ_LABELS + ["Medoid"]
        fig.legend(handles=legend_handles, labels=legend_labels,
                   loc="upper center", bbox_to_anchor=(0.5, 0.958),
                   ncol=4, fontsize=14, frameon=False, columnspacing=1.5)

        # Data grid — top pushed down to leave clear space for title + legend
        gs = gridspec.GridSpec(4, 4, figure=fig,
                               width_ratios=[3, 1, 3, 1],
                               top=0.910, bottom=0.05, left=0.06, right=0.98,
                               hspace=0.50, wspace=0.30)

        for row_idx, (key, cond_title) in enumerate(labels_order):
            if key not in conditions_data:
                continue

            pos_ids, trans, rots, medoid_idx = conditions_data[key]

            # ── Translation Line ──────────────────────────────────────────────
            ax_t_line = fig.add_subplot(gs[row_idx, 0])
            for j, lab in enumerate(XYZ_LABELS):
                ax_t_line.plot(pos_ids, trans[:, j], color=XYZ_COLORS[j], linewidth=LINE_WIDTH)
            _add_y_ref_lines(ax_t_line, SUMMARY_YLIM)
            ax_t_line.axvline(pos_ids[medoid_idx], color=MEDOID_COLOR,
                              linewidth=MEDOID_LINE_WIDTH, linestyle="--", zorder=4)
            ax_t_line.set_ylim(-SUMMARY_YLIM, SUMMARY_YLIM)
            ax_t_line.yaxis.set_major_locator(MultipleLocator(1.0))
            ax_t_line.set_ylabel("Δ position (mm)", fontsize=15)
            ax_t_line.set_title(f"{cond_title}\nTranslation vs position ID", fontsize=15)
            if row_idx == 3:
                ax_t_line.set_xlabel("Position ID", fontsize=15)
            _ggpubr_open_ax(ax_t_line)

            # ── Translation Box ───────────────────────────────────────────────
            ax_t_box = fig.add_subplot(gs[row_idx, 1])
            _boxplot_summary(ax_t_box, [trans[:, 0], trans[:, 1], trans[:, 2]],
                             ["ΔX", "ΔY", "ΔZ"], XYZ_COLORS,
                             "Δ position (mm)", "Translation summary", SUMMARY_YLIM)
            ax_t_box.set_ylim(-SUMMARY_YLIM, SUMMARY_YLIM)
            ax_t_box.yaxis.set_major_locator(MultipleLocator(1.0))
            ax_t_box.set_title("Translation summary", fontsize=15)
            ax_t_box.set_ylabel("")
            _annotate_sd(ax_t_box, [trans[:, 0], trans[:, 1], trans[:, 2]], ["ΔX", "ΔY", "ΔZ"])

            # ── Rotation Line ─────────────────────────────────────────────────
            ax_r_line = fig.add_subplot(gs[row_idx, 2])
            for j, lab in enumerate(XYZ_LABELS):
                ax_r_line.plot(pos_ids, rots[:, j], color=XYZ_COLORS[j], linewidth=LINE_WIDTH)
            _add_y_ref_lines(ax_r_line, SUMMARY_YLIM)
            ax_r_line.axvline(pos_ids[medoid_idx], color=MEDOID_COLOR,
                              linewidth=MEDOID_LINE_WIDTH, linestyle="--", zorder=4)
            ax_r_line.set_ylim(-SUMMARY_YLIM, SUMMARY_YLIM)
            ax_r_line.yaxis.set_major_locator(MultipleLocator(1.0))
            ax_r_line.set_ylabel("Δ orientation (degree)", fontsize=15)
            ax_r_line.set_title(f"{cond_title}\nRotation vs position ID", fontsize=15)
            if row_idx == 3:
                ax_r_line.set_xlabel("Position ID", fontsize=15)
            _ggpubr_open_ax(ax_r_line)

            # ── Rotation Box ──────────────────────────────────────────────────
            ax_r_box = fig.add_subplot(gs[row_idx, 3])
            _boxplot_summary(ax_r_box, [rots[:, 0], rots[:, 1], rots[:, 2]],
                             ["ΔX", "ΔY", "ΔZ"], XYZ_COLORS,
                             "Δ orientation (degree)", "Rotation summary", SUMMARY_YLIM)
            ax_r_box.set_ylim(-SUMMARY_YLIM, SUMMARY_YLIM)
            ax_r_box.yaxis.set_major_locator(MultipleLocator(1.0))
            ax_r_box.set_title("Rotation summary", fontsize=15)
            ax_r_box.set_ylabel("")
            _annotate_sd(ax_r_box, [rots[:, 0], rots[:, 1], rots[:, 2]], ["ΔX", "ΔY", "ΔZ"])

        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

# ── Main Routine ──────────────────────────────────────────────────────────────

def main():
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = repo_root / "data" / "gum" / "actual" / "citrus-offline_participant_ratings - ratings.csv"
    actual_dir = repo_root / "data" / "gum" / "actual"
    out_dir = repo_root / "derivatives" / "medoid_opt_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path).rename(columns={pd.read_csv(csv_path).columns[0]: "subject"})
    subjects = df["subject"].dropna().unique()

    for sub in subjects:
        log.info(f"Processing {sub}...")
        sub_dir = out_dir / sub
        sub_dir.mkdir(parents=True, exist_ok=True)
        
        sub_data = {}
        max_variation = 0.5
        
        for _, row in df[df["subject"] == sub].iterrows():
            label = f"{sub}_{row['condition'].lower()}_{row['hemisphere'].lower()}"
            xml_path = actual_dir / sub / (row["localite_file"] if str(row["localite_file"]).endswith(".xml") else f"{row['localite_file']}.xml")
            if not xml_path.exists(): continue

            tree = etree.parse(str(xml_path), etree.XMLParser(recover=False))
            elements = select_elements_by_frame_range(discover_position_elements(tree.getroot(), POSITION_XPATH), int(row["xml_start"]), int(row["xml_end"]))
            
            translations, rotations = [], []
            for el in elements:
                matrix_text = find_matrix_text_in_element(el)
                if matrix_text:
                    m = matrix_from_text(matrix_text, MATRIX_ORDER)
                    t, r = extract_translation_and_rotation(m)
                    translations.append(t); rotations.append(r)
            
            if not translations:
                log.warning(f"No valid trajectory data found for {label}. Skipping.")
                continue

            translations = np.array(translations)
            rel_trans = translations - translations[0]
            rel_rots = relative_rotations_degrees(np.array(rotations))
            med_idx, _ = compute_medoid_index(translations)
            
            max_variation = max(max_variation, np.abs(rel_trans).max())
            key = f"{row['condition'].lower()[:3]}_{row['hemisphere'].lower()[0]}"
            sub_data[key] = (np.arange(1, len(elements)+1), rel_trans, rel_rots, med_idx)

            # Single condition plots saved to subject folder
            limit = 0.5 if np.abs(rel_trans).max() < 0.4 else (1.0 if np.abs(rel_trans).max() < 0.9 else 2.0)
            plot_translation_rotation_lines(rel_trans, rel_rots, np.arange(1, len(elements)+1), med_idx, sub_dir / f"{label}_drift.png", limit)
            plot_spatial_1x4(rel_trans, med_idx, sub_dir / f"{label}_spatial.png", limit)

        # Adaptive subject limit and comprehensive summary plot
        sub_limit = 0.5 if max_variation < 0.4 else (1.0 if max_variation < 0.9 else 2.0)
        plot_subject_comprehensive_summary(sub, sub_data, sub_limit, sub_dir / f"{sub}_summary.png")

if __name__ == "__main__":
    main()
