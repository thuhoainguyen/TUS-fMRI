# -*- coding: utf-8 -*-
"""
find_medoid_opt.py
==================
Optimized script to find the medoid frame for TUS transducer actual trajectory data
across all participants, preserving the exact functions, plotting logic, colors, 
and styles from extract_affine_medoid.ipynb.

@author Hoai Thu Nguyen
"""

import os
import re
import csv
import math
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Run headlessly //$NON-NLS-1$
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from lxml import etree

try:
    import seaborn as sns
    sns.set_theme(style="white", context="notebook", font="Arial")  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$
except ImportError:
    pass

plt.rcParams.update({
    "font.family": "sans-serif",  #//$NON-NLS-1$
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$
    "figure.facecolor": "white",  #//$NON-NLS-1$
    "axes.facecolor": "white",  #//$NON-NLS-1$
    "axes.edgecolor": "black",  #//$NON-NLS-1$
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "legend.frameon": False,
})

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",  #//$NON-NLS-1$
    datefmt="%H:%M:%S",  #//$NON-NLS-1$
)
log = logging.getLogger("find_medoid_opt")  #//$NON-NLS-1$

FLOAT_RE = re.compile(
    r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"  #//$NON-NLS-1$
)

# Global XML Settings from Notebook
POSITION_XPATH = "//Element[@type='InstrumentMarker']"  #//$NON-NLS-1$
MATRIX_ORDER = "row-major"  # or "column-major"  #//$NON-NLS-1$


def parse_16_floats(text):
    values = [float(x) for x in FLOAT_RE.findall(text or "")]  #//$NON-NLS-1$
    if len(values) == 16:
        return values
    return None


def matrix4d_attrib_to_text(element):
    """Parse Localite Matrix4D data00..data33 attributes into 16 floats."""
    for child in element.iterdescendants():
        tag = etree.QName(child).localname
        if tag != "Matrix4D":  #//$NON-NLS-1$
            continue
        comp = {k: float(v) for k, v in child.attrib.items() if k.startswith("data")}  #//$NON-NLS-1$
        if len(comp) < 16:
            continue
        values = [comp.get(f"data{r}{c}", 0.0) for r in range(4) for c in range(4)]  #//$NON-NLS-1$
        return " ".join(str(v) for v in values)  #//$NON-NLS-1$
    return None


def find_matrix_text_in_element(element):
    likely_attr_names = [
        "matrix", "affine", "affinetransform", "transform",  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$
        "transformation", "stageposition",  #//$NON-NLS-1$  #//$NON-NLS-1$
    ]

    for attr_name, attr_value in element.attrib.items():
        compact = attr_name.lower().replace("_", "").replace("-", "")  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$
        if any(name in compact for name in likely_attr_names):
            values = parse_16_floats(attr_value)
            if values is not None:
                return attr_value

    if element.text:
        values = parse_16_floats(element.text)
        if values is not None:
            return element.text

    for child in element.iterdescendants():
        if child.text:
            values = parse_16_floats(child.text)
            if values is not None:
                return child.text

        for attr_name, attr_value in child.attrib.items():
            compact = attr_name.lower().replace("_", "").replace("-", "")  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$
            if any(name in compact for name in likely_attr_names):
                values = parse_16_floats(attr_value)
                if values is not None:
                    return attr_value

    return matrix4d_attrib_to_text(element)


def matrix_from_text(text, matrix_order):
    values = parse_16_floats(text)
    if values is None:
        raise ValueError("Expected exactly 16 numeric values for affine matrix.")  #//$NON-NLS-1$
    arr = np.array(values, dtype=float)
    if matrix_order == "row-major":  #//$NON-NLS-1$
        return arr.reshape((4, 4))
    if matrix_order == "column-major":  #//$NON-NLS-1$
        return arr.reshape((4, 4), order="F")  #//$NON-NLS-1$
    raise ValueError("matrix_order must be 'row-major' or 'column-major'.")  #//$NON-NLS-1$


def extract_translation_and_rotation(matrix):
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    return translation, rotation


def discover_position_elements(root, position_xpath=None):
    if position_xpath:
        candidates = root.xpath(position_xpath)
    else:
        candidates = list(root.iter())
    found = []
    for element in candidates:
        if find_matrix_text_in_element(element) is not None:
            found.append(element)
    return found


def element_frame_index(element):
    idx = element.get("index")  #//$NON-NLS-1$
    return int(idx) if idx is not None else None


def select_elements_by_frame_range(position_elements, start_frame, end_frame):
    """Keep elements whose <Element index="..."> is in [start_frame, end_frame]."""
    selected = [
        element for element in position_elements
        if (frame := element_frame_index(element)) is not None
        and start_frame <= frame <= end_frame
    ]
    if not selected:
        raise ValueError(
            f"No elements found with <Element index> in [{start_frame}, {end_frame}]."  #//$NON-NLS-1$
        )
    selected.sort(key=element_frame_index)
    return selected


def compute_medoid_index(points):
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        raise ValueError("No points provided.")  #//$NON-NLS-1$
    diff = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    total_distances = distances.sum(axis=1)
    return int(np.argmin(total_distances)), total_distances


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
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0
    return np.degrees([x, y, z])


def relative_rotations_degrees(rotations):
    if len(rotations) == 0:
        return np.empty((0, 3))
    R0 = project_to_rotation_matrix(rotations[0])
    R0_inv = R0.T
    rel_eulers = []
    for R in rotations:
        R_clean = project_to_rotation_matrix(R)
        R_rel = R0_inv @ R_clean
        rel_eulers.append(rotation_matrix_to_euler_xyz_degrees(R_rel))
    return np.array(rel_eulers, dtype=float)


def write_summary_csv(output_file, relative_translations, relative_rotations,
                      medoid_local_index, medoid_frame):
    columns = {
        "translation_x": relative_translations[:, 0],  #//$NON-NLS-1$
        "translation_y": relative_translations[:, 1],  #//$NON-NLS-1$
        "translation_z": relative_translations[:, 2],  #//$NON-NLS-1$
        "rotation_x_deg": relative_rotations[:, 0],  #//$NON-NLS-1$
        "rotation_y_deg": relative_rotations[:, 1],  #//$NON-NLS-1$
        "rotation_z_deg": relative_rotations[:, 2],  #//$NON-NLS-1$
    }
    with open(output_file, "w", newline="", encoding="utf-8") as f:  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$
        writer = csv.writer(f)
        writer.writerow([
            "metric", "count", "mean", "std", "min", "median", "max",  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$
            "medoid_value", "medoid_local_index", "medoid_frame",  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$
        ])
        for metric, values in columns.items():
            values = np.asarray(values, dtype=float)
            writer.writerow([
                metric, len(values), float(np.mean(values)),
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                float(np.min(values)), float(np.median(values)), float(np.max(values)),
                float(values[medoid_local_index]), medoid_local_index, medoid_frame,
            ])


def write_medoid_xml(original_tree, medoid_element, label, output_xml_file):
    """Write a medoid-only XML with a single <Element index="0">."""
    medoid_element.set("index", "0")  #//$NON-NLS-1$  #//$NON-NLS-1$
    medoid_element.set("medoid", "true")  #//$NON-NLS-1$  #//$NON-NLS-1$
    medoid_element.set("label", label)  #//$NON-NLS-1$
    medoid_element.set("description", label)  #//$NON-NLS-1$

    for child in medoid_element.iterdescendants():
        tag = etree.QName(child).localname
        if tag == "InstrumentMarker":  #//$NON-NLS-1$
            child.set("description", label)  #//$NON-NLS-1$

    root = original_tree.getroot()
    for child in list(root):
        if not isinstance(child.tag, str):
            continue
        if etree.QName(child).localname == "Element" and child is not medoid_element:  #//$NON-NLS-1$
            root.remove(child)

    original_tree.write(
        str(output_xml_file), encoding="utf-8",  #//$NON-NLS-1$
        xml_declaration=True, pretty_print=True,
    )


# ==============================================================================
# Styling variables from Notebook cell 162
# ==============================================================================
AXIS_LIMIT_MM = 1.0
BIN_WIDTH_MM = 0.1
POINT_SIZE = 24
POINT_ALPHA = 0.65
MEDOID_SIZE = int(POINT_SIZE * 1.25)
LINE_WIDTH = 1.5
MEDOID_LINE_WIDTH = LINE_WIDTH * 1.25
REF_LINE_WIDTH = 0.5
REF_LEVELS = (-0.5, 0.0, 0.5)
REF_COLOR_MINOR = "#cccccc"  #//$NON-NLS-1$
HIST_SIZE = "24%"  #//$NON-NLS-1$
HIST_PAD = 0.38
HIST_LABEL_SIZE = 9
HIST_TICK_SIZE = 9
GENERAL_COLOR = "#c4c4c4"  #//$NON-NLS-1$
MEDOID_COLOR = "black"  #//$NON-NLS-1$
XYZ_COLORS = ["#66c2a5", "#fc8d62", "#8da0cb"]  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$
XYZ_LABELS = ["X", "Y", "Z"]  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$


from contextlib import contextmanager
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import seaborn as sns
except ImportError:
    sns = None


@contextmanager
def use_ggpubr_theme():
    """ggpubr theme_pubr-like styling for matplotlib."""
    if sns is not None:
        sns.set_theme(style="white", context="notebook", font="Arial")  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$
    with plt.rc_context({
        "font.family": "sans-serif",  #//$NON-NLS-1$
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$
        "figure.facecolor": "white",  #//$NON-NLS-1$
        "axes.facecolor": "white",  #//$NON-NLS-1$
        "axes.edgecolor": "black",  #//$NON-NLS-1$
        "axes.linewidth": 0.8,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.grid": False,
    }):
        yield


def _ggpubr_open_ax(ax):
    ax.spines["top"].set_visible(False)  #//$NON-NLS-1$
    ax.spines["right"].set_visible(False)  #//$NON-NLS-1$
    ax.grid(False)


def _boxed_ax(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.grid(False)


def _set_square_limits(ax, limit_mm=AXIS_LIMIT_MM):
    ax.set_xlim(-limit_mm, limit_mm)
    ax.set_ylim(-limit_mm, limit_mm)
    ax.set_aspect("equal", adjustable="box")  #//$NON-NLS-1$


def _ref_line_kwargs(color):
    return dict(color=color, linewidth=REF_LINE_WIDTH, linestyle=":", alpha=0.75, zorder=0)  #//$NON-NLS-1$


def _add_y_ref_lines(ax, zero_color=REF_COLOR_MINOR):
    for level in REF_LEVELS:
        color = zero_color if level == 0.0 else REF_COLOR_MINOR
        ax.axhline(level, **_ref_line_kwargs(color))


def _add_xy_ref_lines(ax, x_zero_color, y_zero_color):
    for level in REF_LEVELS:
        ax.axhline(level, **_ref_line_kwargs(y_zero_color if level == 0.0 else REF_COLOR_MINOR))
        ax.axvline(level, **_ref_line_kwargs(x_zero_color if level == 0.0 else REF_COLOR_MINOR))


def mm_bin_edges(bin_width_mm, limit_mm=AXIS_LIMIT_MM):
    return np.arange(-limit_mm, limit_mm + bin_width_mm / 2, bin_width_mm)


def apply_mm_limits_3d(ax, limit_mm=AXIS_LIMIT_MM):
    ax.set_xlim(-limit_mm, limit_mm)
    ax.set_ylim(-limit_mm, limit_mm)
    ax.set_zlim(-limit_mm, limit_mm)
    ax.margins(0)


def _filled_scatter(ax, x, y, *, c, s, alpha=1.0, zorder=2):
    ax.scatter(x, y, c=c, s=s, alpha=alpha, linewidths=0, edgecolors="none", zorder=zorder)  #//$NON-NLS-1$


def _style_projection_ax(ax, xlabel, ylabel, title, d0, d1):
    _set_square_limits(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    _boxed_ax(ax)
    _add_xy_ref_lines(ax, XYZ_COLORS[d0], XYZ_COLORS[d1])


def _style_hist_axis_only(ax, orientation="horizontal"):  #//$NON-NLS-1$
    """Marginal histogram: single baseline axis with count ticks."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    if orientation == "horizontal":  #//$NON-NLS-1$
        ax.spines["left"].set_visible(True)  #//$NON-NLS-1$
        ax.set_xlabel("Count", fontsize=HIST_LABEL_SIZE, labelpad=3)  #//$NON-NLS-1$
        ax.tick_params(axis="x", labelsize=HIST_TICK_SIZE, length=3, pad=2)  #//$NON-NLS-1$
        ax.tick_params(axis="y", labelleft=False, length=2, pad=1)  #//$NON-NLS-1$
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    else:
        ax.spines["bottom"].set_visible(True)  #//$NON-NLS-1$
        ax.set_ylabel("Count", fontsize=HIST_LABEL_SIZE, labelpad=3)  #//$NON-NLS-1$
        ax.tick_params(axis="x", labelbottom=False, length=2, pad=1)  #//$NON-NLS-1$
        ax.tick_params(axis="y", labelsize=HIST_TICK_SIZE, length=3, pad=2)  #//$NON-NLS-1$
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))


def _add_marginal_histograms(ax, x, y, color_x, color_y):
    bins = mm_bin_edges(BIN_WIDTH_MM)
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", size=HIST_SIZE, pad=HIST_PAD, sharex=ax)  #//$NON-NLS-1$
    ax_histy = divider.append_axes("right", size=HIST_SIZE, pad=HIST_PAD, sharey=ax)  #//$NON-NLS-1$
    ax_histx.hist(x, bins=bins, color=color_x, alpha=0.8, edgecolor="none")  #//$NON-NLS-1$
    ax_histy.hist(y, bins=bins, orientation="horizontal", color=color_y, alpha=0.8, edgecolor="none")  #//$NON-NLS-1$  #//$NON-NLS-1$
    ax_histx.set_xlim(-AXIS_LIMIT_MM, AXIS_LIMIT_MM)
    ax_histy.set_ylim(-AXIS_LIMIT_MM, AXIS_LIMIT_MM)
    _style_hist_axis_only(ax_histx, orientation="vertical")  #//$NON-NLS-1$
    _style_hist_axis_only(ax_histy, orientation="horizontal")  #//$NON-NLS-1$
    ax.set_aspect("equal", adjustable="box")  #//$NON-NLS-1$


def _scatter_points_2d(ax, x, y, medoid_idx):
    mask = np.ones(len(x), dtype=bool)
    mask[medoid_idx] = False
    _filled_scatter(ax, x[mask], y[mask], c=GENERAL_COLOR, s=POINT_SIZE, alpha=POINT_ALPHA)
    _filled_scatter(ax, [x[medoid_idx]], [y[medoid_idx]], c=MEDOID_COLOR, s=MEDOID_SIZE, alpha=1.0, zorder=5)


def _boxplot_summary(ax, data_by_axis, axis_labels, colors, ylabel, title):
    bp = ax.boxplot(data_by_axis, tick_labels=axis_labels, showfliers=False, patch_artist=True, widths=0.55)  #//$NON-NLS-1$
    for patch, color in zip(bp["boxes"], colors):  #//$NON-NLS-1$
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
    for med, color in zip(bp["medians"], colors):  #//$NON-NLS-1$
        med.set_color(color)
        med.set_linewidth(1.5)
    means = [float(np.mean(d)) for d in data_by_axis]
    for i, (m, color) in enumerate(zip(means, colors)):
        ax.scatter(i + 1, m, marker="D", s=30, c=color, linewidths=0, edgecolors="none", zorder=5)  #//$NON-NLS-1$  #//$NON-NLS-1$
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _add_y_ref_lines(ax)
    _ggpubr_open_ax(ax)


def plot_translation_rotation_lines(trans_values, rot_values, position_ids, medoid_idx, save_path=None):
    with use_ggpubr_theme():
        fig = plt.figure(figsize=(14, 7), facecolor="white")  #//$NON-NLS-1$
        fig.suptitle("Transducer variation relative to starting position", fontsize=13, y=0.98)  #//$NON-NLS-1$
        gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[3, 1], hspace=0.35, wspace=0.25)
        panels = [
            (gs[0, 0], gs[0, 1], np.asarray(trans_values),
             "Δ position (mm)", "Translation vs position ID", "Translation summary", False),  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$
            (gs[1, 0], gs[1, 1], np.asarray(rot_values),
             "Δ orientation (degree)", "Rotation vs position ID", "Rotation summary", True),  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$
        ]
        for ax_line_spec, ax_box_spec, values, ylabel, line_title, box_title, show_xlabel in panels:
            ax_line = fig.add_subplot(ax_line_spec)
            ax_box = fig.add_subplot(ax_box_spec)
            for j, lab in enumerate(XYZ_LABELS):
                ax_line.plot(position_ids, values[:, j], label=lab,
                             color=XYZ_COLORS[j], linewidth=LINE_WIDTH)
            _add_y_ref_lines(ax_line)
            ax_line.axvline(position_ids[medoid_idx], color=MEDOID_COLOR,
                            linewidth=MEDOID_LINE_WIDTH, linestyle="--", alpha=1.0,  #//$NON-NLS-1$
                            label="Medoid", zorder=4)  #//$NON-NLS-1$
            ax_line.set_ylim(-AXIS_LIMIT_MM, AXIS_LIMIT_MM)
            ax_line.set_ylabel(ylabel)
            ax_line.set_title(line_title)
            _ggpubr_open_ax(ax_line)
            ax_line.legend(loc="best")  #//$NON-NLS-1$
            _boxplot_summary(ax_box, [values[:, 0], values[:, 1], values[:, 2]],
                             ["ΔX", "ΔY", "ΔZ"], XYZ_COLORS, ylabel, box_title)  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$
            ax_box.set_ylim(-AXIS_LIMIT_MM, AXIS_LIMIT_MM)
            if show_xlabel:
                ax_line.set_xlabel("Position ID")  #//$NON-NLS-1$
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")  #//$NON-NLS-1$
            plt.close(fig)
        else:
            plt.show()


def plot_spatial_1x4(coords, medoid_idx, save_path=None):
    coords = np.asarray(coords, dtype=float)
    mask = np.ones(len(coords), dtype=bool)
    mask[medoid_idx] = False

    with use_ggpubr_theme():
        fig = plt.figure(figsize=(14, 9), facecolor="white")  #//$NON-NLS-1$
        fig.suptitle("Transducer variation relative to starting position", fontsize=13, y=0.98)  #//$NON-NLS-1$
        gs = gridspec.GridSpec(
            2, 3, figure=fig, height_ratios=[0.88, 1.12],
            hspace=0.35, wspace=0.25,
        )

        ax3d = fig.add_subplot(gs[0, :], projection="3d")  #//$NON-NLS-1$
        ax3d.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2],
                     c=GENERAL_COLOR, s=POINT_SIZE, alpha=POINT_ALPHA, depthshade=False,
                     linewidths=0, edgecolors="none")  #//$NON-NLS-1$
        ax3d.scatter(coords[medoid_idx, 0], coords[medoid_idx, 1], coords[medoid_idx, 2],
                     c=MEDOID_COLOR, s=MEDOID_SIZE, alpha=1.0, depthshade=False,
                     linewidths=0, edgecolors="none", zorder=10)  #//$NON-NLS-1$
        ax3d.set_xlabel("ΔX (mm)", color=XYZ_COLORS[0], labelpad=4)  #//$NON-NLS-1$
        ax3d.set_ylabel("ΔY (mm)", color=XYZ_COLORS[1], labelpad=4)  #//$NON-NLS-1$
        ax3d.set_zlabel("ΔZ (mm)", color=XYZ_COLORS[2], labelpad=4)  #//$NON-NLS-1$
        ax3d.set_title("3D translation scatter", pad=6)  #//$NON-NLS-1$
        apply_mm_limits_3d(ax3d)
        try:
            ax3d.set_box_aspect((1, 1, 1), zoom=1.2)  #//$NON-NLS-1$
        except TypeError:
            ax3d.set_box_aspect((1, 1, 1))
            ax3d.dist = 7
        ax3d.view_init(elev=22, azim=-58)
        for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#dddddd")  #//$NON-NLS-1$
        ax3d.grid(False)

        projections = [
            (gs[1, 0], 0, 1, "ΔX (mm)", "ΔY (mm)", "XY projection"),  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$
            (gs[1, 1], 0, 2, "ΔX (mm)", "ΔZ (mm)", "XZ projection"),  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$
            (gs[1, 2], 1, 2, "ΔY (mm)", "ΔZ (mm)", "YZ projection"),  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$
        ]
        for spec, d0, d1, xl, yl, title in projections:
            ax = fig.add_subplot(spec)
            _scatter_points_2d(ax, coords[:, d0], coords[:, d1], medoid_idx)
            _style_projection_ax(ax, xl, yl, title, d0, d1)
            _add_marginal_histograms(ax, coords[:, d0], coords[:, d1],
                                     XYZ_COLORS[d0], XYZ_COLORS[d1])

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")  #//$NON-NLS-1$
            plt.close(fig)
        else:
            plt.show()


# ==============================================================================
# Main Routine
# ==============================================================================
def main() -> None:
    # Setup working folders relative to CITRUS workspace
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = (
        repo_root
        / "data"
        / "gum"
        / "actual"
        / "citrus-offline_participant_ratings - ratings.csv"
    )  #//$NON-NLS-1$
    actual_dir = repo_root / "data" / "gum" / "actual"  #//$NON-NLS-1$

    out_dir = repo_root / "derivatives" / "medoid_opt"  #//$NON-NLS-1$
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        log.error("Ratings CSV file not found at: %s", csv_path)  #//$NON-NLS-1$
        return

    # Load ratings sheet
    log.info("Loading ratings CSV from %s", csv_path)  #//$NON-NLS-1$
    df_ratings = pd.read_csv(csv_path)
    df_ratings = df_ratings.rename(
        columns={df_ratings.columns[0]: "subject"}
    )  #//$NON-NLS-1$

    subjects = df_ratings["subject"].dropna().unique()  #//$NON-NLS-1$
    log.info("Processing subjects: %s", list(subjects))  #//$NON-NLS-1$

    medoid_records = []

    for sub in subjects:
        log.info("=== Processing %s ===", sub)  #//$NON-NLS-1$
        sub_ratings = df_ratings[df_ratings["subject"] == sub]  #//$NON-NLS-1$

        for _, row in sub_ratings.iterrows():
            cond = row["condition"]  #//$NON-NLS-1$
            hemi = row["hemisphere"]  #//$NON-NLS-1$
            localite_file = row["localite_file"]  #//$NON-NLS-1$
            if not str(localite_file).endswith(".xml"):  #//$NON-NLS-1$
                localite_file = f"{localite_file}.xml"

            actual_xml_path = actual_dir / sub / localite_file
            if not actual_xml_path.exists():
                log.error(
                    "Actual XML file not found: %s", actual_xml_path
                )  #//$NON-NLS-1$
                continue

            xml_start = int(row["xml_start"])  #//$NON-NLS-1$
            xml_end = int(row["xml_end"])  #//$NON-NLS-1$

            label = f"{sub}_{cond.lower()}_{hemi.lower()}"
            log.info(
                "Analyzing %s [frames %d-%d]", label, xml_start, xml_end
            )  #//$NON-NLS-1$

            try:
                # Parse using lxml etree exactly as in notebook
                parser_config = etree.XMLParser(
                    remove_blank_text=False, remove_comments=False,
                    remove_pis=False, recover=False,
                )
                tree = etree.parse(str(actual_xml_path), parser_config)
                root = tree.getroot()

                position_elements = discover_position_elements(root, position_xpath=POSITION_XPATH)
                if not position_elements:
                    log.error("No XML elements containing 4x4 affine matrices were found in %s", actual_xml_path)  #//$NON-NLS-1$
                    continue

                selected_elements = select_elements_by_frame_range(
                    position_elements, xml_start, xml_end
                )
                frame_indices = [element_frame_index(el) for el in selected_elements]

                matrices, translations, rotations = [], [], []
                for element in selected_elements:
                    matrix_text = find_matrix_text_in_element(element)
                    if matrix_text is None:
                        raise ValueError("Selected XML element does not contain a parseable 4x4 matrix.")  #//$NON-NLS-1$
                    matrix = matrix_from_text(matrix_text, MATRIX_ORDER)
                    translation, rotation = extract_translation_and_rotation(matrix)
                    matrices.append(matrix)
                    translations.append(translation)
                    rotations.append(rotation)

                translations = np.array(translations, dtype=float)
                rotations = np.array(rotations, dtype=float)

                # Compute medoid
                medoid_local_index, total_distances = compute_medoid_index(translations)
                medoid_element = selected_elements[medoid_local_index]
                medoid_frame = element_frame_index(medoid_element)

                # Relative displacements from start translation
                start_translation = translations[0]
                relative_translations = translations - start_translation
                relative_rotations = relative_rotations_degrees(rotations)

                # Record results
                medoid_records.append({
                    "subject": sub,  #//$NON-NLS-1$
                    "condition": cond,  #//$NON-NLS-1$
                    "hemisphere": hemi,  #//$NON-NLS-1$
                    "medoid_frame": medoid_frame,  #//$NON-NLS-1$
                })

                # Write XML output (forces index 0)
                output_xml = out_dir / f"{label}_medoid.xml"
                write_medoid_xml(tree, medoid_element, label, output_xml)

                # Write summary CSV
                summary_csv = out_dir / f"{label}_summary_stats.csv"
                write_summary_csv(
                    summary_csv, relative_translations, relative_rotations,
                    medoid_local_index, medoid_frame,
                )

                # Save plots
                pos_ids = np.arange(1, len(frame_indices) + 1)
                rel_trans_start = translations - start_translation
                rel_rot_start = relative_rotations_degrees(rotations)

                drift_plot_path = out_dir / f"{label}_translation_rotation_drift.png"
                plot_translation_rotation_lines(
                    rel_trans_start, rel_rot_start, pos_ids, medoid_local_index, save_path=str(drift_plot_path)
                )

                spatial_plot_path = out_dir / f"{label}_spatial_projections.png"
                plot_spatial_1x4(
                    rel_trans_start, medoid_local_index, save_path=str(spatial_plot_path)
                )

                log.info(
                    "Successfully processed %s (Medoid frame: %d)",  #//$NON-NLS-1$
                    label,
                    medoid_frame,
                )

            except Exception as e:
                log.exception("Failed to analyze condition %s: %s", label, e)  #//$NON-NLS-1$

    # Print summary and save txt
    table_lines = [
        "=" * 55,
        "   IDENTIFIED MEDOID FRAME INDICES (OPTIMIZED)",  #//$NON-NLS-1$
        "=" * 55,
        f"   {'Subject':<10} {'Condition':<12} {'Hemi':<6} {'Medoid Frame':<12}",  #//$NON-NLS-1$
        "   " + "─" * 49,
    ]
    for r in medoid_records:
        table_lines.append(
            f"   {r['subject']:<10} {r['condition']:<12} {r['hemisphere']:<6} {r['medoid_frame']:<12}"
        )
    table_lines.append("=" * 55)

    table_text = "\n".join(table_lines) + "\n"
    print("\n" + table_text)

    txt_path = out_dir / "medoid.txt"  #//$NON-NLS-1$
    with open(txt_path, "w") as f:  #//$NON-NLS-1$
        f.write(table_text)
    log.info("Saved medoid text summary to: %s", txt_path)  #//$NON-NLS-1$


if __name__ == "__main__":
    main()
