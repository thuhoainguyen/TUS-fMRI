#!/usr/bin/env python3
"""
find_medoid.py
==============
Script to find the medoid frame for TUS transducer actual trajectory data
for all subjects, sessions (CON and EXP), and hemispheres (L and R),
and output quality control visualization plots.

@author Hoai Thu Nguyen
"""

import os
import re
import glob
import math
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("find_medoid")

# ── Color Palette ─────────────────────────────────────────────────────────────
BG    = "#1a1a2e"
PAN   = "#0d0d1a"
GOLD  = "#FFD700"
WHITE = "#FFFFFF"
GRAY  = "#aaaaaa"
C_L   = "#4C9BE8"
C_R   = "#E8834C"
CLUSTER_COLORS = ["#4C9BE8", "#55A868", "#C44E52", "#DD8452",
                  "#8172B2", "#937860", "#64B5CD", "#E8C34C"]
NOISE_COLOR    = "#666666"
CMAP_TIME      = "plasma"


# ══════════════════════════════════════════════════════════════════════════════
# 1. XML PARSING
# ══════════════════════════════════════════════════════════════════════════════

def _matrix4d_to_components(mat_el: ET.Element) -> Dict[str, float]:
    """
    Extract numeric values from XML Matrix4D element attributes.
    """
    return {k: float(v) for k, v in mat_el.attrib.items()}


def _rotation_to_euler_xyz(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert 3x3 rotation matrix to Euler angles in degrees.
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        rx = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        ry = np.degrees(np.arctan2(-R[2, 0], sy))
        rz = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        rx = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        ry = np.degrees(np.arctan2(-R[2, 0], sy))
        rz = 0.0
    return rx, ry, rz


def _rotation_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Convert 3x3 rotation matrix to quaternion (w, x, y, z).
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return w, x, y, z


def parse_gummarker_xml(xml_path: str, convert_lps_to_ras: bool = True) -> pd.DataFrame:
    """
    Parse a Localite GUMMarkers XML file into a pandas DataFrame.
    Flips coordinates from LPS to RAS space if coordinateSpace is LPS.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    source = Path(xml_path).name
    records = []

    coord_system = (root.get("coordinateSpace", "RAS") or "RAS").upper()
    flip = np.eye(4)
    if convert_lps_to_ras and coord_system == "LPS":
        flip = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)

    for elem in root.findall("Element"):
        frame = int(elem.get("index", -1))
        im    = elem.find(".//InstrumentMarker")
        if im is None:
            continue

        mat_el = im.find("Matrix4D")
        if mat_el is None:
            continue

        desc = im.get("description", "")
        # Parse time from description if present
        m = re.search(r"([\d.]+)\s*s\b", desc)
        time_s = float(m.group(1)) if m else np.nan

        comp = _matrix4d_to_components(mat_el)
        M = np.array([[comp[f"data{r}{c}"] for c in range(4)]
                      for r in range(4)], dtype=float)

        # Coordinate transformation
        M = flip @ M

        x, y, z = M[0, 3], M[1, 3], M[2, 3]
        R = M[:3, :3]
        rx, ry, rz   = _rotation_to_euler_xyz(R)
        qw, qx, qy, qz = _rotation_to_quaternion(R)

        row: Dict = dict(
            frame=frame,
            description=desc,
            time_s=time_s,
            x=x, y=y, z=z,
            rot_x=rx, rot_y=ry, rot_z=rz,
            quat_w=qw, quat_x=qx, quat_y=qy, quat_z=qz,
            source_file=source,
        )
        for r in range(4):
            for c in range(4):
                row[f"mat_{r}{c}"] = M[r, c]

        records.append(row)

    df = pd.DataFrame(records)
    log.debug("Parsed %d rows from %s", len(df), xml_path)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. DISPLACEMENT & ANGULAR DRIFT
# ══════════════════════════════════════════════════════════════════════════════

def compute_displacement(df: pd.DataFrame, ref_row: pd.Series) -> pd.DataFrame:
    """
    Calculate translational and rotational deviation relative to a planned reference.
    """
    df = df.copy()

    ref_xyz = np.array([ref_row["x"], ref_row["y"], ref_row["z"]])
    R_ref   = np.array([[ref_row[f"mat_{r}{c}"] for c in range(3)]
                        for r in range(3)])

    xyz = df[["x", "y", "z"]].to_numpy()
    df["displacement_mm"] = np.linalg.norm(xyz - ref_xyz, axis=1)

    angles = []
    for _, row in df.iterrows():
        R_act   = np.array([[row[f"mat_{r}{c}"] for c in range(3)]
                            for r in range(3)])
        R_delta = R_ref.T @ R_act
        cos_val = np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)
        angles.append(float(np.degrees(np.arccos(cos_val))))
    df["angle_deg"] = angles

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. CLUSTERING & MEDOID SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def auto_eps(coords: np.ndarray, k: int = 4) -> float:
    """
    Select DBSCAN eps automatically using the k-NN elbow method.
    """
    if len(coords) <= k:
        return float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)) * 0.10)

    nbrs  = NearestNeighbors(n_neighbors=k).fit(coords)
    d, _  = nbrs.kneighbors(coords)
    knn   = np.sort(d[:, k - 1])
    x     = np.linspace(0.0, 1.0, len(knn))
    y     = (knn - knn.min()) / (knn.max() - knn.min() + 1e-12)
    d2    = np.gradient(np.gradient(y, x), x)
    mg    = max(3, len(knn) // 10)
    if len(knn) - 2 * mg <= 0:
        elbow = int(np.argmax(d2))
    else:
        elbow = int(np.argmax(d2[mg:-mg]) + mg)
    return float(knn[elbow])


def cluster_positions(coords: np.ndarray, eps: Union[float, str], min_samples: int) -> Tuple[np.ndarray, float]:
    """
    Compute clusters using DBSCAN.
    """
    if isinstance(eps, str) and eps.lower() == "auto":
        eps_used = auto_eps(coords)
    else:
        eps_used = float(eps)

    labels = DBSCAN(eps=eps_used, min_samples=min_samples,
                    metric="euclidean").fit_predict(coords)
    return labels, eps_used


def find_medoid_index(coords: np.ndarray, mask: np.ndarray) -> int:
    """
    Find the medoid of the selected cluster mask.
    """
    pts   = coords[mask]
    dists = cdist(pts, pts, metric="euclidean")
    local = int(np.argmin(dists.sum(axis=1)))
    return int(np.where(mask)[0][local])


def select_representative(df: pd.DataFrame, labels: np.ndarray, coords: np.ndarray) -> Tuple[int, int, bool]:
    """
    Select the largest cluster and identify its medoid index.
    """
    cluster_ids = [l for l in set(labels) if l != -1]

    if not cluster_ids:
        log.warning("No valid clusters found – falling back to min-displacement frame.")
        fallback = int(df["displacement_mm"].idxmin())
        return fallback, -1, True

    best_id  = max(cluster_ids, key=lambda l: np.sum(labels == l))
    mask     = labels == best_id
    med_idx  = find_medoid_index(coords, mask)
    return med_idx, best_id, False


def analyse_condition(
    actual_df: pd.DataFrame,
    ref_row:   pd.Series,
    indices:   List[int],
    eps:       Union[float, str],
    min_samples: int,
    label:     str,
) -> Dict:
    """
    Perform DBSCAN and medoid selection for one session & hemisphere tracking sequence.
    """
    sub = actual_df[actual_df["frame"].isin(indices)].copy().reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"[{label}] No rows found for indices {indices[:5]}…")

    sub = compute_displacement(sub, ref_row)
    coords = sub[["x", "y", "z"]].to_numpy()
    labels, eps_used = cluster_positions(coords, eps, min_samples)
    sub["cluster"] = labels

    med_idx, best_id, used_fallback = select_representative(sub, labels, coords)
    medoid_row = sub.iloc[med_idx]

    mask_best = labels == best_id if not used_fallback else np.ones(len(sub), bool)
    cluster_ids  = sorted(set(labels))
    n_clusters   = len([l for l in cluster_ids if l != -1])
    n_noise      = int(np.sum(labels == -1))
    cluster_sizes = {l: int(np.sum(labels == l)) for l in cluster_ids if l != -1}

    stats = dict(
        n_total        = len(sub),
        n_clusters     = n_clusters,
        n_noise        = n_noise,
        cluster_sizes  = cluster_sizes,
        best_cluster   = best_id,
        best_cluster_n = int(np.sum(mask_best)),
        eps_used       = eps_used,
        medoid_frame   = int(medoid_row["frame"]),
        medoid_x       = float(medoid_row["x"]),
        medoid_y       = float(medoid_row["y"]),
        medoid_z       = float(medoid_row["z"]),
        medoid_disp_mm = float(medoid_row["displacement_mm"]),
        medoid_ang_deg = float(medoid_row["angle_deg"]),
        disp_mean_all  = float(sub["displacement_mm"].mean()),
        disp_sd_all    = float(sub["displacement_mm"].std()) if len(sub) > 1 else 0.0,
        disp_max_all   = float(sub["displacement_mm"].max()),
        disp_mean_cls  = float(sub.loc[mask_best, "displacement_mm"].mean()),
        disp_sd_cls    = float(sub.loc[mask_best, "displacement_mm"].std()) if np.sum(mask_best) > 1 else 0.0,
        ang_mean_all   = float(sub["angle_deg"].mean()),
        ang_sd_all     = float(sub["angle_deg"].std()) if len(sub) > 1 else 0.0,
        ang_max_all    = float(sub["angle_deg"].max()),
        ang_mean_cls   = float(sub.loc[mask_best, "angle_deg"].mean()),
        ang_sd_cls     = float(sub.loc[mask_best, "angle_deg"].std()) if np.sum(mask_best) > 1 else 0.0,
        used_fallback  = used_fallback,
    )

    return dict(
        df=sub, labels=labels, eps_used=eps_used,
        coords=coords, best_id=best_id,
        medoid_idx=med_idx, used_fallback=used_fallback,
        medoid_row=medoid_row, stats=stats,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. PLOT GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PAN)
    ax.set_title(title, color=WHITE, fontsize=9, pad=4)
    ax.set_xlabel(xlabel, color=GRAY, fontsize=8)
    ax.set_ylabel(ylabel, color=GRAY, fontsize=8)
    ax.tick_params(colors=GRAY, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.grid(color="#2a2a3a", linewidth=0.5, linestyle="--")


def fig_positions_over_time(result: Dict, label: str, save_path: str) -> None:
    """
    Save trajectories vs frame time.
    """
    df        = result["df"]
    med_idx   = result["medoid_idx"]
    mask_best = df["cluster"] == result["best_id"]

    has_time = df["time_s"].notna().any()
    t_col    = "time_s" if has_time else "frame"
    t_label  = "Time (s)" if has_time else "Frame index"
    t        = df[t_col].to_numpy()

    norm  = Normalize(vmin=t.min(), vmax=t.max())
    cmap  = plt.get_cmap(CMAP_TIME)

    fig = plt.figure(figsize=(12, 9), facecolor=BG)
    fig.suptitle(f"Transducer Positions Over Time — {label.upper()}\n"
                 f"(n={len(df)} | DBSCAN eps={result['eps_used']:.3f} mm)",
                 color=WHITE, fontsize=12, fontweight="bold", y=0.98)

    gs   = gridspec.GridSpec(5, 2, figure=fig, hspace=0.6, wspace=0.3,
                             left=0.08, right=0.95, top=0.90, bottom=0.08)
    ax3d = fig.add_subplot(gs[:3, 1], projection="3d")

    row_specs = [
        ("x", "X (mm)", "#88ccff"),
        ("y", "Y (mm)", "#88ffcc"),
        ("z", "Z (mm)", "#ffcc88"),
        ("displacement_mm", "Displacement (mm)", "#ff88cc"),
        ("angle_deg",       "Angle (°)",         "#ccaaff"),
    ]

    for row, (col, ylbl, acol) in enumerate(row_specs):
        ax = fig.add_subplot(gs[row, 0])
        vals = df[col].to_numpy()
        colors_pt = cmap(norm(t))

        ax.plot(t, vals, color=acol, linewidth=0.5, alpha=0.3, zorder=1)
        ax.scatter(t, vals, c=colors_pt, s=12, alpha=0.75, zorder=3, linewidths=0)

        # non-best-cluster red x
        nb = ~mask_best
        if nb.any():
            ax.scatter(t[nb.to_numpy()], vals[nb.to_numpy()],
                       color="#ff4444", s=25, marker="x",
                       linewidths=1.0, alpha=0.65, zorder=5,
                       label="Drift / noise" if row == 0 else "")

        # medoid star
        ax.scatter(t[med_idx], vals[med_idx], color=GOLD, s=200,
                   marker="*", zorder=10, edgecolors="black",
                   linewidths=0.5, label="Medoid" if row == 0 else "")

        _style_ax(ax, title="", xlabel=t_label if row == 4 else "", ylabel=ylbl)
        if row == 0:
            ax.legend(fontsize=7, facecolor="#111", labelcolor=WHITE,
                      loc="upper right", framealpha=0.8)

    # 3D trajectory
    ax3d.set_facecolor(PAN)
    xyz = df[["x", "y", "z"]].to_numpy()
    ax3d.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                 c=cmap(norm(t)), s=16, alpha=0.80, depthshade=True)
    ax3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2],
              color=WHITE, linewidth=0.4, alpha=0.2)
    if (~mask_best).any():
        nb_xyz = xyz[~mask_best.to_numpy()]
        ax3d.scatter(nb_xyz[:, 0], nb_xyz[:, 1], nb_xyz[:, 2],
                     color="#ff4444", s=30, marker="x", linewidths=1.2, alpha=0.6)
    ax3d.scatter([xyz[med_idx, 0]], [xyz[med_idx, 1]], [xyz[med_idx, 2]],
                 color=GOLD, s=250, marker="*", zorder=15,
                 edgecolors="black", linewidths=0.5)
    ax3d.set_xlabel("X", color=GRAY, fontsize=7, labelpad=1)
    ax3d.set_ylabel("Y", color=GRAY, fontsize=7, labelpad=1)
    ax3d.set_zlabel("Z", color=GRAY, fontsize=7, labelpad=1)
    ax3d.tick_params(colors="#555", labelsize=6)
    ax3d.set_title("3D trajectory (color = time)", color=WHITE, fontsize=9)

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def fig_spatial_clusters(result: Dict, label: str, ref_row: pd.Series, save_path: str) -> None:
    """
    Save 3D spatial cluster mapping plots.
    """
    df      = result["df"]
    labels  = result["labels"]
    coords  = result["coords"]
    med_idx = result["medoid_idx"]
    best_id = result["best_id"]

    unique_labels = sorted(set(labels))

    def lcolor(l):
        if l == -1:
            return NOISE_COLOR
        ordered = [best_id] + [x for x in unique_labels if x != -1 and x != best_id]
        return CLUSTER_COLORS[ordered.index(l) % len(CLUSTER_COLORS)]

    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    fig.suptitle(
        f"Spatial Clusters — {label.upper()}\n"
        f"DBSCAN eps={result['eps_used']:.3f} mm | selected={result['stats']['best_cluster_n']} pts",
        color=WHITE, fontsize=12, fontweight="bold", y=0.98)

    gs   = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.3,
                             left=0.06, right=0.94, top=0.88, bottom=0.08)
    ax3d = fig.add_subplot(gs[0, :2], projection="3d")
    ax_bar = fig.add_subplot(gs[0, 2])
    ax_xy  = fig.add_subplot(gs[1, 0])
    ax_xz  = fig.add_subplot(gs[1, 1])
    ax_yz  = fig.add_subplot(gs[1, 2])

    ax3d.set_facecolor(PAN)
    for l in unique_labels:
        m   = labels == l
        col = lcolor(l)
        nm  = (f"Cluster {l} (Selected)" if l == best_id
               else "Noise" if l == -1
               else f"Cluster {l} (Drift)")
        ax3d.scatter(coords[m, 0], coords[m, 1], coords[m, 2],
                     color=col, s=20 if l != -1 else 10,
                     alpha=0.80 if l == best_id else 0.40,
                     depthshade=True, label=nm)

    # Reference diamond
    ax3d.scatter([ref_row["x"]], [ref_row["y"]], [ref_row["z"]],
                 color=WHITE, s=120, marker="D", zorder=12,
                 edgecolors="black", linewidths=0.5, label="Target ref")

    # Medoid star
    ax3d.scatter([coords[med_idx, 0]], [coords[med_idx, 1]], [coords[med_idx, 2]],
                 color=GOLD, s=250, marker="*", zorder=15,
                 edgecolors="black", linewidths=0.6, label="★ Medoid")

    ax3d.set_xlabel("X (mm)", color=GRAY, fontsize=7, labelpad=2)
    ax3d.set_ylabel("Y (mm)", color=GRAY, fontsize=7, labelpad=2)
    ax3d.set_zlabel("Z (mm)", color=GRAY, fontsize=7, labelpad=2)
    ax3d.tick_params(colors="#555", labelsize=6)
    ax3d.legend(loc="upper left", fontsize=7, facecolor="#111", labelcolor=WHITE, framealpha=0.8)

    # Bar chart
    ax_bar.set_facecolor(PAN)
    cids   = [l for l in unique_labels if l != -1]
    cnts   = [int(np.sum(labels == l)) for l in cids]
    cnames = [f"Cluster {l}\n(★)" if l == best_id else f"Cluster {l}" for l in cids]
    ccols  = [lcolor(l) for l in cids]
    if result["stats"]["n_noise"] > 0:
        cids.append(-1)
        cnts.append(result["stats"]["n_noise"])
        cnames.append("Noise")
        ccols.append(NOISE_COLOR)

    bars = ax_bar.bar(cnames, cnts, color=ccols, edgecolor=WHITE, linewidth=0.5, width=0.5)
    for bar, ct in zip(bars, cnts):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(ct),
                    ha="center", va="bottom", color=WHITE, fontsize=8, fontweight="bold")
    ax_bar.set_ylim(0, max(cnts) * 1.25)
    _style_ax(ax_bar, title="Cluster sizes", ylabel="Count")

    # 2D Projections
    proj_axes = [
        (ax_xy, (0, 1), ("X (mm)", "Y (mm)")),
        (ax_xz, (0, 2), ("X (mm)", "Z (mm)")),
        (ax_yz, (1, 2), ("Y (mm)", "Z (mm)")),
    ]
    for ax, (d0, d1), (xl, yl) in proj_axes:
        for l in unique_labels:
            m   = labels == l
            col = lcolor(l)
            ax.scatter(coords[m, d0], coords[m, d1],
                       color=col, s=15 if l != -1 else 8,
                       alpha=0.80 if l == best_id else 0.35, linewidths=0)
        # reference
        ref_c = [ref_row["x"], ref_row["y"], ref_row["z"]]
        ax.scatter(ref_c[d0], ref_c[d1], color=WHITE, s=80, marker="D", zorder=10, edgecolors="black", linewidths=0.5)
        # medoid
        ax.scatter(coords[med_idx, d0], coords[med_idx, d1], color=GOLD, s=180, marker="*", zorder=12, edgecolors="black", linewidths=0.5)
        _style_ax(ax, title=f"{xl.split()[0]}{yl.split()[0]} projection", xlabel=xl, ylabel=yl)

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def fig_cluster_size_summary(results: Dict[str, Dict], save_path: str) -> None:
    """
    Save group cluster sizes summary comparison.
    """
    labels = list(results.keys())
    n = len(labels)
    colors_map = {l: C_L if "left" in l or "_l" in l else C_R for l in labels}

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), facecolor=BG)
    if n == 1:
        axes = [axes]
    fig.suptitle("Cluster Size Summary", color=WHITE, fontsize=12, fontweight="bold")

    for ax, lbl in zip(axes, labels):
        res  = results[lbl]
        stats = res["stats"]
        labels_arr = res["labels"]
        unique = sorted(set(labels_arr))

        cids  = [l for l in unique if l != -1]
        cnts  = [int(np.sum(labels_arr == l)) for l in cids]
        if stats["n_noise"] > 0:
            cids.append(-1)
            cnts.append(stats["n_noise"])
        cnames = [f"C{l}\n(★)" if l == stats["best_cluster"] else (f"C{l}" if l != -1 else "Noise") for l in cids]
        ccols  = [colors_map[lbl] if l == stats["best_cluster"] else (NOISE_COLOR if l == -1 else "#888888") for l in cids]

        bars = ax.bar(cnames, cnts, color=ccols, edgecolor=WHITE, linewidth=0.5, width=0.5)
        for bar, ct in zip(bars, cnts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(ct),
                    ha="center", va="bottom", color=WHITE, fontsize=8, fontweight="bold")

        _style_ax(ax, title=lbl.upper(), ylabel="Count")
        ax.set_facecolor(PAN)
        ax.set_ylim(0, max(cnts) * 1.25)

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def fig_displacement_summary(results: Dict[str, Dict], save_path: str) -> None:
    """
    Save group translation & rotation deviations boxplots.
    """
    labels = list(results.keys())
    n = len(labels)
    colors_map = {l: C_L if "left" in l or "_l" in l else C_R for l in labels}

    fig = plt.figure(figsize=(15, 8), facecolor=BG)
    fig.suptitle(
        "Displacement & Angular Drift Summary\n"
        "◆ = mean±SD of best cluster | ★ = selected medoid | ■ = box: all points",
        color=WHITE, fontsize=11, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3, left=0.08, right=0.95, top=0.88, bottom=0.20)

    for col, (metric, unit) in enumerate([
        ("displacement_mm", "mm"),
        ("angle_deg",       "°"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(PAN)

        data_all = [results[l]["df"][metric].to_numpy() for l in labels]
        positions = list(range(1, n + 1))
        clrs = [colors_map[l] for l in labels]

        bp = ax.boxplot(
            data_all, positions=positions, patch_artist=True, widths=0.35,
            medianprops  =dict(color=GOLD, linewidth=2.0),
            whiskerprops =dict(color=GRAY, linewidth=1.0, linestyle="--"),
            capprops     =dict(color=GRAY, linewidth=1.2),
            flierprops   =dict(marker="o", markerfacecolor=GRAY, markeredgecolor="none", markersize=3, alpha=0.4),
        )
        for patch, c in zip(bp["boxes"], clrs):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)

        np.random.seed(0)
        for pos, arr, c, lbl in zip(positions, data_all, clrs, labels):
            mask = results[lbl]["labels"] == results[lbl]["best_id"]
            jitter = np.random.normal(0, 0.04, len(arr))
            ax.scatter(pos + jitter, arr, color=c, alpha=0.2, s=8, zorder=3, linewidths=0)
            ax.scatter(pos + jitter[mask], arr[mask], color=c, alpha=0.5, s=12, zorder=4, linewidths=0)

        # mean +- SD
        for pos, lbl, c in zip(positions, labels, clrs):
            res  = results[lbl]
            arr  = res["df"][metric].to_numpy()
            mask = res["labels"] == res["best_id"]
            m, s = arr[mask].mean(), arr[mask].std()
            ax.errorbar(pos, m, yerr=s, fmt="D", color=c,
                        markersize=6, markeredgecolor=WHITE, markeredgewidth=0.5,
                        ecolor=WHITE, elinewidth=1.2, capsize=4, capthick=1.2, zorder=8)

        # medoid star
        for pos, lbl in zip(positions, labels):
            res = results[lbl]
            mv  = float(res["medoid_row"][metric])
            ax.scatter(pos, mv, color=GOLD, s=200, marker="*", zorder=15, edgecolors="black", linewidths=0.5)

        y_top = max(arr.max() for arr in data_all) * 1.30
        ax.set_ylim(0, y_top)

        # Statistics text annotation on top of the bars
        for pos, lbl, c in zip(positions, labels, clrs):
            res  = results[lbl]
            arr  = res["df"][metric].to_numpy()
            mask = res["labels"] == res["best_id"]
            m_a, s_a = arr.mean(), arr.std()
            m_c, s_c = arr[mask].mean(), arr[mask].std()
            txt = (f"All: {m_a:.2f}±{s_a:.2f}{unit}\n"
                   f"Cls: {m_c:.2f}±{s_c:.2f}{unit}")
            ax.text(pos, y_top * 0.98, txt, ha="center", va="top",
                    color=c, fontsize=7, fontweight="bold", linespacing=1.4,
                    bbox=dict(facecolor=PAN, edgecolor=c, boxstyle="round,pad=0.2", alpha=0.8))

        ax.set_xticks(positions)
        ax.set_xticklabels([l.upper().replace("-", "\n").replace("_", "\n") for l in labels], color=WHITE, fontsize=8)
        title_str = "Translational Displacement" if metric == "displacement_mm" else "Angular Drift"
        ax.set_title(f"Maximum {title_str} ({unit})", color=WHITE, fontsize=10, pad=6)
        ax.set_ylabel(f"Displacement ({unit})", color=GRAY, fontsize=8)
        ax.tick_params(colors=GRAY)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.grid(axis="y", color="#2a2a3a", linewidth=0.5, linestyle="--")
        ax.set_xlim(0.4, n + 0.6)

    # Legend
    legend_elems = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor=GOLD, markersize=10, linestyle="none", label="★ Medoid"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#aaa", markeredgecolor=WHITE, markersize=6, linestyle="none", label="◆ Mean±SD (best cluster)"),
        Line2D([0], [0], color=GOLD, linewidth=2.0, label="Median (box)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_L, markersize=5, alpha=0.6, linestyle="none", label="Left hemisphere"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_R, markersize=5, alpha=0.6, linestyle="none", label="Right hemisphere"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=5, facecolor=BG, labelcolor=WHITE, fontsize=8, framealpha=0.8, bbox_to_anchor=(0.5, 0.02))

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN ROUTINE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # Setup working folders relative to CITRUS workspace
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = repo_root / "data" / "gum" / "actual" / "citrus-offline_participant_ratings - ratings.csv"
    actual_dir = repo_root / "data" / "gum" / "actual"
    planned_idx_csv = repo_root / "data" / "input" / "planned_positions_index.csv"
    planned_xml_dir = repo_root / "data" / "input"
    out_dir = repo_root / "derivatives" / "medoid"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        log.error("Ratings CSV file not found at: %s", csv_path)
        return
    if not planned_idx_csv.exists():
        log.error("Planned indices CSV file not found at: %s", planned_idx_csv)
        return

    # 1. Load CSV data
    log.info("Loading ratings CSV from %s", csv_path)
    df_ratings = pd.read_csv(csv_path)
    df_ratings = df_ratings.rename(columns={df_ratings.columns[0]: "subject"})

    log.info("Loading planned indices CSV from %s", planned_idx_csv)
    df_planned_indices = pd.read_csv(planned_idx_csv)

    subjects = df_ratings["subject"].dropna().unique()
    log.info("Processing subjects: %s", list(subjects))

    medoid_records = []

    for sub in subjects:
        log.info("=== Processing %s ===", sub)
        sub_ratings = df_ratings[df_ratings["subject"] == sub]
        sub_planned_info = df_planned_indices[df_planned_indices["Subject"] == sub]
        if sub_planned_info.empty:
            log.error("No planned indices row found for %s in %s", sub, planned_idx_csv.name)
            continue

        # Get planned XML path for the subject
        planned_xml_pattern = planned_xml_dir / sub / f"{sub}_GUMMarkers*.xml"
        planned_xml_files = [f for f in glob.glob(str(planned_xml_pattern))
                             if not ("_ses-" in os.path.basename(f) or "_medoid" in os.path.basename(f))]
        if not planned_xml_files:
            log.error("Planned GUMMarkers XML file not found under %s", planned_xml_dir / sub)
            continue
        plan_xml_path = planned_xml_files[0]
        log.info("Using planned reference XML: %s", Path(plan_xml_path).name)

        # Parse planned XML
        planned_df = parse_gummarker_xml(plan_xml_path)

        # Resolve planned targets
        idx_plan_l = int(sub_planned_info["index_left"].values[0])
        idx_plan_r = int(sub_planned_info["index_right"].values[0])
        planned_ref = {
            "L": planned_df[planned_df["frame"] == idx_plan_l].iloc[0],
            "R": planned_df[planned_df["frame"] == idx_plan_r].iloc[0],
        }

        # Keep results for subject summaries
        subject_results = {}

        # Loop through ratings rows
        for _, row in sub_ratings.iterrows():
            cond = row["condition"]
            hemi = row["hemisphere"]
            localite_file = row["localite_file"]
            if not str(localite_file).endswith(".xml"):
                localite_file = f"{localite_file}.xml"

            actual_xml_path = actual_dir / sub / localite_file
            if not actual_xml_path.exists():
                log.error("Actual XML file not found: %s", actual_xml_path)
                continue

            xml_start = int(row["xml_start"])
            xml_end = int(row["xml_end"])
            actual_indices = list(range(xml_start, xml_end + 1))

            # Label for logging and filenames
            label = f"{sub}_{cond.lower()}_{hemi.lower()}"
            log.info("Analyzing %s [frames %d-%d]", label, xml_start, xml_end)

            # Parse actual tracking data
            actual_df = parse_gummarker_xml(str(actual_xml_path))

            # Run analysis pipeline
            try:
                res = analyse_condition(
                    actual_df=actual_df,
                    ref_row=planned_ref[hemi],
                    indices=actual_indices,
                    eps="auto",
                    min_samples=3,
                    label=label,
                )
                subject_results[f"{cond.lower()}_{hemi.lower()}"] = res
                medoid_records.append({
                    "subject": sub,
                    "condition": cond,
                    "hemisphere": hemi,
                    "medoid_frame": res["stats"]["medoid_frame"]
                })

                # Generate individual figures
                fig_time_path = out_dir / f"positions_over_time_{label}.png"
                fig_positions_over_time(res, label, str(fig_time_path))

                fig_spatial_path = out_dir / f"spatial_clusters_{label}.png"
                fig_spatial_clusters(res, label, planned_ref[hemi], str(fig_spatial_path))

            except Exception as e:
                log.exception("Failed to analyze condition %s: %s", label, e)

        # Generate subject level summary plots if we have results
        if subject_results:
            log.info("Generating group summary plots for %s", sub)
            cls_sum_path = out_dir / f"{sub}_cluster_size_summary.png"
            fig_cluster_size_summary(subject_results, str(cls_sum_path))

            disp_sum_path = out_dir / f"{sub}_displacement_summary.png"
            fig_displacement_summary(subject_results, str(disp_sum_path))

    log.info("find_medoid pipeline finished. All images saved to: %s", out_dir)

    # Print medoid frames table
    table_lines = [
        "="*50,
        "  IDENTIFIED MEDOID FRAME INDICES",
        "="*50,
        f"  {'Subject':<10} {'Condition':<10} {'Hemi':<6} {'Medoid Frame':<12}",
        "  " + "─"*46
    ]
    for r in medoid_records:
        table_lines.append(f"  {r['subject']:<10} {r['condition']:<10} {r['hemisphere']:<6} {r['medoid_frame']:<12}")
    table_lines.append("="*50)
    
    table_text = "\n".join(table_lines) + "\n"
    print("\n" + table_text)

    # Write to derivatives/medoid/medoid.txt
    txt_path = out_dir / "medoid.txt"
    with open(txt_path, "w") as f:
        f.write(table_text)
    log.info("Saved medoid text summary to: %s", txt_path)


if __name__ == "__main__":
    main()
