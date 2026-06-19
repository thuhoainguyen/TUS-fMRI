"""
transducer_qc.py
================
Core library for TUS transducer-position quality control.

Provides:
  - GUMMarker XML parsing  → pandas DataFrame
  - Displacement & angular-drift computation
  - Auto-eps DBSCAN clustering + medoid selection
  - All figure generators
  - HTML report builder

All functions are stateless and accept plain numpy/pandas objects so they
can be used independently from the CLI.
"""

from __future__ import annotations

import logging
import re
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

log = logging.getLogger(__name__)

# ── Colour palette ─────────────────────────────────────────────────────────────
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
# 1.  XML PARSING
# ══════════════════════════════════════════════════════════════════════════════

def inspect_xml(xml_path: str) -> str:
    """
    Print a human-readable summary of a GUMMarker XML file.
    Returns the summary string (also suitable for printing).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    lines.append(f"File       : {xml_path}")
    lines.append(f"Root tag   : <{root.tag}>  attribs: {dict(root.attrib)}")

    # ── tag frequency ──
    tag_count: Dict[str, int] = {}
    for el in root.iter():
        tag_count[el.tag] = tag_count.get(el.tag, 0) + 1
    lines.append("\nTag frequency:")
    for tag, cnt in sorted(tag_count.items(), key=lambda x: -x[1]):
        lines.append(f"  <{tag}>  ×{cnt}")

    # ── first Element in detail ──
    elements = root.findall("Element")
    lines.append(f"\nTotal <Element> nodes : {len(elements)}")
    if elements:
        e = elements[0]
        lines.append(f"\nFirst <Element> attribs: {dict(e.attrib)}")
        for child in e.iter():
            if child is e:
                continue
            attrs = {k: v[:60] if len(v) > 60 else v
                     for k, v in child.attrib.items()}
            lines.append(f"  <{child.tag}>  {attrs}")

    # ── detect Matrix4D ──
    mat_nodes = root.findall(".//Matrix4D")
    if mat_nodes:
        lines.append(f"\nMatrix4D nodes found: {len(mat_nodes)}")
        sample = mat_nodes[0]
        lines.append(f"  Sample keys: {list(sample.attrib.keys())[:8]} ...")
        lines.append(f"  Translation column (data03, data13, data23):")
        lines.append(f"    X={sample.get('data03')}  "
                     f"Y={sample.get('data13')}  "
                     f"Z={sample.get('data23')}")
    else:
        lines.append("\nNo Matrix4D nodes found – manual parser adjustment needed.")

    return "\n".join(lines)


def _matrix4d_to_components(mat_el: ET.Element) -> Dict[str, float]:
    """Extract all numeric fields from a <Matrix4D> element."""
    return {k: float(v) for k, v in mat_el.attrib.items()}


def _rotation_to_euler_xyz(R: np.ndarray) -> Tuple[float, float, float]:
    """Convert 3×3 rotation matrix to Euler angles (rx, ry, rz) in degrees."""
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
    """Convert 3×3 rotation matrix to quaternion (w, x, y, z)."""
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


def parse_gummarker_xml(xml_path: str) -> pd.DataFrame:
    """
    Parse a GUMMarker XML file into a pandas DataFrame.

    Each row corresponds to one <Element> node.

    Columns returned
    ----------------
    frame        : int   — XML element index attribute
    description  : str   — InstrumentMarker description attribute
    time_s       : float — time in seconds parsed from description (NaN if absent)
    x, y, z      : float — translation (mm) from Matrix4D columns data03/13/23
    rot_x, rot_y, rot_z : float — Euler angles (degrees) derived from rotation block
    quat_w, quat_x, quat_y, quat_z : float — quaternion derived from rotation block
    mat_*        : float — all 16 raw Matrix4D values (mat_00 … mat_33)
    source_file  : str   — basename of the source XML
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    source = Path(xml_path).name
    records = []

    for elem in root.findall("Element"):
        frame = int(elem.get("index", -1))
        im    = elem.find(".//InstrumentMarker")
        if im is None:
            log.debug("Element %d: no InstrumentMarker – skipping", frame)
            continue

        mat_el = im.find("Matrix4D")
        if mat_el is None:
            log.warning("Element %d: no Matrix4D – skipping", frame)
            continue

        desc = im.get("description", "")

        # ── time from description  e.g. "Response: NaN µV, 42.1 s" ──────────
        m = re.search(r"([\d.]+)\s*s\b", desc)
        time_s = float(m.group(1)) if m else np.nan

        # ── 4×4 matrix ────────────────────────────────────────────────────────
        comp = _matrix4d_to_components(mat_el)
        M = np.array([[comp[f"data{r}{c}"] for c in range(4)]
                      for r in range(4)], dtype=float)

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
        # raw matrix columns
        for r in range(4):
            for c in range(4):
                row[f"mat_{r}{c}"] = M[r, c]

        records.append(row)

    df = pd.DataFrame(records)
    log.info("Parsed %d rows from %s", len(df), xml_path)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DISPLACEMENT & ANGULAR DRIFT
# ══════════════════════════════════════════════════════════════════════════════

def compute_displacement(df: pd.DataFrame,
                         ref_row: pd.Series) -> pd.DataFrame:
    """
    Add displacement_mm and angle_deg columns to *df* relative to *ref_row*.

    Parameters
    ----------
    df      : actual position DataFrame (subset of parse_gummarker_xml output)
    ref_row : single-row Series from parse_gummarker_xml representing reference

    Returns
    -------
    df with new columns: displacement_mm, angle_deg
    """
    df = df.copy()

    ref_xyz = np.array([ref_row["x"], ref_row["y"], ref_row["z"]])
    R_ref   = np.array([[ref_row[f"mat_{r}{c}"] for c in range(3)]
                        for r in range(3)])

    # translational
    xyz = df[["x", "y", "z"]].to_numpy()
    df["displacement_mm"] = np.linalg.norm(xyz - ref_xyz, axis=1)

    # angular
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
# 3.  CLUSTERING & MEDOID SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def auto_eps(coords: np.ndarray, k: int = 4) -> float:
    """
    Automatically select DBSCAN eps via the k-NN elbow method.

    Sorts the distance-to-k-th-nearest-neighbour for all points, normalises
    to [0,1], computes the second derivative, and returns the distance at the
    point of maximum curvature (the elbow).
    """
    if len(coords) <= k:
        # Fall back to 10 % of cloud diameter
        return float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)) * 0.10)

    nbrs  = NearestNeighbors(n_neighbors=k).fit(coords)
    d, _  = nbrs.kneighbors(coords)
    knn   = np.sort(d[:, k - 1])
    x     = np.linspace(0.0, 1.0, len(knn))
    y     = (knn - knn.min()) / (knn.max() - knn.min() + 1e-12)
    d2    = np.gradient(np.gradient(y, x), x)
    mg    = max(3, len(knn) // 10)
    elbow = int(np.argmax(d2[mg:-mg]) + mg)
    return float(knn[elbow])


def cluster_positions(coords: np.ndarray,
                      eps: Union[float, str],
                      min_samples: int) -> Tuple[np.ndarray, float]:
    """
    Run DBSCAN on (N,3) coords.

    Parameters
    ----------
    coords      : (N, 3) float array of x, y, z positions
    eps         : neighbourhood radius in mm, or "auto" to use auto_eps()
    min_samples : minimum points to form a core point

    Returns
    -------
    labels : (N,) int array  — DBSCAN labels (-1 = noise)
    eps_used : float          — eps value that was used
    """
    if isinstance(eps, str) and eps.lower() == "auto":
        eps_used = auto_eps(coords)
        log.info("Auto eps selected: %.4f mm", eps_used)
    else:
        eps_used = float(eps)

    labels = DBSCAN(eps=eps_used, min_samples=min_samples,
                    metric="euclidean").fit_predict(coords)
    return labels, eps_used


def find_medoid(coords: np.ndarray, mask: np.ndarray) -> int:
    """
    Return the index (into coords) of the medoid of the masked subset.

    Medoid = point minimising sum of Euclidean distances to all other points
    in the same cluster.
    """
    pts   = coords[mask]
    dists = cdist(pts, pts, metric="euclidean")
    local = int(np.argmin(dists.sum(axis=1)))
    return int(np.where(mask)[0][local])


def select_representative(df: pd.DataFrame,
                           labels: np.ndarray,
                           coords: np.ndarray) -> Tuple[int, int, bool]:
    """
    Select the best cluster and its medoid.

    Returns
    -------
    medoid_row_idx   : int  — position in df (iloc index)
    best_cluster_id  : int  — DBSCAN cluster label
    used_fallback    : bool — True if no valid cluster found, fell back to
                              closest-to-reference point
    """
    cluster_ids = [l for l in set(labels) if l != -1]

    if not cluster_ids:
        log.warning("No valid clusters found – falling back to min-displacement point.")
        fallback = int(df["displacement_mm"].idxmin())
        return fallback, -1, True

    best_id  = max(cluster_ids, key=lambda l: np.sum(labels == l))
    mask     = labels == best_id
    med_idx  = find_medoid(coords, mask)
    return med_idx, best_id, False


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FULL ANALYSIS FOR ONE CONDITION × HEMISPHERE
# ══════════════════════════════════════════════════════════════════════════════

def analyse_condition(
    actual_df: pd.DataFrame,
    ref_row:   pd.Series,
    indices:   Sequence[int],
    eps:       Union[float, str],
    min_samples: int,
    label:     str,
) -> Dict:
    """
    Run the full QC pipeline for one (condition, hemisphere) combination.

    Parameters
    ----------
    actual_df    : full DataFrame from parse_gummarker_xml (actual session)
    ref_row      : reference position Series (from planned DataFrame)
    indices      : list of frame indices to include
    eps          : DBSCAN eps in mm or "auto"
    min_samples  : DBSCAN min_samples
    label        : human label e.g. "exp-left"

    Returns
    -------
    dict with keys:
        df          : actual positions subset with displacement columns
        labels      : DBSCAN labels array
        eps_used    : float
        coords      : (N,3) array
        best_id     : int
        medoid_idx  : int  (iloc into df)
        used_fallback : bool
        medoid_row  : Series
        stats       : dict of summary statistics
    """
    # ── subset ──────────────────────────────────────────────────────────────
    sub = actual_df[actual_df["frame"].isin(indices)].copy().reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"[{label}] No rows found for indices {indices[:5]}…")

    # ── displacements ────────────────────────────────────────────────────────
    sub = compute_displacement(sub, ref_row)

    # ── clustering ───────────────────────────────────────────────────────────
    coords = sub[["x", "y", "z"]].to_numpy()
    labels, eps_used = cluster_positions(coords, eps, min_samples)
    sub["cluster"] = labels

    # ── medoid ───────────────────────────────────────────────────────────────
    med_idx, best_id, used_fallback = select_representative(sub, labels, coords)
    medoid_row = sub.iloc[med_idx]

    if used_fallback:
        log.warning("[%s] Fallback medoid: frame=%d", label, medoid_row["frame"])
    else:
        log.info("[%s] Medoid: frame=%d  cluster=%d  best_cluster_size=%d",
                 label, medoid_row["frame"], best_id,
                 int(np.sum(labels == best_id)))

    # ── summary stats ────────────────────────────────────────────────────────
    mask_best = labels == best_id if not used_fallback else np.ones(len(sub), bool)
    cluster_ids  = sorted(set(labels))
    n_clusters   = len([l for l in cluster_ids if l != -1])
    n_noise      = int(np.sum(labels == -1))
    cluster_sizes = {l: int(np.sum(labels == l))
                     for l in cluster_ids if l != -1}

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
        disp_sd_all    = float(sub["displacement_mm"].std()),
        disp_max_all   = float(sub["displacement_mm"].max()),
        disp_mean_cls  = float(sub.loc[mask_best, "displacement_mm"].mean()),
        disp_sd_cls    = float(sub.loc[mask_best, "displacement_mm"].std()),
        ang_mean_all   = float(sub["angle_deg"].mean()),
        ang_sd_all     = float(sub["angle_deg"].std()),
        ang_max_all    = float(sub["angle_deg"].max()),
        ang_mean_cls   = float(sub.loc[mask_best, "angle_deg"].mean()),
        ang_sd_cls     = float(sub.loc[mask_best, "angle_deg"].std()),
        used_fallback  = used_fallback,
    )

    return dict(
        df=sub, labels=labels, eps_used=eps_used,
        coords=coords, best_id=best_id,
        medoid_idx=med_idx, used_fallback=used_fallback,
        medoid_row=medoid_row, stats=stats,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FIGURES
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


def fig_positions_over_time(result: Dict, label: str,
                             save_path: str) -> None:
    """
    Figure: X, Y, Z, displacement, and angle vs. time/frame.
    Best-cluster points brighter; non-cluster points marked with red ×.
    """
    df        = result["df"]
    med_idx   = result["medoid_idx"]
    mask_best = df["cluster"] == result["best_id"]

    # use time_s if available, else frame number
    has_time = df["time_s"].notna().any()
    t_col    = "time_s" if has_time else "frame"
    t_label  = "Time (s)" if has_time else "Frame index"
    t        = df[t_col].to_numpy()

    norm  = Normalize(vmin=t.min(), vmax=t.max())
    cmap  = plt.get_cmap(CMAP_TIME)

    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    fig.suptitle(f"Transducer Positions Over Time — {label.upper()}\n"
                 f"(n={len(df)}  |  DBSCAN eps={result['eps_used']:.3f} mm  |  "
                 f"best cluster={result['stats']['best_cluster_n']} pts)",
                 color=WHITE, fontsize=13, fontweight="bold", y=0.99)

    gs   = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.30,
                             left=0.07, right=0.97, top=0.93, bottom=0.06)
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

        ax.plot(t, vals, color=acol, linewidth=0.5, alpha=0.25, zorder=1)
        ax.scatter(t, vals, c=colors_pt, s=18, alpha=0.75, zorder=3,
                   linewidths=0)

        # non-best-cluster red ×
        nb = ~mask_best
        if nb.any():
            ax.scatter(t[nb.to_numpy()], vals[nb.to_numpy()],
                       color="#ff4444", s=30, marker="x",
                       linewidths=1.2, alpha=0.65, zorder=5,
                       label="Drift / noise")

        # medoid star
        ax.scatter(t[med_idx], vals[med_idx], color=GOLD, s=260,
                   marker="*", zorder=10, edgecolors="black",
                   linewidths=0.6, label=f"Medoid")

        _style_ax(ax, title="", xlabel=t_label if row == 4 else "",
                  ylabel=ylbl)
        if row == 0:
            ax.legend(fontsize=7, facecolor="#111", labelcolor=WHITE,
                      loc="upper right", framealpha=0.8)

    # ── 3D trajectory coloured by time ──────────────────────────────────────
    ax3d.set_facecolor(PAN)
    xyz = df[["x", "y", "z"]].to_numpy()
    ax3d.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                 c=cmap(norm(t)), s=22, alpha=0.80, depthshade=True)
    ax3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2],
              color=WHITE, linewidth=0.4, alpha=0.15)
    if (~mask_best).any():
        nb_xyz = xyz[~mask_best.to_numpy()]
        ax3d.scatter(nb_xyz[:, 0], nb_xyz[:, 1], nb_xyz[:, 2],
                     color="#ff4444", s=40, marker="x",
                     linewidths=1.5, alpha=0.65)
    ax3d.scatter([xyz[med_idx, 0]], [xyz[med_idx, 1]], [xyz[med_idx, 2]],
                 color=GOLD, s=300, marker="*", zorder=15,
                 edgecolors="black", linewidths=0.7)
    ax3d.set_xlabel("X", color=GRAY, fontsize=7, labelpad=1)
    ax3d.set_ylabel("Y", color=GRAY, fontsize=7, labelpad=1)
    ax3d.set_zlabel("Z", color=GRAY, fontsize=7, labelpad=1)
    ax3d.tick_params(colors="#555", labelsize=6)
    ax3d.set_title("3D trajectory (colour = time)", color=WHITE, fontsize=9)

    # colour-bar
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                      ax=fig.get_axes()[:-1], shrink=0.5, pad=0.01,
                      orientation="vertical")
    cb.set_label(t_label, color=GRAY)
    cb.ax.tick_params(colors=GRAY)

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info("Saved: %s", save_path)


def fig_spatial_clusters(result: Dict, label: str,
                          ref_row: pd.Series, save_path: str) -> None:
    """
    Figure: 3D scatter + 3 orthogonal projections coloured by cluster,
    reference position shown as a diamond, medoid as a gold star.
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
        ordered = [best_id] + [x for x in unique_labels
                               if x != -1 and x != best_id]
        return CLUSTER_COLORS[ordered.index(l) % len(CLUSTER_COLORS)]

    fig = plt.figure(figsize=(20, 13), facecolor=BG)
    fig.suptitle(
        f"Spatial Clusters — {label.upper()}\n"
        f"DBSCAN eps={result['eps_used']:.3f} mm  |  "
        f"{result['stats']['n_clusters']} cluster(s)  |  "
        f"noise={result['stats']['n_noise']}  |  "
        f"selected (blue) = {result['stats']['best_cluster_n']} pts",
        color=WHITE, fontsize=12, fontweight="bold", y=0.99)

    gs   = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32,
                             left=0.05, right=0.97, top=0.91, bottom=0.07)
    ax3d = fig.add_subplot(gs[0, :2], projection="3d")
    ax_bar = fig.add_subplot(gs[0, 2])
    ax_xy  = fig.add_subplot(gs[1, 0])
    ax_xz  = fig.add_subplot(gs[1, 1])
    ax_yz  = fig.add_subplot(gs[1, 2])

    # ── 3D ──────────────────────────────────────────────────────────────────
    ax3d.set_facecolor(PAN)
    for l in unique_labels:
        m   = labels == l
        col = lcolor(l)
        nm  = (f"Cluster {l} — SELECTED (n={m.sum()})" if l == best_id
               else f"Noise (n={m.sum()})" if l == -1
               else f"Cluster {l} — drift (n={m.sum()})")
        ax3d.scatter(coords[m, 0], coords[m, 1], coords[m, 2],
                     color=col, s=28 if l != -1 else 16,
                     alpha=0.80 if l == best_id else 0.45,
                     depthshade=True, label=nm)

    # reference diamond
    ax3d.scatter([ref_row["x"]], [ref_row["y"]], [ref_row["z"]],
                 color=WHITE, s=160, marker="D", zorder=12,
                 edgecolors="black", linewidths=0.7, label="Planned ref")

    # medoid star
    ax3d.scatter([coords[med_idx, 0]], [coords[med_idx, 1]],
                 [coords[med_idx, 2]],
                 color=GOLD, s=320, marker="*", zorder=15,
                 edgecolors="black", linewidths=0.8,
                 label=f"★ Medoid (frame {int(df.iloc[med_idx]['frame'])})")

    ax3d.set_xlabel("X (mm)", color=GRAY, fontsize=7, labelpad=2)
    ax3d.set_ylabel("Y (mm)", color=GRAY, fontsize=7, labelpad=2)
    ax3d.set_zlabel("Z (mm)", color=GRAY, fontsize=7, labelpad=2)
    ax3d.tick_params(colors="#555", labelsize=6)
    ax3d.legend(loc="upper left", fontsize=7.5, facecolor="#111",
                labelcolor=WHITE, framealpha=0.85)

    # ── cluster bar chart ────────────────────────────────────────────────────
    ax_bar.set_facecolor(PAN)
    cids   = [l for l in unique_labels if l != -1]
    cnts   = [int(np.sum(labels == l)) for l in cids]
    cnames = [f"Cluster {l}\n(★ SELECTED)" if l == best_id
              else f"Cluster {l}" for l in cids]
    ccols  = [lcolor(l) for l in cids]
    if result["stats"]["n_noise"] > 0:
        cids.append(-1); cnts.append(result["stats"]["n_noise"])
        cnames.append("Noise"); ccols.append(NOISE_COLOR)

    bars = ax_bar.bar(cnames, cnts, color=ccols,
                      edgecolor=WHITE, linewidth=0.6, width=0.55)
    for bar, ct in zip(bars, cnts):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2, str(ct),
                    ha="center", va="bottom", color=WHITE,
                    fontsize=10, fontweight="bold")
    ax_bar.set_ylim(0, max(cnts) * 1.20)
    _style_ax(ax_bar, title="Cluster sizes", ylabel="Count")

    # ── 2D projections ───────────────────────────────────────────────────────
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
                       color=col, s=20 if l != -1 else 12,
                       alpha=0.80 if l == best_id else 0.38,
                       linewidths=0)
        # reference
        ref_c = [ref_row["x"], ref_row["y"], ref_row["z"]]
        ax.scatter(ref_c[d0], ref_c[d1], color=WHITE, s=120,
                   marker="D", zorder=10, edgecolors="black",
                   linewidths=0.5)
        # medoid
        ax.scatter(coords[med_idx, d0], coords[med_idx, d1],
                   color=GOLD, s=240, marker="*", zorder=12,
                   edgecolors="black", linewidths=0.5)
        _style_ax(ax, title=f"{xl.split()[0]}{yl.split()[0]} projection",
                  xlabel=xl, ylabel=yl)

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info("Saved: %s", save_path)


def fig_cluster_size_summary(results: Dict[str, Dict],
                              save_path: str) -> None:
    """Bar chart comparing cluster sizes across all four conditions."""
    labels = list(results.keys())
    n = len(labels)
    colors_map = {l: C_L if "left" in l else C_R for l in labels}

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), facecolor=BG)
    if n == 1:
        axes = [axes]
    fig.suptitle("Cluster Size Summary — All Conditions",
                 color=WHITE, fontsize=13, fontweight="bold")

    for ax, lbl in zip(axes, labels):
        res  = results[lbl]
        stats = res["stats"]
        labels_arr = res["labels"]
        unique = sorted(set(labels_arr))

        cids  = [l for l in unique if l != -1]
        cnts  = [int(np.sum(labels_arr == l)) for l in cids]
        if stats["n_noise"] > 0:
            cids.append(-1); cnts.append(stats["n_noise"])
        cnames = [f"C{l}\n(★)" if l == stats["best_cluster"]
                  else (f"C{l}" if l != -1 else "Noise") for l in cids]
        ccols  = [colors_map[lbl] if l == stats["best_cluster"]
                  else (NOISE_COLOR if l == -1
                        else "#888888") for l in cids]

        bars = ax.bar(cnames, cnts, color=ccols, edgecolor=WHITE,
                      linewidth=0.6, width=0.55)
        for bar, ct in zip(bars, cnts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2, str(ct),
                    ha="center", va="bottom", color=WHITE,
                    fontsize=9, fontweight="bold")

        _style_ax(ax, title=lbl.upper(), ylabel="Count")
        ax.set_facecolor(PAN)
        ax.set_ylim(0, max(cnts) * 1.22)

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info("Saved: %s", save_path)


def fig_displacement_summary(results: Dict[str, Dict],
                              save_path: str) -> None:
    """
    Summary boxplot: translational displacement and angular drift
    for all four conditions side-by-side.
    """
    labels = list(results.keys())
    n = len(labels)
    colors_map = {l: C_L if "left" in l else C_R for l in labels}

    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    fig.suptitle(
        "Displacement & Angular Drift Summary — All Conditions\n"
        "◆ = mean±SD of best cluster  |  ★ = selected medoid  |  "
        "■ = box: all recorded points",
        color=WHITE, fontsize=12, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35,
                           left=0.07, right=0.96, top=0.88, bottom=0.22)

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
            data_all, positions=positions, patch_artist=True, widths=0.42,
            medianprops  =dict(color=GOLD, linewidth=2.5),
            whiskerprops =dict(color=GRAY, linewidth=1.4, linestyle="--"),
            capprops     =dict(color=GRAY, linewidth=1.8),
            flierprops   =dict(marker="o", markerfacecolor=GRAY,
                               markeredgecolor="none",
                               markersize=4, alpha=0.45),
        )
        for patch, c in zip(bp["boxes"], clrs):
            patch.set_facecolor(c); patch.set_alpha(0.40)

        np.random.seed(0)
        for pos, arr, c, lbl in zip(positions, data_all, clrs, labels):
            mask = results[lbl]["labels"] == results[lbl]["best_id"]
            jitter = np.random.normal(0, 0.05, len(arr))
            ax.scatter(pos + jitter, arr,
                       color=c, alpha=0.25, s=12, zorder=3, linewidths=0)
            ax.scatter(pos + jitter[mask], arr[mask],
                       color=c, alpha=0.60, s=16, zorder=4, linewidths=0)

        # mean ± SD diamond for best cluster
        for pos, lbl, c in zip(positions, labels, clrs):
            res  = results[lbl]
            arr  = res["df"][metric].to_numpy()
            mask = res["labels"] == res["best_id"]
            m, s = arr[mask].mean(), arr[mask].std()
            ax.errorbar(pos, m, yerr=s, fmt="D", color=c,
                        markersize=8, markeredgecolor=WHITE,
                        markeredgewidth=0.7,
                        ecolor=WHITE, elinewidth=1.8,
                        capsize=5, capthick=1.6, zorder=8)

        # medoid star
        for pos, lbl in zip(positions, labels):
            res = results[lbl]
            mv  = float(res["medoid_row"][metric])
            ax.scatter(pos, mv, color=GOLD, s=280, marker="*",
                       zorder=15, edgecolors="black", linewidths=0.7)

        y_top = max(arr.max() for arr in data_all) * 1.28
        ax.set_ylim(0, y_top)

        # stat annotation
        for pos, lbl, c in zip(positions, labels, clrs):
            res  = results[lbl]
            arr  = res["df"][metric].to_numpy()
            mask = res["labels"] == res["best_id"]
            m_a, s_a = arr.mean(), arr.std()
            m_c, s_c = arr[mask].mean(), arr[mask].std()
            txt = (f"All: {m_a:.2f}±{s_a:.2f}{unit}\n"
                   f"Cls: {m_c:.2f}±{s_c:.2f}{unit}")
            ax.text(pos, y_top * 0.97, txt, ha="center", va="top",
                    color=c, fontsize=7.5, fontweight="bold",
                    linespacing=1.5,
                    bbox=dict(facecolor=PAN, edgecolor=c,
                              boxstyle="round,pad=0.3", alpha=0.85))

        ax.set_xticks(positions)
        ax.set_xticklabels([l.upper().replace("-", "\n") for l in labels],
                           color=WHITE, fontsize=9)
        title_str = ("Translational Displacement" if metric == "displacement_mm"
                     else "Angular Drift")
        ax.set_title(f"Maximum {title_str} ({unit})",
                     color=WHITE, fontsize=11, pad=7)
        ax.set_ylabel(f"Displacement ({unit})", color=GRAY, fontsize=9)
        ax.tick_params(colors=GRAY)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.grid(axis="y", color="#2a2a3a", linewidth=0.7, linestyle="--")
        ax.set_xlim(0.4, n + 0.6)

    # legend
    legend_elems = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor=GOLD,
               markersize=13, linestyle="none",
               label="★ Medoid (post-hoc kPlan)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#aaa",
               markeredgecolor=WHITE, markersize=7, linestyle="none",
               label="◆ Mean±SD (best cluster)"),
        Line2D([0], [0], color=GOLD, linewidth=2.5,
               label="Median (box)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_L,
               markersize=6, alpha=0.6, linestyle="none",
               label="Left hemisphere"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_R,
               markersize=6, alpha=0.6, linestyle="none",
               label="Right hemisphere"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=5,
               facecolor=BG, labelcolor=WHITE, fontsize=9,
               framealpha=0.85, bbox_to_anchor=(0.5, 0.01))

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info("Saved: %s", save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  HTML REPORT
# ══════════════════════════════════════════════════════════════════════════════

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{{ subject }} — Transducer Position QC Report</title>
<style>
  body { font-family: Arial, sans-serif; background: #121212; color: #e0e0e0;
         margin: 40px; line-height: 1.6; }
  h1   { color: #FFD700; border-bottom: 2px solid #333; padding-bottom: 8px; }
  h2   { color: #4C9BE8; margin-top: 36px; }
  h3   { color: #aaddff; }
  table{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }
  th   { background: #1e3050; color: #FFD700; padding: 7px 12px; text-align: left; }
  td   { padding: 6px 12px; border-bottom: 1px solid #2a2a2a; }
  tr:nth-child(even) td { background: #1a1a2a; }
  .highlight { color: #FFD700; font-weight: bold; }
  .warn      { color: #ff7777; font-weight: bold; }
  img  { max-width: 100%; margin: 12px 0; border: 1px solid #333;
         border-radius: 4px; }
  pre  { background: #1a1a2a; padding: 14px; border-radius: 6px;
         font-size: 12px; overflow-x: auto; color: #ccc; }
  .methods { background: #0d1a2a; border-left: 4px solid #4C9BE8;
             padding: 14px 20px; margin: 16px 0; border-radius: 4px;
             font-style: italic; }
</style>
</head>
<body>

<h1>Transducer Position QC Report — {{ subject }}</h1>
<p>Generated: {{ generated }}</p>

<!-- ── 1. Overview ───────────────────────────────────────────────────── -->
<h2>1. Analysis Overview</h2>
<table>
  <tr><th>Field</th><th>Value</th></tr>
  <tr><td>Subject</td><td>{{ subject }}</td></tr>
  {% for k, v in overview.items() %}
  <tr><td>{{ k }}</td><td>{{ v }}</td></tr>
  {% endfor %}
</table>

<!-- ── 2. Positions over time ────────────────────────────────────────── -->
<h2>2. Transducer Positions Over Time</h2>
<p>X, Y, Z coordinates and displacement/angular drift plotted against frame
   time. Red × markers indicate frames outside the selected stable cluster.
   ★ denotes the recommended medoid position.</p>
{% for lbl, fig_path in figs_time.items() %}
<h3>{{ lbl.upper() }}</h3>
<img src="{{ fig_path }}" alt="Positions over time — {{ lbl }}">
{% endfor %}

<!-- ── 3. Spatial clusters ───────────────────────────────────────────── -->
<h2>3. Spatial Distribution and Clustering</h2>
<p>All recorded positions shown in 3D and three orthogonal projections,
   coloured by DBSCAN cluster. ◆ = planned reference. ★ = selected medoid.</p>
{% for lbl, fig_path in figs_spatial.items() %}
<h3>{{ lbl.upper() }}</h3>
<img src="{{ fig_path }}" alt="Spatial clusters — {{ lbl }}">
{% endfor %}

<!-- ── 4. Cluster summary ────────────────────────────────────────────── -->
<h2>4. Cluster Summary</h2>
<img src="{{ fig_cluster_summary }}" alt="Cluster size summary">
{% for lbl, stats in all_stats.items() %}
<h3>{{ lbl.upper() }}</h3>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Total actual positions</td><td>{{ stats.n_total }}</td></tr>
  <tr><td>Number of DBSCAN clusters</td><td>{{ stats.n_clusters }}</td></tr>
  <tr><td>Noise points</td>
      <td>{{ stats.n_noise }} ({{ "%.1f"|format(100*stats.n_noise/stats.n_total) }}%)</td></tr>
  <tr><td>Selected cluster ID</td>
      <td class="highlight">{{ stats.best_cluster }}</td></tr>
  <tr><td>Selected cluster size</td>
      <td class="highlight">{{ stats.best_cluster_n }}
        ({{ "%.1f"|format(100*stats.best_cluster_n/stats.n_total) }}%)</td></tr>
  <tr><td>DBSCAN eps used</td><td>{{ "%.4f"|format(stats.eps_used) }} mm</td></tr>
  <tr><td>★ Medoid frame index</td>
      <td class="highlight">{{ stats.medoid_frame }}</td></tr>
  <tr><td>Medoid X / Y / Z (mm)</td>
      <td>{{ "%.4f"|format(stats.medoid_x) }} /
          {{ "%.4f"|format(stats.medoid_y) }} /
          {{ "%.4f"|format(stats.medoid_z) }}</td></tr>
  <tr><td>Medoid displacement from ref</td>
      <td class="highlight">{{ "%.3f"|format(stats.medoid_disp_mm) }} mm</td></tr>
  <tr><td>Medoid angle from ref</td>
      <td class="highlight">{{ "%.3f"|format(stats.medoid_ang_deg) }}°</td></tr>
  {% if stats.used_fallback %}
  <tr><td>Clustering note</td>
      <td class="warn">⚠ No valid cluster found — fallback to min-displacement point</td></tr>
  {% endif %}
</table>
{% endfor %}

<!-- ── 5. Displacement summary ──────────────────────────────────────── -->
<h2>5. Displacement and Angular Drift Summary</h2>
<img src="{{ fig_displacement_summary }}" alt="Displacement summary">
<table>
  <tr>
    <th>Condition</th>
    <th>Trans. all — mean±SD (mm)</th>
    <th>Trans. cluster — mean±SD (mm)</th>
    <th>Trans. max (mm)</th>
    <th>Ang. all — mean±SD (°)</th>
    <th>Ang. cluster — mean±SD (°)</th>
    <th>Ang. max (°)</th>
    <th>★ Medoid trans (mm)</th>
    <th>★ Medoid ang (°)</th>
  </tr>
  {% for lbl, stats in all_stats.items() %}
  <tr>
    <td>{{ lbl.upper() }}</td>
    <td>{{ "%.2f"|format(stats.disp_mean_all) }} ± {{ "%.2f"|format(stats.disp_sd_all) }}</td>
    <td>{{ "%.2f"|format(stats.disp_mean_cls) }} ± {{ "%.2f"|format(stats.disp_sd_cls) }}</td>
    <td>{{ "%.2f"|format(stats.disp_max_all) }}</td>
    <td>{{ "%.2f"|format(stats.ang_mean_all) }} ± {{ "%.2f"|format(stats.ang_sd_all) }}</td>
    <td>{{ "%.2f"|format(stats.ang_mean_cls) }} ± {{ "%.2f"|format(stats.ang_sd_cls) }}</td>
    <td>{{ "%.2f"|format(stats.ang_max_all) }}</td>
    <td class="highlight">{{ "%.3f"|format(stats.medoid_disp_mm) }}</td>
    <td class="highlight">{{ "%.3f"|format(stats.medoid_ang_deg) }}</td>
  </tr>
  {% endfor %}
</table>

<!-- ── 6. Methods text ───────────────────────────────────────────────── -->
<h2>6. Methods — Post-Hoc Position Selection</h2>
<div class="methods">{{ methods_text }}</div>

</body>
</html>
"""


def build_html_report(
    subject: str,
    overview: Dict,
    all_stats: Dict[str, Dict],
    results: Dict[str, Dict],
    figs_time: Dict[str, str],
    figs_spatial: Dict[str, str],
    fig_cluster_summary: str,
    fig_displacement_summary: str,
    out_path: str,
) -> None:
    """Render and save the HTML report."""
    try:
        from jinja2 import Environment
    except ImportError:
        raise ImportError("jinja2 is required for HTML reports. "
                          "pip install jinja2")

    from datetime import datetime

    env  = Environment()
    tmpl = env.from_string(_HTML_TEMPLATE)

    # ── Methods text (auto-adapted if fallback used) ─────────────────────────
    any_fallback = any(r["used_fallback"] for r in results.values())
    if any_fallback:
        fallback_note = (
            " Note: for one or more conditions, DBSCAN did not identify a "
            "valid cluster; in those cases the position with the smallest "
            "Euclidean distance to the planned reference was selected as "
            "the representative position and clearly flagged in the report."
        )
    else:
        fallback_note = ""

    eps_vals = ", ".join(
        f"{lbl}: {r['stats']['eps_used']:.3f} mm"
        for lbl, r in results.items()
    )

    methods_text = textwrap.dedent(f"""
        The planned transducer reference position was defined from the
        selected GUMMarker index for each hemisphere and condition.
        Actual transducer positions recorded during stimulation were
        extracted from the corresponding recorded GUMMarker file and
        compared with the planned reference position. Translational
        displacement was computed as the Euclidean distance between the
        actual and planned translation vectors (mm). Angular drift was
        computed from the geodesic angle between the actual and planned
        rotation matrices: θ = arccos((trace(R_ref^T · R_actual) − 1) / 2).

        To avoid selecting transient outlier frames, actual positions were
        clustered in Cartesian (x, y, z) coordinate space using DBSCAN
        (Ester et al., 1996). The neighbourhood radius (ε) was selected
        automatically per condition using the 4-nearest-neighbour elbow
        method, which identifies the natural spatial scale of each point
        cloud without requiring a manually tuned threshold
        ({eps_vals}; min_samples = {list(results.values())[0]['stats'].get('min_samples', 'see config')}).
        The largest non-noise cluster was interpreted as the most
        representative stable transducer placement during the stimulation
        period. The medoid of this cluster — the actual observed position
        minimising the mean Euclidean distance to all other positions in
        the stable cluster — was selected for post-hoc acoustic simulation
        in kPlan, because it corresponds to a physically realised
        transducer pose rather than an interpolated centroid.
        Positional and angular drift relative to the planned reference
        were summarised to evaluate placement stability and
        reproducibility.{fallback_note}
    """).strip()

    html = tmpl.render(
        subject=subject,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        overview=overview,
        all_stats={lbl: r["stats"] for lbl, r in results.items()},
        figs_time=figs_time,
        figs_spatial=figs_spatial,
        fig_cluster_summary=fig_cluster_summary,
        fig_displacement_summary=fig_displacement_summary,
        methods_text=methods_text,
    )

    Path(out_path).write_text(html, encoding="utf-8")
    log.info("HTML report saved: %s", out_path)
