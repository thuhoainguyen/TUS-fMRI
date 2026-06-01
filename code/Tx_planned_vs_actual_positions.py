"""
Visualize planned vs actual (medoid) transducer positions on the subject head scalp mesh.

This script loads the SimNIBS head mesh, resolves the planned transducer indices from
a CSV index, parses actual medoid transducer indices from the medoid XML filenames,
computes their circular disc projections with outward normals snapped to the surface,
and paints their intersection in green using a 2-D path grid rasterization algorithm.

@author Hoai Thu Nguyen
"""

import os
import re
import sys
import glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch as _Patch
from matplotlib.tri import Triangulation as _Tri
from matplotlib.path import Path as _MplPath
import numpy as np
import pandas as pd

# Set up logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Tx_planned_vs_actual")

# Attempt importing meshio for reading GMSH mesh files
try:
    import meshio
    HAS_MESHIO = True
except ImportError:
    meshio = None
    HAS_MESHIO = False


# ── data structures ──────────────────────────────────────────────────────────

@dataclass
class TxMatrix:
    """
    Represents a transducer marker with an index, label description, and 4x4 matrix.

    Attributes:
        index (int): Transducer numerical index.
        description (str): Transducer name/position description.
        matrix (np.ndarray): 4x4 coordinate transform matrix in RAS coordinate space.
    """
    index: int
    description: str
    matrix: np.ndarray

    @property
    def center(self) -> np.ndarray:
        """Return the 3-D center coordinate of the transducer in millimeters."""
        return self.matrix[:3, 3].astype(float)


# ── helper coordinate functions ──────────────────────────────────────────────

def lps_to_ras_matrix() -> np.ndarray:
    """
    Return the 4x4 LPS to RAS coordinate-system transformation matrix.

    Returns:
        np.ndarray: The diagonal coordinate flip matrix.
    """
    return np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)


def parse_gummarkers(xml_path: Path, convert_lps_to_ras: bool = True) -> List[TxMatrix]:
    """
    Parse a Localite GUMMarkers XML file into a list of TxMatrix objects.

    Args:
        xml_path (Path): Path to the GUMMarkers XML file.
        convert_lps_to_ras (bool): Whether to convert coordinates from LPS space to RAS space.

    Returns:
        List[TxMatrix]: List of parsed transducer matrices.
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    coord_system = (root.get("coordinateSpace", "RAS") or "RAS").upper()
    flip = np.eye(4)
    if convert_lps_to_ras and coord_system == "LPS":
        flip = lps_to_ras_matrix()
    
    out: List[TxMatrix] = []
    for elem in root.findall("Element"):
        idx = int(elem.get("index", len(out)))
        im = elem.find("InstrumentMarker")
        descr = ""
        if im is not None:
            descr = im.get("description", "") or ""
        mat_node = elem.find(".//Matrix4D")
        if mat_node is None:
            continue
        
        M = np.zeros((4, 4), dtype=float)
        for r in range(4):
            for c in range(4):
                val = mat_node.get(f"data{r}{c}")
                if val is None:
                    raise ValueError(f"Missing Matrix4D data{r}{c} in {xml_path}, element index {idx}")
                M[r, c] = float(val)
        
        M = flip @ M
        out.append(TxMatrix(index=idx, description=descr, matrix=M))
        
    if not out:
        raise ValueError(f"No Matrix4D entries found in XML: {xml_path}")
    return out


# ── mesh loading & projection functions ───────────────────────────────────────

def load_mesh(mesh_path: Path, scalp_tag: int = 1005, max_triangles: int = 75000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Load a GMSH .msh file and extract the scalp boundary triangulation mesh.

    Args:
        mesh_path (Path): Path to the SimNIBS .msh file.
        scalp_tag (int): Cell physical tag representing scalp (default 1005).
        max_triangles (int): Target maximum triangles to downsample for quick plotting.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, float]: 
            - points: array of 3-D vertex positions.
            - tris_plot: array of triangulation index triplets.
            - mid: center point of the mesh bounding box.
            - rng: half of the maximum dimension of the bounding box.
    """
    if not HAS_MESHIO:
        raise ImportError("meshio is required for mesh plotting. Install with: pip install meshio")
    
    mesh = meshio.read(str(mesh_path))
    points = mesh.points
    cells = {c.type: c.data for c in mesh.cells}
    if "triangle" not in cells:
        raise RuntimeError("Mesh has no triangle cells")
    
    triangles = cells["triangle"]
    tri_phys = None
    if hasattr(mesh, "cell_data_dict") and "gmsh:physical" in mesh.cell_data_dict:
        tri_phys = mesh.cell_data_dict["gmsh:physical"].get("triangle", None)
        
    if tri_phys is not None and scalp_tag in np.unique(tri_phys):
        tris = triangles[tri_phys == scalp_tag]
    else:
        tris = triangles
        
    # Downsample triangles if the triangulation is extremely high resolution
    step = max(1, tris.shape[0] // max_triangles)
    tris_plot = tris[::step]
    
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    mid = (xyz_min + xyz_max) / 2.0
    rng = float((xyz_max - xyz_min).max() / 2.0)
    return points, tris_plot, mid, rng


def _snap_tx_to_surface(tx_center: np.ndarray, mesh_points: np.ndarray) -> np.ndarray:
    """
    Snap the transducer center to the nearest vertex on the scalp surface mesh.

    Args:
        tx_center (np.ndarray): Original 3-D transducer coordinates.
        mesh_points (np.ndarray): Scalp mesh boundary vertices.

    Returns:
        np.ndarray: Snapped coordinates on the scalp surface.
    """
    dists = np.linalg.norm(mesh_points - tx_center[None, :], axis=1)
    return mesh_points[int(np.argmin(dists))].copy()


def _outward_normal(surface_point: np.ndarray, mesh_mid: np.ndarray) -> np.ndarray:
    """
    Estimate the outward-facing surface normal vector from skull centroid to target point.

    Args:
        surface_point (np.ndarray): Snapped surface coordinate.
        mesh_mid (np.ndarray): Centroid of the mesh bounding box.

    Returns:
        np.ndarray: Normalized outward-facing 3-D vector.
    """
    n = surface_point - mesh_mid
    return n / (np.linalg.norm(n) + 1e-9)


def _project_disc_to_2d(origin: np.ndarray, normal: np.ndarray,
                         radius_mm: float, view_name: str,
                         n_pts: int = 72) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project a 3-D flat disc ring onto a 2-D orthographic projection coordinate plane.

    Args:
        origin (np.ndarray): 3-D snapped transducer center.
        normal (np.ndarray): 3-D outward normal vector.
        radius_mm (float): Radial size of the disk in mm.
        view_name (str): Slicing viewport camera angle (left, right, front, top).
        n_pts (int): Circle angular resolution.

    Returns:
        Tuple[np.ndarray, np.ndarray]: x and y coordinates on the 2-D plot plane.
    """
    n = normal / (np.linalg.norm(normal) + 1e-9)
    ref = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([1., 0., 0.])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u) + 1e-9
    v = np.cross(n, u)
    
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=True)
    rim3d = origin[None, :] + radius_mm * (np.cos(theta)[:, None] * u +
                                            np.sin(theta)[:, None] * v)
    
    # Project and discard depth depending on the view angle
    if view_name in ("left", "right"):
        xs_r = rim3d[:, 1] * (-1.0 if view_name == "right" else 1.0)
        return xs_r, rim3d[:, 2]
    elif view_name in ("front", "back"):
        return rim3d[:, 0], rim3d[:, 2]
    else:
        return rim3d[:, 0], rim3d[:, 1]


def _mesh_silhouette_2d(points: np.ndarray, tris: np.ndarray,
                          view_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2-D vertices and sort triangles back-to-front for Painter's algorithm rendering.

    Args:
        points (np.ndarray): 3-D scalp vertices.
        tris (np.ndarray): Triangle connectivity triplets.
        view_name (str): Viewport string.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            - Projected x coordinates.
            - Projected y coordinates.
            - Depth array per vertex.
            - Reordered triangles (back-to-front).
    """
    if view_name in ("left", "right"):
        ys = points[:, 2]
        depth = points[:, 0] * (1.0 if view_name == "right" else -1.0)
        xs = points[:, 1] * (-1.0 if view_name == "right" else 1.0)
    elif view_name in ("front", "back"):
        xs, ys = points[:, 0], points[:, 2]
        depth  = points[:, 1] * (1.0 if view_name == "front" else -1.0)
    else:
        xs, ys = points[:, 0], points[:, 1]
        depth  = points[:, 2] * (1.0 if view_name == "top" else -1.0)
        
    order = np.argsort(depth[tris].mean(axis=1))
    return xs, ys, depth, tris[order]


# ── plotting & overlap function ──────────────────────────────────────────────

def plot_planned_vs_actual_mesh(subject: str,
                                mesh_path: Path,
                                planned_txs: List[TxMatrix],
                                actual_txs: List[TxMatrix],
                                out_path: Path) -> None:
    """
    Generate and save a 4-view comparison layout comparing planned and actual medoid positions.

    Args:
        subject (str): Subject identifier.
        mesh_path (Path): Path to SimNIBS .msh file.
        planned_txs (List[TxMatrix]): List of resolved planned transducers (Left and Right).
        actual_txs (List[TxMatrix]): List of resolved actual medoid transducers (Left and Right).
        out_path (Path): Output filename path to save PNG figure.
    """
    log.info(f"[{subject}] Rendering planned vs actual transducer positions...")
    points, tris, mid, rng = load_mesh(mesh_path)

    PLANNED_COL = "#ffc107"  # Yellow for planned
    ACTUAL_COL  = "#1e90ff"  # Blue for actual medoid
    OVERLAP_COL = "#22c55e"  # Green for overlap
    DISC_RADIUS_MM = 31.0    # 62 mm diameter

    four_views = [
        ("Left / lateral",  "left"),
        ("Right / lateral", "right"),
        ("Top",             "top"),
        ("Front",           "front"),
    ]

    # Pre-compute continuous planned centers (unsnapped for projection) and surface normal unit vectors
    planned_snapped: List[Tuple[np.ndarray, np.ndarray]] = []
    for tx in planned_txs:
        sp = _snap_tx_to_surface(tx.center, points)
        normal = _outward_normal(sp, mid)
        planned_snapped.append((tx.center, normal))

    # Pre-compute continuous actual medoid centers (unsnapped for projection) and surface normal unit vectors
    actual_snapped: List[Tuple[np.ndarray, np.ndarray]] = []
    for tx in actual_txs:
        sp = _snap_tx_to_surface(tx.center, points)
        normal = _outward_normal(sp, mid)
        actual_snapped.append((tx.center, normal))

    # Set up matplotlib figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), facecolor="white")

    for ax, (view_title, view_name) in zip(axes, four_views):
        ax.set_facecolor("#f8f9fa")
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(view_title, fontsize=11, fontweight="bold")

        # ── Scalp silhouette rendering (Greys_r depth shading) ──────────
        xs2d, ys2d, depth, tris_s = _mesh_silhouette_2d(points, tris, view_name)
        d_min, d_max = depth.min(), depth.max()
        d_norm = (depth - d_min) / (d_max - d_min + 1e-9)
        tri_d  = d_norm[tris_s].mean(axis=1)
        triang = _Tri(xs2d, ys2d, tris_s)
        ax.tripcolor(triang, facecolors=tri_d,
                     cmap="Greys_r", vmin=0.0, vmax=1.0,
                     edgecolors="none", rasterized=True)

        # ── View camera direction vector ──────────────────────────────────
        if   view_name == "left":   cam = np.array([-1., 0.,  0.])
        elif view_name == "right":  cam = np.array([ 1., 0.,  0.])
        elif view_name == "top":    cam = np.array([ 0., 0.,  1.])
        else:                       cam = np.array([ 0.,  1.,  0.])  # front

        # Collect projected polygon outline coordinates
        plan_xys: List[Tuple[np.ndarray, np.ndarray]] = []
        act_xys: List[Tuple[np.ndarray, np.ndarray]] = []

        # Back-face culling: only project if outward normal faces the camera
        for origin, out_n in planned_snapped:
            if float(np.dot(out_n, cam)) >= 0.05:
                xs_r, ys_r = _project_disc_to_2d(origin, out_n, DISC_RADIUS_MM, view_name)
                plan_xys.append((xs_r, ys_r))

        for origin, out_n in actual_snapped:
            if float(np.dot(out_n, cam)) >= 0.05:
                xs_r, ys_r = _project_disc_to_2d(origin, out_n, DISC_RADIUS_MM, view_name)
                act_xys.append((xs_r, ys_r))

        # Layer 1: render planned yellow discs
        for xs_r, ys_r in plan_xys:
            ax.fill(xs_r, ys_r, color=PLANNED_COL, alpha=0.90, zorder=3, linewidth=0)

        # Layer 2: render actual blue medoid discs
        for xs_r, ys_r in act_xys:
            ax.fill(xs_r, ys_r, color=ACTUAL_COL, alpha=0.85, zorder=4, linewidth=0)

        # Layer 3: render exact rasterized green overlap zone
        pad = max(rng * 0.08, DISC_RADIUS_MM + 5.0)
        if plan_xys and act_xys:
            x_min, x_max = xs2d.min() - pad, xs2d.max() + pad
            y_min, y_max = ys2d.min() - pad, ys2d.max() + pad

            nx, ny = 900, 900
            gx = np.linspace(x_min, x_max, nx)
            gy = np.linspace(y_min, y_max, ny)
            xx, yy = np.meshgrid(gx, gy)
            grid_points = np.column_stack([xx.ravel(), yy.ravel()])

            planned_mask = np.zeros(grid_points.shape[0], dtype=bool)
            actual_mask = np.zeros(grid_points.shape[0], dtype=bool)

            for xs_r, ys_r in plan_xys:
                poly = np.column_stack([xs_r, ys_r])
                planned_mask |= _MplPath(poly).contains_points(grid_points)

            for xs_r, ys_r in act_xys:
                poly = np.column_stack([xs_r, ys_r])
                actual_mask |= _MplPath(poly).contains_points(grid_points)

            overlap_mask = (planned_mask & actual_mask).reshape(ny, nx)

            if np.any(overlap_mask):
                rgba = np.zeros((ny, nx, 4), dtype=float)
                rgba[overlap_mask] = [
                    int(OVERLAP_COL[1:3], 16) / 255,
                    int(OVERLAP_COL[3:5], 16) / 255,
                    int(OVERLAP_COL[5:7], 16) / 255,
                    1.0,
                ]
                ax.imshow(rgba, origin="lower", extent=[x_min, x_max, y_min, y_max],
                          interpolation="nearest", zorder=20)

        # Set axes viewport window tightly around skull
        ax.set_xlim(xs2d.min() - pad, xs2d.max() + pad)
        ax.set_ylim(ys2d.min() - pad, ys2d.max() + pad)

    # ── Legend display ────────────────────────────────────────────────────────
    legend_handles = [
        _Patch(facecolor=PLANNED_COL, edgecolor="none", label="Planned (Yellow)"),
        _Patch(facecolor=ACTUAL_COL,  edgecolor="none", label="Actual Medoid (Blue)"),
        _Patch(facecolor=OVERLAP_COL, edgecolor="none", label="Overlap (Green)"),
    ]
    fig.legend(handles=legend_handles, loc="center left",
               bbox_to_anchor=(0.88, 0.50), fontsize=9, frameon=True,
               title="Transducer (62mm)", title_fontsize=9)

    fig.suptitle(f"{subject} | Head mesh planned vs actual medoid positions",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 0.87, 0.97])
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"[saved]  {out_path}")


# ── main block ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_DIR = os.path.join(BASE, "data", "input")
    MEDOID_DIR = os.path.join(BASE, "data", "gum", "medoid")
    SIMNIBS_DIR = os.path.join(BASE, "data", "simnibs")
    OUTPUT_DIR = os.path.join(BASE, "derivatives", "planned_vs_actual_positions")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load planned position indices from CSV
    csv_path = os.path.join(INPUT_DIR, "planned_positions_index.csv")
    if not os.path.exists(csv_path):
        log.error(f"planned_positions_index.csv file not found at: {csv_path}")
        sys.exit(1)
        
    log.info(f"Loading planned position indices from {csv_path}")
    df_indices = pd.read_csv(csv_path)
    df_indices.columns = [c.strip() for c in df_indices.columns]
    
    SUBJECTS = ["sub-03", "sub-04", "sub-05", "sub-06", "sub-11"]
    
    log.info("Starting planned vs actual transducer positions mesh plotting pipeline...")
    
    for sub in SUBJECTS:
        log.info(f"Processing {sub} ...")
        
        # 1. Locate GMSH scalp head mesh file
        mesh_path = Path(os.path.join(SIMNIBS_DIR, sub, f"{sub}.msh"))
        if not mesh_path.exists():
            log.error(f"SimNIBS scalp head mesh file not found for {sub}: {mesh_path}")
            continue
            
        # 2. Get planned transducer indices from CSV
        sub_row = df_indices[df_indices["Subject"] == sub]
        if sub_row.empty:
            log.error(f"No planned indices row found for {sub} in CSV.")
            continue
            
        idx_left_plan = int(sub_row["index_left"].values[0])
        idx_right_plan = int(sub_row["index_right"].values[0])
        
        # 3. Locate and load planned XML positions
        planned_xml_pattern = os.path.join(INPUT_DIR, sub, f"{sub}_GUMMarkers*.xml")
        planned_xml_files = glob.glob(planned_xml_pattern)
        if not planned_xml_files:
            log.error(f"Missing planned GUMMarkers XML in input for {sub} using pattern: {planned_xml_pattern}")
            continue
            
        plan_xml_path = Path(planned_xml_files[0])
        try:
            planned_markers = parse_gummarkers(plan_xml_path)
        except Exception as e:
            log.exception(f"Failed to parse planned XML {plan_xml_path}: {e}")
            continue
            
        plan_left_tx = next((elem for elem in planned_markers if elem.index == idx_left_plan), None)
        plan_right_tx = next((elem for elem in planned_markers if elem.index == idx_right_plan), None)
        
        if plan_left_tx is None or plan_right_tx is None:
            log.error(f"Could not find planned Left (index {idx_left_plan}) or Right (index {idx_right_plan}) markers in {plan_xml_path.name}")
            continue
            
        planned_txs = [plan_left_tx, plan_right_tx]
        
        # 4. Locate and load actual medoid XML positions
        medoid_pattern = os.path.join(MEDOID_DIR, f"{sub}*.xml")
        medoid_files = glob.glob(medoid_pattern)
        if not medoid_files:
            log.error(f"Missing actual medoid XML in medoid folder for {sub} using pattern: {medoid_pattern}")
            continue
            
        med_xml_path = Path(medoid_files[0])
        
        try:
            actual_markers = parse_gummarkers(med_xml_path)
        except Exception as e:
            log.exception(f"Failed to parse actual medoid XML {med_xml_path}: {e}")
            continue
            
        # Robust coordinate-based hemisphere resolution (Left has RAS X < 120.0 mm, Right has RAS X >= 120.0 mm)
        act_left_tx = next((elem for elem in actual_markers if elem.center[0] < 120.0), None)
        act_right_tx = next((elem for elem in actual_markers if elem.center[0] >= 120.0), None)
        
        if act_left_tx is None or act_right_tx is None:
            log.error(f"Could not resolve Left or Right actual medoid markers by coordinate in {med_xml_path.name}")
            continue
            
        log.info(f"  [{sub}] Planned: Left={idx_left_plan} (center: {plan_left_tx.center.round(1)}), Right={idx_right_plan} (center: {plan_right_tx.center.round(1)})")
        log.info(f"  [{sub}] Actual: Left={act_left_tx.index} (center: {act_left_tx.center.round(1)}), Right={act_right_tx.index} (center: {act_right_tx.center.round(1)})")
        
        actual_txs = [act_left_tx, act_right_tx]
        
        # 5. Render and save comparative planned vs actual medoid head mesh figure
        try:
            out_img_path = Path(os.path.join(OUTPUT_DIR, f"{sub}_planned_vs_actual_positions.png"))
            plot_planned_vs_actual_mesh(sub, mesh_path, planned_txs, actual_txs, out_img_path)
        except Exception as e:
            log.exception(f"Failed to generate planned vs actual positions mesh plot for {sub}: {e}")
            
    log.info("Mesh comparative plotting pipeline complete.")
