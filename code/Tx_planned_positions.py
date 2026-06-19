"""
Plot planned transducer positions on the subject head scalp surface mesh.

This script loads the SimNIBS head mesh file (.msh) and the Localite GUMMarkers XML 
to render a high-quality 2-D orthographic projection of the skull with 10 planned 
transducer positions (5 left, 5 right).

@author Hoai Thu Nguyen
"""

import os
import glob
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch as _Patch
from matplotlib.tri import Triangulation as _Tri
import numpy as np

# Set up logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Tx_planned")

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
        xs_r = rim3d[:, 1] * (1.0 if view_name == "right" else -1.0)
        return xs_r, rim3d[:, 2]
    elif view_name in ("front", "back"):
        # Match neurological convention: negate X so patient-left is on viewer's right.
        return -rim3d[:, 0], rim3d[:, 2]
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
        xs = points[:, 1] * (1.0 if view_name == "right" else -1.0)
    elif view_name in ("front", "back"):
        # Negate X so the image matches neurological convention:
        # patient's Left appears on the viewer's right side of the frame.
        xs, ys = -points[:, 0], points[:, 2]
        depth  = points[:, 1] * (1.0 if view_name == "front" else -1.0)
    else:
        xs, ys = points[:, 0], points[:, 1]
        depth  = points[:, 2] * (1.0 if view_name == "top" else -1.0)
        
    order = np.argsort(depth[tris].mean(axis=1))
    return xs, ys, depth, tris[order]


# ── plotting function ────────────────────────────────────────────────────────

def plot_mesh_planned_all_positions(subject: str,
                                    mesh_path: Path,
                                    planned_txs: List[TxMatrix],
                                    out_path: Path) -> None:
    """
    Generate and save a 4-view orthographic 2-D scalp projection showing planned positions.

    Args:
        subject (str): Subject identifier.
        mesh_path (Path): Path to SimNIBS .msh file.
        planned_txs (List[TxMatrix]): List of all loaded planned transducer positions.
        out_path (Path): Output filename path to save PNG figure.
    """
    log.info(f"[{subject}] Projecting and drawing head mesh + transducer positions...")
    points, tris, mid, rng = load_mesh(mesh_path)

    # 4 distinct standard projections
    four_views = [
        ("Left / lateral",  "left"),
        ("Right / lateral", "right"),
        ("Top",             "top"),
        ("Front",           "front"),
    ]

    # Filter to target named planned Tx positions (L-pos1..5, R-pos1..5)
    selected = [tx for tx in planned_txs
                if "Tx" in (tx.description or "") and "pos" in (tx.description or "")]
    if not selected:
        selected = planned_txs
    selected = sorted(selected, key=lambda t: t.description)

    # Coherent color palette (tab10/tab20 styled hex colors for high distinction)
    POSITION_COLORS = {
        "L_1": "#e6194b",  # Red
        "L_2": "#3cb44b",  # Lime
        "L_3": "#4363d8",  # Orange
        "L_4": "#f58231",  # Gray
        "L_5": "#911eb4",  # Pink
        "R_1": "#42d4f4",  # Cyan
        "R_2": "#f032e6",  # Magenta
        "R_3": "#bfef45",  # Green
        "R_4": "#fabed4",  # Soft Rose
        "R_5": "#a9a9a9",  # Blue
    }
    
    import re
    color_map: Dict[int, np.ndarray] = {}
    for tx in selected:
        match = re.match(r"Tx-2_([LR])_pos-(\d+)", tx.description or "")
        label_name = f"{match.group(1)}_{match.group(2)}" if match else tx.description
        hex_col = POSITION_COLORS.get(label_name, "#888888")
        r = int(hex_col[1:3], 16) / 255
        g = int(hex_col[3:5], 16) / 255
        b = int(hex_col[5:7], 16) / 255
        color_map[tx.index] = np.array([r, g, b, 1.0])

    # Pre-compute snapped centers and surface normal unit vectors
    snapped: Dict[int, np.ndarray] = {}
    normals: Dict[int, np.ndarray] = {}
    for tx in selected:
        sp = _snap_tx_to_surface(tx.center, points)
        snapped[tx.index] = sp
        normals[tx.index] = _outward_normal(sp, mid)

    DISC_RADIUS_MM = 31.0  # ~62 mm diameter transducers

    # Set up matplotlib figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), facecolor="white")

    for ax, (view_title, view_name) in zip(axes, four_views):
        ax.set_facecolor("#f8f9fa")
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(view_title, fontsize=18, pad=6)
        if view_name == "top":
            ax.text(0.05, 0.90, "L", color="black", fontsize=16.0, fontweight="bold",
                    transform=ax.transAxes, va="center", ha="left")
            ax.text(0.95, 0.90, "R", color="black", fontsize=16.0, fontweight="bold",
                    transform=ax.transAxes, va="center", ha="right")
        elif view_name == "front":
            # Neurological convention: patient-left on viewer's right
            ax.text(0.05, 0.90, "R", color="black", fontsize=16.0, fontweight="bold",
                    transform=ax.transAxes, va="center", ha="left")
            ax.text(0.95, 0.90, "L", color="black", fontsize=16.0, fontweight="bold",
                    transform=ax.transAxes, va="center", ha="right")

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

        # ── Transducer discs rendering (with back-face culling) ──────────
        for tx in selected:
            sp    = snapped[tx.index]
            out_n = normals[tx.index]
            col   = color_map.get(tx.index, np.array([0.5, 0.5, 0.5, 1.0]))

            # Back-face culling: only render if outward normal faces the camera
            if float(np.dot(out_n, cam)) < 0.05:
                continue

            xs_r, ys_r = _project_disc_to_2d(sp, out_n, DISC_RADIUS_MM, view_name)
            # Render slightly transparent to show overlaps
            ax.fill(xs_r, ys_r, color=col, alpha=0.75, zorder=4, linewidth=0)

        # Set axes viewport window tightly around skull
        pad = rng * 0.08
        ax.set_xlim(xs2d.min() - pad, xs2d.max() + pad)
        ax.set_ylim(ys2d.min() - pad, ys2d.max() + pad)

    # ── Legend display ────────────────────────────────────────────────────────
    legend_handles = []
    for tx in selected:
        col = color_map.get(tx.index, np.array([0.5, 0.5, 0.5, 1.0]))
        match = re.match(r"Tx-2_([LR])_pos-(\d+)", tx.description or "")
        label_name = f"{match.group(1)}_{match.group(2)}" if match else tx.description
        legend_handles.append(_Patch(facecolor=col, edgecolor="none",
                                     label=label_name))
    if legend_handles:
        fig.legend(handles=legend_handles, loc="center left",
                   bbox_to_anchor=(0.88, 0.50), fontsize=16.0, frameon=True,
                   title="Planned positions", title_fontsize=16.0)

    #fig.suptitle(f"{subject} | Head mesh with all planned transducer positions",
                # fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 0.87, 0.97])
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"[saved]  {out_path}")


# ── main block ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Setup directory paths relative to root CITRUS folder
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_DIR   = os.path.join(BASE, "data", "input")
    SIMNIBS_DIR = os.path.join(BASE, "data", "simnibs")
    OUTPUT_DIR  = os.path.join(BASE, "derivatives", "planned_positions")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    SUBJECTS = ["sub-03", "sub-04", "sub-05", "sub-06", "sub-11"]
    
    log.info("Starting planned transducer position mesh plotting pipeline...")
    for sub in SUBJECTS:
        log.info(f"Processing {sub} ...")
        
        # 1. Locate GMSH head mesh .msh file
        mesh_path = Path(os.path.join(SIMNIBS_DIR, sub, f"{sub}.msh"))
        if not mesh_path.exists():
            log.error(f"SimNIBS head mesh file not found for {sub}: {mesh_path}")
            continue
            
        # 2. Find planned markers GUMMarkers XML file inside input folder
        sub_input_dir = os.path.join(INPUT_DIR, sub)
        xml_pattern = os.path.join(sub_input_dir, f"{sub}_GUMMarkers*.xml")
        xml_files = glob.glob(xml_pattern)
        
        if not xml_files:
            log.error(f"GUMMarkers XML planned file not found in {sub_input_dir} using pattern f'{sub}_GUMMarkers*.xml'")
            continue
        
        xml_path = Path(xml_files[0])
        log.info(f"Found GUMMarkers XML: {xml_path.name}")
        
        # 3. Parse XML and run projection plot
        try:
            planned_txs = parse_gummarkers(xml_path)
            out_img_path = Path(os.path.join(OUTPUT_DIR, f"{sub}_planned_positions.png"))
            plot_mesh_planned_all_positions(sub, mesh_path, planned_txs, out_img_path)
        except Exception as e:
            log.exception(f"Failed to generate planned positions plot for {sub}: {e}")
            
    log.info("Mesh plotting pipeline complete.")
