"""
Plot actual recorded transducer position density heatmaps on the subject head scalp surface mesh.

This script parses the participant ratings CSV and subjects' session experimental (ses-exp) and
control (ses-con) Localite GUMMarkers XML files. It extracts the actual recorded transducer 
frame centers for Left and Right sides based on start/end XML indices, projects their spatial
density onto the scalp mesh, and renders 4-view orthographic 3D projection plots.

@author Hoai Thu Nguyen
"""

import os
import sys
import glob
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd

# Set up logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Tx_actual")

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


def set_mesh_view(ax, view: str):
    """
    Set camera elevation and azimuth angles to standard projection directions.

    Args:
        ax (Axes3D): Matplotlib 3D axes object.
        view (str): Viewing camera angle (left, right, front, top).
    """
    views = {
        "left": (0, 180),
        "right": (0, 0),
        "front": (0, 90),
        "top": (90, -90),
    }
    elev, azim = views.get(view, views["left"])
    ax.view_init(elev=elev, azim=azim)


# ── plotting function ────────────────────────────────────────────────────────

def plot_mesh_heatmap(subject: str,
                      mesh_path: Path,
                      actual_points: List[np.ndarray],
                      condition: str,
                      out_path: Path) -> None:
    """
    Calculate spatial point density using a Gaussian kernel and render a 4-view 3D projection.

    Args:
        subject (str): Subject identifier.
        mesh_path (Path): Path to SimNIBS .msh file.
        actual_points (List[np.ndarray]): List of all 3-D actual positions.
        condition (str): Condition identifier (EXP or CON).
        out_path (Path): Output filename path to save PNG figure.
    """
    log.info(f"[{subject} | {condition}] Rendering scalp mesh heatmap with {len(actual_points)} coordinates...")
    points, tris, mid, rng = load_mesh(mesh_path)

    # 1. Calculate spatial density on scalp triangle centroids using Gaussian kernel (sigma = 15.0 mm)
    tri_centroids = points[tris].mean(axis=1)
    if actual_points:
        centers_arr = np.array(actual_points, dtype=float)
        diff = tri_centroids[:, None, :] - centers_arr[None, :, :]
        sq_dist = np.sum(diff ** 2, axis=2)
        density = np.exp(-sq_dist / (15.0 ** 2)).sum(axis=1)
        # Min-max normalization for colormap mapping
        density = (density - density.min()) / (density.max() - density.min() + 1e-9)
    else:
        density = np.zeros(len(tris), dtype=float)

    # Map density values to hot colormap
    hot_rgba = plt.cm.hot(density)
    tri_verts = points[tris]

    # 4 distinct standard projections
    four_views = [
        ("Left / lateral",  "left"),
        ("Right / lateral", "right"),
        ("Front",           "front"),
        ("Top",             "top"),
    ]

    # Set up matplotlib figure
    fig = plt.figure(figsize=(20, 5.5), facecolor="white")

    for i, (view_title, view_name) in enumerate(four_views):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        ax.set_facecolor("white")

        # Set up standard 3D triangulation polygon collection with back-face depth sorting
        coll = Poly3DCollection(tri_verts, zsort="min")
        coll.set_facecolor(hot_rgba)
        coll.set_edgecolor("none")
        coll.set_alpha(1.0)
        ax.add_collection3d(coll)

        # Set equal bounding volume bounds
        ax.set_xlim(mid[0] - rng, mid[0] + rng)
        ax.set_ylim(mid[1] - rng, mid[1] + rng)
        ax.set_zlim(mid[2] - rng, mid[2] + rng)

        set_mesh_view(ax, view_name)
        ax.set_axis_off()
        ax.set_title(view_title, color="black", fontsize=11, fontweight="bold")

    # Custom colorbar for density representation
    sm = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap="hot")
    sm.set_array([])
    cbar_ax = fig.add_axes([0.905, 0.18, 0.012, 0.60])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Position density\n(brighter = more frequent)", fontsize=8, color="black")
    cb.ax.tick_params(labelsize=7, colors="black")
    cb.ax.yaxis.set_tick_params(color="black")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="black")

    cond_label = "Experimental" if condition.upper() == "EXP" else "Control"
    fig.suptitle(
        f"{subject} | {cond_label} | "
        f"Actual transducer position heatmap (all recorded frames)",
        fontsize=13, fontweight="bold", y=1.02, color="black",
    )
    fig.tight_layout(rect=[0, 0, 0.90, 0.97])
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"[saved]  {out_path}")


# ── main block ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Setup directory paths relative to root CITRUS folder
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    GUM_DIR = os.path.join(BASE, "data", "gum")
    SIMNIBS_DIR = os.path.join(BASE, "data", "simnibs")
    OUTPUT_DIR = os.path.join(BASE, "derivatives", "actual_positions")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Parse participant ratings CSV containing start/end frame indices
    csv_path = os.path.join(GUM_DIR, "citrus-offline_participant_ratings - ratings.csv")
    if not os.path.exists(csv_path):
        log.error(f"Participant ratings CSV file not found at: {csv_path}")
        sys.exit(1)
        
    log.info(f"Loading participant ratings CSV from {csv_path}")
    df_ratings = pd.read_csv(csv_path)
    
    # Standardize column headers and unconditionally name the first column as 'subject'
    df_ratings.columns = [c.strip() for c in df_ratings.columns]
    df_ratings.rename(columns={df_ratings.columns[0]: "subject"}, inplace=True)
        
    # Group ratings by subject and condition
    grouped = df_ratings.groupby(["subject", "condition"])

    log.info("Starting actual transducer position heatmap plotting pipeline...")
    
    for (subject, condition), group_df in grouped:
        log.info(f"Processing {subject} | {condition} ...")
        
        # Locate GMSH scalp head mesh file
        mesh_path = Path(os.path.join(SIMNIBS_DIR, subject, f"{subject}.msh"))
        if not mesh_path.exists():
            log.error(f"SimNIBS scalp head mesh file not found for {subject}: {mesh_path}")
            continue
            
        actual_points = []
        parsed_cache = {}
        
        # Accumulate coordinates for Left and Right hemispheres from the GUMMarkers XML
        for _, row in group_df.iterrows():
            localite_file = str(row["localite_file"]).strip()
            if not localite_file.lower().endswith(".xml"):
                localite_file += ".xml"
                
            xml_path = Path(os.path.join(GUM_DIR, subject, localite_file))
            if not xml_path.exists():
                log.error(f"XML actual markers file not found: {xml_path}")
                continue
                
            # Parse XML if not already cached
            if xml_path not in parsed_cache:
                try:
                    log.info(f"Parsing GUMMarkers XML: {xml_path.name}")
                    parsed_cache[xml_path] = parse_gummarkers(xml_path)
                except Exception as e:
                    log.exception(f"Failed to parse XML actual markers file {xml_path}: {e}")
                    continue
            
            parsed_markers = parsed_cache[xml_path]
            
            # Extract start and end frame indices
            xml_start = int(row["xml_start"])
            xml_end = int(row["xml_end"])
            hemisphere = row["hemisphere"]
            
            # Filter matrices by inclusive index range
            hem_points = [elem.center for elem in parsed_markers if xml_start <= elem.index <= xml_end]
            log.info(f"  Hemisphere {hemisphere}: Extracted {len(hem_points)} coordinates in index range [{xml_start}, {xml_end}]")
            actual_points.extend(hem_points)
            
        # Draw and save heatmap
        if not actual_points:
            log.warning(f"No coordinates found for {subject} | {condition}, skipping heatmap rendering.")
            continue
            
        try:
            out_img_path = Path(os.path.join(OUTPUT_DIR, f"{subject}_actual_positions_{condition.lower()}.png"))
            plot_mesh_heatmap(subject, mesh_path, actual_points, condition, out_img_path)
        except Exception as e:
            log.exception(f"Failed to generate actual positions heatmap for {subject} | {condition}: {e}")
            
    log.info("Mesh actual positions heatmap plotting pipeline complete.")
