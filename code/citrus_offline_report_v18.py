#!/usr/bin/env python3
"""
citrus_offline_report.py
========================

One-subject, bash-runnable report generator for CITRUS offline acoustic
simulation and post-hoc planned-vs-actual transducer analysis.

Inputs are direct file paths. No fixed folder tree or citrus_config.py is needed.

Main outputs:
  - report-ready PNG figures
  - CSV metric tables
  - one HTML report collecting the figures/tables

The plotting style follows the CITRUS report examples:
  - T1w grayscale anatomy
  - sgACC filled red in anatomy-only figure
  - sgACC white contour in pressure/temperature/focal figures
  - shared subject-level color ranges for pressure and temperature
  - planned vs actual -3 dB focal contours
  - head mesh with planned and actual transducer positions

Requirements:
  pip install nibabel numpy matplotlib scipy pandas meshio

Optional but useful:
  pip install pyvista

Author: generated for CITRUS offline protocol report automation.
"""

from __future__ import annotations

import argparse
import base64
import html
import logging
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import map_coordinates, label as nd_label, center_of_mass

try:
    import meshio
    HAS_MESHIO = True
except Exception:
    meshio = None
    HAS_MESHIO = False

try:
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union as shapely_union
    HAS_SHAPELY = True
except Exception:
    HAS_SHAPELY = False


# ------------------------- logging and constants -------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("citrus_offline_report")

SIDES = ["left", "right"]
CONDS = ["exp", "con"]
SIDE_TITLE = {"left": "Left", "right": "Right"}
COND_TITLE = {"exp": "Experimental", "con": "Control"}

PRESSURE_CMAP = "turbo"
TEMP_CMAP = "hot"
BG = "#0b0b12"
WHITE = "#ffffff"
RED = "#ff3030"
CYAN = "#00e5ff"
PLANNED = "#ff6a00"  # orange-red  – planned focal zone
ACTUAL  = "#00d4ff"  # cyan        – actual focal zone
GOLD = "#ffd700"
GRAY = "#bbbbbb"
GREEN = "#22c55e"
PURPLE = "#b66dff"

MINUS3_AMP = 10 ** (-3.0 / 20.0)  # 0.7079 for pressure amplitude
SCRIPT_VERSION = "v5-2026-05-30-pressure-skull-exclusion-brain-focal"


# ------------------------- data structures -------------------------

@dataclass
class TxMatrix:
    index: int
    description: str
    matrix: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return self.matrix[:3, 3].astype(float)


@dataclass
class FocalMetrics:
    subject: str
    condition: str
    side: str
    planned_peak_mpa: float
    actual_peak_mpa: float
    planned_threshold_mpa: float
    actual_threshold_mpa: float
    planned_focal_voxels: int
    actual_focal_voxels: int
    planned_focal_volume_mm3: float
    actual_focal_volume_mm3: float
    dice_planned_actual: float
    actual_percent_inside_planned: float
    planned_percent_inside_actual: float
    actual_percent_inside_sgacc: float
    sgacc_percent_covered_by_actual: float
    planned_center_to_sgacc_mm: float
    actual_center_to_sgacc_mm: float
    planned_fwhm_major_mm: float
    planned_fwhm_middle_mm: float
    planned_fwhm_minor_mm: float
    actual_fwhm_major_mm: float
    actual_fwhm_middle_mm: float
    actual_fwhm_minor_mm: float


# ------------------------- utility helpers -------------------------

def ensure_path(path: Optional[str], label: str, required: bool = True) -> Optional[Path]:
    if path is None or str(path).strip() == "":
        if required:
            raise FileNotFoundError(f"Missing required argument: {label}")
        return None
    p = Path(path).expanduser()
    if not p.exists():
        if required:
            raise FileNotFoundError(f"File not found for {label}: {p}")
        log.warning("Optional file not found for %s: %s", label, p)
        return None
    return p


def load_img(path: Path) -> nib.Nifti1Image:
    return nib.load(str(path))


def img_data(img: nib.Nifti1Image, dtype=np.float32) -> np.ndarray:
    return img.get_fdata(dtype=dtype)


def voxel_sizes(img: nib.Nifti1Image) -> np.ndarray:
    return np.array(img.header.get_zooms()[:3], dtype=float)


def voxel_volume_mm3(img: nib.Nifti1Image) -> float:
    return float(abs(np.linalg.det(img.affine[:3, :3])))


def voxel_to_world(vox_xyz: Sequence[float], affine: np.ndarray) -> np.ndarray:
    return (affine @ np.array([vox_xyz[0], vox_xyz[1], vox_xyz[2], 1.0], dtype=float))[:3]


def world_to_voxel(world_xyz: Sequence[float], affine: np.ndarray) -> np.ndarray:
    return (np.linalg.inv(affine) @ np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1.0], dtype=float))[:3]


def resample_to_target(src_img: nib.Nifti1Image, tgt_img: nib.Nifti1Image, order: int = 1) -> np.ndarray:
    """Resample src image to target voxel grid using scipy.map_coordinates."""
    src = img_data(src_img, dtype=np.float32)
    tgt_shape = tgt_img.shape[:3]
    ii, jj, kk = np.meshgrid(
        np.arange(tgt_shape[0]),
        np.arange(tgt_shape[1]),
        np.arange(tgt_shape[2]),
        indexing="ij",
    )
    tgt_vox = np.stack([ii, jj, kk, np.ones_like(ii)], axis=-1).reshape(-1, 4)
    world = (tgt_img.affine @ tgt_vox.T).T
    src_vox = (np.linalg.inv(src_img.affine) @ world.T).T[:, :3]
    out = map_coordinates(
        src,
        [src_vox[:, 0], src_vox[:, 1], src_vox[:, 2]],
        order=order,
        mode="constant",
        cval=0.0,
    )
    return out.reshape(tgt_shape).astype(np.float32)


def percentile_nonzero(arrays: Iterable[np.ndarray], q: float, default: float = 1.0) -> float:
    vals = []
    for a in arrays:
        x = np.asarray(a, dtype=float)
        x = x[np.isfinite(x)]
        x = x[x > 0]
        if x.size:
            vals.append(x.ravel())
    if not vals:
        return default
    cat = np.concatenate(vals)
    if cat.size == 0:
        return default
    return float(np.percentile(cat, q))


def robust_t1_limits(t1: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(t1, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return 0.0, 1.0
    positive = finite[finite > np.percentile(finite, 5)]
    if positive.size < 100:
        positive = finite
    return tuple(float(v) for v in np.percentile(positive, [1, 99]))


def mask_centroid_vox(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    coords = np.argwhere(mask > threshold)
    if coords.size == 0:
        raise ValueError("Mask has no voxels above threshold")
    return coords.mean(axis=0)


def slice2d(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    idx = int(np.clip(idx, 0, vol.shape[axis] - 1))
    if axis == 2:      # axial z: x/y plane
        return vol[:, :, idx].T
    if axis == 1:      # coronal y: x/z plane
        return vol[:, idx, :].T
    if axis == 0:      # sagittal x: y/z plane
        return vol[idx, :, :].T
    raise ValueError("axis must be 0, 1, or 2")


def point_xy(vox_xyz: Sequence[float], axis: int) -> Tuple[float, float]:
    x, y, z = vox_xyz
    if axis == 2:
        return float(x), float(y)
    if axis == 1:
        return float(x), float(z)
    if axis == 0:
        return float(y), float(z)
    raise ValueError("axis must be 0, 1, or 2")


def crop_limits(shape2d: Tuple[int, int], center_xy: Tuple[float, float], half_width_vox: int) -> Optional[Tuple[int, int, int, int]]:
    if half_width_vox <= 0:
        return None
    cx, cy = center_xy
    x0 = max(0, int(round(cx)) - half_width_vox)
    x1 = min(shape2d[1], int(round(cx)) + half_width_vox)
    y0 = max(0, int(round(cy)) - half_width_vox)
    y1 = min(shape2d[0], int(round(cy)) + half_width_vox)
    return x0, x1, y0, y1


def apply_crop(arr2d: np.ndarray, lim: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if lim is None:
        return arr2d
    x0, x1, y0, y1 = lim
    return arr2d[y0:y1, x0:x1]


def adjust_xy(xy: Tuple[float, float], lim: Optional[Tuple[int, int, int, int]]) -> Tuple[float, float]:
    if lim is None:
        return xy
    x0, _x1, y0, _y1 = lim
    return xy[0] - x0, xy[1] - y0


def safe_contour(ax, arr2d: np.ndarray, level: float, color: str, linestyle: str = "solid", linewidth: float = 1.5, alpha: float = 1.0):
    a = np.asarray(arr2d, dtype=float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return None
    if not (float(np.nanmin(finite)) <= level <= float(np.nanmax(finite))):
        return None
    return ax.contour(a, levels=[level], colors=[color], linestyles=linestyle, linewidths=linewidth, alpha=alpha)


def safe_imshow_overlay(ax, arr2d: np.ndarray, vmin: float, vmax: float, cmap: str, alpha: float = 0.55):
    arr = np.asarray(arr2d, dtype=float)
    masked = np.ma.masked_where(~np.isfinite(arr) | (arr <= vmin), arr)
    if masked.count() == 0:
        return None
    return ax.imshow(masked, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, interpolation="nearest")



def pressure_to_mpa(arr: np.ndarray) -> np.ndarray:
    """Return pressure in MPa.

    k-Plan exports may be in MPa already or in Pa. If the robust maximum is
    very large (>1000), treat the values as Pa and convert to MPa.
    """
    x = np.asarray(arr, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if finite.size and float(np.nanpercentile(np.abs(finite), 99.9)) > 1000.0:
        return (x / 1_000_000.0).astype(np.float32)
    return x.astype(np.float32)


def load_pressure_mpa(path: Path) -> np.ndarray:
    return pressure_to_mpa(img_data(load_img(path)))


def brain_crop_limits_2d(arr2d: np.ndarray, pad: int = 10) -> Optional[Tuple[int, int, int, int]]:
    """Crop a 2D anatomy-like slice around non-air tissue."""
    a = np.asarray(arr2d, dtype=float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return None
    thr = np.percentile(finite, 8)
    mask = a > thr
    if int(mask.sum()) < 25:
        return None
    ys, xs = np.where(mask)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(a.shape[1], int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(a.shape[0], int(ys.max()) + pad + 1)
    return x0, x1, y0, y1


def add_lr_labels(ax, axis: int, color: str = "white"):
    """Add simple radiological/anatomical direction labels on axial/coronal slices."""
    if axis in (1, 2):
        ax.text(0.02, 0.50, "L", transform=ax.transAxes, color=color, fontsize=11,
                fontweight="bold", va="center", ha="left",
                bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1.5))
        ax.text(0.98, 0.50, "R", transform=ax.transAxes, color=color, fontsize=11,
                fontweight="bold", va="center", ha="right",
                bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1.5))


# ------------------------- XML / transducer functions -------------------------

def lps_to_ras_matrix() -> np.ndarray:
    return np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)


def parse_gummarkers(xml_path: Path, convert_lps_to_ras: bool = True) -> List[TxMatrix]:
    """Parse Localite GUMMarkers XML into matrices in RAS world space.

    If coordinateSpace="LPS", matrices are converted to RAS by pre-multiplying
    the LPS->RAS flip. If coordinateSpace="RAS", matrices are used as-is.
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


def select_tx_by_index_or_label(txs: List[TxMatrix], index: Optional[int], label: Optional[str], side_label: str) -> TxMatrix:
    if label:
        matches = [t for t in txs if label in t.description]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            desc = ", ".join(f"{t.index}:{t.description}" for t in matches[:10])
            raise ValueError(f"Multiple planned markers match {side_label} label {label!r}: {desc}")
        raise ValueError(f"No planned marker matches {side_label} label {label!r}")
    if index is not None:
        matches = [t for t in txs if t.index == index]
        if len(matches) == 1:
            return matches[0]
        available = ", ".join(str(t.index) for t in txs[:50])
        raise ValueError(f"Planned marker index {index} not found for {side_label}. Available starts: {available}")
    raise ValueError(f"Provide either planned {side_label} label or index")


def select_range(txs: List[TxMatrix], frame_range: Optional[Sequence[int]], label_filter: Optional[str] = None) -> List[TxMatrix]:
    selected = txs
    if frame_range:
        if len(frame_range) != 2:
            raise ValueError("Frame range must have START END")
        lo, hi = int(frame_range[0]), int(frame_range[1])
        if hi < lo:
            raise ValueError("Frame range END must be >= START")
        selected = [t for t in selected if lo <= t.index <= hi]
    if label_filter:
        selected = [t for t in selected if label_filter in t.description]
    if not selected:
        log.warning("No actual frames found for range=%s, label_filter=%r. Returning dummy RAS frame.", frame_range, label_filter)
        return [TxMatrix(index=frame_range[0] if frame_range else 0, description="No Data Dummy", matrix=np.eye(4))]
    return selected


def rotation_angle_deg(R: np.ndarray) -> float:
    val = (np.trace(R) - 1.0) / 2.0
    val = float(np.clip(val, -1.0, 1.0))
    return math.degrees(math.acos(val))


def rotation_vector_xyz_deg(R: np.ndarray) -> np.ndarray:
    """Small-angle rotation vector approximation from rotation matrix, degrees."""
    angle = math.acos(float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    if abs(angle) < 1e-12:
        return np.zeros(3)
    denom = 2.0 * math.sin(angle)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / denom
    return axis * math.degrees(angle)


def all_points_medoid(txs: List[TxMatrix]) -> TxMatrix:
    centers = np.array([t.center for t in txs], dtype=float)
    if len(txs) == 1:
        return txs[0]
    # Pairwise Euclidean distances. The medoid minimizes total distance to all other points.
    diffs = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diffs, axis=2)
    med_idx = int(np.argmin(d.sum(axis=1)))
    return txs[med_idx]


def drift_dataframe(subject: str, condition: str, side: str, txs: List[TxMatrix], planned_tx: Optional[TxMatrix], medoid_tx: TxMatrix) -> pd.DataFrame:
    ref = txs[0]
    rows = []
    R0 = ref.matrix[:3, :3]
    for t in txs:
        dxyz = t.center - ref.center
        Rdiff = t.matrix[:3, :3] @ R0.T
        rvec = rotation_vector_xyz_deg(Rdiff)
        planned_dist = float(np.linalg.norm(t.center - planned_tx.center)) if planned_tx is not None else np.nan
        medoid_dist = float(np.linalg.norm(t.center - medoid_tx.center))
        rows.append({
            "subject": subject,
            "condition": condition,
            "side": side,
            "frame_index": t.index,
            "description": t.description,
            "x_mm": t.center[0],
            "y_mm": t.center[1],
            "z_mm": t.center[2],
            "dx_from_first_mm": dxyz[0],
            "dy_from_first_mm": dxyz[1],
            "dz_from_first_mm": dxyz[2],
            "translation_from_first_mm": float(np.linalg.norm(dxyz)),
            "rot_x_from_first_deg": rvec[0],
            "rot_y_from_first_deg": rvec[1],
            "rot_z_from_first_deg": rvec[2],
            "rotation_angle_from_first_deg": rotation_angle_deg(Rdiff),
            "distance_to_planned_mm": planned_dist,
            "distance_to_medoid_mm": medoid_dist,
            "is_medoid": t.index == medoid_tx.index,
        })
    return pd.DataFrame(rows)


def drift_summary(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col in ["dx_from_first_mm", "dy_from_first_mm", "dz_from_first_mm", "translation_from_first_mm",
                "rot_x_from_first_deg", "rot_y_from_first_deg", "rot_z_from_first_deg", "rotation_angle_from_first_deg",
                "distance_to_planned_mm"]:
        if col not in df:
            continue
        x = df[col].astype(float).to_numpy()
        out[f"{col}_mean"] = float(np.nanmean(x))
        out[f"{col}_sd"] = float(np.nanstd(x, ddof=1)) if np.sum(np.isfinite(x)) > 1 else 0.0
        out[f"{col}_max_abs"] = float(np.nanmax(np.abs(x))) if np.any(np.isfinite(x)) else np.nan
    out["n_frames"] = int(len(df))
    return out


# ------------------------- focal analysis -------------------------

def sphere_mask_around_world(ref_img: nib.Nifti1Image, center_world: np.ndarray, radius_mm: float) -> np.ndarray:
    shape = ref_img.shape[:3]
    ijk = np.indices(shape, dtype=float).reshape(3, -1).T
    world = nib.affines.apply_affine(ref_img.affine, ijk)
    dist = np.linalg.norm(world - center_world[None, :], axis=1)
    return (dist <= radius_mm).reshape(shape)


def largest_component_containing_peak(binary: np.ndarray, peak_vox: np.ndarray) -> np.ndarray:
    lab, n = nd_label(binary.astype(bool))
    if n == 0:
        return np.zeros_like(binary, dtype=bool)
    peak_idx = tuple(int(round(v)) for v in peak_vox)
    peak_idx = tuple(np.clip(peak_idx[i], 0, binary.shape[i] - 1) for i in range(3))
    peak_label = int(lab[peak_idx])
    if peak_label > 0:
        return lab == peak_label
    # fallback: largest component
    counts = np.bincount(lab.ravel())
    if len(counts) <= 1:
        return np.zeros_like(binary, dtype=bool)
    counts[0] = 0
    return lab == int(np.argmax(counts))


def focal_volume_from_pressure(pressure: np.ndarray,
                               ref_img: nib.Nifti1Image,
                               sgacc_mask: np.ndarray,
                               sgacc_center_world: np.ndarray,
                               search_radius_mm: float,
                               brain_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    """Return the connected -3 dB focal volume near sgACC.

    Simple, robust pipeline:
    1. Restrict candidates to a sphere of radius search_radius_mm around sgACC centroid.
    2. Within that sphere, apply brain_mask (hard constraint: skull voxels excluded).
    3. Find the peak pressure within brain-masked candidates.
    4. Threshold at peak * MINUS3_AMP (-3 dB for amplitude).
    5. Take the largest connected component containing the peak.

    The binary zone is always bounded by the search sphere AND brain_mask so it
    cannot expand to fill the whole brain regardless of threshold value.
    """
    # ── Step 1: search sphere ────────────────────────────────────────────────────
    search = sphere_mask_around_world(ref_img, sgacc_center_world, search_radius_mm)
    candidate = search & np.isfinite(pressure) & (pressure > 0)

    # ── Step 2: brain mask (hard constraint) ─────────────────────────────────────
    use_brain = brain_mask is not None
    if use_brain:
        c_brain = candidate & brain_mask.astype(bool)
        if np.any(c_brain):
            candidate = c_brain
            log.debug("Brain mask applied: %d candidates in sphere.", int(c_brain.sum()))
        else:
            log.warning("brain_mask has no positive voxels inside the search sphere "
                        "(radius %.0f mm). Falling back to sphere-only candidates. "
                        "Pass --segmentation for better skull exclusion.", search_radius_mm)
            use_brain = False

    if not np.any(candidate):
        raise ValueError("No positive pressure voxels found near sgACC. "
                         "Check --focal-search-radius-mm and sgACC mask alignment.")

    # ── Step 3: find peak within brain-masked search sphere ──────────────────────
    masked_p = np.where(candidate, pressure, -np.inf)
    peak_vox_int = np.array(np.unravel_index(np.argmax(masked_p), pressure.shape), dtype=float)
    peak = float(pressure[tuple(peak_vox_int.astype(int))])
    thr  = peak * MINUS3_AMP
    log.debug("Focal peak: %.4g MPa at vox %s  |  -3dB thr: %.4g MPa", peak, peak_vox_int, thr)

    # ── Step 4: build -3 dB binary, bounded by sphere + brain_mask ───────────────
    binary = (pressure >= thr) & search
    if use_brain and brain_mask is not None:
        binary &= brain_mask.astype(bool)

    # ── Step 5: largest connected component containing the peak ──────────────────
    focal = largest_component_containing_peak(binary, peak_vox_int)
    if not np.any(focal):
        focal = binary   # fallback: all binary voxels (still bounded by sphere)

    com_vox = np.array(center_of_mass(focal.astype(float)), dtype=float)
    if not np.all(np.isfinite(com_vox)):
        com_vox = peak_vox_int.copy()
    log.debug("Focal volume: %d voxels", int(focal.sum()))
    return focal.astype(bool), peak, thr, peak_vox_int, com_vox


def pca_dimensions_mm(binary: np.ndarray, img: nib.Nifti1Image) -> Tuple[float, float, float]:
    coords = np.argwhere(binary)
    if coords.shape[0] < 3:
        return 0.0, 0.0, 0.0
    world = nib.affines.apply_affine(img.affine, coords)
    centered = world - world.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    proj = centered @ vecs
    dims = proj.max(axis=0) - proj.min(axis=0)
    dims = np.sort(dims)[::-1]
    return float(dims[0]), float(dims[1]), float(dims[2])


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    denom = int(a.sum() + b.sum())
    if denom == 0:
        return np.nan
    return float(2 * np.logical_and(a, b).sum() / denom)


def percent(part: int, whole: int) -> float:
    return float(100.0 * part / whole) if whole > 0 else np.nan


def compute_focal_metrics(subject: str,
                          condition: str,
                          side: str,
                          planned_pressure: np.ndarray,
                          actual_pressure: np.ndarray,
                          ref_img: nib.Nifti1Image,
                          sgacc_mask: np.ndarray,
                          sgacc_center_world: np.ndarray,
                          search_radius_mm: float,
                          brain_mask: Optional[np.ndarray]) -> Tuple[FocalMetrics, Dict[str, np.ndarray]]:
    plan_focal, plan_peak, plan_thr, plan_peak_vox, plan_com_vox = focal_volume_from_pressure(
        planned_pressure, ref_img, sgacc_mask, sgacc_center_world, search_radius_mm, brain_mask
    )
    act_focal, act_peak, act_thr, act_peak_vox, act_com_vox = focal_volume_from_pressure(
        actual_pressure, ref_img, sgacc_mask, sgacc_center_world, search_radius_mm, brain_mask
    )
    vvol = voxel_volume_mm3(ref_img)
    inter = np.logical_and(plan_focal, act_focal)
    act_in_sg = np.logical_and(act_focal, sgacc_mask > 0.5)
    sg_voxels = int((sgacc_mask > 0.5).sum())
    plan_com_world = voxel_to_world(plan_com_vox, ref_img.affine)
    act_com_world = voxel_to_world(act_com_vox, ref_img.affine)
    plan_dims = pca_dimensions_mm(plan_focal, ref_img)
    act_dims = pca_dimensions_mm(act_focal, ref_img)
    m = FocalMetrics(
        subject=subject,
        condition=condition,
        side=side,
        planned_peak_mpa=plan_peak,
        actual_peak_mpa=act_peak,
        planned_threshold_mpa=plan_thr,
        actual_threshold_mpa=act_thr,
        planned_focal_voxels=int(plan_focal.sum()),
        actual_focal_voxels=int(act_focal.sum()),
        planned_focal_volume_mm3=float(plan_focal.sum() * vvol),
        actual_focal_volume_mm3=float(act_focal.sum() * vvol),
        dice_planned_actual=dice(plan_focal, act_focal),
        actual_percent_inside_planned=percent(int(inter.sum()), int(act_focal.sum())),
        planned_percent_inside_actual=percent(int(inter.sum()), int(plan_focal.sum())),
        actual_percent_inside_sgacc=percent(int(act_in_sg.sum()), int(act_focal.sum())),
        sgacc_percent_covered_by_actual=percent(int(act_in_sg.sum()), sg_voxels),
        planned_center_to_sgacc_mm=float(np.linalg.norm(plan_com_world - sgacc_center_world)),
        actual_center_to_sgacc_mm=float(np.linalg.norm(act_com_world - sgacc_center_world)),
        planned_fwhm_major_mm=plan_dims[0],
        planned_fwhm_middle_mm=plan_dims[1],
        planned_fwhm_minor_mm=plan_dims[2],
        actual_fwhm_major_mm=act_dims[0],
        actual_fwhm_middle_mm=act_dims[1],
        actual_fwhm_minor_mm=act_dims[2],
    )
    aux = {
        "planned_focal": plan_focal,
        "actual_focal": act_focal,
        "planned_com_vox": plan_com_vox,
        "actual_com_vox": act_com_vox,
        "planned_peak_vox": plan_peak_vox,
        "actual_peak_vox": act_peak_vox,
    }
    return m, aux


# ------------------------- plotting functions -------------------------

def plot_anatomy(subject: str,
                 t1_img: nib.Nifti1Image,
                 seg_img: Optional[nib.Nifti1Image],
                 sgacc_left_img: nib.Nifti1Image,
                 sgacc_right_img: nib.Nifti1Image,
                 out_path: Path,
                 target_threshold: float = 0.5):
    log.info("Creating anatomy/segmentation/sgACC figure using nilearn")

    from nilearn import plotting
    from nilearn.image import resample_to_img
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Resample sgACC masks to the T1w grid for anatomy visualization.
    left_mask_img = resample_to_img(
        sgacc_left_img,
        t1_img,
        interpolation="nearest",
    )
    right_mask_img = resample_to_img(
        sgacc_right_img,
        t1_img,
        interpolation="nearest",
    )

    left_data = left_mask_img.get_fdata()
    right_data = right_mask_img.get_fdata()

    left_data = (left_data > 0).astype(np.float32)
    right_data = (right_data > 0).astype(np.float32)

    # Re-create clean binary NIfTI masks after thresholding.
    left_mask_img = nib.Nifti1Image(left_data, t1_img.affine, t1_img.header)
    right_mask_img = nib.Nifti1Image(right_data, t1_img.affine, t1_img.header)

    # Use the actual mask centroids in the T1w grid so the cuts pass through the visible masks.
    left_vox = mask_centroid_vox(left_data, 0.0)
    right_vox = mask_centroid_vox(right_data, 0.0)

    left_world = voxel_to_world(left_vox, t1_img.affine)
    right_world = voxel_to_world(right_vox, t1_img.affine)

    print("DEBUG anatomy L centroid voxel in T1w grid:", left_vox)
    print("DEBUG anatomy R centroid voxel in T1w grid:", right_vox)
    print("DEBUG anatomy L centroid world/mm:", left_world)
    print("DEBUG anatomy R centroid world/mm:", right_world)
    print("DEBUG anatomy L positive voxels:", int(np.nansum(left_data > 0)))
    print("DEBUG anatomy R positive voxels:", int(np.nansum(right_data > 0)))

    red_cmap = mcolors.ListedColormap(["red"])

    fig = plt.figure(figsize=(14, 10), dpi=300, facecolor="white")

    rows = [
        ("A", f"{subject} T1w at left sgACC", t1_img, None, left_world),
        ("B", f"{subject} segmentation at left sgACC", seg_img if seg_img is not None else t1_img, None, left_world),
        ("C", f"{subject} left sgACC ROI", t1_img, left_mask_img, left_world),
        ("D", f"{subject} right sgACC ROI", t1_img, right_mask_img, right_world),
    ]

    from nilearn.image import crop_img

    for i, (letter, title, bg_img, overlay_img, cut_coords) in enumerate(rows):
        # Bốn hàng xếp chồng lên nhau.
        y0 = 0.755 - i * 0.235
        ax = fig.add_axes([0.06, y0, 0.82, 0.20])

        is_seg = "segmentation" in title.lower()

        # Ép tự động cắt bỏ phần cổ thừa (vùng chứa zero/background thừa)
        # để tất cả các ảnh có chung một góc nhìn (FOV) sát vào sọ não giống ảnh B.
        if seg_img is not None:
            # Cắt ảnh hiện tại dựa trên bo góc tự động (loại bỏ vùng viền đen trống)
            bg_img_cropped = crop_img(bg_img, rtol=0.0)
        else:
            bg_img_cropped = bg_img

        display = plotting.plot_anat(
            anat_img=bg_img_cropped,
            figure=fig,
            axes=ax,
            display_mode="ortho",
            cut_coords=tuple(cut_coords),
            cmap="gray" if not is_seg else "bone",
            black_bg=True,
            draw_cross=False,
            annotate=True,
            title=f"{letter}. {title}",
            colorbar=False,  # Không hiển thị thanh thang độ xám bên phải
        )

        if overlay_img is not None:
            display.add_overlay(
                overlay_img,
                cmap=red_cmap,
                threshold=0.0,
                alpha=1.0,
            )

        # Đồng bộ canvas tối đa
        try:
            display.trim_canvas()
        except:
            pass

        # Nhãn tọa độ bên phải
        fig.text(
            0.90,
            y0 + 0.085,
            f"x={cut_coords[0]:.1f} mm\n"
            f"y={cut_coords[1]:.1f} mm\n"
            f"z={cut_coords[2]:.1f} mm",
            fontsize=8,
            color="black",
            ha="left",
            va="center",
        )

    fig.suptitle(
        "Individual anatomy, segmentation, and left/right sgACC target mask",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_mosaic_indices(center: np.ndarray, shape: Tuple[int, int, int], n_each: int = 7) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for axis in [2, 0, 1]:
        c = int(round(center[axis]))
        span = max(1, shape[axis] // 5)
        vals = np.linspace(c - span, c + span, n_each).round().astype(int)
        vals = np.clip(vals, 0, shape[axis] - 1)
        out[axis] = sorted(set(int(v) for v in vals))
        while len(out[axis]) < n_each:
            out[axis].append(out[axis][-1])
    return out


def plot_map_mosaic(subject: str,
                    title: str,
                    t1: np.ndarray,
                    mapvol: np.ndarray,
                    sgacc: np.ndarray,
                    center_vox: np.ndarray,
                    cmap: str,
                    vmin: float,
                    vmax: float,
                    unit: str,
                    out_path: Path,
                    overlay_alpha: float = 0.56,
                    mosaic_x: Optional[Sequence[int]] = None,
                    mosaic_y: Optional[Sequence[int]] = None,
                    mosaic_z: Optional[Sequence[int]] = None):
    """Whole-head mosaic of pressure or temperature map.

    Layout: 3 rows (axial z, sagittal x, coronal y), user-specified columns.
    Colorbar shows the true global max value at the top tick.
    sgACC is always shown as white contour.
    No special column highlighting — just plain slice grids.
    """
    axes_order = [(2, "z"), (0, "x"), (1, "y")]

    # Shared colorbar ceiling passed in from the caller (same across planned/actual for comparability).
    # The per-volume true peak is annotated directly on the colorbar.
    true_vmax = vmax
    true_vmin = vmin

    # Per-volume true peak.
    peak_val = float(np.nanmax(mapvol)) if mapvol.size else true_vmax

    # Build index lists per axis from user-supplied mosaic args.
    idxs: Dict[int, List[int]] = {}
    base_lists = {2: mosaic_z, 0: mosaic_x, 1: mosaic_y}
    axis_map = {2: t1.shape[2], 0: t1.shape[0], 1: t1.shape[1]}
    for axis, base in base_lists.items():
        if base is not None:
            idxs[axis] = [int(np.clip(v, 0, axis_map[axis] - 1)) for v in base]
        else:
            c_ax = int(round(center_vox[axis]))
            span = max(1, axis_map[axis] // 5)
            vals = np.linspace(c_ax - span, c_ax + span, 7).round().astype(int)
            vals = np.clip(vals, 0, axis_map[axis] - 1)
            idxs[axis] = sorted(set(int(v) for v in vals))

    ncols = max(len(idxs[a]) for a in [2, 0, 1])
    for axis in [2, 0, 1]:
        while len(idxs[axis]) < ncols:
            idxs[axis].append(idxs[axis][-1])

    t1_vmin, t1_vmax = robust_t1_limits(t1)
    fig, axes = plt.subplots(3, ncols, figsize=(2.1 * ncols + 1.5, 7.2), facecolor="black")
    if ncols == 1:
        axes = np.array(axes).reshape(3, 1)
    axis_names = {2: "z", 0: "x", 1: "y"}
    for r, (axis, _) in enumerate(axes_order):
        aname = axis_names[axis]
        for c, idx in enumerate(idxs[axis][:ncols]):
            ax = axes[r, c]
            ax.set_facecolor("black")
            ax.imshow(slice2d(t1, axis, idx), cmap="gray", origin="lower", vmin=t1_vmin, vmax=t1_vmax)
            safe_imshow_overlay(ax, slice2d(mapvol, axis, idx), vmin=true_vmin, vmax=true_vmax, cmap=cmap, alpha=overlay_alpha)
            safe_contour(ax, slice2d(sgacc, axis, idx), 0.5, WHITE, "solid", 1.4)
            add_lr_labels(ax, axis)
            ax.text(0.03, 0.04, f"{aname}={idx}", transform=ax.transAxes, color="white", fontsize=8,
                    bbox=dict(facecolor="black", alpha=0.75, edgecolor="none"))
            for sp in ax.spines.values():
                sp.set_color("black")
            ax.set_xticks([]); ax.set_yticks([])

    # Colorbar: show max value at the top.
    fig.subplots_adjust(left=0.02, right=0.89, top=0.90, bottom=0.04, wspace=0.02, hspace=0.08)
    cax = fig.add_axes([0.905, 0.12, 0.016, 0.70])
    sm = cm.ScalarMappable(norm=Normalize(vmin=true_vmin, vmax=true_vmax), cmap=cmap)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(colors="white", labelsize=8)
    cbar.set_label(unit, color="white", fontsize=9)
    # Annotate peak value at the top of the bar.
    cbar.ax.text(0.5, 1.02, f"max\n{peak_val:.3g}", transform=cbar.ax.transAxes,
                 color="white", fontsize=7, ha="center", va="bottom", fontweight="bold")

    fig.suptitle(f"{subject} | {title}", color="white", fontsize=12, fontweight="bold")
    fig.savefig(out_path, dpi=220, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_planned_actual_focal_overlay(subject: str,
                                       condition: str,
                                       side: str,
                                       t1: np.ndarray,
                                       sgacc: np.ndarray,
                                       center_vox: np.ndarray,
                                       planned_focal: np.ndarray,
                                       actual_focal: np.ndarray,
                                       metrics: "FocalMetrics",
                                       out_path: Path,
                                       crop_vox: int = 80,
                                       planned_com_vox: Optional[np.ndarray] = None,
                                       actual_com_vox: Optional[np.ndarray] = None):
    """Overlay planned (orange-red) and actual (cyan) -3 dB focal volumes.

    Two-row layout:
    - TOP ROW: slices centred at the FOCAL ZONE (midpoint of planned+actual CoM).
      Shows the -3 dB zones clearly with a tight crop.
    - BOTTOM ROW: slices centred at the sgACC CENTROID.
      Shows targeting accuracy — how far the beam landed from the target.

    Both rows show: Coronal | Sagittal | Axial.
    sgACC contour (white) is drawn in both rows.
    """
    from scipy.ndimage import center_of_mass as scipy_com

    # ── Focal zone centre (for top row) ─────────────────────────────────────
    if planned_com_vox is not None and np.all(np.isfinite(planned_com_vox)):
        plan_com = np.asarray(planned_com_vox, dtype=float)
    elif np.any(planned_focal):
        plan_com = np.array(scipy_com(planned_focal.astype(float)))
    else:
        plan_com = center_vox.copy()

    if actual_com_vox is not None and np.all(np.isfinite(actual_com_vox)):
        act_com = np.asarray(actual_com_vox, dtype=float)
    elif np.any(actual_focal):
        act_com = np.array(scipy_com(actual_focal.astype(float)))
    else:
        act_com = center_vox.copy()

    focal_ctr = (plan_com + act_com) / 2.0

    # ── Slice indices ────────────────────────────────────────────────────────
    # Top row: cut through focal zone; bottom row: cut through sgACC
    VIEWS = [("Coronal", 1), ("Sagittal", 0), ("Axial", 2)]
    focal_axes_info = [(v, ax, int(round(focal_ctr[ax])))  for v, ax in VIEWS]
    sgacc_axes_info = [(v, ax, int(round(center_vox[ax]))) for v, ax in VIEWS]

    t1_vmin, t1_vmax = robust_t1_limits(t1)
    overlap      = np.logical_and(planned_focal, actual_focal)
    planned_only = np.logical_and(planned_focal, ~actual_focal)
    actual_only  = np.logical_and(actual_focal,  ~planned_focal)
    ALPHA = 0.65

    def _render_row(axes_row, axes_info, crop_half, row_label):
        for ax, (view, axis, idx) in zip(axes_row, axes_info):
            ax.set_facecolor("black")
            t1s = slice2d(t1, axis, idx)
            cxy = point_xy(np.array([focal_ctr[0], focal_ctr[1], focal_ctr[2]]
                                    if "Focal" in row_label else center_vox), axis)
            lim = crop_limits(t1s.shape, cxy, crop_half)

            ax.imshow(apply_crop(t1s, lim), cmap="gray", origin="lower",
                      vmin=t1_vmin, vmax=t1_vmax, interpolation="bilinear")

            for vol_bin, colour in [
                (planned_only, PLANNED),
                (actual_only,  ACTUAL),
                (overlap,      GREEN),
            ]:
                sl = apply_crop(slice2d(vol_bin.astype(float), axis, idx), lim)
                if np.any(sl >= 0.5):
                    rgba = np.zeros((*sl.shape, 4), dtype=float)
                    rgba[sl >= 0.5] = [
                        int(colour[1:3], 16)/255,
                        int(colour[3:5], 16)/255,
                        int(colour[5:7], 16)/255,
                        ALPHA,
                    ]
                    ax.imshow(rgba, origin="lower")

            # Always draw sgACC contour in white
            safe_contour(ax, apply_crop(slice2d(sgacc, axis, idx), lim),
                         0.5, WHITE, "solid", 1.8)
            safe_contour(ax, apply_crop(slice2d(planned_focal.astype(float), axis, idx), lim),
                         0.5, PLANNED, "solid", 2.0)
            safe_contour(ax, apply_crop(slice2d(actual_focal.astype(float), axis, idx), lim),
                         0.5, ACTUAL, "dashed", 2.0)
            safe_contour(ax, apply_crop(slice2d(overlap.astype(float), axis, idx), lim),
                         0.5, GREEN, "solid", 1.4)

            add_lr_labels(ax, axis)
            ax.set_title(view, color="white", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

        # Row label on the leftmost axis
        axes_row[0].set_ylabel(row_label, color="white", fontsize=8,
                               labelpad=4, rotation=90, va="center")

    fig, axes2d = plt.subplots(2, 3, figsize=(14, 9),
                               gridspec_kw={"hspace": 0.12, "wspace": 0.04},
                               facecolor="black")

    # Top row — focal zone view (tight crop so small zones are visible)
    _render_row(axes2d[0], focal_axes_info, crop_half=50, row_label="Focal zone")
    # Bottom row — sgACC target view (wider crop to show anatomy context)
    _render_row(axes2d[1], sgacc_axes_info, crop_half=crop_vox, row_label="sgACC target")

    # Legend
    handles = [
        Patch(facecolor=PLANNED, alpha=ALPHA, edgecolor=PLANNED,
              label=f"Planned -3 dB ({metrics.planned_focal_volume_mm3:.0f} mm³)"),
        Patch(facecolor=ACTUAL,  alpha=ALPHA, edgecolor=ACTUAL,
              label=f"Actual  -3 dB ({metrics.actual_focal_volume_mm3:.0f} mm³)"),
        Patch(facecolor=GREEN,   alpha=ALPHA, edgecolor=GREEN, label="Overlap"),
        Line2D([0], [0], color=WHITE, lw=1.6, label="sgACC (BA25)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, facecolor="black",
               labelcolor="white", framealpha=0.85, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0))

    # Metrics text
    txt = (
        f"Dice (planned ∩ actual): {metrics.dice_planned_actual:.3f}   |   "
        f"Actual in planned: {metrics.actual_percent_inside_planned:.1f}%   |   "
        f"Planned in actual: {metrics.planned_percent_inside_actual:.1f}%\n"
        f"Actual focal in sgACC: {metrics.actual_percent_inside_sgacc:.1f}%   |   "
        f"sgACC covered by actual: {metrics.sgacc_percent_covered_by_actual:.1f}%   |   "
        f"Actual centre → sgACC: {metrics.actual_center_to_sgacc_mm:.2f} mm\n"
        f"Actual FWHM: {metrics.actual_fwhm_major_mm:.1f} × "
        f"{metrics.actual_fwhm_middle_mm:.1f} × {metrics.actual_fwhm_minor_mm:.1f} mm   |   "
        f"Planned FWHM: {metrics.planned_fwhm_major_mm:.1f} × "
        f"{metrics.planned_fwhm_middle_mm:.1f} × {metrics.planned_fwhm_minor_mm:.1f} mm"
    )
    fig.text(0.5, 0.07, txt, color="white", ha="center", fontsize=8,
             bbox=dict(facecolor="#111111", edgecolor="#555555", alpha=0.92, pad=4))

    fig.subplots_adjust(bottom=0.22, top=0.93, left=0.05, right=0.98)
    fig.suptitle(
        f"{subject} | {COND_TITLE[condition]} | {SIDE_TITLE[side]} | "
        f"Planned vs Actual -3 dB focal volume overlay",
        color="white", fontsize=11, fontweight="bold",
    )
    fig.savefig(out_path, dpi=220, facecolor="black", bbox_inches="tight")
    plt.close(fig)


# Keep the old planned_actual function name for backward compatibility but redirect to overlay.
def plot_planned_actual_focal(subject: str,
                              condition: str,
                              side: str,
                              t1: np.ndarray,
                              map_planned: np.ndarray,
                              map_actual: np.ndarray,
                              sgacc: np.ndarray,
                              center_vox: np.ndarray,
                              planned_focal: np.ndarray,
                              actual_focal: np.ndarray,
                              cmap: str,
                              vmin: float,
                              vmax: float,
                              unit: str,
                              out_path: Path,
                              crop_vox: int = 80,
                              is_pressure: bool = True,
                              metrics: Optional["FocalMetrics"] = None):
    """Kept for backward compat; the overlay version is preferred for pressure."""
    axes_info = [("Coronal", 1, int(round(center_vox[1]))), ("Sagittal", 0, int(round(center_vox[0]))), ("Axial", 2, int(round(center_vox[2])))]
    t1_vmin, t1_vmax = robust_t1_limits(t1)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), facecolor="black")
    rows = [("Planning", map_planned, planned_focal), ("Post-hoc / actual", map_actual, actual_focal)]
    for r, (row_label, vol, focal) in enumerate(rows):
        for c, (view, axis, idx) in enumerate(axes_info):
            ax = axes[r, c]
            ax.set_facecolor("black")
            t1s = slice2d(t1, axis, idx)
            center_xy = point_xy(center_vox, axis)
            lim = crop_limits(t1s.shape, center_xy, crop_vox)
            ax.imshow(apply_crop(t1s, lim), cmap="gray", origin="lower", vmin=t1_vmin, vmax=t1_vmax)
            safe_imshow_overlay(ax, apply_crop(slice2d(vol, axis, idx), lim), vmin=vmin, vmax=vmax, cmap=cmap, alpha=0.55)
            safe_contour(ax, apply_crop(slice2d(sgacc, axis, idx), lim), 0.5, WHITE, "solid", 1.6)
            if is_pressure:
                safe_contour(ax, apply_crop(slice2d(planned_focal.astype(float), axis, idx), lim), 0.5, PLANNED, "solid", 2.0)
                safe_contour(ax, apply_crop(slice2d(actual_focal.astype(float), axis, idx), lim), 0.5, ACTUAL, "dashed", 2.0)
            cx, cy = adjust_xy(point_xy(center_vox, axis), lim)
            if c == 0:
                ax.set_ylabel(row_label, color="white", fontsize=11, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
    sm = cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    fig.subplots_adjust(right=0.86, bottom=0.12, top=0.88, wspace=0.10, hspace=0.18)
    cbar_ax = fig.add_axes([0.885, 0.20, 0.025, 0.60])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(colors="white", labelsize=8)
    cbar.set_label(unit, color="white")
    handles = [Line2D([0], [0], color=WHITE, lw=1.8, label="sgACC mask")]
    if is_pressure:
        handles += [
            Line2D([0], [0], color=PLANNED, lw=2.2, label="Planned -3 dB focal volume"),
            Line2D([0], [0], color=ACTUAL, lw=2.2, linestyle="--", label="Actual -3 dB focal volume"),
        ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), facecolor="black", labelcolor="white", framealpha=0.75)
    fig.suptitle(f"{subject} | {COND_TITLE[condition]} | {SIDE_TITLE[side]} | planned vs post-hoc {unit}",
                 color="white", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=220, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    axes_info = [("Coronal", 1, int(round(center_vox[1]))), ("Sagittal", 0, int(round(center_vox[0]))), ("Axial", 2, int(round(center_vox[2])))]
    t1_vmin, t1_vmax = robust_t1_limits(t1)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), facecolor="black")
    rows = [("Planning", map_planned, planned_focal), ("Post-hoc / actual", map_actual, actual_focal)]
    for r, (row_label, vol, focal) in enumerate(rows):
        for c, (view, axis, idx) in enumerate(axes_info):
            ax = axes[r, c]
            ax.set_facecolor("black")
            t1s = slice2d(t1, axis, idx)
            center_xy = point_xy(center_vox, axis)
            lim = crop_limits(t1s.shape, center_xy, crop_vox)
            ax.imshow(apply_crop(t1s, lim), cmap="gray", origin="lower", vmin=t1_vmin, vmax=t1_vmax)
            safe_imshow_overlay(ax, apply_crop(slice2d(vol, axis, idx), lim), vmin=vmin, vmax=vmax, cmap=cmap, alpha=0.55)
            safe_contour(ax, apply_crop(slice2d(sgacc, axis, idx), lim), 0.5, WHITE, "solid", 1.6)
            if is_pressure:
                # show both planned and actual -3 dB contours in both rows for direct comparison
                safe_contour(ax, apply_crop(slice2d(planned_focal.astype(float), axis, idx), lim), 0.5, PLANNED, "solid", 2.0)
                safe_contour(ax, apply_crop(slice2d(actual_focal.astype(float), axis, idx), lim), 0.5, ACTUAL, "dashed", 2.0)
            cx, cy = adjust_xy(center_xy, lim)
            #ax.scatter(cx, cy, marker="+", c=WHITE, s=55, linewidths=1.4)
            #ax.set_title(view, color="white", fontsize=10)
            if c == 0:
                ax.set_ylabel(row_label, color="white", fontsize=11, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
    sm = cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    # Put colorbar outside the 2x3 image grid, not on top of the axial panel.
    fig.subplots_adjust(right=0.86, bottom=0.12, top=0.88, wspace=0.10, hspace=0.18)

    cbar_ax = fig.add_axes([0.885, 0.20, 0.025, 0.60])
    cbar = fig.colorbar(sm, cax=cbar_ax)

    cbar.ax.tick_params(colors="white", labelsize=8)
    cbar.set_label(unit, color="white")
    handles = [
        Line2D([0], [0], color=WHITE, lw=1.8, label="sgACC mask"),
    ]
    if is_pressure:
        handles += [
            Line2D([0], [0], color=PLANNED, lw=2.2, label="Planned -3 dB focal volume"),
            Line2D([0], [0], color=ACTUAL, lw=2.2, linestyle="--", label="Actual -3 dB focal volume"),
        ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), facecolor="black", labelcolor="white", framealpha=0.75)
    fig.suptitle(f"{subject} | {COND_TITLE[condition]} | {SIDE_TITLE[side]} | planned vs post-hoc {unit}",
                 color="white", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=220, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def plot_overlap_figure(subject: str,
                        condition: str,
                        side: str,
                        t1: np.ndarray,
                        sgacc: np.ndarray,
                        center_vox: np.ndarray,
                        planned_focal: np.ndarray,
                        actual_focal: np.ndarray,
                        metrics: FocalMetrics,
                        out_path: Path,
                        crop_vox: int = 80):
    axes_info = [("Coronal", 1, int(round(center_vox[1]))), ("Sagittal", 0, int(round(center_vox[0]))), ("Axial", 2, int(round(center_vox[2])))]
    t1_vmin, t1_vmax = robust_t1_limits(t1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.7), facecolor="black")
    overlap = np.logical_and(planned_focal, actual_focal).astype(float)
    for ax, (view, axis, idx) in zip(axes, axes_info):
        ax.set_facecolor("black")
        t1s = slice2d(t1, axis, idx)
        lim = crop_limits(t1s.shape, point_xy(center_vox, axis), crop_vox)
        ax.imshow(apply_crop(t1s, lim), cmap="gray", origin="lower", vmin=t1_vmin, vmax=t1_vmax)
        safe_contour(ax, apply_crop(slice2d(sgacc, axis, idx), lim), 0.5, WHITE, "solid", 1.8)
        safe_contour(ax, apply_crop(slice2d(planned_focal.astype(float), axis, idx), lim), 0.5, PLANNED, "solid", 2.2)
        safe_contour(ax, apply_crop(slice2d(actual_focal.astype(float), axis, idx), lim), 0.5, ACTUAL, "dashed", 2.2)
        safe_contour(ax, apply_crop(slice2d(overlap, axis, idx), lim), 0.5, GREEN, "solid", 1.2)
        cx, cy = adjust_xy(point_xy(center_vox, axis), lim)
        #ax.scatter(cx, cy, marker="+", c=WHITE, s=60, linewidths=1.6)
        ax.set_title(view, color="white")
        ax.set_xticks([]); ax.set_yticks([])
    txt = (
        f"Dice planned/actual = {metrics.dice_planned_actual:.3f}\n"
        f"Actual in planned = {metrics.actual_percent_inside_planned:.1f}% | Planned in actual = {metrics.planned_percent_inside_actual:.1f}%\n"
        f"Actual focal in sgACC = {metrics.actual_percent_inside_sgacc:.1f}% | sgACC covered = {metrics.sgacc_percent_covered_by_actual:.1f}%\n"
        f"Actual focal center to sgACC = {metrics.actual_center_to_sgacc_mm:.2f} mm\n"
        f"Actual FWHM PCA axes = {metrics.actual_fwhm_major_mm:.1f} x {metrics.actual_fwhm_middle_mm:.1f} x {metrics.actual_fwhm_minor_mm:.1f} mm"
    )
    fig.text(0.5, 0.02, txt, color="white", ha="center", fontsize=9,
             bbox=dict(facecolor="#111111", edgecolor="#444444", alpha=0.9))
    handles = [
        Line2D([0], [0], color=WHITE, lw=2, label="sgACC"),
        Line2D([0], [0], color=PLANNED, lw=2.2, label="Planned -3 dB"),
        Line2D([0], [0], color=ACTUAL, lw=2.2, linestyle="--", label="Actual -3 dB"),
        Line2D([0], [0], color=GREEN, lw=1.7, label="Overlap contour"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, facecolor="black", labelcolor="white", framealpha=0.75)
    fig.suptitle(f"{subject} | {COND_TITLE[condition]} | {SIDE_TITLE[side]} | -3 dB focal overlap", color="white", fontsize=13, fontweight="bold", y=0.93)
    fig.tight_layout(rect=[0, 0.17, 1, 0.87])
    fig.savefig(out_path, dpi=220, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def get_drift_stats(df: pd.DataFrame) -> Dict[str, Tuple[float, float, float]]:
    stats = {}
    cols = [
        "dx_from_first_mm", "dy_from_first_mm", "dz_from_first_mm",
        "rot_x_from_first_deg", "rot_y_from_first_deg", "rot_z_from_first_deg"
    ]
    for col in cols:
        vals = df[col].astype(float).to_numpy()
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        max_abs = float(np.nanmax(np.abs(vals))) if len(vals) > 0 else 0.0
        stats[col] = (mean, std, max_abs)
    return stats


def draw_panel_statistics(ax, stats: Dict[str, Tuple[float, float, float]]):
    ax.axis("off")
    ax.text(0.01, 0.99, "", bbox=dict(boxstyle="square,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9))
    ax.text(0.02, 0.94, "Stability Statistics (relative to first position):", fontsize=13, fontweight="bold", va="top")
    y = 0.82
    dy = 0.05
    labels_trans = [
        ("dx_from_first_mm", "• X deviation:"),
        ("dy_from_first_mm", "• Y deviation:"),
        ("dz_from_first_mm", "• Z deviation:")
    ]
    for col, label in labels_trans:
        mean, std, max_abs = stats[col]
        ax.text(0.03, y, label, fontsize=11, fontweight="semibold", va="center")
        y -= dy
        stat_text = f"• mean = {mean:.3f} mm, std = {std:.3f} mm, max_abs = {max_abs:.3f} mm"
        ax.text(0.08, y, stat_text, fontsize=10.5, va="center")
        y -= dy * 1.3
        
    y -= 0.01
    labels_rot = [
        ("rot_x_from_first_deg", "• Rot-X deviation:"),
        ("rot_y_from_first_deg", "• Rot-Y deviation:"),
        ("rot_z_from_first_deg", "• Rot-Z deviation:")
    ]
    for col, label in labels_rot:
        mean, std, max_abs = stats[col]
        ax.text(0.03, y, label, fontsize=11, fontweight="semibold", va="center")
        y -= dy
        stat_text = f"• mean = {mean:.3f}°, std = {std:.3f}°, max_abs = {max_abs:.3f}°"
        ax.text(0.08, y, stat_text, fontsize=10.5, va="center")
        y -= dy * 1.3


def plot_drift_combined(
    df_exp: pd.DataFrame, 
    df_con: Optional[pd.DataFrame], 
    subject: str, 
    side: str, 
    out_path: Path
):
    # Determine if both conditions are available
    has_con = df_con is not None and not df_con.empty
    
    fig_height = 12.0 if has_con else 6.2
    fig = plt.figure(figsize=(19, fig_height), facecolor="white")
    
    cols_config = [
        ("dx_from_first_mm", "X position deviation", "Deviation (mm)"),
        ("dy_from_first_mm", "Y position deviation", "Deviation (mm)"),
        ("dz_from_first_mm", "Z position deviation", "Deviation (mm)"),
        ("rot_x_from_first_deg", "Rot-X deviation", "Deviation (degrees)"),
        ("rot_y_from_first_deg", "Rot-Y deviation", "Deviation (degrees)"),
        ("rot_z_from_first_deg", "Rot-Z deviation", "Deviation (degrees)")
    ]
    
    if has_con:
        gs_outer = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.35)
        gs_a = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs_outer[0], width_ratios=[1, 1, 1, 1.4], hspace=0.28, wspace=0.22)
        gs_b = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs_outer[1], width_ratios=[1, 1, 1, 1.4], hspace=0.28, wspace=0.22)
    else:
        gs_outer = gridspec.GridSpec(1, 1)
        gs_a = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs_outer[0], width_ratios=[1, 1, 1, 1.4], hspace=0.28, wspace=0.22)
        
    # ── PANEL A (Experimental Condition) ──
    exp_color = "#2563eb"
    exp_n = len(df_exp)
    fig.text(0.35, 0.965 if has_con else 0.93, f"{subject}_exp - Stability Analysis (n={exp_n})", fontsize=13, fontweight="bold", ha="center")
    
    # Vertical panel labels
    fig.text(0.015, 0.91 if has_con else 0.88, "A", fontsize=24, fontweight="bold", ha="center")
    fig.text(0.015, 0.70 if has_con else 0.45, "experimental condition", fontsize=15, fontweight="bold", rotation="vertical", ha="center", va="center", color="#4b5563")
    
    exp_stats = get_drift_stats(df_exp)
    
    for idx, (col, title, ylabel) in enumerate(cols_config):
        row = 0 if idx < 3 else 1
        c = idx % 3
        ax = fig.add_subplot(gs_a[row, c])
        
        y_vals = df_exp[col].astype(float).to_numpy()
        x_vals = np.arange(len(df_exp))
        
        ax.plot(x_vals, y_vals, color=exp_color, marker="o", markersize=3, linewidth=1.0, alpha=0.9)
        ax.axhline(0, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
        
        ax.grid(True, linestyle="-", linewidth=0.5, color="#e5e7eb", alpha=0.8)
        ax.set_facecolor("#f9fafb")
        ax.tick_params(axis="both", labelsize=9)
        
        ax.set_ylim(-2.0, 2.0)
        ax.set_yticks([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
        ax.set_title(title, fontsize=10, fontweight="semibold")
        ax.set_xlabel("Position Index" if row == 1 else "", fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        
    ax_stats_a = fig.add_subplot(gs_a[:, 3])
    draw_panel_statistics(ax_stats_a, exp_stats)
    
    # ── PANEL B (Control Condition) ──
    if has_con and df_con is not None:
        con_color = "#16a34a"
        con_n = len(df_con)
        fig.text(0.35, 0.485, f"{subject}_con - Stability Analysis (n={con_n})", fontsize=13, fontweight="bold", ha="center")
        
        fig.text(0.015, 0.43, "B", fontsize=24, fontweight="bold", ha="center")
        fig.text(0.015, 0.22, "control condition", fontsize=15, fontweight="bold", rotation="vertical", ha="center", va="center", color="#4b5563")
        
        con_stats = get_drift_stats(df_con)
        
        for idx, (col, title, ylabel) in enumerate(cols_config):
            row = 0 if idx < 3 else 1
            c = idx % 3
            ax = fig.add_subplot(gs_b[row, c])
            
            y_vals = df_con[col].astype(float).to_numpy()
            x_vals = np.arange(len(df_con))
            
            ax.plot(x_vals, y_vals, color=con_color, marker="o", markersize=3, linewidth=1.0, alpha=0.9)
            ax.axhline(0, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
            
            ax.grid(True, linestyle="-", linewidth=0.5, color="#e5e7eb", alpha=0.8)
            ax.set_facecolor("#f9fafb")
            ax.tick_params(axis="both", labelsize=9)
            
            ax.set_ylim(-2.0, 2.0)
            ax.set_yticks([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
            ax.set_title(title, fontsize=10, fontweight="semibold")
            ax.set_xlabel("Position Index" if row == 1 else "", fontsize=8.5)
            ax.set_ylabel(ylabel, fontsize=8.5)
            
        ax_stats_b = fig.add_subplot(gs_b[:, 3])
        draw_panel_statistics(ax_stats_b, con_stats)
        
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93 if has_con else 0.88, bottom=0.05 if has_con else 0.10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------- mesh plotting -------------------------

def load_mesh(mesh_path: Path, scalp_tag: int = 1005, max_triangles: int = 75000):
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
    step = max(1, tris.shape[0] // max_triangles)
    tris_plot = tris[::step]
    xyz_min = points.min(axis=0); xyz_max = points.max(axis=0)
    mid = (xyz_min + xyz_max) / 2
    rng = (xyz_max - xyz_min).max() / 2
    return points, tris_plot, mid, rng


def set_mesh_view(ax, view: str):
    views = {
        "left": (0, 180),
        "right": (0, 0),
        "front": (0, 90),
        "top": (90, -90),
    }
    elev, azim = views.get(view, views["left"])
    ax.view_init(elev=elev, azim=azim)


def plot_mesh_positions(subject: str,
                        mesh_path: Path,
                        planned: Dict[str, TxMatrix],
                        actual_groups: Dict[Tuple[str, str], List[TxMatrix]],
                        medoids: Dict[Tuple[str, str], TxMatrix],
                        out_path: Path,
                        view: str = "left"):
    """Plot planned (yellow) and actual (blue) discs with green overlap — 4 views.

    Uses the same proven 2-D projection approach as plot_mesh_planned_all_positions.
    Overlap region computed via shapely if available, otherwise visual layering.
    Layout: Left lateral | Right lateral | Front | Top.
    """
    log.info("Creating head mesh planned + actual transducer position plot")
    points, tris, mid, rng = load_mesh(mesh_path)
    from matplotlib.tri import Triangulation as _Tri

    PLANNED_COL = "#f5c518"
    ACTUAL_COL  = "#1e90ff"
    OVERLAP_COL = "#22c55e"
    DISC_R = 22.0

    four_views = [
        ("Left / lateral",  "left"),
        ("Right / lateral", "right"),
        ("Front",           "front"),
        ("Top",             "top"),
    ]

    # Snap planned (one per side) to mesh surface.
    planned_snapped: Dict[str, tuple] = {}
    for side, tx in planned.items():
        sp = _snap_tx_to_surface(tx.center, points)
        planned_snapped[side] = (sp, _outward_normal(sp, mid))

    # Actual: one representative disc per (cond, side) — mean of all frame centres.
    actual_rep: Dict[tuple, tuple] = {}
    for key, txs in actual_groups.items():
        if not txs:
            continue
        mean_c = np.array([t.center for t in txs], dtype=float).mean(axis=0)
        sp = _snap_tx_to_surface(mean_c, points)
        actual_rep[key] = (sp, _outward_normal(sp, mid))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), facecolor="white")

    for ax, (view_title, view_name) in zip(axes, four_views):
        ax.set_facecolor("#f0f0f0")
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(view_title, fontsize=11, fontweight="bold")

        # Mesh shading
        xs2d, ys2d, depth, tris_s = _mesh_silhouette_2d(points, tris, view_name)
        d_min, d_max = depth.min(), depth.max()
        d_norm = (depth - d_min) / (d_max - d_min + 1e-9)
        tri_d  = d_norm[tris_s].mean(axis=1)
        triang = _Tri(xs2d, ys2d, tris_s)
        ax.tripcolor(triang, facecolors=tri_d, cmap="Greys_r",
                     vmin=0.0, vmax=1.0, edgecolors="none", rasterized=True)

        if   view_name == "left":   cam = np.array([-1.,  0.,  0.])
        elif view_name == "right":  cam = np.array([ 1.,  0.,  0.])
        elif view_name == "front":  cam = np.array([ 0.,  1.,  0.])
        elif view_name == "back":   cam = np.array([ 0., -1.,  0.])
        else:                       cam = np.array([ 0.,  0.,  1.])

        # Collect visible disc polygons as (xs, ys) arrays
        plan_xys, act_xys = [], []

        for side, (sp, out_n) in planned_snapped.items():
            if float(np.dot(out_n, cam)) < -0.1:
                continue
            xs_r, ys_r = _project_disc_to_2d(sp, out_n, DISC_R, view_name)
            plan_xys.append((xs_r, ys_r))

        for key, (sp, out_n) in actual_rep.items():
            if float(np.dot(out_n, cam)) < -0.1:
                continue
            xs_r, ys_r = _project_disc_to_2d(sp, out_n, DISC_R, view_name)
            act_xys.append((xs_r, ys_r))

        # Layer 1: planned yellow
        for xs_r, ys_r in plan_xys:
            ax.fill(xs_r, ys_r, color=PLANNED_COL, alpha=0.90, zorder=3, linewidth=0)

        # Layer 2: actual blue
        for xs_r, ys_r in act_xys:
            ax.fill(xs_r, ys_r, color=ACTUAL_COL, alpha=0.85, zorder=4, linewidth=0)

        # Layer 3: overlap green (shapely intersection if available)
        # Layer 3: overlap green (shapely intersection if available)
        # Layer 3: overlap green, no Shapely needed
        # Rasterize planned and actual projected discs, then paint their intersection green.
        pad = rng * 0.08 
        if plan_xys and act_xys:
            from matplotlib.path import Path as MplPath

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
                planned_mask |= MplPath(poly).contains_points(grid_points)

            for xs_r, ys_r in act_xys:
                poly = np.column_stack([xs_r, ys_r])
                actual_mask |= MplPath(poly).contains_points(grid_points)

            overlap_mask = (planned_mask & actual_mask).reshape(ny, nx)

            if np.any(overlap_mask):
                rgba = np.zeros((ny, nx, 4), dtype=float)
                rgba[overlap_mask] = [
                    int(OVERLAP_COL[1:3], 16) / 255,
                    int(OVERLAP_COL[3:5], 16) / 255,
                    int(OVERLAP_COL[5:7], 16) / 255,
                    1.0,
                ]

                ax.imshow(
                    rgba,
                    origin="lower",
                    extent=[x_min, x_max, y_min, y_max],
                    interpolation="nearest",
                    zorder=20,
                )

        ax.set_xlim(xs2d.min() - pad, xs2d.max() + pad)
        ax.set_ylim(ys2d.min() - pad, ys2d.max() + pad)

    from matplotlib.patches import Patch as _Patch
    fig.legend(handles=[
        _Patch(facecolor=PLANNED_COL, edgecolor="none", label="Planned (L & R)"),
        _Patch(facecolor=ACTUAL_COL,  edgecolor="none", label="Actual (L & R)"),
        _Patch(facecolor=OVERLAP_COL, edgecolor="none", label="Overlap"),
    ], loc="center left", bbox_to_anchor=(0.88, 0.50), fontsize=9, frameon=True)

    fig.suptitle(f"{subject} | Planned and actual transducer positions",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 0.87, 0.97])
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _draw_transducer_disc(ax, tx: "TxMatrix", color, disc_radius_mm: float = 22.0,
                           n_pts: int = 36, alpha: float = 0.92, label: Optional[str] = None):
    """Draw a filled circular disc at the transducer position in 3-D.

    The disc lies in the plane perpendicular to the transducer's local Z-axis
    (column 2 of the rotation matrix).  This faithfully represents a circular
    flat transducer face as visible in the reference images.
    """
    origin = tx.center.astype(float)
    normal = tx.matrix[:3, 2].astype(float)
    normal = normal / (np.linalg.norm(normal) + 1e-9)

    # Build two orthogonal basis vectors in the disc plane.
    ref = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, ref); u /= np.linalg.norm(u) + 1e-9
    v = np.cross(normal, u)

    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=True)
    rim = origin[None, :] + disc_radius_mm * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)

    # Filled polygon (Poly3DCollection).
    poly = Poly3DCollection([rim], alpha=alpha, zorder=5)
    poly.set_facecolor(color)
    poly.set_edgecolor("black")
    poly.set_linewidth(0.8)
    ax.add_collection3d(poly)

    # Invisible scatter just to get a legend handle.
    if label is not None:
        ax.scatter([], [], [], s=120, color=color, edgecolors="black", label=label)

    # Short normal arrow showing beam direction (away from scalp).
    arrow_len = 18.0
    tip = origin + normal * arrow_len
    ax.plot([origin[0], tip[0]], [origin[1], tip[1]], [origin[2], tip[2]],
            color=color, linewidth=2.5, alpha=0.95, zorder=6)


def _snap_tx_to_surface(tx_center: np.ndarray,
                         mesh_points: np.ndarray) -> np.ndarray:
    """Return the mesh vertex closest to the transducer centre.

    This snaps the displayed disc to the skull surface regardless of any
    coordinate-system offset between the GUMMarker XML and the mesh file.
    """
    dists = np.linalg.norm(mesh_points - tx_center[None, :], axis=1)
    return mesh_points[int(np.argmin(dists))].copy()


def _outward_normal(surface_point: np.ndarray, mesh_mid: np.ndarray) -> np.ndarray:
    """Estimate the outward surface normal as the unit vector from mesh
    centroid → surface point.  Works for any convex head mesh."""
    n = surface_point - mesh_mid
    return n / (np.linalg.norm(n) + 1e-9)


def _project_disc_to_2d(origin: np.ndarray, normal: np.ndarray,
                         radius_mm: float, view_name: str,
                         n_pts: int = 72) -> Tuple[np.ndarray, np.ndarray]:
    """Project a 3-D disc rim onto the 2-D screen plane for the given view.

    Builds the disc in 3-D then drops the depth axis.  Result is an ellipse
    (circle when viewed face-on) with no z-order artefacts.
    """
    n = normal / (np.linalg.norm(normal) + 1e-9)
    ref = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([1., 0., 0.])
    u = np.cross(n, ref);  u /= np.linalg.norm(u) + 1e-9
    v = np.cross(n, u)
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=True)
    rim3d = origin[None, :] + radius_mm * (np.cos(theta)[:, None] * u +
                                            np.sin(theta)[:, None] * v)
    # Drop depth axis per view: (horizontal, vertical) on screen.
    if view_name in ("left", "right"):
        xs_r = rim3d[:, 1] * (-1.0 if view_name == "right" else 1.0)
        return xs_r, rim3d[:, 2]
    elif view_name in ("front", "back"):  return rim3d[:, 0], rim3d[:, 2]
    else:                                  return rim3d[:, 0], rim3d[:, 1]  # top/bottom


def _point_2d(p: np.ndarray, view_name: str) -> Tuple[float, float]:
    """Project a single 3-D point to 2-D screen coords."""
    if view_name in ("left", "right"):
        return float(p[1]) * (-1.0 if view_name == "right" else 1.0), float(p[2])
    elif view_name in ("front", "back"):  return float(p[0]), float(p[2])
    else:                                  return float(p[0]), float(p[1])


def _mesh_silhouette_2d(points: np.ndarray, tris: np.ndarray,
                          view_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project mesh to 2-D and sort triangles back-to-front for correct painter's order."""
    if view_name in ("left", "right"):
        ys = points[:, 2]
        depth = points[:, 0] * (1.0 if view_name == "right" else -1.0)
        # Mirror horizontal axis for right view so the head faces the correct
        # direction: left-lateral → anterior (Y+) on the right; right-lateral →
        # anterior (Y+) on the left (mirror image, as anatomically expected).
        xs = points[:, 1] * (-1.0 if view_name == "right" else 1.0)
    elif view_name in ("front", "back"):
        xs, ys = points[:, 0], points[:, 2]
        # RAS: +Y = anterior (face). front camera at +Y → anterior = near → depth = +Y.
        depth  = points[:, 1] * (1.0 if view_name == "front" else -1.0)
    else:  # top / bottom
        xs, ys = points[:, 0], points[:, 1]
        depth  = points[:, 2] * (1.0 if view_name == "top" else -1.0)
    order = np.argsort(depth[tris].mean(axis=1))   # back → front
    return xs, ys, depth, tris[order]


def plot_mesh_planned_all_positions(subject: str,
                                    mesh_path: Path,
                                    planned_txs: List[TxMatrix],
                                    out_path: Path,
                                    view: str = "left"):
    """Plot all planned transducer positions on the head mesh in 4 views.

    Rendering strategy
    ------------------
    We use a **2-D orthographic projection** rather than matplotlib 3-D axes.
    matplotlib's Poly3DCollection has a fundamental z-ordering bug that makes
    discs appear clipped or on the wrong side of the mesh.  The 2-D approach
    avoids this entirely.

    Three fixes vs the previous version
    ------------------------------------
    1. **Surface snapping**: each transducer centre is snapped to the nearest
       mesh vertex so that the disc always sits ON the skull regardless of any
       coordinate-system offset between the GUMMarker XML and the mesh file.

    2. **Outward normal from mesh centroid**: the disc orientation is computed
       as (surface_point − mesh_centroid), not from the GUMMarker rotation
       matrix column-2 (which points INWARD = beam direction).  Using the
       beam direction for the visual normal caused discs to appear edge-on or
       face-away from the camera.

    3. **Correct back-face culling**: a disc is drawn only when its outward
       normal has a positive dot-product with the camera direction, i.e. it
       faces the viewer.  Previously the beam (inward) normal was used, which
       inverted the culling so left-side discs appeared in the right-lateral
       view and vice-versa.

    Layout: Left lateral | Right lateral | Top | Bottom.
    No medoid stars, no actual-frame clouds — planned positions only.
    """
    log.info("Creating planned-only head mesh transducer position plot (2-D projection)")
    points, tris, mid, rng = load_mesh(mesh_path)
    from matplotlib.tri import Triangulation as _Tri

    four_views = [
        ("Left / lateral",  "left"),
        ("Right / lateral", "right"),
        ("Top",             "top"),
        ("Front",           "front"),
    ]

    # Filter to named planned Tx positions.
    selected = [tx for tx in planned_txs
                if "Tx" in (tx.description or "") and "pos" in (tx.description or "")]
    if not selected:
        selected = planned_txs
    selected = sorted(selected, key=lambda t: t.description)

    # Assign a distinct color to every position using tab10/tab20 palette.
    # With up to 10 positions per side (20 total), each gets a unique color.
    # ── Manual colors per planned position ──────────────────────────
    POSITION_COLORS = {
        "Tx-2_L_pos-1": "#e6194b",  # violet
        "Tx-2_L_pos-2": "#3cb44b",  # green
        "Tx-2_L_pos-3": "#4363d8",  # blue
        "Tx-2_L_pos-4": "#f58231",  # orange
        "Tx-2_L_pos-5": "#911eb4",  # pink
        "Tx-2_R_pos-1": "#42d4f4",  # red
        "Tx-2_R_pos-2": "#f032e6",  # magenta
        "Tx-2_R_pos-3": "#bfef45",  # brown
        "Tx-2_R_pos-4": "#fabed4",  # cyan
        "Tx-2_R_pos-5": "#a9a9a9",  # purple
    }
    color_map: Dict[int, np.ndarray] = {}
    for tx in selected:
        hex_col = POSITION_COLORS.get(tx.description, "#888888")
        r = int(hex_col[1:3], 16) / 255
        g = int(hex_col[3:5], 16) / 255
        b = int(hex_col[5:7], 16) / 255
        color_map[tx.index] = np.array([r, g, b, 1.0])

    # Pre-compute surface-snapped positions and outward normals for every tx.
    snapped: Dict[int, np.ndarray] = {}
    normals: Dict[int, np.ndarray] = {}
    n_sel = len(selected)
    for tx in selected:
        sp = _snap_tx_to_surface(tx.center, points)
        snapped[tx.index] = sp
        normals[tx.index] = _outward_normal(sp, mid)

    DISC_RADIUS_MM = 22.0   # all positions same physical size (~44 mm diameter)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), facecolor="white")

    for ax, (view_title, view_name) in zip(axes, four_views):
        ax.set_facecolor("#f0f0f0")
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(view_title, fontsize=11, fontweight="bold")

        # ── Mesh shading (depth-based grayscale) ─────────────────────────
        xs2d, ys2d, depth, tris_s = _mesh_silhouette_2d(points, tris, view_name)
        d_min, d_max = depth.min(), depth.max()
        d_norm = (depth - d_min) / (d_max - d_min + 1e-9)
        tri_d  = d_norm[tris_s].mean(axis=1)
        triang = _Tri(xs2d, ys2d, tris_s)
        ax.tripcolor(triang, facecolors=tri_d,
                     cmap="Greys_r", vmin=0.0, vmax=1.0,
                     edgecolors="none", rasterized=True)

        # ── Camera direction for this view ───────────────────────────────
        if   view_name == "left":   cam = np.array([-1., 0.,  0.])
        elif view_name == "right":  cam = np.array([ 1., 0.,  0.])
        elif view_name == "top":    cam = np.array([ 0., 0.,  1.])
        else:                       cam = np.array([ 0.,  1.,  0.])  # RAS +Y = anterior/face

        # ── Transducer discs ─────────────────────────────────────────────
        for tx in selected:
            sp    = snapped[tx.index]
            out_n = normals[tx.index]
            col   = color_map.get(tx.index, np.array([0.5, 0.5, 0.5, 1.0]))

            # Back-face culling: skip if outward normal faces away from camera.
            if float(np.dot(out_n, cam)) < 0.05:
                continue

            xs_r, ys_r = _project_disc_to_2d(sp, out_n, DISC_RADIUS_MM, view_name)
            # Each position slightly transparent (0.75) so overlapping ones show through
            ax.fill(xs_r, ys_r, color=col, alpha=0.75, zorder=4, linewidth=0)

        # ── Axis limits from the mesh projection ─────────────────────────
        pad = rng * 0.08
        ax.set_xlim(xs2d.min() - pad, xs2d.max() + pad)
        ax.set_ylim(ys2d.min() - pad, ys2d.max() + pad)

    # ── Legend ───────────────────────────────────────────────────────────
    from matplotlib.patches import Patch as _Patch
    legend_handles = []
    for tx in selected:
        col = color_map.get(tx.index, np.array([0.5, 0.5, 0.5, 1.0]))
        legend_handles.append(_Patch(facecolor=col, edgecolor="none",
                                     label=tx.description))
    if legend_handles:
        fig.legend(handles=legend_handles, loc="center left",
                   bbox_to_anchor=(0.88, 0.50), fontsize=8, frameon=True,
                   title="Planned positions", title_fontsize=8)

    fig.suptitle(f"{subject} | Head mesh with all planned transducer positions",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 0.87, 0.97])
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_mesh_heatmap(subject: str,
                      mesh_path: Path,
                      actual_groups: Dict[Tuple[str, str], List[TxMatrix]],
                      medoids: Dict[Tuple[str, str], TxMatrix],
                      condition: str,
                      out_path: Path,
                      view: str = "left"):
    points, tris, mid, rng = load_mesh(mesh_path)

    all_centers = []
    for side in SIDES:
        key = (condition, side)
        if key in actual_groups:
            all_centers.extend([t.center for t in actual_groups[key]])

    tri_centroids = points[tris].mean(axis=1)
    if all_centers:
        centers_arr = np.array(all_centers, dtype=float)
        diff    = tri_centroids[:, None, :] - centers_arr[None, :, :]
        sq_dist = np.sum(diff ** 2, axis=2)
        density = np.exp(-sq_dist / (15.0 ** 2)).sum(axis=1)
        density = (density - density.min()) / (density.max() - density.min() + 1e-9)
    else:
        density = np.zeros(len(tris), dtype=float)

    hot_rgba  = plt.cm.hot(density)
    tri_verts = points[tris]

    four_views = [
        ("Left / lateral",  "left"),
        ("Right / lateral", "right"),
        ("Front",           "front"),
        ("Top",             "top"),
    ]

    fig = plt.figure(figsize=(20, 5.5), facecolor="white")

    for i, (view_title, view_name) in enumerate(four_views):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        ax.set_facecolor("white")

        coll = Poly3DCollection(tri_verts, zsort="min")
        coll.set_facecolor(hot_rgba)
        coll.set_edgecolor("none")
        coll.set_alpha(1.0)
        ax.add_collection3d(coll)

        ax.set_xlim(mid[0] - rng, mid[0] + rng)
        ax.set_ylim(mid[1] - rng, mid[1] + rng)
        ax.set_zlim(mid[2] - rng, mid[2] + rng)
        set_mesh_view(ax, view_name)
        ax.set_axis_off()
        ax.set_title(view_title, color="black", fontsize=11, fontweight="bold")

    sm = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap="hot")
    sm.set_array([])
    cbar_ax = fig.add_axes([0.905, 0.18, 0.012, 0.60])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Position density\n(brighter = more frequent)", fontsize=8, color="black")
    cb.ax.tick_params(labelsize=7, colors="black")
    cb.ax.yaxis.set_tick_params(color="black")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="black")

    fig.suptitle(
        f"{subject} | {COND_TITLE[condition]} | "
        f"Actual transducer position heatmap (all recorded frames)",
        fontsize=13, fontweight="bold", y=1.02, color="black",
    )
    fig.tight_layout(rect=[0, 0, 0.90, 0.97])
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ------------------------- report HTML -------------------------

def relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)


def write_html_report(subject: str, outdir: Path, figure_paths: List[Path], table_paths: List[Path], notes: List[str]):
    html_path = outdir / f"{subject}_citrus_offline_report.html"
    fig_html = []
    for p in figure_paths:
        rp = html.escape(relpath(p, outdir))
        title = html.escape(p.stem)
        fig_html.append(f"<section><h3>{title}</h3><img src='{rp}' alt='{title}'></section>")
    table_html = []
    for p in table_paths:
        try:
            df = pd.read_csv(p)
            table_html.append(f"<section><h3>{html.escape(p.name)}</h3>{df.head(200).to_html(index=False, escape=True)}</section>")
        except Exception:
            table_html.append(f"<p><a href='{html.escape(relpath(p, outdir))}'>{html.escape(p.name)}</a></p>")
    notes_html = "".join(f"<li>{html.escape(n)}</li>" for n in notes)
    page = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(subject)} CITRUS offline report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 28px; line-height: 1.45; color: #222; }}
h1 {{ border-bottom: 3px solid #333; padding-bottom: 8px; }}
img {{ max-width: 100%; border: 1px solid #ddd; box-shadow: 0 1px 5px rgba(0,0,0,.15); }}
section {{ margin: 28px 0; }}
table {{ border-collapse: collapse; font-size: 12px; max-width: 100%; overflow-x: auto; display: block; }}
th, td {{ border: 1px solid #ccc; padding: 4px 6px; }}
th {{ background: #f3f3f3; }}
.note {{ background: #fff9db; border: 1px solid #f0d36c; padding: 12px; }}
</style>
</head>
<body>
<h1>{html.escape(subject)} CITRUS Offline Simulation and Post-hoc Report</h1>
<div class="note"><strong>Notes</strong><ul>{notes_html}</ul></div>
<h2>Figures</h2>
{''.join(fig_html)}
<h2>Tables</h2>
{''.join(table_html)}
</body>
</html>
"""
    html_path.write_text(page, encoding="utf-8")
    return html_path


# ------------------------- CLI -------------------------

def add_map_args(p: argparse.ArgumentParser, prefix: str):
    for cond in CONDS:
        for side in SIDES:
            p.add_argument(f"--{prefix}-{cond}-{side}", default=None,
                           help=f"{prefix.replace('-', ' ')} map for {cond} {side}")


def get_arg(args, name: str):
    return getattr(args, name.replace("-", "_"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate CITRUS offline simulation and post-hoc planned-vs-actual report for one subject."
    )
    p.add_argument("--version", action="version", version=SCRIPT_VERSION)
    p.add_argument("--subject", required=True)
    p.add_argument("--t1w", required=True)
    p.add_argument("--segmentation", default=None)
    p.add_argument("--mesh", default=None)
    p.add_argument("--sgacc-left", required=True)
    p.add_argument("--sgacc-right", required=True)

    p.add_argument("--planned-xml", required=True)
    p.add_argument("--actual-exp-xml", required=True)
    p.add_argument("--actual-con-xml", default=None)
    p.add_argument("--conditions", nargs="+", choices=["exp", "con"], default=["exp", "con"],
                   help="Which conditions to process. Use --conditions exp to skip control.")
    p.add_argument("--planned-left-label", default=None)
    p.add_argument("--planned-right-label", default=None)
    p.add_argument("--planned-left-index", type=int, default=None)
    p.add_argument("--planned-right-index", type=int, default=None)

    # Frame ranges are inclusive, by XML Element index.
    for cond in CONDS:
        for side in SIDES:
            p.add_argument(f"--{cond}-{side}-range", nargs=2, type=int, default=None,
                           metavar=("START", "END"), help=f"Inclusive actual XML frame index range for {cond} {side}")
            p.add_argument(f"--actual-{cond}-{side}-medoid-frame", type=int, default=None,
                           help=f"Manual medoid frame override for {cond} {side}; otherwise all-points medoid is used")
            p.add_argument(f"--{cond}-{side}-label-filter", default=None,
                           help=f"Optional label substring filter for actual frames in {cond} {side}")

    add_map_args(p, "planned-pressure")
    add_map_args(p, "actual-pressure")
    add_map_args(p, "planned-temperature")
    add_map_args(p, "actual-temperature")

    p.add_argument("--mosaic-z", nargs="+", type=int, default=None,
                   help="Axial z slice indices for pressure/temperature mosaics.")
    p.add_argument("--mosaic-x", nargs="+", type=int, default=None,
                   help="Sagittal x slice indices for pressure/temperature mosaics.")
    p.add_argument("--mosaic-y", nargs="+", type=int, default=None,
                   help="Coronal y slice indices for pressure/temperature mosaics.")
    p.add_argument("--outdir", default="reports")
    p.add_argument("--target-threshold", type=float, default=0.5)
    p.add_argument("--focal-search-radius-mm", type=float, default=25.0,
                   help="Search radius around sgACC centroid for local brain/focal peak (default 25 mm keeps the search inside brain for deep targets like sgACC)")
    p.add_argument("--pressure-vmin", type=float, default=0.0)
    p.add_argument("--pressure-vmax", type=float, default=None,
                   help="Manual pressure colorbar maximum; otherwise shared percentile is used")
    p.add_argument("--pressure-percentile", type=float, default=99.9)
    p.add_argument("--temperature-vmin", type=float, default=37.0)
    p.add_argument("--temperature-vmax", type=float, default=None,
                   help="Manual absolute temperature colorbar maximum; otherwise shared percentile is used")
    p.add_argument("--temperature-percentile", type=float, default=99.9)
    p.add_argument("--crop-vox", type=int, default=80)
    p.add_argument("--mesh-view", choices=["left", "right", "front", "top"], default="left")
    p.add_argument("--scalp-tag", type=int, default=1005)
    p.add_argument("--skip-mesh", action="store_true")
    p.add_argument("--skip-mosaics", action="store_true")
    p.add_argument("--skip-html", action="store_true")
    return p


def path_for_map(args, kind: str, cond: str, side: str, required: bool = False) -> Optional[Path]:
    name = f"{kind}_{cond}_{side}".replace("-", "_")
    val = getattr(args, name, None)
    return ensure_path(val, f"--{kind}-{cond}-{side}", required=required)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    active_conds = list(dict.fromkeys(args.conditions))
    subject = args.subject
    outdir = Path(args.outdir).expanduser() / subject
    figdir = outdir / "figures"
    tabdir = outdir / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)
    figure_paths: List[Path] = []
    table_paths: List[Path] = []
    notes: List[str] = []

    # Load core images
    t1_path = ensure_path(args.t1w, "--t1w")
    seg_path = ensure_path(args.segmentation, "--segmentation", required=False)
    mesh_path = ensure_path(args.mesh, "--mesh", required=False)
    sg_l_path = ensure_path(args.sgacc_left, "--sgacc-left")
    sg_r_path = ensure_path(args.sgacc_right, "--sgacc-right")
    planned_xml_path = ensure_path(args.planned_xml, "--planned-xml")
    actual_exp_xml_path = ensure_path(args.actual_exp_xml, "--actual-exp-xml")
    if "con" in active_conds:
        actual_con_xml_path = ensure_path(args.actual_con_xml, "--actual-con-xml")
    else:
        actual_con_xml_path = None

    t1_img = load_img(t1_path)
    seg_img = load_img(seg_path) if seg_path is not None else None
    sgacc_imgs = {"left": load_img(sg_l_path), "right": load_img(sg_r_path)}

    # Anatomy figure
    anatomy_path = figdir / f"{subject}_01_anatomy_segmentation_sgacc.png"
    plot_anatomy(subject, t1_img, seg_img, sgacc_imgs["left"], sgacc_imgs["right"], anatomy_path, args.target_threshold)
    figure_paths.append(anatomy_path)

    # XML parsing and transducer groups
    planned_txs = parse_gummarkers(planned_xml_path)
    actual_xmls = {
        "exp": parse_gummarkers(actual_exp_xml_path),
    }
    if "con" in active_conds:
        actual_xmls["con"] = parse_gummarkers(actual_con_xml_path)
    planned = {
        "left": select_tx_by_index_or_label(planned_txs, args.planned_left_index, args.planned_left_label, "left"),
        "right": select_tx_by_index_or_label(planned_txs, args.planned_right_index, args.planned_right_label, "right"),
    }

    actual_groups: Dict[Tuple[str, str], List[TxMatrix]] = {}
    medoids: Dict[Tuple[str, str], TxMatrix] = {}
    drift_tables = []
    drift_summary_rows = []
    medoid_rows = []
    dfs_dict: Dict[Tuple[str, str], pd.DataFrame] = {}
    for cond in active_conds:
        for side in SIDES:
            frame_range = getattr(args, f"{cond}_{side}_range")
            label_filter = getattr(args, f"{cond}_{side}_label_filter")
            group = select_range(actual_xmls[cond], frame_range, label_filter)
            actual_groups[(cond, side)] = group
            manual = getattr(args, f"actual_{cond}_{side}_medoid_frame")
            if manual is not None:
                matches = [t for t in group if t.index == manual]
                if not matches:
                    raise ValueError(f"Manual medoid frame {manual} for {cond} {side} is not inside selected range")
                med = matches[0]
                method = "manual"
            else:
                med = all_points_medoid(group)
                method = "all_points_medoid"
            medoids[(cond, side)] = med
            df = drift_dataframe(subject, cond, side, group, planned[side], med)
            drift_tables.append(df)
            summ = drift_summary(df)
            summ.update({"subject": subject, "condition": cond, "side": side, "medoid_frame": med.index, "medoid_method": method})
            drift_summary_rows.append(summ)
            medoid_rows.append({
                "subject": subject, "condition": cond, "side": side, "medoid_frame": med.index,
                "medoid_method": method, "medoid_x": med.center[0], "medoid_y": med.center[1], "medoid_z": med.center[2],
                "planned_frame": planned[side].index, "planned_description": planned[side].description,
                "planned_x": planned[side].center[0], "planned_y": planned[side].center[1], "planned_z": planned[side].center[2],
                "medoid_to_planned_mm": float(np.linalg.norm(med.center - planned[side].center)),
            })
            dfs_dict[(cond, side)] = df

    # Generate combined drift stability plots for Left and Right sides
    for side in SIDES:
        df_exp = dfs_dict.get(("exp", side))
        df_con = dfs_dict.get(("con", side))
        if df_exp is not None:
            drift_fig = figdir / f"{subject}_drift_combined_{side}.png"
            plot_drift_combined(df_exp, df_con, subject, side, drift_fig)
            figure_paths.append(drift_fig)

    drift_all = pd.concat(drift_tables, ignore_index=True)
    drift_csv = tabdir / f"{subject}_transducer_drift_all_frames.csv"
    drift_all.to_csv(drift_csv, index=False)
    table_paths.append(drift_csv)
    drift_summary_csv = tabdir / f"{subject}_transducer_drift_summary.csv"
    pd.DataFrame(drift_summary_rows).to_csv(drift_summary_csv, index=False)
    table_paths.append(drift_summary_csv)
    medoid_csv = tabdir / f"{subject}_selected_medoid_frames.csv"
    pd.DataFrame(medoid_rows).to_csv(medoid_csv, index=False)
    table_paths.append(medoid_csv)

    # Mesh figures if available
    # Mesh figures if available
    if mesh_path is not None and not args.skip_mesh:
        # Each figure in its own try block so one failure doesn't kill the rest.
        try:
            planned_all_path = figdir / f"{subject}_02_headmesh_all_planned_positions.png"
            plot_mesh_planned_all_positions(
                subject=subject, mesh_path=mesh_path, planned_txs=planned_txs,
                out_path=planned_all_path, view=args.mesh_view,
            )
            figure_paths.append(planned_all_path)
        except Exception as e:
            log.exception("plot_mesh_planned_all_positions failed"); notes.append(f"Planned mesh failed: {e}")

        try:
            mesh_pos_path = figdir / f"{subject}_03_headmesh_planned_actual_positions.png"
            plot_mesh_positions(
                subject, mesh_path, planned, actual_groups, medoids,
                mesh_pos_path, view=args.mesh_view,
            )
            figure_paths.append(mesh_pos_path)
        except Exception as e:
            log.exception("plot_mesh_positions failed"); notes.append(f"Planned+actual mesh failed: {e}")

        for cond in active_conds:
            try:
                hp = figdir / f"{subject}_04_headmesh_actual_position_heatmap_{cond}.png"
                plot_mesh_heatmap(
                    subject, mesh_path, actual_groups, medoids, cond, hp, view=args.mesh_view,
                )
                figure_paths.append(hp)
            except Exception as e:
                log.exception("plot_mesh_heatmap failed for %s", cond); notes.append(f"Heatmap {cond} failed: {e}")
    else:
        notes.append("Mesh figures skipped or no mesh file provided.")

    # Load maps and determine shared color ranges.
    pressure_data_for_range = []
    temperature_data_for_range = []
    available_cases = []
    for cond in active_conds:
        for side in SIDES:
            pplan = path_for_map(args, "planned-pressure", cond, side, required=False)
            pact = path_for_map(args, "actual-pressure", cond, side, required=False)
            tplan = path_for_map(args, "planned-temperature", cond, side, required=False)
            tact = path_for_map(args, "actual-temperature", cond, side, required=False)
            if all(x is not None for x in [pplan, pact, tplan, tact]):
                available_cases.append((cond, side, pplan, pact, tplan, tact))
                pressure_data_for_range.append(load_pressure_mpa(pplan))
                pressure_data_for_range.append(load_pressure_mpa(pact))
                temperature_data_for_range.append(img_data(load_img(tplan)))
                temperature_data_for_range.append(img_data(load_img(tact)))
            else:
                notes.append(f"Missing one or more pressure/temperature maps for {cond} {side}; skipped simulation figures for this case.")

    if not available_cases:
        raise ValueError("No complete planned/actual pressure/temperature map sets were provided.")

    pressure_vmin = args.pressure_vmin
    #pressure_vmax = args.pressure_vmax if args.pressure_vmax is not None else percentile_nonzero(pressure_data_for_range, args.pressure_percentile, default=1.0)
    temperature_vmin = args.temperature_vmin
    #temperature_vmax = args.temperature_vmax if args.temperature_vmax is not None else percentile_nonzero(temperature_data_for_range, args.temperature_percentile, default=temperature_vmin + 1.0)
    if args.pressure_vmax is not None:
        pressure_vmax = args.pressure_vmax
    else:
        pressure_vmax = max(float(np.nanmax(x)) for x in pressure_data_for_range)

    if args.temperature_vmax is not None:
        temperature_vmax = args.temperature_vmax
    else:
        temperature_vmax = max(float(np.nanmax(x)) for x in temperature_data_for_range)
    if temperature_vmax <= temperature_vmin:
        temperature_vmax = temperature_vmin + 1.0
    notes.append(f"Pressure color range: {pressure_vmin:.3g} to {pressure_vmax:.3g} MPa. True maxima are saved in CSV tables.")
    notes.append(f"Temperature color range: {temperature_vmin:.3g} to {temperature_vmax:.3g} C. True maxima are saved in CSV tables.")
    notes.append("-3 dB focal volumes are computed from the local pressure peak near sgACC, not from the skull/global peak.")
    notes.append("Actual representative transducer position defaults to all-points medoid, so no recorded positions are discarded as DBSCAN noise.")

    focal_rows = []
    sim_rows = []

    for cond, side, pplan_path, pact_path, tplan_path, tact_path in available_cases:
        log.info("Processing maps: %s %s", cond, side)
        pplan_img = load_img(pplan_path)
        pact_img = load_img(pact_path)
        tplan_img = load_img(tplan_path)
        tact_img = load_img(tact_path)
        ref_img = pplan_img

        # Resample all relevant images to planned pressure grid.
        t1_ref = resample_to_target(t1_img, ref_img, order=1)
        sg_ref = resample_to_target(sgacc_imgs[side], ref_img, order=0)
        pplan = pressure_to_mpa(img_data(pplan_img))
        pact = pressure_to_mpa(resample_to_target(pact_img, ref_img, order=1))
        tplan = resample_to_target(tplan_img, ref_img, order=1)
        tact = resample_to_target(tact_img, ref_img, order=1)
        from scipy.ndimage import binary_erosion as _bin_erode, binary_fill_holes as _fill_holes
        # Build a conservative brain-tissue mask to exclude skull/scalp/air from
        # focal-zone peak detection.  Two strategies depending on available inputs.
        if seg_img is not None:
            seg_ref = resample_to_target(seg_img, ref_img, order=0)
            # Restrict to GM/WM/CSF labels (1–3 in fMRIPrep/FAST).
            # Skull/background are label 0 or >3; dura/meninges may be label 4+.
            seg_int = np.round(seg_ref).astype(int)
            brain_mask = (seg_int >= 1) & (seg_int <= 3)
            # Erode 2 voxels to shed partial-volume skull/boundary contamination.
            brain_mask = _bin_erode(brain_mask, iterations=2)
        else:
            # No segmentation: build a soft-tissue mask from T1 intensity.
            # In a skull-stripped T1 the background is 0; brain is the largest
            # connected region of non-zero voxels.  We threshold at p20 (lower
            # than before) and fill holes so CSF spaces are included, then erode.
            t1_finite = t1_ref[np.isfinite(t1_ref) & (t1_ref > 0)]
            # p20 of positive voxels keeps brain tissue and excludes air gaps.
            t1_lo = float(np.percentile(t1_finite, 20)) if t1_finite.size > 0 else 0.0
            # p98 ceiling removes very bright skull bone (compact bone appears
            # bright in T1 and is the dominant source of skull pressure artifacts).
            t1_hi = float(np.percentile(t1_finite, 98)) if t1_finite.size > 0 else np.inf
            brain_mask = (t1_ref > t1_lo) & (t1_ref < t1_hi)
            # Fill interior holes (ventricles, CSF pockets) and then erode.
            brain_mask = _fill_holes(brain_mask)
            brain_mask = _bin_erode(brain_mask, iterations=2)

        center_vox = mask_centroid_vox(sg_ref, args.target_threshold)
        center_world = voxel_to_world(center_vox, ref_img.affine)

        # Whole-head mosaics
        if not args.skip_mosaics:
            for kind, vol, cmap, vmin, vmax, unit, label in [
                ("planned_pressure", pplan, PRESSURE_CMAP, pressure_vmin, pressure_vmax, "Pressure (MPa)", "planning pressure"),
                ("actual_pressure", pact, PRESSURE_CMAP, pressure_vmin, pressure_vmax, "Pressure (MPa)", "post-hoc pressure"),
                ("planned_temperature", tplan, TEMP_CMAP, temperature_vmin, temperature_vmax, "Temperature (C)", "planning temperature"),
                ("actual_temperature", tact, TEMP_CMAP, temperature_vmin, temperature_vmax, "Temperature (C)", "post-hoc temperature"),
            ]:
                op = figdir / f"{subject}_{kind}_{cond}_{side}_mosaic.png"
                plot_map_mosaic(subject, f"{COND_TITLE[cond]} {SIDE_TITLE[side]} {label}", t1_ref, vol, sg_ref,
                                center_vox, cmap, vmin, vmax, unit, op,
                                mosaic_x=args.mosaic_x, mosaic_y=args.mosaic_y, mosaic_z=args.mosaic_z)
                figure_paths.append(op)

        metrics, aux = compute_focal_metrics(
            subject=subject,
            condition=cond,
            side=side,
            planned_pressure=pplan,
            actual_pressure=pact,
            ref_img=ref_img,
            sgacc_mask=sg_ref,
            sgacc_center_world=center_world,
            search_radius_mm=args.focal_search_radius_mm,
            brain_mask=brain_mask,
        )
        focal_rows.append(metrics.__dict__)

        # Planned vs actual pressure: overlay (planned=blue, actual=orange, overlap=green).
        pressure_comp = figdir / f"{subject}_planned_vs_actual_pressure_{cond}_{side}.png"
        plot_planned_actual_focal_overlay(
            subject, cond, side, t1_ref, sg_ref, center_vox,
            aux["planned_focal"], aux["actual_focal"], metrics, pressure_comp,
            crop_vox=args.crop_vox,
            planned_com_vox=aux.get("planned_com_vox"),
            actual_com_vox=aux.get("actual_com_vox"),
        )
        figure_paths.append(pressure_comp)
        # NOTE: planned_vs_actual_temperature and focal_overlap figures are intentionally omitted.

        # Summary values for simulation table
        sgmask = sg_ref > args.target_threshold
        for stage, pvol, tvol in [("planned", pplan, tplan), ("actual", pact, tact)]:
            sim_rows.append({
                "subject": subject,
                "condition": cond,
                "side": side,
                "stage": stage,
                "pressure_global_max_mpa": float(np.nanmax(pvol)),
                "pressure_sgacc_max_mpa": float(np.nanmax(pvol[sgmask])) if np.any(sgmask) else np.nan,
                "pressure_sgacc_mean_mpa": float(np.nanmean(pvol[sgmask])) if np.any(sgmask) else np.nan,
                "temperature_global_max_c": float(np.nanmax(tvol)),
                "temperature_sgacc_max_c": float(np.nanmax(tvol[sgmask])) if np.any(sgmask) else np.nan,
                "temperature_sgacc_mean_c": float(np.nanmean(tvol[sgmask])) if np.any(sgmask) else np.nan,
                "sgacc_centroid_world_x": center_world[0],
                "sgacc_centroid_world_y": center_world[1],
                "sgacc_centroid_world_z": center_world[2],
            })

    sim_csv = tabdir / f"{subject}_simulation_summary.csv"
    pd.DataFrame(sim_rows).to_csv(sim_csv, index=False)
    table_paths.append(sim_csv)
    focal_csv = tabdir / f"{subject}_focal_overlap_metrics.csv"
    pd.DataFrame(focal_rows).to_csv(focal_csv, index=False)
    table_paths.append(focal_csv)

    html_path = None
    if not args.skip_html:
        html_path = write_html_report(subject, outdir, figure_paths, table_paths, notes)

    print("\n" + "=" * 78)
    print(f"CITRUS offline report completed for {subject}")
    print("=" * 78)
    print(f"Output folder: {outdir}")
    print(f"Figures:       {figdir}")
    print(f"Tables:        {tabdir}")
    if html_path:
        print(f"HTML report:   {html_path}")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
