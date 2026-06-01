"""
Extract, analyze, and visualize actual (post-hoc) pressure and temperature maps.

This script processes actual post-hoc simulation maps in NIfTI format. It resamples them
to a unified grid, identifies the actual -3 dB focal zone within a conservative
brain mask, and renders multi-view slice mosaics of actual pressure and temperature.

@author Hoai Thu Nguyen
"""

import os
import glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates, label as nd_label, center_of_mass
from scipy.ndimage import binary_erosion as _bin_erode, binary_fill_holes as _fill_holes

# Set up logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Tx_actual_maps")

# ── constants ────────────────────────────────────────────────────────────────

SIDES = ["left", "right"]
CONDS = ["exp", "con"]
SIDE_TITLE = {"left": "Left", "right": "Right"}
COND_TITLE = {"exp": "Experimental", "con": "Control"}

PRESSURE_CMAP = "turbo"
TEMP_CMAP = "hot"
WHITE = "#ffffff"
ACTUAL  = "#00d4ff"  # cyan        - actual focal zone

MINUS3_AMP = 10 ** (-3.0 / 20.0)  # 0.7079 amplitude ratio for -3 dB power half-maximum


# ── general NIfTI utility functions ──────────────────────────────────────────

def ensure_path(path: Optional[str], label: str, required: bool = True) -> Optional[Path]:
    """
    Convert a string path to a Path object and check if it exists.

    Args:
        path (str): String file path.
        label (str): Argument name or description for logging.
        required (bool): If True, raise FileNotFoundError if missing.

    Returns:
        Optional[Path]: Absolute Path object, or None if optional and missing.
    """
    if path is None or str(path).strip() == "":
        if required:
            raise FileNotFoundError(f"Missing required path: {label}")
        return None
    p = Path(path).expanduser()
    if not p.exists():
        if required:
            raise FileNotFoundError(f"File not found for {label}: {p}")
        log.warning(f"Optional file not found for {label}: {p}")
        return None
    return p


def load_img(path: Path) -> nib.Nifti1Image:
    """Load a NIfTI-1 image file."""
    return nib.load(str(path))


def img_data(img: nib.Nifti1Image, dtype=np.float32) -> np.ndarray:
    """Extract floating-point voxel data from a NIfTI image."""
    return img.get_fdata(dtype=dtype)


def voxel_volume_mm3(img: nib.Nifti1Image) -> float:
    """Calculate the cubic volume of a single voxel in cubic millimeters."""
    return float(abs(np.linalg.det(img.affine[:3, :3])))


def voxel_to_world(vox_xyz: Sequence[float], affine: np.ndarray) -> np.ndarray:
    """
    Transform 3-D integer voxel index coordinates to 3-D world coordinate space (RAS mm).

    Args:
        vox_xyz (Sequence[float]): Voxel index coordinates.
        affine (np.ndarray): 4x4 coordinate transform matrix.

    Returns:
        np.ndarray: 3-D spatial coordinate vector.
    """
    return (affine @ np.array([vox_xyz[0], vox_xyz[1], vox_xyz[2], 1.0], dtype=float))[:3]


def resample_to_target(src_img: nib.Nifti1Image, tgt_img: nib.Nifti1Image, order: int = 1) -> np.ndarray:
    """
    Resample a source NIfTI volume to match a target volume's voxel grid.

    Args:
        src_img (nib.Nifti1Image): Source volume image.
        tgt_img (nib.Nifti1Image): Target destination volume.
        order (int): Trilinear (1) or nearest-neighbor (0) interpolation.

    Returns:
        np.ndarray: Resampled source volume voxel grid array.
    """
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


# ── rendering and cropping helpers ───────────────────────────────────────────

def robust_t1_limits(t1: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the 2nd and 98th intensity percentiles for robust grayscale scaling.

    Args:
        t1 (np.ndarray): Background brain image voxel array.

    Returns:
        Tuple[float, float]: Lower and upper clip limits.
    """
    finite = t1[np.isfinite(t1) & (t1 > 0)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, 2)), float(np.percentile(finite, 98))


def mask_centroid_vox(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Calculate the center-of-mass voxel coordinate for a binary mask.

    Args:
        mask (np.ndarray): Voxel mask volume.
        threshold (float): Threshold for binarization.

    Returns:
        np.ndarray: 3-D centroid coordinate.
    """
    binary = mask > threshold
    if not np.any(binary):
        return np.array(mask.shape, dtype=float) / 2.0
    coords = np.argwhere(binary)
    return coords.mean(axis=0)


def slice2d(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    """
    Extract a 2-D slice from a 3-D volume and transpose it for standard viewing orientation.

    Args:
        vol (np.ndarray): 3-D voxel array.
        axis (int): Projection axis (0: Sagittal, 1: Coronal, 2: Axial).
        idx (int): Slice index.

    Returns:
        np.ndarray: Transposed 2-D slice.
    """
    idx = int(np.clip(idx, 0, vol.shape[axis] - 1))
    if axis == 2:      # axial z: x/y plane
        return vol[:, :, idx].T
    if axis == 1:      # coronal y: x/z plane
        return vol[:, idx, :].T
    if axis == 0:      # sagittal x: y/z plane
        return vol[idx, :, :].T
    raise ValueError("axis must be 0, 1, or 2")


def point_xy(vox_xyz: Sequence[float], axis: int) -> Tuple[float, float]:
    """
    Extract 2-D coordinates from a 3-D voxel coordinate depending on projection axis.

    Args:
        vox_xyz (Sequence[float]): 3-D voxel coordinate.
        axis (int): Slice projection direction.

    Returns:
        Tuple[float, float]: Projected coordinates.
    """
    x, y, z = vox_xyz
    if axis == 2:
        return float(x), float(y)
    if axis == 1:
        return float(x), float(z)
    if axis == 0:
        return float(y), float(z)
    raise ValueError("axis must be 0, 1, or 2")


def crop_limits(shape2d: Tuple[int, int], center_xy: Tuple[float, float], half_width_vox: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute rectangular slice index crop limits around a 2-D center point.

    Args:
        shape2d (Tuple[int, int]): Size of 2-D canvas.
        center_xy (Tuple[float, float]): 2-D coordinate centroid.
        half_width_vox (int): Crop box radius size in pixels.

    Returns:
        Optional[Tuple[int, int, int, int]]: Bounding coordinates (x0, x1, y0, y1), or None.
    """
    if half_width_vox <= 0:
        return None
    cx, cy = center_xy
    x0 = max(0, int(round(cx)) - half_width_vox)
    x1 = min(shape2d[1], int(round(cx)) + half_width_vox)
    y0 = max(0, int(round(cy)) - half_width_vox)
    y1 = min(shape2d[0], int(round(cy)) + half_width_vox)
    return x0, x1, y0, y1


def apply_crop(arr2d: np.ndarray, lim: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    """Crop a 2-D slice with bounding limits."""
    if lim is None:
        return arr2d
    x0, x1, y0, y1 = lim
    return arr2d[y0:y1, x0:x1]


def safe_contour(ax, arr2d: np.ndarray, level: float, color: str, linestyle: str = "solid", linewidth: float = 1.5, alpha: float = 1.0):
    """Safely draw a single contour line on matplotlib axes without crashing."""
    a = np.asarray(arr2d, dtype=float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return None
    if not (float(np.nanmin(finite)) <= level <= float(np.nanmax(finite))):
        return None
    return ax.contour(a, levels=[level], colors=[color], linestyles=linestyle, linewidths=linewidth, alpha=alpha)


def safe_imshow_overlay(ax, arr2d: np.ndarray, vmin: float, vmax: float, cmap: str, alpha: float = 0.55):
    """Overlay an intensity map on matplotlib axes safely ignoring invalid values."""
    arr = np.asarray(arr2d, dtype=float)
    masked = np.ma.masked_where(~np.isfinite(arr) | (arr <= vmin), arr)
    if masked.count() == 0:
        return None
    return ax.imshow(masked, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, interpolation="nearest")


def add_lr_labels(ax, axis: int, color: str = "white"):
    """Annotate Left (L) and Right (R) anatomical labels on axial or coronal slices."""
    if axis in (1, 2):
        ax.text(0.02, 0.50, "L", transform=ax.transAxes, color=color, fontsize=11,
                fontweight="bold", va="center", ha="left",
                bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1.5))
        ax.text(0.98, 0.50, "R", transform=ax.transAxes, color=color, fontsize=11,
                fontweight="bold", va="center", ha="right",
                bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1.5))


# ── map processing & focal analytics ─────────────────────────────────────────

def pressure_to_mpa(arr: np.ndarray) -> np.ndarray:
    """
    Convert pressure array values to MPa. Automatically scales down values if recorded in Pa.

    Args:
        arr (np.ndarray): Raw pressure array.

    Returns:
        np.ndarray: Scaled MPa pressure array.
    """
    x = np.asarray(arr, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if finite.size and float(np.nanpercentile(np.abs(finite), 99.9)) > 1000.0:
        return (x / 1_000_000.0).astype(np.float32)
    return x.astype(np.float32)


def sphere_mask_around_world(ref_img: nib.Nifti1Image, center_world: np.ndarray, radius_mm: float) -> np.ndarray:
    """
    Generate a 3-D sphere boundary mask in voxel grid space using millimeter coordinate distance.

    Args:
        ref_img (nib.Nifti1Image): Reference grid template.
        center_world (np.ndarray): 3-D RAS coordinates centroid.
        radius_mm (float): Millimeter size radius of sphere boundary.

    Returns:
        np.ndarray: Binary mask array.
    """
    shape = ref_img.shape[:3]
    ijk = np.indices(shape, dtype=float).reshape(3, -1).T
    world = nib.affines.apply_affine(ref_img.affine, ijk)
    dist = np.linalg.norm(world - center_world[None, :], axis=1)
    return (dist <= radius_mm).reshape(shape)


def largest_component_containing_peak(binary: np.ndarray, peak_vox: np.ndarray) -> np.ndarray:
    """
    Extract the largest connected component of non-zero voxels that contains a peak coordinate.

    Args:
        binary (np.ndarray): Binary mask array.
        peak_vox (np.ndarray): Voxel coordinate of peak.

    Returns:
        np.ndarray: Single connected component mask.
    """
    lab, n = nd_label(binary.astype(bool))
    if n == 0:
        return np.zeros_like(binary, dtype=bool)
    peak_idx = tuple(int(round(v)) for v in peak_vox)
    peak_idx = tuple(np.clip(peak_idx[i], 0, binary.shape[i] - 1) for i in range(3))
    peak_label = int(lab[peak_idx])
    if peak_label > 0:
        return lab == peak_label
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
    """
    Extract the connected -3 dB focal volume zone of pressure near the sgACC ROI target.

    Args:
        pressure (np.ndarray): Pressure MPa voxel array.
        ref_img (nib.Nifti1Image): Spatial coordinate template image.
        sgacc_mask (np.ndarray): sgACC target volume.
        sgacc_center_world (np.ndarray): centroid target RAS coordinates.
        search_radius_mm (float): Millimeter boundary range.
        brain_mask (Optional[np.ndarray]): Soft tissue mask.

    Returns:
        Tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
            - Binary focal zone array.
            - Peak focal value.
            - -3 dB amplitude threshold value.
            - Peak voxel coordinate.
            - Center-of-mass voxel coordinate.
    """
    search = sphere_mask_around_world(ref_img, sgacc_center_world, search_radius_mm)
    candidate = search & np.isfinite(pressure) & (pressure > 0)

    use_brain = brain_mask is not None
    if use_brain:
        c_brain = candidate & brain_mask.astype(bool)
        if np.any(c_brain):
            candidate = c_brain
        else:
            use_brain = False

    if not np.any(candidate):
        raise ValueError("No positive pressure voxels found inside targeted sphere.")

    masked_p = np.where(candidate, pressure, -np.inf)
    peak_vox_int = np.array(np.unravel_index(np.argmax(masked_p), pressure.shape), dtype=float)
    peak = float(pressure[tuple(peak_vox_int.astype(int))])
    thr  = peak * MINUS3_AMP

    binary = (pressure >= thr) & search
    if use_brain and brain_mask is not None:
        binary &= brain_mask.astype(bool)

    focal = largest_component_containing_peak(binary, peak_vox_int)
    if not np.any(focal):
        focal = binary

    com_vox = np.array(center_of_mass(focal.astype(float)), dtype=float)
    if not np.all(np.isfinite(com_vox)):
        com_vox = peak_vox_int.copy()
    return focal.astype(bool), peak, thr, peak_vox_int, com_vox


# ── plotting & visualization ─────────────────────────────────────────────────

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
    """
    Generate and save a 3-row orthogonal slice grid mosaic for an individual map.

    Args:
        subject (str): Subject identifier.
        title (str): Suptitle title label.
        t1 (np.ndarray): T1w anatomical background array.
        mapvol (np.ndarray): Pressure or temperature volume.
        sgacc (np.ndarray): sgACC target volume.
        center_vox (np.ndarray): 3-D centroid coordinates of target.
        cmap (str): Color map name.
        vmin (float): Color minimum bound.
        vmax (float): Color maximum bound.
        unit (str): Unit string (e.g. MPa, C).
        out_path (Path): Output filename path to save PNG figure.
        overlay_alpha (float): Overlay transparency.
        mosaic_x (Optional[Sequence[int]]): Sagittal slice indices.
        mosaic_y (Optional[Sequence[int]]): Coronal slice indices.
        mosaic_z (Optional[Sequence[int]]): Axial slice indices.
    """
    axes_order = [(2, "z"), (0, "x"), (1, "y")]
    true_vmax = vmax
    true_vmin = vmin
    peak_val = float(np.nanmax(mapvol)) if mapvol.size else true_vmax

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

    fig.subplots_adjust(left=0.02, right=0.89, top=0.90, bottom=0.04, wspace=0.02, hspace=0.08)
    cax = fig.add_axes([0.905, 0.12, 0.016, 0.70])
    sm = cm.ScalarMappable(norm=Normalize(vmin=true_vmin, vmax=true_vmax), cmap=cmap)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(colors="white", labelsize=8)
    cbar.set_label(unit, color="white", fontsize=9)
    cbar.ax.text(0.5, 1.02, f"max\n{peak_val:.3g}", transform=cbar.ax.transAxes,
                 color="white", fontsize=7, ha="center", va="bottom", fontweight="bold")

    fig.suptitle(f"{subject} | {title}", color="white", fontsize=12, fontweight="bold")
    fig.savefig(out_path, dpi=220, facecolor="black", bbox_inches="tight")
    plt.close(fig)


# ── map finding logic ────────────────────────────────────────────────────────

def find_map_file(dir_path: Path, side_letter: str, map_type: str) -> Optional[Path]:
    """
    Search robustly in a directory using a wildcard pattern for the specific transducer mapping files.

    Args:
        dir_path (Path): Directory folder.
        side_letter (str): Side code ('L' or 'R').
        map_type (str): Map category ('Pressure' or 'Temperature').

    Returns:
        Optional[Path]: Located filename Path, or None.
    """
    if not dir_path.exists():
        return None
    pattern = f"*Tx-2_{side_letter}_pos-* - {map_type}.nii.gz"
    files = list(dir_path.glob(pattern))
    if files:
        return files[0]
    return None


# ── main block ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_DIR = os.path.join(BASE, "data", "input")
    OUTPUT_DIR = os.path.join(BASE, "data", "output")
    DERIVATIVES_DIR = os.path.join(BASE, "derivatives", "actual_maps")

    os.makedirs(DERIVATIVES_DIR, exist_ok=True)

    SUBJECTS = ["sub-03", "sub-04", "sub-05", "sub-06", "sub-11"]

    log.info("Starting ACTUAL (post-hoc) pressure & temperature maps processing pipeline...")

    for sub in SUBJECTS:
        log.info(f"Processing {sub} ...")

        # 1. Locate and load individual structural and target region volumes
        sub_in = os.path.join(INPUT_DIR, sub)
        t1_path = Path(os.path.join(sub_in, f"{sub}_T1w_kplan.nii.gz"))
        density_path = Path(os.path.join(sub_in, f"{sub}_density_kplan.nii.gz"))
        sg_l_path = Path(os.path.join(sub_in, f"sgACC_BA25_L_kplan.nii.gz"))
        sg_r_path = Path(os.path.join(sub_in, f"sgACC_BA25_R_kplan.nii.gz"))

        if not t1_path.exists():
            log.error(f"Missing required background anatomical T1w volume: {t1_path}")
            continue
        if not sg_l_path.exists() or not sg_r_path.exists():
            log.error(f"Missing required sgACC target ROI volumes in: {sub_in}")
            continue

        t1_img = load_img(t1_path)
        sgacc_imgs = {"left": load_img(sg_l_path), "right": load_img(sg_r_path)}

        # Load optional soft-tissue/segmentation density volume for skull boundary masking
        seg_img = load_img(density_path) if density_path.exists() else None

        # 2. Iterate through experimental conditions and hemisphere sides to resolve posthoc maps
        for cond in CONDS:
            for side in SIDES:
                side_letter = "L" if side == "left" else "R"
                cond_folder = "exp-focused" if cond == "exp" else "con-defocused"

                act_dir = Path(os.path.join(OUTPUT_DIR, sub, "posthoc", cond_folder))

                pact_path = find_map_file(act_dir, side_letter, "Pressure")
                tact_path = find_map_file(act_dir, side_letter, "Temperature")

                # Verify both map components are available
                if pact_path is None or tact_path is None:
                    log.info(f"  [{cond} | {side}] Skipping: missing one or both actual map files.")
                    continue

                log.info(f"  [{cond} | {side}] Processing complete actual map set...")

                # Resample and load arrays
                pact_img = load_img(pact_path)
                tact_img = load_img(tact_path)
                ref_img = pact_img

                t1_ref = resample_to_target(t1_img, ref_img, order=1)
                sg_ref = resample_to_target(sgacc_imgs[side], ref_img, order=0)
                pact = pressure_to_mpa(img_data(pact_img))
                tact = resample_to_target(tact_img, ref_img, order=1)

                # 3. Build soft-tissue scalp/skull boundary exclusion mask
                if seg_img is not None:
                    seg_ref = resample_to_target(seg_img, ref_img, order=0)
                    seg_int = np.round(seg_ref).astype(int)
                    # Label 1 to 3 represent brain tissue classes GM/WM/CSF
                    brain_mask = (seg_int >= 1) & (seg_int <= 3)
                    brain_mask = _bin_erode(brain_mask, iterations=2)
                else:
                    # Construct intensity-based soft-tissue mask from T1w background
                    t1_finite = t1_ref[np.isfinite(t1_ref) & (t1_ref > 0)]
                    t1_lo = float(np.percentile(t1_finite, 20)) if t1_finite.size > 0 else 0.0
                    t1_hi = float(np.percentile(t1_finite, 98)) if t1_finite.size > 0 else np.inf
                    brain_mask = (t1_ref > t1_lo) & (t1_ref < t1_hi)
                    brain_mask = _fill_holes(brain_mask)
                    brain_mask = _bin_erode(brain_mask, iterations=2)

                center_vox = mask_centroid_vox(sg_ref, 0.5)
                center_world = voxel_to_world(center_vox, ref_img.affine)

                # Define standard scale maximums
                pressure_vmax = float(np.nanmax(pact))
                temperature_vmin = 37.0
                temperature_vmax = float(np.nanmax(tact))
                if temperature_vmax <= temperature_vmin:
                    temperature_vmax = temperature_vmin + 1.0

                # 4. Generate whole-head mosaics
                for kind, vol, cmap, vmin, vmax, unit, label in [
                    ("actual_pressure", pact, PRESSURE_CMAP, 0.0, pressure_vmax, "Pressure (MPa)", "post-hoc pressure"),
                    ("actual_temperature", tact, TEMP_CMAP, temperature_vmin, temperature_vmax, "Temperature (C)", "post-hoc temperature"),
                ]:
                    op = Path(os.path.join(DERIVATIVES_DIR, f"{sub}_{kind}_{cond}_{side}_mosaic.png"))
                    log.info(f"    Generating actual whole-head mosaic: {op.name}")
                    plot_map_mosaic(sub, f"{COND_TITLE[cond]} {SIDE_TITLE[side]} {label}", t1_ref, vol, sg_ref,
                                    center_vox, cmap, vmin, vmax, unit, op)

                # 5. Extract actual focal volume details
                try:
                    focal_bin, peak, thr, peak_vox, com_vox = focal_volume_from_pressure(
                        pact, ref_img, sg_ref, center_world, 25.0, brain_mask
                    )
                    vvol = voxel_volume_mm3(ref_img)
                    focal_vol_mm3 = float(focal_bin.sum() * vvol)
                    log.info(f"    Actual Focal Analysis:")
                    log.info(f"      Peak pressure: {peak:.4f} MPa")
                    log.info(f"      -3 dB threshold: {thr:.4f} MPa")
                    log.info(f"      Focal volume: {focal_vol_mm3:.1f} mm3 ({int(focal_bin.sum())} voxels)")
                except Exception as e:
                    log.warning(f"    Focal analysis skipped or failed: {e}")

    log.info("ACTUAL (post-hoc) pressure & temperature maps processing complete.")
