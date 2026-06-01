"""ROI vs amplitude focus (−3/−6 dB) overlap and pressure summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import nibabel as nib

from ok_plan.nii_utils import squeeze_to_3d


def _aligned_bool(mask_img: nib.Nifti1Image) -> np.ndarray:
    return np.asanyarray(squeeze_to_3d(mask_img).dataobj, dtype=np.float64) != 0


def _aligned_pressure(pressure_img: nib.Nifti1Image) -> np.ndarray:
    return np.squeeze(
        np.asanyarray(squeeze_to_3d(pressure_img).dataobj, dtype=np.float64)
    )


def _voxel_volume_mm3(img: nib.Nifti1Image) -> float:
    """Voxel volume in mm³ from the image affine diagonal."""
    zooms = img.header.get_zooms()[:3]
    return float(abs(zooms[0] * zooms[1] * zooms[2]))


@dataclass(frozen=True)
class FocusRoiPressureStats:
    """Pressure and overlap for one focus definition (e.g. −3 dB)."""

    label: str
    # Overlap geometry
    n_roi_voxels: int
    n_focus_voxels: int
    n_overlap_voxels: int
    voxel_vol_mm3: float
    overlap_pct_of_roi: float   # target coverage: % of ROI in focus
    on_target_pct: float        # % of focus that is on-target
    off_target_pct: float       # % of focus that is off-target
    overlap_vol_mm3: float
    # Pressure in whole ROI
    p_max_in_roi_pa: float
    p_min_in_roi_pa: float
    p_mean_in_roi_pa: float
    # Pressure in overlap (focus ∩ ROI)
    p_max_in_overlap_pa: float
    p_min_in_overlap_pa: float
    p_mean_in_overlap_pa: float
    # Pressure in focus ∖ ROI
    p_max_in_focus_not_roi_pa: float
    p_min_in_focus_not_roi_pa: float
    p_mean_in_focus_not_roi_pa: float


def _pressure_min_max_mean(
    pressure_pa: np.ndarray, region: np.ndarray
) -> tuple[float, float, float]:
    vals = pressure_pa[region & np.isfinite(pressure_pa)]
    if vals.size == 0:
        return (float("nan"),) * 3
    return float(np.max(vals)), float(np.min(vals)), float(np.mean(vals))


def focus_roi_pressure_stats(
    pressure_img: nib.Nifti1Image,
    roi_img: nib.Nifti1Image,
    focus_img: nib.Nifti1Image,
    *,
    label: str,
) -> FocusRoiPressureStats:
    """Overlap of ROI with focus mask; pressure (Pa) in ROI, overlap, and focus∖ROI."""
    p = _aligned_pressure(pressure_img)
    r = _aligned_bool(roi_img)
    f = _aligned_bool(focus_img)
    if p.shape != r.shape or p.shape != f.shape:
        raise ValueError(
            f"Shape mismatch pressure {p.shape} vs roi {r.shape} vs focus {f.shape}"
        )
    n_roi = int(np.count_nonzero(r))
    n_focus = int(np.count_nonzero(f))
    inter = r & f
    n_ov = int(np.count_nonzero(inter))
    overlap_pct = 100.0 * float(n_ov) / float(max(n_roi, 1))
    on_target = 100.0 * float(n_ov) / float(max(n_focus, 1))
    off_target = 100.0 - on_target

    vvol = _voxel_volume_mm3(pressure_img)
    overlap_vol = float(n_ov) * vvol

    pmax_roi, pmin_roi, pmean_roi = _pressure_min_max_mean(p, r)
    pmax_ov, pmin_ov, pmean_ov = _pressure_min_max_mean(p, inter)
    focus_excl_roi = f & ~r
    pmax_fr, pmin_fr, pmean_fr = _pressure_min_max_mean(p, focus_excl_roi)

    return FocusRoiPressureStats(
        label=label,
        n_roi_voxels=n_roi,
        n_focus_voxels=n_focus,
        n_overlap_voxels=n_ov,
        voxel_vol_mm3=vvol,
        overlap_pct_of_roi=overlap_pct,
        on_target_pct=on_target,
        off_target_pct=off_target,
        overlap_vol_mm3=overlap_vol,
        p_max_in_roi_pa=pmax_roi,
        p_min_in_roi_pa=pmin_roi,
        p_mean_in_roi_pa=pmean_roi,
        p_max_in_overlap_pa=pmax_ov,
        p_min_in_overlap_pa=pmin_ov,
        p_mean_in_overlap_pa=pmean_ov,
        p_max_in_focus_not_roi_pa=pmax_fr,
        p_min_in_focus_not_roi_pa=pmin_fr,
        p_mean_in_focus_not_roi_pa=pmean_fr,
    )
