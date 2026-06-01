"""NIfTI helpers."""

from __future__ import annotations

import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine


def squeeze_to_3d(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Return a 3D image (first volume if 4D)."""
    data = np.squeeze(np.asanyarray(img.dataobj))
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D (or squeezable 4D) volume, got array shape {data.shape}"
        )
    return nib.Nifti1Image(data.astype(np.float32), img.affine, img.header)


def pressure_img_mpa_from_pa(pressure_pa_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Return a new image with data in MPa (input pressure in Pascal)."""
    img = squeeze_to_3d(pressure_pa_img)
    d = np.asanyarray(img.dataobj, dtype=np.float64) * 1e-6
    return nib.Nifti1Image(d.astype(np.float32), img.affine, img.header)


def pa_to_mpa(pa: float) -> float:
    """Convert pressure from Pascal to Megapascal."""
    return pa * 1e-6


def pa_to_isppa_w_per_cm2(
    pa: float, *, rho: float = 1000.0, c: float = 1500.0
) -> float:
    """Spatial-peak pulse-average intensity in W/cm² from peak pressure amplitude.

    Assumes plane-wave / free-field relation Isppa = p₀² / (2ρc).
    Default medium: water (ρ = 1000 kg/m³, c = 1500 m/s).
    """
    return (pa ** 2) / (2.0 * rho * c) * 1e-4  # W/m² → W/cm²


def max_delta_t_coords_mm(
    temp_abs_img: nib.Nifti1Image,
    baseline_body_temp_c: float = 37.0,
) -> tuple[float, float, float]:
    """World-space (mm) coordinates of the voxel with maximum temperature rise."""
    img = squeeze_to_3d(temp_abs_img)
    data = np.asanyarray(img.dataobj, dtype=np.float64)
    delta = data - float(baseline_body_temp_c)
    delta[~np.isfinite(delta)] = -np.inf
    idx = np.unravel_index(np.argmax(delta), delta.shape)
    ijk = np.array(idx, dtype=np.float64) + 0.5
    xyz = apply_affine(img.affine, ijk)
    return (float(xyz[0]), float(xyz[1]), float(xyz[2]))


def roi_centroid_mm(roi_img: nib.Nifti1Image) -> tuple[float, float, float]:
    """World-space (mm) coordinates of the ROI mask centroid for plot cuts.

    Uses voxel centers: mean index (i,j,k) then ``affine @ (i+0.5, j+0.5, k+0.5, 1)``.
    """
    img = squeeze_to_3d(roi_img)
    data = np.asanyarray(img.dataobj)
    m = np.isfinite(data) & (data != 0)
    if not np.any(m):
        raise ValueError("ROI image has no non-zero voxels.")
    coords = np.argwhere(m)
    c_ijk = coords.mean(axis=0).astype(np.float64)
    ijk_center = c_ijk + 0.5
    xyz = apply_affine(img.affine, ijk_center)
    return (float(xyz[0]), float(xyz[1]), float(xyz[2]))
