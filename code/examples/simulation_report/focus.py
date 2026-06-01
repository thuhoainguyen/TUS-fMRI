"""Pressure focus masks (−3 dB / −6 dB relative to peak amplitude inside ROI)."""

from __future__ import annotations

import nibabel as nib
import numpy as np

from ok_plan.nii_utils import squeeze_to_3d


def peak_amplitude_inside(
    pressure_img: nib.Nifti1Image, inside_mask_img: nib.Nifti1Image
) -> float:
    pressure_img = squeeze_to_3d(pressure_img)
    inside_mask_img = squeeze_to_3d(inside_mask_img)
    p = np.asanyarray(pressure_img.dataobj, dtype=np.float64)
    inside = np.asanyarray(inside_mask_img.dataobj) != 0
    if not np.any(inside):
        raise ValueError("inside_skull mask is empty.")
    vals = np.abs(p[inside])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("No finite pressure values inside inside_skull.")
    pmax = float(np.max(vals))
    if pmax <= 0:
        raise ValueError("Peak |pressure| inside inside_skull is not positive.")
    return pmax


def build_focus_masks_amplitude_db(
    pressure_img: nib.Nifti1Image,
    inside_mask_img: nib.Nifti1Image,
    *,
    db_levels: tuple[float, float] = (-3.0, -6.0),
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Binary masks where |p| ≥ 10^(db/20) × peak(|p|) inside ``inside_mask``.

    Interprets the pressure field as **amplitude** (Pa or arbitrary linear
    units). −3 dB and −6 dB refer to amplitude ratios relative to the peak
    |p| **restricted to inside_skull** voxels.
    """
    pressure_img = squeeze_to_3d(pressure_img)
    inside_mask_img = squeeze_to_3d(inside_mask_img)
    p = np.asanyarray(pressure_img.dataobj, dtype=np.float64)
    inside = np.asanyarray(inside_mask_img.dataobj) != 0
    pmax = peak_amplitude_inside(pressure_img, inside_mask_img)
    aff = pressure_img.affine
    hdr = pressure_img.header

    out = []
    for db in db_levels:
        thr = (10.0 ** (db / 20.0)) * pmax
        m = ((np.abs(p) >= thr) & inside).astype(np.float32)
        out.append(nib.Nifti1Image(m, aff, hdr))
    return out[0], out[1]
