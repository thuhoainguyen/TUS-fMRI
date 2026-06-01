"""Scalar metrics inside binary masks (pressure / temperature safety reporting)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import nibabel as nib

from ok_plan.nii_utils import squeeze_to_3d


@dataclass(frozen=True)
class MaskSafetyMetrics:
    """Per-mask summary for HTML tables."""

    mask_key: str
    mask_name: str
    mi: float
    cem43_max: float
    cem43_mean: float
    temp_abs_max: float
    temp_rise_min: float
    temp_rise_max: float
    temp_rise_mean: float


def _aligned_data(img: nib.Nifti1Image) -> np.ndarray:
    return np.squeeze(np.asanyarray(squeeze_to_3d(img).dataobj, dtype=np.float64))


def cem43_rate_per_equivalent_minute(t_abs_c: np.ndarray) -> np.ndarray:
    """Sapareto-Dewey rate R(T): equivalent minutes at 43 °C per real minute.

    CEM43 = ∫ R^(43-T) dt, with R = 0.5 for T < 43 °C and R = 0.25 for T ≥ 43 °C.

    Equivalently:
    - T < 43: rate = 0.5^(43-T)  (small but non-zero even at body temperature)
    - T ≥ 43: rate = 4^(T-43)

    This is the same formulation used by k-Wave / k-Plan for thermal dose
    computation.  k-Plan integrates this rate over the full simulated thermal
    time series, whereas we apply a single-time-point steady-state assumption
    (each voxel held at its mapped temperature for the exposure duration).

    See Sapareto & Dewey, Int J Radiat Oncol Biol Phys (1984).
    """
    t = np.asarray(t_abs_c, dtype=np.float64)
    out = np.zeros_like(t, dtype=np.float64)
    valid = np.isfinite(t)
    below = valid & (t < 43.0)
    out[below] = np.power(0.5, 43.0 - t[below])
    above = valid & (t >= 43.0)
    out[above] = np.power(4.0, t[above] - 43.0)
    return out


def temperature_delta_stats_in_mask(
    temp_abs_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
    baseline_body_temp_c: float,
) -> tuple[float, float, float, int]:
    """Return (min, max, mean, n_voxels) of ΔT = T_abs − baseline inside mask."""
    t_abs = _aligned_data(temp_abs_img)
    delta = t_abs - float(baseline_body_temp_c)
    m = _aligned_data(mask_img) != 0
    if delta.shape != m.shape:
        raise ValueError(
            f"Shape mismatch temperature {delta.shape} vs mask {m.shape}"
        )
    vals = delta[m & np.isfinite(delta)]
    n = int(vals.size)
    if n == 0:
        return (float("nan"),) * 3 + (0,)
    return (
        float(np.min(vals)),
        float(np.max(vals)),
        float(np.mean(vals)),
        n,
    )


def peak_abs_pressure_pa_in_mask(
    pressure_img: nib.Nifti1Image, mask_img: nib.Nifti1Image
) -> float:
    """Peak |p| (Pa) over voxels in mask."""
    p = _aligned_data(pressure_img)
    msk = _aligned_data(mask_img) != 0
    if p.shape != msk.shape:
        raise ValueError(
            f"Shape mismatch pressure {p.shape} vs mask {msk.shape}"
        )
    vals = np.abs(p[msk & np.isfinite(p)])
    if vals.size == 0:
        return float("nan")
    return float(np.max(vals))


def mechanical_index_proxy(
    peak_rarefactional_pressure_pa: float, center_frequency_mhz: float
) -> float:
    """Diagnostic-style MI from peak pressure and center frequency.

    ``MI ≈ P_MPa / sqrt(f_MHz)`` with ``P_MPa = |p|_peak / 1e6`` (*p* in Pascal).
    """
    if not np.isfinite(peak_rarefactional_pressure_pa) or peak_rarefactional_pressure_pa < 0:
        return float("nan")
    if center_frequency_mhz <= 0:
        return float("nan")
    p_mpa = peak_rarefactional_pressure_pa * 1e-6
    return float(p_mpa / np.sqrt(center_frequency_mhz))


def cem43_mean_in_mask(
    temp_abs_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
    *,
    exposure_duration_min: float,
) -> float:
    """Mean voxel CEM43 = exposure × mean(R(T)) over masked finite voxels."""
    if exposure_duration_min < 0 or not np.isfinite(exposure_duration_min):
        return float("nan")
    t_abs = _aligned_data(temp_abs_img)
    m = _aligned_data(mask_img) != 0
    tv = t_abs[m & np.isfinite(t_abs)]
    if tv.size == 0:
        return float("nan")
    rates = cem43_rate_per_equivalent_minute(tv)
    return float(np.mean(rates) * float(exposure_duration_min))


def temp_abs_max_in_mask(
    temp_abs_img: nib.Nifti1Image, mask_img: nib.Nifti1Image
) -> float:
    t_abs = _aligned_data(temp_abs_img)
    m = _aligned_data(mask_img) != 0
    vals = t_abs[m & np.isfinite(t_abs)]
    if vals.size == 0:
        return float("nan")
    return float(np.max(vals))


def cem43_max_in_mask(
    temp_abs_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
    *,
    exposure_duration_min: float,
) -> float:
    """Maximum CEM43 (equivalent minutes at 43 °C) in mask for uniform exposure.

    Assumes each voxel stays at its mapped absolute temperature for
    ``exposure_duration_min`` minutes (single-time-point steady assumption).
    """
    if exposure_duration_min < 0 or not np.isfinite(exposure_duration_min):
        return float("nan")
    t_abs = _aligned_data(temp_abs_img)
    m = _aligned_data(mask_img) != 0
    if t_abs.shape != m.shape:
        raise ValueError(
            f"Shape mismatch temperature {t_abs.shape} vs mask {m.shape}"
        )
    tv = t_abs[m & np.isfinite(t_abs)]
    if tv.size == 0:
        return float("nan")
    rates = cem43_rate_per_equivalent_minute(tv)
    return float(np.max(rates) * float(exposure_duration_min))


def safety_metrics_for_mask(
    mask_key: str,
    display_name: str,
    pressure_img: nib.Nifti1Image,
    temp_abs_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
    *,
    center_frequency_mhz: float,
    baseline_body_temp_c: float,
    exposure_duration_min: float,
) -> MaskSafetyMetrics:
    compute_mi = mask_key != "skull"
    if compute_mi:
        p_peak = peak_abs_pressure_pa_in_mask(pressure_img, mask_img)
        mi = mechanical_index_proxy(p_peak, center_frequency_mhz)
    else:
        mi = float("nan")
    cem = cem43_max_in_mask(
        temp_abs_img,
        mask_img,
        exposure_duration_min=exposure_duration_min,
    )
    cem_mean = cem43_mean_in_mask(
        temp_abs_img,
        mask_img,
        exposure_duration_min=exposure_duration_min,
    )
    t_abs_max = temp_abs_max_in_mask(temp_abs_img, mask_img)
    tmin, tmax, tmean, _n = temperature_delta_stats_in_mask(
        temp_abs_img, mask_img, baseline_body_temp_c
    )
    return MaskSafetyMetrics(
        mask_key=mask_key,
        mask_name=display_name,
        mi=mi,
        cem43_max=cem,
        cem43_mean=cem_mean,
        temp_abs_max=t_abs_max,
        temp_rise_min=tmin,
        temp_rise_max=tmax,
        temp_rise_mean=tmean,
    )
