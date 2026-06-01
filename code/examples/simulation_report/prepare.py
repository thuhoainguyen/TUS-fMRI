"""Prepare per-subject output folder: copy inputs and resample to simulation grid."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img

from ok_plan.geometry import assert_same_space
from ok_plan.tissues import build_derived_masks, find_final_tissues


@dataclass(frozen=True)
class SubjectWorkspace:
    subject: str
    root: Path
    nifti_dir: Path
    derived_masks_dir: Path
    report_path: Path
    t1_path: Path
    roi_path: Path
    pressure_path: Path
    temperature_path: Path
    seg_path: Path
    charm_dir_effective: Path  # nifti_dir (holds final_tissues + LUT for report)


def _save_resampled(
    source: Path,
    reference: nib.Nifti1Image,
    dest: Path,
    *,
    interpolation: str,
) -> None:
    img = nib.load(source)
    out = resample_to_img(
        img,
        reference,
        interpolation=interpolation,
        force_resample=True,
        copy_header=True,
    )
    nib.save(out, dest)


def _copy_sidecar_charm_files(charm_dir: Path, dest_dir: Path) -> None:
    """Copy small SimNIBS text/config files into dest_dir (flat)."""
    for name in ("final_tissues_LUT.txt", "settings.ini"):
        src = Path(charm_dir) / name
        if src.is_file():
            shutil.copy2(src, dest_dir / name)


def prepare_subject_workspace(
    *,
    output_root: Path,
    subject: str,
    charm_dir: Path,
    t1_path: Path,
    roi_path: Path,
    pressure_path: Path,
    temperature_path: Path,
    report_name: str = "report.html",
) -> SubjectWorkspace:
    """Create ``{output_root}/{subject}/nifti_files/``, copy pressure/temperature,
    resample T1w / ROI / final_tissues to the pressure grid, copy LUT sidecars.

    Pressure and temperature must already share shape and affine.
    """
    output_root = Path(output_root).resolve()
    root = output_root / subject
    nifti_dir = root / "nifti_files"
    nifti_dir.mkdir(parents=True, exist_ok=True)

    pressure_img = nib.load(pressure_path)
    temperature_img = nib.load(temperature_path)
    assert_same_space(
        (pressure_img, "pressure"),
        (temperature_img, "temperature"),
    )

    shutil.copy2(pressure_path, nifti_dir / "pressure.nii.gz")
    shutil.copy2(temperature_path, nifti_dir / "temperature.nii.gz")

    _save_resampled(
        t1_path,
        pressure_img,
        nifti_dir / "T1w.nii.gz",
        interpolation="continuous",
    )
    _save_resampled(
        roi_path,
        pressure_img,
        nifti_dir / "roi.nii.gz",
        interpolation="nearest",
    )

    seg_src = find_final_tissues(charm_dir)
    _save_resampled(
        seg_src,
        pressure_img,
        nifti_dir / "final_tissues.nii.gz",
        interpolation="nearest",
    )

    derived_masks_dir = root / "derived_masks"
    derived_masks_dir.mkdir(parents=True, exist_ok=True)
    native_seg = nib.load(seg_src)
    native_data = np.asanyarray(native_seg.dataobj)
    for name, arr in build_derived_masks(native_data).items():
        native_bin = nib.Nifti1Image(
            arr.astype(np.float32), native_seg.affine, native_seg.header
        )
        on_sim = resample_to_img(
            native_bin,
            pressure_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
        nib.save(on_sim, derived_masks_dir / f"{name}.nii.gz")

    _copy_sidecar_charm_files(charm_dir, nifti_dir)

    report_path = root / report_name
    return SubjectWorkspace(
        subject=subject,
        root=root,
        nifti_dir=nifti_dir,
        derived_masks_dir=derived_masks_dir,
        report_path=report_path,
        t1_path=nifti_dir / "T1w.nii.gz",
        roi_path=nifti_dir / "roi.nii.gz",
        pressure_path=nifti_dir / "pressure.nii.gz",
        temperature_path=nifti_dir / "temperature.nii.gz",
        seg_path=nifti_dir / "final_tissues.nii.gz",
        charm_dir_effective=nifti_dir,
    )
