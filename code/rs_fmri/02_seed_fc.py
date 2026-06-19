"""
rs_fmri/02_seed_fc.py
======================
Step 2: Seed-based functional connectivity from bilateral sgACC.

Input:  derivatives/rs_fmri/clean_bold/{sub}/{ses}/{sub}_{ses}_acq-{acq}_desc-clean_bold.nii.gz
Output: derivatives/rs_fmri/seed_fc/{sub}/{ses}/
          {sub}_{ses}_acq-{acq}_seed-{seed}_desc-fc_map.nii.gz   (r-to-z map)
          {sub}_{ses}_acq-{acq}_seed-{seed}_desc-fc_timeseries.npy
"""

import sys
import logging
import numpy as np
import nibabel as nib
from pathlib import Path
from nilearn import image, maskers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_rs import (
    OUT_ROOT, SUBJECTS, SESSIONS, TIMEPOINTS,
    SGACC_SEEDS, SEED_RADIUS_MM
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("seed_fc")

CLEAN_DIR = OUT_ROOT / "clean_bold"
FC_DIR    = OUT_ROOT / "seed_fc"


def seed_fc_map(bold_img, seed_coord_mm: tuple, radius_mm: float, brain_mask_img) -> nib.Nifti1Image:
    """Compute whole-brain seed-based FC map (Fisher z-transformed r)."""
    # Extract seed time series
    seed_masker = maskers.NiftiSpheresMasker(
        seeds=[seed_coord_mm],
        radius=radius_mm,
        mask_img=brain_mask_img,
        standardize=True,
        detrend=False,
    )
    seed_ts = seed_masker.fit_transform(bold_img).squeeze()  # (n_TRs,)

    # Extract all brain voxels
    brain_masker = maskers.NiftiMasker(
        mask_img=brain_mask_img,
        standardize=True,
        detrend=False,
    )
    brain_ts = brain_masker.fit_transform(bold_img)  # (n_TRs, n_voxels)

    # Pearson correlation with seed
    r = np.dot(seed_ts, brain_ts) / (len(seed_ts) - 1)
    r = np.clip(r, -0.999, 0.999)

    # Fisher r-to-z transform
    z = np.arctanh(r)

    # Reconstruct 3D image
    z_img = brain_masker.inverse_transform(z)
    return z_img, seed_ts


def main():
    for sub in SUBJECTS:
        for ses in SESSIONS:
            clean_dir = CLEAN_DIR / sub / ses
            if not clean_dir.exists():
                log.warning("No clean BOLD for %s %s — run 01_confound_regression.py first", sub, ses)
                continue

            for acq in TIMEPOINTS:
                bold_path = clean_dir / f"{sub}_{ses}_acq-{acq}_desc-clean_bold.nii.gz"
                if not bold_path.exists():
                    log.warning("MISSING clean BOLD: %s %s %s", sub, ses, acq)
                    continue

                # Load brain mask (use from MEPrep output)
                from config_rs import MEPREP_ROOT, MEPREP_SSD, BOLD_PROC, BOLD_SPACE
                for root in [MEPREP_ROOT / ses / "func", MEPREP_SSD / ses / "func"]:
                    mask_path = root / f"{sub}_{ses}_task-rest_acq-{acq}_proc-{BOLD_PROC}_space-{BOLD_SPACE}_desc-brain_mask.nii.gz"
                    if mask_path.exists():
                        break

                bold_img = image.load_img(str(bold_path))
                mask_img = image.load_img(str(mask_path)) if mask_path.exists() else None

                out_dir = FC_DIR / sub / ses
                out_dir.mkdir(parents=True, exist_ok=True)

                for seed_name, seed_coord in SGACC_SEEDS.items():
                    out_fc   = out_dir / f"{sub}_{ses}_acq-{acq}_seed-{seed_name}_desc-fc_map.nii.gz"
                    out_ts   = out_dir / f"{sub}_{ses}_acq-{acq}_seed-{seed_name}_timeseries.npy"

                    if out_fc.exists():
                        log.info("EXISTS (skip): %s %s %s %s", sub, ses, acq, seed_name)
                        continue

                    log.info("FC map: %s | %s | %s | seed=%s", sub, ses, acq, seed_name)
                    z_img, seed_ts = seed_fc_map(bold_img, seed_coord, SEED_RADIUS_MM, mask_img)
                    z_img.to_filename(str(out_fc))
                    np.save(str(out_ts), seed_ts)
                    log.info("  Saved FC map → %s", out_fc.name)


if __name__ == "__main__":
    main()
