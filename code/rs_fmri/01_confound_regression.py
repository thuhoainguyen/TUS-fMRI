"""
rs_fmri/01_confound_regression.py
===================================
Step 1: Confound regression + bandpass filtering + spatial smoothing.

Input:  proc-pmeica preprocessed BOLD (MNI space)
Output: derivatives/rs_fmri/{sub}/{ses}/
          {sub}_{ses}_acq-{acq}_desc-clean_bold.nii.gz

Strategy: 24 HMP + 5 aCompCor + cosine drift + motion outliers
          Bandpass 0.01–0.10 Hz, smooth 6 mm FWHM
          MEICA already handles physio noise → no WM/CSF regression
"""

import os
import sys
import logging
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from nilearn import image, signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_rs import (
    MEPREP_ROOT, MEPREP_SSD, OUT_ROOT,
    SUBJECTS, SESSIONS, TIMEPOINTS,
    BOLD_SPACE, BOLD_PROC,
    CONFOUND_COLS, BANDPASS_LOW, BANDPASS_HIGH, SMOOTHING_FWHM, TR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("confound_regression")


def get_meprep_root(sub: str, ses: str) -> Path:
    """Return the correct MEPrep root (local or SSD)."""
    local = MEPREP_ROOT / ses / "func"
    ssd   = MEPREP_SSD  / ses / "func"
    # Check which root has the subject's data
    for root in [local, ssd]:
        if root.exists() and any(root.glob(f"{sub}_*")):
            return root
    return local  # fallback


def find_bold(func_dir: Path, sub: str, ses: str, acq: str) -> tuple[Path | None, Path | None, Path | None]:
    """Find BOLD, brain mask, and confounds TSV for a given run."""
    stem = f"{sub}_{ses}_task-rest_acq-{acq}_proc-{BOLD_PROC}"
    bold = func_dir / f"{stem}_space-{BOLD_SPACE}_desc-preproc_bold.nii.gz"
    mask = func_dir / f"{stem}_space-{BOLD_SPACE}_desc-brain_mask.nii.gz"
    conf = func_dir / f"{stem}_desc-confounds_timeseries.tsv"
    return (bold if bold.exists() else None,
            mask if mask.exists() else None,
            conf if conf.exists() else None)


def build_confound_matrix(conf_tsv: Path) -> np.ndarray:
    """Select and fill confound columns; add motion outlier dummies."""
    df = pd.read_csv(conf_tsv, sep="\t")
    # Base columns
    cols = [c for c in CONFOUND_COLS if c in df.columns]
    # Add all motion outlier columns (variable count per run)
    outlier_cols = [c for c in df.columns if c.startswith("motion_outlier")]
    cols += outlier_cols
    # Fill NaN (first row of derivatives is always NaN)
    confounds = df[cols].fillna(0).values
    log.info("  Confounds: %d timepoints × %d regressors (%d outliers)",
             confounds.shape[0], confounds.shape[1], len(outlier_cols))
    return confounds


def clean_bold(bold_path: Path, mask_path: Path, conf_tsv: Path, out_path: Path) -> None:
    """Run confound regression + bandpass + smoothing → save cleaned BOLD."""
    log.info("  BOLD:     %s", bold_path.name)

    bold_img   = image.load_img(str(bold_path))
    mask_img   = image.load_img(str(mask_path))
    confounds  = build_confound_matrix(conf_tsv)

    # Smooth first (before regression — preserve more signal)
    log.info("  Smoothing %g mm FWHM...", SMOOTHING_FWHM)
    bold_smooth = image.smooth_img(bold_img, SMOOTHING_FWHM)

    # Confound regression + bandpass in one step
    log.info("  Confound regression + bandpass %.3f–%.3f Hz...", BANDPASS_LOW, BANDPASS_HIGH)
    cleaned = image.clean_img(
        bold_smooth,
        confounds=confounds,
        low_pass=BANDPASS_HIGH,
        high_pass=BANDPASS_LOW,
        t_r=TR,
        mask_img=mask_img,
        standardize=False,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_filename(str(out_path))
    log.info("  Saved → %s", out_path.name)


def main():
    for sub in SUBJECTS:
        for ses in SESSIONS:
            func_dir = get_meprep_root(sub, ses)
            for acq in TIMEPOINTS:
                bold, mask, conf = find_bold(func_dir, sub, ses, acq)
                if bold is None:
                    log.warning("MISSING: %s %s %s — skipping", sub, ses, acq)
                    continue

                out_dir  = OUT_ROOT / "clean_bold" / sub / ses
                out_path = out_dir / f"{sub}_{ses}_acq-{acq}_desc-clean_bold.nii.gz"

                if out_path.exists():
                    log.info("EXISTS (skip): %s %s %s", sub, ses, acq)
                    continue

                log.info("Processing: %s | %s | %s", sub, ses, acq)
                try:
                    clean_bold(bold, mask, conf, out_path)
                except Exception as e:
                    log.exception("FAILED %s %s %s: %s", sub, ses, acq, e)


if __name__ == "__main__":
    main()
