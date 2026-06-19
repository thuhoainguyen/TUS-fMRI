"""
rs_fmri/03_roi_network.py
==========================
Step 3: Extract sgACC network ROI-to-ROI connectivity matrix.

For each subject × session × timepoint:
  - Extract mean time series from each network node (sphere ROI)
  - Compute pairwise Pearson r between all ROI pairs
  - Save connectivity matrix + sgACC-to-each-node vector for radar chart

Output: derivatives/rs_fmri/roi_network/{sub}/{ses}/
          {sub}_{ses}_acq-{acq}_connectivity_matrix.npy
          {sub}_{ses}_acq-{acq}_connectivity_matrix.tsv   (readable)
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from nilearn import image, maskers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_rs import (
    OUT_ROOT, SUBJECTS, SESSIONS, TIMEPOINTS,
    NETWORK_ROIS, NETWORK_RADIUS_MM,
    SGACC_SEEDS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("roi_network")

CLEAN_DIR  = OUT_ROOT / "clean_bold"
NET_DIR    = OUT_ROOT / "roi_network"
ROI_NAMES  = list(NETWORK_ROIS.keys())
ROI_COORDS = list(NETWORK_ROIS.values())


def extract_roi_timeseries(bold_img, mask_img) -> np.ndarray:
    """Extract mean time series for each network ROI. Returns (n_TRs, n_ROIs)."""
    masker = maskers.NiftiSpheresMasker(
        seeds=ROI_COORDS,
        radius=NETWORK_RADIUS_MM,
        mask_img=mask_img,
        standardize=True,
        detrend=False,
    )
    return masker.fit_transform(bold_img)  # (n_TRs, n_ROIs)


def connectivity_matrix(roi_ts: np.ndarray) -> np.ndarray:
    """Compute Fisher-z correlation matrix. Returns (n_ROIs, n_ROIs)."""
    n = roi_ts.shape[1]
    r = np.corrcoef(roi_ts.T)           # (n_ROIs, n_ROIs)
    np.fill_diagonal(r, 0)              # zero self-connectivity
    r = np.clip(r, -0.999, 0.999)
    return np.arctanh(r)                # Fisher z


def main():
    # Collect all results for group-level analysis
    all_records = []

    for sub in SUBJECTS:
        for ses in SESSIONS:
            clean_dir = CLEAN_DIR / sub / ses
            if not clean_dir.exists():
                log.warning("No clean BOLD for %s %s", sub, ses)
                continue

            for acq in TIMEPOINTS:
                bold_path = clean_dir / f"{sub}_{ses}_acq-{acq}_desc-clean_bold.nii.gz"
                if not bold_path.exists():
                    log.warning("MISSING: %s %s %s", sub, ses, acq)
                    continue

                # Brain mask
                from config_rs import MEPREP_ROOT, MEPREP_SSD, BOLD_PROC, BOLD_SPACE
                mask_path = None
                for root in [MEPREP_ROOT / ses / "func", MEPREP_SSD / ses / "func"]:
                    p = root / f"{sub}_{ses}_task-rest_acq-{acq}_proc-{BOLD_PROC}_space-{BOLD_SPACE}_desc-brain_mask.nii.gz"
                    if p.exists():
                        mask_path = p
                        break

                log.info("ROI network: %s | %s | %s", sub, ses, acq)
                bold_img = image.load_img(str(bold_path))
                mask_img = image.load_img(str(mask_path)) if mask_path else None

                roi_ts = extract_roi_timeseries(bold_img, mask_img)  # (n_TRs, n_ROIs)
                conn_z = connectivity_matrix(roi_ts)                  # (n_ROIs, n_ROIs)

                # Save per-subject outputs
                out_dir = NET_DIR / sub / ses
                out_dir.mkdir(parents=True, exist_ok=True)

                np.save(str(out_dir / f"{sub}_{ses}_acq-{acq}_connectivity_matrix.npy"), conn_z)
                pd.DataFrame(conn_z, index=ROI_NAMES, columns=ROI_NAMES).to_csv(
                    str(out_dir / f"{sub}_{ses}_acq-{acq}_connectivity_matrix.tsv"), sep="\t", float_format="%.4f"
                )

                # Also save individual ROI time series
                np.save(str(out_dir / f"{sub}_{ses}_acq-{acq}_roi_timeseries.npy"), roi_ts)

                # Collect sgACC-to-network values for radar chart
                sgacc_l_idx = ROI_NAMES.index("sgACC_L")
                sgacc_r_idx = ROI_NAMES.index("sgACC_R")
                sgacc_mean_conn = (conn_z[sgacc_l_idx] + conn_z[sgacc_r_idx]) / 2

                for roi_i, roi_name in enumerate(ROI_NAMES):
                    if roi_name in ("sgACC_L", "sgACC_R"):
                        continue
                    all_records.append({
                        "subject":    sub,
                        "session":    ses,
                        "timepoint":  acq,
                        "roi":        roi_name,
                        "fc_z":       float(sgacc_mean_conn[roi_i]),
                    })

    # Save group-level FC table
    df = pd.DataFrame(all_records)
    NET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(NET_DIR / "sgacc_network_fc_all.tsv"), sep="\t", index=False, float_format="%.4f")
    log.info("Saved group FC table → %s", NET_DIR / "sgacc_network_fc_all.tsv")


if __name__ == "__main__":
    main()
