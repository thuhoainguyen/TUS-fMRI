"""
rs_fmri/00_config.py
====================
Shared configuration for all resting-state fMRI analysis scripts.
CITRUS study — sgACC network functional connectivity across TUS timepoints.

Data layout (MEPrep output, proc-pmeica, MNI152NLin2009cAsym):
  ses-exp / ses-con
    └── func/
        └── sub-{ID}_ses-{ses}_task-rest_acq-{acq}_proc-pmeica_
              space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
              space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz
              proc-pmeica_desc-confounds_timeseries.tsv

Timepoints:
  preTUS15   — 15 min pre-TUS  (within-session baseline)
  postTUS15  — 15 min post-TUS
  postTUS30  — 30 min post-TUS
  postTUS45  — 45 min post-TUS

Sessions:  ses-exp (focused TUS), ses-con (defocused TUS)
Subjects:  sub-03, sub-04, sub-05, sub-06, sub-11
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# Primary: external SSD (all preprocessed fMRI data lives here)
MEPREP_ROOT = Path("/Volumes/Extreme SSD/THESIS MSC/MEPrep output")
# Fallback: local copy (currently only sub-05 partial)
MEPREP_SSD  = Path("/Users/hoaithunguyen/Documents/Masters/thesis/MEPrep_output")

CITRUS_ROOT = Path("/Users/hoaithunguyen/Projects/Master-thesis/CITRUS")
OUT_ROOT    = CITRUS_ROOT / "derivatives" / "rs_fmri"

# ── Study design ──────────────────────────────────────────────────────────────
SUBJECTS   = ["sub-03", "sub-04", "sub-05", "sub-06", "sub-11"]
SESSIONS   = ["ses-exp", "ses-con"]
TIMEPOINTS = ["preTUS15", "postTUS15", "postTUS30", "postTUS45"]
# Human-readable labels for plots
TP_LABELS  = {
    "preTUS15":  "Pre-TUS\n(−15 min)",
    "postTUS15": "Post-TUS\n(+15 min)",
    "postTUS30": "Post-TUS\n(+30 min)",
    "postTUS45": "Post-TUS\n(+45 min)",
}
SES_LABELS = {"ses-exp": "Experimental (focused)", "ses-con": "Control (defocused)"}
SES_COLORS = {"ses-exp": "#e05c5c", "ses-con": "#5c7de0"}
TP_COLORS  = ["#9ecae1", "#fc9272", "#e34a33", "#b30000"]  # preTUS → post45

# ── MRI / space ───────────────────────────────────────────────────────────────
BOLD_SPACE = "MNI152NLin2009cAsym"
BOLD_PROC  = "pmeica"
TR         = 1.50  # seconds (confirmed from BOLD JSON sidecar RepetitionTime)

# ── Confound regression strategy ──────────────────────────────────────────────
# 24 HMP + 5 aCompCor + cosine drift + motion outliers
# MEICA already removes physio noise — do NOT include WM/CSF signals
CONFOUND_COLS = (
    ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    + [f"trans_{a}_derivative1"        for a in ["x","y","z"]]
    + [f"trans_{a}_power2"             for a in ["x","y","z"]]
    + [f"trans_{a}_derivative1_power2" for a in ["x","y","z"]]
    + [f"rot_{a}_derivative1"          for a in ["x","y","z"]]
    + [f"rot_{a}_power2"               for a in ["x","y","z"]]
    + [f"rot_{a}_derivative1_power2"   for a in ["x","y","z"]]
    + [f"a_comp_cor_{i:02d}" for i in range(5)]   # 5 aCompCor components
    + [f"cosine{i:02d}"      for i in range(8)]   # cosine drift basis
    # motion_outlier* columns added dynamically (variable count per run)
)
BANDPASS_LOW  = 0.01  # Hz
BANDPASS_HIGH = 0.10  # Hz
SMOOTHING_FWHM = 6.0  # mm (applied after confound regression)

# ── sgACC seed ROIs (MNI152NLin2009cAsym coordinates) ────────────────────────
# BA25 / sgACC bilateral seeds — standard MNI coords, 6mm sphere
# Source: Fox et al. 2012, Mayberg et al. 2005
SGACC_SEEDS = {
    "sgACC_L": (-6, 22, -8),
    "sgACC_R": ( 6, 22, -8),
}
SEED_RADIUS_MM = 6.0

# ── sgACC network ROIs (for radar chart) ─────────────────────────────────────
# Key nodes of the sgACC / default mode network and affective network
# Coordinates in MNI152NLin2009cAsym, sphere radius 8mm
NETWORK_ROIS = {
    "sgACC_L":       (-6,  22,  -8),   # seed — subgenual ACC left
    "sgACC_R":       ( 6,  22,  -8),   # seed — subgenual ACC right
    "mPFC":          ( 0,  52,  -6),   # medial prefrontal cortex
    "PCC":           ( 0, -52,  26),   # posterior cingulate / precuneus
    "Hippo_L":       (-26, -20, -18),  # left hippocampus
    "Hippo_R":       ( 26, -20, -18),  # right hippocampus
    "Amygdala_L":    (-22,  -4, -20),  # left amygdala
    "Amygdala_R":    ( 22,  -4, -20),  # right amygdala
    "Insula_L":      (-38,   2,   2),  # left insula
    "Insula_R":      ( 38,   2,   2),  # right insula
    "vlPFC_L":       (-44,  30,  -8),  # ventrolateral PFC left
    "dACC":          (  0,  24,  30),  # dorsal ACC
}
NETWORK_RADIUS_MM = 8.0
