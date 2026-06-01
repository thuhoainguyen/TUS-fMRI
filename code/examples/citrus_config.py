# citrus_config.py
from pathlib import Path

# -------------------------
# Basic paths
# -------------------------
BASE_DIR = Path("/Volumes/engram/CITRUS/online")

FMRIPREP_DIR = BASE_DIR / "data" / "derivatives" / "nordic-tedana" / "fmriprep"
TEDANA_DIR = BASE_DIR / "data" / "derivatives" / "nordic-tedana" / "tedana"
ANALYSIS_DIR = BASE_DIR / "scratch" / "analysis"

SESS_FUNC = "ses-online"
SESS_ANAT = "ses-intake" 

# Simulation-related paths
SIMNIBS_DIR = BASE_DIR / "data" / "derivatives" / "simnibs"
KPLAN_INPUT_DIR = BASE_DIR / "data" / "derivatives" / "kplan" / "input"
KPLAN_OUTPUT_DIR = BASE_DIR / "data" / "derivatives" / "kplan" / "output"
LOCALITE_DIR = BASE_DIR / "data" / "derivatives" / "kplan" / "input"

SUBJECTS = ["sub-01", "sub-03", "sub-04", "sub-05", "sub-06", "sub-08", "sub-09", "sub-10", "sub-11", "sub-12", "sub-13"]

# Subject-specific Localite XML file paths
# These are needed because filenames contain dates that differ per subject
LOCALITE_XML_PATHS = {
    "sub-01": KPLAN_INPUT_DIR / "sub-01" / "sub-01_GUMMarkers20251030.xml",
    "sub-03": KPLAN_INPUT_DIR / "sub-03" / "sub-03_GUMMarkers20251112.xml",
    "sub-04": KPLAN_INPUT_DIR / "sub-04" / "sub-04_GUMMarkers20251114.xml",
    "sub-05": KPLAN_INPUT_DIR / "sub-05" / "sub-05_GUMMarkers20251120.xml",
    "sub-06": KPLAN_INPUT_DIR / "sub-06" / "sub-06_GUMMarkers20251205.xml",    
    "sub-08": KPLAN_INPUT_DIR / "sub-08" / "sub-08_GUMMarkers20260305.xml",
    "sub-09": KPLAN_INPUT_DIR / "sub-09" / "sub-09_GUMMarkers20260305.xml",
    "sub-10": KPLAN_INPUT_DIR / "sub-10" / "sub-10_GUMMarkers20260306.xml",
    "sub-11": KPLAN_INPUT_DIR / "sub-11" / "sub-11_GUMMarkers20251120.xml",
    "sub-12": KPLAN_INPUT_DIR / "sub-12" / "sub-12_GUMMarkers20251120.xml",
    "sub-13": KPLAN_INPUT_DIR / "sub-13" / "sub-13_GUMMarkers20260321.xml",
}
    
# Imaging parameters
TR = 1.8  # seconds
TUS_ONSET_OFFSET = 1.53  # seconds after TR onset
BLOCK_TRS = 10          # 10 TRs per block
BLOCK_DURATION = BLOCK_TRS * TR  # 18 s

# ROI templates (in MNI152NLin2009cAsym)
ROI_TEMPLATE_DIR = BASE_DIR / "data" / "roi" / "template"

# List of template-space ROIs and logical names
# We'll output them in functional/boldref space as:
# {sub}/ses-onlineTUS/roi/{roi_name}_space-boldref.nii.gz
ROI_DEFS = [
    {
        "template_fname": "sgACC_BA25_L.nii.gz",
        "roi_name": "sgACC_BA25_L",
    },
    {
        "template_fname": "sgACC_BA25_R.nii.gz",
        "roi_name": "sgACC_BA25_R",
    },
]

# -------------------------
# Helpers to parse filenames
# -------------------------
def parse_task_and_acq(fname):
    """
    Given a fMRIPrep func filename, extract task (prot1exp/prot2con...)
    and acq (onoff/offon).
    Assumes pattern: sub-XX_ses-online_task-XXX_acq-YYY_desc-...tsv
    """
    name = fname.name
    parts = name.split("_")
    task = None
    acq = None
    for p in parts:
        if p.startswith("task-"):
            task = p.split("task-")[1]
        if p.startswith("acq-"):
            acq = p.split("acq-")[1]
    if task is None or acq is None:
        raise ValueError(f"Could not parse task/acq from {name}")
    return task, acq


def task_to_protocol_and_condition(task):
    """
    task looks like 'prot1exp', 'prot2con', etc.
    protocol: 'prot1', condition: 'exp' or 'con'
    """
    if not task.startswith("prot"):
        raise ValueError(f"Unexpected task format: {task}")
    # last 3 chars are 'exp' or 'con'
    cond = task[-3:]
    prot = task[:-3]  # 'prot1', 'prot2', etc.
    return prot, cond


def acq_to_pattern(acq):
    """
    acq: 'onoff' or 'offon'
    pattern for ON blocks:
      - 'onoff': ON blocks are 0,2,4,6,8,...
      - 'offon': ON blocks are 1,3,5,7,9,...
    """
    if acq not in ("onoff", "offon"):
        raise ValueError(f"Unexpected acq pattern: {acq}")
    return acq

