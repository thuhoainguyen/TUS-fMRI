# citrus_functions.py
"""
Reusable functions for CITRUS TUS-fMRI analysis.
Load this module like the config: importlib.util.spec_from_file_location(...)
"""
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.affines import apply_affine
from pathlib import Path
from nilearn.glm.first_level import FirstLevelModel
from nilearn import plotting
import xml.etree.ElementTree as ET
import meshio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch
try:
    import ants
    HAS_ANTS = True
except ImportError:
    HAS_ANTS = False
    ants = None


# ============================================================================
# Path helpers (using config)
# ============================================================================

def get_subject_paths(subject, cfg):
    """
    Get all standard paths for a subject using config.
    
    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'sub-pilot02')
    cfg : module
        Loaded config module (citrus_config)
    
    Returns
    -------
    dict : Dictionary with path keys
    """
    return {
        'func_dir': cfg.FMRIPREP_DIR / subject / cfg.SESS_FUNC / "func",
        'anat_dir': cfg.FMRIPREP_DIR / subject / cfg.SESS_ANAT / "anat",
        'tedana_dir': cfg.TEDANA_DIR / subject / cfg.SESS_FUNC,
        'prep_dir': cfg.ANALYSIS_DIR / subject / cfg.SESS_FUNC / "prep",
        'glm_dir': cfg.ANALYSIS_DIR / subject / cfg.SESS_FUNC / "glm",
        'kplan_input_dir': cfg.KPLAN_INPUT_DIR / subject,
        # 'kplan_output_dir': cfg.KPLAN_OUTPUT_DIR / subject,
        'simnibs_dir': cfg.SIMNIBS_DIR / subject,
        # 'localite_dir': cfg.LOCALITE_DIR / subject,
    }


def get_target_coord_path(subject, cfg):
    """Path to kplan ``roi_centroids.txt`` (subject, roi, x, y, z per row)."""
    paths = get_subject_paths(subject, cfg)
    kplan_subdir = paths['kplan_input_dir']
    return kplan_subdir / "roi_centroids.txt"


def load_target_coords(subject, cfg, roi_name=None):
    """
    Return ``[x, y, z]`` for ``subject`` and ROI from ``roi_centroids.txt``.

    If ``roi_name`` is None, uses ``cfg.ROI_DEFS[0]['roi_name']``.
    """
    if roi_name is None:
        roi_name = cfg.ROI_DEFS[0]["roi_name"]
    path = get_target_coord_path(subject, cfg)
    if not path.is_file():
        raise FileNotFoundError(f"ROI centroids file not found: {path}")
    df = pd.read_csv(path, sep=r"\s+")
    mask = (df["subject"] == subject) & (df["roi"] == roi_name)
    rows = df.loc[mask]
    if rows.empty:
        raise ValueError(
            f"No row in {path} for subject={subject!r} roi={roi_name!r}"
        )
    if len(rows) > 1:
        raise ValueError(
            f"Multiple rows in {path} for subject={subject!r} roi={roi_name!r}"
        )
    row = rows.iloc[0]
    return np.asarray([row["x"], row["y"], row["z"]], dtype=float)


def get_t1_path(subject, cfg, space='kplan'):
    """
    Get T1 path in different spaces.
    
    Parameters
    ----------
    subject : str
    cfg : module
    space : str
        'kplan', 'fmriprep', or 'kplan_output'
    """
    paths = get_subject_paths(subject, cfg)
    if space == 'kplan':
        kplan_subdir = paths['kplan_input_dir']
        return kplan_subdir / f"{subject}_T1w_kplan.nii.gz"
    elif space == 'fmriprep':
        return next(
            paths['anat_dir'].glob(f"{subject}_{cfg.SESS_ANAT}_acq-HCP_desc-preproc_T1w.nii.gz"),
            None
        )
    elif space == 'kplan_output':
        # This would need more specific logic based on your kplan output structure
        return None
    return None


def get_roi_path(subject, cfg, roi_name):
    """Get path to ROI file in kplan space."""
    paths = get_subject_paths(subject, cfg)
    kplan_subdir = paths['kplan_input_dir']
    return kplan_subdir / f"{roi_name}_kplan.nii.gz"


# ============================================================================
# Trimming functions
# ============================================================================

def decide_trimming(n_vols, block_trs, label=None):
    """
    Decide whether to drop first+last 5 (Option A) or only first 5 (Option B)
    to keep as many full blocks as possible.
    
    Parameters
    ----------
    n_vols : int
        Total number of volumes
    block_trs : int
        Number of TRs per block (e.g., 10)
    label : str, optional
        Label for logging
    
    Returns
    -------
    tuple : (trim_start, trim_end, n_trimmed, choice)
    """
    usable_A = n_vols - 10  # drop first 5 + last 5
    usable_B = n_vols - 5   # drop only first 5

    if label:
        print(f"=== {label} ===")
        print(f"Total vols: {n_vols}")
        print(f"  Option A (drop first+last 5): usable {usable_A}, mod{block_trs}={usable_A % block_trs if usable_A>0 else 'NA'}")
        print(f"  Option B (drop first 5 only): usable {usable_B}, mod{block_trs}={usable_B % block_trs if usable_B>0 else 'NA'}")

    choice = None
    if usable_A > 0 and usable_A % block_trs == 0:
        choice = "A"
    if usable_B > 0 and usable_B % block_trs == 0:
        choice = "B"

    if choice is None:
        choice = "B" if usable_B > usable_A else "A"

    if choice == "A":
        trim_start = 5
        trim_end = n_vols - 5
    else:
        trim_start = 5
        trim_end = n_vols

    n_trimmed = trim_end - trim_start

    if label:
        print(f"  -> choice: {choice}, trim_start={trim_start}, trim_end={trim_end}, n_trimmed={n_trimmed}")
        print("")

    return trim_start, trim_end, n_trimmed, choice


def trim_bold_and_confounds(subject, cfg, verbose=True):
    """
    Trim BOLD and confounds for all runs of a subject.
    
    Returns
    -------
    pd.DataFrame : Metadata about trimming
    """
    paths = get_subject_paths(subject, cfg)
    func_dir = paths['func_dir']
    prep_dir = paths['prep_dir']
    tedana_dir = paths['tedana_dir']
    
    prep_dir.mkdir(parents=True, exist_ok=True)
    
    if not func_dir.exists():
        if verbose:
            print(f"  WARNING: {func_dir} does not exist, skipping.")
        return pd.DataFrame()
    
    conf_files = sorted(
        func_dir.glob(f"{subject}_{cfg.SESS_FUNC}_task-*_acq-*_desc-confounds_timeseries.tsv")
    )
    if not conf_files:
        if verbose:
            print(f"  No confounds files found in {func_dir}, skipping subject.")
        return pd.DataFrame()
    
    records = []
    
    for conf_file in conf_files:
        task, acq = cfg.parse_task_and_acq(conf_file)
        label = f"{subject} task-{task} acq-{acq}"
        
        # tedana denoised BOLD
        bold_file = (
            tedana_dir / f"task-{task}" / "tedpca-aic" /
            "desc-denoised_bold.nii.gz"
        )
        if not bold_file.exists():
            if verbose:
                print(f"  WARNING: no tedana denoised file for {label}: {bold_file}, skipping this run.")
            continue
        
        if verbose:
            print(f"Processing {label}")
        img = nib.load(str(bold_file))
        n_vols = img.shape[-1]
        
        trim_start, trim_end, n_trimmed, choice = decide_trimming(
            n_vols, cfg.BLOCK_TRS, label if verbose else None
        )
        if n_trimmed <= 0:
            if verbose:
                print(f"  ERROR: non-positive trimmed length for {label}, skipping this run.")
            continue
        
        # Trim BOLD
        bold_data = img.get_fdata()
        bold_trim = bold_data[..., trim_start:trim_end]
        bold_trim_img = nib.Nifti1Image(bold_trim, img.affine, img.header)
        
        bold_trim_file = (
            prep_dir / f"{subject}_{cfg.SESS_FUNC}_task-{task}_acq-{acq}_bold_trim.nii.gz"
        )
        nib.save(bold_trim_img, bold_trim_file)
        if verbose:
            print(f"  Saved trimmed BOLD: {bold_trim_file}")
        
        # Trim confounds
        conf_df = pd.read_csv(conf_file, sep="\t")
        if conf_df.shape[0] != n_vols:
            if verbose:
                print(f"  WARNING: confounds rows ({conf_df.shape[0]}) != n_vols ({n_vols}) for {label}")
        conf_trim = conf_df.iloc[trim_start:trim_end].reset_index(drop=True)
        
        conf_trim_file = (
            prep_dir / f"{subject}_{cfg.SESS_FUNC}_task-{task}_acq-{acq}_confounds_trim.tsv"
        )
        conf_trim.to_csv(conf_trim_file, sep="\t", index=False)
        if verbose:
            print(f"  Saved trimmed confounds: {conf_trim_file}")
        
        # Copy brain mask
        mask_src = func_dir / f"{subject}_{cfg.SESS_FUNC}_task-{task}_acq-{acq}_desc-brain_mask.nii.gz"
        if mask_src.exists():
            mask_dst = (
                prep_dir / f"{subject}_{cfg.SESS_FUNC}_task-{task}_acq-{acq}_brain_mask.nii.gz"
            )
            if not mask_dst.exists():
                mask_dst.write_bytes(mask_src.read_bytes())
            if verbose:
                print(f"  Copied brain mask to: {mask_dst}")
        else:
            if verbose:
                print(f"  WARNING: brain mask not found for {label}: {mask_src}")
        
        records.append({
            "subject": subject,
            "task": task,
            "acq": acq,
            "n_vols_original": n_vols,
            "trim_start": trim_start,
            "trim_end": trim_end,
            "n_vols_trimmed": n_trimmed,
            "choice": choice,
        })
    
    if records:
        meta_df = pd.DataFrame(records)
        meta_file = prep_dir / f"{subject}_{cfg.SESS_FUNC}_trimming_metadata.tsv"
        meta_df.to_csv(meta_file, sep="\t", index=False)
        if verbose:
            print(f"  Wrote trimming metadata: {meta_file}")
        return meta_df
    return pd.DataFrame()


# ============================================================================
# Block design event generation
# ============================================================================

def generate_block_events(subject, task, acq, n_vols_trimmed, cfg, verbose=True):
    """
    Generate block-design events for a TUS run.
    
    Parameters
    ----------
    subject : str
    task : str
        e.g., 'prot1exp', 'prot2con'
    acq : str
        'onoff' or 'offon'
    n_vols_trimmed : int
        Number of volumes after trimming
    cfg : module
    verbose : bool
    
    Returns
    -------
    pd.DataFrame or None : Events DataFrame
    """
    pattern = cfg.acq_to_pattern(acq)
    label = f"{subject} task-{task} acq-{acq}"
    
    n_blocks_possible = n_vols_trimmed // cfg.BLOCK_TRS
    if verbose:
        print(f"  n_vols_trimmed={n_vols_trimmed}, blocks_possible={n_blocks_possible}")
    
    onsets = []
    durations = []
    trial_types = []
    
    for b in range(n_blocks_possible):
        first_tr = b * cfg.BLOCK_TRS
        
        if pattern == "onoff":
            is_on = (b % 2 == 0)
        elif pattern == "offon":
            is_on = (b % 2 == 1)
        else:
            raise ValueError(f"Unexpected pattern for {label}: {pattern}")
        
        if not is_on:
            continue
        
        onset_sec = cfg.TUS_ONSET_OFFSET + first_tr * cfg.TR
        onsets.append(onset_sec)
        durations.append(cfg.BLOCK_DURATION)
        trial_types.append(f"TUS_ON_{task}")
    
    if not onsets:
        if verbose:
            print(f"  WARNING: no ON blocks detected for {label}, no events file written.")
        return None
    
    events_df = pd.DataFrame({
        "trial_type": trial_types,
        "onset": onsets,
        "duration": durations,
    })
    
    return events_df


def generate_all_block_events(subject, cfg, verbose=True):
    """
    Generate block events for all trimmed runs of a subject.
    """
    paths = get_subject_paths(subject, cfg)
    prep_dir = paths['prep_dir']
    
    if not prep_dir.exists():
        if verbose:
            print(f"  WARNING: prep dir {prep_dir} does not exist, did you run trimming first?")
        return
    
    # Find trimmed bold files
    bold_trim_files = sorted(
        prep_dir.glob(f"{subject}_{cfg.SESS_FUNC}_task-*_*_bold_trim.nii.gz")
    )
    if not bold_trim_files:
        if verbose:
            print(f"  No trimmed BOLD files found for {subject}, skipping.")
        return
    
    for bold_trim_file in bold_trim_files:
        name = bold_trim_file.name
        parts = name.split("_")
        task = None
        acq = None
        for p in parts:
            if p.startswith("task-"):
                task = p.split("task-")[1]
            if p.startswith("acq-"):
                acq = p.split("acq-")[1]
        if task is None or acq is None:
            if verbose:
                print(f"  Could not parse task/acq from {name}, skipping.")
            continue
        
        if verbose:
            print(f"Processing {subject} task-{task} acq-{acq}")
        
        img = nib.load(str(bold_trim_file))
        n_vols_trimmed = img.shape[-1]
        
        events_df = generate_block_events(subject, task, acq, n_vols_trimmed, cfg, verbose)
        
        if events_df is not None:
            events_file = prep_dir / f"{subject}_{cfg.SESS_FUNC}_task-{task}_acq-{acq}_events.tsv"
            events_df.to_csv(events_file, sep="\t", index=False)
            if verbose:
                print(f"  Wrote events: {events_file}")


# ============================================================================
# GLM functions
# ============================================================================

MOTION_BASE_COLS = [
    "trans_x", "trans_y", "trans_z",
    "rot_x", "rot_y", "rot_z",
]


def build_friston24_from_confounds(conf_df):
    """
    Select Friston-24 motion regressors from a fMRIPrep confounds DataFrame.
    
    Returns
    -------
    tuple : (conf_24_df, motion_cols_list)
    """
    cols = list(conf_df.columns)
    motion_cols = []
    
    for base in MOTION_BASE_COLS:
        for c in cols:
            if c == base or c.startswith(base + "_"):
                motion_cols.append(c)
    
    motion_cols = sorted(set(motion_cols))
    conf_24 = conf_df[motion_cols].copy()
    return conf_24, motion_cols


def get_condition_regressors(design_mats):
    """
    Collect all unique TUS_ON_* regressors across runs.
    
    Returns
    -------
    list : Sorted list of regressor names
    """
    regset = set()
    for dm in design_mats:
        for col in dm.columns:
            if col.startswith("TUS_ON_"):
                regset.add(col)
    return sorted(regset)


def split_exp_con(reg_names):
    """
    Split regressor names into exp and con groups.
    
    Returns
    -------
    tuple : (exp_regs, con_regs)
    """
    exp_regs = [r for r in reg_names if r.endswith("exp")]
    con_regs = [r for r in reg_names if r.endswith("con")]
    return exp_regs, con_regs


def get_protocols(reg_names):
    """
    From names like 'TUS_ON_prot1exp', extract protocols: 'prot1', 'prot2', etc.
    
    Returns
    -------
    list : Sorted list of protocol names
    """
    prots = set()
    for r in reg_names:
        if not r.startswith("TUS_ON_"):
            continue
        task = r[len("TUS_ON_"):]  # 'prot1exp'
        prot = task[:-3]           # 'prot1'
        prots.add(prot)
    return sorted(prots)


def fit_glm(subject, cfg, verbose=True):
    """
    Fit FirstLevelModel GLM for a subject.
    
    Returns
    -------
    tuple : (FirstLevelModel, list of design matrices, list of run labels)
        run_labels: list of strings like "sub-pilot02 task-prot1con acq-onoff"
    """
    paths = get_subject_paths(subject, cfg)
    prep_dir = paths['prep_dir']
    glm_dir = paths['glm_dir']
    
    glm_dir.mkdir(parents=True, exist_ok=True)
    
    # Trimmed TEDANA BOLD files
    bold_files = sorted(
        prep_dir.glob(f"{subject}_{cfg.SESS_FUNC}_task-*_*_bold_trim.nii.gz")
    )
    if not bold_files:
        if verbose:
            print(f"  No trimmed BOLD files found, skipping {subject}.")
        return None, None
    
    meta_file = prep_dir / f"{subject}_{cfg.SESS_FUNC}_trimming_metadata.tsv"
    if not meta_file.exists():
        if verbose:
            print(f"  Trimming metadata not found: {meta_file}")
        return None, None
    meta_df = pd.read_csv(meta_file, sep="\t")
    
    run_imgs = []
    run_events = []
    run_confounds = []
    run_labels = []  # Store run labels for visualization
    
    for bold_file in bold_files:
        name = bold_file.name
        parts = name.split("_")
        task = None
        acq = None
        for p in parts:
            if p.startswith("task-"):
                task = p.split("task-")[1]
            if p.startswith("acq-"):
                acq = p.split("acq-")[1]
        if task is None or acq is None:
            if verbose:
                print(f"  Could not parse task/acq from {name}, skipping this run.")
            continue
        
        label = f"{subject} task-{task} acq-{acq}"
        if verbose:
            print(f"\n  Adding run: {label}")
        
        row = meta_df.query("task == @task and acq == @acq")
        if row.empty:
            if verbose:
                print(f"    WARNING: no trimming metadata for {label}, skipping this run.")
            continue
        
        # Events
        events_file = prep_dir / f"{subject}_{cfg.SESS_FUNC}_task-{task}_acq-{acq}_events.tsv"
        if not events_file.exists():
            if verbose:
                print(f"    WARNING: events file missing for {label}, skipping this run.")
            continue
        events_df = pd.read_csv(events_file, sep="\t")
        
        # Confounds
        conf_file_trim = prep_dir / f"{subject}_{cfg.SESS_FUNC}_task-{task}_acq-{acq}_confounds_trim.tsv"
        if not conf_file_trim.exists():
            if verbose:
                print(f"    WARNING: trimmed confounds missing for {label}, skipping this run.")
            continue
        conf_df_trim = pd.read_csv(conf_file_trim, sep="\t")
        
        if verbose:
            print(f"    Trimmed confounds shape: {conf_df_trim.shape}")
        conf_24, motion_cols = build_friston24_from_confounds(conf_df_trim)
        if verbose:
            print(f"    Using {len(motion_cols)} motion cols as Friston-24 regressors.")
        
        # BOLD
        img = nib.load(str(bold_file))
        
        run_imgs.append(img)
        run_events.append(events_df)
        run_confounds.append(conf_24)
        run_labels.append(label)  # Store label for this run
    
    if not run_imgs:
        if verbose:
            print(f"  No valid runs for {subject}, skipping GLM.")
        return None, None, None
    
    # Fit GLM
    if verbose:
        print(f"\n  Fitting FirstLevelModel for {subject} with {len(run_imgs)} runs...")
    fm = FirstLevelModel(
        t_r=cfg.TR,
        drift_model="cosine",
        high_pass=1/128,
        hrf_model="spm",
        minimize_memory=False,
    )
    fm = fm.fit(run_imgs, events=run_events, confounds=run_confounds)
    
    design_mats = fm.design_matrices_
    return fm, design_mats, run_labels


def plot_design_matrices(design_mats, run_labels=None, subject=None, show=True):
    """
    Plot design matrices with run names instead of run numbers.
    
    Parameters
    ----------
    design_mats : list
        List of design matrices (DataFrames)
    run_labels : list of str, optional
        Labels for each run (e.g., ["sub-pilot02 task-prot1con acq-onoff", ...])
        If None, uses "run 1", "run 2", etc.
    subject : str, optional
        Subject ID for title (if provided and run_label already contains subject, it's removed to avoid redundancy)
    show : bool
        Whether to call plt.show() for each plot
    
    Returns
    -------
    list : List of matplotlib axes objects
    """
    axes = []
    
    for i, dm in enumerate(design_mats):
        if run_labels and i < len(run_labels):
            run_name = run_labels[i]
            # Remove subject prefix from run_name if it matches subject to avoid redundancy
            if subject and run_name.startswith(f"{subject} "):
                run_name = run_name[len(f"{subject} "):]  # Remove "sub-pilot02 " prefix
            
            if subject:
                title = f"{subject} - {run_name}"
            else:
                title = run_name
        else:
            run_name = f"run {i+1}"
            if subject:
                title = f"{subject} {run_name} design matrix"
            else:
                title = f"{run_name} design matrix"
        
        print(f"\n  Design matrix for {title}")
        ax = plotting.plot_design_matrix(dm)
        ax.set_title(title)
        axes.append(ax)
        
        if show:
            plt.show()
    
    return axes


# ============================================================================
# Contrast functions
# ============================================================================

def contrast_dict_to_arrays(cdef, design_mats):
    """
    Convert a dict {regressor_name: weight} into a list of numpy arrays.
    
    Parameters
    ----------
    cdef : dict
        Contrast definition, e.g., {'TUS_ON_prot1exp': 1.0, 'TUS_ON_prot1con': -1.0}
    design_mats : list
        List of design matrices (DataFrames)
    
    Returns
    -------
    list : List of numpy arrays, one per run
    """
    weights_per_run = []
    
    for dm in design_mats:
        cols = list(dm.columns)
        col_index = {name: i for i, name in enumerate(cols)}
        w = np.zeros(len(cols))
        
        for reg_name, weight in cdef.items():
            if reg_name in col_index:
                w[col_index[reg_name]] = weight
        
        weights_per_run.append(w)
    
    return weights_per_run


def generate_contrasts(design_mats, verbose=True):
    """
    Generate all possible contrasts based on available regressors.
    Robust for 4 or 8 functional runs (prot1-4, exp/con).
    
    Parameters
    ----------
    design_mats : list
        List of design matrices
    verbose : bool
    
    Returns
    -------
    dict : {contrast_name: contrast_definition_dict}
    """
    cond_regs = get_condition_regressors(design_mats)
    if not cond_regs:
        if verbose:
            print("  No TUS_ON_* regressors found, no contrasts generated.")
        return {}
    
    if verbose:
        print(f"  Condition regressors: {cond_regs}")
    
    exp_regs, con_regs = split_exp_con(cond_regs)
    prots = get_protocols(cond_regs)
    
    if verbose:
        print(f"  Exp regs: {exp_regs}")
        print(f"  Con regs: {con_regs}")
        print(f"  Protocols detected: {prots}")
    
    contrasts = {}
    
    # Overall exp > con
    if exp_regs and con_regs:
        w = {}
        for r in exp_regs:
            w[r] = 1.0 / len(exp_regs)
        for r in con_regs:
            w[r] = -1.0 / len(con_regs)
        contrasts["exp_gt_con_overall"] = w
    
    # Per-protocol exp > con and >baseline
    for prot in prots:
        exp_name = f"TUS_ON_{prot}exp"
        con_name = f"TUS_ON_{prot}con"
        
        if exp_name in cond_regs and con_name in cond_regs:
            w = {exp_name: 1.0, con_name: -1.0}
            contrasts[f"{prot}_exp_gt_con"] = w
        
        if exp_name in cond_regs:
            contrasts[f"{prot}_exp_gt_baseline"] = {exp_name: 1.0}
        if con_name in cond_regs:
            contrasts[f"{prot}_con_gt_baseline"] = {con_name: 1.0}
    
    # Protocol differences (exp vs exp, con vs con)
    if len(prots) > 1:
        for i in range(len(prots) - 1):
            p1 = prots[i]
            p2 = prots[i+1]
            e1 = f"TUS_ON_{p1}exp"
            e2 = f"TUS_ON_{p2}exp"
            if e1 in cond_regs and e2 in cond_regs:
                contrasts[f"{p1}exp_gt_{p2}exp"] = {e1: 1.0, e2: -1.0}
            c1 = f"TUS_ON_{p1}con"
            c2 = f"TUS_ON_{p2}con"
            if c1 in cond_regs and c2 in cond_regs:
                contrasts[f"{p1}con_gt_{p2}con"] = {c1: 1.0, c2: -1.0}
    
    return contrasts


# List of contrasts we expect to attempt for every subject/run set.
EXPECTED_CONTRASTS = [
    "exp_gt_con_overall",
    "prot1_con_gt_baseline",
    "prot1_exp_gt_baseline",
    "prot1_exp_gt_con",
    "prot2_con_gt_baseline",
    "prot2_exp_gt_baseline",
    "prot2_exp_gt_con",
    "prot3_con_gt_baseline",
    "prot3_exp_gt_baseline",
    "prot3_exp_gt_con",
    "prot4_con_gt_baseline",
    "prot4_exp_gt_baseline",
    "prot4_exp_gt_con",
]


def _contrast_requirements(name):
    """
    Return a set of regressor names required for a contrast name,
    or a tuple of two sets when any-of is acceptable.
    """
    if name == "exp_gt_con_overall":
        return {"any_exp"}, {"any_con"}
    if name.endswith("_exp_gt_con"):
        prot = name.split("_exp_gt_con")[0]
        return {f"TUS_ON_{prot}exp"}, {f"TUS_ON_{prot}con"}
    if name.endswith("_exp_gt_baseline"):
        prot = name.split("_exp_gt_baseline")[0]
        return {f"TUS_ON_{prot}exp"}
    if name.endswith("_con_gt_baseline"):
        prot = name.split("_con_gt_baseline")[0]
        return {f"TUS_ON_{prot}con"}
    if "exp_gt_" in name and "exp" in name.split("_gt_")[1]:
        p1, p2 = name.split("_exp_gt_")
        return {f"TUS_ON_{p1}exp", f"TUS_ON_{p2}exp"}
    if "con_gt_" in name and "con" in name.split("_gt_")[1]:
        p1, p2 = name.split("_con_gt_")
        return {f"TUS_ON_{p1}con", f"TUS_ON_{p2}con"}
    return set()


def generate_expected_contrasts(design_mats, expected_list=None, verbose=True):
    """
    Generate contrasts limited to an expected list, logging skips with reasons.
    """
    if expected_list is None:
        expected_list = EXPECTED_CONTRASTS

    cond_regs = get_condition_regressors(design_mats)
    if not cond_regs:
        if verbose:
            print("  No TUS_ON_* regressors found, no contrasts generated.")
        return {}

    base_generated = generate_contrasts(design_mats, verbose=False)
    contrasts = {}

    for name in expected_list:
        if name in base_generated:
            contrasts[name] = base_generated[name]
            continue

        reqs = _contrast_requirements(name)
        cond_set = set(cond_regs)

        def _req_str(req):
            return ", ".join(sorted(req)) if req else "none"

        missing_msg = None
        if isinstance(reqs, tuple) and len(reqs) == 2:
            req_a, req_b = reqs
            if req_a == {"any_exp"}:
                has_exp = any(r.endswith("exp") for r in cond_regs)
                has_con = any(r.endswith("con") for r in cond_regs)
                if not has_exp or not has_con:
                    missing_msg = f"needs at least one exp and one con regressor (have exp={has_exp}, con={has_con})"
            else:
                missing = req_a - cond_set
                missing |= req_b - cond_set
                if missing:
                    missing_msg = f"missing regressors: {_req_str(missing)}"
        else:
            missing = reqs - cond_set
            if missing:
                missing_msg = f"missing regressors: {_req_str(missing)}"
            elif not reqs:
                missing_msg = "requirements could not be determined for this contrast name"

        if verbose:
            print(f"  [SKIP] {name}: {missing_msg if missing_msg else 'not generated from available regressors'}")

    if verbose and contrasts:
        print(f"  Generated {len(contrasts)} of {len(expected_list)} expected contrasts.")
    return contrasts


def compute_and_save_contrasts(subject, fm, design_mats, cfg, verbose=True):
    """
    Compute all contrasts and save effect, t-stat, and z-score maps.
    """
    paths = get_subject_paths(subject, cfg)
    glm_dir = paths['glm_dir']
    
    contrasts = generate_expected_contrasts(design_mats, EXPECTED_CONTRASTS, verbose)
    
    if not contrasts:
        if verbose:
            print(f"  No contrasts defined for {subject}, skipping.")
        return
    
    # Compute and save contrasts
    for cname, cdef in contrasts.items():
        if verbose:
            print(f"  Computing contrast: {cname} -> {cdef}")
        
        weights_per_run = contrast_dict_to_arrays(cdef, design_mats)
        
        eff_img = fm.compute_contrast(weights_per_run, output_type="effect_size")
        eff_file = glm_dir / f"{subject}_{cfg.SESS_FUNC}_contrast-{cname}_effect.nii.gz"
        eff_img.to_filename(eff_file)
        if verbose:
            print(f"    Saved effect image: {eff_file}")
        
        t_img = fm.compute_contrast(weights_per_run, output_type="stat")
        t_file = glm_dir / f"{subject}_{cfg.SESS_FUNC}_contrast-{cname}_tstat.nii.gz"
        t_img.to_filename(t_file)
        if verbose:
            print(f"    Saved t-stat image: {t_file}")
        
        z_img = fm.compute_contrast(weights_per_run, output_type="z_score")
        z_file = glm_dir / f"{subject}_{cfg.SESS_FUNC}_contrast-{cname}_zmap.nii.gz"
        z_img.to_filename(z_file)
        if verbose:
            print(f"    Saved z-score image: {z_file}")


# ============================================================================
# Visualization helpers
# ============================================================================

def auto_vmax(img, q=99):
    """
    Auto-determine vmax for visualization based on percentile.
    
    Parameters
    ----------
    img : nibabel image
    q : float
        Percentile to use (default 99)
    
    Returns
    -------
    float : vmax value
    """
    data = np.nan_to_num(img.get_fdata()).ravel()
    data = np.abs(data[data != 0])
    if data.size == 0:
        return 1.0
    return np.percentile(data, q)


# ============================================================================
# Contrast visualization on T1w
# ============================================================================

def list_available_contrasts(subject, cfg, map_type='zmap'):
    """
    List available contrast maps for a subject.
    
    Parameters
    ----------
    subject : str
    cfg : module
    map_type : str
        'zmap', 'tstat', or 'effect'
    
    Returns
    -------
    list : List of contrast names (strings)
    """
    paths = get_subject_paths(subject, cfg)
    glm_dir = paths['glm_dir']
    
    pattern = f"{subject}_{cfg.SESS_FUNC}_contrast-*_{map_type}.nii.gz"
    map_files = sorted(glm_dir.glob(pattern))
    
    contrasts = []
    for map_file in map_files:
        # Extract contrast name
        cname = (
            map_file.name
            .replace(f"{subject}_{cfg.SESS_FUNC}_contrast-", "")
            .replace(f"_{map_type}.nii.gz", "")
        )
        contrasts.append(cname)
    
    return sorted(contrasts)


def warp_contrast_to_t1w(subject, contrast_name, cfg, map_type='zmap', verbose=True):
    """
    Warp a contrast map from boldref space to T1w space.
    
    Parameters
    ----------
    subject : str
    contrast_name : str
        Name of contrast (e.g., 'prot1_exp_gt_con')
    cfg : module
    map_type : str
        'zmap', 'tstat', or 'effect'
    verbose : bool
    
    Returns
    -------
    tuple : (warped_nibabel_image, t1_file_path) or (None, None) if failed
    """
    if not HAS_ANTS:
        raise ImportError("ANTsPy is required for warping. Install with: pip install antspyx")
    
    paths = get_subject_paths(subject, cfg)
    glm_dir = paths['glm_dir']
    func_dir = paths['func_dir']
    anat_dir = paths['anat_dir']
    
    # Load source map
    map_file = glm_dir / f"{subject}_{cfg.SESS_FUNC}_contrast-{contrast_name}_{map_type}.nii.gz"
    if not map_file.exists():
        if verbose:
            print(f"  Contrast map not found: {map_file}")
        return None, None
    
    # Load T1
    t1_file = next(
        anat_dir.glob(f"{subject}_{cfg.SESS_ANAT}_acq-HCP_desc-preproc_T1w.nii.gz"),
        None
    )
    if t1_file is None:
        if verbose:
            print(f"  No T1 found for {subject}")
        return None, None
    
    t1_img_fmriprep = nib.load(t1_file)
    t1_ants = ants.image_read(str(t1_file))
    
    # Get transform
    xfm = next(
        func_dir.glob(f"{subject}_{cfg.SESS_FUNC}_task-*_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt"),
        None
    )
    if xfm is None:
        if verbose:
            print(f"  No boldref→T1w transform found for {subject}")
        return None, None
    
    # Warp
    map_ants = ants.image_read(str(map_file))
    warped_ants = ants.apply_transforms(
        fixed=t1_ants,
        moving=map_ants,
        transformlist=[str(xfm)],
        interpolator="linear"
    )
    
    warped_nib = nib.Nifti1Image(
        warped_ants.numpy(),
        affine=t1_img_fmriprep.affine
    )
    
    # Save warped map
    warped_dir = glm_dir / "warped_to_T1w"
    warped_dir.mkdir(exist_ok=True)
    warped_path = warped_dir / f"{subject}_{cfg.SESS_FUNC}_contrast-{contrast_name}_space-T1w_{map_type}.nii.gz"
    nib.save(warped_nib, warped_path)
    
    if verbose:
        print(f"    Saved warped map → {warped_path.name}")
    
    return warped_nib, t1_file


# ============================================================================
# Warping contrasts to MNI space
# ============================================================================

def warp_contrast_to_mni(subject, contrast_name, cfg, map_type='zmap', verbose=True):
    """
    Warp a contrast map from boldref space to MNI152NLin2009cAsym space.
    Uses ANTsPy (Python API) with two transforms: boldref→T1w and T1w→MNI.
    
    Parameters
    ----------
    subject : str
    contrast_name : str
        Name of contrast (e.g., 'prot1_exp_gt_con')
    cfg : module
    map_type : str
        'zmap', 'tstat', or 'effect'
    verbose : bool
    
    Returns
    -------
    Path or None : Path to warped MNI file if successful, None otherwise
    """
    if not HAS_ANTS:
        if verbose:
            print(f"  [ERROR] {contrast_name} ({map_type}): ANTsPy not available. Install with: pip install antspyx")
        return None
    
    paths = get_subject_paths(subject, cfg)
    glm_dir = paths['glm_dir']
    func_dir = paths['func_dir']
    anat_dir = paths['anat_dir']
    
    # Output directory
    mni_dir = glm_dir.parent / 'glm_MNI'
    mni_dir.mkdir(parents=True, exist_ok=True)
    
    # Source map (in boldref space)
    map_file = glm_dir / f"{subject}_{cfg.SESS_FUNC}_contrast-{contrast_name}_{map_type}.nii.gz"
    if not map_file.exists():
        if verbose:
            print(f"  [SKIP] {contrast_name} ({map_type}): source map not found: {map_file.name}")
        return None
    
    # Output file (include map_type to avoid overwriting different map types)
    out_file = mni_dir / f"{subject}_{cfg.SESS_FUNC}_contrast-{contrast_name}_{map_type}_space-MNI152NLin2009cAsym.nii.gz"
    if out_file.exists():
        if verbose:
            print(f"  [SKIP] {contrast_name} ({map_type}): MNI file already exists: {out_file.name}")
        return out_file
    
    # Reference T1w in MNI space
    ref_mni = next(
        anat_dir.glob(f"{subject}_{cfg.SESS_ANAT}_acq-HCP_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz"),
        None
    )
    if ref_mni is None:
        if verbose:
            print(f"  [SKIP] {contrast_name} ({map_type}): MNI T1w reference not found for {subject}")
        return None
    
    # T1w → MNI transform
    xfm_t1_to_mni = next(
        anat_dir.glob(f"{subject}_{cfg.SESS_ANAT}_acq-HCP_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"),        
        None
    )
    if xfm_t1_to_mni is None:
        if verbose:
            print(f"  [SKIP] {contrast_name} ({map_type}): T1w→MNI transform not found for {subject}")
        return None
    
    # boldref → T1w transform (try to find any task/acq)
    xfm_boldref_to_t1w = next(
        func_dir.glob(f"{subject}_{cfg.SESS_FUNC}_task-*_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt"),
        None
    )
    if xfm_boldref_to_t1w is None:
        if verbose:
            print(f"  [SKIP] {contrast_name} ({map_type}): boldref→T1w transform not found for {subject}")
        return None
    
    # Warp using ANTsPy Python API
    # Transforms are applied in the order listed: boldref→T1w first, then T1w→MNI
    if verbose:
        print(f"  Warping {contrast_name} ({map_type}) to MNI...")
    
    try:
        # Load images using ANTsPy
        moving_img = ants.image_read(str(map_file))
        fixed_img = ants.image_read(str(ref_mni))
        
        # Apply transforms: boldref→T1w, then T1w→MNI
        # Note: ants.apply_transforms applies transforms in reverse order (last transform first)
        # So we list T1w→MNI first, then boldref→T1w
        warped_img = ants.apply_transforms(
            fixed=fixed_img,
            moving=moving_img,
            transformlist=[str(xfm_t1_to_mni), str(xfm_boldref_to_t1w)],
            interpolator='linear'
        )
        
        # Save the warped image
        ants.image_write(warped_img, str(out_file))
        
        if verbose:
            print(f"    Saved: {out_file.name}")
        return out_file
    except Exception as e:
        if verbose:
            print(f"  [ERROR] {contrast_name} ({map_type}): Warping failed: {str(e)}")
        return None


def warp_all_contrasts_to_mni(subject, cfg, contrast_list=None, map_types=None, verbose=True):
    """
    Warp all contrast maps for a subject to MNI space.
    
    Parameters
    ----------
    subject : str
    cfg : module
    contrast_list : list, optional
        List of contrast names to warp. If None, uses EXPECTED_CONTRASTS.
    map_types : list, optional
        List of map types to warp. Default: ['zmap', 'tstat', 'effect']
    verbose : bool
    
    Returns
    -------
    dict : Summary of warping results
    """
    if contrast_list is None:
        contrast_list = EXPECTED_CONTRASTS
    if map_types is None:
        map_types = ['zmap', 'tstat', 'effect']
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Warping contrasts to MNI for {subject}")
        print(f"{'='*60}")
    
    results = {'success': [], 'skipped': [], 'failed': []}
    
    for contrast in contrast_list:
        for map_type in map_types:
            result = warp_contrast_to_mni(subject, contrast, cfg, map_type, verbose)
            if result is None:
                # Check if it was skipped (already exists) or failed
                mni_dir = (cfg.ANALYSIS_DIR / subject / cfg.SESS_FUNC / 'glm_MNI')
                out_file = mni_dir / f"{subject}_{cfg.SESS_FUNC}_contrast-{contrast}_space-MNI152NLin2009cAsym.nii.gz"
                if out_file.exists():
                    results['skipped'].append((contrast, map_type))
                else:
                    results['failed'].append((contrast, map_type))
            else:
                results['success'].append((contrast, map_type))
    
    if verbose:
        print(f"\nSummary for {subject}:")
        print(f"  Success: {len(results['success'])}")
        print(f"  Skipped: {len(results['skipped'])}")
        print(f"  Failed: {len(results['failed'])}")
    
    return results


def plot_contrast_on_t1w(
    subject, contrast_name, cfg, map_type='zmap',
    display_mode='mosaic', cut_coords=None, threshold=None, draw_cross=False, transparency=0.5,
    t1_file=None, warped_map=None, verbose=True, show=True
):
    """
    Plot a contrast map on T1w background.
    
    Parameters
    ----------
    subject : str
    contrast_name : str
        Name of contrast
    cfg : module
    map_type : str
        'zmap', 'tstat', or 'effect'
    display_mode : str
        'mosaic' or 'ortho'
    cut_coords : int or list
        For 'mosaic': number of cuts (e.g., 5, 7)
        For 'ortho': list of [x, y, z] coordinates
    threshold : float, optional
        Threshold for display
    t1_file : Path, optional
        Path to T1 file (if None, will be loaded)
    warped_map : nibabel image, optional
        Pre-warped map (if None, will be warped)
    verbose : bool
    draw_cross : bool
        Whether to draw a crosshair on the plot
    transparency : float
        Transparency of the overlay
    show : bool
        Whether to call plt.show()
    
    Returns
    -------
    matplotlib display object
    """
    # Warp map if not provided
    if warped_map is None:
        warped_map, t1_file = warp_contrast_to_t1w(subject, contrast_name, cfg, map_type, verbose)
        if warped_map is None:
            return None
    
    # Load T1 if not provided
    if t1_file is None:
        paths = get_subject_paths(subject, cfg)
        anat_dir = paths['anat_dir']
        t1_file = next(
            anat_dir.glob(f"{subject}_{cfg.SESS_ANAT}_acq-HCP_desc-preproc_T1w.nii.gz"),
            None
        )
        if t1_file is None:
            if verbose:
                print(f"  No T1 found for {subject}")
            return None
    
    # Auto-determine vmax
    vmax = auto_vmax(warped_map)
    vmin = -vmax
    
    # Set default cut_coords if not provided
    if cut_coords is None:
        if display_mode == 'mosaic':
            cut_coords = 7
        else:
            # For ortho, use center of image
            data = warped_map.get_fdata()
            center_idx = [s // 2 for s in data.shape]
            # Convert to world coordinates
            center_homogeneous = np.array([center_idx[0], center_idx[1], center_idx[2], 1.0])
            cut_coords = (warped_map.affine @ center_homogeneous)[:3].tolist()
    
    # Plot
    if verbose:
        print(f"    Visualizing {contrast_name} ({map_type}) ...")
    
    plot_kwargs = {
        'stat_map_img': warped_map,
        'bg_img': t1_file,
        'vmin': vmin,
        'vmax': vmax,
        'cmap': 'cold_hot',
        'symmetric_cbar': True,
        'display_mode': display_mode,
        'cut_coords': cut_coords,
        'draw_cross': draw_cross,
        'transparency': transparency,
        'title': f"{subject} {contrast_name} ({map_type}, thr={threshold}) on T1w",
    }
    
    if threshold is not None:
        plot_kwargs['threshold'] = threshold
    
    display = plotting.plot_stat_map(**plot_kwargs)
    
    if show:
        plt.show()
    
    return display

def plot_contrast_on_t1w_all(
    subject, contrast_name, cfg, map_type="zmap",
    display_mode="mosaic", cut_coords=None,
    threshold=None, draw_cross=False, transparency=0.5,
    t1_file=None, warped_map=None,
    scale_mode="robust",        # "robust" | "raw" | "manual"
    robust_pct=(1, 99),         # used if scale_mode="robust"
    vmin=None, vmax=None,       # used if scale_mode="manual"
    symmetric_cbar=True,        # keep True for z/t maps; can set False if desired
    verbose=True, show=True
):
    """
    Plot a contrast map on T1w background with flexible scaling.

    scale_mode:
      - "robust": use percentiles (default) to avoid outliers dominating
      - "raw": use true min/max of the image data
      - "manual": use provided vmin/vmax

    threshold:
      - None => plot all values (no threshold)
      - float => nilearn thresholding for display
    """
    import numpy as np
    import nibabel as nib
    import matplotlib.pyplot as plt
    from nilearn import plotting

    # Warp map if not provided
    if warped_map is None:
        warped_map, t1_file = warp_contrast_to_t1w(subject, contrast_name, cfg, map_type, verbose)
        if warped_map is None:
            return None

    # Load T1 if not provided
    if t1_file is None:
        paths = get_subject_paths(subject, cfg)
        anat_dir = paths["anat_dir"]
        t1_file = next(
            anat_dir.glob(f"{subject}_{cfg.SESS_ANAT}_acq-HCP_desc-preproc_T1w.nii.gz"),
            None
        )
        if t1_file is None:
            if verbose:
                print(f"  No T1 found for {subject}")
            return None

    # Compute display scaling
    data = warped_map.get_fdata()
    data = data[np.isfinite(data)]
    if data.size == 0:
        if verbose:
            print("  Map is empty/NaN only.")
        return None

    if scale_mode == "raw":
        raw_min = float(data.min())
        raw_max = float(data.max())

        if symmetric_cbar:
            m = max(abs(raw_min), abs(raw_max))
            vmin_use, vmax_use = -m, m
        else:
            vmin_use, vmax_use = raw_min, raw_max

        if verbose:
            p1, p99 = np.percentile(data, [1, 99])
            print(f"  RAW range: min={raw_min:.3f} max={raw_max:.3f} | p01={p1:.3f} p99={p99:.3f}")

    elif scale_mode == "manual":
        if vmin is None or vmax is None:
            raise ValueError("scale_mode='manual' requires vmin and vmax.")
        vmin_use, vmax_use = float(vmin), float(vmax)

        if verbose:
            print(f"  MANUAL range: vmin={vmin_use:.3f} vmax={vmax_use:.3f}")

    elif scale_mode == "robust":
        lo, hi = robust_pct
        plo, phi = np.percentile(data, [lo, hi])

        if symmetric_cbar:
            m = max(abs(float(plo)), abs(float(phi)))
            vmin_use, vmax_use = -m, m
        else:
            vmin_use, vmax_use = float(plo), float(phi)

        if verbose:
            raw_min = float(data.min())
            raw_max = float(data.max())
            print(f"  ROBUST range (p{lo}/p{hi}): {plo:.3f} .. {phi:.3f} | raw min/max={raw_min:.3f}/{raw_max:.3f}")

    else:
        raise ValueError("scale_mode must be one of: 'robust', 'raw', 'manual'.")

    # Default cut coords
    if cut_coords is None:
        cut_coords = 7 if display_mode == "mosaic" else None  # nilearn will pick sensible defaults if None

    # Plot
    if verbose:
        thr_txt = "None" if threshold is None else str(threshold)
        print(f"    Visualizing {contrast_name} ({map_type}) on T1w | threshold={thr_txt} | scale_mode={scale_mode}")

    plot_kwargs = dict(
        stat_map_img=warped_map,
        bg_img=t1_file,
        cmap="cold_hot",
        symmetric_cbar=symmetric_cbar,
        display_mode=display_mode,
        cut_coords=cut_coords,
        draw_cross=draw_cross,
        transparency=transparency,
        vmin=vmin_use,
        vmax=vmax_use,
        title=f"{subject} {contrast_name} ({map_type}) on T1w | thr={threshold} | scale={scale_mode}",
    )
    if threshold is not None:
        plot_kwargs["threshold"] = threshold

    display = plotting.plot_stat_map(**plot_kwargs)
    if show:
        plt.show()
    return display

# ============================================================================
# Transducer position visualization
# ============================================================================

def _lps2ras():
    """Helper: LPS to RAS transformation matrix."""
    return np.array([
        [-1, +0, +0, +0],
        [+0, -1, +0, +0],
        [+0, +0, +1, +0],
        [0, 0, 0, 1]
    ])


def _simnibs2localite():
    """Helper: SimNIBS to Localite coil axes transformation matrix."""
    return np.array([
        [+0, +0, +1, +0],  # +z -> +x
        [+0, -1, +0, +0],  # +y -> -y
        [+1, +0, +0, +0],  # +x -> +z
        [0, 0, 0, 1]
    ])


def parse_gummarker_xml_to_matsimnibs(xml_path):
    """
    Parse Localite GUMMarkers XML file and convert to SimNIBS-compatible matrices.
    
    Parameters
    ----------
    xml_path : str or Path
        Path to GUMMarkers XML file
    
    Returns
    -------
    tuple : (mats, descrs)
        mats : np.ndarray, shape (n_markers, 4, 4)
            Transformation matrices in SimNIBS space
        descrs : list of str
            Descriptions for each marker
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    coord_system = root.get("coordinateSpace", "").upper()
    if coord_system == "RAS":
        lps2ras_flipmat = np.eye(4)
    elif coord_system == "LPS":
        lps2ras_flipmat = _lps2ras()
    else:
        raise ValueError(f"coordinateSpace='{coord_system}' not supported")
    
    coil_axes_flipmat = _simnibs2localite()
    mats = []
    descrs = []
    
    for elem in root.findall("Element"):
        im_node = elem.find("InstrumentMarker")
        if im_node is None:
            continue
        descr = im_node.get("description", "")
        mat_node = im_node.find("Matrix4D")
        if mat_node is None:
            continue
        
        M = np.zeros((4, 4), dtype=float)
        for i in range(4):
            for j in range(4):
                M[i, j] = float(mat_node.get(f"data{i}{j}"))
        
        # Localite → RAS
        M = lps2ras_flipmat @ M
        # Localite coil axes → SimNIBS coil axes
        M = M @ coil_axes_flipmat
        
        mats.append(M)
        descrs.append(descr)
    
    return np.stack(mats, axis=0), descrs


def get_localite_xml_path(subject, cfg, session="ses-intake", pattern="GUMMarkers*.xml"):
    """
    Get path to Localite GUMMarkers XML file.
    
    First checks LOCALITE_XML_PATHS dictionary in config (subject-specific paths).
    If not found, falls back to pattern search in the session directory.
    
    Parameters
    ----------
    subject : str
    cfg : module
    session : str
        Session name (default: "ses-intake")
    pattern : str
        Filename pattern to search for (fallback only)
    
    Returns
    -------
    Path or None
    """
    # First, check if subject has a specific path in config
    if hasattr(cfg, 'LOCALITE_XML_PATHS') and subject in cfg.LOCALITE_XML_PATHS:
        xml_path = cfg.LOCALITE_XML_PATHS[subject]
        if xml_path.exists():
            return xml_path
        # If path doesn't exist, warn and fall through to pattern search
        import warnings
        warnings.warn(f"Config path for {subject} does not exist: {xml_path}, trying pattern search")
    
    # Fallback: search by pattern
    paths = get_subject_paths(subject, cfg)
    localite_dir = paths['localite_dir'] / session
    if not localite_dir.exists():
        return None
    
    # Try to find XML file matching pattern
    xml_files = list(localite_dir.glob(pattern))
    if xml_files:
        return sorted(xml_files)[0]  # Return first match
    return None


def get_simnibs_mesh_path(subject, cfg):
    """
    Get path to SimNIBS mesh file.
    
    Parameters
    ----------
    subject : str
    cfg : module
    
    Returns
    -------
    Path
    """
    paths = get_subject_paths(subject, cfg)
    msh_path = paths['simnibs_dir'] / f"m2m_{subject}" / f"{subject}.msh"
    return msh_path


def load_transducer_positions(subject, cfg, wanted_prefixes=None, disc_radius_mm=31.0, n_circle_pts=200, verbose=True):
    """
    Load transducer positions from Localite XML and generate disc geometries.
    
    Parameters
    ----------
    subject : str
    cfg : module
    wanted_prefixes : list of str, optional
        Prefixes to filter markers (e.g., ["Tx-2_pos-1", "Tx-2_pos-2", ...])
        If None, uses ["Tx-2_pos-1", "Tx-2_pos-2", "Tx-2_pos-3", "Tx-2_pos-4", "Tx-2_pos-5"]
    disc_radius_mm : float
        Disc radius in mm (default: 31.0, which is 6.2 cm / 2)
    n_circle_pts : int
        Number of points for disc circle (default: 200)
    verbose : bool
    
    Returns
    -------
    dict with keys:
        'discs_world' : list of np.ndarray, shape (n_pts, 3)
            Disc outlines in world coordinates
        'centers_world' : list of np.ndarray, shape (3,)
            Disc centers in world coordinates
        'descriptions' : list of str
            Marker descriptions
    """
    if wanted_prefixes is None:
        wanted_prefixes = [f"Tx-2_L_pos-{k}" for k in (1, 2, 3, 4, 5)]
    
    xml_path = get_localite_xml_path(subject, cfg)
    if xml_path is None:
        raise FileNotFoundError(f"Could not find Localite XML file for {subject}")
    
    if verbose:
        print(f"Loading transducer positions from: {xml_path.name}")
    
    mats_all, descrs_all = parse_gummarker_xml_to_matsimnibs(xml_path)
    
    # Filter by wanted prefixes
    sel_mats, sel_descrs = [], []
    for M, d in zip(mats_all, descrs_all):
        if any(pref in d for pref in wanted_prefixes):
            sel_mats.append(M)
            sel_descrs.append(d)
    
    sel_mats = np.array(sel_mats)
    if verbose:
        print(f"Selected {len(sel_mats)} markers:")
        for d in sel_descrs:
            print(f"  {d}")
    
    # Generate disc geometries
    angles = np.linspace(0, 2 * np.pi, n_circle_pts, endpoint=True)
    circle_local = np.vstack([
        disc_radius_mm * np.cos(angles),
        disc_radius_mm * np.sin(angles),
        np.zeros_like(angles),
        np.ones_like(angles),
    ])  # (4, N)
    
    discs_world = []
    centers_world = []
    for M in sel_mats:
        circle_world = (M @ circle_local)[:3, :].T  # (N, 3)
        center = M[:3, 3]  # (3,)
        discs_world.append(circle_world)
        centers_world.append(center)
    
    return {
        'discs_world': discs_world,
        'centers_world': centers_world,
        'descriptions': sel_descrs,
    }


def load_simnibs_mesh(subject, cfg, scalp_tag=1005, max_triangles=75000, verbose=True):
    """
    Load SimNIBS mesh and extract scalp surface.
    
    Parameters
    ----------
    subject : str
    cfg : module
    scalp_tag : int
        Physical tag for scalp (default: 1005)
    max_triangles : int
        Maximum number of triangles to plot (default: 75000)
    verbose : bool
    
    Returns
    -------
    dict with keys:
        'points' : np.ndarray, shape (n_points, 3)
            Mesh points
        'scalp_tris' : np.ndarray, shape (n_tris, 3)
            Scalp triangle indices
        'scalp_tris_plot' : np.ndarray
            Subsampled triangles for plotting
        'mid' : np.ndarray, shape (3,)
            Center point for visualization bounds
        'rng' : float
            Range for visualization bounds
    """
    msh_path = get_simnibs_mesh_path(subject, cfg)
    if not msh_path.exists():
        raise FileNotFoundError(f"SimNIBS mesh not found: {msh_path}")
    
    if verbose:
        print(f"Loading SimNIBS mesh from: {msh_path.name}")
    
    mesh = meshio.read(msh_path)
    points = mesh.points
    
    cells_dict = {c.type: c.data for c in mesh.cells}
    if "triangle" not in cells_dict:
        raise RuntimeError("No triangle cells in mesh.")
    triangles = cells_dict["triangle"]
    
    tri_phys = None
    if "gmsh:physical" in mesh.cell_data_dict:
        tri_phys = mesh.cell_data_dict["gmsh:physical"].get("triangle", None)
    
    if tri_phys is not None and scalp_tag in np.unique(tri_phys):
        mask = tri_phys == scalp_tag
        scalp_tris = triangles[mask]
        if verbose:
            print(f"Using scalp tag {scalp_tag}, n_tris = {scalp_tris.shape[0]}")
    else:
        scalp_tris = triangles
        if verbose:
            print("No scalp tag 1005 found – using all triangles.")
    
    k = max(1, scalp_tris.shape[0] // max_triangles)
    scalp_tris_plot = scalp_tris[::k]
    if verbose:
        print(f"Plotting {scalp_tris_plot.shape[0]} triangles")
    
    # Precompute bounds
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    mid = (xyz_min + xyz_max) / 2
    rng = (xyz_max - xyz_min).max() / 2
    
    return {
        'points': points,
        'scalp_tris': scalp_tris,
        'scalp_tris_plot': scalp_tris_plot,
        'mid': mid,
        'rng': rng,
    }


def plot_transducer_positions(
    mesh_data, transducer_data, view='left', 
    figsize=(6, 6), disc_colors=None, title=None, show=True
):
    """
    Plot transducer positions on head mesh.
    
    Parameters
    ----------
    mesh_data : dict
        Output from load_simnibs_mesh()
    transducer_data : dict
        Output from load_transducer_positions()
    view : str
        'left', 'front', or 'top'
    figsize : tuple
        Figure size
    disc_colors : list of str, optional
        Colors for discs (default: uses predefined palette)
    title : str, optional
        Plot title (default: auto-generated from view)
    show : bool
        Whether to call plt.show()
    
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    if disc_colors is None:
        disc_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    points = mesh_data['points']
    scalp_tris_plot = mesh_data['scalp_tris_plot']
    mid = mesh_data['mid']
    rng = mesh_data['rng']
    
    discs_world = transducer_data['discs_world']
    centers_world = transducer_data['centers_world']
    sel_descrs = transducer_data['descriptions']
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    
    # Head surface
    ax.plot_trisurf(
        points[:, 0], points[:, 1], points[:, 2],
        triangles=scalp_tris_plot,
        color="lightgray",
        linewidth=0.2,
        antialiased=False,
        alpha=0.9,
    )
    
    # Transducer discs
    legend_elements = []  # For legend
    for idx, (cw, center, descr) in enumerate(zip(discs_world, centers_world, sel_descrs)):
        col = disc_colors[idx % len(disc_colors)]
        
        # Outline
        ax.plot(cw[:, 0], cw[:, 1], cw[:, 2], color=col, linewidth=2)
        
        # Solid disc
        verts = []
        for j in range(len(cw) - 1):
            verts.append([center, cw[j], cw[j + 1]])
        verts.append([center, cw[-1], cw[0]])
        
        disc = Poly3DCollection(verts, alpha=0.5)
        disc.set_facecolor(col)
        disc.set_edgecolor("none")
        ax.add_collection3d(disc)
        
        # Create legend entry (proxy patch for 3D plot)
        # Extract position identifier from description (e.g., "Tx-2_pos-1" -> "Tx-2_pos-1")
        # Try to extract the transducer and position identifier
        if "pos-" in descr:
            # Extract parts containing transducer and position (e.g., "Tx-2_pos-1")
            parts = descr.split("_")
            # Find parts with "Tx-" and "pos-"
            tx_part = [p for p in parts if "Tx-" in p]
            pos_part = [p for p in parts if "pos-" in p]
            if tx_part and pos_part:
                label = f"{tx_part[0]}_{pos_part[0]}"  # e.g., "Tx-2_pos-1"
            elif pos_part:
                label = pos_part[0]  # e.g., "pos-1"
            else:
                label = descr
        else:
            # Fallback: use description as-is
            label = descr
        
        legend_elements.append(Patch(facecolor=col, edgecolor=col, alpha=0.45, label=label))
    
    # Add legend
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=9)
    
    # Clean look
    ax.set_axis_off()
    ax.grid(False)
    
    # Equal aspect
    ax.set_xlim(mid[0] - rng, mid[0] + rng)
    ax.set_ylim(mid[1] - rng, mid[1] + rng)
    ax.set_zlim(mid[2] - rng, mid[2] + rng)
    
    # Set view
    view_params = {
        'left': {'elev': 0, 'azim': 180, 'title': 'Left view'},
        'front': {'elev': 0, 'azim': 90, 'title': 'Front view'},
        'top': {'elev': 120, 'azim': -90, 'title': 'Top view'},
    }
    
    if view not in view_params:
        raise ValueError(f"Unknown view: {view}. Must be one of {list(view_params.keys())}")
    
    params = view_params[view]
    ax.view_init(elev=params['elev'], azim=params['azim'])
    ax.set_title(title if title is not None else params['title'])
    
    # Adjust layout to accommodate legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave 15% space on right for legend
    if show:
        plt.show()
    
    return fig, ax
