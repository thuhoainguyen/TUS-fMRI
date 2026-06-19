# -*- coding: utf-8 -*-
"""
export_dosimetry_summary.py
===========================
Extract actual peak pressure, peak temperature, and target sgACC pressure/intensity
safety and efficacy statistics across all subjects and conditions. Exports summaries
in tab-separated text and Typst table syntax.

@author Hoai Thu Nguyen
"""

import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
from scipy import ndimage
from nilearn import image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",  #//$NON-NLS-1$
    datefmt="%H:%M:%S",  #//$NON-NLS-1$
)
log = logging.getLogger("export_dosimetry_summary")  #//$NON-NLS-1$


def pa_to_isppa_w_per_cm2(
    pa: float, rho: float = 1000.0, c: float = 1500.0
) -> float:
    """
    Spatial-peak pulse-average intensity in W/cm² from peak pressure amplitude.
    Assumes plane-wave / free-field relation Isppa = p0^2 / (2*rho*c).
    """
    return (pa ** 2) / (2.0 * rho * c) * 1e-4  # W/m2 -> W/cm2


def find_map_file(dir_path: Path, side_letter: str, map_type: str) -> Optional[Path]:
    """Robust search for NIfTI maps in directory using wildcard patterns."""
    if not dir_path.exists():
        return None
    for pattern in [
        f"*_{side_letter}_pos-medoid-*-{map_type}.nii.gz",  #//$NON-NLS-1$
        f"*_{side_letter}_pos-medoid-* - {map_type}.nii.gz",  #//$NON-NLS-1$
        f"*Tx-2_{side_letter}_pos-*-{map_type}.nii.gz",  #//$NON-NLS-1$
        f"*Tx-2_{side_letter}_pos-* - {map_type}.nii.gz",  #//$NON-NLS-1$
        f"*_{side_letter}_pos-*-{map_type}.nii.gz",  #//$NON-NLS-1$
        f"*_{side_letter}_pos-* - {map_type}.nii.gz",  #//$NON-NLS-1$
    ]:
        files = list(dir_path.glob(pattern))
        if files:
            return files[0]
    return None


def main() -> None:
    # Setup working folders relative to CITRUS workspace
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = (
        repo_root
        / "data"
        / "gum"
        / "actual"
        / "citrus-offline_participant_ratings - ratings.csv"
    )  #//$NON-NLS-1$
    input_dir = repo_root / "data" / "input"  #//$NON-NLS-1$
    output_dir = repo_root / "data" / "output"  #//$NON-NLS-1$

    tables_dir = repo_root / "derivatives" / "tables"  #//$NON-NLS-1$
    tables_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        log.error("Ratings CSV file not found at: %s", csv_path)  #//$NON-NLS-1$
        return

    # Load ratings sheet
    log.info("Loading ratings CSV from %s", csv_path)  #//$NON-NLS-1$
    df_ratings = pd.read_csv(csv_path) if 'pd' in globals() else None
    if df_ratings is None:
        import pandas as pd
        df_ratings = pd.read_csv(csv_path)
    df_ratings = df_ratings.rename(
        columns={df_ratings.columns[0]: "subject"}
    )  #//$NON-NLS-1$

    subjects = df_ratings["subject"].dropna().unique()  #//$NON-NLS-1$
    log.info("Processing subjects: %s", list(subjects))  #//$NON-NLS-1$

    records = []

    for sub in subjects:
        log.info("Processing subject %s ...", sub)  #//$NON-NLS-1$

        sub_in = input_dir / sub
        t1w_path = sub_in / f"{sub}_T1w_kplan.nii.gz"  #//$NON-NLS-1$
        brain_mask_path = sub_in / f"{sub}_T1w_kplan_brain_mask.nii.gz"  #//$NON-NLS-1$

        if not t1w_path.exists() or not brain_mask_path.exists():
            log.error("Missing T1w or brain mask files for %s", sub)  #//$NON-NLS-1$
            continue

        # Load brain mask and erode it 3 times to pull away from skull boundary
        brain_mask_img = nib.load(str(brain_mask_path))
        mask_data = brain_mask_img.get_fdata() > 0.5
        eroded_mask = ndimage.binary_erosion(mask_data, iterations=3)

        # Loop through conditions and hemispheres
        for cond in ["exp", "con"]:  #//$NON-NLS-1$  #//$NON-NLS-1$
            cond_folder = "exp-focused" if cond == "exp" else "con-defocused"  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$
            post_dir = output_dir / sub / "posthoc" / cond_folder  #//$NON-NLS-1$
            plan_dir = output_dir / sub / "planning" / cond_folder  #//$NON-NLS-1$

            for side in ["left", "right"]:  #//$NON-NLS-1$  #//$NON-NLS-1$
                side_letter = "L" if side == "left" else "R"  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$
                roi_path = sub_in / f"sgACC_BA25_{side_letter}_kplan.nii.gz"  #//$NON-NLS-1$

                if not roi_path.exists():
                    log.error("sgACC ROI mask not found: %s", roi_path)  #//$NON-NLS-1$
                    continue

                pact_path = find_map_file(post_dir, side_letter, "Pressure")  #//$NON-NLS-1$
                tact_path = find_map_file(post_dir, side_letter, "Temperature")  #//$NON-NLS-1$
                pplan_path = find_map_file(plan_dir, side_letter, "Pressure")  #//$NON-NLS-1$
                tplan_path = find_map_file(plan_dir, side_letter, "Temperature")  #//$NON-NLS-1$

                if pact_path is None or tact_path is None or pplan_path is None or tplan_path is None:
                    log.warning(
                        "[%s | %s | %s] Missing planned or actual pressure/temperature post-hoc/planning NIfTI. Skipping.",  #//$NON-NLS-1$
                        sub, cond.upper(), side.upper()
                    )
                    continue

                try:
                    pact_img = nib.load(str(pact_path))
                    tact_img = nib.load(str(tact_path))
                    pplan_img = nib.load(str(pplan_path))
                    tplan_img = nib.load(str(tplan_path))
                    roi_img = nib.load(str(roi_path))

                    # 1. ACTUAL calculations
                    # Resample masks to actual pressure image space (nearest-neighbor)
                    eroded_ref_img = nib.Nifti1Image(
                        eroded_mask.astype(np.float32), brain_mask_img.affine, brain_mask_img.header
                    )
                    brain_res = image.resample_to_img(eroded_ref_img, pact_img, interpolation="nearest")  #//$NON-NLS-1$
                    eroded_res_data = brain_res.get_fdata() > 0.5

                    roi_res = image.resample_to_img(roi_img, pact_img, interpolation="nearest")  #//$NON-NLS-1$
                    roi_res_data = roi_res.get_fdata() > 0.5

                    # Resample temperature map to pressure image space (continuous)
                    tact_res = image.resample_to_img(tact_img, pact_img, interpolation="continuous")  #//$NON-NLS-1$

                    p_data = pact_img.get_fdata()
                    t_data = tact_res.get_fdata()

                    # Global Brain peak pressure (actual)
                    p_brain_act = np.where(eroded_res_data, p_data, 0.0)
                    max_p_brain_act_raw = float(p_brain_act.max())

                    # Check if raw pressure is in Pa or MPa
                    if max_p_brain_act_raw > 1000.0:
                        p_brain_act_mpa = max_p_brain_act_raw * 1e-6
                    else:
                        p_brain_act_mpa = max_p_brain_act_raw

                    # Target sgACC peak pressure and peak intensity (Isppa) (actual)
                    p_target_act = np.where(roi_res_data, p_data, 0.0)
                    max_p_target_act_raw = float(p_target_act.max())

                    if max_p_target_act_raw > 1000.0:
                        p_target_act_mpa = max_p_target_act_raw * 1e-6
                        p_target_act_pa = max_p_target_act_raw
                    else:
                        p_target_act_mpa = max_p_target_act_raw
                        p_target_act_pa = max_p_target_act_raw * 1e6

                    isppa_target_act = pa_to_isppa_w_per_cm2(p_target_act_pa)

                    # Global Brain peak temperature (actual)
                    t_brain_act = np.where(eroded_res_data, t_data, np.nan)
                    max_t_brain_act = float(np.nanmax(t_brain_act))
                    delta_t_max_brain_act = max_t_brain_act - 37.0

                    # 2. PLANNED calculations
                    # Resample masks to planned pressure image space (nearest-neighbor)
                    brain_plan_res = image.resample_to_img(eroded_ref_img, pplan_img, interpolation="nearest")  #//$NON-NLS-1$
                    eroded_plan_res_data = brain_plan_res.get_fdata() > 0.5

                    roi_plan_res = image.resample_to_img(roi_img, pplan_img, interpolation="nearest")  #//$NON-NLS-1$
                    roi_plan_res_data = roi_plan_res.get_fdata() > 0.5

                    # Resample temperature map to planned pressure image space (continuous)
                    tplan_res = image.resample_to_img(tplan_img, pplan_img, interpolation="continuous")  #//$NON-NLS-1$

                    p_plan_data = pplan_img.get_fdata()
                    t_plan_data = tplan_res.get_fdata()

                    # Target sgACC peak pressure and peak intensity (Isppa) (planned)
                    p_target_plan = np.where(roi_plan_res_data, p_plan_data, 0.0)
                    max_p_target_plan_raw = float(p_target_plan.max())

                    if max_p_target_plan_raw > 1000.0:
                        p_target_plan_mpa = max_p_target_plan_raw * 1e-6
                        p_target_plan_pa = max_p_target_plan_raw
                    else:
                        p_target_plan_mpa = max_p_target_plan_raw
                        p_target_plan_pa = max_p_target_plan_raw * 1e6

                    isppa_target_plan = pa_to_isppa_w_per_cm2(p_target_plan_pa)

                    # Global Brain peak temperature (planned)
                    t_brain_plan = np.where(eroded_plan_res_data, t_plan_data, np.nan)
                    max_t_brain_plan = float(np.nanmax(t_brain_plan))
                    delta_t_max_brain_plan = max_t_brain_plan - 37.0

                    records.append({
                        "subject": sub,  #//$NON-NLS-1$
                        "condition": cond.upper(),  #//$NON-NLS-1$
                        "hemisphere": side.capitalize(),  #//$NON-NLS-1$
                        "p_target_planned_mpa": p_target_plan_mpa,  #//$NON-NLS-1$
                        "p_max_brain_actual_mpa": p_brain_act_mpa,  #//$NON-NLS-1$
                        "p_max_target_actual_mpa": p_target_act_mpa,  #//$NON-NLS-1$
                        "isppa_target_planned_w_cm2": isppa_target_plan,  #//$NON-NLS-1$
                        "isppa_max_target_actual_w_cm2": isppa_target_act,  #//$NON-NLS-1$
                        "delta_t_max_brain_planned_c": delta_t_max_brain_plan,  #//$NON-NLS-1$
                        "delta_t_max_brain_actual_c": delta_t_max_brain_act,  #//$NON-NLS-1$
                    })

                    log.info(
                        "  [%s | %s | %s] P_target_planned=%.4f, P_brain_act=%.4f, P_target_act=%.4f MPa, Isppa_plan=%.3f, Isppa_act=%.3f W/cm2, deltaT_plan=%.2f, deltaT_act=%.2f C",  #//$NON-NLS-1$
                        sub, cond.upper(), side.upper(), p_target_plan_mpa, p_brain_act_mpa, p_target_act_mpa, isppa_target_plan, isppa_target_act, delta_t_max_brain_plan, delta_t_max_brain_act
                    )

                except Exception as e:
                    log.exception(
                        "Failed to extract dosimetry for [%s | %s | %s]: %s",  #//$NON-NLS-1$
                        sub, cond.upper(), side.upper(), e
                    )

    # 1. Save Tab-Separated plain text table
    txt_path = tables_dir / "dosimetry_summary.txt"  #//$NON-NLS-1$
    with open(txt_path, "w", encoding="utf-8") as f:  #//$NON-NLS-1$  #//$NON-NLS-1$
        f.write(
            f"{'Subject':<10}\t{'Condition':<10}\t{'Hemi':<10}\t"  #//$NON-NLS-1$
            f"{'P_target_planned (MPa)':<24}\t{'P_max_brain_actual (MPa)':<26}\t"  #//$NON-NLS-1$
            f"{'P_max_target_actual (MPa)':<26}\t{'Isppa_target_planned (W/cm²)':<30}\t"  #//$NON-NLS-1$
            f"{'Isppa_max_target_actual (W/cm²)':<32}\t{'delta_T_max_brain_planned (°C)':<32}\t"  #//$NON-NLS-1$
            f"{'delta_T_max_brain_actual (°C)':<30}\n"  #//$NON-NLS-1$
        )
        for r in records:
            f.write(
                f"{r['subject']:<10}\t{r['condition']:<10}\t{r['hemisphere']:<10}\t"
                f"{r['p_target_planned_mpa']:<24.4f}\t{r['p_max_brain_actual_mpa']:<26.4f}\t"
                f"{r['p_max_target_actual_mpa']:<26.4f}\t{r['isppa_target_planned_w_cm2']:<30.4f}\t"
                f"{r['isppa_max_target_actual_w_cm2']:<32.4f}\t{r['delta_t_max_brain_planned_c']:<32.4f}\t"
                f"{r['delta_t_max_brain_actual_c']:<30.4f}\n"
            )
    log.info("Saved plain text dosimetry table to: %s", txt_path)  #//$NON-NLS-1$

    # 2. Save Typst formatted table
    typst_path = tables_dir / "dosimetry_summary_typst.txt"  #//$NON-NLS-1$
    typst_path_typ = tables_dir / "dosimetry_summary.typ"  #//$NON-NLS-1$
    typst_lines = [
        "// Post-Hoc Planned & Actual Transducer Dosimetry & Safety Summary Table (-3 dB Focus)",  #//$NON-NLS-1$
        "#table(",  #//$NON-NLS-1$
        "  columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),",  #//$NON-NLS-1$
        "  align: horizon + center,",  #//$NON-NLS-1$
        "  fill: (x, y) => if y == 0 { rgb(\"e0e0e0\") } else if calc.even(y) { rgb(\"f9f9f9\") } else { rgb(\"ffffff\") },",  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$  #//$NON-NLS-1$
        "  [*Subject*], [*Condition*], [*Hemi*], [*sgACC $P_(\"plan\")$ (MPa)*], [*sgACC $P_(\"act\")$ (MPa)*], [*sgACC $I_(\"plan\")$ (W/cm²)*], [*sgACC $I_(\"act\")$ (W/cm²)*], [*Brain $Delta T_(\"plan\")$ (°C)*], [*Brain $Delta T_(\"act\")$ (°C)*],",  #//$NON-NLS-1$
    ]
    for r in records:
        typst_lines.append(
            f"  [{r['subject']}], [{r['condition']}], [{r['hemisphere']}], "
            f"[{r['p_target_planned_mpa']:.4f}], "
            f"[{r['p_max_target_actual_mpa']:.4f}], [{r['isppa_target_planned_w_cm2']:.4f}], "
            f"[{r['isppa_max_target_actual_w_cm2']:.4f}], [{r['delta_t_max_brain_planned_c']:.4f}], "
            f"[{r['delta_t_max_brain_actual_c']:.4f}],"
        )
    typst_lines.append(")")  #//$NON-NLS-1$

    with open(typst_path, "w", encoding="utf-8") as f:  #//$NON-NLS-1$  #//$NON-NLS-1$
        f.write("\n".join(typst_lines) + "\n")  #//$NON-NLS-1$
    log.info("Saved Typst formatted dosimetry table to: %s", typst_path)  #//$NON-NLS-1$

    with open(typst_path_typ, "w", encoding="utf-8") as f:  #//$NON-NLS-1$  #//$NON-NLS-1$
        f.write("\n".join(typst_lines) + "\n")  #//$NON-NLS-1$
    log.info("Saved Typst formatted dosimetry table to: %s", typst_path_typ)  #//$NON-NLS-1$

    print("\nDosimetry extraction complete! Files saved to derivatives/tables/.")  #//$NON-NLS-1$


if __name__ == "__main__":
    main()
