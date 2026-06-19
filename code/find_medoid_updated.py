#!/usr/bin/env python3
"""
find_medoid_updated.py
======================
Simplified script to find the medoid frame for TUS transducer actual trajectory data
for all subjects, sessions (CON and EXP), and hemispheres (L and R),
and output simple quality control visualization plots.

@author Hoai Thu Nguyen
"""

import os
import re
import glob
import math
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import cdist

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("find_medoid_updated")


# ══════════════════════════════════════════════════════════════════════════════
# 1. XML PARSING
# ══════════════════════════════════════════════════════════════════════════════

def _matrix4d_to_components(mat_el: ET.Element) -> Dict[str, float]:
    """
    Extract numeric values from XML Matrix4D element attributes.
    """
    return {k: float(v) for k, v in mat_el.attrib.items()}

def parse_gummarker_xml(xml_path: str, convert_lps_to_ras: bool = True) -> pd.DataFrame:
    """
    Parse a Localite GUMMarkers XML file into a pandas DataFrame.
    Flips coordinates from LPS to RAS space if coordinateSpace is LPS.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    source = Path(xml_path).name
    records = []

    coord_system = (root.get("coordinateSpace", "RAS") or "RAS").upper()
    flip = np.eye(4)
    if convert_lps_to_ras and coord_system == "LPS":
        flip = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)

    for elem in root.findall("Element"):
        frame = int(elem.get("index", -1))
        im    = elem.find(".//InstrumentMarker")
        if im is None:
            continue

        mat_el = im.find("Matrix4D")
        if mat_el is None:
            continue

        comp = _matrix4d_to_components(mat_el)
        M = np.array([[comp[f"data{r}{c}"] for c in range(4)]
                      for r in range(4)], dtype=float)

        # Coordinate transformation
        M = flip @ M
        x, y, z = M[0, 3], M[1, 3], M[2, 3]

        row: Dict = dict(
            frame=frame,
            x=x, y=y, z=z,
            source_file=source,
        )
        records.append(row)

    df = pd.DataFrame(records)
    log.debug("Parsed %d rows from %s", len(df), xml_path)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. MEDOID SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def find_medoid_index(coords: np.ndarray) -> int:
    """
    Find the medoid of the provided coordinates.
    """
    dists = cdist(coords, coords, metric="euclidean")
    local_idx = int(np.argmin(dists.sum(axis=1)))
    return local_idx


def analyse_condition(
    actual_df: pd.DataFrame,
    indices:   List[int],
    label:     str,
) -> Dict:
    """
    Find medoid for one session & hemisphere tracking sequence.
    """
    sub = actual_df[actual_df["frame"].isin(indices)].copy().reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"[{label}] No rows found for indices {indices[:5]}…")

    coords = sub[["x", "y", "z"]].to_numpy()

    med_idx = find_medoid_index(coords)
    medoid_row = sub.iloc[med_idx]

    stats = dict(
        n_total        = len(sub),
        medoid_frame   = int(medoid_row["frame"]),
        medoid_x       = float(medoid_row["x"]),
        medoid_y       = float(medoid_row["y"]),
        medoid_z       = float(medoid_row["z"]),
    )

    return dict(
        df=sub,
        coords=coords,
        medoid_idx=med_idx,
        medoid_row=medoid_row,
        stats=stats,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. PLOT GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, color='black', fontsize=10, pad=4)
    ax.set_xlabel(xlabel, color='black', fontsize=9)
    ax.set_ylabel(ylabel, color='black', fontsize=9)
    ax.tick_params(colors='black', labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("black")
    ax.grid(color="#dddddd", linewidth=0.5, linestyle="--")

def fig_spatial_summary(result: Dict, label: str, save_path: str) -> None:
    """
    Save 4-panel spatial summary plot (3D, XY, XZ, YZ).
    """
    df      = result["df"]
    coords  = result["coords"]
    med_idx = result["medoid_idx"]
    stats   = result["stats"]

    medoid_frame = stats['medoid_frame']

    # Parse label back to condition and hemi for display
    parts = label.split('_')
    cond = parts[1].upper()
    hemi = parts[2].upper()

    fig = plt.figure(figsize=(10, 8), facecolor="white")
    fig.suptitle(
        f"Spatial Summary — {label.upper()}\n"
        f"Total points: {stats['n_total']}",
        color="black", fontsize=12, fontweight="bold", y=0.98)

    gs   = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4,
                             left=0.06, right=0.94, top=0.88, bottom=0.08)

    # 3D plot takes up top left and middle
    ax3d = fig.add_subplot(gs[0, :2], projection="3d")

    # Empty space for top right or we could put legend there
    ax_leg = fig.add_subplot(gs[0, 2])
    ax_leg.axis('off')

    # 2D projections on bottom row
    ax_xy  = fig.add_subplot(gs[1, 0])
    ax_xz  = fig.add_subplot(gs[1, 1])
    ax_yz  = fig.add_subplot(gs[1, 2])

    # Plot 3D
    ax3d.set_facecolor("white")

    # All points
    ax3d.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                 color="black", s=10, alpha=0.5, depthshade=True, label="Actual positions")

    # Medoid
    ax3d.scatter([coords[med_idx, 0]], [coords[med_idx, 1]], [coords[med_idx, 2]],
                 color="red", s=80, marker="o", zorder=15, label=f"Medoid: {cond}-{hemi} - {medoid_frame}")

    ax3d.set_xlabel("X (mm)", color="black", fontsize=8, labelpad=2)
    ax3d.set_ylabel("Y (mm)", color="black", fontsize=8, labelpad=2)
    ax3d.set_zlabel("Z (mm)", color="black", fontsize=8, labelpad=2)
    ax3d.tick_params(colors="black", labelsize=7)

    # Put legend in the dedicated empty axes
    handles, labels_leg = ax3d.get_legend_handles_labels()
    ax_leg.legend(handles, labels_leg, loc="center", fontsize=9, framealpha=1.0)

    # 2D Projections
    proj_axes = [
        (ax_xy, (0, 1), ("X (mm)", "Y (mm)")),
        (ax_xz, (0, 2), ("X (mm)", "Z (mm)")),
        (ax_yz, (1, 2), ("Y (mm)", "Z (mm)")),
    ]

    for ax, (d0, d1), (xl, yl) in proj_axes:
        # All points
        ax.scatter(coords[:, d0], coords[:, d1],
                   color="black", s=8, alpha=0.5, linewidths=0)

        # Medoid
        ax.scatter(coords[med_idx, d0], coords[med_idx, d1],
                   color="red", s=50, marker="o", zorder=12)

        _style_ax(ax, title=f"{xl.split()[0]}{yl.split()[0]} projection", xlabel=xl, ylabel=yl)

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 4. XML MEDOID EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

def extract_medoid_xml(xml_path: str, medoid_frame: int, save_path: str) -> None:
    """
    Extract the single Element with index == medoid_frame from the source XML
    and write it to save_path as a valid GUMMarkerList XML file.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    medoid_elem = None
    for elem in root.findall("Element"):
        if int(elem.get("index", -1)) == medoid_frame:
            medoid_elem = elem
            break

    if medoid_elem is None:
        raise ValueError(f"Element with index {medoid_frame} not found in {xml_path}")

    # Build a new root preserving the coordinateSpace attribute
    new_root = ET.Element("GUMMarkerList")
    coord_space = root.get("coordinateSpace")
    if coord_space:
        new_root.set("coordinateSpace", coord_space)

    new_root.append(medoid_elem)

    ET.indent(new_root, space="  ")
    new_tree = ET.ElementTree(new_root)
    ET.indent(new_tree.getroot(), space="  ")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    new_tree.write(save_path, encoding="UTF-8", xml_declaration=True)
    log.info("Saved medoid XML to: %s", save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN ROUTINE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # Setup working folders relative to CITRUS workspace
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = repo_root / "data" / "gum" / "actual" / "citrus-offline_participant_ratings - ratings.csv"
    actual_dir = repo_root / "data" / "gum" / "actual"

    out_dir = repo_root / "derivatives" / "medoid_updated"
    out_dir.mkdir(parents=True, exist_ok=True)

    medoid_xml_dir = repo_root / "data" / "gum" / "medoid_updated"
    medoid_xml_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        log.error("Ratings CSV file not found at: %s", csv_path)
        return

    # 1. Load CSV data
    log.info("Loading ratings CSV from %s", csv_path)
    df_ratings = pd.read_csv(csv_path)
    df_ratings = df_ratings.rename(columns={df_ratings.columns[0]: "subject"})

    subjects = df_ratings["subject"].dropna().unique()
    log.info("Processing subjects: %s", list(subjects))

    medoid_records = []

    for sub in subjects:
        log.info("=== Processing %s ===", sub)
        sub_ratings = df_ratings[df_ratings["subject"] == sub]

        # Loop through ratings rows
        for _, row in sub_ratings.iterrows():
            cond = row["condition"]
            hemi = row["hemisphere"]
            localite_file = row["localite_file"]
            if not str(localite_file).endswith(".xml"):
                localite_file = f"{localite_file}.xml"

            actual_xml_path = actual_dir / sub / localite_file
            if not actual_xml_path.exists():
                log.error("Actual XML file not found: %s", actual_xml_path)
                continue

            xml_start = int(row["xml_start"])
            xml_end = int(row["xml_end"])
            actual_indices = list(range(xml_start, xml_end + 1))

            # Label for logging and filenames
            label = f"{sub}_{cond.lower()}_{hemi.lower()}"
            log.info("Analyzing %s [frames %d-%d]", label, xml_start, xml_end)

            # Parse actual tracking data
            actual_df = parse_gummarker_xml(str(actual_xml_path))

            # Run analysis pipeline
            try:
                res = analyse_condition(
                    actual_df=actual_df,
                    indices=actual_indices,
                    label=label,
                )

                medoid_records.append({
                    "subject": sub,
                    "condition": cond,
                    "hemisphere": hemi,
                    "medoid_frame": res["stats"]["medoid_frame"]
                })

                # Extract medoid XML element
                medoid_xml_path = medoid_xml_dir / f"{label}_medoid.xml"
                extract_medoid_xml(
                    str(actual_xml_path),
                    res["stats"]["medoid_frame"],
                    str(medoid_xml_path),
                )

                # Generate individual figures
                fig_spatial_path = out_dir / f"{label}_spatial_summary.png"
                fig_spatial_summary(res, label, str(fig_spatial_path))

            except Exception as e:
                log.exception("Failed to analyze condition %s: %s", label, e)


    log.info("find_medoid_updated pipeline finished. All images saved to: %s", out_dir)

    # Print medoid frames table
    table_lines = [
        "="*50,
        "  IDENTIFIED MEDOID FRAME INDICES (UPDATED)",
        "="*50,
        f"  {'Subject':<10} {'Condition':<10} {'Hemi':<6} {'Medoid Frame':<12}",
        "  " + "─"*46
    ]
    for r in medoid_records:
        table_lines.append(f"  {r['subject']:<10} {r['condition']:<10} {r['hemisphere']:<6} {r['medoid_frame']:<12}")
    table_lines.append("="*50)

    table_text = "\n".join(table_lines) + "\n"
    print("\n" + table_text)

    # Write to derivatives/medoid_updated/medoid.txt
    txt_path = out_dir / "medoid.txt"
    with open(txt_path, "w") as f:
        f.write(table_text)
    log.info("Saved medoid text summary to: %s", txt_path)


if __name__ == "__main__":
    main()
