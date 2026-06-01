#!/usr/bin/env python3
"""
stability_analysis.py
=====================

Standalone script to compute and plot transducer stability (drift analysis) 
for both Experimental (EXP) and Control (CON) sessions, for Left and Right sides.

This script parses GUMMarkers XML files, extracts specific frame ranges specified 
in a ratings CSV, computes translational and rotational deviations relative to 
the first frame, and generates high-fidelity combined plots matching the user mockup.

@author Hoai Thu Nguyen
"""

import os
import sys
import math
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ------------------------- Logging setup -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stability_analysis")


# ------------------------- Data Structures -------------------------
@dataclass
class TxMatrix:
    """
    Represents a transducer coordinate frame/matrix.
    
    Attributes:
        index: The index of the frame from the XML.
        description: Description label of the marker.
        matrix: 4x4 transform matrix in RAS space.
    """
    index: int
    description: str
    matrix: np.ndarray

    @property
    def center(self) -> np.ndarray:
        """Returns the 3D position vector (translation part) of the transform."""
        return self.matrix[:3, 3].astype(float)


# ------------------------- XML & Rotation Math -------------------------
def lps_to_ras_matrix() -> np.ndarray:
    """Returns the LPS to RAS coordinate space conversion matrix."""
    return np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)


def parse_gummarkers(xml_path: Path, convert_lps_to_ras: bool = True) -> List[TxMatrix]:
    """
    Parses a Localite GUMMarkers XML file into RAS space TxMatrix objects.
    
    Args:
        xml_path: Path to the XML file.
        convert_lps_to_ras: If True and space is LPS, converts to RAS.
        
    Returns:
        A list of parsed TxMatrix objects.
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")
        
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    coord_system = (root.get("coordinateSpace", "RAS") or "RAS").upper()
    flip = np.eye(4)
    if convert_lps_to_ras and coord_system == "LPS":
        flip = lps_to_ras_matrix()
        
    out: List[TxMatrix] = []
    for elem in root.findall("Element"):
        idx = int(elem.get("index", len(out)))
        im = elem.find("InstrumentMarker")
        descr = ""
        if im is not None:
            descr = im.get("description", "") or ""
        mat_node = elem.find(".//Matrix4D")
        if mat_node is None:
            continue
        M = np.zeros((4, 4), dtype=float)
        for r in range(4):
            for c in range(4):
                val = mat_node.get(f"data{r}{c}")
                if val is None:
                    raise ValueError(f"Missing Matrix4D data{r}{c} in {xml_path}, element index {idx}")
                M[r, c] = float(val)
        M = flip @ M
        out.append(TxMatrix(index=idx, description=descr, matrix=M))
        
    if not out:
        raise ValueError(f"No Matrix4D entries found in XML: {xml_path}")
    return out


def rotation_angle_deg(R: np.ndarray) -> float:
    """Computes the overall rotation angle in degrees from a rotation matrix."""
    val = (np.trace(R) - 1.0) / 2.0
    val = float(np.clip(val, -1.0, 1.0))
    return math.degrees(math.acos(val))


def rotation_vector_xyz_deg(R: np.ndarray) -> np.ndarray:
    """
    Computes the small-angle rotation vector (X, Y, Z degrees) from a rotation matrix.
    
    Args:
        R: 3x3 rotation matrix.
        
    Returns:
        3-element numpy array representing rotation degrees around X, Y, Z.
    """
    angle = math.acos(float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    if abs(angle) < 1e-12:
        return np.zeros(3)
    denom = 2.0 * math.sin(angle)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / denom
    return axis * math.degrees(angle)


def select_range(txs: List[TxMatrix], start_idx: int, end_idx: int) -> List[TxMatrix]:
    """Filters the TxMatrix list within the given start and end index range."""
    selected = [t for t in txs if start_idx <= t.index <= end_idx]
    if not selected:
        log.warning("No frames found within range [%d, %d] for GUMMarkers. Returning dummy RAS frame.", start_idx, end_idx)
        return [TxMatrix(index=start_idx, description="No Data Dummy", matrix=np.eye(4))]
    return selected


# ------------------------- Metric Computations -------------------------
def compute_drift_dataframe(subject: str, condition: str, side: str, txs: List[TxMatrix]) -> pd.DataFrame:
    """
    Computes translation and rotation deviations relative to the first frame.
    
    Args:
        subject: Subject identifier.
        condition: EXP or CON session.
        side: left or right.
        txs: List of TxMatrix frames.
        
    Returns:
        A pandas DataFrame with frame indices and calculated deviations.
    """
    ref = txs[0]
    rows = []
    R0 = ref.matrix[:3, :3]
    for t in txs:
        dxyz = t.center - ref.center
        Rdiff = t.matrix[:3, :3] @ R0.T
        rvec = rotation_vector_xyz_deg(Rdiff)
        rows.append({
            "subject": subject,
            "condition": condition,
            "side": side,
            "frame_index": t.index,
            "description": t.description,
            "x_mm": t.center[0],
            "y_mm": t.center[1],
            "z_mm": t.center[2],
            "dx_from_first_mm": dxyz[0],
            "dy_from_first_mm": dxyz[1],
            "dz_from_first_mm": dxyz[2],
            "translation_from_first_mm": float(np.linalg.norm(dxyz)),
            "rot_x_from_first_deg": rvec[0],
            "rot_y_from_first_deg": rvec[1],
            "rot_z_from_first_deg": rvec[2],
            "rotation_angle_from_first_deg": rotation_angle_deg(Rdiff),
        })
    return pd.DataFrame(rows)


def get_drift_stats(df: pd.DataFrame) -> Dict[str, Tuple[float, float, float]]:
    """
    Computes mean, standard deviation, and max absolute deviation for translation/rotation.
    
    Args:
        df: The calculated drift DataFrame.
        
    Returns:
        A dictionary mapping columns to (mean, std, max_abs) tuples.
    """
    stats = {}
    cols = [
        "dx_from_first_mm", "dy_from_first_mm", "dz_from_first_mm",
        "rot_x_from_first_deg", "rot_y_from_first_deg", "rot_z_from_first_deg"
    ]
    for col in cols:
        vals = df[col].astype(float).to_numpy()
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        max_abs = float(np.nanmax(np.abs(vals))) if len(vals) > 0 else 0.0
        stats[col] = (mean, std, max_abs)
    return stats


# ------------------------- High-Fidelity Plotting -------------------------
def draw_panel_statistics(ax, stats: Dict[str, Tuple[float, float, float]]):
    """
    Renders custom, high-fidelity nested bullet points for the stability stats box.
    
    Args:
        ax: Matplotlib axes object with axis off.
        stats: Statistics dictionary computed by get_drift_stats.
    """
    ax.axis("off")
    # Draw background bounding box matching mockup style
    ax.text(0.01, 0.99, "", bbox=dict(boxstyle="square,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9))
    
    # Title
    ax.text(0.02, 0.94, "Stability Statistics (relative to first position):", fontsize=13, fontweight="bold", va="top")
    
    # Coordinates of bullets
    y = 0.82
    dy = 0.05
    
    labels_trans = [
        ("dx_from_first_mm", "• X deviation:"),
        ("dy_from_first_mm", "• Y deviation:"),
        ("dz_from_first_mm", "• Z deviation:")
    ]
    
    for col, label in labels_trans:
        mean, std, max_abs = stats[col]
        ax.text(0.03, y, label, fontsize=11, fontweight="semibold", va="center")
        y -= dy
        stat_text = f"• mean = {mean:.3f} mm, std = {std:.3f} mm, max_abs = {max_abs:.3f} mm"
        ax.text(0.08, y, stat_text, fontsize=10.5, va="center")
        y -= dy * 1.3
        
    y -= 0.01  # small gap between translation and rotation
    
    labels_rot = [
        ("rot_x_from_first_deg", "• Rot-X deviation:"),
        ("rot_y_from_first_deg", "• Rot-Y deviation:"),
        ("rot_z_from_first_deg", "• Rot-Z deviation:")
    ]
    
    for col, label in labels_rot:
        mean, std, max_abs = stats[col]
        ax.text(0.03, y, label, fontsize=11, fontweight="semibold", va="center")
        y -= dy
        stat_text = f"• mean = {mean:.3f}°, std = {std:.3f}°, max_abs = {max_abs:.3f}°"
        ax.text(0.08, y, stat_text, fontsize=10.5, va="center")
        y -= dy * 1.3


def plot_drift_combined(
    df_exp: pd.DataFrame, 
    df_con: pd.DataFrame, 
    subject: str, 
    side: str, 
    out_path: Path
):
    """
    Plots the combined high-fidelity double-panel vertical stability analysis.
    
    Panel A (EXP) uses Blue lines, and Panel B (CON) uses Green lines.
    
    Args:
        df_exp: Experimental session drift DataFrame.
        df_con: Control session drift DataFrame.
        subject: Subject ID.
        side: Left or Right.
        out_path: Destination path to save the generated PNG.
    """
    fig = plt.figure(figsize=(19, 12), facecolor="white")
    
    # Outer GridSpec: 2 rows for Panel A and Panel B
    gs_outer = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.35)
    
    # Panel subgrids: 2 rows of plots, 4 columns (col 3 is stats)
    gs_a = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs_outer[0], width_ratios=[1, 1, 1, 1.4], hspace=0.28, wspace=0.22)
    gs_b = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs_outer[1], width_ratios=[1, 1, 1, 1.4], hspace=0.28, wspace=0.22)
    
    # Configuration lists
    cols_config = [
        ("dx_from_first_mm", "X position deviation", "Deviation (mm)"),
        ("dy_from_first_mm", "Y position deviation", "Deviation (mm)"),
        ("dz_from_first_mm", "Z position deviation", "Deviation (mm)"),
        ("rot_x_from_first_deg", "Rot-X deviation", "Deviation (degrees)"),
        ("rot_y_from_first_deg", "Rot-Y deviation", "Deviation (degrees)"),
        ("rot_z_from_first_deg", "Rot-Z deviation", "Deviation (degrees)")
    ]
    
    # ── PANEL A (Experimental Condition) ──
    exp_color = "#2563eb"  # Elegant vibrant blue
    exp_n = len(df_exp)
    
    # Text annotation panel titles
    fig.text(0.35, 0.965, f"{subject}_exp - Stability Analysis (n={exp_n})", fontsize=13, fontweight="bold", ha="center")
    
    # Vertical panel labels
    fig.text(0.015, 0.91, "A", fontsize=24, fontweight="bold", ha="center")
    fig.text(0.015, 0.70, "experimental condition", fontsize=15, fontweight="bold", rotation="vertical", ha="center", va="center", color="#4b5563")
    
    exp_stats = get_drift_stats(df_exp)
    
    # Draw plots for Panel A
    for idx, (col, title, ylabel) in enumerate(cols_config):
        row = 0 if idx < 3 else 1
        c = idx % 3
        ax = fig.add_subplot(gs_a[row, c])
        
        y_vals = df_exp[col].astype(float).to_numpy()
        x_vals = np.arange(len(df_exp))
        
        ax.plot(x_vals, y_vals, color=exp_color, marker="o", markersize=3, linewidth=1.0, alpha=0.9)
        ax.axhline(0, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
        
        # Grid and design polish
        ax.grid(True, linestyle="-", linewidth=0.5, color="#e5e7eb", alpha=0.8)
        ax.set_facecolor("#f9fafb")
        ax.tick_params(axis="both", labelsize=9)
        
        # Labels and Y limits matching mockup exactly
        ax.set_ylim(-2.0, 2.0)
        ax.set_yticks([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
        ax.set_title(title, fontsize=10, fontweight="semibold")
        ax.set_xlabel("Position Index" if row == 1 else "", fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        
    # Draw stats box for Panel A
    ax_stats_a = fig.add_subplot(gs_a[:, 3])
    draw_panel_statistics(ax_stats_a, exp_stats)
    
    # ── PANEL B (Control Condition) ──
    con_color = "#16a34a"  # Elegant grass green
    con_n = len(df_con)
    
    fig.text(0.35, 0.485, f"{subject}_con - Stability Analysis (n={con_n})", fontsize=13, fontweight="bold", ha="center")
    
    fig.text(0.015, 0.43, "B", fontsize=24, fontweight="bold", ha="center")
    fig.text(0.015, 0.22, "control condition", fontsize=15, fontweight="bold", rotation="vertical", ha="center", va="center", color="#4b5563")
    
    con_stats = get_drift_stats(df_con)
    
    # Draw plots for Panel B
    for idx, (col, title, ylabel) in enumerate(cols_config):
        row = 0 if idx < 3 else 1
        c = idx % 3
        ax = fig.add_subplot(gs_b[row, c])
        
        y_vals = df_con[col].astype(float).to_numpy()
        x_vals = np.arange(len(df_con))
        
        ax.plot(x_vals, y_vals, color=con_color, marker="o", markersize=3, linewidth=1.0, alpha=0.9)
        ax.axhline(0, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
        
        ax.grid(True, linestyle="-", linewidth=0.5, color="#e5e7eb", alpha=0.8)
        ax.set_facecolor("#f9fafb")
        ax.tick_params(axis="both", labelsize=9)
        
        ax.set_ylim(-2.0, 2.0)
        ax.set_yticks([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
        ax.set_title(title, fontsize=10, fontweight="semibold")
        ax.set_xlabel("Position Index" if row == 1 else "", fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        
    # Draw stats box for Panel B
    ax_stats_b = fig.add_subplot(gs_b[:, 3])
    draw_panel_statistics(ax_stats_b, con_stats)
    
    # Adjust outer canvas boundaries
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.05)
    
    # Save output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved combined drift plot to: %s", out_path)


# ------------------------- Main Script Execution -------------------------
def main():
    """
    Main entry point to read ratings CSV, find matching subject GUMMarkers XML files,
    compute deviations, and generate combined plots for Left and Right hemispheres.
    """
    # Define paths relative to repo root
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = repo_root / "data" / "gum" / "actual" / "citrus-offline_participant_ratings - ratings.csv"
    actual_dir = repo_root / "data" / "gum" / "actual"
    output_dir = repo_root / "derivatives" / "stability_analysis"
    
    if not csv_path.exists():
        log.error("Ratings CSV file not found at: %s", csv_path)
        sys.exit(1)
        
    log.info("Reading ratings from CSV: %s", csv_path)
    df_csv = pd.read_csv(csv_path)
    # Rename the first column which holds the subject ID
    df_csv = df_csv.rename(columns={df_csv.columns[0]: "subject"})
    
    # Group subjects
    subjects = df_csv["subject"].dropna().unique()
    log.info("Found %d subjects in ratings file: %s", len(subjects), list(subjects))
    
    # Loop over subjects and hemispheres to construct combined plots
    for sub in subjects:
        sub_df = df_csv[df_csv["subject"] == sub]
        
        for side_label, side_key in [("L", "left"), ("R", "right")]:
            log.info("Processing subject %s side %s...", sub, side_key)
            
            # Filter rows for EXP and CON for this side
            exp_row = sub_df[(sub_df["condition"] == "EXP") & (sub_df["hemisphere"] == side_label)]
            con_row = sub_df[(sub_df["condition"] == "CON") & (sub_df["hemisphere"] == side_label)]
            
            if exp_row.empty or con_row.empty:
                log.warning("Skipping subject %s side %s due to missing condition data in CSV.", sub, side_label)
                continue
                
            try:
                # ── Extract EXP data ───────────────────────────────────────────
                exp_file = exp_row.iloc[0]["localite_file"]
                if not exp_file.endswith(".xml"):
                    exp_file += ".xml"
                exp_xml_path = actual_dir / sub / exp_file
                exp_start = int(exp_row.iloc[0]["xml_start"])
                exp_end = int(exp_row.iloc[0]["xml_end"])
                
                log.info("  Loading EXP XML: %s [frames %d-%d]", exp_xml_path.name, exp_start, exp_end)
                exp_txs = parse_gummarkers(exp_xml_path)
                exp_selected = select_range(exp_txs, exp_start, exp_end)
                df_exp = compute_drift_dataframe(sub, "EXP", side_key, exp_selected)
                
                # ── Extract CON data ───────────────────────────────────────────
                con_file = con_row.iloc[0]["localite_file"]
                if not con_file.endswith(".xml"):
                    con_file += ".xml"
                con_xml_path = actual_dir / sub / con_file
                con_start = int(con_row.iloc[0]["xml_start"])
                con_end = int(con_row.iloc[0]["xml_end"])
                
                log.info("  Loading CON XML: %s [frames %d-%d]", con_xml_path.name, con_start, con_end)
                con_txs = parse_gummarkers(con_xml_path)
                con_selected = select_range(con_txs, con_start, con_end)
                df_con = compute_drift_dataframe(sub, "CON", side_key, con_selected)
                
                # ── Generate Combined Plot ─────────────────────────────────────
                plot_name = f"{sub}_drift_combined_{side_key}.png"
                out_plot_path = output_dir / plot_name
                plot_drift_combined(df_exp, df_con, sub, side_key, out_plot_path)
                
            except Exception as e:
                log.error("Failed to process subject %s side %s: %s", sub, side_key, str(e), exc_info=True)


if __name__ == "__main__":
    main()
