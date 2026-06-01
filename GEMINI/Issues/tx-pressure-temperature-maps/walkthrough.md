# GEMINI Issue Walkthrough: Tx Planned and Actual Maps Extraction & Anatomy Alignment

## Phase: Verification
Completed by: @author Hoai Thu Nguyen

This document concludes the verification of the standalone planned and actual pressure/temperature map extraction pipelines, as well as the L/R anatomical display alignment inside `code/anatomy.py`.

---

### 1. Verification Details
- **Script Locations**: 
  - `code/Tx_planned_maps.py`
  - `code/Tx_actual_maps.py`
  - `code/anatomy.py`
- **Output Artifacts**:
  - `derivatives/planned_maps/sub-XX_planned_pressure_cond_side_mosaic.png`
  - `derivatives/planned_maps/sub-XX_planned_temperature_cond_side_mosaic.png`
  - `derivatives/actual_maps/sub-XX_actual_pressure_cond_side_mosaic.png`
  - `derivatives/actual_maps/sub-XX_actual_temperature_cond_side_mosaic.png`
  - `derivatives/anatomy/sub-XX_anatomy_report.png`
  - `derivatives/anatomy/sub-XX_sgacc_overlay_report.png`
- **Logic Documents**:
  - `GEMINI/Issues/tx-pressure-temperature-maps/analysis_design_planning.md`

---

### 2. Implementation Summary

Consistent with the `Tx_planned_positions.py` and `Tx_actual_positions.py` separation, we have created two standalone, dedicated map processing scripts and refactored anatomical slices to be universally coherent:

1. **[Tx_planned_maps.py](file:///Users/hoaithunguyen/Projects/Master%20thesis/CITRUS/code/Tx_planned_maps.py)**:
   - Processes planned pressure and temperature simulation maps from `data/output/{sub}/planning/`.
   - Snaps coordinates to a conservative brain tissue mask and extracts planned -3 dB focal zone metrics.
   - Generates whole-head orthographic slice mosaics under `derivatives/planned_maps/`.

2. **[Tx_actual_maps.py](file:///Users/hoaithunguyen/Projects/Master%20thesis/CITRUS/code/Tx_actual_maps.py)**:
   - Processes actual post-hoc pressure and temperature maps from `data/output/{sub}/posthoc/`.
   - Automatically handles cases where post-hoc simulation outputs are available (such as `sub-05` experimental runs).
   - Generates whole-head orthographic slice mosaics under `derivatives/actual_maps/`.

3. **[anatomy.py](file:///Users/hoaithunguyen/Projects/Master%20thesis/CITRUS/code/anatomy.py) Orientation Alignment**:
   - Refactored the anatomical slice plotting code to align with the neurological convention used by the map mosaic plots.
   - Horizontally inverted the coronal (`'y'`) and axial (`'z'`) slice viewports using Matplotlib's `ax.invert_xaxis()` function so that the left hemisphere is consistently displayed on the left, and the right hemisphere is displayed on the right.
   - Updated the manual text labels from `"R"` (left) and `"L"` (right) to `"L"` (left) and `"R"` (right) to ensure complete coherence.

All scripts are fully self-contained, robustly designed, and tagged with `@author Hoai Thu Nguyen`.
