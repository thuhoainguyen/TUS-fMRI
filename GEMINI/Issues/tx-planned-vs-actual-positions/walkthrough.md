# GEMINI Issue Walkthrough: Planned vs Actual Transducer Positions Visualisation

## Phase: Verification
Completed by: @author Hoai Thu Nguyen

This document concludes the verification of the standalone planned vs actual medoid transducer positions mesh plotting pipeline.

---

### 1. Verification Details
- **Script Location**: 
  - `code/Tx_planned_vs_actual_positions.py`
- **Output Artifacts**:
  - `derivatives/planned_vs_actual_positions/sub-XX_planned_vs_actual_positions.png`
- **Logic Documents**:
  - `GEMINI/Issues/tx-planned-vs-actual-positions/analysis_design_planning.md`

---

### 2. Implementation Summary
- Successfully created a standalone script to load and overlay the planned vs actual medoid transducer positions in 4 orthogonal views:
  - **Planned indices**: Resolved from `data/input/planned_positions_index.csv`.
  - **Planned XML file**: Loaded from `data/input/{subject}/{subject}_GUMMarkers*.xml`.
  - **Actual (Medoid) XML file**: Unique `.xml` resolved under `data/gum/medoid/` for each subject. Since some filenames contained L/R prefix typos relative to the physical coordinates (e.g. `sub-03` has `L59-R146` but index 59 is Right and 146 is Left), actual Left/Right transducers are resolved robustly by checking physical RAS X-coordinates ($X < 120.0$ mm for Left, $X \ge 120.0$ mm for Right).
- **Disk Geometry & Coloring**:
  - Enforced a uniform **`62 mm`** diameter (31.0 mm radius) for all circular disc projections on the scalp boundary mesh.
  - Used **Yellow** (`#ffc107`) for planned, **Blue** (`#1e90ff`) for actual, and **Green** (`#22c55e`) for visual overlap.
- **Robust Path Intersection Rasterization**:
  - Applied the exact 2-D matplotlib path contains-points rasterization logic from `citrus_offline_report_v18.py` to paint the exact green overlap mask cleanly without any Shapely library dependencies.

All code is fully self-contained, robustly written, and tagged with the `@author Hoai Thu Nguyen` signature.
