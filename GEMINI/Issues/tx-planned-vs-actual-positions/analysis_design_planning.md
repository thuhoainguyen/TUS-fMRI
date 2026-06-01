# GEMINI Issue Lifecycle: Tx Planned vs Actual Positions Visualisation

**Issue ID**: `tx-planned-vs-actual-positions`
**Phase**: Analysis, Design, & Planning
**Author**: @author Hoai Thu Nguyen

---

## 1. Analysis Phase

### 1.1 Goal
Create a standalone Python script `code/Tx_planned_vs_actual_positions.py` that visualizes planned vs actual transducer positions side-by-side on left and right head mesh, highlighting the spatial overlap in green, using 62 mm diameter transducers.

### 1.2 Input Data Sources
- **Planned indices**: `data/input/planned_positions_index.csv` (Left and Right planned XML index).
- **Planned XML file**: `data/input/{subject}/{subject}_GUMMarkers*.xml`.
- **Actual XML file (Medoid)**: A unique `.xml` file under `data/gum/medoid/` for each subject.
  - The Left and Right actual indices are resolved robustly by checking physical RAS coordinates ($X < 120.0$ mm for Left hemisphere, $X \ge 120.0$ mm for Right hemisphere) inside the XML to prevent filename typo mismatches.
- **Triangulation Mesh**: SimNIBS scalp head mesh `data/simnibs/{subject}/{subject}.msh`.

---

## 2. Design Phase

### 2.1 Color Coding & Sizing
- Transducer size: `DISC_R = 31.0` mm (62 mm diameter) for both planned and actual.
- **Yellow** (`#ffc107`) for planned transducer discs.
- **Blue** (`#1e90ff`) for actual transducer discs.
- **Green** (`#22c55e`) for visual overlap.

### 2.2 Overlap Computation
We will project the 3-D circular discs onto the 2-D orthographic projection planes, and then compute their intersection using Matplotlib's Path rasterization. Points inside both projected planned and actual paths will be painted green.

---

## 3. Planning & Verification Phase

### 3.1 Step-by-Step Plan
1. **Initialize Issue and Plan**: Awaiting approval.
2. **Draft Script**: Write `code/Tx_planned_vs_actual_positions.py`.
3. **Execute and Verify**: Generate and check the overlay PNG figures in `derivatives/planned_vs_actual_positions/`.
