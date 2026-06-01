# GEMINI Issue Lifecycle: Tx Pressure & Temperature Maps Extraction

**Issue ID**: `tx-pressure-temperature-maps`
**Phase**: Analysis, Design, & Planning
**Author**: @author Hoai Thu Nguyen

---

## 1. Analysis Phase

### 1.1 Goal
Extract all logic and visualization functions from `code/citrus_offline_report_v18.py` related to planned/actual pressure and temperature maps, and compile them into a new, self-contained standalone Python script: `code/Tx_pressure_temperature_maps.py`.

### 1.2 Identified Functions for Extraction
From `citrus_offline_report_v18.py`, we need to extract:
1. **Core Data Structures**:
   - `FocalMetrics` dataclass
2. **Coordinate & Spatial Helpers**:
   - `voxel_to_world`, `world_to_voxel`, `voxel_volume_mm3`, `resample_to_target`
   - `robust_t1_limits`, `mask_centroid_vox`
   - `slice2d`, `point_xy`, `crop_limits`, `apply_crop`, `adjust_xy`
   - `safe_contour`, `safe_imshow_overlay`, `add_lr_labels`
3. **Map Processing & Focal Analytics**:
   - `pressure_to_mpa`, `load_pressure_mpa`
   - `sphere_mask_around_world`, `largest_component_containing_peak`
   - `focal_volume_from_pressure`, `pca_dimensions_mm`
   - `dice`, `percent`
   - `compute_focal_metrics`
4. **Plotting & Visualization**:
   - `plot_map_mosaic` (generates the 3xN axial/sagittal/coronal grid for a map)
   - `plot_planned_actual_focal_overlay` (generates the 2x3 planned/actual overlay figure)
   - `plot_planned_actual_focal` (backward-compatible two-row grid)
   - `plot_overlap_figure` (single-row planned/actual overlay)

### 1.3 Target Directory Structure
- The extracted code will live in [Tx_pressure_temperature_maps.py](file:///Users/hoaithunguyen/Projects/Master%20thesis/CITRUS/code/Tx_pressure_temperature_maps.py).
- Output figures and reports will be saved under `derivatives/pressure_temperature_maps/`.

---

## 2. Design Phase

### 2.1 File Structure and Imports
The new script `Tx_pressure_temperature_maps.py` will have the following outline:
- File header with description and `@author Hoai Thu Nguyen` tagging.
- Independent, clean logging setup using a logger named `Tx_maps`.
- Imports: standard library + `numpy`, `pandas`, `nibabel`, `scipy.ndimage`, `matplotlib`, `nilearn`.
- Extracted utility helpers, metrics computation, and plotting functions.
- Self-contained `__main__` entry block designed to automatically find maps, run metrics, and render figures for all five standard subjects (`sub-03`, `sub-04`, `sub-05`, `sub-06`, `sub-11`).

### 2.2 Glob-Based Map Matching Logic
For each subject, condition (`exp`, `con`), and hemisphere (`left`, `right`), we will resolve file paths:
- **T1w Image**: `data/input/{subject}/{subject}_T1w_kplan.nii.gz`
- **sgACC ROI Mask**: `data/input/{subject}/sgACC_BA25_{side_capitalized}_kplan.nii.gz`
- **Search Folder**: `data/output/{subject}/planning/{cond_folder}/` and `data/output/{subject}/posthoc/{cond_folder}/`
- **Planned Pressure**: `*Tx-2_{side_code}_pos-*{cond} - Pressure.nii.gz` under `planning/`
- **Planned Temperature**: `*Tx-2_{side_code}_pos-*{cond} - Temperature.nii.gz` under `planning/`
- **Actual Pressure**: `*Tx-2_{side_code}_pos-medoid-*{post-hoc/actual} - Pressure.nii.gz` under `posthoc/`
- **Actual Temperature**: `*Tx-2_{side_code}_pos-medoid-*{post-hoc/actual} - Temperature.nii.gz` under `posthoc/`

If all 4 maps (`pplan`, `pact`, `tplan`, `tact`) exist for a combination, it is processed. Otherwise, the script will gracefully log a message and continue.

---

## 3. Planning & Verification Phase

### 3.1 Step-by-Step Plan
1. **Bootstrap Verification**: Verify codebase environment.
2. **Draft Plan & Get Approval**: Submit this plan for user feedback.
3. **Write Script**: Implement `Tx_pressure_temperature_maps.py` containing all extracted logic, with a fully automated `__main__` block.
4. **Compile & Run Test**: Perform a compilation check and execute the script for all available subjects.
5. **Verify Outputs**: Ensure high-quality figures are successfully generated and saved to `derivatives/pressure_temperature_maps/`.

### 3.2 Verification Gate
- Compilation is successful with `python -m py_compile code/Tx_pressure_temperature_maps.py`.
- Running `python code/Tx_pressure_temperature_maps.py` finishes with exit code `0` and populates `derivatives/pressure_temperature_maps/` with mosaic and overlay PNG figures.
