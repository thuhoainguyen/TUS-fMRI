# GEMINI Issue: Tx Actual Positions Heatmap

## Phase: Planning
Created by: @author Hoai Thu Nguyen

This issue tracks the extraction, creation, and documentation of the standalone actual transducer position mesh heatmap plotting script `Tx_actual_positions.py`.

### 1. Analysis Phase
We reviewed the original `plot_mesh_heatmap` function and its related configurations. We found:
- Actual positions are loaded from `data/gum/{subject}/{localite_file}`.
- Left and Right transducer recorded frames are filtered by inclusive start and end indices specified in `./data/gum/citrus-offline_participant_ratings - ratings.csv`.
- Scalp mesh is extracted from `data/simnibs/{subject}/{subject}.msh` with physical tag `1005`.
- A Gaussian spatial kernel of size $\sigma = 15.0$ mm is used to calculate the spatial density of actual transducer centers over all scalp triangle centroids.
- A 1x4 subplot grid displays 4 orthogonal 3D projections (Left, Right, Front, Top) with a ScalarMappable custom colorbar using the `hot` colormap.

### 2. Design Phase
We will design a standalone pipeline that:
- Reads the ratings CSV file using standard python/pandas.
- Locates XML GUMMarker files and `.msh` mesh files.
- Computes standard LPS-to-RAS flipped transformation matrices.
- Snaps centers and computes Gaussian kernel density.
- Automatically handles both EXP and CON conditions for all subjects (`sub-03`, `sub-04`, `sub-05`, `sub-06`, `sub-11`).
- Saves high-quality figures to `derivatives/actual_positions/` (e.g. `sub-03_actual_positions_exp.png`).

### 3. Execution Plan
1. Create `code/Tx_actual_positions.py` containing the logic and multi-subject runner.
2. Create `logic code/actual_positions.md` documenting the mathematical logic.
3. Execute the code to render the figures.
4. Verify the outputs.
