# ok-plan — features

A small Python toolbox that turns a single transcranial-ultrasound planning run
(SimNIBS CHARM + pressure / temperature NIfTIs + a target ROI) into a
self-contained **HTML QC report**. No servers, no manual plotting — one CLI
call produces an auditable artefact you can drop into a study folder.

## 1. Inputs it accepts

Per subject, a minimal input pack:

- **SimNIBS CHARM / m2m** folder (`final_tissues.nii.gz`)
- **Subject T1w** NIfTI (kplan space)
- **Pressure** NIfTI (Pa, real or complex amplitude)
- **Temperature** NIfTI (absolute °C)
- **Target ROI** NIfTI (binary or labelled)
- *(optional)* ANTs **transforms** (`*_kplan2std_*`) for atlas warping
- *(optional)* **Julich Brain Atlas 3.1** directory (LH/RH + XML) for
  anatomical context

All inputs are declared either as CLI flags or via a single JSON project file
(`--config project.json`); relative paths resolve against the config's
directory for easy relocatable layouts.

## 2. What it produces

Under `out/<subject>/`:

- `nifti_files/` — every input resampled to the pressure-map grid
  (nearest-neighbour for labels, trilinear for continuous maps)
- `derived_masks/` — binary tissue masks extracted from
  `final_tissues.nii.gz`: `scalp`, `skull`, `inside_skull` (brain + CSF + GM +
  WM), `eyes`
- `report.html` — the QC report, with all PNGs embedded as base64 (one file,
  portable, no external assets)

## 3. Report sections

1. **Summary cards** — ITRUSST NSR-threshold checks in two rows
   - *Mechanical* (MI for scalp / brain+ / eyes)
   - *Thermal* (ΔT max, T abs max, CEM43 max brain, CEM43 max bone)
   - *Targeting efficacy* (−3 dB / −6 dB ROI coverage, on-target %, P max at
     target, I<sub>sppa</sub>)
   - Each row tagged **OK** / **CHECK** against the published NSR limits.
2. **Targeting and exposure**
   - ROI ortho view on the T1w
   - Pressure overlay (MPa)
   - −3 dB focus ∪ ROI at 50 % transparency (and −6 dB when `--include-6db`)
3. **Overlap and pressure statistics**
   - Target coverage table (voxel counts, %ROI, %focus, on-target %)
   - Pressure statistics inside the focus ∩ ROI intersection
4. **Temperature** — ortho view through the hot-spot, ΔT and absolute T maps
5. **Derived tissue masks** — mosaic of scalp / skull / brain+ / eyes on a
   black background with a calming fixed palette
6. **Per-mask safety metrics** — MI, CEM43<sub>max</sub>, ΔT<sub>max</sub>,
   T<sub>abs,max</sub> per tissue, each compared to its ITRUSST NSR limit,
   with an expanded Sapareto–Dewey CEM43 footnote
7. **Anatomical context** (when atlas + transforms supplied)
   - Target-ROI atlas label (name from the Julich XML, marked `*` as the
     target)
   - Table of all atlas regions overlapping the focus, with L/R hemisphere
     suffixes derived from a separate warped hemisphere mask (no label
     collisions, 207-region space preserved)
   - A single ortho overlay of the −3 dB focus with all overlapping atlas
     regions at 50 % transparency, categorical colours, and a legend

## 4. Metrics implemented

- **Mechanical Index**: `MI = p_max[Pa] / 10⁶ / √f[MHz]`
- **ΔT**: `T − T_baseline` (default 37 °C)
- **CEM43 (Sapareto–Dewey)**: steady-state single-map form
  `CEM43_max = exposure · R^(43 − T_max)` with `R = 0.25` for `T ≥ 43 °C`,
  `R = 0.5` otherwise; same formulation as k-Wave / k-Plan
- **Focus thresholds**: −3 dB and (optional) −6 dB of the peak inside the
  head, binarised and intersected with the ROI
- **ITRUSST NSR thresholds** baked in: MI ≤ 1.9 (brain+/scalp) / 0.4 (eyes),
  ΔT ≤ 2 °C, T<sub>abs</sub> ≤ 39 °C, CEM43 ≤ 2 (brain) / 16 (bone) / 21
  (skin)

## 5. Atlas / anatomical pipeline

- Merges Julich 3.1 LH + RH NIfTIs preserving the original 1–207 grayvalue
  space and stores hemisphere origin in a separate mask (`1 = L`, `2 = R`)
- Warps both the merged atlas and the hemisphere mask MNI→subject with
  `antsApplyTransforms` (NearestNeighbor)
- Splits region statistics by `(label, hemisphere)` so bilateral regions get
  distinct L / R rows with correct voxel counts
- Derives the report's ROI name from the atlas itself (highest-overlap
  region), not a filename heuristic

## 6. Formatting touches

- HTML layout: left TOC + main panel, wraps on narrow screens
- CEM43 rendered as `M × 10⁻ⁿ` (unicode superscript + non-breaking space) so
  column widths stay stable in narrow tables
- Status badges: green **OK**, red **CHECK** (no false "EXCEEDS" certainty)
- All figures share the simulation domain grid; ROI and focus cuts pass
  through the ROI centroid; temperature cuts pass through the ΔT<sub>max</sub>
  voxel

## 7. Usage

```bash
ok-plan --config examples/ok-plan.sub-06.exp-focused-R.json --include-6db
```

or, fully CLI-driven:

```bash
ok-plan \
  --subject sub-02 \
  --charm-dir /path/to/m2m_sub-02 \
  --t1 T1w_kplan.nii.gz \
  --pressure pressure.nii.gz \
  --temperature temperature.nii.gz \
  --roi target_ROI_kplan.nii.gz \
  --transforms-dir /path/to/transforms/sub-02 \
  --atlas-dir /path/to/julich_atlas \
  --center-frequency-mhz 0.286 \
  --exposure-duration-min 1.0 \
  --include-6db
```

CLI flags override JSON defaults, so the same config can be reused across
protocols with a one-line override (e.g. a different pressure map for each
sonication protocol).

## 8. Design choices

- **One file, one run**: the HTML report is fully self-contained (base64
  PNGs); no server, no relative-asset breakage when emailed or archived
- **Everything on the sim grid**: stats and figures share a single reference
  grid, avoiding subtle resampling mismatches between pressure, temperature,
  ROI, and tissue masks
- **Explicit, published thresholds**: ITRUSST NSR values are hard-coded and
  referenced in-line so a reviewer can see what the tool is checking against
- **No silent guesses**: the atlas pipeline preserves the published 207-label
  space; hemispheres come from a separate deterministic mask, not label
  offsets
