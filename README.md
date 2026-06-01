# TUS-fMRI — CITRUS Offline Protocol

**Author:** Hoai Thu Nguyen  
**Affiliation:** Neuroimaging Center (NIC), Medical Center, Johannes Gutenberg University Mainz

---

## About

This repository contains the analysis code for the CITRUS master's thesis project, investigating **low-intensity transcranial focused ultrasound (TUS)** targeting the **subgenual anterior cingulate cortex (sgACC)** using an offline, pre-fMRI protocol. The study follows an experimental vs. control session design across multiple subjects.

## Project Structure

```
.
├── code/               # Main analysis scripts
├── knowledge/          # Notes and documentation
└── logic code/         # Step-by-step logic breakdowns of each script
```

## Scripts

| Script | Purpose |
|---|---|
| `anatomy.py` | Generates T1w/density overlay and sgACC target localization figures per subject |
| `Tx_planned_positions.py` | Visualizes planned transducer positions on head mesh |
| `Tx_actual_positions.py` | Visualizes actual transducer positions recorded during sessions |
| `Tx_planned_vs_actual_positions.py` | Compares planned vs. actual transducer placement |
| `Tx_planned_maps.py` | Generates pressure & temperature maps for planned sonication |
| `Tx_actual_maps.py` | Generates pressure & temperature maps for actual sonication |
| `blinding_analysis.py` | Statistical analysis of blinding success and subjective ratings (sensation, sound, tiredness) |
| `stability_analysis.py` | Stability analysis across sessions and subjects |
| `reference-function_v18.py` | Reference functions and examples |

## Installation

```bash
pip install -r requirements.txt
```

**Python dependencies:** nibabel, nilearn, numpy, matplotlib, scipy, pandas, seaborn, meshio, pyvista
