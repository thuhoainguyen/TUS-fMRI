#!/bin/bash
# Run full resting-state fMRI analysis pipeline for CITRUS sgACC network
# Run from project root: bash code/rs_fmri/run_all.sh

set -e
cd "$(dirname "$0")"

echo "======================================================"
echo " CITRUS rsFMRI Pipeline — sgACC Network Analysis"
echo "======================================================"
echo ""

echo "[Step 1] Confound regression + bandpass + smoothing..."
python 01_confound_regression.py
echo ""

echo "[Step 2] Seed-based FC maps (sgACC_L, sgACC_R)..."
python 02_seed_fc.py
echo ""

echo "[Step 3] ROI network connectivity extraction..."
python 03_roi_network.py
echo ""

echo "[Step 4] Temporal visualisation (line plots + heatmap + brain maps)..."
python 04_temporal_plots.py
echo ""

echo "[Step 5] Radar charts (group + delta + per-subject)..."
python 05_radar_chart.py
echo ""

echo "[Step 6] Group-level statistics..."
python 06_group_analysis.py
echo ""

echo "======================================================"
echo " Done. Outputs in: derivatives/rs_fmri/"
echo "======================================================"
