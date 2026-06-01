# GEMINI Issue Lifecycle: Blinding and Participant Ratings Analysis

**Issue ID**: `blinding-analysis`
**Phase**: Analysis, Design, & Planning
**Author**: @author Hoai Thu Nguyen

---

## 1. Analysis Phase

### 1.1 Goal
Create a comprehensive, robust Python analysis pipeline `code/blinding_analysis.py` that processes the `data/blinding/` CSV files:
1. `sensation_sound_tiredness.csv`
2. `forced_choice_confidence.csv`

The pipeline will perform within-subject and between-subject statistical analyses to assess participant blinding quality, compare physical/auditory sensations across conditions, track tiredness trajectories over time, and output professional, publication-ready figures to `derivatives/blinding/`.

### 1.2 Data Sources & Variables
* **`sensation_sound_tiredness.csv`**:
  * $N = 5$ subjects (`sub-03`, `sub-04`, `sub-05`, `sub-06`, `sub-11`).
  * Structured as 4 rows per subject, mapping:
    * `EXP, L` $\rightarrow$ $T_0$ tiredness.
    * `EXP, R` $\rightarrow$ $T_{15}$ tiredness.
    * `CON, L` $\rightarrow$ $T_{30}$ tiredness.
    * `CON, R` $\rightarrow$ $T_{45}$ tiredness.
  * *Scale*: Sound (1-5), Sensation (1-5), Tiredness (1-5).
* **`forced_choice_confidence.csv`**:
  * Correct condition order (`condition_order` e.g., `EXP-CON` or `CON-EXP`) and participant's final guess (`forced_choice` e.g., `CON-EXP`).
  * *Scale*: Confidence (1-5).

---

## 2. Design Phase

### 2.1 Statistical Calculations

1. **Blinding Efficacy (Between-Subject)**:
   * **Accuracy Rate**: Compute the percentage of subjects who guessed correctly (`forced_choice == condition_order`).
   * **Binomial Test**: Test the null hypothesis that guess accuracy is at chance level ($p_0 = 0.50$).
   * **Bang's Blinding Index (BBI)**:
     $$BBI = \frac{N_{\text{correct}} - N_{\text{incorrect}}}{N_{\text{total}}}$$
     * BBI values range from $-1$ (opposite guess) to $+1$ (complete unblinding), where $0$ indicates perfect blinding.

2. **Within-Subject Sensation & Sound Contrast**:
   * **Paired Comparison**: Wilcoxon signed-rank tests (due to small $N=5$) comparing `sound` and `sensation` ratings between `EXP` and `CON` sessions.

3. **Tiredness Trajectory (Within-Subject)**:
   * Track sequential tiredness rating changes ($T_0 \rightarrow T_{15} \rightarrow T_{30} \rightarrow T_{45}$) for each subject.
   * Calculate mean tiredness and standard error at each timepoint.

4. **Sensation Contrast vs. Blinding Efficacy (Between-Subject)**:
   * Calculate absolute differences for each subject:
     $$\Delta\text{Sensation} = \text{Sensation}_{\text{EXP}} - \text{Sensation}_{\text{CON}}$$
     $$\Delta\text{Sound} = \text{Sound}_{\text{EXP}} - \text{Sound}_{\text{CON}}$$
   * Compare mean $\Delta\text{Sensation}$ and $\Delta\text{Sound}$ between **Correct Guessers** and **Incorrect Guessers** using Mann-Whitney U tests.

5. **Blinding Guess vs. Confidence (Between-Subject)**:
   * Compare confidence rates between Correct and Incorrect Guessers to determine if correct guesses were accompanied by higher subjective certainty.

### 2.2 Premium Visualization Aesthetics
Using a cohesive, elegant theme with HSL-tailored colors (similar to the project's aesthetics):
* **Plots to generate**:
  1. `tiredness_trajectory.png`: Line plot with error bands displaying the sleepiness timeline.
  2. `sound_sensation_comparison.png`: Side-by-side box/swarm plots for EXP vs. CON.
  3. `blinding_accuracy_confidence.png`: Elegant grouped bar/scatter chart representing each participant's guess, correctness, and confidence.
  4. `sensory_contrast_scatter.png`: Scatter plot of $\Delta\text{Sensation}$ vs. $\Delta\text{Sound}$ categorized by guess accuracy.

---

## 3. Planning & Verification Phase

### 3.1 Step-by-Step Plan
1. **Initialize Issue and Plan**: Create the implementation plan and obtain user approval.
2. **Draft Script**: Write `code/blinding_analysis.py` with rigorous numpy/pandas data processing and scipy statistical validation.
3. **Execute & Generate Outputs**: Run the script to produce tabular output files and premium PNG figures in `derivatives/blinding/`.
4. **Verification**: Confirm correctness of binomial statistics, Bang's Blinding Index, paired Wilcoxon tests, and visually inspect generated plots.
5. **Walkthrough**: Document the final results, embedding plots, and write the lifecycle summary in `GEMINI/Issues/blinding-analysis/walkthrough.md`.
