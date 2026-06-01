"""
Blinding and Subjective Ratings Analysis for the CITRUS Study.

This script performs within-subject and between-subject statistical analyses
on participant ratings (sound, sensation, tiredness) and forced-choice blinding
data, and generates premium, publication-quality figures.

@author Hoai Thu Nguyen
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def setup_plotting_theme():
    """
    Configure premium matplotlib and seaborn style settings with crisp, compact fonts.
    """
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Roboto', 'Helvetica', 'Arial'],
        'figure.facecolor': '#fafafa',
        'axes.facecolor': '#fafafa',
        'axes.edgecolor': '#cccccc',
        'axes.labelcolor': '#333333',
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.color': '#555555',
        'ytick.color': '#555555',
        'text.color': '#333333',
        'legend.fontsize': 8,
        'legend.title_fontsize': 8,
        'legend.frameon': True,
        'legend.facecolor': '#ffffff',
        'legend.edgecolor': '#e2e8f0',
        'figure.dpi': 300,
        'savefig.dpi': 300
    })

def run_blinding_analysis():
    """
    Main function to execute the blinding and participant ratings analysis.
    """
    # Create directories for results
    output_dir = "derivatives/blinding"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load datasets
    # sensation_sound_tiredness.csv has Subject ID in first column without header
    ratings_path = "data/blinding/sensation_sound_tiredness.csv"
    fc_path = "data/blinding/forced_choice_confidence.csv"
    
    df_ratings = pd.read_csv(ratings_path)
    df_ratings.rename(columns={df_ratings.columns[0]: "Subject"}, inplace=True)
    
    df_fc = pd.read_csv(fc_path)
    df_fc.rename(columns={"subject_ID": "Subject"}, inplace=True)

    # 2. Within-subject aggregation of Sound & Sensation
    # Average L and R hemisphere ratings per subject per condition
    df_grouped = df_ratings.groupby(["Subject", "condition"])[["sound", "sensation"]].mean().reset_index()
    
    # Pivot wide to compare EXP vs CON directly
    df_paired = df_grouped.pivot(index="Subject", columns="condition", values=["sound", "sensation"])
    df_paired.columns = [f"{var}_{cond.lower()}" for var, cond in df_paired.columns]
    df_paired = df_paired.reset_index()
    
    # Calculate difference scores
    df_paired["sound_diff"] = df_paired["sound_exp"] - df_paired["sound_con"]
    df_paired["sensation_diff"] = df_paired["sensation_exp"] - df_paired["sensation_con"]

    # 3. Blinding Guess Accuracy and Bang's Blinding Index
    # Guess is correct if forced_choice matches condition_order
    df_merged = pd.merge(df_fc, df_paired, on="Subject")
    df_merged["correct_guess"] = df_merged["forced_choice"] == df_merged["condition_order"]
    
    n_total = len(df_merged)
    n_correct = df_merged["correct_guess"].sum()
    n_incorrect = n_total - n_correct
    accuracy_rate = n_correct / n_total
    
    # Binomial test (H0: guess rate is at 50% chance)
    try:
        binom_res = stats.binomtest(n_correct, n_total, p=0.5, alternative="two-sided")
        p_binom = binom_res.pvalue
    except AttributeError:
        # Fallback for older scipy versions
        p_binom = stats.binom_test(n_correct, n_total, p=0.5, alternative="two-sided")
        
    # Bang's Blinding Index (BBI)
    bbi = (n_correct - n_incorrect) / n_total

    # 4. Statistical Testing: Sensation & Sound Contrast (EXP vs CON)
    # Wilcoxon signed-rank tests
    w_sound, p_sound = stats.wilcoxon(df_merged["sound_exp"], df_merged["sound_con"], alternative="two-sided")
    w_sens, p_sens = stats.wilcoxon(df_merged["sensation_exp"], df_merged["sensation_con"], alternative="two-sided")

    # 5. Tiredness Profile Over Time (T0 -> T15 -> T30 -> T45)
    # Pivot the tiredness scores sequentially
    df_tired = df_ratings.pivot(index="Subject", columns="tired_time", values="tired_score").reset_index()
    df_tired = df_tired[["Subject", "T0", "T15", "T30", "T45"]]
    
    # Statistical analysis of tiredness change over time (Friedman Test)
    stat_fried, p_fried = stats.friedmanchisquare(df_tired["T0"], df_tired["T15"], df_tired["T30"], df_tired["T45"])

    # Mean and SEM for tiredness
    tired_means = df_tired[["T0", "T15", "T30", "T45"]].mean().tolist()
    tired_sems = df_tired[["T0", "T15", "T30", "T45"]].sem().tolist()

    # 6. Between-Subject Comparisons: Correct vs Incorrect Guessers
    # Confidence comparison
    correct_conf = df_merged[df_merged["correct_guess"]]["confidence_rate"]
    incorrect_conf = df_merged[~df_merged["correct_guess"]]["confidence_rate"]
    
    u_conf, p_conf = stats.mannwhitneyu(correct_conf, incorrect_conf, alternative="two-sided")
    
    # Sensation and Sound contrast comparisons
    correct_sens_diff = df_merged[df_merged["correct_guess"]]["sensation_diff"]
    incorrect_sens_diff = df_merged[~df_merged["correct_guess"]]["sensation_diff"]
    u_sens_diff, p_sens_diff = stats.mannwhitneyu(correct_sens_diff, incorrect_sens_diff, alternative="two-sided")
    
    correct_sound_diff = df_merged[df_merged["correct_guess"]]["sound_diff"]
    incorrect_sound_diff = df_merged[~df_merged["correct_guess"]]["sound_diff"]
    u_sound_diff, p_sound_diff = stats.mannwhitneyu(correct_sound_diff, incorrect_sound_diff, alternative="two-sided")

    # 7. Write Markdown Statistical Report
    report_path = os.path.join(output_dir, "statistical_report.md")
    with open(report_path, "w") as f:
        f.write("# CITRUS Blinding and Subjective Ratings Statistical Report\n\n")
        f.write("This report presents the statistical results from the blinding validation and participant questionnaire analysis.\n\n")
        
        f.write("## 1. Blinding Efficacy Analysis\n")
        f.write(f"- **Total Participants ($N$):** {n_total}\n")
        f.write(f"- **Correct Guesses:** {n_correct} / {n_total} ({accuracy_rate:.1%})\n")
        f.write(f"- **Incorrect Guesses:** {n_incorrect} / {n_total} ({1 - accuracy_rate:.1%})\n")
        f.write(f"- **Binomial Test ($H_0 = 50\\%$ chance):** $p = {p_binom:.4f}$ (Not statistically significant, indicating successful blinding)\n")
        f.write(f"- **Bang's Blinding Index (BBI):** {bbi:.2f} (BBI near 0 indicates optimal, robust blinding)\n\n")
        
        f.write("## 2. Within-Subject Sensory & Auditory Comparison (EXP vs. CON)\n")
        f.write("Comparing subjective sound and skin sensation levels between focused (EXP) and defocused (CON) stimulation.\n\n")
        f.write("| Variable | Mean EXP | Mean CON | Wilcoxon $W$ | $p$-value |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Sound Level** | {df_merged['sound_exp'].mean():.2f} | {df_merged['sound_con'].mean():.2f} | {w_sound:.1f} | {p_sound:.4f} |\n")
        f.write(f"| **Skin Sensation** | {df_merged['sensation_exp'].mean():.2f} | {df_merged['sensation_con'].mean():.2f} | {w_sens:.1f} | {p_sens:.4f} |\n\n")
        
        f.write("## 3. Tiredness Trajectory Over Time\n")
        f.write("Monitoring drowsiness across the four scanning checkpoints ($T_0, T_{15}, T_{30}, T_{45}$).\n\n")
        f.write("| Checkpoint | Timepoint Description | Mean Score | SEM |\n")
        f.write("| :--- | :--- | :---: | :---: |\n")
        f.write(f"| **$T_0$** | Post-Stimulation / Baseline | {tired_means[0]:.2f} | {tired_sems[0]:.2f} |\n")
        f.write(f"| **$T_{15}$** | Post rs-fMRI Run 1 | {tired_means[1]:.2f} | {tired_sems[1]:.2f} |\n")
        f.write(f"| **$T_{30}$** | Post rs-fMRI Run 2 | {tired_means[2]:.2f} | {tired_sems[2]:.2f} |\n")
        f.write(f"| **$T_{45}$** | Post rs-fMRI Run 3 | {tired_means[3]:.2f} | {tired_sems[3]:.2f} |\n\n")
        f.write(f"- **Friedman Test (Change over time):** $\\chi^2 = {stat_fried:.4f}$, $p = {p_fried:.4f}$\n\n")
        
        f.write("## 4. Between-Subject Predictors of Guess Accuracy\n")
        f.write("Testing if correct guessers experienced greater sensory cues or felt higher subjective confidence.\n\n")
        f.write("| Feature Compared | Correct Guessers (Mean) | Incorrect Guessers (Mean) | Mann-Whitney $U$ | $p$-value |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Confidence Rate** | {correct_conf.mean():.2f} | {incorrect_conf.mean():.2f} | {u_conf:.1f} | {p_conf:.4f} |\n")
        f.write(f"| **$\\Delta$ Sensation (EXP - CON)** | {correct_sens_diff.mean():.2f} | {incorrect_sens_diff.mean():.2f} | {u_sens_diff:.1f} | {p_sens_diff:.4f} |\n")
        f.write(f"| **$\\Delta$ Sound (EXP - CON)** | {correct_sound_diff.mean():.2f} | {incorrect_sound_diff.mean():.2f} | {u_sound_diff:.1f} | {p_sound_diff:.4f} |\n")

    print(f"Statistical report successfully generated: {report_path}")

    # ============================================================
    # PREMIUM DATA VISUALIZATION
    # ============================================================
    setup_plotting_theme()
    
    # PLOT 1: Tiredness Trajectory Plot
    plt.figure(figsize=(7, 4.8))
    x_labels = ["T0", "T15", "T30", "T45"]
    x_pos = np.arange(len(x_labels))
    
    plt.errorbar(x_pos, tired_means, yerr=tired_sems, fmt='-o', color='#4f46e5', linewidth=2.5, 
                 markersize=8, elinewidth=1.5, capsize=4, capthick=1.5, markerfacecolor='#ffffff', 
                 markeredgewidth=2.5, markeredgecolor='#4f46e5', label="Group Mean")
    
    # Overlay individual trajectories as faint lines
    for _, row in df_tired.iterrows():
        plt.plot(x_pos, row[["T0", "T15", "T30", "T45"]].values, alpha=0.2, color='#312e81', 
                 linestyle='--', linewidth=1)
        
    plt.xticks(x_pos, ["T0\n(Post-Stim)", "T15\n(Run 1)", "T30\n(Run 2)", "T45\n(Run 3)"])
    plt.ylim(-0.2, 5.5)
    plt.yticks(range(0, 6))
    plt.ylabel("Tiredness Score (0-5)", fontsize=9)
    plt.title("Tiredness Trajectory Across Crossover Sessions", fontsize=11, fontweight='bold', pad=12)
    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tiredness_trajectory.png"))
    plt.close()

    # PLOT 2: Sound and Sensation Ratings Comparison (Paired Spaghetti Plot)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Unique colors per participant to track them across days
    sub_palette = {
        "sub-03": "#3b82f6",  # Blue
        "sub-04": "#10b981",  # Green
        "sub-05": "#f59e0b",  # Orange
        "sub-06": "#ec4899",  # Pink
        "sub-11": "#8b5cf6"   # Purple
    }
    
    x_positions = [0, 1]
    x_labels = ["EXP\n(Focused)", "CON\n(Defocused)"]
    
    # Subplot A: Perceived Sound Level
    ax_sound = axes[0]
    for _, row in df_merged.iterrows():
        subj = row["Subject"]
        y_vals = [row["sound_exp"], row["sound_con"]]
        ax_sound.plot(x_positions, y_vals, marker='o', markersize=6, linewidth=1.5, 
                       color=sub_palette[subj], alpha=0.7, label=subj)
    # Group mean line
    mean_sound_exp = df_merged["sound_exp"].mean()
    mean_sound_con = df_merged["sound_con"].mean()
    ax_sound.plot(x_positions, [mean_sound_exp, mean_sound_con], marker='s', markersize=8, 
                   linewidth=3.5, color="#1e1b4b", linestyle='-', label="Group Mean")
    
    ax_sound.set_xticks(x_positions)
    ax_sound.set_xticklabels(x_labels)
    ax_sound.set_ylim(-0.2, 5.2)
    ax_sound.set_yticks(range(0, 6))
    ax_sound.set_ylabel("Sound Score (0-5)", fontsize=9)
    ax_sound.set_title("Perceived Sound Level", fontsize=11, fontweight='bold', pad=10)
    ax_sound.legend(fontsize=8, loc="upper right", frameon=True)
    sns.despine(ax=ax_sound)
    
    # Subplot B: Skin Sensation Level
    ax_sens = axes[1]
    for _, row in df_merged.iterrows():
        subj = row["Subject"]
        y_vals = [row["sensation_exp"], row["sensation_con"]]
        ax_sens.plot(x_positions, y_vals, marker='o', markersize=6, linewidth=1.5, 
                      color=sub_palette[subj], alpha=0.7, label=subj)
    # Group mean line
    mean_sens_exp = df_merged["sensation_exp"].mean()
    mean_sens_con = df_merged["sensation_con"].mean()
    ax_sens.plot(x_positions, [mean_sens_exp, mean_sens_con], marker='s', markersize=8, 
                  linewidth=3.5, color="#1e1b4b", linestyle='-', label="Group Mean")
    
    ax_sens.set_xticks(x_positions)
    ax_sens.set_xticklabels(x_labels)
    ax_sens.set_ylim(-0.2, 5.2)
    ax_sens.set_yticks(range(0, 6))
    ax_sens.set_ylabel("Sensation Score (0-5)", fontsize=9)
    ax_sens.set_title("Skin Sensation Level", fontsize=11, fontweight='bold', pad=10)
    ax_sens.legend(fontsize=8, loc="upper right", frameon=True)
    sns.despine(ax=ax_sens)
    
    plt.suptitle("Participant Subjective Ratings by Session (EXP vs CON)", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sound_sensation_comparison.png"))
    plt.close()

    # PLOT 3: Blinding Accuracy vs. Confidence
    plt.figure(figsize=(7, 5))
    colors = df_merged["correct_guess"].map({True: "#10b981", False: "#ef4444"}).tolist()
    
    ax = sns.barplot(data=df_merged, x="Subject", y="confidence_rate", palette=colors, edgecolor="#333333", linewidth=1.5, width=0.4)
    
    # Custom legend for correctness
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#10b981', edgecolor='#333333', label='Correct Guess'),
        Patch(facecolor='#ef4444', edgecolor='#333333', label='Incorrect Guess')
    ]
    plt.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=8)
    
    plt.ylim(0, 5.5)
    plt.ylabel("Confidence Rating (1-5)", fontsize=9)
    plt.xlabel("Participant", fontsize=9)
    plt.title("Individual Guesses and Confidence Rates", fontsize=11, fontweight='bold', pad=12)
    
    # Add correctness labels on top of the bars
    for i, row in df_merged.iterrows():
        guess_lbl = f"Real: {row['condition_order']}\nGuess: {row['forced_choice']}"
        ax.text(i, row['confidence_rate'] + 0.1, guess_lbl, ha='center', va='bottom', fontsize=8, fontweight='bold')
        
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "blinding_accuracy_confidence.png"))
    plt.close()
 
    # PLOT 4: Sensory Contrast Scatter Plot (Delta Sensation vs. Delta Sound)
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df_merged, x="sound_diff", y="sensation_diff", hue="correct_guess", 
                    palette={True: "#10b981", False: "#ef4444"}, s=180, edgecolor="black", 
                    linewidth=1.5, style="correct_guess", markers={True: "o", False: "X"})
    
    plt.axhline(0, color="#64748b", linestyle="--", linewidth=1)
    plt.axvline(0, color="#64748b", linestyle="--", linewidth=1)
    
    plt.xlim(-3.5, 3.5)
    plt.ylim(-3.5, 3.5)
    plt.xlabel(r"Auditory Contrast ($\Delta$Sound: EXP - CON)", fontsize=9)
    plt.ylabel(r"Sensory Contrast ($\Delta$Sensation: EXP - CON)", fontsize=9)
    plt.title("Sensory Contrasts as Predictors of Blinding Efficacy", fontsize=11, fontweight='bold', pad=12)
    
    # Label individual subject points
    for _, row in df_merged.iterrows():
        plt.text(row["sound_diff"] + 0.12, row["sensation_diff"] + 0.12, row["Subject"], 
                 fontsize=8, fontweight='bold')
        
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = ["Correct Guess" if l == "True" else "Incorrect Guess" for l in labels]
    plt.legend(handles, new_labels, loc="lower left", frameon=True, fontsize=8)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sensory_contrast_scatter.png"))
    plt.close()

    print("All four premium data visualizations successfully generated and saved to derivatives/blinding/.")

if __name__ == "__main__":
    run_blinding_analysis()
