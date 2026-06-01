# ============================================================
# EXP − CON heatmaps for blinding-risk analysis
# Input: EXP CON data, citrus-online_participant_ratings.xlsx


# Output:
#   1. heatmap
#   2. Bar plot: Number of subjects with EXP > CON
# ============================================================

# ---- Install packages if needed ----
# install.packages(c("readxl", "dplyr", "tidyr", "ggplot2", "stringr"))

library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)

# ---- Load data ----
file_path <- "C:/Local files/JGU Master Doc/Dr. Bergmann-FUS/Judy CITRUS FUS/Blinding-Analysis/citrus-online_participant_ratings.xlsx"

df <- read_excel(file_path, sheet = "Sheet1")

# Expected columns:
# subject_ID, run, protocol, condition, block, sound, sensation, tired

# ---- Prepare data ----
df_clean <- df %>%
  mutate(
    protocol_num = as.numeric(str_extract(protocol, "\\d+")),
    condition = tolower(condition),
    subject_num = as.numeric(str_extract(subject_ID, "\\d+"))
  )

# ---- Convert EXP and CON into paired columns ----
paired_df <- df_clean %>%
  select(subject_ID, subject_num, protocol_num, condition, sound, sensation) %>%
  pivot_wider(
    names_from = condition,
    values_from = c(sound, sensation)
  ) %>%
  mutate(
    sound_diff = sound_exp - sound_con,
    sensation_diff = sensation_exp - sensation_con
  )

# ---- Long-format data for plotting ----
plot_df <- paired_df %>%
  select(subject_ID, subject_num, protocol_num, sound_diff, sensation_diff) %>%
  pivot_longer(
    cols = c(sound_diff, sensation_diff),
    names_to = "measure",
    values_to = "exp_con_diff"
  ) %>%
  mutate(
    measure = recode(
      measure,
      "sound_diff" = "Sound",
      "sensation_diff" = "Sensation"
    ),
    protocol_label = paste0("Protocol ", protocol_num),
    subject_ID = factor(
      subject_ID,
      levels = paired_df %>%
        arrange(subject_num) %>%
        pull(subject_ID) %>%
        unique()
    ),
    protocol_label = factor(
      protocol_label,
      levels = paste0("Protocol ", sort(unique(protocol_num)))
    )
  )

# ============================================================
# Helper function for heatmap
# ============================================================

plot_exp_con_heatmap <- function(data, selected_measure, output_file) {
  
  plot_data <- data %>%
    filter(measure == selected_measure)
  
  # Symmetric colour scale around zero
  # This makes 0 white, negative blue, positive red
  max_abs <- max(abs(plot_data$exp_con_diff), na.rm = TRUE)
  
  p <- ggplot(
    plot_data,
    aes(
      x = protocol_label,
      y = subject_ID,
      fill = exp_con_diff
    )
  ) +
    geom_tile(color = "white", linewidth = 0.7) +
    geom_text(
      aes(label = exp_con_diff),
      size = 4,
      color = "black"
    ) +
    scale_fill_gradient2(
      low = "blue",
      mid = "white",
      high = "red",
      midpoint = 0,
      limits = c(-max_abs, max_abs),
      name = "EXP − CON\nrating difference"
    ) +
    labs(
      title = paste0(selected_measure, " Difference by Subject and Protocol (EXP − CON)"),
      x = "Protocols",
      y = "Subjects"
    ) +
    theme_classic(base_size = 14) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text.x = element_text(angle = 0, hjust = 0.5),
      axis.text.y = element_text(size = 11),
      legend.title = element_text(size = 11),
      legend.text = element_text(size = 10),
      panel.grid = element_blank()
    )
  
  print(p)
  
  ggsave(
    filename = output_file,
    plot = p,
    width = 8.8,
    height = 6.2,
    dpi = 300
  )
}

# ---- Generate plots ----
plot_exp_con_heatmap(
  data = plot_df,
  selected_measure = "Sound",
  output_file = "sound_exp_con_heatmap_blue_white_red.png"
)

plot_exp_con_heatmap(
  data = plot_df,
  selected_measure = "Sensation",
  output_file = "sensation_exp_con_heatmap_blue_white_red.png"
)

# ============================================================
# Optional: print the EXP − CON tables
# ============================================================

sound_table <- paired_df %>%
  select(subject_ID, protocol_num, sound_diff) %>%
  pivot_wider(
    names_from = protocol_num,
    values_from = sound_diff,
    names_prefix = "Protocol_"
  ) %>%
  arrange(as.numeric(str_extract(subject_ID, "\\d+")))

sensation_table <- paired_df %>%
  select(subject_ID, protocol_num, sensation_diff) %>%
  pivot_wider(
    names_from = protocol_num,
    values_from = sensation_diff,
    names_prefix = "Protocol_"
  ) %>%
  arrange(as.numeric(str_extract(subject_ID, "\\d+")))

print("Sound EXP − CON table:")
print(sound_table)

print("Sensation EXP − CON table:")
print(sensation_table)




# ============================================================
# Protocol-level bar plots:
# Number of subjects with EXP > CON, EXP = CON, EXP < CON
# Uses the existing plot_df from your heatmap script
# ============================================================

# ---- Categorize EXP − CON difference ----
# Rule:
#   EXP > CON  if EXP − CON > 0.5
#   EXP = CON  if -0.5 <= EXP − CON <= 0.5
#   EXP < CON  if EXP − CON < -0.5
#

bar_df <- plot_df %>%
  mutate(
    category = case_when(
      exp_con_diff > 0.5  ~ "EXP > CON",
      exp_con_diff < -0.5 ~ "EXP < CON",
      TRUE                ~ "EXP = CON"
    ),
    category = factor(
      category,
      levels = c("EXP > CON", "EXP = CON", "EXP < CON")
    )
  ) %>%
  group_by(measure, protocol_label, category) %>%
  summarise(
    n = n(),
    .groups = "drop"
  ) %>%
  # make sure missing categories appear as 0
  complete(
    measure,
    protocol_label,
    category,
    fill = list(n = 0)
  )

# ============================================================
# Plotting function
# ============================================================

plot_protocol_bar <- function(data, selected_measure, output_file) {
  
  plot_data <- data %>%
    filter(measure == selected_measure)
  
  p <- ggplot(
    plot_data,
    aes(
      x = protocol_label,
      y = n,
      fill = category
    )
  ) +
    geom_col(
      position = position_dodge(width = 0.8),
      width = 0.7,
      color = "black",
      linewidth = 0.25
    ) +
    geom_text(
      aes(label = n),
      position = position_dodge(width = 0.8),
      vjust = -0.3,
      size = 4
    ) +
    scale_fill_manual(
      values = c(
        "EXP > CON" = "red",
        "EXP = CON" = "white",
        "EXP < CON" = "blue"
      )
    ) +
    scale_y_continuous(
      breaks = 0:11,
      limits = c(0, 11.8),
      expand = expansion(mult = c(0, 0.03))
    ) +
    labs(
      title = paste0(
        selected_measure,
        ": Number of Subjects Rating EXP > CON, EXP = CON, or EXP < CON"
      ),
      x = "Protocol",
      y = "Number of subjects",
      fill = NULL
    ) +
    theme_classic(base_size = 14) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      legend.position = "top",
      legend.text = element_text(size = 11),
      axis.text = element_text(size = 11),
      axis.title = element_text(size = 13)
    )
  
  print(p)
  
  ggsave(
    filename = output_file,
    plot = p,
    width = 8.5,
    height = 5.8,
    dpi = 300
  )
}

# ============================================================
# Generate bar plots
# ============================================================

plot_protocol_bar(
  data = bar_df,
  selected_measure = "Sound",
  output_file = "sound_protocol_bar_EXP_CON_categories.png"
)

plot_protocol_bar(
  data = bar_df,
  selected_measure = "Sensation",
  output_file = "sensation_protocol_bar_EXP_CON_categories.png"
)

# ============================================================
# Optional: print count tables
# ============================================================

sound_counts <- bar_df %>%
  filter(measure == "Sound") %>%
  pivot_wider(
    names_from = category,
    values_from = n
  )

sensation_counts <- bar_df %>%
  filter(measure == "Sensation") %>%
  pivot_wider(
    names_from = category,
    values_from = n
  )

cat("\nSound category counts by protocol:\n")
print(sound_counts)

cat("\nSensation category counts by protocol:\n")
print(sensation_counts)



# ============================================================
# Grouped bar plot:
# For each protocol, show 2 bars:
#   1. Sound: EXP > CON
#   2. Sensation: EXP > CON
# y-axis = number of subjects
# Uses the existing plot_df from your script
# ============================================================

# ---- Count number of subjects with EXP > CON ----
# Because ratings are integers, exp_con_diff > 0.5 means EXP is at least 1 point higher than CON.

exp_gt_con_df <- plot_df %>%
  mutate(
    exp_gt_con = exp_con_diff > 0.5
  ) %>%
  group_by(protocol_label, measure) %>%
  summarise(
    n_subjects = sum(exp_gt_con, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    measure_label = paste0(measure, ": EXP > CON"),
    measure_label = factor(
      measure_label,
      levels = c("Sound: EXP > CON", "Sensation: EXP > CON")
    )
  )

# ---- Plot grouped bar chart ----
p_exp_gt_con <- ggplot(
  exp_gt_con_df,
  aes(
    x = protocol_label,
    y = n_subjects,
    fill = measure_label
  )
) +
  geom_col(
    position = position_dodge(width = 0.8),
    width = 0.7,
    color = "black",
    linewidth = 0.25
  ) +
  geom_text(
    aes(label = n_subjects),
    position = position_dodge(width = 0.8),
    vjust = -0.3,
    size = 4
  ) +
  scale_fill_manual(
    values = c(
      "Sound: EXP > CON" = "blue",
      "Sensation: EXP > CON" = "red"
    )
  ) +
  scale_y_continuous(
    breaks = 0:11,
    limits = c(0, 11.8),
    expand = expansion(mult = c(0, 0.03))
  ) +
  labs(
    title = "Number of Subjects with EXP > CON by Protocol",
    x = "Protocol",
    y = "Number of subjects",
    fill = NULL
  ) +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "top",
    legend.text = element_text(size = 11),
    axis.text = element_text(size = 11),
    axis.title = element_text(size = 13)
  )

print(p_exp_gt_con)

# ---- Save plot ----
ggsave(
  filename = "protocol_barplot_sound_vs_sensation_EXP_gt_CON.png",
  plot = p_exp_gt_con,
  width = 8.5,
  height = 5.8,
  dpi = 300
)

# ============================================================
# Optional: print count table
# ============================================================

exp_gt_con_table <- exp_gt_con_df %>%
  select(protocol_label, measure_label, n_subjects) %>%
  pivot_wider(
    names_from = measure_label,
    values_from = n_subjects
  )

cat("\nNumber of subjects with EXP > CON by protocol:\n")
print(exp_gt_con_table)












