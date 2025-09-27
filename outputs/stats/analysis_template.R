
# GDPR DPA Decisions Analysis Template
# Legal data analysis with proper handling of typed missingness

library(dplyr)
library(ggplot2)
library(forcats)
library(lubridate)

# Load the data (run load_gdpr_data.R first)
source("load_gdpr_data.R")

# ============================================================================
# 1. DESCRIPTIVE ANALYSIS
# ============================================================================

# Fine distribution by country group
gdpr_data %>%
  filter(fine_eur > 0) %>%
  ggplot(aes(x = country_group, y = fine_eur)) +
  geom_boxplot() +
  scale_y_log10() +
  labs(title = "Fine Distribution by Geographic Region",
       y = "Fine Amount (EUR, log scale)",
       x = "Geographic Group")

# Temporal trends in enforcement
temporal_trends <- gdpr_data %>%
  filter(!is.na(decision_year)) %>%
  group_by(decision_year, country_group) %>%
  summarise(
    n_decisions = n(),
    n_fines = sum(fine_positive_flag, na.rm = TRUE),
    median_fine = median(fine_eur[fine_eur > 0], na.rm = TRUE),
    .groups = "drop"
  )

# ============================================================================
# 2. VIOLATION ANALYSIS (Respecting Legal Semantics)
# ============================================================================

# Article 5 violations (only where discussed)
article5_analysis <- gdpr_data %>%
  filter(q31_violated_status == "DISCUSSED") %>%  # Only analyze where violations were assessed
  select(decision_id, starts_with("q31_violated_")) %>%
  select(-ends_with("_status"), -ends_with("_coverage_status")) %>%
  pivot_longer(cols = -c(decision_id), names_to = "principle", values_to = "violated") %>%
  filter(!is.na(violated) & violated == 1) %>%
  count(principle, sort = TRUE)

# ============================================================================
# 3. ENFORCEMENT PATTERNS
# ============================================================================

# Fine determinants (basic model - extend as needed)
fine_model <- gdpr_data %>%
  filter(fine_status == "DISCUSSED") %>%  # Only where fines were assessed
  lm(fine_log ~ country_group + breach_case + n_principles_violated +
     enforcement_severity_index, data = .)

summary(fine_model)

# ============================================================================
# 4. LEGAL INTERPRETATION NOTES
# ============================================================================

# IMPORTANT: This dataset preserves legal nuance through typed missingness
# - NOT_DISCUSSED != NOT_VIOLATED
# - Always filter by status variables before analysis
# - Use appropriate statistical methods for legal data

cat("Analysis template completed. Modify as needed for specific research questions.\n")
