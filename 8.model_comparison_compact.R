# ============================================================================
# COMPACT MODEL COMPARISON: BINOMIAL GLMM vs GAUSSIAN LMM (LOG2 RATIO)
# ============================================================================
# Compares two models with combined visualization:
#   1. Binomial GLMM: cbind(IntronCount, ExonCount) ~ Condition + (1|sample)
#   2. Gaussian LMM:  Log2IntronExonRatio ~ Condition + (1|sample)
# ============================================================================

library(lme4)
library(lmerTest)
library(DHARMa)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(grid)

# === CONFIGURATION ===
gene_of_interest <- "STMN2"
base_output_dir <- "C:/Users/catta/Desktop/ALS/THESIS/Transcriptome_integrity_general"
output_dir <- file.path(base_output_dir, paste0("model_comparison_compact_", gene_of_interest))
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Theme
theme_pres <- theme_bw(base_size = 14) +
  theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
        axis.title = element_text(face = "bold", size = 11),
        legend.position = "bottom")

cat("\n=== COMPACT MODEL COMPARISON ===\n")
cat(sprintf("Gene: %s\n\n", gene_of_interest))

# === LOAD DATA ===
df <- read.csv(file.path(base_output_dir, "valid_genes_df.csv"))
dfg <- df %>% filter(geneName == gene_of_interest)
dfg$sample <- factor(dfg$sample)
dfg$Condition <- factor(dfg$Condition, levels = c("CTR", "ALS"))

cat(sprintf("Cells: %d | Samples: CTR=%d, ALS=%d\n",
            nrow(dfg),
            length(unique(dfg$sample[dfg$Condition == "CTR"])),
            length(unique(dfg$sample[dfg$Condition == "ALS"]))))

# === FIT MODELS ===
cat("\nFitting models...\n")

# Model 1: Binomial GLMM
glmer_mod <- glmer(cbind(IntronCount, ExonCount) ~ Condition + (1 | sample),
                   data = dfg, family = binomial(link = "logit"),
                   control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))

# Model 2: Gaussian LMM on Log2 Ratio
lmm_log2 <- lmer(Log2IntronExonRatio ~ Condition + (1 | sample),
                 data = dfg, REML = FALSE,
                 control = lmerControl(optimizer = "bobyqa"))

# === EXTRACT RESULTS ===
# Binomial GLMM
glmer_coef <- fixef(glmer_mod)["ConditionALS"]
glmer_or <- exp(glmer_coef)
glmer_ci_log <- confint(glmer_mod, parm = "ConditionALS", method = "Wald", quiet = TRUE)
glmer_ci <- exp(glmer_ci_log)
glmer_pval <- coef(summary(glmer_mod))["ConditionALS", "Pr(>|z|)"]

# Log2 LMM
log2_coef <- fixef(lmm_log2)["ConditionALS"]
log2_ci <- confint(lmm_log2, parm = "ConditionALS", method = "Wald", quiet = TRUE)
log2_pval <- coef(summary(lmm_log2))["ConditionALS", "Pr(>|t|)"]
fold_change <- 2^log2_coef
fold_change_ci <- 2^log2_ci

# AIC/BIC
aic_glmer <- AIC(glmer_mod); bic_glmer <- BIC(glmer_mod)
aic_log2 <- AIC(lmm_log2); bic_log2 <- BIC(lmm_log2)

# DHARMa
sim_glmer <- simulateResiduals(glmer_mod, n = 1000)
sim_log2 <- simulateResiduals(lmm_log2, n = 1000)

# Predictions
dfg$pred_glmer <- predict(glmer_mod, type = "response")
dfg$pred_log2 <- predict(lmm_log2)

# === ADDITIONAL DIAGNOSTICS ===
cat("\n--- Model Diagnostics ---\n")

# 1. Random effects variance components
cat("\n[Random Effects]\n")
vc_glmer <- as.data.frame(VarCorr(glmer_mod))
vc_log2 <- as.data.frame(VarCorr(lmm_log2))
cat(sprintf("  Binomial GLMM - Sample variance: %.4f (SD = %.4f)\n",
            vc_glmer$vcov[1], vc_glmer$sdcor[1]))
cat(sprintf("  Log2 LMM      - Sample variance: %.4f (SD = %.4f)\n",
            vc_log2$vcov[1], vc_log2$sdcor[1]))

# ICC for Log2 LMM (proportion of variance explained by sample)
residual_var_log2 <- sigma(lmm_log2)^2
sample_var_log2 <- vc_log2$vcov[1]
icc_log2 <- sample_var_log2 / (sample_var_log2 + residual_var_log2)
cat(sprintf("  Log2 LMM ICC (sample): %.3f (%.1f%% of variance)\n", icc_log2, icc_log2 * 100))

# 2. DHARMa formal tests
cat("\n[DHARMa Tests]\n")
dharma_tests_glmer <- testResiduals(sim_glmer, plot = FALSE)
dharma_tests_log2 <- testResiduals(sim_log2, plot = FALSE)

cat("  Binomial GLMM:\n")
cat(sprintf("    - Uniformity (KS test): p = %.4f %s\n",
            dharma_tests_glmer$uniformity$p.value,
            ifelse(dharma_tests_glmer$uniformity$p.value < 0.05, "[!]", "[OK]")))
cat(sprintf("    - Dispersion test: p = %.4f %s\n",
            dharma_tests_glmer$dispersion$p.value,
            ifelse(dharma_tests_glmer$dispersion$p.value < 0.05, "[!]", "[OK]")))
cat(sprintf("    - Outlier test: p = %.4f %s\n",
            dharma_tests_glmer$outliers$p.value,
            ifelse(dharma_tests_glmer$outliers$p.value < 0.05, "[!]", "[OK]")))

cat("  Log2 LMM:\n")
cat(sprintf("    - Uniformity (KS test): p = %.4f %s\n",
            dharma_tests_log2$uniformity$p.value,
            ifelse(dharma_tests_log2$uniformity$p.value < 0.05, "[!]", "[OK]")))
cat(sprintf("    - Dispersion test: p = %.4f %s\n",
            dharma_tests_log2$dispersion$p.value,
            ifelse(dharma_tests_log2$dispersion$p.value < 0.05, "[!]", "[OK]")))
cat(sprintf("    - Outlier test: p = %.4f %s\n",
            dharma_tests_log2$outliers$p.value,
            ifelse(dharma_tests_log2$outliers$p.value < 0.05, "[!]", "[OK]")))

# 3. Overdispersion check for binomial GLMM
overdisp_glmer <- sum(residuals(glmer_mod, type = "pearson")^2) / df.residual(glmer_mod)
cat(sprintf("\n[Overdispersion]\n  Binomial GLMM Pearson dispersion: %.3f %s\n",
            overdisp_glmer,
            ifelse(overdisp_glmer > 1.5, "[overdispersed!]",
                   ifelse(overdisp_glmer < 0.5, "[underdispersed!]", "[OK]"))))

# 4. Model convergence
cat("\n[Convergence]\n")
cat(sprintf("  Binomial GLMM converged: %s\n",
            ifelse(length(glmer_mod@optinfo$conv$lme4) == 0, "YES", "NO - check warnings")))
cat(sprintf("  Log2 LMM converged: %s\n",
            ifelse(length(lmm_log2@optinfo$conv$lme4) == 0, "YES", "NO - check warnings")))

# 5. Sample-level residuals
dfg$resid_glmer <- residuals(sim_glmer)
dfg$resid_log2 <- residuals(sim_log2)

sample_resid <- dfg %>%
  group_by(sample, Condition) %>%
  summarize(
    mean_resid_glmer = mean(resid_glmer),
    mean_resid_log2 = mean(resid_log2),
    n_cells = n(),
    .groups = "drop"
  )

cat("\n[Sample-level Residual Summary]\n")
cat("  Binomial GLMM residuals by condition:\n")
cat(sprintf("    CTR: mean = %.4f, SD = %.4f\n",
            mean(sample_resid$mean_resid_glmer[sample_resid$Condition == "CTR"]),
            sd(sample_resid$mean_resid_glmer[sample_resid$Condition == "CTR"])))
cat(sprintf("    ALS: mean = %.4f, SD = %.4f\n",
            mean(sample_resid$mean_resid_glmer[sample_resid$Condition == "ALS"]),
            sd(sample_resid$mean_resid_glmer[sample_resid$Condition == "ALS"])))

cat("  Log2 LMM residuals by condition:\n")
cat(sprintf("    CTR: mean = %.4f, SD = %.4f\n",
            mean(sample_resid$mean_resid_log2[sample_resid$Condition == "CTR"]),
            sd(sample_resid$mean_resid_log2[sample_resid$Condition == "CTR"])))
cat(sprintf("    ALS: mean = %.4f, SD = %.4f\n",
            mean(sample_resid$mean_resid_log2[sample_resid$Condition == "ALS"]),
            sd(sample_resid$mean_resid_log2[sample_resid$Condition == "ALS"])))

# 6. R-squared (marginal and conditional) for Log2 LMM
r2_marginal <- var(predict(lmm_log2, re.form = NA)) / var(dfg$Log2IntronExonRatio)
r2_conditional <- var(predict(lmm_log2)) / var(dfg$Log2IntronExonRatio)
cat(sprintf("\n[Model Fit - Log2 LMM]\n  R² marginal (fixed only): %.3f\n  R² conditional (fixed + random): %.3f\n",
            r2_marginal, r2_conditional))

# 7. Pseudo R² for Binomial GLMM (McFadden's R²)
null_glmer <- glmer(cbind(IntronCount, ExonCount) ~ 1 + (1 | sample),
                    data = dfg, family = binomial(link = "logit"),
                    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
mcfadden_r2 <- 1 - (logLik(glmer_mod) / logLik(null_glmer))
cat(sprintf("\n[Model Fit - Binomial GLMM]\n  McFadden's pseudo R²: %.4f\n", as.numeric(mcfadden_r2)))

# 8. Likelihood Ratio Tests vs Null Models
cat("\n[Likelihood Ratio Tests]\n")
lrt_glmer <- anova(null_glmer, glmer_mod)
null_lmm <- lmer(Log2IntronExonRatio ~ 1 + (1 | sample), data = dfg, REML = FALSE,
                 control = lmerControl(optimizer = "bobyqa"))
lrt_lmm <- anova(null_lmm, lmm_log2)
cat(sprintf("  Binomial GLMM vs null: χ² = %.2f, df = %d, p = %.2e\n",
            lrt_glmer$Chisq[2], lrt_glmer$Df[2], lrt_glmer$`Pr(>Chisq)`[2]))
cat(sprintf("  Log2 LMM vs null:      χ² = %.2f, df = %d, p = %.2e\n",
            lrt_lmm$Chisq[2], lrt_lmm$Df[2], lrt_lmm$`Pr(>Chisq)`[2]))

# 9. Zero-inflation check
n_zero_intron <- sum(dfg$IntronCount == 0)
n_zero_exon <- sum(dfg$ExonCount == 0)
n_zero_ratio <- sum(is.infinite(dfg$Log2IntronExonRatio) | is.na(dfg$Log2IntronExonRatio))
cat(sprintf("\n[Zero/Extreme Values Check]\n  Cells with IntronCount = 0: %d (%.1f%%)\n",
            n_zero_intron, 100 * n_zero_intron / nrow(dfg)))
cat(sprintf("  Cells with ExonCount = 0: %d (%.1f%%)\n",
            n_zero_exon, 100 * n_zero_exon / nrow(dfg)))
cat(sprintf("  Cells with undefined Log2 ratio: %d (%.1f%%)\n",
            n_zero_ratio, 100 * n_zero_ratio / nrow(dfg)))

# 10. Random Effects Normality (Shapiro-Wilk test on BLUPs)
re_glmer <- ranef(glmer_mod)$sample[, 1]
re_lmm <- ranef(lmm_log2)$sample[, 1]
cat("\n[Random Effects Normality - Shapiro-Wilk]\n")
if (length(re_glmer) >= 3 & length(re_glmer) <= 5000) {
  sw_glmer <- shapiro.test(re_glmer)
  cat(sprintf("  Binomial GLMM: W = %.4f, p = %.4f %s\n",
              sw_glmer$statistic, sw_glmer$p.value,
              ifelse(sw_glmer$p.value < 0.05, "[non-normal!]", "[OK]")))
} else {
  cat("  Binomial GLMM: insufficient samples for Shapiro-Wilk\n")
}
if (length(re_lmm) >= 3 & length(re_lmm) <= 5000) {
  sw_lmm <- shapiro.test(re_lmm)
  cat(sprintf("  Log2 LMM:      W = %.4f, p = %.4f %s\n",
              sw_lmm$statistic, sw_lmm$p.value,
              ifelse(sw_lmm$p.value < 0.05, "[non-normal!]", "[OK]")))
} else {
  cat("  Log2 LMM: insufficient samples for Shapiro-Wilk\n")
}

# 11. Influential Observations (Cook's Distance for LMM)
cooks_lmm <- cooks.distance(lmm_log2)
n_influential <- sum(cooks_lmm > 4 / nrow(dfg), na.rm = TRUE)
cat(sprintf("\n[Influential Observations - Log2 LMM]\n  Cook's D threshold: %.4f (4/n)\n",
            4 / nrow(dfg)))
cat(sprintf("  Observations exceeding threshold: %d (%.2f%%)\n",
            n_influential, 100 * n_influential / nrow(dfg)))
if (n_influential > 0) {
  top_influential <- head(order(cooks_lmm, decreasing = TRUE), 5)
  cat("  Top 5 most influential observations:\n")
  for (i in top_influential) {
    cat(sprintf("    Row %d: Cook's D = %.4f, Sample = %s, Condition = %s\n",
                i, cooks_lmm[i], dfg$sample[i], dfg$Condition[i]))
  }
}

# 12. Leave-One-Sample-Out Sensitivity Analysis
cat("\n[Leave-One-Sample-Out Sensitivity]\n")
samples <- unique(dfg$sample)
loo_results <- data.frame(
  sample = samples,
  condition = sapply(samples, function(s) as.character(dfg$Condition[dfg$sample == s][1])),
  glmer_or = NA,
  glmer_p = NA,
  lmm_fc = NA,
  lmm_p = NA
)

for (i in seq_along(samples)) {
  s <- samples[i]
  dfg_loo <- dfg[dfg$sample != s, ]

  tryCatch({
    mod_loo <- glmer(cbind(IntronCount, ExonCount) ~ Condition + (1 | sample),
                     data = dfg_loo, family = binomial(link = "logit"),
                     control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
    loo_results$glmer_or[i] <- exp(fixef(mod_loo)["ConditionALS"])
    loo_results$glmer_p[i] <- coef(summary(mod_loo))["ConditionALS", "Pr(>|z|)"]
  }, error = function(e) {})

  tryCatch({
    mod_loo <- lmer(Log2IntronExonRatio ~ Condition + (1 | sample),
                    data = dfg_loo, REML = FALSE,
                    control = lmerControl(optimizer = "bobyqa"))
    loo_results$lmm_fc[i] <- 2^fixef(mod_loo)["ConditionALS"]
    loo_results$lmm_p[i] <- coef(summary(mod_loo))["ConditionALS", "Pr(>|t|)"]
  }, error = function(e) {})
}

cat("  Effect size range when each sample is removed:\n")
cat(sprintf("    Binomial GLMM OR: [%.2f, %.2f] (full model: %.2f)\n",
            min(loo_results$glmer_or, na.rm = TRUE),
            max(loo_results$glmer_or, na.rm = TRUE), glmer_or))
cat(sprintf("    Log2 LMM FC:      [%.2f, %.2f] (full model: %.2f)\n",
            min(loo_results$lmm_fc, na.rm = TRUE),
            max(loo_results$lmm_fc, na.rm = TRUE), fold_change))

# Find most influential samples
glmer_or_diff <- abs(loo_results$glmer_or - glmer_or)
lmm_fc_diff <- abs(loo_results$lmm_fc - fold_change)
most_infl_glmer <- which.max(glmer_or_diff)
most_infl_lmm <- which.max(lmm_fc_diff)
cat(sprintf("  Most influential sample (Binomial): %s (%s) - OR changes to %.2f\n",
            loo_results$sample[most_infl_glmer], loo_results$condition[most_infl_glmer],
            loo_results$glmer_or[most_infl_glmer]))
cat(sprintf("  Most influential sample (Log2 LMM): %s (%s) - FC changes to %.2f\n",
            loo_results$sample[most_infl_lmm], loo_results$condition[most_infl_lmm],
            loo_results$lmm_fc[most_infl_lmm]))

# Save LOO results
write.csv(loo_results, file.path(output_dir, "loo_sensitivity_results.csv"), row.names = FALSE)

# 13. Effective sample size / Design effect
n_cells <- nrow(dfg)
n_samples <- length(unique(dfg$sample))
avg_cells_per_sample <- n_cells / n_samples
design_effect <- 1 + (avg_cells_per_sample - 1) * icc_log2
eff_sample_size <- n_cells / design_effect
cat(sprintf("\n[Effective Sample Size]\n  Total cells: %d\n  Number of samples: %d\n",
            n_cells, n_samples))
cat(sprintf("  Avg cells per sample: %.1f\n", avg_cells_per_sample))
cat(sprintf("  Design effect (DEFF): %.2f\n", design_effect))
cat(sprintf("  Effective sample size: %.0f\n", eff_sample_size))

# Print results
cat("\n--- Results ---\n")
cat(sprintf("Binomial GLMM:  OR = %.2f [%.2f, %.2f], p = %.2e, AIC = %.0f\n",
            glmer_or, glmer_ci[1], glmer_ci[2], glmer_pval, aic_glmer))
cat(sprintf("Log2 LMM:       FC = %.2f [%.2f, %.2f], p = %.2e, AIC = %.0f\n",
            fold_change, fold_change_ci[1], fold_change_ci[2], log2_pval, aic_log2))

# === PLOT 1: DATA OVERVIEW (side-by-side) ===
cat("\nCreating plots...\n")

pb <- dfg %>% group_by(sample, Condition) %>%
  summarize(IntronFraction = median(IntronFraction),
            Log2Ratio = median(Log2IntronExonRatio), .groups = "drop")

p1a <- ggplot(pb, aes(x = Condition, y = IntronFraction, fill = Condition)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.15, size = 3, alpha = 0.8) +
  scale_fill_manual(values = c("CTR" = "#4DAF4A", "ALS" = "#E41A1C")) +
  labs(title = "Intron Fraction", y = "Median per Sample") +
  theme_pres + theme(legend.position = "none")

p1b <- ggplot(pb, aes(x = Condition, y = Log2Ratio, fill = Condition)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.15, size = 3, alpha = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  scale_fill_manual(values = c("CTR" = "#4DAF4A", "ALS" = "#E41A1C")) +
  labs(title = "Log2(Intron/Exon)", y = "Median per Sample") +
  theme_pres + theme(legend.position = "none")

png(file.path(output_dir, "01_data_overview.png"), width = 10, height = 5, units = "in", res = 300)
grid.arrange(p1a, p1b, ncol = 2,
             top = textGrob(paste0(gene_of_interest, ": Sample-Level Data"),
                           gp = gpar(fontsize = 16, fontface = "bold")))
dev.off()

# === PLOT 2: DHARMA COMPARISON (2x2) ===
png(file.path(output_dir, "02_dharma_comparison.png"), width = 12, height = 8, units = "in", res = 300)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))

plotQQunif(sim_glmer, main = "Binomial GLMM: Q-Q")
plotResiduals(sim_glmer, main = "Binomial GLMM: Residuals")
plotQQunif(sim_log2, main = "Log2 LMM: Q-Q")
plotResiduals(sim_log2, main = "Log2 LMM: Residuals")

mtext(paste0(gene_of_interest, ": DHARMa Diagnostics"), outer = TRUE, cex = 1.5, font = 2)
dev.off()

# === PLOT 3: PREDICTIONS COMPARISON ===
p3a <- ggplot(dfg, aes(x = pred_glmer, y = IntronFraction)) +
  geom_point(alpha = 0.3, size = 2, color = "darkgreen") +
  geom_abline(slope = 1, intercept = 0, color = "red", lwd = 1.2, linetype = "dashed") +
  labs(title = "Binomial GLMM", subtitle = sprintf("AIC = %.0f", aic_glmer),
       x = "Predicted", y = "Observed IntronFraction") +
  theme_pres + coord_fixed(xlim = c(0, 1), ylim = c(0, 1))

p3b <- ggplot(dfg, aes(x = pred_log2, y = Log2IntronExonRatio)) +
  geom_point(alpha = 0.3, size = 2, color = "darkorange") +
  geom_abline(slope = 1, intercept = 0, color = "red", lwd = 1.2, linetype = "dashed") +
  geom_hline(yintercept = 0, linetype = "dotted", color = "grey50") +
  labs(title = "Log2 LMM", subtitle = sprintf("AIC = %.0f | FC = %.2f", aic_log2, fold_change),
       x = "Predicted", y = "Observed Log2(Intron/Exon)") +
  theme_pres

png(file.path(output_dir, "03_predictions_comparison.png"), width = 12, height = 5, units = "in", res = 300)
grid.arrange(p3a, p3b, ncol = 2,
             top = textGrob(paste0(gene_of_interest, ": Model Predictions"),
                           gp = gpar(fontsize = 16, fontface = "bold")))
dev.off()

# === PLOT 4: COMBINED EFFECT SIZE FOREST PLOT ===
effect_df <- data.frame(
  Model = c("Binomial GLMM\n(Odds Ratio)", "Log2 LMM\n(Fold Change)"),
  Effect = c(glmer_or, fold_change),
  Lower = c(glmer_ci[1], fold_change_ci[1]),
  Upper = c(glmer_ci[2], fold_change_ci[2]),
  P_value = c(glmer_pval, log2_pval),
  Color = c("darkgreen", "darkorange")
)
effect_df$Model <- factor(effect_df$Model, levels = rev(effect_df$Model))

p4 <- ggplot(effect_df, aes(y = Model, x = Effect, color = Model)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey50", lwd = 1) +
  geom_errorbarh(aes(xmin = Lower, xmax = Upper), height = 0.2, lwd = 1.5) +
  geom_point(size = 6) +
  scale_color_manual(values = c("Binomial GLMM\n(Odds Ratio)" = "darkgreen",
                                 "Log2 LMM\n(Fold Change)" = "darkorange")) +
  scale_x_log10(breaks = c(0.5, 1, 2, 4, 8, 16, 32)) +
  geom_text(aes(label = sprintf("%.2f [%.2f, %.2f]\np = %.2e", Effect, Lower, Upper, P_value)),
            hjust = -0.15, size = 4, fontface = "bold", color = "black") +
  labs(title = paste0(gene_of_interest, ": Effect Size Comparison (ALS vs CTR)"),
       subtitle = "Both metrics show ratio scale: OR for counts, FC for log2 ratio | Reference = 1",
       x = "Effect Size (log scale)", y = "") +
  theme_pres +
  theme(legend.position = "none", axis.text.y = element_text(size = 12, face = "bold")) +
  coord_cartesian(xlim = c(min(effect_df$Lower) * 0.5, max(effect_df$Upper) * 3))

ggsave(file.path(output_dir, "04_effect_comparison.png"), p4, width = 12, height = 4, dpi = 300)

# === PLOT 5: RANDOM EFFECTS DIAGNOSTICS ===
re_df <- data.frame(
  sample = rownames(ranef(glmer_mod)$sample),
  re_glmer = ranef(glmer_mod)$sample[, 1],
  re_lmm = ranef(lmm_log2)$sample[, 1]
)
re_df$Condition <- sapply(re_df$sample, function(s) as.character(dfg$Condition[dfg$sample == s][1]))

p5a <- ggplot(re_df, aes(sample = re_glmer)) +
  stat_qq() + stat_qq_line(color = "red") +
  labs(title = "Binomial GLMM", x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_pres

p5b <- ggplot(re_df, aes(sample = re_lmm)) +
  stat_qq() + stat_qq_line(color = "red") +
  labs(title = "Log2 LMM", x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_pres

p5c <- ggplot(re_df, aes(x = reorder(sample, re_glmer), y = re_glmer, fill = Condition)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("CTR" = "#4DAF4A", "ALS" = "#E41A1C")) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Binomial GLMM BLUPs", x = "Sample", y = "Random Effect") +
  theme_pres + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8))

p5d <- ggplot(re_df, aes(x = reorder(sample, re_lmm), y = re_lmm, fill = Condition)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("CTR" = "#4DAF4A", "ALS" = "#E41A1C")) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Log2 LMM BLUPs", x = "Sample", y = "Random Effect") +
  theme_pres + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8))

png(file.path(output_dir, "05_random_effects_diagnostics.png"), width = 14, height = 10, units = "in", res = 300)
grid.arrange(p5a, p5b, p5c, p5d, ncol = 2,
             top = textGrob(paste0(gene_of_interest, ": Random Effects Diagnostics"),
                           gp = gpar(fontsize = 16, fontface = "bold")))
dev.off()

# === PLOT 6: COOK'S DISTANCE ===
dfg$cooks_d <- cooks_lmm
cooks_threshold <- 4 / nrow(dfg)

p6 <- ggplot(dfg, aes(x = seq_along(cooks_d), y = cooks_d, color = Condition)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = cooks_threshold, linetype = "dashed", color = "red") +
  scale_color_manual(values = c("CTR" = "#4DAF4A", "ALS" = "#E41A1C")) +
  labs(title = paste0(gene_of_interest, ": Cook's Distance (Log2 LMM)"),
       subtitle = sprintf("Threshold = 4/n = %.4f | %d observations above threshold",
                         cooks_threshold, n_influential),
       x = "Observation Index", y = "Cook's Distance") +
  theme_pres

ggsave(file.path(output_dir, "06_cooks_distance.png"), p6, width = 10, height = 5, dpi = 300)

# === PLOT 7: LEAVE-ONE-OUT SENSITIVITY ===
loo_long <- rbind(
  data.frame(sample = loo_results$sample, condition = loo_results$condition,
             Model = "Binomial GLMM", Effect = loo_results$glmer_or),
  data.frame(sample = loo_results$sample, condition = loo_results$condition,
             Model = "Log2 LMM", Effect = loo_results$lmm_fc)
)
full_effects <- data.frame(Model = c("Binomial GLMM", "Log2 LMM"),
                           FullEffect = c(glmer_or, fold_change))

p7 <- ggplot(loo_long, aes(x = reorder(sample, Effect), y = Effect, fill = condition)) +
  geom_bar(stat = "identity") +
  geom_hline(data = full_effects, aes(yintercept = FullEffect), linetype = "dashed", color = "blue", lwd = 1) +
  geom_hline(yintercept = 1, linetype = "dotted", color = "grey50") +
  scale_fill_manual(values = c("CTR" = "#4DAF4A", "ALS" = "#E41A1C")) +
  facet_wrap(~ Model, scales = "free_y", ncol = 1) +
  labs(title = paste0(gene_of_interest, ": Leave-One-Sample-Out Sensitivity"),
       subtitle = "Blue dashed line = full model estimate | Each bar = effect when that sample removed",
       x = "Sample Removed", y = "Effect Size (OR or FC)") +
  theme_pres +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8))

ggsave(file.path(output_dir, "07_loo_sensitivity.png"), p7, width = 12, height = 8, dpi = 300)

# === PLOT 8: RESIDUALS BY SAMPLE ===
p8a <- ggplot(dfg, aes(x = sample, y = resid_glmer, fill = Condition)) +
  geom_boxplot(outlier.size = 0.5) +
  scale_fill_manual(values = c("CTR" = "#4DAF4A", "ALS" = "#E41A1C")) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey50") +
  labs(title = "Binomial GLMM", x = "Sample", y = "DHARMa Residual") +
  theme_pres + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8))

p8b <- ggplot(dfg, aes(x = sample, y = resid_log2, fill = Condition)) +
  geom_boxplot(outlier.size = 0.5) +
  scale_fill_manual(values = c("CTR" = "#4DAF4A", "ALS" = "#E41A1C")) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey50") +
  labs(title = "Log2 LMM", x = "Sample", y = "DHARMa Residual") +
  theme_pres + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8))

png(file.path(output_dir, "08_residuals_by_sample.png"), width = 14, height = 8, units = "in", res = 300)
grid.arrange(p8a, p8b, ncol = 1,
             top = textGrob(paste0(gene_of_interest, ": Residuals by Sample"),
                           gp = gpar(fontsize = 16, fontface = "bold")))
dev.off()

# === SUMMARY TABLE ===
summary_df <- data.frame(
  Metric = c("Effect (ALS vs CTR)", "95% CI Lower", "95% CI Upper", "P-value", "AIC", "BIC",
             "DHARMa KS p", "LRT p-value", "R² / pseudo-R²",
             "Random Effect SD", "ICC", "LOO Effect Range"),
  `Binomial GLMM` = c(sprintf("OR = %.2f", glmer_or), sprintf("%.2f", glmer_ci[1]),
                       sprintf("%.2f", glmer_ci[2]), sprintf("%.2e", glmer_pval),
                       sprintf("%.0f", aic_glmer), sprintf("%.0f", bic_glmer),
                       sprintf("%.4f", dharma_tests_glmer$uniformity$p.value),
                       sprintf("%.2e", lrt_glmer$`Pr(>Chisq)`[2]),
                       sprintf("%.4f (McFadden)", as.numeric(mcfadden_r2)),
                       sprintf("%.4f", vc_glmer$sdcor[1]),
                       "N/A",
                       sprintf("[%.2f, %.2f]", min(loo_results$glmer_or, na.rm = TRUE),
                               max(loo_results$glmer_or, na.rm = TRUE))),
  `Log2 LMM` = c(sprintf("FC = %.2f", fold_change), sprintf("%.2f", fold_change_ci[1]),
                  sprintf("%.2f", fold_change_ci[2]), sprintf("%.2e", log2_pval),
                  sprintf("%.0f", aic_log2), sprintf("%.0f", bic_log2),
                  sprintf("%.4f", dharma_tests_log2$uniformity$p.value),
                  sprintf("%.2e", lrt_lmm$`Pr(>Chisq)`[2]),
                  sprintf("%.3f (cond) / %.3f (marg)", r2_conditional, r2_marginal),
                  sprintf("%.4f", vc_log2$sdcor[1]),
                  sprintf("%.3f", icc_log2),
                  sprintf("[%.2f, %.2f]", min(loo_results$lmm_fc, na.rm = TRUE),
                          max(loo_results$lmm_fc, na.rm = TRUE))),
  check.names = FALSE
)
write.csv(summary_df, file.path(output_dir, "model_comparison_summary.csv"), row.names = FALSE)

# === FINAL OUTPUT ===
cat("\n=== ANALYSIS COMPLETE ===\n")
cat("Output:", output_dir, "\n\n")
cat("Files:\n")
cat("  01_data_overview.png              - Side-by-side data visualization\n")
cat("  02_dharma_comparison.png          - DHARMa diagnostics (2x2)\n")
cat("  03_predictions_comparison.png     - Model predictions\n")
cat("  04_effect_comparison.png          - Combined forest plot\n")
cat("  05_random_effects_diagnostics.png - Random effects Q-Q and BLUPs\n")
cat("  06_cooks_distance.png             - Influential observations\n")
cat("  07_loo_sensitivity.png            - Leave-one-out sensitivity\n")
cat("  08_residuals_by_sample.png        - Residuals by sample\n")
cat("  model_comparison_summary.csv      - Comprehensive summary table\n")
cat("  loo_sensitivity_results.csv       - LOO analysis results\n\n")

cat("Summary:\n")
cat(sprintf("  Binomial GLMM: OR = %.2f, p = %.2e (McFadden R² = %.4f)\n",
            glmer_or, glmer_pval, as.numeric(mcfadden_r2)))
cat(sprintf("  Log2 LMM:      FC = %.2f, p = %.2e (R² cond = %.3f)\n",
            fold_change, log2_pval, r2_conditional))
cat(sprintf("\n  Effective sample size: %.0f (of %d cells, DEFF = %.2f)\n",
            eff_sample_size, n_cells, design_effect))
cat(sprintf("  LOO stability - OR range: [%.2f, %.2f], FC range: [%.2f, %.2f]\n",
            min(loo_results$glmer_or, na.rm = TRUE), max(loo_results$glmer_or, na.rm = TRUE),
            min(loo_results$lmm_fc, na.rm = TRUE), max(loo_results$lmm_fc, na.rm = TRUE)))
cat("\nNote: Both models show increased intron retention in ALS\n")
cat("      OR and FC are on comparable ratio scales (reference = 1)\n")
