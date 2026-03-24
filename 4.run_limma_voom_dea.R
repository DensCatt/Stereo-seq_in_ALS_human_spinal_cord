#!/usr/bin/env Rscript
#
# run_limma_voom_dea.R
# Standalone limma-voom differential expression analysis for pseudobulk data
#
# Usage:
#   Rscript run_limma_voom_dea.R \
#     --counts pseudobulk_counts.tsv \
#     --meta pseudobulk_meta.tsv \
#     --outdir dea_results \
#     --condition condition \
#     --covariates batch,n_cells \
#     --fdr 0.05 \
#     --logfc 1
#

suppressPackageStartupMessages({
  library(edgeR)
  library(limma)
  library(optparse)
  library(ggplot2)
  library(ggrepel)
})

# Parse command-line arguments
option_list <- list(
  make_option(c("--counts"), type="character", help="Pseudobulk counts file (TSV)"),
  make_option(c("--meta"), type="character", help="Sample metadata file (TSV)"),
  make_option(c("--outdir"), type="character", default="dea_results", help="Output directory"),
  make_option(c("--condition"), type="character", default="condition", help="Condition column name"),
  make_option(c("--covariates"), type="character", default=NULL,
              help="Comma-separated covariates (e.g., 'batch,n_cells')"),
  make_option(c("--fdr"), type="numeric", default=0.05, help="FDR threshold"),
  make_option(c("--logfc"), type="numeric", default=1, help="Log2FC threshold"),
  make_option(c("--layer-name"), type="character", default="counts",
              help="Layer name (for output file naming)"),
  make_option(c("--counts-other"), type="character", default=NULL,
              help="Comma-separated paths to other layer counts for joint filtering (filterByExpr on all layers)")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

# Check required arguments
if (is.null(opt$counts) | is.null(opt$meta)) {
  print_help(opt_parser)
  stop("--counts and --meta are required")
}

msg <- function(...) cat("[", format(Sys.time(), "%H:%M:%S"), "]", ..., "\n")

dir.create(opt$outdir, showWarnings=FALSE, recursive=TRUE)

# ============================================================================
# LOAD DATA
# ============================================================================
msg("Loading data...")
counts <- read.delim(opt$counts, row.names=1, check.names=FALSE)
meta <- read.delim(opt$meta, row.names=1, check.names=FALSE)

msg("Counts: ", nrow(counts), " genes x ", ncol(counts), " samples")
msg("Metadata: ", nrow(meta), " samples")

# Ensure order matches
meta <- meta[colnames(counts), , drop=FALSE]

# Set factor levels explicitly with Control/CTR as reference
# This ensures the comparison is "ALS - Control" (or disease - control)
cond_values <- unique(as.character(meta[[opt$condition]]))

# Find control condition (can be "Control", "CTR", "CTRL", "control", etc.)
control_patterns <- c("Control", "control", "CONTROL", "CTR", "Ctr", "ctr", "CTRL", "ctrl")
control_cond <- intersect(cond_values, control_patterns)

if (length(control_cond) > 0) {
  # Put control first (reference level), then other conditions alphabetically
  cond_levels <- c(control_cond[1], setdiff(sort(cond_values), control_cond))
  msg("Setting ", control_cond[1], " as reference (control) condition")
} else {
  # If no obvious control, use alphabetical order
  cond_levels <- sort(cond_values)
  msg("No standard control label found, using alphabetical ordering")
}
meta[[opt$condition]] <- factor(as.character(meta[[opt$condition]]), levels = cond_levels)

msg("Condition levels: ", paste(levels(meta[[opt$condition]]), collapse=", "))
msg("  Reference level (first): ", levels(meta[[opt$condition]])[1])
msg("  Comparison level(s): ", paste(levels(meta[[opt$condition]])[-1], collapse=", "))

# Sample counts
sample_counts <- table(meta[[opt$condition]])
msg("Samples per condition:")
for (i in seq_along(sample_counts)) {
  msg("  ", names(sample_counts)[i], ": ", sample_counts[i])
}

# ============================================================================
# PREPARE DESIGN
# ============================================================================
covariates <- NULL
if (!is.null(opt$covariates)) {
  covariates <- unlist(strsplit(opt$covariates, ","))
  covariates <- trimws(covariates)

  # Validate covariates exist
  missing <- setdiff(covariates, colnames(meta))
  if (length(missing) > 0) {
    msg("WARNING: Covariates not found: ", paste(missing, collapse=", "))
    covariates <- intersect(covariates, colnames(meta))
  }

  if (length(covariates) > 0) {
    msg("Covariates: ", paste(covariates, collapse=", "))
    design_formula <- paste0("~ ", paste(c(covariates, opt$condition), collapse=" + "))
  } else {
    msg("No valid covariates, using condition only")
    design_formula <- paste0("~ ", opt$condition)
  }
} else {
  design_formula <- paste0("~ ", opt$condition)
}

msg("Design formula: ", design_formula)

# ============================================================================
# FILTERING & NORMALIZATION
# ============================================================================
msg("Creating DGEList...")
y <- DGEList(counts=counts)

# ============================================================================
# GENE FILTERING
# ============================================================================
if (!is.null(opt$`counts-other`)) {
  # Multi-layer filtering: filterByExpr on each layer, keep intersection
  # (Lee et al. 2020 approach)
  msg("Multi-layer gene filtering (filterByExpr on all layers)...")

  # Create design matrix for filterByExpr
  design_for_filter <- model.matrix(as.formula(design_formula), data=meta)

  other_files <- unlist(strsplit(opt$`counts-other`, ","))
  other_files <- trimws(other_files)

  # Start with filterByExpr on primary layer
  keep_primary <- filterByExpr(y, design=design_for_filter)
  msg("  ", opt$`layer-name`, " (primary): ", sum(keep_primary), " genes pass filterByExpr")

  # Combined mask starts with primary layer
  keep_all <- keep_primary

  # Apply filterByExpr to each other layer
  for (other_file in other_files) {
    if (!file.exists(other_file)) {
      msg("  WARNING: File not found: ", other_file, " - skipping")
      next
    }

    # Extract layer name from filename
    other_name <- gsub(".*pseudobulk_(.*)_counts\\.tsv", "\\1", basename(other_file))

    # Load other layer counts
    other_counts <- read.delim(other_file, row.names=1, check.names=FALSE)

    # Ensure same samples in same order
    other_counts <- other_counts[, colnames(counts), drop=FALSE]

    # Ensure same genes (use intersection)
    common_genes <- intersect(rownames(counts), rownames(other_counts))
    if (length(common_genes) < nrow(counts)) {
      msg("  WARNING: ", other_name, " has ", length(common_genes), "/", nrow(counts), " genes in common")
    }

    # Create DGEList for other layer
    y_other <- DGEList(counts=other_counts[common_genes, , drop=FALSE])

    # Run filterByExpr on this layer
    keep_other <- filterByExpr(y_other, design=design_for_filter)

    # Map back to full gene set (genes not in other layer fail filter)
    keep_other_full <- rep(FALSE, nrow(y))
    names(keep_other_full) <- rownames(y)
    keep_other_full[common_genes] <- keep_other

    msg("  ", other_name, ": ", sum(keep_other), " genes pass filterByExpr")

    # Intersect with combined mask
    keep_all <- keep_all & keep_other_full
  }

  msg("Genes passing filterByExpr in ALL layers: ", sum(keep_all), " / ", nrow(y))
  keep <- keep_all

} else {
  # Original single-layer filtering (default behavior)
  msg("Filtering low-expression genes...")
  keep <- rowSums(cpm(y) > 1) >= 2
  msg("  Keeping ", sum(keep), " / ", nrow(y), " genes")
}

y <- y[keep, , keep.lib.sizes=FALSE]

msg("TMM normalization...")
y <- calcNormFactors(y, method="TMM")

msg("Normalization factors:")
print(data.frame(sample=colnames(y), norm.factor=y$samples$norm.factors))

# ============================================================================
# VOOM TRANSFORMATION
# ============================================================================
msg("Creating design matrix...")
design <- model.matrix(as.formula(design_formula), data=meta)
colnames(design) <- make.names(colnames(design))

msg(strrep("=", 70))
msg("MODEL SPECIFICATION")
msg(strrep("=", 70))
msg("Formula: ", design_formula)
msg("Design matrix dimensions: ", nrow(design), " samples x ", ncol(design), " coefficients")
msg("Design matrix columns: ", paste(colnames(design), collapse=", "))

# Check for rank deficiency
design_rank <- qr(design)$rank
msg("Design matrix rank: ", design_rank, " / ", ncol(design))
if (design_rank < ncol(design)) {
  msg("WARNING: Design matrix is not full rank (collinearity detected)")
  msg("         ", ncol(design) - design_rank, " coefficients are redundant")
}

# Print design matrix
msg("\nDesign matrix (first 10 samples):")
print(head(design, 10))

# Show metadata for context
msg("\nMetadata summary:")
for (col in colnames(meta)) {
  if (is.numeric(meta[[col]])) {
    msg("  ", col, ": ", paste(round(range(meta[[col]], na.rm=TRUE), 2), collapse=" to "))
  } else {
    msg("  ", col, ": ", paste(unique(meta[[col]]), collapse=", "))
  }
}

# Check correlation between covariates and condition
if (!is.null(covariates) && length(covariates) > 0) {
  msg("\nCovariate-Condition associations:")
  for (cov in covariates) {
    if (cov %in% colnames(meta)) {
      if (is.numeric(meta[[cov]])) {
        # For numeric: show mean per condition
        cond_means <- tapply(meta[[cov]], meta[[opt$condition]], mean, na.rm=TRUE)
        msg("  ", cov, " by condition: ", paste(names(cond_means), "=", round(cond_means, 2), collapse=", "))
      } else {
        # For categorical: show contingency table
        ct <- table(meta[[cov]], meta[[opt$condition]])
        msg("  ", cov, " x ", opt$condition, ":")
        print(ct)
      }
    }
  }
}
msg(strrep("=", 70))

msg("Running voom with quality weights...")
vqw <- tryCatch({
  voomWithQualityWeights(y, design=design, plot=FALSE)
}, error = function(e) {
  msg("voomWithQualityWeights failed, using standard voom")
  msg("Error: ", e$message)
  voom(y, design=design, plot=FALSE)
})

# Extract sample weights
if (!is.null(vqw$targets$sample.weights)) {
  sample_weights <- data.frame(
    sample = rownames(vqw$targets),
    weight = vqw$targets$sample.weights
  )
  sample_weights <- sample_weights[order(sample_weights$weight), ]

  write.table(sample_weights,
              file=file.path(opt$outdir, paste0("dea_", opt$`layer-name`, "_sample_weights.tsv")),
              sep="\t", quote=FALSE, row.names=FALSE)

  msg("Sample weights (lowest first):")
  print(head(sample_weights))

  if (any(sample_weights$weight < 0.8)) {
    low_weight <- sample_weights[sample_weights$weight < 0.8, ]
    msg("WARNING: ", nrow(low_weight), " samples with weight < 0.8:")
    print(low_weight)
  }
}

# ============================================================================
# DIFFERENTIAL EXPRESSION
# ============================================================================
msg("Fitting linear model...")
fit <- lmFit(vqw, design)
fit <- eBayes(fit)

msg(strrep("=", 70))
msg("COEFFICIENT TESTING")
msg(strrep("=", 70))

# Find condition coefficient
cond_cols <- grep(paste0("^", opt$condition), colnames(design), value=TRUE)
if (length(cond_cols) == 0) {
  stop("No condition coefficient found in design matrix")
}
coef_name <- cond_cols[length(cond_cols)]

msg("All coefficients in model: ", paste(colnames(design), collapse=", "))
msg("Testing coefficient: ", coef_name)
msg("\nInterpretation:")
msg("  This coefficient represents the effect of ", opt$condition)
msg("  Reference level: ", levels(meta[[opt$condition]])[1])
msg("  Comparison: ", levels(meta[[opt$condition]])[length(levels(meta[[opt$condition]]))], " - ", levels(meta[[opt$condition]])[1])
msg("  LogFC > 0: Higher expression in ", levels(meta[[opt$condition]])[length(levels(meta[[opt$condition]]))])
msg("  LogFC < 0: Lower expression in ", levels(meta[[opt$condition]])[length(levels(meta[[opt$condition]]))], " (higher in ", levels(meta[[opt$condition]])[1], ")")

# Show coefficient estimates for a few genes
msg("\nExample fitted coefficients (first 5 genes):")
coef_idx <- which(colnames(fit$coefficients) == coef_name)
example_genes <- head(fit$coefficients[, coef_idx, drop=FALSE], 5)
print(round(example_genes, 3))
msg(strrep("=", 70))

msg("Extracting results...")
top <- topTable(fit, coef=coef_name, number=Inf, sort.by="P")

# Add significance annotation
top$significant <- (top$adj.P.Val < opt$fdr) & (abs(top$logFC) > opt$logfc)
top$direction <- ifelse(top$logFC > 0, "up", "down")

# Summary
n_sig <- sum(top$significant)
n_up <- sum(top$significant & top$direction == "up")
n_down <- sum(top$significant & top$direction == "down")

msg(strrep("=", 70))
msg("RESULTS SUMMARY")
msg(strrep("=", 70))
msg("Total genes tested: ", nrow(top))
msg("Significant (FDR < ", opt$fdr, ", |logFC| > ", opt$logfc, "): ", n_sig)
msg("  Up-regulated: ", n_up)
msg("  Down-regulated: ", n_down)

# ============================================================================
# SAVE RESULTS
# ============================================================================
msg("Saving results...")

# Full results
results_file <- file.path(opt$outdir, paste0("dea_", opt$`layer-name`, "_results.tsv"))
write.table(top, file=results_file, sep="\t", quote=FALSE, row.names=TRUE)
msg("Saved: ", results_file)

# Significant genes only
if (n_sig > 0) {
  sig_file <- file.path(opt$outdir, paste0("dea_", opt$`layer-name`, "_significant.tsv"))
  write.table(top[top$significant, ], file=sig_file, sep="\t", quote=FALSE, row.names=TRUE)
  msg("Saved: ", sig_file)

  # Top 20 genes
  top20_file <- file.path(opt$outdir, paste0("dea_", opt$`layer-name`, "_top20.tsv"))
  write.table(head(top[top$significant, ], 20), file=top20_file, sep="\t", quote=FALSE, row.names=TRUE)
  msg("Saved: ", top20_file)
}

# Summary statistics
summary_df <- data.frame(
  metric = c("total_genes", "significant", "up_regulated", "down_regulated",
             "median_logFC_up", "median_logFC_down"),
  value = c(nrow(top), n_sig, n_up, n_down,
            ifelse(n_up > 0, median(top$logFC[top$significant & top$direction == "up"]), NA),
            ifelse(n_down > 0, median(top$logFC[top$significant & top$direction == "down"]), NA))
)
write.table(summary_df, file=file.path(opt$outdir, paste0("dea_", opt$`layer-name`, "_summary.tsv")),
            sep="\t", quote=FALSE, row.names=FALSE)

# ============================================================================
# PLOTS
# ============================================================================
msg("Generating plots...")

# Volcano plot with ggplot2 and ggrepel
volcano_file <- file.path(opt$outdir, paste0("dea_", opt$`layer-name`, "_volcano.png"))

# Prepare data for plotting
top$gene <- rownames(top)
top$category <- "Not significant"
top$category[top$significant & top$direction == "up"] <- "Up"
top$category[top$significant & top$direction == "down"] <- "Down"

# For labeling: use FDR < 0.05 regardless of input threshold
# This ensures visual consistency across different threshold settings
top$label_gene <- ifelse(top$adj.P.Val < 0.05, top$gene, "")

# Limit labels if too many
genes_to_label <- top[top$adj.P.Val < 0.05, ]
max_labels <- 50
if (nrow(genes_to_label) > max_labels) {
  msg("FDR < 0.05: ", nrow(genes_to_label), " genes. Labeling top ", max_labels, " by p-value")
  genes_to_label <- genes_to_label[order(genes_to_label$adj.P.Val), ][1:max_labels, ]
  top$label_gene <- ifelse(top$gene %in% genes_to_label$gene, top$gene, "")
} else {
  msg("Labeling all ", nrow(genes_to_label), " genes with FDR < 0.05")
}

# Create volcano plot
p <- ggplot(top, aes(x = logFC, y = -log10(adj.P.Val), color = category, label = label_gene)) +
  geom_point(size = 1.5, alpha = 0.6) +
  scale_color_manual(values = c("Not significant" = "grey70", "Up" = "red", "Down" = "blue"),
                     labels = c(paste0("Down (", n_down, ")"),
                               "Not sig",
                               paste0("Up (", n_up, ")"))) +
  geom_hline(yintercept = -log10(opt$fdr), linetype = "dashed", color = "black", linewidth = 0.5) +
  geom_vline(xintercept = c(-opt$logfc, opt$logfc), linetype = "dashed", color = "black", linewidth = 0.5) +
  geom_text_repel(
    size = 2.5,
    max.overlaps = Inf,
    min.segment.length = 0,
    segment.size = 0.2,
    segment.color = "grey50",
    box.padding = 0.5,
    point.padding = 0.3,
    force = 2,
    force_pull = 0.5
  ) +
  labs(
    title = paste0("Volcano Plot: ", opt$`layer-name`),
    subtitle = paste0(n_sig, " significant genes (FDR < ", opt$fdr, ", |logFC| > ", opt$logfc, ")"),
    x = "log2 Fold Change",
    y = "-log10(FDR)",
    color = "Significance"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 10),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

ggsave(volcano_file, plot = p, width = 12, height = 9, dpi = 300, bg = "white")
msg("Saved: ", volcano_file)

# MA plot
ma_file <- file.path(opt$outdir, paste0("dea_", opt$`layer-name`, "_MA.png"))
png(ma_file, width=1000, height=800, res=120)

# Define colors for MA plot
ma_colors <- c("Not significant" = "grey70", "Up" = "red", "Down" = "blue")

plot(log2(top$AveExpr + 0.1), top$logFC,
     pch=20, cex=0.6, col=ma_colors[top$category],
     xlab="log2(Average Expression)", ylab="log2 Fold Change",
     main=paste0("MA Plot: ", opt$`layer-name`))

abline(h=0, lty=2, col="black")
abline(h=c(-opt$logfc, opt$logfc), lty=3, col="grey50")

dev.off()
msg("Saved: ", ma_file)

# P-value histogram
pval_file <- file.path(opt$outdir, paste0("dea_", opt$`layer-name`, "_pvalue_hist.png"))
png(pval_file, width=800, height=600, res=120)

hist(top$P.Value, breaks=50, col="skyblue", border="white",
     xlab="P-value", main="P-value Distribution")
abline(v=0.05, col="red", lty=2, lwd=2)

dev.off()
msg("Saved: ", pval_file)

# ============================================================================
# DONE
# ============================================================================
msg(strrep("=", 70))
msg("ANALYSIS COMPLETE")
msg(strrep("=", 70))
msg("Results saved to: ", opt$outdir)
msg("\nKey files:")
msg("  - ", basename(results_file), " (all genes)")
if (n_sig > 0) {
  msg("  - ", paste0("dea_", opt$`layer-name`, "_significant.tsv"), " (significant genes)")
  msg("  - ", paste0("dea_", opt$`layer-name`, "_top20.tsv"), " (top 20 genes)")
}
msg("  - ", paste0("dea_", opt$`layer-name`, "_volcano.png"))
msg("  - ", paste0("dea_", opt$`layer-name`, "_MA.png"))
