# Simple RDS to CSV/MTX converter (no special packages needed)
# Run this in R before running the Python comparison

library(Seurat)
library(Matrix)

# Set working directory
setwd("C:/Users/catta/Desktop/ALS/Gautier/")

# Load the RDS file
print("Loading RDS file...")
seurat_obj <- readRDS("GSE228778_gautier_mns.rds")

print(paste("Loaded Seurat object with", ncol(seurat_obj), "cells and", nrow(seurat_obj), "genes"))
print(paste("Available assays:", paste(names(seurat_obj@assays), collapse = ", ")))

# Get the default assay (usually RNA)
default_assay <- DefaultAssay(seurat_obj)
print(paste("Using assay:", default_assay))

# Extract counts matrix (sparse format)
print("\nExtracting counts matrix...")
counts <- GetAssayData(seurat_obj, slot = "counts", assay = default_assay)
print(paste("Counts matrix:", nrow(counts), "genes x", ncol(counts), "cells"))

# Save as Matrix Market format (efficient for sparse matrices)
print("Saving counts matrix...")
writeMM(counts, "gautier_mns_counts.mtx")

# Extract and save metadata
print("Saving metadata...")
metadata <- seurat_obj@meta.data
write.csv(metadata, "gautier_mns_metadata.csv", row.names = TRUE)
print(paste("Saved metadata with", ncol(metadata), "columns:", paste(colnames(metadata), collapse = ", ")))

# Save gene names
print("Saving gene names...")
genes <- rownames(counts)
write.csv(data.frame(gene_name = genes), "gautier_mns_genes.csv", row.names = FALSE)

# Save cell barcodes
print("Saving cell barcodes...")
cells <- colnames(counts)
write.csv(data.frame(barcode = cells), "gautier_mns_cells.csv", row.names = FALSE)

# Try to get normalized data if available
print("Checking for normalized data...")
tryCatch({
  norm_data <- GetAssayData(seurat_obj, slot = "data", assay = default_assay)
  if (!identical(norm_data, counts)) {
    writeMM(norm_data, "gautier_mns_normalized.mtx")
    print("Saved normalized data")
  } else {
    print("No separate normalized data found")
  }
}, error = function(e) {
  print("No normalized data available")
})

# Save dimensional reductions if available
print("\nChecking for dimensional reductions...")
if ("pca" %in% names(seurat_obj@reductions)) {
  pca_embeddings <- Embeddings(seurat_obj, reduction = "pca")
  write.csv(pca_embeddings, "gautier_mns_pca.csv", row.names = TRUE)
  print(paste("Saved PCA embeddings:", nrow(pca_embeddings), "cells x", ncol(pca_embeddings), "PCs"))
}

if ("umap" %in% names(seurat_obj@reductions)) {
  umap_embeddings <- Embeddings(seurat_obj, reduction = "umap")
  write.csv(umap_embeddings, "gautier_mns_umap.csv", row.names = TRUE)
  print(paste("Saved UMAP embeddings:", nrow(umap_embeddings), "cells x", ncol(umap_embeddings), "dims"))
}

print("\n========================================")
print("✓ Export complete!")
print("========================================")
print("Now run the Python script to load the data.")
print("========================================")
