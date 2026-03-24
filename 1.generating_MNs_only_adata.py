#%%
import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

os.chdir("/media/brain/large_drive/Stereoseq_2025/saw_output")

# Sample dictionary (removed s12-186)
samples = {
    "s15-051": ["A05326G4", "CTR"],
    "s15-025": ["Y01048K8", "CTR"],
    "s18-070": ["A05328G6", "ALS"],
    "s13-047": ["Y01048H6", "ALS"],
    "s09-055": ["Y01065A8", "ALS"],
    "s13-133": ["Y01065J9", "CTR"]
}

binsize = "20"
MNs_adatas = []


def remove_duplicate_genes(adata):
    """
    Remove duplicate genes while preserving all layers

    Args:
        adata: AnnData object

    Returns:
        AnnData object with duplicates removed
    """
    # Identify duplicate genes
    gene_counts = adata.var["real_gene_name"].value_counts()
    duplicated_genes = gene_counts[gene_counts > 1].index.tolist()

    # Create mask for genes to keep
    keep_mask = ~adata.var["real_gene_name"].isin(duplicated_genes)

    # Subset the data (this preserves layers)
    adata_filtered = adata[:, keep_mask].copy()

    # Make var_names unique
    adata_filtered.var_names_make_unique()

    return adata_filtered


for sample, info in samples.items():
    print(f"Processing {sample}...")

    # Load FINAL file (motor neurons)
    adata = sc.read_h5ad(f"{sample}/outs/{info[0]}.bin{binsize}.FINAL.h5ad")

    # Check available layers
    available_layers = list(adata.layers.keys()) if adata.layers else []
    print(f"  Available layers: {available_layers}")

    # Check existing metadata
    print(f"  Existing obs columns: {adata.obs.columns.tolist()}")

    # Select motor neurons (label != "0")
    motorneurons = adata[adata.obs["label"] != "0", :].copy()

    # Drop label column to avoid duplicates during concatenation
    motorneurons.obs.drop(columns=["label"], inplace=True)

    # Add/update metadata (preserves existing chip, sample, motorneuron, n_bins_aggregated if present)
    motorneurons.obs["condition"] = info[1]
    motorneurons.obs["type"] = "MN"

    # Ensure critical metadata exists (in case it's missing from older files)
    if "chip" not in motorneurons.obs.columns:
        motorneurons.obs["chip"] = info[0]
    if "sample" not in motorneurons.obs.columns:
        motorneurons.obs["sample"] = sample
    if "n_bins_aggregated" not in motorneurons.obs.columns:
        motorneurons.obs["n_bins_aggregated"] = 1  # default if not aggregated

    # Remove duplicate genes (preserves layers)
    motorneurons = remove_duplicate_genes(motorneurons)

    print(f"  Motor neurons: {motorneurons.n_obs}, Genes: {motorneurons.n_vars}")

    MNs_adatas.append(motorneurons)

    #################################################
    # For s15-051 and s13-133, also load NEURONS file
    #################################################
    if sample in ("s15-051", "s13-133"):
        print(f"  Loading NEURONS file for {sample}...")
        adata = sc.read_h5ad(f"{sample}/outs/{info[0]}.bin{binsize}.NEURONS.h5ad")

        # Select labeled neurons
        neurons = adata[adata.obs["label"] != "0", :].copy()

        # Drop label column
        neurons.obs.drop(columns=["label"], inplace=True)

        # Add/update metadata
        neurons.obs["condition"] = info[1]
        neurons.obs["type"] = "Neuron"

        # Ensure critical metadata exists
        if "chip" not in neurons.obs.columns:
            neurons.obs["chip"] = info[0]
        if "sample" not in neurons.obs.columns:
            neurons.obs["sample"] = sample
        if "n_bins_aggregated" not in neurons.obs.columns:
            neurons.obs["n_bins_aggregated"] = 1  # default if not aggregated

        # Remove duplicate genes (preserves layers)
        neurons = remove_duplicate_genes(neurons)

        print(f"  Neurons: {neurons.n_obs}, Genes: {neurons.n_vars}")

        MNs_adatas.append(neurons)

#################################################
# Concatenate all datasets (OUTSIDE the loop!)
#################################################
print("\nConcatenating all datasets...")
print(f"Total datasets to merge: {len(MNs_adatas)}")

adata_merged = ad.concat(
    MNs_adatas,
    join="inner",  # Only keep genes present in all datasets
    index_unique="_",  # Make observation names unique by adding suffix
    uns_merge="same",  # Merge uns metadata
    merge="first",  # For conflicting metadata, take first
)

print("\n" + "="*60)
print("Merge Summary:")
print("="*60)
print(f"Total observations: {adata_merged.n_obs}")
print(f"Total variables (genes): {adata_merged.n_vars}")
print(f"Layers preserved: {list(adata_merged.layers.keys())}")
print(f"\nMetadata columns: {adata_merged.obs.columns.tolist()}")
print(f"\nSamples: {sorted(adata_merged.obs['sample'].unique())}")
print(f"Chips: {sorted(adata_merged.obs['chip'].unique())}")
print(f"Conditions: {adata_merged.obs['condition'].value_counts().to_dict()}")
print(f"Types: {adata_merged.obs['type'].value_counts().to_dict()}")
if "n_bins_aggregated" in adata_merged.obs.columns:
    print(f"Bins aggregated (mean): {adata_merged.obs['n_bins_aggregated'].mean():.2f}")
if "motorneuron" in adata_merged.obs.columns:
    print(f"Motorneuron flag present: {adata_merged.obs['motorneuron'].value_counts().to_dict()}")
print("="*60)

# Save merged dataset
output_path = f"../motorneurons_bin{binsize}.h5ad"
print(f"\nSaving to {output_path}...")
adata_merged.write(output_path)
print(f"✅ DONE! Merged dataset saved.")

# %%
adata_merged
# %%
