#!/usr/bin/env python3
"""
Aggregate Motor Neurons - Preserving All Layers

This script takes a labeled h5ad file and aggregates motor neuron regions
while preserving all layers (counts, exon, intron).

Usage:
    python 1.aggregate_motorneurons.py --input <input_h5ad> --output <output_h5ad>

Example:
    python 1.aggregate_motorneurons.py --input s18-070_bin20.ALL_layers_labeled.h5ad --output s18-070_bin20.FINAL.h5ad
"""

import os
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix, vstack


def aggregate_with_layers(adata):
    """
    Aggregate motor neuron regions while preserving all layers

    Args:
        adata: AnnData object with labeled regions

    Returns:
        AnnData object with aggregated motor neurons
    """
    print("\n" + "="*60)
    print("Aggregating labeled regions...")
    print("="*60)

    # Fix label column - handle categorical types
    if pd.api.types.is_categorical_dtype(adata.obs["label"]):
        adata.obs["label"] = adata.obs["label"].astype(str)
    # Now fill NaN and convert to int
    adata.obs["label"] = adata.obs["label"].fillna("0").astype("int32")

    # Drop orig.ident if it exists
    if "orig.ident" in adata.obs.columns:
        adata.obs.drop(columns=["orig.ident"], inplace=True)

    # Make var_names unique
    adata.var_names_make_unique()

    # Separate labeled and unlabeled regions
    adata_labeled = adata[adata.obs["label"] != 0].copy()
    adata_zero = adata[adata.obs["label"] == 0].copy()

    labels = np.sort(adata_labeled.obs["label"].unique())

    # Check which layers are available
    available_layers = list(adata.layers.keys()) if adata.layers else []
    print(f"Available layers: {available_layers}")

    # Initialize lists for aggregated data
    counts_list = []
    obs_list = []
    spatial_coords = []

    # Initialize layer dictionaries
    layer_data = {layer_name: [] for layer_name in available_layers}

    print(f"Aggregating {len(labels)} motor neuron regions...")

    for label in labels:
        adata_label = adata_labeled[adata_labeled.obs["label"] == label]

        # Aggregate main X matrix
        summed_counts = adata_label.X.sum(axis=0)
        counts_list.append(summed_counts)

        # Aggregate layers if they exist
        for layer_name in available_layers:
            layer_summed = adata_label.layers[layer_name].sum(axis=0)
            layer_data[layer_name].append(layer_summed)

        # Aggregate observation metadata
        obs_row = {
            "total_counts": adata_label.obs["total_counts"].sum(),
            "n_genes_by_counts": adata_label.obs["n_genes_by_counts"].mean(),
            "pct_counts_mt": adata_label.obs["pct_counts_mt"].mean(),
            "x": adata_label.obs["x"].mean(),
            "y": adata_label.obs["y"].mean(),
            "label": label,
            "n_bins_aggregated": adata_label.n_obs
        }
        obs_list.append(obs_row)
        spatial_coords.append([obs_row["x"], obs_row["y"]])

    # Create aggregated AnnData for labeled regions
    X_agg = vstack([csr_matrix(c) for c in counts_list]).astype(np.float32)
    obs_agg = pd.DataFrame(obs_list)
    obs_agg["label"] = obs_agg["label"].astype(str)
    obs_agg["n_bins_aggregated"] = obs_agg["n_bins_aggregated"].astype(np.int32)
    obs_agg["n_genes_by_counts"] = obs_agg["n_genes_by_counts"].astype(np.float32)
    obs_agg["x"] = obs_agg["x"].astype(np.float32)
    obs_agg["y"] = obs_agg["y"].astype(np.float32)
    obs_agg.index = [f"MN_{i}" for i in range(1, len(obs_agg) + 1)]

    # Create aggregated layers
    layers_agg = {}
    for layer_name in available_layers:
        layers_agg[layer_name] = vstack([csr_matrix(c) for c in layer_data[layer_name]]).astype(np.float32)
        print(f"  Aggregated layer '{layer_name}': {layers_agg[layer_name].shape}")

    # Fix unlabeled bins metadata
    adata_zero.obs["n_bins_aggregated"] = 1
    adata_zero.obs["label"] = adata_zero.obs["label"].astype(str)
    adata_zero.obs["n_bins_aggregated"] = adata_zero.obs["n_bins_aggregated"].astype(np.int32)
    adata_zero.obs["n_genes_by_counts"] = adata_zero.obs["n_genes_by_counts"].astype(np.float32)
    adata_zero.obs["x"] = adata_zero.obs["x"].astype(np.float32)
    adata_zero.obs["y"] = adata_zero.obs["y"].astype(np.float32)
    adata_zero.obs.index = [f"bin_{i}" for i in range(1, adata_zero.n_obs + 1)]

    # Extract spatial coordinates for unlabeled bins
    if "x" in adata_zero.obs.columns and "y" in adata_zero.obs.columns:
        spatial_coords_zero = np.column_stack([adata_zero.obs["x"].values, adata_zero.obs["y"].values]).astype(np.float32)
    else:
        spatial_coords_zero = None

    # Create AnnData object for aggregated motor neurons
    adata_agg = ad.AnnData(
        X=X_agg,
        obs=obs_agg,
        var=adata.var.copy(),
        layers=layers_agg,
        obsm={"spatial": np.array(spatial_coords, dtype=np.float32)},
        uns=adata.uns.copy()
    )

    # Ensure unlabeled bins have correct dtypes
    adata_zero.X = adata_zero.X.astype(np.float32)

    # Convert layers in unlabeled bins to float32
    for layer_name in available_layers:
        if layer_name in adata_zero.layers:
            adata_zero.layers[layer_name] = adata_zero.layers[layer_name].astype(np.float32)

    # Add spatial coordinates to unlabeled bins if available
    if spatial_coords_zero is not None:
        adata_zero.obsm["spatial"] = spatial_coords_zero

    # Concatenate unlabeled and aggregated data
    print("\nConcatenating unlabeled bins and aggregated motor neurons...")
    adata_final = ad.concat(
        [adata_zero, adata_agg],
        merge="same",
        uns_merge="same"
    )

    # Extract chip and sample from input data if available
    chip_name = adata.obs["chip"].iloc[0] if "chip" in adata.obs.columns else "unknown"
    sample_name = adata.obs["sample"].iloc[0] if "sample" in adata.obs.columns else "unknown"

    # Add metadata
    adata_final.obs["chip"] = chip_name
    adata_final.obs["sample"] = sample_name
    adata_final.obs["motorneuron"] = adata_final.obs["label"] != "0"

    # Summary
    print(f"\n{'='*60}")
    print(f"Aggregation Summary:")
    print(f"{'='*60}")
    print(f"Unlabeled bins: {adata_zero.n_obs}")
    print(f"Aggregated motor neurons: {len(labels)}")
    print(f"Total observations: {adata_final.n_obs}")
    print(f"Layers preserved: {list(adata_final.layers.keys())}")
    print(f"{'='*60}\n")

    return adata_final


def main(input_file, output_file):
    """
    Main function to aggregate motor neurons

    Args:
        input_file: Path to input labeled h5ad file
        output_file: Path to output aggregated h5ad file
    """
    print(f"\n{'='*60}")
    print(f"Motor Neuron Aggregation Pipeline")
    print(f"{'='*60}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")

    # Read input file
    print("Reading input file...")
    adata = sc.read_h5ad(input_file)
    print(f"✅ Loaded: {adata.n_obs} observations × {adata.n_vars} variables")
    print(f"Layers: {list(adata.layers.keys())}")

    # Aggregate
    adata_final = aggregate_with_layers(adata)

    # Save output
    print(f"Saving to {output_file}...")
    adata_final.write_h5ad(output_file)

    print(f"\n{'='*60}")
    print(f"✅ Aggregation completed successfully!")
    print(f"{'='*60}")
    print(f"Output saved: {output_file}")
    print(f"Total observations: {adata_final.n_obs}")
    print(f"Total variables: {adata_final.n_vars}")
    print(f"Layers: {list(adata_final.layers.keys())}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate motor neurons while preserving all layers (counts, exon, intron)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--input", required=True, help="Path to input labeled h5ad file")
    parser.add_argument("--output", required=True, help="Path to output aggregated h5ad file")

    args = parser.parse_args()

    main(
        input_file=args.input,
        output_file=args.output
    )
