#!/usr/bin/env python3
"""
Multi-layer pseudobulk DEA pipeline: analyze intron, exon, and total counts separately

Handles h5ad files with multiple count layers (e.g., intron, exon, spliced, unspliced)
Runs separate DEA on each layer to identify:
  - Transcriptional regulation (intron changes)
  - Post-transcriptional regulation (exon changes)
  - Overall expression changes (total counts)

Usage:
    python pseudobulk_pipeline_multilayer.py \\
        --h5ad data.h5ad \\
        --sample-col sample \\
        --condition-col condition \\
        --layers intron,exon,counts \\
        --r-script 3.run_limma_voom_dea.R \\
        --outdir ./multilayer_results \\
        --covariates batch
"""

import os
import sys
import argparse
import subprocess
import shutil
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def safe_mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def print_step(msg):
    print(f"\n{'='*70}\n{msg}\n{'='*70}")

def filter_cells(adata, layer_name, min_counts=500, max_mito=20, min_genes=200):
    """Filter cells based on a specific layer."""
    print_step(f"CELL FILTERING (layer: {layer_name})")
    n_cells_before = adata.n_obs

    # Get counts from layer
    if layer_name in adata.layers:
        X = adata.layers[layer_name]
    elif layer_name == "X":
        X = adata.X
    else:
        raise ValueError(f"Layer '{layer_name}' not found")

    try: X = X.tocsr()
    except: pass

    # Calculate metrics for this layer
    total_counts = np.array(X.sum(axis=1)).ravel()
    total_genes = np.array((X > 0).sum(axis=1)).ravel()

    # Use existing pct_mito if available, else calculate
    if "pct_mito" not in adata.obs.columns:
        mito_genes = [g for g in adata.var_names if str(g).upper().startswith("MT-")]
        if mito_genes:
            idx = [adata.var_names.get_loc(g) for g in mito_genes]
            mito_counts = np.array(X[:, idx].sum(axis=1)).ravel()
            adata.obs["pct_mito"] = mito_counts / (total_counts + 1e-9) * 100
        else:
            adata.obs["pct_mito"] = 0.0

    mask = (
        (total_counts >= min_counts) &
        (adata.obs["pct_mito"] <= max_mito) &
        (total_genes >= min_genes)
    )

    adata_filtered = adata[mask, :].copy()
    n_cells_after = adata_filtered.n_obs

    print(f"Layer: {layer_name}")
    print(f"Cells before: {n_cells_before}")
    print(f"Cells after:  {n_cells_after}")
    print(f"Removed:      {n_cells_before - n_cells_after} ({100*(n_cells_before - n_cells_after)/n_cells_before:.1f}%)")

    return adata_filtered

def filter_genes(adata, layer_name, min_cells=10, remove_mito=True, remove_ribo=True, remove_hb=True):
    """Filter genes based on expression in a specific layer."""
    print_step(f"GENE FILTERING (layer: {layer_name})")
    n_genes_before = adata.n_vars

    # Get layer
    if layer_name in adata.layers:
        X = adata.layers[layer_name]
    elif layer_name == "X":
        X = adata.X
    else:
        raise ValueError(f"Layer '{layer_name}' not found")

    try: X = X.tocsr()
    except: pass

    # Mark gene categories
    if "mt" not in adata.var.columns:
        adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
        adata.var["ribo"] = adata.var_names.str.upper().str.match("^RP[SL]")
        adata.var["hb"] = adata.var_names.str.upper().str.contains("^HB[^(P)]")

    n_mt = adata.var["mt"].sum()
    n_ribo = adata.var["ribo"].sum()
    n_hb = adata.var["hb"].sum()

    # Filter by expression in this layer
    n_cells_expressing = np.array((X > 0).sum(axis=0)).ravel()
    keep_expr = n_cells_expressing >= min_cells

    # Remove unwanted gene types
    mask = keep_expr.copy()
    if remove_mito: mask &= ~adata.var["mt"].values
    if remove_ribo: mask &= ~adata.var["ribo"].values
    if remove_hb: mask &= ~adata.var["hb"].values

    adata_filtered = adata[:, mask].copy()
    n_genes_after = adata_filtered.n_vars

    print(f"Layer: {layer_name}")
    print(f"Genes before: {n_genes_before} (MT:{n_mt}, Ribo:{n_ribo}, Hb:{n_hb})")
    print(f"Genes after:  {n_genes_after}")
    print(f"Removed:      {n_genes_before - n_genes_after} ({100*(n_genes_before - n_genes_after)/n_genes_before:.1f}%)")

    return adata_filtered


def filter_genes_multilayer(adata, layers_to_analyze, min_cells=10, remove_mito=True, remove_ribo=True, remove_hb=True):
    """
    Filter genes based on expression across ALL layers (intersection).

    A gene is kept only if it passes the min_cells threshold in EVERY layer.
    This follows Lee et al. (2020) approach of filterByExpr on both intron and exon.
    """
    print_step("GENE FILTERING (multi-layer intersection)")
    n_genes_before = adata.n_vars

    # Mark gene categories (once)
    if "mt" not in adata.var.columns:
        adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
        adata.var["ribo"] = adata.var_names.str.upper().str.match("^RP[SL]")
        adata.var["hb"] = adata.var_names.str.upper().str.contains("^HB[^(P)]")

    n_mt = adata.var["mt"].sum()
    n_ribo = adata.var["ribo"].sum()
    n_hb = adata.var["hb"].sum()

    # Start with all genes passing
    combined_mask = np.ones(adata.n_vars, dtype=bool)

    # Remove unwanted gene types first
    if remove_mito: combined_mask &= ~adata.var["mt"].values
    if remove_ribo: combined_mask &= ~adata.var["ribo"].values
    if remove_hb: combined_mask &= ~adata.var["hb"].values

    n_after_type_filter = combined_mask.sum()
    print(f"Genes before: {n_genes_before} (MT:{n_mt}, Ribo:{n_ribo}, Hb:{n_hb})")
    print(f"After removing MT/Ribo/Hb: {n_after_type_filter}")

    # For each layer, check expression threshold
    layer_masks = {}
    for layer_name in layers_to_analyze:
        if layer_name in adata.layers:
            X = adata.layers[layer_name]
        elif layer_name == "X":
            X = adata.X
        else:
            print(f"  Warning: Layer '{layer_name}' not found, skipping")
            continue

        try: X = X.tocsr()
        except: pass

        # Count cells expressing each gene in this layer
        n_cells_expressing = np.array((X > 0).sum(axis=0)).ravel()
        layer_mask = n_cells_expressing >= min_cells
        layer_masks[layer_name] = layer_mask

        n_pass = layer_mask.sum()
        print(f"  {layer_name}: {n_pass} genes with >= {min_cells} cells expressing")

        # Intersect with combined mask
        combined_mask &= layer_mask

    # Final count
    n_genes_after = combined_mask.sum()

    print(f"\nGenes passing in ALL layers: {n_genes_after}")
    print(f"Removed: {n_genes_before - n_genes_after} ({100*(n_genes_before - n_genes_after)/n_genes_before:.1f}%)")

    # Show layer-specific breakdown
    print("\nLayer-specific breakdown:")
    for layer_name, layer_mask in layer_masks.items():
        only_this = layer_mask & ~combined_mask
        print(f"  {layer_name}: {only_this.sum()} genes would pass only in this layer (excluded)")

    adata_filtered = adata[:, combined_mask].copy()

    return adata_filtered


def qc_multilayer(adata, sample_col, layers_to_analyze, outdir):
    """QC report comparing multiple layers."""
    safe_mkdir(outdir)

    qc_stats = []

    for layer_name in layers_to_analyze:
        print(f"Computing QC for layer: {layer_name}")

        if layer_name in adata.layers:
            X = adata.layers[layer_name]
        elif layer_name == "X":
            X = adata.X
        else:
            print(f"Warning: Layer '{layer_name}' not found, skipping")
            continue

        try: X = X.tocsr()
        except: pass

        total_counts = np.array(X.sum(axis=1)).ravel()
        total_genes = np.array((X > 0).sum(axis=1)).ravel()

        adata.obs[f"total_counts_{layer_name}"] = total_counts
        adata.obs[f"n_genes_{layer_name}"] = total_genes

        qc_stats.append({
            "layer": layer_name,
            "median_counts": np.median(total_counts),
            "mean_counts": np.mean(total_counts),
            "median_genes": np.median(total_genes),
            "mean_genes": np.mean(total_genes)
        })

    qc_df = pd.DataFrame(qc_stats)
    qc_df.to_csv(os.path.join(outdir, "layer_comparison.tsv"), sep="\t", index=False)

    # Multi-layer comparison plots
    with PdfPages(os.path.join(outdir, "multilayer_QC_report.pdf")) as pdf:
        # Counts comparison
        fig, axes = plt.subplots(1, len(layers_to_analyze), figsize=(5*len(layers_to_analyze), 4))
        if len(layers_to_analyze) == 1:
            axes = [axes]

        for i, layer_name in enumerate(layers_to_analyze):
            col_name = f"total_counts_{layer_name}"
            if col_name in adata.obs.columns:
                axes[i].hist(adata.obs[col_name], bins=50, edgecolor='black', alpha=0.7)
                axes[i].axvline(np.median(adata.obs[col_name]), color='red',
                               linestyle='--', label=f'Median: {np.median(adata.obs[col_name]):.0f}')
                axes[i].set_xlabel("Counts per cell")
                axes[i].set_ylabel("Number of cells")
                axes[i].set_title(f"{layer_name} counts")
                axes[i].legend()
        plt.tight_layout()
        pdf.savefig(); plt.close()

        # Genes comparison
        fig, axes = plt.subplots(1, len(layers_to_analyze), figsize=(5*len(layers_to_analyze), 4))
        if len(layers_to_analyze) == 1:
            axes = [axes]

        for i, layer_name in enumerate(layers_to_analyze):
            col_name = f"n_genes_{layer_name}"
            if col_name in adata.obs.columns:
                axes[i].hist(adata.obs[col_name], bins=50, edgecolor='black', alpha=0.7)
                axes[i].axvline(np.median(adata.obs[col_name]), color='red',
                               linestyle='--', label=f'Median: {np.median(adata.obs[col_name]):.0f}')
                axes[i].set_xlabel("Detected genes per cell")
                axes[i].set_ylabel("Number of cells")
                axes[i].set_title(f"{layer_name} genes")
                axes[i].legend()
        plt.tight_layout()
        pdf.savefig(); plt.close()

        # Scatter plots comparing layers
        if len(layers_to_analyze) >= 2:
            for i in range(len(layers_to_analyze)):
                for j in range(i+1, len(layers_to_analyze)):
                    layer1 = layers_to_analyze[i]
                    layer2 = layers_to_analyze[j]

                    col1 = f"total_counts_{layer1}"
                    col2 = f"total_counts_{layer2}"

                    if col1 in adata.obs.columns and col2 in adata.obs.columns:
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.scatter(adata.obs[col1], adata.obs[col2], s=1, alpha=0.3)
                        ax.set_xlabel(f"{layer1} counts (per cell)")
                        ax.set_ylabel(f"{layer2} counts (per cell)")
                        ax.set_title(f"{layer1} vs {layer2}")

                        corr = np.corrcoef(adata.obs[col1], adata.obs[col2])[0,1]
                        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
                        pdf.savefig(); plt.close()

    return adata, qc_df

def pseudobulk_layer(adata, sample_col, condition_col, layer_name):
    """Aggregate a specific layer to pseudobulk and extract all available covariates."""
    print(f"Pseudobulking layer: {layer_name}")

    if layer_name in adata.layers:
        X = adata.layers[layer_name]
    elif layer_name == "X":
        X = adata.X
    else:
        raise ValueError(f"Layer '{layer_name}' not found")

    try: X = X.tocsr()
    except: pass

    genes = adata.var_names
    samples = adata.obs[sample_col].astype(str).unique()

    # Hardcoded list of meaningful covariates to extract (if present in data)
    meaningful_covariates = ['batch', 'donor', 'patient', 'sex', 'age', 'pct_mito']

    # Check which ones are actually present in the data
    potential_covariates = [col for col in meaningful_covariates
                           if col in adata.obs.columns]

    cols, meta = [], []
    for s in samples:
        idx = np.where(adata.obs[sample_col].astype(str) == s)[0]
        summed = np.asarray(X[idx,:].sum(axis=0)).ravel()
        cols.append(pd.Series(summed, index=genes, name=s))

        sample_obs = adata.obs.iloc[idx]

        # Condition (should be unique per sample)
        condition_vals = sample_obs[condition_col].astype(str).unique()
        if len(condition_vals) > 1:
            print(f"WARNING: Sample {s} has multiple conditions: {condition_vals}. Using first.")
        condition_val = condition_vals[0]

        # Build metadata dictionary
        meta_dict = {
            "sample_id": s,
            condition_col: condition_val,
            "n_cells": len(idx),
            "layer": layer_name
        }

        # Extract all potential covariates
        for col in potential_covariates:
            if col in sample_obs.columns:
                vals = sample_obs[col]

                # For categorical/string: use mode (most frequent)
                if vals.dtype == 'object' or vals.dtype.name == 'category':
                    mode_val = vals.astype(str).mode()
                    meta_dict[col] = mode_val.iloc[0] if not mode_val.empty else None

                # For numeric: use mean for QC metrics, median for counts
                elif pd.api.types.is_numeric_dtype(vals):
                    if col in ['total_counts', 'n_genes_by_counts', 'n_genes']:
                        meta_dict[col] = vals.median()
                    else:
                        meta_dict[col] = vals.mean()

        meta.append(meta_dict)

    counts_df = pd.concat(cols, axis=1)
    meta_df = pd.DataFrame(meta).set_index("sample_id")

    print(f"  {counts_df.shape[0]} genes x {counts_df.shape[1]} samples")
    print(f"  Extracted covariates: {', '.join([c for c in meta_df.columns if c not in [condition_col, 'layer', 'n_cells']])}")

    return counts_df, meta_df


def diagnose_covariates(meta_df, condition_col, outdir):
    """Test which covariates are associated with condition (potential confounders)."""
    print_step("COVARIATE DIAGNOSTICS")

    safe_mkdir(outdir)

    # Exclude non-covariate columns
    exclude = [condition_col, 'layer', 'sample_id']
    potential_covariates = [col for col in meta_df.columns if col not in exclude]

    results = []

    print(f"Testing {len(potential_covariates)} potential covariates for association with {condition_col}...")

    for cov in potential_covariates:
        # Skip if all NaN
        if meta_df[cov].isna().all():
            continue

        result = {
            'covariate': cov,
            'type': 'numeric' if pd.api.types.is_numeric_dtype(meta_df[cov]) else 'categorical',
            'n_unique': meta_df[cov].nunique(),
            'n_missing': meta_df[cov].isna().sum()
        }

        # Numeric covariate
        if pd.api.types.is_numeric_dtype(meta_df[cov]):
            # Remove NaN for testing
            test_df = meta_df[[cov, condition_col]].dropna()

            if len(test_df) < 3:
                result['test'] = 'insufficient_data'
                result['p_value'] = np.nan
                result['recommendation'] = 'SKIP (too few samples)'
            else:
                # Check if there's any variance in the data
                if test_df[cov].nunique() == 1:
                    result['test'] = 'no_variance'
                    result['p_value'] = np.nan
                    result['recommendation'] = 'SKIP (no variance, all values identical)'
                else:
                    # Kruskal-Wallis test (non-parametric ANOVA)
                    from scipy.stats import kruskal
                    groups = [test_df[test_df[condition_col] == cond][cov].values
                             for cond in test_df[condition_col].unique()]
                    groups = [g for g in groups if len(g) > 0]

                    if len(groups) < 2:
                        result['test'] = 'kruskal_wallis'
                        result['p_value'] = np.nan
                        result['recommendation'] = 'SKIP (only one condition)'
                    else:
                        try:
                            stat, pval = kruskal(*groups)
                            result['test'] = 'kruskal_wallis'
                            result['statistic'] = stat
                            result['p_value'] = pval

                            # Recommendation
                            if pval < 0.01:
                                result['recommendation'] = 'INCLUDE (strong confounder, p<0.01)'
                            elif pval < 0.05:
                                result['recommendation'] = 'CONSIDER (moderate confounder, p<0.05)'
                            else:
                                result['recommendation'] = 'OPTIONAL (not confounded, p>0.05)'

                            # Calculate effect size (median difference)
                            medians = {cond: test_df[test_df[condition_col] == cond][cov].median()
                                      for cond in test_df[condition_col].unique()}
                            result['medians'] = str(medians)
                        except ValueError as e:
                            # Handle "All numbers are identical" error
                            result['test'] = 'kruskal_wallis_failed'
                            result['p_value'] = np.nan
                            result['recommendation'] = f'SKIP (test failed: {str(e)})'

        # Categorical covariate
        else:
            # Contingency table
            test_df = meta_df[[cov, condition_col]].dropna()

            if len(test_df) < 3:
                result['test'] = 'insufficient_data'
                result['p_value'] = np.nan
                result['recommendation'] = 'SKIP (too few samples)'
            else:
                from scipy.stats import chi2_contingency, fisher_exact
                contingency = pd.crosstab(test_df[cov], test_df[condition_col])

                result['contingency_table'] = contingency.to_dict()

                # Chi-square test (or Fisher's exact if small counts)
                if contingency.min().min() < 5 or contingency.shape[0] == 2:
                    # Fisher's exact for 2x2
                    if contingency.shape == (2, 2):
                        stat, pval = fisher_exact(contingency)
                        result['test'] = 'fisher_exact'
                        result['p_value'] = pval
                    else:
                        # Chi-square for larger tables
                        stat, pval, dof, expected = chi2_contingency(contingency)
                        result['test'] = 'chi_square'
                        result['statistic'] = stat
                        result['p_value'] = pval
                else:
                    stat, pval, dof, expected = chi2_contingency(contingency)
                    result['test'] = 'chi_square'
                    result['statistic'] = stat
                    result['p_value'] = pval

                # Recommendation
                if pval < 0.01:
                    result['recommendation'] = 'INCLUDE (strong confounder, p<0.01)'
                elif pval < 0.05:
                    result['recommendation'] = 'CONSIDER (moderate confounder, p<0.05)'
                else:
                    result['recommendation'] = 'OPTIONAL (not confounded, p>0.05)'

        results.append(result)

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Sort by p-value
    results_df = results_df.sort_values('p_value', na_position='last')

    # Save results
    results_file = os.path.join(outdir, "covariate_diagnostics.tsv")
    results_df.to_csv(results_file, sep="\t", index=False)

    # Print summary
    print("\n" + "="*70)
    print("COVARIATE DIAGNOSTICS SUMMARY")
    print("="*70)

    for _, row in results_df.iterrows():
        print(f"\n{row['covariate']} ({row['type']}):")
        print(f"  Test: {row['test']}")
        if not pd.isna(row['p_value']):
            print(f"  P-value: {row['p_value']:.4f}")
        print(f"  Recommendation: {row['recommendation']}")

    print(f"\n{'='*70}")
    print(f"Full results saved to: {results_file}")
    print(f"{'='*70}\n")

    # Suggested covariates
    strong_confounders = results_df[(results_df['p_value'] < 0.05) & results_df['recommendation'].str.contains('INCLUDE|CONSIDER')]

    if len(strong_confounders) > 0:
        print("SUGGESTED COVARIATES TO INCLUDE:")
        print("  --covariates " + ",".join(strong_confounders['covariate'].tolist()))
        print()
    else:
        print("No strong confounders detected. You may run without covariates or include")
        print("n_cells as a technical covariate: --covariates n_cells\n")

    return results_df

def find_rscript():
    """Find Rscript executable on Windows or Unix."""
    # Try to find Rscript in PATH first
    rscript = shutil.which("Rscript")
    if rscript:
        return rscript

    # Windows: try common R installation locations
    if sys.platform == "win32":
        possible_paths = [
            r"C:\Program Files\R\R-*\bin\Rscript.exe",
            r"C:\Program Files\R\R-*\bin\x64\Rscript.exe",
            r"C:\R\R-*\bin\Rscript.exe",
            r"C:\R\R-*\bin\x64\Rscript.exe",
        ]

        for pattern in possible_paths:
            matches = glob.glob(pattern)
            if matches:
                # Get the latest version
                latest = sorted(matches)[-1]
                return latest

    return None

def find_r_script_file(script_name="3.run_limma_voom_dea.R"):
    """Find the R script file in common locations."""
    # Check current directory
    if os.path.exists(script_name):
        return os.path.abspath(script_name)

    # Check script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, script_name)
    if os.path.exists(script_path):
        return script_path

    # Check parent directory
    parent_dir = os.path.dirname(script_dir)
    script_path = os.path.join(parent_dir, script_name)
    if os.path.exists(script_path):
        return script_path

    return None

def run_multilayer_dea(layer_results, outdir, condition_col, covariates=None, fdr=0.05, logfc=1, r_script_path=None, multilayer_gene_filter=False):
    """Run DEA for multiple layers using external R script."""
    print_step("RUNNING MULTI-LAYER DEA IN R")

    # Find Rscript
    rscript_exe = find_rscript()
    if not rscript_exe:
        print("\nERROR: Rscript not found!")
        print("\nPlease ensure R is installed. Options:")
        print("1. Add R to your PATH, or")
        print("2. Run the R script manually (see instructions below)")
        print("\nManual R script commands:")
        for layer_name in layer_results.keys():
            counts_file = layer_results[layer_name]['counts_file']
            meta_file = layer_results[layer_name]['meta_file']
            layer_outdir = os.path.join(outdir, layer_name)
            cov_str = ",".join(covariates) if covariates else ""
            print(f'\nRscript 3.run_limma_voom_dea.R \\')
            print(f'  --counts "{counts_file}" \\')
            print(f'  --meta "{meta_file}" \\')
            print(f'  --outdir "{layer_outdir}" \\')
            print(f'  --condition {condition_col} \\')
            if cov_str:
                print(f'  --covariates {cov_str} \\')
            print(f'  --fdr {fdr} \\')
            print(f'  --logfc {logfc} \\')
            print(f'  --layer-name {layer_name}')
        raise FileNotFoundError("Rscript executable not found")

    print(f"Using Rscript: {rscript_exe}")

    # Validate R script file
    if r_script_path is None:
        print("\nERROR: R script path not provided!")
        print("Please specify the R script path using --r-script argument")
        raise ValueError("R script path is required")

    if not os.path.exists(r_script_path):
        print(f"\nERROR: R script not found at: {r_script_path}")
        raise FileNotFoundError(f"R script not found: {r_script_path}")

    print(f"Using R script: {r_script_path}")

    # Run DEA for each layer
    for layer_name, layer_info in layer_results.items():
        print(f"\n{'='*70}")
        print(f"Processing layer: {layer_name}")
        print(f"{'='*70}")

        counts_file = layer_info['counts_file']
        meta_file = layer_info['meta_file']

        # Create layer-specific subdirectory
        layer_outdir = os.path.join(outdir, layer_name)
        safe_mkdir(layer_outdir)
        print(f"Results will be saved to: {layer_outdir}")

        # Build R command
        cmd = [
            rscript_exe,
            r_script_path,
            "--counts", counts_file,
            "--meta", meta_file,
            "--outdir", layer_outdir,
            "--condition", condition_col,
            "--fdr", str(fdr),
            "--logfc", str(logfc),
            "--layer-name", layer_name
        ]

        if covariates:
            cmd.extend(["--covariates", ",".join(covariates)])

        # Add other layer counts for multi-layer filterByExpr (Lee et al. 2020 approach)
        if multilayer_gene_filter and len(layer_results) > 1:
            other_counts_files = [
                info['counts_file'] for name, info in layer_results.items()
                if name != layer_name
            ]
            if other_counts_files:
                cmd.extend(["--counts-other", ",".join(other_counts_files)])

        # Run R script
        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
            if result.stderr:
                print("R messages:", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: R script failed for layer {layer_name}!")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            print("\nContinuing with other layers...")

    # Compare layers if multiple
    if len(layer_results) >= 2:
        print(f"\n{'='*70}")
        print("Comparing layers...")
        print(f"{'='*70}")
        compare_layers(outdir, list(layer_results.keys()), fdr, logfc)

def compare_layers(outdir, layer_names, fdr, logfc):
    """Compare DEA results across multiple layers."""
    # Create comparisons subdirectory
    comp_dir = os.path.join(outdir, "comparisons")
    safe_mkdir(comp_dir)

    # Load results for each layer
    results = {}
    for layer in layer_names:
        # Look for results in layer-specific subdirectory
        results_file = os.path.join(outdir, layer, f"dea_{layer}_results.tsv")
        if os.path.exists(results_file):
            results[layer] = pd.read_csv(results_file, sep="\t", index_col=0)
            print(f"Loaded: {layer} ({len(results[layer])} genes)")
        else:
            print(f"WARNING: Results file not found for layer {layer} at {results_file}")

    if len(results) < 2:
        print("Not enough layers to compare")
        return

    # Find common genes
    gene_sets = [set(df.index) for df in results.values()]
    common_genes = set.intersection(*gene_sets)
    print(f"Common genes across layers: {len(common_genes)}")

    # Build comparison dataframe
    comp_data = {"gene": sorted(common_genes)}

    for layer in layer_names:
        if layer in results:
            df = results[layer]
            comp_data[f"logFC_{layer}"] = [df.loc[g, "logFC"] for g in sorted(common_genes)]
            comp_data[f"adj.P.Val_{layer}"] = [df.loc[g, "adj.P.Val"] for g in sorted(common_genes)]
            comp_data[f"sig_{layer}"] = [(df.loc[g, "adj.P.Val"] < fdr) and (abs(df.loc[g, "logFC"]) > logfc) for g in sorted(common_genes)]

    comp_df = pd.DataFrame(comp_data)
    comp_df.to_csv(os.path.join(comp_dir, "layer_comparison_results.tsv"), sep="\t", index=False)
    print(f"Saved: comparisons/layer_comparison_results.tsv")

    # Summary table
    summary_data = []
    for layer in layer_names:
        if layer in results:
            df = results[layer]
            sig = (df["adj.P.Val"] < fdr) & (df["logFC"].abs() > logfc)
            summary_data.append({
                "layer": layer,
                "n_genes_tested": len(df),
                "n_significant": sig.sum(),
                "n_up": ((df["logFC"] > 0) & sig).sum(),
                "n_down": ((df["logFC"] < 0) & sig).sum()
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(comp_dir, "layer_summary.tsv"), sep="\t", index=False)
    print("\nLayer summary:")
    print(summary_df.to_string(index=False))

    # Pairwise scatter plots
    for i in range(len(layer_names)):
        for j in range(i+1, len(layer_names)):
            layer1 = layer_names[i]
            layer2 = layer_names[j]

            if layer1 not in results or layer2 not in results:
                continue

            lfc1 = f"logFC_{layer1}"
            lfc2 = f"logFC_{layer2}"
            sig1 = f"sig_{layer1}"
            sig2 = f"sig_{layer2}"

            # Calculate correlation
            corr = comp_df[[lfc1, lfc2]].corr().iloc[0, 1]
            print(f"{layer1} vs {layer2} correlation: {corr:.4f}")

            # Scatter plot
            fig, ax = plt.subplots(figsize=(10, 9))

            # Categorize points
            comp_df['category'] = 'NS'
            comp_df.loc[comp_df[sig1] & comp_df[sig2], 'category'] = 'Both sig'
            comp_df.loc[comp_df[sig1] & ~comp_df[sig2], 'category'] = f'Only {layer1}'
            comp_df.loc[~comp_df[sig1] & comp_df[sig2], 'category'] = f'Only {layer2}'

            colors = {'NS': 'grey', 'Both sig': 'red', f'Only {layer1}': 'blue', f'Only {layer2}': 'green'}

            for cat, color in colors.items():
                subset = comp_df[comp_df['category'] == cat]
                ax.scatter(subset[lfc1], subset[lfc2], c=color, label=cat, s=20, alpha=0.5)

            ax.axline((0, 0), slope=1, color='black', linestyle='--', alpha=0.5)
            ax.axhline(0, color='grey', linestyle=':', alpha=0.3)
            ax.axvline(0, color='grey', linestyle=':', alpha=0.3)
            ax.set_xlabel(f'{layer1} log2FC')
            ax.set_ylabel(f'{layer2} log2FC')
            ax.set_title(f'{layer1} vs {layer2} (r={corr:.3f})')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(comp_dir, f"compare_{layer1}_vs_{layer2}.png"), dpi=100)
            plt.close()
            print(f"Saved: comparisons/compare_{layer1}_vs_{layer2}.png")

    # Special analysis: Intron vs Exon logFC comparison (regulatory mechanisms)
    if 'intron' in results and 'exon' in results:
        print(f"\n{'='*70}")
        print("SPECIAL ANALYSIS: Intron vs Exon Regulatory Mechanisms")
        print(f"{'='*70}")

        analyze_intron_exon_regulation(comp_df, results, comp_dir, fdr, logfc)

    print("\nLayer comparison complete!")


def analyze_intron_exon_regulation(comp_df, results, outdir, fdr, logfc):
    """
    Analyze regulatory mechanisms by comparing intron vs exon logFC.

    Classifies genes into regulatory categories:
    - Transcriptional: Both intron and exon change in same direction
    - Splicing defect: Intron increases, exon decreases
    - Post-transcriptional: Exon changes more than intron
    """

    intron_df = results['intron']
    exon_df = results['exon']

    # Get common genes
    common_genes = comp_df['gene'].tolist()

    # Calculate logFC difference and classify genes
    categories = []
    for gene in common_genes:
        intron_lfc = comp_df.loc[comp_df['gene'] == gene, 'logFC_intron'].values[0]
        exon_lfc = comp_df.loc[comp_df['gene'] == gene, 'logFC_exon'].values[0]
        intron_sig = comp_df.loc[comp_df['gene'] == gene, 'sig_intron'].values[0]
        exon_sig = comp_df.loc[comp_df['gene'] == gene, 'sig_exon'].values[0]

        # Classify based on significance and direction
        if not intron_sig and not exon_sig:
            category = "Not significant"
        elif intron_sig and not exon_sig:
            category = "Intron only"
        elif exon_sig and not intron_sig:
            category = "Exon only"
        elif intron_sig and exon_sig:
            # Both significant - look at direction and magnitude
            if (intron_lfc > 0 and exon_lfc > 0) or (intron_lfc < 0 and exon_lfc < 0):
                # Same direction
                if abs(intron_lfc - exon_lfc) < 0.5:
                    category = "Transcriptional (both change)"
                elif abs(intron_lfc) > abs(exon_lfc) * 1.5:
                    category = "Intron retention (intron >> exon)"
                else:
                    category = "Post-transcriptional (exon >> intron)"
            else:
                # Opposite directions
                category = "Splicing defect (opposite directions)"
        else:
            category = "Other"

        categories.append(category)

    comp_df['regulatory_category'] = categories

    # Save categorized results
    comp_df.to_csv(os.path.join(outdir, "intron_exon_regulatory_classification.tsv"),
                   sep="\t", index=False)
    print(f"\nSaved: intron_exon_regulatory_classification.tsv")

    # Count categories
    category_counts = comp_df['regulatory_category'].value_counts()
    print("\nRegulatory mechanism classification:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count} genes ({100*count/len(comp_df):.1f}%)")

    # Create detailed scatter plot with categories
    fig, ax = plt.subplots(figsize=(12, 10))

    # Define colors for categories
    cat_colors = {
        "Not significant": "lightgray",
        "Intron only": "blue",
        "Exon only": "green",
        "Transcriptional (both change)": "purple",
        "Intron retention (intron >> exon)": "red",
        "Post-transcriptional (exon >> intron)": "orange",
        "Splicing defect (opposite directions)": "darkred",
        "Other": "gray"
    }

    for category in comp_df['regulatory_category'].unique():
        subset = comp_df[comp_df['regulatory_category'] == category]
        ax.scatter(subset['logFC_intron'], subset['logFC_exon'],
                  c=cat_colors.get(category, 'gray'),
                  label=f"{category} (n={len(subset)})",
                  s=30, alpha=0.6)

    # Add reference lines
    ax.axline((0, 0), slope=1, color='black', linestyle='--', linewidth=2,
             alpha=0.5, label='Perfect correlation')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.3)

    # Calculate correlation
    corr = comp_df[['logFC_intron', 'logFC_exon']].corr().iloc[0, 1]

    ax.set_xlabel('Intron log2FC (ALS/Control)', fontsize=12)
    ax.set_ylabel('Exon log2FC (ALS/Control)', fontsize=12)
    ax.set_title(f'Regulatory Mechanisms: Intron vs Exon Expression Changes\n(r={corr:.3f})',
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "intron_vs_exon_regulatory_mechanisms.png"),
               dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: intron_vs_exon_regulatory_mechanisms.png")

    # Export top genes in each category
    sig_categories = [c for c in category_counts.index if c != "Not significant"]
    for category in sig_categories:
        cat_genes = comp_df[comp_df['regulatory_category'] == category].copy()
        cat_genes = cat_genes.sort_values('logFC_intron', key=abs, ascending=False)

        # Save top 50 genes in this category
        top_n = min(50, len(cat_genes))
        if top_n > 0:
            cat_file = os.path.join(outdir, f"regulatory_category_{category.replace(' ', '_').replace('(', '').replace(')', '')}_top{top_n}.tsv")
            cat_genes.head(top_n).to_csv(cat_file, sep="\t", index=False)
            print(f"  Saved top {top_n} genes in '{category}': {cat_file}")

    print(f"\n{'='*70}")
    print("Regulatory mechanism analysis complete!")
    print(f"{'='*70}")


def main(args):
    print_step("MULTI-LAYER PSEUDOBULK DEA PIPELINE")
    print(f"Input: {args.h5ad}")
    print(f"Layers to analyze: {args.layers}")

    safe_mkdir(args.outdir)

    # Parse layers
    layers_to_analyze = [l.strip() for l in args.layers.split(',')]

    # Load data
    print_step("LOADING DATA")
    adata = sc.read_h5ad(args.h5ad)
    print(f"Loaded: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"Available layers: {list(adata.layers.keys())}")

    # Validate layers
    for layer in layers_to_analyze:
        if layer != "X" and layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found. Available: {list(adata.layers.keys())}")

    # Multi-layer QC before filtering
    adata, qc_stats = qc_multilayer(adata, args.sample_col, layers_to_analyze,
                                     os.path.join(args.outdir, "qc_before_filtering"))

    # Use primary layer for cell/gene filtering (typically the first one, e.g., total counts)
    primary_layer = layers_to_analyze[0]
    print_step(f"FILTERING BASED ON PRIMARY LAYER: {primary_layer}")

    if not args.skip_cell_filter:
        adata = filter_cells(adata, primary_layer, min_counts=args.min_counts,
                           max_mito=args.max_mito, min_genes=args.min_genes)

    if not args.skip_gene_filter:
        if args.multilayer_gene_filter:
            # Filter based on ALL layers (intersection) - Lee et al. 2020 approach
            adata = filter_genes_multilayer(adata, layers_to_analyze, min_cells=args.min_cells_per_gene)
        else:
            # Filter based on primary layer only (default)
            adata = filter_genes(adata, primary_layer, min_cells=args.min_cells_per_gene)

    # Sample validation
    print_step("SAMPLE VALIDATION")
    cells_per_sample = adata.obs.groupby(args.sample_col).size()
    low_samples = cells_per_sample[cells_per_sample < args.min_cells_per_sample]
    if len(low_samples) > 0:
        print(f"Removing {len(low_samples)} samples with < {args.min_cells_per_sample} cells")
        adata = adata[~adata.obs[args.sample_col].isin(low_samples.index), :].copy()

    # QC after filtering
    adata, qc_stats_post = qc_multilayer(adata, args.sample_col, layers_to_analyze,
                                          os.path.join(args.outdir, "qc_after_filtering"))

    # Pseudobulk each layer
    print_step("PSEUDOBULK AGGREGATION")
    layer_results = {}

    for layer_name in layers_to_analyze:
        counts_df, meta_df = pseudobulk_layer(adata, args.sample_col, args.condition_col, layer_name)

        # Save
        counts_file = os.path.join(args.outdir, f"pseudobulk_{layer_name}_counts.tsv")
        meta_file = os.path.join(args.outdir, f"pseudobulk_{layer_name}_meta.tsv")
        counts_df.to_csv(counts_file, sep="\t")
        meta_df.to_csv(meta_file, sep="\t")

        layer_results[layer_name] = {
            "counts_file": counts_file,
            "meta_file": meta_file,
            "counts_df": counts_df,
            "meta_df": meta_df
        }

        print(f"Saved: {counts_file}")

    # Diagnostic: Test which covariates are confounded with condition
    # Use metadata from first layer (all layers have same sample-level metadata)
    first_layer = list(layer_results.keys())[0]
    meta_for_diagnostics = layer_results[first_layer]["meta_df"]
    diagnose_covariates(meta_for_diagnostics, args.condition_col,
                       os.path.join(args.outdir, "covariate_diagnostics"))

    # Run DEA for all layers
    covariates = args.covariates.split(',') if args.covariates else None
    run_multilayer_dea(layer_results, args.outdir, args.condition_col,
                      covariates=covariates, fdr=args.fdr, logfc=args.logfc,
                      r_script_path=args.r_script,
                      multilayer_gene_filter=args.multilayer_gene_filter)

    print_step("ALL DONE")
    print(f"\nResults in: {args.outdir}")
    print(f"\nKey outputs:")

    print(f"\n1. Layer-specific DEA Results:")
    for layer in layers_to_analyze:
        print(f"  {layer}/")
        print(f"    - dea_{layer}_results.tsv")
        print(f"    - dea_{layer}_significant.tsv")
        print(f"    - dea_{layer}_volcano.png")

    print(f"\n2. Layer Comparisons:")
    print(f"  comparisons/")
    print(f"    - layer_comparison_results.tsv")
    print(f"    - layer_summary.tsv")
    print(f"    - compare_*.png (pairwise layer comparisons)")

    print(f"\n3. Regulatory Mechanism Analysis:")
    if 'intron' in layers_to_analyze and 'exon' in layers_to_analyze:
        print(f"  comparisons/")
        print(f"    - intron_exon_regulatory_classification.tsv")
        print(f"    - intron_vs_exon_regulatory_mechanisms.png")
        print(f"    - regulatory_category_*_top50.tsv (genes by mechanism)")

    print(f"\n4. Diagnostic Outputs:")
    print(f"  covariate_diagnostics/")
    print(f"    - covariate_diagnostics.tsv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Multi-layer pseudobulk DEA pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required
    ap.add_argument("--h5ad", required=True, help="Input h5ad file")
    ap.add_argument("--sample-col", required=True, help="Sample ID column")
    ap.add_argument("--condition-col", required=True, help="Condition column")
    ap.add_argument("--layers", required=True, help="Comma-separated layer names (e.g., 'intron,exon,counts')")

    # Output
    ap.add_argument("--outdir", default="./multilayer_dea", help="Output directory")

    # Filtering (applied to primary layer only)
    ap.add_argument("--min-counts", type=int, default=500, help="Min counts per cell")
    ap.add_argument("--max-mito", type=float, default=20, help="Max mito %%")
    ap.add_argument("--min-genes", type=int, default=200, help="Min genes per cell")
    ap.add_argument("--min-cells-per-gene", type=int, default=10, help="Min cells per gene")
    ap.add_argument("--min-cells-per-sample", type=int, default=10, help="Min cells per sample")
    ap.add_argument("--skip-cell-filter", action="store_true", help="Skip cell filtering")
    ap.add_argument("--skip-gene-filter", action="store_true", help="Skip gene filtering")
    ap.add_argument("--multilayer-gene-filter", action="store_true",
                    help="Filter genes based on ALL layers (intersection). Follows Lee et al. 2020 approach.")

    # DEA
    ap.add_argument("--covariates", type=str, default=None, help="Comma-separated covariates")
    ap.add_argument("--fdr", type=float, default=0.05, help="FDR threshold")
    ap.add_argument("--logfc", type=float, default=1, help="Log2FC threshold")
    ap.add_argument("--r-script", type=str, required=True, help="Path to the R script for DEA (e.g., 3.run_limma_voom_dea.R)")

    args = ap.parse_args()
    main(args)
