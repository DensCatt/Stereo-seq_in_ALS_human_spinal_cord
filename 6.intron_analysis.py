#!/usr/bin/env python3
"""
Comprehensive Motor Neuron Transcriptome Integrity Analysis
============================================================

Analyzes intron retention in motor neurons as a biomarker for RNA processing
defects in ALS vs control samples using spatial transcriptomics data.

Input: h5ad file with motor neurons containing layers: counts, exon, intron
Output: Genome-wide statistical analysis + in-depth analysis of candidate genes

Authors: Combined best practices from multiple analysis scripts
Date: 2024
"""

#%%
# =============================================================================
# IMPORTS
# =============================================================================
import os
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import mannwhitneyu, shapiro, ttest_ind, spearmanr
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# For advanced statistical methods
try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available. Mixed-effects models will be skipped.")

#%%
# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration parameters for the analysis."""

    # Input/Output
    INPUT_H5AD = "C:\\Users\\catta\\Desktop\\ALS\\THESIS\\motorneurons_bin20_ALL_LAYERS_processed.h5ad"
    OUTPUT_DIR = "C:\\Users\\catta\\Desktop\\ALS\\THESIS\\Transcriptome_integrity_general"

    # Sample metadata columns
    SAMPLE_COL = "sample"
    CONDITION_COL = "condition"

    # Gene filtering thresholds
    # Option 1: Absolute thresholds
    MIN_CELLS_PER_GENE_PER_SAMPLE = 5  # Gene must be in ≥5 cells per sample
    MIN_SAMPLES_PER_GROUP = 3          # Gene must be in ≥3 samples per condition
    MIN_CELLS_PER_GROUP = 20  # Gene must have ≥20 total cells per condition

    # Option 2: Relative thresholds (fraction-based, adapts to dataset size)
    MIN_CELLS_FRACTION_PER_SAMPLE = 0.1  # Gene must be in ≥1% of cells per sample
    MIN_CELLS_FRACTION_PER_GROUP = 0.1 # Gene must be in ≥0.5% of cells per condition

    # Toggle: Use relative filtering? (True = fractions, False = absolute numbers)
    USE_RELATIVE_FILTERING = False

    # Intron retention threshold
    RETENTION_THRESHOLD = 0.1  # IntronFraction > this value = "has retention"

    # Statistical testing
    FDR_THRESHOLD = 0.05
    EFFECT_SIZE_THRESHOLD = 0.05  # Minimum difference in retention proportion (5%)
    N_PERMUTATIONS = 5000

    # Candidate genes for in-depth analysis
    CANDIDATE_GENES = ["STMN2", "UNC13A", "MEG8", "SNAP25", "TUBA1B", 
                       "EIF4H", "HSPA8", "NEFL", "CALM1", "UNC5C", "STMN1"]

    # Visualization
    FIGURE_DPI = 300
    MAX_GENES_TO_PLOT = 50


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_output_dirs(base_dir):
    """Create output directory structure."""
    dirs = {
        'base': base_dir,
        'figures': os.path.join(base_dir, 'figures'),
        'tables': os.path.join(base_dir, 'tables'),
        'qc': os.path.join(base_dir, 'qc'),
        'candidate_genes': os.path.join(base_dir, 'candidate_genes')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def detect_condition_labels(conditions):
    """
    Auto-detect control and ALS condition labels from a list of conditions.

    Uses str(c).strip().upper() for safe conversion (handles non-strings and whitespace).

    Parameters
    ----------
    conditions : array-like
        Unique condition values from the data

    Returns
    -------
    control_label, als_label : tuple of str
    """
    conditions = list(conditions)

    # Try to match control label (using str().strip().upper() for robustness)
    control_matches = ['CONTROL', 'CTR', 'CTRL', 'CON', 'HEALTHY', 'NORMAL', 'WT', 'WILDTYPE']
    control_label = None
    for c in conditions:
        if str(c).strip().upper() in control_matches:
            control_label = c
            break

    # Try to match ALS label
    als_matches = ['ALS', 'DISEASE', 'PATIENT', 'CASE']
    als_label = None
    for c in conditions:
        if str(c).strip().upper() in als_matches:
            als_label = c
            break

    # Fallback: if exactly 2 conditions and one identified
    if len(conditions) == 2:
        if als_label and not control_label:
            control_label = [c for c in conditions if c != als_label][0]
        elif control_label and not als_label:
            als_label = [c for c in conditions if c != control_label][0]

    # Final check with informative error
    if control_label is None or als_label is None:
        raise ValueError(
            f"Could not auto-detect condition labels from: {conditions}\n"
            f"Detected: Control='{control_label}', ALS='{als_label}'\n"
            f"Expected control variations: {control_matches}\n"
            f"Expected ALS variations: {als_matches}\n"
            f"Please check your data or update detect_condition_labels()"
        )

    return control_label, als_label


#%%
# =============================================================================
# DATA LOADING AND METRIC CALCULATION
# =============================================================================

def prefilter_genes_fast(adata, sample_col, condition_col,
                         min_cells_per_sample, min_samples_per_group):
    """
    Fast pre-filtering of genes using sparse matrix operations.

    This is much faster than iterating through all genes because it uses
    vectorized sparse matrix operations.

    Returns
    -------
    valid_genes : list
        List of gene names that pass the filtering criteria
    filter_stats : dict
        Statistics about the filtering
    """
    print("\n" + "-"*60)
    print("PRE-FILTERING GENES (fast sparse operations)")
    print("-"*60)

    counts_matrix = adata.layers['counts']
    n_genes_initial = adata.n_vars
    print(f"Initial genes: {n_genes_initial:,}")

    # Get unique samples and conditions
    samples = adata.obs[sample_col].unique()
    conditions = adata.obs[condition_col].unique()

    # For each gene, count cells per sample (vectorized)
    print("Counting cells per gene per sample...")

    gene_sample_counts = {}
    for sample in samples:
        sample_mask = (adata.obs[sample_col] == sample).values
        sample_counts = counts_matrix[sample_mask, :]

        # Count non-zero cells per gene for this sample
        if sparse.issparse(sample_counts):
            cells_per_gene = np.array((sample_counts > 0).sum(axis=0)).ravel()
        else:
            cells_per_gene = np.array((sample_counts > 0).sum(axis=0)).ravel()

        gene_sample_counts[sample] = cells_per_gene

    # Convert to DataFrame for easier filtering
    gene_sample_df = pd.DataFrame(gene_sample_counts, index=adata.var_names)

    # Filter 1: Gene must have >= min_cells_per_sample in at least one sample
    genes_with_enough_cells = (gene_sample_df >= min_cells_per_sample).any(axis=1)
    n_after_f1 = genes_with_enough_cells.sum()
    print(f"After min {min_cells_per_sample} cells/sample filter: {n_after_f1:,} genes")

    # Filter 2: Gene must be present in >= min_samples_per_group samples per condition
    # Build condition -> samples mapping
    condition_samples = {}
    for cond in conditions:
        cond_samples = adata.obs[adata.obs[condition_col] == cond][sample_col].unique()
        condition_samples[cond] = list(cond_samples)

    valid_genes = []
    for gene in adata.var_names[genes_with_enough_cells]:
        gene_counts = gene_sample_df.loc[gene]

        # Check each condition
        passes_all_conditions = True
        for cond, cond_sample_list in condition_samples.items():
            # Count samples where gene has >= min_cells_per_sample
            n_valid_samples = (gene_counts[cond_sample_list] >= min_cells_per_sample).sum()
            if n_valid_samples < min_samples_per_group:
                passes_all_conditions = False
                break

        if passes_all_conditions:
            valid_genes.append(gene)

    n_final = len(valid_genes)
    print(f"After min {min_samples_per_group} samples/condition filter: {n_final:,} genes")
    print(f"\n✓ Pre-filtering complete: {n_final:,} genes will be processed")
    print(f"  (Skipping {n_genes_initial - n_final:,} genes - {100*(n_genes_initial - n_final)/n_genes_initial:.1f}% reduction)")
    print("-"*60)

    filter_stats = {
        'n_initial': n_genes_initial,
        'n_after_cells_filter': n_after_f1,
        'n_final': n_final,
        'pct_reduction': 100 * (n_genes_initial - n_final) / n_genes_initial
    }

    return valid_genes, filter_stats


def load_and_prepare_data(h5ad_path, sample_col="sample", condition_col="condition",
                          min_cells_per_sample=None, min_samples_per_group=None):
    """
    Load h5ad file and calculate intron/exon metrics.

    Now includes fast pre-filtering to only process genes that pass thresholds.

    Parameters
    ----------
    h5ad_path : str
        Path to h5ad file with layers: counts, exon, intron
    sample_col : str
        Column name for sample IDs
    condition_col : str
        Column name for condition (e.g., ALS, Control)
    min_cells_per_sample : int
        Minimum cells per gene per sample for pre-filtering
    min_samples_per_group : int
        Minimum samples per condition for pre-filtering

    Returns
    -------
    adata : AnnData
        Annotated data with calculated metrics
    df : DataFrame
        Long-form DataFrame with metrics (only for filtered genes)
    prefilter_stats : dict
        Pre-filtering statistics
    """
    print_section("LOADING DATA")

    # Load data
    print(f"Loading: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"Available layers: {list(adata.layers.keys())}")

    # Verify required layers exist
    required_layers = ['counts', 'exon', 'intron']
    missing_layers = [l for l in required_layers if l not in adata.layers]
    if missing_layers:
        raise ValueError(f"Missing required layers: {missing_layers}")

    # Verify required columns exist
    required_cols = [sample_col, condition_col]
    missing_cols = [c for c in required_cols if c not in adata.obs.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in obs: {missing_cols}")

    print(f"\nSamples: {adata.obs[sample_col].nunique()}")
    print(f"Conditions: {adata.obs[condition_col].unique()}")
    print(f"Cells per condition:")
    print(adata.obs[condition_col].value_counts())

    # Use Config defaults if not specified
    if min_cells_per_sample is None:
        min_cells_per_sample = Config.MIN_CELLS_PER_GENE_PER_SAMPLE
    if min_samples_per_group is None:
        min_samples_per_group = Config.MIN_SAMPLES_PER_GROUP

    # === FAST PRE-FILTERING ===
    valid_genes, prefilter_stats = prefilter_genes_fast(
        adata, sample_col, condition_col,
        min_cells_per_sample=min_cells_per_sample,
        min_samples_per_group=min_samples_per_group
    )

    if len(valid_genes) == 0:
        print("\nERROR: No genes passed pre-filtering!")
        return adata, pd.DataFrame(), prefilter_stats

    # === CALCULATE METRICS ONLY FOR FILTERED GENES ===
    print_section("CALCULATING METRICS (filtered genes only)")
    print(f"Processing {len(valid_genes):,} genes...")

    # Get indices of valid genes
    gene_to_idx = {gene: i for i, gene in enumerate(adata.var_names)}
    valid_gene_indices = [gene_to_idx[g] for g in valid_genes]

    # Build long-form DataFrame with all cell-level data
    obs_data = adata.obs[[sample_col, condition_col]].copy()
    obs_data['cell_id'] = obs_data.index

    # For each FILTERED gene, extract counts, exon, intron
    records = []

    for idx, gene in enumerate(valid_genes):
        if (idx + 1) % 500 == 0:
            print(f"  Processing gene {idx+1}/{len(valid_genes)}")

        gene_idx = valid_gene_indices[idx]

        # Extract counts for this gene across all cells
        counts_vec = adata.layers['counts'][:, gene_idx]
        exon_vec = adata.layers['exon'][:, gene_idx]
        intron_vec = adata.layers['intron'][:, gene_idx]

        # Convert to dense if sparse
        if sparse.issparse(counts_vec):
            counts_vec = counts_vec.toarray().ravel()
        if sparse.issparse(exon_vec):
            exon_vec = exon_vec.toarray().ravel()
        if sparse.issparse(intron_vec):
            intron_vec = intron_vec.toarray().ravel()

        # Create records for cells with non-zero counts
        for cell_idx in range(len(counts_vec)):
            if counts_vec[cell_idx] > 0:  # Only include expressed genes
                records.append({
                    'cell_id': obs_data.index[cell_idx],
                    'geneName': gene,
                    'sample': obs_data.iloc[cell_idx][sample_col],
                    'Condition': obs_data.iloc[cell_idx][condition_col],
                    'MIDCount': counts_vec[cell_idx],
                    'ExonCount': exon_vec[cell_idx],
                    'IntronCount': intron_vec[cell_idx]
                })

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Calculate fractions and ratios
    print("\nCalculating fractions and ratios...")
    df['ExonFraction'] = df['ExonCount'] / (df['MIDCount'] + 1e-9)
    df['IntronFraction'] = df['IntronCount'] / (df['MIDCount'] + 1e-9)

    # Handle ratios with potential division by zero
    df['ExonIntronRatio'] = np.where(
        df['IntronCount'] > 0,
        df['ExonCount'] / df['IntronCount'],
        np.nan
    )
    df['IntronExonRatio'] = np.where(
        df['ExonCount'] > 0,
        df['IntronCount'] / df['ExonCount'],
        np.nan
    )

    # Log2 intron-exon ratio with pseudocount to handle zeros
    # Pseudocount of 1 is standard in transcriptomics (similar to DESeq2/edgeR)
    pseudocount = 1
    df['Log2IntronExonRatio'] = np.log2(
        (df['IntronCount'] + pseudocount) / (df['ExonCount'] + pseudocount)
    )

    print(f"\nGenerated long-form DataFrame:")
    print(f"  Total records: {len(df):,}")
    print(f"  Unique genes: {df['geneName'].nunique():,}")
    print(f"  Unique cells: {df['cell_id'].nunique():,}")
    print(f"  Samples: {df['sample'].unique()}")

    print("\nMetric summary (all data):")
    print(df[['MIDCount', 'ExonCount', 'IntronCount', 'IntronFraction', 'Log2IntronExonRatio']].describe())

    # Condition-specific summary
    print("\nMetric summary by condition:")
    for cond in df['Condition'].dropna().unique():
        cond_data = df[df['Condition'] == cond]
        print(f"\n  {cond} (n={len(cond_data):,} observations):")
        print(f"    IntronFraction:     median={cond_data['IntronFraction'].median():.4f}, "
              f"mean={cond_data['IntronFraction'].mean():.4f}, "
              f"std={cond_data['IntronFraction'].std():.4f}")
        log2_finite = cond_data['Log2IntronExonRatio'][np.isfinite(cond_data['Log2IntronExonRatio'])]
        print(f"    Log2IntronExonRatio: median={log2_finite.median():.4f}, "
              f"mean={log2_finite.mean():.4f}, "
              f"std={log2_finite.std():.4f}")

    return adata, df, prefilter_stats


#%%
# =============================================================================
# GENE FILTERING
# =============================================================================

def filter_genes(df, min_cells_per_sample=10, min_samples_per_group=3, min_cells_per_group=30,
                 use_relative=False, min_cells_fraction_per_sample=0.01, min_cells_fraction_per_group=0.005):
    """
    Apply three-tier gene filtering strategy.

    Parameters
    ----------
    df : DataFrame
        Long-form data with columns: geneName, sample, Condition
    min_cells_per_sample : int
        Minimum cells expressing gene per sample (absolute)
    min_samples_per_group : int
        Minimum samples per condition where gene is detected
    min_cells_per_group : int
        Minimum total cells per condition (absolute)
    use_relative : bool
        If True, use fraction-based thresholds instead of absolute
    min_cells_fraction_per_sample : float
        Minimum fraction of cells per sample (if use_relative=True)
    min_cells_fraction_per_group : float
        Minimum fraction of cells per condition (if use_relative=True)

    Returns
    -------
    valid_genes : list
        List of genes passing all filters
    filtering_stats : dict
        Statistics about filtering steps
    """
    print_section("GENE FILTERING")

    # Detect condition labels for filtering
    unique_conditions = df['Condition'].dropna().unique()
    ctrl_label, als_label = detect_condition_labels(unique_conditions)
    print(f"Detected conditions: Control='{ctrl_label}', ALS='{als_label}'")

    n_genes_initial = df['geneName'].nunique()
    print(f"Initial genes: {n_genes_initial:,}")
    print(f"Filtering mode: {'RELATIVE (fraction-based)' if use_relative else 'ABSOLUTE'}")

    # FILTER 1: Min cells per gene per sample
    counts_per_sample = df.groupby(['sample', 'geneName']).size().reset_index(name='n_cells')
    df = df.merge(counts_per_sample, on=['sample', 'geneName'], how='left')

    if use_relative:
        # Calculate cells per sample for relative threshold
        cells_per_sample = df.groupby('sample').size()
        df['sample_total_cells'] = df['sample'].map(cells_per_sample)
        df['cells_threshold'] = (df['sample_total_cells'] * min_cells_fraction_per_sample).clip(lower=3)
        df_filtered = df[df['n_cells'] >= df['cells_threshold']].copy()
        print(f"\nFilter 1: Gene must be in ≥{min_cells_fraction_per_sample*100:.1f}% of cells per sample (min 3)")
    else:
        df_filtered = df[df['n_cells'] >= min_cells_per_sample].copy()
        print(f"\nFilter 1: Gene must be in ≥{min_cells_per_sample} cells per sample")

    n_genes_after_f1 = df_filtered['geneName'].nunique()
    print(f"  Genes remaining: {n_genes_after_f1:,} ({100*n_genes_after_f1/n_genes_initial:.1f}%)")

    # FILTER 2: Min samples per condition
    print(f"\nFilter 2: Gene must be in ≥{min_samples_per_group} samples per condition")
    sample_counts = (
        df_filtered
        .groupby(['geneName', 'Condition'])['sample']
        .nunique()
        .unstack(fill_value=0)
    )

    valid_by_samples = sample_counts[
        (sample_counts.get(ctrl_label, 0) >= min_samples_per_group) &
        (sample_counts.get(als_label, 0) >= min_samples_per_group)
    ].index.tolist()

    n_genes_after_f2 = len(valid_by_samples)
    print(f"  Genes remaining: {n_genes_after_f2:,} ({100*n_genes_after_f2/n_genes_initial:.1f}%)")

    # FILTER 3: Min total cells per condition
    total_cells_per_condition = (
        df_filtered
        .groupby(['geneName', 'Condition'])
        .size()
        .unstack(fill_value=0)
    )

    if use_relative:
        # Calculate total cells per condition for relative threshold
        cells_per_condition = df_filtered.groupby('Condition').size()
        ctrl_threshold = max(3, int(cells_per_condition.get(ctrl_label, 0) * min_cells_fraction_per_group))
        als_threshold = max(3, int(cells_per_condition.get(als_label, 0) * min_cells_fraction_per_group))
        print(f"\nFilter 3: Gene must have ≥{min_cells_fraction_per_group*100:.1f}% of cells per condition")
        print(f"  (Control: ≥{ctrl_threshold} cells, ALS: ≥{als_threshold} cells)")
        valid_by_cells = total_cells_per_condition[
            (total_cells_per_condition.get(ctrl_label, 0) >= ctrl_threshold) &
            (total_cells_per_condition.get(als_label, 0) >= als_threshold)
        ].index.tolist()
    else:
        print(f"\nFilter 3: Gene must have ≥{min_cells_per_group} total cells per condition")
        valid_by_cells = total_cells_per_condition[
            (total_cells_per_condition.get(ctrl_label, 0) >= min_cells_per_group) &
            (total_cells_per_condition.get(als_label, 0) >= min_cells_per_group)
        ].index.tolist()

    # Take intersection of all filters
    valid_genes = sorted(list(set(valid_by_samples).intersection(set(valid_by_cells))))

    n_genes_final = len(valid_genes)
    print(f"  Genes remaining: {n_genes_final:,} ({100*n_genes_final/n_genes_initial:.1f}%)")

    print(f"\n{'='*80}")
    print(f"FILTERING SUMMARY")
    print(f"{'='*80}")
    print(f"Initial genes:        {n_genes_initial:>8,}")
    print(f"After filter 1:       {n_genes_after_f1:>8,}  (-{n_genes_initial-n_genes_after_f1:,})")
    print(f"After filter 2:       {n_genes_after_f2:>8,}  (-{n_genes_after_f1-n_genes_after_f2:,})")
    print(f"After filter 3:       {n_genes_final:>8,}  (-{n_genes_after_f2-n_genes_final:,})")
    print(f"{'='*80}")

    filtering_stats = {
        'n_initial': n_genes_initial,
        'n_after_filter1': n_genes_after_f1,
        'n_after_filter2': n_genes_after_f2,
        'n_final': n_genes_final,
        'valid_genes': valid_genes
    }

    # Filter the dataframe
    df_filtered = df_filtered[df_filtered['geneName'].isin(valid_genes)].copy()

    return df_filtered, valid_genes, filtering_stats


#%%
# =============================================================================
# SAMPLE-LEVEL AGGREGATION
# =============================================================================

def aggregate_to_sample_level(df, metric='IntronFraction', agg_func='median', threshold=None):
    """
    Aggregate cell-level data to sample level.

    Parameters
    ----------
    df : DataFrame
        Cell-level data
    metric : str
        Metric to aggregate (default: IntronFraction)
    agg_func : str
        Aggregation function: 'median', 'mean', or 'proportion'
    threshold : float, optional
        For 'proportion' mode: IntronFraction > threshold = "has retention"

    Returns
    -------
    agg_df : DataFrame
        Sample-level aggregated data
    """
    print_section(f"SAMPLE-LEVEL AGGREGATION ({agg_func.upper()})")

    if agg_func == 'proportion':
        if threshold is None:
            raise ValueError("threshold must be specified for proportion aggregation")

        print(f"Calculating proportion of cells with {metric} > {threshold}")

        # Create binary retention indicator
        df['has_retention'] = (df[metric] > threshold).astype(int)

        # Calculate proportion per sample
        agg_df = (
            df.groupby(['geneName', 'sample', 'Condition'])['has_retention']
            .mean()  # Mean of 0/1 = proportion
            .reset_index()
            .rename(columns={'has_retention': 'RetentionProportion'})
        )

        print(f"\nRetention statistics:")
        print(f"  Mean retention proportion across all samples: {agg_df['RetentionProportion'].mean():.3f}")
        print(f"  Median retention proportion: {agg_df['RetentionProportion'].median():.3f}")
        print(f"  Range: [{agg_df['RetentionProportion'].min():.3f}, {agg_df['RetentionProportion'].max():.3f}]")

    elif agg_func == 'median':
        print(f"Aggregating {metric} by sample using {agg_func}...")
        agg_df = (
            df.groupby(['geneName', 'sample', 'Condition'])[metric]
            .median()
            .reset_index()
            .rename(columns={metric: f'{metric}_{agg_func}'})
        )
    elif agg_func == 'mean':
        print(f"Aggregating {metric} by sample using {agg_func}...")
        agg_df = (
            df.groupby(['geneName', 'sample', 'Condition'])[metric]
            .mean()
            .reset_index()
            .rename(columns={metric: f'{metric}_{agg_func}'})
        )
    else:
        raise ValueError(f"Unknown aggregation function: {agg_func}. Use 'median', 'mean', or 'proportion'.")

    print(f"Aggregated data shape: {agg_df.shape}")
    print(f"Unique genes: {agg_df['geneName'].nunique()}")
    print(f"Unique samples: {agg_df['sample'].nunique()}")

    return agg_df


#%%
# =============================================================================
# GENOME-WIDE STATISTICAL TESTING
# =============================================================================

def genome_wide_analysis(agg_df, valid_genes, metric='IntronFraction_median',
                          fdr_threshold=0.05, min_effect_size=0.0):
    """
    Perform genome-wide Mann-Whitney U test with FDR correction.

    Parameters
    ----------
    agg_df : DataFrame
        Sample-level aggregated data
    valid_genes : list
        List of genes to test
    metric : str
        Metric column name in agg_df
    fdr_threshold : float
        FDR threshold for significance
    min_effect_size : float
        Minimum median difference to report

    Returns
    -------
    results_df : DataFrame
        Test results with FDR correction
    """
    print_section("GENOME-WIDE STATISTICAL ANALYSIS")

    print(f"Testing {len(valid_genes):,} genes...")
    print(f"Metric: {metric}")
    print(f"Test: Mann-Whitney U (non-parametric)")
    print(f"Multiple testing correction: Benjamini-Hochberg FDR")
    print(f"FDR threshold: {fdr_threshold}")

    # Auto-detect condition names
    unique_conditions = agg_df['Condition'].dropna().unique()
    control_label, als_label = detect_condition_labels(unique_conditions)

    print(f"Detected condition labels: Control='{control_label}', ALS='{als_label}'")

    results = []

    for i, gene in enumerate(valid_genes):
        if (i + 1) % 500 == 0:
            print(f"  Tested {i+1}/{len(valid_genes)} genes...")

        gene_data = agg_df[agg_df['geneName'] == gene]

        ctrl_vals = gene_data[gene_data['Condition'] == control_label][metric].values
        als_vals = gene_data[gene_data['Condition'] == als_label][metric].values

        # Skip if insufficient data
        if len(ctrl_vals) < 2 or len(als_vals) < 2:
            continue

        # Mann-Whitney U test
        stat, pval = mannwhitneyu(ctrl_vals, als_vals, alternative='two-sided')

        # Calculate summary statistics
        median_ctrl = np.median(ctrl_vals)
        median_als = np.median(als_vals)
        median_diff = median_als - median_ctrl
        fold_change = median_als / median_ctrl if median_ctrl > 0 else np.nan

        results.append({
            'gene': gene,
            'n_samples_control': len(ctrl_vals),
            'n_samples_als': len(als_vals),
            'median_control': median_ctrl,
            'median_als': median_als,
            'median_diff': median_diff,
            'fold_change': fold_change,
            'mannwhitney_stat': stat,
            'p_value': pval
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Check if we have any results
    if len(results_df) == 0:
        print(f"\n{'='*80}")
        print(f"WARNING: No genes passed the filtering criteria!")
        print(f"All genes were skipped due to insufficient samples per condition.")
        print(f"Check that your data has at least 2 samples per condition for each gene.")
        print(f"{'='*80}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['gene', 'n_samples_control', 'n_samples_als',
                                     'median_control', 'median_als', 'median_diff',
                                     'fold_change', 'mannwhitney_stat', 'p_value',
                                     'p_adj', 'significant'])

    # Apply FDR correction
    print(f"\nApplying FDR correction...")
    reject, pvals_corrected, _, _ = multipletests(
        results_df['p_value'],
        alpha=fdr_threshold,
        method='fdr_bh'
    )

    results_df['p_adj'] = pvals_corrected
    results_df['significant'] = reject

    # Apply effect size filter
    if min_effect_size > 0:
        results_df['significant'] = (
            results_df['significant'] &
            (np.abs(results_df['median_diff']) >= min_effect_size)
        )

    # Sort by adjusted p-value
    results_df = results_df.sort_values('p_adj')

    # Summary
    n_significant = results_df['significant'].sum()
    n_upregulated = ((results_df['significant']) & (results_df['median_diff'] > 0)).sum()
    n_downregulated = ((results_df['significant']) & (results_df['median_diff'] < 0)).sum()

    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Total genes tested:        {len(results_df):>8,}")
    print(f"Significant (FDR < {fdr_threshold}):  {n_significant:>8,} ({100*n_significant/len(results_df):.1f}%)")
    print(f"  Higher in ALS:           {n_upregulated:>8,}")
    print(f"  Lower in ALS:            {n_downregulated:>8,}")
    print(f"{'='*80}")

    if n_significant > 0:
        print(f"\nTop 10 most significant genes:")
        print(results_df[results_df['significant']].head(10)[
            ['gene', 'median_diff', 'fold_change', 'p_value', 'p_adj']
        ].to_string(index=False))

    return results_df


#%%
# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_volcano(results_df, fdr_threshold=0.05, outdir=None):
    """Create volcano plot of results."""
    print("\nGenerating volcano plot...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Prepare data
    results_df['neg_log10_p'] = -np.log10(results_df['p_adj'])
    results_df['log2_fc'] = np.log2(results_df['fold_change'])

    # Replace inf values
    results_df['log2_fc'] = results_df['log2_fc'].replace([np.inf, -np.inf], np.nan)

    # Plot
    sig = results_df['significant']

    ax.scatter(
        results_df.loc[~sig, 'log2_fc'],
        results_df.loc[~sig, 'neg_log10_p'],
        c='lightgray', s=10, alpha=0.5, label='Not significant'
    )

    ax.scatter(
        results_df.loc[sig, 'log2_fc'],
        results_df.loc[sig, 'neg_log10_p'],
        c='red', s=20, alpha=0.7, label=f'FDR < {fdr_threshold}'
    )

    # Add threshold line
    ax.axhline(-np.log10(fdr_threshold), color='blue', linestyle='--',
               linewidth=1, alpha=0.5, label=f'FDR = {fdr_threshold}')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    # Labels
    ax.set_xlabel('log2(Fold Change) [ALS / Control]', fontsize=12)
    ax.set_ylabel('-log10(Adjusted P-value)', fontsize=12)
    ax.set_title('Volcano Plot: Intron Retention in ALS vs Control',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if outdir:
        plt.savefig(os.path.join(outdir, 'volcano_plot.png'), dpi=Config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: volcano_plot.png")

    return fig


def plot_top_genes_boxplot(df, agg_df, results_df, top_n=20, metric='IntronFraction_median', outdir=None):
    """Plot boxplots for top significant genes."""
    print(f"\nGenerating boxplot for top {top_n} genes...")

    # Get top significant genes
    sig_genes = results_df[results_df['significant']].head(top_n)['gene'].tolist()

    if len(sig_genes) == 0:
        print("  No significant genes to plot.")
        return None

    # Prepare data
    plot_data = agg_df[agg_df['geneName'].isin(sig_genes)].copy()

    # Calculate number of rows needed
    n_genes = len(sig_genes)
    n_cols = 5
    n_rows = int(np.ceil(n_genes / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*2.5))
    axes = axes.flatten() if n_genes > 1 else [axes]

    for i, gene in enumerate(sig_genes):
        ax = axes[i]
        gene_data = plot_data[plot_data['geneName'] == gene]

        # Boxplot
        sns.boxplot(data=gene_data, x='Condition', y=metric, ax=ax, palette='Set2')

        # Add p-value
        pval = results_df.loc[results_df['gene'] == gene, 'p_adj'].values[0]
        median_diff = results_df.loc[results_df['gene'] == gene, 'median_diff'].values[0]

        # Significance stars
        if pval < 0.0001:
            sig_label = "****"
        elif pval < 0.001:
            sig_label = "***"
        elif pval < 0.01:
            sig_label = "**"
        elif pval < 0.05:
            sig_label = "*"
        else:
            sig_label = "ns"

        ax.set_title(f'{gene}\nΔ={median_diff:.3f}, {sig_label}', fontsize=9)
        ax.set_xlabel('')
        ax.set_ylabel('Retention Proportion' if i % n_cols == 0 else '')

    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Top {len(sig_genes)} Genes with Differential Intron Retention',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    if outdir:
        plt.savefig(os.path.join(outdir, f'top_{top_n}_genes_boxplots.png'),
                   dpi=Config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: top_{top_n}_genes_boxplots.png")

    return fig


def plot_qc_summary(df, valid_genes, outdir=None):
    """Generate QC summary plots including Log2 ratio metrics."""
    print("\nGenerating QC summary plots...")

    fig, axes = plt.subplots(3, 3, figsize=(15, 14))

    # Define consistent color palette using Set2 colors mapped to conditions
    # Sorted alphabetically: ALS, CTR -> Set2 colors [0], [1]
    set2_colors = sns.color_palette('Set2', n_colors=2)
    CONDITION_COLORS = {'CTR': set2_colors[0], 'ALS': set2_colors[1]}

    # Filter out NaN conditions (used by multiple plots)
    valid_data = df[(df['geneName'].isin(valid_genes)) & (df['Condition'].notna())]

    # 1. Intron vs Exon counts (top-left)
    ax = axes[0, 0]
    sample_data = df[df['geneName'].isin(valid_genes)].sample(min(10000, len(df)))
    # Filter out zeros for log scale plotting AND NaN conditions
    sample_data_positive = sample_data[
        (sample_data['ExonCount'] > 0) &
        (sample_data['IntronCount'] > 0) &
        (sample_data['Condition'].notna())
    ]

    if len(sample_data_positive) > 0:
        colors = sample_data_positive['Condition'].map(CONDITION_COLORS)

        ax.scatter(
            sample_data_positive['ExonCount'],
            sample_data_positive['IntronCount'],
            c=colors.tolist(),
            alpha=0.3, s=5
        )
        ax.set_xscale('log')
        ax.set_yscale('log')
        # Add legend for scatter plot
        for cond, color in CONDITION_COLORS.items():
            ax.scatter([], [], c=color, label=cond, alpha=0.6)
        ax.legend(loc='lower right', fontsize=8)
    ax.set_xlabel('Exon Count')
    ax.set_ylabel('Intron Count')
    ax.set_title('Exon vs Intron Counts')

    # 2. Cells per sample
    ax = axes[0, 1]
    sample_counts = df.groupby(['sample', 'Condition']).size().reset_index(name='n_cells')
    sns.barplot(data=sample_counts, x='sample', y='n_cells', hue='Condition', ax=ax, palette=CONDITION_COLORS)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Number of Observations')
    ax.set_title('Cells per Sample')
    ax.tick_params(axis='x', rotation=45)

    # 3. Genes per sample
    ax = axes[0, 2]
    genes_per_sample = df.groupby(['sample', 'Condition'])['geneName'].nunique().reset_index(name='n_genes')
    sns.barplot(data=genes_per_sample, x='sample', y='n_genes', hue='Condition', ax=ax, palette=CONDITION_COLORS)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Unique Genes')
    ax.set_title('Genes Detected per Sample')
    ax.tick_params(axis='x', rotation=45)

    # 4. Intron fraction distribution (middle-left)
    ax = axes[1, 0]
    if len(valid_data) > 0:
        # Plot each condition explicitly to ensure correct legend labels
        for cond in sorted(CONDITION_COLORS.keys()):
            if cond in valid_data['Condition'].values:
                cond_data = valid_data[valid_data['Condition'] == cond]['IntronFraction']
                ax.hist(cond_data, bins=50, alpha=0.6, label=cond, color=CONDITION_COLORS[cond])
        ax.legend()
    ax.set_xlabel('Intron Fraction')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Intron Fraction')

    # 5. Intron fraction by condition
    ax = axes[1, 1]
    # Filter out NaN conditions
    valid_condition_data = df[(df['geneName'].isin(valid_genes)) & (df['Condition'].notna())]
    if len(valid_condition_data) > 0:
        sns.violinplot(data=valid_condition_data,
                       x='Condition', y='IntronFraction', ax=ax, palette=CONDITION_COLORS,
                       order=sorted(CONDITION_COLORS.keys()))
    ax.set_ylabel('Intron Fraction')
    ax.set_title('Intron Fraction by Condition')

    # 6. Sample-level intron fraction
    ax = axes[1, 2]
    # Filter out NaN conditions before grouping
    valid_for_grouping = df[(df['geneName'].isin(valid_genes)) & (df['Condition'].notna())]
    sample_intron = valid_for_grouping.groupby(
        ['sample', 'Condition'])['IntronFraction'].median().reset_index()
    if len(sample_intron) > 0:
        sns.boxplot(data=sample_intron, x='Condition', y='IntronFraction', ax=ax,
                    palette=CONDITION_COLORS, order=sorted(CONDITION_COLORS.keys()))
        sns.stripplot(data=sample_intron, x='Condition', y='IntronFraction',
                     color='black', size=8, ax=ax, order=sorted(CONDITION_COLORS.keys()))
    ax.set_ylabel('Median Intron Fraction per Sample')
    ax.set_title('Sample-Level Intron Retention')

    # 7. Log2 ratio distribution
    ax = axes[2, 0]
    if 'Log2IntronExonRatio' in df.columns and len(valid_data) > 0:
        # Filter out infinite values
        log2_valid = valid_data[np.isfinite(valid_data['Log2IntronExonRatio'])]
        if len(log2_valid) > 0:
            # Plot each condition explicitly to ensure correct legend labels
            for cond in sorted(CONDITION_COLORS.keys()):
                if cond in log2_valid['Condition'].values:
                    cond_data = log2_valid[log2_valid['Condition'] == cond]['Log2IntronExonRatio']
                    ax.hist(cond_data, bins=50, alpha=0.6, label=cond, color=CONDITION_COLORS[cond])
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax.legend()
    ax.set_xlabel('Log2(Intron/Exon)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Log2 Intron/Exon Ratio')

    # 8. Log2 ratio by condition
    ax = axes[2, 1]
    if 'Log2IntronExonRatio' in df.columns and len(valid_condition_data) > 0:
        log2_condition = valid_condition_data[np.isfinite(valid_condition_data['Log2IntronExonRatio'])]
        if len(log2_condition) > 0:
            sns.violinplot(data=log2_condition,
                           x='Condition', y='Log2IntronExonRatio', ax=ax,
                           palette=CONDITION_COLORS, order=sorted(CONDITION_COLORS.keys()))
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Log2(Intron/Exon)')
    ax.set_title('Log2 Ratio by Condition')

    # 9. Sample-level Log2 ratio
    ax = axes[2, 2]
    if 'Log2IntronExonRatio' in df.columns and len(valid_for_grouping) > 0:
        log2_grouping = valid_for_grouping[np.isfinite(valid_for_grouping['Log2IntronExonRatio'])]
        if len(log2_grouping) > 0:
            sample_log2 = log2_grouping.groupby(
                ['sample', 'Condition'])['Log2IntronExonRatio'].median().reset_index()
            if len(sample_log2) > 0:
                sns.boxplot(data=sample_log2, x='Condition', y='Log2IntronExonRatio', ax=ax,
                            palette=CONDITION_COLORS, order=sorted(CONDITION_COLORS.keys()))
                sns.stripplot(data=sample_log2, x='Condition', y='Log2IntronExonRatio',
                             color='black', size=8, ax=ax, order=sorted(CONDITION_COLORS.keys()))
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Median Log2(Intron/Exon) per Sample')
    ax.set_title('Sample-Level Log2 Ratio')

    plt.suptitle('QC Summary - Transcriptome Integrity Analysis',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if outdir:
        plt.savefig(os.path.join(outdir, 'qc_summary.png'), dpi=Config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: qc_summary.png")

    return fig


def plot_candidate_genes_overview(df, candidate_genes, outdir=None):
    """
    Create overview plots for candidate genes.

    Parameters
    ----------
    df : DataFrame
        Cell-level data with IntronFraction
    candidate_genes : list
        List of candidate gene names
    outdir : str
        Output directory for plots
    """
    print("\nGenerating candidate genes overview plots...")

    # Filter to candidate genes that exist in data
    available_genes = [g for g in candidate_genes if g in df['geneName'].unique()]

    if len(available_genes) == 0:
        print("  No candidate genes found in data.")
        return None

    print(f"  Plotting {len(available_genes)} candidate genes...")

    # Auto-detect condition labels
    unique_conditions = df['Condition'].dropna().unique()
    ctrl_label, als_label = detect_condition_labels(unique_conditions)
    print(f"  Detected conditions: Control='{ctrl_label}', ALS='{als_label}'")

    # --- Plot 1: Violin plots for each candidate gene ---
    n_genes = len(available_genes)
    n_cols = min(3, n_genes)
    n_rows = int(np.ceil(n_genes / n_cols))

    fig1, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3.5))
    if n_genes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, gene in enumerate(available_genes):
        ax = axes[i]
        gene_data = df[df['geneName'] == gene]

        # Violin + strip plot (explicit order for consistency)
        condition_order = [ctrl_label, als_label]
        sns.violinplot(data=gene_data, x='Condition', y='IntronFraction',
                      ax=ax, palette='Set2', alpha=0.7, order=condition_order)
        sns.stripplot(data=gene_data, x='Condition', y='IntronFraction',
                     ax=ax, color='black', alpha=0.3, size=2, order=condition_order)

        # Stats using detected labels
        ctrl_median = gene_data[gene_data['Condition'] == ctrl_label]['IntronFraction'].median()
        als_median = gene_data[gene_data['Condition'] == als_label]['IntronFraction'].median()
        n_ctrl = len(gene_data[gene_data['Condition'] == ctrl_label])
        n_als = len(gene_data[gene_data['Condition'] == als_label])

        ax.set_title(f'{gene}\n{ctrl_label}: {ctrl_median:.3f} (n={n_ctrl}) | {als_label}: {als_median:.3f} (n={n_als})',
                    fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('Intron Fraction' if i % n_cols == 0 else '')

    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Candidate Genes: Intron Retention by Condition',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if outdir:
        plt.savefig(os.path.join(outdir, 'candidate_genes_violin.png'),
                   dpi=Config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: candidate_genes_violin.png")

    # --- Plot 2: Sample-level summary (pseudobulk) ---
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3.5))
    if n_genes == 1:
        axes2 = [axes2]
    else:
        axes2 = axes2.flatten()

    for i, gene in enumerate(available_genes):
        ax = axes2[i]
        gene_data = df[df['geneName'] == gene]

        # Aggregate to sample level
        sample_agg = gene_data.groupby(['sample', 'Condition'])['IntronFraction'].median().reset_index()

        # Explicit order for consistency with violin plot
        condition_order = [ctrl_label, als_label]
        sns.boxplot(data=sample_agg, x='Condition', y='IntronFraction',
                   ax=ax, palette='Set2', width=0.5, order=condition_order)
        sns.stripplot(data=sample_agg, x='Condition', y='IntronFraction',
                     ax=ax, color='black', size=8, alpha=0.8, order=condition_order)

        ax.set_title(f'{gene} (Sample-level)', fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Median Intron Fraction' if i % n_cols == 0 else '')

    # Hide unused subplots
    for j in range(i+1, len(axes2)):
        axes2[j].axis('off')

    plt.suptitle('Candidate Genes: Sample-Level Intron Retention (Pseudobulk)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if outdir:
        plt.savefig(os.path.join(outdir, 'candidate_genes_pseudobulk.png'),
                   dpi=Config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: candidate_genes_pseudobulk.png")

    # --- Plot 3: Summary heatmap ---
    fig3, ax3 = plt.subplots(figsize=(10, max(4, len(available_genes) * 0.5)))

    # Calculate summary statistics per gene per condition
    summary_data = []
    for gene in available_genes:
        gene_data = df[df['geneName'] == gene]
        for cond in [ctrl_label, als_label]:
            cond_data = gene_data[gene_data['Condition'] == cond]['IntronFraction']
            summary_data.append({
                'Gene': gene,
                'Condition': cond,
                'Median': cond_data.median(),
                'Mean': cond_data.mean(),
                'N_cells': len(cond_data)
            })

    summary_df = pd.DataFrame(summary_data)

    # Pivot for heatmap
    heatmap_data = summary_df.pivot(index='Gene', columns='Condition', values='Median')
    heatmap_data['Diff (ALS-Ctrl)'] = heatmap_data[als_label] - heatmap_data[ctrl_label]

    # Sort by difference
    heatmap_data = heatmap_data.sort_values('Diff (ALS-Ctrl)', ascending=False)

    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r',
               center=0, ax=ax3, cbar_kws={'label': 'Intron Fraction'})
    ax3.set_title('Candidate Genes: Median Intron Fraction Summary',
                 fontsize=14, fontweight='bold')
    ax3.set_ylabel('')

    plt.tight_layout()

    if outdir:
        plt.savefig(os.path.join(outdir, 'candidate_genes_heatmap.png'),
                   dpi=Config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: candidate_genes_heatmap.png")

        # Also save summary table
        summary_df.to_csv(os.path.join(outdir, 'candidate_genes_summary.csv'), index=False)
        print(f"  Saved: candidate_genes_summary.csv")

    # --- Plot 4: Log2 Intron/Exon Ratio violin plots ---
    # Check if Log2IntronExonRatio exists in data
    if 'Log2IntronExonRatio' not in df.columns:
        print("  Log2IntronExonRatio not found in data, skipping ratio plots.")
        return fig1, fig2, fig3

    fig4, axes4 = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3.5))
    if n_genes == 1:
        axes4 = [axes4]
    else:
        axes4 = axes4.flatten()

    for i, gene in enumerate(available_genes):
        ax = axes4[i]
        gene_data = df[df['geneName'] == gene].dropna(subset=['Log2IntronExonRatio'])

        if len(gene_data) == 0:
            ax.text(0.5, 0.5, f'{gene}\nNo valid data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel('')
            ax.set_ylabel('')
            continue

        # Violin + strip plot
        condition_order = [ctrl_label, als_label]
        sns.violinplot(data=gene_data, x='Condition', y='Log2IntronExonRatio',
                      ax=ax, palette='Set2', alpha=0.7, order=condition_order)
        sns.stripplot(data=gene_data, x='Condition', y='Log2IntronExonRatio',
                     ax=ax, color='black', alpha=0.3, size=2, order=condition_order)

        # Stats
        ctrl_data = gene_data[gene_data['Condition'] == ctrl_label]['Log2IntronExonRatio']
        als_data = gene_data[gene_data['Condition'] == als_label]['Log2IntronExonRatio']
        ctrl_median = ctrl_data.median() if len(ctrl_data) > 0 else np.nan
        als_median = als_data.median() if len(als_data) > 0 else np.nan

        ax.set_title(f'{gene}\n{ctrl_label}: {ctrl_median:.2f} | {als_label}: {als_median:.2f}',
                    fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('Log2(Intron/Exon)' if i % n_cols == 0 else '')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Hide unused subplots
    for j in range(i+1, len(axes4)):
        axes4[j].axis('off')

    plt.suptitle('Candidate Genes: Log2 Intron/Exon Ratio by Condition',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if outdir:
        plt.savefig(os.path.join(outdir, 'candidate_genes_log2ratio_violin.png'),
                   dpi=Config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: candidate_genes_log2ratio_violin.png")

    # --- Plot 5: Log2 Ratio Sample-level (pseudobulk) ---
    fig5, axes5 = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3.5))
    if n_genes == 1:
        axes5 = [axes5]
    else:
        axes5 = axes5.flatten()

    for i, gene in enumerate(available_genes):
        ax = axes5[i]
        gene_data = df[df['geneName'] == gene].dropna(subset=['Log2IntronExonRatio'])

        if len(gene_data) == 0:
            ax.text(0.5, 0.5, f'{gene}\nNo valid data', ha='center', va='center', transform=ax.transAxes)
            continue

        # Aggregate to sample level
        sample_agg = gene_data.groupby(['sample', 'Condition'])['Log2IntronExonRatio'].median().reset_index()

        condition_order = [ctrl_label, als_label]
        sns.boxplot(data=sample_agg, x='Condition', y='Log2IntronExonRatio',
                   ax=ax, palette='Set2', width=0.5, order=condition_order)
        sns.stripplot(data=sample_agg, x='Condition', y='Log2IntronExonRatio',
                     ax=ax, color='black', size=8, alpha=0.8, order=condition_order)

        ax.set_title(f'{gene} (Sample-level)', fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Median Log2(Intron/Exon)' if i % n_cols == 0 else '')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Hide unused subplots
    for j in range(i+1, len(axes5)):
        axes5[j].axis('off')

    plt.suptitle('Candidate Genes: Sample-Level Log2 Intron/Exon Ratio',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if outdir:
        plt.savefig(os.path.join(outdir, 'candidate_genes_log2ratio_pseudobulk.png'),
                   dpi=Config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: candidate_genes_log2ratio_pseudobulk.png")

    # --- Plot 6: Log2 Ratio Summary Heatmap ---
    fig6, ax6 = plt.subplots(figsize=(10, max(4, len(available_genes) * 0.5)))

    log2_summary_data = []
    for gene in available_genes:
        gene_data = df[df['geneName'] == gene].dropna(subset=['Log2IntronExonRatio'])
        for cond in [ctrl_label, als_label]:
            cond_data = gene_data[gene_data['Condition'] == cond]['Log2IntronExonRatio']
            log2_summary_data.append({
                'Gene': gene,
                'Condition': cond,
                'Median': cond_data.median() if len(cond_data) > 0 else np.nan,
                'Mean': cond_data.mean() if len(cond_data) > 0 else np.nan,
                'N_cells': len(cond_data)
            })

    log2_summary_df = pd.DataFrame(log2_summary_data)

    # Pivot for heatmap
    log2_heatmap = log2_summary_df.pivot(index='Gene', columns='Condition', values='Median')
    log2_heatmap['Diff (ALS-Ctrl)'] = log2_heatmap[als_label] - log2_heatmap[ctrl_label]

    # Sort by difference
    log2_heatmap = log2_heatmap.sort_values('Diff (ALS-Ctrl)', ascending=False)

    sns.heatmap(log2_heatmap, annot=True, fmt='.2f', cmap='RdBu_r',
               center=0, ax=ax6, cbar_kws={'label': 'Log2(Intron/Exon)'})
    ax6.set_title('Candidate Genes: Median Log2 Intron/Exon Ratio Summary',
                 fontsize=14, fontweight='bold')
    ax6.set_ylabel('')

    plt.tight_layout()

    if outdir:
        plt.savefig(os.path.join(outdir, 'candidate_genes_log2ratio_heatmap.png'),
                   dpi=Config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: candidate_genes_log2ratio_heatmap.png")

        # Save log2 ratio summary table
        log2_summary_df.to_csv(os.path.join(outdir, 'candidate_genes_log2ratio_summary.csv'), index=False)
        print(f"  Saved: candidate_genes_log2ratio_summary.csv")

    return fig1, fig2, fig3, fig4, fig5, fig6


def plot_stmn2_comprehensive(df, outdir=None):
    """
    Create a comprehensive 4-panel figure for STMN2 showing:
    - Cell-level IntronFraction
    - Sample-level IntronFraction (pseudobulk)
    - Cell-level Log2IntronExonRatio
    - Sample-level Log2IntronExonRatio (pseudobulk)

    Parameters
    ----------
    df : DataFrame
        Cell-level data with IntronFraction and Log2IntronExonRatio
    outdir : str
        Output directory for plots
    """
    print("\nGenerating comprehensive STMN2 figure...")

    gene = 'STMN2'

    # Check if STMN2 exists in data
    if gene not in df['geneName'].unique():
        print(f"  {gene} not found in data.")
        return None

    gene_data = df[df['geneName'] == gene].copy()
    print(f"  {gene}: {len(gene_data)} cells")

    # Auto-detect condition labels
    unique_conditions = gene_data['Condition'].dropna().unique()
    ctrl_label, als_label = detect_condition_labels(unique_conditions)
    condition_order = [ctrl_label, als_label]

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Color palette
    palette = {'Control': '#66c2a5', 'ALS': '#fc8d62', ctrl_label: '#66c2a5', als_label: '#fc8d62'}

    # --- Panel A: Cell-level Intron Fraction ---
    ax = axes[0, 0]
    sns.violinplot(data=gene_data, x='Condition', y='IntronFraction',
                   ax=ax, palette=palette, alpha=0.7, order=condition_order)
    sns.stripplot(data=gene_data, x='Condition', y='IntronFraction',
                  ax=ax, color='black', alpha=0.3, size=2, order=condition_order)

    # Stats
    ctrl_median = gene_data[gene_data['Condition'] == ctrl_label]['IntronFraction'].median()
    als_median = gene_data[gene_data['Condition'] == als_label]['IntronFraction'].median()
    n_ctrl = len(gene_data[gene_data['Condition'] == ctrl_label])
    n_als = len(gene_data[gene_data['Condition'] == als_label])

    # Mann-Whitney test
    from scipy.stats import mannwhitneyu
    ctrl_vals = gene_data[gene_data['Condition'] == ctrl_label]['IntronFraction'].dropna()
    als_vals = gene_data[gene_data['Condition'] == als_label]['IntronFraction'].dropna()
    if len(ctrl_vals) > 0 and len(als_vals) > 0:
        _, pval = mannwhitneyu(ctrl_vals, als_vals, alternative='two-sided')
        pval_str = f"p = {pval:.2e}" if pval < 0.001 else f"p = {pval:.3f}"
    else:
        pval_str = ""

    ax.set_title(f'A) Cell-Level Intron Fraction\n{ctrl_label}: {ctrl_median:.3f} (n={n_ctrl}) | {als_label}: {als_median:.3f} (n={n_als})\n{pval_str}',
                 fontsize=11)
    ax.set_xlabel('')
    ax.set_ylabel('Intron Fraction')

    # --- Panel B: Sample-level Intron Fraction (Pseudobulk) ---
    ax = axes[0, 1]
    sample_agg = gene_data.groupby(['sample', 'Condition'])['IntronFraction'].median().reset_index()

    sns.boxplot(data=sample_agg, x='Condition', y='IntronFraction',
                ax=ax, palette=palette, width=0.5, order=condition_order)
    sns.stripplot(data=sample_agg, x='Condition', y='IntronFraction',
                  ax=ax, color='black', size=10, alpha=0.8, order=condition_order)

    # Stats for sample level
    ctrl_samples = sample_agg[sample_agg['Condition'] == ctrl_label]['IntronFraction']
    als_samples = sample_agg[sample_agg['Condition'] == als_label]['IntronFraction']
    if len(ctrl_samples) > 1 and len(als_samples) > 1:
        _, pval_sample = mannwhitneyu(ctrl_samples, als_samples, alternative='two-sided')
        pval_sample_str = f"p = {pval_sample:.2e}" if pval_sample < 0.001 else f"p = {pval_sample:.3f}"
    else:
        pval_sample_str = ""

    ax.set_title(f'B) Sample-Level Intron Fraction (Pseudobulk)\n{ctrl_label}: n={len(ctrl_samples)} samples | {als_label}: n={len(als_samples)} samples\n{pval_sample_str}',
                 fontsize=11)
    ax.set_xlabel('')
    ax.set_ylabel('Median Intron Fraction')

    # --- Panel C: Cell-level Log2 Intron/Exon Ratio ---
    ax = axes[1, 0]
    gene_data_log2 = gene_data.dropna(subset=['Log2IntronExonRatio'])

    if len(gene_data_log2) > 0:
        sns.violinplot(data=gene_data_log2, x='Condition', y='Log2IntronExonRatio',
                       ax=ax, palette=palette, alpha=0.7, order=condition_order)
        sns.stripplot(data=gene_data_log2, x='Condition', y='Log2IntronExonRatio',
                      ax=ax, color='black', alpha=0.3, size=2, order=condition_order)

        # Stats
        ctrl_log2 = gene_data_log2[gene_data_log2['Condition'] == ctrl_label]['Log2IntronExonRatio']
        als_log2 = gene_data_log2[gene_data_log2['Condition'] == als_label]['Log2IntronExonRatio']
        ctrl_median_log2 = ctrl_log2.median()
        als_median_log2 = als_log2.median()

        if len(ctrl_log2) > 0 and len(als_log2) > 0:
            _, pval_log2 = mannwhitneyu(ctrl_log2, als_log2, alternative='two-sided')
            pval_log2_str = f"p = {pval_log2:.2e}" if pval_log2 < 0.001 else f"p = {pval_log2:.3f}"
        else:
            pval_log2_str = ""

        ax.set_title(f'C) Cell-Level Log2(Intron/Exon) Ratio\n{ctrl_label}: {ctrl_median_log2:.2f} | {als_label}: {als_median_log2:.2f}\n{pval_log2_str}',
                     fontsize=11)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    else:
        ax.text(0.5, 0.5, 'No valid Log2 ratio data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('C) Cell-Level Log2(Intron/Exon) Ratio', fontsize=11)

    ax.set_xlabel('')
    ax.set_ylabel('Log2(Intron/Exon)')

    # --- Panel D: Sample-level Log2 Ratio (Pseudobulk) ---
    ax = axes[1, 1]

    if len(gene_data_log2) > 0:
        sample_agg_log2 = gene_data_log2.groupby(['sample', 'Condition'])['Log2IntronExonRatio'].median().reset_index()

        sns.boxplot(data=sample_agg_log2, x='Condition', y='Log2IntronExonRatio',
                    ax=ax, palette=palette, width=0.5, order=condition_order)
        sns.stripplot(data=sample_agg_log2, x='Condition', y='Log2IntronExonRatio',
                      ax=ax, color='black', size=10, alpha=0.8, order=condition_order)

        # Stats
        ctrl_samples_log2 = sample_agg_log2[sample_agg_log2['Condition'] == ctrl_label]['Log2IntronExonRatio']
        als_samples_log2 = sample_agg_log2[sample_agg_log2['Condition'] == als_label]['Log2IntronExonRatio']

        if len(ctrl_samples_log2) > 1 and len(als_samples_log2) > 1:
            _, pval_sample_log2 = mannwhitneyu(ctrl_samples_log2, als_samples_log2, alternative='two-sided')
            pval_sample_log2_str = f"p = {pval_sample_log2:.2e}" if pval_sample_log2 < 0.001 else f"p = {pval_sample_log2:.3f}"
        else:
            pval_sample_log2_str = ""

        ax.set_title(f'D) Sample-Level Log2(Intron/Exon) (Pseudobulk)\n{ctrl_label}: n={len(ctrl_samples_log2)} | {als_label}: n={len(als_samples_log2)}\n{pval_sample_log2_str}',
                     fontsize=11)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    else:
        ax.text(0.5, 0.5, 'No valid Log2 ratio data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('D) Sample-Level Log2(Intron/Exon) (Pseudobulk)', fontsize=11)

    ax.set_xlabel('')
    ax.set_ylabel('Median Log2(Intron/Exon)')

    # Overall title
    plt.suptitle(f'STMN2 Intron Retention Analysis\nMotor Neurons: ALS vs Control',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if outdir:
        plt.savefig(os.path.join(outdir, 'STMN2_comprehensive_figure.png'),
                    dpi=Config.FIGURE_DPI, bbox_inches='tight')
        plt.savefig(os.path.join(outdir, 'STMN2_comprehensive_figure.pdf'),
                    bbox_inches='tight')
        print(f"  Saved: STMN2_comprehensive_figure.png")
        print(f"  Saved: STMN2_comprehensive_figure.pdf")

    return fig


# =============================================================================
# EFFECT SIZE AND CORRELATION ANALYSIS
# =============================================================================

def rank_genes_by_effect(df, min_cells=10):
    """
    Rank genes by effect size metrics comparing ALS vs Control.

    For each gene, calculates:
    1. Median difference (ALS - CTR) in Log2IntronExonRatio
    2. Fold change (2^median_diff)
    3. Cohen's d effect size

    Parameters
    ----------
    df : DataFrame
        Cell-level data with columns: geneName, Condition, Log2IntronExonRatio
    min_cells : int
        Minimum cells required per condition to include gene (default: 10)

    Returns
    -------
    DataFrame
        Genes ranked by Cohen's d with effect size metrics
    """
    print("\n" + "="*80)
    print("RANKING GENES BY EFFECT SIZE")
    print("="*80)

    results = []
    genes_skipped = 0

    for gene in df['geneName'].unique():
        gdf = df[df['geneName'] == gene]

        als = gdf[gdf['Condition'] == 'ALS']['Log2IntronExonRatio']
        ctr = gdf[gdf['Condition'] == 'CTR']['Log2IntronExonRatio']

        if len(als) < min_cells or len(ctr) < min_cells:
            genes_skipped += 1
            continue

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((als.std()**2 + ctr.std()**2) / 2)
        cohens_d = (als.mean() - ctr.mean()) / pooled_std if pooled_std > 0 else 0

        # Fold change (from median difference in log2 space)
        median_diff = als.median() - ctr.median()
        fc = 2 ** median_diff

        results.append({
            'gene': gene,
            'median_diff': median_diff,
            'fold_change': fc,
            'cohens_d': cohens_d,
            'als_median': als.median(),
            'ctr_median': ctr.median(),
            'als_mean': als.mean(),
            'ctr_mean': ctr.mean(),
            'n_als': len(als),
            'n_ctr': len(ctr)
        })

    results_df = pd.DataFrame(results).sort_values('cohens_d', ascending=False)

    print(f"Genes analyzed: {len(results_df)}")
    print(f"Genes skipped (< {min_cells} cells): {genes_skipped}")

    if len(results_df) > 0:
        # Show top genes with largest effect sizes (both directions)
        print(f"\nTop 10 genes with HIGHEST effect size (more retention in ALS):")
        print(results_df.head(10)[['gene', 'median_diff', 'fold_change', 'cohens_d', 'n_als', 'n_ctr']].to_string(index=False))

        print(f"\nTop 10 genes with LOWEST effect size (less retention in ALS):")
        print(results_df.tail(10)[['gene', 'median_diff', 'fold_change', 'cohens_d', 'n_als', 'n_ctr']].to_string(index=False))

        # Summary statistics
        print(f"\nEffect size distribution:")
        print(f"  Mean Cohen's d: {results_df['cohens_d'].mean():.3f}")
        print(f"  Median Cohen's d: {results_df['cohens_d'].median():.3f}")
        print(f"  Genes with |d| > 0.2 (small effect): {(results_df['cohens_d'].abs() > 0.2).sum()}")
        print(f"  Genes with |d| > 0.5 (medium effect): {(results_df['cohens_d'].abs() > 0.5).sum()}")
        print(f"  Genes with |d| > 0.8 (large effect): {(results_df['cohens_d'].abs() > 0.8).sum()}")

    return results_df


def find_stmn2_correlated_genes(df, min_cells=20):
    """
    Find genes whose intron retention correlates with STMN2 retention at the cell level.

    This analysis identifies genes that show coordinated intron retention patterns
    with STMN2, potentially indicating shared regulatory mechanisms (e.g., TDP-43 targets).

    Parameters
    ----------
    df : DataFrame
        Cell-level data with columns: geneName, cell_id, Log2IntronExonRatio
    min_cells : int
        Minimum cells with both genes measured to include in analysis (default: 20)

    Returns
    -------
    DataFrame
        Genes ranked by Spearman correlation with STMN2, with FDR correction
    """
    print("\n" + "="*80)
    print("FINDING GENES CORRELATED WITH STMN2 RETENTION")
    print("="*80)

    # Check if STMN2 exists in the data
    if 'STMN2' not in df['geneName'].values:
        print("WARNING: STMN2 not found in dataset. Skipping correlation analysis.")
        return pd.DataFrame()

    # Get STMN2 retention per cell
    stmn2 = df[df['geneName'] == 'STMN2'][['cell_id', 'Log2IntronExonRatio']].copy()
    stmn2.columns = ['cell_id', 'STMN2_retention']

    print(f"STMN2 detected in {len(stmn2)} cells")

    results = []
    genes_skipped = 0

    for gene in df['geneName'].unique():
        if gene == 'STMN2':
            continue

        gdf = df[df['geneName'] == gene][['cell_id', 'Log2IntronExonRatio']].copy()

        # Merge with STMN2 to get cells with both measurements
        merged = gdf.merge(stmn2, on='cell_id')

        if len(merged) < min_cells:
            genes_skipped += 1
            continue

        # Calculate Spearman correlation
        corr, pval = spearmanr(merged['Log2IntronExonRatio'],
                               merged['STMN2_retention'])

        results.append({
            'gene': gene,
            'correlation': corr,
            'pval': pval,
            'n_cells': len(merged)
        })

    if len(results) == 0:
        print("No genes with sufficient cell overlap found.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # FDR correction
    results_df['fdr'] = multipletests(results_df['pval'], method='fdr_bh')[1]

    # Sort by correlation (highest positive correlation first)
    results_df = results_df.sort_values('correlation', ascending=False)

    print(f"Genes analyzed: {len(results_df)}")
    print(f"Genes skipped (< {min_cells} shared cells): {genes_skipped}")

    # Summary
    sig_pos = ((results_df['fdr'] < 0.05) & (results_df['correlation'] > 0)).sum()
    sig_neg = ((results_df['fdr'] < 0.05) & (results_df['correlation'] < 0)).sum()

    print(f"\nSignificant correlations (FDR < 0.05):")
    print(f"  Positive correlations: {sig_pos}")
    print(f"  Negative correlations: {sig_neg}")

    if sig_pos > 0:
        print(f"\nTop 10 genes POSITIVELY correlated with STMN2:")
        top_pos = results_df[(results_df['fdr'] < 0.05) & (results_df['correlation'] > 0)].head(10)
        print(top_pos[['gene', 'correlation', 'pval', 'fdr', 'n_cells']].to_string(index=False))

    if sig_neg > 0:
        print(f"\nTop 10 genes NEGATIVELY correlated with STMN2:")
        top_neg = results_df[(results_df['fdr'] < 0.05) & (results_df['correlation'] < 0)].head(10)
        print(top_neg[['gene', 'correlation', 'pval', 'fdr', 'n_cells']].to_string(index=False))

    return results_df


def cross_gene_correlation_analysis(df, candidate_genes, outdir=None):
    """
    Analyze correlations in intron retention across TDP-43 target genes.

    Examines whether cells with high intron retention in one gene (e.g., STMN2)
    also show aberrant patterns in other TDP-43 targets.

    Parameters
    ----------
    df : DataFrame
        Cell-level data with IntronFraction
    candidate_genes : list
        List of candidate gene names to analyze
    outdir : str
        Output directory for plots and tables

    Returns
    -------
    results : dict
        Dictionary containing correlation matrices and statistics
    """
    print_section("CROSS-GENE CORRELATION ANALYSIS")

    # Filter to candidate genes that exist in data
    available_genes = [g for g in candidate_genes if g in df['geneName'].unique()]

    if len(available_genes) < 2:
        print(f"  Need at least 2 candidate genes for correlation analysis.")
        print(f"  Found only: {available_genes}")
        return None

    print(f"Analyzing correlations across {len(available_genes)} genes: {available_genes}")

    # Auto-detect condition labels
    unique_conditions = df['Condition'].dropna().unique()
    ctrl_label, als_label = detect_condition_labels(unique_conditions)

    # Create pivot table: cells x genes with IntronFraction values
    print("\nCreating intron fraction matrix (cells × genes)...")
    intron_matrix = df[df['geneName'].isin(available_genes)].pivot_table(
        values='IntronFraction',
        index='cell_id',
        columns='geneName',
        aggfunc='first'  # Each cell should have one value per gene
    )

    # Get condition for each cell
    cell_conditions = df[['cell_id', 'Condition']].drop_duplicates().set_index('cell_id')
    intron_matrix = intron_matrix.join(cell_conditions)

    print(f"  Matrix shape: {intron_matrix.shape[0]} cells × {len(available_genes)} genes")
    print(f"  Cells with data for all genes: {intron_matrix[available_genes].dropna().shape[0]}")

    # --- Overall correlation matrix ---
    print("\nCalculating correlation matrices...")

    # Use only cells with data for at least 2 genes
    valid_cells = intron_matrix[available_genes].dropna(thresh=2)
    corr_overall = valid_cells[available_genes].corr(method='spearman')

    print(f"\nOverall Spearman correlations (n={len(valid_cells)} cells):")
    print(corr_overall.round(3).to_string())

    # --- Condition-specific correlations ---
    corr_ctrl = None
    corr_als = None

    ctrl_cells = intron_matrix[intron_matrix['Condition'] == ctrl_label][available_genes].dropna(thresh=2)
    als_cells = intron_matrix[intron_matrix['Condition'] == als_label][available_genes].dropna(thresh=2)

    if len(ctrl_cells) >= 10:
        corr_ctrl = ctrl_cells.corr(method='spearman')
        print(f"\n{ctrl_label} correlations (n={len(ctrl_cells)} cells):")
        print(corr_ctrl.round(3).to_string())
    else:
        print(f"\n{ctrl_label}: Too few cells ({len(ctrl_cells)}) for correlation")

    if len(als_cells) >= 10:
        corr_als = als_cells.corr(method='spearman')
        print(f"\n{als_label} correlations (n={len(als_cells)} cells):")
        print(corr_als.round(3).to_string())
    else:
        print(f"\n{als_label}: Too few cells ({len(als_cells)}) for correlation")

    # --- Visualizations ---

    # Plot 1: Overall correlation heatmap
    fig1, ax1 = plt.subplots(figsize=(8, 7))
    mask = np.triu(np.ones_like(corr_overall, dtype=bool), k=1)
    sns.heatmap(corr_overall, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=ax1,
                square=True, linewidths=0.5,
                cbar_kws={'label': 'Spearman ρ'})
    ax1.set_title('Cross-Gene Intron Retention Correlations\n(All Cells)',
                  fontsize=12, fontweight='bold')
    plt.tight_layout()

    if outdir:
        plt.savefig(os.path.join(outdir, 'cross_gene_correlation_overall.png'),
                   dpi=Config.FIGURE_DPI, bbox_inches='tight')
        print(f"\n  Saved: cross_gene_correlation_overall.png")

    # Plot 2: Side-by-side condition-specific correlations
    if corr_ctrl is not None and corr_als is not None:
        fig2, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Control
        sns.heatmap(corr_ctrl, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, ax=axes[0],
                    square=True, linewidths=0.5)
        axes[0].set_title(f'{ctrl_label}\n(n={len(ctrl_cells)} cells)',
                         fontsize=11, fontweight='bold')

        # ALS
        sns.heatmap(corr_als, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, ax=axes[1],
                    square=True, linewidths=0.5)
        axes[1].set_title(f'{als_label}\n(n={len(als_cells)} cells)',
                         fontsize=11, fontweight='bold')

        # Difference (ALS - Control)
        corr_diff = corr_als - corr_ctrl
        sns.heatmap(corr_diff, annot=True, fmt='.2f', cmap='PiYG',
                    center=0, vmin=-0.5, vmax=0.5, ax=axes[2],
                    square=True, linewidths=0.5)
        axes[2].set_title(f'Difference ({als_label} - {ctrl_label})',
                         fontsize=11, fontweight='bold')

        plt.suptitle('Condition-Specific Cross-Gene Correlations',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if outdir:
            plt.savefig(os.path.join(outdir, 'cross_gene_correlation_by_condition.png'),
                       dpi=Config.FIGURE_DPI, bbox_inches='tight')
            print(f"  Saved: cross_gene_correlation_by_condition.png")

    # Plot 3: Pairwise scatter plots for top gene pairs
    # Focus on STMN2 correlations if available
    focus_gene = 'STMN2' if 'STMN2' in available_genes else available_genes[0]
    other_genes = [g for g in available_genes if g != focus_gene][:4]  # Max 4 others

    if len(other_genes) > 0:
        n_cols = min(len(other_genes), 4)
        fig3, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4))
        if n_cols == 1:
            axes = [axes]

        for i, other_gene in enumerate(other_genes):
            ax = axes[i]

            # Get cells with both genes
            pair_data = intron_matrix[[focus_gene, other_gene, 'Condition']].dropna()

            # Plot by condition
            for cond, color, marker in [(ctrl_label, 'steelblue', 'o'), (als_label, 'firebrick', '^')]:
                cond_data = pair_data[pair_data['Condition'] == cond]
                ax.scatter(cond_data[focus_gene], cond_data[other_gene],
                          c=color, marker=marker, alpha=0.4, s=20, label=cond)

            # Add correlation values
            r_overall = pair_data[[focus_gene, other_gene]].corr(method='spearman').iloc[0,1]
            ax.set_xlabel(f'{focus_gene} Intron Fraction')
            ax.set_ylabel(f'{other_gene} Intron Fraction')
            ax.set_title(f'{focus_gene} vs {other_gene}\nρ = {r_overall:.3f}')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(alpha=0.3)

        plt.suptitle(f'Pairwise Intron Retention: {focus_gene} vs Other TDP-43 Targets',
                    fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()

        if outdir:
            plt.savefig(os.path.join(outdir, f'cross_gene_scatter_{focus_gene}.png'),
                       dpi=Config.FIGURE_DPI, bbox_inches='tight')
            print(f"  Saved: cross_gene_scatter_{focus_gene}.png")

    # --- Identify cells with coordinated high retention ---
    print("\n" + "-"*60)
    print("COORDINATED RETENTION ANALYSIS")
    print("-"*60)

    retention_threshold = Config.RETENTION_THRESHOLD

    # Count how many genes show retention per cell
    retention_counts = (intron_matrix[available_genes] > retention_threshold).sum(axis=1)
    intron_matrix['n_genes_with_retention'] = retention_counts

    # Summary by condition
    print(f"\nCells with retention (IntronFraction > {retention_threshold}) in multiple genes:")
    for cond in [ctrl_label, als_label]:
        cond_data = intron_matrix[intron_matrix['Condition'] == cond]
        print(f"\n  {cond}:")
        for n_genes in range(len(available_genes) + 1):
            n_cells = (cond_data['n_genes_with_retention'] == n_genes).sum()
            pct = 100 * n_cells / len(cond_data) if len(cond_data) > 0 else 0
            print(f"    {n_genes} genes: {n_cells} cells ({pct:.1f}%)")

    # Plot 4: Distribution of coordinated retention
    fig4, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax = axes[0]
    for cond, color in [(ctrl_label, 'steelblue'), (als_label, 'firebrick')]:
        cond_data = intron_matrix[intron_matrix['Condition'] == cond]['n_genes_with_retention']
        ax.hist(cond_data, bins=np.arange(-0.5, len(available_genes)+1.5, 1),
               alpha=0.6, label=cond, color=color, edgecolor='black')
    ax.set_xlabel('Number of Genes with Intron Retention')
    ax.set_ylabel('Number of Cells')
    ax.set_title(f'Distribution of Coordinated Retention\n(threshold > {retention_threshold})')
    ax.legend()
    ax.set_xticks(range(len(available_genes)+1))

    # Proportion with 2+ genes
    ax = axes[1]
    summary_data = []
    for cond in [ctrl_label, als_label]:
        cond_data = intron_matrix[intron_matrix['Condition'] == cond]
        for threshold_n in range(1, len(available_genes)+1):
            pct = 100 * (cond_data['n_genes_with_retention'] >= threshold_n).sum() / len(cond_data)
            summary_data.append({'Condition': cond, 'Min_Genes': threshold_n, 'Percent': pct})

    summary_plot_df = pd.DataFrame(summary_data)
    summary_pivot = summary_plot_df.pivot(index='Min_Genes', columns='Condition', values='Percent')
    summary_pivot[[ctrl_label, als_label]].plot(kind='bar', ax=ax, color=['steelblue', 'firebrick'])
    ax.set_xlabel('Minimum Number of Genes with Retention')
    ax.set_ylabel('% of Cells')
    ax.set_title('Cells with Coordinated Multi-Gene Retention')
    ax.legend(title='Condition')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()

    if outdir:
        plt.savefig(os.path.join(outdir, 'coordinated_retention_distribution.png'),
                   dpi=Config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: coordinated_retention_distribution.png")

    # Save correlation tables
    if outdir:
        corr_overall.to_csv(os.path.join(outdir, 'cross_gene_correlation_overall.csv'))
        if corr_ctrl is not None:
            corr_ctrl.to_csv(os.path.join(outdir, f'cross_gene_correlation_{ctrl_label}.csv'))
        if corr_als is not None:
            corr_als.to_csv(os.path.join(outdir, f'cross_gene_correlation_{als_label}.csv'))
        if corr_ctrl is not None and corr_als is not None:
            corr_diff.to_csv(os.path.join(outdir, 'cross_gene_correlation_difference.csv'))
        print(f"  Saved: correlation tables (.csv)")

    results = {
        'correlation_overall': corr_overall,
        'correlation_control': corr_ctrl,
        'correlation_als': corr_als,
        'intron_matrix': intron_matrix,
        'available_genes': available_genes
    }

    return results


#%%
# =============================================================================
# EXPORT STMN2 Log2IntronExonRatio TO ADATA
# =============================================================================

def add_stmn2_log2ratio_to_adata(adata, df):
    """
    Add STMN2-specific Log2IntronExonRatio as obs column to adata.

    Parameters
    ----------
    adata : AnnData
        Motorneuron adata (index like MN_1_0, MN_2_0)
    df : DataFrame
        Long-form intron data with columns: geneName, cell_id, Log2IntronExonRatio

    Returns
    -------
    adata : AnnData
        Updated adata with STMN2_Log2IntronExonRatio column
    """
    print("\n" + "-"*60)
    print("ADDING STMN2 Log2IntronExonRatio TO ADATA")
    print("-"*60)

    # Filter for STMN2
    stmn2_df = df[df['geneName'] == 'STMN2'][['cell_id', 'Log2IntronExonRatio']].copy()

    if len(stmn2_df) == 0:
        print("WARNING: STMN2 not found in data!")
        adata.obs['STMN2_Log2IntronExonRatio'] = np.nan
        return adata

    print(f"  Found STMN2 data for {len(stmn2_df)} cells")

    # Handle duplicates (if any) by taking mean
    stmn2_df = stmn2_df.groupby('cell_id')['Log2IntronExonRatio'].mean().reset_index()
    stmn2_df = stmn2_df.set_index('cell_id')

    # Map to adata obs
    adata.obs['STMN2_Log2IntronExonRatio'] = np.nan
    matching_cells = [idx for idx in adata.obs_names if idx in stmn2_df.index]

    for idx in matching_cells:
        adata.obs.loc[idx, 'STMN2_Log2IntronExonRatio'] = stmn2_df.loc[idx, 'Log2IntronExonRatio']

    n_mapped = adata.obs['STMN2_Log2IntronExonRatio'].notna().sum()
    print(f"  Mapped STMN2_Log2IntronExonRatio to {n_mapped}/{len(adata)} cells")

    # Summary stats
    valid_values = adata.obs['STMN2_Log2IntronExonRatio'].dropna()
    if len(valid_values) > 0:
        print(f"  Value range: [{valid_values.min():.3f}, {valid_values.max():.3f}]")
        print(f"  Median: {valid_values.median():.3f}")

    return adata


#%%
# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def main():
    """Main analysis pipeline."""

    print("\n" + "="*80)
    print("  COMPREHENSIVE MOTOR NEURON TRANSCRIPTOME INTEGRITY ANALYSIS")
    print("="*80)

    # Create output directories
    dirs = create_output_dirs(Config.OUTPUT_DIR)

    # 1. Load and prepare data (with fast pre-filtering using Config thresholds)
    adata, df, prefilter_stats = load_and_prepare_data(
        Config.INPUT_H5AD,
        sample_col=Config.SAMPLE_COL,
        condition_col=Config.CONDITION_COL
        # Uses Config.MIN_CELLS_PER_GENE_PER_SAMPLE and Config.MIN_SAMPLES_PER_GROUP
    )

    # Check if pre-filtering returned empty data
    if len(df) == 0:
        print("\n" + "!"*80)
        print("  ERROR: No genes passed pre-filtering!")
        print("  Consider relaxing the filtering thresholds in Config.")
        print("!"*80)
        return None

    # Save pre-filtering stats
    pd.DataFrame([prefilter_stats]).to_csv(
        os.path.join(dirs['tables'], 'prefiltering_stats.csv'),
        index=False
    )

    # DIAGNOSTIC: Check sample structure
    print("\n" + "="*80)
    print("DIAGNOSTIC: DATA STRUCTURE CHECK")
    print("="*80)
    print(f"\nUnique samples in data: {df['sample'].nunique()}")
    print(f"Samples by condition:")
    print(df.groupby('Condition')['sample'].nunique())
    print(f"\nSample IDs by condition:")
    for cond in df['Condition'].unique():
        samples = df[df['Condition'] == cond]['sample'].unique()
        print(f"  {cond}: {list(samples[:10])}")  # Show first 10
    print(f"\nTotal observations: {len(df):,}")
    print("="*80)

    # 2. Additional filtering
    if Config.MIN_CELLS_PER_GROUP > 0 or Config.USE_RELATIVE_FILTERING:
        df_filtered, valid_genes, filter_stats = filter_genes(
            df,
            min_cells_per_sample=Config.MIN_CELLS_PER_GENE_PER_SAMPLE,
            min_samples_per_group=Config.MIN_SAMPLES_PER_GROUP,
            min_cells_per_group=Config.MIN_CELLS_PER_GROUP,
            use_relative=Config.USE_RELATIVE_FILTERING,
            min_cells_fraction_per_sample=Config.MIN_CELLS_FRACTION_PER_SAMPLE,
            min_cells_fraction_per_group=Config.MIN_CELLS_FRACTION_PER_GROUP
        )
    else:
        # Pre-filtering already done, use df as-is
        df_filtered = df
        valid_genes = df['geneName'].unique().tolist()
        filter_stats = {
            'n_initial': prefilter_stats['n_initial'],
            'n_final': len(valid_genes),
            'valid_genes': valid_genes
        }

    n_genes_remaining = len(valid_genes)
    print(f"\n  ✓ {n_genes_remaining:,} genes ready for analysis")

    # Save filtering stats
    pd.DataFrame([filter_stats]).to_csv(
        os.path.join(dirs['tables'], 'filtering_stats.csv'),
        index=False
    )

    # 3. Aggregate to sample level - PROPORTION MODE
    # Instead of testing median IntronFraction, test proportion of cells with retention
    agg_df = aggregate_to_sample_level(
        df_filtered,
        metric='IntronFraction',
        agg_func='proportion',
        threshold=Config.RETENTION_THRESHOLD
    )

    # DIAGNOSTIC: Check aggregated data structure
    print("\n" + "="*80)
    print("DIAGNOSTIC: AGGREGATED DATA CHECK")
    print("="*80)
    print(f"\nAggregated data shape: {agg_df.shape}")
    print(f"Unique genes in aggregated data: {agg_df['geneName'].nunique()}")
    print(f"\nSamples per condition in aggregated data:")
    print(agg_df.groupby('Condition')['sample'].nunique())
    print(f"\nFor a random gene, how many samples do we have?")
    if len(agg_df) > 0:
        test_gene = agg_df['geneName'].iloc[0]
        test_gene_data = agg_df[agg_df['geneName'] == test_gene]
        print(f"  Gene: {test_gene}")
        print(f"  Total rows: {len(test_gene_data)}")
        print(f"  Samples by condition:")
        print(test_gene_data.groupby('Condition')['sample'].apply(list))
    print("="*80)

    # Save aggregated data
    agg_df.to_csv(os.path.join(dirs['tables'], 'sample_level_aggregated_data.csv'), index=False)

    # 4. Genome-wide analysis
    # Test retention proportion (% of cells with IntronFraction > threshold)
    results_df = genome_wide_analysis(
        agg_df,
        valid_genes,
        metric='RetentionProportion',
        fdr_threshold=Config.FDR_THRESHOLD,
        min_effect_size=Config.EFFECT_SIZE_THRESHOLD
    )

    # Save results
    results_df.to_csv(
        os.path.join(dirs['tables'], 'genome_wide_results.csv'),
        index=False
    )

    # Save significant genes only
    sig_results = results_df[results_df['significant']].copy()
    sig_results.to_csv(
        os.path.join(dirs['tables'], 'significant_genes.csv'),
        index=False
    )

    # 4b. Effect size ranking analysis
    print_section("EFFECT SIZE ANALYSIS")
    effect_size_df = rank_genes_by_effect(df_filtered, min_cells=Config.MIN_CELLS_PER_GROUP)
    effect_size_df.to_csv(
        os.path.join(dirs['tables'], 'genes_ranked_by_effect_size.csv'),
        index=False
    )

    # 4c. STMN2 correlation analysis
    print_section("STMN2 CORRELATION ANALYSIS")
    stmn2_corr_df = find_stmn2_correlated_genes(df_filtered, min_cells=Config.MIN_CELLS_PER_GROUP)
    if len(stmn2_corr_df) > 0:
        stmn2_corr_df.to_csv(
            os.path.join(dirs['tables'], 'stmn2_correlated_genes.csv'),
            index=False
        )

    # 5. Generate visualizations
    plot_qc_summary(df_filtered, valid_genes, outdir=dirs['qc'])
    # Volcano plot removed - not useful for intron fraction analysis

    if results_df['significant'].sum() > 0:
        plot_top_genes_boxplot(
            df_filtered, agg_df, results_df,
            top_n=min(Config.MAX_GENES_TO_PLOT, results_df['significant'].sum()),
            metric='RetentionProportion',
            outdir=dirs['figures']
        )

    # 6. Export data for R analysis (GLMM)
    print_section("EXPORTING DATA FOR R ANALYSIS")

    # Export cell-level data for candidate genes (for R mixed models)
    candidate_genes_data = df_filtered[df_filtered['geneName'].isin(Config.CANDIDATE_GENES)].copy()

    if len(candidate_genes_data) > 0:
        # Select relevant columns for R analysis
        r_export = candidate_genes_data[[
            'geneName', 'sample', 'Condition',
            'MIDCount', 'ExonCount', 'IntronCount',
            'IntronFraction', 'ExonFraction', 'Log2IntronExonRatio'
        ]].copy()

        # Save for R (in the main output directory)
        r_export_path = os.path.join(Config.OUTPUT_DIR, 'valid_genes_df.csv')
        r_export.to_csv(r_export_path, index=False)

        print(f"Exported {len(candidate_genes_data):,} cells from {len(Config.CANDIDATE_GENES)} candidate genes")
        print(f"Saved to: {r_export_path}")
        print(f"\nCandidate genes exported:")
        for gene in Config.CANDIDATE_GENES:
            n_cells = len(candidate_genes_data[candidate_genes_data['geneName'] == gene])
            if n_cells > 0:
                print(f"  - {gene}: {n_cells} cells")
            else:
                print(f"  - {gene}: NOT FOUND in filtered data")

        print(f"\nRun the R script for in-depth GLMM analysis:")
        print(f"  Rscript C:/Users/catta/Desktop/ALS/THESIS/final_scripts/LMM-binGLMM_presentation.R")

        # Generate candidate genes overview plots
        plot_candidate_genes_overview(
            candidate_genes_data,
            Config.CANDIDATE_GENES,
            outdir=dirs['candidate_genes']
        )

        # Generate comprehensive STMN2 figure
        plot_stmn2_comprehensive(
            candidate_genes_data,
            outdir=dirs['candidate_genes']
        )

        # Cross-gene correlation analysis
        correlation_results = cross_gene_correlation_analysis(
            candidate_genes_data,
            Config.CANDIDATE_GENES,
            outdir=dirs['candidate_genes']
        )
    else:
        print("  No candidate genes found in filtered data.")
        correlation_results = None

    # 7. Add STMN2 Log2IntronExonRatio to adata and save
    adata = add_stmn2_log2ratio_to_adata(adata, df_filtered)

    # Save updated adata with STMN2_Log2IntronExonRatio
    output_h5ad = Config.INPUT_H5AD.replace('.h5ad', '_with_STMN2ratio.h5ad')
    adata.write(output_h5ad)
    print(f"\nSaved updated adata with STMN2_Log2IntronExonRatio to:")
    print(f"  {output_h5ad}")

    # 8. Final summary
    print_section("ANALYSIS COMPLETE")
    print(f"\nResults saved to: {Config.OUTPUT_DIR}/")
    print(f"\nKey outputs:")
    print(f"  - genome_wide_results.csv: All tested genes with statistics")
    print(f"  - significant_genes.csv: Genes passing FDR threshold")
    print(f"  - genes_ranked_by_effect_size.csv: Genes ranked by Cohen's d effect size")
    print(f"  - stmn2_correlated_genes.csv: Genes correlated with STMN2 retention")
    print(f"  - figures/: Volcano plot, boxplots, QC plots")
    print(f"  - qc/: Quality control summary plots")
    print(f"  - candidate_genes/: Overview plots and cross-gene correlations")
    print(f"\nFor in-depth candidate gene analysis:")
    print(f"  Run the R script: LMM-binGLMM_presentation.R")
    print(f"  (Uses GLMM with binomial family for rigorous statistical testing)")

    return {
        'adata': adata,
        'df': df_filtered,
        'agg_df': agg_df,
        'results_df': results_df,
        'effect_size_df': effect_size_df,
        'stmn2_corr_df': stmn2_corr_df,
        'correlation_results': correlation_results,
        'dirs': dirs
    }


#%%
# =============================================================================
# EXECUTE
# =============================================================================

if __name__ == '__main__':
    # Run the complete analysis
    analysis_output = main()

    print("\n" + "="*80)
    print("  All analyses complete!")
    print("="*80)

