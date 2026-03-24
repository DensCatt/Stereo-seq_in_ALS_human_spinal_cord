#%% COMPREHENSIVE QUALITY & EXPLORATORY ANALYSIS OF MOTOR NEURONS
# This script performs detailed QC analysis for motor neuron cells
# Data contains 3 layers: counts, exon, intron
# Analysis is tailored for downstream pseudobulk DEA

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set working directory
os.chdir("C:\\Users\\catta\\Desktop\\ALS\\THESIS")
motorneuron_data_path = "motorneurons_bin20_ALL_LAYERS.h5ad"

# Create output folder for QC figures
QC_DIR = "MNs_QC"
os.makedirs(QC_DIR, exist_ok=True)

# Configure plotting
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100, facecolor='white', frameon=False)
sns.set_style("whitegrid")

#%% ============================================================================
# 1. DATA LOADING AND INITIAL INSPECTION
# =============================================================================
print("\n" + "="*80)
print("1. DATA LOADING AND INITIAL INSPECTION")
print("="*80)

adata = sc.read_h5ad(motorneuron_data_path)

# Set gene names as index
adata.var = adata.var.reset_index()
adata.var = adata.var.set_index("real_gene_name")

#--------------------------------------------
# Check current samples
print("Samples before filtering:")
print(adata.obs['sample'].value_counts())
# Remove s12-186
adata = adata[adata.obs['sample'] != 's12-186'].copy()
# Verify removal
print("\nSamples after filtering:")
print(adata.obs['sample'].value_counts())
print(f"\nTotal cells remaining: {adata.n_obs}")
#--------------------------------------------

print(f"\nDataset shape: {adata.n_obs} cells × {adata.n_vars} genes")
print(f"\nAvailable layers: {list(adata.layers.keys())}")
print(f"\nSamples: {adata.obs['sample'].unique()}")
print(f"\nConditions: {adata.obs['condition'].unique()}")

# Check data structure
print("\n--- Data Structure ---")
print(f"X matrix type: {type(adata.X)}")
print(f"X matrix shape: {adata.X.shape}")
print("\nObservations (cells) metadata columns:")
print(adata.obs.columns.tolist())
print("\nVariables (genes) metadata columns:")
print(adata.var.columns.tolist())

#%% ============================================================================
# 2. LAYER INSPECTION AND COMPARISON
# =============================================================================
print("\n" + "="*80)
print("2. LAYER INSPECTION (counts, exon, intron)")
print("="*80)

# Check if layers exist
expected_layers = ['counts', 'exon', 'intron']
available_layers = [l for l in expected_layers if l in adata.layers.keys()]

if len(available_layers) == 0:
    print("\nWARNING: Expected layers (counts, exon, intron) not found!")
    print(f"Available layers: {list(adata.layers.keys())}")
    print("\nNote: Will analyze available data structure")
else:
    print(f"\nFound layers: {available_layers}")

# Display info about each layer
for layer_name in adata.layers.keys():
    layer = adata.layers[layer_name]
    print(f"\n--- Layer: {layer_name} ---")
    print(f"Type: {type(layer)}")
    print(f"Shape: {layer.shape}")
    if sparse.issparse(layer):
        print(f"Sparsity: {(1 - layer.nnz / (layer.shape[0] * layer.shape[1])) * 100:.2f}%")
    print(f"Mean: {layer.mean():.2f}")
    print(f"Max: {np.max(layer.toarray() if sparse.issparse(layer) else layer):.2f}")

#%% ============================================================================
# 3. SAMPLE METADATA SETUP
# =============================================================================
print("\n" + "="*80)
print("3. ADDING SAMPLE METADATA")
print("="*80)

# Add batch information
sample_batch = {
    "s09-055": "03",
    "s13-133": "03",
    "s13-047": "06",
    "s15-025": "06",
    "s15-051": "07",
    "s18-070": "07"
}

# Add sex information
sample_sex = {
    "s09-055": "f",
    "s13-133": "f",
    "s13-047": "m",
    "s15-025": "m",
    "s15-051": "f",
    "s18-070": "f"
}

# Add age information
sample_age = {
    "s09-055": 54,
    "s13-133": 54,
    "s13-047": 73,
    "s15-025": 74,
    "s15-051": 72,
    "s18-070": 70
}

adata.obs["batch"] = adata.obs["sample"].map(sample_batch)
adata.obs["sex"] = adata.obs["sample"].map(sample_sex)
adata.obs["age"] = adata.obs["sample"].map(sample_age).astype(int)

print("\nSample metadata summary:")
print(adata.obs[['sample', 'condition', 'batch', 'sex', 'age']].drop_duplicates().sort_values('sample'))

#%% ============================================================================
# 4. CELL DISTRIBUTION ACROSS SAMPLES AND CONDITIONS
# =============================================================================
print("\n" + "="*80)
print("4. CELL DISTRIBUTION ANALYSIS")
print("="*80)

# Cells per sample
cells_per_sample = adata.obs['sample'].value_counts().sort_index()
print("\nCells per sample:")
print(cells_per_sample)

# Cells per condition
cells_per_condition = adata.obs['condition'].value_counts()
print("\nCells per condition:")
print(cells_per_condition)

# Cells per condition per sample
cells_per_cond_sample = adata.obs.groupby(['condition', 'sample']).size()
print("\nCells per condition-sample combination:")
print(cells_per_cond_sample)

# Plot cell distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Cells per sample
mn_counts_df = cells_per_sample.reset_index()
mn_counts_df.columns = ['sample', 'MN_count']
sns.barplot(data=mn_counts_df, x='sample', y='MN_count', palette='viridis', ax=axes[0])
axes[0].set_title('Motor Neurons per Sample', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Sample')
axes[0].set_ylabel('Number of Motor Neurons')
axes[0].tick_params(axis='x', rotation=45)
for i in axes[0].containers:
    axes[0].bar_label(i, fmt='%d', label_type='edge', fontsize=9)

# Plot 2: Cells per condition
condition_df = cells_per_condition.reset_index()
condition_df.columns = ['condition', 'MN_count']
sns.barplot(data=condition_df, x='condition', y='MN_count', palette=sns.color_palette('Set2')[1::-1], ax=axes[1])
axes[1].set_title('Motor Neurons per Condition', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Condition')
axes[1].set_ylabel('Number of Motor Neurons')
for i in axes[1].containers:
    axes[1].bar_label(i, fmt='%d', label_type='edge', fontsize=10)

# Plot 3: Cells per sample colored by condition
sample_condition_df = adata.obs.groupby(['sample', 'condition']).size().reset_index(name='MN_count')
sns.barplot(data=sample_condition_df, x='sample', y='MN_count', hue='condition', palette=sns.color_palette('Set2')[1::-1], ax=axes[2])
axes[2].set_title('Motor Neurons per Sample by Condition', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Sample')
axes[2].set_ylabel('Number of Motor Neurons')
axes[2].tick_params(axis='x', rotation=45)
axes[2].legend(title='Condition')

plt.tight_layout()
plt.show()

#%% ============================================================================
# 5. BASIC QC METRICS CALCULATION
# =============================================================================
print("\n" + "="*80)
print("5. CALCULATING QC METRICS")
print("="*80)

# Identify mitochondrial, ribosomal, and hemoglobin genes
adata.var["mt"] = adata.var_names.str.startswith("MT-")
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

print(f"\nMitochondrial genes: {adata.var['mt'].sum()}")
print(f"Ribosomal genes: {adata.var['ribo'].sum()}")
print(f"Hemoglobin genes: {adata.var['hb'].sum()}")

# Calculate QC metrics
sc.pp.calculate_qc_metrics(
    adata,
    qc_vars=["mt", "ribo", "hb"],
    inplace=True,
    percent_top=[20],
    log1p=True
)

# Display QC metrics summary (BEFORE filtering)
print("\n--- QC Metrics Summary (before filtering) ---")
qc_metrics = ['total_counts', 'n_genes_by_counts', 'pct_counts_mt',
              'pct_counts_ribo', 'pct_counts_hb']
print(adata.obs[qc_metrics].describe())

#%% ============================================================================
# 6. CELL-LEVEL QUALITY FILTERING
# =============================================================================
print("\n" + "="*80)
print("6. CELL-LEVEL QUALITY FILTERING")
print("="*80)

print(f"\nCells before filtering: {adata.n_obs}")

# --- Define filtering thresholds ---
MIN_COUNTS = 1500          # Minimum total counts per cell
MAX_COUNTS = None          # Maximum total counts (None = no upper limit, useful for doublet removal)
MIN_GENES = 500            # Minimum genes detected per cell
MAX_GENES = None           # Maximum genes detected (None = no upper limit)
MAX_MT_PCT = 25            # Maximum mitochondrial percentage
MAX_RIBO_PCT = 50          # Maximum ribosomal percentage (high ribo can indicate low-quality/dying cells)
MAX_HB_PCT = 5             # Maximum hemoglobin percentage (high hb indicates blood contamination)

# Store initial cell count per sample for comparison
initial_cells_per_sample = adata.obs['sample'].value_counts()

# Create filter masks
print("\n--- Applying Cell Filters ---")

# Filter 1: Minimum counts
mask_min_counts = adata.obs['total_counts'] >= MIN_COUNTS
n_fail_min_counts = (~mask_min_counts).sum()
print(f"  Cells with < {MIN_COUNTS} counts: {n_fail_min_counts}")

# Filter 2: Maximum counts (optional - useful for removing doublets)
if MAX_COUNTS is not None:
    mask_max_counts = adata.obs['total_counts'] <= MAX_COUNTS
    n_fail_max_counts = (~mask_max_counts).sum()
    print(f"  Cells with > {MAX_COUNTS} counts: {n_fail_max_counts}")
else:
    mask_max_counts = np.ones(adata.n_obs, dtype=bool)

# Filter 3: Minimum genes detected
mask_min_genes = adata.obs['n_genes_by_counts'] >= MIN_GENES
n_fail_min_genes = (~mask_min_genes).sum()
print(f"  Cells with < {MIN_GENES} genes: {n_fail_min_genes}")

# Filter 4: Maximum genes (optional - useful for removing doublets)
if MAX_GENES is not None:
    mask_max_genes = adata.obs['n_genes_by_counts'] <= MAX_GENES
    n_fail_max_genes = (~mask_max_genes).sum()
    print(f"  Cells with > {MAX_GENES} genes: {n_fail_max_genes}")
else:
    mask_max_genes = np.ones(adata.n_obs, dtype=bool)

# Filter 5: Maximum MT percentage
mask_mt = adata.obs['pct_counts_mt'] <= MAX_MT_PCT
n_fail_mt = (~mask_mt).sum()
print(f"  Cells with > {MAX_MT_PCT}% MT: {n_fail_mt}")

# Filter 6: Maximum ribosomal percentage
mask_ribo = adata.obs['pct_counts_ribo'] <= MAX_RIBO_PCT
n_fail_ribo = (~mask_ribo).sum()
print(f"  Cells with > {MAX_RIBO_PCT}% ribosomal: {n_fail_ribo}")

# Filter 7: Maximum hemoglobin percentage
mask_hb = adata.obs['pct_counts_hb'] <= MAX_HB_PCT
n_fail_hb = (~mask_hb).sum()
print(f"  Cells with > {MAX_HB_PCT}% hemoglobin: {n_fail_hb}")

# Combine all masks
combined_mask = (mask_min_counts & mask_max_counts & mask_min_genes &
                 mask_max_genes & mask_mt & mask_ribo & mask_hb)

n_cells_removed = (~combined_mask).sum()
print(f"\n  Total cells failing QC: {n_cells_removed}")

# Apply filter
if n_cells_removed > 0:
    adata = adata[combined_mask, :].copy()
    print(f"\nCells after filtering: {adata.n_obs}")

    # Show cells removed per sample
    final_cells_per_sample = adata.obs['sample'].value_counts()
    cells_removed_per_sample = initial_cells_per_sample - final_cells_per_sample.reindex(initial_cells_per_sample.index, fill_value=0)

    print("\n--- Cells Removed per Sample ---")
    removal_summary = []
    for sample in sorted(initial_cells_per_sample.index):
        initial = initial_cells_per_sample[sample]
        removed = cells_removed_per_sample[sample]
        remaining = initial - removed
        pct_removed = (removed / initial * 100) if initial > 0 else 0
        removal_summary.append({'sample': sample, 'initial': initial, 'remaining': remaining,
                                'removed': removed, 'pct_removed': pct_removed})
        print(f"  {sample}: {initial} → {remaining} ({removed} removed, {pct_removed:.1f}%)")

    # Check if any sample lost too many cells
    removal_df = pd.DataFrame(removal_summary)
    high_removal_samples = removal_df[removal_df['pct_removed'] > 50]
    if len(high_removal_samples) > 0:
        print(f"\n⚠ WARNING: The following samples lost >50% of cells:")
        print(high_removal_samples[['sample', 'initial', 'remaining', 'pct_removed']])
else:
    print("\n✓ All cells passed quality control!")

# Display QC metrics summary after filtering
print("\n--- QC Metrics Summary (after filtering) ---")
print(adata.obs[qc_metrics].describe())

print(f"\n--- Cell Filtering Summary ---")
print(f"Thresholds used:")
print(f"  - Min counts: {MIN_COUNTS}")
print(f"  - Min genes: {MIN_GENES}")
print(f"  - Max MT%: {MAX_MT_PCT}")
print(f"  - Max ribosomal%: {MAX_RIBO_PCT}")
print(f"  - Max hemoglobin%: {MAX_HB_PCT}")

#%% ============================================================================
# 7. QC METRICS VISUALIZATION
# =============================================================================
print("\n" + "="*80)
print("7. QC METRICS VISUALIZATION")
print("="*80)

# Violin plots by sample
sc.pl.violin(adata, keys=['total_counts', 'n_genes_by_counts', 'pct_counts_mt'],
             jitter=False, multi_panel=True, rotation=45, groupby="sample")

# Violin plots by condition
sc.pl.violin(adata, keys=['total_counts', 'n_genes_by_counts', 'pct_counts_mt'],
             jitter=False, multi_panel=True, rotation=45, groupby="condition")

# Violin plots by sex
sc.pl.violin(adata, keys=['total_counts', 'n_genes_by_counts', 'pct_counts_mt'],
             jitter=False, multi_panel=True, rotation=45, groupby="sex")

# Violin plots by batch
sc.pl.violin(adata, keys=['total_counts', 'n_genes_by_counts', 'pct_counts_mt'],
             jitter=False, multi_panel=True, rotation=45, groupby="batch")

#%% Scatter plots to identify relationships
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Total counts vs genes detected
sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts",
              color="condition", ax=axes[0, 0], show=False)
axes[0, 0].set_title('Total Counts vs Genes Detected')

# Total counts vs MT percentage
sc.pl.scatter(adata, x="total_counts", y="pct_counts_mt",
              color="pct_counts_mt", ax=axes[0, 1], show=False)
axes[0, 1].set_title('Total Counts vs MT%')

# Genes detected vs MT percentage
sc.pl.scatter(adata, x="n_genes_by_counts", y="pct_counts_mt",
              color="condition", ax=axes[1, 0], show=False)
axes[1, 0].set_title('Genes Detected vs MT%')

# Total counts vs ribosomal percentage
sc.pl.scatter(adata, x="total_counts", y="pct_counts_ribo",
              color="condition", ax=axes[1, 1], show=False)
axes[1, 1].set_title('Total Counts vs Ribosomal%')

plt.tight_layout()
plt.show()

#%% ============================================================================
# 8. IDENTIFY AND REMOVE BAD QUALITY SAMPLES
# =============================================================================
print("\n" + "="*80)
print("8. SAMPLE QUALITY ASSESSMENT AND FILTERING")
print("="*80)

# Calculate per-sample QC metrics
sample_qc = adata.obs.groupby('sample').agg({
    'total_counts': ['mean', 'median', 'std'],
    'n_genes_by_counts': ['mean', 'median', 'std'],
    'pct_counts_mt': ['mean', 'median', 'std'],
    'pct_counts_ribo': ['mean', 'median'],
}).round(2)

# Flatten column names
sample_qc.columns = ['_'.join(col).strip() for col in sample_qc.columns.values]

# Add cell count per sample
sample_qc['n_cells'] = adata.obs['sample'].value_counts()

# Add condition, batch, sex, age
sample_info = adata.obs.groupby('sample')[['condition', 'batch', 'sex', 'age']].first()
sample_qc = pd.concat([sample_qc, sample_info], axis=1)

print("\n--- Per-Sample QC Metrics ---")
print(sample_qc)

# Identify outlier samples using MAD (Median Absolute Deviation)
def is_outlier_mad(data, threshold=3):
    """Identify outliers using MAD method"""
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad == 0:
        return np.zeros(len(data), dtype=bool)
    modified_z_scores = 0.6745 * (data - median) / mad
    return np.abs(modified_z_scores) > threshold

# Initialize flags
bad_samples = set()
sample_flags = pd.DataFrame(index=sample_qc.index)

# Flag 1: Very few cells (< 10 cells)
MIN_CELLS_PER_SAMPLE = 10
sample_flags['too_few_cells'] = sample_qc['n_cells'] < MIN_CELLS_PER_SAMPLE
if sample_flags['too_few_cells'].any():
    flagged = sample_flags[sample_flags['too_few_cells']].index.tolist()
    bad_samples.update(flagged)
    print(f"\n⚠ Samples with < {MIN_CELLS_PER_SAMPLE} cells: {flagged}")

# Flag 2: Low total counts (outliers on the low end)
low_counts_outliers = is_outlier_mad(sample_qc['total_counts_mean'].values)
# Only flag if it's LOW (below median)
low_counts_mask = (sample_qc['total_counts_mean'] < sample_qc['total_counts_mean'].median()) & low_counts_outliers
sample_flags['low_total_counts'] = low_counts_mask
if sample_flags['low_total_counts'].any():
    flagged = sample_flags[sample_flags['low_total_counts']].index.tolist()
    bad_samples.update(flagged)
    print(f"\n⚠ Samples with abnormally low total counts: {flagged}")

# Flag 3: Low genes detected (outliers on the low end)
low_genes_outliers = is_outlier_mad(sample_qc['n_genes_by_counts_mean'].values)
low_genes_mask = (sample_qc['n_genes_by_counts_mean'] < sample_qc['n_genes_by_counts_mean'].median()) & low_genes_outliers
sample_flags['low_genes_detected'] = low_genes_mask
if sample_flags['low_genes_detected'].any():
    flagged = sample_flags[sample_flags['low_genes_detected']].index.tolist()
    bad_samples.update(flagged)
    print(f"\n⚠ Samples with abnormally low genes detected: {flagged}")

# Flag 4: High MT percentage (outliers on the high end)
high_mt_outliers = is_outlier_mad(sample_qc['pct_counts_mt_mean'].values)
high_mt_mask = (sample_qc['pct_counts_mt_mean'] > sample_qc['pct_counts_mt_mean'].median()) & high_mt_outliers
sample_flags['high_mt_pct'] = high_mt_mask
if sample_flags['high_mt_pct'].any():
    flagged = sample_flags[sample_flags['high_mt_pct']].index.tolist()
    bad_samples.update(flagged)
    print(f"\n⚠ Samples with abnormally high MT%: {flagged}")

# Summary of flagged samples
sample_flags['total_flags'] = sample_flags.sum(axis=1)
sample_flags['is_bad_quality'] = sample_flags.index.isin(bad_samples)

print("\n" + "="*80)
print("SAMPLE QUALITY FLAGS SUMMARY")
print("="*80)
print(sample_flags[sample_flags['total_flags'] > 0])

if len(bad_samples) > 0:
    print(f"\n{'='*80}")
    print(f"SAMPLES TO BE REMOVED: {sorted(bad_samples)}")
    print(f"Total: {len(bad_samples)} samples")
    print(f"{'='*80}")

    # Show detailed info for flagged samples
    print("\n--- Details of Flagged Samples ---")
    print(sample_qc.loc[sorted(bad_samples)])

    # Visualize flagged samples
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Total counts
    ax = axes[0, 0]
    colors = ['red' if s in bad_samples else 'steelblue' for s in sample_qc.index]
    ax.bar(range(len(sample_qc)), sample_qc['total_counts_mean'], color=colors)
    ax.set_xticks(range(len(sample_qc)))
    ax.set_xticklabels(sample_qc.index, rotation=45, ha='right')
    ax.set_ylabel('Mean Total Counts')
    ax.set_title('Average Total Counts per Sample')
    ax.axhline(sample_qc['total_counts_mean'].median(), color='black',
               linestyle='--', linewidth=1, label='Median')
    ax.legend()

    # Plot 2: Genes detected
    ax = axes[0, 1]
    ax.bar(range(len(sample_qc)), sample_qc['n_genes_by_counts_mean'], color=colors)
    ax.set_xticks(range(len(sample_qc)))
    ax.set_xticklabels(sample_qc.index, rotation=45, ha='right')
    ax.set_ylabel('Mean Genes Detected')
    ax.set_title('Average Genes Detected per Sample')
    ax.axhline(sample_qc['n_genes_by_counts_mean'].median(), color='black',
               linestyle='--', linewidth=1, label='Median')
    ax.legend()

    # Plot 3: MT percentage
    ax = axes[1, 0]
    ax.bar(range(len(sample_qc)), sample_qc['pct_counts_mt_mean'], color=colors)
    ax.set_xticks(range(len(sample_qc)))
    ax.set_xticklabels(sample_qc.index, rotation=45, ha='right')
    ax.set_ylabel('Mean MT%')
    ax.set_title('Average MT% per Sample')
    ax.axhline(sample_qc['pct_counts_mt_mean'].median(), color='black',
               linestyle='--', linewidth=1, label='Median')
    ax.legend()

    # Plot 4: Cell count
    ax = axes[1, 1]
    ax.bar(range(len(sample_qc)), sample_qc['n_cells'], color=colors)
    ax.set_xticks(range(len(sample_qc)))
    ax.set_xticklabels(sample_qc.index, rotation=45, ha='right')
    ax.set_ylabel('Number of Cells')
    ax.set_title('Cell Count per Sample')
    if MIN_CELLS_PER_SAMPLE > 0:
        ax.axhline(MIN_CELLS_PER_SAMPLE, color='red',
                   linestyle='--', linewidth=1, label=f'Min threshold ({MIN_CELLS_PER_SAMPLE})')
    ax.legend()

    plt.suptitle('Sample Quality Assessment (Red = Flagged for Removal)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Remove bad quality samples
    print(f"\nRemoving {len(bad_samples)} bad quality samples...")
    print(f"Cells before filtering: {adata.n_obs}")

    # Keep only cells from good samples
    good_samples_mask = ~adata.obs['sample'].isin(bad_samples)
    adata = adata[good_samples_mask, :].copy()

    print(f"Cells after filtering: {adata.n_obs}")
    print(f"Samples remaining: {adata.obs['sample'].nunique()}")
    print(f"Remaining samples: {sorted(adata.obs['sample'].unique())}")

    # Check condition balance after removal
    print("\n--- Samples per Condition (after removal) ---")
    print(adata.obs.groupby('condition')['sample'].nunique())

    # Warning if imbalanced
    samples_per_condition = adata.obs.groupby('condition')['sample'].nunique()
    if (samples_per_condition < 3).any():
        print("\n⚠ WARNING: After sample removal, some conditions have < 3 replicates!")
        print("This may affect statistical power in pseudobulk DEA.")

else:
    print(f"\n✓ All samples passed quality control!")
    print(f"No samples were flagged for removal.")

#%% ============================================================================
# 8b. CELL DISTRIBUTION AFTER FILTERING
# =============================================================================
print("\n" + "="*80)
print("8b. CELL DISTRIBUTION AFTER FILTERING")
print("="*80)

# Cells per sample (after filtering)
cells_per_sample_filtered = adata.obs['sample'].value_counts().sort_index()
print("\nCells per sample (after filtering):")
print(cells_per_sample_filtered)

# Cells per condition (after filtering)
cells_per_condition_filtered = adata.obs['condition'].value_counts()
print("\nCells per condition (after filtering):")
print(cells_per_condition_filtered)

# Plot cell distribution after filtering
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Cells per sample
mn_counts_df_filt = cells_per_sample_filtered.reset_index()
mn_counts_df_filt.columns = ['sample', 'MN_count']
sns.barplot(data=mn_counts_df_filt, x='sample', y='MN_count', palette='viridis', ax=axes[0])
axes[0].set_title('Motor Neurons per Sample (After Filtering)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Sample')
axes[0].set_ylabel('Number of Motor Neurons')
axes[0].tick_params(axis='x', rotation=45)
for i in axes[0].containers:
    axes[0].bar_label(i, fmt='%d', label_type='edge', fontsize=9)

# Plot 2: Cells per condition
condition_df_filt = cells_per_condition_filtered.reset_index()
condition_df_filt.columns = ['condition', 'MN_count']
sns.barplot(data=condition_df_filt, x='condition', y='MN_count', palette=sns.color_palette('Set2')[1::-1], ax=axes[1])
axes[1].set_title('Motor Neurons per Condition (After Filtering)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Condition')
axes[1].set_ylabel('Number of Motor Neurons')
for i in axes[1].containers:
    axes[1].bar_label(i, fmt='%d', label_type='edge', fontsize=10)

# Plot 3: Cells per sample colored by condition
sample_condition_df_filt = adata.obs.groupby(['sample', 'condition']).size().reset_index(name='MN_count')
sns.barplot(data=sample_condition_df_filt, x='sample', y='MN_count', hue='condition', palette=sns.color_palette('Set2')[1::-1], ax=axes[2])
axes[2].set_title('Motor Neurons per Sample by Condition (After Filtering)', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Sample')
axes[2].set_ylabel('Number of Motor Neurons')
axes[2].tick_params(axis='x', rotation=45)
axes[2].legend(title='Condition')

plt.tight_layout()
plt.savefig(f'{QC_DIR}/cell_distribution_after_filtering.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{QC_DIR}/cell_distribution_after_filtering.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved: {QC_DIR}/cell_distribution_after_filtering.pdf/png")

#%% ============================================================================
# 9. LAYER-SPECIFIC ANALYSIS (if available)
# =============================================================================
print("\n" + "="*80)
print("9. LAYER-SPECIFIC ANALYSIS")
print("="*80)

# Check for missing condition labels
n_missing_condition = adata.obs['condition'].isna().sum()
if n_missing_condition > 0:
    print(f"\n⚠ WARNING: {n_missing_condition} cells have missing 'condition' labels!")
    print("These cells will be shown in gray in condition-colored plots.")
    missing_samples = adata.obs[adata.obs['condition'].isna()]['sample'].unique()
    print(f"Affected samples: {list(missing_samples)}")

if 'exon' in adata.layers.keys() and 'intron' in adata.layers.keys():
    print("\nAnalyzing exon and intron layers...")

    # Calculate total exon and intron counts per cell
    exon_counts = np.array(adata.layers['exon'].sum(axis=1)).flatten()
    intron_counts = np.array(adata.layers['intron'].sum(axis=1)).flatten()

    adata.obs['exon_counts'] = exon_counts
    adata.obs['intron_counts'] = intron_counts
    adata.obs['intron_exon_ratio'] = intron_counts / (exon_counts + 1)  # +1 to avoid division by zero

    # Calculate intron proportion (fraction of total counts from introns)
    total_counts = np.array(adata.X.sum(axis=1)).flatten()
    adata.obs['intron_proportion'] = intron_counts / (total_counts + 1)  # +1 to avoid division by zero

    print("\n--- Exon/Intron Summary ---")
    print(adata.obs[['exon_counts', 'intron_counts', 'intron_exon_ratio', 'intron_proportion']].describe())

    # Visualize exon/intron distribution
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Exon counts distribution
    axes[0, 0].hist(exon_counts, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('Exon Counts')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Exon Counts')
    axes[0, 0].axvline(np.median(exon_counts), color='red', linestyle='--',
                       label=f'Median: {np.median(exon_counts):.0f}')
    axes[0, 0].legend()

    # Intron counts distribution
    axes[0, 1].hist(intron_counts, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Intron Counts')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Intron Counts')
    axes[0, 1].axvline(np.median(intron_counts), color='red', linestyle='--',
                       label=f'Median: {np.median(intron_counts):.0f}')
    axes[0, 1].legend()

    # Intron/Exon ratio distribution
    axes[0, 2].hist(adata.obs['intron_exon_ratio'], bins=50, alpha=0.7,
                    color='purple', edgecolor='black')
    axes[0, 2].set_xlabel('Intron/Exon Ratio')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Distribution of Intron/Exon Ratio')
    axes[0, 2].axvline(np.median(adata.obs['intron_exon_ratio']), color='red',
                       linestyle='--', label=f'Median: {np.median(adata.obs["intron_exon_ratio"]):.2f}')
    axes[0, 2].legend()

    # Intron proportion distribution
    axes[1, 0].hist(adata.obs['intron_proportion'], bins=50, alpha=0.7,
                    color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('Intron Proportion')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Intron Proportion')
    axes[1, 0].axvline(np.median(adata.obs['intron_proportion']), color='red',
                       linestyle='--', label=f'Median: {np.median(adata.obs["intron_proportion"]):.2f}')
    axes[1, 0].legend()

    # Exon vs Intron scatter
    # Handle potential NaN values in condition column
    condition_colors = adata.obs['condition'].map({'ALS': 'red', 'CTR': 'blue'})
    axes[1, 1].scatter(exon_counts, intron_counts, alpha=0.5, s=10, c=condition_colors)
    axes[1, 1].set_xlabel('Exon Counts')
    axes[1, 1].set_ylabel('Intron Counts')
    axes[1, 1].set_title('Exon vs Intron Counts')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='ALS'),
                       Patch(facecolor='blue', label='control')]
    
    axes[1, 1].legend(handles=legend_elements, loc='upper left')

    # Add correlation
    corr, pval = pearsonr(exon_counts, intron_counts)
    axes[1, 1].text(0.05, 0.95, f'Pearson r: {corr:.3f}\np-val: {pval:.2e}',
                    transform=axes[1, 1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Total counts vs Intron proportion scatter
    axes[1, 2].scatter(total_counts, adata.obs['intron_proportion'], alpha=0.5, s=10, c=condition_colors)
    axes[1, 2].set_xlabel('Total Counts')
    axes[1, 2].set_ylabel('Intron Proportion')
    axes[1, 2].set_title('Total Counts vs Intron Proportion')
    axes[1, 2].set_xscale('log')
    axes[1, 2].legend(handles=legend_elements, loc='upper right')

    # Add correlation
    corr_ip, pval_ip = pearsonr(total_counts, adata.obs['intron_proportion'])
    axes[1, 2].text(0.05, 0.95, f'Pearson r: {corr_ip:.3f}\np-val: {pval_ip:.2e}',
                    transform=axes[1, 2].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

    # Violin plots by condition
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for i, metric in enumerate(['exon_counts', 'intron_counts', 'intron_exon_ratio', 'intron_proportion']):
        sc.pl.violin(adata, keys=metric, groupby='condition', ax=axes[i], show=False)
        axes[i].set_title(f'{metric} by Condition')

    plt.tight_layout()
    plt.show()

else:
    print("\nExon and intron layers not found. Skipping layer-specific analysis.")
    print("Available layers:", list(adata.layers.keys()))

#%% ============================================================================
# 10. FILTER GENES AND PREPARE FOR ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("10. FILTERING GENES")
print("="*80)

print(f"\nGenes before filtering: {adata.n_vars}")

# Filter genes expressed in at least 1 cell
sc.pp.filter_genes(adata, min_cells=1, inplace=True)
print(f"Genes after min_cells=1 filter: {adata.n_vars}")

# Filter MT, ribosomal, and hemoglobin genes
mask = ~(adata.var['mt'] | adata.var['ribo'] | adata.var['hb'])
adata = adata[:, mask]
print(f"Genes after removing MT/ribo/hb: {adata.n_vars}")

#%% ============================================================================
# 11. GENE-LEVEL QUALITY CONTROL (POST-FILTERING)
# =============================================================================
print("\n" + "="*80)
print("11. GENE-LEVEL QUALITY CONTROL (POST-FILTERING)")
print("="*80)

# Calculate number of cells expressing each gene
MIN_CELLS = 1
cell_num = sc.pp.filter_genes(adata, min_cells=MIN_CELLS, inplace=False)[1]
adata.var["n_cells_expr"] = cell_num

print(f"\nGenes expressed in at least {MIN_CELLS} cell(s): {(adata.var['n_cells_expr'] >= MIN_CELLS).sum()}")
print(f"Genes not expressed in any cell: {(adata.var['n_cells_expr'] == 0).sum()}")

# Gene expression statistics
print("\n--- Gene Expression Statistics ---")
print(adata.var['n_cells_expr'].describe())

# Highly expressed genes
print("\n--- Top 20 Most Widely Expressed Genes ---")
top_genes = adata.var.nlargest(20, 'n_cells_expr')[['n_cells_expr']]
print(top_genes)

# Plot gene expression distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of cells per gene
axes[0].hist(adata.var['n_cells_expr'], bins=100, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].set_xlabel('Number of Cells Expressing Gene')
axes[0].set_ylabel('Number of Genes')
axes[0].set_title('Gene Expression Breadth Distribution')
axes[0].axvline(np.median(adata.var['n_cells_expr']), color='red', linestyle='--',
                label=f'Median: {np.median(adata.var["n_cells_expr"]):.0f}')
axes[0].legend()
axes[0].set_yscale('log')

# Cumulative distribution
sorted_expr = np.sort(adata.var['n_cells_expr'])
cumulative = np.arange(1, len(sorted_expr) + 1) / len(sorted_expr)
axes[1].plot(sorted_expr, cumulative, linewidth=2, color='steelblue')
axes[1].set_xlabel('Number of Cells Expressing Gene')
axes[1].set_ylabel('Cumulative Fraction of Genes')
axes[1].set_title('Cumulative Gene Expression Distribution')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Show most ubiquitously expressed genes (lightweight alternative)
print("\n--- Top 10 Most Ubiquitously Expressed Genes ---")
print("(Genes expressed in the most cells)")
top_ubiquitous = adata.var.nlargest(10, 'n_cells_expr')[['n_cells_expr']]
top_ubiquitous['pct_cells'] = (top_ubiquitous['n_cells_expr'] / adata.n_obs * 100).round(2)
print(top_ubiquitous)

#%% ============================================================================
# 12. NORMALIZATION AND LAYER PREPARATION
# =============================================================================
print("\n" + "="*80)
print("12. NORMALIZATION")
print("="*80)

# Store raw counts
if 'counts' not in adata.layers:
    adata.layers["counts"] = adata.X.copy()
    print("Stored raw counts in layer 'counts'")

# Log-normalization
sc.pp.normalize_total(adata, target_sum=1e4, exclude_highly_expressed=True)
sc.pp.log1p(adata)
adata.layers["lognorm"] = adata.X.copy()
print("Created log-normalized layer 'lognorm'")

#%% ============================================================================
# 13. HIGHLY VARIABLE GENES
# =============================================================================
print("\n" + "="*80)
print("13. IDENTIFYING HIGHLY VARIABLE GENES")
print("="*80)

sc.pp.highly_variable_genes(
    adata,
    layer="lognorm",
    n_top_genes=3000,
    flavor="seurat_v3",
    batch_key="sample"
)

print(f"\nHighly variable genes identified: {adata.var['highly_variable'].sum()}")

# Plot highly variable genes
sc.pl.highly_variable_genes(adata)

#%% ============================================================================
# 14. DIMENSIONALITY REDUCTION (PCA)
# =============================================================================
print("\n" + "="*80)
print("14. PCA ANALYSIS")
print("="*80)

sc.pp.pca(adata, svd_solver="arpack", use_highly_variable=True)

# PCA scatter plots
sc.pl.pca_scatter(adata, color=["condition", "sample", "batch", "sex"])

# PCA colored by QC metrics
sc.pl.pca_scatter(adata, color=["total_counts", "n_genes_by_counts", "pct_counts_mt"])

# Variance ratio
sc.pl.pca_variance_ratio(adata, n_pcs=50)

# PCA loadings
sc.pl.pca_loadings(adata, components='1,2,3,4,5,6,7,8,9,10')

#%% ============================================================================
# 15. BATCH EFFECT ASSESSMENT
# =============================================================================
print("\n" + "="*80)
print("15. BATCH EFFECT ASSESSMENT")
print("="*80)

# PCA by batch
sc.pl.pca(adata, color=['batch', 'sample'], size=50)

# Check if batches cluster separately
from sklearn.metrics import silhouette_score

batch_sil = silhouette_score(adata.obsm['X_pca'][:, :10], adata.obs['batch'])
condition_sil = silhouette_score(adata.obsm['X_pca'][:, :10], adata.obs['condition'])
sample_sil = silhouette_score(adata.obsm['X_pca'][:, :10], adata.obs['sample'])

print(f"\nSilhouette scores (higher = more separated):")
print(f"  Batch: {batch_sil:.3f}")
print(f"  Condition: {condition_sil:.3f}")
print(f"  Sample: {sample_sil:.3f}")
print("\nInterpretation:")
print("  - Batch silhouette should be LOW (batches should NOT separate)")
print("  - Condition silhouette can be moderate (biological signal)")
print("  - High sample silhouette may indicate sample-specific effects")

#%% ============================================================================
# 16. UMAP VISUALIZATION
# =============================================================================
print("\n" + "="*80)
print("16. UMAP VISUALIZATION")
print("="*80)

# Calculate optimal number of neighbors
PCs = 7
N_NB = int(0.5 * len(adata) ** 0.5)
if N_NB > 80:
    N_NB = 80
print(f"\nUsing {N_NB} neighbors and {PCs} PCs")

# Compute neighbors and UMAP
sc.pp.neighbors(adata, n_neighbors=N_NB, n_pcs=PCs, use_rep="X_pca")
sc.tl.umap(adata, random_state=42, min_dist=0.2)

# UMAP colored by metadata
sc.pl.umap(adata, color=['condition', 'sample', 'batch', 'sex', 'age'], ncols=2, size=200)

# UMAP colored by QC metrics
sc.pl.umap(adata, color=['total_counts', 'n_genes_by_counts', 'pct_counts_mt',
                          'pct_counts_ribo'], ncols=2, size=200)

# UMAP with layer-specific metrics if available
if 'intron_exon_ratio' in adata.obs.columns:
    sc.pl.umap(adata, color=['intron_exon_ratio', 'intron_proportion', 'exon_counts', 'intron_counts'], ncols=2, size=200)

#%% ============================================================================
# 17. MARKER GENE EXPRESSION
# =============================================================================
print("\n" + "="*80)
print("17. MOTOR NEURON MARKER GENE EXPRESSION")
print("="*80)

# Define motor neuron markers
mn_markers = ["SLC5A7", "MT1X", "PALMD", "ISL1", "IL33", "CHAT", "TPH2", "AMTN", "CREB5", "PARD3B", "VIPR2", "STMN2", "MEG8"
              ,"NEFH", "NEFL", "NEFM", "PRPH", "HB9", "LHX3", "FOXP1", "NTRK2", "NTRK3", "GAP43","GFAP"]

# Check which markers are present
mn_markers_present = [g for g in mn_markers if g in adata.var_names]
mn_markers_missing = [g for g in mn_markers if g not in adata.var_names]

print(f"\nMarkers present: {mn_markers_present}")
if mn_markers_missing:
    print(f"Markers missing: {mn_markers_missing}")

if len(mn_markers_present) > 0:
    # Violin plot by condition
    sc.pl.violin(adata, keys=mn_markers_present, groupby='condition',
                 jitter=True, multi_panel=True, rotation=45)

    # UMAP with marker expression
    sc.pl.umap(adata, color=mn_markers_present, ncols=3, size=200,
               color_map='viridis')

    # Dotplot by sample
    sc.pl.dotplot(adata, var_names=mn_markers_present, groupby='sample')

# Plot selected genes on UMAP
selected_genes = ["NEAT1", "GFAP", "SNAP25", "KCNC1", "STMN1", "STMN2"]
selected_genes_present = [g for g in selected_genes if g in adata.var_names]
if len(selected_genes_present) > 0:
    print(f"\nPlotting selected genes on UMAP: {selected_genes_present}")
    sc.pl.umap(adata, color=selected_genes_present, ncols=3, size=200,
               color_map='viridis')

#%% Additional interesting gene sets
# ALS-related genes from literature
als_genes = ["F3", "C3AR1", "SERPINA3", "APLNR", "SGK1", "AQP1", "MAPK4",
             "MED13L", "XRN1", "CLIC5", "SRGN", "STOM", "TJP2", "LAMA2"]
als_genes_present = [g for g in als_genes if g in adata.var_names]

if len(als_genes_present) > 0:
    print(f"\nALS-related genes present: {len(als_genes_present)}/{len(als_genes)}")
    sc.pl.violin(adata, keys=als_genes_present[:8], groupby='condition',
                 jitter=True, multi_panel=True, rotation=45)

#%% ============================================================================
# 18. CLUSTERING (optional - for QC purposes)
# =============================================================================
print("\n" + "="*80)
print("18. LEIDEN CLUSTERING (for QC)")
print("="*80)

sc.tl.leiden(adata, resolution=0.2)
sc.pl.umap(adata, color=['leiden', 'condition', 'sample'], ncols=3, size=200)

# Cluster composition
cluster_comp = pd.crosstab(adata.obs['leiden'], adata.obs['condition'], normalize='index')
print("\nCluster composition by condition:")
print(cluster_comp)

cluster_comp_sample = pd.crosstab(adata.obs['leiden'], adata.obs['sample'], normalize='index')
print("\nCluster composition by sample:")
print(cluster_comp_sample)

#%% Calculate marker genes for each cluster
print("\n--- Calculating Marker Genes per Cluster ---")

# Use Wilcoxon rank-sum test for marker detection
sc.tl.rank_genes_groups(
    adata,
    groupby='leiden',
    method='wilcoxon',
    layer='lognorm',  # Use log-normalized data
    pts=True  # Calculate percentage of cells expressing each gene
)

# Plot top markers per cluster
sc.pl.rank_genes_groups(adata, n_genes=15, sharey=False)

# Get marker genes as DataFrame
print("\n--- Top 10 Marker Genes per Cluster ---")
markers_df = sc.get.rank_genes_groups_df(adata, group=None)

# Show top 10 markers per cluster with statistics
for cluster in sorted(adata.obs['leiden'].unique()):
    cluster_markers = markers_df[markers_df['group'] == cluster].head(10)
    print(f"\n=== Cluster {cluster} ===")
    print(cluster_markers[['names', 'scores', 'logfoldchanges', 'pvals_adj', 'pct_nz_group', 'pct_nz_reference']].to_string(index=False))

# Dotplot of top markers
top_n = 5
top_markers = {}
for cluster in sorted(adata.obs['leiden'].unique()):
    cluster_markers = markers_df[markers_df['group'] == cluster].head(top_n)['names'].tolist()
    top_markers[f'Cluster {cluster}'] = cluster_markers

# Flatten for dotplot
all_top_markers = []
for cluster in sorted(adata.obs['leiden'].unique()):
    cluster_genes = markers_df[markers_df['group'] == cluster].head(top_n)['names'].tolist()
    all_top_markers.extend(cluster_genes)
# Remove duplicates while preserving order
all_top_markers = list(dict.fromkeys(all_top_markers))

sc.pl.dotplot(adata, var_names=all_top_markers, groupby='leiden',
              standard_scale='var', title='Top Marker Genes per Cluster')

# Heatmap of top markers
sc.pl.rank_genes_groups_heatmap(adata, n_genes=5, groupby='leiden',
                                 show_gene_labels=True, standard_scale='var')

# Save marker genes to CSV
markers_df.to_csv(f"{QC_DIR}/cluster_markers.csv", index=False)
print(f"\nMarker genes saved to: {QC_DIR}/cluster_markers.csv")

#%% ============================================================================
# 18b. THESIS-GRADE CLUSTER ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("18b. THESIS-GRADE CLUSTER ANALYSIS")
print("="*80)

# -----------------------------------------------------------------------------
# A. CLUSTER COMPOSITION: % ALS vs CONTROL per cluster
# -----------------------------------------------------------------------------
print("\n--- A. Cluster Composition by Condition ---")

# Calculate cluster composition
cluster_condition_counts = pd.crosstab(adata.obs['leiden'], adata.obs['condition'])
cluster_condition_pct = cluster_condition_counts.div(cluster_condition_counts.sum(axis=1), axis=0) * 100

# Calculate % ALS for ordering
if 'ALS' in cluster_condition_pct.columns:
    cluster_als_pct = cluster_condition_pct['ALS'].sort_values(ascending=False)
    print("\n% ALS cells per cluster:")
    print(cluster_als_pct.round(1))

# Create stacked barplot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Stacked bar (percentage)
cluster_order = cluster_condition_pct.index.tolist()
colors = {'ALS': '#E74C3C', 'CTR': '#3498DB'}  # Red for ALS, Blue for CTR

bottom = np.zeros(len(cluster_order))
for condition in ['CTR', 'ALS']:
    if condition in cluster_condition_pct.columns:
        values = cluster_condition_pct.loc[cluster_order, condition].values
        axes[0].bar(cluster_order, values, bottom=bottom, label=condition,
                   color=colors.get(condition, 'gray'), edgecolor='white', linewidth=0.5)
        bottom += values

axes[0].set_xlabel('Cluster', fontsize=12)
axes[0].set_ylabel('% of Cells', fontsize=12)
axes[0].set_title('Cluster Composition: ALS vs CTR', fontsize=14, fontweight='bold')
axes[0].legend(title='Condition', loc='upper right')
axes[0].set_ylim(0, 100)

# Add percentage labels
for i, cluster in enumerate(cluster_order):
    if 'ALS' in cluster_condition_pct.columns:
        als_pct = cluster_condition_pct.loc[cluster, 'ALS']
        axes[0].text(i, 50, f'{als_pct:.0f}%', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')

# Plot 2: Number of cells per cluster (split by condition)
cluster_condition_counts_melted = cluster_condition_counts.reset_index().melt(
    id_vars='leiden', var_name='condition', value_name='n_cells')
x = np.arange(len(cluster_order))
width = 0.35

for i, condition in enumerate(['CTR', 'ALS']):
    if condition in cluster_condition_counts.columns:
        values = cluster_condition_counts.loc[cluster_order, condition].values
        offset = width * (i - 0.5)
        bars = axes[1].bar(x + offset, values, width, label=condition,
                          color=colors.get(condition, 'gray'), edgecolor='white')
        # Add count labels
        for bar, val in zip(bars, values):
            if val > 0:
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{val}', ha='center', va='bottom', fontsize=9)

axes[1].set_xlabel('Cluster', fontsize=12)
axes[1].set_ylabel('Number of Cells', fontsize=12)
axes[1].set_title('Cell Counts per Cluster by Condition', fontsize=14, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(cluster_order)
axes[1].legend(title='Condition')

plt.tight_layout()
plt.savefig(f'{QC_DIR}/cluster_composition_ALS_vs_CTR.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{QC_DIR}/cluster_composition_ALS_vs_CTR.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved: {QC_DIR}/cluster_composition_ALS_vs_CTR.pdf/png")

# -----------------------------------------------------------------------------
# B. PATHWAY SCORES: Synaptic, Stress, Neurotrophic
# -----------------------------------------------------------------------------
print("\n--- B. Calculating Pathway Scores ---")

# Define gene sets for pathways (human gene symbols)
# These are curated from literature - adjust based on your data availability

pathway_genes = {
    'Synaptic': [
        'SYP', 'SYN1', 'SYN2', 'SNAP25', 'STX1A', 'VAMP2', 'SLC17A7', 'SLC17A6',
        'DLG4', 'GRIN1', 'GRIN2A', 'GRIN2B', 'GRIA1', 'GRIA2', 'GABRA1', 'GABRB2',
        'SV2A', 'SV2B', 'NRXN1', 'NLGN1', 'SHANK1', 'SHANK2', 'SHANK3', 'HOMER1',
        'CAMK2A', 'CAMK2B', 'BSN', 'PCLO', 'CPLX1', 'CPLX2', 'SYT1', 'RAB3A'
    ],
    'Stress_Response': [
        'HSP90AA1', 'HSP90AB1', 'HSPA1A', 'HSPA1B', 'HSPA5', 'HSPA8', 'HSPB1',
        'DNAJB1', 'DNAJC3', 'ATF4', 'ATF6', 'XBP1', 'DDIT3', 'EIF2AK3', 'ERN1',
        'NFE2L2', 'KEAP1', 'HMOX1', 'NQO1', 'GCLC', 'GCLM', 'SOD1', 'SOD2',
        'CAT', 'GPX1', 'PRDX1', 'TXN', 'TXNRD1', 'SQSTM1', 'MAP1LC3B', 'BECN1'
    ],
    'Neurotrophic': [
        'BDNF', 'NGF', 'NTF3', 'NTF4', 'GDNF', 'CNTF', 'LIF', 'NTRK1', 'NTRK2',
        'NTRK3', 'NGFR', 'RET', 'GFRA1', 'GFRA2', 'LIFR', 'CNTFR', 'AKT1',
        'MAPK1', 'MAPK3', 'CREB1', 'BCL2', 'BAX', 'CASP3', 'ARC', 'FOS', 'JUN'
    ],
    'MN_Identity': [
        'CHAT', 'SLC5A7', 'ISL1', 'ISL2', 'MNX1', 'LHX3', 'LHX4', 'PHOX2A',
        'PHOX2B', 'PRPH', 'NEFH', 'NEFL', 'NEFM', 'SMN1', 'SMN2', 'CHODL'
    ],
    'Apoptosis': [
        'CASP3', 'CASP7', 'CASP8', 'CASP9', 'BAX', 'BAK1', 'BCL2', 'BCL2L1',
        'BID', 'BAD', 'CYCS', 'APAF1', 'DIABLO', 'XIAP', 'BIRC5', 'TP53',
        'MDM2', 'PMAIP1', 'BBC3', 'FAS', 'FASLG', 'TNFRSF1A', 'RIPK1', 'RIPK3'
    ]
}

# Calculate scores for each pathway
for pathway_name, genes in pathway_genes.items():
    # Filter genes present in dataset
    genes_present = [g for g in genes if g in adata.var_names]
    genes_missing = [g for g in genes if g not in adata.var_names]

    print(f"\n{pathway_name}: {len(genes_present)}/{len(genes)} genes present")
    if len(genes_missing) > 0 and len(genes_missing) <= 10:
        print(f"  Missing: {genes_missing}")

    if len(genes_present) >= 3:  # Need at least 3 genes for reliable score
        sc.tl.score_genes(adata, gene_list=genes_present,
                         score_name=f'{pathway_name}_score', use_raw=False)
    else:
        print(f"  ⚠ Skipping {pathway_name}: insufficient genes")

# Get list of calculated scores
score_columns = [col for col in adata.obs.columns if col.endswith('_score')]
print(f"\nCalculated pathway scores: {score_columns}")

# Visualize pathway scores on UMAP
if len(score_columns) > 0:
    sc.pl.umap(adata, color=score_columns, ncols=3, cmap='RdYlBu_r', size=200,
               title=[s.replace('_', ' ').title() for s in score_columns])

# Violin plots: pathway scores by cluster
if len(score_columns) > 0:
    fig, axes = plt.subplots(1, len(score_columns), figsize=(5*len(score_columns), 5))
    if len(score_columns) == 1:
        axes = [axes]

    for i, score in enumerate(score_columns):
        sc.pl.violin(adata, keys=score, groupby='leiden', ax=axes[i],
                    show=False, rotation=0)
        axes[i].set_title(score.replace('_', ' ').title(), fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{QC_DIR}/pathway_scores_by_cluster.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{QC_DIR}/pathway_scores_by_cluster.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nSaved: {QC_DIR}/pathway_scores_by_cluster.pdf/png")

# Violin plots: pathway scores by condition
if len(score_columns) > 0:
    fig, axes = plt.subplots(1, len(score_columns), figsize=(4*len(score_columns), 5))
    if len(score_columns) == 1:
        axes = [axes]

    for i, score in enumerate(score_columns):
        sc.pl.violin(adata, keys=score, groupby='condition', ax=axes[i],
                    show=False, rotation=0, palette={'ALS': '#E74C3C', 'CTR': '#3498DB'})
        axes[i].set_title(score.replace('_', ' ').title(), fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{QC_DIR}/pathway_scores_by_condition.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{QC_DIR}/pathway_scores_by_condition.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nSaved: {QC_DIR}/pathway_scores_by_condition.pdf/png")

# -----------------------------------------------------------------------------
# C. ORDER CLUSTERS ALONG TRAJECTORY (CTR → compensatory → stress)
# -----------------------------------------------------------------------------
print("\n--- C. Ordering Clusters Along Disease Trajectory ---")

# Calculate mean pathway scores per cluster
cluster_scores = adata.obs.groupby('leiden')[score_columns].mean()

# Calculate % ALS per cluster
cluster_als_fraction = cluster_condition_pct['ALS'] / 100 if 'ALS' in cluster_condition_pct.columns else None

# Create trajectory score: combine stress and ALS enrichment
if 'Stress_Response_score' in cluster_scores.columns and cluster_als_fraction is not None:
    # Normalize scores to 0-1
    stress_norm = (cluster_scores['Stress_Response_score'] - cluster_scores['Stress_Response_score'].min()) / \
                  (cluster_scores['Stress_Response_score'].max() - cluster_scores['Stress_Response_score'].min() + 1e-10)

    # Trajectory score: combination of stress and ALS enrichment
    trajectory_score = 0.5 * stress_norm + 0.5 * cluster_als_fraction

    # Order clusters by trajectory
    cluster_order_trajectory = trajectory_score.sort_values().index.tolist()

    print("\nCluster ordering (CTR → compensatory → stress):")
    print(f"Order: {cluster_order_trajectory}")

    # Create summary table
    trajectory_summary = pd.DataFrame({
        'cluster': cluster_order_trajectory,
        'trajectory_score': trajectory_score[cluster_order_trajectory].values,
        'pct_ALS': cluster_als_fraction[cluster_order_trajectory].values * 100,
        'stress_score': cluster_scores.loc[cluster_order_trajectory, 'Stress_Response_score'].values,
        'n_cells': adata.obs['leiden'].value_counts()[cluster_order_trajectory].values
    })

    if 'Synaptic_score' in cluster_scores.columns:
        trajectory_summary['synaptic_score'] = cluster_scores.loc[cluster_order_trajectory, 'Synaptic_score'].values
    if 'Neurotrophic_score' in cluster_scores.columns:
        trajectory_summary['neurotrophic_score'] = cluster_scores.loc[cluster_order_trajectory, 'Neurotrophic_score'].values

    print("\nTrajectory Summary Table:")
    print(trajectory_summary.round(3).to_string(index=False))

    # Save trajectory summary
    trajectory_summary.to_csv(f'{QC_DIR}/cluster_trajectory_summary.csv', index=False)
    print(f"\nSaved: {QC_DIR}/cluster_trajectory_summary.csv")

    # Create ordered category for plotting
    adata.obs['leiden_ordered'] = pd.Categorical(
        adata.obs['leiden'],
        categories=cluster_order_trajectory,
        ordered=True
    )

    # Comprehensive trajectory figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Trajectory score barplot
    colors_trajectory = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(cluster_order_trajectory)))
    axes[0, 0].bar(range(len(cluster_order_trajectory)), trajectory_summary['trajectory_score'],
                   color=colors_trajectory, edgecolor='black', linewidth=0.5)
    axes[0, 0].set_xticks(range(len(cluster_order_trajectory)))
    axes[0, 0].set_xticklabels(cluster_order_trajectory)
    axes[0, 0].set_xlabel('Cluster (ordered)', fontsize=12)
    axes[0, 0].set_ylabel('Trajectory Score', fontsize=12)
    axes[0, 0].set_title('Disease Trajectory Score per Cluster\n(CTR → Compensatory → Stress)',
                         fontsize=14, fontweight='bold')

    # Plot 2: Stacked pathway scores
    score_cols_plot = [s for s in ['Synaptic_score', 'Stress_Response_score', 'Neurotrophic_score']
                       if s in cluster_scores.columns]
    if len(score_cols_plot) > 0:
        scores_ordered = cluster_scores.loc[cluster_order_trajectory, score_cols_plot]
        # Normalize for stacking
        scores_norm = scores_ordered.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10))

        x = np.arange(len(cluster_order_trajectory))
        width = 0.25
        colors_pathway = ['#2ECC71', '#E74C3C', '#9B59B6']  # Green, Red, Purple

        for i, (col, color) in enumerate(zip(score_cols_plot, colors_pathway)):
            axes[0, 1].bar(x + i*width, scores_norm[col], width, label=col.replace('_score', ''),
                          color=color, edgecolor='white', alpha=0.8)

        axes[0, 1].set_xticks(x + width)
        axes[0, 1].set_xticklabels(cluster_order_trajectory)
        axes[0, 1].set_xlabel('Cluster (ordered)', fontsize=12)
        axes[0, 1].set_ylabel('Normalized Score', fontsize=12)
        axes[0, 1].set_title('Pathway Scores Along Trajectory', fontsize=14, fontweight='bold')
        axes[0, 1].legend(title='Pathway')

    # Plot 3: % ALS along trajectory
    axes[1, 0].bar(range(len(cluster_order_trajectory)), trajectory_summary['pct_ALS'],
                   color=colors_trajectory, edgecolor='black', linewidth=0.5)
    axes[1, 0].set_xticks(range(len(cluster_order_trajectory)))
    axes[1, 0].set_xticklabels(cluster_order_trajectory)
    axes[1, 0].set_xlabel('Cluster (ordered)', fontsize=12)
    axes[1, 0].set_ylabel('% ALS Cells', fontsize=12)
    axes[1, 0].set_title('ALS Enrichment Along Trajectory', fontsize=14, fontweight='bold')
    axes[1, 0].axhline(50, color='gray', linestyle='--', alpha=0.5, label='50%')
    axes[1, 0].legend()

    # Plot 4: UMAP with ordered clusters
    # Create numeric leiden for colormap
    leiden_numeric = adata.obs['leiden'].map({c: i for i, c in enumerate(cluster_order_trajectory)})
    scatter = axes[1, 1].scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1],
                                  c=leiden_numeric, cmap='RdYlBu_r', s=10, alpha=0.7)
    axes[1, 1].set_xlabel('UMAP1', fontsize=12)
    axes[1, 1].set_ylabel('UMAP2', fontsize=12)
    axes[1, 1].set_title('UMAP: Clusters Ordered by Trajectory', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=axes[1, 1], ticks=range(len(cluster_order_trajectory)))
    cbar.ax.set_yticklabels(cluster_order_trajectory)
    cbar.set_label('Cluster (CTR → Stress)')

    plt.tight_layout()
    plt.savefig(f'{QC_DIR}/cluster_trajectory_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{QC_DIR}/cluster_trajectory_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nSaved: {QC_DIR}/cluster_trajectory_analysis.pdf/png")

else:
    print("\n⚠ Cannot calculate trajectory: missing stress scores or ALS labels")

# -----------------------------------------------------------------------------
# D. FINAL THESIS FIGURE: Combined Summary
# -----------------------------------------------------------------------------
print("\n--- D. Creating Final Thesis Summary Figure ---")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.35, wspace=0.3)

# Row 1, Col 1: UMAP by leiden with labels
ax1 = fig.add_subplot(gs[0, 0])
sc.pl.umap(adata, color='leiden', ax=ax1, show=False, title='Clusters', legend_loc='on data', size=200)

# Row 1, Col 2: UMAP by condition
ax2 = fig.add_subplot(gs[0, 1])
sc.pl.umap(adata, color='condition', ax=ax2, show=False, title='Condition', size=200,
          palette={'ALS': '#E74C3C', 'CTR': '#3498DB'})

# Row 1, Col 3: Proportion barplot
ax3 = fig.add_subplot(gs[0, 2])
cluster_order = cluster_condition_pct.index.tolist()
bottom = np.zeros(len(cluster_order))
for condition in ['CTR', 'ALS']:
    if condition in cluster_condition_pct.columns:
        values = cluster_condition_pct.loc[cluster_order, condition].values
        ax3.bar(cluster_order, values, bottom=bottom, label=condition,
               color={'CTR': '#3498DB', 'ALS': '#E74C3C'}[condition], edgecolor='white')
        bottom += values
ax3.set_xlabel('Cluster')
ax3.set_ylabel('% of Cells')
ax3.set_title('Cluster Composition')
ax3.legend(title='Condition')
ax3.set_ylim(0, 100)

# Row 2: Marker gene dotplot (full width)
ax4 = fig.add_subplot(gs[1, :])
# Get top 3 markers per cluster
top_n_markers = 3
marker_genes_for_plot = []
for cluster in sorted(adata.obs['leiden'].unique()):
    genes = markers_df[markers_df['group'] == cluster].head(top_n_markers)['names'].tolist()
    marker_genes_for_plot.extend(genes)
marker_genes_for_plot = list(dict.fromkeys(marker_genes_for_plot))  # Remove duplicates

sc.pl.dotplot(adata, var_names=marker_genes_for_plot, groupby='leiden', ax=ax4,
              show=False, standard_scale='var', title='Top Marker Genes')

plt.suptitle('Motor Neuron Cluster Analysis', fontsize=14, fontweight='bold', y=0.98)
plt.savefig(f'{QC_DIR}/thesis_figure_cluster_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{QC_DIR}/thesis_figure_cluster_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved: {QC_DIR}/thesis_figure_cluster_analysis.pdf/png")

print("\n" + "="*80)
print("THESIS-GRADE ANALYSIS COMPLETED!")
print("="*80)

#%% ============================================================================
# 19. PRE-PSEUDOBULK CHECKS
# =============================================================================
print("\n" + "="*80)
print("19. PRE-PSEUDOBULK DEA CHECKS")
print("="*80)

# Verify data integrity: each sample should belong to only one condition
print("\n--- Data Integrity Check ---")
sample_condition_map = adata.obs.groupby('sample')['condition'].unique()
multi_condition_samples = sample_condition_map[sample_condition_map.apply(len) > 1]
if len(multi_condition_samples) > 0:
    print("⚠ WARNING: The following samples appear in multiple conditions:")
    print(multi_condition_samples)
else:
    print("✓ Each sample belongs to exactly one condition")

# Create comprehensive sample summary table
print("\n--- Sample Summary Table ---")
sample_summary = adata.obs.groupby('sample').agg({
    'condition': 'first',
    'batch': 'first',
    'sex': 'first',
    'age': 'first',
    'n_genes_by_counts': 'mean',
    'total_counts': 'mean',
    'pct_counts_mt': 'mean'
}).round(2)

# Add cell count per sample
sample_summary['n_cells'] = adata.obs.groupby('sample').size()

# Reorder columns for better readability
sample_summary = sample_summary[['condition', 'n_cells', 'n_genes_by_counts',
                                   'total_counts', 'pct_counts_mt', 'batch', 'sex', 'age']]
sample_summary.columns = ['condition', 'n_cells', 'mean_genes', 'mean_counts',
                          'mean_mt_pct', 'batch', 'sex', 'age']
sample_summary = sample_summary.sort_values(['condition', 'n_cells'], ascending=[True, False])
print(sample_summary)

# Check for minimum cell requirements per sample
min_cells_threshold = 10
samples_below_threshold = sample_summary[sample_summary['n_cells'] < min_cells_threshold]
if len(samples_below_threshold) > 0:
    print(f"\n⚠ WARNING: The following samples have < {min_cells_threshold} cells:")
    print(samples_below_threshold)
else:
    print(f"\n✓ All samples have >= {min_cells_threshold} cells")

# Biological replicates per condition (critical for pseudobulk DEA)
print("\n--- Biological Replicates per Condition ---")
replicates_per_condition = adata.obs.groupby('condition')['sample'].nunique()
print(replicates_per_condition)

# List samples per condition
print("\n--- Samples in Each Condition ---")
for condition in sorted(adata.obs['condition'].unique()):
    samples = sorted(adata.obs[adata.obs['condition'] == condition]['sample'].unique())
    print(f"{condition}: {samples} (n={len(samples)} samples)")

# Check replicate sufficiency
if (replicates_per_condition < 3).any():
    print("\n⚠ WARNING: Some conditions have fewer than 3 biological replicates!")
    print("Minimum 3 replicates per condition is recommended for robust DEA.")
else:
    print("\n✓ All conditions have >= 3 biological replicates")

# Summary statistics for pseudobulk
print("\n" + "="*80)
print("PSEUDOBULK DEA SUMMARY")
print("="*80)
print(f"\nTotal samples: {adata.obs['sample'].nunique()}")
print(f"Total cells: {adata.n_obs}")
print(f"Total genes (after filtering): {adata.n_vars}")
print(f"\nConditions: {sorted(adata.obs['condition'].unique())}")
print(f"Batches: {sorted(adata.obs['batch'].unique())}")

# Cells per condition breakdown
print("\n--- Cells per Condition ---")
for condition in sorted(adata.obs['condition'].unique()):
    n_cells = (adata.obs['condition'] == condition).sum()
    n_samples = adata.obs[adata.obs['condition'] == condition]['sample'].nunique()
    mean_cells = n_cells / n_samples if n_samples > 0 else 0
    print(f"{condition}: {n_cells} cells across {n_samples} samples (mean: {mean_cells:.1f} cells/sample)")

# Experimental design table
print("\n--- Experimental Design for DEA ---")
design_summary = adata.obs.groupby(['condition', 'batch']).agg({
    'sample': 'nunique',
}).reset_index()
design_summary.columns = ['condition', 'batch', 'n_samples']
print(design_summary.sort_values(['condition', 'batch']))

print("\n" + "="*80)
print("✓ Data ready for pseudobulk aggregation and DEA")
print("="*80)

#%% ============================================================================
# 20. SAVE PROCESSED DATA
# =============================================================================
print("\n" + "="*80)
print("20. SAVING PROCESSED DATA")
print("="*80)

# Save the processed AnnData object
output_file = motorneuron_data_path.replace(".h5ad", "_processed.h5ad")
adata.write_h5ad(output_file)
print(f"\nProcessed data saved to: {output_file}")

# Save QC summary to CSV
qc_summary = adata.obs[['sample', 'condition', 'batch', 'sex', 'age',
                         'total_counts', 'n_genes_by_counts', 'pct_counts_mt',
                         'pct_counts_ribo', 'pct_counts_hb']]
qc_summary.to_csv(f"{QC_DIR}/motorneurons_qc_summary.csv")
print(f"QC summary saved in: {QC_DIR}")

# Save gene statistics
gene_stats = adata.var[['n_cells_expr', 'highly_variable', 'mt', 'ribo', 'hb']]
gene_stats.to_csv(f"{QC_DIR}/motorneurons_gene_stats.csv")
print(f"Gene statistics saved in: {QC_DIR}")

print("\n" + "="*80)
print("COMPREHENSIVE QC ANALYSIS COMPLETED!")
print("="*80)

# %%
