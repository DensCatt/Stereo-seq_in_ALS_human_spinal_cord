#%%
#################################################
######## MOTORNEURON COMPARISON WITH THESIS #####
#################################################
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None  # Will skip text adjustment if not installed

# Set plot style for better aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# Define color palette
COLORS = {
    'gautier': '#E63946',      # Red
    'thesis': '#457B9D',       # Blue
    'gautier_light': '#F4A5A5',
    'thesis_light': '#A8DADC',
    'significant_up': '#E63946',
    'significant_down': '#457B9D',
    'non_significant': '#CCCCCC'
}

# Set output directory
output_dir = "C:\\Users\\catta\\Desktop\\ALS\\THESIS\\Spatial_vs_Gautier"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# Load the THESIS motorneurons
os.chdir("C:\\Users\\catta\\Desktop\\ALS\\THESIS\\")
adata_thesis_mns = sc.read_h5ad("motorneurons_bin20_ALL_LAYERS_processed.h5ad")
print(f"THESIS motorneurons shape (all): {adata_thesis_mns.shape}")
print(f"THESIS motorneurons obs columns: {adata_thesis_mns.obs.columns.tolist()}")

# Filter to only CTR condition
print(f"\nCondition values in THESIS data: {adata_thesis_mns.obs['condition'].value_counts().to_dict()}")
adata_thesis_mns = adata_thesis_mns[adata_thesis_mns.obs['condition'] == 'CTR'].copy()
print(f"THESIS motorneurons shape (CTR only): {adata_thesis_mns.shape}")

#%%
### Step 1: Load Gautier motorneurons from RDS file ###
os.chdir("C:\\Users\\catta\\Desktop\\ALS\\Gautier\\")

# Check if h5ad file already exists (converted from RDS)
if os.path.exists("gautier_mns.h5ad"):
    print("Loading pre-converted h5ad file...")
    adata_gautier_mns = sc.read_h5ad("gautier_mns.h5ad")
    print(f"Gautier motorneurons shape: {adata_gautier_mns.shape}")
    print(f"Gautier motorneurons obs columns: {adata_gautier_mns.obs.columns.tolist()}")

# Check if CSV/MTX files exist (exported from R)
elif os.path.exists("gautier_mns_counts.mtx"):
    print("Loading from CSV/MTX files...")
    from scipy.io import mmread

    # Load components
    counts = mmread("gautier_mns_counts.mtx").T.tocsr()
    metadata = pd.read_csv("gautier_mns_metadata.csv", index_col=0)
    genes = pd.read_csv("gautier_mns_genes.csv")
    cells = pd.read_csv("gautier_mns_cells.csv")

    # Create AnnData object
    adata_gautier_mns = ad.AnnData(X=counts, obs=metadata, var=genes.set_index('gene_name'))
    adata_gautier_mns.obs_names = cells['barcode'].values

    # Add counts layer
    adata_gautier_mns.layers['counts'] = adata_gautier_mns.X.copy()

    # Load normalized data if available
    if os.path.exists("gautier_mns_normalized.mtx"):
        norm_data = mmread("gautier_mns_normalized.mtx").T.tocsr()
        adata_gautier_mns.layers['lognorm'] = norm_data
        adata_gautier_mns.X = norm_data  # Set as default

    # Load PCA if available
    if os.path.exists("gautier_mns_pca.csv"):
        pca_df = pd.read_csv("gautier_mns_pca.csv", index_col=0)
        adata_gautier_mns.obsm['X_pca'] = pca_df.values

    # Load UMAP if available
    if os.path.exists("gautier_mns_umap.csv"):
        umap_df = pd.read_csv("gautier_mns_umap.csv", index_col=0)
        adata_gautier_mns.obsm['X_umap'] = umap_df.values

    print(f"\n✓ Loaded Gautier motorneurons from CSV/MTX files")
    print(f"Shape: {adata_gautier_mns.shape}")
    print(f"Obs columns: {adata_gautier_mns.obs.columns.tolist()}")
    print(f"Layers: {list(adata_gautier_mns.layers.keys())}")
    print(f"Obsm keys: {list(adata_gautier_mns.obsm.keys())}")

    # Save as h5ad for future use
    adata_gautier_mns.write("gautier_mns.h5ad")
    print("✓ Saved as gautier_mns.h5ad for future use")

else:
    print("❌ Gautier MN data not found!")
    print("\nPlease run the following in R to convert the RDS file:")
    print("  source('convert_rds_simple.R')")
    print("\nThis will export the data to CSV/MTX format.")
    raise FileNotFoundError("Run convert_rds_simple.R in R first")

#%%
### Step 2: Explore Gautier MN annotations ###
# Check what metadata is available
print("\nGautier MN metadata columns:")
print(adata_gautier_mns.obs.columns.tolist())

# Check for cell type annotations, condition, etc.
for col in adata_gautier_mns.obs.columns:
    if adata_gautier_mns.obs[col].dtype == 'object' or adata_gautier_mns.obs[col].dtype.name == 'category':
        print(f"\n{col} unique values:")
        print(adata_gautier_mns.obs[col].value_counts())

#%%
### Step 3: QC Comparison between Gautier and THESIS ###
print("\n" + "="*60)
print("QC METRICS COMPARISON: GAUTIER vs THESIS")
print("="*60)

# Calculate QC metrics if not present
for adata, name in [(adata_gautier_mns, 'Gautier'), (adata_thesis_mns, 'THESIS')]:
    if 'total_counts' not in adata.obs.columns or 'n_genes_by_counts' not in adata.obs.columns:
        print(f"\nCalculating QC metrics for {name}...")
        sc.pp.calculate_qc_metrics(adata, inplace=True, log1p=False)

# Summary statistics comparison
print("\n" + "-"*60)
print("GAUTIER Summary:")
gautier_qc_cols = ['total_counts', 'n_genes_by_counts']
if 'pct_counts_mt' in adata_gautier_mns.obs.columns:
    gautier_qc_cols.append('pct_counts_mt')
if 'pct_counts_ribo' in adata_gautier_mns.obs.columns:
    gautier_qc_cols.append('pct_counts_ribo')
print(adata_gautier_mns.obs[gautier_qc_cols].describe())

print("\n" + "-"*60)
print("THESIS Summary:")
thesis_qc_cols = ['total_counts', 'n_genes_by_counts']
if 'pct_counts_mt' in adata_thesis_mns.obs.columns:
    thesis_qc_cols.append('pct_counts_mt')
if 'pct_counts_ribo' in adata_thesis_mns.obs.columns:
    thesis_qc_cols.append('pct_counts_ribo')
print(adata_thesis_mns.obs[thesis_qc_cols].describe())

# Visual comparison
common_qc_metrics = list(set(gautier_qc_cols) & set(thesis_qc_cols))
n_metrics = len(common_qc_metrics)

fig, axes = plt.subplots(2, n_metrics, figsize=(5*n_metrics, 10))
if n_metrics == 1:
    axes = axes.reshape(-1, 1)

for idx, metric in enumerate(common_qc_metrics):
    # Prepare data for seaborn
    df_qc = pd.DataFrame({
        'Dataset': ['Gautier'] * len(adata_gautier_mns) + ['Spatial (CTR)'] * len(adata_thesis_mns),
        metric: np.concatenate([adata_gautier_mns.obs[metric].values,
                                adata_thesis_mns.obs[metric].values])
    })

    # Violin plots with seaborn
    sns.violinplot(data=df_qc, x='Dataset', y=metric, ax=axes[0, idx],
                   palette=[COLORS['gautier'], COLORS['thesis']],
                   inner='box', linewidth=1.5, saturation=0.9)
    axes[0, idx].set_xlabel('')
    axes[0, idx].set_ylabel(metric.replace('_', ' ').title())
    axes[0, idx].set_title(f'{metric.replace("_", " ").title()}')

    # Add median annotations
    gautier_median = adata_gautier_mns.obs[metric].median()
    thesis_median = adata_thesis_mns.obs[metric].median()
    axes[0, idx].annotate(f'med: {gautier_median:.0f}', xy=(0, gautier_median),
                          xytext=(0.25, gautier_median), fontsize=9, color=COLORS['gautier'],
                          fontweight='bold')
    axes[0, idx].annotate(f'med: {thesis_median:.0f}', xy=(1, thesis_median),
                          xytext=(1.1, thesis_median), fontsize=9, color=COLORS['thesis'],
                          fontweight='bold')

    # KDE plots (overlaid) - cleaner than histograms
    sns.kdeplot(data=adata_gautier_mns.obs[metric], ax=axes[1, idx],
                color=COLORS['gautier'], fill=True, alpha=0.4, linewidth=2,
                label=f'Gautier (n={len(adata_gautier_mns)})')
    sns.kdeplot(data=adata_thesis_mns.obs[metric], ax=axes[1, idx],
                color=COLORS['thesis'], fill=True, alpha=0.4, linewidth=2,
                label=f'Spatial CTR (n={len(adata_thesis_mns)})')
    axes[1, idx].set_xlabel(metric.replace('_', ' ').title())
    axes[1, idx].set_ylabel('Density')
    axes[1, idx].set_title(f'{metric.replace("_", " ").title()} Distribution')
    axes[1, idx].legend(frameon=True, fancybox=True, shadow=True)

plt.suptitle('QC Metrics Comparison: Gautier vs Spatial (CTR)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Statistical comparison
print("\n" + "="*60)
print("STATISTICAL TESTS (Mann-Whitney U)")
print("="*60)
from scipy.stats import mannwhitneyu

for metric in common_qc_metrics:
    stat, pval = mannwhitneyu(adata_gautier_mns.obs[metric],
                              adata_thesis_mns.obs[metric],
                              alternative='two-sided')
    gautier_med = adata_gautier_mns.obs[metric].median()
    thesis_med = adata_thesis_mns.obs[metric].median()
    fold_change = gautier_med / thesis_med if thesis_med != 0 else float('inf')

    print(f"\n{metric}:")
    print(f"  Gautier median: {gautier_med:.2f}")
    print(f"  THESIS median: {thesis_med:.2f}")
    print(f"  Fold change: {fold_change:.2f}x")
    print(f"  p-value: {pval:.2e}")
    if pval < 0.001:
        print(f"  *** Highly significant difference")
    elif pval < 0.05:
        print(f"  * Significant difference")
    else:
        print(f"  ns (not significant)")

print("="*60 + "\n")

#%%
### Step 4: Prepare datasets for comparison ###
# Add dataset labels
adata_gautier_mns.obs['dataset'] = 'Gautier'
adata_thesis_mns.obs['dataset'] = 'THESIS'

# Remove spatial-specific layers if they exist
if 'intron' in adata_thesis_mns.layers:
    del adata_thesis_mns.layers['intron']
if 'exon' in adata_thesis_mns.layers:
    del adata_thesis_mns.layers['exon']

# Check normalization status
print(f"\nGautier MNs layers: {list(adata_gautier_mns.layers.keys())}")
print(f"THESIS MNs layers: {list(adata_thesis_mns.layers.keys())}")

#%%
### Most Abundant Genes in Each Dataset ###
print("\n" + "="*60)
print("MOST ABUNDANT GENES BY MEAN EXPRESSION")
print("="*60)

# Calculate mean expression for Gautier
X_gautier_abund = adata_gautier_mns.X.toarray() if hasattr(adata_gautier_mns.X, 'toarray') else adata_gautier_mns.X
gautier_mean_expr = pd.Series(X_gautier_abund.mean(axis=0), index=adata_gautier_mns.var_names)
gautier_top = gautier_mean_expr.sort_values(ascending=False).head(20)

# Calculate mean expression for Spatial/THESIS
X_thesis_abund = adata_thesis_mns.X.toarray() if hasattr(adata_thesis_mns.X, 'toarray') else adata_thesis_mns.X
thesis_mean_expr = pd.Series(X_thesis_abund.mean(axis=0), index=adata_thesis_mns.var_names)
thesis_top = thesis_mean_expr.sort_values(ascending=False).head(20)

# Get overlap info
gautier_top_genes = set(gautier_top.index)
thesis_top_genes = set(thesis_top.index)
overlap = gautier_top_genes & thesis_top_genes

print(f"Shared in top 20: {len(overlap)} genes")
print(f"Unique to Gautier: {len(gautier_top_genes - thesis_top_genes)}")
print(f"Unique to Spatial: {len(thesis_top_genes - gautier_top_genes)}")

#%%
### Visualization: Most Abundant Genes - Side-by-Side Bar Plot ###
from matplotlib.patches import Patch

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Plot 1: Gautier top genes (horizontal bar)
ax1 = axes[0]
colors_gautier = [COLORS['gautier'] if g not in thesis_top_genes else COLORS['gautier_light'] for g in gautier_top.index]
bars1 = ax1.barh(range(len(gautier_top)), gautier_top.values[::-1], color=colors_gautier[::-1],
                  edgecolor='white', linewidth=0.5)
ax1.set_yticks(range(len(gautier_top)))
ax1.set_yticklabels(gautier_top.index[::-1], fontsize=10)
ax1.set_xlabel('Mean Expression', fontsize=11)
ax1.set_title('Top 20 Abundant Genes\nGAUTIER', fontsize=14, fontweight='bold', color=COLORS['gautier'])
ax1.invert_xaxis()

# Add value labels for Gautier
for bar, val in zip(bars1, gautier_top.values[::-1]):
    ax1.text(val + 0.02 * gautier_top.max(), bar.get_y() + bar.get_height()/2, f'{val:.2f}',
             va='center', ha='left', fontsize=8, color='#333333')

# Plot 2: Spatial top genes (horizontal bar)
ax2 = axes[1]
colors_thesis = [COLORS['thesis'] if g not in gautier_top_genes else COLORS['thesis_light'] for g in thesis_top.index]
bars2 = ax2.barh(range(len(thesis_top)), thesis_top.values[::-1], color=colors_thesis[::-1],
                  edgecolor='white', linewidth=0.5)
ax2.set_yticks(range(len(thesis_top)))
ax2.set_yticklabels(thesis_top.index[::-1], fontsize=10)
ax2.set_xlabel('Mean Expression', fontsize=11)
ax2.set_title('Top 20 Abundant Genes\nSPATIAL (CTR)', fontsize=14, fontweight='bold', color=COLORS['thesis'])

# Add value labels for Spatial
for bar, val in zip(bars2, thesis_top.values[::-1]):
    ax2.text(val + 0.02 * thesis_top.max(), bar.get_y() + bar.get_height()/2, f'{val:.2f}',
             va='center', ha='left', fontsize=8, color='#333333')

# Add legend
legend_elements = [
    Patch(facecolor=COLORS['gautier'], label='Unique to Gautier top 20'),
    Patch(facecolor=COLORS['gautier_light'], label='Shared (shown in Gautier)'),
    Patch(facecolor=COLORS['thesis'], label='Unique to Spatial top 20'),
    Patch(facecolor=COLORS['thesis_light'], label='Shared (shown in Spatial)')
]
fig.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.02),
           frameon=True, fontsize=9)

plt.suptitle('Most Abundant Genes Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout(rect=[0, 0.05, 1, 0.98])
plt.savefig(os.path.join(output_dir, "abundant_genes_barplot.png"), dpi=150, bbox_inches='tight')
plt.show()

#%%
### Heatmap: Expression of Union of Top Genes Across Both Datasets ###
# Get union of top genes that exist in both datasets
common_var = adata_gautier_mns.var_names.intersection(adata_thesis_mns.var_names)
union_top_genes = list((gautier_top_genes | thesis_top_genes) & set(common_var))

# Calculate mean expression for union genes in both datasets
heatmap_data = pd.DataFrame({
    'Gautier': [gautier_mean_expr.get(g, 0) for g in union_top_genes],
    'Spatial (CTR)': [thesis_mean_expr.get(g, 0) for g in union_top_genes]
}, index=union_top_genes)

# Sort by max expression across datasets
heatmap_data['max'] = heatmap_data.max(axis=1)
heatmap_data = heatmap_data.sort_values('max', ascending=False).drop('max', axis=1)

# Z-score normalize for relative comparison
heatmap_zscore = heatmap_data.apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0, axis=1)

fig, axes = plt.subplots(1, 2, figsize=(10, 12), gridspec_kw={'width_ratios': [1, 1]})

# Heatmap 1: Raw mean expression (log scale)
ax1 = axes[0]
heatmap_log = np.log1p(heatmap_data)
im1 = ax1.imshow(heatmap_log.values, aspect='auto', cmap='YlOrRd')
ax1.set_xticks([0, 1])
ax1.set_xticklabels(['Gautier', 'Spatial (CTR)'], fontsize=11)
ax1.set_yticks(range(len(heatmap_data)))
ax1.set_yticklabels(heatmap_data.index, fontsize=9)
ax1.set_title('Mean Expression\n(log1p scale)', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=ax1, shrink=0.5, label='log1p(mean expr)')

# Heatmap 2: Z-score (relative expression)
ax2 = axes[1]
im2 = ax2.imshow(heatmap_zscore.values, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['Gautier', 'Spatial (CTR)'], fontsize=11)
ax2.set_yticks(range(len(heatmap_zscore)))
ax2.set_yticklabels(heatmap_zscore.index, fontsize=9)
ax2.set_title('Relative Expression\n(Z-score)', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=ax2, shrink=0.5, label='Z-score')

plt.suptitle('Top Abundant Genes: Expression Comparison', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "abundant_genes_heatmap.png"), dpi=150, bbox_inches='tight')
plt.show()

print(f"\nVisualizations saved to: {output_dir}")

#%%
### Step 5: Compare motorneuron marker expression ###
# Define motorneuron markers (including neurofilament genes)
mn_markers = ['CHAT', 'SLC5A7', 'MNX1', 'PALMD', 'MT1X', 'VIPR2']
 # Neurofilament heavy, light, medium, peripherin, internexin
mn_markers = mn_markers

# Get common genes between datasets
common_genes = adata_gautier_mns.var_names.intersection(adata_thesis_mns.var_names)
print(f"\nNumber of common genes: {len(common_genes)}")

# Subset both datasets to common genes
adata_gautier_mns_common = adata_gautier_mns[:, common_genes].copy()
adata_thesis_mns_common = adata_thesis_mns[:, common_genes].copy()

# Compare expression of MN markers
common_mn_markers = [gene for gene in mn_markers if gene in common_genes]
print(f"Common MN markers available: {common_mn_markers}")

if common_mn_markers:
    n_markers = len(common_mn_markers)
    n_cols = min(2, n_markers)
    n_rows = (n_markers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = np.atleast_2d(axes).flatten()

    for idx, gene in enumerate(common_mn_markers):
        gautier_expr = adata_gautier_mns_common[:, gene].X.toarray().flatten() if hasattr(adata_gautier_mns_common.X, 'toarray') else adata_gautier_mns_common[:, gene].X.flatten()
        thesis_expr = adata_thesis_mns_common[:, gene].X.toarray().flatten() if hasattr(adata_thesis_mns_common.X, 'toarray') else adata_thesis_mns_common[:, gene].X.flatten()

        # Create dataframe for seaborn
        df_expr = pd.DataFrame({
            'Dataset': ['Gautier'] * len(gautier_expr) + ['Spatial (CTR)'] * len(thesis_expr),
            'Expression': np.concatenate([gautier_expr, thesis_expr])
        })

        sns.violinplot(data=df_expr, x='Dataset', y='Expression', ax=axes[idx],
                       palette=[COLORS['gautier'], COLORS['thesis']],
                       inner='box', linewidth=1.5, saturation=0.9)
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('Expression')
        axes[idx].set_title(gene, fontsize=13, fontweight='bold')

    # Hide unused axes
    for idx in range(len(common_mn_markers), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Motor Neuron Marker Expression', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

#%%
### Step 6: Differential expression between datasets ###
print("\n" + "="*60)
print("DIFFERENTIAL EXPRESSION ANALYSIS")
print("="*60)

# Filter genes: require minimum expression in BOTH datasets to avoid extreme LFC values
print("\nFiltering genes for DE analysis...")
min_cells_pct = 0.05  # Gene must be detected in at least 5% of cells
min_mean_expr = 0.1   # Minimum mean expression threshold

# Get expression matrices
X_gautier = adata_gautier_mns_common.X.toarray() if hasattr(adata_gautier_mns_common.X, 'toarray') else adata_gautier_mns_common.X
X_thesis = adata_thesis_mns_common.X.toarray() if hasattr(adata_thesis_mns_common.X, 'toarray') else adata_thesis_mns_common.X

# Calculate detection rate and mean expression for each gene
gautier_pct_cells = (X_gautier > 0).mean(axis=0)
thesis_pct_cells = (X_thesis > 0).mean(axis=0)
gautier_mean = X_gautier.mean(axis=0)
thesis_mean = X_thesis.mean(axis=0)

# Keep genes detected in both datasets
gene_mask = (
    (gautier_pct_cells >= min_cells_pct) &
    (thesis_pct_cells >= min_cells_pct) &
    (gautier_mean >= min_mean_expr) &
    (thesis_mean >= min_mean_expr)
)

# Handle array shape
if hasattr(gene_mask, 'A1'):
    gene_mask = gene_mask.A1
gene_mask = np.array(gene_mask).flatten()

filtered_genes = adata_gautier_mns_common.var_names[gene_mask]
print(f"Genes before filtering: {adata_gautier_mns_common.n_vars}")
print(f"Genes after filtering: {len(filtered_genes)}")
print(f"Genes removed (low/no expression in one dataset): {adata_gautier_mns_common.n_vars - len(filtered_genes)}")

# Subset to filtered genes
adata_gautier_filtered = adata_gautier_mns_common[:, filtered_genes].copy()
adata_thesis_filtered = adata_thesis_mns_common[:, filtered_genes].copy()

# Concatenate datasets for DE analysis
adata_combined = ad.concat([adata_gautier_filtered, adata_thesis_filtered],
                           join='inner', label='dataset',
                           keys=['Gautier', 'THESIS'])

print(f"\nCombined dataset shape: {adata_combined.shape}")

# Ensure proper normalization for DE
if 'counts' in adata_combined.layers:
    adata_combined.X = adata_combined.layers['counts'].copy()
    sc.pp.normalize_total(adata_combined, target_sum=1e4)
    sc.pp.log1p(adata_combined)
    print("Normalized combined data for DE analysis")

# Differential expression between datasets
sc.tl.rank_genes_groups(adata_combined, groupby='dataset',
                        method='wilcoxon', key_added='de_datasets')

# Get results for both directions
print("\n" + "-"*60)
print("Top genes higher in GAUTIER vs THESIS:")
print("-"*60)
de_gautier = sc.get.rank_genes_groups_df(adata_combined, group='Gautier', key='de_datasets')
de_gautier_sorted = de_gautier.sort_values('pvals_adj')
print(de_gautier_sorted.head(20)[['names', 'scores', 'pvals_adj', 'logfoldchanges']])

print("\n" + "-"*60)
print("Top genes higher in THESIS vs GAUTIER:")
print("-"*60)
de_thesis = sc.get.rank_genes_groups_df(adata_combined, group='THESIS', key='de_datasets')
de_thesis_sorted = de_thesis.sort_values('pvals_adj')
print(de_thesis_sorted.head(20)[['names', 'scores', 'pvals_adj', 'logfoldchanges']])

# Visualize top DE genes (filtered by significance AND LFC direction)
# Genes ACTUALLY higher in Gautier (positive LFC, significant)
top_gautier_genes = de_gautier_sorted[
    (de_gautier_sorted['logfoldchanges'] > 0) &
    (de_gautier_sorted['pvals_adj'] < 0.05)
].head(5)['names'].tolist()

# Genes ACTUALLY higher in Spatial (negative LFC, significant)
top_spatial_genes = de_gautier_sorted[
    (de_gautier_sorted['logfoldchanges'] < 0) &
    (de_gautier_sorted['pvals_adj'] < 0.05)
].head(3)['names'].tolist()

# Add NEFL to Spatial row (if not already there and if present in data)
if 'NEFL' not in top_spatial_genes and 'NEFL' in filtered_genes:
    top_spatial_genes.append('NEFL')
if 'TUBA1B' not in top_spatial_genes and 'TUBA1B' in filtered_genes:
    top_spatial_genes.append('TUBA1B')

# top_de_genes = top_gautier_genes + top_spatial_genes
top_de_genes = ["GRIK2","SLC44A5","PDE4B",
                "NEFL","STMN2","TUBA1B"]
if top_de_genes:
    fig, axes = plt.subplots(2, 3, figsize=(22, 9))

    for idx, gene in enumerate(top_de_genes[:10]):
        row = idx // 3
        col = idx % 3

        gautier_expr = adata_gautier_filtered[:, gene].X.toarray().flatten() if hasattr(adata_gautier_filtered.X, 'toarray') else adata_gautier_filtered[:, gene].X.flatten()
        thesis_expr = adata_thesis_filtered[:, gene].X.toarray().flatten() if hasattr(adata_thesis_filtered.X, 'toarray') else adata_thesis_filtered[:, gene].X.flatten()

        # Create dataframe for seaborn
        df_de = pd.DataFrame({
            'Dataset': ['Gautier'] * len(gautier_expr) + ['Spatial (CTR)'] * len(thesis_expr),
            'Expression': np.concatenate([gautier_expr, thesis_expr])
        })

        sns.violinplot(data=df_de, x='Dataset', y='Expression', ax=axes[row, col],
                       palette=[COLORS['gautier'], COLORS['thesis']],
                       inner='box', linewidth=1.5, saturation=0.9)
        axes[row, col].set_xlabel('')
        axes[row, col].set_ylabel('Expression')

        # Color title based on which dataset has higher expression
        title_color = COLORS['gautier'] if row == 0 else COLORS['thesis']
        axes[row, col].set_title(gene, fontsize=12, fontweight='bold', color=title_color)

    # Add row labels
    fig.text(0.02, 0.75, 'Higher in\nGautier', fontsize=12, fontweight='bold',
             color=COLORS['gautier'], ha='center', va='center', rotation=90)
    fig.text(0.02, 0.25, 'Higher in\nSpatial (CTR)', fontsize=12, fontweight='bold',
             color=COLORS['thesis'], ha='center', va='center', rotation=90)

    plt.suptitle('Top Differentially Expressed Genes', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0.03, 0, 1, 0.98])
    plt.show()

# Volcano plot
print("\n" + "-"*60)
print("Creating volcano plot...")
print("-"*60)

fig, ax = plt.subplots(figsize=(12, 9))

# Prepare data for volcano plot
logfc = de_gautier['logfoldchanges'].values
pvals = de_gautier['pvals_adj'].values
names = de_gautier['names'].values

# Transform p-values
neg_log_pvals = -np.log10(pvals + 1e-300)  # Add small value to avoid log(0)

# Categorize points
categories = []
for lfc, pval in zip(logfc, pvals):
    if pval < 0.05 and lfc > 0.5:
        categories.append('up')      # Upregulated in Gautier
    elif pval < 0.05 and lfc < -0.5:
        categories.append('down')    # Downregulated in Gautier (higher in Spatial)
    else:
        categories.append('ns')      # Not significant

# Plot each category separately for better legend
for cat, color, label, alpha, size in [
    ('ns', COLORS['non_significant'], 'Not significant', 0.4, 15),
    ('up', COLORS['significant_up'], f'Higher in Gautier (n={(np.array(categories)=="up").sum()})', 0.7, 30),
    ('down', COLORS['significant_down'], f'Higher in Spatial CTR (n={(np.array(categories)=="down").sum()})', 0.7, 30)
]:
    mask = np.array(categories) == cat
    ax.scatter(logfc[mask], neg_log_pvals[mask], c=color, alpha=alpha, s=size,
               label=label, edgecolors='white', linewidths=0.3)

# Add threshold lines
ax.axhline(-np.log10(0.05), color='#666666', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axvline(0.5, color='#666666', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axvline(-0.5, color='#666666', linestyle='--', linewidth=1.5, alpha=0.7)

# Add threshold labels
ax.text(ax.get_xlim()[1] * 0.95, -np.log10(0.05) + 0.5, 'p = 0.05',
        fontsize=9, color='#666666', ha='right')
ax.text(0.5, ax.get_ylim()[1] * 0.95, 'LFC = 0.5', fontsize=9,
        color='#666666', ha='center', rotation=90)
ax.text(-0.5, ax.get_ylim()[1] * 0.95, 'LFC = -0.5', fontsize=9,
        color='#666666', ha='center', rotation=90)

# Label top genes
n_top_label = 15
top_indices = np.argsort(pvals)[:n_top_label]
texts = []
for idx in top_indices:
    color = COLORS['significant_up'] if logfc[idx] > 0 else COLORS['significant_down']
    texts.append(ax.annotate(names[idx], (logfc[idx], neg_log_pvals[idx]),
                             fontsize=9, fontweight='bold', color=color))

# Adjust text positions if adjustText is available
if adjust_text is not None:
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))

ax.set_xlabel('Log2 Fold Change (Gautier vs Spatial CTR)', fontsize=12)
ax.set_ylabel('-Log10(adjusted p-value)', fontsize=12)
ax.set_title('Volcano Plot: Gautier vs Spatial (CTR)', fontsize=16, fontweight='bold', pad=15)
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)

# Add subtle background shading for significant regions
ax.axvspan(0.5, ax.get_xlim()[1], alpha=0.05, color=COLORS['significant_up'])
ax.axvspan(ax.get_xlim()[0], -0.5, alpha=0.05, color=COLORS['significant_down'])

plt.tight_layout()
plt.show()


#%%
### Step 7: Save results ###
os.chdir(output_dir)

# Save Gautier MNs
adata_gautier_mns.write("Gautier_MNs_only.h5ad")
print("\n✅ Gautier motorneurons saved as 'Gautier_MNs_only.h5ad'")

# Save combined dataset (for reference)
adata_combined.write("Gautier_Spatial_MNs_combined.h5ad")
print("✅ Combined dataset saved as 'Gautier_Spatial_MNs_combined.h5ad'")

# Save DE results
de_gautier_sorted.to_csv("DE_Gautier_vs_Spatial.csv", index=False)
de_thesis_sorted.to_csv("DE_Spatial_vs_Gautier.csv", index=False)
print("✅ DE results saved:")
print("   - DE_Gautier_vs_Spatial.csv (genes higher in Gautier)")
print("   - DE_Spatial_vs_Gautier.csv (genes higher in Spatial)")

# Save QC comparison summary
qc_summary = pd.DataFrame({
    'Metric': common_qc_metrics,
    'Gautier_median': [adata_gautier_mns.obs[m].median() for m in common_qc_metrics],
    'Spatial_median': [adata_thesis_mns.obs[m].median() for m in common_qc_metrics],
    'Fold_change': [adata_gautier_mns.obs[m].median() / adata_thesis_mns.obs[m].median()
                    if adata_thesis_mns.obs[m].median() != 0 else float('inf')
                    for m in common_qc_metrics]
})
qc_summary.to_csv("QC_comparison_summary.csv", index=False)
print("✅ QC comparison summary saved as 'QC_comparison_summary.csv'")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print(f"\nOutput directory: {output_dir}")
print(f"\nDatasets compared:")
print(f"  - Gautier: {adata_gautier_mns.n_obs} cells, {adata_gautier_mns.n_vars} genes")
print(f"  - Spatial (CTR only): {adata_thesis_mns.n_obs} cells, {adata_thesis_mns.n_vars} genes")
print(f"  - Common genes: {len(common_genes)}")
print(f"  - Significant DE genes (p<0.05): {(de_gautier_sorted['pvals_adj'] < 0.05).sum()}")
print("="*60)