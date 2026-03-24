#%% COMPREHENSIVE GENE SET ENRICHMENT ANALYSIS
# Following methods from ALS motor neuron meta-analysis
# Analyzes up and down regulated DEGs for pathway/ontology enrichment

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# GSEA packages
import gseapy as gp

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

#%% ============================================================================
# CONFIGURATION
# =============================================================================

WORK_DIR = r"C:\Users\catta\Desktop\ALS\THESIS"
DEA_RESULTS_DIR = os.path.join(WORK_DIR, "multilayer_DEA_01")  # Adjust path
OUTPUT_DIR = os.path.join(WORK_DIR, "GSEA_results")

# DEA thresholds
FDR_THRESHOLD = 0.05
LOGFC_THRESHOLD = 1.0

# Which layer to use for GSEA (counts, exon, or intron)
LAYER = "counts"

# Background gene universe (if available - set to None to use default)
# This should be the set of all expressed genes in your dataset
BACKGROUND_GENES_FILE = None  # or path to file with one gene per line

# Enrichr libraries to query
ENRICHR_LIBRARIES = [
    # Gene Ontology
    'GO_Biological_Process_2023',
    'GO_Molecular_Function_2023',
    'GO_Cellular_Component_2023',
    # Pathways
    'KEGG_2021_Human',
    'WikiPathway_2023_Human',
    'Reactome_2022',
    # Disease
    'DisGeNET',
    'OMIM_Disease',
    # Drug/Chemical
    'DSigDB',
    # MSigDB collections
    'MSigDB_Hallmark_2020',
    'MSigDB_Oncogenic_Signatures',
    # Cell type markers
    'CellMarker_Augmented_2021',
    'PanglaoDB_Augmented_2021',
    # Transcription factors
    'TRANSFAC_and_JASPAR_PWMs',
    'ChEA_2022',
    # miRNA targets
    'miRTarBase_2017',
    'TargetScan_microRNA_2017',
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

#%% ============================================================================
# LOAD DEA RESULTS
# =============================================================================
print("="*70)
print("LOADING DEA RESULTS")
print("="*70)

# Load significant genes
sig_file = os.path.join(DEA_RESULTS_DIR, LAYER, f'dea_{LAYER}_significant.tsv')
all_file = os.path.join(DEA_RESULTS_DIR, LAYER, f'dea_{LAYER}_results.tsv')

if os.path.exists(sig_file):
    sig_df = pd.read_csv(sig_file, sep='\t', index_col=0)
    print(f"Loaded {len(sig_df)} significant genes from {LAYER} layer")
else:
    print(f"Significant genes file not found: {sig_file}")
    sig_df = None

if os.path.exists(all_file):
    all_df = pd.read_csv(all_file, sep='\t', index_col=0)
    print(f"Loaded {len(all_df)} total genes from {LAYER} layer")
else:
    print(f"All results file not found: {all_file}")
    all_df = None

#%% Separate up and down regulated genes
if sig_df is not None:
    # Use the significance thresholds
    up_genes = sig_df[sig_df['logFC'] > 0].index.tolist()
    down_genes = sig_df[sig_df['logFC'] < 0].index.tolist()

    print(f"\nUpregulated genes (ALS > CTR): {len(up_genes)}")
    print(f"Downregulated genes (ALS < CTR): {len(down_genes)}")

    # Show top genes
    print("\nTop 10 upregulated genes:")
    top_up = sig_df[sig_df['logFC'] > 0].nlargest(10, 'logFC')[['logFC', 'adj.P.Val']]
    print(top_up)

    print("\nTop 10 downregulated genes:")
    top_down = sig_df[sig_df['logFC'] < 0].nsmallest(10, 'logFC')[['logFC', 'adj.P.Val']]
    print(top_down)

#%% Load background gene universe (if provided)
if BACKGROUND_GENES_FILE and os.path.exists(BACKGROUND_GENES_FILE):
    with open(BACKGROUND_GENES_FILE, 'r') as f:
        background_genes = [line.strip() for line in f if line.strip()]
    print(f"\nBackground gene universe: {len(background_genes)} genes")
elif all_df is not None:
    # Use all tested genes as background
    background_genes = all_df.index.tolist()
    print(f"\nUsing all tested genes as background: {len(background_genes)} genes")
else:
    background_genes = None
    print("\nNo background gene set defined - using Enrichr defaults")

#%% ============================================================================
# ENRICHR OVER-REPRESENTATION ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("ENRICHR OVER-REPRESENTATION ANALYSIS")
print("="*70)

def run_enrichr_analysis(gene_list, gene_set_name, libraries, output_subdir):
    """Run Enrichr analysis for a gene list across multiple libraries."""
    if len(gene_list) == 0:
        print(f"No genes in {gene_set_name} list, skipping...")
        return None

    out_dir = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    all_results = []

    for library in libraries:
        print(f"  Querying {library}...")
        try:
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=library,
                organism='human',
                outdir=None,  # Don't save individual files
                cutoff=0.05,
                no_plot=True
            )

            if enr.results is not None and len(enr.results) > 0:
                results = enr.results.copy()
                results['Library'] = library
                results['Gene_Set'] = gene_set_name
                all_results.append(results)
                n_sig = (results['Adjusted P-value'] < 0.05).sum()
                print(f"    Found {n_sig} significant terms (FDR < 0.05)")

        except Exception as e:
            print(f"    Error querying {library}: {e}")

    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        # Save combined results
        combined_results.to_csv(os.path.join(out_dir, f'{gene_set_name}_enrichr_results.tsv'),
                               sep='\t', index=False)
        return combined_results
    return None


#%% Run Enrichr for upregulated genes
print("\n--- Analyzing UPREGULATED genes ---")
up_results = run_enrichr_analysis(up_genes, 'upregulated', ENRICHR_LIBRARIES, 'upregulated')

#%% Run Enrichr for downregulated genes
print("\n--- Analyzing DOWNREGULATED genes ---")
down_results = run_enrichr_analysis(down_genes, 'downregulated', ENRICHR_LIBRARIES, 'downregulated')

#%% ============================================================================
# GSEA (PRE-RANKED) ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("GSEA PRE-RANKED ANALYSIS")
print("="*70)

if all_df is not None:
    # Create ranked gene list based on signed -log10(p-value)
    # This preserves direction (up vs down) and significance
    ranked_genes = all_df.copy()
    ranked_genes['rank_metric'] = -np.log10(ranked_genes['P.Value'] + 1e-300) * np.sign(ranked_genes['logFC'])
    ranked_genes = ranked_genes.sort_values('rank_metric', ascending=False)

    # Save ranked list
    rank_file = os.path.join(OUTPUT_DIR, 'ranked_gene_list.rnk')
    ranked_genes[['rank_metric']].to_csv(rank_file, sep='\t', header=False)
    print(f"Saved ranked gene list: {rank_file}")

    # Run GSEA prerank for key gene sets
    gsea_libraries = [
        'GO_Biological_Process_2023',
        'KEGG_2021_Human',
        'Reactome_2022',
        'MSigDB_Hallmark_2020',
    ]

    gsea_results = {}

    for library in gsea_libraries:
        print(f"\nRunning GSEA prerank for {library}...")
        try:
            pre_res = gp.prerank(
                rnk=ranked_genes[['rank_metric']],
                gene_sets=library,
                threads=4,
                min_size=15,
                max_size=500,
                permutation_num=1000,
                outdir=os.path.join(OUTPUT_DIR, 'gsea_prerank', library),
                seed=42,
                verbose=True
            )

            gsea_results[library] = pre_res.res2d
            n_sig = (pre_res.res2d['FDR q-val'] < 0.25).sum()
            print(f"  Found {n_sig} significant terms (FDR < 0.25)")

        except Exception as e:
            print(f"  Error running GSEA for {library}: {e}")

    # Combine GSEA results
    if gsea_results:
        all_gsea = []
        for lib, res in gsea_results.items():
            res_copy = res.copy()
            res_copy['Library'] = lib
            all_gsea.append(res_copy)

        gsea_combined = pd.concat(all_gsea, ignore_index=True)
        gsea_combined.to_csv(os.path.join(OUTPUT_DIR, 'gsea_prerank_combined_results.tsv'),
                            sep='\t', index=False)
        print(f"\nSaved combined GSEA results")

#%% ============================================================================
# VISUALIZATION: ENRICHMENT BARPLOTS
# =============================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

def plot_enrichment_barplot(results_df, title, output_file, top_n=15):
    """Create horizontal barplot of top enriched terms."""
    if results_df is None or len(results_df) == 0:
        print(f"No results to plot for {title}")
        return

    # Filter significant results
    sig_results = results_df[results_df['Adjusted P-value'] < 0.05].copy()

    if len(sig_results) == 0:
        print(f"No significant results for {title}")
        return

    # Get top terms by adjusted p-value
    top_results = sig_results.nsmallest(top_n, 'Adjusted P-value')

    # Create plot
    fig, ax = plt.subplots(figsize=(12, max(6, len(top_results) * 0.4)))

    # Calculate -log10(adjusted p-value)
    top_results['-log10(FDR)'] = -np.log10(top_results['Adjusted P-value'])

    # Create barplot
    colors = plt.cm.RdYlBu_r(top_results['-log10(FDR)'] / top_results['-log10(FDR)'].max())

    bars = ax.barh(range(len(top_results)), top_results['-log10(FDR)'],
                   color=colors, edgecolor='black', linewidth=0.5)

    ax.set_yticks(range(len(top_results)))
    ax.set_yticklabels(top_results['Term'].str[:60])  # Truncate long names
    ax.invert_yaxis()
    ax.set_xlabel('-log10(Adjusted P-value)', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=12)

    # Add gene count labels
    for i, (idx, row) in enumerate(top_results.iterrows()):
        genes_in_term = row['Genes'].split(';') if pd.notna(row['Genes']) else []
        ax.text(row['-log10(FDR)'] + 0.1, i, f"n={len(genes_in_term)}",
               va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.show()

#%% Plot top enriched terms for upregulated genes
if up_results is not None:
    # GO Biological Process
    go_bp_up = up_results[up_results['Library'].str.contains('Biological_Process')]
    plot_enrichment_barplot(go_bp_up, 'GO Biological Process - Upregulated in ALS',
                           os.path.join(OUTPUT_DIR, 'GO_BP_upregulated.png'))

    # KEGG
    kegg_up = up_results[up_results['Library'].str.contains('KEGG')]
    plot_enrichment_barplot(kegg_up, 'KEGG Pathways - Upregulated in ALS',
                           os.path.join(OUTPUT_DIR, 'KEGG_upregulated.png'))

    # Reactome
    reactome_up = up_results[up_results['Library'].str.contains('Reactome')]
    plot_enrichment_barplot(reactome_up, 'Reactome Pathways - Upregulated in ALS',
                           os.path.join(OUTPUT_DIR, 'Reactome_upregulated.png'))

#%% ORA Dot Plot - Publication quality
def plot_ora_dotplot(results_df, title, output_file, top_n=20):
    """
    Create publication-quality dot plot for ORA (Enrichr) results.
    - X-axis: -log10(Adjusted P-value)
    - Y-axis: Gene set terms
    - Dot size: Significance (-log10 FDR) - larger = more significant
    - Dot color: Gene count
    """
    if results_df is None or len(results_df) == 0:
        print(f"No ORA results to plot for {title}")
        return

    # Filter significant results
    sig_results = results_df[results_df['Adjusted P-value'] < 0.05].copy()

    if len(sig_results) == 0:
        print(f"No significant ORA results (FDR < 0.05) for {title}")
        return

    # Get top terms by adjusted p-value
    top_results = sig_results.nsmallest(top_n, 'Adjusted P-value').copy()

    # Calculate -log10(adjusted p-value)
    top_results['-log10(FDR)'] = -np.log10(top_results['Adjusted P-value'].clip(lower=1e-50))

    # Extract gene count
    top_results['gene_count'] = top_results['Genes'].apply(
        lambda x: len(str(x).split(';')) if pd.notna(x) else 0)

    # Sort by significance for display
    top_results = top_results.sort_values('-log10(FDR)', ascending=True)

    # Normalize -log10(FDR) for dot sizing
    min_size, max_size = 80, 500
    if top_results['-log10(FDR)'].nunique() > 1:
        sizes = (top_results['-log10(FDR)'] - top_results['-log10(FDR)'].min()) / \
                (top_results['-log10(FDR)'].max() - top_results['-log10(FDR)'].min())
        sizes = sizes * (max_size - min_size) + min_size
    else:
        sizes = [200] * len(top_results)

    # Create figure
    fig, ax = plt.subplots(figsize=(9, max(5, len(top_results) * 0.35)))

    # Create scatter plot
    scatter = ax.scatter(
        top_results['-log10(FDR)'],
        range(len(top_results)),
        c=top_results['gene_count'],
        s=sizes,
        cmap='viridis',
        edgecolors='black',
        linewidths=0.5,
        alpha=0.85
    )

    # Formatting
    ax.set_yticks(range(len(top_results)))
    # Add library info to term name if available
    if 'Library' in top_results.columns:
        labels = [f"{row['Term'][:45]} ({row['Library'].split('_')[0]})"
                  for _, row in top_results.iterrows()]
    else:
        labels = top_results['Term'].str[:55].tolist()
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('-log₁₀(Adjusted P-value)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    # Add colorbar for gene count
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('Gene Count', fontsize=10)

    # Add size legend for FDR
    fdr_legend_vals = [top_results['Adjusted P-value'].max(),
                       top_results['Adjusted P-value'].median(),
                       top_results['Adjusted P-value'].min()]
    fdr_legend_vals = [v for v in fdr_legend_vals if pd.notna(v)]

    if len(set(fdr_legend_vals)) > 1:
        legend_sizes = []
        for val in fdr_legend_vals:
            log_val = -np.log10(max(val, 1e-50))
            if top_results['-log10(FDR)'].max() != top_results['-log10(FDR)'].min():
                norm_val = (log_val - top_results['-log10(FDR)'].min()) / \
                           (top_results['-log10(FDR)'].max() - top_results['-log10(FDR)'].min())
                legend_sizes.append(norm_val * (max_size - min_size) + min_size)
            else:
                legend_sizes.append(200)

        for val, size in zip(fdr_legend_vals, legend_sizes):
            ax.scatter([], [], s=size, c='grey', edgecolors='black',
                      linewidths=0.5, label=f'FDR={val:.2e}')
        ax.legend(title='Significance', loc='lower right', fontsize=8,
                 title_fontsize=9, framealpha=0.9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.show()


#%% Generate ORA dot plots
if up_results is not None:
    plot_ora_dotplot(up_results, 'ORA: Top Enriched Terms - Upregulated in ALS',
                     os.path.join(OUTPUT_DIR, 'ORA_dotplot_upregulated.png'), top_n=25)

if down_results is not None:
    plot_ora_dotplot(down_results, 'ORA: Top Enriched Terms - Downregulated in ALS',
                     os.path.join(OUTPUT_DIR, 'ORA_dotplot_downregulated.png'), top_n=25)


#%% Combined ORA plots (top terms across all libraries)
if up_results is not None:
    plot_enrichment_barplot(up_results, 'Top Enriched Terms - Upregulated in ALS (All Libraries)',
                           os.path.join(OUTPUT_DIR, 'ORA_upregulated_combined.png'), top_n=20)

if down_results is not None:
    plot_enrichment_barplot(down_results, 'Top Enriched Terms - Downregulated in ALS (All Libraries)',
                           os.path.join(OUTPUT_DIR, 'ORA_downregulated_combined.png'), top_n=20)


#%% Plot GSEA prerank results - Publication-quality dot plot
def plot_gsea_dotplot(gsea_df, title, output_file, top_n=20, fdr_threshold=0.25):
    """
    Create publication-quality dot plot for GSEA results.
    - X-axis: Normalized Enrichment Score (NES)
    - Y-axis: Gene set terms
    - Dot size: Significance (-log10 FDR) - larger = more significant
    - Dot color: Gene count in leading edge
    """
    if gsea_df is None or len(gsea_df) == 0:
        print(f"No GSEA results to plot for {title}")
        return

    # Ensure numeric columns are properly typed
    gsea_df = gsea_df.copy()
    gsea_df['NES'] = pd.to_numeric(gsea_df['NES'], errors='coerce')
    gsea_df['FDR q-val'] = pd.to_numeric(gsea_df['FDR q-val'], errors='coerce')

    # Filter by FDR threshold
    sig_results = gsea_df[gsea_df['FDR q-val'] < fdr_threshold].copy()

    if len(sig_results) == 0:
        print(f"No significant GSEA results (FDR < {fdr_threshold}) for {title}")
        return

    # Get top terms by absolute NES (using sort instead of nlargest for robustness)
    sig_results['abs_NES'] = sig_results['NES'].abs()
    top_results = sig_results.sort_values('abs_NES', ascending=False).head(top_n).copy()
    top_results = top_results.sort_values('NES', ascending=True)

    # Calculate -log10(FDR) for size mapping (larger = more significant)
    top_results['-log10(FDR)'] = -np.log10(top_results['FDR q-val'].clip(lower=1e-10))

    # Extract gene set size from the results (if available) for color mapping
    if 'Lead_genes' in top_results.columns:
        top_results['gene_count'] = top_results['Lead_genes'].apply(
            lambda x: len(str(x).split(';')) if pd.notna(x) else 10)
    elif 'Gene %' in top_results.columns:
        top_results['gene_count'] = 50  # Default
    else:
        top_results['gene_count'] = 50

    # Normalize -log10(FDR) for dot sizing (larger dots = more significant)
    min_size, max_size = 80, 500
    if top_results['-log10(FDR)'].nunique() > 1:
        sizes = (top_results['-log10(FDR)'] - top_results['-log10(FDR)'].min()) / \
                (top_results['-log10(FDR)'].max() - top_results['-log10(FDR)'].min())
        sizes = sizes * (max_size - min_size) + min_size
    else:
        sizes = [200] * len(top_results)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, max(5, len(top_results) * 0.32)))

    # Create scatter plot: size = FDR significance, color = gene count
    scatter = ax.scatter(
        top_results['NES'],
        range(len(top_results)),
        c=top_results['gene_count'],
        s=sizes,
        cmap='viridis',
        edgecolors='black',
        linewidths=0.5,
        alpha=0.85
    )

    # Formatting
    ax.set_yticks(range(len(top_results)))
    ax.set_yticklabels(top_results['Term'].str[:55], fontsize=9)
    ax.set_xlabel('Normalized Enrichment Score (NES)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.axvline(x=0, color='grey', linewidth=0.8, linestyle='--', alpha=0.7)

    # Add colorbar for gene count
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('Gene Count', fontsize=10)

    # Add size legend for FDR
    fdr_legend_vals = [top_results['FDR q-val'].max(),
                       top_results['FDR q-val'].median(),
                       top_results['FDR q-val'].min()]
    fdr_legend_vals = [v for v in fdr_legend_vals if pd.notna(v)]

    if len(set(fdr_legend_vals)) > 1:
        legend_sizes = []
        for val in fdr_legend_vals:
            log_val = -np.log10(max(val, 1e-10))
            if top_results['-log10(FDR)'].max() != top_results['-log10(FDR)'].min():
                norm_val = (log_val - top_results['-log10(FDR)'].min()) / \
                           (top_results['-log10(FDR)'].max() - top_results['-log10(FDR)'].min())
                legend_sizes.append(norm_val * (max_size - min_size) + min_size)
            else:
                legend_sizes.append(200)

        for val, size in zip(fdr_legend_vals, legend_sizes):
            ax.scatter([], [], s=size, c='grey', edgecolors='black',
                      linewidths=0.5, label=f'FDR={val:.2e}')
        ax.legend(title='Significance', loc='lower right', fontsize=8,
                 title_fontsize=9, framealpha=0.9)

    # Add direction annotations
    xlim = ax.get_xlim()
    ax.text(xlim[1] * 0.95, len(top_results) + 0.5, 'ALS ↑', ha='right',
            fontsize=9, color='#d62728', fontweight='bold')
    ax.text(xlim[0] * 0.95, len(top_results) + 0.5, 'ALS ↓', ha='left',
            fontsize=9, color='#1f77b4', fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.show()


#%% Generate GSEA dot plots
if 'gsea_results' in dir() and gsea_results:
    for library, res_df in gsea_results.items():
        lib_name = library.replace('_', ' ').replace('2023', '').replace('2022', '').replace('2021', '').strip()
        plot_gsea_dotplot(res_df, f'GSEA: {lib_name}',
                          os.path.join(OUTPUT_DIR, f'GSEA_{library}.png'))

    # Combined top hits
    if 'gsea_combined' in dir() and gsea_combined is not None:
        plot_gsea_dotplot(gsea_combined, 'GSEA: Top Enriched Pathways',
                          os.path.join(OUTPUT_DIR, 'GSEA_combined_top.png'), top_n=25)


#%% Plot top enriched terms for downregulated genes
if down_results is not None:
    # GO Biological Process
    go_bp_down = down_results[down_results['Library'].str.contains('Biological_Process')]
    plot_enrichment_barplot(go_bp_down, 'GO Biological Process - Downregulated in ALS',
                           os.path.join(OUTPUT_DIR, 'GO_BP_downregulated.png'))

    # GO Cellular Component
    go_cc_down = down_results[down_results['Library'].str.contains('Cellular_Component')]
    plot_enrichment_barplot(go_cc_down, 'GO Cellular Component - Downregulated in ALS',
                           os.path.join(OUTPUT_DIR, 'GO_CC_downregulated.png'))

    # GO Molecular Function
    go_mf_down = down_results[down_results['Library'].str.contains('Molecular_Function')]
    plot_enrichment_barplot(go_mf_down, 'GO Molecular Function - Downregulated in ALS',
                           os.path.join(OUTPUT_DIR, 'GO_MF_downregulated.png'))

    # KEGG
    kegg_down = down_results[down_results['Library'].str.contains('KEGG')]
    plot_enrichment_barplot(kegg_down, 'KEGG Pathways - Downregulated in ALS',
                           os.path.join(OUTPUT_DIR, 'KEGG_downregulated.png'))

    # Reactome
    reactome_down = down_results[down_results['Library'].str.contains('Reactome')]
    plot_enrichment_barplot(reactome_down, 'Reactome Pathways - Downregulated in ALS',
                           os.path.join(OUTPUT_DIR, 'Reactome_downregulated.png'))

#%% ============================================================================
# SUMMARY REPORT
# =============================================================================
print("\n" + "="*70)
print("GSEA ANALYSIS SUMMARY")
print("="*70)

summary_data = []

# Summarize Enrichr results
for direction, results in [('Upregulated', up_results), ('Downregulated', down_results)]:
    if results is None:
        continue

    for library in results['Library'].unique():
        lib_results = results[results['Library'] == library]
        n_tested = len(lib_results)
        n_sig = (lib_results['Adjusted P-value'] < 0.05).sum()

        summary_data.append({
            'Direction': direction,
            'Library': library,
            'Terms_Tested': n_tested,
            'Terms_Significant': n_sig,
            'Percent_Significant': f"{100*n_sig/n_tested:.1f}%" if n_tested > 0 else "0%"
        })

summary_df = pd.DataFrame(summary_data)
print("\nEnrichment Summary:")
print(summary_df.to_string(index=False))

# Save summary
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'enrichment_summary.tsv'), sep='\t', index=False)

#%% Save top terms across all analyses
print("\n--- Top 10 Enriched Terms (All Libraries) ---")

for direction, results in [('Upregulated', up_results), ('Downregulated', down_results)]:
    if results is None:
        continue

    print(f"\n{direction}:")
    top_terms = results.nsmallest(10, 'Adjusted P-value')[['Term', 'Library', 'Adjusted P-value', 'Genes']]
    top_terms['N_Genes'] = top_terms['Genes'].apply(lambda x: len(x.split(';')) if pd.notna(x) else 0)
    print(top_terms[['Term', 'Library', 'Adjusted P-value', 'N_Genes']].to_string(index=False))

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print(f"Results saved to: {OUTPUT_DIR}")
print("="*70)

# %%
