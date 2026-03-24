#!/usr/bin/env python3
"""
Effect Size and STMN2 Correlation Analysis
===========================================

Analyzes and visualizes the results from:
- genes_ranked_by_effect_size.csv (Cohen's d, fold change, median difference)
- stmn2_correlated_genes.csv (Spearman correlations with STMN2 retention)

Generates publication-ready figures exploring:
1. Effect size distributions and top genes
2. STMN2 correlation patterns
3. Overlap between high-effect genes and STMN2-correlated genes

Author: Analysis pipeline
Date: 2024
"""

#%%
# =============================================================================
# IMPORTS
# =============================================================================
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
# from adjustText import adjust_text
import warnings
warnings.filterwarnings('ignore')

# Try to import adjustText for label positioning
try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False
    print("Note: adjustText not installed. Labels may overlap. Install with: pip install adjustText")

#%%
# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration parameters."""

    # Input/Output paths
    INPUT_DIR = "C:\\Users\\catta\\Desktop\\ALS\\THESIS\\Transcriptome_integrity_general\\tables"
    OUTPUT_DIR = "C:\\Users\\catta\\Desktop\\ALS\\THESIS\\Transcriptome_integrity_general\\effect_correlation_analysis"

    # Input files
    EFFECT_SIZE_FILE = "genes_ranked_by_effect_size.csv"
    STMN2_CORR_FILE = "stmn2_correlated_genes.csv"

    # Analysis parameters
    FDR_THRESHOLD = 0.05
    EFFECT_SIZE_SMALL = 0.2   # Cohen's d threshold for small effect
    EFFECT_SIZE_MEDIUM = 0.5  # Cohen's d threshold for medium effect
    EFFECT_SIZE_LARGE = 0.8   # Cohen's d threshold for large effect

    # Stringent thresholds for high-confidence gene selection
    LOG2FC_THRESHOLD = 1.0    # log2 fold change threshold (FC > 2)
    COHENS_D_THRESHOLD = 0.8  # Cohen's d threshold for stringent filtering

    # Candidate genes to highlight
    CANDIDATE_GENES = ["STMN2", "UNC13A", "MEG8", "SNAP25", "TUBA1B", 
                       "EIF4H", "HSPA8", "NEFL", "CALM1", "UNC5C", "STMN1"]

    # Visualization
    FIGURE_DPI = 300
    TOP_N_GENES = 20  # Number of top genes to show in bar plots

    # Color palette
    COLORS = {
        'positive': '#E74C3C',    # Red for increased retention in ALS
        'negative': '#3498DB',    # Blue for decreased retention in ALS
        'neutral': '#95A5A6',     # Gray for non-significant
        'highlight': '#F39C12',   # Orange for candidate genes
        'als': '#E74C3C',
        'ctr': '#3498DB'
    }


#%%
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_data(config):
    """Load the analysis results."""
    print("="*80)
    print("LOADING DATA")
    print("="*80)

    effect_path = os.path.join(config.INPUT_DIR, config.EFFECT_SIZE_FILE)
    corr_path = os.path.join(config.INPUT_DIR, config.STMN2_CORR_FILE)

    # Load effect size data
    if os.path.exists(effect_path):
        effect_df = pd.read_csv(effect_path)
        print(f"Loaded effect size data: {len(effect_df)} genes")
    else:
        print(f"WARNING: Effect size file not found: {effect_path}")
        effect_df = None

    # Load STMN2 correlation data
    if os.path.exists(corr_path):
        corr_df = pd.read_csv(corr_path)
        print(f"Loaded STMN2 correlation data: {len(corr_df)} genes")
    else:
        print(f"WARNING: STMN2 correlation file not found: {corr_path}")
        corr_df = None

    return effect_df, corr_df


def create_output_dir(config):
    """Create output directory."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {config.OUTPUT_DIR}")
    return config.OUTPUT_DIR


#%%
# =============================================================================
# EFFECT SIZE ANALYSIS PLOTS
# =============================================================================

def plot_effect_size_distribution(effect_df, config, outdir):
    """Plot distribution of Cohen's d effect sizes."""
    print("\nGenerating effect size distribution plot...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Histogram of Cohen's d
    ax1 = axes[0]
    ax1.hist(effect_df['cohens_d'], bins=50, color=config.COLORS['neutral'],
             edgecolor='white', alpha=0.7)
    ax1.axvline(0, color='black', linestyle='-', linewidth=1)
    ax1.axvline(config.EFFECT_SIZE_SMALL, color='green', linestyle='--',
                linewidth=1, label=f'Small effect (d={config.EFFECT_SIZE_SMALL})')
    ax1.axvline(-config.EFFECT_SIZE_SMALL, color='green', linestyle='--', linewidth=1)
    ax1.axvline(config.EFFECT_SIZE_MEDIUM, color='orange', linestyle='--',
                linewidth=1, label=f'Medium effect (d={config.EFFECT_SIZE_MEDIUM})')
    ax1.axvline(-config.EFFECT_SIZE_MEDIUM, color='orange', linestyle='--', linewidth=1)
    ax1.set_xlabel("Cohen's d (ALS - CTR)", fontsize=11)
    ax1.set_ylabel("Number of genes", fontsize=11)
    ax1.set_title("Distribution of Effect Sizes", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)

    # 2. Histogram of fold change (log2)
    ax2 = axes[1]
    log2_fc = np.log2(effect_df['fold_change'])
    log2_fc = log2_fc.replace([np.inf, -np.inf], np.nan).dropna()
    ax2.hist(log2_fc, bins=50, color=config.COLORS['neutral'],
             edgecolor='white', alpha=0.7)
    ax2.axvline(0, color='black', linestyle='-', linewidth=1)
    ax2.axvline(1, color='red', linestyle='--', linewidth=1, label='2-fold change')
    ax2.axvline(-1, color='red', linestyle='--', linewidth=1)
    ax2.set_xlabel("log2(Fold Change)", fontsize=11)
    ax2.set_ylabel("Number of genes", fontsize=11)
    ax2.set_title("Distribution of Fold Changes", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)

    # 3. Histogram of median difference
    ax3 = axes[2]
    ax3.hist(effect_df['median_diff'], bins=50, color=config.COLORS['neutral'],
             edgecolor='white', alpha=0.7)
    ax3.axvline(0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel("Median Difference (Log2 Intron/Exon)", fontsize=11)
    ax3.set_ylabel("Number of genes", fontsize=11)
    ax3.set_title("Distribution of Median Differences", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'effect_size_distributions.png'),
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'effect_size_distributions.pdf'),
                bbox_inches='tight')
    print("  Saved: effect_size_distributions.png/pdf")

    return fig


def plot_top_effect_size_genes(effect_df, config, outdir):
    """Plot top genes by effect size (both directions)."""
    print("\nGenerating top effect size genes plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    n = config.TOP_N_GENES

    # 1. Top genes with INCREASED retention in ALS (positive Cohen's d)
    ax1 = axes[0]
    top_positive = effect_df.nlargest(n, 'cohens_d')
    colors = [config.COLORS['highlight'] if g in config.CANDIDATE_GENES
              else config.COLORS['positive'] for g in top_positive['gene']]

    bars1 = ax1.barh(range(n), top_positive['cohens_d'].values, color=colors)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(top_positive['gene'].values)
    ax1.invert_yaxis()
    ax1.set_xlabel("Cohen's d", fontsize=11)
    ax1.set_title(f"Top {n} Genes: INCREASED Retention in ALS",
                  fontsize=12, fontweight='bold')
    ax1.axvline(config.EFFECT_SIZE_SMALL, color='green', linestyle='--',
                alpha=0.5, label='Small')
    ax1.axvline(config.EFFECT_SIZE_MEDIUM, color='orange', linestyle='--',
                alpha=0.5, label='Medium')
    ax1.axvline(config.EFFECT_SIZE_LARGE, color='red', linestyle='--',
                alpha=0.5, label='Large')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, top_positive['cohens_d'].values)):
        ax1.text(val + 0.02, i, f'{val:.2f}', va='center', fontsize=8)

    # 2. Top genes with DECREASED retention in ALS (negative Cohen's d)
    ax2 = axes[1]
    top_negative = effect_df.nsmallest(n, 'cohens_d')
    colors = [config.COLORS['highlight'] if g in config.CANDIDATE_GENES
              else config.COLORS['negative'] for g in top_negative['gene']]

    bars2 = ax2.barh(range(n), top_negative['cohens_d'].values, color=colors)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(top_negative['gene'].values)
    ax2.invert_yaxis()
    ax2.set_xlabel("Cohen's d", fontsize=11)
    ax2.set_title(f"Top {n} Genes: DECREASED Retention in ALS",
                  fontsize=12, fontweight='bold')
    ax2.axvline(-config.EFFECT_SIZE_SMALL, color='green', linestyle='--',
                alpha=0.5)
    ax2.axvline(-config.EFFECT_SIZE_MEDIUM, color='orange', linestyle='--',
                alpha=0.5)
    ax2.axvline(-config.EFFECT_SIZE_LARGE, color='red', linestyle='--',
                alpha=0.5)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, top_negative['cohens_d'].values)):
        ax2.text(val - 0.02, i, f'{val:.2f}', va='center', ha='right', fontsize=8)

    # Add legend for candidate genes
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=config.COLORS['highlight'], label='Candidate gene'),
                       Patch(facecolor=config.COLORS['positive'], label='Other (increased)'),
                       Patch(facecolor=config.COLORS['negative'], label='Other (decreased)')]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(os.path.join(outdir, 'top_effect_size_genes.png'),
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'top_effect_size_genes.pdf'),
                bbox_inches='tight')
    print("  Saved: top_effect_size_genes.png/pdf")

    return fig


def plot_candidate_genes_effect(effect_df, config, outdir):
    """Plot effect sizes for candidate genes specifically."""
    print("\nGenerating candidate genes effect size plot...")

    # Filter to candidate genes
    candidates = effect_df[effect_df['gene'].isin(config.CANDIDATE_GENES)].copy()
    candidates = candidates.sort_values('cohens_d', ascending=True)

    if len(candidates) == 0:
        print("  No candidate genes found in effect size data.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by direction
    colors = [config.COLORS['positive'] if d > 0 else config.COLORS['negative']
              for d in candidates['cohens_d']]

    bars = ax.barh(range(len(candidates)), candidates['cohens_d'].values, color=colors)
    ax.set_yticks(range(len(candidates)))
    ax.set_yticklabels(candidates['gene'].values)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(config.EFFECT_SIZE_SMALL, color='green', linestyle='--',
               alpha=0.5, label=f'Small (d={config.EFFECT_SIZE_SMALL})')
    ax.axvline(-config.EFFECT_SIZE_SMALL, color='green', linestyle='--', alpha=0.5)
    ax.axvline(config.EFFECT_SIZE_MEDIUM, color='orange', linestyle='--',
               alpha=0.5, label=f'Medium (d={config.EFFECT_SIZE_MEDIUM})')
    ax.axvline(-config.EFFECT_SIZE_MEDIUM, color='orange', linestyle='--', alpha=0.5)

    ax.set_xlabel("Cohen's d (ALS - CTR)", fontsize=12)
    ax.set_ylabel("Gene", fontsize=12)
    ax.set_title("Effect Sizes for Candidate TDP-43 Target Genes",
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')

    # Add value labels
    for i, (bar, val, fc) in enumerate(zip(bars, candidates['cohens_d'].values,
                                            candidates['fold_change'].values)):
        x_pos = val + 0.02 if val > 0 else val - 0.02
        ha = 'left' if val > 0 else 'right'
        ax.text(x_pos, i, f'd={val:.2f}, FC={fc:.2f}', va='center', ha=ha, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'candidate_genes_effect_sizes.png'),
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'candidate_genes_effect_sizes.pdf'),
                bbox_inches='tight')
    print("  Saved: candidate_genes_effect_sizes.png/pdf")

    return fig


#%%
# =============================================================================
# STMN2 CORRELATION ANALYSIS PLOTS
# =============================================================================

def plot_stmn2_correlation_volcano(corr_df, config, outdir):
    """Create volcano-style plot for STMN2 correlations."""
    print("\nGenerating STMN2 correlation volcano plot...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate -log10(FDR)
    corr_df = corr_df.copy()
    corr_df['neg_log10_fdr'] = -np.log10(corr_df['fdr'].clip(lower=1e-300))

    # Classify points
    sig_pos = (corr_df['fdr'] < config.FDR_THRESHOLD) & (corr_df['correlation'] > 0)
    sig_neg = (corr_df['fdr'] < config.FDR_THRESHOLD) & (corr_df['correlation'] < 0)
    not_sig = ~(sig_pos | sig_neg)

    # Plot non-significant
    ax.scatter(corr_df.loc[not_sig, 'correlation'],
               corr_df.loc[not_sig, 'neg_log10_fdr'],
               c=config.COLORS['neutral'], s=20, alpha=0.4, label='Not significant')

    # Plot significant positive
    ax.scatter(corr_df.loc[sig_pos, 'correlation'],
               corr_df.loc[sig_pos, 'neg_log10_fdr'],
               c=config.COLORS['positive'], s=30, alpha=0.7,
               label=f'Positive corr (FDR<{config.FDR_THRESHOLD})')

    # Plot significant negative
    ax.scatter(corr_df.loc[sig_neg, 'correlation'],
               corr_df.loc[sig_neg, 'neg_log10_fdr'],
               c=config.COLORS['negative'], s=30, alpha=0.7,
               label=f'Negative corr (FDR<{config.FDR_THRESHOLD})')

    # Highlight candidate genes
    candidates = corr_df[corr_df['gene'].isin(config.CANDIDATE_GENES)]
    if len(candidates) > 0:
        ax.scatter(candidates['correlation'], candidates['neg_log10_fdr'],
                   c=config.COLORS['highlight'], s=100, marker='*',
                   edgecolors='black', linewidths=0.5, label='Candidate genes', zorder=5)

        # Add labels for candidate genes
        texts = []
        for _, row in candidates.iterrows():
            texts.append(ax.text(row['correlation'], row['neg_log10_fdr'],
                                row['gene'], fontsize=9, fontweight='bold'))
        if HAS_ADJUSTTEXT and len(texts) > 0:
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # Add threshold line
    ax.axhline(-np.log10(config.FDR_THRESHOLD), color='gray', linestyle='--',
               linewidth=1, alpha=0.7)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Spearman Correlation with STMN2", fontsize=12)
    ax.set_ylabel("-log10(FDR)", fontsize=12)
    ax.set_title("Genes Correlated with STMN2 Intron Retention",
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'stmn2_correlation_volcano.png'),
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'stmn2_correlation_volcano.pdf'),
                bbox_inches='tight')
    print("  Saved: stmn2_correlation_volcano.png/pdf")

    return fig


def plot_stmn2_correlation_distribution(corr_df, config, outdir):
    """Plot distribution of STMN2 correlations."""
    print("\nGenerating STMN2 correlation distribution plot...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Histogram of correlations
    ax1 = axes[0]
    ax1.hist(corr_df['correlation'], bins=50, color=config.COLORS['neutral'],
             edgecolor='white', alpha=0.7)
    ax1.axvline(0, color='black', linestyle='-', linewidth=1)
    ax1.axvline(corr_df['correlation'].mean(), color='red', linestyle='--',
                linewidth=2, label=f"Mean = {corr_df['correlation'].mean():.3f}")
    ax1.set_xlabel("Spearman Correlation with STMN2", fontsize=11)
    ax1.set_ylabel("Number of genes", fontsize=11)
    ax1.set_title("Distribution of STMN2 Correlations", fontsize=12, fontweight='bold')
    ax1.legend()

    # 2. Distribution of significant correlations only
    ax2 = axes[1]
    sig = corr_df[corr_df['fdr'] < config.FDR_THRESHOLD]

    if len(sig) > 0:
        sig_pos = sig[sig['correlation'] > 0]
        sig_neg = sig[sig['correlation'] < 0]

        ax2.hist(sig_pos['correlation'], bins=30, color=config.COLORS['positive'],
                 edgecolor='white', alpha=0.7, label=f'Positive (n={len(sig_pos)})')
        ax2.hist(sig_neg['correlation'], bins=30, color=config.COLORS['negative'],
                 edgecolor='white', alpha=0.7, label=f'Negative (n={len(sig_neg)})')
        ax2.axvline(0, color='black', linestyle='-', linewidth=1)
        ax2.legend()

    ax2.set_xlabel("Spearman Correlation with STMN2", fontsize=11)
    ax2.set_ylabel("Number of genes", fontsize=11)
    ax2.set_title(f"Significant Correlations (FDR < {config.FDR_THRESHOLD})",
                  fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'stmn2_correlation_distribution.png'),
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'stmn2_correlation_distribution.pdf'),
                bbox_inches='tight')
    print("  Saved: stmn2_correlation_distribution.png/pdf")

    return fig


def plot_top_stmn2_correlated_genes(corr_df, config, outdir):
    """Plot top genes correlated with STMN2."""
    print("\nGenerating top STMN2 correlated genes plot...")

    # Filter to significant only
    sig = corr_df[corr_df['fdr'] < config.FDR_THRESHOLD].copy()

    if len(sig) == 0:
        print("  No significant correlations found.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    n = min(config.TOP_N_GENES, len(sig))

    # 1. Top positively correlated
    ax1 = axes[0]
    top_pos = sig.nlargest(n, 'correlation')
    colors = [config.COLORS['highlight'] if g in config.CANDIDATE_GENES
              else config.COLORS['positive'] for g in top_pos['gene']]

    bars1 = ax1.barh(range(len(top_pos)), top_pos['correlation'].values, color=colors)
    ax1.set_yticks(range(len(top_pos)))
    ax1.set_yticklabels(top_pos['gene'].values)
    ax1.invert_yaxis()
    ax1.set_xlabel("Spearman Correlation", fontsize=11)
    ax1.set_title(f"Top {n} Genes POSITIVELY Correlated with STMN2",
                  fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 1)

    # Add FDR values
    for i, (val, fdr) in enumerate(zip(top_pos['correlation'].values, top_pos['fdr'].values)):
        ax1.text(val + 0.02, i, f'r={val:.2f}\nFDR={fdr:.1e}', va='center', fontsize=8)

    # 2. Top negatively correlated
    ax2 = axes[1]
    top_neg = sig.nsmallest(n, 'correlation')

    if len(top_neg) > 0 and top_neg['correlation'].min() < 0:
        colors = [config.COLORS['highlight'] if g in config.CANDIDATE_GENES
                  else config.COLORS['negative'] for g in top_neg['gene']]

        bars2 = ax2.barh(range(len(top_neg)), top_neg['correlation'].values, color=colors)
        ax2.set_yticks(range(len(top_neg)))
        ax2.set_yticklabels(top_neg['gene'].values)
        ax2.invert_yaxis()
        ax2.set_xlabel("Spearman Correlation", fontsize=11)
        ax2.set_title(f"Top {n} Genes NEGATIVELY Correlated with STMN2",
                      fontsize=12, fontweight='bold')
        ax2.set_xlim(-1, 0)

        for i, (val, fdr) in enumerate(zip(top_neg['correlation'].values, top_neg['fdr'].values)):
            ax2.text(val - 0.02, i, f'r={val:.2f}\nFDR={fdr:.1e}', va='center', ha='right', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No significant negative\ncorrelations',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title("Negatively Correlated Genes", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'top_stmn2_correlated_genes.png'),
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'top_stmn2_correlated_genes.pdf'),
                bbox_inches='tight')
    print("  Saved: top_stmn2_correlated_genes.png/pdf")

    return fig


#%%
# =============================================================================
# INTEGRATED ANALYSIS PLOTS
# =============================================================================

def plot_effect_vs_correlation(effect_df, corr_df, config, outdir):
    """Plot effect size vs STMN2 correlation."""
    print("\nGenerating effect size vs correlation plot...")

    # Merge the two datasets
    merged = effect_df.merge(corr_df, on='gene', how='inner')

    if len(merged) == 0:
        print("  No overlapping genes found.")
        return None

    print(f"  {len(merged)} genes with both effect size and correlation data")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Classify by significance
    sig_corr = merged['fdr'] < config.FDR_THRESHOLD
    large_effect = merged['cohens_d'].abs() > config.EFFECT_SIZE_LARGE

    # Plot categories
    neither = ~sig_corr & ~large_effect
    only_effect = ~sig_corr & large_effect
    only_corr = sig_corr & ~large_effect
    both = sig_corr & large_effect

    ax.scatter(merged.loc[neither, 'cohens_d'], merged.loc[neither, 'correlation'],
               c=config.COLORS['neutral'], s=20, alpha=0.3, label='Neither')
    ax.scatter(merged.loc[only_effect, 'cohens_d'], merged.loc[only_effect, 'correlation'],
               c='purple', s=40, alpha=0.6, label=f'Large effect only (|d|>{config.EFFECT_SIZE_LARGE})')
    ax.scatter(merged.loc[only_corr, 'cohens_d'], merged.loc[only_corr, 'correlation'],
               c='green', s=40, alpha=0.6, label=f'Sig. correlation only (FDR<{config.FDR_THRESHOLD})')
    ax.scatter(merged.loc[both, 'cohens_d'], merged.loc[both, 'correlation'],
               c='red', s=60, alpha=0.8, label='Both')

    # Highlight candidate genes
    candidates = merged[merged['gene'].isin(config.CANDIDATE_GENES)]
    if len(candidates) > 0:
        ax.scatter(candidates['cohens_d'], candidates['correlation'],
                   c=config.COLORS['highlight'], s=150, marker='*',
                   edgecolors='black', linewidths=1, label='Candidate genes', zorder=5)

        texts = []
        for _, row in candidates.iterrows():
            texts.append(ax.text(row['cohens_d'], row['correlation'],
                                row['gene'], fontsize=10, fontweight='bold'))
        if HAS_ADJUSTTEXT and len(texts) > 0:
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # Reference lines
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    # Threshold lines
    ax.axvline(config.COHENS_D_THRESHOLD, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(-config.COHENS_D_THRESHOLD, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Cohen's d (Effect Size: ALS - CTR)", fontsize=12)
    ax.set_ylabel("Spearman Correlation with STMN2", fontsize=12)
    ax.set_title("Effect Size vs STMN2 Correlation", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

    # Add correlation between effect size and STMN2 correlation
    r, p = stats.spearmanr(merged['cohens_d'], merged['correlation'])
    ax.text(0.95, 0.05, f'Overall Spearman r = {r:.3f}\np = {p:.2e}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'effect_vs_correlation.png'),
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'effect_vs_correlation.pdf'),
                bbox_inches='tight')
    print("  Saved: effect_vs_correlation.png/pdf")

    return fig, merged


def plot_venn_overlap(effect_df, corr_df, config, outdir):
    """Visualize overlap between high-effect and STMN2-correlated genes."""
    print("\nGenerating overlap analysis...")

    # Define gene sets using MEDIUM thresholds (for general overview)
    large_effect_pos = set(effect_df[effect_df['cohens_d'] > config.EFFECT_SIZE_MEDIUM]['gene'])
    large_effect_neg = set(effect_df[effect_df['cohens_d'] < -config.EFFECT_SIZE_MEDIUM]['gene'])
    sig_corr_pos = set(corr_df[(corr_df['fdr'] < config.FDR_THRESHOLD) &
                                (corr_df['correlation'] > 0)]['gene'])
    sig_corr_neg = set(corr_df[(corr_df['fdr'] < config.FDR_THRESHOLD) &
                                (corr_df['correlation'] < 0)]['gene'])

    # Calculate overlaps
    overlap_pos_pos = large_effect_pos & sig_corr_pos  # Both increased in ALS and correlated with STMN2
    overlap_neg_neg = large_effect_neg & sig_corr_neg

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Summary bar chart
    ax1 = axes[0]
    categories = ['Large positive\neffect', 'Positive STMN2\ncorrelation',
                  'Both', 'Large negative\neffect', 'Negative STMN2\ncorrelation']
    values = [len(large_effect_pos), len(sig_corr_pos), len(overlap_pos_pos),
              len(large_effect_neg), len(sig_corr_neg)]
    colors = [config.COLORS['positive'], config.COLORS['positive'], 'purple',
              config.COLORS['negative'], config.COLORS['negative']]

    bars = ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel("Number of genes", fontsize=11)
    ax1.set_title("Gene Set Sizes and Overlap", fontsize=12, fontweight='bold')

    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(val), ha='center', va='bottom', fontsize=10)

    # 2. List overlapping genes
    ax2 = axes[1]
    ax2.axis('off')

    text = "GENES WITH BOTH LARGE EFFECT AND STMN2 CORRELATION\n"
    text += "="*50 + "\n\n"

    text += f"Increased in ALS + Positive STMN2 correlation ({len(overlap_pos_pos)} genes):\n"
    if len(overlap_pos_pos) > 0:
        for g in sorted(overlap_pos_pos)[:15]:
            marker = " *" if g in config.CANDIDATE_GENES else ""
            text += f"  • {g}{marker}\n"
        if len(overlap_pos_pos) > 15:
            text += f"  ... and {len(overlap_pos_pos) - 15} more\n"
    else:
        text += "  (none)\n"

    text += f"\nDecreased in ALS + Negative STMN2 correlation ({len(overlap_neg_neg)} genes):\n"
    if len(overlap_neg_neg) > 0:
        for g in sorted(overlap_neg_neg)[:15]:
            marker = " *" if g in config.CANDIDATE_GENES else ""
            text += f"  • {g}{marker}\n"
        if len(overlap_neg_neg) > 15:
            text += f"  ... and {len(overlap_neg_neg) - 15} more\n"
    else:
        text += "  (none)\n"

    text += "\n* = Candidate TDP-43 target gene"

    ax2.text(0.05, 0.95, text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'overlap_analysis.png'),
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'overlap_analysis.pdf'),
                bbox_inches='tight')
    print("  Saved: overlap_analysis.png/pdf")

    # Save overlap genes to CSV (using MEDIUM thresholds)
    overlap_data = []
    for gene in overlap_pos_pos:
        eff = effect_df[effect_df['gene'] == gene].iloc[0]
        corr = corr_df[corr_df['gene'] == gene].iloc[0]
        overlap_data.append({
            'gene': gene,
            'cohens_d': eff['cohens_d'],
            'fold_change': eff['fold_change'],
            'log2_fc': np.log2(eff['fold_change']) if eff['fold_change'] > 0 else np.nan,
            'stmn2_correlation': corr['correlation'],
            'stmn2_fdr': corr['fdr'],
            'is_candidate': gene in config.CANDIDATE_GENES,
            'category': 'increased_positive_corr'
        })

    for gene in overlap_neg_neg:
        eff = effect_df[effect_df['gene'] == gene].iloc[0]
        corr = corr_df[corr_df['gene'] == gene].iloc[0]
        overlap_data.append({
            'gene': gene,
            'cohens_d': eff['cohens_d'],
            'fold_change': eff['fold_change'],
            'log2_fc': np.log2(eff['fold_change']) if eff['fold_change'] > 0 else np.nan,
            'stmn2_correlation': corr['correlation'],
            'stmn2_fdr': corr['fdr'],
            'is_candidate': gene in config.CANDIDATE_GENES,
            'category': 'decreased_negative_corr'
        })

    if len(overlap_data) > 0:
        overlap_df = pd.DataFrame(overlap_data)
        overlap_df.to_csv(os.path.join(outdir, 'overlap_medium_threshold_genes.csv'),
                          index=False)
        print("  Saved: overlap_medium_threshold_genes.csv")

    return fig


def plot_stringent_high_effect_genes(effect_df, corr_df, config, outdir):
    """
    Identify and plot genes meeting STRINGENT criteria:
    - Fold Change > 2 (i.e., log2FC > 1)
    - Cohen's d > 0.8
    - Significant STMN2 correlation (FDR < 0.05)
    """
    print("\nGenerating STRINGENT high-effect gene analysis...")
    print(f"  Criteria: log2FC > {config.LOG2FC_THRESHOLD}, Cohen's d > {config.COHENS_D_THRESHOLD}, FDR < {config.FDR_THRESHOLD}")

    # Calculate log2FC for effect_df
    effect_df = effect_df.copy()
    effect_df['log2_fc'] = np.log2(effect_df['fold_change'].clip(lower=1e-10))

    # Apply STRINGENT filters for INCREASED retention in ALS
    stringent_increased = effect_df[
        (effect_df['log2_fc'] > config.LOG2FC_THRESHOLD) &
        (effect_df['cohens_d'] > config.COHENS_D_THRESHOLD)
    ]['gene'].tolist()

    # Apply STRINGENT filters for DECREASED retention in ALS
    stringent_decreased = effect_df[
        (effect_df['log2_fc'] < -config.LOG2FC_THRESHOLD) &
        (effect_df['cohens_d'] < -config.COHENS_D_THRESHOLD)
    ]['gene'].tolist()

    print(f"  Genes with log2FC > {config.LOG2FC_THRESHOLD} AND Cohen's d > {config.COHENS_D_THRESHOLD}: {len(stringent_increased)}")
    print(f"  Genes with log2FC < -{config.LOG2FC_THRESHOLD} AND Cohen's d < -{config.COHENS_D_THRESHOLD}: {len(stringent_decreased)}")

    # Find overlap with significant STMN2 correlations
    sig_corr_pos = set(corr_df[(corr_df['fdr'] < config.FDR_THRESHOLD) &
                                (corr_df['correlation'] > 0)]['gene'])
    sig_corr_neg = set(corr_df[(corr_df['fdr'] < config.FDR_THRESHOLD) &
                                (corr_df['correlation'] < 0)]['gene'])

    # High-confidence genes: stringent effect + STMN2 correlated
    high_conf_increased = set(stringent_increased) & sig_corr_pos
    high_conf_decreased = set(stringent_decreased) & sig_corr_neg

    print(f"  High-confidence INCREASED (+ STMN2 corr): {len(high_conf_increased)}")
    print(f"  High-confidence DECREASED (+ STMN2 corr): {len(high_conf_decreased)}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 1. Scatter plot: log2FC vs Cohen's d with stringent thresholds
    ax1 = axes[0]

    # Merge effect and correlation data
    merged = effect_df.merge(corr_df[['gene', 'correlation', 'fdr']], on='gene', how='left')

    # Use consistent color scheme from config
    color_stmn2 = '#9B59B6'  # Purple for STMN2 (consistent with pathway colors)

    # Plot all genes
    ax1.scatter(merged['log2_fc'], merged['cohens_d'],
                c=config.COLORS['neutral'], s=15, alpha=0.4, label='All genes')

    # Highlight stringent genes (increased)
    stringent_inc_df = merged[merged['gene'].isin(stringent_increased)]
    ax1.scatter(stringent_inc_df['log2_fc'], stringent_inc_df['cohens_d'],
                c=config.COLORS['positive'], s=45, alpha=0.75,
                edgecolors='white', linewidths=0.3,
                label=f'Stringent increased (n={len(stringent_increased)})')

    # Highlight stringent genes (decreased)
    stringent_dec_df = merged[merged['gene'].isin(stringent_decreased)]
    ax1.scatter(stringent_dec_df['log2_fc'], stringent_dec_df['cohens_d'],
                c=config.COLORS['negative'], s=45, alpha=0.75,
                edgecolors='white', linewidths=0.3,
                label=f'Stringent decreased (n={len(stringent_decreased)})')

    # Highlight high-confidence genes
    high_conf_df = merged[merged['gene'].isin(high_conf_increased | high_conf_decreased)]
    if len(high_conf_df) > 0:
        ax1.scatter(high_conf_df['log2_fc'], high_conf_df['cohens_d'],
                    c=config.COLORS['highlight'], s=80, alpha=0.9,
                    edgecolors='black', linewidths=0.8,
                    label=f'High-confidence (n={len(high_conf_df)})', zorder=5)

    # Highlight and label STMN2 specifically
    stmn2_row = merged[merged['gene'] == 'STMN2']
    if len(stmn2_row) > 0:
        ax1.scatter(stmn2_row['log2_fc'], stmn2_row['cohens_d'],
                    c=color_stmn2, s=150, marker='*',
                    edgecolors='black', linewidths=1,
                    label='STMN2', zorder=10)
        # Add label for STMN2
        ax1.annotate('STMN2',
                     xy=(stmn2_row['log2_fc'].values[0], stmn2_row['cohens_d'].values[0]),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=11, fontweight='bold', color=color_stmn2,
                     arrowprops=dict(arrowstyle='->', color=color_stmn2, lw=1.5))

    # Add threshold lines using config colors
    ax1.axhline(config.COHENS_D_THRESHOLD, color=config.COLORS['positive'], linestyle='--', linewidth=1.5, alpha=0.6)
    ax1.axhline(-config.COHENS_D_THRESHOLD, color=config.COLORS['positive'], linestyle='--', linewidth=1.5, alpha=0.6)
    ax1.axvline(config.LOG2FC_THRESHOLD, color=config.COLORS['negative'], linestyle='--', linewidth=1.5, alpha=0.6)
    ax1.axvline(-config.LOG2FC_THRESHOLD, color=config.COLORS['negative'], linestyle='--', linewidth=1.5, alpha=0.6)
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    ax1.set_xlabel("log2(Fold Change)", fontsize=12)
    ax1.set_ylabel("Cohen's d", fontsize=12)
    ax1.set_title(f"Stringent Filtering: log2FC > {config.LOG2FC_THRESHOLD}, |d| > {config.COHENS_D_THRESHOLD}",
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.grid(alpha=0.2)

    # 2. Summary bar plot showing gene COUNTS meeting effect size criteria
    ax2 = axes[1]

    # Calculate counts for different effect size criteria
    n_cohens_only = len(effect_df[effect_df['cohens_d'].abs() > config.COHENS_D_THRESHOLD])
    n_fc_only = len(effect_df[effect_df['log2_fc'].abs() > config.LOG2FC_THRESHOLD])
    n_both_effect = len(stringent_increased) + len(stringent_decreased)

    # Create bar plot with config colors
    categories = [
        f"|Cohen's d| > {config.COHENS_D_THRESHOLD}",
        f"|log2FC| > {config.LOG2FC_THRESHOLD}\n(FC > 2)",
        "Both criteria"
    ]
    counts = [n_cohens_only, n_fc_only, n_both_effect]
    bar_colors = [config.COLORS['positive'], config.COLORS['negative'], config.COLORS['highlight']]

    bars = ax2.bar(range(len(categories)), counts, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, fontsize=11)
    ax2.set_ylabel("Number of genes", fontsize=12)
    ax2.set_title("Gene Counts Meeting Effect Size Criteria", fontsize=12, fontweight='bold')

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax2.set_ylim(0, max(counts) * 1.15)  # Add space for labels
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'stringent_high_effect_genes.png'),
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'stringent_high_effect_genes.pdf'),
                bbox_inches='tight')
    print("  Saved: stringent_high_effect_genes.png/pdf")

    # Save high-confidence genes to CSV
    high_conf_data_list = []

    for gene in high_conf_increased:
        eff = effect_df[effect_df['gene'] == gene].iloc[0]
        corr_row = corr_df[corr_df['gene'] == gene].iloc[0]
        high_conf_data_list.append({
            'gene': gene,
            'cohens_d': eff['cohens_d'],
            'fold_change': eff['fold_change'],
            'log2_fc': eff['log2_fc'],
            'median_diff': eff['median_diff'],
            'als_median': eff['als_median'],
            'ctr_median': eff['ctr_median'],
            'n_als': eff['n_als'],
            'n_ctr': eff['n_ctr'],
            'stmn2_correlation': corr_row['correlation'],
            'stmn2_pval': corr_row['pval'],
            'stmn2_fdr': corr_row['fdr'],
            'is_candidate': gene in config.CANDIDATE_GENES,
            'category': 'increased_in_ALS'
        })

    for gene in high_conf_decreased:
        eff = effect_df[effect_df['gene'] == gene].iloc[0]
        corr_row = corr_df[corr_df['gene'] == gene].iloc[0]
        high_conf_data_list.append({
            'gene': gene,
            'cohens_d': eff['cohens_d'],
            'fold_change': eff['fold_change'],
            'log2_fc': eff['log2_fc'],
            'median_diff': eff['median_diff'],
            'als_median': eff['als_median'],
            'ctr_median': eff['ctr_median'],
            'n_als': eff['n_als'],
            'n_ctr': eff['n_ctr'],
            'stmn2_correlation': corr_row['correlation'],
            'stmn2_pval': corr_row['pval'],
            'stmn2_fdr': corr_row['fdr'],
            'is_candidate': gene in config.CANDIDATE_GENES,
            'category': 'decreased_in_ALS'
        })

    if len(high_conf_data_list) > 0:
        high_conf_df = pd.DataFrame(high_conf_data_list)
        high_conf_df = high_conf_df.sort_values('cohens_d', ascending=False)
        high_conf_df.to_csv(os.path.join(outdir, 'high_effect_stmn2_correlated_genes.csv'),
                            index=False)
        print("  Saved: high_effect_stmn2_correlated_genes.csv")

        # Print summary counts
        n_increased = len([g for g in high_conf_data_list if g['category'] == 'increased_in_ALS'])
        n_decreased = len([g for g in high_conf_data_list if g['category'] == 'decreased_in_ALS'])
        n_candidates = len([g for g in high_conf_data_list if g['is_candidate']])
        print(f"\n  HIGH-CONFIDENCE GENES SUMMARY:")
        print(f"    Total genes meeting all criteria: {len(high_conf_data_list)}")
        print(f"    - Increased in ALS: {n_increased}")
        print(f"    - Decreased in ALS: {n_decreased}")
        print(f"    - Known TDP-43 targets: {n_candidates}")
    else:
        print(f"\n  No genes meet stringent criteria (FC > 2, |d| > {config.COHENS_D_THRESHOLD}, FDR < {config.FDR_THRESHOLD})")
        # Save empty file with header
        pd.DataFrame(columns=['gene', 'cohens_d', 'fold_change', 'log2_fc', 'median_diff',
                              'als_median', 'ctr_median', 'n_als', 'n_ctr',
                              'stmn2_correlation', 'stmn2_pval', 'stmn2_fdr',
                              'is_candidate', 'category']).to_csv(
            os.path.join(outdir, 'high_effect_stmn2_correlated_genes.csv'), index=False)

    return fig, high_conf_data_list


def create_summary_table(effect_df, corr_df, config, outdir):
    """Create summary statistics table."""
    print("\nGenerating summary statistics...")

    summary = []

    # Effect size statistics
    summary.append(("EFFECT SIZE ANALYSIS", ""))
    summary.append(("Total genes analyzed", len(effect_df)))
    summary.append(("Mean Cohen's d", f"{effect_df['cohens_d'].mean():.4f}"))
    summary.append(("Median Cohen's d", f"{effect_df['cohens_d'].median():.4f}"))
    summary.append((f"Genes with |d| > {config.EFFECT_SIZE_SMALL} (small)",
                    (effect_df['cohens_d'].abs() > config.EFFECT_SIZE_SMALL).sum()))
    summary.append((f"Genes with |d| > {config.EFFECT_SIZE_MEDIUM} (medium)",
                    (effect_df['cohens_d'].abs() > config.EFFECT_SIZE_MEDIUM).sum()))
    summary.append((f"Genes with |d| > {config.EFFECT_SIZE_LARGE} (large)",
                    (effect_df['cohens_d'].abs() > config.EFFECT_SIZE_LARGE).sum()))
    summary.append(("Genes with d > 0 (increased in ALS)", (effect_df['cohens_d'] > 0).sum()))
    summary.append(("Genes with d < 0 (decreased in ALS)", (effect_df['cohens_d'] < 0).sum()))

    summary.append(("", ""))
    summary.append(("STMN2 CORRELATION ANALYSIS", ""))

    if corr_df is not None and len(corr_df) > 0:
        summary.append(("Total genes analyzed", len(corr_df)))
        summary.append(("Mean correlation", f"{corr_df['correlation'].mean():.4f}"))
        summary.append(("Median correlation", f"{corr_df['correlation'].median():.4f}"))
        sig = corr_df['fdr'] < config.FDR_THRESHOLD
        summary.append((f"Significant correlations (FDR < {config.FDR_THRESHOLD})", sig.sum()))
        summary.append(("  Positive correlations", (sig & (corr_df['correlation'] > 0)).sum()))
        summary.append(("  Negative correlations", (sig & (corr_df['correlation'] < 0)).sum()))
    else:
        summary.append(("No STMN2 correlation data available", ""))

    # Create DataFrame and save
    summary_df = pd.DataFrame(summary, columns=['Metric', 'Value'])
    summary_df.to_csv(os.path.join(outdir, 'analysis_summary.csv'), index=False)

    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    for metric, value in summary:
        if metric:
            print(f"{metric}: {value}")
        else:
            print()
    print("="*60)

    return summary_df


#%%
# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Run the complete analysis."""
    print("\n" + "="*80)
    print("EFFECT SIZE AND STMN2 CORRELATION ANALYSIS")
    print("="*80 + "\n")

    config = Config()

    # Create output directory
    outdir = create_output_dir(config)

    # Load data
    effect_df, corr_df = load_data(config)

    if effect_df is None:
        print("ERROR: Effect size data not found. Exiting.")
        return None

    # Effect size analysis plots
    print("\n" + "="*80)
    print("EFFECT SIZE ANALYSIS PLOTS")
    print("="*80)

    plot_effect_size_distribution(effect_df, config, outdir)
    plot_top_effect_size_genes(effect_df, config, outdir)
    plot_candidate_genes_effect(effect_df, config, outdir)

    # STMN2 correlation analysis plots
    if corr_df is not None and len(corr_df) > 0:
        print("\n" + "="*80)
        print("STMN2 CORRELATION ANALYSIS PLOTS")
        print("="*80)

        plot_stmn2_correlation_volcano(corr_df, config, outdir)
        plot_stmn2_correlation_distribution(corr_df, config, outdir)
        plot_top_stmn2_correlated_genes(corr_df, config, outdir)

        # Integrated analysis
        print("\n" + "="*80)
        print("INTEGRATED ANALYSIS")
        print("="*80)

        plot_effect_vs_correlation(effect_df, corr_df, config, outdir)
        plot_venn_overlap(effect_df, corr_df, config, outdir)

        # Stringent high-effect gene analysis (log2FC > 2, Cohen's d > 0.8)
        print("\n" + "="*80)
        print("STRINGENT HIGH-EFFECT GENE ANALYSIS")
        print("="*80)
        plot_stringent_high_effect_genes(effect_df, corr_df, config, outdir)

    # Summary statistics
    create_summary_table(effect_df, corr_df, config, outdir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {outdir}")
    print(f"\nGenerated files:")
    for f in sorted(os.listdir(outdir)):
        print(f"  - {f}")

    return {
        'effect_df': effect_df,
        'corr_df': corr_df,
        'outdir': outdir
    }


#%%
# =============================================================================
# EXECUTE
# =============================================================================

if __name__ == '__main__':
    results = main()
