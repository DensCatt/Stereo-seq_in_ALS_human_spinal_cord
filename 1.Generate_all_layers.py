#!/usr/bin/env python3
"""
Automated Stereoseq Pipeline
Converts GEF to GEM, splits into EXON/INTRON layers, converts to h5ad, and unifies all layers

Usage:
    python Generate_all_layers.py --gef <gef_file> --output <output_dir> --sample <sample_name> --chip <chip_name> --binsize <bin_size> [--lasso-geojson <geojson_file>] [--n-lasso <n>] [--aggregate]

Examples:
    python Generate_all_layers.py --gef ./path/to/file.gef --output ./output_dir --sample s18-070 --chip A05328G6 --binsize 20
    python Generate_all_layers.py --gef ./path/to/file.gef --output ./output_dir --sample s18-070 --chip A05328G6 --binsize 20 --lasso-geojson ./path/to/lasso.geojson --n-lasso 5 --aggregate

    # If labeled file already exists and you just want to aggregate:
    python Generate_all_layers.py --gef <gef_file> --output <output_dir> --sample s18-070 --chip A05328G6 --binsize 20 --aggregate
"""

import os
import argparse
import numpy as np
import pandas as pd
import subprocess
import scanpy as sc

# ============================================================================
# CONFIGURATION
# ============================================================================
SAW_BIN = "../Stereoseq_2025/saw/saw-8.1.3/bin/saw"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def fix_data_types(adata, chip_name, sample_name):
    """
    Fix data types and add metadata

    Args:
        adata: AnnData object to process
        chip_name: Name of the chip
        sample_name: Name of the sample

    Returns:
        Processed AnnData object
    """
    print("\n" + "="*60)
    print("Fixing data types and metadata...")
    print("="*60)

    # Fix label column
    adata.obs["label"] = adata.obs["label"].fillna(0).astype("int32")

    # Drop orig.ident if it exists
    if "orig.ident" in adata.obs.columns:
        adata.obs.drop(columns=["orig.ident"], inplace=True)

    # Make var_names unique
    adata.var_names_make_unique()

    # Fix dtypes
    adata.obs["label"] = adata.obs["label"].astype(str)
    adata.obs["n_bins_aggregated"] = 1
    adata.obs["n_bins_aggregated"] = adata.obs["n_bins_aggregated"].astype(np.int32)

    if "n_genes_by_counts" in adata.obs.columns:
        adata.obs["n_genes_by_counts"] = adata.obs["n_genes_by_counts"].astype(np.float32)
    if "x" in adata.obs.columns:
        adata.obs["x"] = adata.obs["x"].astype(np.float32)
    if "y" in adata.obs.columns:
        adata.obs["y"] = adata.obs["y"].astype(np.float32)

    # Add metadata
    adata.obs["chip"] = chip_name
    adata.obs["sample"] = sample_name
    adata.obs["motorneuron"] = adata.obs["label"] != "0"

    print("✅ Data types fixed")
    return adata


def header_count(file):
    """Count the number of header lines starting with '#' in a GEM file"""
    header = 0
    with open(file, "r") as f:
        for _ in range(10):
            if f.readline().startswith("#"):
                header += 1
    return header


def add_header(header_file, gem_df, output_file):
    """Add header from header_file to a gem dataframe and save to output_file"""
    with open(header_file, 'r') as f:
        h = f.read()

    with open(output_file, 'w') as f:
        f.write(h)
        if not h.endswith('\n'):
            f.write('\n')

    gem_df.to_csv(output_file, sep='\t', index=True, mode='a')


def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Error: {description} failed")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")
    else:
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"STDOUT: {result.stdout}")

    return result


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(gef_file, output_dir, sample_name, chip_name, bin_size, lasso_geojson=None, n_lasso=None, aggregate=False):
    """
    Run the complete Stereoseq pipeline

    Args:
        gef_file: Path to input GEF file
        output_dir: Path to output directory
        sample_name: Name of the sample (e.g., 's18-070')
        chip_name: Name of the chip (e.g., 'A05328G6')
        bin_size: Bin size for processing (e.g., 20)
        lasso_geojson: (Optional) Path to lasso geojson file. Required if labeled.h5ad doesn't exist.
        n_lasso: (Optional) Number of lasso regions/motor neurons (e.g., 5). Required if labeled.h5ad doesn't exist.
        aggregate: (Optional) Whether to aggregate motor neurons after processing
    """
    # Convert to absolute paths
    gef_file = os.path.abspath(gef_file)
    output_dir = os.path.abspath(output_dir)
    if lasso_geojson:
        lasso_geojson = os.path.abspath(lasso_geojson)

    # Convert bin_size to int
    bin_size = int(bin_size)
    if n_lasso is not None:
        n_lasso = int(n_lasso)

    # Define paths based on input GEF file and output directory
    gef_dir = os.path.dirname(gef_file)
    GEF_FILE = gef_file
    GEM_FILE = os.path.join(gef_dir, f"{chip_name}.tissue.gem")
    OUTPUT_DIR = os.path.join(output_dir, f"ALL_LAYERS_bin{bin_size}")
    LASSO_OUTPUT_DIR = os.path.join(output_dir, f"lasso_output_bin{bin_size}") if lasso_geojson else None
    BASE_H5AD = os.path.join(output_dir, f"{chip_name}.bin{bin_size}.h5ad")
    LABELED_H5AD = os.path.join(OUTPUT_DIR, f"{chip_name}.bin{bin_size}.labeled.h5ad")
    LASSO_GEOJSON = lasso_geojson
    UNIFIED_LABELED_FILE = f"{OUTPUT_DIR}/{sample_name}_bin{bin_size}.ALL_layers_labeled.h5ad"
    FINAL_FILE = f"{OUTPUT_DIR}/{sample_name}_bin{bin_size}.FINAL.h5ad"

    print(f"\n{'='*60}")
    print(f"Stereoseq Pipeline")
    print(f"{'='*60}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Sample: {sample_name}")
    print(f"Chip: {chip_name}")
    print(f"Bin size: {bin_size}")
    print(f"GEF file: {GEF_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    if n_lasso is not None:
        print(f"N lasso regions: {n_lasso}")
    if lasso_geojson is not None:
        print(f"Lasso geojson: {lasso_geojson}")
    print(f"Aggregate: {aggregate}")
    print(f"{'='*60}\n")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if unified labeled file already exists
    skip_generation = False
    if os.path.exists(UNIFIED_LABELED_FILE):
        print(f"\n{'='*60}")
        print(f"✅ Unified labeled file already exists: {UNIFIED_LABELED_FILE}")
        print(f"{'='*60}")
        if aggregate:
            print("Skipping generation steps and proceeding directly to aggregation...")
            skip_generation = True
        else:
            print("File already exists. Skipping pipeline.")
            return

    if not skip_generation:
        # ========================================================================
        # STEP 1: Convert GEF to GEM
        # ========================================================================
        gef2gem_cmd = f"{SAW_BIN} convert gef2gem --gef={GEF_FILE} --bin-size={bin_size} --gem={GEM_FILE}"
        run_command(gef2gem_cmd, "GEF to GEM conversion")

        # ========================================================================
        # STEP 2: Load GEM file and process
        # ========================================================================
        print("\n" + "="*60)
        print("Loading GEM file...")
        print("="*60)

        filename = GEM_FILE
        gem_file = pd.read_csv(filename, sep="\t", index_col=0,
                              skiprows=header_count(filename))
        print(f"✅ Loaded GEM file with shape: {gem_file.shape}")
        print(f"Columns: {gem_file.columns.tolist()}")

        # ========================================================================
        # STEP 3: Extract and save header
        # ========================================================================
        print("\n" + "="*60)
        print("Extracting header...")
        print("="*60)

        header_file = f"{sample_name}/header.gem"

        with open(filename, "r") as f, open(header_file, "w") as h:
            for line in f:
                if line.startswith("#"):
                    h.write(line)
                else:
                    break

        print("Header preview:")
        with open(header_file, "r") as f:
            for i in range(min(10, sum(1 for _ in open(header_file)))):
                print(f.readline(), end='')

        # ========================================================================
        # STEP 4: Create EXON and INTRON GEM files
        # ========================================================================
        print("\n" + "="*60)
        print("Creating EXON and INTRON GEM files...")
        print("="*60)

        # Create ExonCounts file
        Ex_gem = gem_file.drop("MIDCount", axis=1)
        Ex_gem.rename(columns={"ExonCount": "MIDCount"}, inplace=True)

        # Create IntronCounts file
        gem_file["IntronCount"] = gem_file["MIDCount"] - gem_file["ExonCount"]
        In_gem = gem_file.drop(["MIDCount", "ExonCount"], axis=1)
        In_gem.rename(columns={"IntronCount": "MIDCount"}, inplace=True)

        # Save with headers
        exon_gem_file = f"{OUTPUT_DIR}/{chip_name}_bin{bin_size}.EXON.gem"
        intron_gem_file = f"{OUTPUT_DIR}/{chip_name}_bin{bin_size}.INTRON.gem"

        add_header(header_file, Ex_gem, exon_gem_file)
        add_header(header_file, In_gem, intron_gem_file)

        print(f"✅ Created EXON GEM file: {exon_gem_file}")
        print(f"✅ Created INTRON GEM file: {intron_gem_file}")

        # ========================================================================
        # STEP 5: Convert GEM files to H5AD
        # ========================================================================
        print("\n" + "="*60)
        print("Converting GEM files to H5AD...")
        print("="*60)

        # Convert EXON GEM to H5AD
        exon_h5ad = f"{OUTPUT_DIR}/{sample_name}_bin{bin_size}.EXON.h5ad"
        gem2h5ad_exon_cmd = f"{SAW_BIN} convert gem2h5ad --gem={exon_gem_file} --bin-size={bin_size} --h5ad={exon_h5ad}"
        run_command(gem2h5ad_exon_cmd, "EXON GEM to H5AD conversion")

        # Convert INTRON GEM to H5AD
        intron_h5ad = f"{OUTPUT_DIR}/{sample_name}_bin{bin_size}.INTRON.h5ad"
        gem2h5ad_intron_cmd = f"{SAW_BIN} convert gem2h5ad --gem={intron_gem_file} --bin-size={bin_size} --h5ad={intron_h5ad}"
        run_command(gem2h5ad_intron_cmd, "INTRON GEM to H5AD conversion")

        # ========================================================================
        # STEP 5.5: Generate labeled.h5ad if it doesn't exist
        # ========================================================================
        print("\n" + "="*60)
        print("Checking for labeled.h5ad file...")
        print("="*60)

        if not os.path.exists(LABELED_H5AD):
            print(f"⚠️  Labeled H5AD not found: {LABELED_H5AD}")

            if n_lasso is None or lasso_geojson is None:
                print("❌ ERROR: labeled.h5ad does not exist and lasso parameters not provided!")
                print("Please provide --n-lasso and --lasso-geojson arguments to generate labeled.h5ad")
                raise RuntimeError("Missing labeled.h5ad and lasso parameters")

            print("Generating labeled.h5ad...")

            # Check if lasso_output directory exists
            if not os.path.exists(LASSO_OUTPUT_DIR):
                print(f"⚠️  Lasso output directory not found: {LASSO_OUTPUT_DIR}")
                print("Generating lasso output...")

                # Check if base h5ad exists, if not create it
                if not os.path.exists(BASE_H5AD):
                    print(f"⚠️  Base H5AD not found: {BASE_H5AD}")
                    print("Converting tissue GEM to H5AD...")

                    gem2h5ad_base_cmd = f"{SAW_BIN} convert gem2h5ad --gem={GEM_FILE} --bin-size={bin_size} --h5ad={BASE_H5AD}"
                    run_command(gem2h5ad_base_cmd, "Base GEM to H5AD conversion")
                else:
                    print(f"✅ Base H5AD already exists: {BASE_H5AD}")

                # Run SAW lasso reanalyze
                lasso_cmd = f"{SAW_BIN} reanalyze lasso --gef={GEF_FILE} --lasso-geojson={LASSO_GEOJSON} --bin-size={bin_size} --output={LASSO_OUTPUT_DIR}"
                run_command(lasso_cmd, "SAW lasso reanalyze")
            else:
                print(f"✅ Lasso output directory already exists: {LASSO_OUTPUT_DIR}")

            # Run lasso_to_label.py
            lasso_to_label_cmd = f"python denis_scripts/lasso_to_label.py --lasso_dir {LASSO_OUTPUT_DIR} --n_lasso {n_lasso} --chip {chip_name} --binsize {bin_size} --adata {BASE_H5AD} --output_adata {LABELED_H5AD}"
            run_command(lasso_to_label_cmd, "Generate labeled H5AD from lasso")
        else:
            print(f"✅ Labeled H5AD already exists: {LABELED_H5AD}")

        # ========================================================================
        # STEP 6: Unify all layers into a single H5AD file
        # ========================================================================
        print("\n" + "="*60)
        print("Unifying all layers into single H5AD...")
        print("="*60)

        # Paths to the three h5ad files
        count_path = LABELED_H5AD
        exon_path = exon_h5ad
        intron_path = intron_h5ad

        # Load the three AnnData objects
        print(f"Loading: {count_path}")
        count_ad = sc.read_h5ad(count_path)
        print(f"Loading: {exon_path}")
        exon_ad = sc.read_h5ad(exon_path)
        print(f"Loading: {intron_path}")
        intron_ad = sc.read_h5ad(intron_path)

        # Verify consistency (cells and genes must have the same order)
        assert all(count_ad.obs_names == exon_ad.obs_names) and \
               all(count_ad.obs_names == intron_ad.obs_names), \
               "❌ obs_names do not match"
        assert all(count_ad.var_names == exon_ad.var_names) and \
               all(count_ad.var_names == intron_ad.var_names), \
               "❌ var_names do not match"

        print("✅ All files have matching obs_names and var_names")

        # Create a new unified AnnData object
        adata = count_ad.copy()

        # Add counts to different layers
        adata.layers["counts"] = count_ad.X
        adata.layers["exon"] = exon_ad.X
        adata.layers["intron"] = intron_ad.X

        # Set .X to counts (optional)
        adata.X = adata.layers["counts"]

        # ========================================================================
        # STEP 7: Fix data types and save
        # ========================================================================
        adata = fix_data_types(adata, chip_name, sample_name)
        output_file = f"{OUTPUT_DIR}/{sample_name}_bin{bin_size}.ALL_layers_labeled.h5ad"
        adata.write_h5ad(output_file)

        print(f"\n{'='*60}")
        print("✅ Pipeline completed successfully!")
        print(f"{'='*60}")
        print(f"Unified file created with layers: counts, exon, intron")
        print(f"Output: {output_file}")
        print(f"{'='*60}\n")

    # ========================================================================
    # STEP 8: Aggregate motor neurons if requested
    # ========================================================================
    if aggregate:
        print(f"\n{'='*60}")
        print("Calling aggregation script...")
        print(f"{'='*60}")

        # Path to aggregation script
        aggregate_script = os.path.join(os.path.dirname(__file__), "1.aggregate_motorneurons.py")

        # Call the aggregation script
        aggregate_cmd = f"python {aggregate_script} --input {UNIFIED_LABELED_FILE} --output {FINAL_FILE}"
        run_command(aggregate_cmd, "Aggregate motor neurons")

        print(f"\n{'='*60}")
        print("✅ Aggregation completed!")
        print(f"{'='*60}")
        print(f"Labeled file: {UNIFIED_LABELED_FILE}")
        print(f"Final file: {FINAL_FILE}")
        print(f"{'='*60}\n")


# ============================================================================
# RUN PIPELINE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated Stereoseq Pipeline - Converts GEF to GEM, splits into EXON/INTRON layers, and unifies all layers",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument("--gef", required=True, help="Path to input GEF file")
    parser.add_argument("--output", required=True, help="Path to output directory")
    parser.add_argument("--sample", required=True, help="Sample name (e.g., 's18-070')")
    parser.add_argument("--chip", required=True, help="Chip name (e.g., 'A05328G6')")
    parser.add_argument("--binsize", required=True, type=int, help="Bin size (e.g., 20)")

    # Optional arguments
    parser.add_argument("--lasso-geojson", help="Path to lasso geojson file (required if labeled.h5ad doesn't exist)")
    parser.add_argument("--n-lasso", type=int, help="Number of lasso regions/motor neurons (required if labeled.h5ad doesn't exist)")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate motor neurons after processing")

    args = parser.parse_args()

    # Validate lasso arguments
    if args.lasso_geojson and not args.n_lasso:
        parser.error("--lasso-geojson requires --n-lasso")
    if args.n_lasso and not args.lasso_geojson:
        parser.error("--n-lasso requires --lasso-geojson")

    main(
        gef_file=args.gef,
        output_dir=args.output,
        sample_name=args.sample,
        chip_name=args.chip,
        bin_size=args.binsize,
        lasso_geojson=args.lasso_geojson,
        n_lasso=args.n_lasso,
        aggregate=args.aggregate
    )
