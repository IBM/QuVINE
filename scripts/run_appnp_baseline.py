#!/usr/bin/env python3
"""
Run APPNP baseline on all graphml files in a directory.

This script:
1. Finds all .graphml files in the specified directory
2. For each graph, runs APPNP baseline if not already done
3. Uses the existing run_single_network_analysis infrastructure
4. Results are saved to individual network directories
5. After all graphs are processed, regenerates comprehensive_results.csv

Usage:
    python scripts/run_appnp_baseline.py \
        --input-dir /path/to/hard_negatives_v4 \
        --resume

    python scripts/run_appnp_baseline.py \
        --input-dir /path/to/ppi_disease_v3 \
        --resume
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict
import networkx as nx
import pandas as pd

# Allow importing from src/ regardless of install state
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quvine.comprehensive_embedding_analysis import (
    run_single_network_analysis,
    collect_and_aggregate_results
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_all_graphml_files(directory: Path) -> List[Path]:
    """Find all .graphml files in directory and subdirectories."""
    graphml_files = list(directory.rglob("*.graphml"))
    logger.info(f"Found {len(graphml_files)} graphml files in {directory}")
    return sorted(graphml_files)


def extract_network_metadata(graphml_path: Path) -> Dict:
    """
    Extract network metadata from graphml file path and location.
    
    For hard_negatives_v4: network_id like QW1_modular_strong_rep00
    For ppi_disease_v3: network_id like HumanNet_autism_rep00
    """
    network_id = graphml_path.stem
    parent_dir = graphml_path.parent.name
    
    # Determine network type from path
    if "hard_negatives" in str(graphml_path):
        # Extract case name from network_id (e.g., QW1_modular_strong from QW1_modular_strong_rep00)
        parts = network_id.split('_rep')
        case_name = parts[0] if parts else network_id
        
        # Try to determine network type from case name
        if 'modular' in case_name.lower():
            network_type = 'modular'
        elif 'scale_free' in case_name.lower() or 'barabasi' in case_name.lower():
            network_type = 'scale_free'
        elif 'core_periphery' in case_name.lower():
            network_type = 'core_periphery'
        elif 'watts_strogatz' in case_name.lower() or '_ws_' in case_name.lower():
            network_type = 'watts_strogatz'
        elif 'erdos_renyi' in case_name.lower():
            network_type = 'erdos_renyi'
        elif 'random_geometric' in case_name.lower():
            network_type = 'random_geometric'
        elif 'stochastic_block' in case_name.lower() or 'sbm' in case_name.lower():
            network_type = 'stochastic_block_model'
        elif 'powerlaw' in case_name.lower():
            network_type = 'powerlaw_cluster'
        else:
            network_type = 'unknown'
        
        # Determine negative strategy (default to random for hard negatives)
        negative_strategy = 'hard_2hop'  # Most hard negatives use 2-hop
        
    elif "ppi_disease" in str(graphml_path):
        # Real PPI networks
        network_type = 'real_ppi'
        negative_strategy = 'random'
    else:
        network_type = 'unknown'
        negative_strategy = 'random'
    
    metadata = {
        'network_id': network_id,
        'type': network_type,
        'negative_strategy': negative_strategy,
        'case': network_id,
    }
    
    return metadata


def process_single_graph(
    graphml_path: Path,
    output_base_dir: Path,
    resume: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Process a single graphml file with APPNP baseline.
    
    Parameters
    ----------
    graphml_path : Path
        Path to the .graphml file
    output_base_dir : Path
        Base output directory (e.g., hard_negatives_v4/results)
    resume : bool
        If True, skip if APPNP results already exist
    verbose : bool
        Print progress messages
        
    Returns
    -------
    dict
        Summary of processing results
    """
    network_id = graphml_path.stem
    
    # Determine output directory for this network
    # Results go in results/<network_id>/ subdirectory
    network_output_dir = output_base_dir / network_id
    
    if verbose:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {network_id}")
        logger.info(f"GraphML: {graphml_path}")
        logger.info(f"Output: {network_output_dir}")
        logger.info(f"{'='*80}")
    
    # Check if APPNP already done (resume mode)
    if resume:
        appnp_ranking_file = network_output_dir / f"{network_id}_ranking_results.csv"
        if appnp_ranking_file.exists():
            try:
                df = pd.read_csv(appnp_ranking_file)
                if 'method' in df.columns and 'appnp' in df['method'].values:
                    logger.info(f"  ✓ APPNP already completed for {network_id}, skipping")
                    return {
                        'network_id': network_id,
                        'status': 'skipped',
                        'reason': 'already_done'
                    }
            except Exception as e:
                logger.warning(f"  Could not check existing results: {e}")
    
    try:
        # Load graph
        if verbose:
            logger.info(f"  Loading graph from {graphml_path}")
        G = nx.read_graphml(str(graphml_path))
        G = nx.convert_node_labels_to_integers(G)
        
        if verbose:
            logger.info(f"  Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Extract metadata
        metadata = extract_network_metadata(graphml_path)
        
        # Run analysis with APPNP only
        summary = run_single_network_analysis(
            G=G,
            network_id=network_id,
            network_metadata=metadata,
            output_dir=str(network_output_dir),
            embedding_methods=['appnp'],  # Only run APPNP
            embedding_dim=128,
            num_seeds=15,
            num_targets=25,
            verbose=verbose,
            resume=resume,
            method_hyperparams=None,
        )
        
        return {
            'network_id': network_id,
            'status': 'success',
            'summary': summary
        }
        
    except Exception as e:
        logger.error(f"  ✗ Failed to process {network_id}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'network_id': network_id,
            'status': 'failed',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run APPNP baseline on all graphml files in a directory"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Input directory containing graphml files (e.g., /path/to/hard_negatives_v4)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip graphs that already have APPNP results (default: True)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Process all graphs even if APPNP results exist"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose logging (default: True)"
    )
    parser.add_argument(
        "--regenerate-csv",
        action="store_true",
        default=True,
        help="Regenerate comprehensive_results.csv after processing (default: True)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Determine results directory
    results_dir = input_dir / "results"
    if not results_dir.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
        logger.error("Expected structure: <input_dir>/results/<network_id>/")
        sys.exit(1)
    
    logger.info(f"="*80)
    logger.info(f"APPNP BASELINE RUNNER")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Resume mode: {args.resume}")
    logger.info(f"="*80)
    
    # Find all graphml files
    graphml_files = find_all_graphml_files(results_dir)
    
    if not graphml_files:
        logger.error("No graphml files found!")
        sys.exit(1)
    
    logger.info(f"Found {len(graphml_files)} graphml files to process")
    
    # Process each graph
    results = []
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for i, graphml_path in enumerate(graphml_files, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Progress: {i}/{len(graphml_files)}")
        logger.info(f"{'='*80}")
        
        result = process_single_graph(
            graphml_path=graphml_path,
            output_base_dir=results_dir,
            resume=args.resume,
            verbose=args.verbose
        )
        
        results.append(result)
        
        if result['status'] == 'success':
            success_count += 1
        elif result['status'] == 'skipped':
            skip_count += 1
        else:
            fail_count += 1
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"PROCESSING COMPLETE")
    logger.info(f"Total graphs: {len(graphml_files)}")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Skipped: {skip_count}")
    logger.info(f"  Failed: {fail_count}")
    logger.info(f"{'='*80}")
    
    # Save processing summary
    summary_path = results_dir / "appnp_processing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'total': len(graphml_files),
            'success': success_count,
            'skipped': skip_count,
            'failed': fail_count,
            'results': results
        }, f, indent=2)
    logger.info(f"Processing summary saved to: {summary_path}")
    
    # Regenerate comprehensive_results.csv
    if args.regenerate_csv and success_count > 0:
        logger.info(f"\n{'='*80}")
        logger.info("REGENERATING comprehensive_results.csv")
        logger.info(f"{'='*80}")
        
        try:
            comprehensive_df = collect_and_aggregate_results(
                results_dir=str(results_dir),
                output_file="comprehensive_results.csv",
                verbose=True
            )
            logger.info(f"✓ comprehensive_results.csv regenerated successfully")
            logger.info(f"  Total rows: {len(comprehensive_df)}")
            if 'method' in comprehensive_df.columns:
                logger.info(f"  Methods: {sorted(comprehensive_df['method'].unique())}")
                appnp_count = (comprehensive_df['method'] == 'appnp').sum()
                logger.info(f"  APPNP rows: {appnp_count}")
        except Exception as e:
            logger.error(f"✗ Failed to regenerate comprehensive_results.csv: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*80}")
    logger.info("ALL DONE!")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()

# Made with Bob
