#!/usr/bin/env python3
"""
Aggregate Extended Generator Results by Network Type

Collects all individual network CSVs and creates per-network-type compiled files
with all complexity measures, all embedding methods, network metadata, etc.

Usage:
    python scripts/aggregate_extended_generators.py \
        --results-dir /path/to/results/extended_generators/results \
        --output-dir /path/to/results/extended_generators/aggregated \
        [--verbose]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def collect_network_results(results_dir: Path, network_type: str, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Collect all CSV files for a specific network type.
    
    Returns
    -------
    dict
        Dictionary with keys: 'complexity', 'ranking', 'classification', 
        'link_prediction', 'link_prediction_tidy', 'nc_stratified', 'timing'
    """
    results = {
        'complexity': [],
        'ranking': [],
        'classification': [],
        'link_prediction': [],
        'link_prediction_tidy': [],
        'nc_stratified': [],
        'timing': [],
    }
    
    # Find all network directories for this type
    network_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith(network_type)]
    
    if verbose:
        logger.info(f"Found {len(network_dirs)} networks for type '{network_type}'")
    
    for network_dir in sorted(network_dirs):
        network_id = network_dir.name
        
        # Collect each CSV type
        csv_files = {
            'complexity': network_dir / f"{network_id}_complexity.csv",
            'ranking': network_dir / f"{network_id}_ranking_results.csv",
            'classification': network_dir / f"{network_id}_classification_results.csv",
            'link_prediction': network_dir / f"{network_id}_link_prediction_results.csv",
            'link_prediction_tidy': network_dir / f"{network_id}_link_prediction.csv",
            'nc_stratified': network_dir / f"{network_id}_nc_stratified.csv",
            'timing': network_dir / f"{network_id}_timing_results.csv",
        }
        
        for result_type, csv_path in csv_files.items():
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    results[result_type].append(df)
                except Exception as e:
                    logger.warning(f"Failed to read {csv_path}: {e}")
    
    # Concatenate all dataframes
    aggregated = {}
    for result_type, dfs in results.items():
        if dfs:
            aggregated[result_type] = pd.concat(dfs, ignore_index=True)
            if verbose:
                logger.info(f"  {result_type}: {len(aggregated[result_type])} rows")
        else:
            aggregated[result_type] = pd.DataFrame()
            if verbose:
                logger.warning(f"  {result_type}: No data found")
    
    return aggregated


def merge_with_complexity(
    task_df: pd.DataFrame,
    complexity_df: pd.DataFrame,
    on: str = 'network_id'
) -> pd.DataFrame:
    """Merge task results with complexity metrics."""
    if task_df.empty or complexity_df.empty:
        return task_df
    
    # Merge on network_id
    merged = task_df.merge(complexity_df, on=on, how='left', suffixes=('', '_complexity'))
    
    # Remove duplicate columns (keep the one from task_df)
    duplicate_cols = [col for col in merged.columns if col.endswith('_complexity')]
    for col in duplicate_cols:
        base_col = col.replace('_complexity', '')
        if base_col in merged.columns:
            # Keep task_df version, drop complexity version
            merged = merged.drop(columns=[col])
    
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate extended generator results by network type"
    )
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing individual network result folders")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for aggregated CSVs")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return 1
    
    # Network types to aggregate
    network_types = [
        'random_regular',
        'heterophilic_sbm',
        'degree_corrected_sbm',
        'grid_torus',
        'configuration_model',
    ]
    
    logger.info("="*80)
    logger.info("AGGREGATING EXTENDED GENERATOR RESULTS")
    logger.info("="*80)
    logger.info(f"Results dir: {results_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Network types: {len(network_types)}")
    logger.info("="*80)
    
    for network_type in network_types:
        logger.info(f"\nProcessing network type: {network_type}")
        logger.info("-"*80)
        
        # Collect all results for this network type
        results = collect_network_results(results_dir, network_type, verbose=args.verbose)
        
        if results['complexity'].empty:
            logger.warning(f"No results found for {network_type}, skipping")
            continue
        
        # Save complexity metrics (standalone)
        complexity_path = output_dir / f"{network_type}_complexity.csv"
        results['complexity'].to_csv(complexity_path, index=False)
        logger.info(f"✓ Saved complexity: {complexity_path}")
        
        # Merge each task with complexity and save
        for task in ['ranking', 'classification', 'link_prediction', 'link_prediction_tidy', 'nc_stratified', 'timing']:
            if not results[task].empty:
                # Merge with complexity metrics
                merged = merge_with_complexity(results[task], results['complexity'])
                
                # Save merged results
                output_path = output_dir / f"{network_type}_{task}.csv"
                merged.to_csv(output_path, index=False)
                logger.info(f"✓ Saved {task}: {output_path} ({len(merged)} rows)")
        
        # Create a comprehensive file with all tasks combined
        comprehensive_rows = []
        
        # For each network_id and method, create one row with all metrics
        if not results['ranking'].empty:
            for _, row in results['ranking'].iterrows():
                network_id = row['network_id']
                method = row['method']
                
                # Start with ranking metrics
                comp_row = row.to_dict()
                
                # Add classification metrics
                if not results['classification'].empty:
                    class_match = results['classification'][
                        (results['classification']['network_id'] == network_id) &
                        (results['classification']['method'] == method)
                    ]
                    if not class_match.empty:
                        for col in class_match.columns:
                            if col not in ['network_id', 'method']:
                                comp_row[f'class_{col}'] = class_match.iloc[0][col]
                
                # Add link prediction metrics
                if not results['link_prediction'].empty:
                    lp_match = results['link_prediction'][
                        (results['link_prediction']['network_id'] == network_id) &
                        (results['link_prediction']['method'] == method)
                    ]
                    if not lp_match.empty:
                        for col in lp_match.columns:
                            if col not in ['network_id', 'method']:
                                comp_row[f'lp_{col}'] = lp_match.iloc[0][col]
                
                # Add timing
                if not results['timing'].empty:
                    timing_match = results['timing'][
                        (results['timing']['network_id'] == network_id) &
                        (results['timing']['method'] == method)
                    ]
                    if not timing_match.empty:
                        comp_row['embedding_time_s'] = timing_match.iloc[0]['embedding_time_s']
                
                comprehensive_rows.append(comp_row)
        
        if comprehensive_rows:
            comprehensive_df = pd.DataFrame(comprehensive_rows)
            # Merge with complexity
            comprehensive_df = merge_with_complexity(comprehensive_df, results['complexity'])
            
            comprehensive_path = output_dir / f"{network_type}_comprehensive.csv"
            comprehensive_df.to_csv(comprehensive_path, index=False)
            logger.info(f"✓ Saved comprehensive: {comprehensive_path} ({len(comprehensive_df)} rows)")
    
    logger.info("\n" + "="*80)
    logger.info("AGGREGATION COMPLETE")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

