#!/usr/bin/env python3
"""
Aggregate PPI Comprehensive Results
====================================

Collects results from all PPI network × disease × replicate combinations and
creates comprehensive CSV files with complexity metrics merged.

Similar to aggregate_extended_generators.py but handles PPI-specific structure:
- 5 networks: BioPlex3, HumanNet, ProteomeHD, STRING, PCNet
- 3 diseases: asthma, autism, schizophrenia
- 30 replicates per (network, disease) combination

Output structure:
- Per (network, disease) aggregated CSVs
- Comprehensive CSV with all networks and diseases
- Complexity metrics merged from individual network CSVs

Usage:
    python scripts/aggregate_ppi_comprehensive.py \
        --results-dir /path/to/results \
        --output-dir /path/to/output \
        --networks "BioPlex3 HumanNet ProteomeHD STRING PCNet" \
        --diseases "asthma autism schizophrenia"
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import numpy as np


def find_network_dirs(
    results_dir: Path,
    networks: List[str],
    diseases: List[str],
) -> Dict[str, List[Path]]:
    """
    Find all network directories matching pattern: {network}_{disease}_rep{N}
    
    Returns:
        Dict mapping (network, disease) -> list of replicate directories
    """
    network_dirs: Dict[str, List[Path]] = {}
    
    for network in networks:
        for disease in diseases:
            key = f"{network}_{disease}"
            pattern = f"{network}_{disease}_rep*"
            dirs = sorted(results_dir.glob(pattern))
            if dirs:
                network_dirs[key] = dirs
                print(f"Found {len(dirs)} replicates for {key}")
            else:
                print(f"WARNING: No directories found for {key}")
    
    return network_dirs


def load_task_results(
    network_dir: Path,
    task: str,
) -> Optional[pd.DataFrame]:
    """Load results for a specific task from a network directory."""
    csv_path = network_dir / f"{network_dir.name}_{task}_results.csv"
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path)
        # Add network_id if not present
        if 'network_id' not in df.columns:
            df['network_id'] = network_dir.name
        return df
    except Exception as e:
        print(f"  ERROR loading {csv_path}: {e}")
        return None


def load_complexity_metrics(network_dir: Path) -> Optional[pd.DataFrame]:
    """Load complexity metrics from network directory."""
    csv_path = network_dir / f"{network_dir.name}_complexity.csv"
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path)
        if 'network_id' not in df.columns:
            df['network_id'] = network_dir.name
        return df
    except Exception as e:
        print(f"  ERROR loading {csv_path}: {e}")
        return None


def aggregate_by_network_disease(
    network_dirs: Dict[str, List[Path]],
    output_dir: Path,
    tasks: List[str] = ['ranking', 'classification', 'link_prediction'],
) -> Dict[str, pd.DataFrame]:
    """
    Aggregate results by (network, disease) combination.
    
    Returns:
        Dict mapping (network, disease) -> aggregated DataFrame
    """
    aggregated: Dict[str, pd.DataFrame] = {}
    
    for key, dirs in network_dirs.items():
        print(f"\nAggregating {key} ({len(dirs)} replicates)...")
        
        # Collect all task results
        all_dfs: List[pd.DataFrame] = []
        
        for task in tasks:
            task_dfs: List[pd.DataFrame] = []
            for net_dir in dirs:
                df = load_task_results(net_dir, task)
                if df is not None:
                    df['task'] = task
                    task_dfs.append(df)
            
            if task_dfs:
                print(f"  {task}: {len(task_dfs)} replicates")
                all_dfs.extend(task_dfs)
        
        if not all_dfs:
            print(f"  WARNING: No results found for {key}")
            continue
        
        # Concatenate all tasks
        combined = pd.concat(all_dfs, ignore_index=True)
        
        # Load complexity metrics from first replicate
        complexity_df = load_complexity_metrics(dirs[0])
        if complexity_df is not None:
            # Merge complexity metrics (should be same for all reps of same network)
            # Drop network_id from complexity to avoid conflicts
            complexity_cols = [c for c in complexity_df.columns if c != 'network_id']
            if complexity_cols:
                # Take first row (all rows should be identical for same network)
                complexity_row = complexity_df[complexity_cols].iloc[0:1]
                # Add to combined via cross join
                combined = combined.assign(key=1).merge(
                    complexity_row.assign(key=1),
                    on='key'
                ).drop('key', axis=1)
                print(f"  Merged {len(complexity_cols)} complexity metrics")
        
        # Save aggregated results
        output_path = output_dir / f"{key}_aggregated.csv"
        combined.to_csv(output_path, index=False)
        print(f"  Saved: {output_path} ({len(combined)} rows)")
        
        aggregated[key] = combined
    
    return aggregated


def create_comprehensive_csv(
    aggregated: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Combine all (network, disease) aggregated results into one comprehensive CSV.
    """
    print("\nCreating comprehensive CSV...")
    
    if not aggregated:
        print("ERROR: No aggregated results to combine")
        return pd.DataFrame()
    
    # Concatenate all
    all_dfs = list(aggregated.values())
    comprehensive = pd.concat(all_dfs, ignore_index=True)
    
    # Add network and disease columns if not present
    if 'network' not in comprehensive.columns:
        comprehensive['network'] = comprehensive['network_id'].str.extract(
            r'^([^_]+)_'
        )[0]
    
    if 'disease' not in comprehensive.columns:
        comprehensive['disease'] = comprehensive['network_id'].str.extract(
            r'_([^_]+)_rep'
        )[0]
    
    # Save
    output_path = output_dir / "ppi_comprehensive_results.csv"
    comprehensive.to_csv(output_path, index=False)
    
    print(f"Saved comprehensive results: {output_path}")
    print(f"  Total rows: {len(comprehensive)}")
    print(f"  Networks: {comprehensive['network'].nunique()}")
    print(f"  Diseases: {comprehensive['disease'].nunique()}")
    print(f"  Tasks: {comprehensive['task'].nunique()}")
    print(f"  Methods: {comprehensive['method'].nunique()}")
    
    return comprehensive


def print_summary_statistics(
    comprehensive: pd.DataFrame,
    networks: List[str],
    diseases: List[str],
) -> None:
    """Print summary statistics for the comprehensive results."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    for network in networks:
        net_df = comprehensive[comprehensive['network'] == network]
        if len(net_df) == 0:
            continue
        
        print(f"\n{network}:")
        for disease in diseases:
            dis_df = net_df[net_df['disease'] == disease]
            if len(dis_df) == 0:
                continue
            
            n_reps = dis_df['network_id'].nunique()
            n_methods = dis_df['method'].nunique()
            n_tasks = dis_df['task'].nunique()
            
            print(f"  {disease}:")
            print(f"    Replicates: {n_reps}")
            print(f"    Methods: {n_methods}")
            print(f"    Tasks: {n_tasks}")
            print(f"    Total rows: {len(dis_df)}")
            
            # Average performance by task
            for task in dis_df['task'].unique():
                task_df = dis_df[dis_df['task'] == task]
                if 'score' in task_df.columns:
                    avg_score = task_df['score'].mean()
                    print(f"      {task} avg score: {avg_score:.4f}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate PPI comprehensive results"
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        required=True,
        help='Directory containing network result subdirectories'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Directory for aggregated output files'
    )
    parser.add_argument(
        '--networks',
        type=str,
        default='BioPlex3 HumanNet ProteomeHD STRING PCNet',
        help='Space-separated list of network names'
    )
    parser.add_argument(
        '--diseases',
        type=str,
        default='asthma autism schizophrenia',
        help='Space-separated list of disease names'
    )
    
    args = parser.parse_args()
    
    # Parse networks and diseases
    networks = args.networks.split()
    diseases = args.diseases.split()
    
    print("=" * 70)
    print("PPI COMPREHENSIVE RESULTS AGGREGATION")
    print("=" * 70)
    print(f"Results dir: {args.results_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Networks: {networks}")
    print(f"Diseases: {diseases}")
    print("=" * 70)
    
    # Validate directories
    if not args.results_dir.exists():
        print(f"ERROR: Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all network directories
    network_dirs = find_network_dirs(args.results_dir, networks, diseases)
    
    if not network_dirs:
        print("ERROR: No network directories found")
        sys.exit(1)
    
    # Aggregate by (network, disease)
    aggregated = aggregate_by_network_disease(
        network_dirs,
        args.output_dir,
    )
    
    # Create comprehensive CSV
    comprehensive = create_comprehensive_csv(aggregated, args.output_dir)
    
    if len(comprehensive) > 0:
        # Print summary statistics
        print_summary_statistics(comprehensive, networks, diseases)
        
        print("\n" + "=" * 70)
        print("AGGREGATION COMPLETE")
        print("=" * 70)
        print(f"Output files:")
        print(f"  - Per (network, disease): {args.output_dir}/*_aggregated.csv")
        print(f"  - Comprehensive: {args.output_dir}/ppi_comprehensive_results.csv")
        print("=" * 70)
    else:
        print("\nERROR: No results aggregated")
        sys.exit(1)


if __name__ == '__main__':
    main()

# Made with Bob
