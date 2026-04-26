#!/usr/bin/env python3
"""
Aggregate Simulated Data Results
=================================

Collects results from both hard negatives cases and extended generators,
creating comprehensive CSV files with complexity metrics merged.

Handles:
- 16 hard negatives cases (QW1-9, NC1-4, RN1-3) × 30 reps × 3 sizes
- 5 extended generator types × 30 reps × 3 sizes

Output structure:
- Per-case/type aggregated CSVs
- Comprehensive CSV with all results
- Complexity metrics merged from individual network CSVs

Usage:
    python scripts/aggregate_simulated_data.py \
        --results-dir /path/to/results \
        --output-dir /path/to/output
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np


def find_network_dirs(results_dir: Path) -> Dict[str, List[Path]]:
    """
    Find all network directories and group by case/type.
    
    Returns:
        Dict mapping case/type name -> list of replicate directories
    """
    network_dirs: Dict[str, List[Path]] = {}
    
    # Find all directories matching pattern: {name}_n{size}_rep{N}
    for dir_path in sorted(results_dir.glob("*_n*_rep*")):
        if not dir_path.is_dir():
            continue
        
        # Extract case/type name (everything before _n{size})
        name_parts = dir_path.name.split('_n')
        if len(name_parts) < 2:
            continue
        
        case_type = name_parts[0]
        
        if case_type not in network_dirs:
            network_dirs[case_type] = []
        network_dirs[case_type].append(dir_path)
    
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


def aggregate_by_case_type(
    network_dirs: Dict[str, List[Path]],
    output_dir: Path,
    tasks: List[str] = ['ranking', 'classification', 'link_prediction'],
) -> Dict[str, pd.DataFrame]:
    """
    Aggregate results by case/type.
    
    Returns:
        Dict mapping case/type -> aggregated DataFrame
    """
    aggregated: Dict[str, pd.DataFrame] = {}
    
    for case_type, dirs in network_dirs.items():
        print(f"\nAggregating {case_type} ({len(dirs)} replicates)...")
        
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
            print(f"  WARNING: No results found for {case_type}")
            continue
        
        # Concatenate all tasks
        combined = pd.concat(all_dfs, ignore_index=True)
        
        # Load complexity metrics from first replicate
        complexity_df = load_complexity_metrics(dirs[0])
        if complexity_df is not None:
            # Merge complexity metrics
            complexity_cols = [c for c in complexity_df.columns if c != 'network_id']
            if complexity_cols:
                complexity_row = complexity_df[complexity_cols].iloc[0:1]
                combined = combined.assign(key=1).merge(
                    complexity_row.assign(key=1),
                    on='key'
                ).drop('key', axis=1)
                print(f"  Merged {len(complexity_cols)} complexity metrics")
        
        # Save aggregated results
        output_path = output_dir / f"{case_type}_aggregated.csv"
        combined.to_csv(output_path, index=False)
        print(f"  Saved: {output_path} ({len(combined)} rows)")
        
        aggregated[case_type] = combined
    
    return aggregated


def create_comprehensive_csv(
    aggregated: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Combine all case/type aggregated results into one comprehensive CSV.
    """
    print("\nCreating comprehensive CSV...")
    
    if not aggregated:
        print("ERROR: No aggregated results to combine")
        return pd.DataFrame()
    
    # Concatenate all
    all_dfs = list(aggregated.values())
    comprehensive = pd.concat(all_dfs, ignore_index=True)
    
    # Add case/type column if not present
    if 'case_type' not in comprehensive.columns:
        comprehensive['case_type'] = comprehensive['network_id'].str.extract(
            r'^([^_]+(?:_[^_]+)*?)_n\d+'
        )[0]
    
    # Add node size column
    if 'n_nodes' not in comprehensive.columns:
        comprehensive['n_nodes'] = comprehensive['network_id'].str.extract(
            r'_n(\d+)_'
        )[0].astype(int)
    
    # Save
    output_path = output_dir / "simulated_data_comprehensive.csv"
    comprehensive.to_csv(output_path, index=False)
    
    print(f"Saved comprehensive results: {output_path}")
    print(f"  Total rows: {len(comprehensive)}")
    print(f"  Cases/Types: {comprehensive['case_type'].nunique()}")
    print(f"  Node sizes: {sorted(comprehensive['n_nodes'].unique())}")
    print(f"  Tasks: {comprehensive['task'].nunique()}")
    print(f"  Methods: {comprehensive['method'].nunique()}")
    
    return comprehensive


def print_summary_statistics(comprehensive: pd.DataFrame) -> None:
    """Print summary statistics for the comprehensive results."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    # Group by case/type
    for case_type in sorted(comprehensive['case_type'].unique()):
        ct_df = comprehensive[comprehensive['case_type'] == case_type]
        
        print(f"\n{case_type}:")
        
        # By node size
        for n_nodes in sorted(ct_df['n_nodes'].unique()):
            size_df = ct_df[ct_df['n_nodes'] == n_nodes]
            n_reps = size_df['network_id'].nunique()
            n_methods = size_df['method'].nunique()
            n_tasks = size_df['task'].nunique()
            
            print(f"  n={n_nodes}:")
            print(f"    Replicates: {n_reps}")
            print(f"    Methods: {n_methods}")
            print(f"    Tasks: {n_tasks}")
            print(f"    Total rows: {len(size_df)}")
            
            # Average performance by task
            for task in size_df['task'].unique():
                task_df = size_df[size_df['task'] == task]
                if 'score' in task_df.columns:
                    avg_score = task_df['score'].mean()
                    print(f"      {task} avg score: {avg_score:.4f}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate simulated data results"
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
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SIMULATED DATA RESULTS AGGREGATION")
    print("=" * 70)
    print(f"Results dir: {args.results_dir}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 70)
    
    # Validate directories
    if not args.results_dir.exists():
        print(f"ERROR: Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all network directories
    network_dirs = find_network_dirs(args.results_dir)
    
    if not network_dirs:
        print("ERROR: No network directories found")
        sys.exit(1)
    
    print(f"\nFound {len(network_dirs)} cases/types:")
    for case_type, dirs in sorted(network_dirs.items()):
        print(f"  {case_type}: {len(dirs)} replicates")
    
    # Aggregate by case/type
    aggregated = aggregate_by_case_type(network_dirs, args.output_dir)
    
    # Create comprehensive CSV
    comprehensive = create_comprehensive_csv(aggregated, args.output_dir)
    
    if len(comprehensive) > 0:
        # Print summary statistics
        print_summary_statistics(comprehensive)
        
        print("\n" + "=" * 70)
        print("AGGREGATION COMPLETE")
        print("=" * 70)
        print(f"Output files:")
        print(f"  - Per case/type: {args.output_dir}/*_aggregated.csv")
        print(f"  - Comprehensive: {args.output_dir}/simulated_data_comprehensive.csv")
        print("=" * 70)
    else:
        print("\nERROR: No results aggregated")
        sys.exit(1)


if __name__ == '__main__':
    main()

