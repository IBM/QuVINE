#!/usr/bin/env python3
"""
Check APPNP baseline progress in a results directory.

This script scans the results directory and counts how many graphs
have APPNP results completed.

Usage:
    python scripts/check_appnp_progress.py \
        --input-dir /path/to/hard_negatives_v4

    python scripts/check_appnp_progress.py \
        --input-dir /path/to/ppi_disease_v3
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

def check_progress(results_dir: Path) -> dict:
    """
    Check how many graphs have APPNP results.
    
    Returns:
        dict with progress statistics
    """
    # Find all network subdirectories
    network_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    total_graphs = len(network_dirs)
    appnp_completed = 0
    appnp_pending = 0
    
    completed_networks = []
    pending_networks = []
    
    for network_dir in sorted(network_dirs):
        network_id = network_dir.name
        
        # Check if APPNP results exist in ranking file
        ranking_file = network_dir / f"{network_id}_ranking_results.csv"
        
        has_appnp = False
        if ranking_file.exists():
            try:
                df = pd.read_csv(ranking_file)
                if 'method' in df.columns and 'appnp' in df['method'].values:
                    has_appnp = True
                    appnp_completed += 1
                    completed_networks.append(network_id)
            except Exception:
                pass
        
        if not has_appnp:
            appnp_pending += 1
            pending_networks.append(network_id)
    
    return {
        'total_graphs': total_graphs,
        'appnp_completed': appnp_completed,
        'appnp_pending': appnp_pending,
        'percent_complete': (appnp_completed / total_graphs * 100) if total_graphs > 0 else 0,
        'completed_networks': completed_networks,
        'pending_networks': pending_networks,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check APPNP baseline progress"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Input directory (e.g., /path/to/hard_negatives_v4)"
    )
    parser.add_argument(
        "--show-pending",
        action="store_true",
        help="Show list of pending networks"
    )
    parser.add_argument(
        "--show-completed",
        action="store_true",
        help="Show list of completed networks"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: Directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    results_dir = input_dir / "results"
    if not results_dir.exists():
        print(f"ERROR: Results directory does not exist: {results_dir}", file=sys.stderr)
        sys.exit(1)
    
    print("="*80)
    print("APPNP BASELINE PROGRESS CHECK")
    print(f"Directory: {input_dir}")
    print("="*80)
    
    progress = check_progress(results_dir)
    
    print(f"\nTotal graphs: {progress['total_graphs']}")
    print(f"APPNP completed: {progress['appnp_completed']}")
    print(f"APPNP pending: {progress['appnp_pending']}")
    print(f"Progress: {progress['percent_complete']:.1f}%")
    
    if args.show_completed and progress['completed_networks']:
        print(f"\nCompleted networks ({len(progress['completed_networks'])}):")
        for net in progress['completed_networks'][:10]:
            print(f"  ✓ {net}")
        if len(progress['completed_networks']) > 10:
            print(f"  ... and {len(progress['completed_networks']) - 10} more")
    
    if args.show_pending and progress['pending_networks']:
        print(f"\nPending networks ({len(progress['pending_networks'])}):")
        for net in progress['pending_networks'][:10]:
            print(f"  ○ {net}")
        if len(progress['pending_networks']) > 10:
            print(f"  ... and {len(progress['pending_networks']) - 10} more")
    
    print("\n" + "="*80)
    
    # Also check if comprehensive_results.csv has APPNP
    comp_results_file = results_dir / "comprehensive_results.csv"
    if comp_results_file.exists():
        try:
            df = pd.read_csv(comp_results_file)
            if 'method' in df.columns:
                total_rows = len(df)
                appnp_rows = (df['method'] == 'appnp').sum()
                print(f"\ncomprehensive_results.csv:")
                print(f"  Total rows: {total_rows}")
                print(f"  APPNP rows: {appnp_rows}")
                if appnp_rows > 0:
                    print(f"  ✓ APPNP is present in comprehensive_results.csv")
                else:
                    print(f"  ○ APPNP not yet in comprehensive_results.csv (run with --regenerate-csv)")
        except Exception as e:
            print(f"\nCould not read comprehensive_results.csv: {e}")
    else:
        print(f"\ncomprehensive_results.csv not found (will be created when script completes)")
    
    print("="*80)


if __name__ == "__main__":
    main()

