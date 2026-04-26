#!/usr/bin/env python3
"""
Run APPNP baseline on a single network.

This script is designed to be called by LSF job submission scripts.
It loads an existing graphml file and runs APPNP baseline evaluation.

Usage:
    python scripts/run_appnp_single_network.py \
        --graphml-path /path/to/network.graphml \
        --output-dir /path/to/output \
        --network-id network_name \
        --network-type type_name \
        --negative-strategy strategy \
        [--resume] \
        [--verbose]
"""

import argparse
import sys
import json
import logging
from pathlib import Path
import networkx as nx

# Allow importing from src/ regardless of install state
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quvine.comprehensive_embedding_analysis import run_single_network_analysis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run APPNP baseline on a single network"
    )
    parser.add_argument("--graphml-path", required=True, help="Path to graphml file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--network-id", required=True, help="Network identifier")
    parser.add_argument("--network-type", required=True, help="Network type")
    parser.add_argument("--negative-strategy", default="random", help="Negative sampling strategy")
    parser.add_argument("--resume", action="store_true", help="Skip if APPNP already done")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose logging")
    
    args = parser.parse_args()
    
    graphml_path = Path(args.graphml_path)
    output_dir = Path(args.output_dir)
    
    if not graphml_path.exists():
        logger.error(f"GraphML file not found: {graphml_path}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"="*80)
    logger.info(f"APPNP Baseline - Single Network")
    logger.info(f"Network ID: {args.network_id}")
    logger.info(f"Network Type: {args.network_type}")
    logger.info(f"GraphML: {graphml_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"="*80)
    
    # Load graph
    logger.info("Loading graph...")
    G = nx.read_graphml(str(graphml_path))
    G = nx.convert_node_labels_to_integers(G)
    logger.info(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Metadata
    metadata = {
        'network_id': args.network_id,
        'type': args.network_type,
        'negative_strategy': args.negative_strategy,
    }
    
    # Run APPNP analysis
    try:
        summary = run_single_network_analysis(
            G=G,
            network_id=args.network_id,
            network_metadata=metadata,
            output_dir=str(output_dir),
            embedding_methods=['appnp'],  # Only APPNP
            embedding_dim=128,
            num_seeds=15,
            num_targets=25,
            verbose=args.verbose,
            resume=args.resume,
            method_hyperparams=None,
        )
        
        logger.info(f"="*80)
        logger.info(f"✓ APPNP analysis complete for {args.network_id}")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"✗ APPNP analysis failed for {args.network_id}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

