#!/usr/bin/env python3
"""
Package Individual Embedding .npy Files into .npz Archives

For each network, collects all {network_id}_{method}_embedding.npy files
and packages them into a single {network_id}_embeddings.npz file for
easy loading and recomputation.

Usage:
    python scripts/package_embeddings_to_npz.py \
        --results-dir /path/to/results/extended_generators/results \
        [--verbose] \
        [--remove-npy]  # Remove individual .npy files after packaging
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def package_network_embeddings(network_dir: Path, remove_npy: bool = False, verbose: bool = True) -> bool:
    """
    Package all embedding .npy files for a single network into .npz.
    
    Parameters
    ----------
    network_dir : Path
        Directory containing network results
    remove_npy : bool
        If True, remove individual .npy files after packaging
    verbose : bool
        Print progress messages
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    network_id = network_dir.name
    
    # Find all embedding .npy files
    embedding_files = list(network_dir.glob(f"{network_id}_*_embedding.npy"))
    
    if not embedding_files:
        if verbose:
            logger.warning(f"No embeddings found for {network_id}")
        return False
    
    # Load all embeddings into a dictionary
    embeddings = {}
    for npy_file in embedding_files:
        # Extract method name from filename
        # Format: {network_id}_{method}_embedding.npy
        method = npy_file.stem.replace(f"{network_id}_", "").replace("_embedding", "")
        
        try:
            embeddings[method] = np.load(str(npy_file))
        except Exception as e:
            logger.warning(f"Failed to load {npy_file}: {e}")
            continue
    
    if not embeddings:
        logger.warning(f"No valid embeddings loaded for {network_id}")
        return False
    
    # Save as .npz
    npz_path = network_dir / f"{network_id}_embeddings.npz"
    try:
        np.savez_compressed(str(npz_path), **embeddings)
        if verbose:
            logger.info(f"✓ Packaged {len(embeddings)} embeddings: {npz_path}")
        
        # Optionally remove individual .npy files
        if remove_npy:
            for npy_file in embedding_files:
                try:
                    npy_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove {npy_file}: {e}")
            if verbose:
                logger.info(f"  Removed {len(embedding_files)} .npy files")
        
        return True
    except Exception as e:
        logger.error(f"Failed to save {npz_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Package embedding .npy files into .npz archives"
    )
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing network result folders")
    parser.add_argument("--remove-npy", action="store_true",
                        help="Remove individual .npy files after packaging")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return 1
    
    logger.info("="*80)
    logger.info("PACKAGING EMBEDDINGS TO .NPZ")
    logger.info("="*80)
    logger.info(f"Results dir: {results_dir}")
    logger.info(f"Remove .npy: {args.remove_npy}")
    logger.info("="*80)
    
    # Find all network directories
    network_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    if not network_dirs:
        logger.error(f"No network directories found in {results_dir}")
        return 1
    
    logger.info(f"Found {len(network_dirs)} network directories")
    
    success_count = 0
    fail_count = 0
    
    for network_dir in sorted(network_dirs):
        if package_network_embeddings(network_dir, remove_npy=args.remove_npy, verbose=args.verbose):
            success_count += 1
        else:
            fail_count += 1
    
    logger.info("\n" + "="*80)
    logger.info("PACKAGING COMPLETE")
    logger.info(f"Success: {success_count} networks")
    logger.info(f"Failed:  {fail_count} networks")
    logger.info("="*80)
    
    # Example usage message
    if success_count > 0:
        logger.info("\nTo load embeddings later:")
        logger.info("  import numpy as np")
        logger.info("  data = np.load('path/to/{network_id}_embeddings.npz')")
        logger.info("  quvine_rwr_emb = data['quvine_rwr']")
        logger.info("  node2vec_emb = data['node2vec']")
        logger.info("  # etc.")
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

