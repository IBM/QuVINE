#!/usr/bin/env python3
"""
Download QuVINE benchmark graphs from DGL/DGI, SNAP, and BioSNAP.

Usage:
    python download_quvine_graphs.py --out ./quvine_raw_graphs

Recommended:
    pip install dgl requests beautifulsoup4 tqdm

Notes:
    - DGL datasets are downloaded/cached by DGL under <out>/dgl.
    - SNAP/BioSNAP raw files are downloaded under <out>/snap and <out>/biosnap.
    - This script downloads raw files only. Conversion to simple undirected
      unweighted graphs should be handled in your preprocessing script.
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# ---------------------------------------------------------------------
# 1. DGL / DGI datasets
# ---------------------------------------------------------------------

DGL_DATASETS = [
    "CoraGraphDataset",
    "CiteseerGraphDataset",
    "PubmedGraphDataset",
    "CoraFullDataset",
    "AmazonCoBuyComputerDataset",
    "AmazonCoBuyPhotoDataset",
    "CoauthorPhysicsDataset",
    # Recommended third DGL add-on:
    "CoauthorCSDataset",
    "FlickrDataset",
    "YelpDataset",
    "WikiCSDataset",
    "ChameleonDataset",
    "SquirrelDataset",
    "ActorDataset",
    "CornellDataset",
    "TexasDataset",
    "WisconsinDataset",
    "RomanEmpireDataset",
    "AmazonRatingsDataset",
    "MinesweeperDataset",
    "TolokersDataset",
    "QuestionsDataset",
    "FraudYelpDataset",
    "FraudAmazonDataset",
]


def download_dgl_datasets(out_dir: Path, force_reload: bool = False) -> None:
    """
    Download DGL datasets by instantiating dataset classes.
    DGL stores them in raw_dir.
    """
    try:
        import dgl.data as dgldata
    except ImportError as e:
        print("\n[ERROR] DGL is not installed. Install with: pip install dgl")
        print(f"Original error: {e}")
        return

    dgl_dir = out_dir / "dgl"
    dgl_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Downloading DGL datasets ===")

    for name in DGL_DATASETS:
        print(f"\n[DGL] {name}")
        cls = getattr(dgldata, name, None)
        if cls is None:
            print(f"  [SKIP] {name} not found in your installed DGL version.")
            continue

        try:
            # Most DGL datasets support raw_dir, force_reload, verbose.
            ds = cls(raw_dir=str(dgl_dir), force_reload=force_reload, verbose=True)

            try:
                print(f"  Loaded {len(ds)} graph(s).")
            except Exception:
                print("  Loaded successfully.")

        except TypeError:
            # Some older/newer datasets have slightly different constructors.
            try:
                ds = cls(raw_dir=str(dgl_dir))
                print("  Loaded successfully with raw_dir only.")
            except Exception as e:
                print(f"  [FAILED] {name}: {e}")

        except Exception as e:
            print(f"  [FAILED] {name}: {e}")


# ---------------------------------------------------------------------
# 2. SNAP datasets
# ---------------------------------------------------------------------

SNAP_URL_CANDIDATES = {
    # Road networks
    "roadNet-CA": [
        "https://snap.stanford.edu/data/roadNet-CA.txt.gz",
    ],
    "roadNet-PA": [
        "https://snap.stanford.edu/data/roadNet-PA.txt.gz",
    ],
    "roadNet-TX": [
        "https://snap.stanford.edu/data/roadNet-TX.txt.gz",
    ],

    # Autonomous systems / internet topology
    "as-733": [
        "https://snap.stanford.edu/data/as-733.tar.gz",
    ],
    "Oregon-1": [
        "https://snap.stanford.edu/data/Oregon-1.txt.gz",
        "https://snap.stanford.edu/data/oregon1_010331.txt.gz",
        "https://snap.stanford.edu/data/Oregon-1.tar.gz",
    ],
    "Oregon-2": [
        "https://snap.stanford.edu/data/Oregon-2.txt.gz",
        "https://snap.stanford.edu/data/oregon2_010331.txt.gz",
        "https://snap.stanford.edu/data/Oregon-2.tar.gz",
    ],

    # Community networks
    "com-Amazon": [
        "https://snap.stanford.edu/data/com-Amazon.ungraph.txt.gz",
    ],
    "com-DBLP": [
        "https://snap.stanford.edu/data/com-DBLP.ungraph.txt.gz",
    ],
    "com-Youtube": [
        "https://snap.stanford.edu/data/com-Youtube.ungraph.txt.gz",
    ],

    # Communication / collaboration
    "email-Enron": [
        "https://snap.stanford.edu/data/email-Enron.txt.gz",
    ],
    "ca-GrQc": [
        "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
    ],
    "ca-HepTh": [
        "https://snap.stanford.edu/data/ca-HepTh.txt.gz",
    ],
    "ca-HepPh": [
        "https://snap.stanford.edu/data/ca-HepPh.txt.gz",
    ],
    "ca-CondMat": [
        "https://snap.stanford.edu/data/ca-CondMat.txt.gz",
    ],
    "ca-AstroPh": [
        "https://snap.stanford.edu/data/ca-AstroPh.txt.gz",
    ],

    # Recommended additional SNAP datasets
    "loc-Gowalla": [
        "https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz",
    ],
    "loc-Brightkite": [
        "https://snap.stanford.edu/data/loc-brightkite_edges.txt.gz",
    ],
}


def safe_filename_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def download_file(url: str, dest: Path, timeout: int = 60) -> bool:
    """
    Download a URL to dest.
    Returns True on success, False on HTTP/network failure.
    """
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [EXISTS] {dest.name}")
        return True

    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                print(f"  [MISS] {url} -> HTTP {r.status_code}")
                return False

            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=f"  {dest.name}",
                leave=False,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"  [OK] {dest.name}")
        return True

    except Exception as e:
        print(f"  [FAILED] {url}: {e}")
        return False


def extract_archive(path: Path, remove_archive: bool = False) -> None:
    """
    Extract .tar.gz or .zip files. Leave .txt.gz compressed because these are
    easy to stream during preprocessing.
    """
    try:
        if path.suffix == ".zip":
            extract_dir = path.with_suffix("")
            extract_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(extract_dir)
            print(f"  [EXTRACTED] {path.name} -> {extract_dir.name}")

        elif path.name.endswith(".tar.gz") or path.name.endswith(".tgz"):
            extract_dir = path.parent / path.name.replace(".tar.gz", "").replace(".tgz", "")
            extract_dir.mkdir(exist_ok=True)
            with tarfile.open(path, "r:gz") as tf:
                tf.extractall(extract_dir)
            print(f"  [EXTRACTED] {path.name} -> {extract_dir.name}")

        if remove_archive:
            path.unlink()

    except Exception as e:
        print(f"  [EXTRACT FAILED] {path}: {e}")


def download_snap_datasets(out_dir: Path) -> None:
    snap_dir = out_dir / "snap"
    snap_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Downloading SNAP datasets ===")

    for graph_name, urls in SNAP_URL_CANDIDATES.items():
        graph_dir = snap_dir / graph_name
        graph_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[SNAP] {graph_name}")
        success = False

        for url in urls:
            filename = safe_filename_from_url(url)
            dest = graph_dir / filename
            if download_file(url, dest):
                success = True
                if filename.endswith(".tar.gz") or filename.endswith(".tgz") or filename.endswith(".zip"):
                    extract_archive(dest)
                break

        if not success:
            print(f"  [WARNING] Could not download {graph_name}. Check SNAP URL manually.")


# ---------------------------------------------------------------------
# 3. BioSNAP datasets
# ---------------------------------------------------------------------

BIOSNAP_NAMES = [
    "CC-Neuron",
    "DD-Miner",
    "ChG-Miner",
    "ChG-InterDecagon",
    "DCh-Miner",
    "ChSe-Decagon",
    "DG-AssocMiner",
    "FF-Miner",
    "DF-Miner",
]


def find_biosnap_dataset_pages(index_url: str = "https://snap.stanford.edu/biodata/index.html") -> dict[str, str]:
    """
    Parse BioSNAP index and return mapping:
        dataset name -> dataset page URL
    """
    print("\n[BioSNAP] Fetching index...")
    r = requests.get(index_url, timeout=60)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    pages: dict[str, str] = {}
    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        href = a["href"]

        for name in BIOSNAP_NAMES:
            if name.lower() in text.lower() or name.lower() in href.lower():
                pages[name] = urljoin(index_url, href)

    return pages


def find_download_links_on_page(page_url: str) -> list[str]:
    """
    Get likely downloadable files from a BioSNAP dataset page.
    """
    r = requests.get(page_url, timeout=60)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(page_url, href)

        # BioSNAP pages typically expose downloadable CSV/TSV/TXT archives.
        if re.search(r"\.(csv|tsv|txt|gz|zip|tar\.gz)$", full, flags=re.IGNORECASE):
            links.append(full)

    # Deduplicate while preserving order.
    seen = set()
    deduped = []
    for x in links:
        if x not in seen:
            seen.add(x)
            deduped.append(x)

    return deduped


def download_biosnap_datasets(out_dir: Path) -> None:
    biosnap_dir = out_dir / "biosnap"
    biosnap_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Downloading BioSNAP datasets ===")

    try:
        pages = find_biosnap_dataset_pages()
    except Exception as e:
        print(f"[ERROR] Could not fetch BioSNAP index: {e}")
        return

    for name in BIOSNAP_NAMES:
        print(f"\n[BioSNAP] {name}")
        graph_dir = biosnap_dir / name
        graph_dir.mkdir(parents=True, exist_ok=True)

        page = pages.get(name)
        if page is None:
            print(f"  [WARNING] Could not find BioSNAP page for {name}")
            continue

        print(f"  Page: {page}")

        try:
            links = find_download_links_on_page(page)
        except Exception as e:
            print(f"  [FAILED] Could not parse page: {e}")
            continue

        if not links:
            print("  [WARNING] No downloadable file links found.")
            continue

        downloaded_any = False
        for url in links:
            filename = safe_filename_from_url(url)
            dest = graph_dir / filename

            ok = download_file(url, dest)
            if ok:
                downloaded_any = True
                if filename.endswith(".tar.gz") or filename.endswith(".tgz") or filename.endswith(".zip"):
                    extract_archive(dest)

        if not downloaded_any:
            print(f"  [WARNING] No files downloaded for {name}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default="./quvine_raw_graphs",
        help="Output directory for all downloaded datasets.",
    )
    parser.add_argument(
        "--force-dgl",
        action="store_true",
        help="Force reload DGL datasets.",
    )
    parser.add_argument(
        "--skip-dgl",
        action="store_true",
        help="Skip DGL downloads.",
    )
    parser.add_argument(
        "--skip-snap",
        action="store_true",
        help="Skip SNAP downloads.",
    )
    parser.add_argument(
        "--skip-biosnap",
        action="store_true",
        help="Skip BioSNAP downloads.",
    )

    args = parser.parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out_dir}")

    if not args.skip_dgl:
        download_dgl_datasets(out_dir, force_reload=args.force_dgl)

    if not args.skip_snap:
        download_snap_datasets(out_dir)

    if not args.skip_biosnap:
        download_biosnap_datasets(out_dir)

    print("\nDone.")
    print("Next step: preprocess each raw graph into simple undirected unweighted LCC subgraphs.")


if __name__ == "__main__":
    main()