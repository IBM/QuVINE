#!/bin/bash
################################################################################
# PPI Disease Network Comprehensive Job Submission
#
# Submits jobs for all 5 PPI networks × 3 diseases × 30 replicates = 450 jobs.
# Similar structure to submit_extended_generators_jobs.sh for consistency.
#
# Networks: BioPlex3, HumanNet, ProteomeHD, STRING, PCNet
# Diseases: asthma, autism, schizophrenia
# Replicates: 30 per (network, disease) combination
#
# Each job:
#   - Loads network from CSV edge list
#   - Loads disease seeds and GWAS targets
#   - Subsamples to max_nodes if needed (preserving seeds/targets)
#   - Runs run_single_network_analysis with all embedding methods
#   - Saves GraphML, embeddings (.npy), and 7 CSV result types
#
# After all jobs complete:
#   - Aggregation job collects results by network type
#   - Packaging job creates .npz archives for embeddings
#
# Usage:
#   bash scripts/submit_ppi_comprehensive_jobs.sh [OPTIONS]
#
# Options:
#   --queue QUEUE         LSF queue (default: normal)
#   --walltime TIME       Wall time per job (default: 72:00)
#   --memory MEM          Memory in GB (default: 12)
#   --max-nodes N         Max nodes per subgraph (default: 4000)
#   --n-replicates N      Number of replicates (default: 30)
#   --methods METHODS     Comma-separated method list (default: all 33)
#   --python-env PATH     Path to venv activate script
#   --dry-run             Print jobs without submitting
#
################################################################################

set -e

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
QUEUE="normal"
WALLTIME="72:00"
MEMORY="12"
MAX_NODES="4000"
N_REPS="30"
DRY_RUN=false
PYTHON_ENV="../Python-3.12.2/venv_quvine/bin/activate"

# All 33 embedding methods
METHODS="quvine_fused,quvine_ctqw,quvine_dtqw,quvine_rwr,quvine_heat,quvine_poly,quvine_hgcnmf,quvine_pgcnmf,netmf,node2vec,baseline_gcnmf,baseline_filter,graphsage,gat,gcn,gin,graphgps,sage_mean,sage_gcn,sage_pool,sage_lstm,gat_v2,transformer,pna,edge_cnn,tag_conv,arma,sg_conv,appnp,dna,film,super_gat,general_conv"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --queue)        QUEUE="$2";        shift 2 ;;
        --walltime)     WALLTIME="$2";     shift 2 ;;
        --memory)       MEMORY="$2";       shift 2 ;;
        --max-nodes)    MAX_NODES="$2";    shift 2 ;;
        --n-replicates) N_REPS="$2";       shift 2 ;;
        --methods)      METHODS="$2";      shift 2 ;;
        --python-env)   PYTHON_ENV="$2";   shift 2 ;;
        --dry-run)      DRY_RUN=true;      shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--queue Q] [--walltime T] [--memory M] [--max-nodes N]"
            echo "          [--n-replicates R] [--methods M] [--python-env P] [--dry-run]"
            exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
DATA_ROOT="/dccstor/boseukb/Q/NetMed/Aug21/GWAS_NetworkPropagation/processed_data"
HPARAM_BASE="/dccstor/boseukb/Q/NetMed/QuVINE/results/hparam_tuning"
OUTPUT_BASE="/dccstor/boseukb/Q/NetMed/QuVINE/results/ppi_comprehensive"
LOG_DIR="${OUTPUT_BASE}/logs"

mkdir -p "$LOG_DIR" "${OUTPUT_BASE}/results"

# ---------------------------------------------------------------------------
# Network definitions (all 5 PPI networks)
# ---------------------------------------------------------------------------
declare -A NET_PATHS
NET_PATHS["BioPlex3"]="${DATA_ROOT}/networks/BioPlex3_shared/edges_list_ncbi.csv"
NET_PATHS["HumanNet"]="${DATA_ROOT}/networks/HumanNet/edges_list_ncbi.csv"
NET_PATHS["ProteomeHD"]="${DATA_ROOT}/networks/ProteomeHD/edges_list_ncbi.csv"
NET_PATHS["STRING"]="${DATA_ROOT}/networks/STRING/edges_list_ncbi.csv"
NET_PATHS["PCNet"]="${DATA_ROOT}/networks/PCNet/edges_list_ncbi.csv"

NETWORKS="BioPlex3 HumanNet ProteomeHD STRING PCNet"
DISEASES="asthma autism schizophrenia"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
TOTAL_JOBS=$(( 5 * 3 * N_REPS ))
echo "======================================================"
echo " PPI Disease Network Comprehensive Job Submission"
echo "======================================================"
echo " Project dir  : $PROJECT_DIR"
echo " Output dir   : $OUTPUT_BASE"
echo " Queue        : $QUEUE"
echo " Wall time    : $WALLTIME"
echo " Memory       : ${MEMORY}GB"
echo " Networks     : $NETWORKS"
echo " Diseases     : $DISEASES"
echo " Max nodes    : ${MAX_NODES} (seeds+targets always kept)"
echo " Replicates   : ${N_REPS} per (network, disease)"
echo " Total jobs   : ${TOTAL_JOBS}"
echo " Methods      : $(echo $METHODS | tr ',' ' ' | wc -w) methods"
echo " Dry run      : $DRY_RUN"
echo "======================================================"
echo ""
[ "$DRY_RUN" = true ] && echo "DRY RUN MODE — no jobs will be submitted" && echo ""

# ---------------------------------------------------------------------------
# Submit jobs
# ---------------------------------------------------------------------------
JOB_IDS=()
JOB_COUNT=0

for NET in $NETWORKS; do
    NET_PATH="${NET_PATHS[$NET]}"
    HPARAM_JSON="${HPARAM_BASE}/real_${NET}/best_hyperparams.json"

    for DISEASE in $DISEASES; do
        for REP in $(seq -w 0 $((N_REPS-1))); do
            SEED=$((10#$REP))
            NET_ID="${NET}_${DISEASE}_rep${REP}"
            JOB_NAME="ppi_${NET_ID}"
            JOB_OUT="${LOG_DIR}/${JOB_NAME}.out"
            JOB_ERR="${LOG_DIR}/${JOB_NAME}.err"
            JOB_SH="${LOG_DIR}/${JOB_NAME}.sh"
            RESULT_DIR="${OUTPUT_BASE}/results/${NET_ID}"
            mkdir -p "$RESULT_DIR"

            cat > "$JOB_SH" << 'BSUBEOF'
#!/bin/bash
#BSUB -J JOB_NAME_PLACEHOLDER
#BSUB -o JOB_OUT_PLACEHOLDER
#BSUB -e JOB_ERR_PLACEHOLDER
#BSUB -q QUEUE_PLACEHOLDER
#BSUB -W WALLTIME_PLACEHOLDER
#BSUB -M MEMORY_PLACEHOLDERGB
#BSUB -R "rusage[mem=MEMORY_PLACEHOLDERGB]"

source PROJECT_DIR_PLACEHOLDER/PYTHON_ENV_PLACEHOLDER
cd PROJECT_DIR_PLACEHOLDER

python - << 'PYEOF'
import sys, json, warnings
import numpy as np
import networkx as nx
import pandas as pd
sys.path.insert(0, 'PROJECT_DIR_PLACEHOLDER/src')

from quvine.comprehensive_embedding_analysis import run_single_network_analysis
from quvine.data.subgraph import subsample_nodes

MAX_NODES = MAX_NODES_PLACEHOLDER
SEED      = SEED_PLACEHOLDER
warnings.filterwarnings("ignore")

# ── Load per-network hyperparameters ─────────────────────────────────────
hparam_path = 'HPARAM_JSON_PLACEHOLDER'
try:
    with open(hparam_path) as f:
        hp = json.load(f)
    method_hyperparams = hp['best_params'].get('NET_PLACEHOLDER', {})
    method_hyperparams.pop('_scores', None)
    print(f'Hyperparameters for NET_PLACEHOLDER: {list(method_hyperparams.keys())}')
except FileNotFoundError:
    method_hyperparams = {}
    print(f'WARNING: no per-network hparam file at {hparam_path} — using defaults')

# ── Fix ctqw/dtqw: clamp view size so quantum walks always have neighbors ─
# Tuned hparams may have max_nodes=160, max_edges=200, which creates views
# too sparse for dense PPI subgraphs — root nodes end up with zero neighbors.
# Override to allow views covering most of the subgraph.
if 'quvine_walks' in method_hyperparams:
    method_hyperparams['quvine_walks']['max_nodes'] = 250
    method_hyperparams['quvine_walks']['max_edges'] = 5000
    print(f'Overrode view constraints: max_nodes=250, max_edges=5000')
else:
    # No tuned hparams for walks — inject the fix as a new entry
    method_hyperparams['quvine_walks'] = {'max_nodes': 250, 'max_edges': 5000}
    print('Injected quvine_walks view constraints (no tuned hparams found)')

# ── Load network ──────────────────────────────────────────────────────────
print('Loading network: NET_PLACEHOLDER from NET_PATH_PLACEHOLDER')
df = pd.read_csv('NET_PATH_PLACEHOLDER', usecols=[0, 1], dtype=str)
df.columns = ['node1', 'node2']
df[['node1', 'node2']] = df[['node1', 'node2']].apply(pd.to_numeric, errors='coerce')
n_before = len(df)
df = df.dropna(subset=['node1', 'node2'])
if n_before > len(df):
    print(f'Dropped {n_before - len(df)} rows with non-integer node IDs')
df[['node1', 'node2']] = df[['node1', 'node2']].astype(int)

G_full = nx.from_pandas_edgelist(df, source='node1', target='node2')
G_full = nx.convert_node_labels_to_integers(G_full, label_attribute='ncbi_id')
print(f'Full network: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges')

ncbi_to_node = {G_full.nodes[v]['ncbi_id']: v for v in G_full.nodes()}

# ── Load disease seeds and targets ───────────────────────────────────────
seed_path   = 'DATA_ROOT_PLACEHOLDER/gene_seeds/DISEASE_PLACEHOLDER_ncbi_seeds.json'
target_path = 'DATA_ROOT_PLACEHOLDER/gwas_catalog_targets/DISEASE_PLACEHOLDER_targets_ncbi_gwas_catalog.json'

with open(seed_path)   as f: raw_seeds   = json.load(f)
with open(target_path) as f: raw_targets = json.load(f)

seeds_full   = [ncbi_to_node[int(g)] for g in raw_seeds   if int(g) in ncbi_to_node]
targets_full = [ncbi_to_node[int(g)] for g in raw_targets if int(g) in ncbi_to_node]
print(f'Disease DISEASE_PLACEHOLDER: {len(seeds_full)}/{len(raw_seeds)} seeds, '
      f'{len(targets_full)}/{len(raw_targets)} targets in full network')

if len(seeds_full) == 0 or len(targets_full) == 0:
    print('ERROR: no seeds or targets map into this network — aborting.')
    sys.exit(1)

# ── Guard: trim if seeds+targets exceed budget (seeds take priority) ──────
combined = list(dict.fromkeys(seeds_full + targets_full))
if len(combined) > MAX_NODES:
    print(f'WARNING: {len(combined)} protected nodes > MAX_NODES={MAX_NODES}. Trimming.')
    seeds_full   = seeds_full[:MAX_NODES // 2]
    targets_full = targets_full[:MAX_NODES - len(seeds_full)]
    print(f'After trim: {len(seeds_full)} seeds, {len(targets_full)} targets')

# ── Subsample network (rep-specific seed) ────────────────────────────────
if G_full.number_of_nodes() > MAX_NODES:
    print(f'Subsampling to {MAX_NODES} nodes (seed={SEED}) ...')
    rng = np.random.default_rng(SEED)
    G = subsample_nodes(
        G=G_full,
        seeds=seeds_full,
        targets=targets_full,
        max_nodes=MAX_NODES,
        radius=2,
        rng=rng,
    )
    G = nx.convert_node_labels_to_integers(G)
    print(f'Subsampled: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')

    ncbi_to_node_sub = {G.nodes[v].get('ncbi_id'): v
                        for v in G.nodes() if 'ncbi_id' in G.nodes[v]}
    seeds   = [ncbi_to_node_sub[G_full.nodes[s]['ncbi_id']]
               for s in seeds_full
               if G_full.nodes[s]['ncbi_id'] in ncbi_to_node_sub]
    targets = [ncbi_to_node_sub[G_full.nodes[t]['ncbi_id']]
               for t in targets_full
               if G_full.nodes[t]['ncbi_id'] in ncbi_to_node_sub]
    print(f'After subsampling: {len(seeds)} seeds, {len(targets)} targets retained')
else:
    G = G_full
    seeds, targets = seeds_full, targets_full
    print(f'Network fits within {MAX_NODES} nodes — no subsampling.')

del G_full

if len(seeds) == 0 or len(targets) == 0:
    print('ERROR: no seeds or targets in subgraph — aborting.')
    sys.exit(1)

# ── Metadata ──────────────────────────────────────────────────────────────
metadata = {
    'type':              'real_ppi',
    'network':           'NET_PLACEHOLDER',
    'disease':           'DISEASE_PLACEHOLDER',
    'network_id':        'NET_ID_PLACEHOLDER',
    'replicate':         SEED,
    'seeds':             seeds,
    'targets':           targets,
    'negative_strategy': 'random',
}

# ── Run analysis ──────────────────────────────────────────────────────────
results = run_single_network_analysis(
    G=G,
    network_id='NET_ID_PLACEHOLDER',
    network_metadata=metadata,
    output_dir='RESULT_DIR_PLACEHOLDER',
    embedding_methods='METHODS_PLACEHOLDER'.split(','),
    verbose=True,
    resume=False,
    method_hyperparams=method_hyperparams,
)

print(f'Analysis complete: NET_ID_PLACEHOLDER')
print(f'Results: RESULT_DIR_PLACEHOLDER')
PYEOF

exit $?
BSUBEOF

            # Replace placeholders
            sed -i "s|JOB_NAME_PLACEHOLDER|${JOB_NAME}|g" "$JOB_SH"
            sed -i "s|JOB_OUT_PLACEHOLDER|${JOB_OUT}|g" "$JOB_SH"
            sed -i "s|JOB_ERR_PLACEHOLDER|${JOB_ERR}|g" "$JOB_SH"
            sed -i "s|QUEUE_PLACEHOLDER|${QUEUE}|g" "$JOB_SH"
            sed -i "s|WALLTIME_PLACEHOLDER|${WALLTIME}|g" "$JOB_SH"
            sed -i "s|MEMORY_PLACEHOLDER|${MEMORY}|g" "$JOB_SH"
            sed -i "s|PROJECT_DIR_PLACEHOLDER|${PROJECT_DIR}|g" "$JOB_SH"
            sed -i "s|PYTHON_ENV_PLACEHOLDER|${PYTHON_ENV}|g" "$JOB_SH"
            sed -i "s|MAX_NODES_PLACEHOLDER|${MAX_NODES}|g" "$JOB_SH"
            sed -i "s|SEED_PLACEHOLDER|${SEED}|g" "$JOB_SH"
            sed -i "s|HPARAM_JSON_PLACEHOLDER|${HPARAM_JSON}|g" "$JOB_SH"
            sed -i "s|NET_PLACEHOLDER|${NET}|g" "$JOB_SH"
            sed -i "s|NET_PATH_PLACEHOLDER|${NET_PATH}|g" "$JOB_SH"
            sed -i "s|DATA_ROOT_PLACEHOLDER|${DATA_ROOT}|g" "$JOB_SH"
            sed -i "s|DISEASE_PLACEHOLDER|${DISEASE}|g" "$JOB_SH"
            sed -i "s|NET_ID_PLACEHOLDER|${NET_ID}|g" "$JOB_SH"
            sed -i "s|RESULT_DIR_PLACEHOLDER|${RESULT_DIR}|g" "$JOB_SH"
            sed -i "s|METHODS_PLACEHOLDER|${METHODS}|g" "$JOB_SH"

            chmod +x "$JOB_SH"

            if [ "$DRY_RUN" = true ]; then
                echo "  [DRY RUN] $JOB_NAME"
            else
                JOB_ID=$(bsub < "$JOB_SH" 2>&1 | grep -oP 'Job <\K[0-9]+')
                if [ -n "$JOB_ID" ]; then
                    echo "  Submitted $JOB_ID: $JOB_NAME"
                    JOB_IDS+=("$JOB_ID")
                    JOB_COUNT=$((JOB_COUNT + 1))
                else
                    echo "  ERROR: failed to submit $JOB_NAME"
                fi
            fi
        done
    done
done

echo ""
echo "======================================================"
echo " Jobs submitted: $JOB_COUNT / $TOTAL_JOBS"
echo "======================================================"

# ---------------------------------------------------------------------------
# Aggregation job (depends on all analysis jobs)
# ---------------------------------------------------------------------------
if [ "$DRY_RUN" = false ] && [ ${#JOB_IDS[@]} -gt 0 ]; then
    AGG_NAME="ppi_aggregate"
    AGG_SH="${LOG_DIR}/${AGG_NAME}.sh"

    DEPENDENCY_STRING=""
    for JID in "${JOB_IDS[@]}"; do
        if [ -z "$DEPENDENCY_STRING" ]; then
            DEPENDENCY_STRING="ended($JID)"
        else
            DEPENDENCY_STRING="${DEPENDENCY_STRING} && ended($JID)"
        fi
    done

    cat > "$AGG_SH" << BSUBEOF
#!/bin/bash
#BSUB -J ${AGG_NAME}
#BSUB -o ${LOG_DIR}/${AGG_NAME}.out
#BSUB -e ${LOG_DIR}/${AGG_NAME}.err
#BSUB -q ${QUEUE}
#BSUB -W 2:00
#BSUB -M 32GB
#BSUB -R "rusage[mem=32GB]"
#BSUB -w "${DEPENDENCY_STRING}"

source ${PROJECT_DIR}/${PYTHON_ENV}
cd ${PROJECT_DIR}

echo "======================================================"
echo " PPI Comprehensive Results Aggregation"
echo "======================================================"
echo " Results dir: ${OUTPUT_BASE}/results"
echo " Networks: ${NETWORKS}"
echo " Diseases: ${DISEASES}"
echo ""

python scripts/aggregate_ppi_comprehensive.py \\
    --results-dir ${OUTPUT_BASE}/results \\
    --output-dir ${OUTPUT_BASE} \\
    --networks "${NETWORKS}" \\
    --diseases "${DISEASES}"

echo ""
echo "Aggregation complete."
echo "Results: ${OUTPUT_BASE}/ppi_comprehensive_results.csv"
exit \$?
BSUBEOF

    chmod +x "$AGG_SH"
    AGG_JOB_ID=$(bsub < "$AGG_SH" 2>&1 | grep -oP 'Job <\K[0-9]+')
    if [ -n "$AGG_JOB_ID" ]; then
        echo ""
        echo "Submitted aggregation job: $AGG_JOB_ID"
        echo "  Depends on: ${#JOB_IDS[@]} analysis jobs"
    else
        echo ""
        echo "ERROR: failed to submit aggregation job"
    fi

    # -----------------------------------------------------------------------
    # Packaging job (depends on aggregation)
    # -----------------------------------------------------------------------
    PKG_NAME="ppi_package_embeddings"
    PKG_SH="${LOG_DIR}/${PKG_NAME}.sh"

    cat > "$PKG_SH" << BSUBEOF
#!/bin/bash
#BSUB -J ${PKG_NAME}
#BSUB -o ${LOG_DIR}/${PKG_NAME}.out
#BSUB -e ${LOG_DIR}/${PKG_NAME}.err
#BSUB -q ${QUEUE}
#BSUB -W 1:00
#BSUB -M 16GB
#BSUB -R "rusage[mem=16GB]"
#BSUB -w "ended(${AGG_JOB_ID})"

source ${PROJECT_DIR}/${PYTHON_ENV}
cd ${PROJECT_DIR}

echo "======================================================"
echo " PPI Embedding Packaging"
echo "======================================================"
echo " Results dir: ${OUTPUT_BASE}/results"
echo ""

python scripts/package_embeddings_to_npz.py \\
    --results-dir ${OUTPUT_BASE}/results

echo ""
echo "Packaging complete."
echo "Created .npz archives for all networks."
exit \$?
BSUBEOF

    chmod +x "$PKG_SH"
    PKG_JOB_ID=$(bsub < "$PKG_SH" 2>&1 | grep -oP 'Job <\K[0-9]+')
    if [ -n "$PKG_JOB_ID" ]; then
        echo "Submitted packaging job: $PKG_JOB_ID"
        echo "  Depends on: aggregation job $AGG_JOB_ID"
    else
        echo "ERROR: failed to submit packaging job"
    fi
fi

echo ""
echo "======================================================"
echo " Submission complete"
echo "======================================================"
echo " Analysis jobs : $JOB_COUNT"
echo " Log directory : $LOG_DIR"
echo " Results dir   : ${OUTPUT_BASE}/results"
echo ""
echo "Monitor with: bjobs -w"
echo "======================================================"

