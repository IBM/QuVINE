#!/bin/bash
################################################################################
# PPI Disease Network Job Submission — v4
#
# 2 PPI networks × 3 diseases × 50 replicates = 300 analysis jobs.
# Focused on STRING and BioPlex3 for quick results.
#
# Key fix vs v3:
#   ctqw/dtqw were failing because tuned hyperparameters had
#   max_nodes=160, max_edges=200 — views too sparse for dense PPI subgraphs.
#   Fix: after loading hparams, clamp max_nodes=250, max_edges=5000 so every
#   view contains enough edges for quantum walk root nodes to have neighbors.
#
# Parameters vs v3:
#   - Networks   : STRING, BioPlex3 only  (v3: all 5)
#   - Max nodes  : 300  (v3: 500)
#   - Replicates : 50   (v3: 30)
#   - Walltime   : 1:30 (v3: 2:00)
#   - Memory     : 8GB
#
# Usage:
#   bash scripts/submit_ppi_disease_jobs_v4.sh [--dry-run] [--queue QUEUE]
#                                               [--walltime TIME] [--memory MEM]
#                                               [--max-nodes N] [--n-reps N]
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
METHODS="quvine_fused,quvine_ctqw,quvine_dtqw,quvine_rwr,quvine_heat,quvine_poly,quvine_hgcnmf,quvine_pgcnmf,netmf,node2vec,baseline_gcnmf,baseline_filter,graphsage"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --queue)      QUEUE="$2";      shift 2 ;;
        --walltime)   WALLTIME="$2";   shift 2 ;;
        --memory)     MEMORY="$2";     shift 2 ;;
        --max-nodes)  MAX_NODES="$2";  shift 2 ;;
        --n-reps)     N_REPS="$2";     shift 2 ;;
        --python-env) PYTHON_ENV="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true;    shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--queue Q] [--walltime T] [--memory M] [--max-nodes N] [--n-reps R] [--dry-run]"
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
OUTPUT_BASE="/dccstor/boseukb/Q/NetMed/QuVINE/results/ppi_disease_v4"
LOG_DIR="${OUTPUT_BASE}/logs"

mkdir -p "$LOG_DIR" "${OUTPUT_BASE}/results"

# ---------------------------------------------------------------------------
# Network definitions (STRING + BioPlex3 only)
# ---------------------------------------------------------------------------
declare -A NET_PATHS
NET_PATHS["STRING"]="${DATA_ROOT}/networks/STRING/edges_list_ncbi.csv"
NET_PATHS["BioPlex3"]="${DATA_ROOT}/networks/BioPlex3_shared/edges_list_ncbi.csv"

NETWORKS="STRING BioPlex3"
DISEASES="asthma autism schizophrenia"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
TOTAL_JOBS=$(( 2 * 3 * N_REPS ))
echo "======================================================"
echo " PPI Disease Network Job Submission — v4"
echo "======================================================"
echo " Project dir  : $PROJECT_DIR"
echo " Output dir   : $OUTPUT_BASE"
echo " Queue        : $QUEUE"
echo " Wall time    : $WALLTIME"
echo " Memory       : ${MEMORY}GB"
echo " Networks     : $NETWORKS"
echo " Max nodes    : ${MAX_NODES} (seeds+targets always kept)"
echo " Replicates   : ${N_REPS} (seeds 0..$((N_REPS-1)))"
echo " Total jobs   : ${TOTAL_JOBS}"
echo " ctqw/dtqw fix: max_nodes=250, max_edges=5000 (overrides tuned hparams)"
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
            JOB_NAME="ppi4_${NET_ID}"
            JOB_OUT="${LOG_DIR}/${JOB_NAME}.out"
            JOB_ERR="${LOG_DIR}/${JOB_NAME}.err"
            JOB_SH="${LOG_DIR}/${JOB_NAME}.sh"
            RESULT_DIR="${OUTPUT_BASE}/results/${NET_ID}"
            mkdir -p "$RESULT_DIR"

            cat > "$JOB_SH" << BSUBEOF
#!/bin/bash
#BSUB -J ${JOB_NAME}
#BSUB -o ${JOB_OUT}
#BSUB -e ${JOB_ERR}
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -M ${MEMORY}GB
#BSUB -R "rusage[mem=${MEMORY}GB]"

source ${PROJECT_DIR}/${PYTHON_ENV}
cd ${PROJECT_DIR}

python - << 'PYEOF'
import sys, json, warnings
import numpy as np
import networkx as nx
import pandas as pd
sys.path.insert(0, '${PROJECT_DIR}/src')

from quvine.comprehensive_embedding_analysis import run_single_network_analysis
from quvine.data.subgraph import subsample_nodes

MAX_NODES = ${MAX_NODES}
SEED      = ${SEED}
warnings.filterwarnings("ignore")

# ── Load per-network hyperparameters ─────────────────────────────────────
hparam_path = '${HPARAM_JSON}'
try:
    with open(hparam_path) as f:
        hp = json.load(f)
    method_hyperparams = hp['best_params'].get('${NET}', {})
    method_hyperparams.pop('_scores', None)
    print(f'Hyperparameters for ${NET}: {list(method_hyperparams.keys())}')
except FileNotFoundError:
    method_hyperparams = {}
    print(f'WARNING: no per-network hparam file at {hparam_path} — using defaults')

# ── Fix ctqw/dtqw: clamp view size so quantum walks always have neighbors ─
# Tuned hparams had max_nodes=160, max_edges=200, which creates views too
# sparse for dense PPI subgraphs — root nodes end up with zero neighbors.
# Override to allow views covering most of the 300-node graph.
if 'quvine_walks' in method_hyperparams:
    method_hyperparams['quvine_walks']['max_nodes'] = 250
    method_hyperparams['quvine_walks']['max_edges'] = 5000
    print(f'Overrode view constraints: max_nodes=250, max_edges=5000')
else:
    # No tuned hparams for walks — inject the fix as a new entry so the
    # default config also picks it up through the same code path.
    method_hyperparams['quvine_walks'] = {'max_nodes': 250, 'max_edges': 5000}
    print('Injected quvine_walks view constraints (no tuned hparams found)')

# ── Load network ──────────────────────────────────────────────────────────
print('Loading network: ${NET} from ${NET_PATH}')
df = pd.read_csv('${NET_PATH}', usecols=[0, 1], dtype=str)
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
seed_path   = '${DATA_ROOT}/gene_seeds/${DISEASE}_ncbi_seeds.json'
target_path = '${DATA_ROOT}/gwas_catalog_targets/${DISEASE}_targets_ncbi_gwas_catalog.json'

with open(seed_path)   as f: raw_seeds   = json.load(f)
with open(target_path) as f: raw_targets = json.load(f)

seeds_full   = [ncbi_to_node[int(g)] for g in raw_seeds   if int(g) in ncbi_to_node]
targets_full = [ncbi_to_node[int(g)] for g in raw_targets if int(g) in ncbi_to_node]
print(f'Disease ${DISEASE}: {len(seeds_full)}/{len(raw_seeds)} seeds, '
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
    'network':           '${NET}',
    'disease':           '${DISEASE}',
    'network_id':        '${NET_ID}',
    'replicate':         ${SEED},
    'seeds':             seeds,
    'targets':           targets,
    'negative_strategy': 'random',
}

# ── Run analysis ──────────────────────────────────────────────────────────
results = run_single_network_analysis(
    G=G,
    network_id='${NET_ID}',
    network_metadata=metadata,
    output_dir='${RESULT_DIR}',
    embedding_methods='${METHODS}'.split(','),
    verbose=True,
    resume=False,
    method_hyperparams=method_hyperparams,
)

print(f'Analysis complete: ${NET_ID}')
print(f'Results: ${RESULT_DIR}')
PYEOF

exit \$?
BSUBEOF

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
AGG_NAME="ppi4_aggregate"
AGG_SH="${LOG_DIR}/${AGG_NAME}.sh"

DEPENDENCY_STRING=""
for JID in "${JOB_IDS[@]}"; do
    if [ -z "$DEPENDENCY_STRING" ]; then
        DEPENDENCY_STRING="done($JID)"
    else
        DEPENDENCY_STRING="${DEPENDENCY_STRING} && done($JID)"
    fi
done

cat > "$AGG_SH" << BSUBEOF
#!/bin/bash
#BSUB -J ${AGG_NAME}
#BSUB -o ${LOG_DIR}/${AGG_NAME}.out
#BSUB -e ${LOG_DIR}/${AGG_NAME}.err
#BSUB -q ${QUEUE}
#BSUB -W 1:00
#BSUB -M 16GB
#BSUB -R "rusage[mem=16GB]"
$([ -n "$DEPENDENCY_STRING" ] && echo "#BSUB -w \"${DEPENDENCY_STRING}\"")

source ${PROJECT_DIR}/${PYTHON_ENV}
cd ${PROJECT_DIR}

echo "Aggregating PPI v4 results ..."
python scripts/collect_hpc_results.py \
    --results-dir ${OUTPUT_BASE}/results \
    --viz-dir     ${OUTPUT_BASE}/visualizations \
    --n-networks  ${TOTAL_JOBS}

echo "Done. Results: ${OUTPUT_BASE}/results/comprehensive_results.csv"
exit \$?
BSUBEOF

chmod +x "$AGG_SH"

if [ "$DRY_RUN" = true ]; then
    echo "  [DRY RUN] Aggregation: $AGG_NAME"
elif [ ${#JOB_IDS[@]} -gt 0 ]; then
    echo ""
    echo "Submitting aggregation job ..."
    AGG_ID=$(bsub < "$AGG_SH" 2>&1 | grep -oP 'Job <\K[0-9]+')
    if [ -n "$AGG_ID" ]; then
        echo "  Submitted aggregation job $AGG_ID (depends on $JOB_COUNT jobs)"
    else
        echo "  ERROR: failed to submit aggregation job"
    fi
fi

echo ""
echo "======================================================"
echo " SUBMISSION COMPLETE"
echo " Monitor: bjobs -u \$USER"
echo " Results: ${OUTPUT_BASE}/results/"
echo "======================================================"
