#!/bin/bash
################################################################################
# OGB arXiv Dataset Job Submission
#
# Processes QuVINE-compatible CSV/JSON graph bundles generated from
# notebooks/ogbn_arxiv_preprocessing.ipynb.
#
# Processes subsampled graphs (2K, 5K, 10K nodes with 30 repeats each).
# Each CSV edgelist + JSON metadata pair becomes one analysis job.
#
# Usage:
#   bash scripts/submit_ogbn_arxiv_jobs.sh [--dry-run]
#       [--queue QUEUE] [--walltime TIME] [--memory MEM]
#       [--seed-pct PCT] [--target-pct PCT]
#       [--data-dir DIR] [--output-dir DIR] [--python-env PATH]
#       [--target-size SIZE]
################################################################################

set -e

QUEUE="normal"
WALLTIME="72:00"
MEMORY="16"
SEED_PCT="10"
TARGET_PCT="10"
DRY_RUN=false
PYTHON_ENV="/u/futro/envs/py311/bin/activate"
METHODS="quvine_fused,quvine_ctqw,quvine_dtqw,quvine_rwr,quvine_heat,quvine_poly,quvine_hgcnmf,quvine_pgcnmf,netmf,node2vec,baseline_gcnmf,baseline_filter,graphsage"
TARGET_SIZE=""

DATA_ROOT="/dccstor/cgq4hls/Q/ogbn_arxiv/processed"
HPARAM_BASE="/dccstor/boseukb/Q/NetMed/QuVINE/results/hparam_tuning"
OUTPUT_BASE="/dccstor/cgq4hls/Q/ogbn_arxiv/results"

while [[ $# -gt 0 ]]; do
    case $1 in
        --queue)       QUEUE="$2";       shift 2 ;;
        --walltime)    WALLTIME="$2";    shift 2 ;;
        --memory)      MEMORY="$2";      shift 2 ;;
        --seed-pct)    SEED_PCT="$2";    shift 2 ;;
        --target-pct)  TARGET_PCT="$2";  shift 2 ;;
        --python-env)  PYTHON_ENV="$2";  shift 2 ;;
        --data-dir)    DATA_ROOT="$2";   shift 2 ;;
        --output-dir)  OUTPUT_BASE="$2"; shift 2 ;;
        --target-size) TARGET_SIZE="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=true;     shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--queue Q] [--walltime T] [--memory M] [--seed-pct P] [--target-pct P] [--target-size SIZE] [--dry-run]"
            exit 1 ;;
    esac
done

PYTHON_ENV="${PYTHON_ENV/#\~/$HOME}"
DATA_ROOT="${DATA_ROOT/#\~/$HOME}"
OUTPUT_BASE="${OUTPUT_BASE/#\~/$HOME}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
LOG_DIR="${OUTPUT_BASE}/logs"

mkdir -p "$LOG_DIR" "${OUTPUT_BASE}/results"

echo "Discovering OGB arXiv graph files in ${DATA_ROOT}..."

if [ -n "$TARGET_SIZE" ]; then
    CSV_FILES=($(find "$DATA_ROOT/n${TARGET_SIZE}" -name "*.csv" -type f ! -name "*_node_index.csv" 2>/dev/null | sort))
    SIZE_TAG="n${TARGET_SIZE}"
else
    CSV_FILES=($(find "$DATA_ROOT" -name "*.csv" -type f ! -name "*_node_index.csv" | sort))
    SIZE_TAG="all"
fi

TOTAL_JOBS=${#CSV_FILES[@]}

if [ $TOTAL_JOBS -eq 0 ]; then
    echo "ERROR: No CSV files found in ${DATA_ROOT}"
    [ -n "$TARGET_SIZE" ] && echo "       (searched in n${TARGET_SIZE} subdirectory)"
    exit 1
fi

echo "======================================================"
echo " OGB arXiv Dataset Job Submission"
echo "======================================================"
echo " Project dir  : $PROJECT_DIR"
echo " Data dir     : $DATA_ROOT"
echo " Output dir   : $OUTPUT_BASE"
echo " Target size  : ${TARGET_SIZE:-all sizes}"
echo " Queue        : $QUEUE"
echo " Wall time    : $WALLTIME"
echo " Memory       : ${MEMORY}GB"
echo " Seed %       : ${SEED_PCT}%"
echo " Target %     : ${TARGET_PCT}%"
echo " Total jobs   : ${TOTAL_JOBS}"
echo " Methods      : ${METHODS}"
echo " Dry run      : $DRY_RUN"
echo "======================================================"
echo ""
[ "$DRY_RUN" = true ] && echo "DRY RUN MODE — no jobs will be submitted" && echo ""

JOB_IDS=()
JOB_COUNT=0

for CSV_PATH in "${CSV_FILES[@]}"; do
    CSV_FILE=$(basename "$CSV_PATH")
    GRAPH_NAME="${CSV_FILE%.csv}"
    JSON_PATH="${CSV_PATH%.csv}.json"

    if [ ! -f "$JSON_PATH" ]; then
        echo "WARNING: No metadata file for ${CSV_FILE}, skipping"
        continue
    fi

    JOB_NAME="ogbn_arxiv_${GRAPH_NAME}"
    JOB_OUT="${LOG_DIR}/${JOB_NAME}.out"
    JOB_ERR="${LOG_DIR}/${JOB_NAME}.err"
    JOB_SH="${LOG_DIR}/${JOB_NAME}.sh"
    RESULT_DIR="${OUTPUT_BASE}/results/${GRAPH_NAME}"
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

warnings.filterwarnings("ignore")

SEED_PCT = ${SEED_PCT}
TARGET_PCT = ${TARGET_PCT}
BASE_SEED = 42
CSV_PATH = '${CSV_PATH}'
JSON_PATH = '${JSON_PATH}'
GRAPH_NAME = '${GRAPH_NAME}'
RESULT_DIR = '${RESULT_DIR}'

with open(JSON_PATH) as f:
    graph_metadata = json.load(f)

print(f'Loaded metadata from {JSON_PATH}')
print(f'Graph: {GRAPH_NAME}')
print(f'Dataset: {graph_metadata.get("dataset_name", "ogbn-arxiv")}')
print(f'Type: {graph_metadata.get("type", "ogb_subsample")}')

df = pd.read_csv(CSV_PATH)
if 'node1' in df.columns and 'node2' in df.columns:
    source_col, target_col = 'node1', 'node2'
elif 'source' in df.columns and 'target' in df.columns:
    source_col, target_col = 'source', 'target'
elif len(df.columns) >= 2:
    source_col, target_col = df.columns[0], df.columns[1]
else:
    print('ERROR: Cannot identify edge columns')
    sys.exit(1)

df[[source_col, target_col]] = df[[source_col, target_col]].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=[source_col, target_col])
df[[source_col, target_col]] = df[[source_col, target_col]].astype(int)

G = nx.from_pandas_edgelist(df, source=source_col, target=target_col)
G.add_nodes_from(range(int(graph_metadata.get('n_nodes', G.number_of_nodes()))))

print(f'Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')

if G.number_of_nodes() > 0 and not nx.is_connected(G):
    print('Graph is disconnected, taking largest connected component')
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    G = nx.convert_node_labels_to_integers(G)
    print(f'LCC: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')
else:
    G = nx.convert_node_labels_to_integers(G)

rng = np.random.default_rng(BASE_SEED)
all_nodes = list(G.nodes())
n_nodes = len(all_nodes)

n_seeds = max(1, int(n_nodes * SEED_PCT / 100))
n_targets = max(1, int(n_nodes * TARGET_PCT / 100))
rng.shuffle(all_nodes)
seeds = all_nodes[:n_seeds]
targets = all_nodes[n_seeds:n_seeds + n_targets]

print(f'Generated {len(seeds)} seeds ({SEED_PCT}%) and {len(targets)} targets ({TARGET_PCT}%)')

method_hyperparams = {}
hparam_candidates = [
    '${HPARAM_BASE}/real_STRING/best_hyperparams.json',
    '${HPARAM_BASE}/real_BioPlex3/best_hyperparams.json',
]

for hparam_path in hparam_candidates:
    try:
        with open(hparam_path) as f:
            hp = json.load(f)
        for net_name, params in hp.get('best_params', {}).items():
            if params and not method_hyperparams:
                method_hyperparams = params.copy()
                method_hyperparams.pop('_scores', None)
                print(f'Using hyperparameters from {hparam_path} ({net_name})')
                break
        if method_hyperparams:
            break
    except FileNotFoundError:
        continue

if not method_hyperparams:
    print('No tuned hyperparameters found, using defaults')

# Adjust quantum walk parameters for larger graphs
target_size = graph_metadata.get('sampling', {}).get('target_size', n_nodes)
if target_size >= 10000:
    max_nodes, max_edges = 500, 10000
elif target_size >= 5000:
    max_nodes, max_edges = 350, 7500
else:
    max_nodes, max_edges = 250, 5000

if 'quvine_walks' in method_hyperparams:
    method_hyperparams['quvine_walks']['max_nodes'] = max_nodes
    method_hyperparams['quvine_walks']['max_edges'] = max_edges
else:
    method_hyperparams['quvine_walks'] = {'max_nodes': max_nodes, 'max_edges': max_edges}

print(f'Quantum walk limits: max_nodes={max_nodes}, max_edges={max_edges}')

analysis_metadata = {
    'type': graph_metadata.get('type', 'ogb_subsample'),
    'graph_name': GRAPH_NAME,
    'network_id': graph_metadata.get('network_id', GRAPH_NAME),
    'dataset_family': 'ogb',
    'dataset_name': 'ogbn-arxiv',
    'csv_path': CSV_PATH,
    'json_path': JSON_PATH,
    'seeds': seeds,
    'targets': targets,
    'negative_strategy': 'random',
    'seed_pct': SEED_PCT,
    'target_pct': TARGET_PCT,
    'base_seed': BASE_SEED,
}
analysis_metadata.update(graph_metadata)

results = run_single_network_analysis(
    G=G,
    network_id=analysis_metadata['network_id'],
    network_metadata=analysis_metadata,
    output_dir=RESULT_DIR,
    embedding_methods='${METHODS}'.split(','),
    verbose=True,
    resume=False,
    method_hyperparams=method_hyperparams,
)

print(f'Analysis complete: {GRAPH_NAME}')
print(f'Results: {RESULT_DIR}')
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

echo ""
echo "======================================================"
echo " Jobs submitted: $JOB_COUNT / $TOTAL_JOBS"
echo "======================================================"

AGG_NAME="ogbn_arxiv_${SIZE_TAG}_aggregate"
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
#BSUB -M 24GB
#BSUB -R "rusage[mem=24GB]"
$([ -n "$DEPENDENCY_STRING" ] && echo "#BSUB -w \"${DEPENDENCY_STRING}\"")

source ${PROJECT_DIR}/${PYTHON_ENV}
cd ${PROJECT_DIR}

echo "Aggregating OGB arXiv results ..."
python scripts/collect_hpc_results.py \
    --results-dir ${OUTPUT_BASE}/results \
    --viz-dir     ${OUTPUT_BASE}/visualizations \
    --n-networks  ${JOB_COUNT}

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

# Made with Bob
