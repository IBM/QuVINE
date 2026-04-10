#!/bin/bash
################################################################################
# PPI Disease Network Job Submission
#
# Submits 15 LSF jobs (5 PPI networks × 3 diseases) using tuned hyperparameters
# from results/hparam_tuning/best_hyperparams.json.
#
# Networks  : STRING, BioPlex3, HumanNet, PCNet, ProteomeHD
# Diseases  : asthma, autism, schizophrenia
# Methods   : all (quvine_fused, quvine_ctqw, quvine_dtqw, quvine_rwr,
#              quvine_heat, quvine_poly, quvine_hgcnmf, quvine_pgcnmf,
#              netmf, node2vec, baseline_gcnmf, baseline_filter, graphsage)
#
# Usage:
#   bash scripts/submit_ppi_disease_jobs.sh [--dry-run] [--queue QUEUE]
#                                            [--walltime TIME] [--memory MEM]
#
# Author: QuVINE Team
# Date: 2026-04-09
################################################################################

set -e

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
QUEUE="normal"
WALLTIME="12:00"
MEMORY="4"
DRY_RUN=false
PYTHON_ENV="../Python-3.12.2/venv_quvine/bin/activate"
RESUME=false

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --queue)     QUEUE="$2";    shift 2 ;;
        --walltime)  WALLTIME="$2"; shift 2 ;;
        --memory)    MEMORY="$2";   shift 2 ;;
        --python-env) PYTHON_ENV="$2"; shift 2 ;;
        --resume)    RESUME=true;   shift ;;
        --dry-run)   DRY_RUN=true;  shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--queue QUEUE] [--walltime TIME] [--memory MEM] [--resume] [--dry-run]"
            exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
DATA_ROOT="/dccstor/boseukb/Q/NetMed/Aug21/GWAS_NetworkPropagation/processed_data"
HPARAM_JSON="/dccstor/boseukb/Q/NetMed/QuVINE/results/hparam_tuning/best_hyperparams.json"
OUTPUT_BASE="/dccstor/boseukb/Q/NetMed/QuVINE/results/ppi_disease"
LOG_DIR="${OUTPUT_BASE}/logs"
METHODS="quvine_fused,quvine_ctqw,quvine_dtqw,quvine_rwr,quvine_heat,quvine_poly,quvine_hgcnmf,quvine_pgcnmf,netmf,node2vec,baseline_gcnmf,baseline_filter,graphsage"

mkdir -p "$LOG_DIR"
mkdir -p "${OUTPUT_BASE}/results"

# ---------------------------------------------------------------------------
# Network definitions  (display_name → edge-list path, hparam key)
# ---------------------------------------------------------------------------
declare -A NET_PATHS
NET_PATHS["STRING"]="${DATA_ROOT}/networks/STRING/edges_list_ncbi.csv"
NET_PATHS["BioPlex3"]="${DATA_ROOT}/networks/BioPlex3_shared/edges_list_ncbi.csv"
NET_PATHS["HumanNet"]="${DATA_ROOT}/networks/HumanNetV3/edges_list_ncbi.csv"
NET_PATHS["PCNet"]="${DATA_ROOT}/networks/PCNet/edges_list_ncbi.csv"
NET_PATHS["ProteomeHD"]="${DATA_ROOT}/networks/ProteomeHD/edges_list_ncbi.csv"

NETWORKS="STRING BioPlex3 HumanNet PCNet ProteomeHD"
DISEASES="asthma autism schizophrenia"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "======================================================"
echo " PPI Disease Network Job Submission"
echo "======================================================"
echo " Project dir  : $PROJECT_DIR"
echo " Data root    : $DATA_ROOT"
echo " Hparam file  : $HPARAM_JSON"
echo " Output dir   : $OUTPUT_BASE"
echo " Queue        : $QUEUE"
echo " Wall time    : $WALLTIME"
echo " Memory       : ${MEMORY}GB"
echo " Methods      : $METHODS"
echo " Resume       : $RESUME"
echo " Dry run      : $DRY_RUN"
echo "======================================================"
echo ""
[ "$DRY_RUN" = true ] && echo "DRY RUN MODE — no jobs will be submitted" && echo ""

# ---------------------------------------------------------------------------
# Submit one job per (network, disease)
# ---------------------------------------------------------------------------
ANALYSIS_JOB_IDS=()
JOB_COUNT=0

for NET in $NETWORKS; do
    NET_PATH="${NET_PATHS[$NET]}"
    for DISEASE in $DISEASES; do
        NET_ID="${NET}_${DISEASE}"
        JOB_NAME="quvine_ppi_${NET_ID}"
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

python -c "
import sys, json, numpy as np
sys.path.insert(0, '${PROJECT_DIR}/src')

import networkx as nx
import pandas as pd
from quvine.comprehensive_embedding_analysis import run_single_network_analysis

# ── Load tuned hyperparameters ────────────────────────────────────────────
with open('${HPARAM_JSON}') as f:
    all_hparams = json.load(f)

method_hyperparams = all_hparams['best_params'].get('${NET}', {})
# Remove internal _scores key if present
method_hyperparams.pop('_scores', None)
print(f'Loaded tuned hyperparameters for ${NET}: {list(method_hyperparams.keys())}')

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
G = nx.from_pandas_edgelist(df, source='node1', target='node2')
G = nx.convert_node_labels_to_integers(G, label_attribute='ncbi_id')
print(f'Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')

# Build ncbi_id → integer-node mapping
ncbi_to_node = {G.nodes[v]['ncbi_id']: v for v in G.nodes()}

# ── Load disease seeds and targets ───────────────────────────────────────
seed_path   = '${DATA_ROOT}/gene_seeds/${DISEASE}_ncbi_seeds.json'
target_path = '${DATA_ROOT}/gwas_catalog_targets/${DISEASE}_targets_ncbi_gwas_catalog.json'

with open(seed_path)   as f: raw_seeds   = json.load(f)
with open(target_path) as f: raw_targets = json.load(f)

# Map to integer node IDs (only those present in the network)
seeds   = [ncbi_to_node[int(g)] for g in raw_seeds   if int(g) in ncbi_to_node]
targets = [ncbi_to_node[int(g)] for g in raw_targets if int(g) in ncbi_to_node]
print(f'Disease ${DISEASE}: {len(seeds)}/{len(raw_seeds)} seeds in network, '
      f'{len(targets)}/{len(raw_targets)} targets in network')

if len(seeds) == 0 or len(targets) == 0:
    print('ERROR: no seeds or targets map into this network — aborting.')
    sys.exit(1)

# ── Metadata ──────────────────────────────────────────────────────────────
metadata = {
    'type': 'real_ppi',
    'network': '${NET}',
    'disease': '${DISEASE}',
    'network_id': '${NET_ID}',
    'seeds': seeds,
    'targets': targets,
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
    resume=${RESUME^},
    method_hyperparams=method_hyperparams,
)

print(f'Analysis complete for ${NET_ID}')
print(f'Results saved to: ${RESULT_DIR}')
"

exit \$?
BSUBEOF

        chmod +x "$JOB_SH"

        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY RUN] Would submit: $JOB_NAME"
        else
            JOB_ID=$(bsub < "$JOB_SH" 2>&1 | grep -oP 'Job <\K[0-9]+')
            if [ -n "$JOB_ID" ]; then
                echo "  Submitted job $JOB_ID: $JOB_NAME"
                ANALYSIS_JOB_IDS+=("$JOB_ID")
                JOB_COUNT=$((JOB_COUNT + 1))
            else
                echo "  ERROR: Failed to submit $JOB_NAME"
            fi
        fi
    done
done

echo ""
echo "======================================================"
echo " Jobs submitted: $JOB_COUNT / 15"
echo "======================================================"

# ---------------------------------------------------------------------------
# Aggregation job — runs after all analysis jobs complete
# ---------------------------------------------------------------------------
AGG_NAME="quvine_ppi_aggregate"
AGG_SH="${LOG_DIR}/${AGG_NAME}.sh"

DEPENDENCY_STRING=""
for JID in "${ANALYSIS_JOB_IDS[@]}"; do
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
#BSUB -W 2:00
#BSUB -M 32GB
#BSUB -R "rusage[mem=32GB]"
$([ -n "$DEPENDENCY_STRING" ] && echo "#BSUB -w \"${DEPENDENCY_STRING}\"")

source ${PROJECT_DIR}/${PYTHON_ENV}
cd ${PROJECT_DIR}

echo "================================================"
echo "Aggregating PPI disease network results"
echo "================================================"

python scripts/collect_hpc_results.py \
    --results-dir ${OUTPUT_BASE}/results \
    --viz-dir     ${OUTPUT_BASE}/visualizations \
    --n-networks  15

echo "================================================"
echo "Aggregation complete"
echo "Results : ${OUTPUT_BASE}/results/comprehensive_results.csv"
echo "Plots   : ${OUTPUT_BASE}/visualizations/"
echo "================================================"
exit \$?
BSUBEOF

chmod +x "$AGG_SH"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "  [DRY RUN] Would submit aggregation job: $AGG_NAME"
    [ -n "$DEPENDENCY_STRING" ] && echo "  [DRY RUN] Dependencies: $DEPENDENCY_STRING"
elif [ ${#ANALYSIS_JOB_IDS[@]} -gt 0 ]; then
    echo ""
    echo "Submitting aggregation job..."
    AGG_ID=$(bsub < "$AGG_SH" 2>&1 | grep -oP 'Job <\K[0-9]+')
    if [ -n "$AGG_ID" ]; then
        echo "  Submitted aggregation job $AGG_ID: $AGG_NAME"
        echo "  Depends on ${#ANALYSIS_JOB_IDS[@]} analysis jobs"
    else
        echo "  ERROR: Failed to submit aggregation job"
    fi
else
    echo "No analysis jobs submitted — skipping aggregation job."
fi

echo ""
echo "======================================================"
echo " SUBMISSION COMPLETE"
echo "======================================================"
echo " Analysis jobs : $JOB_COUNT"
if [ "$DRY_RUN" = false ] && [ ${#ANALYSIS_JOB_IDS[@]} -gt 0 ]; then
    echo " Aggregation   : 1 (runs after all analysis jobs)"
    echo ""
    echo " Monitor:  bjobs -u \$USER"
    echo " Results:  ${OUTPUT_BASE}/results/"
    echo " Plots:    ${OUTPUT_BASE}/visualizations/"
fi
echo "======================================================"
