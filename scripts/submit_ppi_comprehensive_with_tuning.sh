#!/bin/bash
################################################################################
# PPI Comprehensive Job Submission with Hyperparameter Tuning
#
# Phase 1: Hyperparameter Tuning (5 jobs — one per PPI network)
#   tune_hyperparameters.py tunes per network topology (not per task/disease),
#   so a single tuning job per network covers ranking, classification, and LP.
#
# Phase 2: Main Analysis (750 jobs, depends on that network's tuning job)
#   - Ranking         : 5 networks × 3 diseases × 30 reps = 450 jobs
#   - Classification  : 5 networks × 30 reps              = 150 jobs
#   - Link Prediction : 5 networks × 30 reps              = 150 jobs
#
# Phase 3: Aggregation (6 jobs)
#   - Per-network     : 5 jobs (wait for all analysis jobs for that network)
#   - Comprehensive   : 1 job  (wait for all 5 per-network jobs)
#
# Total: 761 jobs
#
# Networks  : BioPlex3, HumanNet, STRING, ProteomeHD, PCNet
# Diseases  : asthma, autism, schizophrenia  (ranking only)
# Methods   : 15 embedding methods
#
# Usage:
#   bash scripts/submit_ppi_comprehensive_with_tuning.sh [OPTIONS]
#
# Options:
#   --queue Q            LSF queue             (default: normal)
#   --walltime T         Wall time per job     (default: 48:00)
#   --tune-walltime T    Wall time tuning job  (default: 72:00)
#   --memory M           Memory in GB          (default: 16)
#   --max-nodes N        Max nodes per network (default: 4000)
#   --n-reps N           Replicates per run    (default: 30)
#   --n-trials N         Optuna trials         (default: 30)
#   --pilot-nodes N      Tuning pilot nodes    (default: 150)
#   --methods M          Comma-separated list  (default: all 15)
#   --python-env PATH    Path to venv activate
#   --dry-run            Print jobs, don't submit
#   --skip-tuning        Skip Phase 1 (use existing hparam files)
#   --skip-ranking       Skip ranking analysis jobs
#   --skip-classification  Skip classification analysis jobs
#   --skip-link-prediction Skip link prediction analysis jobs
################################################################################

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
QUEUE="normal"
WALLTIME="48:00"
TUNE_WALLTIME="72:00"
MEMORY="16"
MAX_NODES="4000"
N_REPS="30"
N_TRIALS="30"
PILOT_NODES="150"
DRY_RUN=false
SKIP_TUNING=false
SKIP_RANKING=false
SKIP_CLASSIFICATION=false
SKIP_LINK_PREDICTION=false
PYTHON_ENV="../Python-3.12.2/venv_quvine/bin/activate"
METHODS="quvine_rwr,quvine_ctqw,quvine_dtqw,quvine_baseline_heat,quvine_baseline_poly,quvine_rwr_heat,quvine_rwr_poly,quvine_ctqw_heat,quvine_ctqw_poly,gat_baseline,gat_heat,gat_poly,gat_rwr,gat_ctqw,gat_dtqw,gat_rwr_heat,gat_rwr_poly,gat_ctqw_heat,gat_ctqw_poly,gat_dtqw_heat,gat_dtqw_poly,graphgps_baseline,graphgps_heat,graphgps_poly,graphgps_rwr,graphgps_ctqw,graphgps_dtqw,graphgps_rwr_heat,graphgps_rwr_poly,graphgps_ctqw_heat,graphgps_ctqw_poly,graphgps_dtqw_heat,graphgps_dtqw_poly,node2vec,netmf,graphsage,appnp,baseline_filter,baseline_gcnmf"

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --queue)                QUEUE="$2";          shift 2 ;;
        --walltime)             WALLTIME="$2";        shift 2 ;;
        --tune-walltime)        TUNE_WALLTIME="$2";   shift 2 ;;
        --memory)               MEMORY="$2";          shift 2 ;;
        --max-nodes)            MAX_NODES="$2";       shift 2 ;;
        --n-reps)               N_REPS="$2";          shift 2 ;;
        --n-trials)             N_TRIALS="$2";        shift 2 ;;
        --pilot-nodes)          PILOT_NODES="$2";     shift 2 ;;
        --methods)              METHODS="$2";         shift 2 ;;
        --python-env)           PYTHON_ENV="$2";      shift 2 ;;
        --dry-run)              DRY_RUN=true;         shift ;;
        --skip-tuning)          SKIP_TUNING=true;     shift ;;
        --skip-ranking)         SKIP_RANKING=true;    shift ;;
        --skip-classification)  SKIP_CLASSIFICATION=true; shift ;;
        --skip-link-prediction) SKIP_LINK_PREDICTION=true; shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--queue Q] [--walltime T] [--tune-walltime T] [--memory M]"
            echo "          [--max-nodes N] [--n-reps R] [--n-trials N] [--pilot-nodes N]"
            echo "          [--methods M] [--python-env P]"
            echo "          [--dry-run] [--skip-tuning]"
            echo "          [--skip-ranking] [--skip-classification] [--skip-link-prediction]"
            exit 1 ;;
    esac
done

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
DATA_ROOT="/dccstor/boseukb/Q/NetMed/Aug21/GWAS_NetworkPropagation/processed_data"
OUTPUT_BASE="/dccstor/boseukb/Q/NetMed/QuVINE/results/ppi_comprehensive"
HPARAM_DIR="${OUTPUT_BASE}/hparam_tuning"
LOG_DIR="${OUTPUT_BASE}/logs"
RESULTS_DIR="${OUTPUT_BASE}/results"
AGG_DIR="${OUTPUT_BASE}/aggregated"

mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$HPARAM_DIR" "$AGG_DIR"

# Space-separated methods for tune_hyperparameters.py (which uses nargs="*")
METHODS_SPACE="${METHODS//,/ }"

# ── Network and disease definitions ───────────────────────────────────────────
NETWORKS=(BioPlex3 HumanNet STRING ProteomeHD PCNet)
DISEASES=(asthma autism schizophrenia)

declare -A NET_PATHS
NET_PATHS["BioPlex3"]="${DATA_ROOT}/networks/BioPlex3_shared/edges_list_ncbi.csv"
NET_PATHS["HumanNet"]="${DATA_ROOT}/networks/HumanNetV3/edges_list_ncbi.csv"
NET_PATHS["STRING"]="${DATA_ROOT}/networks/STRING/edges_list_ncbi.csv"
NET_PATHS["ProteomeHD"]="${DATA_ROOT}/networks/ProteomeHD/edges_list_ncbi.csv"
NET_PATHS["PCNet"]="${DATA_ROOT}/networks/PCNet/edges_list_ncbi.csv"

# ── Helper: submit a BSub job with optional dependency ────────────────────────
submit_bsub() {
    local script="$1"
    local dep="$2"
    if [ "$DRY_RUN" = true ]; then
        echo "DRYRUN"
        return
    fi
    if [ -n "$dep" ]; then
        bsub -w "$dep" < "$script" 2>&1 | grep -oP 'Job <\K[0-9]+'
    else
        bsub < "$script" 2>&1 | grep -oP 'Job <\K[0-9]+'
    fi
}

# ── Job counts ────────────────────────────────────────────────────────────────
N_NETS=${#NETWORKS[@]}
N_DISEASES=${#DISEASES[@]}
ANALYSIS_RANKING=$(( N_NETS * N_DISEASES * N_REPS ))
ANALYSIS_CLASS=$(( N_NETS * N_REPS ))
ANALYSIS_LINK=$(( N_NETS * N_REPS ))
TOTAL_ANALYSIS=0
[ "$SKIP_RANKING" = false ]         && TOTAL_ANALYSIS=$(( TOTAL_ANALYSIS + ANALYSIS_RANKING ))
[ "$SKIP_CLASSIFICATION" = false ]  && TOTAL_ANALYSIS=$(( TOTAL_ANALYSIS + ANALYSIS_CLASS ))
[ "$SKIP_LINK_PREDICTION" = false ] && TOTAL_ANALYSIS=$(( TOTAL_ANALYSIS + ANALYSIS_LINK ))

echo "============================================================"
echo " PPI Comprehensive Job Submission with Hyperparameter Tuning"
echo "============================================================"
echo " Networks   : ${NETWORKS[*]}"
echo " Diseases   : ${DISEASES[*]}"
echo ""
echo " Phase 1 (tuning)    : ${N_NETS} jobs  [skip: ${SKIP_TUNING}]"
echo " Phase 2 (analysis)  :"
echo "   Ranking           : ${ANALYSIS_RANKING} jobs  [skip: ${SKIP_RANKING}]"
echo "   Classification    : ${ANALYSIS_CLASS} jobs  [skip: ${SKIP_CLASSIFICATION}]"
echo "   Link Prediction   : ${ANALYSIS_LINK} jobs  [skip: ${SKIP_LINK_PREDICTION}]"
echo "   Total analysis    : ${TOTAL_ANALYSIS} jobs"
echo " Phase 3 (agg)       : $((N_NETS + 1)) jobs"
echo ""
echo " Replicates : ${N_REPS}"
echo " Trials     : ${N_TRIALS}"
echo " Queue      : ${QUEUE}"
echo " Wall time  : ${WALLTIME}  (tuning: ${TUNE_WALLTIME})"
echo " Memory     : ${MEMORY}GB"
echo " Max nodes  : ${MAX_NODES}"
echo " Methods    : $(echo "$METHODS" | tr ',' '\n' | wc -l)"
echo " Dry run    : ${DRY_RUN}"
echo "============================================================"
echo ""
[ "$DRY_RUN" = true ] && echo "DRY RUN MODE — no jobs will be submitted" && echo ""

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: HYPERPARAMETER TUNING  (5 jobs, one per PPI network)
# ══════════════════════════════════════════════════════════════════════════════

declare -A TUNE_JOB_ID

if [ "$SKIP_TUNING" = false ]; then
    echo "PHASE 1: Hyperparameter Tuning"
    echo "================================"
    echo ""

    for NET in "${NETWORKS[@]}"; do
        JOB_NAME="tune_ppi_${NET}"
        JOB_SH="${LOG_DIR}/${JOB_NAME}.sh"
        HPARAM_OUT="${HPARAM_DIR}/${NET}"
        mkdir -p "$HPARAM_OUT"

        cat > "$JOB_SH" << TUNEEOF
#!/bin/bash
#BSUB -J ${JOB_NAME}
#BSUB -o ${LOG_DIR}/${JOB_NAME}.out
#BSUB -e ${LOG_DIR}/${JOB_NAME}.err
#BSUB -q ${QUEUE}
#BSUB -W ${TUNE_WALLTIME}
#BSUB -M 32GB
#BSUB -R "rusage[mem=32GB]"

source ${PROJECT_DIR}/${PYTHON_ENV}
cd ${PROJECT_DIR}

echo "======================================"
echo " Tuning: ${NET}"
echo " Output : ${HPARAM_OUT}"
echo " Started: \$(date)"
echo "======================================"

python scripts/tune_hyperparameters.py \\
    --network-type real_${NET} \\
    --output-dir   ${HPARAM_OUT} \\
    --n-trials     ${N_TRIALS} \\
    --pilot-nodes  ${PILOT_NODES} \\
    --pilot-seeds  3 \\
    --methods      ${METHODS_SPACE}

echo "======================================"
echo " Finished: ${NET}  at \$(date)"
echo "======================================"
exit \$?
TUNEEOF

        chmod +x "$JOB_SH"

        JOB_ID=$( submit_bsub "$JOB_SH" "" )
        TUNE_JOB_ID["$NET"]="${JOB_ID}"
        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY RUN] ${JOB_NAME}"
        else
            echo "  Submitted ${JOB_ID}: ${JOB_NAME}"
        fi
    done

    echo ""
    echo "Phase 1: ${#TUNE_JOB_ID[@]} / ${N_NETS} tuning jobs submitted."
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: MAIN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "PHASE 2: Main Analysis"
echo "======================"

RANK_JOB_NAMES=()
CLASS_JOB_NAMES=()
LINK_JOB_NAMES=()

# ── 2A: Ranking (5 nets × 3 diseases × N_REPS) ───────────────────────────────
if [ "$SKIP_RANKING" = false ]; then
    echo ""
    echo "2A: Ranking (${ANALYSIS_RANKING} jobs)"

    for NET in "${NETWORKS[@]}"; do
        NET_PATH="${NET_PATHS[$NET]}"
        HPARAM_JSON="${HPARAM_DIR}/${NET}/best_hyperparams.json"

        # Dependency: wait for this network's tuning job
        TUNE_DEP=""
        if [ "$SKIP_TUNING" = false ]; then
            TID="${TUNE_JOB_ID[$NET]:-}"
            [ -n "$TID" ] && [ "$TID" != "DRYRUN" ] && TUNE_DEP="ended(${TID})"
        fi

        echo "  Network: ${NET}"
        for DISEASE in "${DISEASES[@]}"; do
            for REP in $( seq -w 0 $(( N_REPS - 1 )) ); do
                SEED=$(( 10#$REP ))
                NET_ID="${NET}_${DISEASE}_rep${REP}"
                JOB_NAME="ppi_rank_${NET}_${DISEASE}_r${REP}"
                JOB_SH="${LOG_DIR}/${JOB_NAME}.sh"
                RESULT_DIR="${RESULTS_DIR}/${NET_ID}"
                mkdir -p "$RESULT_DIR"

                cat > "$JOB_SH" << 'BSUBEOF'
#!/bin/bash
#BSUB -J JOB_NAME_PH
#BSUB -o LOG_DIR_PH/JOB_NAME_PH.out
#BSUB -e LOG_DIR_PH/JOB_NAME_PH.err
#BSUB -q QUEUE_PH
#BSUB -W WALLTIME_PH
#BSUB -M MEMORY_PHGB
#BSUB -R "rusage[mem=MEMORY_PHGB]"

source PROJECT_DIR_PH/PYTHON_ENV_PH
cd PROJECT_DIR_PH

python - << 'PYEOF'
import sys, json, warnings
import numpy as np
import networkx as nx
import pandas as pd
sys.path.insert(0, 'PROJECT_DIR_PH/src')

from quvine.comprehensive_embedding_analysis import run_single_network_analysis
from quvine.data.subgraph import subsample_nodes

MAX_NODES = MAX_NODES_PH
SEED      = SEED_PH
warnings.filterwarnings("ignore")

# Load per-network hyperparameters
hparam_path = 'HPARAM_JSON_PH'
try:
    with open(hparam_path) as f:
        hp = json.load(f)
    method_hyperparams = hp['best_params'].get('real_NET_PH', {})
    method_hyperparams.pop('_scores', None)
    print(f'Hyperparameters loaded for real_NET_PH: {list(method_hyperparams.keys())}')
except FileNotFoundError:
    method_hyperparams = {}
    print(f'WARNING: no hparam file at {hparam_path} — using defaults')

# Ensure view constraints are sane for PPI subgraphs
if 'quvine_walks' not in method_hyperparams:
    method_hyperparams['quvine_walks'] = {'max_nodes': 250, 'max_edges': 5000}

# Load network
df = pd.read_csv('NET_PATH_PH', usecols=[0, 1], dtype=str)
df.columns = ['node1', 'node2']
df[['node1', 'node2']] = df[['node1', 'node2']].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=['node1', 'node2'])
df[['node1', 'node2']] = df[['node1', 'node2']].astype(int)
G_full = nx.from_pandas_edgelist(df, source='node1', target='node2')
G_full = nx.convert_node_labels_to_integers(G_full, label_attribute='ncbi_id')
ncbi_to_node = {G_full.nodes[v]['ncbi_id']: v for v in G_full.nodes()}
print(f'Network NET_PH: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges')

# Load disease seeds and targets
seed_path   = 'DATA_ROOT_PH/gene_seeds/DISEASE_PH_ncbi_seeds.json'
target_path = 'DATA_ROOT_PH/gwas_catalog_targets/DISEASE_PH_targets_ncbi_gwas_catalog.json'
with open(seed_path)   as f: raw_seeds   = json.load(f)
with open(target_path) as f: raw_targets = json.load(f)

seeds_full   = [ncbi_to_node[int(g)] for g in raw_seeds   if int(g) in ncbi_to_node]
targets_full = [ncbi_to_node[int(g)] for g in raw_targets if int(g) in ncbi_to_node]
print(f'Disease DISEASE_PH: {len(seeds_full)} seeds, {len(targets_full)} targets')

if not seeds_full or not targets_full:
    print('ERROR: no seeds or targets map into this network — aborting.')
    sys.exit(1)

# Trim if seeds+targets exceed budget (seeds take priority)
combined = list(dict.fromkeys(seeds_full + targets_full))
if len(combined) > MAX_NODES:
    seeds_full   = seeds_full[:MAX_NODES // 2]
    targets_full = targets_full[:MAX_NODES - len(seeds_full)]

# Subsample network preserving disease seeds and targets
if G_full.number_of_nodes() > MAX_NODES:
    rng = np.random.default_rng(SEED)
    G = subsample_nodes(G=G_full, seeds=seeds_full, targets=targets_full,
                        max_nodes=MAX_NODES, radius=2, rng=rng)
    G = nx.convert_node_labels_to_integers(G)
    ncbi_sub = {G.nodes[v].get('ncbi_id'): v for v in G.nodes() if 'ncbi_id' in G.nodes[v]}
    seeds   = [ncbi_sub[G_full.nodes[s]['ncbi_id']] for s in seeds_full
               if G_full.nodes[s]['ncbi_id'] in ncbi_sub]
    targets = [ncbi_sub[G_full.nodes[t]['ncbi_id']] for t in targets_full
               if G_full.nodes[t]['ncbi_id'] in ncbi_sub]
    print(f'Subsampled: {G.number_of_nodes()} nodes, {len(seeds)} seeds, {len(targets)} targets')
else:
    G = G_full
    seeds, targets = seeds_full, targets_full

del G_full

if not seeds or not targets:
    print('ERROR: no seeds or targets in subgraph — aborting.')
    sys.exit(1)

metadata = {
    'type':              'real_ppi',
    'network':           'NET_PH',
    'disease':           'DISEASE_PH',
    'network_id':        'NET_ID_PH',
    'replicate':         SEED,
    'seeds':             seeds,
    'targets':           targets,
    'negative_strategy': 'random',
}

results = run_single_network_analysis(
    G=G,
    network_id='NET_ID_PH',
    network_metadata=metadata,
    output_dir='RESULT_DIR_PH',
    embedding_methods='METHODS_PH'.split(','),
    verbose=True,
    resume=False,
    method_hyperparams=method_hyperparams,
)
print(f'Done: NET_ID_PH  →  RESULT_DIR_PH')
PYEOF

exit $?
BSUBEOF

                sed -i "s|JOB_NAME_PH|${JOB_NAME}|g"       "$JOB_SH"
                sed -i "s|LOG_DIR_PH|${LOG_DIR}|g"           "$JOB_SH"
                sed -i "s|QUEUE_PH|${QUEUE}|g"               "$JOB_SH"
                sed -i "s|WALLTIME_PH|${WALLTIME}|g"         "$JOB_SH"
                sed -i "s|MEMORY_PH|${MEMORY}|g"             "$JOB_SH"
                sed -i "s|PROJECT_DIR_PH|${PROJECT_DIR}|g"   "$JOB_SH"
                sed -i "s|PYTHON_ENV_PH|${PYTHON_ENV}|g"     "$JOB_SH"
                sed -i "s|MAX_NODES_PH|${MAX_NODES}|g"       "$JOB_SH"
                sed -i "s|SEED_PH|${SEED}|g"                 "$JOB_SH"
                sed -i "s|HPARAM_JSON_PH|${HPARAM_JSON}|g"   "$JOB_SH"
                sed -i "s|NET_PATH_PH|${NET_PATH}|g"         "$JOB_SH"
                sed -i "s|DATA_ROOT_PH|${DATA_ROOT}|g"       "$JOB_SH"
                sed -i "s|NET_ID_PH|${NET_ID}|g"             "$JOB_SH"
                sed -i "s|NET_PH|${NET}|g"                   "$JOB_SH"
                sed -i "s|DISEASE_PH|${DISEASE}|g"           "$JOB_SH"
                sed -i "s|RESULT_DIR_PH|${RESULT_DIR}|g"     "$JOB_SH"
                sed -i "s|METHODS_PH|${METHODS}|g"           "$JOB_SH"
                chmod +x "$JOB_SH"

                JOB_ID=$( submit_bsub "$JOB_SH" "$TUNE_DEP" )
                RANK_JOB_NAMES+=("$JOB_NAME")
                [ "$DRY_RUN" = false ] && echo "    Submitted ${JOB_ID}: ${JOB_NAME}"
            done
        done
    done
fi

# ── 2B: Classification (5 nets × N_REPS) ─────────────────────────────────────
if [ "$SKIP_CLASSIFICATION" = false ]; then
    echo ""
    echo "2B: Classification (${ANALYSIS_CLASS} jobs)"

    for NET in "${NETWORKS[@]}"; do
        NET_PATH="${NET_PATHS[$NET]}"
        HPARAM_JSON="${HPARAM_DIR}/${NET}/best_hyperparams.json"

        TUNE_DEP=""
        if [ "$SKIP_TUNING" = false ]; then
            TID="${TUNE_JOB_ID[$NET]:-}"
            [ -n "$TID" ] && [ "$TID" != "DRYRUN" ] && TUNE_DEP="ended(${TID})"
        fi

        echo "  Network: ${NET}"
        for REP in $( seq -w 0 $(( N_REPS - 1 )) ); do
            SEED=$(( 10#$REP ))
            NET_ID="${NET}_class_rep${REP}"
            JOB_NAME="ppi_class_${NET}_r${REP}"
            JOB_SH="${LOG_DIR}/${JOB_NAME}.sh"
            RESULT_DIR="${RESULTS_DIR}/${NET_ID}"
            mkdir -p "$RESULT_DIR"

            cat > "$JOB_SH" << 'BSUBEOF'
#!/bin/bash
#BSUB -J JOB_NAME_PH
#BSUB -o LOG_DIR_PH/JOB_NAME_PH.out
#BSUB -e LOG_DIR_PH/JOB_NAME_PH.err
#BSUB -q QUEUE_PH
#BSUB -W WALLTIME_PH
#BSUB -M MEMORY_PHGB
#BSUB -R "rusage[mem=MEMORY_PHGB]"

source PROJECT_DIR_PH/PYTHON_ENV_PH
cd PROJECT_DIR_PH

python - << 'PYEOF'
import sys, json, warnings
import numpy as np
import networkx as nx
import pandas as pd
sys.path.insert(0, 'PROJECT_DIR_PH/src')

from quvine.comprehensive_embedding_analysis import run_single_network_analysis

MAX_NODES = MAX_NODES_PH
SEED      = SEED_PH
warnings.filterwarnings("ignore")

# Load per-network hyperparameters
hparam_path = 'HPARAM_JSON_PH'
try:
    with open(hparam_path) as f:
        hp = json.load(f)
    method_hyperparams = hp['best_params'].get('real_NET_PH', {})
    method_hyperparams.pop('_scores', None)
    print(f'Hyperparameters loaded for real_NET_PH: {list(method_hyperparams.keys())}')
except FileNotFoundError:
    method_hyperparams = {}
    print(f'WARNING: no hparam file at {hparam_path} — using defaults')

if 'quvine_walks' not in method_hyperparams:
    method_hyperparams['quvine_walks'] = {'max_nodes': 250, 'max_edges': 5000}

# Load and subsample network (no disease seeds)
df = pd.read_csv('NET_PATH_PH', usecols=[0, 1], dtype=str)
df.columns = ['node1', 'node2']
df[['node1', 'node2']] = df[['node1', 'node2']].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=['node1', 'node2'])
df[['node1', 'node2']] = df[['node1', 'node2']].astype(int)
G_full = nx.from_pandas_edgelist(df, source='node1', target='node2')
G_full = nx.convert_node_labels_to_integers(G_full, label_attribute='ncbi_id')
print(f'Network NET_PH: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges')

if G_full.number_of_nodes() > MAX_NODES:
    rng  = np.random.default_rng(SEED)
    keep = rng.choice(list(G_full.nodes()), size=MAX_NODES, replace=False).tolist()
    G = G_full.subgraph(keep).copy()
    G = nx.convert_node_labels_to_integers(G)
    print(f'Subsampled: {G.number_of_nodes()} nodes')
else:
    G = G_full

del G_full

metadata = {
    'type':              'real_ppi',
    'network':           'NET_PH',
    'network_id':        'NET_ID_PH',
    'replicate':         SEED,
    'negative_strategy': 'random',
}

results = run_single_network_analysis(
    G=G,
    network_id='NET_ID_PH',
    network_metadata=metadata,
    output_dir='RESULT_DIR_PH',
    embedding_methods='METHODS_PH'.split(','),
    verbose=True,
    resume=False,
    method_hyperparams=method_hyperparams,
)
print(f'Done: NET_ID_PH  →  RESULT_DIR_PH')
PYEOF

exit $?
BSUBEOF

            sed -i "s|JOB_NAME_PH|${JOB_NAME}|g"       "$JOB_SH"
            sed -i "s|LOG_DIR_PH|${LOG_DIR}|g"           "$JOB_SH"
            sed -i "s|QUEUE_PH|${QUEUE}|g"               "$JOB_SH"
            sed -i "s|WALLTIME_PH|${WALLTIME}|g"         "$JOB_SH"
            sed -i "s|MEMORY_PH|${MEMORY}|g"             "$JOB_SH"
            sed -i "s|PROJECT_DIR_PH|${PROJECT_DIR}|g"   "$JOB_SH"
            sed -i "s|PYTHON_ENV_PH|${PYTHON_ENV}|g"     "$JOB_SH"
            sed -i "s|MAX_NODES_PH|${MAX_NODES}|g"       "$JOB_SH"
            sed -i "s|SEED_PH|${SEED}|g"                 "$JOB_SH"
            sed -i "s|HPARAM_JSON_PH|${HPARAM_JSON}|g"   "$JOB_SH"
            sed -i "s|NET_PATH_PH|${NET_PATH}|g"         "$JOB_SH"
            sed -i "s|NET_ID_PH|${NET_ID}|g"             "$JOB_SH"
            sed -i "s|NET_PH|${NET}|g"                   "$JOB_SH"
            sed -i "s|RESULT_DIR_PH|${RESULT_DIR}|g"     "$JOB_SH"
            sed -i "s|METHODS_PH|${METHODS}|g"           "$JOB_SH"
            chmod +x "$JOB_SH"

            JOB_ID=$( submit_bsub "$JOB_SH" "$TUNE_DEP" )
            CLASS_JOB_NAMES+=("$JOB_NAME")
            [ "$DRY_RUN" = false ] && echo "    Submitted ${JOB_ID}: ${JOB_NAME}"
        done
    done
fi

# ── 2C: Link Prediction (5 nets × N_REPS) ────────────────────────────────────
if [ "$SKIP_LINK_PREDICTION" = false ]; then
    echo ""
    echo "2C: Link Prediction (${ANALYSIS_LINK} jobs)"

    for NET in "${NETWORKS[@]}"; do
        NET_PATH="${NET_PATHS[$NET]}"
        HPARAM_JSON="${HPARAM_DIR}/${NET}/best_hyperparams.json"

        TUNE_DEP=""
        if [ "$SKIP_TUNING" = false ]; then
            TID="${TUNE_JOB_ID[$NET]:-}"
            [ -n "$TID" ] && [ "$TID" != "DRYRUN" ] && TUNE_DEP="ended(${TID})"
        fi

        echo "  Network: ${NET}"
        for REP in $( seq -w 0 $(( N_REPS - 1 )) ); do
            SEED=$(( 10#$REP ))
            NET_ID="${NET}_link_rep${REP}"
            JOB_NAME="ppi_link_${NET}_r${REP}"
            JOB_SH="${LOG_DIR}/${JOB_NAME}.sh"
            RESULT_DIR="${RESULTS_DIR}/${NET_ID}"
            mkdir -p "$RESULT_DIR"

            cat > "$JOB_SH" << 'BSUBEOF'
#!/bin/bash
#BSUB -J JOB_NAME_PH
#BSUB -o LOG_DIR_PH/JOB_NAME_PH.out
#BSUB -e LOG_DIR_PH/JOB_NAME_PH.err
#BSUB -q QUEUE_PH
#BSUB -W WALLTIME_PH
#BSUB -M MEMORY_PHGB
#BSUB -R "rusage[mem=MEMORY_PHGB]"

source PROJECT_DIR_PH/PYTHON_ENV_PH
cd PROJECT_DIR_PH

python - << 'PYEOF'
import sys, json, warnings
import numpy as np
import networkx as nx
import pandas as pd
sys.path.insert(0, 'PROJECT_DIR_PH/src')

from quvine.comprehensive_embedding_analysis import run_single_network_analysis

MAX_NODES = MAX_NODES_PH
SEED      = SEED_PH
warnings.filterwarnings("ignore")

# Load per-network hyperparameters
hparam_path = 'HPARAM_JSON_PH'
try:
    with open(hparam_path) as f:
        hp = json.load(f)
    method_hyperparams = hp['best_params'].get('real_NET_PH', {})
    method_hyperparams.pop('_scores', None)
    print(f'Hyperparameters loaded for real_NET_PH: {list(method_hyperparams.keys())}')
except FileNotFoundError:
    method_hyperparams = {}
    print(f'WARNING: no hparam file at {hparam_path} — using defaults')

if 'quvine_walks' not in method_hyperparams:
    method_hyperparams['quvine_walks'] = {'max_nodes': 250, 'max_edges': 5000}

# Load and subsample network (no disease seeds)
df = pd.read_csv('NET_PATH_PH', usecols=[0, 1], dtype=str)
df.columns = ['node1', 'node2']
df[['node1', 'node2']] = df[['node1', 'node2']].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=['node1', 'node2'])
df[['node1', 'node2']] = df[['node1', 'node2']].astype(int)
G_full = nx.from_pandas_edgelist(df, source='node1', target='node2')
G_full = nx.convert_node_labels_to_integers(G_full, label_attribute='ncbi_id')
print(f'Network NET_PH: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges')

if G_full.number_of_nodes() > MAX_NODES:
    rng  = np.random.default_rng(SEED)
    keep = rng.choice(list(G_full.nodes()), size=MAX_NODES, replace=False).tolist()
    G = G_full.subgraph(keep).copy()
    G = nx.convert_node_labels_to_integers(G)
    print(f'Subsampled: {G.number_of_nodes()} nodes')
else:
    G = G_full

del G_full

metadata = {
    'type':              'real_ppi',
    'network':           'NET_PH',
    'network_id':        'NET_ID_PH',
    'replicate':         SEED,
    'negative_strategy': 'random',
}

results = run_single_network_analysis(
    G=G,
    network_id='NET_ID_PH',
    network_metadata=metadata,
    output_dir='RESULT_DIR_PH',
    embedding_methods='METHODS_PH'.split(','),
    verbose=True,
    resume=False,
    method_hyperparams=method_hyperparams,
)
print(f'Done: NET_ID_PH  →  RESULT_DIR_PH')
PYEOF

exit $?
BSUBEOF

            sed -i "s|JOB_NAME_PH|${JOB_NAME}|g"       "$JOB_SH"
            sed -i "s|LOG_DIR_PH|${LOG_DIR}|g"           "$JOB_SH"
            sed -i "s|QUEUE_PH|${QUEUE}|g"               "$JOB_SH"
            sed -i "s|WALLTIME_PH|${WALLTIME}|g"         "$JOB_SH"
            sed -i "s|MEMORY_PH|${MEMORY}|g"             "$JOB_SH"
            sed -i "s|PROJECT_DIR_PH|${PROJECT_DIR}|g"   "$JOB_SH"
            sed -i "s|PYTHON_ENV_PH|${PYTHON_ENV}|g"     "$JOB_SH"
            sed -i "s|MAX_NODES_PH|${MAX_NODES}|g"       "$JOB_SH"
            sed -i "s|SEED_PH|${SEED}|g"                 "$JOB_SH"
            sed -i "s|HPARAM_JSON_PH|${HPARAM_JSON}|g"   "$JOB_SH"
            sed -i "s|NET_PATH_PH|${NET_PATH}|g"         "$JOB_SH"
            sed -i "s|NET_ID_PH|${NET_ID}|g"             "$JOB_SH"
            sed -i "s|NET_PH|${NET}|g"                   "$JOB_SH"
            sed -i "s|RESULT_DIR_PH|${RESULT_DIR}|g"     "$JOB_SH"
            sed -i "s|METHODS_PH|${METHODS}|g"           "$JOB_SH"
            chmod +x "$JOB_SH"

            JOB_ID=$( submit_bsub "$JOB_SH" "$TUNE_DEP" )
            LINK_JOB_NAMES+=("$JOB_NAME")
            [ "$DRY_RUN" = false ] && echo "    Submitted ${JOB_ID}: ${JOB_NAME}"
        done
    done
fi

RANK_COUNT=${#RANK_JOB_NAMES[@]}
CLASS_COUNT=${#CLASS_JOB_NAMES[@]}
LINK_COUNT=${#LINK_JOB_NAMES[@]}
echo ""
echo "Phase 2: submitted ${RANK_COUNT} ranking + ${CLASS_COUNT} classification + ${LINK_COUNT} LP = $(( RANK_COUNT + CLASS_COUNT + LINK_COUNT )) total."

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "PHASE 3: Aggregation"
echo "===================="

# Build per-network wildcard dependencies using job-name prefixes.
# ppi_rank_NET_* covers all ranking reps for that network.
# ppi_class_NET_* and ppi_link_NET_* cover classification and LP reps.

declare -A PER_NET_AGG_JOB_ID

# ── 3A: Per-network aggregation (one job per network) ─────────────────────────
for NET in "${NETWORKS[@]}"; do
    # Collect which task types were actually submitted for this network
    NET_DEP_PARTS=()
    [ "$SKIP_RANKING" = false ]         && NET_DEP_PARTS+=("ended(ppi_rank_${NET}_*)")
    [ "$SKIP_CLASSIFICATION" = false ]  && NET_DEP_PARTS+=("ended(ppi_class_${NET}_*)")
    [ "$SKIP_LINK_PREDICTION" = false ] && NET_DEP_PARTS+=("ended(ppi_link_${NET}_*)")

    NET_ANA_DEP=""
    for _part in "${NET_DEP_PARTS[@]}"; do
        if [ -z "$NET_ANA_DEP" ]; then
            NET_ANA_DEP="$_part"
        else
            NET_ANA_DEP="${NET_ANA_DEP} && ${_part}"
        fi
    done

    # Skip if nothing was submitted for this network
    SKIP_NET_AGG=true
    [ "$SKIP_RANKING" = false ]         && [ $RANK_COUNT -gt 0 ] && SKIP_NET_AGG=false
    [ "$SKIP_CLASSIFICATION" = false ]  && [ $CLASS_COUNT -gt 0 ] && SKIP_NET_AGG=false
    [ "$SKIP_LINK_PREDICTION" = false ] && [ $LINK_COUNT -gt 0 ] && SKIP_NET_AGG=false

    if [ "$SKIP_NET_AGG" = true ] && [ "$DRY_RUN" = false ]; then
        continue
    fi

    JOB_NAME="ppi_agg_${NET}"
    JOB_SH="${LOG_DIR}/${JOB_NAME}.sh"
    NET_AGG_DIR="${AGG_DIR}/${NET}"
    mkdir -p "$NET_AGG_DIR"

    cat > "$JOB_SH" << BSUBEOF
#!/bin/bash
#BSUB -J ${JOB_NAME}
#BSUB -o ${LOG_DIR}/${JOB_NAME}.out
#BSUB -e ${LOG_DIR}/${JOB_NAME}.err
#BSUB -q ${QUEUE}
#BSUB -W 02:00
#BSUB -M 16GB
#BSUB -R "rusage[mem=16GB]"

source ${PROJECT_DIR}/${PYTHON_ENV}
cd ${PROJECT_DIR}

echo "========================================"
echo " Per-network aggregation: ${NET}"
echo " Results dir : ${RESULTS_DIR}"
echo " Output dir  : ${NET_AGG_DIR}"
echo " Started     : \$(date)"
echo "========================================"

python - << 'PYEOF'
import sys, pandas as pd
from pathlib import Path

results_dir = Path('${RESULTS_DIR}')
output_dir  = Path('${NET_AGG_DIR}')
output_dir.mkdir(parents=True, exist_ok=True)
net = '${NET}'

for task in ('ranking', 'classification', 'link_prediction'):
    files = sorted(results_dir.glob(f'{net}_*/{net}_*_{task}_results.csv'))
    if not files:
        print(f'[{net}] No {task} result files found — skipping')
        continue
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f'WARNING: could not read {f}: {e}')
    if not dfs:
        continue
    combined = pd.concat(dfs, ignore_index=True)
    out = output_dir / f'{net}_{task}_aggregated.csv'
    combined.to_csv(out, index=False)
    print(f'[{net}] {task}: {len(files)} files, {len(combined)} rows -> {out}')

print(f'Per-network aggregation complete for {net}')
PYEOF

echo "========================================"
echo " Finished: ${NET}  at \$(date)"
echo "========================================"
exit \$?
BSUBEOF

    chmod +x "$JOB_SH"

    JOB_ID=$( submit_bsub "$JOB_SH" "$NET_ANA_DEP" )
    PER_NET_AGG_JOB_ID["$NET"]="${JOB_ID}"
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] ${JOB_NAME}  dep: ${NET_ANA_DEP:-none}"
    else
        echo "  Submitted ${JOB_ID}: ${JOB_NAME}  dep: ${NET_ANA_DEP:-none}"
    fi
done

# ── 3B: Comprehensive aggregation (waits for all per-network agg jobs) ─────────
COMP_DEP_PARTS=()
for NET in "${NETWORKS[@]}"; do
    TID="${PER_NET_AGG_JOB_ID[$NET]:-}"
    [ -n "$TID" ] && [ "$TID" != "DRYRUN" ] && COMP_DEP_PARTS+=("ended(${TID})")
done

COMP_DEP=""
for _part in "${COMP_DEP_PARTS[@]}"; do
    if [ -z "$COMP_DEP" ]; then
        COMP_DEP="$_part"
    else
        COMP_DEP="${COMP_DEP} && ${_part}"
    fi
done

JOB_NAME="ppi_agg_all"
JOB_SH="${LOG_DIR}/${JOB_NAME}.sh"

cat > "$JOB_SH" << BSUBEOF
#!/bin/bash
#BSUB -J ${JOB_NAME}
#BSUB -o ${LOG_DIR}/${JOB_NAME}.out
#BSUB -e ${LOG_DIR}/${JOB_NAME}.err
#BSUB -q ${QUEUE}
#BSUB -W 02:00
#BSUB -M 32GB
#BSUB -R "rusage[mem=32GB]"

source ${PROJECT_DIR}/${PYTHON_ENV}
cd ${PROJECT_DIR}

echo "========================================"
echo " Comprehensive aggregation: all networks"
echo " Agg dir : ${AGG_DIR}"
echo " Started : \$(date)"
echo "========================================"

python - << 'PYEOF'
import sys, pandas as pd
from pathlib import Path

agg_dir    = Path('${AGG_DIR}')
output_dir = Path('${OUTPUT_BASE}')
output_dir.mkdir(parents=True, exist_ok=True)
networks = ['${NETWORKS[0]}', '${NETWORKS[1]}', '${NETWORKS[2]}', '${NETWORKS[3]}', '${NETWORKS[4]}']

for task in ('ranking', 'classification', 'link_prediction'):
    all_dfs = []
    for net in networks:
        f = agg_dir / net / f'{net}_{task}_aggregated.csv'
        if f.exists():
            try:
                df = pd.read_csv(f)
                df['network'] = net
                all_dfs.append(df)
                print(f'  [{net}] {task}: {len(df)} rows')
            except Exception as e:
                print(f'WARNING: could not read {f}: {e}')
        else:
            print(f'[SKIP] {f} not found')
    if not all_dfs:
        print(f'No {task} data — skipping')
        continue
    combined = pd.concat(all_dfs, ignore_index=True)
    out = output_dir / f'ppi_comprehensive_{task}.csv'
    combined.to_csv(out, index=False)
    print(f'{task}: {len(combined)} total rows -> {out}')

# Also run the existing aggregate_ppi_comprehensive.py for ranking
# (it handles the disease × network directory pattern)
import subprocess, sys
cmd = [
    sys.executable, 'scripts/aggregate_ppi_comprehensive.py',
    '--results-dir', '${RESULTS_DIR}',
    '--output-dir',  '${OUTPUT_BASE}',
    '--networks',    'BioPlex3 HumanNet STRING ProteomeHD PCNet',
    '--diseases',    'asthma autism schizophrenia',
]
result = subprocess.run(cmd, capture_output=False)
if result.returncode != 0:
    print(f'WARNING: aggregate_ppi_comprehensive.py exited {result.returncode}')

print('Comprehensive aggregation complete.')
PYEOF

echo "========================================"
echo " Finished: comprehensive aggregation  at \$(date)"
echo "========================================"
exit \$?
BSUBEOF

chmod +x "$JOB_SH"

COMP_JOB_ID=$( submit_bsub "$JOB_SH" "$COMP_DEP" )
if [ "$DRY_RUN" = true ]; then
    echo "  [DRY RUN] ${JOB_NAME}  dep: ${COMP_DEP:-none}"
else
    echo "  Submitted ${COMP_JOB_ID}: ${JOB_NAME}  dep: ${COMP_DEP:-none}"
fi

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "============================================================"
echo " Submission Complete"
echo "  Phase 1 (tuning)      : ${#TUNE_JOB_ID[@]} jobs"
echo "  Phase 2 (analysis)    : $(( RANK_COUNT + CLASS_COUNT + LINK_COUNT )) jobs"
echo "    Ranking             : ${RANK_COUNT}"
echo "    Classification      : ${CLASS_COUNT}"
echo "    Link Prediction     : ${LINK_COUNT}"
echo "  Phase 3 (aggregation) : $((N_NETS + 1)) jobs"
echo ""
echo "  Output base  : ${OUTPUT_BASE}"
echo "  Results      : ${RESULTS_DIR}"
echo "  Hparams      : ${HPARAM_DIR}"
echo "  Aggregated   : ${AGG_DIR}"
echo "  Logs         : ${LOG_DIR}"
echo ""
echo "  Monitor      : bjobs -u \$USER"
echo "  Check logs   : ls ${LOG_DIR}/*.out"
echo "============================================================"
