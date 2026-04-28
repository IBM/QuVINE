#!/bin/bash
################################################################################
# PPI Network Hyperparameter Tuning Job Submission — LSF
#
# DEFAULT (Parallel Mode):
#   Submits one job per method × network × disease combination.
#   Each job tunes hyperparameters for all 3 tasks (node_classification,
#   link_prediction, node_ranking) for a single method on a single network-disease pair.
#   Total jobs: N_METHODS × N_NETWORKS × N_DISEASES (default: 12 × 5 × 3 = 180 jobs)
#
# SERIAL MODE (--serial flag):
#   Submits one job per network × disease, processing all methods sequentially.
#   Total jobs: N_NETWORKS × N_DISEASES (default: 5 × 3 = 15 jobs)
#
# Networks: STRING, BioPlex3, HumanNet, PCNet, ProteomeHD (5 networks)
# Diseases: asthma, autism, schizophrenia (3 diseases)
# Methods: 12 unified embedding methods (3 quantum walks + 9 baselines)
#
# Usage:
#   # Parallel mode (default) - 180 jobs (12 methods × 5 networks × 3 diseases)
#   bash scripts/submit_ppi_tuning_jobs.sh
#
#   # Serial mode - 15 jobs (5 networks × 3 diseases, all methods sequential)
#   bash scripts/submit_ppi_tuning_jobs.sh --serial
#
#   # Tune only on STRING network - 36 jobs (12 methods × 3 diseases)
#   bash scripts/submit_ppi_tuning_jobs.sh --networks STRING
#
#   # Tune on specific network-disease pairs - 12 jobs
#   bash scripts/submit_ppi_tuning_jobs.sh --networks STRING --diseases asthma
#
#   # With options
#   bash scripts/submit_ppi_tuning_jobs.sh --queue normal --walltime 72:00 --memory 32
#   bash scripts/submit_ppi_tuning_jobs.sh --serial --n-replicates 5 --dry-run
#
# Options:
#   --networks NET1,NET2,...  Networks to tune (default: all 5)
#   --diseases DIS1,DIS2,...  Diseases to tune (default: all 3)
#   --methods MET1,MET2,...   Methods to tune (default: all 12)
#   --serial                  Run all methods in one job per network-disease (default: parallel)
#   --queue QUEUE             LSF queue name (default: normal)
#   --walltime TIME           Wall time limit (default: 72:00)
#   --memory MEM              Memory in GB (default: 32)
#   --n-replicates N          Number of subsampling replicates (default: 3)
#   --n-trials N              Number of Optuna trials per method (default: from config)
#   --config FILE             Config file path (default: scripts/unified_tuning_config.yaml)
#   --dry-run                 Show what would be submitted without submitting
#
################################################################################

set -e

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
QUEUE="normal"
WALLTIME="72:00"
MEMORY="32"
N_REPLICATES="3"
N_TRIALS=""  # Empty means use config defaults
DRY_RUN=false
SERIAL_MODE=false  # Default: parallel (one job per method × network × disease)
PYTHON_ENV="../Python-3.12.2/venv_quvine/bin/activate"
CONFIG_FILE="scripts/unified_tuning_config.yaml"
USE_GPU=false       # Request GPU nodes for DNN methods
GPU_MEMORY="32"     # Memory for GPU DNN jobs (GPU handles compute)
DNN_CPU_MEMORY="48" # Memory for CPU DNN jobs (forces node spreading)
DNN_NCORES="4"      # CPU cores for DNN jobs (enables multithreading + spreading)

# DNN methods that benefit from GPU and/or separate nodes
DNN_METHODS=("gat_baseline" "graphgps_baseline" "appnp" "graphsage" "baseline_gcnmf")

# Methods to tune (unified 12-method configuration)
METHODS=(
    "quvine_rwr"
    "quvine_ctqw"
    "quvine_dtqw"
    "baseline_filter_heat"
    "baseline_filter_poly"
    "baseline_gcnmf"
    "gat_baseline"
    "graphgps_baseline"
    "node2vec"
    "netmf"
    "graphsage"
    "appnp"
)

# Networks to tune on (5 PPI networks)
NETWORKS=(
    "STRING"
    "BioPlex3"
    "HumanNet"
    "PCNet"
    "ProteomeHD"
)

# Diseases to tune on (3 diseases)
DISEASES=(
    "asthma"
    "autism"
    "schizophrenia"
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --queue)      QUEUE="$2";      shift 2 ;;
        --walltime)   WALLTIME="$2";   shift 2 ;;
        --memory)     MEMORY="$2";     shift 2 ;;
        --n-replicates) N_REPLICATES="$2"; shift 2 ;;
        --n-trials)   N_TRIALS="$2";   shift 2 ;;
        --python-env) PYTHON_ENV="$2"; shift 2 ;;
        --config)     CONFIG_FILE="$2"; shift 2 ;;
        --networks)
            IFS=',' read -ra NETWORKS <<< "$2"
            shift 2 ;;
        --diseases)
            IFS=',' read -ra DISEASES <<< "$2"
            shift 2 ;;
        --methods)
            IFS=',' read -ra METHODS <<< "$2"
            shift 2 ;;
        --dry-run)    DRY_RUN=true;    shift ;;
        --serial)     SERIAL_MODE=true; shift ;;
        --use-gpu)    USE_GPU=true;     shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--queue Q] [--walltime T] [--memory M] [--n-replicates N]"
            echo "          [--networks NET1,NET2,...] [--diseases DIS1,DIS2,...]"
            echo "          [--methods MET1,MET2,...] [--serial] [--use-gpu] [--dry-run]"
            exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
OUTPUT_BASE="/dccstor/boseukb/Q/NetMed/QuVINE/ppi_tuning_by_task"
LOG_DIR="${OUTPUT_BASE}/logs"

mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
if [ "$SERIAL_MODE" = true ]; then
    TOTAL_JOBS=$((${#NETWORKS[@]} * ${#DISEASES[@]}))
    MODE_DESC="Serial (all methods per job)"
else
    TOTAL_JOBS=$((${#METHODS[@]} * ${#NETWORKS[@]} * ${#DISEASES[@]}))
    MODE_DESC="Parallel (one method per job)"
fi

echo "======================================================"
echo " PPI Network Hyperparameter Tuning Job Submission"
echo "======================================================"
echo " Project dir  : $PROJECT_DIR"
echo " Output dir   : $OUTPUT_BASE"
echo " Config file  : $CONFIG_FILE"
echo " Queue        : $QUEUE"
echo " Wall time    : $WALLTIME"
echo " Memory       : ${MEMORY}GB"
echo " Replicates   : ${N_REPLICATES}"
echo " Methods      : ${#METHODS[@]} (${METHODS[*]})"
echo " Networks     : ${#NETWORKS[@]} (${NETWORKS[*]})"
echo " Diseases     : ${#DISEASES[@]} (${DISEASES[*]})"
echo " Mode         : ${MODE_DESC}"
echo " Total jobs   : ${TOTAL_JOBS}"
echo " Dry run      : $DRY_RUN"
echo " Use GPU      : $USE_GPU (for DNN methods: ${DNN_METHODS[*]})"
echo "======================================================"
echo ""
[ "$DRY_RUN" = true ] && echo "DRY RUN MODE — no jobs will be submitted" && echo ""

# ---------------------------------------------------------------------------
# Helper: check if a method is a DNN method
# ---------------------------------------------------------------------------
is_dnn_method() {
    local method="$1"
    for dm in "${DNN_METHODS[@]}"; do
        [ "$method" = "$dm" ] && return 0
    done
    return 1
}

# ---------------------------------------------------------------------------
# Submit jobs
# ---------------------------------------------------------------------------
JOB_IDS=()
JOB_COUNT=0

if [ "$SERIAL_MODE" = true ]; then
    # Serial mode: One job per network × disease, processing all methods
    for NETWORK in "${NETWORKS[@]}"; do
        for DISEASE in "${DISEASES[@]}"; do
            JOB_NAME="ppi_tune_all_${NETWORK}_${DISEASE}"
            JOB_OUT="${LOG_DIR}/${JOB_NAME}.out"
            JOB_ERR="${LOG_DIR}/${JOB_NAME}.err"
            JOB_SH="${LOG_DIR}/${JOB_NAME}.sh"

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

export OMP_NUM_THREADS=\${LSB_DJOB_NUMPROC:-4}
export OPENBLAS_NUM_THREADS=\${LSB_DJOB_NUMPROC:-4}
export MKL_NUM_THREADS=\${LSB_DJOB_NUMPROC:-4}

echo "======================================================"
echo " Tuning: ALL METHODS on ${NETWORK} - ${DISEASE} (SERIAL)"
echo "======================================================"
echo " Start time: \$(date)"
echo " Config: ${CONFIG_FILE}"
echo " Methods: ${METHODS[*]}"
echo " Replicates: ${N_REPLICATES}"
echo "======================================================"
echo ""

python scripts/tune_ppi_by_task.py \\
    --config ${CONFIG_FILE} \\
    --network ${NETWORK} \\
    --disease ${DISEASE} \\
    --n-replicates ${N_REPLICATES} \\
    --output-dir ${OUTPUT_BASE}$([ -n "$N_TRIALS" ] && echo " \\\\" && echo "    --n-trials ${N_TRIALS}")

EXIT_CODE=\$?

echo ""
echo "======================================================"
echo " Tuning complete: ALL METHODS on ${NETWORK} - ${DISEASE}"
echo " Exit code: \$EXIT_CODE"
echo " End time: \$(date)"
echo "======================================================"

exit \$EXIT_CODE
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
else
    # Parallel mode (default): One job per method × network × disease
    for METHOD in "${METHODS[@]}"; do
        for NETWORK in "${NETWORKS[@]}"; do
            for DISEASE in "${DISEASES[@]}"; do
                JOB_NAME="ppi_tune_${METHOD}_${NETWORK}_${DISEASE}"
                JOB_OUT="${LOG_DIR}/${JOB_NAME}.out"
                JOB_ERR="${LOG_DIR}/${JOB_NAME}.err"
                JOB_SH="${LOG_DIR}/${JOB_NAME}.sh"
                RESULT_FILE="${OUTPUT_BASE}/${NETWORK}_${DISEASE}_${METHOD}_tuning_by_task.json"

                # Determine resource allocation based on method type
                if is_dnn_method "$METHOD"; then
                    if [ "$USE_GPU" = true ]; then
                        JOB_MEM="$GPU_MEMORY"
                        JOB_NCORES=1
                        GPU_BSUB_LINES='#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[ngpus_excl_p>0] rusage[ngpus_excl_p=1]"'
                    else
                        JOB_MEM="$DNN_CPU_MEMORY"
                        JOB_NCORES="$DNN_NCORES"
                        GPU_BSUB_LINES='#BSUB -x'  # exclusive node: prevents other jobs sharing this node
                    fi
                    DEVICE_ARG="--device auto"
                else
                    JOB_MEM="$MEMORY"
                    JOB_NCORES=4  # multi-core for OMP_NUM_THREADS / BLAS speedup
                    GPU_BSUB_LINES=""
                    DEVICE_ARG=""
                fi

                cat > "$JOB_SH" << BSUBEOF
#!/bin/bash
#BSUB -J ${JOB_NAME}
#BSUB -o ${JOB_OUT}
#BSUB -e ${JOB_ERR}
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -n ${JOB_NCORES}
#BSUB -M ${JOB_MEM}GB
#BSUB -R "rusage[mem=${JOB_MEM}GB]"
${GPU_BSUB_LINES}

source ${PROJECT_DIR}/${PYTHON_ENV}
cd ${PROJECT_DIR}

export OMP_NUM_THREADS=\${LSB_DJOB_NUMPROC:-4}
export OPENBLAS_NUM_THREADS=\${LSB_DJOB_NUMPROC:-4}
export MKL_NUM_THREADS=\${LSB_DJOB_NUMPROC:-4}

echo "======================================================"
echo " Tuning: ${METHOD} on ${NETWORK} - ${DISEASE}"
echo "======================================================"
echo " Start time: \$(date)"
echo " Config: ${CONFIG_FILE}"
echo " Output: ${RESULT_FILE}"
echo " Replicates: ${N_REPLICATES}"
echo " Device: ${DEVICE_ARG:-cpu}"
echo "======================================================"
echo ""

python scripts/tune_ppi_by_task.py \\
    --config ${CONFIG_FILE} \\
    --network ${NETWORK} \\
    --disease ${DISEASE} \\
    --methods ${METHOD} \\
    --n-replicates ${N_REPLICATES} \\
    --output-dir ${OUTPUT_BASE} \\
    ${DEVICE_ARG}$([ -n "$N_TRIALS" ] && echo " \\\\" && echo "    --n-trials ${N_TRIALS}")

EXIT_CODE=\$?

echo ""
echo "======================================================"
echo " Tuning complete: ${METHOD} on ${NETWORK} - ${DISEASE}"
echo " Exit code: \$EXIT_CODE"
echo " End time: \$(date)"
echo "======================================================"

exit \$EXIT_CODE
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
fi

echo ""
echo "======================================================"
echo " Jobs submitted: $JOB_COUNT / $TOTAL_JOBS"
echo "======================================================"

# ---------------------------------------------------------------------------
# Aggregation job (depends on all tuning jobs)
# ---------------------------------------------------------------------------
AGG_NAME="ppi_tune_aggregate"
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
#BSUB -M 8GB
#BSUB -R "rusage[mem=8GB]"
$([ -n "$DEPENDENCY_STRING" ] && echo "#BSUB -w \"${DEPENDENCY_STRING}\"")

source ${PROJECT_DIR}/${PYTHON_ENV}
cd ${PROJECT_DIR}

echo "======================================================"
echo " Aggregating PPI tuning results"
echo "======================================================"
echo " Start time: \$(date)"
echo " Output dir: ${OUTPUT_BASE}"
echo "======================================================"
echo ""

python - << 'PYEOF'
import json
import os
from pathlib import Path
from collections import defaultdict

output_dir = Path('${OUTPUT_BASE}')
results_by_network_disease = defaultdict(dict)

# All known methods (unified 12-method configuration)
known_methods = [
    'quvine_rwr', 'quvine_ctqw', 'quvine_dtqw',
    'baseline_filter_heat', 'baseline_filter_poly', 'baseline_gcnmf',
    'gat_baseline', 'graphgps_baseline', 'appnp',
    'node2vec', 'netmf', 'graphsage'
]

# Collect all individual method results
for json_file in output_dir.glob('*_tuning_by_task.json'):
    print(f'Loading: {json_file.name}')
    
    # Parse filename: NETWORK_DISEASE_METHOD_tuning_by_task.json
    # Example: BioPlex3_asthma_node2vec_tuning_by_task.json
    parts = json_file.stem.split('_')
    
    # Find where 'tuning' starts
    try:
        tuning_idx = parts.index('tuning')
    except ValueError:
        print(f'  Skipping {json_file.name}: no "tuning" in filename')
        continue
    
    network_disease_method_parts = parts[:tuning_idx]
    
    # Last part before 'tuning' should be method name
    if len(network_disease_method_parts) < 2:
        print(f'  Skipping {json_file.name}: insufficient parts')
        continue
    
    potential_method = network_disease_method_parts[-1]
    
    # Check if last part is a known method
    if potential_method in known_methods:
        # NETWORK_DISEASE_METHOD format
        network_disease = '_'.join(network_disease_method_parts[:-1])
        
        with open(json_file) as f:
            data = json.load(f)
        
        # Merge method data into network_disease results
        results_by_network_disease[network_disease].update(data)
        print(f'  -> {network_disease}: added {len(data)} method(s)')
    else:
        # Might be aggregated file (NETWORK_DISEASE_tuning_by_task.json)
        # Skip to avoid double-counting
        print(f'  Skipping {json_file.name}: appears to be aggregated file')

# Save aggregated results per network-disease pair
print('')
print('Saving aggregated results...')
for network_disease, methods_data in sorted(results_by_network_disease.items()):
    output_file = output_dir / f'{network_disease}_tuning_by_task.json'
    with open(output_file, 'w') as f:
        json.dump(methods_data, f, indent=2)
    print(f'Saved: {output_file}')
    print(f'  Methods: {len(methods_data)}')
    if methods_data:
        sample_method = list(methods_data.keys())[0]
        tasks = list(methods_data[sample_method].keys())
        print(f'  Tasks: {tasks}')

# Save overall summary
summary = {
    'network_disease_pairs': len(results_by_network_disease),
    'total_methods': sum(len(methods) for methods in results_by_network_disease.values()),
    'pairs': sorted(results_by_network_disease.keys()),
    'methods_per_pair': {k: len(v) for k, v in sorted(results_by_network_disease.items())}
}

summary_file = output_dir / 'tuning_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print('')
print('Aggregation complete!')
print(f'Network-disease pairs: {summary["network_disease_pairs"]}')
print(f'Total method-pair combinations: {summary["total_methods"]}')
print(f'Pairs: {summary["pairs"]}')
PYEOF

EXIT_CODE=\$?

echo ""
echo "======================================================"
echo " Aggregation complete"
echo " Exit code: \$EXIT_CODE"
echo " End time: \$(date)"
echo "======================================================"

exit \$EXIT_CODE
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
echo " Results: ${OUTPUT_BASE}/"
echo "======================================================"

