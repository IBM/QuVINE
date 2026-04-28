#!/bin/bash
################################################################################
# Submit Remaining Hyperparameter Tuning Jobs (Version 3 - FINAL)
# 
# This script submits only the jobs that were not completed in previous runs.
# Based on the last submitted job: tune_graphsage_modular_many_communities (1357448)
#
# Remaining jobs:
# - graphsage: 9 networks (core_periphery through configuration_model)
# - appnp: 16 networks (all)
# Total: 25 jobs
################################################################################

set -e

# ---------------------------------------------------------------------------
# Configuration (match original script)
# ---------------------------------------------------------------------------
QUEUE="normal"
WALLTIME="48:00"
MEMORY="32"
N_GRAPHS="10"
PYTHON_ENV="../Python-3.12.2/venv_quvine/bin/activate"
CONFIG_FILE="scripts/unified_tuning_config.yaml"
USE_GPU=false
GPU_MEMORY="32"
DNN_CPU_MEMORY="48"
DNN_NCORES="4"

# Networks remaining for graphsage (starting from core_periphery)
GRAPHSAGE_NETWORKS=(
    "core_periphery"
    "scale_free"
    "powerlaw_cluster"
    "stochastic_block_model"
    "random_regular"
    "heterophilic_sbm"
    "degree_corrected_sbm"
    "grid_torus"
    "configuration_model"
)

# All networks for appnp
APPNP_NETWORKS=(
    "erdos_renyi"
    "watts_strogatz_high_p"
    "watts_strogatz_low_p"
    "random_geometric"
    "modular_strong"
    "modular_medium"
    "modular_many_communities"
    "core_periphery"
    "scale_free"
    "powerlaw_cluster"
    "stochastic_block_model"
    "random_regular"
    "heterophilic_sbm"
    "degree_corrected_sbm"
    "grid_torus"
    "configuration_model"
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
OUTPUT_BASE="/dccstor/boseukb/Q/NetMed/QuVINE/tuning_by_task"
LOG_DIR="${OUTPUT_BASE}/logs"

mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "======================================================"
echo " Submitting FINAL Hyperparameter Tuning Jobs (v3)"
echo "======================================================"
echo " Project dir  : $PROJECT_DIR"
echo " Output dir   : $OUTPUT_BASE"
echo " Config file  : $CONFIG_FILE"
echo " Queue        : $QUEUE"
echo " Wall time    : $WALLTIME"
echo ""
echo " Remaining methods:"
echo "   - graphsage: 9 networks (core_periphery onwards)"
echo "   - appnp: 16 networks (all)"
echo " Total jobs   : 25"
echo "======================================================"
echo ""

# ---------------------------------------------------------------------------
# Submit jobs
# ---------------------------------------------------------------------------
JOB_IDS=()
JOB_COUNT=0

# Submit graphsage remaining jobs
echo "Submitting graphsage remaining jobs..."
for NET_TYPE in "${GRAPHSAGE_NETWORKS[@]}"; do
    METHOD="graphsage"
    JOB_NAME="tune_${METHOD}_${NET_TYPE}"
    JOB_OUT="${LOG_DIR}/${JOB_NAME}.out"
    JOB_ERR="${LOG_DIR}/${JOB_NAME}.err"
    JOB_SH="${LOG_DIR}/${JOB_NAME}.sh"
    RESULT_FILE="${OUTPUT_BASE}/${NET_TYPE}_${METHOD}_tuning_by_task.json"

    # graphsage is a DNN method
    if [ "$USE_GPU" = true ]; then
        JOB_MEM="$GPU_MEMORY"
        JOB_NCORES=1
        GPU_BSUB_LINES='#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[ngpus_excl_p>0] rusage[ngpus_excl_p=1]"'
    else
        JOB_MEM="$DNN_CPU_MEMORY"
        JOB_NCORES="$DNN_NCORES"
        GPU_BSUB_LINES='#BSUB -x'
    fi
    DEVICE_ARG="--device auto"

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
echo " Tuning: ${METHOD} on ${NET_TYPE}"
echo "======================================================"
echo " Start time: \$(date)"
echo " Config: ${CONFIG_FILE}"
echo " Output: ${RESULT_FILE}"
echo " Device: ${DEVICE_ARG}"
echo "======================================================"
echo ""

python scripts/tune_by_task_with_config.py \\
    --config ${CONFIG_FILE} \\
    --methods ${METHOD} \\
    --network-type ${NET_TYPE} \\
    --n-graphs ${N_GRAPHS} \\
    --output-dir ${OUTPUT_BASE} \\
    ${DEVICE_ARG}

EXIT_CODE=\$?

echo ""
echo "======================================================"
echo " Tuning complete: ${METHOD} on ${NET_TYPE}"
echo " Exit code: \$EXIT_CODE"
echo " End time: \$(date)"
echo "======================================================"

exit \$EXIT_CODE
BSUBEOF

    chmod +x "$JOB_SH"

    JOB_ID=$(bsub < "$JOB_SH" 2>&1 | grep -oP 'Job <\K[0-9]+')
    if [ -n "$JOB_ID" ]; then
        echo "  Submitted $JOB_ID: $JOB_NAME"
        JOB_IDS+=("$JOB_ID")
        JOB_COUNT=$((JOB_COUNT + 1))
    else
        echo "  ERROR: failed to submit $JOB_NAME"
    fi
done

# Submit appnp jobs
echo ""
echo "Submitting appnp jobs..."
for NET_TYPE in "${APPNP_NETWORKS[@]}"; do
    METHOD="appnp"
    JOB_NAME="tune_${METHOD}_${NET_TYPE}"
    JOB_OUT="${LOG_DIR}/${JOB_NAME}.out"
    JOB_ERR="${LOG_DIR}/${JOB_NAME}.err"
    JOB_SH="${LOG_DIR}/${JOB_NAME}.sh"
    RESULT_FILE="${OUTPUT_BASE}/${NET_TYPE}_${METHOD}_tuning_by_task.json"

    # appnp is a DNN method
    if [ "$USE_GPU" = true ]; then
        JOB_MEM="$GPU_MEMORY"
        JOB_NCORES=1
        GPU_BSUB_LINES='#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[ngpus_excl_p>0] rusage[ngpus_excl_p=1]"'
    else
        JOB_MEM="$DNN_CPU_MEMORY"
        JOB_NCORES="$DNN_NCORES"
        GPU_BSUB_LINES='#BSUB -x'
    fi
    DEVICE_ARG="--device auto"

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
echo " Tuning: ${METHOD} on ${NET_TYPE}"
echo "======================================================"
echo " Start time: \$(date)"
echo " Config: ${CONFIG_FILE}"
echo " Output: ${RESULT_FILE}"
echo " Device: ${DEVICE_ARG}"
echo "======================================================"
echo ""

python scripts/tune_by_task_with_config.py \\
    --config ${CONFIG_FILE} \\
    --methods ${METHOD} \\
    --network-type ${NET_TYPE} \\
    --n-graphs ${N_GRAPHS} \\
    --output-dir ${OUTPUT_BASE} \\
    ${DEVICE_ARG}

EXIT_CODE=\$?

echo ""
echo "======================================================"
echo " Tuning complete: ${METHOD} on ${NET_TYPE}"
echo " Exit code: \$EXIT_CODE"
echo " End time: \$(date)"
echo "======================================================"

exit \$EXIT_CODE
BSUBEOF

    chmod +x "$JOB_SH"

    JOB_ID=$(bsub < "$JOB_SH" 2>&1 | grep -oP 'Job <\K[0-9]+')
    if [ -n "$JOB_ID" ]; then
        echo "  Submitted $JOB_ID: $JOB_NAME"
        JOB_IDS+=("$JOB_ID")
        JOB_COUNT=$((JOB_COUNT + 1))
    else
        echo "  ERROR: failed to submit $JOB_NAME"
    fi
done

echo ""
echo "======================================================"
echo " Jobs submitted: $JOB_COUNT / 25"
echo "======================================================"

# ---------------------------------------------------------------------------
# Aggregation job (depends on all tuning jobs)
# ---------------------------------------------------------------------------
AGG_NAME="tune_aggregate_final"
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
echo " Aggregating ALL tuning results (FINAL)"
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
results_by_network = defaultdict(dict)

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
    
    # Parse filename: NETWORK_METHOD_tuning_by_task.json
    parts = json_file.stem.split('_')
    
    # Find where 'tuning' starts
    try:
        tuning_idx = parts.index('tuning')
    except ValueError:
        print(f'  Skipping {json_file.name}: no "tuning" in filename')
        continue
    
    network_method_parts = parts[:tuning_idx]
    
    # Last part before 'tuning' should be method name
    if len(network_method_parts) < 2:
        print(f'  Skipping {json_file.name}: insufficient parts')
        continue
    
    potential_method = network_method_parts[-1]
    
    # Check if last part is a known method
    if potential_method in known_methods:
        # NETWORK_METHOD format
        network_type = '_'.join(network_method_parts[:-1])
        
        with open(json_file) as f:
            data = json.load(f)
        
        # Merge method data into network results
        results_by_network[network_type].update(data)
        print(f'  -> {network_type}: added {len(data)} method(s)')
    else:
        # Might be aggregated file (NETWORK_tuning_by_task.json)
        # Skip to avoid double-counting
        print(f'  Skipping {json_file.name}: appears to be aggregated file')

# Save aggregated results per network type
print('')
print('Saving aggregated results...')
for network_type, methods_data in sorted(results_by_network.items()):
    output_file = output_dir / f'{network_type}_tuning_by_task.json'
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
    'network_types': len(results_by_network),
    'total_methods': sum(len(methods) for methods in results_by_network.values()),
    'networks': sorted(results_by_network.keys()),
    'methods_per_network': {k: len(v) for k, v in sorted(results_by_network.items())}
}

summary_file = output_dir / 'tuning_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print('')
print('='*60)
print('FINAL AGGREGATION COMPLETE!')
print('='*60)
print(f'Network types: {summary["network_types"]}')
print(f'Total method-network combinations: {summary["total_methods"]}')
print(f'Expected: 192 (12 methods × 16 networks)')
print(f'Networks: {summary["networks"]}')
print('')
print('Methods per network:')
for net, count in sorted(summary["methods_per_network"].items()):
    status = "✓ COMPLETE" if count == 12 else f"⚠ INCOMPLETE ({count}/12)"
    print(f'  {net}: {count} methods {status}')
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

if [ ${#JOB_IDS[@]} -gt 0 ]; then
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
echo " SUBMISSION COMPLETE - THIS IS THE FINAL BATCH"
echo " Monitor: bjobs -u \$USER"
echo " Results: ${OUTPUT_BASE}/"
echo ""
echo " After completion, you should have:"
echo "   - 12 methods × 16 networks = 192 total combinations"
echo "======================================================"

# Made with Bob
