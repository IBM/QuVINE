#!/bin/bash
################################################################################
# Submit Remaining Hyperparameter Tuning Jobs (Version 5 - FINAL 20)
# 
# Based on last submitted: tune_graphsage_random_regular (1357696)
#
# Remaining jobs:
# - graphsage: 4 networks (heterophilic_sbm through configuration_model)
# - appnp: 16 networks (all)
# Total: 20 jobs
################################################################################

set -e

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
QUEUE="normal"
WALLTIME="48:00"
N_GRAPHS="10"
PYTHON_ENV="../Python-3.12.2/venv_quvine/bin/activate"
CONFIG_FILE="scripts/unified_tuning_config.yaml"
USE_GPU=false
DNN_CPU_MEMORY="48"
DNN_NCORES="4"

# Remaining graphsage networks
GRAPHSAGE_NETWORKS=(
    "heterophilic_sbm"
    "degree_corrected_sbm"
    "grid_torus"
    "configuration_model"
)

# All appnp networks
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

echo "======================================================"
echo " Submitting FINAL 20 Tuning Jobs (v5)"
echo "======================================================"
echo " graphsage: 4 networks"
echo " appnp: 16 networks"
echo "======================================================"
echo ""

# ---------------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------------
submit_job() {
    local METHOD=$1
    local NET_TYPE=$2
    
    local JOB_NAME="tune_${METHOD}_${NET_TYPE}"
    local JOB_SH="${LOG_DIR}/${JOB_NAME}.sh"

    cat > "$JOB_SH" << JOBEOF
#!/bin/bash
#BSUB -J ${JOB_NAME}
#BSUB -o ${LOG_DIR}/${JOB_NAME}.out
#BSUB -e ${LOG_DIR}/${JOB_NAME}.err
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -n ${DNN_NCORES}
#BSUB -M ${DNN_CPU_MEMORY}GB
#BSUB -R "rusage[mem=${DNN_CPU_MEMORY}GB]"
#BSUB -x

source ${PROJECT_DIR}/${PYTHON_ENV}
cd ${PROJECT_DIR}

export OMP_NUM_THREADS=\${LSB_DJOB_NUMPROC:-4}

echo "Tuning: ${METHOD} on ${NET_TYPE} | Start: \$(date)"

python scripts/tune_by_task_with_config.py \\
    --config ${CONFIG_FILE} \\
    --methods ${METHOD} \\
    --network-type ${NET_TYPE} \\
    --n-graphs ${N_GRAPHS} \\
    --output-dir ${OUTPUT_BASE} \\
    --device auto

echo "Exit: \$? | End: \$(date)"
JOBEOF

    chmod +x "$JOB_SH"
    local JOB_ID=$(bsub < "$JOB_SH" 2>&1 | grep -oP 'Job <\K[0-9]+' || echo "")
    if [ -n "$JOB_ID" ]; then
        echo "  ✓ $JOB_ID: $JOB_NAME"
        echo "$JOB_ID"
    else
        echo "  ✗ FAILED: $JOB_NAME"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Submit jobs
# ---------------------------------------------------------------------------
JOB_IDS=()

echo "Submitting graphsage (4 remaining)..."
for NET_TYPE in "${GRAPHSAGE_NETWORKS[@]}"; do
    JOB_ID=$(submit_job "graphsage" "$NET_TYPE")
    [ $? -eq 0 ] && [ -n "$JOB_ID" ] && JOB_IDS+=("$JOB_ID")
done

echo ""
echo "Submitting appnp (16 networks)..."
for NET_TYPE in "${APPNP_NETWORKS[@]}"; do
    JOB_ID=$(submit_job "appnp" "$NET_TYPE")
    [ $? -eq 0 ] && [ -n "$JOB_ID" ] && JOB_IDS+=("$JOB_ID")
done

echo ""
echo "======================================================"
echo " Submitted: ${#JOB_IDS[@]} / 20 jobs"
echo "======================================================"

# ---------------------------------------------------------------------------
# Aggregation job
# ---------------------------------------------------------------------------
if [ ${#JOB_IDS[@]} -gt 0 ]; then
    echo ""
    echo "Creating aggregation job..."
    
    AGG_SH="${LOG_DIR}/tune_aggregate_final.sh"
    
    DEPS=""
    for JID in "${JOB_IDS[@]}"; do
        [ -z "$DEPS" ] && DEPS="done($JID)" || DEPS="${DEPS} && done($JID)"
    done
    
    cat > "$AGG_SH" << 'AGGEOF'
#!/bin/bash
AGGEOF

    cat >> "$AGG_SH" << AGGEOF
#BSUB -J tune_aggregate_final
#BSUB -o ${LOG_DIR}/tune_aggregate_final.out
#BSUB -e ${LOG_DIR}/tune_aggregate_final.err
#BSUB -q ${QUEUE}
#BSUB -W 1:00
#BSUB -M 8GB
#BSUB -R "rusage[mem=8GB]"
#BSUB -w "${DEPS}"

source ${PROJECT_DIR}/${PYTHON_ENV}
cd ${PROJECT_DIR}

python -c "
import json
from pathlib import Path
from collections import defaultdict

output_dir = Path('${OUTPUT_BASE}')
results_by_network = defaultdict(dict)
known_methods = ['quvine_rwr', 'quvine_ctqw', 'quvine_dtqw', 'baseline_filter_heat', 
                 'baseline_filter_poly', 'baseline_gcnmf', 'gat_baseline', 'graphgps_baseline',
                 'appnp', 'node2vec', 'netmf', 'graphsage']

print('Aggregating results...')
for json_file in output_dir.glob('*_tuning_by_task.json'):
    parts = json_file.stem.split('_')
    try:
        tuning_idx = parts.index('tuning')
        potential_method = parts[tuning_idx-1]
        if potential_method in known_methods:
            network_type = '_'.join(parts[:tuning_idx-1])
            with open(json_file) as f:
                results_by_network[network_type].update(json.load(f))
            print(f'  Loaded: {json_file.name}')
    except: pass

print('\\nSaving aggregated files...')
for network_type, methods_data in sorted(results_by_network.items()):
    output_file = output_dir / f'{network_type}_tuning_by_task.json'
    with open(output_file, 'w') as f:
        json.dump(methods_data, f, indent=2)
    status = '✓' if len(methods_data) == 12 else f'⚠ {len(methods_data)}/12'
    print(f'  {network_type}: {len(methods_data)} methods {status}')

summary = {
    'networks': len(results_by_network), 
    'total_combinations': sum(len(m) for m in results_by_network.values()),
    'expected': 192,
    'methods_per_network': {k: len(v) for k, v in sorted(results_by_network.items())}
}
with open(output_dir / 'tuning_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\\n=== FINAL SUMMARY ===')
print(f'Networks: {summary[\"networks\"]}')
print(f'Total combinations: {summary[\"total_combinations\"]} / {summary[\"expected\"]}')
complete = sum(1 for c in summary['methods_per_network'].values() if c == 12)
print(f'Complete networks: {complete} / {summary[\"networks\"]}')
"
AGGEOF

    chmod +x "$AGG_SH"
    AGG_ID=$(bsub < "$AGG_SH" 2>&1 | grep -oP 'Job <\K[0-9]+' || echo "")
    [ -n "$AGG_ID" ] && echo "  ✓ Aggregation: $AGG_ID" || echo "  ✗ Aggregation failed"
fi

echo ""
echo "======================================================"
echo " DONE - This completes all 192 tuning jobs!"
echo " Monitor: bjobs -u \$USER"
echo "======================================================"

# Made with Bob
