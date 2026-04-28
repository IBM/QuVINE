#!/bin/bash
################################################################################
# Submit Remaining Hyperparameter Tuning Jobs (Version 6 - FIXED OUTPUT)
# 
# Based on last known submitted: tune_graphsage_random_regular (1357696)
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
echo " Submitting FINAL 20 Tuning Jobs (v6)"
echo "======================================================"
echo " graphsage: 4 networks"
echo " appnp: 16 networks"
echo "======================================================"
echo ""

# ---------------------------------------------------------------------------
# Submit jobs with visible output
# ---------------------------------------------------------------------------
JOB_IDS=()
JOB_COUNT=0

echo "Submitting graphsage (4 remaining)..."
for NET_TYPE in "${GRAPHSAGE_NETWORKS[@]}"; do
    METHOD="graphsage"
    JOB_NAME="tune_${METHOD}_${NET_TYPE}"
    JOB_SH="${LOG_DIR}/${JOB_NAME}.sh"

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
    
    # Submit and show output immediately
    SUBMIT_OUT=$(bsub < "$JOB_SH" 2>&1)
    JOB_ID=$(echo "$SUBMIT_OUT" | grep -oP 'Job <\K[0-9]+')
    
    if [ -n "$JOB_ID" ]; then
        echo "  Submitted $JOB_ID: $JOB_NAME"
        JOB_IDS+=("$JOB_ID")
        JOB_COUNT=$((JOB_COUNT + 1))
    else
        echo "  ERROR: failed to submit $JOB_NAME"
        echo "  Output: $SUBMIT_OUT"
    fi
done

echo ""
echo "Submitting appnp (16 networks)..."
for NET_TYPE in "${APPNP_NETWORKS[@]}"; do
    METHOD="appnp"
    JOB_NAME="tune_${METHOD}_${NET_TYPE}"
    JOB_SH="${LOG_DIR}/${JOB_NAME}.sh"

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
    
    # Submit and show output immediately
    SUBMIT_OUT=$(bsub < "$JOB_SH" 2>&1)
    JOB_ID=$(echo "$SUBMIT_OUT" | grep -oP 'Job <\K[0-9]+')
    
    if [ -n "$JOB_ID" ]; then
        echo "  Submitted $JOB_ID: $JOB_NAME"
        JOB_IDS+=("$JOB_ID")
        JOB_COUNT=$((JOB_COUNT + 1))
    else
        echo "  ERROR: failed to submit $JOB_NAME"
        echo "  Output: $SUBMIT_OUT"
    fi
done

echo ""
echo "======================================================"
echo " Jobs submitted: $JOB_COUNT / 20"
echo "======================================================"

# Save job IDs for reference
echo "${JOB_IDS[@]}" > "${LOG_DIR}/submitted_job_ids_v6.txt"
echo "Job IDs saved to: ${LOG_DIR}/submitted_job_ids_v6.txt"

echo ""
echo "======================================================"
echo " DONE - Monitor with: bjobs -u \$USER"
echo " This completes all 192 tuning jobs!"
echo "======================================================"

# Made with Bob
