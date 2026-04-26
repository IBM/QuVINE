#!/bin/bash
################################################################################
# Extended Random Graph Generators Experiment
#
# Submits LSF jobs for 5 new synthetic graph families:
#   1. Random Regular / Expander-like
#   2. Heterophilic SBM (disassortative)
#   3. Degree-Corrected SBM
#   4. Grid/Torus Lattice
#   5. Configuration Model (power-law/log-normal)
#
# Structure:
#   - One LSF job per (network_type, n_nodes, replicate)
#   - 30 replicates × 3 node sizes = 90 jobs per network type
#   - 5 network types × 90 jobs = 450 total jobs
#   - Each job calls scripts/run_extended_generator_network.py
#   - One aggregation job with ended() dependency on all analysis jobs
#
# Usage:
#   bash scripts/submit_extended_generators_jobs.sh [options]
#
# Options:
#   --n-replicates NUM   Replicates per (type, size) (default: 30)
#   --n-nodes SIZES      Node sizes (comma-separated)  (default: 500,2000,5000)
#   --output-dir DIR     Output root                   (default: outputs/extended_generators)
#   --queue QUEUE        LSF queue                     (default: normal)
#   --walltime TIME      Wall time per job             (default: 48:00)
#   --memory MEM         Memory in GB                  (default: 4)
#   --methods METHODS    Methods to run                (default: all)
#   --python-env PATH    Path to python binary         (default: ../Python-3.12.2/venv_quvine/bin/python)
#   --resume             Pass --resume to analysis jobs
#   --dry-run            Print scripts, do not submit
#
################################################################################

set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
N_REPLICATES=30
N_NODES_LIST="500,2000,5000"
OUTPUT_DIR="/dccstor/boseukb/Q/NetMed/QuVINE/results/extended_generators/"
QUEUE="normal"
WALLTIME="48:00"
MEMORY="4"
METHODS="all"
PYTHON_ENV="../Python-3.12.2/venv_quvine/bin/python"
HPARAM_FILE="/dccstor/boseukb/Q/NetMed/QuVINE/results/hparam_tuning/best_hyperparams.json"
RESUME=false
DRY_RUN=false

# ── Method presets ─────────────────────────────────────────────────────────────
ALL_METHODS="quvine_fused-walk,quvine_ctqw,quvine_dtqw,quvine_rwr,quvine_heat,quvine_poly,quvine_fused-filt,quvine_hgcnmf,quvine_pgcnmf,quvine_fused-gcnmf,netmf,node2vec,graphsage,baseline_gcnmf,baseline_filter"
QUANTUM_METHODS="quvine_fused-walk,quvine_ctqw,quvine_dtqw,quvine_rwr,quvine_heat,quvine_poly,quvine_fused-filt,quvine_hgcnmf,quvine_pgcnmf,quvine_fused-gcnmf"
CLASSICAL_METHODS="netmf,node2vec,graphsage,baseline_gcnmf,baseline_filter"

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-replicates) N_REPLICATES="$2"; shift 2 ;;
        --n-nodes)      N_NODES_LIST="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --queue)        QUEUE="$2";        shift 2 ;;
        --walltime)     WALLTIME="$2";     shift 2 ;;
        --memory)       MEMORY="$2";       shift 2 ;;
        --methods)      METHODS="$2";      shift 2 ;;
        --python-env)   PYTHON_ENV="$2";   shift 2 ;;
        --hparam-file)  HPARAM_FILE="$2";  shift 2 ;;
        --resume)       RESUME=true;       shift ;;
        --dry-run)      DRY_RUN=true;      shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--n-replicates N] [--n-nodes SIZES] [--output-dir DIR]"
            echo "          [--queue Q] [--walltime T] [--memory M]"
            echo "          [--methods all|quantum|classical|<list>]"
            echo "          [--python-env PATH] [--resume] [--dry-run]"
            exit 1 ;;
    esac
done

case "$METHODS" in
    all)       SELECTED_METHODS="$ALL_METHODS"       ;;
    quantum)   SELECTED_METHODS="$QUANTUM_METHODS"   ;;
    classical) SELECTED_METHODS="$CLASSICAL_METHODS" ;;
    *)         SELECTED_METHODS="$METHODS"            ;;
esac

RESUME_FLAG=""
[ "$RESUME" = true ] && RESUME_FLAG="--resume"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
ANALYSIS_SCRIPT="${PROJECT_DIR}/scripts/run_extended_generator_network.py"

# Derive the activate script from PYTHON_ENV
VENV_ACTIVATE="$( cd "${PROJECT_DIR}" && realpath -m "${PYTHON_ENV%/bin/python}/bin/activate" )"

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/results"

# ── Network type definitions ───────────────────────────────────────────────────
# 5 extended generator families
NETWORK_TYPES=(
    random_regular
    heterophilic_sbm
    degree_corrected_sbm
    grid_torus
    configuration_model
)

# Seed offset bands: 10000 apart per type so replicates never share a seed
NETWORK_SEED_OFFSETS=(
    10000  # random_regular
    20000  # heterophilic_sbm
    30000  # degree_corrected_sbm
    40000  # grid_torus
    50000  # configuration_model
)

# Convert comma-separated node sizes to array
IFS=',' read -ra N_NODES_ARRAY <<< "$N_NODES_LIST"

# ── Banner ─────────────────────────────────────────────────────────────────────
N_TYPES=${#NETWORK_TYPES[@]}
N_SIZES=${#N_NODES_ARRAY[@]}
JOBS_PER_TYPE=$(( N_REPLICATES * N_SIZES ))
TOTAL_JOBS=$(( N_TYPES * JOBS_PER_TYPE ))

echo "============================================================"
echo " Extended Generators Experiment — Job Submission"
echo "============================================================"
echo " Project dir    : $PROJECT_DIR"
echo " Output dir     : $OUTPUT_DIR"
echo " Network types  : $N_TYPES (${NETWORK_TYPES[*]})"
echo " Node sizes     : $N_SIZES (${N_NODES_ARRAY[*]})"
echo " Replicates     : $N_REPLICATES (per type-size combo)"
echo " Jobs per type  : $JOBS_PER_TYPE"
echo " Total jobs     : $TOTAL_JOBS"
echo " LSF queue      : $QUEUE"
echo " Wall time      : $WALLTIME"
echo " Memory         : ${MEMORY}GB"
echo " Python env     : $PYTHON_ENV"
echo " Hparam file    : $HPARAM_FILE"
echo " Methods        : $SELECTED_METHODS"
echo " Resume         : $RESUME"
echo " Dry run        : $DRY_RUN"
echo "============================================================"
echo ""
[ "$DRY_RUN" = true ] && echo "DRY RUN — no jobs will be submitted" && echo ""

ANALYSIS_JOB_IDS=()
JOB_COUNT=0

# ── _submit_network_jobs TYPE_IDX ──────────────────────────────────────────────
_submit_network_jobs() {
    local type_idx="$1"
    local network_type="${NETWORK_TYPES[$type_idx]}"
    local seed_offset="${NETWORK_SEED_OFFSETS[$type_idx]}"

    echo ""
    echo "Network Type: $network_type"

    for n_nodes in "${N_NODES_ARRAY[@]}"; do
        echo "  Node size: $n_nodes"
        
        for rep in $(seq 0 $(( N_REPLICATES - 1 ))); do
            # Unique seed: offset + (size_index * 1000) + rep
            local size_idx=0
            for i in "${!N_NODES_ARRAY[@]}"; do
                if [ "${N_NODES_ARRAY[$i]}" = "$n_nodes" ]; then
                    size_idx=$i
                    break
                fi
            done
            local seed=$(( seed_offset + (size_idx * 1000) + rep ))
            
            local network_id="${network_type}_n${n_nodes}_rep$(printf '%02d' $rep)"
            local job_output_dir="$OUTPUT_DIR/results/$network_id"
            local job_name="ext_${network_type}_${n_nodes}_$(printf '%02d' $rep)"
            local log_file="$OUTPUT_DIR/logs/${job_name}.out"
            local err_file="$OUTPUT_DIR/logs/${job_name}.err"
            local job_script="$OUTPUT_DIR/logs/${job_name}.sh"

            mkdir -p "$job_output_dir"

            # ── Write LSF job script ────────────────────────────────────────────
            cat > "$job_script" << JOBEOF
#!/bin/bash
#BSUB -J ${job_name}
#BSUB -o ${log_file}
#BSUB -e ${err_file}
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -M ${MEMORY}GB
#BSUB -R "rusage[mem=${MEMORY}GB]"

source ${VENV_ACTIVATE}
cd ${PROJECT_DIR}

python ${ANALYSIS_SCRIPT} \
    --network-type ${network_type} \
    --network-id   ${network_id} \
    --output-dir   ${job_output_dir} \
    --methods      ${SELECTED_METHODS} \
    --n-nodes      ${n_nodes} \
    --seed         ${seed} \
    --hparam-file  ${HPARAM_FILE} \
    ${RESUME_FLAG} \
    --verbose

exit \$?
JOBEOF

            chmod +x "$job_script"

            if [ "$DRY_RUN" = true ]; then
                echo "    [DRY RUN] bsub < $job_script"
            else
                local job_id
                job_id=$(bsub < "$job_script" 2>&1 | grep -oP 'Job <\K[0-9]+')
                if [ -n "$job_id" ]; then
                    echo "    Submitted $job_id: $job_name"
                    ANALYSIS_JOB_IDS+=("$job_id")
                    JOB_COUNT=$(( JOB_COUNT + 1 ))
                else
                    echo "    ERROR: submission failed for $job_name"
                fi
            fi
        done
    done
}

# ── Submit all network types ───────────────────────────────────────────────────
for type_idx in "${!NETWORK_TYPES[@]}"; do
    _submit_network_jobs "$type_idx"
done

echo ""
echo "============================================================"
echo " Analysis jobs submitted: $JOB_COUNT"
echo "============================================================"

# ── Build ended() dependency string ───────────────────────────────────────────
DEPENDENCY_STRING=""
for job_id in "${ANALYSIS_JOB_IDS[@]}"; do
    if [ -z "$DEPENDENCY_STRING" ]; then
        DEPENDENCY_STRING="ended($job_id)"
    else
        DEPENDENCY_STRING="$DEPENDENCY_STRING && ended($job_id)"
    fi
done

# ── Aggregation + visualisation job ───────────────────────────────────────────
agg_job_name="ext_aggregate"
agg_job_script="$OUTPUT_DIR/logs/${agg_job_name}.sh"

cat > "$agg_job_script" << AGGEOF
#!/bin/bash
#BSUB -J ${agg_job_name}
#BSUB -o ${OUTPUT_DIR}/logs/${agg_job_name}.out
#BSUB -e ${OUTPUT_DIR}/logs/${agg_job_name}.err
#BSUB -q ${QUEUE}
#BSUB -W 2:00
#BSUB -M 32GB
#BSUB -R "rusage[mem=32GB]"
$([ -n "$DEPENDENCY_STRING" ] && echo "#BSUB -w \"${DEPENDENCY_STRING}\"")

source ${VENV_ACTIVATE}
cd ${PROJECT_DIR}

echo "========================================"
echo " AGGREGATING EXTENDED GENERATORS RESULTS"
echo "========================================"

# Aggregate all per-network CSVs into comprehensive_results.csv
python scripts/collect_hpc_results.py \\
    --results-dir ${OUTPUT_DIR}/results \\
    --viz-dir     ${OUTPUT_DIR}/visualizations \\
    --n-networks  ${TOTAL_JOBS}

echo "========================================"
echo " AGGREGATION COMPLETE"
echo " Results CSV : ${OUTPUT_DIR}/results/comprehensive_results.csv"
echo " Plots       : ${OUTPUT_DIR}/visualizations/"
echo "========================================"

exit \$?
AGGEOF

chmod +x "$agg_job_script"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "  [DRY RUN] bsub < $agg_job_script"
    [ -n "$DEPENDENCY_STRING" ] && echo "  [DRY RUN] Dependencies: $DEPENDENCY_STRING"
elif [ ${#ANALYSIS_JOB_IDS[@]} -gt 0 ]; then
    echo ""
    echo "Submitting aggregation job..."
    agg_id=$(bsub < "$agg_job_script" 2>&1 | grep -oP 'Job <\K[0-9]+')
    if [ -n "$agg_id" ]; then
        echo "  Submitted aggregation job $agg_id  (waits for ${#ANALYSIS_JOB_IDS[@]} jobs via ended())"
    else
        echo "  ERROR: aggregation job submission failed"
    fi
else
    echo "  No analysis jobs submitted — skipping aggregation job."
fi

echo ""
echo "============================================================"
echo " SUBMISSION SUMMARY"
echo " Analysis jobs : $JOB_COUNT / $TOTAL_JOBS"
echo " Output        : $OUTPUT_DIR/results/"
echo " Visualisations: $OUTPUT_DIR/visualizations/"
echo " Monitor with  : bjobs -u \$USER"
echo "============================================================"

# Made with Bob
