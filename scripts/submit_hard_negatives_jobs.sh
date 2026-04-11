#!/bin/bash
################################################################################
# Hard-Negatives Experiment: Where Quantum Wins (and Where It Doesn't)
#
# Structure mirrors submit_hpc_jobs_complete.sh:
#   - One LSF job per (case, replicate)
#   - Each job calls scripts/run_hard_negative_network.py with plain CLI args
#     (no inline Python in bash)
#   - run_hard_negative_network.py loads the .graphml if already present,
#     otherwise generates the network from configs/hard_negatives_cases.json
#   - --resume skips embedding methods already computed in output CSVs
#   - One aggregation job with ended() dependency on every analysis job
#   - collect_hpc_results.py produces all visualisations
#
# EXPERIMENT CASES
# ────────────────
# QUANTUM-WINS:
#   QW1  modular_strong       + same_community
#   QW2  modular_medium       + same_community
#   QW3  scale_free           + hard_2hop
#   QW4  scale_free           + same_community
#   QW5  core_periphery       + hard_2hop
#   QW6  watts_strogatz_low_p + hard_2hop
#
# CLASSICAL-WINS (negative controls):
#   NC1  erdos_renyi          + random
#   NC2  erdos_renyi          + hard_2hop
#   NC3  watts_strogatz_high_p + hard_2hop
#
# Network config : configs/hard_negatives_cases.json
# Analysis script: scripts/run_hard_negative_network.py
#
# Usage:
#   bash scripts/submit_hard_negatives_jobs.sh [options]
#
# Options:
#   --n-replicates NUM   Replicates per case     (default: 10)
#   --n-nodes NUM        Target nodes per graph  (default: 300)
#   --output-dir DIR     Output root             (default: outputs/hard_negatives)
#   --queue QUEUE        LSF queue               (default: normal)
#   --walltime TIME      Wall time per job        (default: 4:00)
#   --memory MEM         Memory in GB            (default: 16)
#   --methods METHODS    Methods to run          (default: all)
#                        Presets: all | quantum | classical
#                        Or comma-separated list
#   --python-env PATH    Path to python binary   (default: ../Python-3.12.2/venv_quvine/bin/python)
#   --resume             Pass --resume to analysis jobs (skip done methods)
#   --dry-run            Print scripts, do not submit
#
################################################################################

set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
N_REPLICATES=30
N_NODES=1000
OUTPUT_DIR="/dccstor/boseukb/Q/NetMed/QuVINE/results/hard_negatives_v4/"
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
        --n-nodes)      N_NODES="$2";      shift 2 ;;
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
            echo "Usage: $0 [--n-replicates N] [--n-nodes N] [--output-dir DIR]"
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
CASES_CONFIG="${PROJECT_DIR}/configs/hard_negatives_cases.json"
ANALYSIS_SCRIPT="${PROJECT_DIR}/scripts/run_hard_negative_network.py"

# Derive the activate script from PYTHON_ENV (strip /bin/python, append /bin/activate)
VENV_ACTIVATE="$( cd "${PROJECT_DIR}" && realpath -m "${PYTHON_ENV%/bin/python}/bin/activate" )"

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/results"

# ── Case definitions (parallel arrays) ────────────────────────────────────────
# Quantum-wins cases (QW): embeddings should capture community/hierarchical/hub structure
# Negative controls   (NC): classical methods should win on random/near-random graphs
# Real networks       (RN): well-known benchmark graphs with ground-truth communities
CASE_NAMES=(
    QW1_modular_strong
    QW2_modular_medium
    QW3_scale_free_2hop
    QW4_scale_free_comm
    QW5_core_periphery
    QW6_ws_low_p
    QW7_modular_many_comm
    QW8_sbm_assortative
    QW9_powerlaw_cluster
    NC1_erdos_renyi_rand
    NC2_erdos_renyi_2hop
    NC3_ws_high_p
    NC4_random_geometric
    RN1_karate_club
    RN2_les_miserables
    RN3_polbooks
)

# Seed-offset bands: 1000 apart per case so replicates never share a seed
CASE_SEED_OFFSETS=(
    1000   # QW1
    2000   # QW2
    3000   # QW3
    4000   # QW4
    5000   # QW5
    6000   # QW6
    10000  # QW7
    11000  # QW8
    12000  # QW9
    7000   # NC1
    8000   # NC2
    9000   # NC3
    13000  # NC4
    14000  # RN1  (real networks use seed only for sampling/eval, not generation)
    15000  # RN2
    16000  # RN3
)

# ── Banner ─────────────────────────────────────────────────────────────────────
N_CASES=${#CASE_NAMES[@]}
TOTAL_JOBS=$(( N_CASES * N_REPLICATES ))

echo "============================================================"
echo " Hard-Negatives Experiment — Job Submission"
echo "============================================================"
echo " Project dir    : $PROJECT_DIR"
echo " Cases config   : $CASES_CONFIG"
echo " Output dir     : $OUTPUT_DIR"
echo " Cases          : $N_CASES"
echo " Replicates     : $N_REPLICATES  (per case)"
echo " Total jobs     : $TOTAL_JOBS"
echo " Nodes / graph  : $N_NODES"
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

# ── _submit_case_jobs CASE_IDX ─────────────────────────────────────────────────
_submit_case_jobs() {
    local case_idx="$1"
    local case_name="${CASE_NAMES[$case_idx]}"
    local seed_offset="${CASE_SEED_OFFSETS[$case_idx]}"

    echo ""
    echo "Case $case_name"

    for rep in $(seq 0 $(( N_REPLICATES - 1 ))); do
        local seed=$(( seed_offset + rep ))
        local network_id="${case_name}_rep$(printf '%02d' $rep)"
        local job_output_dir="$OUTPUT_DIR/results/$network_id"
        local job_name="hn_${network_id}"
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
    --case-name   ${case_name} \
    --network-id  ${network_id} \
    --config      ${CASES_CONFIG} \
    --output-dir  ${job_output_dir} \
    --methods     ${SELECTED_METHODS} \
    --n-nodes     ${N_NODES} \
    --seed        ${seed} \
    --hparam-file ${HPARAM_FILE} \
    ${RESUME_FLAG} \
    --verbose

exit \$?
JOBEOF

        chmod +x "$job_script"

        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY RUN] bsub < $job_script"
        else
            local job_id
            job_id=$(bsub < "$job_script" 2>&1 | grep -oP 'Job <\K[0-9]+')
            if [ -n "$job_id" ]; then
                echo "  Submitted $job_id: $job_name"
                ANALYSIS_JOB_IDS+=("$job_id")
                JOB_COUNT=$(( JOB_COUNT + 1 ))
            else
                echo "  ERROR: submission failed for $job_name"
            fi
        fi
    done
}

# ── Submit all cases ───────────────────────────────────────────────────────────
for case_idx in "${!CASE_NAMES[@]}"; do
    _submit_case_jobs "$case_idx"
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
agg_job_name="hn_aggregate"
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
echo " AGGREGATING HARD-NEGATIVES RESULTS"
echo "========================================"

# collect_hpc_results.py handles everything in one pass:
#   1. Aggregates all per-network CSVs into comprehensive_results.csv
#   2. Per-network-type task boxplots (ranking / classification / link prediction)
#   3. Timing boxplots (per method, per type, scaling curve)
#   4. Ranking@K precision/recall curves (per network type)
#   5. Degree- and distance-matched binned AUC-PR plots + heatmaps
python scripts/collect_hpc_results.py \\
    --results-dir ${OUTPUT_DIR}/results \\
    --viz-dir     ${OUTPUT_DIR}/visualizations \\
    --n-networks  ${N_REPLICATES}

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
