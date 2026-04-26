#!/bin/bash
################################################################################
# Simulated Data Comprehensive Job Submission with Hyperparameter Tuning
#
# Phase 1: Hyperparameter Tuning  (21 jobs)
#   Hard-Negatives  : 16 cases  (one job per case, uses matching synthetic type)
#   Extended Gens   :  5 types  (one job per type, uses closest proxy type)
#
# Phase 2: Main Analysis  (1440 + 450 = 1890 jobs, depends on tuning)
#   Hard-Negatives  : 16 cases × 3 sizes × 30 reps = 1440 jobs
#   Extended Gens   :  5 types × 3 sizes × 30 reps =  450 jobs
#   Analysis jobs depend on their corresponding tuning job.
#
# Phase 3: Aggregation  (2 jobs, depends on all analysis)
#   HN aggregation  : waits for all hn_* analysis jobs
#   EG aggregation  : waits for all eg_* analysis jobs
#
# Total: 1913 jobs
#
# Usage:
#   bash scripts/submit_simulated_data_jobs_with_tuning.sh [options]
#
# Options:
#   --n-replicates N      Replicates per case/type/size   (default: 30)
#   --n-nodes LIST        Comma-separated sizes           (default: 500,2000,5000)
#   --output-dir DIR      Base output directory
#   --queue Q             LSF queue                       (default: normal)
#   --walltime T          Wall time per analysis job      (default: 24:00)
#   --tune-walltime T     Wall time per tuning job        (default: 48:00)
#   --memory M            Memory in GB per job            (default: 8)
#   --methods LIST        Comma-separated method list     (default: all 15)
#   --python-env PATH     Path to venv python binary
#   --n-trials N          Optuna trials per method        (default: 30)
#   --resume              Pass --resume to analysis jobs
#   --dry-run             Print commands, do not submit
#   --skip-tuning         Skip Phase 1, use existing hparam files
#   --skip-hard-negatives Skip HN cases
#   --skip-extended-gens  Skip EG types
################################################################################

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
N_REPLICATES=30
N_NODES_LIST="500,2000,5000"
OUTPUT_DIR="/dccstor/boseukb/Q/NetMed/QuVINE/results/simulated_data"
QUEUE="normal"
WALLTIME="24:00"
TUNE_WALLTIME="48:00"
MEMORY="8"
N_TRIALS=30
METHODS="quvine_rwr,quvine_ctqw,quvine_dtqw,quvine_baseline_heat,quvine_baseline_poly,quvine_rwr_heat,quvine_rwr_poly,quvine_ctqw_heat,quvine_ctqw_poly,gat_baseline,gat_heat,gat_poly,gat_rwr,gat_ctqw,gat_dtqw,gat_rwr_heat,gat_rwr_poly,gat_ctqw_heat,gat_ctqw_poly,gat_dtqw_heat,gat_dtqw_poly,graphgps_baseline,graphgps_heat,graphgps_poly,graphgps_rwr,graphgps_ctqw,graphgps_dtqw,graphgps_rwr_heat,graphgps_rwr_poly,graphgps_ctqw_heat,graphgps_ctqw_poly,graphgps_dtqw_heat,graphgps_dtqw_poly,node2vec,netmf,graphsage,appnp,baseline_filter,baseline_gcnmf"
PYTHON_ENV="../Python-3.12.2/venv_quvine/bin/python"
RESUME=false
DRY_RUN=false
SKIP_TUNING=false
SKIP_HARD_NEGATIVES=false
SKIP_EXTENDED_GENS=false

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-replicates)       N_REPLICATES="$2";       shift 2 ;;
        --n-nodes)            N_NODES_LIST="$2";       shift 2 ;;
        --output-dir)         OUTPUT_DIR="$2";         shift 2 ;;
        --queue)              QUEUE="$2";              shift 2 ;;
        --walltime)           WALLTIME="$2";           shift 2 ;;
        --tune-walltime)      TUNE_WALLTIME="$2";      shift 2 ;;
        --memory)             MEMORY="$2";             shift 2 ;;
        --methods)            METHODS="$2";            shift 2 ;;
        --python-env)         PYTHON_ENV="$2";         shift 2 ;;
        --n-trials)           N_TRIALS="$2";           shift 2 ;;
        --resume)             RESUME=true;             shift ;;
        --dry-run)            DRY_RUN=true;            shift ;;
        --skip-tuning)        SKIP_TUNING=true;        shift ;;
        --skip-hard-negatives) SKIP_HARD_NEGATIVES=true; shift ;;
        --skip-extended-gens) SKIP_EXTENDED_GENS=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
VENV_ACTIVATE="${PROJECT_DIR}/$( dirname "$PYTHON_ENV" )/activate"

HN_RESULTS_DIR="${OUTPUT_DIR}/hard_negatives/results"
EG_RESULTS_DIR="${OUTPUT_DIR}/extended_generators/results"
HPARAM_DIR="${OUTPUT_DIR}/hparam_tuning"
AGG_DIR="${OUTPUT_DIR}/aggregated"
LOG_DIR="${OUTPUT_DIR}/logs"
HN_CONFIG="${PROJECT_DIR}/configs/hard_negatives_cases.json"

mkdir -p "$HN_RESULTS_DIR" "$EG_RESULTS_DIR" "$HPARAM_DIR" "$AGG_DIR" "$LOG_DIR"

IFS=',' read -ra N_NODES_ARRAY <<< "$N_NODES_LIST"

RESUME_FLAG=""
[ "$RESUME" = true ] && RESUME_FLAG="--resume"

# ── Network definitions ────────────────────────────────────────────────────────
# Hard-negatives: case name → seed offset → tuner network type
HN_CASES=(
    "QW1_modular_strong:1000:modular_strong"
    "QW2_modular_medium:2000:modular_medium"
    "QW3_scale_free_2hop:3000:scale_free"
    "QW4_scale_free_comm:4000:scale_free"
    "QW5_core_periphery:5000:core_periphery"
    "QW6_ws_low_p:6000:watts_strogatz_low_p"
    "QW7_modular_many_comm:10000:modular_many_communities"
    "QW8_sbm_assortative:11000:stochastic_block_model"
    "QW9_powerlaw_cluster:12000:powerlaw_cluster"
    "NC1_erdos_renyi_rand:7000:erdos_renyi"
    "NC2_erdos_renyi_2hop:8000:erdos_renyi"
    "NC3_ws_high_p:9000:watts_strogatz_high_p"
    "NC4_random_geometric:13000:random_geometric"
    "RN1_karate_club:14000:real_karate"
    "RN2_les_miserables:15000:real_lesmis"
    "RN3_polbooks:16000:real_polbooks"
)

# Extended generators: type → seed offset → tuner proxy type
EG_CASES=(
    "random_regular:100000:erdos_renyi"
    "heterophilic_sbm:200000:stochastic_block_model"
    "degree_corrected_sbm:300000:stochastic_block_model"
    "grid_torus:400000:watts_strogatz_low_p"
    "configuration_model:500000:scale_free"
)

# ── Helper: submit_bsub <script> <dep_str> → prints job ID ────────────────────
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

# ── Job script helper (writes LSF job script, no variable expansion in body) ──
# Usage: write_job_script <path> <job_name> <queue> <walltime> <mem_gb>
#        Then append the actual command block and close BSUBEOF.
# We instead write the full job file directly in each section for clarity.

echo "============================================================"
echo " Simulated Data Job Submission with Hyperparameter Tuning"
echo "============================================================"
echo " HN cases      : ${#HN_CASES[@]}"
echo " EG types      : ${#EG_CASES[@]}"
echo " Sizes         : ${N_NODES_ARRAY[*]}"
echo " Replicates    : ${N_REPLICATES}"
echo " Tuning trials : ${N_TRIALS}"
echo " Queue         : ${QUEUE}"
echo " Walltime      : ${WALLTIME} (tuning: ${TUNE_WALLTIME})"
echo " Memory        : ${MEMORY}GB"
echo " Skip tuning   : ${SKIP_TUNING}"
echo " Skip HN       : ${SKIP_HARD_NEGATIVES}"
echo " Skip EG       : ${SKIP_EXTENDED_GENS}"
echo " Dry run       : ${DRY_RUN}"
echo "============================================================"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════

declare -A TUNE_JOB_ID   # TUNE_JOB_ID[case_name] = job_id

if [ "$SKIP_TUNING" = false ]; then
    echo "PHASE 1: Hyperparameter Tuning"
    echo "================================"

    # ── 1A: Hard-Negatives tuning ────────────────────────────────────────────
    if [ "$SKIP_HARD_NEGATIVES" = false ]; then
        echo ""
        echo "1A: Hard-Negatives tuning (${#HN_CASES[@]} jobs)"
        for ENTRY in "${HN_CASES[@]}"; do
            IFS=':' read -r CASE_NAME SEED_OFFSET TUNER_TYPE <<< "$ENTRY"
            JOB_NAME="tune_hn_${CASE_NAME}"
            JOB_SH="${LOG_DIR}/${JOB_NAME}.sh"
            HPARAM_OUT="${HPARAM_DIR}/hn_${CASE_NAME}/best_hyperparams.json"
            mkdir -p "$( dirname "$HPARAM_OUT" )"

            cat > "$JOB_SH" << BSUBEOF
#!/bin/bash
#BSUB -J ${JOB_NAME}
#BSUB -o ${LOG_DIR}/${JOB_NAME}.out
#BSUB -e ${LOG_DIR}/${JOB_NAME}.err
#BSUB -q ${QUEUE}
#BSUB -W ${TUNE_WALLTIME}
#BSUB -M ${MEMORY}GB
#BSUB -R "rusage[mem=${MEMORY}GB]"

source ${VENV_ACTIVATE}
cd ${PROJECT_DIR}

${PYTHON_ENV} scripts/tune_hyperparameters.py \\
    --output-dir ${HPARAM_DIR}/hn_${CASE_NAME} \\
    --network-type ${TUNER_TYPE} \\
    --n-trials ${N_TRIALS} \\
    --skip-real

exit \$?
BSUBEOF
            chmod +x "$JOB_SH"

            JOB_ID=$( submit_bsub "$JOB_SH" "" )
            TUNE_JOB_ID["${CASE_NAME}"]="${JOB_ID:-DRYRUN}"
            if [ "$DRY_RUN" = true ]; then
                echo "  [DRY RUN] ${JOB_NAME}  type=${TUNER_TYPE}"
            else
                echo "  Submitted ${JOB_ID}: ${JOB_NAME}  type=${TUNER_TYPE}"
            fi
        done
    fi

    # ── 1B: Extended-Generators tuning ──────────────────────────────────────
    if [ "$SKIP_EXTENDED_GENS" = false ]; then
        echo ""
        echo "1B: Extended-Generators tuning (${#EG_CASES[@]} jobs)"
        for ENTRY in "${EG_CASES[@]}"; do
            IFS=':' read -r EG_TYPE SEED_OFFSET PROXY_TYPE <<< "$ENTRY"
            JOB_NAME="tune_eg_${EG_TYPE}"
            JOB_SH="${LOG_DIR}/${JOB_NAME}.sh"
            HPARAM_OUT="${HPARAM_DIR}/eg_${EG_TYPE}/best_hyperparams.json"
            mkdir -p "$( dirname "$HPARAM_OUT" )"

            cat > "$JOB_SH" << BSUBEOF
#!/bin/bash
#BSUB -J ${JOB_NAME}
#BSUB -o ${LOG_DIR}/${JOB_NAME}.out
#BSUB -e ${LOG_DIR}/${JOB_NAME}.err
#BSUB -q ${QUEUE}
#BSUB -W ${TUNE_WALLTIME}
#BSUB -M ${MEMORY}GB
#BSUB -R "rusage[mem=${MEMORY}GB]"

source ${VENV_ACTIVATE}
cd ${PROJECT_DIR}

${PYTHON_ENV} scripts/tune_hyperparameters.py \\
    --output-dir ${HPARAM_DIR}/eg_${EG_TYPE} \\
    --network-type ${PROXY_TYPE} \\
    --n-trials ${N_TRIALS} \\
    --skip-real

exit \$?
BSUBEOF
            chmod +x "$JOB_SH"

            JOB_ID=$( submit_bsub "$JOB_SH" "" )
            TUNE_JOB_ID["${EG_TYPE}"]="${JOB_ID:-DRYRUN}"
            if [ "$DRY_RUN" = true ]; then
                echo "  [DRY RUN] ${JOB_NAME}  proxy=${PROXY_TYPE}"
            else
                echo "  Submitted ${JOB_ID}: ${JOB_NAME}  proxy=${PROXY_TYPE}"
            fi
        done
    fi

    echo ""
    echo "Phase 1 complete: $(( ${#HN_CASES[@]} + ${#EG_CASES[@]} )) tuning jobs submitted."
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: MAIN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "PHASE 2: Main Analysis"
echo "======================"

HN_ANA_JOB_NAMES=()
EG_ANA_JOB_NAMES=()

# ── 2A: Hard-Negatives analysis ───────────────────────────────────────────────
if [ "$SKIP_HARD_NEGATIVES" = false ]; then
    echo ""
    echo "2A: Hard-Negatives analysis  (${#HN_CASES[@]} cases × ${#N_NODES_ARRAY[@]} sizes × ${N_REPLICATES} reps)"

    for ENTRY in "${HN_CASES[@]}"; do
        IFS=':' read -r CASE_NAME SEED_OFFSET TUNER_TYPE <<< "$ENTRY"

        HPARAM_FILE="${HPARAM_DIR}/hn_${CASE_NAME}/best_hyperparams.json"

        # Dependency: wait for this case's tuning job (if not skipping tuning)
        TUNE_DEP=""
        if [ "$SKIP_TUNING" = false ] && [ -n "${TUNE_JOB_ID[$CASE_NAME]+x}" ]; then
            TID="${TUNE_JOB_ID[$CASE_NAME]}"
            [ "$TID" != "DRYRUN" ] && TUNE_DEP="ended(${TID})"
        fi

        echo "  Case: ${CASE_NAME}"
        for N_NODES in "${N_NODES_ARRAY[@]}"; do
            for REP in $( seq -w 0 $(( N_REPLICATES - 1 )) ); do
                SEED=$(( SEED_OFFSET + 10#$REP ))
                NET_ID="${CASE_NAME}_n${N_NODES}_r${REP}"
                JOB_NAME="hn_${CASE_NAME}_n${N_NODES}_r${REP}"
                JOB_SH="${LOG_DIR}/${JOB_NAME}.sh"
                RESULT_DIR="${HN_RESULTS_DIR}/${NET_ID}"
                mkdir -p "$RESULT_DIR"

                cat > "$JOB_SH" << BSUBEOF
#!/bin/bash
#BSUB -J ${JOB_NAME}
#BSUB -o ${LOG_DIR}/${JOB_NAME}.out
#BSUB -e ${LOG_DIR}/${JOB_NAME}.err
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -M ${MEMORY}GB
#BSUB -R "rusage[mem=${MEMORY}GB]"

source ${VENV_ACTIVATE}
cd ${PROJECT_DIR}

${PYTHON_ENV} scripts/run_hard_negative_network.py \\
    --case-name   ${CASE_NAME} \\
    --network-id  ${NET_ID} \\
    --config      ${HN_CONFIG} \\
    --output-dir  ${RESULT_DIR} \\
    --methods     ${METHODS} \\
    --n-nodes     ${N_NODES} \\
    --seed        ${SEED} \\
    --hparam-file ${HPARAM_FILE} \\
    ${RESUME_FLAG}

exit \$?
BSUBEOF
                chmod +x "$JOB_SH"

                JOB_ID=$( submit_bsub "$JOB_SH" "$TUNE_DEP" )
                HN_ANA_JOB_NAMES+=("$JOB_NAME")
                if [ "$DRY_RUN" = true ]; then
                    : # suppress per-job output in dry-run for brevity
                else
                    echo "    Submitted ${JOB_ID}: ${JOB_NAME}"
                fi
            done
        done
    done
fi

# ── 2B: Extended-Generators analysis ─────────────────────────────────────────
if [ "$SKIP_EXTENDED_GENS" = false ]; then
    echo ""
    echo "2B: Extended-Generators analysis  (${#EG_CASES[@]} types × ${#N_NODES_ARRAY[@]} sizes × ${N_REPLICATES} reps)"

    for ENTRY in "${EG_CASES[@]}"; do
        IFS=':' read -r EG_TYPE SEED_OFFSET PROXY_TYPE <<< "$ENTRY"

        HPARAM_FILE="${HPARAM_DIR}/eg_${EG_TYPE}/best_hyperparams.json"

        TUNE_DEP=""
        if [ "$SKIP_TUNING" = false ] && [ -n "${TUNE_JOB_ID[$EG_TYPE]+x}" ]; then
            TID="${TUNE_JOB_ID[$EG_TYPE]}"
            [ "$TID" != "DRYRUN" ] && TUNE_DEP="ended(${TID})"
        fi

        echo "  Type: ${EG_TYPE}"
        for N_NODES in "${N_NODES_ARRAY[@]}"; do
            for REP in $( seq -w 0 $(( N_REPLICATES - 1 )) ); do
                SEED=$(( SEED_OFFSET + 10#$REP ))
                NET_ID="${EG_TYPE}_n${N_NODES}_r${REP}"
                JOB_NAME="eg_${EG_TYPE}_n${N_NODES}_r${REP}"
                JOB_SH="${LOG_DIR}/${JOB_NAME}.sh"
                RESULT_DIR="${EG_RESULTS_DIR}/${NET_ID}"
                mkdir -p "$RESULT_DIR"

                cat > "$JOB_SH" << BSUBEOF
#!/bin/bash
#BSUB -J ${JOB_NAME}
#BSUB -o ${LOG_DIR}/${JOB_NAME}.out
#BSUB -e ${LOG_DIR}/${JOB_NAME}.err
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -M ${MEMORY}GB
#BSUB -R "rusage[mem=${MEMORY}GB]"

source ${VENV_ACTIVATE}
cd ${PROJECT_DIR}

${PYTHON_ENV} scripts/run_extended_generator_network.py \\
    --network-type ${EG_TYPE} \\
    --network-id   ${NET_ID} \\
    --output-dir   ${RESULT_DIR} \\
    --methods      ${METHODS} \\
    --n-nodes      ${N_NODES} \\
    --seed         ${SEED} \\
    --hparam-file  ${HPARAM_FILE} \\
    ${RESUME_FLAG}

exit \$?
BSUBEOF
                chmod +x "$JOB_SH"

                JOB_ID=$( submit_bsub "$JOB_SH" "$TUNE_DEP" )
                EG_ANA_JOB_NAMES+=("$JOB_NAME")
                if [ "$DRY_RUN" = true ]; then
                    :
                else
                    echo "    Submitted ${JOB_ID}: ${JOB_NAME}"
                fi
            done
        done
    done
fi

# Summary
HN_COUNT=${#HN_ANA_JOB_NAMES[@]}
EG_COUNT=${#EG_ANA_JOB_NAMES[@]}
echo ""
echo "Phase 2: submitted ${HN_COUNT} HN jobs + ${EG_COUNT} EG jobs = $(( HN_COUNT + EG_COUNT )) total."

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "PHASE 3: Aggregation"
echo "===================="

# Build dependency on all HN analysis jobs (by job-name wildcard pattern).
# BSub -w "ended(name*)" matches all jobs whose name starts with that prefix.
HN_ANA_DEP=""
[ "$SKIP_HARD_NEGATIVES" = false ] && [ ${#HN_ANA_JOB_NAMES[@]} -gt 0 ] && \
    HN_ANA_DEP="ended(hn_*)"

EG_ANA_DEP=""
[ "$SKIP_EXTENDED_GENS" = false ] && [ ${#EG_ANA_JOB_NAMES[@]} -gt 0 ] && \
    EG_ANA_DEP="ended(eg_*)"

# ── 3A: Hard-Negatives aggregation ───────────────────────────────────────────
if [ "$SKIP_HARD_NEGATIVES" = false ] && [ ${#HN_ANA_JOB_NAMES[@]} -gt 0 ]; then
    JOB_NAME="agg_hn_all"
    JOB_SH="${LOG_DIR}/${JOB_NAME}.sh"

    cat > "$JOB_SH" << BSUBEOF
#!/bin/bash
#BSUB -J ${JOB_NAME}
#BSUB -o ${LOG_DIR}/${JOB_NAME}.out
#BSUB -e ${LOG_DIR}/${JOB_NAME}.err
#BSUB -q ${QUEUE}
#BSUB -W 04:00
#BSUB -M 16GB
#BSUB -R "rusage[mem=16GB]"

source ${VENV_ACTIVATE}
cd ${PROJECT_DIR}

echo "=== Aggregating Hard-Negatives results ==="

${PYTHON_ENV} scripts/aggregate_simulated_data.py \\
    --results-dir ${HN_RESULTS_DIR} \\
    --output-dir  ${AGG_DIR}/hard_negatives

echo "HN aggregation complete. Output: ${AGG_DIR}/hard_negatives"
exit \$?
BSUBEOF
    chmod +x "$JOB_SH"

    JOB_ID=$( submit_bsub "$JOB_SH" "$HN_ANA_DEP" )
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] ${JOB_NAME}  depends-on: ${HN_ANA_DEP:-none}"
    else
        echo "  Submitted ${JOB_ID}: ${JOB_NAME}"
    fi
fi

# ── 3B: Extended-Generators aggregation ──────────────────────────────────────
if [ "$SKIP_EXTENDED_GENS" = false ] && [ ${#EG_ANA_JOB_NAMES[@]} -gt 0 ]; then
    JOB_NAME="agg_eg_all"
    JOB_SH="${LOG_DIR}/${JOB_NAME}.sh"

    cat > "$JOB_SH" << BSUBEOF
#!/bin/bash
#BSUB -J ${JOB_NAME}
#BSUB -o ${LOG_DIR}/${JOB_NAME}.out
#BSUB -e ${LOG_DIR}/${JOB_NAME}.err
#BSUB -q ${QUEUE}
#BSUB -W 04:00
#BSUB -M 16GB
#BSUB -R "rusage[mem=16GB]"

source ${VENV_ACTIVATE}
cd ${PROJECT_DIR}

echo "=== Aggregating Extended-Generators results ==="

${PYTHON_ENV} scripts/aggregate_simulated_data.py \\
    --results-dir ${EG_RESULTS_DIR} \\
    --output-dir  ${AGG_DIR}/extended_generators

echo "EG aggregation complete. Output: ${AGG_DIR}/extended_generators"
exit \$?
BSUBEOF
    chmod +x "$JOB_SH"

    JOB_ID=$( submit_bsub "$JOB_SH" "$EG_ANA_DEP" )
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] ${JOB_NAME}  depends-on: ${EG_ANA_DEP:-none}"
    else
        echo "  Submitted ${JOB_ID}: ${JOB_NAME}"
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

TUNE_COUNT=0
[ "$SKIP_TUNING" = false ] && [ "$SKIP_HARD_NEGATIVES" = false ] && \
    TUNE_COUNT=$(( TUNE_COUNT + ${#HN_CASES[@]} ))
[ "$SKIP_TUNING" = false ] && [ "$SKIP_EXTENDED_GENS" = false ] && \
    TUNE_COUNT=$(( TUNE_COUNT + ${#EG_CASES[@]} ))

echo ""
echo "============================================================"
echo " Submission complete"
echo "  Phase 1 (tuning)   : ${TUNE_COUNT} jobs"
echo "  Phase 2 (analysis) : $(( HN_COUNT + EG_COUNT )) jobs"
echo "  Phase 3 (agg)      : 2 jobs"
echo ""
echo " Output base : ${OUTPUT_DIR}"
echo " Logs        : ${LOG_DIR}"
echo " Hparams     : ${HPARAM_DIR}"
echo " Aggregated  : ${AGG_DIR}"
echo ""
echo " Monitor     : bjobs -u \$USER"
echo " Check logs  : ls ${LOG_DIR}/*.out"
echo "============================================================"
