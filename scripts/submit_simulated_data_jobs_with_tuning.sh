#!/bin/bash
################################################################################
# Simulated Data Job Submission — Dispatcher Pattern
#
# Phase 1 : Hyperparameter Tuning  (18 jobs submitted immediately)
#   Hard-Negatives  : 13 tuning jobs
#   Extended Gens   :  5 tuning jobs
#
# Phase 2 : Dispatcher jobs  (18 jobs, each depends on its tuning job)
#   Each dispatcher runs AFTER its tuning job completes.
#   It queries the cluster state at that moment, then submits
#   N_REPLICATES × N_SIZES analysis jobs with fresh node assignments.
#   This prevents queue flooding and keeps node assignments current.
#
# Phase 3 : Aggregation scripts (written to disk, submitted manually)
#   After all analysis completes:
#     bsub -w "ended(hn_*)" < logs/agg_hn_all.sh
#     bsub -w "ended(eg_*)" < logs/agg_eg_all.sh
#
# NOTE: Memory unit on this cluster is GB (bare integer = GB).
#
# Usage:
#   bash scripts/submit_simulated_data_jobs_with_tuning.sh [options]
#
# Options:
#   --n-replicates N      Replicates per case/size        (default: 30)
#   --n-nodes LIST        Comma-separated sizes           (default: 500,2000,5000)
#   --output-dir DIR      Base output directory
#   --queue Q             LSF queue                       (default: normal)
#   --walltime T          Wall time per analysis job      (default: 120:00)
#   --tune-walltime T     Wall time per tuning job        (default: 48:00)
#   --memory M            Memory in GB per job            (default: 8)
#   --methods LIST        Comma-separated method list
#   --python-env PATH     Path to venv python binary
#   --n-trials N          Optuna trials per method        (default: 30)
#   --resume              Pass --resume to analysis jobs
#   --dry-run             Print commands, do not submit
#   --skip-tuning         Skip Phase 1; dispatchers run immediately (no dep)
#   --skip-hard-negatives Skip HN cases
#   --skip-extended-gens  Skip EG types
################################################################################

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
N_REPLICATES=30
N_NODES_LIST="500,2000,5000"
OUTPUT_DIR="/dccstor/boseukb/Q/NetMed/QuVINE/results/simulated_data"
QUEUE="normal"
WALLTIME="120:00"
TUNE_WALLTIME="48:00"
MEMORY="8"
N_TRIALS=30
# All 39 methods for analysis (used in analysis jobs)
ALL_METHODS="quvine_rwr,quvine_ctqw,quvine_dtqw,quvine_baseline_heat,quvine_baseline_poly,quvine_rwr_heat,quvine_rwr_poly,quvine_ctqw_heat,quvine_ctqw_poly,quvine_dtqw_heat,quvine_dtqw_poly,gat_baseline,gat_heat,gat_poly,gat_rwr,gat_ctqw,gat_dtqw,gat_rwr_heat,gat_rwr_poly,gat_ctqw_heat,gat_ctqw_poly,gat_dtqw_heat,gat_dtqw_poly,graphgps_baseline,graphgps_heat,graphgps_poly,graphgps_rwr,graphgps_ctqw,graphgps_dtqw,graphgps_rwr_heat,graphgps_rwr_poly,graphgps_ctqw_heat,graphgps_ctqw_poly,graphgps_dtqw_heat,graphgps_dtqw_poly,node2vec,netmf,graphsage,baseline_gcnmf"

# Representative methods for tuning (tune these 8, reuse params for all 39)
TUNE_METHODS="quvine_walks,baseline_filter_heat,baseline_filter_poly,baseline_gcnmf,node2vec,netmf,graphsage,appnp"

# Default to all methods for analysis
METHODS="$ALL_METHODS"
PYTHON_ENV="../Python-3.12.2/venv_quvine/bin/python"
RESUME=false
DRY_RUN=false
SKIP_TUNING=false
SKIP_HARD_NEGATIVES=false
SKIP_EXTENDED_GENS=false

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-replicates)        N_REPLICATES="$2";        shift 2 ;;
        --n-nodes)             N_NODES_LIST="$2";        shift 2 ;;
        --output-dir)          OUTPUT_DIR="$2";          shift 2 ;;
        --queue)               QUEUE="$2";               shift 2 ;;
        --walltime)            WALLTIME="$2";            shift 2 ;;
        --tune-walltime)       TUNE_WALLTIME="$2";       shift 2 ;;
        --memory)              MEMORY="$2";              shift 2 ;;
        --methods)             METHODS="$2";             shift 2 ;;
        --python-env)          PYTHON_ENV="$2";          shift 2 ;;
        --n-trials)            N_TRIALS="$2";            shift 2 ;;
        --resume)              RESUME=true;              shift ;;
        --dry-run)             DRY_RUN=true;             shift ;;
        --skip-tuning)         SKIP_TUNING=true;         shift ;;
        --skip-hard-negatives) SKIP_HARD_NEGATIVES=true; shift ;;
        --skip-extended-gens)  SKIP_EXTENDED_GENS=true;  shift ;;
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

AGG_MEMORY=16
RESUME_FLAG=""
[ "$RESUME" = true ] && RESUME_FLAG="--resume"

# ── Network definitions ────────────────────────────────────────────────────────
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
)

EG_CASES=(
    "random_regular:100000:erdos_renyi"
    "heterophilic_sbm:200000:stochastic_block_model"
    "degree_corrected_sbm:300000:stochastic_block_model"
    "grid_torus:400000:watts_strogatz_low_p"
    "configuration_model:500000:scale_free"
)

# ── submit_bsub <script> <dep_str> → prints job ID ────────────────────────────
submit_bsub() {
    local script="$1"
    local dep="$2"
    if [ "$DRY_RUN" = true ]; then
        echo "DRYRUN"
        return 0
    fi
    local cmd=(bsub)
    [ -n "$dep" ] && cmd+=(-w "$dep")
    local out
    if ! out=$( "${cmd[@]}" < "$script" 2>&1 ); then
        echo "  BSUB ERROR: $out" >&2
        echo ""
        return 0
    fi
    echo "$out" | grep -oP 'Job <\K[0-9]+' || true
}

# ── make_dispatcher <dispatch_sh> <job_name> ──────────────────────────────────
# Reads the global _DISPATCH_JOBS array (set by caller).
# Writes a self-contained script that:
#   1. Queries bhosts at run time (after tuning completed → fresh state)
#   2. Round-robins jobs across nodes by available capacity
#   3. Submits each pre-written job script with bsub -m <node>
declare -a _DISPATCH_JOBS=()

make_dispatcher() {
    local dispatch_sh="$1"
    local job_name="$2"

    {
        echo "#!/bin/bash"
        echo "#BSUB -J ${job_name}"
        echo "#BSUB -o ${LOG_DIR}/${job_name}.out"
        echo "#BSUB -e ${LOG_DIR}/${job_name}.err"
        echo "#BSUB -q ${QUEUE}"
        echo "#BSUB -W 02:00"
        echo "#BSUB -n 1"
        echo "#BSUB -M 1"
        echo '#BSUB -R "rusage[mem=1] span[hosts=1]"'
        echo ""
        echo "# Job scripts baked in at generation time"
        echo "declare -a JOBS=("
        for jf in "${_DISPATCH_JOBS[@]}"; do
            printf "    '%s'\n" "$jf"
        done
        echo ")"
        echo ""
        cat << 'DISPATCHER_BODY'
set -euo pipefail

# Query cluster state now that tuning has just finished
declare -a _N=()
declare -A _F=()
_I=0
_C=""

while IFS=' ' read -r h s j m n r su us v _rest; do
    [[ "$s" != "ok" || "$m" == "-" ]] && continue
    (( m == 0 )) && continue
    r_v=${r//-/0}; v_v=${v//-/0}
    f=$(( m - r_v - v_v - 2 ))
    (( f > 0 )) || continue
    _N+=("$h"); _F["$h"]=$f
done < <(bhosts -w 2>/dev/null | tail -n +2)

_nxt() {
    _C=""
    local n=${#_N[@]}
    (( n == 0 )) && return
    local a=0
    while (( a < n )); do
        local nd="${_N[$_I]}"
        _I=$(( (_I + 1) % n ))
        if (( ${_F[$nd]:-0} > 0 )); then
            _F["$nd"]=$(( _F["$nd"] - 1 ))
            _C="$nd"; return
        fi
        a=$(( a + 1 ))
    done
}

echo "=== Dispatcher: ${#JOBS[@]} jobs | ${#_N[@]} nodes available ==="
submitted=0
for js in "${JOBS[@]}"; do
    [ -f "$js" ] || { echo "  WARN (not found): $js"; continue; }
    _nxt
    if [ -n "$_C" ]; then
        jid=$(bsub -m "$_C" < "$js" 2>&1 | grep -oP 'Job <\K[0-9]+' || true)
    else
        jid=$(bsub         < "$js" 2>&1 | grep -oP 'Job <\K[0-9]+' || true)
    fi
    if [ -n "$jid" ]; then
        echo "  $jid: $(basename "$js") -> ${_C:-<scheduler>}"
        submitted=$(( submitted + 1 ))
    else
        echo "  FAIL: $(basename "$js")" >&2
    fi
done
echo "=== Submitted: $submitted / ${#JOBS[@]} ==="
exit 0
DISPATCHER_BODY
    } > "$dispatch_sh"
    chmod +x "$dispatch_sh"
}

# ── Print header ───────────────────────────────────────────────────────────────
echo "============================================================"
echo " Simulated Data Job Submission (Dispatcher Pattern)"
echo "============================================================"
echo " HN cases      : ${#HN_CASES[@]}"
echo " EG types      : ${#EG_CASES[@]}"
echo " Sizes         : ${N_NODES_ARRAY[*]}"
echo " Replicates    : ${N_REPLICATES}"
echo " Tuning trials : ${N_TRIALS}"
echo " Queue         : ${QUEUE}"
echo " Walltime      : ${WALLTIME}  (tuning: ${TUNE_WALLTIME})"
echo " Memory        : ${MEMORY} GB  (aggregation: ${AGG_MEMORY} GB)"
echo " Skip tuning   : ${SKIP_TUNING}"
echo " Skip HN       : ${SKIP_HARD_NEGATIVES}"
echo " Skip EG       : ${SKIP_EXTENDED_GENS}"
echo " Dry run       : ${DRY_RUN}"
echo "============================================================"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════

declare -A TUNE_JOB_ID

if [ "$SKIP_TUNING" = false ]; then
    echo "PHASE 1: Hyperparameter Tuning"
    echo "================================"

    if [ "$SKIP_HARD_NEGATIVES" = false ]; then
        echo ""
        echo "1A: Hard-Negatives tuning (${#HN_CASES[@]} jobs)"
        for ENTRY in "${HN_CASES[@]}"; do
            IFS=':' read -r CASE_NAME SEED_OFFSET TUNER_TYPE <<< "$ENTRY"
            JOB_NAME="tune_hn_${CASE_NAME}"
            JOB_SH="${LOG_DIR}/${JOB_NAME}.sh"
            mkdir -p "${HPARAM_DIR}/hn_${CASE_NAME}"

            cat > "$JOB_SH" << BSUBEOF
#!/bin/bash
#BSUB -J ${JOB_NAME}
#BSUB -o ${LOG_DIR}/${JOB_NAME}.out
#BSUB -e ${LOG_DIR}/${JOB_NAME}.err
#BSUB -q ${QUEUE}
#BSUB -W ${TUNE_WALLTIME}
#BSUB -n 1
#BSUB -M ${MEMORY}
#BSUB -R "rusage[mem=${MEMORY}] span[hosts=1]"

source ${VENV_ACTIVATE}
cd ${PROJECT_DIR}

${PYTHON_ENV} scripts/tune_hyperparameters.py \
    --output-dir  ${HPARAM_DIR}/hn_${CASE_NAME} \
    --network-type ${TUNER_TYPE} \
    --methods     ${TUNE_METHODS} \
    --n-trials    ${N_TRIALS} \
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

    if [ "$SKIP_EXTENDED_GENS" = false ]; then
        echo ""
        echo "1B: Extended-Generators tuning (${#EG_CASES[@]} jobs)"
        for ENTRY in "${EG_CASES[@]}"; do
            IFS=':' read -r EG_TYPE SEED_OFFSET PROXY_TYPE <<< "$ENTRY"
            JOB_NAME="tune_eg_${EG_TYPE}"
            JOB_SH="${LOG_DIR}/${JOB_NAME}.sh"
            mkdir -p "${HPARAM_DIR}/eg_${EG_TYPE}"

            cat > "$JOB_SH" << BSUBEOF
#!/bin/bash
#BSUB -J ${JOB_NAME}
#BSUB -o ${LOG_DIR}/${JOB_NAME}.out
#BSUB -e ${LOG_DIR}/${JOB_NAME}.err
#BSUB -q ${QUEUE}
#BSUB -W ${TUNE_WALLTIME}
#BSUB -n 1
#BSUB -M ${MEMORY}
#BSUB -R "rusage[mem=${MEMORY}] span[hosts=1]"

source ${VENV_ACTIVATE}
cd ${PROJECT_DIR}

${PYTHON_ENV} scripts/tune_hyperparameters.py \
    --output-dir  ${HPARAM_DIR}/eg_${EG_TYPE} \
    --network-type ${PROXY_TYPE} \
    --methods     ${TUNE_METHODS} \
    --n-trials    ${N_TRIALS} \
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
# PHASE 2: DISPATCHERS
# Each dispatcher waits for its tuning job, then submits 90 analysis jobs
# using a fresh bhosts query for node assignment.
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "PHASE 2: Dispatchers"
echo "===================="

HN_ANA_JOB_NAMES=()
EG_ANA_JOB_NAMES=()

# ── 2A: Hard-Negatives ────────────────────────────────────────────────────────
if [ "$SKIP_HARD_NEGATIVES" = false ]; then
    echo ""
    echo "2A: Hard-Negatives dispatchers  (${#HN_CASES[@]} cases)"

    for ENTRY in "${HN_CASES[@]}"; do
        IFS=':' read -r CASE_NAME SEED_OFFSET TUNER_TYPE <<< "$ENTRY"

        HPARAM_FILE="${HPARAM_DIR}/hn_${CASE_NAME}/best_hyperparams.json"
        DISPATCH_NAME="dispatch_hn_${CASE_NAME}"
        DISPATCH_SH="${LOG_DIR}/${DISPATCH_NAME}.sh"

        # Write all replicate job scripts (not submitted here — dispatcher does it)
        _DISPATCH_JOBS=()
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
#BSUB -n 1
#BSUB -M ${MEMORY}
#BSUB -R "rusage[mem=${MEMORY}] span[hosts=1]"

source ${VENV_ACTIVATE}
cd ${PROJECT_DIR}

${PYTHON_ENV} scripts/run_hard_negative_network.py \
    --case-name   ${CASE_NAME} \
    --network-id  ${NET_ID} \
    --config      ${HN_CONFIG} \
    --output-dir  ${RESULT_DIR} \
    --methods     ${METHODS} \
    --n-nodes     ${N_NODES} \
    --seed        ${SEED} \
    --hparam-file ${HPARAM_FILE} \
    ${RESUME_FLAG}

exit \$?
BSUBEOF
                chmod +x "$JOB_SH"
                _DISPATCH_JOBS+=("$JOB_SH")
                HN_ANA_JOB_NAMES+=("$JOB_NAME")
            done
        done

        # Generate self-contained dispatcher script
        make_dispatcher "$DISPATCH_SH" "$DISPATCH_NAME"

        # Submit dispatcher — depends on tuning job (or runs immediately if --skip-tuning)
        TUNE_DEP=""
        if [ "$SKIP_TUNING" = false ] && [ -n "${TUNE_JOB_ID[$CASE_NAME]+x}" ]; then
            TID="${TUNE_JOB_ID[$CASE_NAME]}"
            [ "$TID" != "DRYRUN" ] && TUNE_DEP="ended(${TID})"
        fi

        DISP_ID=$( submit_bsub "$DISPATCH_SH" "$TUNE_DEP" )
        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY RUN] ${DISPATCH_NAME}  (${#_DISPATCH_JOBS[@]} jobs queued)"
        else
            echo "  Submitted dispatcher ${DISP_ID}: ${DISPATCH_NAME}  dep='${TUNE_DEP:-none}'  jobs=${#_DISPATCH_JOBS[@]}"
        fi
    done
fi

# ── 2B: Extended-Generators ───────────────────────────────────────────────────
if [ "$SKIP_EXTENDED_GENS" = false ]; then
    echo ""
    echo "2B: Extended-Generators dispatchers  (${#EG_CASES[@]} types)"

    for ENTRY in "${EG_CASES[@]}"; do
        IFS=':' read -r EG_TYPE SEED_OFFSET PROXY_TYPE <<< "$ENTRY"

        HPARAM_FILE="${HPARAM_DIR}/eg_${EG_TYPE}/best_hyperparams.json"
        DISPATCH_NAME="dispatch_eg_${EG_TYPE}"
        DISPATCH_SH="${LOG_DIR}/${DISPATCH_NAME}.sh"

        _DISPATCH_JOBS=()
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
#BSUB -n 1
#BSUB -M ${MEMORY}
#BSUB -R "rusage[mem=${MEMORY}] span[hosts=1]"

source ${VENV_ACTIVATE}
cd ${PROJECT_DIR}

${PYTHON_ENV} scripts/run_extended_generator_network.py \
    --network-type ${EG_TYPE} \
    --network-id   ${NET_ID} \
    --output-dir   ${RESULT_DIR} \
    --methods      ${METHODS} \
    --n-nodes      ${N_NODES} \
    --seed         ${SEED} \
    --hparam-file  ${HPARAM_FILE} \
    ${RESUME_FLAG}

exit \$?
BSUBEOF
                chmod +x "$JOB_SH"
                _DISPATCH_JOBS+=("$JOB_SH")
                EG_ANA_JOB_NAMES+=("$JOB_NAME")
            done
        done

        make_dispatcher "$DISPATCH_SH" "$DISPATCH_NAME"

        TUNE_DEP=""
        if [ "$SKIP_TUNING" = false ] && [ -n "${TUNE_JOB_ID[$EG_TYPE]+x}" ]; then
            TID="${TUNE_JOB_ID[$EG_TYPE]}"
            [ "$TID" != "DRYRUN" ] && TUNE_DEP="ended(${TID})"
        fi

        DISP_ID=$( submit_bsub "$DISPATCH_SH" "$TUNE_DEP" )
        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY RUN] ${DISPATCH_NAME}  (${#_DISPATCH_JOBS[@]} jobs queued)"
        else
            echo "  Submitted dispatcher ${DISP_ID}: ${DISPATCH_NAME}  dep='${TUNE_DEP:-none}'  jobs=${#_DISPATCH_JOBS[@]}"
        fi
    done
fi

echo ""
echo "Phase 2: ${#HN_ANA_JOB_NAMES[@]} HN + ${#EG_ANA_JOB_NAMES[@]} EG analysis scripts written."
echo "         Dispatchers will submit them after tuning completes."

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: AGGREGATION  (scripts written; submit manually after analysis done)
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "PHASE 3: Aggregation scripts (written — submit manually after analysis)"
echo "========================================================================="

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
#BSUB -n 1
#BSUB -M ${AGG_MEMORY}
#BSUB -R "rusage[mem=${AGG_MEMORY}] span[hosts=1]"

source ${VENV_ACTIVATE}
cd ${PROJECT_DIR}

${PYTHON_ENV} scripts/aggregate_simulated_data.py \
    --results-dir ${HN_RESULTS_DIR} \
    --output-dir  ${AGG_DIR}/hard_negatives

exit \$?
BSUBEOF
    chmod +x "$JOB_SH"
    echo "  HN agg script : ${JOB_SH}"
    echo "  Submit when done: bsub -w \"ended(hn_*)\" < ${JOB_SH}"
fi

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
#BSUB -n 1
#BSUB -M ${AGG_MEMORY}
#BSUB -R "rusage[mem=${AGG_MEMORY}] span[hosts=1]"

source ${VENV_ACTIVATE}
cd ${PROJECT_DIR}

${PYTHON_ENV} scripts/aggregate_simulated_data.py \
    --results-dir ${EG_RESULTS_DIR} \
    --output-dir  ${AGG_DIR}/extended_generators

exit \$?
BSUBEOF
    chmod +x "$JOB_SH"
    echo "  EG agg script : ${JOB_SH}"
    echo "  Submit when done: bsub -w \"ended(eg_*)\" < ${JOB_SH}"
fi

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

TUNE_COUNT=0
[ "$SKIP_TUNING" = false ] && [ "$SKIP_HARD_NEGATIVES" = false ] && \
    TUNE_COUNT=$(( TUNE_COUNT + ${#HN_CASES[@]} ))
[ "$SKIP_TUNING" = false ] && [ "$SKIP_EXTENDED_GENS" = false ] && \
    TUNE_COUNT=$(( TUNE_COUNT + ${#EG_CASES[@]} ))

DISP_COUNT=0
[ "$SKIP_HARD_NEGATIVES" = false ] && \
    DISP_COUNT=$(( DISP_COUNT + ${#HN_CASES[@]} ))
[ "$SKIP_EXTENDED_GENS" = false ] && \
    DISP_COUNT=$(( DISP_COUNT + ${#EG_CASES[@]} ))

echo ""
echo "============================================================"
echo " Submission complete"
echo "  Phase 1 (tuning)      : ${TUNE_COUNT} jobs submitted now"
echo "  Phase 2 (dispatchers) : ${DISP_COUNT} jobs submitted now"
echo "    → each dispatcher submits $(( N_REPLICATES * ${#N_NODES_ARRAY[@]} )) analysis jobs after tuning"
echo "  Phase 3 (aggregation) : submit manually (scripts in logs/)"
echo ""
echo " Output  : ${OUTPUT_DIR}"
echo " Logs    : ${LOG_DIR}"
echo " Monitor : bjobs -u \$USER"
echo "============================================================"
