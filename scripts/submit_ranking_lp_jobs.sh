#!/bin/bash
################################################################################
# Ranking + Link-Prediction Jobs for All Baselines — PPI Disease v3
#
# Submits one LSF job per network to run ranking and link-prediction evaluation
# for all baseline methods using pre-computed .npy embeddings.
# APPNP results (already in the CSVs) are preserved via --resume.
#
# After all per-network jobs complete, an aggregation job rebuilds
# comprehensive_results_all_methods.csv with the full method set.
#
# Usage:
#   bash scripts/submit_ranking_lp_jobs.sh [options]
#
# Options:
#   --queue QUEUE         LSF queue             (default: normal)
#   --walltime TIME       Wall time per job     (default: 2:00)
#   --memory MEM          Memory in GB          (default: 8)
#   --python-env PATH     Path to venv activate (default: ../Python-3.12.2/venv_quvine/bin/activate)
#   --methods LIST        Comma-separated methods to evaluate
#                         (default: all 13 baselines)
#   --resume              Skip methods already present in ranking_results.csv
#   --dry-run             Print job scripts, do not submit
################################################################################

set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
QUEUE="normal"
WALLTIME="18:00"
MEMORY="8"
PYTHON_ENV="../Python-3.12.2/venv_quvine/bin/activate"
METHODS="quvine_fused,quvine_rwr,quvine_heat,quvine_poly,quvine_hgcnmf,quvine_pgcnmf,netmf,node2vec,baseline_filter,baseline_gcnmf,graphsage,quvine_ctqw,quvine_dtqw"
RESUME=false
DRY_RUN=false

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --queue)      QUEUE="$2";      shift 2 ;;
        --walltime)   WALLTIME="$2";   shift 2 ;;
        --memory)     MEMORY="$2";     shift 2 ;;
        --python-env) PYTHON_ENV="$2"; shift 2 ;;
        --methods)    METHODS="$2";    shift 2 ;;
        --resume)     RESUME=true;     shift ;;
        --dry-run)    DRY_RUN=true;    shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--queue Q] [--walltime T] [--memory M] [--methods LIST] [--resume] [--dry-run]"
            exit 1 ;;
    esac
done

RESUME_FLAG=""
[ "$RESUME" = true ] && RESUME_FLAG="--resume"

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
RESULTS_DIR="/dccstor/boseukb/Q/NetMed/QuVINE/results/ppi_disease_v3/results"
LOG_DIR="/dccstor/boseukb/Q/NetMed/QuVINE/results/ppi_disease_v3/logs_ranking_lp"
WORKER="${PROJECT_DIR}/scripts/run_ranking_lp_single_network.py"
AGG_SCRIPT="${PROJECT_DIR}/scripts/build_comprehensive_all_methods.py"

mkdir -p "$LOG_DIR"

# ── Banner ─────────────────────────────────────────────────────────────────────
echo "======================================================================"
echo " Ranking + Link-Prediction Jobs — All Baselines (PPI Disease v3)"
echo "======================================================================"
echo " Project dir    : $PROJECT_DIR"
echo " Results dir    : $RESULTS_DIR"
echo " Log dir        : $LOG_DIR"
echo " Queue          : $QUEUE"
echo " Wall time      : $WALLTIME"
echo " Memory         : ${MEMORY}GB"
echo " Methods        : $METHODS"
echo " Resume         : $RESUME"
echo " Dry run        : $DRY_RUN"
echo "======================================================================"
echo ""
[ "$DRY_RUN" = true ] && echo "DRY RUN MODE — no jobs will be submitted" && echo ""

# ── Find all graphml files ─────────────────────────────────────────────────────
GRAPHML_FILES=()
while IFS= read -r -d '' file; do
    GRAPHML_FILES+=("$file")
done < <(find "$RESULTS_DIR" -maxdepth 2 -name "*.graphml" -type f -print0 | sort -z)

TOTAL_NETWORKS=${#GRAPHML_FILES[@]}
echo "Found $TOTAL_NETWORKS networks"
echo ""

if [ "$TOTAL_NETWORKS" -eq 0 ]; then
    echo "ERROR: No graphml files found in $RESULTS_DIR"
    exit 1
fi

# ── Submit one job per network ─────────────────────────────────────────────────
JOB_IDS=()
JOB_COUNT=0
SKIPPED=0

for graphml_path in "${GRAPHML_FILES[@]}"; do
    network_dir="$(dirname "$graphml_path")"
    network_id="$(basename "$network_dir")"

    # In resume mode: skip if all methods already have ranking results
    if [ "$RESUME" = true ]; then
        rank_csv="${network_dir}/${network_id}_ranking_results.csv"
        if [ -f "$rank_csv" ]; then
            # Count distinct methods in file vs requested
            n_done=$(tail -n +2 "$rank_csv" 2>/dev/null | cut -d',' -f2 | sort -u | wc -l)
            n_requested=$(echo "$METHODS" | tr ',' '\n' | wc -l)
            if [ "$n_done" -ge "$n_requested" ]; then
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
        fi
    fi

    job_name="rlp_${network_id}"
    job_sh="${LOG_DIR}/${job_name}.sh"
    job_out="${LOG_DIR}/${job_name}.out"
    job_err="${LOG_DIR}/${job_name}.err"

    cat > "$job_sh" << JOBEOF
#!/bin/bash
#BSUB -J ${job_name}
#BSUB -o ${job_out}
#BSUB -e ${job_err}
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -M ${MEMORY}GB
#BSUB -R "rusage[mem=${MEMORY}GB]"

source ${PROJECT_DIR}/${PYTHON_ENV}
cd ${PROJECT_DIR}

python ${WORKER} \\
    --graphml-path     "${graphml_path}" \\
    --output-dir       "${network_dir}" \\
    --network-id       "${network_id}" \\
    --network-type     "real_ppi" \\
    --methods          "${METHODS}" \\
    --negative-strategy random \\
    --num-seeds        15 \\
    --num-targets      25 \\
    ${RESUME_FLAG} \\
    --verbose

exit \$?
JOBEOF

    chmod +x "$job_sh"

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] $job_name"
    else
        job_id=$(bsub < "$job_sh" 2>&1 | grep -oP 'Job <\K[0-9]+')
        if [ -n "$job_id" ]; then
            JOB_IDS+=("$job_id")
            JOB_COUNT=$((JOB_COUNT + 1))
            if [ $((JOB_COUNT % 50)) -eq 0 ]; then
                echo "  Submitted $JOB_COUNT jobs ..."
            fi
        else
            echo "  ERROR: failed to submit $job_name"
        fi
    fi
done

echo ""
echo "======================================================================"
echo " Jobs submitted : $JOB_COUNT"
echo " Skipped        : $SKIPPED  (already complete)"
echo "======================================================================"

# ── Build dependency string ────────────────────────────────────────────────────
DEPENDENCY_STRING=""
for jid in "${JOB_IDS[@]}"; do
    if [ -z "$DEPENDENCY_STRING" ]; then
        DEPENDENCY_STRING="ended($jid)"
    else
        DEPENDENCY_STRING="${DEPENDENCY_STRING} && ended($jid)"
    fi
done

# ── Aggregation job ────────────────────────────────────────────────────────────
AGG_NAME="rlp_aggregate"
AGG_SH="${LOG_DIR}/${AGG_NAME}.sh"

cat > "$AGG_SH" << AGGEOF
#!/bin/bash
#BSUB -J ${AGG_NAME}
#BSUB -o ${LOG_DIR}/${AGG_NAME}.out
#BSUB -e ${LOG_DIR}/${AGG_NAME}.err
#BSUB -q ${QUEUE}
#BSUB -W 0:30
#BSUB -M 32GB
#BSUB -R "rusage[mem=32GB]"
$([ -n "$DEPENDENCY_STRING" ] && echo "#BSUB -w \"${DEPENDENCY_STRING}\"")

source ${PROJECT_DIR}/${PYTHON_ENV}
cd ${PROJECT_DIR}

echo "Rebuilding comprehensive_results_all_methods.csv ..."
python ${AGG_SCRIPT}

echo "Done."
exit \$?
AGGEOF

chmod +x "$AGG_SH"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "  [DRY RUN] Aggregation job: $AGG_NAME (depends on $JOB_COUNT jobs)"
elif [ "${#JOB_IDS[@]}" -gt 0 ]; then
    echo ""
    echo "Submitting aggregation job ..."
    agg_id=$(bsub < "$AGG_SH" 2>&1 | grep -oP 'Job <\K[0-9]+')
    if [ -n "$agg_id" ]; then
        echo "  Submitted aggregation job $agg_id (depends on ${#JOB_IDS[@]} jobs)"
    else
        echo "  ERROR: failed to submit aggregation job"
    fi
else
    echo "  No jobs submitted — skipping aggregation."
fi

echo ""
echo "======================================================================"
echo " SUBMISSION COMPLETE"
echo " Monitor  : bjobs -u \$USER"
echo " Results  : ${RESULTS_DIR}/comprehensive_results_all_methods.csv"
echo "======================================================================"
