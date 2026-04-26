#!/bin/bash
################################################################################
# Submit APPNP Baseline Jobs for PPI Disease v3
#
# This script submits one LSF job per network to run APPNP baseline.
# After all jobs complete, an aggregation job regenerates comprehensive_results.csv
# with APPNP integrated.
#
# Usage:
#   bash scripts/submit_appnp_ppi_disease.sh [options]
#
# Options:
#   --queue QUEUE        LSF queue               (default: normal)
#   --walltime TIME      Wall time per job       (default: 2:00)
#   --memory MEM         Memory in GB            (default: 8)
#   --python-env PATH    Path to venv activate   (default: ../Python-3.12.2/venv_quvine/bin/activate)
#   --resume             Skip networks with APPNP already done
#   --dry-run            Print scripts, do not submit
################################################################################

set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
QUEUE="normal"
WALLTIME="12:00"
MEMORY="8"
PYTHON_ENV="../Python-3.12.2/venv_quvine/bin/activate"
RESUME=false
DRY_RUN=false

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --queue)      QUEUE="$2";      shift 2 ;;
        --walltime)   WALLTIME="$2";   shift 2 ;;
        --memory)     MEMORY="$2";     shift 2 ;;
        --python-env) PYTHON_ENV="$2"; shift 2 ;;
        --resume)     RESUME=true;     shift ;;
        --dry-run)    DRY_RUN=true;    shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--queue Q] [--walltime T] [--memory M] [--python-env PATH] [--resume] [--dry-run]"
            exit 1 ;;
    esac
done

RESUME_FLAG=""
[ "$RESUME" = true ] && RESUME_FLAG="--resume"

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
RESULTS_DIR="/dccstor/boseukb/Q/NetMed/QuVINE/results/ppi_disease_v3/results"
LOG_DIR="/dccstor/boseukb/Q/NetMed/QuVINE/results/ppi_disease_v3/logs_appnp"
ANALYSIS_SCRIPT="${PROJECT_DIR}/scripts/run_appnp_single_network.py"

mkdir -p "$LOG_DIR"

# ── Banner ─────────────────────────────────────────────────────────────────────
echo "============================================================"
echo " APPNP Baseline - PPI Disease v3"
echo "============================================================"
echo " Project dir    : $PROJECT_DIR"
echo " Results dir    : $RESULTS_DIR"
echo " Log dir        : $LOG_DIR"
echo " Queue          : $QUEUE"
echo " Wall time      : $WALLTIME"
echo " Memory         : ${MEMORY}GB"
echo " Python env     : $PYTHON_ENV"
echo " Resume         : $RESUME"
echo " Dry run        : $DRY_RUN"
echo "============================================================"
echo ""
[ "$DRY_RUN" = true ] && echo "DRY RUN MODE — no jobs will be submitted" && echo ""

# ── Find all graphml files ─────────────────────────────────────────────────────
GRAPHML_FILES=()
while IFS= read -r -d '' file; do
    GRAPHML_FILES+=("$file")
done < <(find "$RESULTS_DIR" -name "*.graphml" -type f -print0 | sort -z)

TOTAL_NETWORKS=${#GRAPHML_FILES[@]}
echo "Found $TOTAL_NETWORKS graphml files"
echo ""

if [ $TOTAL_NETWORKS -eq 0 ]; then
    echo "ERROR: No graphml files found in $RESULTS_DIR"
    exit 1
fi

# ── Submit one job per network ─────────────────────────────────────────────────
ANALYSIS_JOB_IDS=()
JOB_COUNT=0

for graphml_path in "${GRAPHML_FILES[@]}"; do
    # Extract network info from path
    # Path format: .../results/NETWORK_ID/NETWORK_ID.graphml
    network_dir=$(dirname "$graphml_path")
    network_id=$(basename "$network_dir")
    
    # Skip if APPNP already done (when --resume is set)
    if [ "$RESUME" = true ]; then
        ranking_file="${network_dir}/${network_id}_ranking_results.csv"
        if [ -f "$ranking_file" ]; then
            if grep -q ",appnp," "$ranking_file" 2>/dev/null; then
                continue  # Skip this network
            fi
        fi
    fi
    
    # All PPI disease networks are real_ppi type with random negative strategy
    network_type="real_ppi"
    negative_strategy="random"
    
    job_name="appnp_${network_id}"
    log_file="${LOG_DIR}/${job_name}.out"
    err_file="${LOG_DIR}/${job_name}.err"
    job_script="${LOG_DIR}/${job_name}.sh"
    
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

source ${PROJECT_DIR}/${PYTHON_ENV}
cd ${PROJECT_DIR}

python ${ANALYSIS_SCRIPT} \
    --graphml-path ${graphml_path} \
    --output-dir ${network_dir} \
    --network-id ${network_id} \
    --network-type ${network_type} \
    --negative-strategy ${negative_strategy} \
    ${RESUME_FLAG} \
    --verbose

exit \$?
JOBEOF

    chmod +x "$job_script"
    
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] Would submit: $job_name"
    else
        job_id=$(bsub < "$job_script" 2>&1 | grep -oP 'Job <\K[0-9]+')
        if [ -n "$job_id" ]; then
            ANALYSIS_JOB_IDS+=("$job_id")
            JOB_COUNT=$((JOB_COUNT + 1))
            # Print progress every 50 jobs
            if [ $((JOB_COUNT % 50)) -eq 0 ]; then
                echo "  Submitted $JOB_COUNT / $TOTAL_NETWORKS jobs..."
            fi
        else
            echo "  ERROR: Failed to submit $job_name"
        fi
    fi
done

echo ""
echo "============================================================"
echo " Analysis jobs submitted: $JOB_COUNT / $TOTAL_NETWORKS"
echo "============================================================"

# ── Build dependency string ────────────────────────────────────────────────────
DEPENDENCY_STRING=""
for job_id in "${ANALYSIS_JOB_IDS[@]}"; do
    if [ -z "$DEPENDENCY_STRING" ]; then
        DEPENDENCY_STRING="ended($job_id)"
    else
        DEPENDENCY_STRING="$DEPENDENCY_STRING && ended($job_id)"
    fi
done

# ── Aggregation job ────────────────────────────────────────────────────────────
agg_job_name="appnp_ppi_aggregate"
agg_job_script="${LOG_DIR}/${agg_job_name}.sh"

cat > "$agg_job_script" << AGGEOF
#!/bin/bash
#BSUB -J ${agg_job_name}
#BSUB -o ${LOG_DIR}/${agg_job_name}.out
#BSUB -e ${LOG_DIR}/${agg_job_name}.err
#BSUB -q ${QUEUE}
#BSUB -W 2:00
#BSUB -M 32GB
#BSUB -R "rusage[mem=32GB]"
$([ -n "$DEPENDENCY_STRING" ] && echo "#BSUB -w \"${DEPENDENCY_STRING}\"")

source ${PROJECT_DIR}/${PYTHON_ENV}
cd ${PROJECT_DIR}

echo "========================================"
echo " REGENERATING comprehensive_results.csv"
echo " WITH APPNP INTEGRATED"
echo "========================================"

python - << 'PYEOF'
import sys
sys.path.insert(0, '${PROJECT_DIR}/src')
from quvine.comprehensive_embedding_analysis import collect_and_aggregate_results

comprehensive_df = collect_and_aggregate_results(
    results_dir='${RESULTS_DIR}',
    output_file='comprehensive_results.csv',
    verbose=True
)

print(f"\\n✓ comprehensive_results.csv regenerated")
print(f"  Total rows: {len(comprehensive_df)}")
if 'method' in comprehensive_df.columns:
    print(f"  Methods: {sorted(comprehensive_df['method'].unique())}")
    appnp_count = (comprehensive_df['method'] == 'appnp').sum()
    print(f"  APPNP rows: {appnp_count}")
PYEOF

echo "========================================"
echo " AGGREGATION COMPLETE"
echo " Results: ${RESULTS_DIR}/comprehensive_results.csv"
echo "========================================"

exit \$?
AGGEOF

chmod +x "$agg_job_script"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "  [DRY RUN] Would submit aggregation job: $agg_job_name"
    [ -n "$DEPENDENCY_STRING" ] && echo "  [DRY RUN] Dependencies: ${#ANALYSIS_JOB_IDS[@]} analysis jobs"
elif [ ${#ANALYSIS_JOB_IDS[@]} -gt 0 ]; then
    echo ""
    echo "Submitting aggregation job..."
    agg_id=$(bsub < "$agg_job_script" 2>&1 | grep -oP 'Job <\K[0-9]+')
    if [ -n "$agg_id" ]; then
        echo "  Submitted aggregation job $agg_id"
        echo "  Depends on ${#ANALYSIS_JOB_IDS[@]} analysis jobs"
    else
        echo "  ERROR: Failed to submit aggregation job"
    fi
else
    echo "  No analysis jobs submitted — skipping aggregation job."
fi

echo ""
echo "============================================================"
echo " SUBMISSION COMPLETE"
echo " Analysis jobs : $JOB_COUNT"
if [ "$DRY_RUN" = false ] && [ ${#ANALYSIS_JOB_IDS[@]} -gt 0 ]; then
    echo " Aggregation   : 1 (runs after all analysis jobs)"
    echo ""
    echo " Monitor:  bjobs -u \$USER"
    echo " Results:  ${RESULTS_DIR}/comprehensive_results.csv"
fi
echo "============================================================"

