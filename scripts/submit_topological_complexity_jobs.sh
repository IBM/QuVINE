#!/bin/bash
################################################################################
# Topological Complexity Job Submission
#
# Submits 560 parallel jobs (one per network) to compute topological complexity
# metrics (Betti numbers and persistence entropy) for all graphs in ppi_disease_v3.
#
# After all jobs complete, an aggregation job merges the results into
# comprehensive_results_ppi3.csv.
#
# Usage:
#   bash scripts/submit_topological_complexity_jobs.sh [--dry-run] [--queue QUEUE]
#                                                        [--walltime TIME] [--memory MEM]
################################################################################

set -e

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
QUEUE="normal"
WALLTIME="12:00"
MEMORY="8"
DRY_RUN=false
PYTHON_ENV="../Python-3.12.2/venv_quvine/bin/activate"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --queue)      QUEUE="$2";      shift 2 ;;
        --walltime)   WALLTIME="$2";   shift 2 ;;
        --memory)     MEMORY="$2";     shift 2 ;;
        --python-env) PYTHON_ENV="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true;    shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--queue Q] [--walltime T] [--memory M] [--dry-run]"
            exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
RESULTS_BASE="/dccstor/boseukb/Q/NetMed/QuVINE/results/ppi_disease_v3/results"
LOG_DIR="${RESULTS_BASE}/logs_topology"

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Find all networks
# ---------------------------------------------------------------------------
GRAPHML_FILES=()
while IFS= read -r -d '' file; do
    GRAPHML_FILES+=("$file")
done < <(find "$RESULTS_BASE" -maxdepth 2 -name "*.graphml" -print0 | sort -z)

TOTAL_JOBS=${#GRAPHML_FILES[@]}

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "======================================================"
echo " Topological Complexity Job Submission"
echo "======================================================"
echo " Project dir  : $PROJECT_DIR"
echo " Results dir  : $RESULTS_BASE"
echo " Queue        : $QUEUE"
echo " Wall time    : $WALLTIME"
echo " Memory       : ${MEMORY}GB"
echo " Total jobs   : ${TOTAL_JOBS}"
echo " Dry run      : $DRY_RUN"
echo "======================================================"
echo ""
[ "$DRY_RUN" = true ] && echo "DRY RUN MODE — no jobs will be submitted" && echo ""

# ---------------------------------------------------------------------------
# Submit jobs
# ---------------------------------------------------------------------------
JOB_IDS=()
JOB_COUNT=0

for GRAPHML_PATH in "${GRAPHML_FILES[@]}"; do
    # Extract network ID from path
    NETWORK_ID=$(basename "$GRAPHML_PATH" .graphml)
    COMPLEXITY_CSV="${RESULTS_BASE}/${NETWORK_ID}/${NETWORK_ID}_complexity.csv"
    
    # Check if complexity file exists
    if [ ! -f "$COMPLEXITY_CSV" ]; then
        echo "  WARNING: $COMPLEXITY_CSV not found, skipping $NETWORK_ID"
        continue
    fi
    
    JOB_NAME="topo_${NETWORK_ID}"
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

# Install ripser if not available
pip install ripser --quiet 2>/dev/null || true

python scripts/compute_single_network_topology.py \\
    --graphml "${GRAPHML_PATH}" \\
    --network-id "${NETWORK_ID}" \\
    --output-csv "${COMPLEXITY_CSV}"

exit \$?
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

echo ""
echo "======================================================"
echo " Jobs submitted: $JOB_COUNT / $TOTAL_JOBS"
echo "======================================================"

# ---------------------------------------------------------------------------
# Aggregation job (depends on all topology jobs)
# ---------------------------------------------------------------------------
AGG_NAME="topo_aggregate"
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
#BSUB -W 0:30
#BSUB -M 16GB
#BSUB -R "rusage[mem=16GB]"
$([ -n "$DEPENDENCY_STRING" ] && echo "#BSUB -w \"${DEPENDENCY_STRING}\"")

source ${PROJECT_DIR}/${PYTHON_ENV}
cd ${PROJECT_DIR}

echo "Aggregating topological complexity results ..."
python scripts/merge_topological_to_comprehensive.py

echo "Done. Results: ${RESULTS_BASE}/comprehensive_results_ppi3_with_topology.csv"
exit \$?
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
echo " Results: ${RESULTS_BASE}/comprehensive_results_ppi3_with_topology.csv"
echo "======================================================"
