#!/bin/bash
################################################################################
# Submit QBC complexity computation job for hard_negatives_v3
#
# Usage:
#   bash scripts/submit_qbc_complexity_job.sh [--n-jobs N] [--reset-checkpoint]
################################################################################

set -e

N_JOBS=8
RESET=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --n-jobs)          N_JOBS="$2"; shift 2 ;;
        --reset-checkpoint) RESET="--reset-checkpoint"; shift ;;
        --dry-run)         DRY_RUN=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
LOG_DIR="/dccstor/boseukb/Q/NetMed/QuVINE/results/hard_negatives_v3/logs"
mkdir -p "$LOG_DIR"

JOB_SH="${LOG_DIR}/qbc_complexity.sh"

cat > "$JOB_SH" << EOF
#!/bin/bash
#BSUB -J quvine_qbc_complexity
#BSUB -o ${LOG_DIR}/qbc_complexity.out
#BSUB -e ${LOG_DIR}/qbc_complexity.err
#BSUB -q normal
#BSUB -W 4:00
#BSUB -n ${N_JOBS}
#BSUB -M 32GB
#BSUB -R "rusage[mem=32GB] span[hosts=1]"

source ${PROJECT_DIR}/../Python-3.12.2/venv_quvine/bin/activate
cd ${PROJECT_DIR}

echo "=== QBC complexity job started: \$(date) ==="
echo "Workers: ${N_JOBS}"

python scripts/add_qbc_complexity_to_results.py --n-jobs ${N_JOBS} ${RESET}

echo "=== QBC complexity job done: \$(date) ==="
EOF

chmod +x "$JOB_SH"

echo "Job script: $JOB_SH"

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would submit: quvine_qbc_complexity"
    cat "$JOB_SH"
else
    JOB_ID=$(bsub < "$JOB_SH" 2>&1 | grep -oP 'Job <\K[0-9]+')
    if [ -n "$JOB_ID" ]; then
        echo "Submitted job $JOB_ID: quvine_qbc_complexity"
        echo "Monitor: bjobs $JOB_ID"
        echo "Logs:    tail -f ${LOG_DIR}/qbc_complexity.out"
        echo "$JOB_ID"
    else
        echo "ERROR: Failed to submit job"
        exit 1
    fi
fi
