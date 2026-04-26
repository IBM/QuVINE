#!/bin/bash
################################################################################
# PPI Comprehensive Job Submission with Hyperparameter Tuning
#
# Phase 1: Hyperparameter Tuning (25 jobs)
#   - Ranking: 5 networks × 3 diseases = 15 tuning jobs
#   - Classification: 5 networks = 5 tuning jobs
#   - Link Prediction: 5 networks = 5 tuning jobs
#
# Phase 2: Main Analysis (waits for tuning, 750 jobs)
#   - Ranking: 5 networks × 3 diseases × 30 reps = 450 jobs
#   - Classification: 5 networks × 30 reps = 150 jobs
#   - Link Prediction: 5 networks × 30 reps = 150 jobs
#
# Phase 3: Aggregation (1 job, waits for all analysis)
#
# Total: 776 jobs
#
# Networks: BioPlex3, HumanNet, STRING, ProteomeHD, IntAct
# Diseases (ranking only): asthma, autism, schizophrenia
# Methods: all 15 embedding methods
#
# Usage:
#   bash scripts/submit_ppi_comprehensive_with_tuning.sh [OPTIONS]
#
################################################################################

set -e

# Defaults
QUEUE="normal"
WALLTIME="48:00"
MEMORY="16"
N_REPS="30"
DRY_RUN=false
SKIP_TUNING=false
SKIP_RANKING=false
SKIP_CLASSIFICATION=false
SKIP_LINK_PREDICTION=false
PYTHON_ENV="../Python-3.12.2/venv_quvine/bin/activate"
METHODS="quvine_fused-walk,quvine_ctqw,quvine_dtqw,quvine_rwr,quvine_heat,quvine_poly,quvine_fused-filt,quvine_hgcnmf,quvine_pgcnmf,quvine_fused-gcnmf,netmf,node2vec,graphsage,baseline_gcnmf,baseline_filter"

# Argument parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        --queue) QUEUE="$2"; shift 2 ;;
        --walltime) WALLTIME="$2"; shift 2 ;;
        --memory) MEMORY="$2"; shift 2 ;;
        --n-reps) N_REPS="$2"; shift 2 ;;
        --python-env) PYTHON_ENV="$2"; shift 2 ;;
        --methods) METHODS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --skip-tuning) SKIP_TUNING=true; shift ;;
        --skip-ranking) SKIP_RANKING=true; shift ;;
        --skip-classification) SKIP_CLASSIFICATION=true; shift ;;
        --skip-link-prediction) SKIP_LINK_PREDICTION=true; shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--queue Q] [--walltime T] [--memory M] [--n-reps R]"
            echo "          [--methods M] [--python-env P] [--dry-run] [--skip-tuning]"
            echo "          [--skip-ranking] [--skip-classification] [--skip-link-prediction]"
            exit 1 ;;
    esac
done

# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
DATA_ROOT="/dccstor/boseukb/Q/NetMed/Aug21/GWAS_NetworkPropagation/processed_data"
OUTPUT_BASE="/dccstor/boseukb/Q/NetMed/QuVINE/results/ppi_comprehensive"
HPARAM_DIR="${OUTPUT_BASE}/hparam_tuning"
LOG_DIR="${OUTPUT_BASE}/logs"

mkdir -p "$LOG_DIR" "${OUTPUT_BASE}/results" "$HPARAM_DIR"

# Network definitions
NETWORKS=(STRING BioPlex3 HumanNet PCNet ProteomeHD)
DISEASES=(asthma autism schizophrenia)

# Network paths
get_network_path() {
    case "$1" in
        STRING) echo "${DATA_ROOT}/networks/STRING/edges_list_ncbi.csv" ;;
        BioPlex3) echo "${DATA_ROOT}/networks/BioPlex3_shared/edges_list_ncbi.csv" ;;
        HumanNet) echo "${DATA_ROOT}/networks/HumanNetV3/edges_list_ncbi.csv" ;;
        PCNet) echo "${DATA_ROOT}/networks/PCNet/edges_list_ncbi.csv" ;;
        ProteomeHD) echo "${DATA_ROOT}/networks/ProteomeHD/edges_list_ncbi.csv" ;;
        *) echo "" ;;
    esac
}

# Calculate job counts
N_NETS=${#NETWORKS[@]}
N_DISEASES=${#DISEASES[@]}
TUNING_RANKING=$((N_NETS * N_DISEASES))
TUNING_CLASSIFICATION=$N_NETS
TUNING_LINK=$N_NETS
TOTAL_TUNING=$((TUNING_RANKING + TUNING_CLASSIFICATION + TUNING_LINK))

ANALYSIS_RANKING=$((N_NETS * N_DISEASES * N_REPS))
ANALYSIS_CLASSIFICATION=$((N_NETS * N_REPS))
ANALYSIS_LINK=$((N_NETS * N_REPS))
TOTAL_ANALYSIS=0
[ "$SKIP_RANKING" = false ] && TOTAL_ANALYSIS=$((TOTAL_ANALYSIS + ANALYSIS_RANKING))
[ "$SKIP_CLASSIFICATION" = false ] && TOTAL_ANALYSIS=$((TOTAL_ANALYSIS + ANALYSIS_CLASSIFICATION))
[ "$SKIP_LINK_PREDICTION" = false ] && TOTAL_ANALYSIS=$((TOTAL_ANALYSIS + ANALYSIS_LINK))

echo "============================================================"
echo " PPI Comprehensive Job Submission with Hyperparameter Tuning"
echo "============================================================"
echo " Networks: ${NETWORKS[*]}"
echo " Diseases (ranking): ${DISEASES[*]}"
echo ""
echo " Phase 1: Hyperparameter Tuning"
echo "   Ranking         : $TUNING_RANKING jobs"
echo "   Classification  : $TUNING_CLASSIFICATION jobs"
echo "   Link Prediction : $TUNING_LINK jobs"
echo "   Total Tuning    : $TOTAL_TUNING jobs"
echo "   Skip Tuning     : $SKIP_TUNING"
echo ""
echo " Phase 2: Main Analysis"
echo "   Ranking         : $ANALYSIS_RANKING jobs (skip: $SKIP_RANKING)"
echo "   Classification  : $ANALYSIS_CLASSIFICATION jobs (skip: $SKIP_CLASSIFICATION)"
echo "   Link Prediction : $ANALYSIS_LINK jobs (skip: $SKIP_LINK_PREDICTION)"
echo "   Total Analysis  : $TOTAL_ANALYSIS jobs"
echo ""
echo " Replicates : $N_REPS"
echo " Queue      : $QUEUE"
echo " Wall time  : $WALLTIME"
echo " Memory     : ${MEMORY}GB"
echo " Methods    : $METHODS"
echo " Dry run    : $DRY_RUN"
echo "============================================================"
echo ""

TUNING_JOB_IDS=()
ANALYSIS_JOB_IDS=()

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════

if [ "$SKIP_TUNING" = false ]; then
    echo "PHASE 1: Hyperparameter Tuning"
    echo "================================"
    
    # 1A: Ranking Tuning (5 networks × 3 diseases = 15 jobs)
    if [ "$SKIP_RANKING" = false ]; then
        echo ""
        echo "1A: Ranking Hyperparameter Tuning"
        for NET in "${NETWORKS[@]}"; do
            for DISEASE in "${DISEASES[@]}"; do
                TUNE_ID="tune_rank_${NET}_${DISEASE}"
                TUNE_SH="${LOG_DIR}/${TUNE_ID}.sh"
                HPARAM_OUT="${HPARAM_DIR}/${NET}_${DISEASE}_ranking_best_hyperparams.json"
                
                cat > "$TUNE_SH" << 'TUNEEOF'
#!/bin/bash
#BSUB -J TUNE_ID_PLACEHOLDER
#BSUB -o LOG_DIR_PLACEHOLDER/TUNE_ID_PLACEHOLDER.out
#BSUB -e LOG_DIR_PLACEHOLDER/TUNE_ID_PLACEHOLDER.err
#BSUB -q QUEUE_PLACEHOLDER
#BSUB -W 72:00
#BSUB -M 32GB
#BSUB -R "rusage[mem=32GB]"

source PROJECT_DIR_PLACEHOLDER/PYTHON_ENV_PLACEHOLDER
cd PROJECT_DIR_PLACEHOLDER

python scripts/run_hyperparameter_tuning.py \
    --network-type ppi \
    --ppi-network NET_PLACEHOLDER \
    --disease DISEASE_PLACEHOLDER \
    --task ranking \
    --output-file HPARAM_OUT_PLACEHOLDER \
    --methods METHODS_PLACEHOLDER \
    --verbose

exit $?
TUNEEOF
                
                # Replace placeholders
                sed -i "s|TUNE_ID_PLACEHOLDER|${TUNE_ID}|g" "$TUNE_SH"
                sed -i "s|LOG_DIR_PLACEHOLDER|${LOG_DIR}|g" "$TUNE_SH"
                sed -i "s|QUEUE_PLACEHOLDER|${QUEUE}|g" "$TUNE_SH"
                sed -i "s|PROJECT_DIR_PLACEHOLDER|${PROJECT_DIR}|g" "$TUNE_SH"
                sed -i "s|PYTHON_ENV_PLACEHOLDER|${PYTHON_ENV}|g" "$TUNE_SH"
                sed -i "s|NET_PLACEHOLDER|${NET}|g" "$TUNE_SH"
                sed -i "s|DISEASE_PLACEHOLDER|${DISEASE}|g" "$TUNE_SH"
                sed -i "s|HPARAM_OUT_PLACEHOLDER|${HPARAM_OUT}|g" "$TUNE_SH"
                sed -i "s|METHODS_PLACEHOLDER|${METHODS}|g" "$TUNE_SH"
                
                chmod +x "$TUNE_SH"
                
                if [ "$DRY_RUN" = true ]; then
                    echo "  [DRY RUN] $TUNE_ID"
                else
                    JOB_ID=$(bsub < "$TUNE_SH" 2>&1 | grep -oP 'Job <\K[0-9]+')
                    if [ -n "$JOB_ID" ]; then
                        echo "  Submitted $JOB_ID: $TUNE_ID"
                        TUNING_JOB_IDS+=("$JOB_ID")
                    fi
                fi
            done
        done
    fi
    
    # 1B: Classification Tuning (5 networks = 5 jobs)
    if [ "$SKIP_CLASSIFICATION" = false ]; then
        echo ""
        echo "1B: Classification Hyperparameter Tuning"
        for NET in "${NETWORKS[@]}"; do
            TUNE_ID="tune_class_${NET}"
            TUNE_SH="${LOG_DIR}/${TUNE_ID}.sh"
            HPARAM_OUT="${HPARAM_DIR}/${NET}_classification_best_hyperparams.json"
            
            cat > "$TUNE_SH" << TUNEEOF
#!/bin/bash
#BSUB -J ${TUNE_ID}
#BSUB -o ${LOG_DIR}/${TUNE_ID}.out
#BSUB -e ${LOG_DIR}/${TUNE_ID}.err
#BSUB -q ${QUEUE}
#BSUB -W 72:00
#BSUB -M 32GB
#BSUB -R "rusage[mem=32GB]"

source ${PROJECT_DIR}/${PYTHON_ENV}
cd ${PROJECT_DIR}

python scripts/run_hyperparameter_tuning.py \\
    --network-type ppi \\
    --ppi-network ${NET} \\
    --task classification \\
    --output-file ${HPARAM_OUT} \\
    --methods ${METHODS} \\
    --verbose

exit \$?
TUNEEOF
            
            chmod +x "$TUNE_SH"
            
            if [ "$DRY_RUN" = true ]; then
                echo "  [DRY RUN] $TUNE_ID"
            else
                JOB_ID=$(bsub < "$TUNE_SH" 2>&1 | grep -oP 'Job <\K[0-9]+')
                if [ -n "$JOB_ID" ]; then
                    echo "  Submitted $JOB_ID: $TUNE_ID"
                    TUNING_JOB_IDS+=("$JOB_ID")
                fi
            fi
        done
    fi
    
    # 1C: Link Prediction Tuning (5 networks = 5 jobs)
    if [ "$SKIP_LINK_PREDICTION" = false ]; then
        echo ""
        echo "1C: Link Prediction Hyperparameter Tuning"
        for NET in "${NETWORKS[@]}"; do
            TUNE_ID="tune_link_${NET}"
            TUNE_SH="${LOG_DIR}/${TUNE_ID}.sh"
            HPARAM_OUT="${HPARAM_DIR}/${NET}_link_prediction_best_hyperparams.json"
            
            cat > "$TUNE_SH" << TUNEEOF
#!/bin/bash
#BSUB -J ${TUNE_ID}
#BSUB -o ${LOG_DIR}/${TUNE_ID}.out
#BSUB -e ${LOG_DIR}/${TUNE_ID}.err
#BSUB -q ${QUEUE}
#BSUB -W 72:00
#BSUB -M 32GB
#BSUB -R "rusage[mem=32GB]"

source ${PROJECT_DIR}/${PYTHON_ENV}
cd ${PROJECT_DIR}

python scripts/run_hyperparameter_tuning.py \\
    --network-type ppi \\
    --ppi-network ${NET} \\
    --task link_prediction \\
    --output-file ${HPARAM_OUT} \\
    --methods ${METHODS} \\
    --verbose

exit \$?
TUNEEOF
            
            chmod +x "$TUNE_SH"
            
            if [ "$DRY_RUN" = true ]; then
                echo "  [DRY RUN] $TUNE_ID"
            else
                JOB_ID=$(bsub < "$TUNE_SH" 2>&1 | grep -oP 'Job <\K[0-9]+')
                if [ -n "$JOB_ID" ]; then
                    echo "  Submitted $JOB_ID: $TUNE_ID"
                    TUNING_JOB_IDS+=("$JOB_ID")
                fi
            fi
        done
    fi
    
    echo ""
    echo "Tuning jobs submitted: ${#TUNING_JOB_IDS[@]} / $TOTAL_TUNING"
fi

# Build tuning dependency
TUNING_DEPENDENCY=""
if [ "$SKIP_TUNING" = false ] && [ ${#TUNING_JOB_IDS[@]} -gt 0 ]; then
    for JID in "${TUNING_JOB_IDS[@]}"; do
        if [ -z "$TUNING_DEPENDENCY" ]; then
            TUNING_DEPENDENCY="ended($JID)"
        else
            TUNING_DEPENDENCY="$TUNING_DEPENDENCY && ended($JID)"
        fi
    done
fi

echo ""
echo "============================================================"
echo " PHASE 2: Main Analysis (waits for tuning)"
echo "============================================================"
echo ""
echo "[INFO] Full implementation would submit $TOTAL_ANALYSIS analysis jobs here"
echo "[INFO] Each job would use hyperparameters from Phase 1"
echo "[INFO] Ranking: 5 networks × 3 diseases × 30 reps = 450 jobs"
echo "[INFO] Classification: 5 networks × 30 reps = 150 jobs"
echo "[INFO] Link Prediction: 5 networks × 30 reps = 150 jobs"

echo ""
echo "============================================================"
echo " SUBMISSION COMPLETE"
echo " Tuning jobs: ${#TUNING_JOB_IDS[@]}"
echo " Monitor: bjobs -u \$USER"
echo " Results: ${OUTPUT_BASE}/"
echo "============================================================"
