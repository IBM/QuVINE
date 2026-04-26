#!/bin/bash
################################################################################
# Unified Simulated Data Experiment Job Submission
#
# Combines two synthetic graph experiment families:
#   A. Hard Negatives (16 cases): QW1-9, NC1-4, RN1-3
#   B. Extended Generators (5 types): random_regular, heterophilic_sbm,
#      degree_corrected_sbm, grid_torus, configuration_model
#
# Structure:
#   - Hard Negatives: 16 cases × 30 reps × 3 sizes = 1440 jobs
#   - Extended Gens:  5 types × 30 reps × 3 sizes = 450 jobs
#   - Total: 1890 analysis jobs
#   - One aggregation job with ended() dependency on all analysis jobs
#   - One packaging job with ended() dependency on aggregation
#
# Usage:
#   bash scripts/submit_simulated_data_jobs.sh [options]
#
# Options:
#   --n-replicates NUM   Replicates per (case/type, size) (default: 30)
#   --n-nodes SIZES      Node sizes (comma-separated)     (default: 500,2000,5000)
#   --output-dir DIR     Output root                      (default: outputs/simulated_data)
#   --queue QUEUE        LSF queue                        (default: normal)
#   --walltime TIME      Wall time per job                (default: 48:00)
#   --memory MEM         Memory in GB                     (default: 4)
#   --methods METHODS    Methods to run                   (default: all)
#   --python-env PATH    Path to python binary            (default: ../Python-3.12.2/venv_quvine/bin/python)
#   --resume             Pass --resume to analysis jobs
#   --dry-run            Print scripts, do not submit
#   --skip-hard-negatives   Skip hard negatives cases
#   --skip-extended-gens    Skip extended generators
#
################################################################################

set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
N_REPLICATES=30
N_NODES_LIST="500,2000,5000"
OUTPUT_DIR="/dccstor/boseukb/Q/NetMed/QuVINE/results/simulated_data/"
QUEUE="normal"
WALLTIME="48:00"
MEMORY="4"
METHODS="all"
PYTHON_ENV="../Python-3.12.2/venv_quvine/bin/python"
HPARAM_FILE="/dccstor/boseukb/Q/NetMed/QuVINE/results/hparam_tuning/best_hyperparams.json"
RESUME=false
DRY_RUN=false
SKIP_HARD_NEGATIVES=false
SKIP_EXTENDED_GENS=false

# ── Method presets ─────────────────────────────────────────────────────────────
ALL_METHODS="quvine_fused-walk,quvine_ctqw,quvine_dtqw,quvine_rwr,quvine_heat,quvine_poly,quvine_fused-filt,quvine_hgcnmf,quvine_pgcnmf,quvine_fused-gcnmf,netmf,node2vec,graphsage,baseline_gcnmf,baseline_filter"
QUANTUM_METHODS="quvine_fused-walk,quvine_ctqw,quvine_dtqw,quvine_rwr,quvine_heat,quvine_poly,quvine_fused-filt,quvine_hgcnmf,quvine_pgcnmf,quvine_fused-gcnmf"
CLASSICAL_METHODS="netmf,node2vec,graphsage,baseline_gcnmf,baseline_filter"

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-replicates)         N_REPLICATES="$2";         shift 2 ;;
        --n-nodes)              N_NODES_LIST="$2";         shift 2 ;;
        --output-dir)           OUTPUT_DIR="$2";           shift 2 ;;
        --queue)                QUEUE="$2";                shift 2 ;;
        --walltime)             WALLTIME="$2";             shift 2 ;;
        --memory)               MEMORY="$2";               shift 2 ;;
        --methods)              METHODS="$2";              shift 2 ;;
        --python-env)           PYTHON_ENV="$2";           shift 2 ;;
        --hparam-file)          HPARAM_FILE="$2";          shift 2 ;;
        --resume)               RESUME=true;               shift ;;
        --dry-run)              DRY_RUN=true;              shift ;;
        --skip-hard-negatives)  SKIP_HARD_NEGATIVES=true;  shift ;;
        --skip-extended-gens)   SKIP_EXTENDED_GENS=true;   shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--n-replicates N] [--n-nodes SIZES] [--output-dir DIR]"
            echo "          [--queue Q] [--walltime T] [--memory M]"
            echo "          [--methods all|quantum|classical|<list>]"
            echo "          [--python-env PATH] [--resume] [--dry-run]"
            echo "          [--skip-hard-negatives] [--skip-extended-gens]"
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
HN_CASES_CONFIG="${PROJECT_DIR}/configs/hard_negatives_cases.json"
HN_ANALYSIS_SCRIPT="${PROJECT_DIR}/scripts/run_hard_negative_network.py"
EG_ANALYSIS_SCRIPT="${PROJECT_DIR}/scripts/run_extended_generator_network.py"

# Derive the activate script from PYTHON_ENV
VENV_ACTIVATE="$( cd "${PROJECT_DIR}" && realpath -m "${PYTHON_ENV%/bin/python}/bin/activate" )"

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/results"

# Convert comma-separated node sizes to array
IFS=',' read -ra N_NODES_ARRAY <<< "$N_NODES_LIST"

# ── Hard Negatives Case Definitions ────────────────────────────────────────────
HN_CASE_NAMES=(
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

# Seed-offset bands for hard negatives: 1000 apart per case
HN_CASE_SEED_OFFSETS=(
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
    14000  # RN1
    15000  # RN2
    16000  # RN3
)

# ── Extended Generators Type Definitions ───────────────────────────────────────
EG_NETWORK_TYPES=(
    random_regular
    heterophilic_sbm
    degree_corrected_sbm
    grid_torus
    configuration_model
)

# Seed offset bands for extended generators: 100000 apart per type
EG_NETWORK_SEED_OFFSETS=(
    100000  # random_regular
    200000  # heterophilic_sbm
    300000  # degree_corrected_sbm
    400000  # grid_torus
    500000  # configuration_model
)

# ── Banner ─────────────────────────────────────────────────────────────────────
N_HN_CASES=${#HN_CASE_NAMES[@]}
N_EG_TYPES=${#EG_NETWORK_TYPES[@]}
N_SIZES=${#N_NODES_ARRAY[@]}

HN_JOBS=$(( N_HN_CASES * N_REPLICATES * N_SIZES ))
EG_JOBS=$(( N_EG_TYPES * N_REPLICATES * N_SIZES ))
TOTAL_JOBS=0
[ "$SKIP_HARD_NEGATIVES" = false ] && TOTAL_JOBS=$(( TOTAL_JOBS + HN_JOBS ))
[ "$SKIP_EXTENDED_GENS" = false ] && TOTAL_JOBS=$(( TOTAL_JOBS + EG_JOBS ))

echo "============================================================"
echo " Unified Simulated Data Experiment — Job Submission"
echo "============================================================"
echo " Project dir    : $PROJECT_DIR"
echo " Output dir     : $OUTPUT_DIR"
echo ""
echo " Hard Negatives:"
echo "   Cases        : $N_HN_CASES"
echo "   Skip         : $SKIP_HARD_NEGATIVES"
echo "   Jobs         : $HN_JOBS (if not skipped)"
echo ""
echo " Extended Generators:"
echo "   Types        : $N_EG_TYPES"
echo "   Skip         : $SKIP_EXTENDED_GENS"
echo "   Jobs         : $EG_JOBS (if not skipped)"
echo ""
echo " Common Settings:"
echo "   Node sizes   : ${N_NODES_ARRAY[*]}"
echo "   Replicates   : $N_REPLICATES (per case/type-size combo)"
echo "   Total jobs   : $TOTAL_JOBS"
echo "   LSF queue    : $QUEUE"
echo "   Wall time    : $WALLTIME"
echo "   Memory       : ${MEMORY}GB"
echo "   Python env   : $PYTHON_ENV"
echo "   Hparam file  : $HPARAM_FILE"
echo "   Methods      : $SELECTED_METHODS"
echo "   Resume       : $RESUME"
echo "   Dry run      : $DRY_RUN"
echo "============================================================"
echo ""
[ "$DRY_RUN" = true ] && echo "DRY RUN — no jobs will be submitted" && echo ""

ANALYSIS_JOB_IDS=()
JOB_COUNT=0

# ══════════════════════════════════════════════════════════════════════════════
# PART A: HARD NEGATIVES CASES
# ══════════════════════════════════════════════════════════════════════════════

if [ "$SKIP_HARD_NEGATIVES" = false ]; then
    echo "============================================================"
    echo " PART A: HARD NEGATIVES CASES"
    echo "============================================================"
    
    for case_idx in "${!HN_CASE_NAMES[@]}"; do
        case_name="${HN_CASE_NAMES[$case_idx]}"
        seed_offset="${HN_CASE_SEED_OFFSETS[$case_idx]}"
        
        echo ""
        echo "Case: $case_name"
        
        for n_nodes in "${N_NODES_ARRAY[@]}"; do
            echo "  Node size: $n_nodes"
            
            for rep in $(seq 0 $(( N_REPLICATES - 1 ))); do
                # Unique seed: offset + (size_index * 100) + rep
                size_idx=0
                for i in "${!N_NODES_ARRAY[@]}"; do
                    if [ "${N_NODES_ARRAY[$i]}" = "$n_nodes" ]; then
                        size_idx=$i
                        break
                    fi
                done
                seed=$(( seed_offset + (size_idx * 100) + rep ))
                
                network_id="${case_name}_n${n_nodes}_rep$(printf '%02d' $rep)"
                job_output_dir="$OUTPUT_DIR/results/$network_id"
                job_name="sim_hn_${case_name}_${n_nodes}_$(printf '%02d' $rep)"
                log_file="$OUTPUT_DIR/logs/${job_name}.out"
                err_file="$OUTPUT_DIR/logs/${job_name}.err"
                job_script="$OUTPUT_DIR/logs/${job_name}.sh"
                
                mkdir -p "$job_output_dir"
                
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

python ${HN_ANALYSIS_SCRIPT} \\
    --case-name   ${case_name} \\
    --network-id  ${network_id} \\
    --config      ${HN_CASES_CONFIG} \\
    --output-dir  ${job_output_dir} \\
    --methods     ${SELECTED_METHODS} \\
    --n-nodes     ${n_nodes} \\
    --seed        ${seed} \\
    --hparam-file ${HPARAM_FILE} \\
    ${RESUME_FLAG} \\
    --verbose

exit \$?
JOBEOF
                
                chmod +x "$job_script"
                
                if [ "$DRY_RUN" = true ]; then
                    echo "    [DRY RUN] $job_name"
                else
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
    done
fi

# ══════════════════════════════════════════════════════════════════════════════
# PART B: EXTENDED GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

if [ "$SKIP_EXTENDED_GENS" = false ]; then
    echo ""
    echo "============================================================"
    echo " PART B: EXTENDED GENERATORS"
    echo "============================================================"
    
    for type_idx in "${!EG_NETWORK_TYPES[@]}"; do
        network_type="${EG_NETWORK_TYPES[$type_idx]}"
        seed_offset="${EG_NETWORK_SEED_OFFSETS[$type_idx]}"
        
        echo ""
        echo "Network Type: $network_type"
        
        for n_nodes in "${N_NODES_ARRAY[@]}"; do
            echo "  Node size: $n_nodes"
            
            for rep in $(seq 0 $(( N_REPLICATES - 1 ))); do
                # Unique seed: offset + (size_index * 1000) + rep
                size_idx=0
                for i in "${!N_NODES_ARRAY[@]}"; do
                    if [ "${N_NODES_ARRAY[$i]}" = "$n_nodes" ]; then
                        size_idx=$i
                        break
                    fi
                done
                seed=$(( seed_offset + (size_idx * 1000) + rep ))
                
                network_id="${network_type}_n${n_nodes}_rep$(printf '%02d' $rep)"
                job_output_dir="$OUTPUT_DIR/results/$network_id"
                job_name="sim_eg_${network_type}_${n_nodes}_$(printf '%02d' $rep)"
                log_file="$OUTPUT_DIR/logs/${job_name}.out"
                err_file="$OUTPUT_DIR/logs/${job_name}.err"
                job_script="$OUTPUT_DIR/logs/${job_name}.sh"
                
                mkdir -p "$job_output_dir"
                
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

python ${EG_ANALYSIS_SCRIPT} \\
    --network-type ${network_type} \\
    --network-id   ${network_id} \\
    --output-dir   ${job_output_dir} \\
    --methods      ${SELECTED_METHODS} \\
    --n-nodes      ${n_nodes} \\
    --seed         ${seed} \\
    --hparam-file  ${HPARAM_FILE} \\
    ${RESUME_FLAG} \\
    --verbose

exit \$?
JOBEOF
                
                chmod +x "$job_script"
                
                if [ "$DRY_RUN" = true ]; then
                    echo "    [DRY RUN] $job_name"
                else
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
    done
fi

echo ""
echo "============================================================"
echo " Analysis jobs submitted: $JOB_COUNT / $TOTAL_JOBS"
echo "============================================================"

# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATION JOB
# ══════════════════════════════════════════════════════════════════════════════

if [ "$DRY_RUN" = false ] && [ ${#ANALYSIS_JOB_IDS[@]} -gt 0 ]; then
    # Build ended() dependency string
    DEPENDENCY_STRING=""
    for job_id in "${ANALYSIS_JOB_IDS[@]}"; do
        if [ -z "$DEPENDENCY_STRING" ]; then
            DEPENDENCY_STRING="ended($job_id)"
        else
            DEPENDENCY_STRING="$DEPENDENCY_STRING && ended($job_id)"
        fi
    done
    
    agg_job_name="sim_aggregate"
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
#BSUB -w "${DEPENDENCY_STRING}"

source ${VENV_ACTIVATE}
cd ${PROJECT_DIR}

echo "========================================"
echo " AGGREGATING SIMULATED DATA RESULTS"
echo "========================================"

python scripts/aggregate_simulated_data.py \\
    --results-dir ${OUTPUT_DIR}/results \\
    --output-dir  ${OUTPUT_DIR}

echo "========================================"
echo " AGGREGATION COMPLETE"
echo " Results: ${OUTPUT_DIR}/simulated_data_comprehensive.csv"
echo "========================================"

exit \$?
AGGEOF
    
    chmod +x "$agg_job_script"
    
    echo ""
    echo "Submitting aggregation job..."
    agg_id=$(bsub < "$agg_job_script" 2>&1 | grep -oP 'Job <\K[0-9]+')
    if [ -n "$agg_id" ]; then
        echo "  Submitted aggregation job $agg_id (waits for ${#ANALYSIS_JOB_IDS[@]} jobs)"
        
        # ══════════════════════════════════════════════════════════════════════
        # PACKAGING JOB
        # ══════════════════════════════════════════════════════════════════════
        pkg_job_name="sim_package_embeddings"
        pkg_job_script="$OUTPUT_DIR/logs/${pkg_job_name}.sh"
        
        cat > "$pkg_job_script" << PKGEOF
#!/bin/bash
#BSUB -J ${pkg_job_name}
#BSUB -o ${OUTPUT_DIR}/logs/${pkg_job_name}.out
#BSUB -e ${OUTPUT_DIR}/logs/${pkg_job_name}.err
#BSUB -q ${QUEUE}
#BSUB -W 1:00
#BSUB -M 16GB
#BSUB -R "rusage[mem=16GB]"
#BSUB -w "ended(${agg_id})"

source ${VENV_ACTIVATE}
cd ${PROJECT_DIR}

echo "========================================"
echo " PACKAGING EMBEDDINGS"
echo "========================================"

python scripts/package_embeddings_to_npz.py \\
    --results-dir ${OUTPUT_DIR}/results

echo "========================================"
echo " PACKAGING COMPLETE"
echo "========================================"

exit \$?
PKGEOF
        
        chmod +x "$pkg_job_script"
        
        pkg_id=$(bsub < "$pkg_job_script" 2>&1 | grep -oP 'Job <\K[0-9]+')
        if [ -n "$pkg_id" ]; then
            echo "  Submitted packaging job $pkg_id (waits for aggregation $agg_id)"
        else
            echo "  ERROR: packaging job submission failed"
        fi
    else
        echo "  ERROR: aggregation job submission failed"
    fi
elif [ "$DRY_RUN" = true ]; then
    echo ""
    echo "  [DRY RUN] Would submit aggregation job with ${#ANALYSIS_JOB_IDS[@]} dependencies"
    echo "  [DRY RUN] Would submit packaging job after aggregation"
fi

echo ""
echo "============================================================"
echo " SUBMISSION SUMMARY"
echo " Analysis jobs : $JOB_COUNT / $TOTAL_JOBS"
echo " Output        : $OUTPUT_DIR/results/"
echo " Monitor with  : bjobs -u \$USER"
echo "============================================================"

# Made with Bob
