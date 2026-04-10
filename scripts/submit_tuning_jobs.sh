#!/bin/bash
################################################################################
# Hyperparameter Tuning — LSF Job Submission
#
# Submits one LSF job per network type; each job calls
# scripts/tune_hyperparameters.py with Optuna TPE search (or random search
# if optuna is not installed).
#
# Structure:
#   - 14 synthetic network types  (100-node pilot, 3 seeds each)
#   - 5 real PPI networks          (BFS-subsampled to pilot size)
#   - 1 merge job (ended() dependency on all tuning jobs)
#     → merges per-network best_hyperparams.json into one final file
#
# NETWORK TYPES
# ─────────────
# Synthetic (from hard_negatives_v2 experiment):
#   erdos_renyi, watts_strogatz_high_p, watts_strogatz_low_p, random_geometric
#   modular_strong, modular_medium, modular_many_communities, core_periphery
#   scale_free, powerlaw_cluster, stochastic_block_model
#   real_karate, real_lesmis, real_polbooks
#
# Real PPI networks (subsampled to --pilot-nodes):
#   BioPlex3, HumanNet, PCNet, ProteomeHD, STRING
#
# Usage:
#   bash scripts/submit_tuning_jobs.sh [options]
#
# Options:
#   --output-dir   DIR    Output root               (default: results/hparam_tuning)
#   --n-trials     NUM    Optuna trials per method  (default: 30)
#   --pilot-nodes  NUM    Nodes in pilot graphs     (default: 100)
#   --pilot-seeds  NUM    Graph seeds per type      (default: 3)
#   --queue        Q      LSF queue                 (default: normal)
#   --walltime     T      Wall time per job         (default: 6:00)
#   --memory       MEM    Memory in GB              (default: 16)
#   --methods      LIST   Comma-separated methods   (default: all)
#                         Presets: all | quantum | classical
#   --python-env   PATH   Python binary path        (default: venv_quvine)
#   --skip-real           Skip real PPI networks
#   --synthetic-only      Alias for --skip-real
#   --dry-run             Print job scripts, do not submit
#   --merge-only          Only run the merge job (skip tuning submissions)
#
################################################################################

set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
OUTPUT_DIR="/dccstor/boseukb/Q/NetMed/QuVINE/results/hparam_tuning"
N_TRIALS=30
PILOT_NODES=100
SMALL_PILOT_NODES=50    # used for networks that timed out at 100 nodes
PILOT_SEEDS=3
QUEUE="normal"
WALLTIME="12:00"
MEMORY="2"
METHODS="all"
SKIP_REAL=false
DRY_RUN=false
MERGE_ONLY=false

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"
PYTHON_ENV="${PROJECT_DIR}/../Python-3.12.2/venv_quvine/bin/python3"

# ── Method presets ─────────────────────────────────────────────────────────────
ALL_METHODS="quvine_walks,baseline_filter_heat,baseline_filter_poly,baseline_gcnmf,node2vec,netmf,graphsage"
QUANTUM_METHODS="quvine_walks,baseline_filter_heat,baseline_filter_poly,baseline_gcnmf"
CLASSICAL_METHODS="node2vec,netmf,graphsage"

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --n-trials)     N_TRIALS="$2";     shift 2 ;;
        --pilot-nodes)       PILOT_NODES="$2";       shift 2 ;;
        --small-pilot-nodes) SMALL_PILOT_NODES="$2"; shift 2 ;;
        --pilot-seeds)  PILOT_SEEDS="$2";  shift 2 ;;
        --queue)        QUEUE="$2";        shift 2 ;;
        --walltime)     WALLTIME="$2";     shift 2 ;;
        --memory)       MEMORY="$2";       shift 2 ;;
        --methods)      METHODS="$2";      shift 2 ;;
        --python-env)   PYTHON_ENV="$2";   shift 2 ;;
        --skip-real|--synthetic-only) SKIP_REAL=true; shift ;;
        --dry-run)      DRY_RUN=true;      shift ;;
        --merge-only)   MERGE_ONLY=true;   shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--output-dir DIR] [--n-trials N] [--pilot-nodes N]"
            echo "          [--small-pilot-nodes N] [--pilot-seeds N] [--queue Q]"
            echo "          [--walltime T] [--memory M]"
            echo "          [--methods all|quantum|classical|<list>]"
            echo "          [--python-env PATH] [--skip-real] [--dry-run] [--merge-only]"
            exit 1 ;;
    esac
done

# ── Resolve method preset ──────────────────────────────────────────────────────
case "$METHODS" in
    all)       SELECTED_METHODS="$ALL_METHODS"       ;;
    quantum)   SELECTED_METHODS="$QUANTUM_METHODS"   ;;
    classical) SELECTED_METHODS="$CLASSICAL_METHODS" ;;
    *)         SELECTED_METHODS="$METHODS"            ;;
esac

# Convert comma-separated methods to space-separated for argparse
METHODS_ARGS="${SELECTED_METHODS//,/ }"

# ── Derived paths ──────────────────────────────────────────────────────────────
TUNING_SCRIPT="${PROJECT_DIR}/scripts/tune_hyperparameters.py"
VENV_ACTIVATE="${PYTHON_ENV%/python3}/activate"
[ ! -f "$VENV_ACTIVATE" ] && VENV_ACTIVATE="${PYTHON_ENV%/bin/python3}/bin/activate"

mkdir -p "${OUTPUT_DIR}/logs"

# ── Network type arrays ────────────────────────────────────────────────────────
SYNTHETIC_TYPES=(
    erdos_renyi
    watts_strogatz_high_p
    watts_strogatz_low_p
    random_geometric
    modular_strong
    modular_medium
    modular_many_communities
    core_periphery
    scale_free
    powerlaw_cluster
    stochastic_block_model
    real_karate
    real_lesmis
    real_polbooks
)

# Networks that timed out at 100 pilot nodes (GCNMF too slow) → use SMALL_PILOT_NODES
SLOW_SYNTHETIC_TYPES=(
    erdos_renyi
    watts_strogatz_high_p
    watts_strogatz_low_p
    modular_strong
    modular_medium
    modular_many_communities
    scale_free
    powerlaw_cluster
    stochastic_block_model
)

REAL_NETWORKS=(
    BioPlex3
    HumanNet
    PCNet
    ProteomeHD
    STRING
)

# ── Banner ─────────────────────────────────────────────────────────────────────
N_SYNTHETIC=${#SYNTHETIC_TYPES[@]}
N_REAL=0
[ "$SKIP_REAL" = false ] && N_REAL=${#REAL_NETWORKS[@]}
TOTAL_JOBS=$(( N_SYNTHETIC + N_REAL ))

echo "============================================================"
echo " Hyperparameter Tuning — Job Submission"
echo "============================================================"
echo " Project dir    : ${PROJECT_DIR}"
echo " Tuning script  : ${TUNING_SCRIPT}"
echo " Output dir     : ${OUTPUT_DIR}"
echo " Synthetic types: ${N_SYNTHETIC}"
echo " Real networks  : ${N_REAL}"
echo " Total jobs     : ${TOTAL_JOBS}  (+1 merge)"
echo " Trials/method  : ${N_TRIALS}"
echo " Pilot nodes    : ${PILOT_NODES}  (slow nets: ${SMALL_PILOT_NODES})"
echo " Pilot seeds    : ${PILOT_SEEDS}"
echo " LSF queue      : ${QUEUE}"
echo " Wall time      : ${WALLTIME}"
echo " Memory         : ${MEMORY}GB"
echo " Python env     : ${PYTHON_ENV}"
echo " Methods        : ${SELECTED_METHODS}"
echo " Skip real nets : ${SKIP_REAL}"
echo " Dry run        : ${DRY_RUN}"
echo " Merge only     : ${MERGE_ONLY}"
echo "============================================================"
echo ""
[ "$DRY_RUN" = true ] && echo "DRY RUN — no jobs will be submitted" && echo ""

TUNING_JOB_IDS=()
JOB_COUNT=0

# ── _submit_tuning_job <label> <type_arg> <skip_real_flag> [pilot_nodes] ───────
# label          : job name and per-type output subdir
# type_arg       : --network-type value (e.g. "erdos_renyi" or "real_BioPlex3")
# skip_real_flag : "--skip-real" or ""
# pilot_nodes    : override for this job (defaults to global PILOT_NODES)
_submit_tuning_job() {
    local label="$1"
    local type_arg="$2"          # e.g. "erdos_renyi" or "real_BioPlex3"
    local skip_real_flag="$3"    # "--skip-real" or ""
    local job_pilot_nodes="${4:-${PILOT_NODES}}"  # per-job override

    local job_name="tune_${label}"
    local job_out_dir="${OUTPUT_DIR}/${label}"
    local log_file="${OUTPUT_DIR}/logs/${job_name}.out"
    local err_file="${OUTPUT_DIR}/logs/${job_name}.err"
    local job_script="${OUTPUT_DIR}/logs/${job_name}.sh"

    mkdir -p "${job_out_dir}"

    cat > "${job_script}" << JOBEOF
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

echo "========================================"
echo " Tuning: ${label}"
echo " Output : ${job_out_dir}"
echo " Started: \$(date)"
echo "========================================"

${PYTHON_ENV} ${TUNING_SCRIPT} \\
    --network-type ${type_arg} \\
    --output-dir   ${job_out_dir} \\
    --n-trials     ${N_TRIALS} \\
    --pilot-nodes  ${job_pilot_nodes} \\
    --pilot-seeds  ${PILOT_SEEDS} \\
    --methods      ${METHODS_ARGS} \\
    ${skip_real_flag}

echo "========================================"
echo " Finished: ${label}  at \$(date)"
echo "========================================"

exit \$?
JOBEOF

    chmod +x "${job_script}"

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] bsub < ${job_script}"
    else
        local bsub_out job_id
        bsub_out=$(bsub < "${job_script}" 2>&1) || true
        job_id=$(echo "$bsub_out" | grep -oP 'Job <\K[0-9]+') || true
        if [ -n "$job_id" ]; then
            echo "  Submitted ${job_id}: ${job_name}"
            TUNING_JOB_IDS+=("$job_id")
            JOB_COUNT=$(( JOB_COUNT + 1 ))
        else
            echo "  ERROR: submission failed for ${job_name}: ${bsub_out}"
        fi
    fi
}

# ── Helper: check if a network type is in the slow list ───────────────────────
_is_slow_network() {
    local net="$1"
    for slow in "${SLOW_SYNTHETIC_TYPES[@]}"; do
        [ "$slow" = "$net" ] && return 0
    done
    return 1
}

# ── Helper: check if a network already has best_hyperparams.json ──────────────
_is_done() {
    local label="$1"
    [ -f "${OUTPUT_DIR}/${label}/best_hyperparams.json" ]
}

# ── Submit synthetic network jobs ──────────────────────────────────────────────
if [ "$MERGE_ONLY" = false ]; then
    echo "── Synthetic network types ──────────────────────────────────"
    for net_type in "${SYNTHETIC_TYPES[@]}"; do
        if _is_done "${net_type}"; then
            echo "  [SKIP] ${net_type} (already done)"
            continue
        fi
        if _is_slow_network "${net_type}"; then
            _submit_tuning_job "${net_type}" "${net_type}" "--skip-real" "${SMALL_PILOT_NODES}"
        else
            _submit_tuning_job "${net_type}" "${net_type}" "--skip-real" "${PILOT_NODES}"
        fi
    done

    # ── Submit real network jobs ───────────────────────────────────────────────
    if [ "$SKIP_REAL" = false ]; then
        echo ""
        echo "── Real PPI networks ────────────────────────────────────────"
        for net_name in "${REAL_NETWORKS[@]}"; do
            if _is_done "real_${net_name}"; then
                echo "  [SKIP] real_${net_name} (already done)"
                continue
            fi
            _submit_tuning_job "real_${net_name}" "real_${net_name}" ""
        done
    fi

    echo ""
    echo "============================================================"
    echo " Tuning jobs submitted: ${JOB_COUNT} / ${TOTAL_JOBS}"
    echo "============================================================"
fi

# ── Build ended() dependency string ───────────────────────────────────────────
DEPENDENCY_STRING=""
for job_id in "${TUNING_JOB_IDS[@]}"; do
    if [ -z "$DEPENDENCY_STRING" ]; then
        DEPENDENCY_STRING="ended(${job_id})"
    else
        DEPENDENCY_STRING="${DEPENDENCY_STRING} && ended(${job_id})"
    fi
done

# ── Merge job — combines all per-network best_hyperparams.json files ───────────
merge_job_name="tune_merge"
merge_job_script="${OUTPUT_DIR}/logs/${merge_job_name}.sh"

cat > "${merge_job_script}" << MERGEEOF
#!/bin/bash
#BSUB -J ${merge_job_name}
#BSUB -o ${OUTPUT_DIR}/logs/${merge_job_name}.out
#BSUB -e ${OUTPUT_DIR}/logs/${merge_job_name}.err
#BSUB -q ${QUEUE}
#BSUB -W 0:30
#BSUB -M 8GB
#BSUB -R "rusage[mem=8GB]"
$([ -n "$DEPENDENCY_STRING" ] && echo "#BSUB -w \"${DEPENDENCY_STRING}\"")

source ${VENV_ACTIVATE}
cd ${PROJECT_DIR}

echo "========================================"
echo " Merging hyperparameter tuning results"
echo " Started: \$(date)"
echo "========================================"

${PYTHON_ENV} - << 'PYEOF'
import json
import sys
from pathlib import Path

output_root = Path("${OUTPUT_DIR}")
merged_best: dict = {}
merged_full: dict = {"synthetic": {}, "real": {}}
errors = []

# Collect all per-network-type result files
for subdir in sorted(output_root.iterdir()):
    if not subdir.is_dir() or subdir.name == "logs":
        continue
    json_file = subdir / "best_hyperparams.json"
    if not json_file.exists():
        errors.append(f"MISSING: {json_file}")
        continue
    try:
        with open(json_file) as f:
            data = json.load(f)
        net_type = subdir.name
        # Merge best_params section
        bp = data.get("best_params", {})
        for k, v in bp.items():
            if k not in merged_best:
                merged_best[k] = v
            else:
                merged_best[k].update(v)
        # Merge _full section
        full = data.get("_full", {})
        for category in ["synthetic", "real"]:
            for nt, nt_data in full.get(category, {}).items():
                merged_full[category][nt] = nt_data
        print(f"  Merged: {net_type}")
    except Exception as e:
        errors.append(f"ERROR reading {json_file}: {e}")

# Write final merged file
out_file = output_root / "best_hyperparams.json"
with open(out_file, "w") as f:
    json.dump({"_full": merged_full, "best_params": merged_best}, f, indent=2, default=str)

# Write a clean summary CSV
import csv
rows = []
for net_type, methods_data in merged_best.items():
    if net_type.startswith("_"):
        continue
    scores = methods_data.get("_scores", {})
    for method, params in methods_data.items():
        if method.startswith("_") or not isinstance(params, dict):
            continue
        rows.append({
            "network_type": net_type,
            "method": method,
            "score": scores.get(method, ""),
            **{f"param_{k}": v for k, v in params.items()},
        })

if rows:
    csv_file = output_root / "best_hyperparams_summary.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary CSV: {csv_file}")

print(f"\\nMerged {len(merged_best)} network types → {out_file}")
if errors:
    print(f"\\nWarnings ({len(errors)}):")
    for e in errors:
        print(f"  {e}")
sys.exit(len(errors) > 0)
PYEOF

echo "========================================"
echo " Merge complete at \$(date)"
echo " Output: ${OUTPUT_DIR}/best_hyperparams.json"
echo " Summary: ${OUTPUT_DIR}/best_hyperparams_summary.csv"
echo "========================================"

exit \$?
MERGEEOF

chmod +x "${merge_job_script}"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "  [DRY RUN] bsub < ${merge_job_script}"
    [ -n "$DEPENDENCY_STRING" ] && echo "  [DRY RUN] Dependencies: ${DEPENDENCY_STRING}"
elif [ "$MERGE_ONLY" = true ]; then
    echo ""
    echo "Submitting merge job (no dependency — merge-only mode)..."
    merge_id=$(bsub < "${merge_job_script}" 2>&1 | grep -oP 'Job <\K[0-9]+')
    if [ -n "$merge_id" ]; then
        echo "  Submitted merge job ${merge_id}"
    else
        echo "  ERROR: merge job submission failed"
    fi
elif [ ${#TUNING_JOB_IDS[@]} -gt 0 ]; then
    echo ""
    echo "Submitting merge job (waits for all tuning jobs)..."
    merge_id=$(bsub < "${merge_job_script}" 2>&1 | grep -oP 'Job <\K[0-9]+')
    if [ -n "$merge_id" ]; then
        echo "  Submitted merge job ${merge_id}  (waits for ${#TUNING_JOB_IDS[@]} jobs via ended())"
    else
        echo "  ERROR: merge job submission failed"
    fi
else
    echo "  No tuning jobs submitted — skipping merge job."
fi

echo ""
echo "============================================================"
echo " SUBMISSION SUMMARY"
echo " Tuning jobs  : ${JOB_COUNT} / ${TOTAL_JOBS}"
echo " Output root  : ${OUTPUT_DIR}/"
echo " Final params : ${OUTPUT_DIR}/best_hyperparams.json"
echo " Monitor with : bjobs -u \$USER"
echo " Kill all with: bkill -J 'tune_*'"
echo "============================================================"
