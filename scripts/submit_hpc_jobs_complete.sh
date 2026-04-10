#!/bin/bash
################################################################################
# Complete LSF Job Submission Script with Dataset Generation and Aggregation
#
# This script:
# 1. Submits jobs to generate networks (scale-free and modular)
# 2. Each job generates one network and runs complete analysis
# 3. Final aggregation job runs after all analysis jobs complete
# 4. Generates visualizations and recommendations
#
# Usage:
#   bash scripts/submit_hpc_jobs_complete.sh [options]
#
# Options:
#   --n-networks NUM     Number of networks per type (default: 20)
#   --n-nodes NUM        Number of nodes per network (default: 200)
#   --output-dir DIR     Output directory (default: outputs/hpc_results)
#   --queue QUEUE        LSF queue name (default: normal)
#   --walltime TIME      Wall time for analysis jobs (default: 4:00)
#   --memory MEM         Memory per job in GB (default: 16)
#   --methods METHODS    Comma-separated list of methods (default: all)
#                        Available: quvine_fused, quvine_ctqw, quvine_dtqw, quvine_rwr,
#                                   quvine_heat, quvine_poly, quvine_hgcnmf, quvine_pgcnmf,
#                                   netmf, node2vec, baseline_gcnmf, baseline_filter
#                        Use 'all' for all methods, 'quantum' for QuVINE only, 'classical' for baselines
#   --dry-run            Print commands without submitting
#
# Author: QuVINE Team
# Date: 2026-04-02
################################################################################

set -e  # Exit on error

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
N_NETWORKS=40
N_NODES=200
OUTPUT_DIR="outputs/hpc_results"
QUEUE="normal"
WALLTIME="4:00"
MEMORY="8"
METHODS="all"
DRY_RUN=false
PYTHON_ENV="venv_quvine/bin/python"
ALL_NETWORK_TYPES=false   # set by --all-networks

# Available embedding methods
ALL_METHODS="quvine_fused,quvine_ctqw,quvine_dtqw,quvine_rwr,quvine_heat,quvine_poly,quvine_hgcnmf,quvine_pgcnmf,netmf,node2vec,baseline_gcnmf,baseline_filter"
QUANTUM_METHODS="quvine_fused,quvine_ctqw,quvine_dtqw,quvine_rwr,quvine_heat,quvine_poly,quvine_hgcnmf,quvine_pgcnmf"
CLASSICAL_METHODS="netmf,node2vec,baseline_gcnmf,baseline_filter"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-networks)
            N_NETWORKS="$2"
            shift 2
            ;;
        --n-nodes)
            N_NODES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --queue)
            QUEUE="$2"
            shift 2
            ;;
        --walltime)
            WALLTIME="$2"
            shift 2
            ;;
        --memory)
            MEMORY="$2"
            shift 2
            ;;
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --python-env)
            PYTHON_ENV="$2"
            shift 2
            ;;
        --all-networks)
            # Submit jobs for every network type defined in data/random_graphs.py:
            #   scale_free, modular, erdos_renyi, watts_strogatz,
            #   powerlaw_cluster, random_geometric, hierarchical, core_periphery
            ALL_NETWORK_TYPES=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --n-networks NUM     Networks per type            (default: 20)"
            echo "  --n-nodes NUM        Nodes per network            (default: 200)"
            echo "  --output-dir DIR     Output directory             (default: outputs/hpc_results)"
            echo "  --queue QUEUE        LSF queue                    (default: normal)"
            echo "  --walltime TIME      Per-job wall time            (default: 4:00)"
            echo "  --memory MEM         Memory per job in GB         (default: 16)"
            echo "  --methods METHODS    Comma-separated method list  (default: all)"
            echo "                       Presets: all | quantum | classical"
            echo "  --all-networks       Submit all 8 network types from random_graphs.py"
            echo "                       (default: scale_free + modular only)"
            echo "  --python-env PATH    Python executable path       (default: venv_quvine/bin/python)"
            echo "  --dry-run            Print job scripts without submitting"
            exit 1
            ;;
    esac
done

# Resolve method preset
case "$METHODS" in
    all)      SELECTED_METHODS="$ALL_METHODS"     ;;
    quantum)  SELECTED_METHODS="$QUANTUM_METHODS" ;;
    classical) SELECTED_METHODS="$CLASSICAL_METHODS" ;;
    *)        SELECTED_METHODS="$METHODS"          ;;
esac

# Determine which network types to generate
if [ "$ALL_NETWORK_TYPES" = true ]; then
    # All 8 types defined in data/random_graphs.py
    NETWORK_TYPES="scale_free modular erdos_renyi watts_strogatz powerlaw_cluster random_geometric hierarchical core_periphery"
    N_TYPES=8
else
    NETWORK_TYPES="scale_free modular"
    N_TYPES=2
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/results"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
TOTAL_NETWORKS=$((N_NETWORKS * N_TYPES))
echo "=================================="
echo "HPC Complete Pipeline Configuration"
echo "=================================="
echo "Project Directory : $PROJECT_DIR"
echo "Output Directory  : $OUTPUT_DIR"
echo "Network types     : $NETWORK_TYPES"
echo "Networks per type : $N_NETWORKS"
echo "Total networks    : $TOTAL_NETWORKS"
echo "Nodes per network : $N_NODES"
echo "LSF Queue         : $QUEUE"
echo "Wall Time         : $WALLTIME"
echo "Memory per Job    : ${MEMORY}GB"
echo "Python Env        : $PYTHON_ENV"
echo "Methods           : $SELECTED_METHODS"
echo "Dry Run           : $DRY_RUN"
echo "=================================="
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - No jobs will be submitted"
    echo ""
fi

# Array to store all analysis job IDs
ANALYSIS_JOB_IDS=()
JOB_COUNT=0

# ---------------------------------------------------------------------------
# Submit analysis jobs for each network type
# ---------------------------------------------------------------------------
# _submit_network_type_jobs NET_TYPE SEED_OFFSET
# Generates N_NETWORKS jobs for the given network type.
# Each job writes a self-contained bash/Python script and optionally submits it.
_submit_network_type_jobs() {
    local net_type="$1"
    local seed_offset="$2"

    echo ""
    echo "Submitting ${net_type} network jobs..."

    for i in $(seq 0 $((N_NETWORKS - 1))); do
        network_id=$(printf "${net_type}_%02d" $i)
        job_output_dir="$OUTPUT_DIR/results/$network_id"
        mkdir -p "$job_output_dir"

        job_name="quvine_${network_id}"
        log_file="$OUTPUT_DIR/logs/${job_name}.out"
        err_file="$OUTPUT_DIR/logs/${job_name}.err"
        job_script="$OUTPUT_DIR/logs/${job_name}.sh"

        seed=$(( seed_offset + i ))

        # ------------------------------------------------------------------
        # Build the Python network-generation snippet for this type.
        # We use a bash variable PY_GEN that is interpolated into the heredoc.
        # ------------------------------------------------------------------
        case "$net_type" in
            scale_free)
                local m=$(( 2 + (i % 4) ))
                PY_GEN="
from quvine.data.random_graphs import generate_barabasi_albert
print('Generating scale_free network: ${network_id}')
print('Parameters: n=${N_NODES}, m=${m}, seed=${seed}')
G = generate_barabasi_albert(n=${N_NODES}, m=${m}, seed=${seed})
metadata = {'type': 'scale_free', 'n_nodes': ${N_NODES}, 'm': ${m}, 'seed': ${seed}, 'network_id': '${network_id}'}
"
                ;;
            modular)
                local num_communities=$(( 3 + (i % 5) ))
                PY_GEN="
from quvine.data.random_graphs import generate_modular_network
import numpy as np
rng = np.random.default_rng(${seed})
num_communities = ${num_communities}
nodes_per_community = ${N_NODES} // num_communities
p_intra = 0.3 + float(rng.uniform(-0.1, 0.1))
p_inter = 0.01 + float(rng.uniform(-0.005, 0.005))
print(f'Generating modular network: ${network_id} (communities={num_communities}, p_intra={p_intra:.3f}, p_inter={p_inter:.4f})')
G, _ = generate_modular_network(num_communities=num_communities, nodes_per_community=nodes_per_community, p_intra=p_intra, p_inter=p_inter, seed=${seed})
metadata = {'type': 'modular', 'n_nodes': ${N_NODES}, 'num_communities': num_communities, 'p_intra': float(p_intra), 'p_inter': float(p_inter), 'seed': ${seed}, 'network_id': '${network_id}'}
"
                ;;
            erdos_renyi)
                # p from 0.05 to 0.14 across the N_NETWORKS instances
                PY_GEN="
from quvine.data.random_graphs import generate_erdos_renyi
p = 0.05 + (${i} % 10) * 0.01
print(f'Generating erdos_renyi network: ${network_id} (n=${N_NODES}, p={p:.3f}, seed=${seed})')
G = generate_erdos_renyi(n=${N_NODES}, p=p, seed=${seed})
metadata = {'type': 'erdos_renyi', 'n_nodes': ${N_NODES}, 'p': p, 'seed': ${seed}, 'network_id': '${network_id}'}
"
                ;;
            watts_strogatz)
                local k=$(( 4 + (i % 6) * 2 ))   # k in {4,6,8,10,12,14}
                PY_GEN="
from quvine.data.random_graphs import generate_watts_strogatz
p_rewire = 0.1 + (${i} / max(${N_NETWORKS} - 1, 1)) * 0.4
print(f'Generating watts_strogatz network: ${network_id} (n=${N_NODES}, k=${k}, p={p_rewire:.3f}, seed=${seed})')
G = generate_watts_strogatz(n=${N_NODES}, k=${k}, p=p_rewire, seed=${seed})
metadata = {'type': 'watts_strogatz', 'n_nodes': ${N_NODES}, 'k': ${k}, 'p_rewire': p_rewire, 'seed': ${seed}, 'network_id': '${network_id}'}
"
                ;;
            powerlaw_cluster)
                local m_plc=$(( 2 + (i % 4) ))
                PY_GEN="
from quvine.data.random_graphs import generate_powerlaw_cluster
p_tri = 0.1 + (${i} / max(${N_NETWORKS} - 1, 1)) * 0.3
print(f'Generating powerlaw_cluster network: ${network_id} (n=${N_NODES}, m=${m_plc}, p={p_tri:.3f}, seed=${seed})')
G = generate_powerlaw_cluster(n=${N_NODES}, m=${m_plc}, p=p_tri, seed=${seed})
metadata = {'type': 'powerlaw_cluster', 'n_nodes': ${N_NODES}, 'm': ${m_plc}, 'p_tri': p_tri, 'seed': ${seed}, 'network_id': '${network_id}'}
"
                ;;
            random_geometric)
                PY_GEN="
from quvine.data.random_graphs import generate_random_geometric
radius = 0.10 + (${i} / max(${N_NETWORKS} - 1, 1)) * 0.15
print(f'Generating random_geometric network: ${network_id} (n=${N_NODES}, radius={radius:.3f}, seed=${seed})')
G = generate_random_geometric(n=${N_NODES}, radius=radius, seed=${seed})
metadata = {'type': 'random_geometric', 'n_nodes': ${N_NODES}, 'radius': radius, 'seed': ${seed}, 'network_id': '${network_id}'}
"
                ;;
            hierarchical)
                local levels=$(( 4 + (i % 3) ))
                local bf=$(( 2 + (i % 3) ))
                PY_GEN="
from quvine.data.random_graphs import generate_hierarchical_network
p_lvl = 0.01 + (${i} / max(${N_NETWORKS} - 1, 1)) * 0.09
print(f'Generating hierarchical network: ${network_id} (levels=${levels}, bf=${bf}, p_level={p_lvl:.3f}, seed=${seed})')
G, _ = generate_hierarchical_network(levels=${levels}, branching_factor=${bf}, p_level=p_lvl, seed=${seed})
metadata = {'type': 'hierarchical', 'levels': ${levels}, 'branching_factor': ${bf}, 'p_level': p_lvl, 'seed': ${seed}, 'network_id': '${network_id}'}
"
                ;;
            core_periphery)
                PY_GEN="
from quvine.data.random_graphs import generate_core_periphery
n_core = max(10, int(${N_NODES} * (0.1 + (${i} / max(${N_NETWORKS} - 1, 1)) * 0.2)))
n_periphery = ${N_NODES} - n_core
p_core = 0.5 + (${i} / max(${N_NETWORKS} - 1, 1)) * 0.4
p_cp   = 0.05 + (${i} / max(${N_NETWORKS} - 1, 1)) * 0.10
print(f'Generating core_periphery network: ${network_id} (n_core={n_core}, n_periphery={n_periphery}, p_core={p_core:.2f}, seed=${seed})')
G, _, _ = generate_core_periphery(n_core=n_core, n_periphery=n_periphery, p_core=p_core, p_core_periphery=p_cp, seed=${seed})
metadata = {'type': 'core_periphery', 'n_core': n_core, 'n_periphery': n_periphery, 'p_core': p_core, 'p_cp': p_cp, 'seed': ${seed}, 'network_id': '${network_id}'}
"
                ;;
            *)
                echo "  WARNING: Unknown network type '${net_type}' — skipping"
                continue
                ;;
        esac

        cat > "$job_script" << EOF
#!/bin/bash
#BSUB -J ${job_name}
#BSUB -o ${log_file}
#BSUB -e ${err_file}
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -M ${MEMORY}GB
#BSUB -R "rusage[mem=${MEMORY}GB]"

source ${PROJECT_DIR}/${PYTHON_ENV%/bin/python}/bin/activate
cd ${PROJECT_DIR}

python -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}/src')
import networkx as nx

${PY_GEN}

# Ensure graph is connected — keep largest component
if not nx.is_connected(G):
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

print(f'Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')

from quvine.comprehensive_embedding_analysis import run_single_network_analysis
run_single_network_analysis(
    G=G,
    network_id='${network_id}',
    network_metadata=metadata,
    output_dir='${job_output_dir}',
    embedding_methods='${SELECTED_METHODS}'.split(','),
    verbose=True,
)
print('Analysis complete for ${network_id}')
"

exit \$?
EOF

        chmod +x "$job_script"

        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY RUN] Would submit: bsub < $job_script"
        else
            job_id=$(bsub < "$job_script" 2>&1 | grep -oP 'Job <\K[0-9]+')
            if [ -n "$job_id" ]; then
                echo "  Submitted job $job_id: $job_name"
                ANALYSIS_JOB_IDS+=("$job_id")
                JOB_COUNT=$((JOB_COUNT + 1))
            else
                echo "  ERROR: Failed to submit job for $network_id"
            fi
        fi
    done
}

# Each network type gets a distinct seed-offset band (1000 apart) so networks
# are independently reproducible and don't share seeds across types.
TYPE_IDX=0
for NET_TYPE in $NETWORK_TYPES; do
    SEED_OFFSET=$(( 42 + TYPE_IDX * 1000 ))
    _submit_network_type_jobs "$NET_TYPE" "$SEED_OFFSET"
    TYPE_IDX=$(( TYPE_IDX + 1 ))
done

echo ""
echo "=================================="
echo "Analysis Jobs Submitted: $JOB_COUNT"
echo "=================================="

# ---------------------------------------------------------------------------
# Submit aggregation + visualization job (depends on all analysis jobs)
# ---------------------------------------------------------------------------

agg_job_name="quvine_aggregate"
agg_job_script="$OUTPUT_DIR/logs/${agg_job_name}.sh"

# Build LSF dependency string
DEPENDENCY_STRING=""
for job_id in "${ANALYSIS_JOB_IDS[@]}"; do
    if [ -z "$DEPENDENCY_STRING" ]; then
        DEPENDENCY_STRING="done($job_id)"
    else
        DEPENDENCY_STRING="$DEPENDENCY_STRING && done($job_id)"
    fi
done

# Write the aggregation job script (no quoted heredoc — variables expand here)
cat > "$agg_job_script" << EOF
#!/bin/bash
#BSUB -J ${agg_job_name}
#BSUB -o ${OUTPUT_DIR}/logs/${agg_job_name}.out
#BSUB -e ${OUTPUT_DIR}/logs/${agg_job_name}.err
#BSUB -q ${QUEUE}
#BSUB -W 2:00
#BSUB -M 32GB
#BSUB -R "rusage[mem=32GB]"
$([ -n "$DEPENDENCY_STRING" ] && echo "#BSUB -w \"${DEPENDENCY_STRING}\"")

# Activate Python environment
source ${PROJECT_DIR}/${PYTHON_ENV%/bin/python}/bin/activate

# Change to project directory
cd ${PROJECT_DIR}

echo "========================================"
echo "AGGREGATING AND VISUALIZING RESULTS"
echo "========================================"

# Collect, aggregate, and generate all box-plot visualizations.
# collect_hpc_results.py handles both steps in one call.
python scripts/collect_hpc_results.py \\
    --results-dir ${OUTPUT_DIR}/results \\
    --viz-dir     ${OUTPUT_DIR}/visualizations \\
    --n-networks  ${N_NETWORKS}

echo "========================================"
echo "AGGREGATION AND VISUALIZATION COMPLETE"
echo "Results CSV : ${OUTPUT_DIR}/results/comprehensive_results.csv"
echo "Plots       : ${OUTPUT_DIR}/visualizations/"
echo "========================================"

exit \$?
EOF

chmod +x "$agg_job_script"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "  [DRY RUN] Would submit aggregation job: bsub < $agg_job_script"
    [ -n "$DEPENDENCY_STRING" ] && echo "  [DRY RUN] Dependencies: $DEPENDENCY_STRING"
elif [ ${#ANALYSIS_JOB_IDS[@]} -gt 0 ]; then
    echo ""
    echo "Submitting aggregation and visualization job..."
    agg_job_id=$(bsub < "$agg_job_script" 2>&1 | grep -oP 'Job <\K[0-9]+')
    if [ -n "$agg_job_id" ]; then
        echo "  Submitted aggregation job $agg_job_id: $agg_job_name"
        echo "  Depends on ${#ANALYSIS_JOB_IDS[@]} analysis jobs"
    else
        echo "  ERROR: Failed to submit aggregation job"
    fi
else
    echo ""
    echo "No analysis jobs were submitted — skipping aggregation job."
fi

echo ""
echo "=================================="
echo "SUBMISSION COMPLETE"
echo "=================================="
echo "Analysis jobs submitted: $JOB_COUNT"
if [ "$DRY_RUN" = false ] && [ ${#ANALYSIS_JOB_IDS[@]} -gt 0 ]; then
    echo "Aggregation job       : 1 (runs after all analysis jobs)"
    echo ""
    echo "Monitor jobs with:"
    echo "  bjobs -u \$USER"
    echo ""
    echo "Results will be in:"
    echo "  $OUTPUT_DIR/results/comprehensive_results.csv"
    echo "  $OUTPUT_DIR/visualizations/"
fi
echo "=================================="
