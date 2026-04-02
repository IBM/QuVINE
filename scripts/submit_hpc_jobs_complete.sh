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
#   --dry-run            Print commands without submitting
#
# Author: QuVINE Team
# Date: 2026-04-02
################################################################################

set -e  # Exit on error

# Default parameters
N_NETWORKS=20
N_NODES=200
OUTPUT_DIR="outputs/hpc_results"
QUEUE="normal"
WALLTIME="4:00"
MEMORY="16"
DRY_RUN=false
PYTHON_ENV="venv_quvine/bin/python"

# Parse command line arguments
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
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --python-env)
            PYTHON_ENV="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--n-networks NUM] [--n-nodes NUM] [--output-dir DIR] [--queue QUEUE] [--walltime TIME] [--memory MEM] [--dry-run]"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/results"

echo "=================================="
echo "HPC Complete Pipeline Configuration"
echo "=================================="
echo "Project Directory: $PROJECT_DIR"
echo "Output Directory:  $OUTPUT_DIR"
echo "Networks per type: $N_NETWORKS"
echo "Nodes per network: $N_NODES"
echo "LSF Queue:         $QUEUE"
echo "Wall Time:         $WALLTIME"
echo "Memory per Job:    ${MEMORY}GB"
echo "Python Env:        $PYTHON_ENV"
echo "Dry Run:           $DRY_RUN"
echo "=================================="
echo ""

# Calculate total networks
TOTAL_NETWORKS=$((N_NETWORKS * 2))  # scale-free + modular
echo "Total networks to generate and analyze: $TOTAL_NETWORKS"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - No jobs will be submitted"
    echo ""
fi

# Array to store all analysis job IDs
ANALYSIS_JOB_IDS=()
JOB_COUNT=0

# Submit scale-free network jobs
echo "Submitting scale-free network jobs..."
for i in $(seq 0 $((N_NETWORKS - 1))); do
    network_id=$(printf "scale_free_%02d" $i)
    job_output_dir="$OUTPUT_DIR/results/$network_id"
    mkdir -p "$job_output_dir"
    
    job_name="quvine_${network_id}"
    log_file="$OUTPUT_DIR/logs/${job_name}.out"
    err_file="$OUTPUT_DIR/logs/${job_name}.err"
    job_script="$OUTPUT_DIR/logs/${job_name}.sh"
    
    # Vary m parameter for diversity (2-5)
    m=$((2 + (i % 4)))
    seed=$((42 + i))
    
    cat > "$job_script" << EOF
#!/bin/bash
#BSUB -J ${job_name}
#BSUB -o ${log_file}
#BSUB -e ${err_file}
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -M ${MEMORY}GB
#BSUB -R "rusage[mem=${MEMORY}GB]"

# Activate Python environment
source ${PROJECT_DIR}/${PYTHON_ENV%/bin/python}/bin/activate

# Change to project directory
cd ${PROJECT_DIR}

# Run complete analysis: generate network + analyze
python -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}/src')

from quvine.data.random_graphs import generate_barabasi_albert
from quvine.comprehensive_embedding_analysis import run_single_network_analysis
import networkx as nx

# Generate scale-free network
print('Generating scale-free network: ${network_id}')
print('Parameters: n=${N_NODES}, m=${m}, seed=${seed}')
G = generate_barabasi_albert(n=${N_NODES}, m=${m}, seed=${seed})

# Ensure connected
if not nx.is_connected(G):
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

print(f'Network generated: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')

# Metadata
metadata = {
    'type': 'scale_free',
    'n_nodes': ${N_NODES},
    'm': ${m},
    'seed': ${seed},
    'network_id': '${network_id}'
}

# Run complete analysis
results = run_single_network_analysis(
    G=G,
    network_id='${network_id}',
    network_metadata=metadata,
    output_dir='${job_output_dir}',
    embedding_methods=['quvine_fused', 'quvine_rwr', 'quvine_ctqw', 'quvine_dtqw', 'netmf', 'node2vec'],
    verbose=True
)

print(f'Analysis complete for ${network_id}')
"

exit \$?
EOF
    
    chmod +x "$job_script"
    
    # Submit job
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

# Submit modular network jobs
echo ""
echo "Submitting modular network jobs..."
for i in $(seq 0 $((N_NETWORKS - 1))); do
    network_id=$(printf "modular_%02d" $i)
    job_output_dir="$OUTPUT_DIR/results/$network_id"
    mkdir -p "$job_output_dir"
    
    job_name="quvine_${network_id}"
    log_file="$OUTPUT_DIR/logs/${job_name}.out"
    err_file="$OUTPUT_DIR/logs/${job_name}.err"
    job_script="$OUTPUT_DIR/logs/${job_name}.sh"
    
    # Vary modularity parameters
    num_communities=$((3 + (i % 5)))  # 3-7 communities
    p_intra_base=0.3
    p_inter_base=0.01
    seed=$((1042 + i))
    
    cat > "$job_script" << EOF
#!/bin/bash
#BSUB -J ${job_name}
#BSUB -o ${log_file}
#BSUB -e ${err_file}
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -M ${MEMORY}GB
#BSUB -R "rusage[mem=${MEMORY}GB]"

# Activate Python environment
source ${PROJECT_DIR}/${PYTHON_ENV%/bin/python}/bin/activate

# Change to project directory
cd ${PROJECT_DIR}

# Run complete analysis: generate network + analyze
python -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}/src')

from quvine.data.random_graphs import generate_modular_network
from quvine.comprehensive_embedding_analysis import run_single_network_analysis
import networkx as nx
import numpy as np

# Set seed for reproducibility
np.random.seed(${seed})
rng = np.random.default_rng(${seed})

# Generate modular network
print('Generating modular network: ${network_id}')
num_communities = ${num_communities}
nodes_per_community = ${N_NODES} // num_communities
p_intra = ${p_intra_base} + rng.uniform(-0.1, 0.1)
p_inter = ${p_inter_base} + rng.uniform(-0.005, 0.005)

print(f'Parameters: communities={num_communities}, nodes_per_comm={nodes_per_community}')
print(f'            p_intra={p_intra:.3f}, p_inter={p_inter:.4f}, seed=${seed}')

G, communities = generate_modular_network(
    num_communities=num_communities,
    nodes_per_community=nodes_per_community,
    p_intra=p_intra,
    p_inter=p_inter,
    seed=${seed}
)

# Ensure connected
if not nx.is_connected(G):
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

print(f'Network generated: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')

# Metadata
metadata = {
    'type': 'modular',
    'n_nodes': ${N_NODES},
    'num_communities': num_communities,
    'nodes_per_community': nodes_per_community,
    'p_intra': float(p_intra),
    'p_inter': float(p_inter),
    'seed': ${seed},
    'network_id': '${network_id}'
}

# Run complete analysis
results = run_single_network_analysis(
    G=G,
    network_id='${network_id}',
    network_metadata=metadata,
    output_dir='${job_output_dir}',
    embedding_methods=['quvine_fused', 'quvine_rwr', 'quvine_ctqw', 'quvine_dtqw', 'netmf', 'node2vec'],
    verbose=True
)

print(f'Analysis complete for ${network_id}')
"

exit \$?
EOF
    
    chmod +x "$job_script"
    
    # Submit job
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

echo ""
echo "=================================="
echo "Analysis Jobs Submitted: $JOB_COUNT"
echo "=================================="

# Submit aggregation and visualization job (depends on all analysis jobs)
if [ "$DRY_RUN" = false ] && [ ${#ANALYSIS_JOB_IDS[@]} -gt 0 ]; then
    echo ""
    echo "Submitting aggregation and visualization job..."
    
    # Create dependency string
    DEPENDENCY_STRING=$(IFS=" && " ; echo "done(${ANALYSIS_JOB_IDS[*]})")
    
    agg_job_name="quvine_aggregate"
    agg_log_file="$OUTPUT_DIR/logs/${agg_job_name}.out"
    agg_err_file="$OUTPUT_DIR/logs/${agg_job_name}.err"
    agg_job_script="$OUTPUT_DIR/logs/${agg_job_name}.sh"
    
    cat > "$agg_job_script" << 'EOF'
#!/bin/bash
#BSUB -J quvine_aggregate
#BSUB -o OUTPUT_DIR_PLACEHOLDER/logs/quvine_aggregate.out
#BSUB -e OUTPUT_DIR_PLACEHOLDER/logs/quvine_aggregate.err
#BSUB -q QUEUE_PLACEHOLDER
#BSUB -W 2:00
#BSUB -M 32GB
#BSUB -R "rusage[mem=32GB]"
#BSUB -w "DEPENDENCY_PLACEHOLDER"

# Activate Python environment
source PROJECT_DIR_PLACEHOLDER/PYTHON_ENV_PLACEHOLDER/bin/activate

# Change to project directory
cd PROJECT_DIR_PLACEHOLDER

echo "========================================"
echo "AGGREGATING AND VISUALIZING RESULTS"
echo "========================================"

# Step 1: Collect and aggregate all results
python scripts/collect_hpc_results.py --results-dir OUTPUT_DIR_PLACEHOLDER/results

# Step 2: Generate visualizations and recommendations
python -c "
import sys
sys.path.insert(0, 'PROJECT_DIR_PLACEHOLDER/src')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load comprehensive results
results_file = Path('OUTPUT_DIR_PLACEHOLDER/results/comprehensive_results.csv')
df = pd.read_csv(results_file)

print(f'Loaded {len(df)} rows from comprehensive results')
print(f'Networks: {df[\"network_id\"].nunique()}')
print(f'Methods: {df[\"method\"].nunique()}')

# Create visualizations directory
viz_dir = Path('OUTPUT_DIR_PLACEHOLDER/visualizations')
viz_dir.mkdir(exist_ok=True)

# Visualization 1: Method comparison across tasks
print('\\nGenerating method comparison plots...')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Ranking performance
if 'ranking_precision@10' in df.columns:
    df.boxplot(column='ranking_precision@10', by='method', ax=axes[0])
    axes[0].set_title('Ranking: Precision@10')
    axes[0].set_xlabel('Method')
    axes[0].set_ylabel('Precision@10')

# Classification performance
if 'classification_mean_f1_macro' in df.columns:
    df.boxplot(column='classification_mean_f1_macro', by='method', ax=axes[1])
    axes[1].set_title('Classification: Mean F1 (Macro)')
    axes[1].set_xlabel('Method')
    axes[1].set_ylabel('F1 Score')

# Link prediction performance
if 'link_prediction_mean_auc_roc' in df.columns:
    df.boxplot(column='link_prediction_mean_auc_roc', by='method', ax=axes[2])
    axes[2].set_title('Link Prediction: Mean AUC-ROC')
    axes[2].set_xlabel('Method')
    axes[2].set_ylabel('AUC-ROC')

plt.tight_layout()
plt.savefig(viz_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
print(f'Saved: {viz_dir / \"method_comparison.png\"}')

# Visualization 2: Complexity vs Performance
print('\\nGenerating complexity vs performance plots...')
complexity_metrics = ['spectral_gap', 'orc_mean', 'kirchhoff_per_pair']
performance_metrics = {
    'ranking_precision@10': 'Ranking Precision@10',
    'classification_mean_f1_macro': 'Classification F1',
    'link_prediction_mean_auc_roc': 'Link Prediction AUC-ROC'
}

for perf_col, perf_label in performance_metrics.items():
    if perf_col not in df.columns:
        continue
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{perf_label} vs Complexity Metrics', fontsize=14)
    
    for idx, comp_metric in enumerate(complexity_metrics):
        if comp_metric not in df.columns:
            continue
        
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            axes[idx].scatter(method_df[comp_metric], method_df[perf_col], 
                            label=method, alpha=0.6)
        
        axes[idx].set_xlabel(comp_metric)
        axes[idx].set_ylabel(perf_label)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    safe_name = perf_col.replace('@', '_at_').replace('/', '_')
    plt.savefig(viz_dir / f'complexity_vs_{safe_name}.png', dpi=300, bbox_inches='tight')
    print(f'Saved: {viz_dir / f\"complexity_vs_{safe_name}.png\"}')

print('\\n========================================')
print('AGGREGATION AND VISUALIZATION COMPLETE')
print(f'Results: OUTPUT_DIR_PLACEHOLDER/results/comprehensive_results.csv')
print(f'Visualizations: OUTPUT_DIR_PLACEHOLDER/visualizations/')
print('========================================')
"

exit $?
EOF
    
    # Replace placeholders
    sed -i "s|OUTPUT_DIR_PLACEHOLDER|$OUTPUT_DIR|g" "$agg_job_script"
    sed -i "s|PROJECT_DIR_PLACEHOLDER|$PROJECT_DIR|g" "$agg_job_script"
    sed -i "s|QUEUE_PLACEHOLDER|$QUEUE|g" "$agg_job_script"
    sed -i "s|PYTHON_ENV_PLACEHOLDER|${PYTHON_ENV%/bin/python}|g" "$agg_job_script"
    sed -i "s|DEPENDENCY_PLACEHOLDER|$DEPENDENCY_STRING|g" "$agg_job_script"
    
    chmod +x "$agg_job_script"
    
    # Submit aggregation job
    agg_job_id=$(bsub < "$agg_job_script" 2>&1 | grep -oP 'Job <\K[0-9]+')
    if [ -n "$agg_job_id" ]; then
        echo "  Submitted aggregation job $agg_job_id: $agg_job_name"
        echo "  Dependencies: ${#ANALYSIS_JOB_IDS[@]} analysis jobs"
    else
        echo "  ERROR: Failed to submit aggregation job"
    fi
fi

echo ""
echo "=================================="
echo "SUBMISSION COMPLETE"
echo "=================================="
echo "Analysis jobs: $JOB_COUNT"
if [ "$DRY_RUN" = false ] && [ ${#ANALYSIS_JOB_IDS[@]} -gt 0 ]; then
    echo "Aggregation job: 1 (will run after all analysis jobs complete)"
    echo ""
    echo "Monitor jobs with:"
    echo "  bjobs -u \$USER"
    echo ""
    echo "Check specific job:"
    echo "  bjobs ${ANALYSIS_JOB_IDS[0]}"
    echo ""
    echo "View job output:"
    echo "  bpeek ${ANALYSIS_JOB_IDS[0]}"
    echo ""
    echo "Results will be in:"
    echo "  $OUTPUT_DIR/results/comprehensive_results.csv"
    echo "  $OUTPUT_DIR/visualizations/"
fi
echo "=================================="

# Made with Bob
