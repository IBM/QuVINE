#!/bin/bash
################################################################################
# LSF Job Submission Script for Comprehensive Embedding Analysis
#
# This script submits one LSF job per network to enable large-scale parallel
# execution on HPC clusters using IBM's Load Sharing Facility (LSF).
#
# Usage:
#   bash scripts/submit_hpc_jobs.sh [options]
#
# Options:
#   --dataset-dir DIR    Directory containing generated networks (default: data/comprehensive_dataset)
#   --output-dir DIR     Output directory for results (default: outputs/hpc_results)
#   --queue QUEUE        LSF queue name (default: normal)
#   --walltime TIME      Wall time limit (default: 4:00)
#   --memory MEM         Memory per job in GB (default: 16)
#   --dry-run            Print commands without submitting
#
# Requirements:
#   - LSF cluster with bsub command
#   - Python environment with QuVINE installed
#   - Generated network dataset
#
# Author: QuVINE Team
# Date: 2026-04-02
################################################################################

set -e  # Exit on error

# Default parameters
DATASET_DIR="data/comprehensive_dataset"
OUTPUT_DIR="outputs/hpc_results"
QUEUE="normal"
WALLTIME="4:00"
MEMORY="16"
DRY_RUN=false
PYTHON_ENV="venv_quvine/bin/python"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset-dir)
            DATASET_DIR="$2"
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
            echo "Usage: $0 [--dataset-dir DIR] [--output-dir DIR] [--queue QUEUE] [--walltime TIME] [--memory MEM] [--dry-run]"
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
mkdir -p "$OUTPUT_DIR/embeddings"
mkdir -p "$OUTPUT_DIR/results"

echo "=================================="
echo "HPC Job Submission Configuration"
echo "=================================="
echo "Project Directory: $PROJECT_DIR"
echo "Dataset Directory: $DATASET_DIR"
echo "Output Directory:  $OUTPUT_DIR"
echo "LSF Queue:         $QUEUE"
echo "Wall Time:         $WALLTIME"
echo "Memory per Job:    ${MEMORY}GB"
echo "Python Env:        $PYTHON_ENV"
echo "Dry Run:           $DRY_RUN"
echo "=================================="
echo ""

# Check if dataset directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset directory not found: $DATASET_DIR"
    echo "Please generate the dataset first using:"
    echo "  python -c 'from quvine.data.random_graphs import generate_comprehensive_dataset; generate_comprehensive_dataset(save_dir=\"$DATASET_DIR\")'"
    exit 1
fi

# Count total networks
TOTAL_NETWORKS=0
GRAPH_TYPES=()

for type_dir in "$DATASET_DIR"/*; do
    if [ -d "$type_dir" ]; then
        graph_type=$(basename "$type_dir")
        GRAPH_TYPES+=("$graph_type")
        n_graphs=$(find "$type_dir" -name "*.graphml" | wc -l)
        TOTAL_NETWORKS=$((TOTAL_NETWORKS + n_graphs))
        echo "Found $n_graphs networks of type: $graph_type"
    fi
done

echo ""
echo "Total networks to process: $TOTAL_NETWORKS"
echo "Graph types: ${GRAPH_TYPES[@]}"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - No jobs will be submitted"
    echo ""
fi

# Submit jobs for each network
JOB_COUNT=0
SUBMITTED_JOBS=()

for graph_type in "${GRAPH_TYPES[@]}"; do
    type_dir="$DATASET_DIR/$graph_type"
    
    echo "Processing graph type: $graph_type"
    
    # Find all graph files
    for graph_file in "$type_dir"/*.graphml; do
        if [ ! -f "$graph_file" ]; then
            continue
        fi
        
        # Extract network name
        network_name=$(basename "$graph_file" .graphml)
        
        # Create job-specific output directory
        job_output_dir="$OUTPUT_DIR/results/$graph_type/$network_name"
        mkdir -p "$job_output_dir"
        
        # Job name
        job_name="emb_${graph_type}_${network_name}"
        
        # Log files
        log_file="$OUTPUT_DIR/logs/${job_name}.out"
        err_file="$OUTPUT_DIR/logs/${job_name}.err"
        
        # Create job script
        job_script="$OUTPUT_DIR/logs/${job_name}.sh"
        
        cat > "$job_script" << EOF
#!/bin/bash
#BSUB -J ${job_name}
#BSUB -o ${log_file}
#BSUB -e ${err_file}
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -M ${MEMORY}GB
#BSUB -R "rusage[mem=${MEMORY}GB]"

# Load modules (adjust for your HPC environment)
# module load python/3.9
# module load gcc/9.3.0

# Activate Python environment
source ${PROJECT_DIR}/${PYTHON_ENV%/bin/python}/bin/activate

# Change to project directory
cd ${PROJECT_DIR}

# Run analysis for single network
python -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}/src')

from quvine.comprehensive_embedding_analysis import run_single_network_analysis
import networkx as nx
import json

# Load network
G = nx.read_graphml('${graph_file}')

# Load metadata
metadata_file = '${graph_file}'.replace('.graphml', '_metadata.json')
try:
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
except:
    metadata = {'type': '${graph_type}', 'name': '${network_name}'}

# Run analysis
print(f'Processing network: ${network_name}')
print(f'Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}')

results = run_single_network_analysis(
    G=G,
    network_id='${network_name}',
    network_metadata=metadata,
    output_dir='${job_output_dir}',
    embedding_methods=['quvine_fused', 'quvine_rwr', 'quvine_ctqw', 'quvine_dtqw', 'netmf', 'node2vec'],
    verbose=True
)

print(f'Analysis complete for ${network_name}')
print(f'Results saved to: ${job_output_dir}')
"

# Exit with Python's exit code
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
                SUBMITTED_JOBS+=("$job_id")
                JOB_COUNT=$((JOB_COUNT + 1))
            else
                echo "  ERROR: Failed to submit job for $network_name"
            fi
        fi
    done
done

echo ""
echo "=================================="
echo "Job Submission Summary"
echo "=================================="
echo "Total jobs submitted: $JOB_COUNT"

if [ "$DRY_RUN" = false ] && [ ${#SUBMITTED_JOBS[@]} -gt 0 ]; then
    echo "Job IDs: ${SUBMITTED_JOBS[@]}"
    echo ""
    echo "Monitor jobs with:"
    echo "  bjobs -u \$USER"
    echo ""
    echo "Check specific job:"
    echo "  bjobs ${SUBMITTED_JOBS[0]}"
    echo ""
    echo "View job output:"
    echo "  bpeek ${SUBMITTED_JOBS[0]}"
    echo ""
    echo "Kill all jobs:"
    echo "  bkill ${SUBMITTED_JOBS[@]}"
fi

echo ""
echo "Results will be saved to: $OUTPUT_DIR"
echo "=================================="

# Made with Bob
