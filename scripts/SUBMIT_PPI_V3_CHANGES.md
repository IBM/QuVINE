# PPI Disease Job Submission Script v3 - Multi-Scale Modifications

## Summary of Changes

The `submit_ppi_disease_jobs_v3.sh` script has been modified to generate NEW multi-scale PPI network configurations. The 2000-node jobs are already complete and are excluded from this submission.

## Node Count Configurations

### Network-Specific Settings (NEW jobs only):
- **ProteomeHD**: SKIPPED (2000 nodes already complete)
- **BioPlex3**: 5000 nodes only (2000 already complete)
- **STRING**: 5000, 10000, 15000 nodes (2000 already complete)
- **HumanNet**: 5000, 10000, 15000 nodes (2000 already complete)
- **PCNet**: 5000, 10000, 15000 nodes (2000 already complete)

## Job Count Breakdown

### NEW Jobs Only:
- **ProteomeHD**: 0 jobs (2000-node configuration already complete)
- **BioPlex3**: 1 size × 3 diseases × 30 reps = **90 jobs**
- **STRING**: 3 sizes × 3 diseases × 30 reps = **270 jobs**
- **HumanNet**: 3 sizes × 3 diseases × 30 reps = **270 jobs**
- **PCNet**: 3 sizes × 3 diseases × 30 reps = **270 jobs**

### Total NEW Jobs: **900 jobs**

## Key Implementation Details

1. **Function-based Configuration**: Uses `get_node_counts()` function instead of associative arrays for bash 3.x compatibility (macOS default)

2. **Nested Loop Structure**:
   ```bash
   for NET in NETWORKS; do
       for MAX_NODES in NODE_COUNTS; do  # Excludes 2000
           for DISEASE in DISEASES; do
               for REP in 0..29; do
                   # Generate job
               done
           done
       done
   done
   ```

3. **Network ID Format**: `{NET}_n{NODES}_{DISEASE}_rep{REP}`
   - Example: `STRING_n5000_asthma_rep00`
   - Example: `BioPlex3_n5000_autism_rep15`
   - Example: `HumanNet_n10000_schizophrenia_rep29`

4. **Default Parameters**:
   - N_REPS: 30 (as specified)
   - WALLTIME: 240:00 (4 hours)
   - MEMORY: 16GB

## Usage

```bash
# Standard submission (NEW jobs only)
bash scripts/submit_ppi_disease_jobs_v3.sh

# Dry run to preview jobs
bash scripts/submit_ppi_disease_jobs_v3.sh --dry-run

# Custom parameters
bash scripts/submit_ppi_disease_jobs_v3.sh --n-reps 30 --queue normal --memory 16
```

## Output Structure

NEW results will be organized as:
```
results/ppi_disease_v3/
├── results/
│   ├── STRING_n5000_asthma_rep00/
│   ├── STRING_n5000_asthma_rep01/
│   ├── STRING_n10000_asthma_rep00/
│   ├── STRING_n15000_asthma_rep00/
│   ├── BioPlex3_n5000_autism_rep00/
│   ├── HumanNet_n5000_schizophrenia_rep00/
│   ├── HumanNet_n10000_schizophrenia_rep00/
│   ├── HumanNet_n15000_schizophrenia_rep00/
│   ├── PCNet_n5000_asthma_rep00/
│   └── ...
└── logs/
    ├── ppi3_STRING_n5000_asthma_rep00.sh
    ├── ppi3_STRING_n5000_asthma_rep00.out
    └── ...
```

## Complete Dataset After This Submission

After these 900 NEW jobs complete, the full dataset will include:

### ProteomeHD (90 total jobs):
- 2000 nodes: 3 diseases × 30 reps = 90 jobs ✅ (already complete)

### BioPlex3 (180 total jobs):
- 2000 nodes: 3 diseases × 30 reps = 90 jobs ✅ (already complete)
- 5000 nodes: 3 diseases × 30 reps = 90 jobs 🆕 (this submission)

### STRING (360 total jobs):
- 2000 nodes: 3 diseases × 30 reps = 90 jobs ✅ (already complete)
- 5000 nodes: 3 diseases × 30 reps = 90 jobs 🆕 (this submission)
- 10000 nodes: 3 diseases × 30 reps = 90 jobs 🆕 (this submission)
- 15000 nodes: 3 diseases × 30 reps = 90 jobs 🆕 (this submission)

### HumanNet (360 total jobs):
- 2000 nodes: 3 diseases × 30 reps = 90 jobs ✅ (already complete)
- 5000 nodes: 3 diseases × 30 reps = 90 jobs 🆕 (this submission)
- 10000 nodes: 3 diseases × 30 reps = 90 jobs 🆕 (this submission)
- 15000 nodes: 3 diseases × 30 reps = 90 jobs 🆕 (this submission)

### PCNet (360 total jobs):
- 2000 nodes: 3 diseases × 30 reps = 90 jobs ✅ (already complete)
- 5000 nodes: 3 diseases × 30 reps = 90 jobs 🆕 (this submission)
- 10000 nodes: 3 diseases × 30 reps = 90 jobs 🆕 (this submission)
- 15000 nodes: 3 diseases × 30 reps = 90 jobs 🆕 (this submission)

**Grand Total: 1,350 jobs (450 already complete + 900 NEW)**

## Verification

The script has been tested for:
- ✅ Bash syntax validation (`bash -n`)
- ✅ Job count calculation (900 NEW jobs confirmed)
- ✅ Network ID formatting
- ✅ Compatibility with bash 3.x (macOS)
- ✅ ProteomeHD correctly skipped (2000 nodes already done)

## Notes

- Each replicate uses a different random seed (0-29) for stochastic variation
- Seeds and targets are always protected and included in subsampled networks
- The script uses degree-matched subsampling to preserve network topology
- All 13 embedding methods are run for each configuration
- 2000-node configurations are intentionally excluded as they are already complete