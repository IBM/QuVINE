# QuVINE Method Registry - Complete List of 39 Methods

## Overview

Total: **39 methods** (16 quantum, 23 classical)

## Method Categories

### 1. SGNS (Skip-Gram with Negative Sampling) - 3 methods
- `quvine_rwr` - Random Walk with Restart (classical)
- `quvine_ctqw` - Continuous-Time Quantum Walk (quantum)
- `quvine_dtqw` - Discrete-Time Quantum Walk (quantum)

### 2. Graph Filters - 6 methods
- `quvine_baseline_heat` - Heat kernel filter, no walk (classical)
- `quvine_baseline_poly` - Polynomial filter, no walk (classical)
- `quvine_rwr_heat` - RWR + heat filter (classical)
- `quvine_rwr_poly` - RWR + polynomial filter (classical)
- `quvine_ctqw_heat` - CTQW + heat filter (quantum)
- `quvine_ctqw_poly` - CTQW + polynomial filter (quantum)

### 3. GAT (Graph Attention Networks) - 12 methods

**Baseline:**
- `gat_baseline` - GAT without quantum calibration (classical)

**With filters only:**
- `gat_heat` - GAT + heat filter, no walk (classical)
- `gat_poly` - GAT + polynomial filter, no walk (classical)

**With walks only:**
- `gat_rwr` - GAT + RWR (classical)
- `gat_ctqw` - GAT + CTQW (quantum)
- `gat_dtqw` - GAT + DTQW (quantum)

**With walks + filters:**
- `gat_rwr_heat` - GAT + RWR + heat filter (classical)
- `gat_rwr_poly` - GAT + RWR + polynomial filter (classical)
- `gat_ctqw_heat` - GAT + CTQW + heat filter (quantum)
- `gat_ctqw_poly` - GAT + CTQW + polynomial filter (quantum)
- `gat_dtqw_heat` - GAT + DTQW + heat filter (quantum)
- `gat_dtqw_poly` - GAT + DTQW + polynomial filter (quantum)

### 4. GraphGPS (Graph Transformer) - 12 methods

**Baseline:**
- `graphgps_baseline` - GraphGPS without quantum calibration (classical)

**With filters only:**
- `graphgps_heat` - GraphGPS + heat filter, no walk (classical)
- `graphgps_poly` - GraphGPS + polynomial filter, no walk (classical)

**With walks only:**
- `graphgps_rwr` - GraphGPS + RWR (classical)
- `graphgps_ctqw` - GraphGPS + CTQW (quantum)
- `graphgps_dtqw` - GraphGPS + DTQW (quantum)

**With walks + filters:**
- `graphgps_rwr_heat` - GraphGPS + RWR + heat filter (classical)
- `graphgps_rwr_poly` - GraphGPS + RWR + polynomial filter (classical)
- `graphgps_ctqw_heat` - GraphGPS + CTQW + heat filter (quantum)
- `graphgps_ctqw_poly` - GraphGPS + CTQW + polynomial filter (quantum)
- `graphgps_dtqw_heat` - GraphGPS + DTQW + heat filter (quantum)
- `graphgps_dtqw_poly` - GraphGPS + DTQW + polynomial filter (quantum)

### 5. Classical Baselines - 6 methods
- `node2vec` - Node2Vec (classical)
- `netmf` - Network Embedding as Matrix Factorization (classical)
- `graphsage` - GraphSAGE (classical)
- `appnp` - Approximate Personalized Propagation of Neural Predictions (classical)
- `baseline_filter` - Classical filter baseline (classical)
- `baseline_gcnmf` - Classical GCN-MF baseline (classical)

## Quantum vs Classical Breakdown

### Quantum Methods (16 total)
**SGNS (2):**
- quvine_ctqw, quvine_dtqw

**Filters (2):**
- quvine_ctqw_heat, quvine_ctqw_poly

**GAT (6):**
- gat_ctqw, gat_dtqw, gat_ctqw_heat, gat_ctqw_poly, gat_dtqw_heat, gat_dtqw_poly

**GraphGPS (6):**
- graphgps_ctqw, graphgps_dtqw, graphgps_ctqw_heat, graphgps_ctqw_poly, graphgps_dtqw_heat, graphgps_dtqw_poly

### Classical Methods (23 total)
**SGNS (1):**
- quvine_rwr

**Filters (4):**
- quvine_baseline_heat, quvine_baseline_poly, quvine_rwr_heat, quvine_rwr_poly

**GAT (6):**
- gat_baseline, gat_heat, gat_poly, gat_rwr, gat_rwr_heat, gat_rwr_poly

**GraphGPS (6):**
- graphgps_baseline, graphgps_heat, graphgps_poly, graphgps_rwr, graphgps_rwr_heat, graphgps_rwr_poly

**Baselines (6):**
- node2vec, netmf, graphsage, appnp, baseline_filter, baseline_gcnmf

## Method Type Grouping (for Fusion)

### Type 1: SGNS-based (3 methods)
- Quantum: quvine_ctqw, quvine_dtqw
- Classical: quvine_rwr

### Type 2: Filter-based (6 methods)
- Quantum: quvine_ctqw_heat, quvine_ctqw_poly
- Classical: quvine_baseline_heat, quvine_baseline_poly, quvine_rwr_heat, quvine_rwr_poly

### Type 3: GAT-based (12 methods)
- Quantum: gat_ctqw, gat_dtqw, gat_ctqw_heat, gat_ctqw_poly, gat_dtqw_heat, gat_dtqw_poly
- Classical: gat_baseline, gat_heat, gat_poly, gat_rwr, gat_rwr_heat, gat_rwr_poly

### Type 4: GraphGPS-based (12 methods)
- Quantum: graphgps_ctqw, graphgps_dtqw, graphgps_ctqw_heat, graphgps_ctqw_poly, graphgps_dtqw_heat, graphgps_dtqw_poly
- Classical: graphgps_baseline, graphgps_heat, graphgps_poly, graphgps_rwr, graphgps_rwr_heat, graphgps_rwr_poly

### Type 5: Other Baselines (6 methods)
- Classical only: node2vec, netmf, graphsage, appnp, baseline_filter, baseline_gcnmf

## Fusion Strategy

For each method type (SGNS, Filters, GAT, GraphGPS):
1. Generate embeddings for all methods in that type
2. Perform SVD fusion within quantum methods → `fused_quantum_{type}`
3. Perform SVD fusion within classical methods → `fused_classical_{type}`
4. Select best performing quantum method per type (mean across 30 reps)
5. Select best performing classical method per type (mean across 30 reps)
6. Fuse best quantum methods across types → `fused_q`
7. Fuse best classical methods across types → `fused_c`
8. Evaluate `fused_q` and `fused_c` on all three tasks

## Complete Method List (for scripts)

```
quvine_rwr,quvine_ctqw,quvine_dtqw,quvine_baseline_heat,quvine_baseline_poly,quvine_rwr_heat,quvine_rwr_poly,quvine_ctqw_heat,quvine_ctqw_poly,gat_baseline,gat_heat,gat_poly,gat_rwr,gat_ctqw,gat_dtqw,gat_rwr_heat,gat_rwr_poly,gat_ctqw_heat,gat_ctqw_poly,gat_dtqw_heat,gat_dtqw_poly,graphgps_baseline,graphgps_heat,graphgps_poly,graphgps_rwr,graphgps_ctqw,graphgps_dtqw,graphgps_rwr_heat,graphgps_rwr_poly,graphgps_ctqw_heat,graphgps_ctqw_poly,graphgps_dtqw_heat,graphgps_dtqw_poly,node2vec,netmf,graphsage,appnp,baseline_filter,baseline_gcnmf