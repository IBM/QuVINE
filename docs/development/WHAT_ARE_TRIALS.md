# What Are "Trials" in Hyperparameter Tuning?

## Quick Answer

A **trial** is a single attempt to find good hyperparameters by:
1. Picking a set of hyperparameter values
2. Training the model with those values
3. Evaluating performance
4. Recording the results

**30 trials** means we try 30 different combinations of hyperparameters to find the best one.

---

## Detailed Explanation

### The Hyperparameter Tuning Process

For each method (e.g., APPNP), we need to find the best values for its hyperparameters:

**APPNP Hyperparameters:**
- `embedding_dim`: Should it be 32, 64, or 128?
- `hidden_dim`: Should it be 32, 64, or 128?
- `n_layers`: Should it be 1, 2, or 3?
- `alpha`: Should it be 0.05, 0.1, 0.2, or 0.3?
- `K`: Should it be 5, 10, 15, or 20?
- `dropout`: Should it be 0.3, 0.5, or 0.7?
- `lr`: Should it be 0.001, 0.01, or 0.1?
- `epochs`: Should it be 100, 200, or 300?

### What Happens in Each Trial?

**Trial 1:**
```
1. Optuna suggests: embedding_dim=64, hidden_dim=128, n_layers=2, 
                    alpha=0.1, K=10, dropout=0.5, lr=0.01, epochs=200
2. Run APPNP with these parameters on 3 pilot graphs (200 nodes each)
3. Evaluate: Node classification F1 + Link prediction AUC
4. Score: 0.75 (average performance)
5. Record: Trial 1 → score=0.75, params={...}
```

**Trial 2:**
```
1. Optuna suggests: embedding_dim=128, hidden_dim=64, n_layers=3,
                    alpha=0.15, K=15, dropout=0.4, lr=0.005, epochs=300
2. Run APPNP with these NEW parameters on same 3 pilot graphs
3. Evaluate: Node classification F1 + Link prediction AUC
4. Score: 0.82 (better!)
5. Record: Trial 2 → score=0.82, params={...}
```

**Trial 3:**
```
1. Optuna suggests: embedding_dim=128, hidden_dim=128, n_layers=2,
                    alpha=0.12, K=12, dropout=0.45, lr=0.008, epochs=250
   (Optuna is smart - it explores near Trial 2 since it did well)
2. Run APPNP with these parameters
3. Evaluate performance
4. Score: 0.85 (even better!)
5. Record: Trial 3 → score=0.85, params={...}
```

... and so on for 30 trials total.

**After 30 trials:**
- We have tried 30 different hyperparameter combinations
- We pick the one with the highest score (e.g., Trial 23 → score=0.89)
- Those become the "best hyperparameters" for APPNP

---

## Why 30 Trials?

### Too Few Trials (e.g., 5 trials)
- ❌ Might miss good hyperparameter combinations
- ❌ Results depend heavily on luck
- ✅ Very fast (~2 hours)

### Reasonable Trials (e.g., 15-30 trials)
- ✅ Good exploration of hyperparameter space
- ✅ Reliable results
- ✅ Reasonable time (~6-12 hours)
- ⚠️ Moderate computational cost

### Many Trials (e.g., 50-100 trials)
- ✅ Thorough exploration
- ✅ Very reliable results
- ❌ Expensive (~20-40 hours)
- ❌ Diminishing returns after ~30 trials

**Recommendation**: 30 trials is the sweet spot for most use cases.

---

## How Optuna Makes Trials Efficient

Optuna uses **Tree-structured Parzen Estimator (TPE)** - a smart algorithm that:

1. **Starts random**: First few trials explore randomly
2. **Learns patterns**: Identifies which hyperparameters work well
3. **Focuses search**: Later trials explore promising regions
4. **Avoids waste**: Doesn't waste time on obviously bad combinations

### Example: APPNP Learning Process

**Trials 1-5** (Random exploration):
- Try diverse combinations
- Learn: "embedding_dim=128 seems better than 32"
- Learn: "alpha around 0.1-0.15 works well"

**Trials 6-15** (Focused exploration):
- Focus on embedding_dim=128
- Explore alpha values between 0.1-0.15
- Try different n_layers with good embedding_dim

**Trials 16-30** (Fine-tuning):
- Refine the best combinations found
- Explore small variations
- Confirm the best hyperparameters

---

## What Gets Evaluated in Each Trial?

For each trial, we:

1. **Generate 3 pilot graphs** (200 nodes each, different random seeds)
2. **Run the method** with the trial's hyperparameters on all 3 graphs
3. **Evaluate performance**:
   - Node classification: F1-macro score (cross-validation)
   - Link prediction: AUC-ROC score
4. **Compute average score** across the 3 graphs
5. **Return score to Optuna** so it can suggest the next trial

### Why 3 Graphs?

- **Stochasticity**: Graph generation is random
- **Robustness**: Ensures hyperparameters work on different graph instances
- **Reliability**: Average of 3 is more stable than a single graph

---

## Time Breakdown Example: APPNP

**Per Trial (~20 minutes):**
- Generate 3 graphs: ~1 minute
- Train APPNP on graph 1: ~6 minutes
- Train APPNP on graph 2: ~6 minutes
- Train APPNP on graph 3: ~6 minutes
- Evaluate all 3: ~1 minute
- **Total**: ~20 minutes

**30 Trials:**
- 30 trials × 20 min/trial = **600 minutes = 10 hours**

**15 Trials (Fast Mode):**
- 15 trials × 20 min/trial = **300 minutes = 5 hours**

---

## Practical Examples

### Fast Testing (15 trials)
```bash
sbatch scripts/submit_ppi_comprehensive_with_tuning.sh --n-trials 15
```
- **Time**: ~5-6 hours per network
- **Use case**: Quick testing, method comparison
- **Quality**: Good enough for most purposes

### Production (30 trials - default)
```bash
sbatch scripts/submit_ppi_comprehensive_with_tuning.sh
```
- **Time**: ~10-12 hours per network
- **Use case**: Published results, production deployment
- **Quality**: Robust, reliable hyperparameters

### Research (50 trials)
```bash
sbatch scripts/submit_ppi_comprehensive_with_tuning.sh --n-trials 50
```
- **Time**: ~16-20 hours per network
- **Use case**: Method development, thorough analysis
- **Quality**: Very thorough exploration

---

## Summary

**Trials = Attempts to find good hyperparameters**

- Each trial tests one combination of hyperparameter values
- Optuna intelligently suggests which combinations to try
- More trials = better hyperparameters (but diminishing returns)
- 30 trials is a good default (balances quality and time)
- 15 trials is fine for testing (2× faster)
- 50 trials for research (more thorough)

**Think of it like:**
- Trying 30 different recipes to find the best one
- Each recipe (trial) uses different ingredients (hyperparameters)
- You taste each one (evaluate performance)
- You pick the best recipe (best hyperparameters)
- Then you use that recipe for all future cooking (analysis jobs)