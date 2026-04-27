# Optuna Hyperparameter Optimization Explained

## Quick Answer

**`n_trials` is NOT evaluating all combinations!**

Optuna uses **smart sampling** (Bayesian optimization) to intelligently explore the hyperparameter space, not exhaustive grid search.

---

## Grid Search vs Optuna

### Grid Search (Exhaustive)
```python
# Example: node2vec with 5 parameters
embedding_dim: [32, 64, 128, 256]           # 4 values
walk_length: [10, 20, 40, 80]               # 4 values  
num_walks: [10, 20, 40, 80]                 # 4 values
p: [0.25, 0.5, 1.0, 2.0, 4.0]              # 5 values
q: [0.25, 0.5, 1.0, 2.0, 4.0]              # 5 values

Total combinations = 4 × 4 × 4 × 5 × 5 = 1,600 combinations!
```

**Problem**: Evaluating all 1,600 combinations would take days/weeks!

### Optuna (Smart Sampling)
```python
n_trials = 50  # Only evaluate 50 combinations

# Optuna intelligently selects which 50 combinations to try
# Based on results of previous trials
```

**Advantage**: Finds good hyperparameters in 50 trials instead of 1,600!

---

## How Optuna Works

### 1. Tree-structured Parzen Estimator (TPE)

Optuna uses **TPE sampler** by default, which:

1. **Starts with random exploration** (first 10 trials)
   - Tries random combinations to understand the space
   
2. **Builds a probabilistic model** (after 10 trials)
   - Models which hyperparameters lead to good performance
   - Models which hyperparameters lead to bad performance
   
3. **Suggests promising combinations** (remaining trials)
   - Focuses on regions likely to have good performance
   - Still explores to avoid local optima

### Example with node2vec

```
Trial 1:  embedding_dim=64,  p=1.0, q=0.5  → score=0.35 (random)
Trial 2:  embedding_dim=128, p=2.0, q=1.0  → score=0.42 (random)
Trial 3:  embedding_dim=256, p=0.5, q=2.0  → score=0.38 (random)
...
Trial 10: embedding_dim=32,  p=4.0, q=0.25 → score=0.31 (random)

# Now Optuna has learned:
# - embedding_dim=128 seems good
# - p=2.0 seems good
# - Let's try combinations around these values

Trial 11: embedding_dim=128, p=2.0, q=0.5  → score=0.45 (TPE suggests)
Trial 12: embedding_dim=128, p=1.0, q=1.0  → score=0.44 (TPE suggests)
...
Trial 50: embedding_dim=128, p=2.0, q=1.0  → score=0.47 (best found!)
```

---

## Configuration Breakdown

### Your Config File
```yaml
hyperparameters:
  node2vec:
    embedding_dim: [32, 64, 128, 256]       # 4 choices
    walk_length: [10, 20, 40, 80]           # 4 choices
    num_walks: [10, 20, 40, 80]             # 4 choices
    p: [0.25, 0.5, 1.0, 2.0, 4.0]          # 5 choices
    q: [0.25, 0.5, 1.0, 2.0, 4.0]          # 5 choices
    window_size: [5, 10, 15, 20]            # 4 choices

optuna:
  n_trials: 50                               # Only try 50 combinations
  n_startup_trials: 10                       # First 10 are random
```

### What Happens

**Total possible combinations**: 4 × 4 × 4 × 5 × 5 × 4 = **6,400 combinations**

**Optuna evaluates**: **50 combinations** (0.78% of total space!)

**How it chooses**:
- **Trials 1-10**: Random sampling (explore)
- **Trials 11-50**: TPE-guided sampling (exploit + explore)

---

## Comparison Table

| Method | Combinations Evaluated | Time (1000 nodes) | Finds Optimal? |
|--------|----------------------|-------------------|----------------|
| **Grid Search** | ALL (6,400) | ~320 hours | ✅ Guaranteed |
| **Random Search** | 50 | ~2.5 hours | ⚠️ Maybe |
| **Optuna TPE** | 50 | ~2.5 hours | ✅ Very likely |

---

## Why Optuna is Better

### 1. Efficiency
- **Grid search**: Wastes time on bad regions
- **Optuna**: Focuses on promising regions

### 2. Scalability
```python
# With 10 parameters, each with 5 values:
Grid search: 5^10 = 9,765,625 combinations (impossible!)
Optuna: 50-100 trials (feasible!)
```

### 3. Continuous Parameters
```python
# Optuna can sample from continuous ranges
learning_rate: trial.suggest_float('lr', 0.0001, 0.1, log=True)
# Can try ANY value between 0.0001 and 0.1
# Not limited to predefined list!
```

### 4. Adaptive
- Learns from previous trials
- Adjusts strategy based on results
- Balances exploration vs exploitation

---

## n_trials Guidelines

### How Many Trials Do You Need?

**Rule of thumb**: `n_trials ≈ 10 × number_of_parameters`

| Scenario | Parameters | Recommended n_trials | Reasoning |
|----------|-----------|---------------------|-----------|
| **Quick test** | 5 | 20-30 | Get rough idea |
| **Standard** | 5-8 | 50-100 | Good balance |
| **Thorough** | 8-10 | 100-200 | High confidence |
| **Exhaustive** | 10+ | 200-500 | Research-grade |

### Examples

**node2vec** (6 parameters):
```yaml
n_trials: 50   # Quick: 50-60 trials
n_trials: 100  # Standard: 100 trials  
n_trials: 200  # Thorough: 200 trials
```

**graphgps_baseline** (10 parameters):
```yaml
n_trials: 100  # Quick: 100 trials
n_trials: 200  # Standard: 200 trials
n_trials: 500  # Thorough: 500 trials
```

---

## Optuna Configuration Options

### 1. Number of Trials
```yaml
optuna:
  n_trials: 50  # How many combinations to try
```

**Trade-off**:
- More trials = better results but slower
- Fewer trials = faster but might miss optimal

### 2. Startup Trials (Random Exploration)
```yaml
optuna:
  n_startup_trials: 10  # First 10 trials are random
```

**Purpose**: 
- Explore the space before exploiting
- Prevents premature convergence
- Typical: 10-20% of total trials

### 3. Sampler Type
```yaml
optuna:
  sampler: "TPE"  # Tree-structured Parzen Estimator (default)
  # sampler: "Random"  # Pure random search
  # sampler: "CmaEs"  # Covariance Matrix Adaptation
```

**TPE** (recommended):
- Best for most cases
- Balances exploration and exploitation
- Works well with categorical and continuous parameters

### 4. Pruner (Early Stopping)
```yaml
optuna:
  pruner: "MedianPruner"  # Stop unpromising trials early
  pruner_params:
    n_startup_trials: 5
    n_warmup_steps: 10
```

**Purpose**:
- Stop bad trials early to save time
- Example: If score is much worse than median after 10 steps, stop

---

## Practical Example

### Scenario: Tuning node2vec

**Configuration**:
```yaml
hyperparameters:
  node2vec:
    embedding_dim: [32, 64, 128, 256]
    p: [0.25, 0.5, 1.0, 2.0, 4.0]
    q: [0.25, 0.5, 1.0, 2.0, 4.0]

optuna:
  n_trials: 30
  n_startup_trials: 10
```

**What Happens**:

```
Trials 1-10 (Random Exploration):
  Trial 1:  dim=64,  p=1.0, q=0.5  → 0.35
  Trial 2:  dim=128, p=2.0, q=1.0  → 0.42 ⭐
  Trial 3:  dim=256, p=0.5, q=2.0  → 0.38
  Trial 4:  dim=32,  p=4.0, q=0.25 → 0.31
  Trial 5:  dim=128, p=0.5, q=4.0  → 0.36
  Trial 6:  dim=64,  p=2.0, q=2.0  → 0.37
  Trial 7:  dim=256, p=1.0, q=0.5  → 0.40
  Trial 8:  dim=128, p=4.0, q=1.0  → 0.39
  Trial 9:  dim=32,  p=0.25, q=0.5 → 0.29
  Trial 10: dim=256, p=2.0, q=4.0  → 0.41

Optuna learns:
  - dim=128 appears in best trials (2, 5, 8)
  - p=2.0 appears in best trials (2, 6, 10)
  - q=1.0 appears in best trial (2)

Trials 11-30 (TPE-Guided):
  Trial 11: dim=128, p=2.0, q=1.0  → 0.45 ⭐⭐ (TPE suggests)
  Trial 12: dim=128, p=1.0, q=1.0  → 0.44 (explore nearby)
  Trial 13: dim=128, p=2.0, q=0.5  → 0.43 (explore nearby)
  Trial 14: dim=64,  p=2.0, q=1.0  → 0.40 (test dim)
  Trial 15: dim=256, p=2.0, q=1.0  → 0.42 (test dim)
  ...
  Trial 30: dim=128, p=2.0, q=1.0  → 0.47 ⭐⭐⭐ (best!)

Best found: dim=128, p=2.0, q=1.0 → 0.47
```

**Result**: Found excellent hyperparameters in 30 trials instead of trying all 4×5×5 = 100 combinations!

---

## Grid Search vs Optuna: When to Use Each

### Use Grid Search When:
- ✅ Very few parameters (2-3)
- ✅ Few values per parameter (2-3)
- ✅ Total combinations < 50
- ✅ Need to evaluate ALL combinations
- ✅ Have unlimited time

**Example**: Testing 2 learning rates × 3 dropout values = 6 combinations

### Use Optuna When:
- ✅ Many parameters (5+)
- ✅ Many values per parameter (5+)
- ✅ Total combinations > 100
- ✅ Limited time/resources
- ✅ Want intelligent search

**Example**: Tuning 8 parameters with 5 values each = 390,625 combinations

---

## Advanced: How TPE Works (Simplified)

### Step 1: Split Trials into Good and Bad
```python
# After 20 trials, sort by score
good_trials = top 20% of trials  # Best performing
bad_trials = bottom 80% of trials  # Worse performing
```

### Step 2: Model Each Group
```python
# Build probability distributions
P(params | good) = probability of params given good performance
P(params | bad) = probability of params given bad performance
```

### Step 3: Suggest Next Trial
```python
# Suggest params that maximize:
score = P(params | good) / P(params | bad)

# This means: "likely to be good, unlikely to be bad"
```

### Step 4: Repeat
- Evaluate suggested params
- Update good/bad groups
- Suggest next trial

---

## Summary

### Key Points

1. **n_trials ≠ all combinations**
   - Optuna samples intelligently, not exhaustively
   - 50 trials can explore a space of 6,400+ combinations

2. **Optuna is smart**
   - Learns from previous trials
   - Focuses on promising regions
   - Much faster than grid search

3. **Configuration defines search space**
   - Your config lists possible values
   - Optuna chooses which to try
   - n_trials controls how many to evaluate

4. **Recommended n_trials**
   - Quick: 10 × num_parameters
   - Standard: 20 × num_parameters
   - Thorough: 50 × num_parameters

### Example Calculation

**node2vec with 6 parameters**:
```
Possible combinations: 4 × 4 × 4 × 5 × 5 × 4 = 6,400
Optuna n_trials: 50 (0.78% of space)
Time saved: 99.2%
Quality: ~95% of optimal (typically)
```

### Bottom Line

**Optuna is like a smart assistant**:
- You give it a list of options (config file)
- You tell it how many experiments to run (n_trials)
- It intelligently picks which experiments to try
- It learns and adapts as it goes
- It finds good solutions much faster than trying everything

**Grid search is like a brute-force approach**:
- Tries every single combination
- Wastes time on obviously bad combinations
- Guaranteed to find optimal but impractical for large spaces

**For your use case**: With 10 methods × 3 tasks × 2 networks × 50 trials = 3,000 total evaluations, Optuna will find excellent hyperparameters in a reasonable time!