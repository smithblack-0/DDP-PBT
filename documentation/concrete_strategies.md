# Concrete Strategies API Reference

Complete reference for all implemented PBT strategies and their factory functions.

## Navigation

- [AbstractStrategy](abstract_strategy.md) - Base class API reference
- [User Guide](user_guide_outline.md) - Conceptual guide (in progress)
- [Internal Architecture](internal_architecture.md) - Implementation details (contributors)
- [README](../README.md) - Project overview

---

## Overview

DDP-PBT provides five concrete strategy implementations, each with different exploration/exploitation tradeoffs:

| Strategy | Behavior | Best For |
|----------|----------|----------|
| **TopScoreStrategy** | Selects best worker, perturbs | Fast convergence, simple exploration |
| **TopKStrategy** | Random selection from top-K | Moderate exploration, balanced |
| **WeightedAverageStrategy** | Blends all workers by performance | Robust to noise, smooth optimization |
| **TopPopulationAsexualStrategy** | Each worker picks from top-K | Maximum diversity, independent paths |
| **TopPopulationSexualStrategy** | Crossbreeds top-K pairs | Genetic-style exploration, recombination |

All strategies are created via factory functions that handle dependency wiring automatically.

---

## TopScoreStrategy

The simplest PBT strategy. Identifies the best-performing worker and adopts their configuration with perturbation.

### Behavior

At each round end:
1. All workers identify which worker has the lowest validation loss
2. All workers adopt that winner's model parameters and optimizer state
3. All workers adopt that winner's hyperparameters with random perturbation
4. Training continues with synchronized models but perturbed hyperparameters

**Tradeoffs:**
- **Pro**: Fast convergence to good regions
- **Pro**: Simple, deterministic selection
- **Con**: Limited exploration (only via perturbation)
- **Con**: Can get stuck in local minima

### Factory Function

```python
def make_top_score_strategy(
    optimizer: torch.optim.Optimizer,
    max_hyperparameter_search_depth: int = 3,
    config: Optional[Dict[str, Dict[str, Any]]] = None,
    communication_class: Type[Communication] = Communication,
) -> TopScoreStrategy
```

**Parameters:**
- `optimizer` (torch.optim.Optimizer): PyTorch optimizer to manage
- `max_hyperparameter_search_depth` (int): How many layers deep to search `param_groups` for bindable hyperparameters. Default 3.
- `config` (Optional[Dict]): Optional pre-configured hyperparameter schema
- `communication_class` (Type[Communication]): Communication class for distributed ops. Default `Communication`.

**Returns:** Fully-wired `TopScoreStrategy` instance

### Example

```python
import torch
import ddp_pbt

model = MyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

strategy = ddp_pbt.make_top_score_strategy(optimizer)
strategy.bind_log_hyperparameter("lr", std=0.1, min=1e-4, max=1e-1)
strategy.bind_linear_hyperparameter("weight_decay", std=0.001, min=0, max=0.1)

# Training loop
for round_idx in range(num_rounds):
    # ... train for ROUND_LENGTH steps ...
    val_loss = evaluate(model, val_loader)
    strategy.step(val_loss)
```

---

## TopKStrategy

Randomly selects one worker from the top-K performers. Provides more exploration than TopScore while still focusing on good performers.

### Behavior

At each round end:
1. Rank all workers by validation loss
2. Identify the top-K best performers
3. Randomly select one worker from this top-K pool (globally coordinated)
4. All workers adopt the selected worker's configuration with perturbation

**Tradeoffs:**
- **Pro**: More exploration than TopScore
- **Pro**: Still biased toward good performers
- **Con**: Can select sub-optimal workers from top-K
- **Con**: Requires tuning K parameter

### Factory Functions

**Two variants available:**

#### Fixed K

```python
def make_top_k_strategy_out_of_top_k(
    num_k: int,
    optimizer: torch.optim.Optimizer,
    max_hyperparameter_search_depth: int = 3,
    config: Optional[Dict[str, Any]] = None,
    communication_class: Type[Communication] = Communication,
) -> TopKStrategy
```

**Parameters:**
- `num_k` (int): Number of top workers to sample from (must be ≤ world_size)
- Other parameters same as TopScoreStrategy

#### Percentage-Based K

```python
def make_top_k_strategy_by_selection_percentage(
    selection_percentage: float,
    optimizer: torch.optim.Optimizer,
    max_hyperparameter_search_depth: int = 3,
    config: Optional[Dict[str, Any]] = None,
    communication_class: Type[Communication] = Communication,
) -> TopKStrategy
```

**Parameters:**
- `selection_percentage` (float): Fraction of workers to sample from (0.0 to 1.0)
- Other parameters same as TopScoreStrategy

**Calculation:** `K = int(world_size * selection_percentage)`

### Example

```python
import ddp_pbt

# Fixed K: select from top 3 workers
strategy = ddp_pbt.make_top_k_strategy_out_of_top_k(
    num_k=3,
    optimizer=optimizer
)

# Or percentage-based: select from top 25% of workers
strategy = ddp_pbt.make_top_k_strategy_by_selection_percentage(
    selection_percentage=0.25,
    optimizer=optimizer
)

strategy.bind_log_hyperparameter("lr", std=0.1, min=1e-4, max=1e-1)
```

---

## WeightedAverageStrategy

Blends all workers' configurations weighted by their performance. Most robust to noisy validation metrics.

### Behavior

At each round end:
1. Normalize validation losses into weights (lower loss = higher weight)
2. Compute weighted average of all workers' hyperparameters
3. Compute weighted average of all workers' model parameters
4. Compute weighted average of all workers' optimizer states
5. Perturb the averaged hyperparameters for variation

**Weight formula:**

**Let:**
- L_i = validation loss for worker i
- L_min = min(L_1, ..., L_N)

**Formula:**
```
adjusted_i = L_i - L_min
w_i = adjusted_i / sum(adjusted_1, ..., adjusted_N)

# Edge case: if all losses equal, use uniform weights
if sum(adjusted) == 0:
    w_i = 1 / N for all i
```

**Tradeoffs:**
- **Pro**: Robust to noisy metrics (averages out noise)
- **Pro**: All workers contribute (no wasted compute)
- **Pro**: Smooth optimization trajectory
- **Con**: Slower convergence (blending dilutes strong signals)
- **Con**: Can average away promising directions

### Factory Function

```python
def make_weighted_average_strategy(
    optimizer: torch.optim.Optimizer,
    max_hyperparameter_search_depth: int = 3,
    config: Optional[Dict[str, Any]] = None,
    communication_class: Type[Communication] = Communication,
) -> WeightedAverageStrategy
```

**Parameters:** Same as `make_top_score_strategy`

### Example

```python
import ddp_pbt

strategy = ddp_pbt.make_weighted_average_strategy(optimizer)
strategy.bind_log_hyperparameter("lr", std=0.1, min=1e-4, max=1e-1)
strategy.bind_linear_hyperparameter("weight_decay", std=0.001, min=0, max=0.1)
```

---

## TopPopulationAsexualStrategy

Each worker independently selects from top-K performers, creating diverse population evolution.

### Behavior

At each round end:
1. Rank workers by validation loss
2. Identify top-K performers
3. **Each worker independently** selects a different worker from top-K (random selection)
4. Each worker adopts their selected worker's configuration with perturbation
5. Result: K different models exploring different paths

**Tradeoffs:**
- **Pro**: Maximum diversity (K independent exploration paths)
- **Pro**: Parallelizes exploration efficiently
- **Pro**: Can explore multiple promising regions simultaneously
- **Con**: No recombination of good features
- **Con**: Requires careful K tuning (too large = wasteful, too small = limited diversity)

### Factory Function

```python
def make_top_population_asexual_strategy(
    parent_pool_depth: int,
    optimizer: torch.optim.Optimizer,
    max_hyperparameter_search_depth: int = 3,
    config: Optional[Dict[str, Any]] = None,
    communication_class: Type[Communication] = Communication,
) -> TopPopulationAsexualStrategy
```

**Parameters:**
- `parent_pool_depth` (int): Number of top workers each worker can select from (the K)
- Other parameters same as TopScoreStrategy

### Example

```python
import ddp_pbt

# Each worker selects from top 4 performers
strategy = ddp_pbt.make_top_population_asexual_strategy(
    parent_pool_depth=4,
    optimizer=optimizer
)

strategy.bind_log_hyperparameter("lr", std=0.1, min=1e-4, max=1e-1)
```

---

## TopPopulationSexualStrategy

Genetic-style evolution: each worker selects two parents from top-K and crossbreeds their hyperparameters.

### Behavior

At each round end:
1. Rank workers by validation loss
2. Identify top-K performers
3. **Each worker independently** selects two different parents from top-K
4. Crossbreed the parents' hyperparameters (50% chance per hyperparameter)
5. With `mutation_rate` probability, apply additional perturbation
6. Randomly select one parent's model parameters
7. Use that same parent's optimizer state

**Crossbreeding formula (per hyperparameter):**

For each hyperparameter value:
```
if random() < 0.5:
    child_value = parent1_value
else:
    child_value = parent2_value

# Optional mutation
if random() < mutation_rate:
    child_value = perturb(child_value)
```

**Tradeoffs:**
- **Pro**: Combines features from multiple good solutions
- **Pro**: Can discover novel hyperparameter combinations
- **Pro**: Genetic diversity through recombination
- **Con**: More complex selection logic
- **Con**: Model/optimizer chosen randomly (not crossbred like hyperparameters)
- **Con**: May mix incompatible hyperparameter combinations

### Factory Function

```python
def make_top_population_sexual_strategy(
    parent_pool_depth: int,
    mutation_rate: float,
    optimizer: torch.optim.Optimizer,
    max_hyperparameter_search_depth: int = 3,
    config: Optional[Dict[str, Any]] = None,
    communication_class: Type[Communication] = Communication,
) -> TopPopulationSexualStrategy
```

**Parameters:**
- `parent_pool_depth` (int): Number of top workers to select parents from
- `mutation_rate` (float): Probability of additional perturbation after crossbreeding (0.0 to 1.0)
- Other parameters same as TopScoreStrategy

### Example

```python
import ddp_pbt

# Select from top 5, with 10% mutation chance
strategy = ddp_pbt.make_top_population_sexual_strategy(
    parent_pool_depth=5,
    mutation_rate=0.1,
    optimizer=optimizer
)

strategy.bind_log_hyperparameter("lr", std=0.1, min=1e-4, max=1e-1)
strategy.bind_linear_hyperparameter("weight_decay", std=0.001, min=0, max=0.1)
```

---

## Choosing a Strategy

**Start with TopScoreStrategy** if:
- You want simple, fast convergence
- Your validation metrics are reliable (low noise)
- You're doing initial exploration

**Use TopKStrategy** if:
- You want more exploration than TopScore
- You're willing to tune the K parameter
- Your validation metrics have moderate noise

**Use WeightedAverageStrategy** if:
- Your validation metrics are noisy
- You want smooth, stable optimization
- You're training for many rounds and can afford slower convergence

**Use TopPopulationAsexualStrategy** if:
- You want maximum diversity
- You have enough workers to support K independent paths
- You're exploring multiple promising regions

**Use TopPopulationSexualStrategy** if:
- You believe good hyperparameters can be recombined
- You want genetic-style evolution
- You're willing to experiment with crossbreeding

**[TODO: Empirical guidance after experiments - which strategies work best for which problem types]**

---

## Common Parameters

All factory functions share these common parameters:

### `optimizer`
**Type:** `torch.optim.Optimizer`

The PyTorch optimizer instance to manage. Must already be created with your model parameters.

### `max_hyperparameter_search_depth`
**Type:** `int`, default: `3`

How many layers deep to recursively search each `param_group` for float-valued hyperparameters that can be bound.

**Example:**
```python
# param_groups structure:
[{
    "lr": 0.001,                    # depth 0 - found
    "betas": (0.9, 0.999),          # depth 0 - found (tuple unpacked)
    "nested": {                     # depth 1
        "custom_param": 0.5         # depth 1 - found if search_depth >= 1
    }
}]
```

### `config`
**Type:** `Optional[Dict[str, Dict[str, Any]]]`, default: `None`

Pre-configured hyperparameter schema in native JSON format. Alternative to using `bind_*` methods.

**Example:**
```python
config = {
    "lr": {
        "type": "log",
        "std": 0.1,
        "min": 1e-4,
        "max": 1e-1,
        "shared": True
    }
}
strategy = make_top_score_strategy(optimizer, config=config)
# No need to call bind_log_hyperparameter - already configured
```

### `communication_class`
**Type:** `Type[Communication]`, default: `Communication`

Communication class to instantiate for distributed operations. Advanced: Override this to handle communication quirks or custom distributed backends.

---

## Next Steps

- See [AbstractStrategy](abstract_strategy.md) for detailed API reference on binding and configuration
- See [User Guide](user_guide_outline.md) for conceptual understanding and usage patterns
- [TODO: Training loop integration examples]
- [TODO: Checkpointing and resumption patterns]
