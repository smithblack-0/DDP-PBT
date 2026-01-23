# AbstractStrategy API Reference

Base class for all Population Based Training strategies. Provides hyperparameter binding, configuration management, and orchestrates the round-end optimization flow.

## Navigation

- [Concrete Strategies](concrete_strategies.md) - Specific strategy implementations
- [User Guide](user_guide_outline.md) - Conceptual guide (in progress)
- [Internal Architecture](internal_architecture.md) - Implementation details (contributors)
- [README](../README.md) - Project overview

---

## Overview

`AbstractStrategy` is the base class that all concrete PBT strategies inherit from. It acts as a PyTorch LRScheduler and provides:

1. **Hyperparameter binding** - Configure which optimizer parameters to explore
2. **Configuration methods** - Builder pattern for schema setup
3. **Round-end orchestration** - Coordinates distributed strategy execution via `step()`
4. **Checkpointing** - Save/load schema state

Users interact with strategies through factory functions (which return strategy instances) and then call the public methods documented here.

**Important constraint**: Strategies only work correctly in distributed (DDP) environments with 2+ workers. They modify optimizer hyperparameters that affect how gradients are turned into parameter updates. Do not bind to model evaluation properties (like dropout rates) as these won't be properly isolated between workers due to gradient sharing.

---

## Factory Pattern

Strategies are created via factory functions, not direct instantiation:

```python
import ddp_pbt

# Create strategy via factory
strategy = ddp_pbt.make_top_score_strategy(optimizer)

# Configure hyperparameters to explore
strategy.bind_log_hyperparameter("lr", std=0.1, min=1e-4, max=1e-1)
strategy.bind_linear_hyperparameter("weight_decay", std=0.001, min=0, max=0.1)
```

See [Concrete Strategies](concrete_strategies.md) for available factory functions.

---

## Public Methods

### Configuration: Binding Hyperparameters

#### `bind_log_hyperparameter()`

Configure a hyperparameter to be perturbed in log-space.

```python
def bind_log_hyperparameter(
    name: str,
    std: float,
    min: float,
    max: Optional[float] = None,
    shared: bool = True,
) -> None
```

**Use log-space for**: Learning rate, weight decay, any other scalar float hyperparameter that spans orders of magnitude.

**Parameters:**
- `name` (str): Hyperparameter name (must exist in `optimizer.param_groups`)
- `std` (float): Standard deviation for perturbation (must be > 0)
- `min` (float): Minimum bound (required, must be > 0 for log parameters)
- `max` (Optional[float]): Maximum bound (must be > min if provided)
- `shared` (bool): If True, shared across all param groups; if False, per-group values

**Perturbation formula:**

**Let:**
- v = current hyperparameter value
- σ = std (standard deviation)
- ε ~ N(0, σ) = random normal sample

**Formula:**
```
v_new = exp(log(v) + ε)
v_new = clip(v_new, min, max)
```

**Example:**
```python
# Learning rate: explore 1e-4 to 1e-1 with 10% std deviation
strategy.bind_log_hyperparameter("lr", std=0.1, min=1e-4, max=1e-1)
```

**Raises:**
- `ValueError`: If hyperparameter doesn't exist in optimizer or validation fails
- `TypeError`: If parameter types are incorrect

---

#### `bind_linear_hyperparameter()`

Configure a hyperparameter to be perturbed in linear space.

```python
def bind_linear_hyperparameter(
    name: str,
    std: float,
    min: Optional[float] = None,
    max: Optional[float] = None,
    shared: bool = True,
) -> None
```

**Use linear space for**: Weight decay, gradient clipping thresholds, any scalar hyperparameter with linear relationships.

**Parameters:**
- `name` (str): Hyperparameter name (must exist in `optimizer.param_groups`)
- `std` (float): Standard deviation for perturbation (must be > 0)
- `min` (Optional[float]): Optional minimum bound
- `max` (Optional[float]): Optional maximum bound (must be > min if both provided)
- `shared` (bool): If True, shared across all param groups; if False, per-group values

**Perturbation formula:**

**Let:**
- v = current hyperparameter value
- σ = std (standard deviation)
- ε ~ N(0, σ) = random normal sample

**Formula:**
```
v_new = v + ε
v_new = clip(v_new, min, max)  # if bounds provided
```

**Example:**
```python
# Weight decay: explore 0 to 0.1 with 0.001 std deviation
strategy.bind_linear_hyperparameter("weight_decay", std=0.001, min=0, max=0.1)
```

**Raises:**
- `ValueError`: If hyperparameter doesn't exist in optimizer or validation fails
- `TypeError`: If parameter types are incorrect

---

### Training Loop: Round-End Execution

#### `step()`

Execute the round-end strategy step. Call this after evaluating validation metrics at the end of each training round.

```python
def step(validation_metric: Optional[float] = None) -> None
```

**What this does:**
1. Extracts hyperparameters, model parameters, and optimizer state from the local worker
2. Gathers validation metrics and hyperparameters from all workers via distributed communication
3. Calls the strategy's scoring and reduction logic (implementation-specific)
4. Injects the updated hyperparameters, model, and optimizer state back into the local worker

**Parameters:**
- `validation_metric` (float): Local worker's validation metric (e.g., validation loss). Lower is better.

**Example:**
```python
ROUND_LENGTH = 1000

for round_idx in range(num_rounds):
    # Train for ROUND_LENGTH steps
    for step in range(ROUND_LENGTH):
        loss = train_step(model, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Evaluate
    val_loss = evaluate(model, val_loader)

    # Round-end: strategy updates hyperparameters and model
    strategy.step(val_loss)
```

**Raises:**
- `ValueError`: If `validation_metric` is None after initialization

---

### Checkpointing

#### `state_dict()`

Returns the hyperparameter schema for checkpointing.

```python
def state_dict() -> Dict[str, Dict[str, Any]]
```

**Returns:** Schema dictionary suitable for serialization.

**Example:**
```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "strategy": strategy.state_dict(),
}
torch.save(checkpoint, "checkpoint.pt")
```

---

#### `load_state_dict()`

Restores the hyperparameter schema from a checkpoint.

```python
def load_state_dict(schema: Dict[str, Dict[str, Any]]) -> None
```

**Parameters:**
- `schema` (Dict): Schema dictionary to restore

**Example:**
```python
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
strategy.load_state_dict(checkpoint["strategy"])
```

**Raises:**
- `ValueError`: If schema violates invariants

---

## Properties

### `valid_binding_targets`

Returns list of bindable hyperparameter paths available in the optimizer.

```python
@property
def valid_binding_targets() -> List[str]
```

Use this to discover what hyperparameters you can bind to.

**Example:**
```python
strategy = make_top_score_strategy(optimizer)
print("Available hyperparameters:", strategy.valid_binding_targets)
# Output: ['lr', 'weight_decay', 'betas', ...]

# Bind to available parameters
strategy.bind_log_hyperparameter("lr", std=0.1, min=1e-4, max=1e-1)
```

---

### `optimizer`

Access the underlying optimizer instance.

```python
@property
def optimizer() -> torch.optim.Optimizer
```

Inherited from PyTorch's `LRScheduler`. Provides access to the wrapped optimizer.

---

## Schema Format

The schema defines how hyperparameters are perturbed and bounded. It's built via `bind_*` methods but can also be provided directly to factory functions.

**Schema structure:**
```python
{
    "lr": {
        "type": "log",      # "log" or "linear"
        "std": 0.1,         # standard deviation for perturbation
        "min": 1e-4,        # minimum bound (required for log)
        "max": 1e-1,        # maximum bound (optional)
        "shared": True      # shared across param groups
    },
    "weight_decay": {
        "type": "linear",
        "std": 0.001,
        "min": 0,
        "max": 0.1,
        "shared": False     # per-group values
    }
}
```

**Fields:**
- `type`: `"log"` (log-space) or `"linear"` (linear-space)
- `std`: Standard deviation for perturbation (must be > 0)
- `min`: Minimum bound (required for log, optional for linear)
- `max`: Maximum bound (optional, must be > min)
- `shared`: If True, one value shared across all param groups; if False, independent per-group values

---

## Design Notes

**Why strategies inherit from LRScheduler:**

This makes them compatible with PyTorch's scheduler ecosystem and training frameworks that expect scheduler interfaces (checkpointing, learning rate logging, etc.).

**Shared vs Per-Group Hyperparameters:**

- `shared=True`: One value applied to all parameter groups (typical for `lr`, `weight_decay`)
- `shared=False`: Independent values per parameter group (useful for layer-wise learning rates)

**Validation Metric Convention:**

Lower is better. Strategies select/blend workers with lower validation metrics (e.g., validation loss). If using a metric where higher is better (accuracy), negate it before passing to `step()`.

---

## Next Steps

- See [Concrete Strategies](concrete_strategies.md) for specific strategy implementations and their behaviors
- See [User Guide](user_guide_outline.md) for conceptual understanding and usage patterns
- [TODO: Training loop integration examples]
