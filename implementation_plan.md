# DDP-PBT Implementation Plan

## Overview

DDP-PBT implements Population Based Training with gradient sharing via DDP, eliminating compute waste while enabling hyperparameter exploration.

**The problem**: Standard PBT wastes ~(N-1)/N training compute by killing underperforming workers and copying top performers.

**The solution**: Share gradients across all workers via DDP while applying different optimizer hyperparameters to each. All gradients contribute to the final model while enabling hyperparameter exploration through parameter-space divergence.

**This document** is the living specification. As implementation progresses, this must stay synchronized with code - a mismatch is a bug.

## Training Loop Example

```python
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from ddp_pbt import make_top_score_strategy

# User wraps model with DDP
model = DDP(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Create strategy via factory with builder pattern
strategy = make_top_score_strategy(optimizer)
strategy.bind_log_hyperparameter("lr", std=0.1, min=1e-4, max=1e-1)
strategy.bind_linear_hyperparameter("weight_decay", std=0.001, min=0, max=0.1)

# Training loop - user manages rounds
ROUND_LENGTH = 1000
for round_idx in range(num_rounds):
    # Train with DDP (gradient sharing happening)
    for step in range(ROUND_LENGTH):
        loss = train_step(model, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Evaluate
    val_loss = evaluate(model, val_loader)

    # Round end: reduce across workers, perturb for next round
    strategy.step(val_loss)
```

## Design Principles

**Composition over decomposition**: Only abstract what varies between strategies. State extraction, communication, and perturbation are shared infrastructure. Strategy selection logic varies and is the extension point.

**Explicit data structure contracts**: Method signatures declare exact data structures (Schema, Hyperparameter Values, World Weights, Tensor PyTrees). Enables black-box testing and clear boundaries.

**Functional where possible**: Strategies receive data, use Communication for distributed side effects, return new data. AbstractStrategy handles extraction/injection via State. Keeps strategy logic testable.

**Hyperparameters live in optimizer**: No separate storage. Schema defines how to interact with optimizer.param_groups. State extracts/injects as needed.

**Memory-efficient communication**: Leaf-by-leaf operations for large tensors. Filter zero-weight entries before stacking. Process one leaf at a time to avoid memory overhead.

**Schedule interface**: Implements state_dict/load_state_dict, exposes optimizer, provides binding methods. Integrates with existing training patterns.

**Contract Driven Development**: Know what is going to happen, and progressively make finer abstractions.

## Success Criteria

**Architecture**:
- Strategies are easy to implement (minimal boilerplate)
- Adding new strategies doesn't change AbstractStrategy
- Clean separation: strategy logic vs distributed plumbing
- All components testable in isolation
- Integrates cleanly into training loops

**Testing**:
- Comprehensive test coverage (all public APIs tested)
- All tests follow black-box testing contract
- Integration test with toy model running full DDP-PBT rounds
- Tests pass across Python 3.9-3.12

**Code Quality**:
- Type hints on all public methods
- Code formatted with black (line length 100)
- Passes ruff linting
- Clear variable names and minimal comments (code is documentation)

**Documentation**:
- Implementation plan stays synchronized with code
- All public methods have clear contracts
- User-facing documentation created before release (if applicable)

## Core Data Structures

### Hyperparameter Schema

The schema defines how to interact with and mutate hyperparameters in `optimizer.param_groups`. It is NOT stored in the optimizer - it is external configuration that controls perturbation behavior, bounds, and shared vs per-group semantics for hyperparameters that already exist in the optimizer.

The schema is a dictionary where each key is a hyperparameter name (must exist in at least one of the optimizer.param_groups) and each value is a configuration dict.

**Schema format:**
```python
{
    "lr": {
        "type": "log",           # "log" or "linear"
        "std": 0.1,              # standard deviation for perturbation
        "min": 1e-4,             # optional: minimum bound
        "max": 1e-1,             # optional: maximum bound
        "shared": True           # if True: shared across all param groups
    },
    "weight_decay": {
        "type": "linear",
        "std": 0.001,
        "min": 0,                # optional: minimum bound
        "max": 0.1,              # optional: maximum bound
        "shared": False          # if False: per-param-group values
    }
}
```

**Schema field definitions:**
- **type**: "log" (sample in log-space) or "linear" (sample in linear-space)
- **std**: Standard deviation for normal distribution perturbation
- **min** (optional): Minimum bound to enforce after perturbation
- **max** (optional): Maximum bound to enforce after perturbation
- **shared**: If True, hyperparameter is shared across all param groups (broadcast on set, return single value on get). If False, each param group has independent value.

**Hyperparameter values format (dict with lists):**
```python
{
    "lr": [0.001],                    # shared: length 1 list
    "weight_decay": [0.01, 0.005]     # per-group: length = num param groups
}
```

All hyperparameters live in `optimizer.param_groups`. Schema is used to:
- Identify which fields are tunable
- Define perturbation behavior
- Control shared vs per-group semantics

### Hyperparameter Values

The runtime representation of hyperparameter values. This is what flows through the system (extracted from optimizer, perturbed, crossbred, set back).

**Values format (dict with lists):**
```python
{
    "lr": [0.001],                    # shared: length 1 list
    "weight_decay": [0.01, 0.005]     # per-group: length = num param groups
}
```

**Key properties:**
- Always dict of lists (even for single value)
- Shared parameters: length-1 list
- Per-group parameters: length = number of param groups in optimizer
- Permuter and Crossbreeder consume and produce this format
- State extracts this from optimizer and injects it back

### Tensor PyTrees

PyTrees (nested structures of tensors) represent model parameters and optimizer state tensors.

**Key property:**
- Matching leaf structure across devices is the only requirement
- Same tree structure, same tensor shapes at each leaf
- Used for distributed reduction operations (reduce_to_one, reduce_with_weights)
- Leaf-by-leaf operations enable memory-efficient communication

**Example:**
```python
{
    "layer1.weight": Tensor(...),
    "layer1.bias": Tensor(...),
    "layer2.weight": Tensor(...)
}
```

Unstructured pytrees never escape state.

### Dictionary Tree

A dictionary tree is a representation of a pytree parsable by the internal utilities. It consists of a dictionary of unique path strings to value. ell.

**Key property:**
- Flat: One layer exists. There is the path rendered uniquely as a string, and a value
- Base primitive of just about everything above state.

### List of Dictionary Trees

A List of Dictionary Trees is used with hyperparameters to maintain per-parameter-group structure. Model and optimizer tensors use a single Dictionary Tree where paths encode param group membership.

**Key property:**
- Fixed depth: List, then Dictionary Tree.
- Used only for hyperparameters where per-group structure must be preserved explicitly.

### World Weights

A list of floats representing the importance/weight of each worker, used during selection and reduction.

**Format:**
```python
[0.1, 0.3, 0.4, 0.2]  # 4 workers, sums to 1.0
```

**Key properties:**
- Length = world_size (number of distributed workers)
- Sum = 1.0 (normalized weights)
- Used by strategies to:
  - Rank workers (sort by weight)
  - Select top-K workers
  - Weight blending operations (crossbreeding, weighted average)
- Computed from validation metrics by strategies

### World Hyperparameters

A list of Hyperparameter Values, one per world. Used to represent hyperparameters gathered from all workers.

**Format:**
```python
[
    {"lr": [0.001], "weight_decay": [0.01, 0.005]},  # world 0
    {"lr": [0.0008], "weight_decay": [0.009, 0.004]},  # world 1
    {"lr": [0.0012], "weight_decay": [0.011, 0.006]},  # world 2
    ...
]
```

**Key properties:**
- Length = world_size (number of distributed workers)
- Each element is a Hyperparameter Values dict
- Gathered from all workers via Communication.gather_list()
- Will eventually be reduced down to the hyperparameters for the next round by the strategy.
---

## Core Objects

### 1. State
**Responsibility**: Extract and inject tensors and hyperparameter values from optimizer

**Why needed**:
- state_dict() contains metadata (bools, ints, structure) that can't be averaged
- Need clean tensor extraction for communication
- Hyperparameters live in optimizer.param_groups, need clean access
- Handle shared vs per-group hyperparameters transparently

**Interface**:
- `valid_hyperparameter_paths` (property) → returns list of float-valued paths found in param_groups (bindable hyperparameters)
- `setup_schema(schema)` → configure which hyperparameters to manage and their behavior (takes Schema)
- `get_model_tensors()` → returns Dictionary Tree of the model parameters (all param groups in one tree)
- `set_model_tensors(dict_tree)` → inject Dictionary Tree back into model via optimizer in-place
- `get_optimizer_tensors()` → returns Dictionary Tree of optimizer state tensors (momentum buffers, etc., all param groups in one tree)
- `set_optimizer_tensors(dict_tree)` → inject Dictionary Tree back into optimizer state in-place
- `get_hyperparam_values()` → returns Hyperparameter Value
  - If shared=True: returns length-1 list `{"lr": [0.001]}`
  - If shared=False: returns per-group list `{"weight_decay": [0.01, 0.005]}`
- `set_hyperparam_values(values)` → update values in optimizer.param_groups (takes Hyperparameter Values)
  - If shared=True: broadcasts single value to all param groups
  - If shared=False: sets per-group values

**Dependencies**:
- Optimizer (held as reference)
- Schema (held as reference after setup_schema called)

---

### 2. Permuter
**Responsibility**: Hyperparameter mutation logic

**Why separate**: All strategies need perturbation, same logic for all

**Interface**:
- `setup_schema(schema)` → configure perturbation behavior (type, std, bounds)
- `perturb(values)` → returns perturbed Hyperparameter Values dict
  - Perturbs each list element independently (separate random draws per element)
  - Additive perturbation: samples normal(0, std) and adds to current value
  - For log parameters: converts to log-space, adds sample, converts back
  - For linear parameters: adds sample directly
  - Clips to bounds after perturbation to ensure valid structures
  - Same schema config applies to entire list for that hyperparameter

**Dependencies**:
- Schema (held as reference after setup_schema called)

---

### 3. Crossbreeder
**Responsibility**: Parent selection and blending operations with probabilistic mutation

**Why separate**: Shared crossbreeding/blending logic for strategies that average or crossbreed

**Stateful configuration**:
- `parent_pool_depth` (int): Number of top workers to draw parents from
- `mutation_rate` (float): Probability of calling permuter on crossbred result before returning

**Interface**:
1. `setup_schema(schema)` → configure blending behavior for hyperparameters (takes Schema)
2. `setup_permuter(permuter)` → inject Permuter for probabilistic mutation
3. `select_parents(validation_losses)` → returns filtered World Weights
   - Ranks workers by weight
   - Randomly selects 2 parents from top parent_pool_depth entries
   - Returns World Weights with non-zero values only for selected parents (sums to 1.0)
4. `crossbreed_hyperparameters(world_hyperparameters, parent_weights)` → returns Hyperparameter Values
   - Takes World Hyperparameters and parent World Weights
   - Interprets parent_weights to find 2 active indices
   - Blends those two hyperparameter sets with 50% each.
   - For log parameters: converts to log-space, blends, converts back
   - For linear parameters: blends directly
   - With mutation_rate probability: calls permuter on result
   - Returns blended (possibly mutated) Hyperparameter Values

**Pattern**:
- select_parents produces filtered World Weights
- Same World Weights used by Communication.reduce_with_weights for tensor blending
- Hyperparameters need special handling via crossbreed_hyperparameters

**Dependencies**:
- Schema (held as reference for hyperparameter blending)
- Permuter (held as reference for probabilistic mutation)

---

### 4. Communication
**Responsibility**: Distributed gathering and reduction operations

**Why separate**:
- Encapsulates distributed concerns
- Testable via mocking
- Standardized operations for all distributed data types

**Interface**:
- `gather_pytree_list(pytree)` → returns list of pytrees from all workers
  - Walks pytree structure, gathering each leaf from all workers
  - Returns list of reconstructed pytrees (length = world_size)
  - Handles type conversions (tensors and other types)
  - Used for: validation metrics, hyperparameters, any small data

- `reduce_by_world_weights(world_weights, pytree)` → returns single reduced pytree
  - All-gathers each leaf from all workers
  - Filters to non-zero weights before stacking (memory efficient for large tensors)
  - Computes weighted sum: sum(weight_i * leaf_i for non-zero weights)
  - Processes one leaf at a time to avoid large memory overhead
  - Used for: model parameters, optimizer state tensors

**Memory efficiency**:
- Leaf-by-leaf operations avoid gathering full structures at once
- Filtering zero-weight entries before stacking saves memory for sparse weight distributions
- For 2 parents out of 8 workers: stacks [2, ...] instead of [8, ...] per leaf

**Dependencies**: None (wraps torch.distributed)

---

### 5. AbstractStrategy
**Responsibility**: Schema storage, configuration loading, and round-end orchestration.

**Why needed**:
- Central owner of the Schema - builds it through configuration methods
- Orchestrates the round-end flow: extracts data via State, gathers distributed data via Communication, calls strategy methods, injects results back via State
- Defines the contract that all strategies must implement
- Provides schedule interface (state_dict, optimizer property)
- It is a torch schedule. 

**What it holds**:
- **State** (injected) - used to extract/inject optimizer and model data
- **Schema** (built internally) - configuration for hyperparameters
- **Communication** (injected) - used for gathering distributed data

**Constructor**:
- `__init__(state, communication, config={})`
  - State injected (holds optimizer reference)
  - Communication injected (for gathering world data)
  - config dict can provide initial native JSON Schema (default empty)

**Properties**:
- `valid_binding_targets` → list of float field names available in optimizer.param_groups for binding (delegates to State.valid_hyperparameter_paths)
- `optimizer` → exposes the optimizer from State

**Configuration** (builds Schema):
- `bind_log_hyperparameter(name, std, min=None, max=None, shared=True)` → builder method for log parameters
- `bind_linear_hyperparameter(name, std, min=None, max=None, shared=True)` → builder method for linear parameters
- Constructor accepts native JSON config dict
- All methods validate that hyperparameter exists in at least one optimizer.param_group
- Schema is passed to State via `state.setup_schema(schema)` after configuration

**Serialization**:
- `state_dict()` → returns Schema for checkpointing
- `load_state_dict(schema)` → restores Schema from checkpoint

**Flow in step(validation_metric)**:
1. Get local data from State: hyperparams, model_pytree, optimizer_pytree
2. Gather world_hyperparameters via Communication
3. Call abstract `score(validation_metrics, communication)` → returns World Weights
4. Call abstract `reduce_hyperparameters(world_weights, world_hyperparameters, communication)` → returns Hyperparameter Values
5. Call abstract `reduce_models(world_weights, model_pytree, communication)` → returns updated model pytree
6. Call abstract `reduce_optimizer(world_weights, optimizer_pytree, communication)` → returns updated optimizer pytree
7. Inject results back via State

**Abstract methods** (concrete strategies must implement all 4 coherently):
- `score(validation_metrics, communication)` → World Weights
- `reduce_hyperparameters(world_weights, world_hyperparameters, communication)` → Hyperparameter Values
- `reduce_models(world_weights, model_pytree, communication)` → updated model Tensor PyTree
- `reduce_optimizer(world_weights, optimizer_pytree, communication)` → updated optimizer Tensor PyTree

**Pattern**: These 4 methods are one coherent algorithm, must be implemented together. Communication passed as parameter to each method. Strategies never access State directly.

**Dependencies**:
- State (injected) - for extraction/injection
- Communication (injected) - for gathering distributed data
- Concrete strategies receive additional dependencies (Permuter, Crossbreeder) via their own injection

---

### 6. Concrete Strategies

Each strategy implements the 4 abstract methods coherently. Dependencies are injected via factory.

#### TopScoreStrategy

Note to claude: This is clearly out of date. You need to go fix it.

**Intention**:

All workers start with the same model, then permute it.
Best one wins on all workers. We locally permute that 
model on all workers for the next round. 

**Dependencies**: Permuter (injected)

**Implementation**:
- `score(validation_metrics, communication)`:
  - Gather validation metrics from all workers via communication
  - Find argmax (best worker)
  - Return World Weights: 1.0 at best index, 0.0 elsewhere

- `reduce_hyperparameters(world_weights, world_hyperparameters, communication)`:
  - Extract winner's Hyperparameter Values from world_hyperparameters
  - Perturb via Permuter
  - Return perturbed Hyperparameter Values

- `reduce_models(world_weights, model_pytree, communication)`:
  - Use communication.reduce_by_world_weights(world_weights, model_pytree)
  - Returns winner's model (weights filter out others)

- `reduce_optimizer(world_weights, optimizer_pytree, communication)`:
  - Use communication.reduce_by_world_weights(world_weights, optimizer_pytree)
  - Returns winner's optimizer state
---

#### TopKStrategy

**Intention**:

All workers start from the same model. We globally choose one of the topk
workers to move onto the next round. All workers then permute this into
a new variation.

**Dependencies**: Permuter (injected)

**Configuration**: k (int) or percent_chosen (float) - number/fraction of top workers to sample from

**Implementation**:
- `score(validation_metric, communication)`:
  - Gather validation metrics from all workers
  - Rank workers by metric
  - Randomly select one from top-K
  - Propose choice
  - Choose the rank 0's worker's choice. 
  - Return World Weights: 1.0 at selected index, 0.0 elsewhere

- `reduce_hyperparameters(world_weights, world_hyperparameters, communication)`:
  - Extract selected worker's Hyperparameter Values
  - Perturb via Permuter
  - Return perturbed Hyperparameter Values

- `reduce_models(world_weights, model_pytree, communication)`:
  - Use communication.reduce_by_world_weights(world_weights, model_pytree)

- `reduce_optimizer(world_weights, optimizer_pytree, communication)`:
  - Use communication.reduce_by_world_weights(world_weights, optimizer_pytree)

---

#### WeightedAverageStrategy

**Intention**

Average together the better permutation; only one model moves forward per round

All workers receive the scores, and form the same weighted average out of this
score. These weights are then applied in all models producing the same model everywhere.
This model is then permuted.

**Dependencies**: Permuter

**Implementation**:
- `score(validation_metric, communication)`:
  - Gather validation metrics from all workers.
  - Normalize to weights: subtract min, then normalize to sum=1; in edge case all zero all become equal.
  - Return World Weights (all non-zero)

- `reduce_hyperparameters(world_weights, world_hyperparameters, communication)`:
  - Average hyperparameters together.
  - Permute them for unique workers per round

- `reduce_models(world_weights, model_pytree, communication)`:
  - Use communication.reduce_by_world_weights(world_weights, model_pytree)
  - Weighted average of all model parameters

- `reduce_optimizer(world_weights, optimizer_pytree, communication)`:
  - Use communication.reduce_by_world_weights(world_weights, optimizer_pytree)
  - Weighted average of all optimizer states

---

#### TopPopulationAsexualStrategy:

**Intention**

Simple algorithm allowing variation between workers with
a very simple genome strategy. A population is maintained
that is fit.

All workers recieve the validation results. They are ranked. 
Each worker indepedently chooses among the top k to move 
onto the next round. This produces k unique models.
These models are then perturbed to provide mutations.

**Dependencies**: Perturber, for mutation

**Implementation**:
- `score(validation_metric, communication)`:
  - Gather validation metrics from all workers
  - Sort and keep only the topk
  - On each worker, independently select a model to move forward
  - Setup world weights with only that model high (1.0) and all others low (0.0)
  - 
- `reduce_hyperparameters(world_weights, world_hyperparameters, communication)`:
  - Isolate the hyperparameters from the high parent
  - Perturb them for mutation purposes.
  - Returns Hyperparameter Values

- `reduce_models(world_weights, model_pytree, communication)`:
  - Use communication.reduce_by_world_weights(world_weights, optimizer_pytree)

- `reduce_optimizer(world_weights, optimizer_pytree, communication)`:
  - Use communication.reduce_by_world_weights(world_weights, optimizer_pytree)

#### TopPopulationSexualStrategy

**Intention**

More complex algorithm that involves mothers and fathers. 
Each worker ranks the results from all workers and keeps 
the topk, producing a pool of potential mothers and fathers.
Each worker then independently choose a new mother and father,
crossbreeds them by averaging everything 50/50, then mutates
the crossbreed with a random mutation chance.

Note that an important premise of this system is too much
diverity reduces fitness by making gradient updates incompatible
with the model state.

**Dependencies**: Crossbreeder (injected with parent_pool_depth, mutation_rate configs)

**Implementation**:
- `score(validation_metric, communication)`:
  - Gather validation metrics from all workers
  - Call Crossbreeder.select_parents(validation_metrics)
  - Returns filtered World Weights (2 parents selected, each has 505, sum=1.0, rest=0.0)

- `reduce_hyperparameters(world_weights, world_hyperparameters, communication)`:
  - Call Crossbreeder.crossbreed_hyperparameters(world_hyperparameters, world_weights)
  - Averages 2 parents with possibility of mutation per allele.
  - Returns Hyperparameter Values

- `reduce_models(world_weights, model_pytree, communication)`:
  - Use communication.reduce_by_world_weights(world_weights, model_pytree)
  - Averages 2 parent models.

- `reduce_optimizer(world_weights, optimizer_pytree, communication)`:
  - Use communication.reduce_by_world_weights(world_weights, optimizer_pytree)
  - Averages 2 parent optimizer states.

---

## Testing Guide

**Black-box testing only**: Test public methods and documented behavior. Tests validate the contract is honored, not that specific implementation approaches are used.

**What you CAN test:**
- Public methods with their documented input/output behaviors
- Observable state via public properties/methods
- Side effects on injected dependencies (checking optimizer.param_groups is fine - it's part of the contract)
- Documented invariants and error conditions

**What you CANNOT test:**
- Private methods or internal state (unless absolutely required - see below)
- Implementation details (which algorithm, internal data structures)
- Undocumented behavior

**When you MUST access private state:**
- Create ONE helper function at the top of the test file that isolates the access
- Name it clearly: `_test_helper_set_internal_field()` or similar
- This is the ONLY authorized access point - keeps coupling isolated for refactoring
- If you're accessing private state frequently, you're overcoupling - fix the design

**Test fixtures:**
- Helper functions at top of test file are allowed
- Can access/set private state for setup if no public alternative exists
- Individual test methods should still test black-box where possible

**Test naming:**
- Use role-based names: "Get State Test Suite - tests that get_state() retrieves wrapper state values"
- NOT: "Suite 1: Get State Tests" or numbered tests
- Use complete sentences describing what the test verifies

---

## Coding Guide

**Code is documentation**: Decompose properly with excellent variable names so HOW is clear. Comments explain WHY, not WHAT.

**Type hints required**: All public methods must have type hints. Internal methods encouraged.

**Error handling**: Fail fast, fail loud. Errors are your friend - they tell you something is wrong. This is a programmer's API, not an end-user application. Validate on initialization, maybe not in hot paths for performance.

**Style**:
- Follow black formatting (line length 100)
- Use ruff for linting
- Never use numbered enumeration in comments (Step 1, Suite 2) - brittle when reordered
- Comments use complete sentences

**Naming conventions**:
- Classes: PascalCase
- Functions/methods: snake_case
- Private: _leading_underscore
- Constants: UPPER_SNAKE_CASE

**Import style**:
- Test suite: Use `from src.ddp_pbt.base.X import Y` (absolute with src. prefix)
- Modules: Use relative imports `from .utilities import walk_single_pytree`
- Package __init__.py: Use relative imports, order primitives first (e.g., State before AbstractStrategy)
- Circular import resolution: Reorder imports so more primitive components come first
- This style avoids circular import issues by maintaining clear dependency hierarchy

---

## Development Workflow

**This document is the living spec**: As implementation progresses, keep this document synchronized with code. A mismatch between this plan and the code is a bug.

**Workflow**: Implementation Plan → Tests → Implementation → keep plan updated

**Before each release**:
- Version bump in pyproject.toml
- CHANGELOG.md entry
- All tests passing
- Code formatted (black, ruff)
- Documentation updated if needed

**Conflict resolution** (when docs/tests/code disagree):
1. IDENTIFY the conflict
2. GATHER context from all sources
3. ANALYZE what should be correct based on logic/evidence
4. RESOLVE with reasoning (don't just ask "what should it be?")
5. PROPAGATE fix through system


**Implementation Approach**

Start with base components, test in isolation, then build up to strategies:
1. Implement and test individual base components (State, Permuter, Crossbreeder, Communication)
2. Implement AbstractStrategy with configuration loading
3. Implement simplest strategy first (TopScoreStrategy) to validate design
5. End-to-end test with toy model; use GLOO backend. 
6. Document simple objects now their APIs are not changing
7. Implement, Test, Document remaining strategy

---

## Component Organization

**Base components** (reusable infrastructure):
- State - tensor extraction/injection
- Permuter - hyperparameter mutation
- Crossbreeder - hyperparameter blending
- Communication - distributed operations
- AbstractStrategy - base class with orchestration

**Strategy implementations** (concrete algorithms):
- TopScoreStrategy - select best worker
- TopKStrategy - select random from top-K
- WeightedAverageStrategy - blend all workers by score
- TopPopulationStrategy - crossbreed top-K pairs

**Each component colocates its factory** where applicable in the same code file

**Testing needs**:
- Unit tests for each base component (Communication needs mocking for distributed)
- Unit tests for each strategy implementation
- Integration test with toy model doing full training rounds

## Factory Pattern

Each concrete strategy has a factory function colocated in its module that:
- Creates State(optimizer)
- Creates Communication()
- Creates Permuter() and/or Crossbreeder() as needed by that strategy
- Creates ConcreteStrategy(state, communication, config)
- Wires all dependencies together
- Returns ready-to-use strategy instance

Factory accepts optimizer and optional config dict:
- Native JSON Schema config dict
- Builder pattern available on returned strategy object via bind methods

Example factory signature:
```python
def make_top_score_strategy(optimizer, config={}) -> TopScoreStrategy:
    state = State(optimizer)
    communication = Communication()
    permuter = Permuter()
    strategy = TopScoreStrategy(state, communication, config)
    # Permuter injected into strategy during construction
    return strategy
```

## Key Design Decisions

### Why State is separate
- state_dict() contains non-blendable metadata (bools, ints, structure)
- Clean extraction of tensors for communication
- Clean separation of hyperparam values from schema

### Why Communication is separate
- Memory efficiency crucial: leaf-by-leaf operations instead of gathering full state_dicts
- Testable via mocking
- Encapsulates distributed concerns

### Why Permuter and Crossbreeder are separate
- Shared logic across strategies that use them
- Crossbreeder is stateful (holds parent_pool_depth, mutation_rate config)
- Crossbreeder provides two operations: select parents, crossbreed hyperparameters
- Testable in isolation
- Injected into strategies that need them

### Why strategies have 4 methods not decomposed into separate objects
- The 4 methods (score, reduce_hyperparameters, reduce_models, reduce_optimizer) are conceptually ONE algorithm
- They must be implemented together coherently
- TopScore selects same winner for all; WeightedAverage blends all for all
- Not an axis of composition - you can't mix strategies

### Configuration approach
- Hybrid: Native JSON config dict + Builder methods
- User can choose their preferred format
- All convert to internal Schema: type + std + min/max (optional bounds)
- Current values live in optimizer.param_groups, not in Schema
- bind methods validate that hyperparameter exists in at least one optimizer.param_group

### Hyperparameters ARE optimizer parameters
- Hyperparameters (lr, weight_decay, etc.) live in optimizer.param_groups
- No separate hyperparameter storage
- Schema (type, std, min/max) tracked separately for perturbation rules
- State object extracts/injects values from/to optimizer.param_groups based on Schema

## Final Notes:

Implementers should check src/ddp_pbt/base/utilities before coding anything for relevant utilities. Lots of pytree work in this library!