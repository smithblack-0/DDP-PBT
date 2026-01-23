# User Guide Outline

## Questions Needing Answers

**[Q1] What is a Round?**
- Clear definition of round concept
- Typical ROUND_LENGTH values or how to choose
- Too short vs too long symptoms and consequences
- Does it depend on model size, dataset, problem type?

**[Q2] DDP Integration Order**
- Confirmed sequence: model → DDP → optimizer → strategy?
- Does strategy creation order matter relative to DDP wrapping?
- Common mistakes or gotchas?

**[Q3] LR Scheduler Interaction**
- Can strategies coexist with torch.optim.lr_scheduler?
- Do they conflict (both inherit from LRScheduler)?
- Is strategy a replacement or complement?
- Recommended pattern if both needed?

**[Q4] Round Length Tuning**
- Starting recommendations
- How to diagnose too short vs too long
- Empirical guidelines or rules of thumb
- Factors: model size, hyperparameter sensitivity, dataset size?

**[Q5] Perturbation std Tuning**
- Practical guidelines beyond formulas
- How to diagnose std too large vs too small in practice
- Does recommended std depend on strategy choice?

**[Q6] Strategy-Specific Parameters**
- TopK: choosing K (is world_size/4 reasonable?)
- TopPopulation: parent_pool_depth guidelines
- TopPopulationSexual: typical mutation_rate values

**[Q7] Common Problems**
- What does "diverging too much" look like in logs/metrics?
- What does "no exploration" look like?
- Memory issues specific to DDP-PBT?
- Other failure modes from testing/experience?

**[Q8] ScheduleAnything Integration**
- Does this integration exist/work?
- Example use case (adaptive gradient clipping per param group)?
- How to extend optimizer with custom params that DDP-PBT tunes?

**[Q9] Multi-Node Considerations**
- Setup differences for multi-node vs multi-GPU?
- Communication overhead considerations?
- Known issues or limitations?

**[Q10] Empirical Strategy Comparison**
- Have experiments shown which strategies work better for which problems?
- Any patterns observed (e.g., TopScore for CV, WeightedAverage for NLP)?

**[Q11] Guide Structure**
- Problem/solution pairs (like ScheduleAnything)?
- Progressive tutorial?
- Reference with decision trees?

---

## Section-by-Section Plan

### Introduction
**Thrust:** Bridge between API reference and practical usage. Prerequisites: PyTorch DDP, distributed training, hyperparameter optimization basics.

**Modified by:** None - straightforward framing

### Core Concept: Rounds
**Thrust:** A round is N training steps where workers independently train with DDP, then synchronize at round end. Explanation of round lifecycle.

**Modified by:** [Q1] - entire section depends on this answer

### Hyperparameter Spaces: Log vs Linear
**Thrust:**
- Log-space: spans orders of magnitude, multiplicative (lr, momentum, eps). Formula: `v_new = exp(log(v) + noise)`
- Linear-space: additive relationships, narrow range (weight_decay, clipping). Formula: `v_new = v + noise`
- When to use which based on parameter characteristics

**Modified by:** None - have this info from abstract_strategy.md

### Shared vs Per-Group Hyperparameters
**Thrust:**
- shared=True: one value for all param groups (typical: lr, weight_decay)
- shared=False: independent per-group (use for layer-wise tuning)
- Requires manual param_groups setup for per-group

**Modified by:** None - have this info

### Validation Metrics
**Thrust:** Lower is better convention. If using accuracy (higher=better), negate: `strategy.step(-val_accuracy)`

**Modified by:** None - straightforward

### DDP Setup Order
**Thrust:** Step-by-step setup sequence from model creation through strategy binding

**Modified by:** [Q2] - need confirmed correct pattern

### Checkpointing
**Thrust:** Save/load three components (model, optimizer, strategy). Note: strategy.state_dict() is schema only, not current values (those in optimizer state).

**Modified by:** None - have this pattern

### LR Scheduler Interaction
**Thrust:** Explanation of how strategies interact (or don't) with standard schedulers

**Modified by:** [Q3] - entire section depends on this

### Choosing a Strategy
**Thrust:** Decision tree based on problem characteristics:
- Noisy metrics → WeightedAverage
- Max exploration + crossbreed → Sexual
- Max exploration + no crossbreed → Asexual
- Fast convergence → TopScore
- Balanced → TopK

**Modified by:** [Q10] - empirical data could refine this tree

### What NOT to Bind
**Thrust:** Don't bind model evaluation properties (dropout, batchnorm momentum). Only bind optimizer hyperparameters affecting gradient→update. Why: gradient sharing prevents proper isolation.

**Modified by:** None - have this from implementation_plan.md

### Strategy Comparison
**Thrust:** Pro/Con table for each strategy with tradeoffs

**Modified by:** [Q10] - empirical observations could enhance this

### Tuning Round Length
**Thrust:** Guidelines for choosing ROUND_LENGTH

**Modified by:** [Q4] - entire section depends on this

### Tuning Perturbation std
**Thrust:**
- Log-space: std=0.1 (~10%), 0.2 (~20%), 0.5 (~50%)
- Linear-space: rule of thumb `std ≈ (max-min)/10`
- Tradeoff: larger = more exploration, less exploitation
- How to diagnose

**Modified by:** [Q5] - practical diagnostic guidance

### Strategy-Specific Parameter Tuning
**Thrust:** Guidelines for K, parent_pool_depth, mutation_rate

**Modified by:** [Q6] - entire section depends on empirical/theoretical guidance

### Common Problems
**Thrust:** Problem/symptom/solution patterns

**Modified by:** [Q7] - real failure modes from experience

### ScheduleAnything Integration
**Thrust:** Pattern for extending optimizer with custom params that DDP-PBT tunes (e.g., adaptive gradient clipping)

**Modified by:** [Q8] - does this work, example code

### Multi-Node Training
**Thrust:** Considerations for multi-node vs multi-GPU

**Modified by:** [Q9] - specific differences or caveats

### When NOT to Use
**Thrust:**
- Single-GPU training
- Can't structure into rounds
- Validation too expensive
- Need independent trajectories
- Hyperparameters already tuned

**Modified by:** None - clear boundaries

### Best Practices
**Thrust:**
- Start with TopScoreStrategy
- Use smaller std initially
- Monitor hyperparameter trajectories
- Validate on held-out data
- Clear round boundaries in code
- Checkpoint frequently
- Test with small ROUND_LENGTH first

**Modified by:** [Q4, Q5] - round length and std guidance affects recommendations

---

## Can Write Immediately

Sections NOT blocked:
- Introduction
- Log vs Linear Hyperparameter Spaces
- Shared vs Per-Group
- Validation Metrics
- Checkpointing
- What NOT to Bind
- Strategy Comparison (pro/con without empirical data)
- When NOT to Use
- Best Practices (general advice)

## Blocked Sections

Cannot complete without answers:
- Core Concept: Rounds (needs Q1)
- DDP Setup Order (needs Q2)
- LR Scheduler Interaction (needs Q3)
- Tuning Round Length (needs Q4)
- Tuning Perturbation std diagnostics (needs Q5)
- Strategy-Specific Parameters (needs Q6)
- Common Problems (needs Q7)
- ScheduleAnything Integration (needs Q8)
- Multi-Node (needs Q9)
- Choosing a Strategy refinement (could use Q10)

---

## Implementation Notes

**New Section Needed: Using Autobind for Configuration**
- Should be inserted after "DDP Setup Order", before "Checkpointing"
- Thrust: Autobind is primary interface for configuration, explain prototype pattern, note manual binding is for runtime changes only
- Modified by: None - have specification from implementation_plan.md
- Can be written now

**Best Practices Updates:**
- Add bullet: "Use Autobind for configuration, save custom defaults for project/team sharing"
- Add bullet: "Don't mix manual binding + autobind (fighting the library)"

**State Clipping Behavior:**
- State.get_hyperparam_values() clips values to schema bounds during extraction
- Optimizer values are internal/unreliable - State is guardian at read boundary
- Ensures strategies always receive valid values (important for distributed consistency)
- Schema held by reference, can mutate after setup - clipping on get handles this
- Example: optimizer with weight_decay=0, schema min=1e-6 → get_hyperparam_values returns 1e-6
- User doesn't need to manually ensure optimizer values stay in bounds
