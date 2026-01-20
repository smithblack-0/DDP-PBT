# DDP-PBT: Compute-Efficient Population Based Training

## Overview

Population Based Training (PBT) continuously adapts hyperparameters throughout training, tracking the optimal hyperparameter basin as the loss landscape evolves. This eliminates manual learning rate schedules and enables dynamic tuning of weight decay, optimizer betas, and other hyperparameters.

**The problem**: Standard PBT wastes ~(N-1)/N training compute by killing underperforming workers and copying top performers. With 8 workers, 87.5% of compute is thrown away.

**The proposal**: Share gradients across all workers via DDP while applying different optimizer hyperparameters to each. This recaptures wasted compute (all gradients contribute to final model) while still enabling hyperparameter exploration through parameter-space divergence.

**The bet**: Weak per-round signals from limited divergence, accumulated over many rounds, outperform strong per-round signals with massive compute waste; there is sufficient room to detect this difference in training performance while sharing mostly relevant gradients.

**Why it might work**: DDP gradient synchronization eliminates batch subset noise - the primary expected noise source. Validation differences reflect pure hyperparameter effects.

## Core Algorithm

Presume the existance of a STRATEGY object containing the necessary additional abstraction. Note that initialize proceeds the same way on all devices, and reduction produces the same feature on all devices.

```
Initialize:
  - model = initial_model
  - optimizer = initial_optimizer
  - hyperparams = initial_hyperparams
  
  
Each round:
  1. Perturb hyperparams: hyperparams_i = PERTURB(hyperparams)
  2. Train for ROUND_LENGTH steps with DDP gradient synchronization
  3. Evaluate all workers on validation → validation_score_i
  4. Reduce for next round across all devices:
     - scores = STRATEGY.score(validation_score)
     - hyperparams = STRATEGY.reduce_hyperparameters(scores, hyperparameters)
     - model = STRATEGY.reduce_models(scores, model)
     - optimizer = STRATEGY.reduce_optimizer(scores, hyperparams, optimizer)
  5. Repeat
```

The main relevant hyperparameter here is:

- ROUND_LENGTH

## Key Design Principles

**Gradient sharing eliminates compute waste**: Unlike standard PBT where underperforming workers are killed (wasting their training compute), all workers contribute gradients via DDP. Every gradient computed contributes to the final model.

**Short rounds prevent harmful divergence**: Optimizer hyperparameters cause parameter divergence, but keeping rounds short (ROUND_LENGTH) and resetting to a common model by reduction at the end of the round ensures models stay close enough that shared gradients remain relevant to all workers.

**Separation of optimization and parameters**: While all workers receive the same synchronized gradients, they apply different optimizer hyperparameters. This creates measurable performance differences without breaking gradient relevance. Parameter diversion typically occurs much slower than gradient divergent; for instance, overregularization decreases parameter magnitude while gradients still point in roughly the same direction.

**No batch subset noise**: DDP synchronizes gradients before optimizer steps, so all workers effectively see the same batch. Validation differences reflect pure hyperparameter effects, not lucky/unlucky batch assignments.

**No more scheduling or dynamic tuning**: If it is an optimizer trait, it just probes around and follows the best path, no fuss needed.

## Critical Assumption

**There exists a viable operating zone**: Parameter divergence can be large enough to produce detectable hyperparameter performance in validation signals, but small enough that shared gradients remain mostly relevant to all workers. While some training degration is acceptable, it is minor in comparison to the amount of compute lost in a standard PBT run.

**The model starts in a good basin**: The model starts in a basin using conventional tuning where it can train easily. If the model is already in an unstable region this will perform poorly due to the greedy nature of the tuning algorithm. 

## Known Risks / Failure Modes

**No viable operating window**: The window between "diverged enough for signal" and "too diverged for gradient sharing" might not exist or be too narrow to exploit.
    
**Insufficient signal-to-noise**: Hyperparameter effects might be too subtle to detect within ROUND_LENGTH steps, leading to random walks of such magnitude they degrade rather than improve performance.

**Bad Hyperparameter Descent Basins**: It is possible and common for the hyperparameter manifold to present descent tracts that look locally optimal but are globally catastrophic; following these tracts then leads to worse performance in the end. 

## Perturbations and Exploration.

The pertubations and exploration engine is fairly simple. Pertubations are done by the relevant sampling mechanism for that hyperparameter dimension and in an uncorrolated manner. 

- A hyperparameter can be defined as a 'log' or a 'linear' parameter.
- Log parameters can be in zero to infinity, and are declared with a std deviation indicating the percent change on the normal draws that perturb them. They have a "max" and "min" parameter that can be optionally defined.
- Linear parameters are defined with a max, a min, and a std for perturbation. 

To perturb, we convert log parameters into log space, and leave linear parameters alone. Then we draw random normal samples with the indicated deviation and add them to the current state. We apply minimums and maximums. We convert log space back. This is now the new hyperparameter states.

## Strategy Options     

These need to be tested empirically. Note that in cases that average, when averaging logspace hyperparameters the averaging is done in logspace too, then converted back. 

### Top Score

Fairly straightforward. The top score is selected. That model goes onto the next round. If we are lucky this is enough by itself, and it is by far the simplest design. All reductions just copy that model, then mutate the hyperparameters at the beginning of the round.   

### Top-K

One of the Top-K models is randomly chosen for further training. Moderately better at exploration, but slows down tracking and convergence.

*Relevant Hyperparameter*:

 - Percent_Chosen: Percentage of world chosen from

### Weighted Average

Subtract off the minimum validation score from all validation scores, then normalize the resulting scores. This is treated as the weights as we do a weighted combination of the hyperparameters, optimizer, and model parameters by how well the scores did under this metric. Somewhat robust to noise, but not incredibly so.

### Top Average

Take the Top-K elements. Average their hyperparameters, models, and optimizers together. That is now the next rounds model. 

*Relevant Hyperparameter*:

 - Percent_Chosen: Percentage of world chosen from

## Top Population

The top-k entries are selected. Workers randomly select a pair of models from this group, crossbreed them, add a small pertubation, and that becomes the next round's options. This retains some of the advantages of PBT, but it is unknown if the gradient updates will remain mutually intelligible. Still, if species can crossbreed, maybe models can too, and certaintly the groups of models that have mutually intelligible updates should tend to do better.

*Relevant Hyperparameter*:

 - Percent_Chosen: Percentage of world chosen from

## Primary Research Questions

- Is there sufficient operating room where gradient sharing preserves training efficiency while parameter divergence creates detectable hyperparameter performance signals?
- Is one of these performant enough to work?
- In the web of hyperparameter choices, what are reasonable ones?

## Next steps.

- Build a lightning harness on top of GPT-2-small (Kapathy's know good defaults)
- Use, ray-tune with lightning to explore over short runs manifolds in terms of num_rounds, percent chosen, for each strategy.  
- Run control, and DDP-PBT full runs for each best strategy. 