# README.md Outline

## Questions Needing Answers

**[Q1] Autobind API**
- Does autobind utility exist?
- If yes: What's the API? `strategy.autobind(["lr", "weight_decay"])` or similar?
- If no: What manual binding pattern should quick start show?

**[Q2] DDP Setup Order**
- Correct sequence: model → DDP wrap → optimizer → strategy?
- Any gotchas or import requirements to show?

**[Q3] Round Length Guidance**
- Typical ROUND_LENGTH value to show in example?
- Or should example say "# TODO: tune this for your problem"?

**[Q4] Training Loop Integration**
- Complete working example available to reference?
- Where does validation fit? After full round or interleaved?

**[Q5] Target Audience Framing**
- How should "Who Needs This" be framed?
- When NOT to use (boundaries)?
- Prerequisites to state?

**[Q6] Diagrams**
- Should README include diagram of gradient sharing + hyperparameter divergence?
- Or keep text-only?

**[Q7] What NOT to Bind**
- Emphasize caveat about dropout/model properties in main README?
- Or defer to user guide?

**[Q8] Examples Strategy**
- Link to examples/ folder?
- Or inline complete training code in README?

---

## Section-by-Section Plan

### Title & One-Liner
**Thrust:** "DDP-PBT: Compute-Efficient Population Based Training" + one sentence about gradient sharing eliminating compute waste

**Modified by:** None - this is solid

### The Problem
**Thrust:** Standard PBT wastes (N-1)/N compute by killing workers. With 8 workers, 87.5% thrown away.

**Modified by:** None - this exists and is good

### The Solution
**Thrust:** Share gradients via DDP, each worker uses different optimizer hyperparameters, all contribute gradients while exploring.

**Modified by:** [Q6] - diagram yes/no affects how this is presented

### Installation
**Thrust:**
```bash
pip install ddp-pbt
```

**Modified by:** None - straightforward

### Quick Start
**Thrust:** Show minimal working example from imports through one training round

**Modified by:** [Q1] (autobind API), [Q2] (DDP setup order), [Q3] (ROUND_LENGTH value), [Q4] (training loop pattern)

**Current blocker:** All four questions affect this section

### Who Needs This?
**Thrust:** "You need this if..." and "You don't need this if..." to filter audience

**Modified by:** [Q5] - user's framing of audience and boundaries

### How It Works (Current 4 bullets)
**Thrust:** Keep existing 4 conceptual points:
1. Gradient sharing eliminates waste
2. Short rounds prevent divergence
3. Different hyperparameters create measurable differences
4. No batch subset noise

**Modified by:** [Q6] - diagram addition, [Q3] - if we explain "short rounds" need round length context

### What Can You Tune?
**Thrust:** List of optimizer hyperparameters (lr, weight_decay, momentum, etc.) with caveat about NOT binding model properties

**Modified by:** [Q7] - how prominent to make the caveat

### Available Strategies
**Thrust:** Table with 5 strategies, brief "Best For", links to concrete_strategies.md

**Modified by:** None - can write this now from existing docs

### Documentation Links
**Thrust:** Links to:
- API Reference (abstract_strategy.md)
- Strategies (concrete_strategies.md)
- User Guide (user_guide_outline.md - note in progress)
- Internal (implementation_plan.md)

**Modified by:** None - straightforward

### Development Setup
**Thrust:** venv creation, pip install -e .[dev], pytest, black/ruff commands, note about WSL for tests

**Modified by:** None - exists and is good

### Research Status
**Thrust:** Active research project disclaimer, link to proposal.md

**Modified by:** [Q3] quality - how prominent should warning be?

### Citation
**Thrust:** Existing bibtex

**Modified by:** None - straightforward

---

## Can Write Immediately

Sections NOT blocked by questions:
- Title & One-Liner
- The Problem
- The Solution (text, pending diagram decision)
- Installation
- Available Strategies Table
- Documentation Links
- Development Setup
- Citation

## Blocked Sections

Cannot complete until questions answered:
- Quick Start (needs Q1, Q2, Q3, Q4)
- Who Needs This (needs Q5)
- What Can You Tune (needs Q7 for emphasis)
- Potentially "How It Works" (needs Q6 for diagram, Q3 for round context)

---

## Implementation Notes

**Autobind (related to Q1):**
- Autobind object implemented in implementation_plan.md (lines 583+)
- Primary user interface with prototype pattern
- API: `Autobind(file_path, logging_callback)`, bind methods, `__call__(strategy, only=[])`
- Training Loop Example updated to show autobind usage (lines 13-44)
- Q1 partially resolved but still need user input on default hyperparameter values for autobind_defaults.json
