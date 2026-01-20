# DDP-PBT: Compute-Efficient Population Based Training

Population Based Training (PBT) continuously adapts hyperparameters throughout training, tracking the optimal hyperparameter basin as the loss landscape evolves. This eliminates manual learning rate schedules and enables dynamic tuning of weight decay, optimizer betas, and other hyperparameters.

## The Problem

Standard PBT wastes ~(N-1)/N training compute by killing underperforming workers and copying top performers. With 8 workers, 87.5% of compute is thrown away.

## The Proposal

Share gradients across all workers via DDP while applying different optimizer hyperparameters to each. This recaptures wasted compute (all gradients contribute to final model) while still enabling hyperparameter exploration through parameter-space divergence.

## Key Features

- **Zero compute waste**: All workers contribute gradients via DDP
- **Continuous hyperparameter optimization**: No more manual scheduling or dynamic tuning
- **Multiple reduction strategies**: Top-score, Top-K, weighted average, and more
- **Built on PyTorch Lightning and Ray Tune**: Production-ready distributed training

## Installation

```bash
pip install ddp-pbt
```

For development:

```bash
git clone https://github.com/smithblack-0/DDP-PBT.git
cd DDP-PBT
pip install -e .[dev]
```

## Quick Start

```python
# Example coming soon
```

## How It Works

1. **Gradient sharing eliminates compute waste**: Unlike standard PBT where underperforming workers are killed, all workers contribute gradients via DDP
2. **Short rounds prevent harmful divergence**: Keeping rounds short ensures models stay close enough that shared gradients remain relevant
3. **Separation of optimization and parameters**: Different optimizer hyperparameters create measurable performance differences without breaking gradient relevance
4. **No batch subset noise**: DDP synchronizes gradients, so validation differences reflect pure hyperparameter effects

## Development

### Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -e .[dev]
```

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
# Format code
black src/ tests/

# Check linting
ruff check src/ tests/
```

## Research Status

This is an active research project. See [proposal.md](proposal.md) for detailed research questions and methodology.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@software{ddp_pbt,
  title = {DDP-PBT: Compute-Efficient Population Based Training},
  author = {O'Quinn, Christopher},
  year = {2026},
  url = {https://github.com/smithblack-0/DDP-PBT}
}
```
