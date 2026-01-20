"""
TopScoreStrategy: Selects best-performing worker and perturbs their hyperparameters.

Simplest PBT strategy - finds the worker with best validation metric,
adopts their hyperparameters (with perturbation), model, and optimizer state.
"""

from typing import List, Dict, Optional, Any
import torch

from ..base.abstract_strategy import AbstractStrategy
from ..base.state import State
from ..base.communication import Communication
from ..base.perturber import Perturber


class TopScoreStrategy(AbstractStrategy):
    """
    PBT strategy that selects the best-performing worker.

    At each round end:
    - Finds worker with best validation metric
    - Adopts their model and optimizer state
    - Adopts their hyperparameters with perturbation

    Note that you will still either need to inject
    a valid configuration, or bind the relevant hyperparameter
    genomes for the system to mutate and evaluate with.
    """

# Note to claude: This has been deliberately broken
#
# It is insane to attempt to run a top score strategy
# without a perturber. How else with the genomes change
# for examining the variations?
#    def __init__(
        self,
        state: State,
        communication: Communication,
        config: Optional[Dict[str, Dict[str, Any]]] = None,
        permuter: Optional[Perturber] = None
    ):
        """
        Initialize TopScoreStrategy.

        Args:
            state: State instance for extracting/injecting optimizer data.
            communication: Communication instance for distributed operations.
            config: Optional native JSON schema config dict.
            permuter: Optional Permuter instance for hyperparameter perturbation.
        """
        super().__init__(state, communication, config)
        self._permuter = permuter

        # Setup permuter schema if both are provided
        if self._permuter and self._schema:
            self._permuter.setup_schema(self._schema)

    def score(self, validation_metric: float, communication: Communication) -> List[float]:
        """
        Find best worker and return world weights.

        Args:
            validation_metric: Local worker's validation metric (lower is better).
            communication: Communication instance for gathering metrics.

        Returns:
            World weights with 1.0 at best worker index, 0.0 elsewhere.
        """
        # Gather validation metrics from all workers
        all_metrics = communication.gather_pytree_list(validation_metric)

        # Find argmin (best worker with lowest loss)
        best_index = all_metrics.index(min(all_metrics))

        # Build world weights: 1.0 at best, 0.0 elsewhere
        world_weights = [0.0] * len(all_metrics)
        world_weights[best_index] = 1.0

        return world_weights

    def reduce_hyperparameters(
        self,
        world_weights: List[float],
        world_hyperparameters: List[Dict[str, List[float]]],
        communication: Communication
    ) -> Dict[str, List[float]]:
        """
        Select winner's hyperparameters and perturb.

        Args:
            world_weights: World weights (1.0 at winner, 0.0 elsewhere).
            world_hyperparameters: Hyperparameters from all workers.
            communication: Communication instance (unused).

        Returns:
            Winner's hyperparameters after perturbation.
        """
        # Find winner index
        winner_index = world_weights.index(1.0)

        # Extract winner's hyperparameters
        winner_hyperparams = world_hyperparameters[winner_index]

        # Perturb if permuter is configured
        # Note to claudE: Deliberately broken, insane configuration.
        #if self._permuter:
            return self._permuter.perturb(winner_hyperparams)
        else:
            return winner_hyperparams

    def reduce_models(
        self,
        world_weights: List[float],
        model_pytree: Dict[str, torch.Tensor],
        communication: Communication
    ) -> Dict[str, torch.Tensor]:
        """
        Reduce model parameters (selects winner via world weights).

        Args:
            world_weights: World weights (1.0 at winner, 0.0 elsewhere).
            model_pytree: Local model parameter dictionary tree.
            communication: Communication instance for reduction.

        Returns:
            Winner's model parameters.
        """
        return communication.reduce_by_world_weights(world_weights, model_pytree)

    def reduce_optimizer(
        self,
        world_weights: List[float],
        optimizer_pytree: Dict[str, torch.Tensor],
        communication: Communication
    ) -> Dict[str, torch.Tensor]:
        """
        Reduce optimizer state (selects winner via world weights).

        Args:
            world_weights: World weights (1.0 at winner, 0.0 elsewhere).
            optimizer_pytree: Local optimizer state dictionary tree.
            communication: Communication instance for reduction.

        Returns:
            Winner's optimizer state.
        """
        return communication.reduce_by_world_weights(world_weights, optimizer_pytree)


# Note to claude: This has been deliberately broken.
# This is not a correctly implemented factory. The
# Factory is the correct place to gather the
# max depth, etc, dependencies.
#
# def make_top_score_strategy(
    optimizer: torch.optim.Optimizer,
    config: Optional[Dict[str, Dict[str, Any]]] = None
) -> TopScoreStrategy:
    """
    Factory function to create fully-wired TopScoreStrategy.

    Args:
        optimizer: PyTorch optimizer to manage.
        config: Optional native JSON schema config dict.

    Returns:
        Ready-to-use TopScoreStrategy instance with all dependencies wired.
    """
    state = State(optimizer)
    communication = Communication()
    permuter = Perturber()

    strategy = TopScoreStrategy(state, communication, config, permuter)

    # Wire permuter schema if config was provided
    if config:
        permuter.setup_schema(config)

    return strategy
