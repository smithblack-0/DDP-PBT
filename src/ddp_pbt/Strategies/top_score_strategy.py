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

    def __init__(
        self,
        state: State,
        communication: Communication,
        config: Optional[Dict[str, Dict[str, Any]]] = None,
        perturber: Perturber = None
    ):
        """
        Initialize TopScoreStrategy.

        Args:
            state: State instance for extracting/injecting optimizer data.
            communication: Communication instance for distributed operations.
            config: Optional native JSON schema config dict.
            perturber: Perturber instance for hyperparameter perturbation.

        Raises:
            ValueError: If perturber is None.
        """
        if perturber is None:
            raise ValueError("TopScoreStrategy requires a Perturber instance")

        super().__init__(state, communication, config)
        self._perturber = perturber

        # Setup perturber schema if schema configured
        if self._schema:
            self._perturber.setup_schema(self._schema)

    def score(self, validation_metrics: List[float], communication: Communication) -> List[float]:
        """
        Find best worker and return world weights.

        Args:
            validation_metrics: Validation metrics from all workers (lower is better).
            communication: Communication instance (unused).

        Returns:
            World weights with 1.0 at best worker index, 0.0 elsewhere.
        """
        # Find argmin (best worker with lowest loss)
        best_index = validation_metrics.index(min(validation_metrics))

        # Build world weights: 1.0 at best, 0.0 elsewhere
        world_weights = [0.0] * len(validation_metrics)
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

        # Perturb for next round variation
        return self._perturber.perturb(winner_hyperparams)

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


def make_top_score_strategy(
    optimizer: torch.optim.Optimizer,
    max_hyperparameter_search_depth: int = 3,
    config: Optional[Dict[str, Dict[str, Any]]] = None,
    communication_class: type = Communication
) -> TopScoreStrategy:
    """
    Factory function to create fully-wired TopScoreStrategy.

    Creates a TopScoreStrategy with all dependencies properly wired.
    The strategy selects the best-performing worker each round and
    perturbs their hyperparameters for variation.

    Use builder pattern methods to configure bindings:
    - valid_binding_targets: Returns list of bindable hyperparameter paths
    - bind_log_hyperparameter: Bind log-space parameter with std, min, optional max
    - bind_linear_hyperparameter: Bind linear-space parameter with std, optional min/max

    Args:
        optimizer: PyTorch optimizer to manage.
        max_hyperparameter_search_depth: How many layers deep to search param_groups
            for bindable hyperparameters. Default 3.
        config: Optional native JSON schema config dict for pre-configured bindings.
        communication_class: Communication class to instantiate for distributed ops.
            Default Communication.

    Returns:
        Ready-to-use TopScoreStrategy instance with all dependencies wired.
    """
    communicator = communication_class()
    perturber = Perturber()
    state = State(optimizer, max_hyperparameter_search_depth)

    strategy = TopScoreStrategy(state, communicator, config, perturber)

    return strategy
