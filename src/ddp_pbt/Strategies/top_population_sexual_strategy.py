"""
Population-style strategies track separate genomes between
different workers updating them in some way. This is the
sexual variant, which only lets the top k reproduce
and crossbreeds then mutates with some mutation chance.

It should be kept in mind there are some strong potential
downsides here. The gradients must remain mutually intelligent
between the workers or the worker is quickly pruned.

This is for the sexual reproduction strategy, which causes
more averaging and more diversity, but also possibly more
degradation in the averaging process. Empirical
evidence will tell.
"""

from typing import List, Dict, Optional, Any, Type

import torch
import math
import random
import numpy as np
from torch.optim import Optimizer

from ..base import Crossbreeder
from ..base.abstract_strategy import AbstractStrategy
from ..base.state import State
from ..base.communication import Communication
from ..base.perturber import Perturber

class TopPopulationSexualStrategy(AbstractStrategy):
    """
    This strategy takes a biological approach allowing genomes
    , only the top of which are allowed to reproduce.
    Reproduction crossbreeds the hyperparameter genome with a
    50/50 chance of drawing from either model, then randomly
    draws the underlying model and optimizer state from one
    of the two parents.

    Use the following fields to check what can be bound to or bind
    optimizer fields. Keep in mind it is up to you or your optimizer
    to actually respond to the inserted schedule.

    - valid_binding_targets: Returns a list of strings. Each string is a valid target
    - bind_log_hyperparameter: Needs name, std for gaussian sample, min>0, and optionally max.
    - bind_linear_hyperparameter

    Alternatively, if you want to bind directly, you can pass in a config
    instead in the native schema language.
    """

    def __init__(self,
                 num_k: int,
                 state: State,
                 communication: Communication,
                 crossbreeder: Crossbreeder,
                 config: Optional[Dict[str, Any]] = None,
                 ):
        """
        :param num_k: Number of k. Cannot be larger than world size.
        :param state: The state object used for optimizer access
        :param communication: The communication objecct used for distributed work
        :param crossbreeder: The object performing primary crossbreeding and mutation
        :param config: Possibly a valid config
        """
        super().__init__(state, communication, config)
        self._crossbreeder = crossbreeder
        self._crossbreeder.setup_schema(self.schema)
        self._num_k = num_k
        self.root_parent = 0

    def score(self,
              validation_metrics: List[float],
              communication: Communication,
              ) -> List[float]:
        """
        Select 2 parents from top parent_pool_depth workers.

        Args:
            validation_loss: Validation loss list. Lower is better.

        Returns:
            Filtered world weights with exactly 2 non-zero entries (sum = 1.0).
            Non-zero entries are for the 2 randomly selected parents from top-K.
        """
        # Rank workers by weight (descending)
        indexed_weights = [(i, w) for i, w in enumerate(validation_metrics)]
        indexed_weights.sort(key=lambda x: x[1], reverse=False)
        sorted_indexes = [i for i, _ in indexed_weights]

        # Get top parent_pool_depth workers
        # Validate we have enough workers
        if len(validation_metrics) < 2:
            raise ValueError(
                f"Cannot crossbreed with less than 2 workers, got {len(validation_metrics)}"
            )
        if self._num_k > len(validation_metrics):
            raise ValueError(
                f"parent_pool_depth ({self._num_k}) cannot exceed world_size ({len(validation_metrics)})"
            )

        top_pool = sorted_indexes[:self._num_k]

        # Randomly select 2 parents from pool
        # Both have a 50% selection rate.
        selected_parents = random.sample(top_pool, 2)
        result_weights = [0.0] * len(validation_metrics)
        for parent_idx in selected_parents:
            result_weights[parent_idx] = 0.5

        # Store one of the parents as the 'root'
        # Then return
        self.root_parent = random.choice(selected_parents)

        return result_weights

    def reduce_hyperparameters(
            self,
            world_weights: List[float],
            world_hyperparameters: List[Dict[str, List[float]]],
            communication: Communication
    ) -> Dict[str, List[float]]:
        """
        Crossbreed the hyperparameters using the
        crossbreeder
        """
        return self._crossbreeder.crossbreed_hyperparameters(world_hyperparameters, world_weights)

    def reduce_models(
            self,
            world_weights: List[float],
            model_pytree: Dict[str, torch.Tensor],
            communication: Communication
    ) -> Dict[str, torch.Tensor]:
        """
        Reduces the models by their weights using
        the communicator.
        """
        selection_weights = [0.0]*len(world_weights)
        selection_weights[self.root_parent] = 1.0
        return communication.reduce_by_world_weights(selection_weights, model_pytree)

    def reduce_optimizer(
            self,
            world_weights: List[float],
            optimizer_pytree: Dict[str, torch.Tensor],
            communication: Communication
    ) -> Dict[str, torch.Tensor]:
        """"
        Reduces the optimizers by their world weights
        """
        selection_weights = [0.0]*len(world_weights)
        selection_weights[self.root_parent] = 1.0
        return communication.reduce_by_world_weights(selection_weights, optimizer_pytree)

def make_top_population_sexual_strategy(
        reproduction_percentage: float,
        optimizer: Optimizer,
        mutation_rate: float = 0.05,
        max_hyperparameter_search_depth: int = 3,
        communication_class: Type[Communication] = Communication,
        config: Optional[Dict[str, Any]] = None
        )->TopPopulationSexualStrategy:
    """
    Creates a valid and functioning population asexual
    reproduction strategy that keeps separate genomes on
    different devices. This version is sexual, choosing
    with 50% probability each allele from one of the two
    parents.

    A strategy binds to a collection of valid float
    optimizer entries. This strategy promises to
    globally choose a topk entry to mutate and
    try for the next round.

    Use the following to examine available bindings and
    add your own when using the default builder patterns,
    and consult the class for caviots.

    - valid_binding_targets: Returns a list of strings. Each string is a valid target
    - bind_log_hyperparameter: Needs name, std for gaussian sample, min>0, and optionally max.
    - bind_linear_hyperparameter

    Keep in mind this object needs to be fed validation
    metrics after finishing a round to function. Additionally,
    you can extend the optimizer after loading in into the
    object if desired. It is recommended to do that through
    ScheduleAnything (torch-schedule-anything).


    :param reproduction_percentage: A number between 0 to 1.0 indicating the top performing
        percentage to allow reproduction on.
    :param optimizer: The optimizer to use for setup.
    :param mutation_rate: A number from 0 to 1.0 indicating how frequently to mutate. Default is 0.05
        or 5%
    :param max_hyperparameter_search_depth: How many layers deep to search each param group for
         hyperparameters that can be bound to. Default of 3
    :param config: A valid config if provided
    :param communication_class: A possible new communication class implemented with the
        prototype pattern, to handle quirks.
    :return: A valid TopKStrategy object.
    """
    # Handle conversion
    assert reproduction_percentage >= 0.0 and reproduction_percentage <= 1.0
    communicator = communication_class()
    world_size = communicator.world_size
    top_k = int(world_size*reproduction_percentage)

    # Build and return
    crossbreeder = Crossbreeder(top_k, mutation_rate)
    crossbreeder.setup_perturber(Perturber())
    state = State(optimizer, max_hyperparameter_search_depth)
    return TopPopulationAsexualStrategy(top_k, state, communicator, crossbreeder, config)