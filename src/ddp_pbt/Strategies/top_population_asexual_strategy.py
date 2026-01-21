"""
Population-style strategies track separate genomes between
different workers updating them in some way. This is the
asexual variant, which only lets the top k reproduce
and randomly mutates them.

It should be kept in mind there are some strong potential
downsides here. The gradients must remain mutually intelligent
between the workers or the worker is quickly pruned.
"""


from typing import List, Dict, Optional, Any, Type

import torch
import math
import random
import numpy as np
from torch.optim import Optimizer

from ..base.abstract_strategy import AbstractStrategy
from ..base.state import State
from ..base.communication import Communication
from ..base.perturber import Perturber


class TopPopulationAsexualStrategy(AbstractStrategy):
    """
    This strategy takes a bacterially-inspired asexual approach
    to reproduction and modification. Multiple genomes exist, and
    only the top-k of them are allowed to reproduce. No crossbreeding
    occurs. During reproduction the variant is randomly perturbed.

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
                 perturber: Perturber,
                 config: Optional[Dict[str, Any]] = None,
                 ):
        """
        :param num_k: Number of k. Cannot be larger than world size.
        :param state: The state object used for optimizer access
        :param communication: The communication objecct used for distributed work
        :param perturber: The perturbed used to mutate
        :param config: Possibly a valid config
        """
        super().__init__(state, communication, config)
        self._perturber = perturber
        self._perturber.setup_schema(self.schema)
        self._num_k = num_k

    def score(self,
              validation_metrics: List[float],
              communication: Communication,
              ) -> List[float]:
        """
        Perform an independent sorting and top-k
        selection per device. Different wworkers
        can choose different copies

        :param validation_metrics: The list of validation metrics
        :param communication: The communication mechanism
        :return: The top results
        """

        if self._num_k > len(validation_metrics):
            raise RuntimeError("Num k was greater than world size")

        # Sort the metrics and produce a top-k list
        # and make a decision.

        validation_metrics = np.array(validation_metrics)
        ascending_order = np.argsort(validation_metrics)
        top_k = ascending_order[:self._num_k]
        choice = int(np.random.choice(top_k))

        # Create and return the worker's choice. All workers make
        # independent choices
        world_weights = [0.0]*len(validation_metrics)
        world_weights[choice] = 1.0
        return world_weights

    def reduce_hyperparameters(
            self,
            world_weights: List[float],
            world_hyperparameters: List[Dict[str, List[float]]],
            communication: Communication
    ) -> Dict[str, List[float]]:
        """
        Select winner's hyperparameters and perturb.
        """
        # Find winner by index, select it, perturb the
        # hyperparameters
        winner_index = world_weights.index(1.0)
        winner_hyperparams = world_hyperparameters[winner_index]
        return self._perturber.perturb(winner_hyperparams)

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
        return communication.reduce_by_world_weights(world_weights, model_pytree)

    def reduce_optimizer(
            self,
            world_weights: List[float],
            optimizer_pytree: Dict[str, torch.Tensor],
            communication: Communication
    ) -> Dict[str, torch.Tensor]:
        """"
        Reduces the optimizers by their world weights
        """
        return communication.reduce_by_world_weights(world_weights, optimizer_pytree)

def make_top_population_asexual_strategy(
        reproduction_percentage: float,
        optimizer: Optimizer,
        max_hyperparameter_search_depth: int,
        communication_class: Type[Communication] = Communication,
        config: Optional[Dict[str, Any]] = None
        )->TopPopulationAsexualStrategy:
    """
    Creates a valid and functioning population asexual
    reproduction strategy that keeps separate genomes on
    different devices. This comes with more risk of
    gradients becoming incompatible and slowing down training,
    but also increases diversity.

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
    perturber = Perturber()
    state = State(optimizer, max_hyperparameter_search_depth)
    return TopPopulationAsexualStrategy(top_k, state, communicator, perturber, config)