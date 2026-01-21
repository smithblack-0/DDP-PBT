"""
TopKStrategy: Selects random genome among top k or top percentage performers
 then perturbs their hyperparameters.

Straightforward strategy
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

class TopKStrategy(AbstractStrategy):
    """
    A strategy binds to a collection of valid float
    optimizer entries. This strategy promises to
    globally choose a topk entry to mutate and
    try for the next round.

    Keep in mind this will not implemnet the response logic itself, only
    promise to mutate the bound values on change. Consider using
    ScheduleAnything when extending the optimizer for, for instance,
    adaptive gradient clipping per parameter group. Use the following
    Use the following fields to check what can be bound to and bind to it.
    Anything not bound is not responded to.

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
        self._num_k = num_k
        self._perturber = perturber
        self._perturber.setup_schema(self.schema)

    def score(self,
              validation_metrics: List[float],
              communication: Communication,
              ) -> List[float]:
        """
        Score each option, returning a list of world size indicating
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

        # Submit our proposal; first rank wins

        choices = communication.gather_pytree_list(choice)
        choice = choices[0]

        # Choose the top one, ensure it is a float.
        # then create and return the world weights
        # weighting that entry to maximum
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

def make_top_k_strategy_out_of_top_k(
        top_k: int,
        optimizer: Optimizer,
        max_hyperparameter_search_depth: int = 3,
        config: Optional[Dict[str, Any]] = None,
        communication_class: Type[Communication] = Communication,
    )->TopKStrategy:
    """
    Creates a valid and functional top-k strategy that
    can bind various optimizer hyperparameters or, if
    a valid config is passed, is already bound.

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

    :param top_k: The number of top k elements to check
    :param optimizer: The optimizer to use for setup.
    :param max_hyperparameter_search_depth: How many layers deep to search each param group for
         hyperparameters that can be bound to. Default of 3
    :param config: A valid config if provided
    :param communication_class: A possible new communication class implemented with the
        prototype pattern, to handle quirks.
    :return: A valid TopKStrategy object.
    """
    communicator = communication_class()
    perturber = Perturber()
    state = State(optimizer, max_hyperparameter_search_depth)
    return TopKStrategy(top_k, state, communicator, perturber, config)


def make_top_k_strategy_by_selection_percentage(
        selection_percentage: float,
        optimizer: Optimizer,
        max_hyperparameter_search_depth: int,
        communication_class: Type[Communication] = Communication,
        config: Optional[Dict[str, Any]] = None
        )->TopKStrategy:
    """
    Creates a valid and functional top-k strategy that
    can bind various optimizer hyperparameters or, if
    a valid config is passed, is already bound.

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


    :param selection_percentage: A number between 0 and 1.0 indicating what percent
        of the world to randomly, and globally, select from.
    :param optimizer: The optimizer to use for setup.
    :param max_hyperparameter_search_depth: How many layers deep to search each param group for
         hyperparameters that can be bound to. Default of 3
    :param config: A valid config if provided
    :param communication_class: A possible new communication class implemented with the
        prototype pattern, to handle quirks.
    :return: A valid TopKStrategy object.
    """
    # Handle conversion
    assert selection_percentage >= 0 and selection_percentage <= 1.0
    communicator = communication_class()
    world_size = communicator.world_size
    top_k = int(world_size*selection_percentage)

    # Build and return
    perturber = Perturber()
    state = State(optimizer, max_hyperparameter_search_depth)
    return TopKStrategy(top_k, state, communicator, perturber, config)