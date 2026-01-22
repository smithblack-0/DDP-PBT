import math
import random
from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
from torch.optim import Optimizer

from ..base.abstract_strategy import AbstractStrategy
from ..base.communication import Communication
from ..base.perturber import Perturber
from ..base.state import State

class WeightedAverageStrategy(AbstractStrategy):
    """
    A strategy binds to a collection of valid float
    optimizer entries. This strategy promises to handle
    the round by averaging together the hyperparameters
    and models in proportion to how well they do. As such,
    it should use short rounds between steps.

    Keep in mind this will not implement the response logic itself, only
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

    def __init__(
        self,
        state: State,
        communication: Communication,
        perturber: Perturber,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        :param state: The state object used for optimizer access
        :param communication: The communication objecct used for distributed work
        :param perturber: The perturbed used to mutate
        :param config: Possibly a valid config
        """
        super().__init__(state, communication, config)
        self._perturber = perturber
        self._perturber.setup_schema(self.schema)

    def score(
        self,
        validation_metrics: List[float],
        communication: Communication,
    ) -> List[float]:
        """
        Score each option. Since each round has a perturbation
        on the same hyperparameters from last round any effects
        are very subtle.

        :param validation_metrics: The list of validation metrics
        :param communication: The communication mechanism
        :return: The top results
        """

        # Normalize as a probability. Subtract by the
        # maximum score then take the absolute value
        # and normalize to get a sense of how much better
        # the lower loss cases actually are.

        # Basic intuition is anything below the maximum value
        # had a better loss, so we quantify with respect
        # to that.

        maximum = max(validation_metrics)
        validation_metrics = np.array(validation_metrics)
        validation_metrics = validation_metrics - maximum
        validation_metrics = np.abs(validation_metrics)
        norm = validation_metrics.sum()
        if norm > 0:
            distribution = validation_metrics / norm
        else:
            distribution = np.ones_like(validation_metrics) / validation_metrics.size

        # Return the distribution as the world weights
        return distribution.tolist()

    def reduce_hyperparameters(
        self,
        world_weights: List[float],
        world_hyperparameters: List[Dict[str, List[float]]],
        communication: Communication,
    ) -> Dict[str, List[float]]:
        """
        Average the hyperparameters together between every
        world according to their weight, then perturb
        the result for the next round
        """
        # We use full numpy tactics here with matrix multiplication
        # to speed this problem along. Merging is done vectorized
        weights = np.asarray(world_weights)
        result = {}
        for key in world_hyperparameters[0].keys():
            key_entries = [
                hyperparameter_dict[key] for hyperparameter_dict in world_hyperparameters
            ]
            key_entries = np.asarray(key_entries)
            new_subgenome = weights @ key_entries
            result[key] = new_subgenome.tolist()

        # Perturb and return
        result = self._perturber.perturb(result)
        return result

    def reduce_models(
        self,
        world_weights: List[float],
        model_pytree: Dict[str, torch.Tensor],
        communication: Communication,
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
        communication: Communication,
    ) -> Dict[str, torch.Tensor]:
        """ "
        Reduces the optimizers by their world weights
        """
        return communication.reduce_by_world_weights(world_weights, optimizer_pytree)


def make_weighted_average_strategy(
    optimizer: Optimizer,
    max_hyperparameter_search_depth: int = 3,
    config: Optional[Dict[str, Any]] = None,
    communication_class: Type[Communication] = Communication,
) -> WeightedAverageStrategy:
    """
    Creates a valid and functioning weighted average
    strategy capable of averaging the runs together
    by performance. It is recommended to use short
    rounds with this kind of strategy to avoid excessive
    losses when averaging models.

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

    :param optimizer: The optimizer to use for setup.
    :param max_hyperparameter_search_depth: How many layers deep to search each param group for
         hyperparameters that can be bound to. Default of 3
    :param config: A valid config if provided
    :param communication_class: A possible new communication class implemented with the
        prototype pattern, to handle quirks.
    :return: A valid WeightedAverageStrategy object.
    """
    communicator = communication_class()
    perturber = Perturber()
    state = State(optimizer, max_hyperparameter_search_depth)
    return WeightedAverageStrategy(state, communicator, perturber, config)
