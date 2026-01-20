"""
TopKStrategy: Selects random genome among top k or top percentage performers
 then perturbs their hyperparameters.

Straightforward strategy
"""

from typing import List, Dict, Optional, Any
import torch

from ..base.abstract_strategy import AbstractStrategy
from ..base.state import State
from ..base.communication import Communication
from ..base.perturber import Perturber

class TopKStrategy(AbstractStrategy):
    """
    A strategy binds to a collection of valid float
    optimizer entries. This strategy promieses to
    evaluate various gradient application mechanisms
    by producing different genomes on different devices,
    then randomly choosing one of the top devices
    and permuting it.

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
                 state: State,
                 communication: Communication,
                 perturber: Perturber,
                 config: Optional[Dict[str, Any]] = None
                 ):
        """
        :param state: The state object used for optimizer access
        :param communication: The communication objecct used for distributed work
        :param perturber: The perturbed used to mutate
        :param config: Possibly a valid config
        """
        super().__init__(state, communication, config)
        self._perturber = perturber

    def score(self, validation_metrics: float, communication) -> List[float]:
        """
        Score each option, returning a list of world size indicating
        :param validation_metric:
        :param communication:
        :return:
        """