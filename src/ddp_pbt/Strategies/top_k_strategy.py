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
    optimizer entries then promises to update them
    somehow. It is up to the rest of the training
    system
    A scheduling strategy which is stepped periodically
    with a validation statistics in a DDP context, and
    attempts to optimize the hyperparameter genome

    It should generally be used by either initiating
    a new strategy through the factory then binding
    to specific valid genome elements in the
    hyperparameters. You can check what
    the valid binding targets are using

    - valid_binding_targets: Returns a list of strings. Each string is a valid target

    Keep in mind only float features under max_depth inside hyperpameter groups
    can be detected this way. You can bind to them using:

    - bind_log_hyperparameter: Needs name, std for gaussian sample, min>0, and optionally max.
    - bind_linear_hyperparameter

    Alternatively, if you want to bind directly, you can pass in a config
    instead in the native schema language.

    """
