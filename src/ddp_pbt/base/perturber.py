"""
Permuter: Hyperparameter mutation logic.

Handles perturbation of hyperparameter values using normal sampling in either
linear or log-space, with optional bounds clipping.
"""

import random
import math
from typing import Dict, List, Any

class Perturber:
    """
    Perturbs hyperparameter values with normal sampling.

    Supports both linear and log-space perturbation with configurable bounds.
    Each element in a hyperparameter value list is perturbed independently.
    """

    def __init__(self):
        """Initialize Permuter with no schema (must call setup_schema)."""
        self._schema = None

    def setup_schema(self, schema: Dict[str, Dict[str, Any]]) -> None:
        """
        Configure perturbation behavior from schema.

        Args:
            schema: Hyperparameter schema defining type, std, and optional bounds.
                   Format: {
                       "param_name": {
                           "type": "log" | "linear",
                           "std": float,
                           "min": float (optional),
                           "max": float (optional),
                           "shared": bool
                       }
                   }
        """
        self._schema = schema

    def apply_perturbation(self,
                           path: str,
                           value: float
                           )->float:
        """
        Applies a perturbation while respecting the necessary
        clip rules and configuration conditions

        :param path: The path for the hyperparameter. Used to look up schema
        :param value: The current value
        :return: The new value
        :raises: If insane config detected.
        """
        # Unpack config
        config = self._schema[path]
        param_type = config["type"]
        std = config["std"]
        min_bound = config.get("min", None)
        max_bound = config.get("max", None)

        # Get random noise
        noise = random.gauss(0, std)

        # Check config/inputs sane
        # More thorough logic should go in the actual
        # Schema builder in the relevant class.
        if  param_type == "log":
            if min_bound is None:
                raise RuntimeError("Cannot perturb log type with no minimum")
            if min_bound <= 0:
                raise RuntimeError("Minimum value of zero or less is insane in log configuration")
            if value <= 0:
                raise RuntimeError("Value of zero or less is insane in log configuration")

        # Handle linear perturbation
        if param_type == "linear":
            value = value + noise
            if min_bound is not None:
                value = max(min_bound, value)
            if max_bound is not None:
                value = min(max_bound, value)
        elif param_type == "log":
            value = math.log(value)
            value = value + noise
            value = max(math.log(min_bound), value)
            if max_bound is not None:
                value = min(math.log(max_bound), value)
            value = math.exp(value)
        else:
            raise RuntimeError("Unknown perturbation type {}".format(param_type))

        return value

    def perturb(self, values: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        Perturb hyperparameter values with independent normal sampling.

        Args:
            values: Hyperparameter values dict. Format: {"param": [val1, val2, ...]}
                   Each list element is perturbed independently.

        Returns:
            Perturbed hyperparameter values in same format as input.

        Each element in each list receives an independent random draw from normal(0, std).
        For log parameters: converts to log-space, adds noise, converts back.
        For linear parameters: adds noise directly.
        Clips to bounds (min/max) after perturbation if specified in schema.
        """
        if self._schema is None:
            raise RuntimeError("Must call setup_schema before perturb")

        if not values:
            return {}

        result = {}

        for hyperparameter_path, hyperparameters_list in values.items():
            if hyperparameter_path not in self._schema:
                msg = f"Path {hyperparameter_path} wanted but not in schema"
                raise RuntimeError(msg)

            perturbed_values = []
            for hyperparameter in hyperparameters_list:
                perturbed_value = self.apply_perturbation(hyperparameter_path, hyperparameter)
                perturbed_values.append(perturbed_value)

            result[hyperparameter_path] = perturbed_values

        return result
