"""
Permuter: Hyperparameter mutation logic.

Handles perturbation of hyperparameter values using normal sampling in either
linear or log-space, with optional bounds clipping.
"""

import random
import math
from typing import Dict, List, Any


class Permuter:
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

        for param_name, param_values in values.items():
            if param_name not in self._schema:
                continue

            config = self._schema[param_name]
            param_type = config["type"]
            std = config["std"]
            min_bound = config.get("min", None)
            max_bound = config.get("max", None)

            perturbed_values = []

            for value in param_values:
                # Generate independent random sample using standard library
                noise = random.gauss(0, std)

                # Apply perturbation based on type
                if param_type == "log":
                    # Log-space perturbation
                    log_value = math.log(value)
                    perturbed_log = log_value + noise
                    perturbed = math.exp(perturbed_log)
                elif param_type == "linear":
                    # Linear-space perturbation
                    perturbed = value + noise
                else:
                    raise ValueError(f"Unknown parameter type: {param_type}")

                # Clip to bounds
                if min_bound is not None:
                    perturbed = max(perturbed, min_bound)
                if max_bound is not None:
                    perturbed = min(perturbed, max_bound)

                perturbed_values.append(float(perturbed))

            result[param_name] = perturbed_values

        return result
