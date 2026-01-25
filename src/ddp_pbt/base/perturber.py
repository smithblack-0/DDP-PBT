"""
Perturber: Hyperparameter mutation logic.

Handles perturbation of hyperparameter values using normal sampling in either
linear or log-space, with optional bounds clipping.
"""

import math
import random
from typing import Any, Callable, Dict, List

from . import utilities


class Perturber:
    """
    Perturbs hyperparameter values with normal sampling.

    Supports both linear and log-space perturbation with configurable bounds.
    Each element in a hyperparameter value list is perturbed independently.
    """

    def __init__(self):
        """Initialize Perturber with no schema (must call setup_schema)."""
        self._schema = None

    def setup_schema(
        self,
        schema: Dict[str, Dict[str, Any]],
    ) -> None:
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

    def apply_perturbation(
        self,
        path: str,
        value: float,
        default_std: float = None,
        default_param_type: str = None,
        default_min_bound: float = None,
        default_max_bound: float = None,
    ) -> float:
        """
        Applies a perturbation while respecting the necessary
        clip rules and configuration conditions

        :param path: The path for the hyperparameter. Used to look up schema
        :param value: The current value
        :param default_std: Optional fallback std if path not in schema
        :param default_param_type: Optional fallback param_type if path not in schema
        :param default_min_bound: Optional fallback min_bound
        :param default_max_bound: Optional fallback max_bound
        :return: The new value
        :raises: If insane config detected or invalid defaults provided
        """
        # Check if defaults provided together
        if (default_std is None) != (default_param_type is None):
            raise ValueError("default_std and default_param_type must both be provided or both be None")

        # Try schema lookup first
        if self._schema is not None and path in self._schema:
            config = self._schema[path]
            param_type = config["type"]
            std = config["std"]
            min_bound = config.get("min", None)
            max_bound = config.get("max", None)
        elif default_std is not None and default_param_type is not None:
            # Use defaults
            param_type = default_param_type
            std = default_std
            min_bound = default_min_bound
            max_bound = default_max_bound
        else:
            # No schema entry and no defaults - return unchanged
            return value

        # Get random noise
        noise = random.gauss(0, std)

        # Check config/inputs sane
        # More thorough logic should go in the actual
        # Schema builder in the relevant class.
        if param_type == "log":
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

    def perturb_pytree_using_schema(
        self,
        values: Dict[str, List[float]],
        perturb_chance: float = 1.0,
    ) -> Dict[str, List[float]]:
        """
        Perturb hyperparameter values with independent normal sampling.

        Args:
            values: Hyperparameter values dict. Format: {"param": [val1, val2, ...]}
                   Each list element is perturbed independently.
            perturb_chance: Probability of applying perturbation (0.0 = never, 1.0 = always)

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
                # Apply perturbation probabilistically
                if random.random() < perturb_chance:
                    perturbed_value = self.apply_perturbation(hyperparameter_path, hyperparameter)
                else:
                    perturbed_value = hyperparameter
                perturbed_values.append(perturbed_value)

            result[hyperparameter_path] = perturbed_values

        return result

    def perturb_dict_tree_using_defaults(
        self,
        dict_tree: Dict[str, Any],
        std: float,
        param_type: str,
        predicate: Callable[[str, Any], bool],
        perturb_chance: float = 1.0,
        min_bound: float = None,
        max_bound: float = None,
    ) -> Dict[str, Any]:
        """
        Perturb Dictionary Tree using provided defaults, filtering with predicate.

        Args:
            dict_tree: Dictionary Tree to perturb (flat dict with path keys)
            std: Standard deviation for perturbation
            param_type: "log" or "linear" perturbation type
            predicate: Function(path, value) -> bool to filter which paths to perturb
            perturb_chance: Probability of applying perturbation (0.0 = never, 1.0 = always)
            min_bound: Optional minimum bound
            max_bound: Optional maximum bound

        Returns:
            New Dictionary Tree with perturbed values, input unchanged

        Iterates Dictionary Tree, applies predicate to filter paths, perturbs filtered values
        using provided std/param_type (ignores schema).
        """
        result = {}
        for path, value in dict_tree.items():
            if predicate(path, value) and random.random() < perturb_chance:
                result[path] = self.apply_perturbation(
                    path,
                    value,
                    default_std=std,
                    default_param_type=param_type,
                    default_min_bound=min_bound,
                    default_max_bound=max_bound,
                )
            else:
                result[path] = value

        return result
