"""
Crossbreeder: Parent selection and hyperparameter blending with probabilistic mutation.

Handles selecting top parents and blending their hyperparameters with optional mutation.
"""

import random
import math
from typing import Dict, List, Any, Optional


def _blend_log_elements(elements: List[float], weights: List[float]) -> float:
    """
    Blend elements in log-space with given weights.

    Args:
        elements: List of values to blend.
        weights: List of weights (must sum to 1.0).

    Returns:
        Blended value.
    """
    log_elements = [math.log(e) for e in elements]
    blended_log = sum(w * le for w, le in zip(weights, log_elements))
    return math.exp(blended_log)


def _blend_linear_elements(elements: List[float], weights: List[float]) -> float:
    """
    Blend elements in linear-space with given weights.

    Args:
        elements: List of values to blend.
        weights: List of weights (must sum to 1.0).

    Returns:
        Blended value.
    """
    return sum(w * e for w, e in zip(weights, elements))


class Crossbreeder:
    """
    Handles parent selection and hyperparameter crossbreeding with probabilistic mutation.

    Selects parents from top-K workers and blends their hyperparameters.
    Can optionally mutate the result based on mutation_rate.
    """

    def __init__(self, parent_pool_depth: int, mutation_rate: float):
        """
        Initialize Crossbreeder with configuration.

        Args:
            parent_pool_depth: Number of top workers to draw parents from.
            mutation_rate: Probability of mutating crossbred result (0.0 to 1.0).
        """
        self._parent_pool_depth = parent_pool_depth
        self._mutation_rate = mutation_rate
        self._schema = None
        self._permuter = None

    def setup_schema(self, schema: Dict[str, Dict[str, Any]]) -> None:
        """
        Configure blending behavior from schema.

        Args:
            schema: Hyperparameter schema defining type for blending.
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

    def setup_permuter(self, permuter) -> None:
        """
        Inject Permuter dependency for probabilistic mutation.

        Args:
            permuter: Permuter instance for mutating crossbred hyperparameters.
        """
        self._permuter = permuter

    def select_parents(self, world_weights: List[float]) -> List[float]:
        """
        Select 2 parents from top parent_pool_depth workers.

        Args:
            world_weights: World weights (length = world_size, sum = 1.0).

        Returns:
            Filtered world weights with exactly 2 non-zero entries (sum = 1.0).
            Non-zero entries are for the 2 randomly selected parents from top-K.
        """
        # Rank workers by weight (descending)
        indexed_weights = [(i, w) for i, w in enumerate(world_weights)]
        indexed_weights.sort(key=lambda x: x[1], reverse=True)

        # Get top parent_pool_depth workers
        # Validate we have enough workers
        if len(world_weights) < 2:
            raise ValueError(
                f"Cannot crossbreed with less than 2 workers, got {len(world_weights)}"
            )
        if self._parent_pool_depth > len(world_weights):
            raise ValueError(
                f"parent_pool_depth ({self._parent_pool_depth}) cannot exceed world_size ({len(world_weights)})"
            )

        top_pool = indexed_weights[:self._parent_pool_depth]

        # Randomly select 2 parents from pool
        selected = random.sample(top_pool, 2)

        # Build filtered weights with only selected parents
        result_weights = [0.0] * len(world_weights)
        total_weight = sum(w for _, w in selected)

        for idx, weight in selected:
            # Normalize selected weights to sum to 1.0
            result_weights[idx] = weight / total_weight

        return result_weights

    def crossbreed_hyperparameters(
        self,
        world_hyperparameters: List[Dict[str, List[float]]],
        parent_weights: List[float]
    ) -> Dict[str, List[float]]:
        """
        Blend hyperparameters from parents with probabilistic mutation.

        Args:
            world_hyperparameters: List of hyperparameter values from all workers.
            parent_weights: World weights (most entries zero, 2 non-zero for parents).

        Returns:
            Blended hyperparameter values, possibly mutated.

        Blending behavior:
        - Log parameters: blend in log-space, convert back
        - Linear parameters: blend in linear-space
        - With mutation_rate probability: perturb result via permuter
        """
        if self._schema is None:
            raise RuntimeError("Must call setup_schema before crossbreed_hyperparameters")

        if not world_hyperparameters:
            return {}

        # Extract non-zero parent indices and weights
        parents = [(i, w) for i, w in enumerate(parent_weights) if w > 0]

        if len(parents) == 0:
            raise ValueError("No parents selected (all weights are zero)")
        if len(parents) != 2:
            raise ValueError(f"Expected exactly 2 parents, got {len(parents)}")

        # Blend each hyperparameter
        result = {}

        for param_name in self._schema:
            config = self._schema[param_name]
            param_type = config["type"]

            # Get parent hyperparameter values
            parent_values = [world_hyperparameters[i][param_name] for i, w in parents]
            parent_weights_list = [w for i, w in parents]

            # Determine list length from first parent
            num_elements = len(parent_values[0])

            # Blend element-wise for per-group parameters
            blended_values = []
            for elem_idx in range(num_elements):
                # Extract element from each parent
                elements = [pv[elem_idx] for pv in parent_values]

                # Blend based on type using helper functions
                if param_type == "log":
                    blended = _blend_log_elements(elements, parent_weights_list)
                elif param_type == "linear":
                    blended = _blend_linear_elements(elements, parent_weights_list)
                else:
                    raise ValueError(f"Unknown parameter type: {param_type}")

                blended_values.append(float(blended))

            result[param_name] = blended_values

        # Probabilistic mutation
        if random.random() < self._mutation_rate:
            if self._permuter is None:
                raise RuntimeError(
                    "Permuter not configured but mutation_rate > 0. Call setup_permuter()"
                )
            result = self._permuter.perturb(result)

        return result
