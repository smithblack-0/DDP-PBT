"""
Crossbreeder: Parent selection and hyperparameter blending with probabilistic mutation.

Handles selecting top parents and blending their hyperparameters with optional mutation.
"""

import random
import math
from typing import Dict, List, Any, Optional



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
        sorted_indexes = [i for i, _ in indexed_weights]

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

        top_pool = sorted_indexes[:self._parent_pool_depth]

        # Randomly select 2 parents from pool
        # Both have a 50% selection rate.
        selected_parents = random.sample(top_pool, 2)
        result_weights = [0.0] * len(world_weights)
        for parent_idx in selected_parents:
            result_weights[parent_idx] = 0.5
        return result_weights

    def crossbreed_alleles(self,
                          name: str,
                          a_list: List[float],
                          b_list: List[float]
                          )->List[float]:
        """
        Crossbreed a group of alleles of a common type.
        Different parameter groups may or may not be
        configured differently, necessitating list processing.
        Think of them like different parts of the body being configured
        slightly diferently.


        :param name: The name the hyperparameter is called by
        :param a_list: The first list of values
        :param b_list: The second list of values
        :return: The returned list of values
        """
        assert len(a_list) == len(b_list)
        schema_config = self._schema[name]
        allele_type = schema_config["type"]
        if allele_type == "log":
            def average_in_logspace(a: float, b: float)->float:
                a, b = math.log(a), math.log(b)
                output = (a + b)/2.0
                return math.exp(output)
            new_alleles = [average_in_logspace(a, b) for a, b in zip(a_list, b_list)]
            return new_alleles
        elif allele_type == "linear":
            new_allele = [(a + b)/2.0 for a, b in zip(a_list, b_list)]
            return new_allele
        else:
            raise ValueError(f"Unknown allele type {allele_type}")

    def mutate_alleles(self,
                       name: str,
                       allele_group: List[float],
                       )->List[float]:
        """
        Mutates a list of alleles. Each mutation has
        a mutation_chance% percentage of happing. The
        permuter still owns the mutation code.

        :param name: The name of the mutation to do
        :param allele_group: The group to mutate
        :return: A mutated group. Each had a mutation
            chance of actually occuring
        """
        output = []
        for allele in allele_group:
            if random.random() < self._mutation_rate:
                allele = self._permuter.apply_perturbation(name, allele)
            output.append(allele)
        return output

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
        parents = [i for i, w in enumerate(parent_weights) if w > 0]

        if len(parents) == 0:
            raise ValueError("No parents selected (all weights are zero)")
        if len(parents) != 2:
            raise ValueError(f"Expected exactly 2 parents, got {len(parents)}")

        mother_index, father_index = parents
        mother = world_hyperparameters[mother_index]
        father = world_hyperparameters[father_index]

        # The new genomes are 50% of each original
        # genome, with a possible mutation.
        result = {}
        for allele_key in mother.keys():
            # Note these are still divided per
            # hyperparameter group. Sort of like different
            # parts of the body having different epigenomes.
            mother_allele_group = mother[allele_key]
            father_allele_group = father[allele_key]

            if allele_key in self._schema:
                child_allele_group = self.crossbreed_alleles(allele_key, mother_allele_group, father_allele_group)
                child_allele_group = self.mutate_alleles(allele_key, child_allele_group)
            else:
                raise RuntimeError(f"Attempt to crossbreed unconfigured schema: {allele_key}")
            result[allele_key] = child_allele_group
        return result
