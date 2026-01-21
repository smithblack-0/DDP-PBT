"""
Crossbreeder: Parent selection and hyperparameter blending with probabilistic mutation.

Handles selecting top parents and blending their hyperparameters with optional mutation.
"""

import math
import random
from typing import Any, Dict, List, Optional

from .perturber import Perturber


class Crossbreeder:
    """
    Handles parent selection and hyperparameter crossbreeding with probabilistic mutation.

    Selects parents from top-K workers and blends their hyperparameters.
    Can optionally mutate the result based on mutation_rate. New
    hyperparameters are a 50/50
    """

    def __init__(
        self,
        parent_pool_depth: int,
        mutation_rate: float,
    ):
        """
        Initialize Crossbreeder with configuration.

        Args:
            parent_pool_depth: Number of top workers to draw parents from.
            mutation_rate: Probability of mutating crossbred result (0.0 to 1.0).
        """
        self._parent_pool_depth = parent_pool_depth
        self._mutation_rate = mutation_rate
        self._schema = None
        self._perturber = None

    def setup_schema(
        self,
        schema: Dict[str, Dict[str, Any]],
    ) -> None:
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
        # If perturber already set, propagate schema to it
        if self._perturber is not None:
            self._perturber.setup_schema(self._schema)

    def setup_perturber(
        self,
        perturber: Perturber,
    ) -> None:
        """
        Inject Perturber dependency for probabilistic mutation.

        Args:
            perturber: Perturber instance for mutating crossbred hyperparameters.
        """
        self._perturber = perturber
        # If schema already set, propagate to perturber immediately
        if self._schema is not None:
            self._perturber.setup_schema(self._schema)

    def crossbreed_alleles(
        self,
        name: str,
        a_list: List[float],
        b_list: List[float],
    ) -> List[float]:
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
        output = []
        for a, b in zip(a_list, b_list):
            output.append(random.choice([a, b]))
        return output

    def mutate_alleles(
        self,
        name: str,
        allele_group: List[float],
    ) -> List[float]:
        """
        Mutates a list of alleles. Each mutation has
        a mutation_chance% percentage of happing. The
        perturber still owns the mutation code.

        :param name: The name of the mutation to do
        :param allele_group: The group to mutate
        :return: A mutated group. Each had a mutation
            chance of actually occuring
        """
        output = []
        for allele in allele_group:
            if random.random() < self._mutation_rate:
                allele = self._perturber.apply_perturbation(name, allele)
            output.append(allele)
        return output

    def crossbreed_hyperparameters(
        self,
        world_hyperparameters: List[Dict[str, List[float]]],
        parent_weights: List[float],
    ) -> Dict[str, List[float]]:
        """
        Blend hyperparameters from parents with probabilistic mutation.

        Args:
            world_hyperparameters: List of hyperparameter values from all workers.
            parent_weights: World weights (most entries zero, 2 non-zero for parents).

        Returns:
            Blended hyperparameter values, possibly mutated.

        Blending behavior:
        - At each individual allele, choose one of the two parents with
          50% probability. This becomes the new allele
        - With mutation_rate probability: perturb result via perturber.
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
        # genome, with a possible mutation. Alleles
        # are chosen discretely, however.
        result = {}
        for allele_key in mother.keys():
            # Note these are still divided per
            # hyperparameter group. Sort of like different
            # parts of the body having different epigenomes.
            mother_allele_group = mother[allele_key]
            father_allele_group = father[allele_key]

            if allele_key in self._schema:
                child_allele_group = self.crossbreed_alleles(
                    allele_key, mother_allele_group, father_allele_group
                )
                child_allele_group = self.mutate_alleles(allele_key, child_allele_group)
            else:
                raise RuntimeError(f"Attempt to crossbreed unconfigured schema: {allele_key}")
            result[allele_key] = child_allele_group
        return result
