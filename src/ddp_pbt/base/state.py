"""
State manages extraction and injection of tensors and hyperparameters from an optimizer.
"""

import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch

from .utilities import patch_pytree, walk_single_pytree


class State:
    """
    Extracts and injects tensors and hyperparameter values from optimizer.

    Responsibilities:
    - Extract model parameters as Dictionary Tree
    - Inject model parameters from Dictionary Tree
    - Extract optimizer state tensors as Dictionary Tree
    - Inject optimizer state tensors from Dictionary Tree
    - Extract hyperparameter values respecting shared vs per-group semantics
    - Inject hyperparameter values respecting shared vs per-group semantics
    - Provide list of valid hyperparameter paths for binding
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_depth: int = 3,
    ):
        """
        Initialize State with an optimizer reference.

        :param optimizer: PyTorch optimizer to manage state for
        :param max_depth: Maximum depth to walk param_groups for hyperparameters
        """
        self.optimizer = optimizer
        self.schema: Optional[Dict[str, Any]] = None
        self.max_depth = max_depth

    @property
    def valid_hyperparameter_paths(self) -> List[str]:
        """
        Return list of float-valued paths found in param_groups.

        These are the hyperparameters available for binding.

        :return: List of paths to float hyperparameters
        """
        paths_set = set()
        for param_group in self.optimizer.param_groups:
            for path, value in walk_single_pytree(param_group, max_depth=self.max_depth):
                if isinstance(value, float):
                    paths_set.add(path)
        return sorted(paths_set)

    def setup_schema(
        self,
        schema: Dict[str, Any],
    ) -> None:
        """
        Configure which hyperparameters to manage and their behavior.

        :param schema: Schema dict defining hyperparameter configuration
        """
        self.schema = schema

    def get_model_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Extract model parameters as Dictionary Tree.

        Walks optimizer.param_groups and filters to paths containing 'params'.

        :return: Dictionary Tree mapping paths to model parameter tensors
        """
        pytree = self.optimizer.param_groups
        dict_tree = {}

        for path, value in walk_single_pytree(pytree, max_depth=-1):
            if "params" in path and isinstance(value, torch.Tensor):
                dict_tree[path] = value

        return dict_tree

    def set_model_tensors(
        self,
        dict_tree: Dict[str, torch.Tensor],
    ) -> None:
        """
        Inject model parameters from Dictionary Tree in-place.

        Walks optimizer.param_groups and copies tensor data for matching paths.

        :param dict_tree: Dictionary Tree mapping paths to model parameter tensors
        """
        pytree = self.optimizer.param_groups

        for path, original_tensor in walk_single_pytree(pytree, max_depth=-1):
            if path in dict_tree:
                if isinstance(original_tensor, torch.Tensor):
                    original_tensor.data.copy_(dict_tree[path])

    def get_optimizer_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Extract optimizer state tensors as Dictionary Tree.

        Converts optimizer.state parameter keys to str(id(param)) for walking,
        then extracts all tensor values.

        :return: Dictionary Tree mapping paths to optimizer state tensors
        """
        converted_state = {
            str(id(param)): state_dict for param, state_dict in self.optimizer.state.items()
        }

        dict_tree = {}
        for path, value in walk_single_pytree(converted_state, max_depth=-1):
            if isinstance(value, torch.Tensor):
                dict_tree[path] = value

        return dict_tree

    def set_optimizer_tensors(
        self,
        dict_tree: Dict[str, torch.Tensor],
    ) -> None:
        """
        Inject optimizer state tensors from Dictionary Tree in-place.

        Walks converted optimizer.state and copies tensor data for matching paths.

        :param dict_tree: Dictionary Tree mapping paths to optimizer state tensors
        """
        converted_state = {
            str(id(param)): state_dict for param, state_dict in self.optimizer.state.items()
        }

        for path, original_tensor in walk_single_pytree(converted_state, max_depth=-1):
            if path in dict_tree:
                if isinstance(original_tensor, torch.Tensor):
                    original_tensor.copy_(dict_tree[path])

    def get_hyperparam_values(self) -> Dict[str, List[float]]:
        """
        Extract hyperparameter values respecting shared vs per-group semantics.

        For shared hyperparameters: returns length-1 list from first param group.
        For per-group hyperparameters: returns list of values from all param groups.

        :return: Hyperparameter Values dict
        """
        if self.schema is None:
            raise RuntimeError("Schema must be set up before calling get_hyperparam_values")

        output = defaultdict(list)
        for param_group in self.optimizer.param_groups:
            for path, value in walk_single_pytree(param_group, max_depth=self.max_depth):
                if isinstance(value, float) and path in self.schema:
                    output[path].append(value)

        output_dict = {}
        for path, value_list in output.items():
            if len(value_list) != len(self.optimizer.param_groups):
                msg = (
                    f"At pytree path '{path}' the number of valid param groups did not "
                    f"match gathered data, despite being in schema"
                )
                raise RuntimeError(msg)
            else:
                if self.schema[path]["shared"]:
                    value_list = value_list[:1]
                output_dict[path] = value_list

        return output_dict

    def set_hyperparam_values(
        self,
        values: Dict[str, List[float]],
    ) -> None:
        """
        Inject hyperparameter values respecting shared vs per-group semantics.

        For shared hyperparameters: broadcasts single value to all param groups.
        For per-group hyperparameters: sets each param group individually.

        :param values: Hyperparameter Values dict
        """
        if self.schema is None:
            raise RuntimeError("Schema must be set up before calling set_hyperparam_values")

        # Expand shared values to match param group count
        expanded_values = {}
        for path, value_list in values.items():
            if self.schema[path]["shared"]:
                expanded_values[path] = [value_list[0]] * len(self.optimizer.param_groups)
            else:
                expanded_values[path] = value_list

        # Collect patches per param group
        patches_per_group = [[] for _ in self.optimizer.param_groups]
        for path, value_list in expanded_values.items():
            for group_idx, value in enumerate(value_list):
                patches_per_group[group_idx].append((path, value))

        # Patch each param group using utility
        for group_idx, patches in enumerate(patches_per_group):
            patch_pytree(self.optimizer.param_groups[group_idx], patches)
