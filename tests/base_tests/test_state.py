"""
State contract test suite.

These tests validate that State:
- Extracts model parameters as Dictionary Trees
- Injects model parameters from Dictionary Trees in-place
- Extracts optimizer state tensors as Dictionary Trees
- Injects optimizer state tensors from Dictionary Trees in-place
- Extracts hyperparameter values respecting shared vs per-group semantics
- Injects hyperparameter values respecting shared vs per-group semantics
- Clips hyperparameter values to schema bounds
"""

from typing import Any, Dict

import pytest
import torch
import torch.nn as nn

from src.ddp_pbt.base.state import State


@pytest.fixture
def simple_model() -> nn.Module:
    """Returns a simple two-layer model."""
    return nn.Sequential(
        nn.Linear(10, 5),
        nn.Linear(5, 2),
    )


@pytest.fixture
def optimizer_single_group(simple_model) -> torch.optim.Optimizer:
    """Returns an optimizer with a single param group."""
    return torch.optim.SGD(simple_model.parameters(), lr=0.01, momentum=0.9)


@pytest.fixture
def optimizer_multi_group(simple_model) -> torch.optim.Optimizer:
    """Returns an optimizer with multiple param groups."""
    layers = list(simple_model.children())
    return torch.optim.SGD(
        [
            {"params": layers[0].parameters(), "lr": 0.01, "weight_decay": 0.1},
            {"params": layers[1].parameters(), "lr": 0.001, "weight_decay": 0.01},
        ],
        momentum=0.9,
    )


@pytest.fixture
def optimizer_with_state(optimizer_single_group, simple_model) -> torch.optim.Optimizer:
    """Returns an optimizer with populated state (momentum buffers)."""
    # Run a training step to populate optimizer state
    optimizer = optimizer_single_group
    model = simple_model

    x = torch.randn(4, 10)
    y = torch.randn(4, 2)

    output = model(x)
    loss = ((output - y) ** 2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return optimizer


@pytest.fixture
def basic_schema() -> Dict[str, Any]:
    """Returns a basic schema with shared and per-group hyperparameters."""
    return {
        "lr": {
            "type": "log",
            "std": 0.1,
            "min": 1e-4,
            "max": 1e-1,
            "shared": False,
        },
        "weight_decay": {
            "type": "linear",
            "std": 0.001,
            "min": 0.0,
            "max": 0.1,
            "shared": False,
        },
    }


@pytest.fixture
def shared_schema() -> Dict[str, Any]:
    """Returns a schema with a shared hyperparameter."""
    return {
        "lr": {
            "type": "log",
            "std": 0.1,
            "min": 1e-4,
            "max": 1e-1,
            "shared": True,
        },
    }


class TestStateConstruction:
    """Tests that State initializes correctly."""

    def test_state_initializes_with_optimizer_reference(self, optimizer_single_group) -> None:
        """State initializes with optimizer reference and no schema."""
        state = State(optimizer_single_group)

        assert state.optimizer is optimizer_single_group
        assert state.schema is None

    def test_setup_schema_stores_schema_reference(
        self, optimizer_single_group, basic_schema
    ) -> None:
        """State setup schema stores the schema reference."""
        state = State(optimizer_single_group)
        state.setup_schema(basic_schema)

        assert state.schema is basic_schema


class TestGetModelTensors:
    """Tests that get_model_tensors extracts model parameters correctly."""

    def test_get_model_tensors_returns_dictionary_tree_with_params_paths(
        self,
        optimizer_single_group,
    ) -> None:
        """Get model tensors returns Dictionary Tree with paths containing params."""
        state = State(optimizer_single_group)
        dict_tree = state.get_model_tensors()

        # Should have paths like 0/params/0, 0/params/1, etc.
        assert len(dict_tree) > 0
        for path in dict_tree.keys():
            assert "params" in path
            assert isinstance(dict_tree[path], torch.Tensor)

    def test_get_model_tensors_includes_all_parameter_tensors(
        self,
        optimizer_single_group,
    ) -> None:
        """Get model tensors includes all parameter tensors from optimizer."""
        state = State(optimizer_single_group)
        dict_tree = state.get_model_tensors()

        # Count tensors in optimizer param_groups
        total_params = sum(len(pg["params"]) for pg in optimizer_single_group.param_groups)

        assert len(dict_tree) == total_params

    def test_get_model_tensors_works_with_multiple_param_groups(
        self,
        optimizer_multi_group,
    ) -> None:
        """Get model tensors works correctly with multiple param groups."""
        state = State(optimizer_multi_group)
        dict_tree = state.get_model_tensors()

        # Should have paths from both groups: 0/params/... and 1/params/...
        paths_group_0 = [p for p in dict_tree.keys() if p.startswith("0/params/")]
        paths_group_1 = [p for p in dict_tree.keys() if p.startswith("1/params/")]

        assert len(paths_group_0) > 0
        assert len(paths_group_1) > 0


class TestSetModelTensors:
    """Tests that set_model_tensors updates model parameters correctly."""

    def test_set_model_tensors_updates_tensors_in_place(
        self,
        optimizer_single_group,
    ) -> None:
        """Set model tensors updates parameter tensors in-place."""
        state = State(optimizer_single_group)

        # Get original tensors
        original_dict_tree = state.get_model_tensors()
        original_values = {path: tensor.clone() for path, tensor in original_dict_tree.items()}

        # Create modified dict_tree
        modified_dict_tree = {
            path: (tensor + 1.0).detach() for path, tensor in original_dict_tree.items()
        }

        # Set the modified tensors
        state.set_model_tensors(modified_dict_tree)

        # Verify tensors were updated
        updated_dict_tree = state.get_model_tensors()
        for path in original_dict_tree.keys():
            assert torch.allclose(
                updated_dict_tree[path],
                original_values[path] + 1.0,
            )

    def test_set_model_tensors_only_updates_specified_paths(
        self,
        optimizer_multi_group,
    ) -> None:
        """Set model tensors only updates tensors for paths in dict_tree."""
        state = State(optimizer_multi_group)

        # Get all tensors
        all_tensors = state.get_model_tensors()

        # Only modify first tensor
        first_path = list(all_tensors.keys())[0]
        partial_dict_tree = {first_path: (all_tensors[first_path] + 10.0).detach()}

        # Store original values
        original_values = {path: tensor.clone() for path, tensor in all_tensors.items()}

        # Set only the first tensor
        state.set_model_tensors(partial_dict_tree)

        # Verify only first tensor changed
        updated_tensors = state.get_model_tensors()
        assert torch.allclose(updated_tensors[first_path], original_values[first_path] + 10.0)

        for path in list(all_tensors.keys())[1:]:
            assert torch.allclose(updated_tensors[path], original_values[path])


class TestGetOptimizerTensors:
    """Tests that get_optimizer_tensors extracts optimizer state correctly."""

    def test_get_optimizer_tensors_returns_empty_dict_when_state_is_empty(
        self,
        optimizer_single_group,
    ) -> None:
        """Get optimizer tensors returns empty dict when optimizer state is empty."""
        state = State(optimizer_single_group)
        dict_tree = state.get_optimizer_tensors()

        assert dict_tree == {}

    def test_get_optimizer_tensors_returns_dictionary_tree_with_state_tensors(
        self,
        optimizer_with_state,
    ) -> None:
        """Get optimizer tensors returns Dictionary Tree with optimizer state tensors."""
        state = State(optimizer_with_state)
        dict_tree = state.get_optimizer_tensors()

        # Should have state tensors (momentum buffers)
        assert len(dict_tree) > 0
        for path, tensor in dict_tree.items():
            assert isinstance(tensor, torch.Tensor)
            # Path should include param ID
            assert "/" in path

    def test_get_optimizer_tensors_filters_out_non_tensor_values(
        self,
        optimizer_with_state,
    ) -> None:
        """Get optimizer tensors filters out non-tensor values like step counters."""
        # Add a non-tensor value to optimizer state
        first_param = optimizer_with_state.param_groups[0]["params"][0]
        optimizer_with_state.state[first_param]["step"] = 5

        state = State(optimizer_with_state)
        dict_tree = state.get_optimizer_tensors()

        # Verify all values are tensors
        for value in dict_tree.values():
            assert isinstance(value, torch.Tensor)


class TestSetOptimizerTensors:
    """Tests that set_optimizer_tensors updates optimizer state correctly."""

    def test_set_optimizer_tensors_updates_tensors_in_place(
        self,
        optimizer_with_state,
    ) -> None:
        """Set optimizer tensors updates state tensors in-place."""
        state = State(optimizer_with_state)

        # Get original state tensors
        original_dict_tree = state.get_optimizer_tensors()
        original_values = {path: tensor.clone() for path, tensor in original_dict_tree.items()}

        # Create modified dict_tree
        modified_dict_tree = {
            path: (tensor + 1.0).detach() for path, tensor in original_dict_tree.items()
        }

        # Set the modified tensors
        state.set_optimizer_tensors(modified_dict_tree)

        # Verify tensors were updated
        updated_dict_tree = state.get_optimizer_tensors()
        for path in original_dict_tree.keys():
            assert torch.allclose(
                updated_dict_tree[path],
                original_values[path] + 1.0,
            )

    def test_set_optimizer_tensors_only_updates_specified_paths(
        self,
        optimizer_with_state,
    ) -> None:
        """Set optimizer tensors only updates tensors for paths in dict_tree."""
        state = State(optimizer_with_state)

        # Get all state tensors
        all_tensors = state.get_optimizer_tensors()

        if len(all_tensors) < 2:
            pytest.skip("Need at least 2 state tensors for this test")

        # Only modify first tensor
        first_path = list(all_tensors.keys())[0]
        partial_dict_tree = {first_path: (all_tensors[first_path] + 10.0).detach()}

        # Store original values
        original_values = {path: tensor.clone() for path, tensor in all_tensors.items()}

        # Set only the first tensor
        state.set_optimizer_tensors(partial_dict_tree)

        # Verify only first tensor changed
        updated_tensors = state.get_optimizer_tensors()
        assert torch.allclose(updated_tensors[first_path], original_values[first_path] + 10.0)

        for path in list(all_tensors.keys())[1:]:
            assert torch.allclose(updated_tensors[path], original_values[path])


class TestGetHyperparamValues:
    """Tests that get_hyperparam_values extracts hyperparameters correctly."""

    def test_get_hyperparam_values_raises_when_schema_not_set(
        self,
        optimizer_single_group,
    ) -> None:
        """Get hyperparam values raises RuntimeError when schema not set up."""
        state = State(optimizer_single_group)

        with pytest.raises(RuntimeError) as excinfo:
            state.get_hyperparam_values()

        assert "Schema must be set up" in str(excinfo.value)

    def test_get_hyperparam_values_returns_length_one_list_for_shared_hyperparameters(
        self,
        optimizer_multi_group,
        shared_schema,
    ) -> None:
        """Get hyperparam values returns length-1 list for shared hyperparameters."""
        state = State(optimizer_multi_group)
        state.setup_schema(shared_schema)

        values = state.get_hyperparam_values()

        assert "lr" in values
        assert len(values["lr"]) == 1
        assert values["lr"][0] == optimizer_multi_group.param_groups[0]["lr"]

    def test_get_hyperparam_values_returns_per_group_list_for_non_shared_hyperparameters(
        self,
        optimizer_multi_group,
        basic_schema,
    ) -> None:
        """Get hyperparam values returns per-group list for non-shared hyperparameters."""
        state = State(optimizer_multi_group)
        state.setup_schema(basic_schema)

        values = state.get_hyperparam_values()

        assert "lr" in values
        assert len(values["lr"]) == len(optimizer_multi_group.param_groups)
        assert values["lr"][0] == optimizer_multi_group.param_groups[0]["lr"]
        assert values["lr"][1] == optimizer_multi_group.param_groups[1]["lr"]

        assert "weight_decay" in values
        assert len(values["weight_decay"]) == len(optimizer_multi_group.param_groups)

    def test_clips_lr_above_max_bound(self, simple_model) -> None:
        """Get hyperparam values clips learning rate above max bound to max."""
        # Create optimizer with lr above max bound
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.5)

        schema = {
            "lr": {
                "type": "log",
                "std": 0.1,
                "min": 1e-4,
                "max": 0.1,
                "shared": True,
            }
        }

        state = State(optimizer)
        state.setup_schema(schema)

        values = state.get_hyperparam_values()

        # Verify returned value was clipped to max
        assert values["lr"][0] == 0.1

    def test_clips_lr_below_min_bound(self, simple_model) -> None:
        """Get hyperparam values clips learning rate below min bound to min."""
        # Create optimizer with lr below min bound
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=1e-7)

        schema = {
            "lr": {
                "type": "log",
                "std": 0.1,
                "min": 1e-4,
                "max": 0.1,
                "shared": True,
            }
        }

        state = State(optimizer)
        state.setup_schema(schema)

        values = state.get_hyperparam_values()

        # Verify returned value was clipped to min
        assert values["lr"][0] == 1e-4

    def test_lr_within_bounds_unchanged(self, simple_model) -> None:
        """Get hyperparam values leaves learning rate unchanged when within bounds."""
        # Create optimizer with lr within bounds
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)

        schema = {
            "lr": {
                "type": "log",
                "std": 0.1,
                "min": 1e-4,
                "max": 0.1,
                "shared": True,
            }
        }

        state = State(optimizer)
        state.setup_schema(schema)

        values = state.get_hyperparam_values()

        # Verify returned value unchanged
        assert values["lr"][0] == 0.01

    def test_clips_per_group_hyperparameters_independently(self, simple_model) -> None:
        """Get hyperparam values clips per-group hyperparameters independently."""
        layers = list(simple_model.children())
        # First group lr above max, second group lr below min
        optimizer = torch.optim.SGD(
            [
                {"params": layers[0].parameters(), "lr": 0.5},
                {"params": layers[1].parameters(), "lr": 1e-7},
            ]
        )

        schema = {
            "lr": {
                "type": "log",
                "std": 0.1,
                "min": 1e-4,
                "max": 0.1,
                "shared": False,
            }
        }

        state = State(optimizer)
        state.setup_schema(schema)

        values = state.get_hyperparam_values()

        # Verify first group clipped to max, second to min
        assert values["lr"][0] == 0.1
        assert values["lr"][1] == 1e-4

    def test_clips_shared_hyperparameters(self, simple_model) -> None:
        """Get hyperparam values clips shared hyperparameters."""
        layers = list(simple_model.children())
        # Both groups have lr above max (shared means first group value matters)
        optimizer = torch.optim.SGD(
            [
                {"params": layers[0].parameters(), "lr": 0.5},
                {"params": layers[1].parameters(), "lr": 0.6},
            ]
        )

        schema = {
            "lr": {
                "type": "log",
                "std": 0.1,
                "min": 1e-4,
                "max": 0.1,
                "shared": True,
            }
        }

        state = State(optimizer)
        state.setup_schema(schema)

        values = state.get_hyperparam_values()

        # Verify returned value clipped to max (shared returns length-1 list)
        assert len(values["lr"]) == 1
        assert values["lr"][0] == 0.1

    def test_clips_linear_hyperparameter(self, simple_model) -> None:
        """Get hyperparam values clips linear hyperparameter to bounds."""
        # Create optimizer with weight_decay above max
        optimizer = torch.optim.SGD(
            simple_model.parameters(), lr=0.01, weight_decay=0.5
        )

        schema = {
            "weight_decay": {
                "type": "linear",
                "std": 0.001,
                "min": 0.0,
                "max": 0.1,
                "shared": True,
            }
        }

        state = State(optimizer)
        state.setup_schema(schema)

        values = state.get_hyperparam_values()

        # Verify returned weight_decay clipped to max
        assert values["weight_decay"][0] == 0.1

    def test_no_clipping_when_min_max_not_in_schema(self, simple_model) -> None:
        """Get hyperparam values does not clip when min/max not specified."""
        # Create optimizer with lr outside typical bounds
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=5.0)

        schema = {
            "lr": {
                "type": "log",
                "std": 0.1,
                "shared": True,
            }
        }

        state = State(optimizer)
        state.setup_schema(schema)

        values = state.get_hyperparam_values()

        # Verify returned lr unchanged (no bounds to clip to)
        assert values["lr"][0] == 5.0

    def test_clips_multiple_hyperparameters(self, simple_model) -> None:
        """Get hyperparam values clips multiple hyperparameters when needed."""
        # Create optimizer with both lr and weight_decay out of bounds
        optimizer = torch.optim.SGD(
            simple_model.parameters(), lr=0.5, weight_decay=0.5
        )

        schema = {
            "lr": {
                "type": "log",
                "std": 0.1,
                "min": 1e-4,
                "max": 0.1,
                "shared": True,
            },
            "weight_decay": {
                "type": "linear",
                "std": 0.001,
                "min": 0.0,
                "max": 0.1,
                "shared": True,
            },
        }

        state = State(optimizer)
        state.setup_schema(schema)

        values = state.get_hyperparam_values()

        # Verify both returned values clipped to max
        assert values["lr"][0] == 0.1
        assert values["weight_decay"][0] == 0.1


class TestSetHyperparamValues:
    """Tests that set_hyperparam_values updates hyperparameters correctly."""

    def test_set_hyperparam_values_raises_when_schema_not_set(
        self,
        optimizer_single_group,
    ) -> None:
        """Set hyperparam values raises RuntimeError when schema not set up."""
        state = State(optimizer_single_group)

        with pytest.raises(RuntimeError) as excinfo:
            state.set_hyperparam_values({"lr": [0.01]})

        assert "Schema must be set up" in str(excinfo.value)

    def test_set_hyperparam_values_broadcasts_shared_hyperparameter_to_all_groups(
        self,
        optimizer_multi_group,
        shared_schema,
    ) -> None:
        """Set hyperparam values broadcasts shared hyperparameter to all param groups."""
        state = State(optimizer_multi_group)
        state.setup_schema(shared_schema)

        new_lr = 0.05
        state.set_hyperparam_values({"lr": [new_lr]})

        for pg in optimizer_multi_group.param_groups:
            assert pg["lr"] == new_lr

    def test_set_hyperparam_values_sets_per_group_values_individually(
        self,
        optimizer_multi_group,
        basic_schema,
    ) -> None:
        """Set hyperparam values sets per-group hyperparameters individually."""
        state = State(optimizer_multi_group)
        state.setup_schema(basic_schema)

        new_lrs = [0.02, 0.002]
        state.set_hyperparam_values({"lr": new_lrs})

        assert optimizer_multi_group.param_groups[0]["lr"] == new_lrs[0]
        assert optimizer_multi_group.param_groups[1]["lr"] == new_lrs[1]
