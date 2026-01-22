"""
Test suite for WeightedAverageStrategy.

WeightedAverageStrategy creates performance-weighted averages of hyperparameters and models.
Tests validate scoring normalization, weighted averaging, and perturbation.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.ddp_pbt.base.communication import Communication
from src.ddp_pbt.base.perturber import Perturber
from src.ddp_pbt.base.state import State
from src.ddp_pbt.strategies.weighted_average_strategy import (
    WeightedAverageStrategy,
    make_weighted_average_strategy,
)

# Test Fixtures and Helpers


def integration_worker_weighted_average(
    rank,
    world_size,
    output_dir,
    master_addr,
    master_port,
):
    """Worker function for integration test."""
    # Setup environment
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Initialize process group
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    try:
        # Create simple model
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)

        # Create strategy
        strategy = make_weighted_average_strategy(optimizer)
        strategy.bind_linear_hyperparameter("lr", std=0.001, min=0.001, max=0.1)
        strategy.bind_log_hyperparameter("weight_decay", std=0.1, min=1e-5, max=0.01)

        # Run multiple rounds
        results = []
        for round_idx in range(3):
            # Simulate training - just record current hyperparams
            current_lr = optimizer.param_groups[0]["lr"]
            current_wd = optimizer.param_groups[0]["weight_decay"]

            # Simulate validation loss (varying performance each round)
            if round_idx == 0:
                val_loss = 1.0 + rank * 0.5  # rank 0 best
            elif round_idx == 1:
                val_loss = 1.0 + (world_size - rank - 1) * 0.5  # rank 2 best
            else:
                val_loss = 1.5  # all equal

            results.append(
                {
                    "round": round_idx,
                    "lr": current_lr,
                    "weight_decay": current_wd,
                    "val_loss": val_loss,
                }
            )

            # Step strategy
            strategy.step(val_loss)

        # Save results
        output_file = Path(output_dir) / f"rank_{rank}.json"
        with open(output_file, "w") as f:
            json.dump({"results": results}, f)

    finally:
        dist.destroy_process_group()


# Unit Tests


class TestWeightedAverageStrategyScoring:
    """Tests score method that creates performance-weighted distribution."""

    def test_score_creates_normalized_distribution(self):
        """score should return normalized probability distribution favoring better performers."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer

        communication = Mock(spec=Communication)
        perturber = Mock(spec=Perturber)

        strategy = WeightedAverageStrategy(state, communication, perturber=perturber)

        # Validation metrics: [0.5, 0.1, 0.3]
        # Max = 0.5, differences: [0.0, 0.4, 0.2]
        # Normalized: [0.0, 0.4, 0.2] / 0.6 = [0.0, 0.667, 0.333]
        validation_metrics = [0.5, 0.1, 0.3]
        world_weights = strategy.score(validation_metrics, communication)

        # Verify properties
        assert len(world_weights) == 3
        assert sum(world_weights) == pytest.approx(1.0)

        # Worker 1 (best) should have highest weight
        assert world_weights[1] > world_weights[2]
        assert world_weights[1] > world_weights[0]

        # Worker 0 (worst) should have lowest weight (0.0)
        assert world_weights[0] == pytest.approx(0.0)

    def test_score_handles_all_equal_metrics(self):
        """score should create uniform distribution when all metrics are equal."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer

        communication = Mock(spec=Communication)
        perturber = Mock(spec=Perturber)

        strategy = WeightedAverageStrategy(state, communication, perturber=perturber)

        # All equal metrics
        validation_metrics = [0.5, 0.5, 0.5]
        world_weights = strategy.score(validation_metrics, communication)

        # Should create uniform distribution
        assert len(world_weights) == 3
        assert sum(world_weights) == pytest.approx(1.0)
        assert world_weights[0] == pytest.approx(1.0 / 3.0)
        assert world_weights[1] == pytest.approx(1.0 / 3.0)
        assert world_weights[2] == pytest.approx(1.0 / 3.0)

    def test_score_with_negative_metrics(self):
        """score should handle negative metrics correctly."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer

        communication = Mock(spec=Communication)
        perturber = Mock(spec=Perturber)

        strategy = WeightedAverageStrategy(state, communication, perturber=perturber)

        # Negative metrics: [-0.5, -0.1, -0.3]
        # Max = -0.1, differences: [-0.4, 0.0, -0.2]
        # Absolute: [0.4, 0.0, 0.2]
        # Normalized: [0.4, 0.0, 0.2] / 0.6 = [0.667, 0.0, 0.333]
        validation_metrics = [-0.5, -0.1, -0.3]
        world_weights = strategy.score(validation_metrics, communication)

        # Verify properties
        assert len(world_weights) == 3
        assert sum(world_weights) == pytest.approx(1.0)

        # Worker 0 (best, -0.5) should have highest weight
        assert world_weights[0] > world_weights[2]
        assert world_weights[0] > world_weights[1]

        # Worker 1 (worst, -0.1) should have lowest weight (0.0)
        assert world_weights[1] == pytest.approx(0.0)

    def test_score_with_mixed_positive_negative_metrics(self):
        """score should handle mixed positive and negative metrics."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer

        communication = Mock(spec=Communication)
        perturber = Mock(spec=Perturber)

        strategy = WeightedAverageStrategy(state, communication, perturber=perturber)

        # Mixed metrics: [-0.5, 0.5, 0.0]
        # Max = 0.5, differences: [-1.0, 0.0, -0.5]
        # Absolute: [1.0, 0.0, 0.5]
        # Normalized: [1.0, 0.0, 0.5] / 1.5 = [0.667, 0.0, 0.333]
        validation_metrics = [-0.5, 0.5, 0.0]
        world_weights = strategy.score(validation_metrics, communication)

        # Verify properties
        assert len(world_weights) == 3
        assert sum(world_weights) == pytest.approx(1.0)

        # Worker 0 (best) should have highest weight
        assert world_weights[0] > world_weights[2]
        assert world_weights[0] > world_weights[1]

        # Worker 1 (worst) should have lowest weight (0.0)
        assert world_weights[1] == pytest.approx(0.0)


class TestWeightedAverageStrategyReduceHyperparameters:
    """Tests reduce_hyperparameters that performs weighted averaging."""

    def test_reduce_hyperparameters_weighted_average_single_key(self):
        """reduce_hyperparameters should compute weighted average for single key."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer

        communication = Mock(spec=Communication)

        perturber = Mock(spec=Perturber)
        perturber.perturb.return_value = {"lr": [0.0015]}

        strategy = WeightedAverageStrategy(state, communication, perturber=perturber)

        # Weights: [0.5, 0.3, 0.2]
        # Values: [0.001, 0.002, 0.003]
        # Average: 0.5*0.001 + 0.3*0.002 + 0.2*0.003 = 0.0005 + 0.0006 + 0.0006 = 0.0017
        world_weights = [0.5, 0.3, 0.2]
        world_hyperparameters = [
            {"lr": [0.001]},
            {"lr": [0.002]},
            {"lr": [0.003]},
        ]

        result = strategy.reduce_hyperparameters(
            world_weights,
            world_hyperparameters,
            communication,
        )

        # Verify perturber called with averaged values
        perturber.perturb.assert_called_once()
        call_args = perturber.perturb.call_args[0][0]
        assert call_args["lr"][0] == pytest.approx(0.0017)

        # Result should be perturbed value
        assert result == {"lr": [0.0015]}

    def test_reduce_hyperparameters_weighted_average_multiple_keys(self):
        """reduce_hyperparameters should compute weighted average for multiple keys."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer

        communication = Mock(spec=Communication)

        perturber = Mock(spec=Perturber)
        perturber.perturb.return_value = {"lr": [0.0015], "weight_decay": [0.0012]}

        strategy = WeightedAverageStrategy(state, communication, perturber=perturber)

        world_weights = [0.5, 0.3, 0.2]
        world_hyperparameters = [
            {"lr": [0.001], "weight_decay": [0.001]},
            {"lr": [0.002], "weight_decay": [0.002]},
            {"lr": [0.003], "weight_decay": [0.003]},
        ]

        result = strategy.reduce_hyperparameters(
            world_weights,
            world_hyperparameters,
            communication,
        )

        # Verify perturber called with averaged values
        perturber.perturb.assert_called_once()
        call_args = perturber.perturb.call_args[0][0]
        assert call_args["lr"][0] == pytest.approx(0.0017)
        assert call_args["weight_decay"][0] == pytest.approx(0.0017)

        # Result should be perturbed values
        assert result == {"lr": [0.0015], "weight_decay": [0.0012]}

    def test_reduce_hyperparameters_weighted_average_multiple_param_groups(self):
        """reduce_hyperparameters should handle multiple param groups (list values)."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer

        communication = Mock(spec=Communication)

        perturber = Mock(spec=Perturber)
        perturber.perturb.return_value = {"lr": [0.0015, 0.0025]}

        strategy = WeightedAverageStrategy(state, communication, perturber=perturber)

        # Two param groups
        world_weights = [0.5, 0.5]
        world_hyperparameters = [
            {"lr": [0.001, 0.002]},
            {"lr": [0.002, 0.003]},
        ]

        result = strategy.reduce_hyperparameters(
            world_weights,
            world_hyperparameters,
            communication,
        )

        # Verify perturber called with averaged values
        # Group 0: 0.5*0.001 + 0.5*0.002 = 0.0015
        # Group 1: 0.5*0.002 + 0.5*0.003 = 0.0025
        perturber.perturb.assert_called_once()
        call_args = perturber.perturb.call_args[0][0]
        assert call_args["lr"][0] == pytest.approx(0.0015)
        assert call_args["lr"][1] == pytest.approx(0.0025)

        # Result should be perturbed values
        assert result == {"lr": [0.0015, 0.0025]}


class TestWeightedAverageStrategyReduceModels:
    """Tests reduce_models that uses communication to average models."""

    def test_reduce_models_delegates_to_communication(self):
        """reduce_models should use communication.reduce_by_world_weights."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer

        communication = Mock(spec=Communication)
        expected_result = {"param1": torch.tensor([1.0, 2.0])}
        communication.reduce_by_world_weights.return_value = expected_result

        perturber = Mock(spec=Perturber)
        strategy = WeightedAverageStrategy(state, communication, perturber=perturber)

        world_weights = [0.5, 0.3, 0.2]
        model_pytree = {"param1": torch.tensor([1.0, 2.0])}

        result = strategy.reduce_models(world_weights, model_pytree, communication)

        communication.reduce_by_world_weights.assert_called_once_with(
            world_weights,
            model_pytree,
        )
        assert result == expected_result


class TestWeightedAverageStrategyReduceOptimizer:
    """Tests reduce_optimizer that uses communication to average optimizer state."""

    def test_reduce_optimizer_delegates_to_communication(self):
        """reduce_optimizer should use communication.reduce_by_world_weights."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer

        communication = Mock(spec=Communication)
        expected_result = {"state1": torch.tensor([1.0, 2.0])}
        communication.reduce_by_world_weights.return_value = expected_result

        perturber = Mock(spec=Perturber)
        strategy = WeightedAverageStrategy(state, communication, perturber=perturber)

        world_weights = [0.5, 0.3, 0.2]
        optimizer_pytree = {"state1": torch.tensor([1.0, 2.0])}

        result = strategy.reduce_optimizer(world_weights, optimizer_pytree, communication)

        communication.reduce_by_world_weights.assert_called_once_with(
            world_weights,
            optimizer_pytree,
        )
        assert result == expected_result


class TestWeightedAverageStrategyStep:
    """Tests step() orchestration method."""

    def test_step_orchestrates_full_round(self):
        """step() should orchestrate extraction, gathering, scoring, reduction, and injection."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        # Setup mocks
        state = Mock(spec=State)
        state.optimizer = optimizer
        state.get_hyperparam_values.return_value = {"lr": [0.001]}
        state.get_model_tensors.return_value = {"param1": torch.tensor([1.0])}
        state.get_optimizer_tensors.return_value = {"state1": torch.tensor([2.0])}

        communication = Mock(spec=Communication)
        communication.gather_pytree_list.side_effect = [
            [{"lr": [0.001]}, {"lr": [0.002]}],  # world_hyperparameters
            [0.5, 0.3],  # validation_metrics
        ]
        communication.reduce_by_world_weights.side_effect = [
            {"param1": torch.tensor([1.5])},  # reduced model
            {"state1": torch.tensor([2.5])},  # reduced optimizer
        ]

        perturber = Mock(spec=Perturber)
        perturber.perturb.return_value = {"lr": [0.0015]}

        strategy = WeightedAverageStrategy(state, communication, perturber=perturber)

        # Call step
        strategy.step(0.5)

        # Verify orchestration
        state.get_hyperparam_values.assert_called_once()
        state.get_model_tensors.assert_called_once()
        state.get_optimizer_tensors.assert_called_once()

        assert communication.gather_pytree_list.call_count == 2
        perturber.perturb.assert_called_once()

        state.set_hyperparam_values.assert_called_once_with({"lr": [0.0015]})
        state.set_model_tensors.assert_called_once()
        state.set_optimizer_tensors.assert_called_once()


class TestMakeWeightedAverageStrategyFactory:
    """Tests factory function that wires up WeightedAverageStrategy with dependencies."""

    def test_factory_creates_strategy_with_dependencies(self):
        """make_weighted_average_strategy should create fully wired strategy."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        # Mock Communication class since distributed world not initialized
        mock_comm_class = Mock(return_value=Mock(spec=Communication))
        strategy = make_weighted_average_strategy(optimizer, communication_class=mock_comm_class)

        # Should be WeightedAverageStrategy instance
        assert isinstance(strategy, WeightedAverageStrategy)

        # Should have perturber configured
        assert strategy._perturber is not None

        # Should be usable with builder pattern
        strategy.bind_log_hyperparameter("lr", std=0.1, min=1e-5)
        assert "lr" in strategy.schema

    def test_factory_accepts_config_dict(self):
        """make_weighted_average_strategy should accept native config dict."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        config = {
            "lr": {"type": "log", "std": 0.1, "min": 1e-5, "max": 1e-1, "shared": True},
        }

        # Mock Communication class since distributed world not initialized
        mock_comm_class = Mock(return_value=Mock(spec=Communication))
        strategy = make_weighted_average_strategy(
            optimizer,
            config=config,
            communication_class=mock_comm_class,
        )

        # Schema should be loaded
        assert strategy.schema == config


# Integration Tests


@pytest.mark.distributed
@pytest.mark.skipif(sys.platform == "win32", reason="GLOO not supported on Windows")
class TestWeightedAverageStrategyIntegration:
    """Integration tests with real distributed environment."""

    def test_weighted_averaging_over_multiple_rounds(self):
        """Integration test: WeightedAverageStrategy with varying performance over rounds."""
        world_size = 3

        with tempfile.TemporaryDirectory() as tmpdir:
            # Spawn worker processes
            mp.spawn(
                integration_worker_weighted_average,
                args=(world_size, tmpdir, "localhost", "29602"),
                nprocs=world_size,
                join=True,
            )

            # Collect results from all ranks
            results = []
            for rank in range(world_size):
                output_file = Path(tmpdir) / f"rank_{rank}.json"
                with open(output_file, "r") as f:
                    data = json.load(f)
                    results.append(data["results"])

            # Verify all workers completed all rounds
            assert len(results[0]) == 3
            assert len(results[1]) == 3
            assert len(results[2]) == 3

            # Verify hyperparameters changed between rounds (perturbation occurred)
            rank0_lr_round0 = results[0][0]["lr"]
            rank0_lr_round1 = results[0][1]["lr"]
            # LR should have changed due to averaging + perturbation
            assert rank0_lr_round0 != rank0_lr_round1, "LR should change between rounds"

            # Verify hyperparameters stay within bounds
            for rank_results in results:
                for round_result in rank_results:
                    assert 0.001 <= round_result["lr"] <= 0.1, "LR should stay within bounds"
                    assert (
                        1e-5 <= round_result["weight_decay"] <= 0.01
                    ), "Weight decay should stay within bounds"
