"""
Test suite for TopScoreStrategy.

TopScoreStrategy selects the best-performing worker and perturbs their hyperparameters.
Tests validate scoring logic, hyperparameter perturbation, and model/optimizer reduction.
"""

import torch
from unittest.mock import Mock

from src.ddp_pbt.Strategies.top_score_strategy import TopScoreStrategy, make_top_score_strategy
from src.ddp_pbt.base.state import State
from src.ddp_pbt.base.communication import Communication
from src.ddp_pbt.base.perturber import Perturber


class TestTopScoreStrategyScoring:
    """Tests score method that finds best worker."""

    def test_score_selects_worker_with_highest_metric(self):
        """score should return world weights with 1.0 at best worker index."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer

        communication = Mock(spec=Communication)
        # Worker 1 has best validation metric (lowest loss)
        validation_metrics = [0.5, 0.1, 0.3]

        perturber = Mock(spec=Perturber)
        strategy = TopScoreStrategy(state, communication, perturber=perturber)
        world_weights = strategy.score(validation_metrics, communication)

        # Should select worker 1 (index 1) with lowest loss
        assert len(world_weights) == 3
        assert world_weights[1] == 1.0
        assert world_weights[0] == 0.0
        assert world_weights[2] == 0.0
        assert sum(world_weights) == 1.0



class TestTopScoreStrategyReduceHyperparameters:
    """Tests reduce_hyperparameters that selects winner and perturbs."""

    def test_reduce_hyperparameters_selects_winner_and_perturbs(self):
        """reduce_hyperparameters should extract winner's values and perturb."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer

        communication = Mock(spec=Communication)

        # Create strategy with perturber
        perturber = Mock(spec=Perturber)
        perturber.perturb.return_value = {"lr": [0.0015]}

        strategy = TopScoreStrategy(state, communication, perturber=perturber)

        # Winner is worker 1
        world_weights = [0.0, 1.0, 0.0]
        world_hyperparameters = [
            {"lr": [0.001]},
            {"lr": [0.002]},  # Winner
            {"lr": [0.003]}
        ]

        result = strategy.reduce_hyperparameters(
            world_weights, world_hyperparameters, communication
        )

        # Should have perturbed winner's values
        perturber.perturb.assert_called_once_with({"lr": [0.002]})
        assert result == {"lr": [0.0015]}


class TestTopScoreStrategyReduceModels:
    """Tests reduce_models that uses communication to get winner's model."""

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
        strategy = TopScoreStrategy(state, communication, perturber=perturber)

        world_weights = [0.0, 1.0, 0.0]
        model_pytree = {"param1": torch.tensor([1.0, 2.0])}

        result = strategy.reduce_models(world_weights, model_pytree, communication)

        communication.reduce_by_world_weights.assert_called_once_with(
            world_weights, model_pytree
        )
        assert result == expected_result


class TestTopScoreStrategyReduceOptimizer:
    """Tests reduce_optimizer that uses communication to get winner's optimizer state."""

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
        strategy = TopScoreStrategy(state, communication, perturber=perturber)

        world_weights = [0.0, 1.0, 0.0]
        optimizer_pytree = {"state1": torch.tensor([1.0, 2.0])}

        result = strategy.reduce_optimizer(world_weights, optimizer_pytree, communication)

        communication.reduce_by_world_weights.assert_called_once_with(
            world_weights, optimizer_pytree
        )
        assert result == expected_result


class TestMakeTopScoreStrategyFactory:
    """Tests factory function that wires up TopScoreStrategy with dependencies."""

    def test_factory_creates_strategy_with_dependencies(self):
        """make_top_score_strategy should create fully wired strategy."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        # Mock Communication class since distributed world not initialized
        mock_comm_class = Mock(return_value=Mock(spec=Communication))
        strategy = make_top_score_strategy(optimizer, communication_class=mock_comm_class)

        # Should be TopScoreStrategy instance
        assert isinstance(strategy, TopScoreStrategy)

        # Should have perturber configured
        assert strategy._perturber is not None

        # Should be usable with builder pattern
        strategy.bind_log_hyperparameter("lr", std=0.1, min=1e-5)
        assert "lr" in strategy.schema

    def test_factory_accepts_config_dict(self):
        """make_top_score_strategy should accept native config dict."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        config = {
            "lr": {"type": "log", "std": 0.1, "min": 1e-5, "max": 1e-1, "shared": True}
        }

        # Mock Communication class since distributed world not initialized
        mock_comm_class = Mock(return_value=Mock(spec=Communication))
        strategy = make_top_score_strategy(optimizer, config=config, communication_class=mock_comm_class)

        # Schema should be loaded
        assert strategy.schema == config


import os
import sys
import json
import tempfile
from pathlib import Path
import torch.multiprocessing as mp
import torch.distributed as dist


def integration_worker(rank, world_size, output_dir, master_addr, master_port):
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

        # Create strategy and bind hyperparameters
        strategy = make_top_score_strategy(optimizer)
        strategy.bind_linear_hyperparameter("lr", std=0.001, min=0.001, max=0.1)
        strategy.bind_log_hyperparameter("weight_decay", std=0.1, min=1e-5, max=0.01)

        # Run multiple rounds
        results = []
        for round_idx in range(3):
            # Simulate training - just record current hyperparams
            current_lr = optimizer.param_groups[0]['lr']
            current_wd = optimizer.param_groups[0]['weight_decay']

            # Simulate validation loss (rank 0 best in round 0, rank 1 best in round 1)
            if round_idx == 0:
                val_loss = 1.0 if rank == 0 else 2.0
            elif round_idx == 1:
                val_loss = 2.0 if rank == 0 else 1.0
            else:
                val_loss = float(rank + 1)

            results.append({
                "round": round_idx,
                "lr": current_lr,
                "weight_decay": current_wd,
                "val_loss": val_loss
            })

            # Step strategy
            strategy.step(val_loss)

        # Save results
        output_file = Path(output_dir) / f"rank_{rank}.json"
        with open(output_file, "w") as f:
            json.dump({"results": results}, f)

    finally:
        dist.destroy_process_group()


import pytest


@pytest.mark.distributed
@pytest.mark.skipif(sys.platform == "win32", reason="GLOO not supported on Windows")
class TestTopScoreStrategyIntegration:
    """Integration tests with real distributed environment."""

    def test_adamw_with_lr_and_weight_decay_binding(self):
        """Integration test: AdamW with linear lr and log weight_decay over multiple rounds."""
        world_size = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            # Spawn worker processes
            mp.spawn(
                integration_worker,
                args=(world_size, tmpdir, "localhost", "29600"),
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

            # Verify both workers completed all rounds
            assert len(results[0]) == 3
            assert len(results[1]) == 3

            # Verify hyperparameters changed between rounds (perturbation occurred)
            rank0_lr_round0 = results[0][0]["lr"]
            rank0_lr_round1 = results[0][1]["lr"]
            # LR should have changed due to perturbation
            assert rank0_lr_round0 != rank0_lr_round1, "LR should change between rounds"

            # Verify hyperparameters stay within bounds
            for rank_results in results:
                for round_result in rank_results:
                    assert 0.001 <= round_result["lr"] <= 0.1, "LR should stay within bounds"
                    assert 1e-5 <= round_result["weight_decay"] <= 0.01, "Weight decay should stay within bounds"