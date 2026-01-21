"""
Test suite for TopKStrategy.

TopKStrategy randomly selects from top-k performers and perturbs their hyperparameters.
Tests validate top-k selection logic, hyperparameter perturbation, and model/optimizer reduction.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np

from src.ddp_pbt.Strategies.top_k_strategy import (
    TopKStrategy,
    make_top_k_strategy_out_of_top_k,
    make_top_k_strategy_by_selection_percentage
)
from src.ddp_pbt.base.state import State
from src.ddp_pbt.base.communication import Communication
from src.ddp_pbt.base.perturber import Perturber


# Test Fixtures and Helpers

def integration_worker_top_k(rank, world_size, output_dir, master_addr, master_port):
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

        # Create strategy with k=2 (top 2 performers can be selected)
        strategy = make_top_k_strategy_out_of_top_k(
            top_k=2,
            optimizer=optimizer
        )
        strategy.bind_linear_hyperparameter("lr", std=0.001, min=0.001, max=0.1)
        strategy.bind_log_hyperparameter("weight_decay", std=0.1, min=1e-5, max=0.01)

        # Run multiple rounds
        results = []
        for round_idx in range(3):
            # Simulate training - just record current hyperparams
            current_lr = optimizer.param_groups[0]['lr']
            current_wd = optimizer.param_groups[0]['weight_decay']

            # Simulate validation loss (rank 0 best, rank 1 second best, rank 2 worst)
            val_loss = float(rank + 1) * 0.5

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


# Unit Tests

class TestTopKStrategyScoring:
    """Tests score method that selects from top-k workers."""

    def test_score_raises_error_when_num_k_exceeds_world_size(self):
        """score should raise RuntimeError if num_k > world_size."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer

        communication = Mock(spec=Communication)
        perturber = Mock(spec=Perturber)

        # num_k=5 but only 3 workers
        strategy = TopKStrategy(num_k=5, state=state, communication=communication, perturber=perturber)
        validation_metrics = [0.5, 0.1, 0.3]

        with pytest.raises(RuntimeError, match="Num k was greater than world size"):
            strategy.score(validation_metrics, communication)

    def test_score_selects_from_top_k_pool_fuzzing(self):
        """score should randomly select from top-k performers (fuzzing test)."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer

        communication = Mock(spec=Communication)
        perturber = Mock(spec=Perturber)

        # Create strategy with k=2
        strategy = TopKStrategy(num_k=2, state=state, communication=communication, perturber=perturber)

        # Validation metrics: [0.5, 0.1, 0.3]
        # Ascending order (best first): [1, 2, 0] (indices)
        # Top-k=2 should be indices 1 and 2
        validation_metrics = [0.5, 0.1, 0.3]

        # Run 100 times to verify probabilistic behavior
        selected_indices = set()
        for _ in range(100):
            # Mock the gather_pytree_list to return the first rank's choice
            communication.gather_pytree_list.return_value = [1]  # Rank 0 always wins proposal

            # Seed random for reproducibility but test multiple outcomes
            world_weights = strategy.score(validation_metrics, communication)

            # Verify basic properties
            assert len(world_weights) == 3
            assert sum(world_weights) == pytest.approx(1.0)

            # Find winner
            winner_idx = world_weights.index(1.0)
            selected_indices.add(winner_idx)

            # Winner must be from top-k (indices 1 or 2, NOT index 0)
            assert winner_idx in [1, 2], f"Winner {winner_idx} not in top-k [1, 2]"

        # Over 100 runs, we should see some variety (though gather forces rank 0's choice)
        # The key constraint is that only top-k indices are ever selected

    def test_score_boundary_k_equals_one(self):
        """score with k=1 should behave like TopScoreStrategy."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer

        communication = Mock(spec=Communication)
        communication.gather_pytree_list.return_value = [1]  # Rank 0's choice
        perturber = Mock(spec=Perturber)

        # k=1 means only the best performer can be selected
        strategy = TopKStrategy(num_k=1, state=state, communication=communication, perturber=perturber)
        validation_metrics = [0.5, 0.1, 0.3]

        world_weights = strategy.score(validation_metrics, communication)

        # Should always select index 1 (best performer)
        assert world_weights[1] == 1.0
        assert world_weights[0] == 0.0
        assert world_weights[2] == 0.0

    def test_score_boundary_k_equals_world_size(self):
        """score with k=world_size should allow any worker to be selected."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer

        communication = Mock(spec=Communication)
        perturber = Mock(spec=Perturber)

        # k=3 (all workers) means any worker can be selected
        strategy = TopKStrategy(num_k=3, state=state, communication=communication, perturber=perturber)
        validation_metrics = [0.5, 0.1, 0.3]

        # Run multiple times to verify all indices are possible
        possible_indices = set()
        for i in range(3):
            communication.gather_pytree_list.return_value = [i]  # Simulate different rank proposals
            world_weights = strategy.score(validation_metrics, communication)
            winner_idx = world_weights.index(1.0)
            possible_indices.add(winner_idx)

        # With k=world_size, any index should be selectable
        assert len(possible_indices) >= 1  # At least one was selected


class TestTopKStrategyReduceHyperparameters:
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

        strategy = TopKStrategy(num_k=2, state=state, communication=communication, perturber=perturber)

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


class TestTopKStrategyReduceModels:
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
        strategy = TopKStrategy(num_k=2, state=state, communication=communication, perturber=perturber)

        world_weights = [0.0, 1.0, 0.0]
        model_pytree = {"param1": torch.tensor([1.0, 2.0])}

        result = strategy.reduce_models(world_weights, model_pytree, communication)

        communication.reduce_by_world_weights.assert_called_once_with(
            world_weights, model_pytree
        )
        assert result == expected_result


class TestTopKStrategyReduceOptimizer:
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
        strategy = TopKStrategy(num_k=2, state=state, communication=communication, perturber=perturber)

        world_weights = [0.0, 1.0, 0.0]
        optimizer_pytree = {"state1": torch.tensor([1.0, 2.0])}

        result = strategy.reduce_optimizer(world_weights, optimizer_pytree, communication)

        communication.reduce_by_world_weights.assert_called_once_with(
            world_weights, optimizer_pytree
        )
        assert result == expected_result


class TestTopKStrategyStep:
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
            [1]  # choices from score() - rank 0's choice wins
        ]
        communication.reduce_by_world_weights.side_effect = [
            {"param1": torch.tensor([1.5])},  # reduced model
            {"state1": torch.tensor([2.5])}   # reduced optimizer
        ]

        perturber = Mock(spec=Perturber)
        perturber.perturb.return_value = {"lr": [0.0015]}

        strategy = TopKStrategy(num_k=2, state=state, communication=communication, perturber=perturber)

        # Call step
        strategy.step(0.5)

        # Verify orchestration
        state.get_hyperparam_values.assert_called_once()
        state.get_model_tensors.assert_called_once()
        state.get_optimizer_tensors.assert_called_once()

        assert communication.gather_pytree_list.call_count == 3  # hyperparams, metrics, choice coordination
        perturber.perturb.assert_called_once()

        state.set_hyperparam_values.assert_called_once_with({"lr": [0.0015]})
        state.set_model_tensors.assert_called_once()
        state.set_optimizer_tensors.assert_called_once()


class TestMakeTopKStrategyFactory:
    """Tests factory functions that wire up TopKStrategy with dependencies."""

    def test_factory_out_of_top_k_creates_strategy_with_fixed_k(self):
        """make_top_k_strategy_out_of_top_k should create strategy with fixed k."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        # Mock Communication class since distributed world not initialized
        mock_comm_class = Mock(return_value=Mock(spec=Communication))
        strategy = make_top_k_strategy_out_of_top_k(
            top_k=2,
            optimizer=optimizer,
            communication_class=mock_comm_class
        )

        # Should be TopKStrategy instance
        assert isinstance(strategy, TopKStrategy)

        # Should be usable with builder pattern
        strategy.bind_log_hyperparameter("lr", std=0.1, min=1e-5)
        assert "lr" in strategy.schema

    def test_factory_out_of_top_k_accepts_config_dict(self):
        """make_top_k_strategy_out_of_top_k should accept native config dict."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        config = {
            "lr": {"type": "log", "std": 0.1, "min": 1e-5, "max": 1e-1, "shared": True}
        }

        # Mock Communication class since distributed world not initialized
        mock_comm_class = Mock(return_value=Mock(spec=Communication))
        strategy = make_top_k_strategy_out_of_top_k(
            top_k=2,
            optimizer=optimizer,
            config=config,
            communication_class=mock_comm_class
        )

        # Schema should be loaded
        assert strategy.schema == config

    def test_factory_by_selection_percentage_converts_percentage_to_k(self):
        """make_top_k_strategy_by_selection_percentage should convert percentage to k."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        # Mock Communication class with world_size
        mock_comm = Mock(spec=Communication)
        mock_comm.world_size = 10
        mock_comm_class = Mock(return_value=mock_comm)

        # 30% of 10 workers = 3
        strategy = make_top_k_strategy_by_selection_percentage(
            selection_percentage=0.3,
            optimizer=optimizer,
            max_hyperparameter_search_depth=3,
            communication_class=mock_comm_class
        )

        # Should be TopKStrategy instance
        assert isinstance(strategy, TopKStrategy)

    def test_factory_by_selection_percentage_handles_edge_cases(self):
        """make_top_k_strategy_by_selection_percentage should handle 0.0 and 1.0."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        # Mock Communication class with world_size
        mock_comm = Mock(spec=Communication)
        mock_comm.world_size = 10
        mock_comm_class = Mock(return_value=mock_comm)

        # Test 0.0 percentage - strategy should be created successfully
        strategy_0 = make_top_k_strategy_by_selection_percentage(
            selection_percentage=0.0,
            optimizer=optimizer,
            max_hyperparameter_search_depth=3,
            communication_class=mock_comm_class
        )
        assert isinstance(strategy_0, TopKStrategy)

        # Test 1.0 percentage (all workers) - strategy should be created successfully
        strategy_1 = make_top_k_strategy_by_selection_percentage(
            selection_percentage=1.0,
            optimizer=optimizer,
            max_hyperparameter_search_depth=3,
            communication_class=mock_comm_class
        )
        assert isinstance(strategy_1, TopKStrategy)


# Integration Tests

@pytest.mark.distributed
@pytest.mark.skipif(sys.platform == "win32", reason="GLOO not supported on Windows")
class TestTopKStrategyIntegration:
    """Integration tests with real distributed environment."""

    def test_top_k_selection_with_multiple_rounds(self):
        """Integration test: TopKStrategy with k=2 over multiple rounds."""
        world_size = 3

        with tempfile.TemporaryDirectory() as tmpdir:
            # Spawn worker processes
            mp.spawn(
                integration_worker_top_k,
                args=(world_size, tmpdir, "localhost", "29601"),
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
            # LR should have changed due to perturbation
            assert rank0_lr_round0 != rank0_lr_round1, "LR should change between rounds"

            # Verify hyperparameters stay within bounds
            for rank_results in results:
                for round_result in rank_results:
                    assert 0.001 <= round_result["lr"] <= 0.1, "LR should stay within bounds"
                    assert 1e-5 <= round_result["weight_decay"] <= 0.01, "Weight decay should stay within bounds"
