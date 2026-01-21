"""
Test suite for TopPopulationSexualStrategy.

TopPopulationSexualStrategy selects 2 parents from top-k and crossbreeds their hyperparameters.
Tests validate parent selection, crossbreeding, root parent selection, and asymmetric reduction.
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

from src.ddp_pbt.Strategies.top_population_sexual_strategy import (
    TopPopulationSexualStrategy,
    make_top_population_sexual_strategy
)
from src.ddp_pbt.base.state import State
from src.ddp_pbt.base.communication import Communication
from src.ddp_pbt.base.perturber import Perturber
from src.ddp_pbt.base.crossbreeder import Crossbreeder


# Test Fixtures and Helpers

def integration_worker_population_sexual(rank, world_size, output_dir, master_addr, master_port):
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

        # Create strategy with 67% reproduction (ensures top_k=2 with world_size=3), 10% mutation rate
        strategy = make_top_population_sexual_strategy(
            reproduction_percentage=0.67,
            optimizer=optimizer,
            mutation_rate=0.1,
            max_hyperparameter_search_depth=3
        )
        strategy.bind_linear_hyperparameter("lr", std=0.001, min=0.001, max=0.1)
        strategy.bind_log_hyperparameter("weight_decay", std=0.1, min=1e-5, max=0.01)

        # Run multiple rounds
        results = []
        for round_idx in range(3):
            # Simulate training - just record current hyperparams
            current_lr = optimizer.param_groups[0]['lr']
            current_wd = optimizer.param_groups[0]['weight_decay']

            # Simulate validation loss (varying by rank)
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

class TestTopPopulationSexualStrategyScoring:
    """Tests score method that selects 2 parents from top-k."""

    def test_score_raises_error_when_world_size_less_than_two(self):
        """score should raise ValueError if world_size < 2."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer

        communication = Mock(spec=Communication)
        crossbreeder = Mock(spec=Crossbreeder)

        strategy = TopPopulationSexualStrategy(num_k=1, state=state, communication=communication, crossbreeder=crossbreeder)
        validation_metrics = [0.5]  # Only 1 worker

        with pytest.raises(ValueError, match="Cannot crossbreed with less than 2 workers"):
            strategy.score(validation_metrics, communication)

    def test_score_raises_error_when_num_k_exceeds_world_size(self):
        """score should raise ValueError if num_k > world_size."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer

        communication = Mock(spec=Communication)
        crossbreeder = Mock(spec=Crossbreeder)

        # num_k=5 but only 3 workers
        strategy = TopPopulationSexualStrategy(num_k=5, state=state, communication=communication, crossbreeder=crossbreeder)
        validation_metrics = [0.5, 0.1, 0.3]

        with pytest.raises(ValueError, match="parent_pool_depth.*cannot exceed world_size"):
            strategy.score(validation_metrics, communication)

    def test_score_selects_exactly_two_parents_fuzzing(self):
        """score should select exactly 2 parents from top-k (fuzzing test)."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer

        communication = Mock(spec=Communication)
        crossbreeder = Mock(spec=Crossbreeder)

        # Create strategy with k=2
        strategy = TopPopulationSexualStrategy(num_k=2, state=state, communication=communication, crossbreeder=crossbreeder)

        # Validation metrics: [0.5, 0.1, 0.3]
        # Ascending order (best first): [1, 2, 0] (indices)
        # Top-k=2 should be indices 1 and 2
        validation_metrics = [0.5, 0.1, 0.3]

        # Run 100 times to verify probabilistic behavior
        for _ in range(100):
            world_weights = strategy.score(validation_metrics, communication)

            # Verify basic properties
            assert len(world_weights) == 3
            assert sum(world_weights) == pytest.approx(1.0)

            # Exactly 2 parents should be selected
            non_zero_count = sum(1 for w in world_weights if w > 0)
            assert non_zero_count == 2, "Should select exactly 2 parents"

            # Each parent should have weight 0.5
            non_zero_weights = [w for w in world_weights if w > 0]
            assert all(w == 0.5 for w in non_zero_weights), "Each parent should have weight 0.5"

            # Both parents must be from top-k (indices 1 or 2, NOT index 0)
            parent_indices = [i for i, w in enumerate(world_weights) if w > 0]
            for idx in parent_indices:
                assert idx in [1, 2], f"Parent {idx} not in top-k [1, 2]"

    def test_score_stores_root_parent(self):
        """score should store root_parent for model/optimizer selection."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer

        communication = Mock(spec=Communication)
        crossbreeder = Mock(spec=Crossbreeder)

        strategy = TopPopulationSexualStrategy(num_k=3, state=state, communication=communication, crossbreeder=crossbreeder)
        validation_metrics = [0.5, 0.1, 0.3]

        # Run multiple times
        root_parents_seen = set()
        for _ in range(100):
            world_weights = strategy.score(validation_metrics, communication)

            # root_parent should be one of the selected parents
            parent_indices = [i for i, w in enumerate(world_weights) if w > 0]
            assert strategy.root_parent in parent_indices, "root_parent must be one of selected parents"

            root_parents_seen.add(strategy.root_parent)

        # Over many runs, should see variation in root_parent
        assert len(root_parents_seen) >= 2, "Should see different root_parents selected"

    def test_score_boundary_k_equals_two(self):
        """score with k=2 and world_size=2 should select both workers."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer

        communication = Mock(spec=Communication)
        crossbreeder = Mock(spec=Crossbreeder)

        # k=2, world_size=2 means both workers are top-k
        strategy = TopPopulationSexualStrategy(num_k=2, state=state, communication=communication, crossbreeder=crossbreeder)
        validation_metrics = [0.5, 0.1]

        world_weights = strategy.score(validation_metrics, communication)

        # Should select both workers
        assert world_weights[0] == 0.5
        assert world_weights[1] == 0.5


class TestTopPopulationSexualStrategyReduceHyperparameters:
    """Tests reduce_hyperparameters that delegates to Crossbreeder."""

    def test_reduce_hyperparameters_calls_crossbreeder(self):
        """reduce_hyperparameters should call crossbreeder.crossbreed_hyperparameters()."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer

        communication = Mock(spec=Communication)

        # Mock Crossbreeder
        crossbreeder = Mock(spec=Crossbreeder)
        crossbreeder.crossbreed_hyperparameters.return_value = {"lr": [0.0015]}

        strategy = TopPopulationSexualStrategy(num_k=2, state=state, communication=communication, crossbreeder=crossbreeder)

        # Two parents selected
        world_weights = [0.0, 0.5, 0.5]
        world_hyperparameters = [
            {"lr": [0.001]},
            {"lr": [0.002]},  # Parent 1
            {"lr": [0.003]}   # Parent 2
        ]

        result = strategy.reduce_hyperparameters(
            world_weights, world_hyperparameters, communication
        )

        # Should have called crossbreeder with correct arguments
        crossbreeder.crossbreed_hyperparameters.assert_called_once_with(
            world_hyperparameters, world_weights
        )
        assert result == {"lr": [0.0015]}


class TestTopPopulationSexualStrategyReduceModels:
    """Tests reduce_models that uses root_parent for selection."""

    def test_reduce_models_uses_root_parent(self):
        """reduce_models should create one-hot weights using root_parent."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer

        communication = Mock(spec=Communication)
        expected_result = {"param1": torch.tensor([1.0, 2.0])}
        communication.reduce_by_world_weights.return_value = expected_result

        crossbreeder = Mock(spec=Crossbreeder)
        strategy = TopPopulationSexualStrategy(num_k=2, state=state, communication=communication, crossbreeder=crossbreeder)

        # Set root_parent to index 1
        strategy.root_parent = 1

        # Original weights: [0.0, 0.5, 0.5] (2 parents)
        world_weights = [0.0, 0.5, 0.5]
        model_pytree = {"param1": torch.tensor([1.0, 2.0])}

        result = strategy.reduce_models(world_weights, model_pytree, communication)

        # Should have created one-hot weights for root_parent
        call_args = communication.reduce_by_world_weights.call_args[0]
        selection_weights = call_args[0]

        assert len(selection_weights) == 3
        assert selection_weights[0] == 0.0
        assert selection_weights[1] == 1.0  # root_parent
        assert selection_weights[2] == 0.0
        assert sum(selection_weights) == 1.0

        assert result == expected_result


class TestTopPopulationSexualStrategyReduceOptimizer:
    """Tests reduce_optimizer that uses root_parent for selection."""

    def test_reduce_optimizer_uses_root_parent(self):
        """reduce_optimizer should create one-hot weights using root_parent."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer

        communication = Mock(spec=Communication)
        expected_result = {"state1": torch.tensor([1.0, 2.0])}
        communication.reduce_by_world_weights.return_value = expected_result

        crossbreeder = Mock(spec=Crossbreeder)
        strategy = TopPopulationSexualStrategy(num_k=2, state=state, communication=communication, crossbreeder=crossbreeder)

        # Set root_parent to index 2
        strategy.root_parent = 2

        # Original weights: [0.0, 0.5, 0.5] (2 parents)
        world_weights = [0.0, 0.5, 0.5]
        optimizer_pytree = {"state1": torch.tensor([1.0, 2.0])}

        result = strategy.reduce_optimizer(world_weights, optimizer_pytree, communication)

        # Should have created one-hot weights for root_parent
        call_args = communication.reduce_by_world_weights.call_args[0]
        selection_weights = call_args[0]

        assert len(selection_weights) == 3
        assert selection_weights[0] == 0.0
        assert selection_weights[1] == 0.0
        assert selection_weights[2] == 1.0  # root_parent
        assert sum(selection_weights) == 1.0

        assert result == expected_result


class TestTopPopulationSexualStrategyStep:
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
            [0.5, 0.3]  # validation_metrics
        ]
        communication.reduce_by_world_weights.side_effect = [
            {"param1": torch.tensor([1.5])},  # reduced model
            {"state1": torch.tensor([2.5])}   # reduced optimizer
        ]

        crossbreeder = Mock(spec=Crossbreeder)
        crossbreeder.crossbreed_hyperparameters.return_value = {"lr": [0.0015]}

        strategy = TopPopulationSexualStrategy(num_k=2, state=state, communication=communication, crossbreeder=crossbreeder)

        # Call step
        strategy.step(0.5)

        # Verify orchestration
        state.get_hyperparam_values.assert_called_once()
        state.get_model_tensors.assert_called_once()
        state.get_optimizer_tensors.assert_called_once()

        assert communication.gather_pytree_list.call_count == 2
        crossbreeder.crossbreed_hyperparameters.assert_called_once()

        state.set_hyperparam_values.assert_called_once_with({"lr": [0.0015]})
        state.set_model_tensors.assert_called_once()
        state.set_optimizer_tensors.assert_called_once()


class TestMakeTopPopulationSexualStrategyFactory:
    """Tests factory function that wires up TopPopulationSexualStrategy with dependencies."""

    def test_factory_creates_strategy_with_crossbreeder(self):
        """make_top_population_sexual_strategy should create strategy with Crossbreeder."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        # Mock Communication class with world_size
        mock_comm = Mock(spec=Communication)
        mock_comm.world_size = 10
        mock_comm_class = Mock(return_value=mock_comm)

        # 30% of 10 workers = 3
        strategy = make_top_population_sexual_strategy(
            reproduction_percentage=0.3,
            optimizer=optimizer,
            mutation_rate=0.05,
            max_hyperparameter_search_depth=3,
            communication_class=mock_comm_class
        )

        # Should be TopPopulationSexualStrategy instance
        assert isinstance(strategy, TopPopulationSexualStrategy)

    def test_factory_creates_functional_strategy(self):
        """make_top_population_sexual_strategy should create functional strategy."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        # Mock Communication class
        mock_comm = Mock(spec=Communication)
        mock_comm.world_size = 10
        mock_comm_class = Mock(return_value=mock_comm)

        strategy = make_top_population_sexual_strategy(
            reproduction_percentage=0.5,
            optimizer=optimizer,
            mutation_rate=0.1,
            max_hyperparameter_search_depth=3,
            communication_class=mock_comm_class
        )

        # Strategy should be functional - test with builder pattern
        strategy.bind_log_hyperparameter("lr", std=0.1, min=1e-5)
        assert "lr" in strategy.schema

    def test_factory_accepts_config_dict(self):
        """make_top_population_sexual_strategy should accept native config dict."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        config = {
            "lr": {"type": "log", "std": 0.1, "min": 1e-5, "max": 1e-1, "shared": True}
        }

        # Mock Communication class
        mock_comm = Mock(spec=Communication)
        mock_comm.world_size = 10
        mock_comm_class = Mock(return_value=mock_comm)

        strategy = make_top_population_sexual_strategy(
            reproduction_percentage=0.5,
            optimizer=optimizer,
            mutation_rate=0.1,
            max_hyperparameter_search_depth=3,
            config=config,
            communication_class=mock_comm_class
        )

        # Schema should be loaded
        assert strategy.schema == config

    def test_factory_handles_mutation_rate_edge_cases(self):
        """make_top_population_sexual_strategy should handle mutation_rate 0.0 and 1.0."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        # Mock Communication class
        mock_comm = Mock(spec=Communication)
        mock_comm.world_size = 10
        mock_comm_class = Mock(return_value=mock_comm)

        # Test 0.0 mutation rate - strategy should be created successfully
        strategy_0 = make_top_population_sexual_strategy(
            reproduction_percentage=0.5,
            optimizer=optimizer,
            mutation_rate=0.0,
            max_hyperparameter_search_depth=3,
            communication_class=mock_comm_class
        )
        assert isinstance(strategy_0, TopPopulationSexualStrategy)

        # Test 1.0 mutation rate - strategy should be created successfully
        strategy_1 = make_top_population_sexual_strategy(
            reproduction_percentage=0.5,
            optimizer=optimizer,
            mutation_rate=1.0,
            max_hyperparameter_search_depth=3,
            communication_class=mock_comm_class
        )
        assert isinstance(strategy_1, TopPopulationSexualStrategy)


# Integration Tests

@pytest.mark.distributed
@pytest.mark.skipif(sys.platform == "win32", reason="GLOO not supported on Windows")
class TestTopPopulationSexualStrategyIntegration:
    """Integration tests with real distributed environment."""

    def test_sexual_reproduction_with_crossbreeding_over_rounds(self):
        """Integration test: TopPopulationSexualStrategy with crossbreeding and mutation."""
        world_size = 3

        with tempfile.TemporaryDirectory() as tmpdir:
            # Spawn worker processes
            mp.spawn(
                integration_worker_population_sexual,
                args=(world_size, tmpdir, "localhost", "29604"),
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

            # Note: With crossbreeding, hyperparameters may or may not change depending on
            # which parent alleles are selected and whether mutation occurs.
            # We verify the mechanism works without requiring specific probabilistic outcomes.

            # Verify hyperparameters stay within bounds
            for rank_results in results:
                for round_result in rank_results:
                    assert 0.001 <= round_result["lr"] <= 0.1, "LR should stay within bounds"
                    assert 1e-5 <= round_result["weight_decay"] <= 0.01, "Weight decay should stay within bounds"
