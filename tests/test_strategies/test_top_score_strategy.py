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

        strategy = make_top_score_strategy(optimizer)

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

        strategy = make_top_score_strategy(optimizer, config=config)

        # Schema should be loaded
        assert strategy.schema == config

# Note to claude: Hey, where are my integration tests? Lets make an AdamW, bind to it using linear on
# on learning rate and log on weight decay, and step a few times, for a dumb simple model .
# We need integration tests too.