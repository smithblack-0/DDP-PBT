"""
Test suite for AbstractStrategy component.

AbstractStrategy orchestrates the round-end flow and provides configuration methods.
Tests validate schema building, orchestration, serialization, and abstract method contracts.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock

from src.ddp_pbt.base.abstract_strategy import AbstractStrategy
from src.ddp_pbt.base.state import State
from src.ddp_pbt.base.communication import Communication


class ConcreteTestStrategy(AbstractStrategy):
    """Concrete implementation of AbstractStrategy for testing."""

    def __init__(self, state, communication, config=None):
        super().__init__(state, communication, config or {})
        # Track method calls for testing
        self.score_called = False
        self.reduce_hyperparameters_called = False
        self.reduce_models_called = False
        self.reduce_optimizer_called = False

    def score(self, validation_metric, communication):
        """Test implementation: return uniform weights."""
        self.score_called = True
        # Return uniform weights (simple test strategy)
        world_size = 2  # Assume 2 workers for tests
        return [1.0 / world_size] * world_size

    def reduce_hyperparameters(self, world_weights, world_hyperparameters, communication):
        """Test implementation: return first worker's hyperparameters."""
        self.reduce_hyperparameters_called = True
        return world_hyperparameters[0] if world_hyperparameters else {}

    def reduce_models(self, world_weights, model_pytree, communication):
        """Test implementation: return input pytree unchanged."""
        self.reduce_models_called = True
        return model_pytree

    def reduce_optimizer(self, world_weights, optimizer_pytree, communication):
        """Test implementation: return input pytree unchanged."""
        self.reduce_optimizer_called = True
        return optimizer_pytree


class TestAbstractStrategyConfiguration:
    """Tests configuration methods and schema building."""

    def test_bind_log_hyperparameter_builds_schema(self):
        """bind_log_hyperparameter should add entry to schema."""
        # Create mock optimizer with lr parameter
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = State(optimizer)
        communication = Communication()
        strategy = ConcreteTestStrategy(state, communication)

        # Bind log hyperparameter
        strategy.bind_log_hyperparameter("lr", std=0.1, min=1e-4, max=1e-1)

        # Verify schema was built
        schema = strategy.state_dict()
        assert "lr" in schema
        assert schema["lr"]["type"] == "log"
        assert schema["lr"]["std"] == 0.1
        assert schema["lr"]["min"] == 1e-4
        assert schema["lr"]["max"] == 1e-1
        assert schema["lr"]["shared"] is True

    def test_bind_linear_hyperparameter_builds_schema(self):
        """bind_linear_hyperparameter should add entry to schema."""
        # Create mock optimizer with weight_decay parameter
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)

        state = State(optimizer)
        communication = Communication()
        strategy = ConcreteTestStrategy(state, communication)

        # Bind linear hyperparameter
        strategy.bind_linear_hyperparameter("weight_decay", std=0.001, min=0.0, max=0.1, shared=False)

        # Verify schema was built
        schema = strategy.state_dict()
        assert "weight_decay" in schema
        assert schema["weight_decay"]["type"] == "linear"
        assert schema["weight_decay"]["std"] == 0.001
        assert schema["weight_decay"]["min"] == 0.0
        assert schema["weight_decay"]["max"] == 0.1
        assert schema["weight_decay"]["shared"] is False

    def test_bind_multiple_hyperparameters(self):
        """Should support binding multiple hyperparameters."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)

        state = State(optimizer)
        communication = Communication()
        strategy = ConcreteTestStrategy(state, communication)

        # Bind both
        strategy.bind_log_hyperparameter("lr", std=0.1, min=1e-5)
        strategy.bind_linear_hyperparameter("weight_decay", std=0.001)

        # Verify both in schema
        schema = strategy.state_dict()
        assert "lr" in schema
        assert "weight_decay" in schema

    def test_bind_validates_hyperparameter_exists(self):
        """Binding should validate that hyperparameter exists in optimizer."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = State(optimizer)
        communication = Communication()
        strategy = ConcreteTestStrategy(state, communication)

        # Try to bind non-existent hyperparameter
        with pytest.raises((ValueError, KeyError, RuntimeError)):
            strategy.bind_log_hyperparameter("nonexistent_param", std=0.1, min=1e-5)

    def test_constructor_accepts_config_dict(self):
        """Constructor should accept native JSON config dict."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)

        state = State(optimizer)
        communication = Communication()

        config = {
            "lr": {"type": "log", "std": 0.1, "min": 1e-4, "max": 1e-1, "shared": True},
            "weight_decay": {"type": "linear", "std": 0.001, "min": 0.0, "max": 0.1, "shared": False}
        }

        strategy = ConcreteTestStrategy(state, communication, config)

        # Verify schema was loaded
        schema = strategy.state_dict()
        assert schema == config


class TestAbstractStrategyProperties:
    """Tests property accessors."""

    def test_valid_binding_targets_delegates_to_state(self):
        """valid_binding_targets should return bindable hyperparameters from state."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)

        state = State(optimizer)
        communication = Communication()
        strategy = ConcreteTestStrategy(state, communication)

        targets = strategy.valid_binding_targets
        assert isinstance(targets, list)
        # Should include lr and weight_decay
        assert any("lr" in str(t) for t in targets)
        assert any("weight_decay" in str(t) for t in targets)

    def test_optimizer_property_exposes_optimizer(self):
        """optimizer property should expose the optimizer from state."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = State(optimizer)
        communication = Communication()
        strategy = ConcreteTestStrategy(state, communication)

        assert strategy.optimizer is optimizer


class TestAbstractStrategySerialization:
    """Tests state_dict and load_state_dict."""

    def test_state_dict_returns_schema(self):
        """state_dict should return the schema for checkpointing."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)

        state = State(optimizer)
        communication = Communication()
        strategy = ConcreteTestStrategy(state, communication)

        strategy.bind_log_hyperparameter("lr", std=0.1, min=1e-5)
        strategy.bind_linear_hyperparameter("weight_decay", std=0.001)

        schema = strategy.state_dict()
        assert isinstance(schema, dict)
        assert "lr" in schema
        assert "weight_decay" in schema

    def test_load_state_dict_restores_schema(self):
        """load_state_dict should restore schema from checkpoint."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)

        state = State(optimizer)
        communication = Communication()

        # Create first strategy and save schema
        strategy1 = ConcreteTestStrategy(state, communication)
        strategy1.bind_log_hyperparameter("lr", std=0.1, min=1e-4, max=1e-1)
        saved_schema = strategy1.state_dict()

        # Create second strategy and load schema
        strategy2 = ConcreteTestStrategy(state, communication)
        strategy2.load_state_dict(saved_schema)

        # Verify schema was restored
        restored_schema = strategy2.state_dict()
        assert restored_schema == saved_schema


class TestAbstractStrategyOrchestration:
    """Tests step orchestration flow."""

    def test_step_calls_all_four_methods(self):
        """step should call score, reduce_hyperparameters, reduce_models, reduce_optimizer."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer
        state.valid_hyperparameter_paths = ["param_groups[0]['lr']"]
        state.get_hyperparam_values.return_value = {"lr": [0.001]}
        state.get_model_tensors.return_value = {"param1": torch.randn(3, 3)}
        state.get_optimizer_tensors.return_value = {"state1": torch.randn(3, 3)}

        communication = Mock(spec=Communication)
        communication.gather_pytree_list.return_value = [
            {"lr": [0.001]},
            {"lr": [0.002]}
        ]

        strategy = ConcreteTestStrategy(state, communication, {"lr": {"type": "log", "std": 0.1, "min": 1e-5, "shared": True}})

        # Call step
        strategy.step(0.95)

        # Verify all four methods were called
        assert strategy.score_called
        assert strategy.reduce_hyperparameters_called
        assert strategy.reduce_models_called
        assert strategy.reduce_optimizer_called

    def test_step_injects_results_back_via_state(self):
        """step should inject reduced results back via state."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer
        state.valid_hyperparameter_paths = ["param_groups[0]['lr']"]
        state.get_hyperparam_values.return_value = {"lr": [0.001]}
        state.get_model_tensors.return_value = {"param1": torch.randn(3, 3)}
        state.get_optimizer_tensors.return_value = {"state1": torch.randn(3, 3)}

        communication = Mock(spec=Communication)
        communication.gather_pytree_list.return_value = [
            {"lr": [0.001]},
            {"lr": [0.002]}
        ]

        strategy = ConcreteTestStrategy(state, communication, {"lr": {"type": "log", "std": 0.1, "min": 1e-5, "shared": True}})

        # Call step
        strategy.step(0.95)

        # Verify state injection methods were called
        state.set_hyperparam_values.assert_called_once()
        state.set_model_tensors.assert_called_once()
        state.set_optimizer_tensors.assert_called_once()

    def test_step_passes_communication_to_abstract_methods(self):
        """step should pass communication to all four abstract methods."""
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        optimizer = torch.optim.Adam(params, lr=0.001)

        state = Mock(spec=State)
        state.optimizer = optimizer
        state.valid_hyperparameter_paths = ["param_groups[0]['lr']"]
        state.get_hyperparam_values.return_value = {"lr": [0.001]}
        state.get_model_tensors.return_value = {"param1": torch.randn(3, 3)}
        state.get_optimizer_tensors.return_value = {"state1": torch.randn(3, 3)}

        communication = Mock(spec=Communication)
        communication.gather_pytree_list.return_value = [
            {"lr": [0.001]},
            {"lr": [0.002]}
        ]

        class TrackingStrategy(ConcreteTestStrategy):
            """Strategy that tracks communication parameter."""
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.comm_passed = []

            def score(self, validation_metric, communication):
                self.comm_passed.append(communication)
                return super().score(validation_metric, communication)

            def reduce_hyperparameters(self, world_weights, world_hyperparameters, communication):
                self.comm_passed.append(communication)
                return super().reduce_hyperparameters(world_weights, world_hyperparameters, communication)

            def reduce_models(self, world_weights, model_pytree, communication):
                self.comm_passed.append(communication)
                return super().reduce_models(world_weights, model_pytree, communication)

            def reduce_optimizer(self, world_weights, optimizer_pytree, communication):
                self.comm_passed.append(communication)
                return super().reduce_optimizer(world_weights, optimizer_pytree, communication)

        strategy = TrackingStrategy(state, communication, {"lr": {"type": "log", "std": 0.1, "min": 1e-5, "shared": True}})

        # Call step
        strategy.step(0.95)

        # Verify communication was passed to all four methods
        assert len(strategy.comm_passed) == 4
        assert all(c is communication for c in strategy.comm_passed)
