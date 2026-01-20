"""
Test suite for Crossbreeder component.

Crossbreeder handles parent selection and hyperparameter blending with probabilistic mutation.
Tests validate parent selection, blending behavior, and mutation integration.
"""

import random
import pytest
from unittest.mock import Mock

from src.ddp_pbt.base.crossbreeder import Crossbreeder
from src.ddp_pbt.base.perturber import Perturber as Permuter


class TestCrossbreederSetup:
    """Tests setup methods for schema and permuter."""

    def test_setup_schema_stores_reference(self):
        """Crossbreeder should store schema for blending operations."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=0.0)
        schema = {
            "lr": {"type": "log", "std": 0.1, "min": 1e-5, "shared": True}
        }
        crossbreeder.setup_schema(schema)

        # Verify schema is stored by attempting crossbreeding
        world_hyperparameters = [
            {"lr": [0.001]},
            {"lr": [0.002]}
        ]
        parent_weights = [0.5, 0.5]
        result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)
        assert "lr" in result

    def test_setup_permuter_stores_reference(self):
        """Crossbreeder should store permuter for probabilistic mutation."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=1.0)
        schema = {
            "lr": {"type": "linear", "std": 0.1, "shared": True}
        }
        crossbreeder.setup_schema(schema)

        permuter = Mock(spec=Permuter)
        permuter.perturb.return_value = {"lr": [0.0015]}
        crossbreeder.setup_permuter(permuter)

        # With mutation_rate=1.0, permuter should be called
        world_hyperparameters = [{"lr": [0.001]}, {"lr": [0.002]}]
        parent_weights = [0.5, 0.5]
        random.seed(42)
        result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)

        # Permuter should have been called
        assert permuter.perturb.called


class TestCrossbreederParentSelection:
    """Tests parent selection from world weights."""

    def test_selects_two_parents_from_top_k(self):
        """select_parents should randomly choose 2 from top parent_pool_depth workers."""
        crossbreeder = Crossbreeder(parent_pool_depth=3, mutation_rate=0.0)

        # 5 workers with different weights
        world_weights = [0.05, 0.15, 0.30, 0.40, 0.10]  # Sum = 1.0

        random.seed(42)
        parent_weights = crossbreeder.select_parents(world_weights)

        # Should return weights with exactly 2 non-zero entries
        non_zero_count = sum(1 for w in parent_weights if w > 0)
        assert non_zero_count == 2

        # Selected parents should be from top 3 (indices 2, 3, 4 after sorting)
        # which are the workers with original weights [0.40, 0.30, 0.15]
        non_zero_indices = [i for i, w in enumerate(parent_weights) if w > 0]
        top_3_indices = [3, 2, 1]  # Original indices of top 3 workers
        for idx in non_zero_indices:
            assert idx in top_3_indices

        # Weights should sum to 1.0
        assert abs(sum(parent_weights) - 1.0) < 1e-6

    def test_parent_weights_sum_to_one(self):
        """Selected parent weights should be normalized to sum to 1.0."""
        crossbreeder = Crossbreeder(parent_pool_depth=4, mutation_rate=0.0)
        world_weights = [0.1, 0.2, 0.3, 0.4]

        parent_weights = crossbreeder.select_parents(world_weights)

        assert abs(sum(parent_weights) - 1.0) < 1e-6

    def test_with_exactly_two_workers(self):
        """With exactly 2 workers and parent_pool_depth >= 2, should select both."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=0.0)
        world_weights = [0.4, 0.6]

        parent_weights = crossbreeder.select_parents(world_weights)

        # Both should be selected
        non_zero_count = sum(1 for w in parent_weights if w > 0)
        assert non_zero_count == 2
        assert abs(sum(parent_weights) - 1.0) < 1e-6

    def test_parent_pool_depth_larger_than_world_size(self):
        """If parent_pool_depth > world_size, should raise ValueError."""
        crossbreeder = Crossbreeder(parent_pool_depth=10, mutation_rate=0.0)
        world_weights = [0.2, 0.3, 0.5]

        with pytest.raises(ValueError, match="parent_pool_depth.*cannot exceed world_size"):
            crossbreeder.select_parents(world_weights)


class TestCrossbreederHyperparameterBlending:
    """Tests hyperparameter crossbreeding logic."""

    def test_linear_blending_without_mutation(self):
        """Linear parameters should blend directly in linear space."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=0.0)
        schema = {
            "weight_decay": {"type": "linear", "std": 0.001, "shared": True}
        }
        crossbreeder.setup_schema(schema)

        world_hyperparameters = [
            {"weight_decay": [0.01]},
            {"weight_decay": [0.02]}
        ]
        parent_weights = [0.3, 0.7]

        result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)

        # Expected: 0.3 * 0.01 + 0.7 * 0.02 = 0.003 + 0.014 = 0.017
        expected = 0.3 * 0.01 + 0.7 * 0.02
        assert abs(result["weight_decay"][0] - expected) < 1e-6

    def test_log_blending_without_mutation(self):
        """Log parameters should blend in log-space then convert back."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=0.0)
        schema = {
            "lr": {"type": "log", "std": 0.1, "shared": True}
        }
        crossbreeder.setup_schema(schema)

        world_hyperparameters = [
            {"lr": [0.001]},
            {"lr": [0.01]}
        ]
        parent_weights = [0.5, 0.5]

        result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)

        # Expected: exp(0.5 * log(0.001) + 0.5 * log(0.01))
        import math
        log1 = math.log(0.001)
        log2 = math.log(0.01)
        expected = math.exp(0.5 * log1 + 0.5 * log2)
        assert abs(result["lr"][0] - expected) < 1e-6

    def test_per_group_blending(self):
        """Per-group hyperparameters should blend element-wise."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=0.0)
        schema = {
            "weight_decay": {"type": "linear", "std": 0.001, "shared": False}
        }
        crossbreeder.setup_schema(schema)

        world_hyperparameters = [
            {"weight_decay": [0.01, 0.02]},
            {"weight_decay": [0.03, 0.04]}
        ]
        parent_weights = [0.4, 0.6]

        result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)

        # Element-wise blending
        expected_0 = 0.4 * 0.01 + 0.6 * 0.03
        expected_1 = 0.4 * 0.02 + 0.6 * 0.04
        assert abs(result["weight_decay"][0] - expected_0) < 1e-6
        assert abs(result["weight_decay"][1] - expected_1) < 1e-6

    def test_mixed_log_and_linear_blending(self):
        """Should handle mix of log and linear parameters correctly."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=0.0)
        schema = {
            "lr": {"type": "log", "std": 0.1, "shared": True},
            "weight_decay": {"type": "linear", "std": 0.001, "shared": True}
        }
        crossbreeder.setup_schema(schema)

        world_hyperparameters = [
            {"lr": [0.001], "weight_decay": [0.01]},
            {"lr": [0.01], "weight_decay": [0.02]}
        ]
        parent_weights = [0.5, 0.5]

        result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)

        # Both parameters should be blended
        assert "lr" in result
        assert "weight_decay" in result

        # Linear blending for weight_decay
        expected_wd = 0.5 * 0.01 + 0.5 * 0.02
        assert abs(result["weight_decay"][0] - expected_wd) < 1e-6


class TestCrossbreederProbabilisticMutation:
    """Tests probabilistic mutation after crossbreeding."""

    def test_mutation_rate_zero_never_mutates(self):
        """With mutation_rate=0.0, permuter should never be called."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=0.0)
        schema = {
            "lr": {"type": "linear", "std": 0.1, "shared": True}
        }
        crossbreeder.setup_schema(schema)

        permuter = Mock(spec=Permuter)
        permuter.perturb.return_value = {"lr": [0.999]}
        crossbreeder.setup_permuter(permuter)

        world_hyperparameters = [{"lr": [0.001]}, {"lr": [0.002]}]
        parent_weights = [0.5, 0.5]

        # Run multiple times
        for _ in range(10):
            result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)

        # Permuter should never be called
        assert not permuter.perturb.called

    def test_mutation_rate_one_always_mutates(self):
        """With mutation_rate=1.0, permuter should always be called."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=1.0)
        schema = {
            "lr": {"type": "linear", "std": 0.1, "shared": True}
        }
        crossbreeder.setup_schema(schema)

        permuter = Mock(spec=Permuter)
        permuter.perturb.return_value = {"lr": [0.999]}
        crossbreeder.setup_permuter(permuter)

        world_hyperparameters = [{"lr": [0.001]}, {"lr": [0.002]}]
        parent_weights = [0.5, 0.5]

        # Run multiple times
        call_count = 0
        for _ in range(10):
            result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)
            call_count = permuter.perturb.call_count

        # Permuter should be called every time
        assert call_count == 10

    def test_mutation_returns_permuted_result(self):
        """When mutation occurs, should return permuter's result."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=1.0)
        schema = {
            "lr": {"type": "linear", "std": 0.1, "shared": True}
        }
        crossbreeder.setup_schema(schema)

        permuter = Mock(spec=Permuter)
        permuter.perturb.return_value = {"lr": [0.123456]}
        crossbreeder.setup_permuter(permuter)

        world_hyperparameters = [{"lr": [0.001]}, {"lr": [0.002]}]
        parent_weights = [0.5, 0.5]

        result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)

        # Result should be what permuter returned
        assert result["lr"][0] == 0.123456

    def test_mutation_rate_partial_mutates_probabilistically(self):
        """With 0 < mutation_rate < 1, mutation should occur probabilistically."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=0.5)
        schema = {
            "lr": {"type": "linear", "std": 0.1, "shared": True}
        }
        crossbreeder.setup_schema(schema)

        permuter = Mock(spec=Permuter)
        permuter.perturb.return_value = {"lr": [0.999]}
        crossbreeder.setup_permuter(permuter)

        world_hyperparameters = [{"lr": [0.001]}, {"lr": [0.002]}]
        parent_weights = [0.5, 0.5]

        random.seed(42)
        # Run multiple times to check probability
        for _ in range(50):
            result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)

        # With mutation_rate=0.5, permuter should be called roughly half the time
        call_count = permuter.perturb.call_count
        # Allow some variance (between 15 and 35 out of 50)
        assert 15 <= call_count <= 35


class TestCrossbreederEdgeCases:
    """Tests edge cases and error conditions."""

    def test_with_world_weights_filtering(self):
        """Should only use non-zero weighted parents from world_weights."""
        crossbreeder = Crossbreeder(parent_pool_depth=4, mutation_rate=0.0)
        schema = {
            "lr": {"type": "linear", "std": 0.1, "shared": True}
        }
        crossbreeder.setup_schema(schema)

        # World with 4 workers, but only 2 have non-zero weights (already filtered)
        world_hyperparameters = [
            {"lr": [0.001]},
            {"lr": [0.002]},
            {"lr": [0.003]},
            {"lr": [0.004]}
        ]
        parent_weights = [0.0, 0.3, 0.7, 0.0]  # Only workers 1 and 2 selected

        result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)

        # Should blend workers 1 and 2
        expected = 0.3 * 0.002 + 0.7 * 0.003
        assert abs(result["lr"][0] - expected) < 1e-6

    def test_empty_schema(self):
        """Crossbreeder with empty schema should handle empty hyperparameters."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=0.0)
        schema = {}
        crossbreeder.setup_schema(schema)

        world_hyperparameters = [{}, {}]
        parent_weights = [0.5, 0.5]

        result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)

        assert result == {}
