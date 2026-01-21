"""
Test suite for Crossbreeder component.

Crossbreeder handles parent selection and hyperparameter blending with probabilistic mutation.
Tests validate parent selection, blending behavior, and mutation integration.
"""

import random
import pytest
from unittest.mock import Mock

from src.ddp_pbt.base.crossbreeder import Crossbreeder
from src.ddp_pbt.base.perturber import Perturber


class TestCrossbreederSetup:
    """Tests setup methods for schema and perturber."""

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



class TestCrossbreederHyperparameterCrossbreeding:
    """Tests hyperparameter crossbreeding logic."""

    def test_linear_crossbreeding_without_mutation(self):
        """Linear parameters should crossbreed via discrete 50/50 choice per allele."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=0.0)
        schema = {
            "weight_decay": {"type": "linear", "std": 0.001, "shared": True}
        }
        crossbreeder.setup_schema(schema)

        world_hyperparameters = [
            {"weight_decay": [0.01]},
            {"weight_decay": [0.02]}
        ]
        parent_weights = [0.5, 0.5]

        # Run 100 times to verify probabilistic behavior
        results = []
        for _ in range(100):
            result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)
            results.append(result["weight_decay"][0])

        # Should see both parent values selected (discrete choice)
        assert 0.01 in results, "Parent A value should appear"
        assert 0.02 in results, "Parent B value should appear"

        # Roughly 50/50 distribution (allow statistical variance)
        count_parent_a = sum(1 for r in results if r == 0.01)
        assert 30 <= count_parent_a <= 70, f"Expected 30-70% parent A, got {count_parent_a}%"

    def test_log_crossbreeding_without_mutation(self):
        """Log parameters should crossbreed via discrete 50/50 choice per allele."""
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

        # Run 100 times to verify probabilistic behavior
        results = []
        for _ in range(100):
            result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)
            results.append(result["lr"][0])

        # Should see both parent values selected (discrete choice)
        assert 0.001 in results, "Parent A value should appear"
        assert 0.01 in results, "Parent B value should appear"

        # Roughly 50/50 distribution (allow statistical variance)
        count_parent_a = sum(1 for r in results if r == 0.001)
        assert 30 <= count_parent_a <= 70, f"Expected 30-70% parent A, got {count_parent_a}%"

    def test_per_group_crossbreeding(self):
        """Per-group hyperparameters should crossbreed independently per allele."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=0.0)
        schema = {
            "weight_decay": {"type": "linear", "std": 0.001, "shared": False}
        }
        crossbreeder.setup_schema(schema)

        world_hyperparameters = [
            {"weight_decay": [0.01, 0.02]},
            {"weight_decay": [0.03, 0.04]}
        ]
        parent_weights = [0.5, 0.5]

        # Run 100 times to verify probabilistic behavior per allele
        results_0 = []
        results_1 = []
        for _ in range(100):
            result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)
            results_0.append(result["weight_decay"][0])
            results_1.append(result["weight_decay"][1])

        # Each allele should independently choose from parents
        assert 0.01 in results_0 and 0.03 in results_0, "Allele 0 should see both parent values"
        assert 0.02 in results_1 and 0.04 in results_1, "Allele 1 should see both parent values"

    def test_mixed_log_and_linear_crossbreeding(self):
        """Should handle mix of log and linear parameters via discrete crossbreeding."""
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

        # Run 100 times to verify both parameters crossbreed independently
        lr_results = []
        wd_results = []
        for _ in range(100):
            result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)
            lr_results.append(result["lr"][0])
            wd_results.append(result["weight_decay"][0])

        # Both parameters should crossbreed independently
        assert 0.001 in lr_results and 0.01 in lr_results, "lr should see both parent values"
        assert 0.01 in wd_results and 0.02 in wd_results, "weight_decay should see both parent values"


class TestCrossbreederProbabilisticMutation:
    """Tests probabilistic mutation after crossbreeding."""

    def test_mutation_rate_zero_never_mutates(self):
        """With mutation_rate=0.0, perturber should never be called."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=0.0)
        schema = {
            "lr": {"type": "linear", "std": 0.1, "shared": True}
        }
        crossbreeder.setup_schema(schema)

        perturber = Mock(spec=Perturber)
        perturber.apply_perturbation.return_value = 0.999
        crossbreeder.setup_perturber(perturber)

        world_hyperparameters = [{"lr": [0.001]}, {"lr": [0.002]}]
        parent_weights = [0.5, 0.5]

        # Run multiple times
        for _ in range(10):
            result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)

        # Perturber should never be called
        assert perturber.apply_perturbation.call_count == 0

    def test_mutation_rate_one_always_mutates(self):
        """With mutation_rate=1.0, each allele should be mutated."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=1.0)
        schema = {
            "lr": {"type": "linear", "std": 0.1, "shared": True}
        }
        crossbreeder.setup_schema(schema)

        perturber = Mock(spec=Perturber)
        perturber.apply_perturbation.return_value = 0.999
        crossbreeder.setup_perturber(perturber)

        world_hyperparameters = [{"lr": [0.001]}, {"lr": [0.002]}]
        parent_weights = [0.5, 0.5]

        # Run multiple times - each run mutates 1 allele
        for _ in range(10):
            result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)

        # With mutation_rate=1.0 and 1 allele, apply_perturbation called 10 times (once per run)
        assert perturber.apply_perturbation.call_count == 10

    def test_mutation_returns_perturbed_result(self):
        """When mutation occurs, allele should be replaced with perturber's result."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=1.0)
        schema = {
            "lr": {"type": "linear", "std": 0.1, "shared": True}
        }
        crossbreeder.setup_schema(schema)

        perturber = Mock(spec=Perturber)
        perturber.apply_perturbation.return_value = 0.123456
        crossbreeder.setup_perturber(perturber)

        world_hyperparameters = [{"lr": [0.001]}, {"lr": [0.002]}]
        parent_weights = [0.5, 0.5]

        result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)

        # Result should be what perturber returned (mutation_rate=1.0 guarantees mutation)
        assert result["lr"][0] == 0.123456

    def test_mutation_rate_partial_mutates_probabilistically(self):
        """With 0 < mutation_rate < 1, each allele should mutate probabilistically."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=0.5)
        schema = {
            "lr": {"type": "linear", "std": 0.1, "shared": True}
        }
        crossbreeder.setup_schema(schema)

        perturber = Mock(spec=Perturber)
        perturber.apply_perturbation.return_value = 0.999
        crossbreeder.setup_perturber(perturber)

        world_hyperparameters = [{"lr": [0.001]}, {"lr": [0.002]}]
        parent_weights = [0.5, 0.5]

        random.seed(42)
        # Run multiple times to check probability (1 allele per run)
        for _ in range(50):
            result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)

        # With mutation_rate=0.5 and 1 allele, should mutate roughly half the time
        call_count = perturber.apply_perturbation.call_count
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
        parent_weights = [0.0, 0.5, 0.5, 0.0]  # Only workers 1 and 2 selected

        # Run 100 times to verify only selected parents contribute
        results = []
        for _ in range(100):
            result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)
            results.append(result["lr"][0])

        # Should only see values from workers 1 and 2 (0.002 and 0.003)
        assert 0.002 in results and 0.003 in results, "Should see both selected parent values"
        assert 0.001 not in results and 0.004 not in results, "Should not see non-selected parent values"

    def test_empty_schema(self):
        """Crossbreeder with empty schema should handle empty hyperparameters."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=0.0)
        schema = {}
        crossbreeder.setup_schema(schema)

        world_hyperparameters = [{}, {}]
        parent_weights = [0.5, 0.5]

        result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)

        assert result == {}
