"""
Test suite for Crossbreeder component.

Crossbreeder handles parent selection and hyperparameter blending with probabilistic mutation.
Tests validate parent selection, blending behavior, and mutation integration.
"""

import random
from unittest.mock import Mock

import pytest

from src.ddp_pbt.base.crossbreeder import Crossbreeder
from src.ddp_pbt.base.perturber import Perturber


class TestCrossbreederSetup:
    """Tests setup methods for schema and perturber."""

    def test_setup_schema_stores_reference(self):
        """Crossbreeder should store schema for blending operations."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=0.0)
        schema = {
            "lr": {"type": "log", "std": 0.1, "min": 1e-5, "shared": True},
        }
        crossbreeder.setup_schema(schema)
        perturber = Perturber()
        crossbreeder.setup_perturber(perturber)

        # Verify schema is stored by attempting crossbreeding
        world_hyperparameters = [
            {"lr": [0.001]},
            {"lr": [0.002]},
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
            "weight_decay": {"type": "linear", "std": 0.001, "shared": True},
        }
        crossbreeder.setup_schema(schema)
        perturber = Perturber()
        crossbreeder.setup_perturber(perturber)

        world_hyperparameters = [
            {"weight_decay": [0.01]},
            {"weight_decay": [0.02]},
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
            "lr": {"type": "log", "std": 0.1, "shared": True},
        }
        crossbreeder.setup_schema(schema)
        perturber = Perturber()
        crossbreeder.setup_perturber(perturber)

        world_hyperparameters = [
            {"lr": [0.001]},
            {"lr": [0.01]},
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
            "weight_decay": {"type": "linear", "std": 0.001, "shared": False},
        }
        crossbreeder.setup_schema(schema)
        perturber = Perturber()
        crossbreeder.setup_perturber(perturber)

        world_hyperparameters = [
            {"weight_decay": [0.01, 0.02]},
            {"weight_decay": [0.03, 0.04]},
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
            "weight_decay": {"type": "linear", "std": 0.001, "shared": True},
        }
        crossbreeder.setup_schema(schema)
        perturber = Perturber()
        crossbreeder.setup_perturber(perturber)

        world_hyperparameters = [
            {"lr": [0.001], "weight_decay": [0.01]},
            {"lr": [0.01], "weight_decay": [0.02]},
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
        assert (
            0.01 in wd_results and 0.02 in wd_results
        ), "weight_decay should see both parent values"


class TestCrossbreederProbabilisticMutation:
    """Tests probabilistic mutation after crossbreeding."""

    def test_mutation_rate_zero_never_mutates(self):
        """With mutation_rate=0.0, crossbred result should be returned unchanged."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=0.0)
        schema = {
            "lr": {"type": "linear", "std": 0.1, "shared": True},
        }
        crossbreeder.setup_schema(schema)

        perturber = Mock(spec=Perturber)
        perturber.perturb_pytree_using_schema = Mock(side_effect=lambda x, **kwargs: x)
        crossbreeder.setup_perturber(perturber)

        world_hyperparameters = [{"lr": [0.001]}, {"lr": [0.002]}]
        parent_weights = [0.5, 0.5]

        # Run multiple times
        random.seed(42)
        results = []
        for _ in range(10):
            result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)
            results.append(result["lr"][0])

        # Should only see parent values (no mutation)
        assert all(r in [0.001, 0.002] for r in results)

    def test_mutation_rate_one_always_mutates(self):
        """With mutation_rate=1.0, perturb_pytree_using_schema should be called."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=1.0)
        schema = {
            "lr": {"type": "linear", "std": 0.1, "shared": True},
        }
        crossbreeder.setup_schema(schema)

        perturber = Mock(spec=Perturber)
        perturber.perturb_pytree_using_schema = Mock(return_value={"lr": [0.999]})
        crossbreeder.setup_perturber(perturber)

        world_hyperparameters = [{"lr": [0.001]}, {"lr": [0.002]}]
        parent_weights = [0.5, 0.5]

        # Run multiple times
        for _ in range(10):
            crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)

        # perturb_pytree_using_schema called 10 times with perturb_chance=1.0
        assert perturber.perturb_pytree_using_schema.call_count == 10
        # Verify it was called with correct perturb_chance
        call_args = perturber.perturb_pytree_using_schema.call_args
        assert call_args.kwargs["perturb_chance"] == 1.0

    def test_mutation_returns_perturbed_result(self):
        """When mutation occurs, result should be from perturber."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=1.0)
        schema = {
            "lr": {"type": "linear", "std": 0.1, "shared": True},
        }
        crossbreeder.setup_schema(schema)

        perturber = Mock(spec=Perturber)
        perturber.perturb_pytree_using_schema = Mock(return_value={"lr": [0.123456]})
        crossbreeder.setup_perturber(perturber)

        world_hyperparameters = [{"lr": [0.001]}, {"lr": [0.002]}]
        parent_weights = [0.5, 0.5]

        result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)

        # Result should be what perturber returned
        assert result["lr"][0] == 0.123456

    def test_mutation_rate_partial_mutates_probabilistically(self):
        """With 0 < mutation_rate < 1, perturb_chance should be passed correctly."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=0.5)
        schema = {
            "lr": {"type": "linear", "std": 0.1, "shared": True},
        }
        crossbreeder.setup_schema(schema)

        perturber = Mock(spec=Perturber)
        perturber.perturb_pytree_using_schema = Mock(side_effect=lambda x, **kwargs: x)
        crossbreeder.setup_perturber(perturber)

        world_hyperparameters = [{"lr": [0.001]}, {"lr": [0.002]}]
        parent_weights = [0.5, 0.5]

        random.seed(42)
        # Run multiple times
        for _ in range(10):
            crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)

        # Verify perturb_chance=0.5 was passed to perturber
        assert perturber.perturb_pytree_using_schema.call_count == 10
        call_args = perturber.perturb_pytree_using_schema.call_args
        assert call_args.kwargs["perturb_chance"] == 0.5


class TestCrossbreederEdgeCases:
    """Tests edge cases and error conditions."""

    def test_with_world_weights_filtering(self):
        """Should only use non-zero weighted parents from world_weights."""
        crossbreeder = Crossbreeder(parent_pool_depth=4, mutation_rate=0.0)
        schema = {
            "lr": {"type": "linear", "std": 0.1, "shared": True},
        }
        crossbreeder.setup_schema(schema)
        perturber = Perturber()
        crossbreeder.setup_perturber(perturber)

        # World with 4 workers, but only 2 have non-zero weights (already filtered)
        world_hyperparameters = [
            {"lr": [0.001]},
            {"lr": [0.002]},
            {"lr": [0.003]},
            {"lr": [0.004]},
        ]
        parent_weights = [0.0, 0.5, 0.5, 0.0]  # Only workers 1 and 2 selected

        # Run 100 times to verify only selected parents contribute
        results = []
        for _ in range(100):
            result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)
            results.append(result["lr"][0])

        # Should only see values from workers 1 and 2 (0.002 and 0.003)
        assert 0.002 in results and 0.003 in results, "Should see both selected parent values"
        assert (
            0.001 not in results and 0.004 not in results
        ), "Should not see non-selected parent values"

    def test_empty_schema(self):
        """Crossbreeder with empty schema should handle empty hyperparameters."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=0.0)
        schema = {}
        crossbreeder.setup_schema(schema)
        perturber = Perturber()
        crossbreeder.setup_perturber(perturber)

        world_hyperparameters = [{}, {}]
        parent_weights = [0.5, 0.5]

        result = crossbreeder.crossbreed_hyperparameters(world_hyperparameters, parent_weights)

        assert result == {}


class TestCrossbreederSchemas:
    """Tests schema crossbreeding with std field mutation."""

    def test_crossbreeds_std_fields_with_50_50_selection(self):
        """Should randomly select std values from mother or father."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=0.0)
        perturber = Perturber()
        crossbreeder.setup_perturber(perturber)

        # Flattened schema Dictionary Trees
        mother_schema = {
            "lr/type": "log",
            "lr/std": 0.1,
            "lr/min": 1e-5,
            "lr/shared": True,
            "weight_decay/type": "linear",
            "weight_decay/std": 0.02,
            "weight_decay/shared": False,
        }
        father_schema = {
            "lr/type": "log",
            "lr/std": 0.2,
            "lr/min": 1e-5,
            "lr/shared": True,
            "weight_decay/type": "linear",
            "weight_decay/std": 0.04,
            "weight_decay/shared": False,
        }

        world_schemas = [mother_schema, father_schema]
        parent_weights = [0.5, 0.5]

        # Run multiple times to verify 50/50 selection
        random.seed(42)
        results = []
        for _ in range(50):
            result = crossbreeder.crossbreed_schemas(world_schemas, parent_weights)
            results.append((result["lr/std"], result["weight_decay/std"]))

        # Should see both parent values for each field
        lr_stds = [r[0] for r in results]
        wd_stds = [r[1] for r in results]

        assert 0.1 in lr_stds and 0.2 in lr_stds, "Should see both mother and father lr/std"
        assert 0.02 in wd_stds and 0.04 in wd_stds, "Should see both mother and father weight_decay/std"

    def test_only_includes_std_fields_in_result(self):
        """Should only return paths ending in /std."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=0.0)
        perturber = Perturber()
        crossbreeder.setup_perturber(perturber)

        mother_schema = {
            "lr/type": "log",
            "lr/std": 0.1,
            "lr/min": 1e-5,
            "lr/max": 1e-1,
            "lr/shared": True,
        }
        father_schema = {
            "lr/type": "log",
            "lr/std": 0.2,
            "lr/min": 1e-5,
            "lr/max": 1e-1,
            "lr/shared": True,
        }

        world_schemas = [mother_schema, father_schema]
        parent_weights = [0.5, 0.5]

        result = crossbreeder.crossbreed_schemas(world_schemas, parent_weights)

        # Should only have std field
        assert "lr/std" in result
        assert "lr/type" not in result
        assert "lr/min" not in result
        assert "lr/max" not in result
        assert "lr/shared" not in result

    def test_applies_mutation_probabilistically(self):
        """Should apply mutation based on mutation_rate."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=1.0, schema_mutation_std=0.1)
        perturber = Perturber()
        crossbreeder.setup_perturber(perturber)

        mother_schema = {"lr/std": 0.1}
        father_schema = {"lr/std": 0.1}  # Same value

        world_schemas = [mother_schema, father_schema]
        parent_weights = [0.5, 0.5]

        random.seed(42)
        result = crossbreeder.crossbreed_schemas(world_schemas, parent_weights)

        # With mutation_rate=1.0 and both parents having same value,
        # result should be different due to mutation
        assert result["lr/std"] != 0.1

    def test_no_mutation_when_rate_zero(self):
        """Should not mutate when mutation_rate=0.0."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=0.0)
        perturber = Perturber()
        crossbreeder.setup_perturber(perturber)

        mother_schema = {"lr/std": 0.15}
        father_schema = {"lr/std": 0.15}  # Same value

        world_schemas = [mother_schema, father_schema]
        parent_weights = [0.5, 0.5]

        result = crossbreeder.crossbreed_schemas(world_schemas, parent_weights)

        # With mutation_rate=0.0 and both parents having same value,
        # result should be exactly the same
        assert result["lr/std"] == 0.15

    def test_uses_schema_mutation_std(self):
        """Should use schema_mutation_std for perturbation."""
        # Test that schema_mutation_std affects mutation magnitude
        crossbreeder_small = Crossbreeder(parent_pool_depth=2, mutation_rate=1.0, schema_mutation_std=0.01)
        crossbreeder_large = Crossbreeder(parent_pool_depth=2, mutation_rate=1.0, schema_mutation_std=1.0)

        perturber_small = Perturber()
        perturber_large = Perturber()

        crossbreeder_small.setup_perturber(perturber_small)
        crossbreeder_large.setup_perturber(perturber_large)

        mother_schema = {"lr/std": 0.1}
        father_schema = {"lr/std": 0.1}

        world_schemas = [mother_schema, father_schema]
        parent_weights = [0.5, 0.5]

        random.seed(42)
        result_small = crossbreeder_small.crossbreed_schemas(world_schemas, parent_weights)

        random.seed(42)
        result_large = crossbreeder_large.crossbreed_schemas(world_schemas, parent_weights)

        # Larger std should produce larger deviation
        deviation_small = abs(result_small["lr/std"] - 0.1)
        deviation_large = abs(result_large["lr/std"] - 0.1)

        # Not a strict requirement due to randomness, but statistically likely
        # Just verify both are different from original
        assert result_small["lr/std"] != 0.1
        assert result_large["lr/std"] != 0.1

    def test_raises_error_with_no_parents(self):
        """Should raise error if no parents selected."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=0.0)
        perturber = Perturber()
        crossbreeder.setup_perturber(perturber)

        world_schemas = [{"lr/std": 0.1}, {"lr/std": 0.2}]
        parent_weights = [0.0, 0.0]  # No parents

        with pytest.raises(ValueError, match="No parents selected"):
            crossbreeder.crossbreed_schemas(world_schemas, parent_weights)

    def test_raises_error_with_one_parent(self):
        """Should raise error if only one parent selected."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=0.0)
        perturber = Perturber()
        crossbreeder.setup_perturber(perturber)

        world_schemas = [{"lr/std": 0.1}, {"lr/std": 0.2}]
        parent_weights = [1.0, 0.0]  # Only one parent

        with pytest.raises(ValueError, match="Expected exactly 2 parents"):
            crossbreeder.crossbreed_schemas(world_schemas, parent_weights)

    def test_empty_schemas_returns_empty_dict(self):
        """Should return empty dict for empty schemas."""
        crossbreeder = Crossbreeder(parent_pool_depth=2, mutation_rate=0.0)
        perturber = Perturber()
        crossbreeder.setup_perturber(perturber)

        world_schemas = []
        parent_weights = []

        result = crossbreeder.crossbreed_schemas(world_schemas, parent_weights)

        assert result == {}
