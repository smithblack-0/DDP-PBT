"""
Test suite for Perturber component.

Perturber handles hyperparameter mutation with normal sampling.
Tests validate perturbation behavior for log/linear parameters and bounds clipping.
"""

import random

import pytest

from src.ddp_pbt.base.perturber import Perturber

# Deleted illegal alias claude. Go rename everything to perturber instead.
# Stop being lazy. Using random aliases like that is a code smell. No as. No alias. Nope.


class TestPerturberSetupSchema:
    """Tests that setup_schema correctly configures perturbation behavior."""

    def test_stores_schema_reference(self):
        """Perturber should store the schema for use in perturbation."""
        perturber = Perturber()
        schema = {
            "lr": {"type": "log", "std": 0.1, "min": 1e-4, "max": 1e-1, "shared": True},
        }
        perturber.setup_schema(schema)
        # Verify schema is stored by attempting perturbation
        values = {"lr": [0.001]}
        result = perturber.perturb_pytree_using_schema(values)
        assert "lr" in result
        assert len(result["lr"]) == 1


class TestPerturberLinearPerturbation:
    """Tests linear-space perturbation with normal sampling and clipping."""

    def test_linear_parameter_adds_normal_noise(self):
        """Linear parameters should have normal(0, std) added directly."""
        perturber = Perturber()
        schema = {
            "weight_decay": {"type": "linear", "std": 0.001, "shared": False},
        }
        perturber.setup_schema(schema)

        # Use seed for reproducibility
        random.seed(42)
        values = {"weight_decay": [0.01]}
        result = perturber.perturb_pytree_using_schema(values)

        # Value should be different from original
        assert result["weight_decay"][0] != 0.01
        # But should be close (within reasonable std multiples)
        assert abs(result["weight_decay"][0] - 0.01) < 0.01

    def test_linear_clips_to_minimum_bound(self):
        """Linear parameters should clip to min bound after perturbation."""
        perturber = Perturber()
        schema = {
            "weight_decay": {"type": "linear", "std": 0.1, "min": 0.0, "shared": True},
        }
        perturber.setup_schema(schema)

        # Start with value near minimum, perturb many times
        values = {"weight_decay": [0.001]}
        for _ in range(100):
            result = perturber.perturb_pytree_using_schema(values)
            # Should never go below minimum
            assert result["weight_decay"][0] >= 0.0

    def test_linear_clips_to_maximum_bound(self):
        """Linear parameters should clip to max bound after perturbation."""
        perturber = Perturber()
        schema = {
            "weight_decay": {"type": "linear", "std": 0.1, "min": 0.0, "max": 0.1, "shared": True},
        }
        perturber.setup_schema(schema)

        # Start with value near maximum, perturb many times
        values = {"weight_decay": [0.099]}
        for _ in range(100):
            result = perturber.perturb_pytree_using_schema(values)
            # Should never exceed maximum
            assert result["weight_decay"][0] <= 0.1

    def test_linear_without_bounds(self):
        """Linear parameters without bounds should perturb freely."""
        perturber = Perturber()
        schema = {
            "weight_decay": {"type": "linear", "std": 0.001, "shared": True},
        }
        perturber.setup_schema(schema)

        values = {"weight_decay": [0.01]}
        result = perturber.perturb_pytree_using_schema(values)
        # Should return a valid result (no clipping errors)
        assert isinstance(result["weight_decay"][0], float)


class TestPerturberLogPerturbation:
    """Tests log-space perturbation with normal sampling and clipping."""

    def test_log_parameter_perturbs_in_log_space(self):
        """Log parameters should convert to log-space, add noise, convert back."""
        perturber = Perturber()
        schema = {
            "lr": {"type": "log", "std": 0.1, "min": 1e-4, "max": 1e-1, "shared": True},
        }
        perturber.setup_schema(schema)

        random.seed(42)
        values = {"lr": [0.001]}
        result = perturber.perturb_pytree_using_schema(values)

        # Value should be different from original
        assert result["lr"][0] != 0.001
        # Should be positive (log-space operations preserve sign)
        assert result["lr"][0] > 0

    def test_log_clips_to_minimum_bound(self):
        """Log parameters should clip to min bound after converting back from log-space."""
        perturber = Perturber()
        schema = {
            "lr": {"type": "log", "std": 1.0, "min": 1e-4, "max": 1e-1, "shared": True},
        }
        perturber.setup_schema(schema)

        # Start with value near minimum, perturb many times
        values = {"lr": [1.5e-4]}
        for _ in range(100):
            result = perturber.perturb_pytree_using_schema(values)
            # Should never go below minimum
            assert result["lr"][0] >= 1e-4

    def test_log_clips_to_maximum_bound(self):
        """Log parameters should clip to max bound after converting back from log-space."""
        perturber = Perturber()
        schema = {
            "lr": {"type": "log", "std": 1.0, "min": 1e-4, "max": 1e-1, "shared": True},
        }
        perturber.setup_schema(schema)

        # Start with value near maximum, perturb many times
        values = {"lr": [0.09]}
        for _ in range(100):
            result = perturber.perturb_pytree_using_schema(values)
            # Should never exceed maximum (allowing for floating point precision)
            assert result["lr"][0] <= 1e-1 + 1e-9

    def test_log_without_bounds(self):
        """Log parameters without max bound should perturb freely in log-space."""
        perturber = Perturber()
        schema = {
            "lr": {"type": "log", "std": 0.1, "min": 1e-5, "shared": True},
        }
        perturber.setup_schema(schema)

        values = {"lr": [0.001]}
        result = perturber.perturb_pytree_using_schema(values)
        # Should return a valid positive result
        assert result["lr"][0] > 0


class TestPerturberPerElementIndependence:
    """Tests that multiple values in a list are perturbed independently."""

    def test_per_group_values_perturbed_independently(self):
        """Each element in a per-group hyperparameter list should get independent random draws."""
        perturber = Perturber()
        schema = {
            "weight_decay": {"type": "linear", "std": 0.01, "shared": False},
        }
        perturber.setup_schema(schema)

        random.seed(42)
        # Three parameter groups with same initial value
        values = {"weight_decay": [0.05, 0.05, 0.05]}
        result = perturber.perturb_pytree_using_schema(values)

        # All three should be perturbed
        assert len(result["weight_decay"]) == 3
        # They should be different from each other (independent random draws)
        perturbed_values = result["weight_decay"]
        assert (
            perturbed_values[0] != perturbed_values[1] or perturbed_values[1] != perturbed_values[2]
        )

    def test_log_per_group_values_perturbed_independently(self):
        """Each element in a per-group log hyperparameter
        list should get independent random draws."""
        perturber = Perturber()
        schema = {
            "lr": {"type": "log", "std": 0.1, "min": 1e-5, "shared": False},
        }
        perturber.setup_schema(schema)

        random.seed(42)
        # Two parameter groups with same initial value
        values = {"lr": [0.001, 0.001]}
        result = perturber.perturb_pytree_using_schema(values)

        # Both should be perturbed differently
        assert len(result["lr"]) == 2
        assert result["lr"][0] != result["lr"][1]


class TestPerturberMultipleHyperparameters:
    """Tests perturbation of multiple hyperparameters with mixed types."""

    def test_mixed_log_and_linear_parameters(self):
        """Perturber should handle mix of log and linear parameters correctly."""
        perturber = Perturber()
        schema = {
            "lr": {"type": "log", "std": 0.1, "min": 1e-4, "max": 1e-1, "shared": True},
            "weight_decay": {"type": "linear", "std": 0.001, "min": 0, "max": 0.1, "shared": False},
        }
        perturber.setup_schema(schema)

        random.seed(42)
        values = {
            "lr": [0.001],
            "weight_decay": [0.01, 0.005],
        }
        result = perturber.perturb_pytree_using_schema(values)

        # Both hyperparameters should be present
        assert "lr" in result
        assert "weight_decay" in result

        # Structure should be preserved
        assert len(result["lr"]) == 1
        assert len(result["weight_decay"]) == 2

        # Values should be perturbed
        assert result["lr"][0] != 0.001
        assert result["weight_decay"][0] != 0.01 or result["weight_decay"][1] != 0.005

        # Bounds should be respected
        assert 1e-4 <= result["lr"][0] <= 1e-1
        assert 0 <= result["weight_decay"][0] <= 0.1
        assert 0 <= result["weight_decay"][1] <= 0.1

    def test_partial_schema_only_perturbs_configured_parameters(self):
        """Perturber should only perturb parameters in the schema."""
        perturber = Perturber()
        schema = {
            "lr": {"type": "log", "std": 0.1, "min": 1e-5, "shared": True},
        }
        perturber.setup_schema(schema)

        # Only lr is in schema, so only lr should be in result
        values = {"lr": [0.001]}
        result = perturber.perturb_pytree_using_schema(values)

        assert "lr" in result
        assert len(result) == 1


class TestPerturberEdgeCases:
    """Tests edge cases and error conditions."""

    def test_empty_schema(self):
        """Perturber with empty schema should return empty result."""
        perturber = Perturber()
        schema = {}
        perturber.setup_schema(schema)

        values = {}
        result = perturber.perturb_pytree_using_schema(values)
        assert result == {}

    def test_single_value_list(self):
        """Shared hyperparameters (length-1 list) should work correctly."""
        perturber = Perturber()
        schema = {
            "lr": {"type": "log", "std": 0.1, "min": 1e-5, "shared": True},
        }
        perturber.setup_schema(schema)

        values = {"lr": [0.001]}
        result = perturber.perturb_pytree_using_schema(values)

        assert len(result["lr"]) == 1
        assert result["lr"][0] != 0.001

    def test_zero_std_returns_original_value(self):
        """Perturbation with std=0 should return original value (possibly clipped)."""
        perturber = Perturber()
        schema = {
            "lr": {"type": "linear", "std": 0.0, "shared": True},
        }
        perturber.setup_schema(schema)

        values = {"lr": [0.001]}
        result = perturber.perturb_pytree_using_schema(values)

        # With zero std, value should remain unchanged
        assert result["lr"][0] == 0.001


class TestPerturberPerturbChance:
    """Tests probabilistic perturbation with perturb_chance parameter."""

    def test_perturb_chance_zero_returns_unchanged(self):
        """With perturb_chance=0.0, values should remain unchanged."""
        perturber = Perturber()
        schema = {
            "lr": {"type": "log", "std": 1.0, "min": 1e-5, "shared": True},
        }
        perturber.setup_schema(schema)

        values = {"lr": [0.001]}
        result = perturber.perturb_pytree_using_schema(values, perturb_chance=0.0)

        # Value should be unchanged
        assert result["lr"][0] == 0.001

    def test_perturb_chance_one_always_perturbs(self):
        """With perturb_chance=1.0, values should always be perturbed."""
        perturber = Perturber()
        schema = {
            "lr": {"type": "log", "std": 0.1, "min": 1e-5, "shared": True},
        }
        perturber.setup_schema(schema)

        random.seed(42)
        values = {"lr": [0.001]}
        result = perturber.perturb_pytree_using_schema(values, perturb_chance=1.0)

        # Value should be perturbed
        assert result["lr"][0] != 0.001

    def test_perturb_chance_affects_each_element_independently(self):
        """perturb_chance should be applied independently to each list element."""
        perturber = Perturber()
        schema = {
            "lr": {"type": "linear", "std": 0.1, "shared": False},
        }
        perturber.setup_schema(schema)

        # With perturb_chance=0.5 and many elements, some should change and some shouldn't
        random.seed(42)
        values = {"lr": [0.001] * 20}
        result = perturber.perturb_pytree_using_schema(values, perturb_chance=0.5)

        # Not all should be unchanged (statistically very unlikely)
        unchanged_count = sum(1 for i in range(20) if result["lr"][i] == 0.001)
        # With 20 elements at 50% chance, we expect around 10 unchanged
        # Allow range of 3-17 for statistical variation
        assert 3 <= unchanged_count <= 17


class TestPerturberDictTreeDefaults:
    """Tests perturb_dict_tree_using_defaults with predicate filtering."""

    def test_perturbs_filtered_paths_only(self):
        """Should only perturb paths matching the predicate."""
        perturber = Perturber()

        dict_tree = {
            "lr/std": 0.1,
            "lr/min": 1e-5,
            "weight_decay/std": 0.01,
            "weight_decay/min": 0.0,
        }

        def is_std_field(path, value):
            return path.endswith("/std")

        random.seed(42)
        result = perturber.perturb_dict_tree_using_defaults(
            dict_tree,
            std=0.1,
            param_type="log",
            predicate=is_std_field,
            perturb_chance=1.0,
            min_bound=1e-10,
        )

        # std fields should be perturbed
        assert result["lr/std"] != 0.1
        assert result["weight_decay/std"] != 0.01

        # min fields should be unchanged
        assert result["lr/min"] == 1e-5
        assert result["weight_decay/min"] == 0.0

    def test_respects_perturb_chance(self):
        """perturb_dict_tree_using_defaults should respect perturb_chance parameter."""
        perturber = Perturber()

        dict_tree = {"value": 0.1}

        def always_true(path, value):
            return True

        result = perturber.perturb_dict_tree_using_defaults(
            dict_tree,
            std=0.1,
            param_type="linear",
            predicate=always_true,
            perturb_chance=0.0,
        )

        # Value should be unchanged
        assert result["value"] == 0.1

    def test_uses_log_param_type(self):
        """Should correctly apply log-space perturbation."""
        perturber = Perturber()

        dict_tree = {"value": 0.001}

        def always_true(path, value):
            return True

        random.seed(42)
        result = perturber.perturb_dict_tree_using_defaults(
            dict_tree,
            std=0.1,
            param_type="log",
            predicate=always_true,
            perturb_chance=1.0,
            min_bound=1e-5,
        )

        # Value should be perturbed and positive
        assert result["value"] != 0.001
        assert result["value"] > 0
        assert result["value"] >= 1e-5

    def test_uses_linear_param_type(self):
        """Should correctly apply linear-space perturbation."""
        perturber = Perturber()

        dict_tree = {"value": 0.5}

        def always_true(path, value):
            return True

        random.seed(42)
        result = perturber.perturb_dict_tree_using_defaults(
            dict_tree,
            std=0.1,
            param_type="linear",
            predicate=always_true,
            perturb_chance=1.0,
            min_bound=0.0,
            max_bound=1.0,
        )

        # Value should be perturbed and within bounds
        assert result["value"] != 0.5
        assert 0.0 <= result["value"] <= 1.0

    def test_returns_new_dict_tree(self):
        """Should return new Dictionary Tree without modifying input."""
        perturber = Perturber()

        original = {"value": 0.1}

        def always_true(path, value):
            return True

        random.seed(42)
        result = perturber.perturb_dict_tree_using_defaults(
            original,
            std=0.1,
            param_type="linear",
            predicate=always_true,
            perturb_chance=1.0,
        )

        # Original should be unchanged
        assert original["value"] == 0.1
        # Result should be different
        assert result["value"] != 0.1


class TestPerturberApplyPerturbationDefaults:
    """Tests apply_perturbation with default fallback parameters."""

    def test_uses_schema_when_path_exists(self):
        """Should use schema configuration when path exists in schema."""
        perturber = Perturber()
        schema = {
            "lr": {"type": "log", "std": 0.1, "min": 1e-5, "shared": True},
        }
        perturber.setup_schema(schema)

        random.seed(42)
        # Defaults should be ignored when path is in schema
        result = perturber.apply_perturbation(
            "lr",
            0.001,
            default_std=1.0,
            default_param_type="linear",
        )

        # Should be perturbed (not unchanged)
        assert result != 0.001
        # Should be positive (log-space from schema, not linear from defaults)
        assert result > 0

    def test_uses_defaults_when_path_missing(self):
        """Should use default parameters when path not in schema."""
        perturber = Perturber()
        schema = {}
        perturber.setup_schema(schema)

        random.seed(42)
        result = perturber.apply_perturbation(
            "unknown_param",
            0.5,
            default_std=0.1,
            default_param_type="linear",
        )

        # Should be perturbed using defaults
        assert result != 0.5

    def test_returns_unchanged_when_no_schema_no_defaults(self):
        """Should return unchanged when path not in schema and no defaults provided."""
        perturber = Perturber()
        schema = {}
        perturber.setup_schema(schema)

        result = perturber.apply_perturbation("unknown_param", 0.5)

        # Should be unchanged
        assert result == 0.5

    def test_requires_both_std_and_param_type(self):
        """Should raise error if only one of default_std/default_param_type provided."""
        perturber = Perturber()
        schema = {}
        perturber.setup_schema(schema)

        with pytest.raises(ValueError, match="must both be provided"):
            perturber.apply_perturbation(
                "param",
                0.5,
                default_std=0.1,
                # Missing default_param_type
            )

        with pytest.raises(ValueError, match="must both be provided"):
            perturber.apply_perturbation(
                "param",
                0.5,
                # Missing default_std
                default_param_type="linear",
            )
