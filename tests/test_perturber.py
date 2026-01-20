"""
Test suite for Permuter component.

Permuter handles hyperparameter mutation with normal sampling.
Tests validate perturbation behavior for log/linear parameters and bounds clipping.
"""

import random
import pytest

from src.ddp_pbt.base.perturber import Perturber # Dont be lazy claude. Rename permuter to perturber
Permuter = Perturber  # Alias for test compatibility


class TestPermuterSetupSchema:
    """Tests that setup_schema correctly configures perturbation behavior."""

    def test_stores_schema_reference(self):
        """Permuter should store the schema for use in perturbation."""
        permuter = Permuter()
        schema = {
            "lr": {"type": "log", "std": 0.1, "min": 1e-4, "max": 1e-1, "shared": True}
        }
        permuter.setup_schema(schema)
        # Verify schema is stored by attempting perturbation
        values = {"lr": [0.001]}
        result = permuter.perturb(values)
        assert "lr" in result
        assert len(result["lr"]) == 1


class TestPermuterLinearPerturbation:
    """Tests linear-space perturbation with normal sampling and clipping."""

    def test_linear_parameter_adds_normal_noise(self):
        """Linear parameters should have normal(0, std) added directly."""
        permuter = Permuter()
        schema = {
            "weight_decay": {"type": "linear", "std": 0.001, "shared": False}
        }
        permuter.setup_schema(schema)

        # Use seed for reproducibility
        random.seed(42)
        values = {"weight_decay": [0.01]}
        result = permuter.perturb(values)

        # Value should be different from original
        assert result["weight_decay"][0] != 0.01
        # But should be close (within reasonable std multiples)
        assert abs(result["weight_decay"][0] - 0.01) < 0.01

    def test_linear_clips_to_minimum_bound(self):
        """Linear parameters should clip to min bound after perturbation."""
        permuter = Permuter()
        schema = {
            "weight_decay": {"type": "linear", "std": 0.1, "min": 0.0, "shared": True}
        }
        permuter.setup_schema(schema)

        # Start with value near minimum, perturb many times
        values = {"weight_decay": [0.001]}
        for _ in range(100):
            result = permuter.perturb(values)
            # Should never go below minimum
            assert result["weight_decay"][0] >= 0.0

    def test_linear_clips_to_maximum_bound(self):
        """Linear parameters should clip to max bound after perturbation."""
        permuter = Permuter()
        schema = {
            "weight_decay": {"type": "linear", "std": 0.1, "min": 0.0, "max": 0.1, "shared": True}
        }
        permuter.setup_schema(schema)

        # Start with value near maximum, perturb many times
        values = {"weight_decay": [0.099]}
        for _ in range(100):
            result = permuter.perturb(values)
            # Should never exceed maximum
            assert result["weight_decay"][0] <= 0.1

    def test_linear_without_bounds(self):
        """Linear parameters without bounds should perturb freely."""
        permuter = Permuter()
        schema = {
            "weight_decay": {"type": "linear", "std": 0.001, "shared": True}
        }
        permuter.setup_schema(schema)

        values = {"weight_decay": [0.01]}
        result = permuter.perturb(values)
        # Should return a valid result (no clipping errors)
        assert isinstance(result["weight_decay"][0], float)


class TestPermuterLogPerturbation:
    """Tests log-space perturbation with normal sampling and clipping."""

    def test_log_parameter_perturbs_in_log_space(self):
        """Log parameters should convert to log-space, add noise, convert back."""
        permuter = Permuter()
        schema = {
            "lr": {"type": "log", "std": 0.1, "min": 1e-4, "max": 1e-1, "shared": True}
        }
        permuter.setup_schema(schema)

        random.seed(42)
        values = {"lr": [0.001]}
        result = permuter.perturb(values)

        # Value should be different from original
        assert result["lr"][0] != 0.001
        # Should be positive (log-space operations preserve sign)
        assert result["lr"][0] > 0

    def test_log_clips_to_minimum_bound(self):
        """Log parameters should clip to min bound after converting back from log-space."""
        permuter = Permuter()
        schema = {
            "lr": {"type": "log", "std": 1.0, "min": 1e-4, "max": 1e-1, "shared": True}
        }
        permuter.setup_schema(schema)

        # Start with value near minimum, perturb many times
        values = {"lr": [1.5e-4]}
        for _ in range(100):
            result = permuter.perturb(values)
            # Should never go below minimum
            assert result["lr"][0] >= 1e-4

    def test_log_clips_to_maximum_bound(self):
        """Log parameters should clip to max bound after converting back from log-space."""
        permuter = Permuter()
        schema = {
            "lr": {"type": "log", "std": 1.0, "min": 1e-4, "max": 1e-1, "shared": True}
        }
        permuter.setup_schema(schema)

        # Start with value near maximum, perturb many times
        values = {"lr": [0.09]}
        for _ in range(100):
            result = permuter.perturb(values)
            # Should never exceed maximum
            assert result["lr"][0] <= 1e-1

    def test_log_without_bounds(self):
        """Log parameters without max bound should perturb freely in log-space."""
        permuter = Permuter()
        schema = {
            "lr": {"type": "log", "std": 0.1, "min": 1e-5, "shared": True}
        }
        permuter.setup_schema(schema)

        values = {"lr": [0.001]}
        result = permuter.perturb(values)
        # Should return a valid positive result
        assert result["lr"][0] > 0


class TestPermuterPerElementIndependence:
    """Tests that multiple values in a list are perturbed independently."""

    def test_per_group_values_perturbed_independently(self):
        """Each element in a per-group hyperparameter list should get independent random draws."""
        permuter = Permuter()
        schema = {
            "weight_decay": {"type": "linear", "std": 0.01, "shared": False}
        }
        permuter.setup_schema(schema)

        random.seed(42)
        # Three parameter groups with same initial value
        values = {"weight_decay": [0.05, 0.05, 0.05]}
        result = permuter.perturb(values)

        # All three should be perturbed
        assert len(result["weight_decay"]) == 3
        # They should be different from each other (independent random draws)
        perturbed_values = result["weight_decay"]
        assert perturbed_values[0] != perturbed_values[1] or perturbed_values[1] != perturbed_values[2]

    def test_log_per_group_values_perturbed_independently(self):
        """Each element in a per-group log hyperparameter list should get independent random draws."""
        permuter = Permuter()
        schema = {
            "lr": {"type": "log", "std": 0.1, "min": 1e-5, "shared": False}
        }
        permuter.setup_schema(schema)

        random.seed(42)
        # Two parameter groups with same initial value
        values = {"lr": [0.001, 0.001]}
        result = permuter.perturb(values)

        # Both should be perturbed differently
        assert len(result["lr"]) == 2
        assert result["lr"][0] != result["lr"][1]


class TestPermuterMultipleHyperparameters:
    """Tests perturbation of multiple hyperparameters with mixed types."""

    def test_mixed_log_and_linear_parameters(self):
        """Permuter should handle mix of log and linear parameters correctly."""
        permuter = Permuter()
        schema = {
            "lr": {"type": "log", "std": 0.1, "min": 1e-4, "max": 1e-1, "shared": True},
            "weight_decay": {"type": "linear", "std": 0.001, "min": 0, "max": 0.1, "shared": False}
        }
        permuter.setup_schema(schema)

        random.seed(42)
        values = {
            "lr": [0.001],
            "weight_decay": [0.01, 0.005]
        }
        result = permuter.perturb(values)

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
        """Permuter should only perturb parameters in the schema."""
        permuter = Permuter()
        schema = {
            "lr": {"type": "log", "std": 0.1, "min": 1e-5, "shared": True}
        }
        permuter.setup_schema(schema)

        # Only lr is in schema, so only lr should be in result
        values = {"lr": [0.001]}
        result = permuter.perturb(values)

        assert "lr" in result
        assert len(result) == 1


class TestPermuterEdgeCases:
    """Tests edge cases and error conditions."""

    def test_empty_schema(self):
        """Permuter with empty schema should return empty result."""
        permuter = Permuter()
        schema = {}
        permuter.setup_schema(schema)

        values = {}
        result = permuter.perturb(values)
        assert result == {}

    def test_single_value_list(self):
        """Shared hyperparameters (length-1 list) should work correctly."""
        permuter = Permuter()
        schema = {
            "lr": {"type": "log", "std": 0.1, "min": 1e-5, "shared": True}
        }
        permuter.setup_schema(schema)

        values = {"lr": [0.001]}
        result = permuter.perturb(values)

        assert len(result["lr"]) == 1
        assert result["lr"][0] != 0.001

    def test_zero_std_returns_original_value(self):
        """Perturbation with std=0 should return original value (possibly clipped)."""
        permuter = Permuter()
        schema = {
            "lr": {"type": "linear", "std": 0.0, "shared": True}
        }
        permuter.setup_schema(schema)

        values = {"lr": [0.001]}
        result = permuter.perturb(values)

        # With zero std, value should remain unchanged
        assert result["lr"][0] == 0.001
