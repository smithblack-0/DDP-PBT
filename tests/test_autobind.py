"""
Test suite for Autobind component.

Black-box testing: Tests verify behavior through public interfaces only.
Configuration is tested by applying to mock strategies and verifying calls.
No access to internal state - all tests use __call__, save, or error detection.

Responsibilities tested:
- Load schema from default or custom JSON file
- Provide prototype pattern for partial schema updates
- Apply schema to strategies
- Save/load configurations
- Extract schema from strategies
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.ddp_pbt.autobind import Autobind, get_default_schema


@pytest.fixture
def mock_strategy():
    """Create a mock strategy for testing."""
    strategy = Mock()
    strategy.valid_binding_targets = ["lr", "weight_decay", "momentum"]
    strategy.state_dict.return_value = {
        "lr": {"type": "log", "std": 0.2, "min": 1e-5, "max": 0.1, "shared": False},
        "weight_decay": {"type": "log", "std": 0.15, "min": 1e-5, "max": 0.05, "shared": True},
    }
    return strategy


@pytest.fixture
def defaults_schema():
    """Load the default schema for verification."""
    return get_default_schema()


@pytest.fixture
def temp_schema_file():
    """Create a temporary schema file for testing."""
    schema = {
        "lr": {"type": "log", "std": 0.05, "min": 1e-5, "max": 0.01, "shared": True},
        "momentum": {"type": "linear", "std": 0.01, "min": 0.0, "max": 1.0, "shared": True},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(schema, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


class TestAutobindConstructor:
    """Test suite for Autobind constructor and file loading."""

    def test_applies_default_schema_when_no_file_path_provided(
        self, mock_strategy, defaults_schema
    ) -> None:
        """Constructor loads defaults and applies them correctly."""
        autobind = Autobind()
        result = autobind(mock_strategy, only=["lr"])

        # Should return the strategy object
        assert result is mock_strategy

        # Should apply lr from defaults
        mock_strategy.bind_log_hyperparameter.assert_called_once_with(
            "lr",
            std=defaults_schema["lr"]["std"],
            min=defaults_schema["lr"]["min"],
            max=defaults_schema["lr"]["max"],
            shared=defaults_schema["lr"]["shared"],
        )

    def test_applies_custom_schema_when_file_path_provided(
        self, mock_strategy, temp_schema_file
    ) -> None:
        """Constructor loads custom schema from provided file path."""
        autobind = Autobind(file_path=temp_schema_file)
        result = autobind(mock_strategy, only=["lr"])

        # Should return the strategy object
        assert result is mock_strategy

        # Should apply lr from custom schema (std=0.05, not default 0.1)
        mock_strategy.bind_log_hyperparameter.assert_called_once_with(
            "lr", std=0.05, min=1e-5, max=0.01, shared=True
        )

    def test_applies_defaults_when_custom_file_does_not_exist(
        self, mock_strategy, defaults_schema
    ) -> None:
        """Constructor falls back to defaults when custom file path doesn't exist."""
        autobind = Autobind(file_path="/nonexistent/path/to/schema.json")
        result = autobind(mock_strategy, only=["lr"])

        # Should return the strategy object
        assert result is mock_strategy

        # Should fall back to defaults
        mock_strategy.bind_log_hyperparameter.assert_called_once_with(
            "lr",
            std=defaults_schema["lr"]["std"],
            min=defaults_schema["lr"]["min"],
            max=defaults_schema["lr"]["max"],
            shared=defaults_schema["lr"]["shared"],
        )

    def test_uses_default_logging_callback_when_none_provided(self) -> None:
        """Constructor uses print as default logging callback."""
        autobind = Autobind()
        assert autobind.logging_callback == print

    def test_uses_custom_logging_callback_when_provided(self) -> None:
        """Constructor uses provided logging callback."""
        custom_callback = Mock()
        autobind = Autobind(logging_callback=custom_callback)
        assert autobind.logging_callback == custom_callback


class TestBindLogHyperparameter:
    """Test suite for bind_log_hyperparameter method."""

    def test_prototype_pattern_updates_only_std(
        self, mock_strategy, defaults_schema
    ) -> None:
        """Bind log hyperparameter updates only std, preserves other defaults."""
        autobind = Autobind()
        autobind.bind_log_hyperparameter("lr", std=0.9)
        result = autobind(mock_strategy, only=["lr"])
        assert result is mock_strategy

        # Should apply with modified std, other fields from defaults
        mock_strategy.bind_log_hyperparameter.assert_called_once_with(
            "lr",
            std=0.9,  # Modified
            min=defaults_schema["lr"]["min"],  # From defaults
            max=defaults_schema["lr"]["max"],  # From defaults
            shared=defaults_schema["lr"]["shared"],  # From defaults
        )

    def test_prototype_pattern_updates_only_min(
        self, mock_strategy, defaults_schema
    ) -> None:
        """Bind log hyperparameter updates only min, preserves other defaults."""
        autobind = Autobind()
        autobind.bind_log_hyperparameter("lr", min=1e-8)
        result = autobind(mock_strategy, only=["lr"])
        assert result is mock_strategy

        mock_strategy.bind_log_hyperparameter.assert_called_once_with(
            "lr",
            std=defaults_schema["lr"]["std"],  # From defaults
            min=1e-8,  # Modified
            max=defaults_schema["lr"]["max"],  # From defaults
            shared=defaults_schema["lr"]["shared"],  # From defaults
        )

    def test_prototype_pattern_updates_only_max(
        self, mock_strategy, defaults_schema
    ) -> None:
        """Bind log hyperparameter updates only max, preserves other defaults."""
        autobind = Autobind()
        autobind.bind_log_hyperparameter("lr", max=1.0)
        result = autobind(mock_strategy, only=["lr"])
        assert result is mock_strategy

        mock_strategy.bind_log_hyperparameter.assert_called_once_with(
            "lr",
            std=defaults_schema["lr"]["std"],  # From defaults
            min=defaults_schema["lr"]["min"],  # From defaults
            max=1.0,  # Modified
            shared=defaults_schema["lr"]["shared"],  # From defaults
        )

    def test_prototype_pattern_updates_only_shared(
        self, mock_strategy, defaults_schema
    ) -> None:
        """Bind log hyperparameter updates only shared, preserves other defaults."""
        autobind = Autobind()
        autobind.bind_log_hyperparameter("lr", shared=True)
        result = autobind(mock_strategy, only=["lr"])
        assert result is mock_strategy

        mock_strategy.bind_log_hyperparameter.assert_called_once_with(
            "lr",
            std=defaults_schema["lr"]["std"],  # From defaults
            min=defaults_schema["lr"]["min"],  # From defaults
            max=defaults_schema["lr"]["max"],  # From defaults
            shared=True,  # Modified
        )

    def test_prototype_pattern_updates_multiple_fields(self, mock_strategy) -> None:
        """Bind log hyperparameter updates multiple fields when provided."""
        autobind = Autobind()
        autobind.bind_log_hyperparameter("lr", std=0.25, min=1e-8, max=0.5, shared=False)
        result = autobind(mock_strategy, only=["lr"])
        assert result is mock_strategy

        mock_strategy.bind_log_hyperparameter.assert_called_once_with(
            "lr", std=0.25, min=1e-8, max=0.5, shared=False
        )

    def test_creates_new_entry_when_name_not_in_defaults(self, mock_strategy) -> None:
        """Bind log hyperparameter creates new entry when name doesn't exist in defaults."""
        autobind = Autobind()
        autobind.bind_log_hyperparameter("new_param", std=0.1, min=1e-4, max=10.0, shared=False)

        mock_strategy.valid_binding_targets.append("new_param")
        result = autobind(mock_strategy, only=["new_param"])
        assert result is mock_strategy

        mock_strategy.bind_log_hyperparameter.assert_called_once_with(
            "new_param", std=0.1, min=1e-4, max=10.0, shared=False
        )

    def test_requires_std_when_creating_new_entry(self) -> None:
        """Bind log hyperparameter requires std for new entries."""
        autobind = Autobind()

        # When std not provided for new param, should get error
        # (Could be TypeError at call site or validation error - implementation dependent)
        with pytest.raises((TypeError, ValueError)):
            autobind.bind_log_hyperparameter("nonexistent_param", min=1e-4)

    def test_requires_min_when_creating_new_log_entry(self) -> None:
        """Bind log hyperparameter requires min for new log entries."""
        autobind = Autobind()

        with pytest.raises(ValueError, match="min"):
            autobind.bind_log_hyperparameter("nonexistent_param", std=0.1)

    def test_defaults_shared_to_true_when_creating_new_entry(self, mock_strategy) -> None:
        """Bind log hyperparameter defaults shared to True for new entries."""
        autobind = Autobind()
        autobind.bind_log_hyperparameter("new_param", std=0.1, min=1e-4)

        mock_strategy.valid_binding_targets.append("new_param")
        result = autobind(mock_strategy, only=["new_param"])
        assert result is mock_strategy

        # Should have shared=True by default
        call_kwargs = mock_strategy.bind_log_hyperparameter.call_args[1]
        assert call_kwargs["shared"] is True

    def test_validates_std_positive(self) -> None:
        """Bind log hyperparameter validates std is positive."""
        autobind = Autobind()

        with pytest.raises(ValueError, match="std"):
            autobind.bind_log_hyperparameter("lr", std=-0.1)

        with pytest.raises(ValueError, match="std"):
            autobind.bind_log_hyperparameter("lr", std=0.0)

    def test_validates_min_positive_for_log_parameters(self) -> None:
        """Bind log hyperparameter validates min is positive for log parameters."""
        autobind = Autobind()

        with pytest.raises(ValueError, match="min"):
            autobind.bind_log_hyperparameter("lr", min=0.0)

        with pytest.raises(ValueError, match="min"):
            autobind.bind_log_hyperparameter("lr", min=-1e-4)

    def test_validates_max_greater_than_min(self) -> None:
        """Bind log hyperparameter validates max > min."""
        autobind = Autobind()

        with pytest.raises(ValueError, match="max"):
            autobind.bind_log_hyperparameter("lr", min=0.1, max=0.05)

        with pytest.raises(ValueError, match="max"):
            autobind.bind_log_hyperparameter("lr", min=0.1, max=0.1)

    def test_raises_type_error_when_binding_to_existing_linear_entry(self) -> None:
        """Bind log hyperparameter raises TypeError when entry exists with linear type."""
        autobind = Autobind()

        # momentum is linear in defaults
        with pytest.raises(TypeError, match="correct binding method"):
            autobind.bind_log_hyperparameter("momentum", std=0.1, min=1e-4)


class TestBindLinearHyperparameter:
    """Test suite for bind_linear_hyperparameter method."""

    def test_prototype_pattern_updates_only_std(
        self, mock_strategy, defaults_schema
    ) -> None:
        """Bind linear hyperparameter updates only std, preserves other defaults."""
        autobind = Autobind()
        autobind.bind_linear_hyperparameter("momentum", std=0.05)
        result = autobind(mock_strategy, only=["momentum"])
        assert result is mock_strategy

        mock_strategy.bind_linear_hyperparameter.assert_called_once_with(
            "momentum",
            std=0.05,  # Modified
            min=defaults_schema["momentum"]["min"],  # From defaults
            max=defaults_schema["momentum"]["max"],  # From defaults
            shared=defaults_schema["momentum"]["shared"],  # From defaults
        )

    def test_prototype_pattern_updates_only_min(
        self, mock_strategy, defaults_schema
    ) -> None:
        """Bind linear hyperparameter updates only min, preserves other defaults."""
        autobind = Autobind()
        autobind.bind_linear_hyperparameter("momentum", min=0.1)
        result = autobind(mock_strategy, only=["momentum"])
        assert result is mock_strategy

        mock_strategy.bind_linear_hyperparameter.assert_called_once_with(
            "momentum",
            std=defaults_schema["momentum"]["std"],  # From defaults
            min=0.1,  # Modified
            max=defaults_schema["momentum"]["max"],  # From defaults
            shared=defaults_schema["momentum"]["shared"],  # From defaults
        )

    def test_prototype_pattern_updates_only_max(
        self, mock_strategy, defaults_schema
    ) -> None:
        """Bind linear hyperparameter updates only max, preserves other defaults."""
        autobind = Autobind()
        autobind.bind_linear_hyperparameter("momentum", max=0.95)
        result = autobind(mock_strategy, only=["momentum"])
        assert result is mock_strategy

        mock_strategy.bind_linear_hyperparameter.assert_called_once_with(
            "momentum",
            std=defaults_schema["momentum"]["std"],  # From defaults
            min=defaults_schema["momentum"]["min"],  # From defaults
            max=0.95,  # Modified
            shared=defaults_schema["momentum"]["shared"],  # From defaults
        )

    def test_prototype_pattern_updates_multiple_fields(self, mock_strategy) -> None:
        """Bind linear hyperparameter updates multiple fields when provided."""
        autobind = Autobind()
        autobind.bind_linear_hyperparameter("momentum", std=0.1, min=0.5, max=0.99, shared=False)
        result = autobind(mock_strategy, only=["momentum"])
        assert result is mock_strategy

        mock_strategy.bind_linear_hyperparameter.assert_called_once_with(
            "momentum", std=0.1, min=0.5, max=0.99, shared=False
        )

    def test_creates_new_entry_when_name_not_in_defaults(self, mock_strategy) -> None:
        """Bind linear hyperparameter creates new entry when name doesn't exist in defaults."""
        autobind = Autobind()
        autobind.bind_linear_hyperparameter(
            "new_param", std=0.01, min=0.0, max=1.0, shared=False
        )

        mock_strategy.valid_binding_targets.append("new_param")
        result = autobind(mock_strategy, only=["new_param"])
        assert result is mock_strategy

        mock_strategy.bind_linear_hyperparameter.assert_called_once_with(
            "new_param", std=0.01, min=0.0, max=1.0, shared=False
        )

    def test_requires_std_when_creating_new_entry(self) -> None:
        """Bind linear hyperparameter requires std for new entries."""
        autobind = Autobind()

        with pytest.raises((TypeError, ValueError)):
            autobind.bind_linear_hyperparameter("nonexistent_param", min=0.0, max=1.0)

    def test_allows_optional_min_max_for_new_entries(self, mock_strategy) -> None:
        """Bind linear hyperparameter allows optional min/max for new entries."""
        autobind = Autobind()
        autobind.bind_linear_hyperparameter("new_param", std=0.01)

        mock_strategy.valid_binding_targets.append("new_param")
        result = autobind(mock_strategy, only=["new_param"])
        assert result is mock_strategy

        # Should be called with std, without min/max in kwargs
        call_kwargs = mock_strategy.bind_linear_hyperparameter.call_args[1]
        assert call_kwargs["std"] == 0.01
        # min/max might be None or absent - either is fine for optional

    def test_defaults_shared_to_true_when_creating_new_entry(self, mock_strategy) -> None:
        """Bind linear hyperparameter defaults shared to True for new entries."""
        autobind = Autobind()
        autobind.bind_linear_hyperparameter("new_param", std=0.01)

        mock_strategy.valid_binding_targets.append("new_param")
        result = autobind(mock_strategy, only=["new_param"])
        assert result is mock_strategy

        call_kwargs = mock_strategy.bind_linear_hyperparameter.call_args[1]
        assert call_kwargs["shared"] is True

    def test_validates_std_positive(self) -> None:
        """Bind linear hyperparameter validates std is positive."""
        autobind = Autobind()

        with pytest.raises(ValueError, match="std"):
            autobind.bind_linear_hyperparameter("momentum", std=-0.1)

        with pytest.raises(ValueError, match="std"):
            autobind.bind_linear_hyperparameter("momentum", std=0.0)

    def test_validates_min_less_than_max(self) -> None:
        """Bind linear hyperparameter validates min < max when both provided."""
        autobind = Autobind()

        with pytest.raises(ValueError, match="min"):
            autobind.bind_linear_hyperparameter("momentum", min=0.9, max=0.8)

        with pytest.raises(ValueError, match="min"):
            autobind.bind_linear_hyperparameter("momentum", min=0.9, max=0.9)

    def test_raises_type_error_when_binding_to_existing_log_entry(self) -> None:
        """Bind linear hyperparameter raises TypeError when entry exists with log type."""
        autobind = Autobind()

        # lr is log in defaults
        with pytest.raises(TypeError, match="correct binding method"):
            autobind.bind_linear_hyperparameter("lr", std=0.01)


class TestAutobindCall:
    """Test suite for __call__ application method."""

    def test_applies_hyperparameters_in_both_schema_and_valid_targets(
        self, mock_strategy
    ) -> None:
        """Call applies only hyperparameters in both schema and valid_binding_targets."""
        autobind = Autobind()
        autobind.bind_log_hyperparameter("lr", std=0.1, min=1e-6, max=0.1)
        autobind.bind_log_hyperparameter("weight_decay", std=0.1, min=1e-6, max=0.1)
        autobind.bind_linear_hyperparameter("nonexistent", std=0.01, min=0.0, max=1.0)

        result = autobind(mock_strategy)
        assert result is mock_strategy

        # Should apply: lr (modified), weight_decay (modified), momentum (default)
        # Should NOT apply: nonexistent (not in valid_binding_targets)
        assert mock_strategy.bind_log_hyperparameter.call_count == 2  # lr, weight_decay
        assert mock_strategy.bind_linear_hyperparameter.call_count == 1  # momentum (default)

    def test_applies_log_hyperparameters_correctly(self, mock_strategy) -> None:
        """Call applies log hyperparameters using bind_log_hyperparameter."""
        autobind = Autobind()
        autobind.bind_log_hyperparameter("lr", std=0.2, min=1e-5, max=0.05, shared=True)
        result = autobind(mock_strategy, only=["lr"])
        assert result is mock_strategy

        mock_strategy.bind_log_hyperparameter.assert_called_once_with(
            "lr", std=0.2, min=1e-5, max=0.05, shared=True
        )

    def test_applies_linear_hyperparameters_correctly(self, mock_strategy) -> None:
        """Call applies linear hyperparameters using bind_linear_hyperparameter."""
        autobind = Autobind()
        autobind.bind_linear_hyperparameter("momentum", std=0.05, min=0.0, max=1.0, shared=False)
        result = autobind(mock_strategy, only=["momentum"])
        assert result is mock_strategy

        mock_strategy.bind_linear_hyperparameter.assert_called_once_with(
            "momentum", std=0.05, min=0.0, max=1.0, shared=False
        )

    def test_applies_hyperparameters_without_optional_bounds(self, mock_strategy) -> None:
        """Call applies hyperparameters correctly when optional bounds not specified."""
        autobind = Autobind()
        autobind.bind_log_hyperparameter("lr", std=0.1, min=1e-6)  # No max
        result = autobind(mock_strategy, only=["lr"])
        assert result is mock_strategy

        # Should be called, max handling is implementation detail
        assert mock_strategy.bind_log_hyperparameter.call_count == 1

    def test_applies_only_subset_when_only_parameter_provided(self, mock_strategy) -> None:
        """Call applies only specified subset when only parameter provided."""
        autobind = Autobind()
        autobind.bind_log_hyperparameter("lr", std=0.1, min=1e-6, max=0.1)
        autobind.bind_log_hyperparameter("weight_decay", std=0.1, min=1e-6, max=0.1)

        result = autobind(mock_strategy, only=["lr"])
        assert result is mock_strategy

        # Should only apply lr
        mock_strategy.bind_log_hyperparameter.assert_called_once()
        call_args = mock_strategy.bind_log_hyperparameter.call_args[0]
        assert call_args[0] == "lr"

    def test_invokes_logging_callback_for_each_applied_hyperparameter(
        self, mock_strategy
    ) -> None:
        """Call invokes logging callback for each applied hyperparameter."""
        callback = Mock()
        autobind = Autobind(logging_callback=callback)
        autobind.bind_log_hyperparameter("lr", std=0.1, min=1e-6, max=0.1)
        autobind.bind_log_hyperparameter("weight_decay", std=0.1, min=1e-6, max=0.1)

        result = autobind(mock_strategy)
        assert result is mock_strategy

        # Should call logging callback for all applied: lr, weight_decay, momentum (default)
        assert callback.call_count == 3

    def test_overwrites_existing_bindings_on_strategy(self, mock_strategy) -> None:
        """Call overwrites any existing bindings on strategy."""
        autobind = Autobind()
        autobind.bind_log_hyperparameter("lr", std=0.5, min=1e-8, max=1.0)

        # Apply twice
        result1 = autobind(mock_strategy, only=["lr"])
        result2 = autobind(mock_strategy, only=["lr"])
        assert result1 is mock_strategy
        assert result2 is mock_strategy

        # Should be called twice with same args (overwrites on strategy each time)
        assert mock_strategy.bind_log_hyperparameter.call_count == 2

    def test_applies_mixed_log_and_linear_hyperparameters_correctly(
        self, mock_strategy
    ) -> None:
        """Call correctly routes to bind_log vs bind_linear based on type."""
        autobind = Autobind()
        autobind.bind_log_hyperparameter("lr", std=0.1, min=1e-6, max=0.1)
        autobind.bind_linear_hyperparameter("momentum", std=0.05, min=0.0, max=1.0)

        result = autobind(mock_strategy)
        assert result is mock_strategy

        # Should call bind_log for: lr (modified), weight_decay (default)
        # Should call bind_linear for: momentum (modified)
        assert mock_strategy.bind_log_hyperparameter.call_count == 2
        assert mock_strategy.bind_linear_hyperparameter.call_count == 1


class TestAutobindSave:
    """Test suite for save method."""

    def test_saves_schema_to_json_file(self, defaults_schema) -> None:
        """Save writes schema to JSON file in correct format."""
        autobind = Autobind()
        autobind.bind_log_hyperparameter("lr", std=0.9)
        autobind.bind_linear_hyperparameter("momentum", std=0.05)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            autobind.save(temp_path)

            # Read back and verify
            with open(temp_path, "r") as f:
                loaded_schema = json.load(f)

            # Should have modified values
            assert loaded_schema["lr"]["std"] == 0.9
            assert loaded_schema["momentum"]["std"] == 0.05
            # Should preserve other fields from defaults
            assert loaded_schema["lr"]["min"] == defaults_schema["lr"]["min"]
            assert loaded_schema["momentum"]["max"] == defaults_schema["momentum"]["max"]

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_saved_file_is_valid_json(self) -> None:
        """Save produces valid JSON file."""
        autobind = Autobind()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            autobind.save(temp_path)

            # Should be able to parse as JSON without error
            with open(temp_path, "r") as f:
                json.load(f)

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_saved_schema_can_be_loaded_by_new_autobind(self) -> None:
        """Save produces file that can be loaded by another Autobind."""
        autobind1 = Autobind()
        autobind1.bind_log_hyperparameter("lr", std=0.777)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            autobind1.save(temp_path)

            # Load in new Autobind and verify behavior
            autobind2 = Autobind(file_path=temp_path)
            mock_strategy = Mock()
            mock_strategy.valid_binding_targets = ["lr"]
            result = autobind2(mock_strategy, only=["lr"])
            assert result is mock_strategy

            # Should apply with saved std
            call_kwargs = mock_strategy.bind_log_hyperparameter.call_args[1]
            assert call_kwargs["std"] == 0.777

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestAutobindLoadFromStrategy:
    """Test suite for load_from_strategy method."""

    def test_extracts_schema_from_strategy(self, mock_strategy) -> None:
        """Load from strategy extracts schema via strategy.state_dict."""
        autobind = Autobind()

        result = autobind.load_from_strategy(mock_strategy)

        # Should return self for fluent interface
        assert result is autobind

        # Should have called state_dict to extract
        mock_strategy.state_dict.assert_called_once()

        # Apply to verify extraction worked
        new_mock = Mock()
        new_mock.valid_binding_targets = ["lr"]
        result = autobind(new_mock, only=["lr"])
        assert result is new_mock

        # Should apply lr from mock_strategy's state_dict
        new_mock.bind_log_hyperparameter.assert_called_once_with(
            "lr", std=0.2, min=1e-5, max=0.1, shared=False
        )

    def test_merges_with_current_autobind_schema(self, mock_strategy, defaults_schema) -> None:
        """Load from strategy merges strategy schema into current autobind schema."""
        autobind = Autobind()  # Has defaults including momentum

        # mock_strategy has lr and weight_decay, NOT momentum
        result = autobind.load_from_strategy(mock_strategy)

        # Should return self
        assert result is autobind

        # Apply to verify merge
        new_mock = Mock()
        new_mock.valid_binding_targets = ["lr", "momentum"]
        result = autobind(new_mock)
        assert result is new_mock

        # Should have lr from strategy AND momentum from original autobind
        assert new_mock.bind_log_hyperparameter.call_count == 1  # lr
        assert new_mock.bind_linear_hyperparameter.call_count == 1  # momentum

    def test_strategy_schema_overwrites_matching_keys(self, mock_strategy, defaults_schema) -> None:
        """Load from strategy overwrites matching keys with strategy values."""
        autobind = Autobind()  # Has default lr with std=0.1

        # mock_strategy has lr with std=0.2
        result = autobind.load_from_strategy(mock_strategy)

        # Should return self
        assert result is autobind

        new_mock = Mock()
        new_mock.valid_binding_targets = ["lr"]
        result = autobind(new_mock, only=["lr"])
        assert result is new_mock

        # Should use strategy's std=0.2, not default std=0.1
        call_kwargs = new_mock.bind_log_hyperparameter.call_args[1]
        assert call_kwargs["std"] == 0.2  # From mock_strategy

    def test_keys_only_in_autobind_remain_unchanged(
        self, mock_strategy, defaults_schema
    ) -> None:
        """Load from strategy keeps keys only in current autobind unchanged."""
        autobind = Autobind()  # Has momentum

        # mock_strategy does NOT have momentum
        result = autobind.load_from_strategy(mock_strategy)

        # Should return self
        assert result is autobind

        new_mock = Mock()
        new_mock.valid_binding_targets = ["momentum"]
        result = autobind(new_mock, only=["momentum"])
        assert result is new_mock

        # Should apply momentum from original autobind with original values
        new_mock.bind_linear_hyperparameter.assert_called_once_with(
            "momentum",
            std=defaults_schema["momentum"]["std"],
            min=defaults_schema["momentum"]["min"],
            max=defaults_schema["momentum"]["max"],
            shared=defaults_schema["momentum"]["shared"],
        )

    def test_returns_self_for_fluent_interface(self, mock_strategy) -> None:
        """Load from strategy returns self for fluent interface."""
        autobind = Autobind()

        result = autobind.load_from_strategy(mock_strategy)

        assert result is autobind

    def test_modifications_persist_in_same_instance(self, mock_strategy) -> None:
        """Load from strategy mutates in place, subsequent modifications persist."""
        autobind = Autobind()
        result = autobind.load_from_strategy(mock_strategy)

        # Modify after loading
        autobind.bind_log_hyperparameter("lr", std=999.0)

        # Should apply with modified value
        mock1 = Mock()
        mock1.valid_binding_targets = ["lr"]
        result = autobind(mock1, only=["lr"])
        assert result is mock1

        call_kwargs = mock1.bind_log_hyperparameter.call_args[1]
        assert call_kwargs["std"] == 999.0  # Should have modified value
