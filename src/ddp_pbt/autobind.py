"""
Autobind provides user-facing configuration interface with prototype pattern.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base.abstract_strategy import (
    validate_linear_schema_entry,
    validate_log_schema_entry,
    validate_schema,
    AbstractStrategy
)


def get_default_schema() -> Dict[str, Dict[str, Any]]:
    """
    Load and return the default schema from autobind_defaults.json.

    :return: Default schema dictionary
    """
    defaults_path = Path(__file__).parent / "autobind_defaults.json"
    with open(defaults_path, "r") as f:
        schema = json.load(f)
    validate_schema(schema)
    return schema


class Autobind:
    """
    User-facing configuration interface for DDP-PBT strategies.
    The autobind subsystem is intended to be the primary
    user interaction surface for inserting good defaults into
    strategies, supporting many common optimizer types.

    Internally, a "default_schema" is held which is either
    loaded from a file, or loaded from defaults. This schema,
    when invoked with a strategy, will apply its values to
    the strategy for every entry the optimizer supports as
    available.

    ```
    from ddp_pbt import Autobind, make_top_score_strategy

    optimizer = torch.optim.AdamW(lr=1e-4, weight_decay = 1e-2)
    autobind = Autobind()
    strategy =make_top_score_strategy(optimizer)
    strategy = autobind(strategy) # Setup for AdamW now
    ```

    When it is the case the defaults are outside the acceptable
    range, or need to be modified, a prototyping strategy can
    be used

        ```
    from ddp_pbt import Autobind, make_top_score_strategy

    optimizer = torch.optim.AdamW(lr=1e-8, weight_decay = 1e-2) # Lr too low, will be clipped
    autobind = Autobind()
    autobind.bind_log_hyperparameter("lr", min=1e-9)
    strategy =make_top_score_strategy(optimizer)
    strategy = autobind(strategy) # Setup for AdamW now
    ```

    Methods Summary:
    - Load custom defaults → __init__(file_path=...)
    - Tweak log hyperparameter bounds → bind_log_hyperparameter(name, ...)
    - Tweak linear hyperparameter bounds → bind_linear_hyperparameter(name, ...)
    - Apply configuration to strategy → __call__(strategy)
    - Save configuration for reuse → save(file_path)
    - Extract configuration from trained strategy → load_from_strategy(strategy)

    """

    def __init__(
        self,
        file_path: Optional[str] = None,
        logging_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize Autobind with schema from file.

        :param file_path: Path to custom schema JSON file (defaults to autobind_defaults.json)
        :param logging_callback: Optional function for logging applied bindings
        """
        self.logging_callback = logging_callback if logging_callback else print

        # Load schema
        if file_path is None:
            # Use default schema
            self._schema = get_default_schema()
        else:
            # Try custom file, fall back to defaults if doesn't exist
            custom_path = Path(file_path)
            if custom_path.exists():
                with open(custom_path, "r") as f:
                    self._schema: Dict[str, Dict[str, Any]] = json.load(f)
                validate_schema(self._schema)
            else:
                # Fall back to defaults
                self._schema = get_default_schema()

    def bind_log_hyperparameter(
        self,
        name: str,
        std: Optional[float] = None,
        min: Optional[float] = None,
        max: Optional[float] = None,
        shared: Optional[bool] = None,
    ) -> None:
        """
        Configure log-space hyperparameter with prototype pattern.
        If the default_schema already contains the targetted
        entry, defaults, not all features must be provided.
        Those that are not end up sticking with their current
        values.

        If defining a brand new schema not in the defaults,
        all of std, min, and shared must be provided, and
        min must be greater than zero.

        :param name: Hyperparameter name
        :param std: Standard deviation for perturbation
        :param min: Minimum bound (required for new log parameters)
        :param max: Maximum bound
        :param shared: Whether shared across param groups
        """
        if name in self._schema:
            # Prototype pattern: update existing entry with provided fields
            # First verify type compatibility
            if self._schema[name]["type"] != "log":
                raise TypeError(
                    f"Cannot bind log hyperparameter '{name}' - entry exists with different type. "
                    f"Use correct binding method instead."
                )
            entry = self._schema[name].copy()
            if std is not None:
                entry["std"] = std
            if min is not None:
                entry["min"] = min
            if max is not None:
                entry["max"] = max
            if shared is not None:
                entry["shared"] = shared
        else:
            # Creating new entry: require std and min
            if std is None:
                raise TypeError("std is required when creating new log hyperparameter")
            if min is None:
                raise ValueError("Log parameters require 'min' field")

            # Build new entry
            entry = {
                "type": "log",
                "std": std,
                "min": min,
                "shared": shared if shared is not None else True,
            }
            if max is not None:
                entry["max"] = max

        # Validate before storing
        validate_log_schema_entry(entry)

        # Store in schema
        self._schema[name] = entry

    def bind_linear_hyperparameter(
        self,
        name: str,
        std: Optional[float] = None,
        min: Optional[float] = None,
        max: Optional[float] = None,
        shared: Optional[bool] = None,
    ) -> None:
        """
        Configure log-space hyperparameter with prototype pattern.
        If the default_schema already contains the targetted
        entry, defaults, not all features must be provided.
        Those that are not end up sticking with their current
        values.

        If defining a brand new schema not in the defaults,
        all of std, and shared must be provided, and
        min must be greater than zero.

        :param name: Hyperparameter name
        :param std: Standard deviation for perturbation
        :param min: Minimum bound
        :param max: Maximum bound
        :param shared: Whether shared across param groups
        """
        if name in self._schema:
            # Prototype pattern: update existing entry with provided fields
            # First verify type compatibility
            if self._schema[name]["type"] != "linear":
                raise TypeError(
                    f"Cannot bind linear hyperparameter '{name}' - entry exists with different type. "
                    f"Use correct binding method instead."
                )
            entry = self._schema[name].copy()
            if std is not None:
                entry["std"] = std
            if min is not None:
                entry["min"] = min
            if max is not None:
                entry["max"] = max
            if shared is not None:
                entry["shared"] = shared
        else:
            # Creating new entry: require std, min/max optional
            if std is None:
                raise TypeError("std is required when creating new linear hyperparameter")

            # Build new entry
            entry = {
                "type": "linear",
                "std": std,
                "shared": shared if shared is not None else True,
            }
            if min is not None:
                entry["min"] = min
            if max is not None:
                entry["max"] = max

        # Validate before storing
        validate_linear_schema_entry(entry)

        # Store in schema
        self._schema[name] = entry

    def __call__(self,
                 strategy: AbstractStrategy,
                 only: Optional[List[str]] = None,) -> AbstractStrategy:
        """
        Apply schema to strategy.

        :param strategy: AbstractStrategy to configure
        :param only: Optional subset of hyperparameters to apply
        :return: Same strategy object you gave in, with the changes applied.
        """
        # Determine which hyperparameters to apply
        hyperparams_to_apply = only if only is not None else list(self._schema.keys())

        for name in hyperparams_to_apply:
            # Skip if not in schema
            if name not in self._schema:
                continue

            # Skip if not in strategy's valid binding targets
            if name not in strategy.valid_binding_targets:
                continue

            entry = self._schema[name]
            param_type = entry["type"]

            # Extract fields for binding
            std = entry["std"]
            min_val = entry.get("min")
            max_val = entry.get("max")
            shared = entry["shared"]

            # Call appropriate bind method on strategy
            if param_type == "log":
                strategy.bind_log_hyperparameter(
                    name, std=std, min=min_val, max=max_val, shared=shared
                )
            elif param_type == "linear":
                strategy.bind_linear_hyperparameter(
                    name, std=std, min=min_val, max=max_val, shared=shared
                )

            # Log the application
            self.logging_callback(f"Applied {name} of {param_type} type to strategy")
        return strategy

    def save(self, file_path: str) -> None:
        """
        Save current schema to JSON file.

        :param file_path: Path to save schema
        """
        with open(file_path, "w") as f:
            json.dump(self._schema, f, indent=2)

    def load_from_strategy(self, strategy):
        """
        Extract schema from strategy and merge into current schema.

        :param strategy: AbstractStrategy to extract from
        :return: Self for fluent interface
        """
        # Extract strategy schema
        strategy_schema = strategy.state_dict()

        # Merge: strategy schema overwrites self schema (in place)
        self._schema = {**self._schema, **strategy_schema}

        return self
