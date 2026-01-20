"""
AbstractStrategy: Base class for PBT strategies.

Provides schema management, configuration methods, and orchestrates round-end flow.
Concrete strategies implement the four abstract methods: score, reduce_hyperparameters,
reduce_models, and reduce_optimizer.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import torch
from torch.optim.lr_scheduler import LRScheduler


def validate_log_schema_entry(config: Dict[str, Any]) -> None:
    """
    Validate a log-type hyperparameter schema entry.

    Args:
        config: Schema entry dict to validate.

    Raises:
        ValueError: If entry violates log parameter invariants.
        TypeError: If field types are incorrect.
    """
    # Validate std field
    std = config["std"]
    if not isinstance(std, float):
        raise TypeError("std must be float")
    if std <= 0:
        raise ValueError(f"std must be > 0, got {std}")

    # Validate shared field
    shared = config["shared"]
    if not isinstance(shared, bool):
        raise TypeError("shared must be boolean")

    # Log parameters REQUIRE min > 0
    min_val = config.get("min", None)
    if min_val is None:
        raise ValueError("Log parameters require 'min' field")
    if not isinstance(min_val, float):
        raise TypeError("min must be float")
    if min_val <= 0:
        raise ValueError(f"Log parameters require min > 0, got {min_val}")

    # Validate max if present
    max_val = config.get("max", None)
    if max_val is not None:
        if not isinstance(max_val, float):
            raise TypeError("max must be float")
        if max_val <= min_val:
            raise ValueError(f"max must be > min, got max={max_val} <= min={min_val}")


def validate_linear_schema_entry(config: Dict[str, Any]) -> None:
    """
    Validate a linear-type hyperparameter schema entry.

    Args:
        config: Schema entry dict to validate.

    Raises:
        ValueError: If entry violates linear parameter invariants.
        TypeError: If field types are incorrect.
    """
    # Validate std field
    std = config["std"]
    if not isinstance(std, float):
        raise TypeError("std must be float")
    if std <= 0:
        raise ValueError(f"std must be > 0, got {std}")

    # Validate shared field
    shared = config["shared"]
    if not isinstance(shared, bool):
        raise TypeError("shared must be boolean")

    # Validate bounds if present
    min_val = config.get("min", None)
    max_val = config.get("max", None)

    if min_val is not None and not isinstance(min_val, float):
        raise TypeError("min must be float")
    if max_val is not None and not isinstance(max_val, float):
        raise TypeError("max must be float")

    # If both present, check min < max
    if min_val is not None and max_val is not None:
        if min_val >= max_val:
            raise ValueError(f"min must be < max, got min={min_val} >= max={max_val}")


def validate_schema(schema: Dict[str, Dict[str, Any]]) -> None:
    """
    Validate hyperparameter schema for correctness.

    Dispatches to type-specific validators based on 'type' field.

    Args:
        schema: Hyperparameter schema to validate.

    Raises:
        ValueError: If schema violates any invariant.
        TypeError: If field types are incorrect.
        KeyError: If required fields are missing.
    """
    if not isinstance(schema, dict):
        raise TypeError("Schema must be a dictionary")

    for param_name, config in schema.items():
        try:
            # Get type and dispatch to appropriate validator
            param_type = config["type"]

            if param_type == "log":
                validate_log_schema_entry(config)
            elif param_type == "linear":
                validate_linear_schema_entry(config)
            else:
                raise ValueError(f"Invalid type '{param_type}'. Must be 'log' or 'linear'")

        except (ValueError, TypeError, KeyError) as e:
            raise type(e)(f"Error in '{param_name}': {str(e)}") from e


class AbstractStrategy(LRScheduler, ABC):
    """
    Base class for Population Based Training strategies.
    A schedule. With all the baggage.Yes

    Responsibilities:
    - Owns and builds the hyperparameter schema
    - Provides configuration methods (bind_log_hyperparameter, bind_linear_hyperparameter)
    - Orchestrates round-end flow via step() method
    - Defines abstract methods that concrete strategies must implement

    Concrete strategies must implement all four abstract methods coherently.
    """

    def __init__(self,
                 state,
                 communication,
                 config: Optional[Dict[str, Dict[str, Any]]] = None
                 ):
        """
        Initialize AbstractStrategy.

        Args:
            state: State instance for extracting/injecting optimizer data.
            communication: Communication instance for distributed operations.
            config: Optional native JSON schema config dict.
        Raises:
            ValueError: If config violates schema invariants.
        """
        self._finished_setup = False
        super().__init__(state.optimizer)
        self._state = state
        self._communication = communication
        self._schema = config or {}

        # Validate schema if provided
        if self._schema:
            validate_schema(self._schema)
            self._state.setup_schema(self._schema)

        self._finished_setup = True

    @property
    def valid_binding_targets(self) -> List[str]:
        """
        Returns list of bindable hyperparameter paths from optimizer.

        Delegates to State.valid_hyperparameter_paths to get float-valued
        fields available in optimizer.param_groups.

        Returns:
            List of hyperparameter paths that can be bound.
        """
        return self._state.valid_hyperparameter_paths

    @property
    def schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Exposes the hyperparameter schema.

        Returns:
            The schema dictionary.
        """
        return self._schema

    def bind_log_hyperparameter(
        self,
        name: str,
        std: float,
        min: float,
        max: Optional[float] = None,
        shared: bool = True
    ) -> None:
        """
        Builder method to bind a log-space hyperparameter.

        Args:
            name: Hyperparameter name (must exist in optimizer.param_groups).
            std: Standard deviation for perturbation (must be > 0).
            min: Minimum bound (required for log parameters, must be > 0).
            max: Optional maximum bound (must be > min if provided).
            shared: If True, shared across all param groups; if False, per-group.

        Raises:
            ValueError: If hyperparameter doesn't exist or validation fails.
        """
        # Validate hyperparameter exists
        if not any(name in str(path) for path in self.valid_binding_targets):
            raise ValueError(f"Hyperparameter '{name}' not found in optimizer")

        # Build entry
        entry = {
            "type": "log",
            "std": std,
            "min": min,
            "shared": shared
        }
        if max is not None:
            entry["max"] = max

        # Validate BEFORE inserting
        validate_log_schema_entry(entry)

        # Add to schema (State sees update via reference)
        self._schema[name] = entry

    def bind_linear_hyperparameter(
        self,
        name: str,
        std: float,
        min: Optional[float] = None,
        max: Optional[float] = None,
        shared: bool = True
    ) -> None:
        """
        Builder method to bind a linear-space hyperparameter.

        Args:
            name: Hyperparameter name (must exist in optimizer.param_groups).
            std: Standard deviation for perturbation (must be > 0).
            min: Optional minimum bound.
            max: Optional maximum bound (must be > min if both provided).
            shared: If True, shared across all param groups; if False, per-group.

        Raises:
            ValueError: If hyperparameter doesn't exist or validation fails.
        """
        # Validate hyperparameter exists
        if not any(name in str(path) for path in self.valid_binding_targets):
            raise ValueError(f"Hyperparameter '{name}' not found in optimizer")

        # Build entry
        entry = {
            "type": "linear",
            "std": std,
            "shared": shared
        }
        if min is not None:
            entry["min"] = min
        if max is not None:
            entry["max"] = max

        # Validate BEFORE inserting
        validate_linear_schema_entry(entry)

        # Add to schema (State sees update via reference)
        self._schema[name] = entry

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns schema for checkpointing.

        Returns:
            Schema dictionary suitable for serialization.
        """
        return self._schema

    def load_state_dict(self, schema: Dict[str, Dict[str, Any]]) -> None:
        """
        Restores schema from checkpoint.

        Args:
            schema: Schema dictionary to restore.

        Raises:
            ValueError: If schema violates invariants.
        """
        validate_schema(schema)
        self._schema.clear()
        self._schema.update(schema)

    def step(self, validation_metric: Optional[float] = None) -> None:  # type: ignore[override]
        """
        Execute round-end strategy step.

        Orchestrates the four-phase process:
        1. Extract local data from State
        2. Gather distributed data via Communication
        3. Call abstract methods to compute reductions
        4. Inject results back via State

        Args:
            validation_metric: Local worker's validation metric.

        Raises:
            ValueError: If validation_metric is None after initialization completes.
        """
        # Guard: Handle LRScheduler initialization
        if validation_metric is None:
            if not self._finished_setup:
                # Allow during initialization
                return
            else:
                raise ValueError("validation_metric is required after initialization")

        # Phase 1: Extract local data
        local_hyperparams = self._state.get_hyperparam_values()
        local_model_pytree = self._state.get_model_tensors()
        local_optimizer_pytree = self._state.get_optimizer_tensors()

        # Phase 2: Gather world hyperparameters
        world_hyperparameters = self._communication.gather_pytree_list(local_hyperparams)

        # Phase 3: Call abstract methods (concrete strategy implementations)
        world_weights = self.score(validation_metric, self._communication)

        new_hyperparams = self.reduce_hyperparameters(
            world_weights, world_hyperparameters, self._communication
        )

        new_model_pytree = self.reduce_models(
            world_weights, local_model_pytree, self._communication
        )

        new_optimizer_pytree = self.reduce_optimizer(
            world_weights, local_optimizer_pytree, self._communication
        )

        # Phase 4: Inject results back
        self._state.set_hyperparam_values(new_hyperparams)
        self._state.set_model_tensors(new_model_pytree)
        self._state.set_optimizer_tensors(new_optimizer_pytree)

    def get_last_lr(self) -> List[float]:
        """
        Return current learning rates from optimizer.

        Standard LRScheduler interface method.

        Returns:
            List of learning rates, one per param group.
        """
        return [group['lr'] for group in self.optimizer.param_groups]

    @abstractmethod
    def score(self, validation_metric: float, communication) -> List[float]:
        """
        Compute world weights from validation metric.

        Args:
            validation_metric: Local worker's validation metric.
            communication: Communication instance for gathering metrics.

        Returns:
            World weights (length = world_size, sum = 1.0).
        """
        pass

    @abstractmethod
    def reduce_hyperparameters(
        self,
        world_weights: List[float],
        world_hyperparameters: List[Dict[str, List[float]]],
        communication
    ) -> Dict[str, List[float]]:
        """
        Reduce hyperparameters across workers.

        Args:
            world_weights: Weights for each worker.
            world_hyperparameters: Hyperparameters from all workers.
            communication: Communication instance for distributed ops.

        Returns:
            Reduced hyperparameter values for next round.
        """
        pass

    @abstractmethod
    def reduce_models(
        self,
        world_weights: List[float],
        model_pytree: Dict[str, torch.Tensor],
        communication
    ) -> Dict[str, torch.Tensor]:
        """
        Reduce model parameters across workers.

        Args:
            world_weights: Weights for each worker.
            model_pytree: Local model parameter dictionary tree.
            communication: Communication instance for distributed ops.

        Returns:
            Reduced model parameter dictionary tree.
        """
        pass

    @abstractmethod
    def reduce_optimizer(
        self,
        world_weights: List[float],
        optimizer_pytree: Dict[str, torch.Tensor],
        communication
    ) -> Dict[str, torch.Tensor]:
        """
        Reduce optimizer state across workers.

        Args:
            world_weights: Weights for each worker.
            optimizer_pytree: Local optimizer state dictionary tree.
            communication: Communication instance for distributed ops.

        Returns:
            Reduced optimizer state dictionary tree.
        """
        pass
