"""
AbstractStrategy: Base class for PBT strategies.

Provides schema management, configuration methods, and orchestrates round-end flow.
Concrete strategies implement the four abstract methods: score, reduce_hyperparameters,
reduce_models, and reduce_optimizer.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import torch

class AbstractStrategy(ABC):
    """
    Base class for Population Based Training strategies.

    Responsibilities:
    - Owns and builds the hyperparameter schema
    - Provides configuration methods (bind_log_hyperparameter, bind_linear_hyperparameter)
    - Orchestrates round-end flow via step() method
    - Defines abstract methods that concrete strategies must implement

    Concrete strategies must implement all four abstract methods coherently.
    """

    def __init__(self, state, communication, config: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize AbstractStrategy.

        Args:
            state: State instance for extracting/injecting optimizer data.
            communication: Communication instance for distributed operations.
            config: Optional native JSON schema config dict.
                   Format: {
                       "param_name": {
                           "type": "log" | "linear",
                           "std": float,
                           "min": float (optional),
                           "max": float (optional),
                           "shared": bool
                       }
                   }
        """
        self._state = state
        self._communication = communication
        self._schema = config or {}

        # If config provided, setup schema in state
        if self._schema:
            self._state.setup_schema(self._schema)

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
    def optimizer(self) -> torch.optim.Optimizer:
        """
        Exposes the optimizer from State.

        Returns:
            The optimizer instance.
        """
        return self._state.optimizer

    def bind_log_hyperparameter(
        self,
        name: str,
        std: float,
        min: Optional[float] = None,
        max: Optional[float] = None,
        shared: bool = True
    ) -> None:
        """
        Builder method to bind a log-space hyperparameter.

        Args:
            name: Hyperparameter name (must exist in optimizer.param_groups).
            std: Standard deviation for perturbation.
            min: Optional minimum bound.
            max: Optional maximum bound.
            shared: If True, shared across all param groups; if False, per-group.

        Raises:
            ValueError: If hyperparameter doesn't exist in optimizer.
        """
        # Validate hyperparameter exists
        if not any(name in str(path) for path in self.valid_binding_targets):
            raise ValueError(f"Hyperparameter '{name}' not found in optimizer")

        # Add to schema
        self._schema[name] = {
            "type": "log",
            "std": std,
            "shared": shared
        }

        if min is not None:
            self._schema[name]["min"] = min
        if max is not None:
            self._schema[name]["max"] = max

        # Update state with new schema
        self._state.setup_schema(self._schema)

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
            std: Standard deviation for perturbation.
            min: Optional minimum bound.
            max: Optional maximum bound.
            shared: If True, shared across all param groups; if False, per-group.

        Raises:
            ValueError: If hyperparameter doesn't exist in optimizer.
        """
        # Validate hyperparameter exists
        if not any(name in str(path) for path in self.valid_binding_targets):
            raise ValueError(f"Hyperparameter '{name}' not found in optimizer")

        # Add to schema
        self._schema[name] = {
            "type": "linear",
            "std": std,
            "shared": shared
        }

        if min is not None:
            self._schema[name]["min"] = min
        if max is not None:
            self._schema[name]["max"] = max

        # Update state with new schema
        self._state.setup_schema(self._schema)

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
        """
        self._schema = schema
        self._state.setup_schema(self._schema)

    def step(self, validation_metric: float) -> None:
        """
        Execute round-end strategy step.

        Orchestrates the four-phase process:
        1. Extract local data from State
        2. Gather distributed data via Communication
        3. Call abstract methods to compute reductions
        4. Inject results back via State

        Args:
            validation_metric: Local worker's validation metric.
        """
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
