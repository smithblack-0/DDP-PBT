"""
Communication: Distributed gathering and reduction operations.

Handles gathering pytrees from all workers and reducing with weighted sums.
Uses leaf-by-leaf operations for memory efficiency.
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Any


class Communication:
    """
    Handles distributed gathering and reduction operations.

    Wraps torch.distributed primitives for gathering dictionary trees
    and computing weighted reductions across workers.
    """

    def __init__(self):
        """Initialize Communication (no state needed)."""
        pass

    def gather_pytree_list(self, pytree: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Gather dictionary trees from all workers.

        Args:
            pytree: Local dictionary tree with string keys and tensor values.

        Returns:
            List of dictionary trees from all workers (length = world_size).
            Each element is the dictionary tree from that rank.
        """
        if not dist.is_initialized():
            raise NotImplementedError("Cannot use DDP-PBT in non-distributed mode")
        world_size = dist.get_world_size()
        output_list = [pytree]*world_size
        dist.all_gather_object(output_list, pytree)
        return output_list

    def reduce_by_world_weights(
        self,
        world_weights: List[float],
        pytree: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Reduce dictionary tree using weighted sum across workers.

        Args:
            world_weights: Weights for each worker (length = world_size, sum = 1.0).
            pytree: Local dictionary tree with string keys and tensor values.

        Returns:
            Single dictionary tree with weighted sum of tensors.

        Memory-efficient implementation:
        - All-gathers each leaf from all workers
        - Filters to non-zero weights before stacking
        - Computes weighted sum per leaf
        - Processes one leaf at a time to avoid large memory overhead
        """
        if not dist.is_initialized():
            raise NotImplementedError("Cannot use DDP-PBT in non-distributed mode")


        world_size = dist.get_world_size()
        keys = sorted(pytree.keys())

        # Find non-zero weight indices
        non_zero_indices = [i for i, w in enumerate(world_weights) if w > 0]
        non_zero_weights = [world_weights[i] for i in non_zero_indices]

        # Normalize weights (should already be normalized, but ensure it)
        total_weight = sum(non_zero_weights)
        if total_weight > 0:
            non_zero_weights = [w / total_weight for w in non_zero_weights]

        # Reduce each leaf
        result = {}
        for key in keys:
            local_tensor = pytree[key]

            # Gather from all workers
            gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(world_size)]
            dist.all_gather(gathered_tensors, local_tensor)

            # Filter to non-zero weights
            filtered_tensors = [gathered_tensors[i] for i in non_zero_indices]

            # Compute weighted sum
            weighted_sum = torch.zeros_like(local_tensor)
            for tensor, weight in zip(filtered_tensors, non_zero_weights):
                weighted_sum += tensor * weight

            result[key] = weighted_sum

        return result
