"""
Communication: Distributed gathering and reduction operations.

Handles gathering pytrees from all workers and reducing with weighted sums.
Uses leaf-by-leaf operations for memory efficiency.
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Any
from ddp_pbt.base.utilities import PyTree

class Communication:
    """
    Handles distributed gathering and reduction operations.

    Wraps torch.distributed primitives for gathering dictionary trees
    and computing weighted reductions across workers.
    """

    def __init__(self, suppress_error: bool = False):
        """Initialize Communication (no state needed)."""
        if not dist.is_initialized():
            raise EnvironmentError("Distributed world is not initiated")
        if self.world_size == 1 and not suppress_error:
            raise EnvironmentError("No DDP-PBT algorithm can be executed with only one worker")

    @property
    def world_size(self)->int:
        return dist.get_world_size()

    @property
    def rank(self)->int:
        return dist.get_rank()

    def gather_pytree_list(self, pytree: PyTree) -> List[PyTree]:
        """
        Gather arbitrary pytrees from all workers if needed.
        Usually used for transmitting hyperparameter genomes
        around though.

        Args:
            pytree: Pytree which may contain various values.

        Returns:
            List of pytree from all workers (length = world_size).
            Each element is the pytree from that rank.
        """
        if not (dist.is_available() and dist.is_initialized()):
            raise NotImplementedError("Cannot use DDP-PBT in non-distributed mode")
        world_size = dist.get_world_size()
        output_list = [None]*world_size
        dist.all_gather_object(output_list, pytree)
        return output_list

    def reduce_by_world_weights(
        self,
        world_weights: List[float],
        dict_tree: Dict[str, torch.Tensor]
    ):
        """
        Reduce dictionary tree using weighted sum across workers.
        Returns a new dictionary tree.

        Args:
            world_weights: Weights for each worker (length = world_size, sum = 1.0).
            dict_tree: Local dictionary tree with string keys and tensor values.
        Returns:
            Single dictionary tree with weighted sum of tensors across devices
        """
        # Memory, Comm, Compute efficient implementation:
        # - All-gathers each leaf from all workers
        # - Filters to non-zero weights before stacking
        # - Computes weighted sum per leaf
        # - Processes one leaf at a time to avoid large memory overhead
        # - One all gather for all devices keeps comm mesh traffic clean.
        if not dist.is_initialized():
            raise NotImplementedError("Cannot use DDP-PBT in non-distributed mode")

        # Subslice world weights sparsely
        world_weights = torch.tensor(world_weights, dtype=torch.float32)
        active_worlds = world_weights.nonzero(as_tuple=True)[0]
        world_weights = world_weights[active_worlds]

        # Handle communication and in-place reduction.
        world_size = dist.get_world_size()
        output_tree = {}
        for path, tensor in dict_tree.items():
            # Move to whatever is needed.
            if world_weights.device != tensor.device or world_weights.dtype != tensor.dtype:
                world_weights = world_weights.to(dtype=tensor.dtype, device=tensor.device)
                active_worlds = active_worlds.to(device=tensor.device)

            # Primary communication, then pruning down to live worlds
            communication_buffer = [torch.empty_like(tensor) for _ in range(world_size)]
            dist.all_gather(communication_buffer, tensor)
            communication_buffer = torch.stack(communication_buffer, dim=0)
            active_buffer = communication_buffer[active_worlds]

            # Weighted average and storage
            active_buffer = active_buffer.movedim(0, -1)
            weighted_buffer = active_buffer * world_weights
            result = torch.sum(weighted_buffer, dim=-1)
            output_tree[path] = result

        return output_tree
