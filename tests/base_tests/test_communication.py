"""
Test suite for Communication component.

Communication handles distributed gathering and reduction with real GLOO backend.
Tests validate gather_pytree_list and reduce_by_world_weights operations.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.ddp_pbt.base.communication import Communication


def gather_worker(rank, world_size, output_dir, master_addr, master_port):
    """Worker function for testing gather_pytree_list."""
    # Setup environment
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Initialize process group
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    try:
        # Create Communication instance
        comm = Communication()

        # Each worker has unique dictionary tree
        local_dict_tree = {
            "param1": torch.tensor([rank * 1.0]),
            "param2": torch.tensor([rank * 2.0, rank * 3.0])
        }

        # Gather from all workers
        gathered_list = comm.gather_pytree_list(local_dict_tree)

        # Save results
        output_file = Path(output_dir) / f"rank_{rank}.json"
        with open(output_file, "w") as f:
            # Convert tensors to lists for JSON serialization
            result_serializable = [
                {k: v.tolist() for k, v in pytree.items()}
                for pytree in gathered_list
            ]
            json.dump({"gathered": result_serializable}, f)

    finally:
        dist.destroy_process_group()


def reduce_worker(rank, world_size, output_dir, master_addr, master_port):
    """Worker function for testing reduce_by_world_weights."""
    # Setup environment
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Initialize process group
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    try:
        # Create Communication instance
        comm = Communication()

        # Each worker has unique dictionary tree
        local_dict_tree = {
            "param1": torch.tensor([rank * 10.0]),
            "param2": torch.tensor([rank * 20.0, rank * 30.0])
        }

        # World weights (rank 0 has 0.3, rank 1 has 0.7, others have 0.0)
        if world_size == 2:
            world_weights = [0.3, 0.7]
        else:
            world_weights = [0.3, 0.7] + [0.0] * (world_size - 2)

        # Reduce by weights
        reduced_tree = comm.reduce_by_world_weights(world_weights, local_dict_tree)

        # Save results
        output_file = Path(output_dir) / f"rank_{rank}.json"
        with open(output_file, "w") as f:
            result_serializable = {k: v.tolist() for k, v in reduced_tree.items()}
            json.dump({"reduced": result_serializable}, f)

    finally:
        dist.destroy_process_group()


def reduce_uniform_worker(rank, world_size, output_dir, master_addr, master_port):
    """Worker function for testing reduce with uniform weights."""
    # Setup environment
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Initialize process group
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    try:
        # Create Communication instance
        comm = Communication()

        # Each worker has unique dictionary tree
        local_dict_tree = {
            "param1": torch.tensor([float(rank + 1)]),
            "param2": torch.tensor([float(rank * 2), float(rank * 3)])
        }

        # Uniform weights
        world_weights = [1.0 / world_size] * world_size

        # Reduce by weights
        reduced_tree = comm.reduce_by_world_weights(world_weights, local_dict_tree)

        # Save results
        output_file = Path(output_dir) / f"rank_{rank}.json"
        with open(output_file, "w") as f:
            result_serializable = {k: v.tolist() for k, v in reduced_tree.items()}
            json.dump({"reduced": result_serializable}, f)

    finally:
        dist.destroy_process_group()


def filter_worker(rank, world_size, output_dir, master_addr, master_port):
    """Worker function for testing filtering with zero weights."""
    # Setup environment
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Initialize process group
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    try:
        # Create Communication instance
        comm = Communication()

        local_dict_tree = {
            "param1": torch.tensor([float(rank * 100)])
        }

        # Only ranks 1 and 3 have non-zero weights
        world_weights = [0.0, 0.4, 0.0, 0.6]

        # Reduce by weights
        reduced_tree = comm.reduce_by_world_weights(world_weights, local_dict_tree)

        # Save results
        output_file = Path(output_dir) / f"rank_{rank}.json"
        with open(output_file, "w") as f:
            result_serializable = {k: v.tolist() for k, v in reduced_tree.items()}
            json.dump({"reduced": result_serializable}, f)

    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.skipif(sys.platform == "win32", reason="GLOO not supported on Windows")
class TestCommunicationGatherPytreeList:
    """Tests gather_pytree_list with real GLOO backend."""

    def test_gathers_from_two_workers(self):
        """gather_pytree_list should collect dictionary trees from all workers."""
        world_size = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            # Spawn worker processes
            mp.spawn(
                gather_worker,
                args=(world_size, tmpdir, "localhost", "29500"),
                nprocs=world_size,
                join=True,
            )

            # Collect results from all ranks
            results = []
            for rank in range(world_size):
                output_file = Path(tmpdir) / f"rank_{rank}.json"
                with open(output_file, "r") as f:
                    data = json.load(f)
                    results.append(data["gathered"])

            # All ranks should see the same gathered list
            assert results[0] == results[1], "All ranks must agree on gathered data"

            # Verify structure
            gathered = results[0]
            assert len(gathered) == world_size

            # Verify each worker's data
            assert gathered[0]["param1"] == [0.0]
            assert gathered[0]["param2"] == [0.0, 0.0]
            assert gathered[1]["param1"] == [1.0]
            assert gathered[1]["param2"] == [2.0, 3.0]

    def test_gathers_from_four_workers(self):
        """gather_pytree_list should work with more than 2 workers."""
        world_size = 4

        with tempfile.TemporaryDirectory() as tmpdir:
            # Spawn worker processes
            mp.spawn(
                gather_worker,
                args=(world_size, tmpdir, "localhost", "29501"),
                nprocs=world_size,
                join=True,
            )

            # Collect results from all ranks
            results = []
            for rank in range(world_size):
                output_file = Path(tmpdir) / f"rank_{rank}.json"
                with open(output_file, "r") as f:
                    data = json.load(f)
                    results.append(data["gathered"])

            # All ranks should agree
            for i in range(1, world_size):
                assert results[i] == results[0], f"Rank {i} disagrees with rank 0"

            # Verify all workers' data present
            gathered = results[0]
            assert len(gathered) == world_size
            for rank in range(world_size):
                assert gathered[rank]["param1"] == [float(rank)]
                assert gathered[rank]["param2"] == [float(rank * 2), float(rank * 3)]


@pytest.mark.distributed
@pytest.mark.skipif(sys.platform == "win32", reason="GLOO not supported on Windows")
class TestCommunicationReduceByWorldWeights:
    """Tests reduce_by_world_weights with real GLOO backend."""

    def test_reduces_with_weighted_sum(self):
        """reduce_by_world_weights should compute weighted sum of dictionary trees."""
        world_size = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            # Spawn worker processes
            mp.spawn(
                reduce_worker,
                args=(world_size, tmpdir, "localhost", "29502"),
                nprocs=world_size,
                join=True,
            )

            # Collect results from all ranks
            results = []
            for rank in range(world_size):
                output_file = Path(tmpdir) / f"rank_{rank}.json"
                with open(output_file, "r") as f:
                    data = json.load(f)
                    results.append(data["reduced"])

            # All ranks should agree on reduced result
            assert results[0] == results[1], "All ranks must agree on reduced data"

            # Verify weighted sum
            # param1: 0.3 * [0.0] + 0.7 * [10.0] = [7.0]
            # param2: 0.3 * [0.0, 0.0] + 0.7 * [20.0, 30.0] = [14.0, 21.0]
            reduced = results[0]
            assert abs(reduced["param1"][0] - 7.0) < 1e-5
            assert abs(reduced["param2"][0] - 14.0) < 1e-5
            assert abs(reduced["param2"][1] - 21.0) < 1e-5

    def test_reduces_with_uniform_weights(self):
        """reduce_by_world_weights should average when weights are uniform."""
        world_size = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            # Spawn worker processes
            mp.spawn(
                reduce_uniform_worker,
                args=(world_size, tmpdir, "localhost", "29503"),
                nprocs=world_size,
                join=True,
            )

            # Collect results from all ranks
            results = []
            for rank in range(world_size):
                output_file = Path(tmpdir) / f"rank_{rank}.json"
                with open(output_file, "r") as f:
                    data = json.load(f)
                    results.append(data["reduced"])

            # All ranks should agree
            assert results[0] == results[1], "All ranks must agree"

            # Verify average
            # param1: 0.5 * [1.0] + 0.5 * [2.0] = [1.5]
            # param2: 0.5 * [0.0, 0.0] + 0.5 * [2.0, 3.0] = [1.0, 1.5]
            reduced = results[0]
            assert abs(reduced["param1"][0] - 1.5) < 1e-5
            assert abs(reduced["param2"][0] - 1.0) < 1e-5
            assert abs(reduced["param2"][1] - 1.5) < 1e-5

    def test_filters_zero_weight_entries(self):
        """reduce_by_world_weights should only use non-zero weighted workers."""
        world_size = 4

        with tempfile.TemporaryDirectory() as tmpdir:
            mp.spawn(
                filter_worker,
                args=(world_size, tmpdir, "localhost", "29504"),
                nprocs=world_size,
                join=True,
            )

            # Collect results
            results = []
            for rank in range(world_size):
                output_file = Path(tmpdir) / f"rank_{rank}.json"
                with open(output_file, "r") as f:
                    data = json.load(f)
                    results.append(data["reduced"])

            # All ranks should agree
            for i in range(1, world_size):
                assert results[i] == results[0], f"Rank {i} disagrees"

            # Verify only ranks 1 and 3 contributed
            # 0.4 * 100 + 0.6 * 300 = 40 + 180 = 220
            reduced = results[0]
            assert abs(reduced["param1"][0] - 220.0) < 1e-5
