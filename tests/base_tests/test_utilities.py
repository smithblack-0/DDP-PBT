"""
Tree utilities contract test suite.

These tests validate that the public walking utilities:
- Traverse pytrees deterministically with stable paths and ordering.
- Walk multiple pytrees in lockstep under an identical schema requirement.
- Enforce schema identity strictly (container types, dict key sets, sequence lengths, leaf types).
- Respect the max_depth contract, including the unlimited sentinel (-1).
"""

from typing import Any, Dict, List, Tuple

import pytest

from src.ddp_pbt.base.utilities import patch_pytree, walk_pytree_collection, walk_single_pytree


@pytest.fixture
def nested_pair_and_expected() -> Tuple[dict, dict, List[Tuple[str, List[object]]]]:
    """
    Returns two schema-identical pytrees and the expected full walk output.

    This fixture is intentionally small but includes:
    - dict keys
    - list indices
    - nested dict
    """
    a = {"beta": [0.9, 0.99], "state": {"exp_avg": [1, 2]}}
    b = {"beta": [0.8, 0.95], "state": {"exp_avg": [10, 20]}}

    expected = [
        ("beta/0", [0.9, 0.8]),
        ("beta/1", [0.99, 0.95]),
        ("state/exp_avg/0", [1, 10]),
        ("state/exp_avg/1", [2, 20]),
    ]
    return a, b, expected


@pytest.fixture
def deeper_tree() -> dict:
    """
    Returns a single pytree with depth > 3 to validate max_depth truncation behavior.
    """
    return {"x": {"y": {"z": 1}}}


@pytest.fixture
def dict_order_pair() -> Tuple[dict, dict]:
    """
    Returns two dict-only pytrees where insertion order is important.

    The walker must follow root insertion order when yielding leaf paths.
    """
    root = {"b": 2, "a": 1}
    peer = {"b": 20, "a": 10}
    return root, peer


@pytest.fixture
def patch_base_tree_dict() -> Dict[str, Any]:
    """A small dict pytree with both leaves and a nested branch."""
    return {"x": 1, "state": {"exp_avg": [10, 20], "step": 3}}


@pytest.fixture
def patch_base_tree_list() -> List[Any]:
    """A list pytree used to validate list normalization and index patching."""
    return [1, {"k": 2}, 3]


@pytest.fixture
def patch_base_tree_tuple() -> Tuple[Any, ...]:
    """A tuple pytree used to validate tuple normalization and index patching."""
    return (1, {"k": 2}, 3)


# Tests themselves


class TestWalkPytreeCollectionContract:
    """Tests that walk_pytree_collection honors the public traversal and schema contracts."""

    def test_walk_pytree_collection_yields_expected_leaf_stream_in_deterministic_order(
        self,
        nested_pair_and_expected,
    ) -> None:
        """Walk pytree collection yields the full expected leaf stream in a deterministic order."""
        a, b, expected = nested_pair_and_expected
        got = list(walk_pytree_collection(a, b, max_depth=-1))
        assert got == expected

    def test_walk_pytree_collection_preserves_root_dict_insertion_order(
        self, dict_order_pair
    ) -> None:
        """Walk pytree collection yields dict leaves in the root insertion order."""
        root, peer = dict_order_pair
        got = list(walk_pytree_collection(root, peer, max_depth=-1))
        assert got == [
            ("b", [2, 20]),
            ("a", [1, 10]),
        ]

    def test_walk_pytree_collection_raises_when_dict_keys_differ_at_a_nested_path(self) -> None:
        """Walk pytree collection raises when dict key sets differ at a nested path."""
        root = {"state": {"x": 1}}
        peer = {"state": {"x": 2, "extra": 3}}

        with pytest.raises(RuntimeError) as excinfo:
            list(walk_pytree_collection(root, peer, max_depth=-1))

        assert "state" in str(excinfo.value)

    def test_walk_pytree_collection_raises_when_dict_key_is_missing_at_a_nested_path(self) -> None:
        """Walk pytree collection raises when a required dict key is missing at a nested path."""
        root = {"state": {"x": 1, "y": 2}}
        peer = {"state": {"x": 10, "y": 20}}  # start valid
        peer_missing = {"state": {"x": 10}}  # remove "y"

        with pytest.raises(RuntimeError) as excinfo:
            list(walk_pytree_collection(root, peer_missing, max_depth=-1))

        assert "state" in str(excinfo.value)

    def test_walk_pytree_collection_raises_when_sequence_lengths_differ(self) -> None:
        """Walk pytree collection raises when list/tuple lengths differ at a path."""
        root = {"beta": [0.9, 0.99]}
        peer = {"beta": [0.8]}

        with pytest.raises(RuntimeError) as excinfo:
            list(walk_pytree_collection(root, peer, max_depth=-1))

        assert "beta" in str(excinfo.value)

    def test_walk_pytree_collection_raises_when_container_types_differ_at_a_path(self) -> None:
        """Walk pytree collection raises when container types differ at a path (list vs tuple)."""
        root = {"beta": [0.9, 0.99]}
        peer = {"beta": (0.8, 0.95)}

        with pytest.raises(TypeError) as excinfo:
            list(walk_pytree_collection(root, peer, max_depth=-1))

        assert "beta" in str(excinfo.value)

    def test_walk_pytree_collection_raises_when_leaf_vs_container_mismatch_occurs(self) -> None:
        """Walk pytree collection raises when one pytree has a leaf where another has a container."""
        root = {"x": {"y": 1}}
        peer = {"x": 7}

        with pytest.raises(TypeError) as excinfo:
            list(walk_pytree_collection(root, peer, max_depth=-1))

        assert "x" in str(excinfo.value)

    def test_walk_pytree_collection_raises_when_leaf_types_differ_at_a_path(self) -> None:
        """Walk pytree collection raises when leaf Python types differ at a path."""
        root = {"x": 1}
        peer = {"x": 1.0}

        with pytest.raises(TypeError) as excinfo:
            list(walk_pytree_collection(root, peer, max_depth=-1))

        assert "x" in str(excinfo.value)

    def test_walk_pytree_collection_respects_max_depth_truncation(self, deeper_tree) -> None:
        """
        Walk pytree collection truncates traversal once max_depth is exhausted.

        This test asserts the observable behavior: deeper leaves are not yielded.
        """
        got_depth_1 = list(walk_pytree_collection(deeper_tree, max_depth=1))
        got_depth_2 = list(walk_pytree_collection(deeper_tree, max_depth=2))
        got_depth_3 = list(walk_pytree_collection(deeper_tree, max_depth=3))
        got_unlimited = list(walk_pytree_collection(deeper_tree, max_depth=-1))

        assert got_depth_1 == []
        assert got_depth_2 == []
        assert got_depth_3 == []
        assert got_unlimited == [("x/y/z", [1])]


class TestWalkSinglePytreeContract:
    """Tests that walk_single_pytree matches the collection walker semantics for one pytree."""

    def test_walk_single_pytree_yields_expected_leaf_stream(self, nested_pair_and_expected) -> None:
        """Walk single pytree yields the expected leaf stream with unwrapped values."""
        a, _, _ = nested_pair_and_expected
        got = list(walk_single_pytree(a, max_depth=-1))
        assert got == [
            ("beta/0", 0.9),
            ("beta/1", 0.99),
            ("state/exp_avg/0", 1),
            ("state/exp_avg/1", 2),
        ]

    def test_walk_single_pytree_matches_collection_unwrap_semantics(
        self, nested_pair_and_expected
    ) -> None:
        """Walk single pytree matches collection walker output when unwrapped."""
        a, _, _ = nested_pair_and_expected

        single = list(walk_single_pytree(a, max_depth=-1))
        collection = list(walk_pytree_collection(a, max_depth=-1))
        unwrapped = [(p, leaves[0]) for p, leaves in collection]

        assert single == unwrapped

    def test_walk_single_pytree_respects_max_depth_truncation(self, deeper_tree) -> None:
        """Walk single pytree truncates traversal once max_depth is exhausted."""
        got_depth_1 = list(walk_single_pytree(deeper_tree, max_depth=1))
        got_depth_2 = list(walk_single_pytree(deeper_tree, max_depth=2))
        got_depth_3 = list(walk_single_pytree(deeper_tree, max_depth=3))
        got_unlimited = list(walk_single_pytree(deeper_tree, max_depth=-1))

        assert got_depth_1 == []
        assert got_depth_2 == []
        assert got_depth_3 == []
        assert got_unlimited == [("x/y/z", 1)]


class TestPatchPytreeContract:
    """Tests that patch_pytree honors the public patching and schema invariants."""

    def test_patch_pytree_mutates_tree(
        self,
        patch_base_tree_dict: Dict[str, Any],
    ) -> None:
        """Patch pytree returns a new object and leaves the input value-equal to its original."""
        original = patch_base_tree_dict
        original_snapshot = {"x": 1, "state": {"exp_avg": [10, 20], "step": 3}}

        patched = patch_pytree(original, [("x", 999)])

        assert patched == {"x": 999, "state": {"exp_avg": [10, 20], "step": 3}}
        assert original_snapshot != patched
        assert patched is original

    def test_patch_pytree_applies_multiple_leaf_patches_across_branches(
        self,
        patch_base_tree_dict: Dict[str, Any],
    ) -> None:
        """Patch pytree applies multiple leaf patches in different parts of the tree."""
        patched = patch_pytree(
            patch_base_tree_dict,
            [
                ("x", 2),
                ("state/step", 4),
                ("state/exp_avg/1", 200),
            ],
        )
        assert patched == {"x": 2, "state": {"exp_avg": [10, 200], "step": 4}}

    def test_patch_pytree_raises_when_setting_key_does_not_exist(
        self,
        patch_base_tree_dict: Dict[str, Any],
    ) -> None:
        """Patch pytree raises when a patch attempts to set a key that does not exist."""
        with pytest.raises(RuntimeError) as excinfo:
            patch_pytree(patch_base_tree_dict, [("does_not_exist", 1)])

        assert "does_not_exist" in str(excinfo.value)

    def test_patch_pytree_raises_when_nested_key_does_not_exist(
        self,
        patch_base_tree_dict: Dict[str, Any],
    ) -> None:
        """Patch pytree raises when a nested patch targets a non-existent key."""
        with pytest.raises(RuntimeError) as excinfo:
            patch_pytree(patch_base_tree_dict, [("state/missing", 1)])

        assert "missing" in str(excinfo.value)

    def test_patch_pytree_raises_when_attempting_to_descend_into_a_leaf(self) -> None:
        """Patch pytree raises when a patch implies a branch under a leaf."""
        tree = {"a": 7}

        with pytest.raises(TypeError) as excinfo:
            patch_pytree(tree, [("a/b", 1)])

        assert "a" in str(excinfo.value)

    def test_patch_pytree_raises_when_patch_value_is_a_container(
        self,
        patch_base_tree_dict: Dict[str, Any],
    ) -> None:
        """Patch pytree raises when asked to patch in a container value (schema immutability)."""
        with pytest.raises(RuntimeError) as excinfo:
            patch_pytree(patch_base_tree_dict, [("x", {"illegal": "container"})])

        assert "schema" in str(excinfo.value).lower()

    def test_patch_pytree_raises_when_leaf_type_changes(
        self,
        patch_base_tree_dict: Dict[str, Any],
    ) -> None:
        """Patch pytree raises when asked to replace a leaf with a different leaf type."""
        with pytest.raises(TypeError) as excinfo:
            patch_pytree(patch_base_tree_dict, [("x", "not an int anymore")])

        assert "x" in str(excinfo.value)

    def test_patch_pytree_patches_list_indices_via_string_segments(
        self,
        patch_base_tree_list: List[Any],
    ) -> None:
        """Patch pytree patches list elements via numeric string path segments."""
        patched = patch_pytree(patch_base_tree_list, [("0", 111), ("1/k", 222)])
        assert patched == [111, {"k": 222}, 3]
        assert isinstance(patched, list)

    def test_patch_pytree_patches_tuple_indices_via_string_segments(
        self,
        patch_base_tree_tuple: Tuple[Any, ...],
    ) -> None:
        """Patch pytree patches tuple elements via numeric string path segments and returns a tuple."""
        patched = patch_pytree(patch_base_tree_tuple, [("0", 111), ("1/k", 222)])
        assert patched == (111, {"k": 222}, 3)
        assert isinstance(patched, tuple)

    def test_patch_pytree_raises_when_list_index_out_of_range(
        self,
        patch_base_tree_list: List[Any],
    ) -> None:
        """Patch pytree raises when asked to set a list index that does not exist."""
        with pytest.raises(RuntimeError) as excinfo:
            patch_pytree(patch_base_tree_list, [("99", 1)])

        assert "99" in str(excinfo.value)
