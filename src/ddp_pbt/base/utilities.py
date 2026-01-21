"""
A set of utilities commonly used throughout various functions
are located here
"""

from collections import defaultdict
from typing import Any, Dict, Generator, List, Optional, Tuple, TypeAlias, Union

import torch

PyTreeLeaf: TypeAlias = Any
RestrictedPyTreeLeaf: TypeAlias = Union[float, int]
PyTree: TypeAlias = Union[
    PyTreeLeaf,
    Dict[str, "PyTree"],
    List["PyTree"],
    Tuple["PyTree", ...],
]
RestrictedPyTree: TypeAlias = Union[
    RestrictedPyTreeLeaf,
    Dict[str, "PyTree"],
    List["PyTree"],
    Tuple["PyTree", ...],
]
PyTreePatch: TypeAlias = Tuple[str, PyTreeLeaf]


def _recursive_walk_pytrees(
    pytrees: List[PyTree],
    remaining_depth: int,
    path: Optional[str],
) -> Generator[Tuple[str, List[PyTreeLeaf]], None, None]:
    """
    Recursively walks the pytree. Yields the leaf pairs as it goes, and
    throws if keys are insane. It walks the pytree collections in sync.

    Specialized versions of this function are displayed as public functions.

    :param pytrees: A list of the pytree being walked.
    :param remaining_depth: The remaining depth in the search.
        - Note: Negative values can be used to allow searching to infinite depth.
    :param path: The path to this leaf collection.
    :return: A generator providing the path, leaf list pairs for the locations.
    :raise: RuntimeError, if the keys do not correctly match
    :raise: TypeError, if nodes do not have matching types
    """
    # Handle remaining depth break and modification.
    # Numbers less than zero never hit zero triggering
    # the base case and letting us descend forever.
    if remaining_depth == 0:
        return
    remaining_depth = remaining_depth - 1

    # Handle basic checking; same type?
    root_node = pytrees[0]
    if not all(type(node) is type(root_node) for node in pytrees):
        raise TypeError(f"At path {path} not all nodes were the same type. This is illegal")

    # Dictionary case.
    if isinstance(root_node, dict):

        # Key checking of schemas
        keys = list(root_node.keys())
        if not all(keys == list(node.keys()) for node in pytrees):
            raise RuntimeError(f"At path {path} dict nodes found with differing keys")

        # Update path, collect nodes, descend recursively, yield the answers.

        for key in root_node.keys():
            if path is None:
                subpath = key
            else:
                subpath = path + "/" + key

            nodes = [pytree[key] for pytree in pytrees]
            generator = _recursive_walk_pytrees(nodes, remaining_depth, subpath)
            yield from generator

    # List/tuple case
    elif isinstance(root_node, (list, tuple)):

        # Sanity check
        if not all(len(item) == len(root_node) for item in pytrees):
            raise RuntimeError(f"At path {path} list/tuple nodes found with differing length")

        # Get and yield the answer.
        for i, collection in enumerate(zip(*pytrees)):
            if path is None:
                subpath = str(i)
            else:
                subpath = path + "/" + str(i)
            generator = _recursive_walk_pytrees(list(collection), remaining_depth, subpath)
            yield from generator

    # It must be a leaf set then.
    else:
        yield path, pytrees


def patch_pytree(
    pytree: PyTree,
    patches: List[PyTreePatch],
) -> PyTree:
    """
    Mutates the pytree to apply the patch
    A patch is a tuple containing the path,
    and the new value. You provide the pytree
    and a list of patches, and it most efficiently
    rebuilds the tree.

    :param pytree: The pytree to modify
    :return: The newly rebuild pytree
    :raises: If the path does not exist, if trying to change type
    """

    # We begin by isolating patch collections
    # to be passed to subinstances, and the values
    # which need to be set.
    patch_collections = defaultdict(list)
    set_collection = {}
    for patch in patches:
        path, value = patch
        group = path.split("/", 1)

        if isinstance(value, (dict, list, tuple)):
            raise RuntimeError(
                "Attempt to change schema by patching in value. Use namedtuple instead"
            )

        if len(group) > 1:
            # Has suffix, tree goes deeper.
            # More than one leaf can be on the branch.
            prefix, suffix = group
            patch_collections[prefix].append((suffix, value))
        else:
            # This needs to be set here instead
            prefix = group[0]
            set_collection[prefix] = value

    # Patching mechanisms for each of
    # tuple, dict, and list. We standardize
    # to a dict strategy, then convert back
    # at the end.

    if isinstance(pytree, dict):
        processing_pytree = pytree
    elif isinstance(pytree, (list, tuple)):
        processing_pytree = {str(i): node for i, node in enumerate(pytree)}
    else:
        raise TypeError("At root pytree was not dict, tuple, or list: Cannot patch")

    # Process all branches.
    output = {}
    for key in processing_pytree.keys():
        if key in set_collection:
            patch_value = set_collection.pop(key)
            if type(patch_value) is not type(processing_pytree[key]):
                raise TypeError(f"Attempt to overwrite with different type detected at key {key}")
            output[key] = patch_value
        elif key in patch_collections:
            if not isinstance(processing_pytree[key], (list, dict, tuple)):
                raise TypeError(f"Attempt to overwrite a leaf with a branch at path {key}")
            output[key] = patch_pytree(processing_pytree[key], patch_collections.pop(key))
        else:
            output[key] = processing_pytree[key]

    # Verify we used all patches
    if len(patch_collections) > 0:
        msg = f"Patches present that are not used: {patch_collections}"
        raise RuntimeError(msg)
    if len(set_collection) > 0:
        msg = f"Attempted to set values that did not originally exist: {set_collection}"
        raise RuntimeError(msg)

    # Return. Do necessary conversions
    if isinstance(pytree, dict):
        pytree.clear()
        pytree.update(output)
    elif isinstance(pytree, list):
        update = list(output.values())
        pytree.clear()
        pytree.extend(update)
    elif isinstance(pytree, tuple):
        pytree = tuple(output.values())
    else:
        raise RuntimeError("Illegal state detected in patch_pytree: Contact maintainer")

    return pytree


def walk_pytree_collection(
    *pytrees: PyTree,
    max_depth: int = 3,
) -> Generator[Tuple[str, List[PyTreeLeaf]], None, None]:
    """
    Walks a collection of pytrees in parallel, gathering the
    common paths under one key as a list. Dictionary, List, and
    Tuple values are the tree branches, and anything else is
    presumed to be a leaf.

    :param pytrees: A collection of pytrees which can be passed along.
    :param max_depth: Settable only by kwarg. Default of 3. How many layers deep to recur.
        A value =of -1 will recur over everything.\
    :raises: RuntimeError, if insane schema found
    :raises: TypeError, if insane schema found
    """
    yield from _recursive_walk_pytrees(list(pytrees), max_depth, None)


def walk_single_pytree(
    pytree: PyTree,
    max_depth: int = 3,
) -> Generator[Tuple[str, PyTreeLeaf], None, None]:
    """
    Walks a single pytree, yielding the path and the leafs it finds
    as it goes.

    :param pytree: The pytree to walk. Dict, List, Tuple are branches, all else is leafs
    :param max_depth: The maximum depth to recur to. Set to -1 to walk the entire tree
    :return: A generator over str path, PyTreeLeaf tuples
    :raises: RuntimeError, if insane schema found
    :raises: TypeError, if insane schema found
    """
    # We just use recursive walk, then remove the list.
    for path, leaf_list in _recursive_walk_pytrees([pytree], max_depth, None):
        yield path, leaf_list[0]
