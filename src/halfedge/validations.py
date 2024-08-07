"""Ensure meshes are valid.

This is here instead of my test suite to test input I might enter when
using this module. It is here to test me, not the class. That being
said, it's available for the test suite to borrow.

created: 181127
"""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Iterator, TypeVar

from .half_edge_elements import Face, ManifoldMeshError

if TYPE_CHECKING:
    from .half_edge_querries import StaticHalfEdges

_T = TypeVar("_T")


def _faces_neighboring_face(face: Face) -> Iterator[Face]:
    """All faces surrounding :face:."""
    return (edge.pair.face for edge in face.edges)


def _does_reach_all(population: set[_T], f_next: Callable[[_T], Iterator[_T]]) -> bool:
    """Return True if f_next(itm) can reach entire set for each itm in set.

    :param set_: set of items to check if all can be reached
    :param f_next: function to get next items to check
    :return: True if all items can be reached

    Check that each item in population can be reached by recursively calling a
    function that takes that item and returns an iterator of other items in the
    population.
    """
    found: set[Any] = set()
    for itm in population:
        found, not_yet_found = {itm}, {itm}
        while not_yet_found:
            found.update(not_yet_found)
            not_yet_found.update(chain(*(f_next(x) for x in not_yet_found)))
            not_yet_found -= found
    return not bool(found ^ population)


def _confirm_function_laps_do_not_fail(mesh: StaticHalfEdges) -> None:
    """Test any property that uses a function lap does not fail."""
    for vert in mesh.verts:
        if not all(e.orig is vert for e in vert.edges):
            msg = "vert.edges do not all point to vert"
            raise ManifoldMeshError(msg)
    for face in mesh.faces | mesh.holes:
        if not all(e.face is face for e in face.edges):
            msg = "face.edges do not all point to face"
            raise ManifoldMeshError(msg)


def _confirm_edge_have_two_distinct_points(mesh: StaticHalfEdges) -> None:
    """Test that edges have two distinct points."""
    for edge in mesh.edges:
        if edge.orig is edge.dest:
            msg = "loop edge (orig == dest)"
            raise ManifoldMeshError(msg)


def _confirm_edge_dest_lookups_match(mesh: StaticHalfEdges) -> None:
    """Test that both lookup methods for edge.dest are the same."""
    for edge in mesh.edges:
        if edge.next.orig is not edge.pair.orig:
            msg = "next and pair do not share orig point"
            raise ManifoldMeshError(msg)


def _confirm_edges_do_not_overlap(mesh: StaticHalfEdges) -> None:
    """Test that edges do not overlap."""
    edge_tuples = {(x.orig, x.dest) for x in mesh.edges}
    if len(edge_tuples) < len(mesh.edges):
        msg = "overlapping edges"
        raise ManifoldMeshError(msg)


def validate_mesh(mesh: StaticHalfEdges) -> None:
    """Test for a manifold mesh."""
    if not mesh.edges:
        return

    _confirm_edge_have_two_distinct_points(mesh)
    _confirm_edge_dest_lookups_match(mesh)
    if not _does_reach_all(mesh.faces | mesh.holes, _faces_neighboring_face):
        msg = "not all faces can be reached by jumping over edges"
        raise ManifoldMeshError(msg)
    _confirm_edges_do_not_overlap(mesh)
    _confirm_function_laps_do_not_fail(mesh)
