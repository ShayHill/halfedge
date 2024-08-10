"""Ensure meshes are valid.

This is here instead of my test suite to test input I might enter when
using this module. It is here to test me, not the class. That being
said, it's available for the test suite to borrow.

created: 181127
"""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Iterator, TypeVar

from halfedge.half_edge_elements import Edge, Face, ManifoldMeshError

if TYPE_CHECKING:
    from halfedge.half_edge_querries import StaticHalfEdges

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
    """Confirm any property that uses a function lap does not fail."""
    for vert in mesh.verts:
        if not all(e.orig is vert for e in vert.edges):
            msg = "vert.edges do not all point to vert"
            raise ManifoldMeshError(msg)
    for face in mesh.faces | mesh.holes:
        if not all(e.face is face for e in face.edges):
            msg = "face.edges do not all point to face"
            raise ManifoldMeshError(msg)


def _confirm_edge_have_two_distinct_points(mesh: StaticHalfEdges) -> None:
    """Confirm that edges have two distinct points."""
    for edge in mesh.edges:
        if edge.orig is edge.dest:
            msg = "loop edge (orig == dest)"
            raise ManifoldMeshError(msg)


def _confirm_edge_dest_lookups_match(mesh: StaticHalfEdges) -> None:
    """Confirm that both lookup methods for edge.dest are the same."""
    for edge in mesh.edges:
        if edge.next.orig is not edge.pair.orig:
            msg = "next and pair do not share orig point"
            raise ManifoldMeshError(msg)


def _confirm_edges_do_not_overlap(mesh: StaticHalfEdges) -> None:
    """Confirm that edges do not overlap."""
    edge_tuples = {(x.orig, x.dest) for x in mesh.edges}
    if len(edge_tuples) < len(mesh.edges):
        msg = "overlapping edges"
        raise ManifoldMeshError(msg)


def _confirm_pair_points_align(mesh: StaticHalfEdges) -> None:
    """Confirm that pair edges align."""
    for edge in mesh.edges:
        if edge.orig != edge.pair.dest or edge.dest != edge.pair.orig:
            msg = "edge and pair points are not the same"
            raise ManifoldMeshError(msg)


def _confirm_no_ghost_edges(mesh: StaticHalfEdges) -> None:
    """Confirm that every face and vert edge is in the edge list."""
    face_edges: set[Edge] = set()
    vert_edges: set[Edge] = set()
    for face in mesh.faces | mesh.holes:
        face_edges.update(face.edges)
    for vert in mesh.verts:
        vert_edges.update(vert.edges)
    if face_edges ^ mesh.edges:
        msg = "face edges not in edge list"
        raise ManifoldMeshError(msg)
    if vert_edges ^ mesh.edges:
        msg = "vert edges not in edge list"
        raise ManifoldMeshError(msg)


def validate_mesh(mesh: StaticHalfEdges) -> None:
    """Confirm that mesh is a manifold mesh.

    :param mesh: mesh to validate
    :raise ManifoldMeshError: if not all faces can be reached by normal halfedges
        functions
    """
    if not mesh.edges:
        return

    _confirm_edge_have_two_distinct_points(mesh)
    _confirm_edge_dest_lookups_match(mesh)
    if not _does_reach_all(mesh.faces | mesh.holes, _faces_neighboring_face):
        msg = "not all faces can be reached by jumping over edges"
        raise ManifoldMeshError(msg)
    _confirm_edges_do_not_overlap(mesh)
    _confirm_pair_points_align(mesh)
    _confirm_no_ghost_edges(mesh)
    _confirm_function_laps_do_not_fail(mesh)
