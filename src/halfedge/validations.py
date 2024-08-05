"""Ensure meshes are valid.

This is here instead of my test suite to test input I might enter when
using this module. It is here to test me, not the class. That being
said, it's available for the test suite to borrow.

created: 181127
"""

from itertools import chain
from typing import Any, Callable, Iterator, Set, TypeVar

from .half_edge_elements import Face, ManifoldMeshError
from .half_edge_querries import StaticHalfEdges

T = TypeVar("T")


def _faces_neighboring_face(face: Face) -> Iterator[Face]:
    """All faces surrounding :face:"""
    return (edge.pair.face for edge in face.edges)


def _does_reach_all(set_: Set[Any], f_next: Callable[[T], Iterator[T]]) -> bool:
    """Can f_next(itm) reach entire set for each itm in set?"""
    found = set()
    for itm in set_:
        found, new = {itm}, {itm}

        while new:
            found.update(new)
            new.update(chain(*(f_next(x) for x in new)))
            new -= found

    return not bool(found ^ set_)


def validate_mesh(mesh: StaticHalfEdges) -> None:
    """Test for a "legal" mesh."""
    if not mesh.edges:
        return

    for vert in mesh.verts:
        try:
            assert vert.edge in mesh.edges
        except:
            raise ManifoldMeshError("vert points to missing edge")
        try:
            _ = vert.valence
        except:
            raise ManifoldMeshError("cannot loop vert")

    for edge in mesh.edges:

        if edge.next.orig is not edge.pair.orig:
            raise ManifoldMeshError("next or pair error")

        if edge.orig is edge.dest:
            raise ManifoldMeshError("loop edge")

    for face in mesh.faces | mesh.holes:
        try:
            _ = mesh.edges
        except:
            raise ManifoldMeshError("cannot complete edge lap")

        if any(edge.face != face for edge in face.edges):
            raise ManifoldMeshError("edge pointing to wrong face")

        try:
            assert face.edge in mesh.edges
        except:
            raise ManifoldMeshError("face points to missing edge")

    if not _does_reach_all(mesh.faces | mesh.holes, _faces_neighboring_face):
        raise ManifoldMeshError("not all faces can be reached by jumping over edges")

    # TODO: remove this or make it a warning
    edge_tuples = {(x.orig, x.dest) for x in mesh.edges}
    if len(edge_tuples) < len(mesh.edges):
        raise ManifoldMeshError("overlapping edges")
