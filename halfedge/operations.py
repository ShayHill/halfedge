#!/usr/bin/env python3
""" functions which insert, remove, or split edges from a HalfEdges instance
created: 181121 13:43:59

As far as I know. These functions are deprecated, replaced by methods in HalfEdges
TODO: see how much of that is true
"""

from typing import (
    Generator,
    Set,
    Hashable,
    Any,
    Dict,
    cast,
    overload,
    TypeVar,
    Optional,
)

from .half_edge_elements import Edge, Face, ManifoldMeshError, Vert, _MeshElementBase
from .half_edge_querries import StaticHalfEdges
from contextlib import suppress


def array_equal(*args: Any):
    with suppress(ValueError):
        # should work for everything except numpy
        return all(x == args[0] for x in args[1:])
    with suppress(TypeError):
        # for numpy, which would return [True, True, ...]
        return all(all(x == args[0]) for x in args[1:])
    raise ValueError(f"module does not support equality test between {args}")


KeyT = TypeVar("KeyT")


def get_dict_intersection(*dicts: Dict[KeyT, Any]) -> Dict[KeyT, Any]:
    """
    Identical key: value items from multiple dictionaries.

    """
    intersection = {}
    for key in set.intersection(*(set(x.keys()) for x in dicts)):
        if array_equal(*(x[key] for x in dicts)):
            intersection[key] = dicts[0][key]
    return intersection




def full_edges_only(edges: Set[Edge]) -> Generator[Edge, Set[Edge], None]:
    """Edges where edge.pair is in input set.

    Yields either edge or pair where BOTH edge and pair are in input set.

    E.g.,
    `iter_full(mesh.interior_edges)` would yield either (half)edge OR pair where
    both edge and pair are face edges.
    """
    edges = set(edges)
    while edges:
        half = edges.pop()
        if half.pair in edges:
            yield half








def remove_face(mesh: StaticHalfEdges, face: Face) -> None:
    """Remove all edges around a face."""
    edges = tuple(face.edges)
    potential_bridges = [x for x in edges if x.orig.valence > 2]
    if len({x.pair.face for x in edges}) < len(potential_bridges):
        raise ManifoldMeshError(
            "Popping this face would create non-manifold mesh."
            " One of this faces's edges is a bridge edge."
        )

    try:
        for edge in edges:
            mesh.remove_edge(edge)
    except ManifoldMeshError:
        raise RuntimeError(
            "Unexpected error. This should have been caught at the beginning of the "
            "function. A bridge edge was found (e.i., we discovered that this face "
            "cannot be removed), but some face edges may have been removed before this "
            "was realized. We've found a bug in the module."
        )


def insert_edge(
    mesh: StaticHalfEdges,
    orig: Vert,
    dest: Vert,
    face: Optional[Face] = None,
    **edge_kwargs: Any,
) -> Edge:
    """
    Insert a new edge between two verts.

    :face: face to split (or not. see below)
    :orig: origin of new edge
    :dest: destination of new edge
    :returns: newly inserted edge

    Edge face is created.
    Pair face is retained.

    This will only split the face if both orig and dest are new Verts. This function
    will connect:

        * two existing Verts on the same face
        * an existing Vert to a new vert inside the face
        * a new vert inside the face to an existing vert
        * two new verts to create a floating edge (this will result in a non-manifold
          mesh if the face has any other edges)
    """
    if face is None:
        shared_faces = set(orig.faces) | set(dest.faces)
        if len(shared_faces) == 1:
            face = shared_faces.pop()
        else:
            raise ValueError('face cannot be determined from orig and dest')

    face_edges = face.edges
    orig2edge = {x.orig: x for x in face_edges}
    dest2edge = {x.dest: x for x in face_edges}

    if set(face.verts) & {orig, dest} != mesh.verts & {orig, dest}:
        raise ManifoldMeshError("orig or dest in mesh but not on given face")

    if getattr(orig, "edge", None) and dest in orig.neighbors:
        raise ManifoldMeshError("overwriting existing edge")

    if not set(face.verts) & {orig, dest} and face in mesh.faces:
        # TODO: test adding edge to an empty mesh
        raise ManifoldMeshError("adding floating edge to existing face")

    edge = Edge(*face_edges, orig=orig, **edge_kwargs)
    pair = Edge(*face_edges, orig=dest, pair=edge, **edge_kwargs)
    edge.next = orig2edge.get(dest, pair)
    edge.prev = dest2edge.get(orig, pair)
    pair.next = orig2edge.get(orig, edge)
    pair.prev = dest2edge.get(dest, edge)

    if len(mesh.verts & {orig, dest}) == 2:
        edge.face = Face(face)
        for edge_ in edge.face_edges[1:]:
            edge_.face = edge.face

    mesh.edges.update({edge, pair})
    return edge


# TODO: test
def insert_vert(mesh: StaticHalfEdges, face: Face, orig: Vert) -> None:
    """
    Insert a vert into a face then triangulate face.
    """
    for vert in face.verts:
        insert_edge(mesh, orig, vert, face)


def inherit_kwargs(*ancestors: _MeshElementBase, **kwargs: Any) -> Dict[str, Any]:
    """
    Attributes shared by all ancestors.

    :param ancestors: mesh elements from which to inherit attributes
    :param kwargs: new attributes (these overwrite ancestor attributes)
    :returns: attr names mapped to values
    """
    name2val = get_dict_intersection(*(x.__dict__ for x in ancestors))
    name2val.update(kwargs)
    return name2val


def split_edge(mesh: StaticHalfEdges, edge: Edge, **vert_kwargs) -> Vert:
    """ Add a vert to the middle of an edge. """
    new_vert = Vert(**inherit_kwargs(edge.orig, edge.dest, **vert_kwargs))
    for orig, dest in ((edge.dest, new_vert), (new_vert, edge.orig)):
        new_edge = insert_edge(mesh, orig, dest, edge.face)
        new_edge.fill_from(edge.pair)
        new_edge.pair.fill_from(edge)
    mesh.remove_edge(edge)
    return new_vert


def collapse_edge(mesh: StaticHalfEdges, edge: Edge, **vert_kwargs) -> Vert:
    """"""
    new_vert = Vert(edge.orig, edge.dest, **vert_kwargs)
    for edge_ in set(edge.orig.edges) | set(edge.dest.edges):
        edge_.orig = new_vert

    # remove slits
    for face in (x.face for x in new_vert.edges if len(x.face.edges) == 2):
        face_edges = face.edges
        face_edges[0].pair.pair = face_edges[1].pair
        mesh.edges -= set(face_edges)
    return new_vert
