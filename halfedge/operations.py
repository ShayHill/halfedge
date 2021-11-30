#!/usr/bin/env python3
""" functions which insert, remove, or split edges from a HalfEdges instance
created: 181121 13:43:59

As far as I know. These functions are deprecated, replaced by methods in HalfEdges
TODO: see how much of that is true
"""

from contextlib import suppress
from typing import Any, Dict, Generator, Optional, Set, TypeVar

from .half_edge_elements import Edge, Face, ManifoldMeshError, Vert, _MeshElementBase
from .half_edge_querries import StaticHalfEdges


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


def remove_edge(mesh: StaticHalfEdges, edge: Edge) -> None:
    """
    Cut an edge out of the mesh.

    Will not allow you to break (make non-manifold) the mesh. For example,
    here's a mesh with three faces, one in each square, and a third face or
    hole around the outside. If I remove that long edge, the hole would have
    two small, square faces inside of it. The hole would point to a half edge
    around one or the other square, but that edge would just "next" around its
    own small square. The other square could never be found.
     _       _
    |_|_____|_|

    Attempting to pop such edges will raise a ManifoldMeshError.

    Always removes the edge's face and expands the pair's face (if they are
    different).

    """
    pair = edge.pair

    if edge not in mesh.edges:
        raise ValueError("edge {} does not exist in mesh".format(edge.sn))

    if edge.orig.valence > 1 and edge.dest.valence > 1 and edge.face == pair.face:
        raise ManifoldMeshError("would create non-manifold mesh")

    # make sure orig and dest do not point to this edge (if there's another option)
    edge.next.orig = edge.next.orig
    pair.next.orig = pair.next.orig

    # set all faces equal to pair.face
    for edge_ in (x for x in edge.face_edges if x not in (edge, edge.pair)):
        edge_.face = pair.face

    # disconnect from previous edges
    edge.prev.next = pair.next
    pair.prev.next = edge.next
    mesh.edges -= {edge, pair}


def remove_vert(mesh: StaticHalfEdges, vert: Vert) -> None:
    """
    Remove all edges around a vert.

    :raises: ManifoldMeshError if the error was caught before any edges were removed
        (this SHOULD always be the case).
    :raises: RuntimeError if a problem was found after we started removing edges
        (this SHOULD never happen).

    remove_edge checks for bridge faces like so:
        * is orig valence > 1
        * is dest valence > 1
        * is edge.face == pair.face

    If all of these are true, remove_edge assumes the edge is bridging between two
    faces, which would be disjoint if the edge were removed.

    A vert may have peninsula edges radiating from it. That is, edges that do not
    split a face. These would be left floating if any bridge edges were removed.

          |
        --*-----FACE
          |

    Remove peninsula edges first to avoid this.
    """
    # examine the vert edges to see if removing vert is safe
    peninsulas = {x for x in vert.edges if x.dest.valence == 1}
    true_edges = set(vert.edges) - peninsulas
    vert_faces = {x.face for x in true_edges}
    if len(true_edges) != len(vert_faces):
        raise ManifoldMeshError("would create non-manifold mesh")

    # remove peninsula edges then others.
    for edge in peninsulas:
        remove_edge(mesh, edge)
    try:
        for edge in vert.edges:
            if edge.face in mesh.faces:
                remove_edge(mesh, edge)
            else:
                remove_edge(mesh, edge.pair)
    except ManifoldMeshError:
        raise RuntimeError(
            "Unexpected error. This should have been caught at the beginning of the "
            "function. A bridge edge was found (e.i., we discovered that this vert "
            "cannot be removed), but some vert edges may have been removed before this "
            "was realized. We've found a bug in the module."
        )


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
            remove_edge(mesh, edge)
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
            raise ValueError("face cannot be determined from orig and dest")

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
    """Add a vert to the middle of an edge."""
    new_vert = Vert(**inherit_kwargs(edge.orig, edge.dest, **vert_kwargs))
    for orig, dest in ((edge.dest, new_vert), (new_vert, edge.orig)):
        new_edge = insert_edge(mesh, orig, dest, edge.face)
        new_edge.fill_from(edge.pair)
        new_edge.pair.fill_from(edge)
    remove_edge(mesh, edge)
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
