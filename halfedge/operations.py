#!/usr/bin/env python3
""" functions which insert, remove, or split edges from a HalfEdges instance
created: 181121 13:43:59

"""

from typing import Generator, Set

from .classes import Edge, Face, HalfEdges, ManifoldMeshError, Vert


def full_edges_only(edges: Set[Edge]) -> Generator[Edge, Set[Edge], None]:
    """Edges where edge.pair is in input set.
    
    Yields either edge or pair where BOTH edge and pair are in input set.
    
    E.g.,
    `iter_full(mesh.interior_edges)` would yield either (half)edge OR pair where
    both edge and pair are face edges.

    TODO: this looks suspicious for side effects
    """
    while edges:
        half = edges.pop()
        if half.pair in edges:
            yield half


def remove_edge(mesh: HalfEdges, edge: Edge) -> None:
    """Cut an edge out of the mesh.

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
    if edge not in mesh.edges:
        raise ValueError("edge {} does not exist in mesh".format(edge.sn))

    if edge in tuple(edge.pair.face_edges)[2:-1]:
        raise ManifoldMeshError("would create non-manifold mesh")

    for half in edge, edge.pair:
        half.face.assign_new_sn()
        half.orig.assign_new_sn()
        half.orig.edge = half.pair.next
        half.face.edge = half.next.next
        mesh.edges.remove(half)

    if edge.face != edge.pair.face:
        for face_edge in edge.face_edges:
            face_edge.face = edge.pair.face

    pair_prev = edge.pair.prev
    edge.prev.next = edge.pair.next
    pair_prev.next = edge.next


def remove_vert(mesh: HalfEdges, vert: Vert) -> None:
    """Remove all edges around a vert."""
    edges = tuple(vert.edges)
    face_splitting_edges = [x for x in edges if x.dest.valence > 1]
    if len({x.face for x in edges}) < len(face_splitting_edges):
        raise ManifoldMeshError(
            "Popping this vert would create non-manifold mesh."
            " One of this vert's edges is a bridge edge."
        )

    for edge in edges:
        if edge.face in mesh.holes:
            remove_edge(mesh, edge.pair)
        else:
            remove_edge(mesh, edge)


def remove_face(mesh: HalfEdges, face: Face) -> None:
    """Remove all edges around a face."""
    edges = tuple(face.edges)
    potential_bridges = [x for x in edges if x.orig.valence > 2]
    if len({x.pair.face for x in edges}) < len(potential_bridges):
        raise ManifoldMeshError(
            "Popping this face would create non-manifold mesh."
            " One of this faces's edges is a bridge edge."
        )

    for edge in edges:
        remove_edge(mesh, edge)


def insert_edge(mesh: HalfEdges, face: Face, orig: Vert, dest: Vert) -> None:
    """Split a face by inserting a new edge between two verts.

    :face: face to split
    :orig: origin of new edge
    :dest: destination of new edge

    Edge face is created.
    Pair face is retained.

    """
    pair_next = next(x for x in orig.edges if x.face is face)
    pair = Edge(orig=dest, next=pair_next, fill_from=pair_next)

    try:
        edge_next = next(x for x in dest.edges if x.face is face)
        if edge_next.next == pair_next or pair_next.next == edge_next:
            raise ManifoldMeshError("overwriting existing edge")
    except AttributeError:
        edge_next = pair
    edge = Edge(orig=orig, next=edge_next, fill_from=edge_next)

    orig.assign_new_sn()
    dest.assign_new_sn()

    dest.edge = pair
    face.assign_new_sn()
    face.edge = pair

    edge.pair, pair.pair = pair, edge
    edge_prev, pair_prev = pair_next.prev, edge_next.prev
    edge_prev.next, pair_prev.next = edge, pair

    if dest.valence > 1:
        edge_face = Face(edge=edge)
        for face_edge in edge.face_edges:
            face_edge.face = edge_face

    mesh.edges.update((edge, pair))


# def insert_vert(mesh: HalfEdges, face: Face, orig: Vert, vert: Vert) -> None:
#     """Knit a new vert into the mesh by adding edge between orig and new."""
#     insert_edge(mesh, face, orig, vert)
#     return
#     orig.assign_new_sn()
#     face.assign_new_sn()
#     vert.assign_new_sn()  # put new vert at the end of the "list"
#
#     next_edge = next(x for x in orig.edges if x.face is face)
#
#     edge = Edge(fill_from=next_edge)
#     pair = Edge(fill_from=next_edge)
#
#     vert.edge = pair
#
#     pair.orig = vert
#     edge.pair, pair.pair = pair, edge
#     next_edge.prev.next, edge.next, pair.next = edge, pair, next_edge
#
#     mesh.edges.update((edge, pair))


def _replace_edge(mesh: HalfEdges, edge: Edge) -> Edge:
    """Replace an edge completely.

    This goes farther than just replacing the serial number. This will prevent
    a half edge from being altered twice (once from each side). When an edge
    is, for instance, split, the old edge should be destroyed, so `for edge in
    edges: alter(edge)` constructions will raise a ValueError if you don't take
    care to only alter an edge once.

    This beats silently double-splitting edges.
    """
    new_edge = Edge(fill_from=edge)
    edge.orig.edge = new_edge
    edge.pair.pair = new_edge
    edge.face.edge = new_edge
    edge.prev.next = new_edge
    mesh.edges.add(new_edge)
    mesh.edges.remove(edge)
    return new_edge


def split_edge(mesh: HalfEdges, edge: Edge, vert: Vert) -> None:
    """Add a vert to the middle of an edge."""
    edge = _replace_edge(mesh, edge)
    edge.pair = _replace_edge(mesh, edge.pair)

    edge.face.assign_new_sn()
    edge.pair.face.assign_new_sn()
    vert.assign_new_sn()

    edge_next = Edge(orig=vert, face=edge.face, next=edge.next, fill_from=edge)
    pair_next = Edge(
        orig=vert, face=edge.pair.face, next=edge.pair.next, fill_from=edge.pair
    )

    edge.pair.next = pair_next
    edge.next = edge_next
    vert.edge = edge_next

    # self._update_pairs_after_split(edge)
    pair = edge.pair

    pair.pair = edge.next
    edge.next.pair = pair

    edge.pair = pair.next
    pair.next.pair = edge

    mesh.edges.update((edge_next, pair_next))
