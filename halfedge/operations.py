#!/usr/bin/env python3
""" functions which insert, remove, or split edges from a HalfEdges instance
created: 181121 13:43:59

"""

from typing import Generator, Set

from .classes import Edge, Face, HalfEdges, ManifoldMeshError, Vert

from contextlib import suppress

from .validations import validate_mesh

# def remove_edge(mesh: HalfEdges, edge: Edge) -> None:
#     """
#     Remove whole edge and update pointers.
#
#     :param mesh:
#     :param edge:
#     :return:
#     """
#     full_edge = {edge, edge.pair}
#     verts = {x.vert for x in full_edge}
#     faces = {x.face for x in full_edge}
#     mesh.edges -= full_edge
#     for elem in verts | faces:
#         elem.assign_new_sn()
#     for elem in (x for x in verts | faces if x.edge in full_edge):
#         del(elem._edge)

# TODO: delete below
# for half in edge, edge.pair:
#     h
# for half in edge, edge.pair:
#     half.face.assign_new_sn()
#     half.orig.assign_new_sn()
#     mesh.edges.remove(half)
#
# if edge.face != edge.pair.face:
#     for face_edge in edge.face_edges:
#         face_edge.face = edge.pair.face


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

    edge.next.orig = edge.next.orig
    edge.pair.next.orig = edge.pair.next.orig

    face = edge.pair.face
    for edge_ in (x for x in edge.face_edges if x not in (edge, edge.pair)):
        edge_.face = face

    pair_prev = edge.pair.prev
    edge.prev.next = edge.pair.next
    pair_prev.next = edge.next

    full_edge = {edge, edge.pair}
    mesh.edges -= full_edge


def remove_vert(mesh: HalfEdges, vert: Vert) -> None:
    """Remove all edges around a vert."""
    edges = set(vert.edges)
    face_splitting_edges = [x for x in edges if x.dest.valence > 1]
    if len({x.face for x in edges}) < len(face_splitting_edges):
        raise ManifoldMeshError(
            "Popping this vert would create non-manifold mesh."
            " One of this vert's edges is a bridge edge."
        )

    # can cause manifold mesh error if removed in wrong order. Try different sequences.
    len_edges = len(edges)
    while edges:
        for edge in tuple(edges):
            with suppress(ManifoldMeshError):
                if edge.face in mesh.holes:
                    remove_edge(mesh, edge.pair)
                else:
                    remove_edge(mesh, edge)
                edges.remove(edge)
        if len(edges) == len_edges:
            raise ManifoldMeshError(
                "cannot find a safe edge to remove. "
                "The 'face_splitting_edges' check earlier in the function "
                "should have prevented this from happening. This is a bug."
            )
        len_edges = len(edges)


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


from .validations import validate_mesh


def insert_edge(mesh: HalfEdges, face: Face, orig: Vert, dest: Vert) -> None:
    """Split a face by inserting a new edge between two verts.

    :face: face to split
    :orig: origin of new edge
    :dest: destination of new edge

    Edge face is created.
    Pair face is retained.

    """
    face_verts = face.verts
    face_edges = face.edges
    mesh_verts = mesh.verts
    orig2edge = {x.orig: x for x in face_edges}
    dest2edge = {x.dest: x for x in face_edges}

    for v in orig, dest:
        # TODO: test this exception
        if (v in face_verts) == (v not in mesh_verts):
            raise ManifoldMeshError("orig or dest in mesh but not on given face")

    if getattr(orig, "edge", None) and dest in {x.dest for x in orig.edge.vert_edges}:
        raise ManifoldMeshError("overwriting existing edge")

    edge = Edge(orig=orig)
    pair = Edge(orig=dest, pair=edge, face=face)
    edge.next = orig2edge.get(dest, pair)
    edge.prev = dest2edge.get(orig, pair)
    pair.next = orig2edge.get(orig, edge)
    pair.prev = dest2edge.get(dest, edge)

    # orig.assign_new_sn()
    # dest.assign_new_sn()
    # face.assign_new_sn()

    if orig in mesh_verts and dest in mesh_verts:
        edge.face = Face(fill_from=face)
        for e in edge.face_edges[1:]:
            e.face = edge.face
    else:
        edge.face = face

    mesh.edges.update({edge, pair})


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

    This goes further than just replacing the serial number. This will prevent
    a half edge from being altered twice (once from each side). When an edge
    is, for instance, split, the old edge should be destroyed, so `for edge in
    edges: alter(edge)` constructions will raise a ValueError if you don't take
    care to only alter an edge once.

    This beats silently double-splitting edges.
    """
    new_edge = Edge(fill_from=edge)
    edge.pair.pair = new_edge
    edge.prev._next = new_edge
    mesh.edges.add(new_edge)
    mesh.edges.remove(edge)
    return new_edge


def split_edge(mesh: HalfEdges, edge: Edge, vert: Vert) -> None:
    """Add a vert to the middle of an edge."""
    edge = _replace_edge(mesh, edge)
    edge.pair = _replace_edge(mesh, edge.pair)

    edge_next = Edge(orig=vert, face=edge.face, next=edge.next, fill_from=edge)
    pair_next = Edge(
        orig=vert, face=edge.pair.face, next=edge.pair.next, fill_from=edge.pair
    )

    edge.pair.next = pair_next
    edge.next = edge_next

    # self._update_pairs_after_split(edge)
    pair = edge.pair

    pair.pair = edge.next
    edge.next.pair = pair

    edge.pair = pair.next
    pair.next.pair = edge

    edge_next.orig.edge = edge_next
    edge_next.face.edge = edge_next
    pair_next.orig.edge = pair_next
    pair_next.face.edge = pair_next

    mesh.edges.update((edge_next, pair_next))
