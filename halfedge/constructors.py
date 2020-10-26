#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Create HalfEdges instances.

Nothing in this module will ever try to "match" Verts by their coordinate values. The
only Verts that will ever be equal are two references to the identical (by id) Vert
class instance.

This is important when converting from other formats. One cannot simply [Vert(x) for
x in ...] unless identical (by value) x's have already been identified and combined.
For instance, creating a raw mesh by ...

    [[Vert(x) for x in face] for face in cube]

then passing that raw data to mesh_from_vr would create a mesh with 6 faces and
24 (not 8!) Vert instances.
"""


from typing import Any, List, Optional, Set, cast

from .half_edge_elements import Edge, Face, Hole, ManifoldMeshError, Vert
from .half_edge_querries import StaticHalfEdges


def infer_holes(edges: Set[Edge]) -> Set[Edge]:
    """
    Fill in missing hole faces where unambiguous.

    :param edges: Edge instances
    :returns: input edges plus hole edges
    :raises: Manifold mesh error if holes touch at corners. If this happens, holes are
    ambiguous.

    Create pairs for unpaired edges. Try to connect into hole faces.

    This will be most applicable to a 2D mesh defined by positive space (faces,
    not holes). The halfedge data structure requires a manifold mesh, which among
    other things means that every edge must have a pair. This can be accomplished
    by creating a hole face around the positive space (provided the boundary of the
    space is contiguous).

    This function can also fill in holes inside the mesh.
    """
    hole_edges = {Edge(orig=x.dest, pair=x) for x in edges if not hasattr(x, "pair")}
    orig2hole_edge = {x.orig: x for x in hole_edges}

    if len(orig2hole_edge) < len(hole_edges):
        raise ManifoldMeshError(
            "Ambiguous 'next' in inferred pair edge."
            " Inferred holes probably meet at corner."
        )

    while orig2hole_edge:
        orig, edge = next(iter(orig2hole_edge.items()))
        edge.face = Hole()
        while edge.dest in orig2hole_edge:
            edge.next = orig2hole_edge.pop(edge.dest)
            edge.next.face = edge.face
            edge = edge.next
    edges.update(hole_edges)
    return edges


def _create_face_edges(face_verts: List[Vert], face: Face) -> List[Edge]:
    """Create edges around a face defined by vert indices."""
    new_edges = [Edge(orig=vert, face=face) for vert in face_verts]
    for idx, edge in enumerate(new_edges):
        new_edges[idx - 1].next = edge
    return new_edges


def find_pairs(edges: Set[Edge]) -> None:
    """Match edge pairs, where possible."""
    endpoints2edge = {(edge.orig, edge.dest): edge for edge in edges}
    for edge in edges:
        try:
            edge.pair = endpoints2edge[(edge.dest, edge.orig)]
        except KeyError:
            continue


def edges_from_vlvi(
    vl: List[Any], vi: List[List[int]], hi: Optional[List[List[int]]] = None
) -> Set[Edge]:
    """A set of half edges from a vertex list and vertex index.

    :vl (vertex list): a seq of vertices
    [(12.3, 42.02, 4.2), (23.1, 3.55, 3.2) ...]

    :vi (vertex index): a seq of face indices (indices to vl)
    [(0, 1, 3), (4, 2, 5, 7) ...]

    :hi (hole index): empty faces (same format as vi) that will be used to pair
    all edges then sit ignored afterward

    Will attempt to add missing hole edges. This is intended for fields of
    faces on a plane (like a 2D Delaunay triangulation), but the algorithm
    can handle multiple holes (inside and outside) if they are unambiguous.
    One possible ambiguity would be:
     _
    |_|_
      |_|

    Missing edges would have to guess which next edge to follow at the
    shared box corner. The method will fail in such cases.

    ---------------------

    Will silently remove unused verts

    """
    if hi is None:
        hi = set()

    # everything except pairs
    verts = [Vert(coordinate=v) for v in vl]
    edges: Set[Edge] = set()

    for vert_indices in vi | hi:
        face_verts = [verts[idx] for idx in vert_indices]
        if vert_indices in hi:
            edges.update(_create_face_edges(face_verts, Hole()))
        else:
            edges.update(_create_face_edges(face_verts, Face()))

    find_pairs(edges)

    return infer_holes(edges)


def edges_from_vr(
    vr: List[List[Vert]],
    hr: Optional[List[List[Vert]]] = None,
    mesh_type=StaticHalfEdges,
) -> Set[Edge]:
    """A set of half edges from raw mesh information.

    Unlike edges_from_vlvi, this will not create Verts from coordinate (vertex
    list) information. That would involve tests for equality, which are
    problematic with user-defined objects (equal on id) and numpy. Input must
    be a list of lists of Verts. Vert coordinates are not considered here in any
    way.

    :vr (vertex raw): a list of faces, each face a list of Verts
    :hr (holes raw): an optional list of holes, each hole a list of Verts

    Holes will only be used to pair up edges. Will attempt to infer holes if
    none are given.
    """
    if hr is None:
        hr = []

    # everything except pairs
    edges = cast(Set[mesh_type.edge_type], set())
    for hole in hr:
        edges.update(_create_face_edges(hole, mesh_type.hole_type()))
    for face in vr:
        edges.update(_create_face_edges(face, mesh_type.face_type()))

    find_pairs(edges)
    return infer_holes(edges)


def mesh_from_vlvi(
    vl: List[Any], vi: List[List[int]], hi: Optional[List[List[int]]] = None
) -> StaticHalfEdges:
    """A HalfEdges instance from vl, vi, and optionally hi."""
    return StaticHalfEdges(edges_from_vlvi(vl, vi, hi))


def mesh_from_vr(
    vr: List[List[Vert]], hr: Optional[List[List[Vert]]] = None
) -> StaticHalfEdges:
    """A HalfEdges instance from vr and optionally hr."""
    return StaticHalfEdges(edges_from_vr(vr, hr))
