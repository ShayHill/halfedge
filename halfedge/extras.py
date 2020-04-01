#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Transformations, etc. coded for one project, perhaps useful in others

:author: Shay Hill
:created: 3/31/2020

Some of these will require numpy, lots of other deps
"""

# TODO: move this outside to the project folder, too many deps.

from my_geometry import vangle
from my_geometry.lines import gap_corner
from my_halfedge import HalfEdges, Vert, Edge, Face
from typing import Tuple
from my_halfedge.halfedge.validations import validate_mesh
from nptyping import Array


def _get_edge_neighbors(edge: Edge) -> Tuple[Vert, Vert, Vert]:
    """
    Vert before, at, and after face.orig

    :param edge: Edge instance
    :return: three Vert instances
    """
    return (edge.prev.orig, edge.orig, edge.dest)


def _get_edge_gapped(edge: Edge, gap: float) -> Vert:
    """
    A new Vert instance on edge.face at gapped edge.orig

    :param edge: Edge instance
    :return: new Vert instance on edge.face

    Run gap corner for verts on face adjacent to edge.orig
    """
    points = [x.coordinate for x in _get_edge_neighbors(edge)]
    return Vert(coordinate=gap_corner(*points, gap), edge=edge, fill_from=edge.orig)


def bevel(mesh: HalfEdges, gap: float) -> HalfEdges:
    """
    Bevel a mesh, adding a new face for each vert and each edge.

    :param mesh: closed, manifold mesh
    :gap: bevel amount. Faces are beveled by bevel on their surface. This will
        probably not be the width or some consistent multiple of the width of the
        resulting bevel.
    :return: new mesh with original faces plus new face on each edge and vert
    """
    if mesh.holes:
        raise NotImplementedError("behavior not defined for meshes with holes")
    at_sn = mesh.last_issued_sn

    # for each vert map face references to new verts
    vnew = {v: {} for v in mesh.verts}
    enew = {v: {} for v in mesh.verts}
    for face in mesh.faces:
        for edge in face.edges:
            new_vert = _get_edge_gapped(edge, gap)
            vnew[edge.orig][face] = new_vert
            enew[edge.orig][face] = Edge(orig=new_vert)
    for vert in mesh.verts:
        # TODO: might have to reverse faces
        faces = [e.face for e in reversed(vert.edges)]
        edges = [enew[vert][f] for f in faces]
        new_face = Face(edge=edges[0])
        for i, edge in enumerate(edges):
            edge.face = new_face
            edges[i - 1].next = edge
        mesh.edges.update(edges)

    # update to new verts
    cached_endpoints = {
        e: (e.orig, e.dest) for e in (x for x in mesh.edges if x.sn <= at_sn)
    }
    for edge in (x for x in mesh.edges if x.sn <= at_sn):
        edge.orig = vnew[edge.orig][edge.face]

    # a slit face at each edge
    old_edges = [x for x in mesh.edges if x.sn <= at_sn and x.pair.sn < x.sn]
    for edge in old_edges:
        pair = edge.pair
        orig_edge = enew[cached_endpoints[edge][0]][edge.face].prev
        dest_edge = enew[cached_endpoints[pair][0]][pair.face].prev



        new_face = Face()
        new_edges = [
            Edge(orig=edge.dest, pair=edge, face=new_face),
            Edge(orig=edge.orig, pair=orig_edge, face=new_face),
            Edge(orig=pair.dest, pair=pair, face=new_face),
            Edge(orig=pair.orig, pair=dest_edge, face=new_face),
        ]

        kill = False
        if edge.sn in(12, 14) or pair.sn in (12, 14):
            kill = True
        for i, edge_ in enumerate(new_edges):
            new_edges[i - 1].next = edge_
        orig_edge.pair, dest_edge.pair = new_edges[1], new_edges[3]
        edge.pair, pair.pair = new_edges[0], new_edges[2]
        new_face.edge = new_edges[0]
        # if kill is True:
        #     breakpoint()
        mesh.edges.update(new_edges)

    breakpoint()

    validate_mesh(mesh)


    breakpoint()
