# !/usr/bin/env python3
# _*_ coding: utf-8 _*_
# Last modified: 181204 13:07:13
"""A half-edges data container with view methods.

A simple container for a list of half edges. Provides lookups and a serial
number for each mesh element (Vert, Edge, Face, or Hole).

This is a typical halfedges data structure. Exceptions:

    * Faces and Holes are two different (but identical except in name) types. This is
      to simplify working with 2D meshes. You can
          - define a 2d mesh with triangles
          - explicitly or algorithmically define holes to keep the mesh manifold
          - ignore the holes after that. They won't be returned when, for instance,
            iterating over mesh faces.
      Boundary verts, and boundary edges are identified by these holes, but that all
      happens internally, so the holes can again be ignored for most things.

    * Orig, pair, face, and next assignments are mirrored, so a.pair = b will
      set a.pair = b and b.pair = a. This makes edge insertion, etc. cleaner,
      but the whole thing is still easy to break. Hopefully, I've provided enough
      insertion / removal code to get you over the pitfalls. Halfedges is clever when
      it's all built, but a lot has to be temporarily broken down to transform the
      mesh. All I can say is, write a lot of tests if you want to extend the
      insertion / removal methods here.

This module is all the lookups. Transformations elsewhere.

# 2006 June 05
# 2012 September 30
"""
from operator import attrgetter
from typing import Dict, List, Optional, Set, Tuple
from .constructors import BlindHalfEdges

from . import half_edge_elements


class StaticHalfEdges(BlindHalfEdges):
    """
    Basic half edge lookups.

    Some properties require a manifold mesh, but the Edge type does support
    explicitly defined holes. Holes provide enough information to pair and link all
    half edges, but will be ignored in any "for face in" constructs.
    """

    def __init__(self, edges: Optional[Set[half_edge_elements.Edge]] = None) -> None:
        super().__init__(edges)

    @property
    def verts(self) -> Set[half_edge_elements.Vert]:
        """Look up all verts in mesh."""
        return {x.orig for x in self.edges}

    @property
    def faces(self) -> Set[half_edge_elements.Face]:
        """Look up all faces in mesh."""
        return {
            x.face
            for x in self.edges
            if not isinstance(x.face, half_edge_elements.Hole)
        }

    @property
    def holes(self) -> Set[half_edge_elements.Hole]:
        """Look up all holes in mesh."""
        return {
            x.face for x in self.edges if isinstance(x.face, half_edge_elements.Hole)
        }

    @property
    def all_faces(self) -> Set[half_edge_elements.Hole]:
        """ Look up all faces and holes in mesh """
        return {x.face for x in self.edges}

    @property
    def elements(self) -> Set[half_edge_elements._MeshElementBase]:
        """All elements in mesh"""
        return self.verts | self.edges | self.faces | self.holes

    @property
    def last_issued_sn(self) -> int:
        """Look up the last serial number issued to any mesh element."""
        return next(iter(self.edges)).last_issued_sn

    @property
    def boundary_edges(self) -> Set[half_edge_elements.Edge]:
        """Look up edges on holes."""
        return {x for x in self.edges if isinstance(x.face, half_edge_elements.Hole)}

    @property
    def boundary_verts(self) -> Set[half_edge_elements.Vert]:
        """Look up all verts on hole boundaries."""
        return {x.orig for x in self.boundary_edges}

    @property
    def interior_edges(self):
        """Look up edges on faces."""
        return self.edges - self.boundary_edges

    @property
    def interior_verts(self) -> Set[half_edge_elements.Vert]:
        """Look up all verts not on hole boundaries."""
        return self.verts - self.boundary_verts

    @property
    def vl(self) -> List[half_edge_elements.Vert]:
        """ Sorted list of verts """
        return sorted(self.verts, key=attrgetter("sn"))

    @property
    def el(self) -> List[half_edge_elements.Edge]:
        """ Sorted list of edges """
        return sorted(self.edges, key=attrgetter("sn"))

    @property
    def fl(self) -> List[half_edge_elements.Face]:
        """ Sorted list of faces """
        return sorted(self.faces, key=attrgetter("sn"))

    @property
    def hl(self) -> List[half_edge_elements.Hole]:
        """ Sorted list of holes """
        return sorted(self.holes, key=attrgetter("sn"))

    @property
    def _vert2list_index(self) -> Dict[half_edge_elements.Vert, int]:
        """self.vl mapped to list indices."""
        return {vert: cnt for cnt, vert in enumerate(self.vl)}

    @property
    def ei(self) -> Set[Tuple[int, int]]:
        """
        Edges as a set of paired vert indices.

        :returns: {(0, 1), (2, 3), (1, 4), ...}
        """
        v2i = self._vert2list_index
        return {(v2i[edge.orig], v2i[edge.dest]) for edge in self.edges}

    @property
    def fi(self) -> Set[Tuple[int, ...]]:
        """
        Faces as a set of tuples of vl indices.

        :returns: {(0, 1, 2, 3), (1, 0, 4, 5) ...}
        """
        v2i = self._vert2list_index
        return {tuple(v2i[x] for x in face.verts) for face in self.faces}

    @property
    def hi(self) -> Set[Tuple[int, ...]]:
        """
        Holes as a set of tuples of vl indices.

        :returns: {(0, 1, 2, 3), (1, 0, 4, 5) ...}
        """
        v2i = self._vert2list_index
        return {tuple(v2i[x] for x in hole.verts) for hole in self.holes}
