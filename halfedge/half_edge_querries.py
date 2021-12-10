# !/usr/bin/env python3
# _*_ coding: utf-8 _*_
# Last modified: 181204 13:07:13
"""A half-edges data container with view methods.

A simple container for a list of half edges. Provides lookups and a serial
number for each mesh element (Vert, Edge, or Face).

This module is all the lookups. Transformations elsewhere.

# 2006 June 05
# 2012 September 30
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from .constructors import BlindHalfEdges

if TYPE_CHECKING:
    from .half_edge_elements import Edge, Face, Vert

_V = TypeVar("_V", bound="Vert")
_E = TypeVar("_E", bound="Edge")
_F = TypeVar("_F", bound="Face")


class StaticHalfEdges(Generic[_V, _E, _F], BlindHalfEdges[_V, _E, _F]):
    """
    Basic half edge lookups.

    Some properties require a manifold mesh, but the Edge type does support
    explicitly defined holes. These hole (Face(__is_hole=True) provide enough
    information to pair and link all half edges, but will be ignored in any "for face
    in" constructs.
    """

    def __init__(self, edges: Optional[Set[Edge]] = None) -> None:
        super().__init__(edges)

    @property
    def verts(self) -> Set[Vert]:
        """Look up all verts in mesh."""
        return {x.orig for x in self.edges}

    @property
    def faces(self) -> Set[Face]:
        """Look up all faces in mesh."""
        return {x for x in self.all_faces if not x.is_hole}

    @property
    def holes(self):  # -> Set[hole]: # TODO: update hole return type
        """Look up all holes in mesh."""
        return {x for x in self.all_faces if x.is_hole}

    @property
    def all_faces(self):  # -> Set[hole]: # TODO: update hole return type
        """Look up all faces and holes in mesh"""
        return {x.face for x in self.edges}

    @property
    def elements(self) -> Set[Union[Vert, Edge, Face]]:
        """All elements in mesh"""
        return self.verts | self.edges | self.faces | self.holes

    @property
    def boundary_edges(self) -> Set[Edge]:
        """Look up edges on holes."""
        return {x for x in self.edges if x.face.is_hole}

    @property
    def boundary_verts(self) -> Set[Vert]:
        """Look up all verts on hole boundaries."""
        return {x.orig for x in self.boundary_edges}

    # TODO: make these filter back through self.edges
    @property
    def interior_edges(self):
        """Look up edges on faces."""
        return self.edges - self.boundary_edges

    @property
    def interior_verts(self) -> Set[Vert]:
        """Look up all verts not on hole boundaries."""
        return self.verts - self.boundary_verts

    @property
    def vl(self) -> List[Vert]:
        """ "vertex list" - Sorted list of verts"""
        return sorted(self.verts)

    @property
    def _vert2list_index(self) -> Dict[Vert, int]:
        """self.vl mapped to list indices."""
        return {vert: cnt for cnt, vert in enumerate(self.vl)}

    @property
    def ei(self) -> Set[Tuple[int, int]]:
        """
        "edge indices" - Edges as a set of paired vert indices.

        :returns: {(0, 1), (2, 3), (1, 4), ...}
        """
        v2i = self._vert2list_index
        return {(v2i[edge.orig], v2i[edge.dest]) for edge in self.edges}

    @property
    def fi(self) -> Set[Tuple[int, ...]]:
        """
        "face indices" - Faces as a set of tuples of vl indices.

        :returns: {(0, 1, 2, 3), (1, 0, 4, 5) ...}
        """
        v2i = self._vert2list_index
        return {tuple(v2i[x] for x in face.verts) for face in self.faces}

    @property
    def hi(self) -> Set[Tuple[int, ...]]:
        """
        "hole indices" - holes as a set of tuples of vl indices.

        :returns: {(0, 1, 2, 3), (1, 0, 4, 5) ...}
        """
        v2i = self._vert2list_index
        return {tuple(v2i[x] for x in hole.verts) for hole in self.holes}
