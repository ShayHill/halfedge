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

# TODO: test element-switching from Base object (e.g., a different Edge class)

from __future__ import annotations

from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Set,
    TYPE_CHECKING,
    Tuple,
    TypeVar,
    Type,
    Generic,
)

from . import half_edge_elements
from .half_edge_elements import ManifoldMeshError, Vert, Edge, Face

if TYPE_CHECKING:
    from .half_edge_elements import Vert, Edge, Face, _V, _E, _F


_TBlindHalfEdges = TypeVar("_TBlindHalfEdges", bound="BlindHalfEdges")
_TVert = TypeVar("_TVert", bound="Vert")

_V = TypeVar("_V", bound="Vert")
_E = TypeVar("_E", bound="Edge")
_F = TypeVar("_F", bound="Face")


class BlindHalfEdges(Generic[_V, _E, _F]):
    hole_type = half_edge_elements.Hole

    def __init__(self, edges: Optional[Set[_E]] = None) -> None:
        if edges is None:
            self.edges = set()
        else:
            self.edges = edges

    @classmethod
    @property
    def vert_type(cls) -> Type[Vert[_V, _E, _F]]:
        return Vert[_V, _E, _F]

    @classmethod
    @property
    def edge_type(cls) -> Type[Edge[_V, _E, _F]]:
        return Edge[_V, _E, _F]

    @classmethod
    @property
    def face_type(cls) -> Type[Face[_V, _E, _F]]:
        return Face[_V, _E, _F]

    def _infer_holes(self) -> None:
        """
        Fill in missing hole faces where unambiguous.

        :param edges: Edge instances
        :returns: input edges plus hole edges
        :raises: Manifold mesh error if holes touch at corners. If this happens, holes
        are ambiguous.

        Create pairs for unpaired edges. Try to connect into hole faces.

        This will be most applicable to a 2D mesh defined by positive space (faces,
        not holes). The halfedge data structure requires a manifold mesh, which among
        other things means that every edge must have a pair. This can be accomplished
        by creating a hole face around the positive space (provided the boundary of the
        space is contiguous).

        This function can also fill in holes inside the mesh.
        """
        hole_edges = {
            self.edge_type(orig=x.dest, pair=x)
            for x in self.edges
            if not hasattr(x, "pair")
        }
        orig2hole_edge = {x.orig: x for x in hole_edges}

        if len(orig2hole_edge) < len(hole_edges):
            raise ManifoldMeshError(
                "Ambiguous 'next' in inferred pair edge."
                " Inferred holes probably meet at corner."
            )

        while orig2hole_edge:
            orig, edge = next(iter(orig2hole_edge.items()))
            edge.face = self.hole_type()
            while edge.dest in orig2hole_edge:
                edge.next = orig2hole_edge.pop(edge.dest)
                edge.next.face = edge.face
                edge = edge.next
        self.edges.update(hole_edges)

    def _create_face_edges(self, face_verts: Iterable[Vert], face: Face) -> List[Edge]:
        """Create edges around a face defined by vert indices."""
        new_edges = [self.edge_type(orig=vert, face=face) for vert in face_verts]
        for idx, edge in enumerate(new_edges):
            new_edges[idx - 1].next = edge
        return new_edges

    def _find_pairs(self) -> None:
        """Match edge pairs, where possible."""
        endpoints2edge = {(edge.orig, edge.dest): edge for edge in self.edges}
        for edge in self.edges:
            try:
                edge.pair = endpoints2edge[(edge.dest, edge.orig)]
            except KeyError:
                continue

    @classmethod
    def from_vlvi(
        cls: Type[_TBlindHalfEdges],
        vl: List[Any],
        fi: Set[Tuple[int, ...]],
        hi: Optional[Set[Tuple[int, ...]]] = None,
        attr_name: str = "coordinate",
    ) -> _TBlindHalfEdges:
        """A set of half edges from a vertex list and vertex index.

        :param vl: (vertex list) a seq of vertices
        [(12.3, 42.02, 4.2), (23.1, 3.55, 3.2) ...]

        :param fi: (face index) a seq of face indices (indices to vl)
        [(0, 1, 3), (4, 2, 5, 7) ...]

        :param hi: (hole index) optionally provide empty faces (same format as fi)
        that will be used to pair all edges

        :param attr_name: optionally override the default attribute name: "coordinate"

        Presumably, the vl is a list of coordinate values. These will be assigned,
        by default, to ``vert_instance.coordinate.``

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
        hi = hi or set()
        vl = [cls.vert_type(**{attr_name: x}) for x in vl]
        vr = [tuple(vl[x] for x in y) for y in fi]
        hr = [tuple(vl[x] for x in y) for y in hi]

        mesh = cls()
        for face_verts in vr:
            mesh.edges.update(mesh._create_face_edges(face_verts, mesh.face_type()))
        for face_verts in hr:
            mesh.edges.update(mesh._create_face_edges(face_verts, mesh.hole_type()))
        mesh._find_pairs()
        mesh._infer_holes()
        return mesh
