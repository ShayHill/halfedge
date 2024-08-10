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

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any, Iterable, Sequence, TypeVar

from paragraphs import par

from halfedge.half_edge_elements import Edge, Face, ManifoldMeshError, Vert

if TYPE_CHECKING:
    from halfedge.type_attrib import Attrib

_TBlindHalfEdges = TypeVar("_TBlindHalfEdges", bound="BlindHalfEdges")


class BlindHalfEdges:
    """Half-edge structure with no lookups."""

    def __init__(self, edges: set[Edge] | None = None) -> None:
        """Initialize edges with optionally provided Edge instances."""
        if edges is None:
            self.edges: set[Edge] = set()
        else:
            self.edges = edges

    def new_vert(self, *attributes: Attrib[Any], edge: Edge | None = None) -> Vert:
        """Create a new Vert instance.

        :param attributes: attributes to add to the Vert instance
        :param edge: an Edge instance to assign to the Vert instance
        :return: the new Vert instance
        """
        return Vert(*attributes, mesh=self, edge=edge)

    # determine why an edge needs to know its mesh
    def new_edge(
        self,
        *attributes: Attrib[Any],
        orig: Vert | None = None,
        pair: Edge | None = None,
        face: Face | None = None,
        next: Edge | None = None,
        prev: Edge | None = None,
    ) -> Edge:
        """Create a new Edge instance.

        :param attributes: attributes to add to the Edge instance
        :param orig: optionally pass origin Vert instance
        :param pair: optionally pass pair Edge instance
        :param face: optionally pass Face instance
        :param next: optionally pass next Edge instance
        :param prev: optionally pass previous Edge instance
        :return: the new Edge instance

        The Edge instance will not be inserted into the mesh.
        """
        return Edge(
            *attributes,
            mesh=self,
            orig=orig,
            pair=pair,
            face=face,
            next=next,
            prev=prev,
        )

    def new_face(self, *attributes: Attrib[Any], edge: Edge | None = None) -> Face:
        """Create a new Face instance.

        :param attributes: attributes to add to the Face instance
        :param edge: an Edge instance to assign to the Face instance
        :return: the new Face instance
        """
        return Face(*attributes, mesh=self, edge=edge)

    def new_hole(self, *attributes: Attrib[Any], edge: Edge | None = None) -> Face:
        """Create a new Face instance and mark it as a hole.

        :param attributes: attributes to add to the Face instance
        :param edge: an Edge instance to assign to the Face instance
        :return: the new Face instance
        """
        return Face(*attributes, mesh=self, edge=edge, is_hole=True)

    def create_face_edges(self, face_verts: Iterable[Vert], face: Face) -> list[Edge]:
        """Create edges around a face defined by vert indices.

        :param face_verts: an iterable of Vert instances
        :param face: the Face instance to which the edges will belong
        :return: a list of the Edge instances created
        """
        new_edges = [self.new_edge(orig=vert, face=face) for vert in face_verts]
        for idx, edge in enumerate(new_edges):
            new_edges[idx - 1].next = edge
        return new_edges

    def find_pairs(self) -> None:
        """Match edge pairs, where possible."""
        endpoints2edge = {(edge.orig, edge.dest): edge for edge in self.edges}
        for edge in self.edges:
            with suppress(KeyError):
                edge.pair = endpoints2edge[(edge.dest, edge.orig)]

    def infer_holes(self) -> None:
        """Fill in missing hole faces where unambiguous.

        :raise ManifoldMeshError: if holes touch at corners. If this happens, holes
        are ambiguous.

        Create pairs for unpaired edges. Try to connect into hole faces.

        This will be most applicable to a 2D mesh defined by positive space (faces,
        not holes). The halfedge data structure requires a manifold mesh, which among
        other things means that every edge must have a pair. This can be accomplished
        by creating a hole face around the positive space (provided the boundary of the
        space is contiguous).

        This function can also fill in holes inside the mesh.
        """
        hole_edges: set[Edge] = set()
        for edge in self.edges:
            with suppress(AttributeError):
                _ = edge.pair
                continue
            hole_edges.add(self.new_edge(orig=edge.dest, pair=edge))

        orig2hole_edge = {x.orig: x for x in hole_edges}
        if len(orig2hole_edge) < len(hole_edges):
            msg = par(
                """Multiple holes edges start at the same vertex. Ambiguous 'next' in
                inferred pair edge. Inferred holes probably meet at corner."""
            )
            raise ManifoldMeshError(msg)

        while orig2hole_edge:
            _, edge = next(iter(orig2hole_edge.items()))
            edge.face = self.new_hole()
            while edge.dest in orig2hole_edge:
                edge.next = orig2hole_edge.pop(edge.dest)
                edge.next.face = edge.face
                edge = edge.next
        self.edges.update(hole_edges)

    @classmethod
    def from_vlfi(
        cls: type[_TBlindHalfEdges],
        vl: Sequence[Vert],
        fi: Iterable[tuple[int, ...]],
        hi: Iterable[tuple[int, ...]] | None = None,
    ) -> _TBlindHalfEdges:
        """Create a set of half edges from a vertex list and face index.

        :param vl: (vertex list) a seq of vertices
        [(12.3, 42.02, 4.2), (23.1, 3.55, 3.2) ...]

        :param fi: (face index) an iterable of face indices (indices to vl)
        [(0, 1, 3), (4, 2, 5, 7) ...]

        :param hi: (hole index) optionally provide empty faces (same format as fi)
        that will be used to pair all edges

        :return: a BlindHalfEdges instance

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
        vr = [tuple(vl[x] for x in y) for y in fi]
        hr = [tuple(vl[x] for x in y) for y in hi]

        mesh = cls()
        for vert in vl:
            vert.mesh = mesh
        for face_verts in vr:
            mesh.edges.update(mesh.create_face_edges(face_verts, mesh.new_face()))
        for face_verts in hr:
            mesh.edges.update(mesh.create_face_edges(face_verts, mesh.new_hole()))
        mesh.find_pairs()
        mesh.infer_holes()
        return mesh
