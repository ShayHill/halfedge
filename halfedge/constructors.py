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

# TODO: factor out most or all of constructors module
# TODO: test element-switching from Base object (e.g., a different Edge class)

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union, Hashable

from .half_edge_elements import Edge, Face, Hole, ManifoldMeshError, Vert

VertCastable = Union[Vert, Dict[str, Any], Any]
Vr = Iterable[Iterable[Vert]]


class BlindHalfEdges:
    vert_type = Vert
    edge_type = Edge
    face_type = Face
    hole_type = Hole

    def __init__(self, edges: Optional[Set[Edge]] = None) -> None:
        if edges is None:
            self.edges = set()
        else:
            self.edges = edges

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
    def new_vert(cls, source: VertCastable, attr_name: Optional[str] = None) -> Vert:
        """
        A new Vert instance of cls.vert_type

        :param source: attr value or Vert instance or dict {name: val, name: val}
        :param attr_name: optionally pass source as Vert.attr_name = source
        :return: instance of cls.vert_type

        This one is a little "magical"
        if attr_name is given -> new vert with vert.attr_name = source
        elif source is a Vert -> source vert recast as cls.vert_type
        else source is assumed to be a dict -> cls.vert_type(**source)
        """
        if attr_name is not None:
            return cls.vert_type(**{attr_name: source})
        if isinstance(source, Vert):
            return cls.vert_type(source)
        try:
            return cls.vert_type(**source)
        except TypeError:
            raise NotImplementedError(f"no provision for casting {source} into Vert")

    @classmethod
    def new_verts(
        cls,
        sources: Iterable[VertCastable],
        attr_name: Optional[str] = None,
    ) -> List[Vert]:
        """
        Iteratively call self.new_vert

        :param sources: attr values or Vert instances or dicts {name: val, name: val}
        :param attr_name: optionally pass source as Vert.attr_name = source
        :return: list of instances of cls.vert_type

        This one is a little "magical"
        if attr_name is given -> new verts with vert.attr_name = source
        elif sources are Vert instances -> source verts recast as cls.vert_type
        else sources are assumed to be dicts -> cls.vert_type(**source)
        """
        return [cls.new_vert(x, attr_name) for x in sources]

    @classmethod
    def from_vlvi(
        cls,
        vl: List[VertCastable],
        vi: Set[Tuple[int, ...]],
        hi: Optional[Set[Tuple[int, ...]]] = (),
        attr_name: Optional[str] = None,
    ) -> BlindHalfEdges:
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
        vl = cls.new_verts(vl, attr_name)
        vr = {tuple(vl[x] for x in y) for y in vi}
        hr = {tuple(vl[x] for x in y) for y in hi}

        mesh = cls()
        for face_verts in vr:
            mesh.edges.update(mesh._create_face_edges(face_verts, mesh.face_type()))
        for face_verts in hr:
            mesh.edges.update(mesh._create_face_edges(face_verts, mesh.hole_type()))
        mesh._find_pairs()
        mesh._infer_holes()
        return mesh

