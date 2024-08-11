"""A half-edges data container with view methods.

Extend BlindHalfEdges with lookups.

These are all the halfedge tricks (faces around a vert, edges around a face, etc.)
that do not change the mesh.

:author: Shay Hill
:created: 2006 June 05
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from halfedge.half_edge_constructors import BlindHalfEdges

if TYPE_CHECKING:
    from halfedge.half_edge_elements import Edge, Face, Vert


class StaticHalfEdges(BlindHalfEdges):
    """Basic half edge lookups.

    Some properties require a manifold mesh, but the Edge type does support
    explicitly defined holes. These hole (`Face(is_hole=True)`) provide enough
    information to pair and link all half edges, but will be ignored in any "for face
    in" constructs.
    """

    def __init__(self, edges: set[Edge] | None = None) -> None:
        """Initialize the half-edge mesh.

        :param edges: A set of edges to initialize the mesh with. Default is None.
        """
        super().__init__(edges)

    @property
    def verts(self) -> set[Vert]:
        """Look up all vertices in the mesh.

        :return: A set of vertices in the mesh.
        """
        return {x.orig for x in self.edges}

    @property
    def faces(self) -> set[Face]:
        """Look up all faces in the mesh.

        :return: A set of faces in the mesh that are not holes.
        """
        return {x for x in self.all_faces if not x.is_hole}

    @property
    def holes(self) -> set[Face]:
        """Look up all holes in the mesh.

        :return: A set of faces in the mesh that are holes.
        """
        return {x for x in self.all_faces if x.is_hole}

    @property
    def all_faces(self) -> set[Face]:
        """Look up all faces and holes in the mesh.

        :return: A set of all faces (including holes) in the mesh.
        """
        return {x.face for x in self.edges}

    @property
    def elements(self) -> set[Vert | Edge | Face]:
        """Get all elements in the mesh.

        The elements include vertices, edges, and faces.

        :return: A set of all elements (vertices, edges, and faces) in the mesh.
        """
        return self.verts | self.edges | self.faces

    @property
    def boundary_edges(self) -> set[Edge]:
        """Look up edges on holes.

        :return: A set of edges that are on hole boundaries.
        """
        return {x for x in self.edges if x.face.is_hole}

    @property
    def boundary_verts(self) -> set[Vert]:
        """Look up all vertices on hole boundaries.

        :return: A set of vertices that are on hole boundaries.
        """
        return {x.orig for x in self.boundary_edges}

    @property
    def interior_edges(self) -> set[Edge]:
        """Look up edges on faces.

        :return: A set of edges that are on face boundaries but not on hole boundaries.
        """
        return {x for x in self.edges if not x.face.is_hole}

    @property
    def interior_verts(self) -> set[Vert]:
        """Look up all vertices not on hole boundaries.

        :return: A set of vertices that are not on hole boundaries.
        """
        return self.verts - self.boundary_verts

    @property
    def vl(self) -> list[Vert]:
        """Vertex list - Sorted list of vertices.

        :return: A sorted list of vertices in the mesh.
        """
        return sorted(self.verts)

    @property
    def el(self) -> list[Edge]:
        """Edge list - Sorted list of edges.

        :return: A sorted list of edges in the mesh.
        """
        return sorted(self.edges)

    @property
    def fl(self) -> list[Face]:
        """Face list - Sorted list of faces.

        :return: A sorted list of faces in the mesh.
        """
        return sorted(self.faces)

    @property
    def _vert2list_index(self) -> dict[Vert, int]:
        """Map vertices to their indices in the sorted vertex list.

        :return: A dictionary mapping each vertex to its index in the sorted vertex
            list (self.vl).
        """
        return {vert: cnt for cnt, vert in enumerate(self.vl)}

    @property
    def ei(self) -> set[tuple[int, int]]:
        """Edge indices - Edges as a set of paired vertex indices.

        :return: A set of tuples where each tuple represents an edge as paired vertex
            indices.
        """
        v2i = self._vert2list_index
        return {(v2i[edge.orig], v2i[edge.dest]) for edge in self.edges}

    @property
    def fi(self) -> set[tuple[int, ...]]:
        """Face indices - Faces as a set of tuples of vertex list indices.

        :return: A set of tuples where each tuple represents a face as a sequence of
            vertex indices.
        """
        v2i = self._vert2list_index
        return {tuple(v2i[x] for x in face.verts) for face in self.faces}

    @property
    def hi(self) -> set[tuple[int, ...]]:
        """Hole indices - Holes as a set of tuples of vertex list indices.

        :return: A set of tuples where each tuple represents a hole as a sequence of
            vertex indices.
        """
        v2i = self._vert2list_index
        return {tuple(v2i[x] for x in hole.verts) for hole in self.holes}
