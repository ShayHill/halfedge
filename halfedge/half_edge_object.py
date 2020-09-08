from .half_edge_querries import StaticHalfEdges
from .half_edge_elements import Vert, Edge, Face, Hole, ManifoldMeshError
from typing import Optional, Any


def _face_edges(face: Face, edge: Edge) -> None:
    """
    Add or update face attribute for each edge in edge.face_edges
    """
    for edge_ in edge.face_edges:
        edge_.face = face


class HalfEdges(StaticHalfEdges):
    def _infer_face(self, orig: Vert, dest: Vert) -> Face:
        """
        Infer which face two verts lie on.

        :return: face (if unambiguous) on which verts lie

        Able to infer from:
            * both verts on same face: that face
            * empty mesh: a new Hole
        """
        if not self.edges:
            return Hole()
        shared_faces = set(orig.faces) & set(dest.faces)
        if len(shared_faces) == 1:
            return shared_faces.pop()
        raise ValueError("face cannot be determined from orig and dest")

    def insert_edge(
        self, orig: Vert, dest: Vert, face: Optional[Face] = None, **edge_kwargs: Any,
    ) -> Edge:
        """
        Insert a new edge between two verts.

        :param orig: origin of new edge
        :param dest: destination of new edge
        :param face: edge will lie on or split face (will infer in unambiguous)
        :param edge_kwargs: set attributes for new edge
        :returns: newly inserted edge

        :raises: ValueError if no face is given and face is ambiguous
        :raises: ManifoldMeshError if
            * overwriting existing edge
            * any vert in mesh but not on face
            * orig and dest are the same
            * edge is not connected to mesh (and mesh is not empty)

        Edge face is created.
        Pair face is retained.

        This will only split the face if both orig and dest are new Verts. This function
        will connect:

            * two existing Verts on the same face
            * an existing Vert to a new vert inside the face
            * a new vert inside the face to an existing vert
            * two new verts to create a floating edge in an empty mesh.

        Passes attributes:

            * shared face.edges attributes passed to new edge
            * edge_kwargs passed to new edge
            * face attributes passed to new face if face is split
        """
        if face is None:
            face = self._infer_face(orig, dest)

        face_edges = face.edges
        face_verts = face.verts
        orig2edge = {x.orig: x for x in face_edges}
        dest2edge = {x.dest: x for x in face_edges}

        if getattr(orig, "edge", None) and dest in orig.neighbors:
            raise ManifoldMeshError("overwriting existing edge")

        if set(face.verts) & {orig, dest} != self.verts & {orig, dest}:
            raise ManifoldMeshError("orig or dest in mesh but not on given face")

        if orig == dest:
            raise ManifoldMeshError('orig and dest are the same')

        if not set(face.verts) & {orig, dest} and face in self.faces:
            raise ManifoldMeshError("adding floating edge to existing face")

        edge = Edge(*face_edges, orig=orig, **edge_kwargs)
        pair = Edge(*face_edges, orig=dest, pair=edge, **edge_kwargs)
        edge.next = orig2edge.get(dest, pair)
        edge.prev = dest2edge.get(orig, pair)
        pair.next = orig2edge.get(orig, edge)
        pair.prev = dest2edge.get(dest, edge)

        _face_edges(face, pair)
        if len(set(face_verts) & {orig, dest}) == 2:
            _face_edges(Face(face), edge)

        self.edges.update({edge, pair})
        return edge
