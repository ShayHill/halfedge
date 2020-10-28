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
            * empty mesh: a new Hole
            * one vert on a face: that face
            * both verts on same face: that face
        """
        if not self.edges:
            return Hole()
        shared_faces = set(orig.faces) | set(dest.faces)
        if len(shared_faces) == 1:
            return shared_faces.pop()
        raise ValueError("face cannot be determined from orig and dest")

    def insert_edge(
        self, orig: Vert, dest: Vert, face: Optional[Face] = None, **edge_kwargs: Any,
    ) -> Edge:
        """
        Insert a new edge between two verts.

        :orig: origin of new edge
        :dest: destination of new edge
        :face: edge will lie on or split face
        :returns: newly inserted edge

        Edge face is created.
        Pair face is retained.

        This will only split the face if both orig and dest are new Verts. This function
        will connect:

            * two existing Verts on the same face
            * an existing Vert to a new vert inside the face
            * a new vert inside the face to an existing vert
            * two new verts to create a floating edge in an empty mesh.
        """
        if face is None:
            face = self._infer_face(orig, dest)

        face_edges = face.edges
        orig2edge = {x.orig: x for x in face_edges}
        dest2edge = {x.dest: x for x in face_edges}

        if set(face.verts) & {orig, dest} != self.verts & {orig, dest}:
            raise ManifoldMeshError("orig or dest in mesh but not on given face")

        if getattr(orig, "edge", None) and dest in orig.neighbors:
            raise ManifoldMeshError("overwriting existing edge")

        if not set(face.verts) & {orig, dest} and face in self.faces:
            # TODO: test adding edge to an empty mesh
            raise ManifoldMeshError("adding floating edge to existing face")

        edge = Edge(*face_edges, orig=orig, **edge_kwargs)
        pair = Edge(*face_edges, orig=dest, pair=edge, **edge_kwargs)
        edge.next = orig2edge.get(dest, pair)
        edge.prev = dest2edge.get(orig, pair)
        pair.next = orig2edge.get(orig, edge)
        pair.prev = dest2edge.get(dest, edge)

        _face_edges(face, pair)
        if len(set(face.verts) & {orig, dest}) == 2:
            _face_edges(Face(face), edge)

        self.edges.update({edge, pair})
        return edge

    def remove_edge(self, edge: Edge) -> None:
        """
        Cut an edge out of the mesh.

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
        pair = edge.pair

        if edge not in self.edges:
            raise ValueError("edge {} does not exist in mesh".format(edge.sn))

        if edge.orig.valence > 1 and edge.dest.valence > 1 and edge.face == pair.face:
            raise ManifoldMeshError("would create non-manifold mesh")

        # make sure orig and dest do not point to this edge (if there's another option)
        edge.next.orig = edge.next.orig
        pair.next.orig = pair.next.orig

        # set all faces equal to pair.face
        for edge_ in (x for x in edge.face_edges if x not in (edge, edge.pair)):
            edge_.face = pair.face

        # disconnect from previous edges
        edge.prev.next = pair.next
        pair.prev.next = edge.next
        self.edges -= {edge, pair}

    def remove_vert(self, vert: Vert) -> None:
        """
        Remove all edges around a vert.

        :raises: ManifoldMeshError if the error was caught before any edges were removed
            (this SHOULD always be the case).
        :raises: RuntimeError if a problem was found after we started removing edges
            (this SHOULD never happen).

        remove_edge checks for bridge faces like so:
            * is orig valence > 1
            * is dest valence > 1
            * is edge.face == pair.face

        If all of these are true, remove_edge assumes the edge is bridging between two
        faces, which would be disjoint if the edge were removed.

        A vert may have peninsula edges radiating from it. That is, edges that do not
        split a face. These would be left floating if any bridge edges were removed.

              |
            --*-----FACE
              |

        Remove peninsula edges first to avoid this.
        """
        # examine the vert edges to see if removing vert is safe
        peninsulas = {x for x in vert.edges if x.dest.valence == 1}
        true_edges = set(vert.edges) - peninsulas
        vert_faces = {x.face for x in true_edges}
        if len(true_edges) != len(vert_faces):
            raise ManifoldMeshError("would create non-manifold mesh")

        # remove peninsula edges then others.
        for edge in peninsulas:
            self.remove_edge(edge)
        try:
            for edge in vert.edges:
                if edge.face in self.faces:
                    self.remove_edge(edge)
                else:
                    self.remove_edge(edge.pair)
        except ManifoldMeshError:
            raise RuntimeError(
                "Unexpected error. This should have been caught at the beginning of the "
                "function. A bridge edge was found (e.i., we discovered that this vert "
                "cannot be removed), but some vert edges may have been removed before this "
                "was realized. We've found a bug in the module."
            )