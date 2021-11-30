from contextlib import suppress
from typing import Any, Optional, Set, Tuple, Union

from .half_edge_elements import Edge, Face, Hole, ManifoldMeshError, Vert
from .half_edge_querries import StaticHalfEdges

EV = Union[Edge, Vert]


def _face_edges(face: Face, edge: Edge) -> None:
    """
    Add or update face attribute for each edge in edge.face_edges
    """
    for edge_ in edge.face_edges:
        edge_.face = face


def _get_unit_set_item(one: Set) -> Any:
    """
    If a set has exactly one item, return that item.

    :param one: A set with presumably one item
    :return:
    """
    if len(one) == 1:
        return next(iter(one))
    else:
        raise ValueError("argument is not a unit set")


def _get_edge_or_vert_faces(elem: EV) -> Set[Face]:
    """
    Get faces (unordered) adjacent to a vert or edge

    :param elem: Vert or Edge instance
    :return: Face instances adjacent

    This is a subroutine for insert_edge.
    """
    if isinstance(elem, Edge):
        return {elem.face}
    return set(elem.faces)


class UnrecoverableManifoldMeshError(ValueError):
    """
    Found a problem with an operation AFTER mesh was potentially altered.

    Unexpected error. This should have been caught earlier. We found a something that
    couldn't be added or couldn't be removed, but we didn't find it in time. The mesh
    may have been altered before this discovery. We've found a bug in the module.
    """

    def __init__(self, message: str):
        super().__init__(self, message)


class HalfEdges(StaticHalfEdges):
    def _infer_face(self, orig: EV, dest: EV) -> Face:
        """
        Infer which face two verts lie on.

        :elem: vert or edge (presumably on the face)
        :return: face (if unambiguous) on which vert or edge lies

        Able to infer from:
            * both verts on same face: that face
            * empty mesh: a new Hole
        """
        if not self.edges:
            return self.hole_type()
        orig_faces = _get_edge_or_vert_faces(orig)
        dest_faces = _get_edge_or_vert_faces(dest)
        with suppress(ValueError):
            return _get_unit_set_item(orig_faces & dest_faces)
        raise ValueError("face cannot be determined from orig and dest")

    @staticmethod
    def _infer_wing(elem: EV, face: Face, default: Edge) -> Tuple[Vert, Edge]:
        """
        Given a vert or edge, try to return vert and edge such that edge.dest == vert

        :param elem: vert or edge in the mesh
        :param face: face on which vert or edge lies
        :param default: edge value if vert is new (no connected edges)
            - this will always be the edge pair.
        :return: a vert on the face (or presumed to be) and the edge ENDING at vert
        :raises: ValueError if very and edge are ambiguous

        This is a subroutine of insert_edge, which accepts a vert or edge as origin
        and destination arguments. The wing returned is

            * the origin of the edge to be inserted.
            * the edge *before* the edge to be inserted (the prev edge)

        elem (insert_edge orig or dest argument) is an edge: edge.dest and edge
        elem (insert_edge orig or dest argument) is a vert:

            vert on face? the prev edge is the face edge ending in vert

            vert not on face? (presume floating) the prev edge is default
                (new_edge.pair from outer scope)
        """
        if isinstance(elem, Edge):
            return elem.dest, elem
        if elem not in face.verts:
            return elem, default
        with suppress(ValueError):
            prev_edge = _get_unit_set_item({x for x in face.edges if x.dest is elem})
            return elem, prev_edge
        raise ValueError("edge cannot be determined from orig and face")

    def insert_edge(
        self,
        orig: EV,
        dest: EV,
        face: Optional[Face] = None,
        **edge_kwargs: Any,
    ) -> Edge:
        """
        Insert a new edge between two verts.

        :param orig: origin of new edge (vert or edge such that edge.dest == vert)
        :param dest: destination of new edge (vert or edge as above)
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

        edge = self.edge_type(next=Edge())
        edge.pair = self.edge_type(pair=edge, next=edge, prev=edge)

        edge_orig, edge_prev = self._infer_wing(orig, face, edge.pair)
        edge_dest, pair_prev = self._infer_wing(dest, face, edge)
        edge_next, pair_next = edge_prev.next, pair_prev.next

        if getattr(edge_orig, "edge", None) and edge_dest in edge_orig.neighbors:
            raise ManifoldMeshError("overwriting existing edge")

        if set(face.verts) & {edge_orig, edge_dest} != self.verts & {
            edge_orig,
            edge_dest,
        }:
            raise ManifoldMeshError("orig or dest in mesh but not on given face")

        if edge_orig == edge_dest:
            raise ManifoldMeshError("orig and dest are the same")

        if not set(face.verts) & {edge_orig, edge_dest} and face in self.faces:
            raise ManifoldMeshError("adding floating edge to existing face")

        edge.update(
            *face_edges, orig=edge_orig, prev=edge_prev, next=pair_next, **edge_kwargs
        )
        edge.pair.update(
            *face_edges, orig=edge_dest, prev=pair_prev, next=edge_next, **edge_kwargs
        )

        _face_edges(self.face_type(face), edge)
        _face_edges(face, edge.pair)

        self.edges.update({edge, edge.pair})

        return edge

    def insert_vert(self, face: Face, **vert_kwargs: Any) -> Vert:
        """
        Insert a new vert into face then triangulate face.

        :param face: face to triangulate
        :param edge_kwargs: set attributes for new vert
        :returns: newly inserted vert
        :raises: UnrecoverableManifoldMeshError if an unanticipated
            ManifoldMeshError occurs

        new vert is created on face
        new edges are created from new vert to extant face verts
        new faces are created as face is triangulated

        Passes attributes:

            * face attributes passed to new faces
            * shared face.edges attributes passed to new edges
            * shared face.verts attributes passed to new vert
        """
        new_vert = self.vert_type(*face.verts, **vert_kwargs)
        try:
            for vert in face.verts:
                self.insert_edge(vert, new_vert, face)
        except ManifoldMeshError as exc:
            raise UnrecoverableManifoldMeshError(str(exc))
        return new_vert

    def remove_edge(self, edge: Edge, **face_kwargs: Any) -> Face:
        """
        Cut an edge out of the mesh.

        :param edge: edge to remove
        :param face_kwargs: optional attributes for new face
        :returns: Newly joined (if edge split face) face, else new face that replaces
            previously shared face.
        :raises: ManifoldMeshError if
            * edge not in mesh
            * edge is a "bridge edge"

        Will not allow you to break (make non-manifold) the mesh. For example,
        here's a mesh with three faces, one in each square, and a third face or
        hole around the outside. If I remove that long edge, the hole would have
        two small, square faces inside of it. The hole would point to a half edge
        around one or the other square, but that edge would just "next" around its
        own small square. The other square could never be found.
         _       _
        |_|_____|_|

        Attempting to remove such edges will raise a ManifoldMeshError.

        Always removes the edge's face and expands the pair's face (if they are
        different).

        Passes attributes:
            * shared face attributes passed to new face

        """
        if edge not in self.edges:
            raise ManifoldMeshError("edge {} does not exist in mesh".format(id(edge)))

        pair = edge.pair

        if edge.orig.valence > 1 and edge.dest.valence > 1 and edge.face == pair.face:
            raise ManifoldMeshError("would create non-manifold mesh")

        edge_face_edges = set(edge.face_edges)
        pair_face_edges = set(pair.face_edges)

        # make sure orig and dest do not point to this edge (if there's another option)
        edge.next.orig = edge.next.orig
        pair.next.orig = pair.next.orig

        # set all faces equal to new face
        if isinstance(pair.face, Hole):
            new_face = self.hole_type(*{edge.face, pair.face}, **face_kwargs)
        else:
            new_face = self.face_type(*{edge.face, pair.face}, **face_kwargs)
        for edge_ in (edge_face_edges | pair_face_edges) - {edge, pair}:
            edge_.face = new_face

        # disconnect from previous edges
        edge.prev.next = pair.next
        pair.prev.next = edge.next
        self.edges -= {edge, pair}

        return new_face

    def remove_vert(self, vert: Vert, **face_kwargs: Any) -> Face:
        """
        Remove all edges around a vert.

        :raises: ManifoldMeshError if the error was caught before any edges were removed
            (this SHOULD always be the case).
        :raises: UnrecoverableManifoldMeshError if a problem was found after we started
            removing edges (this SHOULD never happen).

        How does this differ from consecutive calls to remove_edge?
            * checks (successfully as far as I can determine) that all edges are safe
              to remove before removing any.
            * orients remove_edge(orig, dest) calls so that holes fill hole-adjacent
              spaces rather than faces fill holes.
            * identifies and removes (one edge long) peninsula edges first

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

        Passes attributes:
            * shared face attributes passed to new face
        """
        peninsulas = {x for x in vert.edges if x.dest.valence == 1}
        true_edges = set(vert.edges) - peninsulas
        vert_faces = {x.face for x in true_edges}
        if len(true_edges) != len(vert_faces):
            raise ManifoldMeshError("removing vert would create non-manifold mesh")

        # remove peninsula edges then others.
        for edge in peninsulas:
            face = self.remove_edge(edge, **face_kwargs)
        try:
            for edge in vert.edges:
                if edge.face in self.faces:
                    face = self.remove_edge(edge, **face_kwargs)
                else:
                    face = self.remove_edge(edge.pair, **face_kwargs)
        except ManifoldMeshError as exc:
            raise UnrecoverableManifoldMeshError(str(exc))
        return face

    def remove_face(self, face: Face) -> Face:
        """
        Remove all edges around a face.

        :returns: face or hole remaining
        :raises: ManifoldMeshError if the error was caught before any edges were removed
            (this SHOULD always be the case).
        :raises: UnrecoverableManifoldMeshError if a problem was found after we started
            removing edges (this SHOULD never happen).

        How does this differ from consecutive calls to remove_edge?
            * checks (successfully as far as I can determine) that all edges are safe
              to remove before removing any.

        Passes attributes:
            * shared face attributes passed to new face
        """
        edges = tuple(face.edges)
        potential_bridges = [x for x in edges if x.orig.valence > 2]
        if len({x.pair.face for x in edges}) < len(potential_bridges):
            raise ManifoldMeshError(
                "Removing this face would create non-manifold mesh."
                " One of this faces's edges is a bridge edge."
            )

        try:
            for edge in edges:
                self.remove_edge(edge)
        except ManifoldMeshError as exc:
            raise UnrecoverableManifoldMeshError(str(exc))
        return face

    def split_edge(self, edge: Edge, **vert_kwargs) -> Vert:
        """
        Add a vert to the middle of an edge.

        :param edge: edge to be split
        :param vert_kwargs: attributes for new vert
        :return:

        Passes attributes:
            * shared vert attributes passed to new vert
            * edge attributes passed to new edges
            * pair attributes passed to new pairs

        remove_edge will replace the original faces, so these are restored at the end
        of the method.
        """
        new_vert = self.vert_type(*{edge.orig, edge.dest}, **vert_kwargs)
        edge_face = edge.face
        pair_face = edge.pair.face
        for orig, dest in ((edge.dest, new_vert), (new_vert, edge.orig)):
            new_edge = self.insert_edge(orig, dest, edge.face)
            new_edge.extend(edge.pair)
            new_edge.pair.extend(edge)
        self.remove_edge(edge)
        _face_edges(edge_face, new_edge.pair)
        _face_edges(pair_face, new_edge)
        return new_vert

    def flip_edge(self, edge: Edge) -> Edge:
        """
        Flip an edge between two triangles.

        :param edge: Edge instance in self
        :return: None

        * The edge must be between two triangles.
        * Only shared edge attributes will remain.
        * Only shared face attributes will remain.

        Warning: This can break your mesh if the quadrangle formed by the triangles
        on either side of the flipped edge is not convex. Not useful without some
        coordinate information to avoid this non-convex case.
        """
        pair = edge.pair
        if len(edge.face_edges) != 3 or len(pair.face_edges) != 3:
            raise ValueError("can only flip an edge between two triangles")
        new_orig = edge.next.dest
        new_dest = pair.next.dest
        face = self.remove_edge(edge)
        new_edge = self.insert_edge(new_orig, new_dest, face)
        new_edge.update(edge, pair)
        return new_edge

    def collapse_edge(self, edge: Edge, **vert_kwargs) -> Vert:
        """
        Collapse an Edge into a Vert.

        :param edge: Edge instance in self
        :param vert_kwargs: attributes for the new Vert
        :return: Vert where edge used to be.

        Passes attributes:
            * shared vert attributes passed to new vert

        Warning: Some ugly things can happen here than can only be recognized and
        avoided by examining the geometry. This module only addresses connectivity,
        not geometry, but I've included this operation to experiment with and use
        carefully. Can really, really break things.
        """
        new_vert = self.vert_type(edge.orig, edge.dest, **vert_kwargs)
        for edge_ in set(edge.orig.edges) | set(edge.dest.edges):
            edge_.orig = new_vert
        self.edges.remove(edge)

        # remove slits
        for face in (x.face for x in new_vert.edges if len(x.face.edges) == 2):
            face_edges = face.edges
            face_edges[0].pair.pair = face_edges[1].pair
            self.edges -= set(face_edges)
        return new_vert
