from __future__ import annotations

from contextlib import suppress
from typing import Any, Optional, Set, TYPE_CHECKING, Tuple, TypeVar, Union

import copy
from .half_edge_elements import ManifoldMeshError
from .half_edge_querries import StaticHalfEdges
from .validations import validate_mesh

if TYPE_CHECKING:
    from .half_edge_elements import Vert, Edge, Face

# TODO: remove LOG
log = []


def _update_face_edges(face: Face, edge: Edge) -> None:
    """
    Add or update face attribute for each edge in edge.face_edges

    :param face: each edge will point to this face
    :param edge: one edge on the face (even if the edge doesn't point to the face yet)

    This is the only way to add a face to a mesh, because faces only exist as long as
    there is an edge pointing to them.
    """
    for edge_ in edge.face_edges:
        edge_.face = face


def _get_singleton_item(one: Set) -> Any:
    """
    If a set has exactly one item, return that item.

    :param one: A set with presumably one item
    :return:
    """
    (item,) = one
    return item


class UnrecoverableManifoldMeshError(ValueError):
    """
    Found a problem with an operation AFTER mesh was potentially altered.

    Unexpected error. This should have been caught earlier. We found a something that
    couldn't be added or couldn't be removed, but we didn't find it in time. The mesh
    may have been altered before this discovery. We've found a bug in the module.
    """

    def __init__(self, message: str):
        super().__init__(self, message)


_TMeshElem = TypeVar("_TMeshElem", bound="MeshElementBase")


class HalfEdges(StaticHalfEdges):
    def _get_edge_or_vert_faces(self, elem: Union[Edge, Vert]) -> Set[Face]:
        """
        Get faces (unordered) adjacent to a vert or edge

        :param elem: Vert or Edge instance
        :return: Face instances adjacent

        This is a subroutine for insert_edge.
        """
        if isinstance(elem, self.edge_type):
            return {elem.face}
        return set(elem.faces)

    def _infer_face(
        self: _TMeshElem,
        orig: Union[Edge, Vert],
        dest: Union[Edge, Vert],
    ) -> Face:
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
        orig_faces = self._get_edge_or_vert_faces(orig)
        dest_faces = self._get_edge_or_vert_faces(dest)
        with suppress(ValueError):
            return _get_singleton_item(orig_faces & dest_faces)
        raise ValueError("face cannot be determined from orig and dest")

    def _infer_wing(
        self, elem: Union[Edge, Vert], face: Face, default: Edge
    ) -> Tuple[Vert, Edge]:
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
        if isinstance(elem, self.edge_type):
            return elem.dest, elem
        if elem not in face.verts:
            return elem, default
        with suppress(ValueError):
            prev_edge = _get_singleton_item({x for x in face.edges if x.dest is elem})
            return elem, prev_edge
        raise ValueError("edge cannot be inferred from orig and face")

    def _point_away_from_edge(self, edge: Edge) -> None:
        """
        Prepare edge to be removed. Remove vert and face pointers to edge.

        :param edge: any edge in mesh
        :effects: points edge.orig and edge.face to another edge

        Each vert and each face point to an adjacent edge. *Which* adjacent edge is
        incidental. This method tries to point an edge's origin and face to
        *something else*.

        This method requires an intact mesh and produces an intact mesh. After this
        method, the mesh will be perfectly equivalent to its previous state. However,
        this method has to be called *before* we start other preparation to remove
        the edge, because *those* preparations *will* alter the mesh and prevent
        *this* method from working.

        The method will fail silently if the edge.orig or edge.face doesn't have
        another edge to point to. But that won't matter, because that orig or face
        will go out of scope when the edge is removed.
        """
        pair = edge.pair
        for edge_ in (edge, pair):
            edge_.orig.edge = edge_.pair.next
            safe_edges = set(edge_.face.edges) - {edge, pair}
            edge_.face.edge = next(iter(safe_edges), edge_)

    def insert_edge(
        self,
        orig: Union[Edge, Vert],
        dest: Union[Edge, Vert],
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

        edge = self.edge_type(next=self.edge_type())
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

        # if face is not split, new face will be created then immediately written over
        _update_face_edges(self.face_type(face), edge)
        _update_face_edges(face, edge.pair)

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
        # TODO: incorporate point_Away_From_edge
        # TODO: new hole every time
        if edge not in self.edges:
            raise ManifoldMeshError("edge {} does not exist in mesh".format(id(edge)))

        pair = edge.pair

        if edge.orig.valence > 1 and edge.dest.valence > 1 and edge.face == pair.face:
            raise ManifoldMeshError("would create non-manifold mesh")

        self._point_away_from_edge(edge)

        edge_face_edges = set(edge.face_edges)
        pair_face_edges = set(pair.face_edges)

        # # make sure orig and dest do not point to this edge (if there's another option)
        # edge.next.orig = edge.next.orig
        # pair.next.orig = pair.next.orig

        # set all faces equal to new face
        if not isinstance(pair.face, self.hole_type):
            new_face = self.face_type(*{edge.face, pair.face}, **face_kwargs)
        else:
            new_face = self.hole_type(*{edge.face, pair.face}, **face_kwargs)
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
        # TODO: review why peninsulas need to be removed.
        peninsulas = {x for x in vert.edges if x.dest.valence == 1}
        true_edges = set(vert.edges) - peninsulas
        vert_faces = {x.face for x in true_edges}
        if len(true_edges) != len(vert_faces):
            raise ManifoldMeshError("removing vert would create non-manifold mesh")

        # remove peninsula edges then others.
        for edge in peninsulas:
            face = self.remove_edge(edge, **face_kwargs)
        try:
            # remove face edges, not hole edges, so holes will fill faces.
            for edge in vert.edges:
                if isinstance(edge.face, self.hole_type):
                    face = self.remove_edge(edge.pair, **face_kwargs)
                else:
                    face = self.remove_edge(edge, **face_kwargs)
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
        _update_face_edges(edge_face, new_edge.pair)
        _update_face_edges(pair_face, new_edge)
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

    def _is_stitchable(self, edge: Edge) -> bool:
        """
        Can two edges be stitched (middle 2-side face removed)?

        :param edge_a:
        :param edge_b:
        :return:

        Two edges can be stitched if their outside faces don't share > 1 vert.

        A mesh is not manifold if two adjacent faces point in opposite directions.
        From a connectivity perspective (this package only deals with connectivity,
        not geometry), two faces that share more that two vertices are facing in
        opposite directions--assuming faces are flat and consecutive sides are not
        linear. Whether or not we accept that, more than two connected vertices
        breaks the halfedge data structure, if not immediately, then eventually.
        """
        # breakpoint()
        pair = edge.pair
        tris = sum(len(x.face_edges) == 3 for x in (edge, pair))
        # breakpoint()
        orig_verts = set(edge.orig.neighbors)
        dest_verts = set(edge.dest.neighbors)
        # breakpoint()
        if len(orig_verts & dest_verts) == tris:
            return True
        return False
        return
        if len(edge.face_edges) > 3:
            return True
        face_a = edge.pair.face
        face_b = edge.next.pair.face
        if face_a == face_b:
            return True
        if len(set(face_a.verts) & set(face_b.verts)) < 2:
            return True
        return False

    def collapse_edge(self, edge: Edge, **vert_kwargs) -> Vert:
        """
        Collapse an Edge into a Vert.

        :param edge: Edge instance in self
        :param vert_kwargs: attribute/s for the new Vert
        :return: Vert where edge used to be.

        Passes attributes:
            * shared vert attributes passed to new vert

        Warning: Some ugly things can happen here than can only be recognized and
        avoided by examining the geometry. This module only addresses connectivity,
        not geometry, but I've included this operation to experiment with and use
        carefully. Can flip faces and create linear faces.
        """
        global log
        # TODO: raise error if collapsing missing edge
        # breakpoint()
        pair = edge.pair
        # breakpoint()
        safe = self._is_stitchable(edge)
        log.append(f"{safe=}")
        # safe = safe and self._is_stitchable(pair.next)
        if not safe:
            log.append("not safe")
            raise ManifoldMeshError("would create non-manifold mesh")

        log.append(f"passed safe test")

        if edge.orig.valence == 1 or edge.dest.valence == 1:
            log.append("trying to remove edge")
            try:
                validate_mesh(self)
            except:
                breakpoint()
            try:
                self.remove_edge(edge)
            except:
                breakpoint()
            return
        log.append(f"passed attempted edge removal")

        for edge_ in set(edge.orig.edges + edge.dest.edges):
            if edge_.dest.valence == 1:
                try:
                    log.append(f"multi-edge removal")
                except:
                    breakpoint()
                self.remove_edge(edge_)
        log.append(f"passed multi-edge removal")

        if len(self.edges) == 2:
            self.edges -= self.edges
            breakpoint()
            return
        props = [len(x.edges) for x in self.all_faces], len(self.verts), len(self.edges)
        log.append(props)
        self._point_away_from_edge(edge)

        if any(x.orig is x.dest for x in self.edges):
            breakpoint()

        mesh_back = copy.deepcopy(self)

        new_vert = self.vert_type(edge.orig, edge.dest, **vert_kwargs)
        for edge_ in (set(edge.orig.edges) | set(edge.dest.edges)) - {edge, edge.pair}:
            edge_.orig = new_vert

        pair = edge.pair
        aaa = edge.prev
        bbb = edge.next
        edge.prev.next = edge.next
        # if any(x.orig is x.dest for x in self.edges - {edge, pair}):
        #     breakpoint()
        pair.prev.next = pair.next
        # if any(x.orig is x.dest for x in self.edges):
        #     breakpoint()
        adjacent_faces = {edge.face, edge.pair.face}
        self.edges -= {edge, pair}
        if any(x.orig is x.dest for x in self.edges):
            breakpoint()

        # remove slits
        while adjacent_faces:
            if any(x.orig is x.dest for x in self.edges):
                props = (
                    [len(x.edges) for x in self.all_faces],
                    len(self.verts),
                    len(self.edges),
                )
                log.append(props)
                breakpoint()
            face = adjacent_faces.pop()
            if face.edge not in self.edges or len(face.edges) > 2:
                log.append("len > 2")
                continue
            if (
                any(x.valence == 2 for x in face.verts)
                and len(face.edge.pair.face.verts) == 4
            ):
                adjacent_faces.add(face.edge.pair.face)
                self.remove_edge(face.edge.next)
                self.remove_edge(face.edge)
                log.append("dart")
                continue
            face_edges = face.edges
            face_edges[0].pair.pair = face_edges[1].pair
            self._point_away_from_edge(face.edge)
            self.edges -= set(face_edges)
            log.append("normal")
        # if any(x.orig is x.dest for x in self.edges):
        props = (
            [len(x.edges) for x in self.all_faces],
            len(self.verts),
            len(self.edges),
        )
        log.append(props)
        # breakpoint()
        log.append("-----")
        # if len(self.edges) > 2 and any(len(x.edges) == 2 for x in self.all_faces):
        #     breakpoint()
        try:
            validate_mesh(self)
        except:
            vl_back = [x.sn for x in mesh_back.vl]
            vi_back = [[x.sn for x in y.verts] for y in mesh_back.all_faces]
            vl = [x.sn for x in self.vl]
            vi = [[x.sn for x in y.verts] for y in self.all_faces]
            breakpoint()
        # breakpoint()
        return new_vert
