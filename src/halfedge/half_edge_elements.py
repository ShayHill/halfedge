"""A half-edges data container with view methods.

A simple container for a list of half edges. Provides lookups and a serial
number for each mesh element (Vert, Edge, or Face).

This is a typical halfedges data structure. Exceptions:

    * Face() is distinct from Face(is_hole=True). This is
      to simplify working with 2D meshes. You can
          - define a 2d mesh with triangles
          - explicitly or algorithmically define holes to keep the mesh manifold
          - ignore the holes after that. They won't be returned when, for instance,
            iterating over mesh faces.
      Boundary verts, and boundary edges are identified by these holes, but that all
      happens internally, so the holes can again be ignored for most things.

    * Orig, pair, face, and next assignments are mirrored, so a.pair = b will set
      a.pair = b and b.pair = a. This makes edge insertion, etc. cleaner, but the
      whole thing is still easy to break if you extend the class. Hopefully, I've
      provided enough insertion / removal code to get you over the pitfalls.
      Halfedges (as a data structure, not just this implementation) is clever when
      it's all built, but a lot has to be temporarily broken down to transform the
      mesh. All I can say is, write a lot of test if you want to extend the insertion
      / removal methods here.

    * Some methods (e.g., Vert.edges or Face.verts) return an empty list, even when
      self.edge would raise an AttributeError. A Vert or Face without an edge is not
      manifold, but a Vert or Face without an edge will never be in a mesh, because a
      mesh only stores edges. These empty Verts or Faces will occur when iteratively
      removing edges around a Vert or Face.

This module is all the base elements (Vert, Edge, and Face).

created: 2006 June 05
"""

from __future__ import annotations

from contextlib import suppress
from itertools import count
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from halfedge.type_attrib import Attrib, ContagionAttrib

if TYPE_CHECKING:
    from halfedge.half_edge_constructors import BlindHalfEdges

_TMeshElem = TypeVar("_TMeshElem", bound="MeshElementBase")

_T = TypeVar("_T")


class IsHole(ContagionAttrib):
    """Flag a Face instance as a hole."""


class ManifoldMeshError(ValueError):
    """Incorrect arguments passed to HalfEdges init.

    ... or something broken along the way. List of edges do not represent a valid
    (manifold) half edge data structure. Some properties will still be available,
    so you might catch this one and continue in some cases, but many operations will
    require valid, manifold mesh data to infer.
    """


class MeshElementBase:
    """Base class for Vert, Edge, and Face."""

    _sn_generator = count()

    def __init__(
        self, *attributes: Attrib[Any], mesh: BlindHalfEdges | None = None
    ) -> None:
        """Create an instance (and copy attrs from fill_from).

        :param attributes: ElemAttribBase instances
        :param pointers: pointers to other mesh elements
            (per typical HalfEdge structure)

        This class does not have pointers. Descendent classes will, and it is
        critical that each have a setter and each setter cache a value as _pointer
        (e.g., _vert, _pair, _face, _next).
        """
        self.sn = next(self._sn_generator)
        self.attrib: dict[str, Attrib[Any]] = {}
        self.mesh = mesh

        for attribute in attributes:
            self.set_attrib(attribute)

    def set_attrib(self, attrib: Attrib[Any]) -> None:
        """Set an attribute."""
        self.attrib[type(attrib).__name__] = attrib.copy_to_element(self)

    def get_attrib(self, attrib: type[Attrib[_T]]) -> Attrib[_T]:
        """Get an attribute."""
        try:
            return self.attrib[attrib.__name__]
        except KeyError as e:
            msg = f"{attrib.__name__} not found in {self.__class__.__name__}"
            raise AttributeError(msg) from e

    def try_attrib(self, attrib: type[Attrib[_T]]) -> Attrib[_T] | None:
        """Get an attribute or return None."""
        try:
            return self.get_attrib(attrib)
        except AttributeError:
            return None

    def merge_from(self: _TMeshElem, *elements: _TMeshElem) -> _TMeshElem:
        """Fill in missing references from other elements.

        :param elements: elements to merge from
        :returns: self with missing attrs dict keys filled in

        For any key present in one or more e.attrib for e in elements, merge
        identical keys and fill in self.attrib. Do not overwrite existing keys. Only
        merge keys that are not already in self.attrib.

        Some attribs, when merged, return None. These these will not be set in
        self.attrib.
        """
        old_attribs: set[type[Attrib[Any]]] = {type(x) for x in self.attrib.values()}
        all_attribs: set[type[Attrib[Any]]] = set()
        for element in elements:
            all_attribs.update({type(x) for x in element.attrib.values()})
        new_attribs = all_attribs - old_attribs
        for attrib in new_attribs:
            merged_attrib = attrib.merge(*(e.try_attrib(attrib) for e in elements))
            if merged_attrib is not None:
                self.set_attrib(merged_attrib)
        return self

    def split_from(self: _TMeshElem, element: _TMeshElem) -> _TMeshElem:
        """Pass attributes when dividing or altering elements.

        Use the 'split' method of Attrib instances to determine how to pass
        attributes child elements when dividing an element.
        """
        elem_attribs = {type(x) for x in element.attrib.values()}
        self_attribs = {type(x) for x in self.attrib.values()}
        for attrib in elem_attribs - self_attribs:
            splitted = element.get_attrib(attrib).split()
            if splitted is not None:
                self.set_attrib(splitted)
        return self

    def __lt__(self: _TMeshElem, other: _TMeshElem) -> bool:
        """Sort by sn.

        You'll want to be able to sort Verts at least to make a vlvi (vertex list,
        vertex index) format.
        """
        return self.sn < other.sn


# argument to a function that returns same type as the input argument
_TFLapArg = TypeVar("_TFLapArg")


def _function_lap(
    func: Callable[[_TFLapArg], _TFLapArg], first_arg: _TFLapArg
) -> list[_TFLapArg]:
    """Repeatedly apply func till first_arg is reached again.

    :param func: function takes one argument and returns a value of the same type
    :returns: [first_arg, func(first_arg), func(func(first_arg)) ... first_arg]
    :raises: ManifoldMeshError if any result except the first repeats
    """
    lap = [first_arg]
    while True:
        lap.append(func(lap[-1]))
        if lap[-1] == lap[0]:
            return lap[:-1]
        if lap[-1] in lap[1:-1]:
            msg = f"infinite loop in {_function_lap.__name__}"
            raise ManifoldMeshError(msg)


class Vert(MeshElementBase):
    """Half-edge mesh vertices."""

    def __init__(
        self,
        *attributes: Attrib[Any],
        mesh: BlindHalfEdges | None = None,
        edge: Edge | None = None,
    ) -> None:
        """Create a Vert instance."""
        super().__init__(*attributes, mesh=mesh)
        self._edge = edge
        if edge is not None:
            self.edge = edge

    @property
    def edge(self) -> Edge:
        """One edge originating at vert."""
        if self._edge is not None:
            return self._edge
        msg = ".edge not set for Vert instance."
        raise AttributeError(msg)

    @edge.setter
    def edge(self, edge_: Edge) -> None:
        self._edge = edge_
        edge_.orig = self

    def set_edge_without_side_effects(self, edge: Edge) -> None:
        """Set edge without setting edge's orig."""
        self._edge = edge

    @property
    def edges(self) -> list[Edge]:
        """Half edges radiating from vert."""
        try:
            vert_edge = self.edge
        except AttributeError:
            return []
        return vert_edge.vert_edges

    @property
    def all_faces(self) -> list[Face]:
        """Faces radiating from vert."""
        try:
            vert_edge = self.edge
        except AttributeError:
            return []
        return vert_edge.vert_all_faces

    @property
    def faces(self) -> list[Face]:
        """Faces radiating from vert."""
        return [x for x in self.all_faces if not x.is_hole]

    @property
    def holes(self) -> list[Face]:
        """Faces radiating from vert."""
        return [x for x in self.all_faces if x.is_hole]

    @property
    def neighbors(self) -> list[Vert]:
        """Evert vert connected to vert by one edge."""
        try:
            vert_edge = self.edge
        except AttributeError:
            return []
        return vert_edge.vert_neighbors

    @property
    def valence(self) -> int:
        """The number of edges incident to vertex."""
        return len(self.edges)


class Edge(MeshElementBase):
    """Half-edge mesh edges."""

    def __init__(
        self,
        *attributes: Attrib[Any],
        mesh: BlindHalfEdges | None = None,
        orig: Vert | None = None,
        pair: Edge | None = None,
        face: Face | None = None,
        next: Edge | None = None,
        prev: Edge | None = None,
    ) -> None:
        """Create an Edge instance."""
        super().__init__(*attributes, mesh=mesh)
        self._orig = orig
        self._pair = pair
        self._face = face
        self._next = next
        if orig is not None:
            self.orig = orig
        if pair is not None:
            self.pair = pair
        if face is not None:
            self.face = face
        if next is not None:
            self.next = next
        if prev is not None:
            self.prev = prev

    @property
    def orig(self) -> Vert:
        """Vert at which edge originates."""
        if self._orig is not None:
            return self._orig
        msg = ".orig vertex not set for Edge instance."
        raise AttributeError(msg)

    @orig.setter
    def orig(self, orig: Vert) -> None:
        self._orig = orig
        orig.set_edge_without_side_effects(self)

    @property
    def pair(self) -> Edge:
        """Edge running opposite direction over same verts."""
        if self._pair is not None:
            return self._pair
        msg = ".pair edge not set for Edge instance."
        raise AttributeError(msg)

    @pair.setter
    def pair(self, pair: Edge) -> None:
        self._pair = pair
        pair.set_pair_without_side_effects(self)

    def set_pair_without_side_effects(self, edge: Edge) -> None:
        """Set pair without setting pair's pair."""
        self._pair = edge

    @property
    def face(self) -> Face:
        """Face to which edge belongs."""
        if self._face is not None:
            return self._face
        msg = ".face not set for Edge instance."
        raise AttributeError(msg)

    @face.setter
    def face(self, face_: Face) -> None:
        self._face = face_
        face_.edge = self

    def set_face_without_side_effects(self, face: Face) -> None:
        """Set face without setting face's edge."""
        self._face = face

    @property
    def next(self: Edge) -> Edge:
        """Next edge along face."""
        if self._next is not None:
            return self._next
        msg = ".next not set for Edge instance."
        raise AttributeError(msg)

    @next.setter
    def next(self: Edge, next_: Edge) -> None:
        self._next = next_

    @property
    def prev(self) -> Edge:
        """Look up the edge before self."""
        try:
            return self.face_edges[-1]
        except (AttributeError, ManifoldMeshError):
            return self.vert_edges[-1].pair

    @prev.setter
    def prev(self, prev: Edge) -> None:
        super(Edge, prev).__setattr__("next", self)

    @property
    def dest(self) -> Vert:
        """Vert at the end of the edge (opposite of orig)."""
        try:
            return self.next.orig
        except AttributeError:
            return self.pair.orig

    @property
    def face_edges(self) -> list[Edge]:
        """All edges around an edge.face."""

        def _get_next(edge: Edge) -> Edge:
            return edge.next

        return _function_lap(_get_next, self)

    @property
    def face_verts(self) -> list[Vert]:
        """All verts around an edge.vert."""
        return [edge.orig for edge in self.face_edges]

    @property
    def vert_edges(self) -> list[Edge]:
        """All half edges radiating from edge.orig.

        These will be returned in the opposite "handedness" of the faces. IOW,
        if the faces are defined ccw, the vert_edges will be returned cw.
        """
        return _function_lap(lambda x: x.pair.next, self)

    @property
    def vert_all_faces(self) -> list[Face]:
        """Return all faces and holes around the edge's vert."""
        return [x.face for x in self.vert_edges]

    @property
    def vert_faces(self) -> list[Face]:
        """Return all faces around the edge's vert."""
        return [x for x in self.vert_all_faces if not x.is_hole]

    @property
    def vert_holes(self) -> list[Face]:
        """Return all holes around the edge's vert."""
        return [x for x in self.vert_all_faces if x.is_hole]

    @property
    def vert_neighbors(self) -> list[Vert]:
        """All verts connected to vert by one edge."""
        return [edge.dest for edge in self.vert_edges]


class Face(MeshElementBase):
    """Half-edge mesh faces."""

    def __init__(
        self,
        *attributes: Attrib[Any],
        mesh: BlindHalfEdges | None = None,
        edge: Edge | None = None,
        is_hole: bool = False,
    ) -> None:
        """Create a Face instance."""
        super().__init__(*attributes, mesh=mesh)
        if is_hole:
            self.set_attrib(IsHole())
        self._edge = edge
        if edge is not None:
            self.edge = edge

    @property
    def edge(self) -> Edge:
        """One edge on the face."""
        if self._edge is not None:
            return self._edge
        msg = ".edge not set for Face instance."
        raise AttributeError(msg)

    @edge.setter
    def edge(self, edge: Edge) -> None:
        """Point face.edge back to face."""
        self._edge = edge
        edge.set_face_without_side_effects(self)

    @property
    def is_hole(self) -> bool:
        """Return True if this face a hole.

        "hole-ness" is assigned at instance creation by passing ``is_hole=True`` to
        ``__init__``
        """
        with suppress(AttributeError):
            return self.get_attrib(IsHole).value
        return False

    @property
    def edges(self) -> list[Edge]:
        """Look up all edges around face."""
        try:
            face_edge = self.edge
        except AttributeError:
            return []
        return face_edge.face_edges

    @property
    def verts(self) -> list[Vert]:
        """Look up all verts around face."""
        try:
            face_edge = self.edge
        except AttributeError:
            return []
        return face_edge.face_verts

    @property
    def sides(self) -> int:
        """Return how many sides the face has.

        This is the equivalent of valence for faces.
        """
        return len(self.verts)
