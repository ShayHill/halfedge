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
      whole thing is still easy to break. Hopefully, I've provided enough insertion /
      removal code to get you over the pitfalls. Halfedges (as a data structure,
      not just this implementation) is clever when it's all built, but a lot has to
      be temporarily broken down to transform the mesh. All I can say is, write a lot
      of test if you want to extend the insertion / removal methods here.

This module is all the base elements (Vert, Edge, and Face).

# 2006 June 05
# 2012 September 30
"""

# TODO: get rid of "last modified" comments across the project

from __future__ import annotations

import itertools as it
from itertools import count
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from halfedge.type_attrib import Attrib, ContagionAttrib

if TYPE_CHECKING:
    from .half_edge_constructors import BlindHalfEdges

_TMeshElem = TypeVar("_TMeshElem", bound="MeshElementBase")

# _TAttrib = TypeVar("_TAttrib", bound=Attrib[Any])
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
        attrib.element = self
        self.attrib[type(attrib).__name__] = attrib

    def get_attrib(self, attrib: type[Attrib[_T]]) -> Attrib[_T]:
        """Get an attribute."""
        return self.attrib[attrib.__name__]

    def try_attrib(self, attrib: type[Attrib[_T]]) -> Attrib[_T] | None:
        """Try to get an attribute."""
        try:
            return self.get_attrib(attrib)
        except KeyError:
            return None

    def try_attrib_value(self, attrib: type[Attrib[_T]]) -> _T | None:
        """Try to get the value of an attribute."""
        attrib_found = self.try_attrib(attrib)
        if attrib_found is None:
            return None
        return attrib_found.value

    def _maybe_set_attrib(self, *attribs: Attrib[Any] | None) -> None:
        """Set an attribute if it is not None."""
        for attrib in attribs:
            if attrib is not None:
                self.set_attrib(attrib)

    def merge_from(self: _TMeshElem, *elements: _TMeshElem) -> _TMeshElem:
        """Fill in missing references from other elements."""
        # TODO: maybe split merge_from into merge_from and fill_from
        all_attrib_names = set(it.chain(*(x.attrib.keys() for x in elements)))
        for new_attrib_name in all_attrib_names - set(self.attrib.keys()):
            maybe_attribs = [x.attrib.get(new_attrib_name) for x in elements]
            elements_attribs = [x for x in maybe_attribs if x is not None]
            if not elements_attribs:
                # shouldn't ever happen
                continue
            merged_attrib = type(elements_attribs[0]).merge(*elements_attribs)
            if merged_attrib is None:
                continue
            self.set_attrib(merged_attrib)
        return self

    def slice_from(self: _TMeshElem, element: _TMeshElem) -> _TMeshElem:
        """Pass attributes when dividing or altering elements.

        Do not pass any pointers. ElemAttribBase instances are passed as defined by
        their classes.
        """
        for key in set(element.attrib) - set(self.attrib):
            self._maybe_set_attrib(element.attrib[key])
        return self

    def __lt__(self: _TMeshElem, other: _TMeshElem) -> bool:
        """Sort by id.

        You'll want to be able to sort Verts at least to make a vlvi (vertex list,
        vertex index) format.
        """
        return id(self) < id(other)


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
    """Half-edge mesh vertices.

    required attributes
    :edge: pointer to one edge originating at vert
    """

    def __init__(
        self,
        *attributes: Attrib[Any],
        mesh: BlindHalfEdges | None = None,
        edge: Edge | None = None,
    ) -> None:
        """Create a vert instance."""
        super().__init__(*attributes, mesh=mesh)
        self._edge = edge
        if edge is not None:
            self.edge = edge

    # TODO: get rid of has_ properties. hasattr works
    @property
    def has_edge(self) -> bool:
        """Return True if .edge is set."""
        return self._edge is not None

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
        if self.has_edge:
            return self.edge.vert_edges
        return []

    @property
    def all_faces(self) -> list[Face]:
        """Faces radiating from vert."""
        if hasattr(self, "edge"):
            return self.edge.vert_all_faces
        return []

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
        if hasattr(self, "edge"):
            return self.edge.vert_neighbors
        return []

    @property
    def valence(self) -> int:
        """The number of edges incident to vertex."""
        return len(self.edges)


class Edge(MeshElementBase):
    """Half-edge mesh edges.

    required attributes
    :orig: pointer to vert at which edge originates
    :pair: pointer to edge running opposite direction over same verts
    :face: pointer to face
    :next: pointer to next edge along face
    """

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
        """Create an edge instance."""
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
    def has_orig(self) -> bool:
        """Return True if .orig is set."""
        return self._orig is not None

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
    def has_pair(self) -> bool:
        """Return True if .pair is set."""
        return self._pair is not None

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
        pair._pair = self

    @property
    def has_face(self) -> bool:
        """Return True if .face is set."""
        return self._face is not None

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
    def has_next(self) -> bool:
        """Return True if .next is set."""
        return self._next is not None

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
    """Half-edge mesh faces.

    required attribute
    :edge: pointer to one edge on the face

    TODO: document how faces are a bit different because they can be valid without an
    edge.
    """

    def __init__(
        self,
        *attributes: Attrib[Any],
        mesh: BlindHalfEdges | None = None,
        edge: Edge | None = None,
        is_hole: bool = False,
    ) -> None:
        """Create a face instance."""
        if is_hole:
            attributes += (IsHole(),)
        super().__init__(*attributes, mesh=mesh)
        self._edge = edge
        if edge is not None:
            self.edge = edge

    @property
    def has_edge(self) -> bool:
        """Does face have one edge identified."""
        return self._edge is not None

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
        return self.try_attrib(IsHole) is not None

    @property
    def edges(self) -> list[Edge]:
        """Look up all edges around face."""
        if self.has_edge:
            return self.edge.face_edges
        return []

    @property
    def verts(self) -> list[Vert]:
        """Look up all verts around face."""
        if self.has_edge:
            return [x.orig for x in self.edges]
        return []

    @property
    def sides(self) -> int:
        """Return how many sides the face has.

        This is the equivalent of valence for faces.
        """
        return len(self.verts)
