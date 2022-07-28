#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# Last modified: 220727 12:51:28
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

from __future__ import annotations

from itertools import count
from typing import Any, Callable, List, Optional, Type, TypeVar

from .element_attributes import ContagionAttributeBase, ElemAttribBase

_TMeshElem = TypeVar("_TMeshElem", bound="MeshElementBase")


class IsHole(ContagionAttributeBase):
    """Flag a Face instance as a hole"""


class ManifoldMeshError(ValueError):
    """
    Incorrect arguments passed to HalfEdges init.

    ... or something broken along the way. List of edges do not represent a valid
    (manifold) half edge data structure. Some properties will still be available,
    so you might catch this one and continue in some cases, but many operations will
    require valid, manifold mesh data to infer.
    """


def _all_is(*args: Any) -> bool:
    """True if all arguments are `a is b`"""
    return args and all(args[0] is x for x in args[1:])


class MeshElementBase:
    _sn_generator = count()
    _pointers: set[str] = set()

    def __init__(
        self,
        *attributes: ElemAttribBase,
        **pointers: MeshElementBase,
    ) -> None:
        """
        Create an instance (and copy attrs from fill_from).

        :param attributes: ElemAttribBase instances
        :param pointers: pointers to other mesh elements
            (per typical HalfEdge structure)

        This class does not have pointers. Descendent classes will, and it is
        critical that each have a setter and each setter cache a value as _pointer
        (e.g., _vert, _pair, _face, _next).
        """
        self.sn = next(self._sn_generator)
        for attribute in attributes:
            self.set_attrib(attribute)
        for k, v in pointers.items():
            if v is not None:
                setattr(self, k, v)

    def __setattr__(self, key: str, value: Any) -> None:
        """To prevent any mistyped attributes, which would clobber fill_from

        This is here to help refactoring, but isn't necessary or necessarily
        Pythonic. Basically, you can only set public attributes which are defined in
        init or have setters.

        I started off allowing element attributes as simple properties, so I had a
        lot of tests with code like `edge_instance.color = "purple"`. Overloading
        setattr this way allowed me to find those quickly. I'm going to leave this in
        for now because it prevents typos and it will help me remember later on that
        I cannot set ElemAttribBase properties with `edge_instance.something =
        ElemAttribBase_instance`.
        """
        allow = key == "sn"
        allow = allow or key in self._pointers
        allow = allow or key.lstrip("_") in self._pointers
        allow = allow or isinstance(value, ElemAttribBase)
        if allow:
            super().__setattr__(key, value)
            return
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")

    def fill_from(self: _TMeshElem, *elements: _TMeshElem) -> _TMeshElem:
        """
        Fill in missing references from other elements.
        """
        keys_seen = {k for k, v in self.__dict__.items() if v is not None}
        for element in elements:
            for key in (x for x in element.__dict__.keys() if x not in keys_seen):
                keys_seen.add(key)
                vals = [getattr(x, key, None) for x in elements]
                if isinstance(getattr(element, key), ElemAttribBase):
                    self.maybe_set_attrib(type(getattr(element, key)).merged(*vals))
                    continue
                if _all_is(*vals):  # will have be something in _pointers
                    setattr(self, key[1:], vals[0])
        return self

    def set_attrib(self: _TMeshElem, *attribs: ElemAttribBase) -> _TMeshElem:
        """Set attribute with an ElemAttribBase instance.

        type(attrib).__name__ : attrib

        :param attribs: ElemAttribBase instances, presumably with a None element
        attribute.
        """
        for attrib in attribs:
            attrib.element = self
            self.__dict__[type(attrib).__name__] = attrib
        return self

    def maybe_set_attrib(self, *attribs: None | ElemAttribBase) -> None:
        """Set attribute if attrib is an ElemAttribBase. Pass silently if None"""
        self.set_attrib(*[x for x in attribs if isinstance(x, ElemAttribBase)])

    def get_attrib(self, type_: Type[ElemAttribBase]) -> Optional[Any]:
        """Try to get an attribute value, None if attrib is not set.

        :param type_: type of ElemAttribBase to seek in the attrib dictionary. This
            takes a type instead of a string to eliminate any possibility of getting
            a None value just because an attrib dictionary key was mistyped.
        """
        if hasattr(self, type_.__name__):
            return getattr(self, type_.__name__).value

    def __lt__(self: _TMeshElem, other: _TMeshElem) -> bool:
        """Sort by id

        You'll want to be able to sort Verts at least to make a vlvi (vertex list,
        vertex index) format.
        """
        return id(self) < id(other)


# argument to a function that returns same type as the input argument
_TFLapArg = TypeVar("_TFLapArg")


def _function_lap(
    func: Callable[[_TFLapArg], _TFLapArg], first_arg: _TFLapArg
) -> List[_TFLapArg]:
    """
    Repeatedly apply func till first_arg is reached again.

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
            raise ManifoldMeshError(f"infinite function lap {[id(x) for x in lap]}")


class Vert(MeshElementBase):
    """Half-edge mesh vertices.

    required attributes
    :edge: pointer to one edge originating at vert
    """

    _pointers = {"edge"}

    def __init__(self, *attributes: ElemAttribBase, edge: Optional[Edge] = None):
        super().__init__(*attributes, edge=edge)

    @property
    def edge(self) -> Edge:
        return self._edge

    @edge.setter
    def edge(self, edge_: Edge) -> None:
        self._edge = edge_
        edge_._orig = self

    @property
    def edges(self) -> List[Edge]:
        """Half edges radiating from vert."""
        if hasattr(self, "edge"):
            return self.edge.vert_edges
        return []

    @property
    def all_faces(self) -> List[Face]:
        """Faces radiating from vert"""
        if hasattr(self, "edge"):
            return self.edge.vert_all_faces
        return []

    @property
    def faces(self) -> List[Face]:
        """Faces radiating from vert"""
        return [x for x in self.all_faces if not x.is_hole]

    @property
    def holes(self) -> List[Face]:
        """Faces radiating from vert"""
        return [x for x in self.all_faces if x.is_hole]

    @property
    def neighbors(self) -> List[Vert]:
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

    _pointers = {"orig", "pair", "face", "next", "prev"}

    @property
    def orig(self) -> Vert:
        return self._orig

    @orig.setter
    def orig(self, orig: Vert) -> None:
        self._orig = orig
        orig._edge = self

    @property
    def pair(self) -> Edge:
        return self._pair

    @pair.setter
    def pair(self, pair: Edge) -> None:
        self._pair = pair
        pair._pair = self

    @property
    def face(self) -> Face:
        return self._face

    @face.setter
    def face(self, face_: Face) -> None:
        self._face = face_
        face_._edge = self

    @property
    def next(self: Edge) -> Edge:
        return self._next

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
    def prev(self, prev) -> None:
        super(Edge, prev).__setattr__("next", self)

    @property
    def dest(self) -> Vert:
        """Vert at the end of the edge (opposite of orig)."""
        try:
            return self.next.orig
        except AttributeError:
            return self.pair.orig

    @property
    def face_edges(self) -> List[Edge]:
        """All edges around an edge.face."""

        def _get_next(edge: Edge) -> Edge:
            return edge.next

        return _function_lap(_get_next, self)

    @property
    def face_verts(self) -> List[Vert]:
        """All verts around an edge.vert."""
        return [edge.orig for edge in self.face_edges]

    @property
    def vert_edges(self) -> List[Edge]:
        """
        All half edges radiating from edge.orig.

        These will be returned in the opposite "handedness" of the faces. IOW,
        if the faces are defined ccw, the vert_edges will be returned cw.
        """
        return _function_lap(lambda x: x.pair.next, self)

    @property
    def vert_all_faces(self) -> List[Face]:
        """
        All faces and holes around the edge's vert
        """
        return [x.face for x in self.vert_edges]

    @property
    def vert_faces(self) -> List[Face]:
        """
        All faces around the edge's vert
        """
        return [x for x in self.vert_all_faces if not x.is_hole]

    @property
    def vert_holes(self) -> List[Face]:
        """
        All holes around the edge's vert
        """
        return [x for x in self.vert_all_faces if x.is_hole]

    @property
    def vert_neighbors(self) -> List[Vert]:
        """All verts connected to vert by one edge."""
        return [edge.dest for edge in self.vert_edges]


class Face(MeshElementBase):
    """Half-edge mesh faces.

    required attribute
    :edge: pointer to one edge on the face
    """

    _pointers = {"edge"}

    def __init__(
        self, *args, edge: Optional[Edge] = None, is_hole: bool = False
    ) -> None:
        if is_hole:
            args += (IsHole(),)
        if isinstance(edge, Edge):
            kwargs = {"edge": edge}
        else:
            kwargs = {}
        super().__init__(*args, **kwargs)

    @property
    def edge(self) -> Edge:
        """One edge on the face"""
        return self._edge

    @edge.setter
    def edge(self, edge: Edge) -> None:
        """Point face.edge back to face."""
        self._edge = edge
        edge._face = self

    @property
    def is_hole(self) -> bool:
        """
        Is this face a hole?

        "hole-ness" is assigned at instance creation by passing ``is_hole=True`` to
        ``__init__``
        """
        return hasattr(self, "IsHole")

    @property
    def edges(self) -> List[Edge]:
        """Look up all edges around face."""
        if hasattr(self, "edge"):
            return self.edge.face_edges
        return []

    @property
    def verts(self) -> List[Vert]:
        """Look up all verts around face."""
        if hasattr(self, "edge"):
            return [x.orig for x in self.edges]
        return []

    @property
    def sides(self) -> int:
        """The equivalent of valence for faces. How many sides does the face have?"""
        return len(self.verts)
