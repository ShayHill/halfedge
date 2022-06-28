#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# Last modified: 220628 12:15:40
"""A half-edges data container with view methods.

A simple container for a list of half edges. Provides lookups and a serial
number for each mesh element (Vert, Edge, or Face).

This is a typical halfedges data structure. Exceptions:

    * Face() is distinct from Face(__is_hole=True). This is
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

from inspect import get_annotations
from contextlib import suppress
from functools import reduce
from itertools import count
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Type,
    TypeVar,
    Hashable,
    Iterable,
    Set,
    Optional,
    Tuple
)

from .element_attributes import ElemAttribBase, find_optional_arg_type


class ManifoldMeshError(ValueError):
    """
    Incorrect arguments passed to HalfEdges init.

    ... or something broken along the way. List of edges do not represent a valid
    (manifold) half edge data structure. Some properties will still be available,
    so you might catch this one and continue in some cases, but many operations will
    require valid, manifold mesh data to infer.
    """


def all_equal(*args: Any):
    """
    Are all arguments equal, including type?

    :param args: will work with items or sequences
    """
    if len({type(x) for x in args}) > 1:
        return False
    with suppress(ValueError, TypeError):
        return all(x == args[0] for x in args[1:])
    with suppress(ValueError, TypeError):
        return all(all_equal(*x) for x in zip(args))
    raise NotImplementedError(f"module does not support equality test between {args}")


KeyT = TypeVar("KeyT")
_TMeshElem = TypeVar("_TMeshElem", bound="MeshElementBase")


def get_dict_intersection(*dicts: Dict) -> Dict:
    """
    Identical key: value items from multiple dictionaries.

    :param dicts: any number of dictionaries

    Return a dictionary of keys: values where key and value are equal for all input
    dicts.

    """
    if not dicts:
        return {}
    intersection = {}
    for key in set.intersection(*(set(x.keys()) for x in dicts)):
        with suppress(KeyError):
            if all_equal(*(x[key] for x in dicts)):
                intersection[key] = dicts[0][key]
    return intersection


def merge_attribs_dicts(
    element: MeshElementBase, *attribs_dicts: Dict[str, ElemAttribBase]
) -> Dict[str, ElemAttribBase]:
    keys = set.union(*(set(x) for x in attribs_dicts))
    for key in keys:
        attrib_per_dict = [x.get(key) for x in attribs_dicts]
        type_ = find_optional_arg_type(attrib_per_dict)
        with suppress(TypeError):
            attrib = type_(None, element, *attrib_per_dict)
            merged[type(attrib).__name__] = attrib
    return merged


_T = TypeVar("_T")
TVert = TypeVar("TVert", bound="Vert")
TEdge = TypeVar("TEdge", bound="Edge")
TFace = TypeVar("TFace", bound="Face")


# TODO: refactor this with slots instead of gaming annotations
# TODO: replace 'attributes' with pointers

def all_is(*args: Any) -> bool:
    """True if all arguments a is b"""
    return args and all(args[0] is x for x in args[1:])


class MeshElementBase(Generic[TVert, TEdge, TFace]):
    _sn_generator = count()

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

        This class does will not have pointers. Descendent classes will, and it is
        critical that each have a setter and each setter cache a value as _pointer
        (e.g., _vert, _pair, _face, _next).
        """
        self.sn = next(self._sn_generator)
        for attribute in attributes:
            self.set_attrib(attribute)
        for k, v in pointers.items():
            if k in pointers:
                setattr(self, k, v)
                continue
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{k}'")


    def __setattr__(self, key: str, value: ElemAttribBase | MeshElementBase) -> None:
        """To prevent any mistyped attributes, which would clobber fill_from
        """
        allow = key == 'sn'
        allow = allow or key.startswith('_')
        allow = allow or key in self.__dict__
        allow = allow or key in dir(self)
        allow = allow or isinstance(value, ElemAttribBase)
        if allow:
            super().__setattr__(key, value)
            return
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")

    def fill_from(self: _TMeshElem, *elements: _TMeshElem) -> None:
        """
        Fill in missing references from other elements.
        """
        keys_seen = {k for k, v in self.__dict__.items() if v is not None}
        for element in elements:
            for key in (x for x in element.__dict__.keys() if x not in keys_seen):
                keys_seen.add(key)
                vals = [getattr(x, key) for x in elements]
                if isinstance(getattr(element, key), ElemAttribBase):
                    self.maybe_set_attrib(type(getattr(element, key)).merged(*vals))
                elif all_is(vals):  # will have to be '_something'
                    setattr(self, key[1:], vals[0])

    def set_attrib(self, *attribs: ElemAttribBase) -> None:
        """Set attribute with an ElemAttribBase instance.

        type(attrib).__name__ : attrib

        :param attrib: an ElemAttribBase instance, presumable with a None element
        attribute.
        """
        for attrib in attribs:
            attrib.element = self
            setattr(self, type(attrib).__name__, attrib)

    def maybe_set_attrib(self, *attribs: Optional[ElemAttribBase]) -> None:
        """Set attribute if attrib is an ElemAttribBase. Pass silently if None
        """
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


class Vert(MeshElementBase[TVert, TEdge, TFace]):
    """Half-edge mesh vertices.

    required attributes
    :edge: pointer to one edge originating at vert
    """

    @property
    def edge(self) -> TEdge:
        return self._edge

    @edge.setter
    def edge(self, edge_: TEdge) -> None:
        self._edge = edge_
        edge_._orig = self

    @property
    def edges(self) -> List[TEdge]:
        """Half edges radiating from vert."""
        if hasattr(self, "edge"):
            return self.edge.vert_edges
        return []

    @property
    def all_faces(self) -> List[TFace]:
        """Faces radiating from vert"""
        if hasattr(self, "edge"):
            return self.edge.vert_all_faces
        return []

    @property
    def faces(self) -> List[TFace]:
        """Faces radiating from vert"""
        return [x for x in self.all_faces if not x.is_hole]

    @property
    def holes(self) -> List[TFace]:
        """Faces radiating from vert"""
        return [x for x in self.all_faces if x.is_hole]

    @property
    def neighbors(self) -> List[TVert]:
        """Evert vert connected to vert by one edge."""
        if hasattr(self, "edge"):
            return self.edge.vert_neighbors
        return []

    @property
    def valence(self) -> int:
        """The number of edges incident to vertex."""
        return len(self.edges)


class Edge(MeshElementBase[TVert, TEdge, TFace]):
    """Half-edge mesh edges.

    required attributes
    :orig: pointer to vert at which edge originates
    :pair: pointer to edge running opposite direction over same verts
    :face: pointer to face
    :next: pointer to next edge along face
    """

    _pointers: Set[str] = {'_orig', '_pair', '_face', '_next', 'prev'}

    @property
    def orig(self) -> TVert:
        return self._orig

    @orig.setter
    def orig(self, orig: TVert) -> None:
        self._orig = orig
        orig._edge = self

    @property
    def pair(self) -> TEdge:
        return self._pair

    @pair.setter
    def pair(self, pair: TEdge) -> None:
        self._pair = pair
        pair._pair = self

    @property
    def face(self) -> TFace:
        return self._face

    @face.setter
    def face(self, face_: TFace) -> None:
        self._face = face_
        face_._edge = self

    @property
    def next(self: TEdge) -> TEdge:
        return self._next

    @next.setter
    def next(self: TEdge, next_: TEdge) -> None:
        self._next = next_

    @property
    def prev(self) -> TEdge:
        """Look up the edge before self."""
        try:
            return self.face_edges[-1]
        except (AttributeError, ManifoldMeshError):
            return self.vert_edges[-1].pair

    @prev.setter
    def prev(self, prev) -> None:
        super(Edge, prev).__setattr__("next", self)

    @property
    def dest(self) -> TVert:
        """Vert at the end of the edge (opposite of orig)."""
        try:
            return self.next.orig
        except AttributeError:
            return self.pair.orig

    @property
    def face_edges(self: TEdge) -> List[TEdge]:
        """All edges around an edge.face."""

        def _get_next(edge: TEdge) -> TEdge:
            return edge.next

        return _function_lap(_get_next, self)

    @property
    def face_verts(self) -> List[TVert]:
        """All verts around an edge.vert."""
        return [edge.orig for edge in self.face_edges]

    @property
    def vert_edges(self: TEdge) -> List[TEdge]:
        """
        All half edges radiating from edge.orig.

        These will be returned in the opposite "handedness" of the faces. IOW,
        if the faces are defined ccw, the vert_edges will be returned cw.
        """
        return _function_lap(lambda x: x.pair.next, self)

    @property
    def vert_all_faces(self) -> List[TFace]:
        """
        All faces and holes around the edge's vert
        """
        return [x.face for x in self.vert_edges]

    @property
    def vert_faces(self) -> List[TFace]:
        """
        All faces around the edge's vert
        """
        return [x for x in self.vert_all_faces if not x.is_hole]

    @property
    def vert_holes(self) -> List[TFace]:
        """
        All holes around the edge's vert
        """
        return [x for x in self.vert_all_faces if x.is_hole]

    @property
    def vert_neighbors(self) -> List[TVert]:
        """All verts connected to vert by one edge."""
        return [edge.dest for edge in self.vert_edges]


class Face(MeshElementBase[TVert, TEdge, TFace]):
    """Half-edge mesh faces.

    required attribute
    :edge: pointer to one edge on the face
    """
    _pointers: Set[str] = {'_edge', '_Face__is_hole'}

    @classmethod
    def factory(cls: Type[TFace]) -> TFace:
        return cls()

    def __init__(self, *args, **kwargs) -> None:
        self.__is_hole = kwargs.pop("__is_hole", False)
        super().__init__(*args, **kwargs)

    @property
    def edge(self) -> TEdge:
        """One edge on the face"""
        return self._edge

    @edge.setter
    def edge(self, edge: TEdge) -> None:
        """Point face.edge back to face."""
        self._edge = edge
        edge._face = self

    @property
    def is_hole(self) -> bool:
        """
        Is this face a hole?

        "hole-ness" is assigned at instance creation by passing ``__is_hole=True`` to
        ``__init__``
        """
        return self.__is_hole

    @property
    def edges(self) -> List[TEdge]:
        """Look up all edges around face."""
        if hasattr(self, "edge"):
            return self.edge.face_edges
        return []

    @property
    def verts(self) -> List[TVert]:
        """Look up all verts around face."""
        if hasattr(self, "edge"):
            return [x.orig for x in self.edges]
        return []

    @property
    def sides(self) -> int:
        """The equivalent of valence for faces. How many sides does the face have?"""
        return len(self.verts)
