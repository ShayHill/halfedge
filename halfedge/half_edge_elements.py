#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# Last modified: 181204 13:07:13
"""A half-edges data container with view methods.

A simple container for a list of half edges. Provides lookups and a serial
number for each mesh element (Vert, Edge, Face, or Hole).

This is a typical halfedges data structure. Exceptions:

    * Faces and Holes are two different (but identical except in name) types. This is
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
      of tests if you want to extend the insertion / removal methods here.

This module is all the base elements (Vert, Edge, Face, Hole).

# 2006 June 05
# 2012 September 30
"""
from __future__ import annotations

from contextlib import suppress
from typing import Any, Callable, Dict, List, TypeVar, Union


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


def get_dict_intersection(*dicts: Dict[KeyT, Any]) -> Dict[KeyT, Any]:
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


_TMeshElem = TypeVar("_TMeshElementBase", bound="_MeshElementBase")


class _MeshElementBase:
    """
    A namespace that == on id, not equivalency.
    """

    def __init__(self: _TMeshElem, *fill_from: _TMeshElem, **kwargs: Any) -> None:
        """
        Create an instance with a new serial number (and copy attrs from fill_from).

        :param fill_from: instances of the same class
        :param kwargs: attributes for new instance

        Priority for attributes
        high: attributes passed as kwargs to init
        low: attributes inherited from fill_from argument
        """
        self.update(*fill_from, **kwargs)

    def update(self: _TMeshElem, *fill_from: _TMeshElem, **kwargs: Any) -> None:
        """
        Add or replace attributes

        :param fill_from: instances of the same type
        :param kwargs: new attribute values (supersede fill_from attributes)
        """
        attrs = get_dict_intersection(*(x.__dict__ for x in fill_from))
        attrs.update(kwargs)
        for key, val in attrs.items():
            setattr(self, key, val)

    def extend(self: _TMeshElem, *fill_from: _TMeshElem, **kwargs: Any) -> None:
        """Add attributes only. Do not replace."""
        attrs = get_dict_intersection(*(x.__dict__ for x in fill_from))
        attrs.update(kwargs)
        attrs.update(self.__dict__)
        for key, val in attrs.items():
            setattr(self, key, val)

    def __lt__(self: _TMeshElem, other: _TMeshElem) -> bool:
        """Sort by id"""
        return id(self) < id(other)


# argument to a function that returns an instance of the input argument
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


class Vert(_MeshElementBase):
    """Half-edge mesh vertices.

    required attributes
    :edge: pointer to one edge originating at vert
    """

    _edge: Edge
    coordinate: Any

    @property
    def edge(self) -> Edge:
        breakpoint()
        return self._edge

    @edge.setter
    def edge(self, edge: Edge) -> None:
        self._edge = edge

    @property
    def edges(self) -> List[Edge]:
        """Half edges radiating from vert."""
        if hasattr(self, "edge"):
            return self.edge.vert_edges
        return []

    @property
    def faces(self) -> List[Face]:
        """Faces radiating from vert"""
        return [x for x in self.all_faces if not isinstance(x, Hole)]

    @property
    def holes(self) -> List[Hole]:
        """Faces radiating from vert"""
        return [x for x in self.all_faces if isinstance(x, Hole)]

    @property
    def all_faces(self) -> List[Union[Face, Hole]]:
        """Faces radiating from vert"""
        return [x.face for x in self.edges]

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


class Edge(_MeshElementBase):
    """Half-edge mesh edges.

    required attributes
    :orig: pointer to vert at which edge originates
    :pair: pointer to edge running opposite direction over same verts
    :face: pointer to face
    :next: pointer to next edge along face
    """

    _orig: Vert
    _pair: Edge
    _face: Union[Face, Hole]
    _next: Edge

    @property
    def orig(self) -> Vert:
        return self._orig

    @orig.setter
    def orig(self, orig: Vert) -> None:
        self._orig = orig
        orig.edge = self

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
    def face(self, face: Face) -> None:
        self._face = face
        face.edge = self

    @property
    def next(self) -> Edge:
        return self._next

    @next.setter
    def next(self, next_: Edge) -> None:
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
        return _function_lap(lambda x: x.next, self)

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
    def vert_neighbors(self) -> List[Vert]:
        """All verts connected to vert by one edge."""
        return [edge.dest for edge in self.vert_edges]


class Face(_MeshElementBase):
    """Half-edge mesh faces.

    required attribute
    :edge: pointer to one edge on the face
    """

    _edge: Edge

    @property
    def edge(self) -> Edge:
        """One edge on the face"""
        return self._edge

    @edge.setter
    def edge(self, edge: Edge) -> None:
        """Point face.edge back to face."""
        self._edge = edge

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


class Hole(Face):
    """A copy of Face to differentiate b/t interior edges and boundaries."""
