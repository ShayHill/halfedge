#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# Last modified: 181204 13:07:13
"""A half-edges data container with view methods.

A simple container for a list of half edges. Provides lookups and a serial
number for each mesh element (Vert, Edge, Face, or Hole).

No transformations, with the exception of _MeshElementBase.assign_new_sn.

# 2006 June 05
# 2012 September 30
"""
from __future__ import annotations
from contextlib import suppress
from operator import attrgetter

from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Dict,
    Optional,
)

T = TypeVar("T")

Coordinate = Any
# TODO: remove coordinate


class ManifoldMeshError(ValueError):
    """Incorrect arguments passed to HalfEdges init.

    ... or something broken along the way. List of edges do not represent a valid
    (manifold) half edge data structure. Some properties will still be available,
    so you might catch this one and continue, but many operations will require valid,
    manifold mesh data to infer.
    """


class _MeshElementBase:
    """A namespace that == on id, not equivalency. counts instances.

    Vert, Edge, Face, and Hole elements

        * are assigned a serial number at creation.
        * are sortable by serial number
        * current highest serial number is available as self.last_issued_sn

    This allows for

        1. capture last issued serial number;
        2. transform mesh; then
        3. filter for new serial numbers.
    """

    sn: int
    last_issued_sn: int = -1

    def __init__(
        self: T,
        mesh: Optional[HalfEdges] = None,
        *,
        fill_from: Optional[T] = None,
        **kwargs: Any,
    ) -> None:
        """
        Create a mesh instance with no

        :param mesh:
        :param fill_from:
        :param kwargs:
        """

        _MeshElementBase.last_issued_sn += 1
        self.sn = self.last_issued_sn
        # TODO: factor out mesh
        if mesh is not None:
            self.mesh = mesh
        for key, val in kwargs.items():
            with suppress(AttributeError):
                setattr(self, key, val)
        if fill_from is not None:
            self.fill_from(fill_from)

    # def assign_new_sn(self) -> None:
    #     """Raise the serial number so this will look like a new element."""
    #     # TODO: factor out
    #     if hasattr(self, 'sn'):
    #         return

    def fill_from(self: T, other: T) -> None:
        """Copy attributes from :other:. Do not overwrite existing self attributes."""
        for key, val in other.__dict__.items():
            setattr(self, key, getattr(self, key, val))


def _function_lap(func: Callable[[T], T], first_arg: T) -> List[T]:
    """Repeatedly apply func till first_arg is reached again.

    [first_arg, func(first_arg), func(func(first_arg)) ... first_arg]
    """
    lap = [first_arg]
    while True:
        lap.append(func(lap[-1]))
        if lap[-1] == lap[0]:
            return lap[:-1]
        if lap[-1] in lap[1:-1]:
            raise ManifoldMeshError(f"infinite function lap {[x.sn for x in lap]}")


class Vert(_MeshElementBase):
    """Half-edge mesh vertices.

    :edge: pointer to one edge originating at vert
    """

    @property
    def edge(self) -> Edge:
        return self._edge

    @edge.setter
    def edge(self, edge: Edge) -> None:
        self._edge = edge

    @property
    def edges(self) -> List[Edge]:
        """Half edges radiating from vert."""
        return self.edge.vert_edges

    @property
    def verts(self) -> List[Vert]:
        """Evert vert connected to vert by one edge."""
        return self.edge.vert_verts

    @property
    def valence(self) -> int:
        """Count the number of edges incident to vertex."""
        return len(self.edges)


class Edge(_MeshElementBase):
    """Half-edge mesh edges.

    required attributes
    :orig: pointer to vert at which edge originates
    :pair: pointer to edge running opposite direction over same verts
    :face: pointer to face
    :next: pointer to next edge along face
    """

    orig: Vert
    pair: Edge
    face: Face
    next: Edge
    fill_from: Edge

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
        """All half edges radiating from edge.orig.

        These will be returned in the opposite "handedness" of the faces. IOW,
        if the faces are defined ccw, the vert_edges will be returned cw.
        """
        return _function_lap(lambda x: x.pair.next, self)

    @property
    def vert_verts(self) -> List[Vert]:
        """All verts connected to vert by one edge."""
        return [edge.dest for edge in self.vert_edges]


class Face(_MeshElementBase):
    """Half-edge mesh faces.

    required attribute
    :edge: pointer to one edge on the face
    """

    fill_from: Face
    edge: Edge

    @property
    def edge(self) -> Edge:
        return self._edge

    @edge.setter
    def edge(self, edge: Edge) -> None:
        self._edge = edge

    @property
    def edges(self) -> List[Edge]:
        """Look up all edges around face."""
        return self.edge.face_edges

    @property
    def verts(self) -> List[Vert]:
        """Look up all verts around face."""
        return self.edge.face_verts


class Hole(Face):
    """A copy of Hole to differentiate b/t interior edges and boundaries."""


class HalfEdges:
    """Basic half edge lookups.

    Some properties require a manifold mesh, but the Edge type does support
    explicitly defined holes. Holes provide enough information to pair and link all
    half edges, but will be ignored in any "for face in" constructs.
    """

    def __init__(self, edges: Optional[Set[Edge]] = None) -> None:
        if edges is None:
            self.edges = set()
        else:
            self.edges = edges

    @property
    def verts(self) -> Set[Vert]:
        """Look up all verts in mesh."""
        return {x.orig for x in self.edges}

    @property
    def faces(self) -> Set[Face]:
        """Look up all faces in mesh."""
        return {x.face for x in self.edges if type(x.face) is Face}

    @property
    def holes(self) -> Set[Face]:
        """Look up all holes in mesh."""
        return {x.face for x in self.edges if type(x.face) is Hole}

    @property
    def elements(self) -> Set[_MeshElementBase]:
        """All elements in mesh"""
        return self.verts | self.edges | self.faces | self.holes

    @property
    def last_issued_sn(self) -> int:
        """Look up the last serial number issued to any mesh element."""
        return next(iter(self.edges)).last_issued_sn

    @property
    def boundary_edges(self) -> Set[Edge]:
        """Look up edges on holes."""
        return {x for x in self.edges if type(x.face) is Hole}

    @property
    def boundary_verts(self) -> Set[Vert]:
        """Look up all verts on hole boundaries."""
        return {x.orig for x in self.boundary_edges}

    @property
    def interior_edges(self):
        """Look up edges on faces."""
        return self.edges - self.boundary_edges

    @property
    def interior_verts(self) -> Set[Vert]:
        """Look up all verts not on hole boundaries."""
        return self.verts - self.boundary_verts

    @property
    def vl(self) -> List[Vert]:
        """ Sorted list of verts """
        return sorted(self.verts, key=attrgetter("sn"))

    @property
    def el(self) -> List[Edge]:
        """ Sorted list of edges """
        return sorted(self.edges, key=attrgetter("sn"))

    @property
    def fl(self) -> List[Edge]:
        """ Sorted list of faces """
        return sorted(self.faces, key=attrgetter("sn"))

    @property
    def hl(self) -> List[Edge]:
        """ Sorted list of holes """
        return sorted(self.holes, key=attrgetter("sn"))

    @property
    def _vert2list_index(self) -> Dict[Vert, int]:
        """self.vl mapped to list indices."""
        return {vert: cnt for cnt, vert in enumerate(self.vl)}

    @property
    def ei(self) -> Set[Tuple[int, int]]:
        """Edges as a set of paired vert indices."""
        v2i = self._vert2list_index
        return {(v2i[edge.orig], v2i[edge.dest]) for edge in self.edges}

    @property
    def fi(self) -> Set[Tuple[int, ...]]:
        """Faces as a set of tuples of vl indices.

        returns [[0, 1, 2, 3], [1, 0, 4, 5] ...]
        """
        v2i = self._vert2list_index
        return {tuple(v2i[x] for x in face.verts) for face in self.faces}

    @property
    def hi(self) -> Set[Tuple[int, ...]]:
        """Holes as a set of tuples of vl indices.

        returns [[0, 1, 2, 3], [1, 0, 4, 5] ...]
        """
        v2i = self._vert2list_index
        return {tuple(v2i[x] for x in hole.verts) for hole in self.holes}
