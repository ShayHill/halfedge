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

from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, TypeVar, Dict

T = TypeVar("T")

Coordinate = Any
# TODO: remove coordinate


class ManifoldMeshError(ValueError):
    """Incorrect arguments passed to HalfEdges init.

    ... or something broken along the way. List of edges do not represent a valid
    (manifold) half edge data structure. Some properties will still be available,
    but many require valid, manifold mesh data to infer.
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
    last_issued_sn = -1

    def __init__(
        self: T,
        mesh: Optional["HalfEdges"] = None,
        *,
        fill_from: Optional[T] = None,
        **kwargs: Any,
    ) -> None:
        self.assign_new_sn()
        if mesh is not None:
            self.mesh = mesh
        for key, val in kwargs.items():
            with suppress(AttributeError):
                setattr(self, key, val)
        if fill_from is not None:
            self.fill_from(fill_from)

    def __lt__(self: _MeshElementBase, other: _MeshElementBase) -> bool:
        return self.sn < other.sn

    def __gt__(self: _MeshElementBase, other: _MeshElementBase) -> bool:
        return self.sn > other.sn

    def assign_new_sn(self) -> None:
        """Raise the serial number so this will look like a new element."""
        _MeshElementBase.last_issued_sn += 1
        setattr(self, "sn", self.last_issued_sn)

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

    some coordinate value is necessary for testing
    :coordinate: coordinate of vert (e.g., np.array([1, -4, 2]))

    required attribute
    :edge: pointer to one edge originating at vert
    """

    coordinate: Sequence[float]
    uv_vector: Sequence[float]
    fill_from: Vert

    @property
    def edge(self) -> Edge:
        """
        Find the first edge that references vert.

        :return: Edge instance such that edge.orig == self

        Looks through every in the unordered set to find the first.
        """
        try:
            return min(e for e in self.mesh.edges if e.orig is self)
        except AttributeError as exc:
            raise AttributeError(
                str(exc)
                + ". This implementation does not store an edge value for each Vert."
                " These can only be found by searching the Vert's 'mesh'."
            )

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
    def prev(self) -> Edge:
        """Look up the edge before self."""
        try:
            return self.face_edges[-1]
        except (AttributeError, ManifoldMeshError):
            return self.vert_edges[-1].pair

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

    edge: Edge
    fill_from: Face

    @property
    def edge(self) -> Edge:
        try:
            return min(e for e in self.mesh.edges if e.face is self)
        except AttributeError as exc:
            raise AttributeError(
                str(exc)
                + ". This implementation does not store an edge value for each Face."
                  " These can only be found by searching the Face's 'mesh'."
            )

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
    def bounding_box(self) -> Tuple[Tuple[float], Tuple[float]]:
        """Look up min and max extent in all dimensions."""
        axes = tuple(zip(*(x.coordinate for x in self.verts)))
        return tuple(float(min(x)) for x in axes), tuple(float(max(x)) for x in axes)

    @property
    def vl(self) -> List[Tuple[float]]:
        """Export verts as a list of coordinate tuples.

        returns [(3, 4.4, -2.22), (13, 42, 344) ...]

        Returned in .sn order.
        """
        return [tuple(vert.coordinate) for vert in sorted(self.verts)]

    @property
    def _vert2list_index(self) -> Dict[Vert, int]:
        """Map Vert instances to indices in a sorted list of verts."""
        return {vert: cnt for cnt, vert in enumerate(sorted(self.verts))}

    @property
    def ei(self) -> List[List[int]]:
        """Export edges as a list of paired vert indices.

        returns [[0, 1], [1, 4] ...]

        Returned in .sn order.
        """
        vert2list_index = self._vert2list_index
        edges = [(edge.orig, edge.dest) for edge in sorted(self.edges)]
        return [[vert2list_index[vert] for vert in edge] for edge in edges]

    @property
    def vi(self) -> List[List[int]]:
        """Export faces as a list lists of vl indices.

        returns [[0, 1, 2, 3], [1, 0, 4, 5] ...]

        Returned in .sn order.
        """
        vert2list_index = self._vert2list_index
        faces = [face.verts for face in sorted(self.faces)]
        return [[vert2list_index[vert] for vert in face] for face in faces]

    @property
    def hi(self) -> List[List[int]]:
        """Export holes as a list of lists of vl indices.

        returns [[0, 1, 2, 3], [1, 0, 4, 5] ...]

        Returned in .sn order.
        """
        vert2list_index = self._vert2list_index
        faces = [face.verts for face in sorted(self.holes)]
        return [[vert2list_index[vert] for vert in face] for face in faces]
