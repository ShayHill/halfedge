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
from typing import TYPE_CHECKING, Any, Callable, List, Optional, TypeVar

from .type_attrib import Attrib, AttribHolder, ContagionAttrib

if TYPE_CHECKING:
    from .half_edge_constructors import BlindHalfEdges

_TMeshElem = TypeVar("_TMeshElem", bound="MeshElementBase")


class IsHole(ContagionAttrib):
    """Flag a Face instance as a hole"""


class ManifoldMeshError(ValueError):
    """Incorrect arguments passed to HalfEdges init.

    ... or something broken along the way. List of edges do not represent a valid
    (manifold) half edge data structure. Some properties will still be available,
    so you might catch this one and continue in some cases, but many operations will
    require valid, manifold mesh data to infer.
    """


def _all_is(*args: Any) -> bool:
    """Return True if all arguments are `a is b`"""
    return bool(args) and all(args[0] is x for x in args[1:])


class MeshElementBase(AttribHolder):
    """Base class for Vert, Edge, and Face."""

    _sn_generator = count()
    _pointers = {"mesh"}

    def __init__(
        self,
        *attributes: Attrib,
        mesh: BlindHalfEdges | None = None,
        **pointers: MeshElementBase | None,
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
        for attribute in attributes:
            self.set_attrib(attribute)
        if mesh is not None:
            self.mesh = mesh
        for k, v in pointers.items():
            if v is not None:
                setattr(self, k, v)

    @property
    def mesh(self) -> BlindHalfEdges:
        """Return the mesh instance."""
        return self._mesh

    @mesh.setter
    def mesh(self, mesh_: BlindHalfEdges) -> None:
        self._mesh = mesh_

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
        allow = allow or isinstance(value, Attrib)
        if allow:
            super().__setattr__(key, value)
            return
        msg = f"'{type(self).__name__}' has no attribute '{key}'"
        raise AttributeError(msg)

    def merge_from(self: _TMeshElem, *elements: _TMeshElem) -> _TMeshElem:
        """Fill in missing references from other elements."""
        keys_seen = {k for k, v in self.__dict__.items() if v is not None}
        for element in elements:
            for key in (x for x in element.__dict__.keys() if x not in keys_seen):
                keys_seen.add(key)
                vals = [getattr(x, key, None) for x in elements]
                if isinstance(getattr(element, key), Attrib):
                    self._maybe_set_attrib(type(getattr(element, key)).merge(*vals))
                    continue
                if _all_is(*vals):  # will have be something in _pointers
                    setattr(self, key.lstrip("_"), vals[0])
        return self

    def slice_from(self: _TMeshElem, element: _TMeshElem) -> _TMeshElem:
        """Pass attributes when dividing or altering elements.

        Do not pass any pointers. ElemAttribBase instances are passed as defined by
        their classes.
        """
        for key in set(element.__dict__) - set(self.__dict__):
            self._maybe_set_attrib(getattr(element, key))
        return self

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
            msg = f"function lap {[id(x) for x in lap]} repeats"
            raise ManifoldMeshError(msg)


class Vert(MeshElementBase):
    """Half-edge mesh vertices.

    required attributes
    :edge: pointer to one edge originating at vert
    """

    _pointers = {"mesh", "edge"}

    def __init__(
        self,
        *attributes: Attrib,
        mesh: BlindHalfEdges | None = None,
        edge: Edge | None = None,
    ):
        """Create a vert instance."""
        super().__init__(*attributes, mesh=mesh, edge=edge)

    @property
    def edge(self) -> Edge:
        """One edge originating at vert."""
        return self._edge

    @edge.setter
    def edge(self, edge_: Edge) -> None:
        self._edge = edge_
        edge_._orig = self

    @property
    def edges(self) -> list[Edge]:
        """Half edges radiating from vert."""
        if hasattr(self, "edge"):
            return self.edge.vert_edges
        return []

    @property
    def all_faces(self) -> list[Face]:
        """Faces radiating from vert"""
        if hasattr(self, "edge"):
            return self.edge.vert_all_faces
        return []

    @property
    def faces(self) -> list[Face]:
        """Faces radiating from vert"""
        return [x for x in self.all_faces if not x.is_hole]

    @property
    def holes(self) -> list[Face]:
        """Faces radiating from vert"""
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

    _pointers = {"mesh", "orig", "pair", "face", "next", "prev"}

    def __init__(
        self,
        *attributes: Attrib,
        mesh: BlindHalfEdges | None = None,
        orig: Vert | None = None,
        pair: Edge | None = None,
        face: Face | None = None,
        next: Edge | None = None,
        prev: Edge | None = None,
    ):
        """Create an edge instance."""
        super().__init__(
            *attributes,
            mesh=mesh,
            orig=orig,
            pair=pair,
            face=face,
            next=next,
            prev=prev,
        )

    @property
    def orig(self) -> Vert:
        """Vert at which edge originates."""
        return self._orig

    @orig.setter
    def orig(self, orig: Vert) -> None:
        self._orig = orig
        orig._edge = self

    @property
    def pair(self) -> Edge:
        """Edge running opposite direction over same verts."""
        return self._pair

    @pair.setter
    def pair(self, pair: Edge) -> None:
        self._pair = pair
        pair._pair = self

    @property
    def face(self) -> Face:
        """Face to which edge belongs."""
        return self._face

    @face.setter
    def face(self, face_: Face) -> None:
        self._face = face_
        face_._edge = self

    @property
    def next(self: Edge) -> Edge:
        """Next edge along face."""
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
    """

    _pointers = {"mesh", "edge"}

    def __init__(
        self,
        *attributes: Attrib,
        mesh: BlindHalfEdges | None = None,
        edge: Edge | None = None,
        is_hole: bool = False,
    ) -> None:
        """Create a face instance."""
        if is_hole:
            attributes += (IsHole(),)
        super().__init__(*attributes, mesh=mesh, edge=edge)

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
        """Is this face a hole?

        "hole-ness" is assigned at instance creation by passing ``is_hole=True`` to
        ``__init__``
        """
        return hasattr(self, "IsHole")

    @property
    def edges(self) -> list[Edge]:
        """Look up all edges around face."""
        if hasattr(self, "edge"):
            return self.edge.face_edges
        return []

    @property
    def verts(self) -> list[Vert]:
        """Look up all verts around face."""
        if hasattr(self, "edge"):
            return [x.orig for x in self.edges]
        return []

    @property
    def sides(self) -> int:
        """The equivalent of valence for faces. How many sides does the face have?"""
        return len(self.verts)
