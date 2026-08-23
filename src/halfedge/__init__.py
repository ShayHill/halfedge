"""Import functions into the package namespace.

:author: Shay Hill
:created: 2026-01-25
"""

from halfedge.half_edge_constructors import BlindHalfEdges
from halfedge.half_edge_elements import Edge, Face, MeshElementBase, Vert
from halfedge.half_edge_object import HalfEdges
from halfedge.type_attrib import (
    Attrib,
    ContagionAttrib,
    IncompatibleAttrib,
    NumericAttrib,
    Vector2Attrib,
    Vector3Attrib,
)

__all__ = [
    "Attrib",
    "BlindHalfEdges",
    "ContagionAttrib",
    "Edge",
    "Face",
    "HalfEdges",
    "IncompatibleAttrib",
    "MeshElementBase",
    "NumericAttrib",
    "Vector2Attrib",
    "Vector3Attrib",
    "Vert",
]
