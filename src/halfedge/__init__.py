"""Allow modules to be imported from top-level."""

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
