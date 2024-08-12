"""Allow modules to be imported from top-level."""

from halfedge.half_edge_constructors import BlindHalfEdges
from halfedge.half_edge_elements import Edge, Face, MeshElementBase, Vert
from halfedge.half_edge_object import HalfEdges
from halfedge.type_attrib import (
    Attrib,
    ContagionAttrib,
    EdgeAttrib,
    FaceAttrib,
    IncompatibleAttrib,
    NumericAttrib,
    Vector2Attrib,
    Vector3Attrib,
    VertAttrib,
)

__all__ = [
    "Attrib",
    "BlindHalfEdges",
    "ContagionAttrib",
    "Edge",
    "EdgeAttrib",
    "Face",
    "FaceAttrib",
    "HalfEdges",
    "IncompatibleAttrib",
    "MeshElementBase",
    "NumericAttrib",
    "Vector2Attrib",
    "Vector3Attrib",
    "Vert",
    "VertAttrib",
]
