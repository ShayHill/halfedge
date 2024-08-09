"""Test exceptions and details of the BlindHalfEdges subclasses.

Most of the BlindHalfEdges subclass code is run to set up other tests. This module
tests exceptions and other details missed by the other tests.

:author: Shay Hill
:created: 2024-08-09
"""

# pyright: reportPrivateUsage=false

from halfedge.half_edge_object import HalfEdges
from halfedge.half_edge_elements import Face, Edge, Vert
import pytest
from .conftest import compare_circular_2, Coordinate, get_canonical_index_tuple


def test_get_edge_or_vert_faces_with_vert_arg() -> None:
    """Return edge.face when edge argument is passed."""
    edge = Edge(face=Face())
    assert HalfEdges()._get_edge_or_vert_faces(edge) == {edge.face}


def test_raise_for_non_tris_in_flip_edge(he_cube: HalfEdges) -> None:
    """Raise if face is not a triangle."""
    edge = next(iter(he_cube.edges))
    with pytest.raises(ValueError) as err:
        _ = he_cube.flip_edge(edge)
    assert "between two triangles" in err.value.args[0]


def test_all_elements(he_cube: HalfEdges) -> None:
    """Return all faces, edges, and verts as elements attribute."""
    assert he_cube.elements == he_cube.faces | he_cube.edges | he_cube.verts


def test_el(he_mesh: HalfEdges) -> None:
    """Return edges as a sorted list mesh.edges."""
    assert he_mesh.el == sorted(he_mesh.edges)


def test_fl(he_mesh: HalfEdges) -> None:
    """Return faces as a sorted list mesh.faces."""
    assert he_mesh.fl == sorted(he_mesh.faces)


def test_fi_explicit(he_cube: HalfEdges) -> None:
    """Return faces as a set of n-tuples of vl indices."""
    expect = {
        (2, 6, 7, 3),
        (0, 4, 5, 1),
        (0, 3, 7, 4),
        (4, 7, 6, 5),
        (0, 1, 2, 3),
        (1, 5, 6, 2),
    }
    fi = {get_canonical_index_tuple(x) for x in he_cube.fi}
    assert fi == expect

def test_hi_explicit(he_grid: HalfEdges) -> None:
    """Return holes as a set of n-tuples of vl indices."""
    expect = {(0, 4, 8, 12, 13, 14, 15, 11, 7, 3, 2, 1)}
    hi = {get_canonical_index_tuple(x) for x in he_grid.hi}
    assert hi == expect

def test_ei() -> None:
    """Return edges as a set of 2-tuples of vl indices."""
    vl = [Vert() for _ in range(3)]
    fi = {(0, 1, 2)}
    mesh = HalfEdges.from_vlfi(vl, fi)
    assert mesh.ei == {(0, 1), (1, 2), (2, 1), (2, 0), (0, 2), (1, 0)}
