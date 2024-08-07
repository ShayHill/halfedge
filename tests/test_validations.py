"""Test functions in classes.py.

created: 170204 14:22:23
"""

from typing import Tuple

import pytest

from halfedge.half_edge_elements import ManifoldMeshError, Vert
from halfedge.half_edge_object import HalfEdges
from halfedge.half_edge_querries import StaticHalfEdges
from halfedge.type_attrib import IncompatibleAttrib
from halfedge.validations import validate_mesh


class Coordinate(IncompatibleAttrib[Tuple[float, ...]]):
    pass


def test_validate_mesh_empty() -> None:
    """Passes on empty mesh."""
    mesh = StaticHalfEdges(edges=set())
    # assert NOT raises
    validate_mesh(mesh)


def test_validate_mesh_next_pair_share_origin(he_mesh: HalfEdges) -> None:
    """Fails if next and pair do not share origin."""
    next(iter(he_mesh.edges)).orig = Vert()
    with pytest.raises(ManifoldMeshError) as err:
        validate_mesh(he_mesh)
    assert "next and pair do not share orig point" in err.value.args[0]


def test_validate_mesh_loop_edge(he_mesh: HalfEdges) -> None:
    """Fails if edge orig and dest are the same."""
    edge = next(iter(he_mesh.edges))
    edge.orig = edge.next.orig
    with pytest.raises(ManifoldMeshError) as err:
        validate_mesh(he_mesh)
    assert "loop edge" in err.value.args[0]


def test_validate_mesh_edge_orig(he_mesh: HalfEdges) -> None:
    """Fails if edge does not point to correct origin."""
    edge = next(iter(he_mesh.edges))
    edge.orig = edge.next.dest
    with pytest.raises(ManifoldMeshError) as err:
        validate_mesh(he_mesh)
    assert "next and pair do not share orig point" in err.value.args[0]


def test_validate_mesh_edge_face(he_mesh: HalfEdges) -> None:
    """Fails if edge points to wrong face."""
    edge = next(iter(he_mesh.edges))
    edge.face = edge.pair.face
    with pytest.raises(ManifoldMeshError) as err:
        validate_mesh(he_mesh)
    assert "face.edges do not all point to face" in err.value.args[0]


def test_disjoint_face() -> None:
    """Fails for disconnected faces."""
    vl = [Vert(Coordinate((0, 0, 0))) for _ in range(6)]
    mesh = StaticHalfEdges.from_vlvi(vl, {(0, 1, 2), (3, 4, 5)})
    with pytest.raises(ManifoldMeshError) as err:
        validate_mesh(mesh)
    assert "not all faces can be reached" in err.value.args[0]
