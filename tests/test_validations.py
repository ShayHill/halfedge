# Last modified: 181126 16:46:28
# _*_ coding: utf-8 _*_
"""Test functions in classes.py.

created: 170204 14:22:23
"""
from typing import Any, Dict

import pytest

from ..halfedge.half_edge_elements import ManifoldMeshError, Vert
from ..halfedge.half_edge_querries import StaticHalfEdges
from ..halfedge.validations import validate_mesh


def test_validate_mesh_empty() -> None:
    """Passes on empty mesh."""
    mesh = StaticHalfEdges(edges=set())
    # assert NOT raises
    validate_mesh(mesh)


def test_validate_mesh_next_pair_share_origin(he_meshes: Dict[str, Any]) -> None:
    """Fails if next and pair do not share origin."""
    cube = he_meshes["cube"]
    next(iter(cube.edges)).orig = Vert()
    with pytest.raises(ManifoldMeshError) as err:
        validate_mesh(cube)
    assert "next or pair error" in err.value.args[0]


def test_validate_mesh_loop_edge(he_meshes: Dict[str, Any]) -> None:
    """Fails if edge orig and dest are the same."""
    cube = he_meshes["cube"]
    edge = next(iter(cube.edges))
    edge.orig = edge.next.orig
    with pytest.raises(ManifoldMeshError) as err:
        validate_mesh(cube)
    assert "loop edge" in err.value.args[0]


def test_validate_mesh_edge_orig(he_meshes: Dict[str, Any]) -> None:
    """Fails if edge does not point to correct origin."""
    cube = he_meshes["cube"]
    edge = next(iter(cube.edges))
    edge.orig = edge.next.dest
    with pytest.raises(ManifoldMeshError) as err:
        validate_mesh(cube)
    assert "next or pair error" in err.value.args[0]


def test_validate_mesh_edge_face(he_meshes: Dict[str, Any]) -> None:
    """Fails if edge points to wrong face."""
    cube = he_meshes["cube"]
    edge = next(iter(cube.edges))
    edge.face = edge.pair.face
    with pytest.raises(ManifoldMeshError) as err:
        validate_mesh(cube)
    assert "wrong face" in err.value.args[0]


def test_disjoint_face() -> None:
    """Fails for disconnected faces."""
    vl = [(0, 0, 0)] * 6
    vl = [{'coordinate': x} for x in vl]
    mesh = StaticHalfEdges.from_vlvi(vl, {(0, 1, 2), (3, 4, 5)})
    with pytest.raises(ManifoldMeshError) as err:
        validate_mesh(mesh)
    assert "not all faces can be reached" in err.value.args[0]
