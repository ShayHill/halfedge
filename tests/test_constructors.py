# Last modified: 181126 16:46:28
# _*_ coding: utf-8 _*_
"""Test functions in constructors.py.

created: 181127
"""

from typing import Dict, Any

import pytest
from halfedge.classes import Hole, ManifoldMeshError, Vert
from halfedge.constructors import (
    edges_from_vlvi,
    edges_from_vr,
    infer_holes,
    mesh_from_vlvi,
    mesh_from_vr,
)
from tests.helpers import are_equivalent_edges


def test_edges_from_vlvi_given_hole(he_triangle: Dict[str, Any]) -> None:
    """Generated edges are equivalent to hand-calculated edges."""
    ctrl = set(he_triangle["edges"])
    vl = [x.coordinate for x in sorted(he_triangle["verts"])]
    test = edges_from_vlvi(vl, [[0, 1, 2]], [[0, 2, 1]])
    assert are_equivalent_edges(ctrl, test)


def test_edges_from_vlvi_infer_holes(meshes_vlvi: Dict[str, Any]) -> None:
    """Inferred holes are equivalent to explicit holes."""
    vl = meshes_vlvi["grid_vl"]
    vi = meshes_vlvi["grid_vi"]
    hi = meshes_vlvi["grid_hi"]

    # make an interior hole (so, two holes in mesh now)
    vi_ = vi[:4] + vi[5:]
    hi_ = hi + vi[4:5]

    ctrl_edges = edges_from_vlvi(vl, vi_, hi_)
    test_edges = edges_from_vlvi(vl, vi_)
    assert are_equivalent_edges(test_edges, ctrl_edges)


def test_infer_holes_fails_kissing(meshes_vlvi: Dict[str, Any]) -> None:
    """Fail for ambiguous case when hole corners meet"""
    part_vi = [y for x, y in enumerate(meshes_vlvi["grid_vi"]) if x not in (0, 4)]
    with pytest.raises(ManifoldMeshError) as err:
        edges_from_vlvi(meshes_vlvi["grid_vl"], part_vi)
    assert "Ambiguous 'next'" in err.value.args[0]


def test_infer_holes_fails_on_paired_edge_on_inferred_hole_boundary(
    meshes_vlvi: Dict[str, Any]
) -> None:
    """Fail when an inferred edge has nothing to connect to."""
    edges = edges_from_vlvi(meshes_vlvi["grid_vl"], meshes_vlvi["grid_vi"])

    edge = next(x for x in edges if isinstance(x.face, Hole))
    del edge.pair
    del edge

    with pytest.raises(ManifoldMeshError) as err:
        infer_holes(edges)
    assert "inferred hole boundary" in err.value.args[0]


def test_mesh_from_vlvi(meshes_vlvi: Dict[str, Any]) -> None:
    """Mesh is equivalent to input edges."""
    vl = meshes_vlvi["grid_vl"]
    vi = meshes_vlvi["grid_vi"]
    hi = meshes_vlvi["grid_hi"]
    edges = edges_from_vlvi(vl, vi, hi=hi)
    mesh = mesh_from_vlvi(vl, vi, hi=hi)
    assert are_equivalent_edges(edges, mesh.edges)


def test_edges_from_vr(meshes_vlvi: Dict[str, Any]) -> None:
    """Edges are equivalent to mesh_from_vlvi."""
    vl = meshes_vlvi["grid_vl"]
    vi = meshes_vlvi["grid_vi"]
    hi = meshes_vlvi["grid_hi"]
    vlvi_edges = edges_from_vlvi(vl, vi, hi=hi)

    verts = [Vert(coordinate=x) for x in meshes_vlvi["grid_vl"]]
    vr = [[verts[x] for x in face] for face in meshes_vlvi["grid_vi"]]
    hr = [[verts[x] for x in face] for face in meshes_vlvi["grid_hi"]]
    vr_edges = edges_from_vr(vr, hr)

    assert are_equivalent_edges(vlvi_edges, vr_edges)


def test_mesh_from_vr(meshes_vlvi: Dict[str, Any]) -> None:
    """Mesh is equivalent to input edges."""
    vl = [Vert(coordinate=x) for x in meshes_vlvi["grid_vl"]]
    vr = [[vl[x] for x in face] for face in meshes_vlvi["grid_vi"]]
    hr = [[vl[x] for x in face] for face in meshes_vlvi["grid_hi"]]
    edges = edges_from_vr(vr, hr)
    mesh = mesh_from_vr(vr, hr)
    assert are_equivalent_edges(edges, mesh.edges)
