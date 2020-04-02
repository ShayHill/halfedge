#!/usr/bin/env python3
""" simple HalfEdges instances for testing

created: 181121 13:14:06
"""

from itertools import product
from typing import Any, Dict, List

import pytest
from ..halfedge import classes
from ..halfedge.constructors import edges_from_vlvi
from ..halfedge.classes import HalfEdges


@pytest.fixture
def he_triangle() -> Dict[str, List[Any]]:
    """A simple triangle (inside and outside faces) for Mesh Element tests"""
    mesh = HalfEdges()
    verts = [classes.Vert(mesh, coordinate=x) for x in ((-1, 0), (1, 0), (0, 1))]
    faces = [classes.Face(mesh), classes.Hole()]
    inner_edges = [classes.Edge(orig=verts[x], face=faces[0]) for x in range(3)]
    outer_edges = [classes.Edge(orig=verts[1 - x], face=faces[1]) for x in range(3)]
    mesh.edges.update(inner_edges, outer_edges)

    for i in range(3):
        inner_edges[i].pair = outer_edges[-i]
        outer_edges[-i].pair = inner_edges[i]
        inner_edges[i - 1].next = inner_edges[i]
        outer_edges[i - 1].next = outer_edges[i]
        # verts[i].edge = inner_edges[i]
    # faces[0].edge = inner_edges[0]
    # faces[1].edge = outer_edges[0]
    # TODO: remove commented lines above once all tests pass

    return {
        "verts": verts,
        "edges": inner_edges + outer_edges,
        "faces": faces[:1],
        "holes": faces[1:],
    }


@pytest.fixture(scope="module")
def meshes_vlvi() -> Dict[str, Any]:
    """A cube and a 3 x 3 grid"""
    # fmt: off
    cube_vl = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
               (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]

    cube_vi = [[0, 1, 2, 3], [0, 3, 7, 4], [0, 4, 5, 1],
               [1, 5, 6, 2], [2, 6, 7, 3], [4, 7, 6, 5]]

    grid_vl = [(x, y) for x, y in product(range(4), range(4))]

    grid_vi = [
        [x + y, x + y + 1, x + y + 5, x + y + 4]
        for y, x in product((0, 4, 8), (0, 1, 2))
    ]

    grid_hi = [[0, 4, 8, 12, 13, 14, 15, 11, 7, 3, 2, 1]]

    return {
        "cube_vl": cube_vl, "cube_vi": cube_vi,
        "grid_vl": grid_vl, "grid_vi": grid_vi, "grid_hi": grid_hi,
    }
    # fmt: on


# noinspection Pylint
@pytest.fixture
def he_meshes(meshes_vlvi: Dict[str, Any]) -> Dict[str, Any]:
    """A cube and a 3 x 3 grid as HalfEdges instances"""
    cube = classes.HalfEdges(
        edges_from_vlvi(meshes_vlvi["cube_vl"], meshes_vlvi["cube_vi"])
    )
    for elem in cube.verts | cube.faces | cube.holes:
        elem.mesh = cube

    grid = classes.HalfEdges(
        edges_from_vlvi(
            meshes_vlvi["grid_vl"], meshes_vlvi["grid_vi"], meshes_vlvi["grid_hi"]
        )
    )
    for elem in grid.verts | grid.faces | grid.holes:
        elem.mesh = grid

    return {"cube": cube, "grid": grid}
