#!/usr/bin/env python3
""" simple HalfEdges instances for testing

created: 181121 13:14:06
"""

from copy import deepcopy
from itertools import product
from typing import Any, Dict, List, Sequence, Set, Tuple, cast

import pytest

from ..halfedge import half_edge_elements

# TODO: fix imports
# from ..halfedge.constructors import edges_from_vlvi

from ..halfedge.half_edge_querries import StaticHalfEdges
import os
import sys

sys.path.append(os.path.join(__file__, "../.."))


@pytest.fixture
def he_triangle() -> Dict[str, List[Any]]:
    """A simple triangle (inside and outside faces) for Mesh Element tests"""
    mesh = StaticHalfEdges()
    verts = [half_edge_elements.Vert(coordinate=x) for x in ((-1, 0), (1, 0), (0, 1))]
    faces = [half_edge_elements.Face(), half_edge_elements.Hole()]
    inner_edges = [
        half_edge_elements.Edge(orig=verts[x], face=faces[0]) for x in range(3)
    ]
    outer_edges = [
        half_edge_elements.Edge(orig=verts[1 - x], face=faces[1]) for x in range(3)
    ]
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

    cube_vi = {(0, 1, 2, 3), (0, 3, 7, 4), (0, 4, 5, 1),
               (1, 5, 6, 2), (2, 6, 7, 3), (4, 7, 6, 5)}

    grid_vl = [(x, y) for x, y in product(range(4), range(4))]

    grid_vi = {
        (x + y, x + y + 1, x + y + 5, x + y + 4)
        for y, x in product((0, 4, 8), (0, 1, 2))
    }

    grid_hi = {(0, 4, 8, 12, 13, 14, 15, 11, 7, 3, 2, 1)}

    return {
        "cube_vl": cube_vl, "cube_vi": cube_vi,
        "grid_vl": grid_vl, "grid_vi": grid_vi, "grid_hi": grid_hi,
    }
    # fmt: on


# noinspection Pylint
@pytest.fixture
def he_meshes(meshes_vlvi: Dict[str, Any]) -> Dict[str, Any]:
    """A cube and a 3 x 3 grid as HalfEdges instances"""
    cube = StaticHalfEdges.mesh_from_vlvi(
        meshes_vlvi["cube_vl"], meshes_vlvi["cube_vi"]
    )
    for elem in cube.verts | cube.faces | cube.holes:
        elem.mesh = cube

    grid = StaticHalfEdges.mesh_from_vlvi(
        meshes_vlvi["grid_vl"], meshes_vlvi["grid_vi"]  # , meshes_vlvi["grid_hi"]
    )
    for elem in grid.verts | grid.faces | grid.holes:
        elem.mesh = grid

    return {"cube": cube, "grid": grid}


def compare_circular(seq_a: Sequence[Any], seq_b: Sequence[Any]) -> bool:
    """Start sequence at lowest value

    To help compare circular sequences
    """
    if not seq_a:
        return ~bool(seq_b)
    beg = seq_a[0]
    if beg not in seq_b:
        return False
    idx = seq_b.index(beg)
    return seq_a == seq_b[idx:] + seq_b[:idx]


def compare_circular_2(seq_a: List[List[Any]], seq_b: List[List[Any]]) -> bool:
    """ "
    Compare_circular with a nested list
    """
    seq_a = deepcopy(seq_a)
    seq_b = deepcopy(seq_b)
    while seq_a and seq_b:
        sub_a = seq_a.pop()
        try:
            seq_b.remove(next(x for x in seq_b if compare_circular(sub_a, x)))
        except StopIteration:
            return False
    if seq_b:
        return False
    return True


def _canon_face_rep(face: half_edge_elements.Face) -> List[Any]:
    """Canonical face representation: value tuples starting at min."""
    coordinates = [x.coordinate for x in face.verts]
    idx_min = coordinates.index(min(coordinates))
    return coordinates[idx_min:] + coordinates[:idx_min]


def _canon_he_rep(edges: Set[half_edge_elements.Edge]) -> Tuple[List[Any], List[Any]]:
    """Canonical mesh representation [faces, holes].

    faces or holes = [canon_face_rep(face), ...]
    """
    faces = set(
        x.face for x in edges if not isinstance(x.face, half_edge_elements.Hole)
    )
    holes = set(x.face for x in edges if isinstance(x.face, half_edge_elements.Hole))
    face_reps = cast(List[Any], [_canon_face_rep(x) for x in faces])
    hole_reps = cast(List[Any], [_canon_face_rep(x) for x in holes])

    return sorted(face_reps), sorted(hole_reps)


def are_equivalent_edges(
    edges_a: Set[half_edge_elements.Edge], edges_b: Set[half_edge_elements.Edge]
) -> bool:
    """Do edges lay on the same vert values with same geometry?"""
    faces_a, holes_a = _canon_he_rep(edges_a)
    faces_b, holes_b = _canon_he_rep(edges_b)

    are_same = len(faces_a) == len(faces_b) and len(holes_a) == len(holes_b)

    for a, b in zip(faces_a + holes_a, faces_b + holes_b):
        are_same = are_same and a == b

    return are_same


def are_equivalent_meshes(mesh_a, mesh_b) -> bool:
    """Do meshes lay on the same vert values with same geometry?"""
    return are_equivalent_edges(mesh_a.edges, mesh_b.edges)
