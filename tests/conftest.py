#!/usr/bin/env python3
""" simple HalfEdges instances for testing

created: 181121 13:14:06
"""
from copy import deepcopy
from itertools import product
from typing import Any, Dict, Hashable, Iterable, List, Sequence, Set, Tuple, TypeVar

import pytest
from pytest_lazyfixture import lazy_fixture

from ..halfedge import half_edge_elements
from ..halfedge.half_edge_elements import Edge, Face
from ..halfedge.half_edge_object import HalfEdges


@pytest.fixture
def he_triangle() -> Dict[str, List[Any]]:
    """A simple triangle (inside and outside faces) for Mesh Element tests"""
    mesh = HalfEdges()
    verts = [half_edge_elements.Vert(coordinate=x) for x in ((-1, 0), (1, 0), (0, 1))]
    faces = [half_edge_elements.Face(), half_edge_elements.Face(__is_hole=True)]
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


@pytest.fixture(scope="function")
def he_cube(meshes_vlvi: Dict[str, Any]) -> HalfEdges:
    return HalfEdges.from_vlvi(
        meshes_vlvi["cube_vl"], meshes_vlvi["cube_vi"], attr_name="coordinate"
    )


@pytest.fixture(scope="function")
def he_grid(meshes_vlvi: Dict[str, Any]) -> HalfEdges:
    return HalfEdges.from_vlvi(
        meshes_vlvi["grid_vl"], meshes_vlvi["grid_vi"], attr_name="coordinate"
    )


@pytest.fixture(params=[lazy_fixture("he_grid"), lazy_fixture("he_cube")])
def he_mesh(request, he_cube, he_grid) -> HalfEdges:
    """A cube and a 3 x 3 grid as HalfEdges instances"""
    return request.param


@pytest.fixture(scope="function", params=range(9))
def grid_faces(request, he_grid) -> Tuple[HalfEdges, Face]:
    """A cube and a 3 x 3 grid as HalfEdges instances"""
    return he_grid, sorted(he_grid.faces)[request.param]


@pytest.fixture(scope="function", params=range(6))
def cube_faces(request, he_cube) -> Tuple[HalfEdges, Face]:
    """A cube and a 3 x 3 grid as HalfEdges instances"""
    return he_cube, sorted(he_cube.faces)[request.param]


@pytest.fixture(params=[lazy_fixture("grid_faces"), lazy_fixture("cube_faces")])
def mesh_faces(request) -> Tuple[HalfEdges, Face]:
    """A cube and a 3 x 3 grid as HalfEdges instances"""
    return request.param


@pytest.fixture(scope="function", params=range(48))
def grid_edges(request, he_grid) -> Tuple[HalfEdges, Edge]:
    """A cube and a 3 x 3 grid as HalfEdges instances"""
    return he_grid, sorted(he_grid.edges)[request.param]


@pytest.fixture(scope="function", params=range(24))
def cube_edges(request, he_cube) -> Tuple[HalfEdges, Edge]:
    """A cube and a 3 x 3 grid as HalfEdges instances"""
    return he_cube, sorted(he_cube.edges)[request.param]


@pytest.fixture(params=[lazy_fixture("grid_edges"), lazy_fixture("cube_edges")])
def mesh_edges(request) -> Tuple[HalfEdges, Edge]:
    """A cube and a 3 x 3 grid as HalfEdges instances"""
    return request.param


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
    return tuple(seq_a) == tuple(seq_b[idx:]) + tuple(seq_b[:idx])


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


_TVert = TypeVar("_TVert", bound=Hashable)


def get_canonical_vr(vr: Set[Tuple[Tuple[_TVert]]]) -> Set[Tuple[Tuple[_TVert]]]:
    """
    Rotate each tuple in a set to start with its min item.

    See docstring for canonical_mesh.
    """
    vr_aligned = set()
    for tuple_ in vr:
        min_item_idx = tuple_.index(min(tuple_))
        vr_aligned.add(tuple_[min_item_idx:] + tuple_[:min_item_idx])
    # noinspection PyTypeChecker
    return vr_aligned


def get_canonical_mesh(
    vl: Sequence[_TVert], vi: Iterable[Tuple[int]]
) -> Set[Tuple[Tuple[_TVert]]]:
    """
    A canonical mesh representation.

    Methods in this library represent meshes as

        * vertex list (vl) - ordered Vert items
        * vertex index (fi) - unordered set of tuples of indices to the vertex list

    The vertex list holds the vertices of the mesh, and the vertex index holds a
    tuple of vertex-list indices for each face. So, a triangle would be

        vl = [point_a, point_b, point_c]
        fi = {(0, 1, 2)}

    Such representations are slightly tricky to compare, because the above is
    equivalent to

        vl = [point_a, point_b, point_c]
        fi = {(1, 2, 0)}

    or

        vl = [point_a, point_c, point_b]
        fi = {(0, 2, 1)}

    The order of the set of input tuples doesn't matter. The order of index tuples
    *does* matter, but the starting point does not (e.i., these indices are
    "circular" -> 012 ~= 120 ~= 201).

    ------

    Methods in this library also produce vr (vertex raw) representations:

        vr = {(point_a, point_b_, point_c)}

    These remove some of the ambiguities of the vl / vi representation, but the
    starting point of each face is still ambiguous. That is:

        {(point_a, point_b, point_c)} is equivalent to
        {(point_b, point_c, point_a)} is equivalent to
        {(point_c, point_a, point_b)}

    This function produces an unambiguous representation so that such methods can be
    tested.
    """
    return get_canonical_vr({tuple(vl[x] for x in y) for y in vi})
