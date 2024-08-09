"""Test functions in classes.py.

created: 170204 14:22:23
"""

# pyright: reportPrivateUsage=false

import itertools
import random
from keyword import iskeyword
from typing import Any, Dict, Tuple

import pytest

from halfedge.half_edge_elements import (
    ManifoldMeshError,
    MeshElementBase,
    Vert,
    _function_lap,
)
from halfedge.half_edge_object import HalfEdges
from halfedge.half_edge_querries import StaticHalfEdges
from halfedge.type_attrib import IncompatibleAttrib, NumericAttrib
from tests.conftest import get_canonical_mesh

alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
identifiers = (
    "".join(random.choice(alphabet) for _ in range(10)) for _ in itertools.count()
)


class Coordinate(IncompatibleAttrib[Tuple[float, ...]]):
    pass


def valid_identifier():
    """Return a strategy which generates a valid Python Identifier"""
    return next(
        filter(
            lambda x: x[0].isalpha() and x.isidentifier() and not (iskeyword(x)),
            identifiers,
        )
    )


class TestFromVlfi:
    def test_raise_on_non_inferable_holes(self) -> None:
        """Raises if holes cannot be inferred."""
        vl = [Vert() for _ in range(7)]
        fi: list[Tuple[int, ...]] = [(0, 2, 3, 1), (3, 5, 6, 4)]
        with pytest.raises(ManifoldMeshError) as err:
            _ = HalfEdges.from_vlfi(vl, fi)
        assert "Ambiguous 'next'" in err.value.args[0]


class TestMeshElementBase:

    def test_fill_attrs_from_fills_missing(self) -> None:
        """Fills attrs if not present."""

        class Flag1(NumericAttrib[int]):
            pass

        class Flag2(NumericAttrib[int]):
            pass

        flag1_defined = MeshElementBase(Flag1(2))
        flag2_defined_a = MeshElementBase(Flag2(3))
        flag2_defined_b = MeshElementBase(Flag2(5))
        _ = flag1_defined.merge_from(flag2_defined_a, flag2_defined_b)
        assert flag1_defined.get_attrib(Flag1).value == 2
        assert flag1_defined.get_attrib(Flag2).value == 4


def test_edge_lap_succeeds(he_triangle: Dict[str, Any]) -> None:
    """Returns to self when (func(func(func(....func(self))))) == self."""
    for edge in he_triangle["edges"]:
        assert _function_lap(lambda x: x.next, edge) == [
            edge,
            edge.next,
            edge.next.next,
        ]


def test_edge_lap_fails(he_triangle: Dict[str, Any]) -> None:
    """Fails when self intersects."""
    edges = he_triangle["edges"]
    with pytest.raises(ManifoldMeshError) as err:
        _ = _function_lap(lambda x: edges[1], edges[0])
    assert "infinite" in err.value.args[0]


class TestElementSubclasses:
    """Test all three _MeshElementBase children."""

    def test_edge_face_edges(self, he_triangle: Dict[str, Any]) -> None:
        """Edge next around face."""
        for edge in he_triangle["edges"]:
            assert tuple(edge.face_edges) == (edge, edge.next, edge.next.next)

    def test_face_edges(self, he_triangle: Dict[str, Any]) -> None:
        """Finds all edges, starting at face.edge."""
        for face in he_triangle["faces"]:
            assert tuple(face.edges) == tuple(face.edge.face_edges)

    def test_edge_face_verts(self, he_triangle: Dict[str, Any]) -> None:
        """Is equivalent to edge.pair.next around orig."""
        for edge in he_triangle["edges"]:
            assert tuple(edge.vert_edges) == (edge, edge.pair.next)

    def test_vert_edges(self, he_triangle: Dict[str, Any]) -> None:
        """Is equivalent to vert_edges for vert.edge."""
        for vert in he_triangle["verts"]:
            assert tuple(vert.edges) == tuple(vert.edge.vert_edges)

    def test_vert_verts(self, he_triangle: Dict[str, Any]) -> None:
        """Is equivalent to vert_edge.dest for vert.edge."""
        for vert in he_triangle["verts"]:
            assert vert.neighbors == [x.dest for x in vert.edge.vert_edges]

    def test_vert_valence(self, he_triangle: Dict[str, Any]) -> None:
        """Valence is two for every corner in a triangle."""
        for vert in he_triangle["verts"]:
            assert vert.valence == 2

    def test_prev_by_face_edges(self, he_triangle: Dict[str, Any]) -> None:
        """Previous edge will 'next' to self."""
        for edge in he_triangle["edges"]:
            assert edge.prev.next == edge

    @staticmethod
    def test_dest_is_next_orig(he_triangle: Dict[str, Any]) -> None:
        """Finds orig of next or pair edge."""
        for edge in he_triangle["edges"]:
            assert edge.dest is edge.next.orig

    @staticmethod
    def test_face_verts(he_triangle: Dict[str, Any]) -> None:
        """Returns orig for every edge in face_verts."""
        for face in he_triangle["faces"]:
            assert tuple(face.verts) == tuple(face.edge.face_verts)


def test_half_edges_init(he_triangle: Dict[str, Any]) -> None:
    """Verts, edges, faces, and holes match hand-calculated coordinates."""
    verts = set(he_triangle["verts"])
    edges = set(he_triangle["edges"])
    faces = set(he_triangle["faces"])
    holes = set(he_triangle["holes"])

    mesh = StaticHalfEdges(edges)

    assert mesh.verts == verts
    assert mesh.edges == edges
    assert mesh.faces == faces
    assert mesh.holes == holes


class TestHalfEdges:
    """Keep the linter happy."""

    def test_vi(
        self, meshes_vlvi: Dict[str, Any], he_grid: HalfEdges, he_cube: HalfEdges
    ) -> None:
        """Convert unaltered mesh faces back to input vi."""
        for mesh, key in ((he_grid, "grid"), (he_cube, "cube")):
            input_vl, input_vi = meshes_vlvi[key + "_vl"], meshes_vlvi[key + "_vi"]
            expect = get_canonical_mesh(input_vl, input_vi)
            result = get_canonical_mesh(
                [x.get_attrib(Coordinate).value for x in mesh.vl], mesh.fi
            )
            assert expect == result

    def test_hi(self, meshes_vlvi: Dict[str, Any], he_grid: HalfEdges) -> None:
        """Convert unaltered mesh holes back to input holes."""
        input_vl, input_hi = meshes_vlvi["grid_vl"], meshes_vlvi["grid_hi"]
        expect = get_canonical_mesh(input_vl, input_hi)
        result = get_canonical_mesh(
            [x.get_attrib(Coordinate).value for x in he_grid.vl], he_grid.hi
        )
        assert expect == result


def test_half_edges_boundary_edges(he_grid: HalfEdges) -> None:
    """12 edges on grid. All face holes."""
    edges = he_grid.boundary_edges
    assert len(edges) == 12
    assert all(x.face.is_hole for x in edges)


def test_half_edges_boundary_verts(he_grid: HalfEdges) -> None:
    """12 verts on grid. All valence 2 or 3."""
    verts = he_grid.boundary_verts
    assert len(verts) == 12
    assert all(x.valence in (2, 3) for x in verts)


def test_half_edges_interior_edges(he_grid: HalfEdges) -> None:
    """36 in grid. All face Faces."""
    edges = he_grid.interior_edges
    assert len(edges) == 36
    assert not any(x.face.is_hole for x in edges)


def test_half_edges_interior_verts(he_grid: HalfEdges) -> None:
    """4 in grid. All valence 4"""
    verts = he_grid.interior_verts
    assert len(verts) == 4
    assert all(x.valence == 4 for x in verts)
