# Last modified: 181126 16:46:28
# _*_ coding: utf-8 _*_
"""Test functions in classes.py.

created: 170204 14:22:23
"""
import itertools
import random
from keyword import iskeyword
from operator import attrgetter
from typing import Any, Callable, Dict

import pytest

# noinspection PyProtectedMember,PyProtectedMember
from .conftest import compare_circular, compare_circular_2, get_canonical_mesh
from ..halfedge.half_edge_elements import (
    Edge,
    Face,
    Hole,
    ManifoldMeshError,
    Vert,
    _MeshElementBase,
    _function_lap,
)
from ..halfedge.half_edge_querries import StaticHalfEdges


alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
identifiers = (
    "".join(random.choice(alphabet) for _ in range(10)) for _ in itertools.count()
)


def valid_identifier():
    """Return a strategy which generates a valid Python Identifier"""
    return next(
        filter(
            lambda x: x[0].isalpha() and x.isidentifier() and not (iskeyword(x)),
            identifiers,
        )
    )


class TestMeshElementBase:
    @pytest.mark.parametrize("name,value", [(valid_identifier(), random.randint(1, 5))])
    def test_kwargs(self, name, value) -> None:
        """Sets kwargs."""
        a = _MeshElementBase(**{name: value})
        assert getattr(a, name) == value

    def test_fill_from_preserves_attrs(self) -> None:
        """Does not overwrite attrs."""
        a_is_1 = _MeshElementBase(a=1)
        a_is_2 = _MeshElementBase(a_is_1, a=2)
        assert getattr(a_is_2, "a") == 2

    def test_fill_attrs_from_fills_missing(self) -> None:
        """Fills attrs if not present."""
        b_is_3 = _MeshElementBase(a=1, b=3)
        a_is_2 = _MeshElementBase(b_is_3, a=2)
        assert getattr(a_is_2, "a") == 2
        assert getattr(a_is_2, "b") == 3


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
        _function_lap(lambda x: edges[1], edges[0])  # type: ignore
    assert "infinite" in err.value.args[0]


class TestElementSubclasses:
    """Test all three _MeshElementBase children."""

    @staticmethod
    def check_init(class_: Callable, potential_kwargs: Dict[str, Any]) -> None:
        """Check values against args dict.

        Pass partial arg sets, each missing one arg. Then pass all args.

        """

        def check_kwarg_subset(kwargs: Dict[str, Any]) -> Any:
            """Run one combination of kwargs."""
            inst = class_(**kwargs)

            for arg in kwargs.keys():
                assert getattr(inst, arg) == kwargs[arg]

            return inst

        for skip in potential_kwargs.keys():
            inst_wo_skip = check_kwarg_subset(
                {k: v for k, v in potential_kwargs.items() if k != skip}
            )

            with pytest.raises(AttributeError):
                getattr(inst_wo_skip, skip)

        # check full init
        check_kwarg_subset(potential_kwargs)

    def test_init_vert(self) -> None:
        """Will not set missing attrs. sets others."""
        self.check_init(Vert, {"coordinate": (0, 0, 0), "some_kwarg": 20})

    def test_init_edge(self) -> None:
        """Will not set missing attrs. sets others."""
        self.check_init(
            Edge,
            {
                "orig": Vert(),
                "pair": Edge(),
                "face": Face(),
                "next": Edge(),
                "some_kwarg": 20,
            },
        )

    def test_init_face(self) -> None:
        """Will not set missing attrs. sets others."""
        self.check_init(Face, {"some_kwarg": 20})

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

    def test_vi(self, meshes_vlvi: Dict[str, Any], he_grid, he_cube) -> None:
        """Convert unaltered mesh faces back to input vi."""
        for mesh, key in ((he_grid, "grid"), (he_cube, "cube")):
            input_vl, input_vi = meshes_vlvi[key + "_vl"], meshes_vlvi[key + "_vi"]
            expect = get_canonical_mesh(input_vl, input_vi)
            result = get_canonical_mesh([x.coordinate for x in mesh.vl], mesh.fi)
            assert expect == result

    def test_hi(self, meshes_vlvi: Dict[str, Any], he_grid) -> None:
        """Convert unaltered mesh holes back to input holes."""
        input_vl, input_hi = meshes_vlvi["grid_vl"], meshes_vlvi["grid_hi"]
        expect = get_canonical_mesh(input_vl, input_hi)
        result = get_canonical_mesh([x.coordinate for x in he_grid.vl], he_grid.hi)
        assert expect == result


def test_half_edges_boundary_edges(he_grid) -> None:
    """12 edges on grid. All face holes."""
    edges = he_grid.boundary_edges
    assert len(edges) == 12
    assert all(isinstance(x.face, Hole) for x in edges)


def test_half_edges_boundary_verts(he_grid) -> None:
    """12 verts on grid. All valence 2 or 3."""
    verts = he_grid.boundary_verts
    assert len(verts) == 12
    assert all(x.valence in (2, 3) for x in verts)


def test_half_edges_interior_edges(he_grid) -> None:
    """36 in grid. All face Faces."""
    edges = he_grid.interior_edges
    assert len(edges) == 36
    assert not any(isinstance(x.face, Hole) for x in edges)


def test_half_edges_interior_verts(he_grid) -> None:
    """4 in grid. All valence 4"""
    verts = he_grid.interior_verts
    assert len(verts) == 4
    assert all(x.valence == 4 for x in verts)
