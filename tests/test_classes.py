# Last modified: 181126 16:46:28
# _*_ coding: utf-8 _*_
"""Test functions in classes.py.

created: 170204 14:22:23
"""
from typing import Any, Callable, Dict

import pytest

# noinspection PyProtectedMember,PyProtectedMember
from halfedge.classes import (
    ManifoldMeshError,
    _MeshElementBase,
    _function_lap,
    HalfEdges,
    Vert,
    Edge,
    Face,
    Hole,
)

from keyword import iskeyword
from hypothesis import given, example, settings
import hypothesis.strategies as strategies


def valid_identifier(**kwargs):
    """Return a strategy which generates a valid Python Identifier"""
    if "min_size" not in kwargs:
        kwargs["min_size"] = 4
    return strategies.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        **kwargs
    ).filter(lambda x: x[0].isalpha() and x.isidentifier() and not (iskeyword(x)))


class TestMeshElementBase:
    """Keep the linter happy."""

    @staticmethod
    @given(strategies.integers(min_value=1, max_value=5))
    @example(0)
    @settings(max_examples=5)
    def test_sequential_serial_numbers(int_) -> None:
        """Assigns sequential serial numbers."""
        sns = [_MeshElementBase().sn for _ in range(int_)]
        assert sns == sorted(sns)

    @staticmethod
    @given(strategies.integers(min_value=1, max_value=5))
    @example(0)
    @settings(max_examples=5)
    def test_last_issued_sn(int_) -> None:
        """The last_issued_sn == last sn issued to ANY instance."""
        instances = [_MeshElementBase() for _ in range(int_)]
        assert all((x.last_issued_sn == instances[-1].sn for x in instances))

    # noinspection Pylint
    @staticmethod
    def test_children_share_serial_numbers() -> None:
        """ child classes share set of serial numbers

        A serial number will never duplicate and will always be sequential
        between _MeshElementBase and any children

        """

        class MeshElemA(_MeshElementBase):
            """Subclass _MeshElementBase to test for sn sharing."""

        class MeshElemB(_MeshElementBase):
            """Subclass _MeshElementBase to test for sn sharing."""

        class MeshElemC(_MeshElementBase):
            """Subclass _MeshElementBase to test for sn sharing."""

        instances = [x() for x in (_MeshElementBase, MeshElemA, MeshElemB, MeshElemC)]
        assert all(x.last_issued_sn == instances[-1].sn for x in instances)
        assert all(instances[x] < instances[x + 1] for x in range(3))

    @staticmethod
    def test_lt_gt() -> None:
        """Sorts by serial number."""
        elem1 = _MeshElementBase()
        elem2 = _MeshElementBase()
        assert elem1 < elem2
        assert elem2 > elem1

    @staticmethod
    @given(valid_identifier(), strategies.integers())
    @settings(max_examples=5)
    def test_kwargs(identifier, value) -> None:
        """Sets kwargs."""
        a = _MeshElementBase(**{identifier: value})
        assert getattr(a, identifier) == value

    @staticmethod
    def test_fill_from_preserves_attrs() -> None:
        """Does not overwrite attrs."""
        a_is_1 = _MeshElementBase(a=1)
        a_is_2 = _MeshElementBase(a=2, fill_from=a_is_1)
        assert getattr(a_is_2, "a") == 2

    @staticmethod
    def test_fill_attrs_from_fills_missing() -> None:
        """Fills attrs if not present."""
        b_is_3 = _MeshElementBase(a=1, b=3)
        a_is_2 = _MeshElementBase(a=2, fill_from=b_is_3)
        assert getattr(a_is_2, "a") == 2
        assert getattr(a_is_2, "b") == 3

    @staticmethod
    def test_assign_new_sn() -> None:
        """New serial number will be higher than previous."""
        instances = [_MeshElementBase() for _ in range(100)]
        instances[0].assign_new_sn()
        sns = [x.sn for x in instances]
        assert sorted(sns) == sns[1:] + sns[:1]


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
        self.check_init(
            Vert, {"coordinate": (0, 0, 0), "edge": Edge(), "some_kwarg": 20}
        )

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
        self.check_init(Face, {"edge": Edge(), "some_kwarg": 20})

    @staticmethod
    def test_edge_face_edges(he_triangle: Dict[str, Any]) -> None:
        """Edge next around face."""
        for edge in he_triangle["edges"]:
            assert tuple(edge.face_edges) == (edge, edge.next, edge.next.next)

    @staticmethod
    def test_face_edges(he_triangle: Dict[str, Any]) -> None:
        """Finds all edges, starting at face.edge."""
        for face in he_triangle["faces"]:
            assert tuple(face.edges) == tuple(face.edge.face_edges)

    @staticmethod
    def test_edge_face_verts(he_triangle: Dict[str, Any]) -> None:
        """Is equivalent to edge.pair.next around orig."""
        for edge in he_triangle["edges"]:
            assert tuple(edge.vert_edges) == (edge, edge.pair.next)

    @staticmethod
    def test_vert_edges(he_triangle: Dict[str, Any]) -> None:
        """Is equivalent to vert_edges for vert.edge."""
        for vert in he_triangle["verts"]:
            assert tuple(vert.edges) == tuple(vert.edge.vert_edges)

    @staticmethod
    def test_vert_verts(he_triangle: Dict[str, Any]) -> None:
        """Is equivalent to vert_edge.dest for vert.edge."""
        for vert in he_triangle["verts"]:
            assert vert.verts == [x.dest for x in vert.edge.vert_edges]

    @staticmethod
    def test_vert_valence(he_triangle: Dict[str, Any]) -> None:
        """Valence is two for every corner in a triangle."""
        for vert in he_triangle["verts"]:
            assert vert.valence == 2

    @staticmethod
    def test_prev_by_face_edges(he_triangle: Dict[str, Any]) -> None:
        """Previous edge will 'next' to self."""
        for edge in he_triangle["edges"]:
            assert edge.prev.next == edge

    @staticmethod
    def test_prev_by_vert_edges(he_triangle: Dict[str, Any]) -> None:
        """Previous edge will 'next' to self."""
        edge = sorted(he_triangle["edges"])[0]
        edge.next = edge  # create an infinite loop
        assert edge.prev.next == edge
        del edge.next  # cause an attribute error
        assert edge.prev.next == edge

    @staticmethod
    def test_dest_is_next_orig(he_triangle: Dict[str, Any]) -> None:
        """Finds orig of next or pair edge."""
        for edge in he_triangle["edges"]:
            assert edge.dest is edge.next.orig

    @staticmethod
    def test_dest_is_pair_orig(he_triangle: Dict[str, Any]) -> None:
        """Returns pair orig if next.orig fails."""
        for edge in he_triangle["edges"]:
            del edge.next
            assert edge.dest is edge.pair.orig

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

    mesh = HalfEdges(edges)

    assert mesh.verts == verts
    assert mesh.edges == edges
    assert mesh.faces == faces
    assert mesh.holes == holes


class TestHalfEdges:
    """Keep the linter happy."""

    @staticmethod
    def test_last_issued_sn(he_meshes: Dict[str, Any]) -> None:
        """Matches highest serial number of any _MeshElementBase instance."""
        for mesh in he_meshes.values():
            assert mesh.last_issued_sn == _MeshElementBase.last_issued_sn

    @staticmethod
    def test_vl(meshes_vlvi: Dict[str, Any], he_meshes: Dict[str, Any]) -> None:
        """Converts unaltered mesh verts back to input vl."""
        assert he_meshes["cube"].vl == meshes_vlvi["cube_vl"]
        assert he_meshes["grid"].vl == meshes_vlvi["grid_vl"]

    @staticmethod
    def test_vi(meshes_vlvi: Dict[str, Any], he_meshes: Dict[str, Any]) -> None:
        """Convert unaltered mesh faces back to input vi."""
        assert he_meshes["cube"].vi == meshes_vlvi["cube_vi"]
        assert he_meshes["grid"].vi == meshes_vlvi["grid_vi"]

    @staticmethod
    def test_hi(meshes_vlvi: Dict[str, Any], he_meshes: Dict[str, Any]) -> None:
        """Convert unaltered mesh holes back to input holes."""
        assert he_meshes["grid"].hi == meshes_vlvi["grid_hi"]


def test_half_edges_boundary_edges(he_meshes: Dict[str, Any]) -> None:
    """12 edges on grid. All face holes."""
    edges = he_meshes["grid"].boundary_edges
    assert len(edges) == 12
    assert all(isinstance(x.face, Hole) for x in edges)


def test_half_edges_boundary_verts(he_meshes: Dict[str, Any]) -> None:
    """12 verts on grid. All valence 2 or 3."""
    verts = he_meshes["grid"].boundary_verts
    assert len(verts) == 12
    assert all(x.valence in (2, 3) for x in verts)


def test_half_edges_interior_edges(he_meshes: Dict[str, Any]) -> None:
    """36 in grid. All face Faces."""
    edges = he_meshes["grid"].interior_edges
    assert len(edges) == 36
    assert not any(isinstance(x.face, Hole) for x in edges)


def test_half_edges_interior_verts(he_meshes: Dict[str, Any]) -> None:
    """4 in grid. All valence 4"""
    verts = he_meshes["grid"].interior_verts
    assert len(verts) == 4
    assert all(x.valence == 4 for x in verts)


def test_half_edges_bounding_box(he_meshes: Dict[str, Any]) -> None:
    """Match hand-calculated values"""
    assert he_meshes["cube"].bounding_box == ((-1, -1, -1), (1, 1, 1))
    assert he_meshes["grid"].bounding_box == ((0, 0), (3, 3))
