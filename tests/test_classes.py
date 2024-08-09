"""Test functions in classes.py.

created: 170204 14:22:23
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import itertools
import random
from typing import Any, Tuple, TypeVar

import pytest

from halfedge.half_edge_elements import (
    Edge,
    Face,
    ManifoldMeshError,
    MeshElementBase,
    Vert,
    _function_lap,
)
from halfedge.half_edge_object import HalfEdges
from halfedge.half_edge_querries import StaticHalfEdges
from halfedge.type_attrib import Attrib, IncompatibleAttrib, NumericAttrib
from tests.conftest import compare_circular_2, get_canonical_mesh

_TElemAttrib = TypeVar("_TElemAttrib", bound="Attrib[Any]")

alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
identifiers = (
    "".join(random.choice(alphabet) for _ in range(10)) for _ in itertools.count()
)


class Flag(IncompatibleAttrib[int]):
    pass


class Score(NumericAttrib[float]):
    pass


class TestAttribBaseClass:
    def test_attribute_error_if_no_value_set(self) -> None:
        """Raise AttributeError if no value set."""
        attrib: Attrib[Any] = Attrib()
        with pytest.raises(AttributeError):
            _ = attrib.value

    def test_merge_returns_none(self) -> None:
        """Return None when attempting to merge Attrib instances."""
        attrib: Attrib[Any] = Attrib()
        new_attrib = attrib.merge(None)
        assert new_attrib is None

    def test_slice_returns_none(self) -> None:
        """Return None when attempting to slice Attrib instances."""
        attrib: Attrib[Any] = Attrib()
        new_attrib = attrib.slice()
        assert new_attrib is None


class TestContagionAttrib:
    def test_return_on_merge_if_no_values(self) -> None:
        """Return None if no values are set."""
        attrib: NumericAttrib[int] = NumericAttrib()
        new_attrib = attrib.merge(None, None, None)
        assert new_attrib is None


class TestIncompatibleAttrib:
    def test_return_self_on_slice(self) -> None:
        """Return self when slicing from."""
        attrib: IncompatibleAttrib[int] = IncompatibleAttrib()
        new_attrib = attrib.slice()
        assert new_attrib is attrib


class TestNumericAttrib:
    def test_return_none_on_empty_merge(self) -> None:
        """Return None if no values are set."""
        attrib: NumericAttrib[int] = NumericAttrib()
        new_attrib = attrib.merge(None, None, None)
        assert new_attrib is None


class TestElemAttribs:
    def test_incompatible_merge_match(self) -> None:
        """Return a new attribute with same value if all values are equal"""
        attribs = [IncompatibleAttrib(7, None) for _ in range(3)]
        new_attrib = IncompatibleAttrib().merge(*attribs)
        assert new_attrib is not None
        assert new_attrib.value == 7

    def test_incompatible_merge_mismatch(self) -> None:
        """Return None if all values are not equal"""
        attribs = [IncompatibleAttrib(7, None) for _ in range(3)]
        attribs.append(IncompatibleAttrib(3))
        new_attrib = IncompatibleAttrib().merge(*attribs)
        assert new_attrib is None

    def test_numeric_all_nos(self) -> None:
        """Return a new attribute with same value if all values are equal"""
        attribs = [NumericAttrib(x) for x in range(1, 6)]
        new_attrib = NumericAttrib().merge(*attribs)
        assert new_attrib is not None
        assert new_attrib.value == 3

    def test_lazy(self) -> None:
        """Given no value, LazyAttrib will try to infer a value from self.element"""

        class LazyAttrib(Attrib[int]):
            @classmethod
            def merge(cls, *merge_from: _TElemAttrib | None) -> _TElemAttrib | None:
                raise NotImplementedError()

            def _infer_value(self) -> int:
                if self.element is None:
                    msg = "no element from which to infer a value"
                    raise AttributeError(msg)
                return self.element.sn

        elem = MeshElementBase()
        elem.set_attrib(LazyAttrib())
        assert elem.get_attrib(LazyAttrib).value == elem.sn


class TestMeshElementBase:
    def test_lt_gt(self) -> None:
        """Sorts by sn."""
        elem1 = MeshElementBase()
        elem2 = MeshElementBase()
        assert (elem1 < elem2) == (elem1.sn < elem2.sn)
        assert (elem2 > elem1) == (elem2.sn > elem1.sn)

    def test_set_attrib(self) -> None:
        """Set an attrib by passing a MeshElementBase instance"""
        elem = MeshElementBase()
        elem_attrib = Flag(8)
        elem.set_attrib(elem_attrib)
        assert elem.get_attrib(Flag).value == 8

    def test_attribs_through_init(self) -> None:
        """MeshElement attributes are captured when passed to init"""
        base_with_attrib = MeshElementBase(Flag(7), Score(8))
        assert base_with_attrib.get_attrib(Flag).value == 7
        assert base_with_attrib.get_attrib(Score).value == 8

    def test_pointers_through_init(self) -> None:
        """Key, val pairs passed as kwargs fail if key does not have a setter"""
        with pytest.raises(TypeError):
            MeshElementBase(edge=MeshElementBase())  # type: ignore

    def test_fill_attrib(self) -> None:
        """Fill missing attrib values from fill_from"""
        elem1 = MeshElementBase(Score(8), Flag(3))
        elem2 = MeshElementBase(Score(6), Flag(3))
        elem3 = MeshElementBase(Flag(1))
        _ = elem3.merge_from(elem1, elem2)
        assert elem3.get_attrib(Flag).value == 1  # unchanged
        assert elem3.get_attrib(Score).value == 7  # filled


def test_edge_lap_succeeds(he_triangle: dict[str, Any]) -> None:
    """Returns to self when (func(func(func(....func(self))))) == self."""
    for edge in he_triangle["edges"]:
        assert _function_lap(lambda x: x.next, edge) == [
            edge,
            edge.next,
            edge.next.next,
        ]


def test_edge_lap_fails(he_triangle: dict[str, Any]) -> None:
    """Fails when self intersects."""
    edges = he_triangle["edges"]
    with pytest.raises(ManifoldMeshError) as err:
        _function_lap(lambda x: edges[1], edges[0])  # type: ignore
    assert "infinite" in err.value.args[0]


class Coordinate(IncompatibleAttrib[Tuple[int, int, int]]):
    pass


class TestInitVert:
    def setup_method(self):
        self.coordinate: Coordinate  # type: ignore
        self.edge: Edge  # type: ignore
        self.vert: Vert  # type: ignore
        self.coordinate = Coordinate((1, 2, 3))
        self.edge = Edge()
        self.vert = Vert(self.coordinate, edge=self.edge)

    def test_fill_from_preserves_pointers(self):
        """fill_from() will not overwrite pointers"""
        edge = Edge()
        vert = Vert(edge=edge)
        filler = Vert(edge=Edge())
        _ = vert.merge_from(filler)
        assert vert.edge is edge

    def test_coordinate_is_attribute(self):
        """Coordinate has been captured as an attribute"""
        result = self.vert.get_attrib(Coordinate).value
        expect = self.coordinate.value
        assert result == expect

    def test_coordinate_element_is_vert(self):
        """Coordinate.element is set during init/"""
        assert self.vert.get_attrib(Coordinate).element is self.vert

    def test_coordinate_value_has_not_changes(self):
        """Coordinate value is still (1, 2, 3)"""
        assert self.vert.get_attrib(Coordinate).value == (1, 2, 3)

    def test_points_to_edge(self):
        """vert.edge points to input edge"""
        assert self.vert.edge is self.edge

    def test_mirrored_assignment(self):
        """vert.edge assignment mirrored in edge.orig"""
        assert self.vert.edge.orig is self.vert


class TestInitEdge:

    def setup_method(self):
        self.coordinate: Coordinate  # type: ignore
        self.edge: Edge  # type: ignore
        self.orig: Vert  # type: ignore
        self.pair: Edge  # type: ignore
        self.face: Face  # type: ignore
        self.next: Edge  # type: ignore
        self.coordinate = Coordinate((1, 2, 3))
        self.edge = Edge()
        self.orig = Vert()
        self.pair = Edge()
        self.face = Face()
        self.next = Edge()
        self.edge = Edge(
            self.coordinate,
            orig=self.orig,
            pair=self.pair,
            face=self.face,
            next=self.next,
        )

    def test_coordinate_is_attribute(self):
        """Coordinate has been captured as an attribute"""
        assert self.edge.get_attrib(Coordinate).value == self.coordinate.value

    def test_coordinate_element_is_vert(self):
        """Coordinate.element is set during init/"""
        assert self.edge.get_attrib(Coordinate).element is self.edge

    def test_coordinate_value_has_not_changes(self):
        """Coordinate value is still (1, 2, 3)"""
        assert self.edge.get_attrib(Coordinate).value == (1, 2, 3)

    def test_points_to_orig(self):
        """vert.edge points to input edge"""
        assert self.edge.orig is self.orig

    def test_mirrored_orig(self):
        """edge.orig assignment mirrored in edge.orig"""
        assert self.edge.orig.edge is self.edge

    def test_points_to_pair(self):
        """vert.pair points to input edge"""
        assert self.edge.pair is self.pair

    def test_mirrored_pair(self):
        """edge.pair assignment mirrored in edge.pair"""
        assert self.edge.pair.pair is self.edge

    def test_points_to_face(self):
        """edge.face points to input face"""
        assert self.edge.face is self.face

    def test_mirrored_face(self):
        """edge.face assignment mirrored in edge.face"""
        assert self.edge.face.edge is self.edge

    def test_points_to_next(self):
        """edge.next points to input edge"""
        assert self.edge.next is self.next


class TestInitFace:

    def setup_method(self):
        self.coordinate: Coordinate  # type: ignore
        self.edge: Edge  # type: ignore
        self.face: Face  # type: ignore
        self.coordinate = Coordinate((1, 2, 3))
        self.edge = Edge()
        self.face = Face(self.coordinate, edge=self.edge)

    def test_coordinate_is_attribute(self):
        """Coordinate has been captured as an attribute"""
        assert self.face.get_attrib(Coordinate).value is self.coordinate.value

    def test_coordinate_element_is_vert(self):
        """Coordinate.element is set during init/"""
        assert self.face.get_attrib(Coordinate).element is self.face

    def test_coordinate_value_has_not_changes(self):
        """Coordinate value is still (1, 2, 3)"""
        assert self.face.get_attrib(Coordinate).value == (1, 2, 3)

    def test_points_to_edge(self):
        """face.edge points to input edge"""
        assert self.face.edge is self.edge

    def test_mirrored_orig(self):
        """face.edge assignment mirrored in face.edge"""
        assert self.face.edge.face is self.face


class TestElementSubclasses:
    """Test all three _MeshElementBase children."""

    def test_edge_face_edges(self, he_triangle: dict[str, Any]) -> None:
        """Edge next around face."""
        for edge in he_triangle["edges"]:
            assert tuple(edge.face_edges) == (edge, edge.next, edge.next.next)

    def test_face_edges(self, he_triangle: dict[str, Any]) -> None:
        """Finds all edges, starting at face.edge."""
        for face in he_triangle["faces"]:
            assert tuple(face.edges) == tuple(face.edge.face_edges)

    def test_edge_face_verts(self, he_triangle: dict[str, Any]) -> None:
        """Is equivalent to edge.pair.next around orig."""
        for edge in he_triangle["edges"]:
            assert tuple(edge.vert_edges) == (edge, edge.pair.next)

    def test_vert_edge(self) -> None:
        """Find vert edge in mesh"""
        vert = Vert()
        edge = Edge(orig=vert)
        _ = StaticHalfEdges({edge})
        assert vert.edge == edge

    def test_vert_edges(self, he_triangle: dict[str, Any]) -> None:
        """Is equivalent to vert_edges for vert.edge."""
        for vert in he_triangle["verts"]:
            assert tuple(vert.edges) == tuple(vert.edge.vert_edges)

    def test_vert_verts(self, he_triangle: dict[str, Any]) -> None:
        """Is equivalent to vert_edge.dest for vert.edge."""
        for vert in he_triangle["verts"]:
            assert vert.neighbors == [x.dest for x in vert.edge.vert_edges]

    def test_vert_valence(self, he_triangle: dict[str, Any]) -> None:
        """Valence is two for every corner in a triangle."""
        for vert in he_triangle["verts"]:
            assert vert.valence == 2

    def test_prev_by_face_edges(self, he_triangle: dict[str, Any]) -> None:
        """Previous edge will 'next' to self."""
        for edge in he_triangle["edges"]:
            assert edge.prev.next == edge

    @staticmethod
    def test_dest_is_next_orig(he_triangle: dict[str, Any]) -> None:
        """Finds orig of next or pair edge."""
        for edge in he_triangle["edges"]:
            assert edge.dest is edge.next.orig

    @staticmethod
    def test_dest_is_pair_orig(he_triangle: dict[str, Any]) -> None:
        """Returns pair orig if next.orig fails."""
        edge = random.choice(he_triangle["edges"])
        edge.next = None
        assert edge.dest is edge.pair.orig

    @staticmethod
    def test_face_verts(he_triangle: dict[str, Any]) -> None:
        """Returns orig for every edge in face_verts."""
        for face in he_triangle["faces"]:
            assert tuple(face.verts) == tuple(face.edge.face_verts)


def test_half_edges_init(he_triangle: dict[str, Any]) -> None:
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

    def test_vl(
        self, meshes_vlvi: dict[str, Any], he_cube: HalfEdges, he_grid: HalfEdges
    ) -> None:
        """Converts unaltered mesh verts back to input vl."""
        assert {x.get_attrib(Coordinate).value for x in he_cube.vl} == set(
            meshes_vlvi["cube_vl"]
        )
        assert {x.get_attrib(Coordinate).value for x in he_grid.vl} == set(
            meshes_vlvi["grid_vl"]
        )

    def test_vi(
        self, meshes_vlvi: dict[str, Any], he_cube: HalfEdges, he_grid: HalfEdges
    ) -> None:
        """Convert unaltered mesh faces back to input vi.
        Demonstrates preservation of face edge beginning point."""
        _ = compare_circular_2(he_cube.fi, meshes_vlvi["cube_vi"])
        _ = compare_circular_2(he_grid.fi, meshes_vlvi["grid_vi"])

    def test_hi(self, meshes_vlvi: dict[str, Any], he_grid: HalfEdges) -> None:
        """Convert unaltered mesh holes back to input holes."""
        expect = get_canonical_mesh(meshes_vlvi["grid_vl"], meshes_vlvi["grid_hi"])
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
