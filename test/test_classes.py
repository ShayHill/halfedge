# Last modified: 220626 22:40:26
# _*_ coding: utf-8 _*_
"""Test functions in classes.py.

created: 170204 14:22:23
"""
import itertools
import random
from keyword import iskeyword
from typing import Any, Callable, Dict, Tuple
from .conftest import get_canonical_mesh

import pytest

from .conftest import compare_circular_2

# noinspection PyProtectedMember,PyProtectedMember
from ..halfedge.element_attributes import (
    IncompatibleAttributeBase,
    NumericAttributeBase,
    ElemAttribBase
)
from ..halfedge.half_edge_elements import (
    Edge,
    Face,
    ManifoldMeshError,
    Vert,
    MeshElementBase,
    _function_lap,
)
from ..halfedge.half_edge_querries import StaticHalfEdges

alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
identifiers = (
    "".join(random.choice(alphabet) for _ in range(10)) for _ in itertools.count()
)


def valid_identifier():
    """Return a valid Python Identifier"""
    return next(
        filter(
            lambda x: x[0].isalpha() and x.isidentifier() and not (iskeyword(x)),
            identifiers,
        )
    )


class TestElemAttribs:
    def test_incompatible_merge_match(self) -> None:
        """Return a new attribute with same value if all values are equal"""
        attribs = [IncompatibleAttributeBase(7, None) for _ in range(3)]
        new_attrib = IncompatibleAttributeBase().merged(*attribs)
        assert new_attrib.value == 7

    def test_incompatible_merge_mismatch(self) -> None:
        """Return None if all values are not equal"""
        attribs = [IncompatibleAttributeBase(7, None) for _ in range(3)]
        attribs.append(IncompatibleAttributeBase(3))
        new_attrib = IncompatibleAttributeBase().merged(*attribs)
        assert new_attrib is None

    def test_numeric_all_nos(self) -> None:
        """Return a new attribute with same value if all values are equal"""
        attribs = [NumericAttributeBase(x) for x in range(1, 6)]
        new_attrib = NumericAttributeBase().merged(*attribs)
        assert new_attrib.value == 3

    def test_lazy(self) -> None:
        """Given no value, LazyAttrib will try to infer a value from self.element"""
        class LazyAttrib(ElemAttribBase):
            @classmethod
            def merged(cls, *merge_from):
                raise NotImplementedError()
            def _infer_value(self):
                return self.element.sn
        elem = MeshElementBase()
        elem.set_attrib(LazyAttrib())
        assert elem.get_attrib(LazyAttrib) == elem.sn


class TestMeshElementBase:
    def test_lt_gt(self) -> None:
        """Sorts by id."""
        elem1 = MeshElementBase()
        elem2 = MeshElementBase()
        assert (elem1 < elem2) == (id(elem1) < id(elem2))
        assert (elem2 > elem1) == (id(elem2) > id(elem1))

    def test_set_attrib(self) -> None:
        """Set an attrib by passing a MeshElementBase instance"""
        elem = MeshElementBase()
        elem_attrib = IncompatibleAttributeBase(8)
        elem.set_attrib(elem_attrib)
        assert getattr(elem, type(elem_attrib).__name__) is elem_attrib

    def test_attribs_through_init(self) -> None:
        """MeshElement attributes are captured when passed to init"""
        base_with_attrib = MeshElementBase(
            IncompatibleAttributeBase(7), NumericAttributeBase(8)
        )
        assert base_with_attrib.get_attrib(IncompatibleAttributeBase) == 7
        assert base_with_attrib.get_attrib(NumericAttributeBase) == 8

    def test_pointers_through_init(self) -> None:
        """Key, val pairs passed as kwargs fail if key does not have a setter"""
        with pytest.raises(AttributeError):
            MeshElementBase(edge=MeshElementBase())

    def test_fill_attrib(self) -> None:
        """Fill missing attrib values from fill_from"""
        elem1 = MeshElementBase(NumericAttributeBase(8), IncompatibleAttributeBase(3))
        elem2 = MeshElementBase(NumericAttributeBase(6), IncompatibleAttributeBase(3))
        elem3 = MeshElementBase(IncompatibleAttributeBase(1))
        elem3.fill_from(elem1, elem2)
        assert elem3.get_attrib(IncompatibleAttributeBase) == 1  # unchanged
        assert elem3.get_attrib(NumericAttributeBase) == 7  # filled


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


class Coordinate(IncompatibleAttributeBase[Tuple[int, int, int]]):
    pass


class TestInitVert:
    def setup_method(self):
        self.coordinate = Coordinate((1, 2, 3))
        self.edge = Edge()
        self.vert = Vert(self.coordinate, edge=self.edge)

    def test_fill_from_preserves_pointers(self):
        """fill_from() will not overwrite pointers"""
        edge = Edge()
        vert = Vert(edge=edge)
        filler = Vert(edge=Edge())
        vert.fill_from(filler)
        assert vert.edge is edge


    def test_coordinate_is_attribute(self):
        """Coordinate has been captured as an attribute"""
        assert self.vert.Coordinate is self.coordinate

    def test_coordinate_element_is_vert(self):
        """Coordinate.element is set during init/"""
        assert self.vert.Coordinate.element is self.vert

    def test_coordinate_value_has_not_changes(self):
        """Coordinate value is still (1, 2, 3)"""
        assert self.vert.get_attrib(Coordinate) == (1, 2, 3)

    def test_points_to_edge(self):
        """vert.edge points to input edge"""
        assert self.vert.edge is self.edge

    def test_mirrored_assignment(self):
        """vert.edge assignment mirrored in edge.orig"""
        assert self.vert.edge.orig is self.vert

class TestInitEdge:
    def setup_method(self):
        self.coordinate = Coordinate((1, 2, 3))
        self.edge = Edge()
        self.orig = Vert()
        self.pair = Edge()
        self.face = Face()
        self.next = Edge()
        self.edge = Edge(self.coordinate, orig=self.orig, pair=self.pair, face=self.face, next=self.next)

    def test_coordinate_is_attribute(self):
        """Coordinate has been captured as an attribute"""
        assert self.edge.Coordinate is self.coordinate

    def test_coordinate_element_is_vert(self):
        """Coordinate.element is set during init/"""
        assert self.edge.Coordinate.element is self.edge

    def test_coordinate_value_has_not_changes(self):
        """Coordinate value is still (1, 2, 3)"""
        assert self.edge.get_attrib(Coordinate) == (1, 2, 3)

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
        self.coordinate = Coordinate((1, 2, 3))
        self.edge = Edge()
        self.face = Face(self.coordinate, edge=self.edge)

    def test_coordinate_is_attribute(self):
        """Coordinate has been captured as an attribute"""
        assert self.face.Coordinate is self.coordinate

    def test_coordinate_element_is_vert(self):
        """Coordinate.element is set during init/"""
        assert self.face.Coordinate.element is self.face

    def test_coordinate_value_has_not_changes(self):
        """Coordinate value is still (1, 2, 3)"""
        assert self.face.get_attrib(Coordinate) == (1, 2, 3)

    def test_points_to_edge(self):
        """face.edge points to input edge"""
        assert self.face.edge is self.edge

    def test_mirrored_orig(self):
        """face.edge assignment mirrored in face.edge"""
        assert self.face.edge.face is self.face


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

    def test_vert_edge(self) -> None:
        """Find vert edge in mesh"""
        vert = Vert()
        edge = Edge(orig=vert)
        mesh = StaticHalfEdges({edge})
        assert vert.edge == edge

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
    def test_dest_is_pair_orig(he_triangle: Dict[str, Any]) -> None:
        """Returns pair orig if next.orig fails."""
        edge = random.choice(he_triangle["edges"])
        edge.next = None
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

    mesh = StaticHalfEdges(edges)

    assert mesh.verts == verts
    assert mesh.edges == edges
    assert mesh.faces == faces
    assert mesh.holes == holes


class TestHalfEdges:
    """Keep the linter happy."""

    def test_vl(self, meshes_vlvi: Dict[str, Any], he_cube, he_grid) -> None:
        """Converts unaltered mesh verts back to input vl."""
        assert {x.get_attrib(Coordinate) for x in he_cube.vl} == set(meshes_vlvi["cube_vl"])
        assert {x.get_attrib(Coordinate) for x in he_grid.vl} == set(meshes_vlvi["grid_vl"])

    def test_vi(self, meshes_vlvi: Dict[str, Any], he_cube, he_grid) -> None:
        """Convert unaltered mesh faces back to input vi.
        Demonstrates preservation of face edge beginning point."""
        compare_circular_2(he_cube.fi, meshes_vlvi["cube_vi"])
        compare_circular_2(he_grid.fi, meshes_vlvi["grid_vi"])

    def test_hi(self, meshes_vlvi: Dict[str, Any], he_grid) -> None:
        """Convert unaltered mesh holes back to input holes."""
        expect = get_canonical_mesh(meshes_vlvi["grid_vl"], meshes_vlvi["grid_hi"])
        result = get_canonical_mesh([x.get_attrib(Coordinate) for x in he_grid.vl], he_grid.hi)
        assert expect == result


def test_half_edges_boundary_edges(he_grid) -> None:
    """12 edges on grid. All face holes."""
    edges = he_grid.boundary_edges
    assert len(edges) == 12
    assert all(x.face.is_hole for x in edges)


def test_half_edges_boundary_verts(he_grid) -> None:
    """12 verts on grid. All valence 2 or 3."""
    verts = he_grid.boundary_verts
    assert len(verts) == 12
    assert all(x.valence in (2, 3) for x in verts)


def test_half_edges_interior_edges(he_grid) -> None:
    """36 in grid. All face Faces."""
    edges = he_grid.interior_edges
    assert len(edges) == 36
    assert not any(x.face.is_hole for x in edges)


def test_half_edges_interior_verts(he_grid) -> None:
    """4 in grid. All valence 4"""
    verts = he_grid.interior_verts
    assert len(verts) == 4
    assert all(x.valence == 4 for x in verts)
