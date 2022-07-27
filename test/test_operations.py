#!/usr/bin/env python3
# Last modified: 211211 08:16:15
""" test halfedge.operations

created: 181121 13:38:46

"""

import random
from contextlib import suppress
from itertools import chain, combinations, permutations
from operator import attrgetter
from typing import Any, Dict

import pytest
import random

from .conftest import get_canonical_mesh
from ..halfedge.half_edge_elements import Edge, ManifoldMeshError, Vert, Face
from ..halfedge.half_edge_object import HalfEdges
from ..halfedge.validations import validate_mesh
from ..halfedge.element_attributes import IncompatibleAttributeBase, NumericAttributeBase


class NamedAttribute(IncompatibleAttributeBase):
    """For color, flags. etc. to ensure attributes are passed"""

class Coordinate(NumericAttributeBase):
    """Hold coordinates when creating a mesh from a list of vertices"""


def sorted_by_sn(elements):
    return sorted(elements, key=attrgetter("sn"))


VERT_IN_ANOTHER_FACE = "orig or dest in mesh but not on given face"


class TestInsertEdge:
    def test_insert_into_empty_mesh(self) -> None:
        """First edge into empty mesh creates Hole"""
        mesh = HalfEdges()
        mesh.insert_edge(Vert(), Vert())
        assert len(tuple(mesh.verts)) == 2
        assert len(tuple(mesh.edges)) == 2
        assert len(tuple(mesh.holes)) == 1
        validate_mesh(mesh)

    @pytest.mark.parametrize("index", range(4))
    def test_infer_face_both_verts(self, index, mesh_faces) -> None:
        """Infer face when two verts on face given"""
        mesh, face = mesh_faces
        new_edge = mesh.insert_edge(face.verts[index], face.verts[index - 2])
        assert new_edge.pair.face == face
        validate_mesh(mesh)

    @pytest.mark.parametrize("index", range(4))
    def test_squares_become_two_triangles(self, index, mesh_faces) -> None:
        """Infer face when two verts on face given"""
        mesh, face = mesh_faces
        new_edge = mesh.insert_edge(face.verts[index], face.verts[index - 2], face)
        assert len(new_edge.face_edges) == 3
        assert len(new_edge.pair.face_edges) == 3
        validate_mesh(mesh)

    @pytest.mark.parametrize("index", range(4))
    def test_orig_on_face(self, index, mesh_faces) -> None:
        """Connect to new vert from orig on face"""
        mesh, face = mesh_faces
        orig = face.verts[index]
        new_edge = mesh.insert_edge(orig, Vert(), face)
        assert new_edge.face == face
        assert new_edge.pair.face == face
        validate_mesh(mesh)

    @pytest.mark.parametrize("index", range(4))
    def test_dest_on_face(self, index, mesh_faces) -> None:
        """Connect to dest on face from new vert"""
        mesh, face = mesh_faces
        dest = face.verts[index]
        new_edge = mesh.insert_edge(Vert(), dest, face)
        assert new_edge.face == face
        assert new_edge.pair.face == face
        validate_mesh(mesh)

    @pytest.mark.parametrize("index", range(4))
    def test_infer_edge(self, index, mesh_faces) -> None:
        """Infer correct vert when edge passed as orig_elem"""
        mesh, face = mesh_faces
        orig = face.edges[index]
        new_edge = mesh.insert_edge(orig, Vert(), face)
        assert new_edge.face == face
        assert new_edge.pair.face == face
        validate_mesh(mesh)

    @pytest.mark.parametrize("index", range(4))
    def test_insert_will_not_overwrite(self, index, mesh_faces) -> None:
        """Raise ManifoldMeshError if attempting to overwrite existing edge."""
        mesh, face = mesh_faces
        with pytest.raises(ManifoldMeshError) as err:
            mesh.insert_edge(face.verts[index], face.verts[index - 1], face)
        assert "overwriting existing edge" in err.value.args[0]

    @pytest.mark.parametrize("index", range(4))
    def test_orig_off_face(self, index, mesh_faces) -> None:
        """Raise ManifoldMeshError if any vert in mesh but not on given face"""
        mesh, face = mesh_faces
        dest = face.verts[index]
        orig = next(
            x for x in mesh.verts if x not in face.verts and x not in dest.neighbors
        )
        with pytest.raises(ManifoldMeshError) as err:
            mesh.insert_edge(orig, dest, face)
        assert VERT_IN_ANOTHER_FACE in err.value.args[0]

    @pytest.mark.parametrize("index", range(4))
    def test_dest_off_face(self, index, mesh_faces) -> None:
        """Raise ManifoldMeshError if any vert in mesh but not on given face"""
        mesh, face = mesh_faces
        orig = face.verts[index]
        dest = next(
            x for x in mesh.verts if x not in face.verts and x not in orig.neighbors
        )
        with pytest.raises(ManifoldMeshError) as err:
            mesh.insert_edge(orig, dest, face)
        assert VERT_IN_ANOTHER_FACE in err.value.args[0]

    def test_orig_and_dest_off_face(self, mesh_faces) -> None:
        """Raise ManifoldMeshError if any vert in mesh but not on given face"""
        mesh, face = mesh_faces
        orig = next(x for x in mesh.verts if x not in face.verts)
        dest = next(
            x
            for x in mesh.verts
            if x not in face.verts and x != orig and x not in orig.neighbors
        )
        with pytest.raises(ManifoldMeshError) as err:
            mesh.insert_edge(orig, dest, face)
        assert VERT_IN_ANOTHER_FACE in err.value.args[0]

    @pytest.mark.parametrize("index", range(4))
    def test_orig_eq_dest(self, index, mesh_faces) -> None:
        """Raise ManifoldMeshError if orig == dest"""
        mesh, face = mesh_faces
        orig = face.verts[index]
        with pytest.raises(ManifoldMeshError) as err:
            mesh.insert_edge(orig, orig, face)
        assert "orig and dest are the same" in err.value.args[0]

    def test_floating_edge(self, mesh_faces) -> None:
        """Raise ManifoldMeshError neither vert in mesh (and mesh not empty)"""
        mesh, face = mesh_faces
        with pytest.raises(ManifoldMeshError) as err:
            mesh.insert_edge(Vert(), Vert(), face)
        assert "adding floating edge to existing face" in err.value.args[0]

    def test_fail_to_infer(self, he_mesh) -> None:
        """Raise ValueError if face not given and not unambiguous"""
        mesh = he_mesh
        with pytest.raises(ValueError) as err:
            mesh.insert_edge(Vert(), Vert())
        assert "face cannot be determined from orig and dest" in err.value.args[0]

    @pytest.mark.parametrize("index", range(4))
    def test_face_attrs_pass(self, index, mesh_faces) -> None:
        """Pass attributes from face when face is split"""
        mesh, face = mesh_faces
        face.set_attrib(IncompatibleAttributeBase('orange'))
        edge = mesh.insert_edge(face.verts[0], face.verts[2])
        assert edge.face.get_attrib(IncompatibleAttributeBase) == "orange"
        assert edge.pair.face.get_attrib(IncompatibleAttributeBase)== "orange"

    @pytest.mark.parametrize("index", range(4))
    def test_edge_kwargs(self, index, mesh_faces) -> None:
        """Shared face edge attributes pass to new edge"""
        mesh, face = mesh_faces
        for edge in face.edges[:2]:
            edge.set_attrib(IncompatibleAttributeBase('blue'))
        for edge in face.edges[2:]:
            edge.set_attrib(IncompatibleAttributeBase('red'))
        edge = mesh.insert_edge(face.verts[0], face.verts[2])
        assert edge.get_attrib(IncompatibleAttributeBase) == "red"
        assert edge.pair.get_attrib(IncompatibleAttributeBase) == "blue"


class TestInsertVert:
    def test_vert_attrs_pass(self, mesh_faces) -> None:
        """Shared face.verts attrs pass to new vert"""
        mesh, face = mesh_faces
        for vert in face.verts:
            vert.set_attrib(NamedAttribute('purple'))
        new_vert = mesh.insert_vert(face)
        assert new_vert.get_attrib(NamedAttribute) == "purple"  # type: ignore

    def test_vert_kwargs_pass(self, mesh_faces) -> None:
        """vert_kwargs assigned to new vert"""
        mesh, face = mesh_faces
        for vert in face.verts:
            vert.set_attrib(NamedAttribute('purple'))
        new_vert = mesh.insert_vert(face)
        assert new_vert.get_attrib(NamedAttribute) == 'purple'  # type: ignore


class TestRemoveEdge:
    def test_face_attributes_passed(self, mesh_edges) -> None:
        """face attributed inherited"""
        mesh, edge = mesh_edges
        edge.face.set_attrib (NamedAttribute("brown"))
        edge.pair.face.set_attrib (NamedAttribute("brown"))
        new_face = mesh.remove_edge(edge)
        validate_mesh(mesh)
        assert new_face.get_attrib(NamedAttribute) == "brown"

    def test_face_kwargs_passed(self, mesh_edges) -> None:
        """face_kwargs become attributes"""
        mesh, edge = mesh_edges
        new_face = mesh.remove_edge(edge).set_attrib(NamedAttribute('green'))
        assert new_face.get_attrib(NamedAttribute) == "green"

    def test_missing_edge(self, he_mesh) -> None:
        """Raise ManifoldMeshError if edge not in mesh"""
        with pytest.raises(ManifoldMeshError) as err:
            he_mesh.remove_edge(Edge())
        assert "does not exist" in err.value.args[0]

    def test_remove_edge_bridge(self, meshes_vlvi: Dict[str, Any]) -> None:
        """Raise an exception if mesh is separated into 'islands'.

        The surrounding hole face will be disjoint (referenced edge will never
        "next" to other edges). See remove_edge docstring.
        """
        row_vl = meshes_vlvi["grid_vl"][:8]
        row_vi = {x for x in meshes_vlvi["grid_vi"] if not any(y > 7 for y in x)}
        vl = [Vert(Coordinate(x)) for x in row_vl]
        mesh = HalfEdges.from_vlvi(vl, row_vi)
        outer_center_edges = [
            x
            for x in mesh.edges
            if x.orig.valence == 3 and x.dest.valence == 3 and x.pair.face in mesh.holes
        ]
        mesh.remove_edge(outer_center_edges[0])
        with pytest.raises(ValueError) as err:
            mesh.remove_edge(outer_center_edges[1])
        assert "would create non-manifold" in err.value.args[0]

    def test_remove_edge_to_empty_mesh(self, he_mesh) -> None:
        """Mesh can be reduced to nothing.

        Can only be reduced this far if edges are removed in proper order
        (never disjoint)
        """
        while he_mesh.edges:
            edges = list(he_mesh.edges)
            random.shuffle(edges)
            for edge in edges:
                with suppress(ValueError):
                    he_mesh.remove_edge(edge)
                validate_mesh(he_mesh)
        assert he_mesh.edges == set()

    def test_pair_type_is_new_face_type(self, mesh_edges) -> None:
        """edge pair is Hole, new_face is Hole"""
        mesh, edge = mesh_edges
        pair_face = edge.pair.face
        new_face = mesh.remove_edge(edge)
        assert type(pair_face).__name__ == type(new_face).__name__


class TestRemoveAddEdge:
    def test_remove_then_add(self, he_mesh) -> None:
        """Remove then re-add each edge"""
        for edge in tuple(x for x in he_mesh.edges if id(x) > id(x.pair)):
            orig, dest = edge.orig, edge.dest
            face = he_mesh.remove_edge(edge)
            validate_mesh(he_mesh)
            he_mesh.insert_edge(orig, dest, face)
            validate_mesh(he_mesh)

    def test_remove_then_add2(self, he_mesh) -> None:
        """Same as above with opposite edges"""
        for edge in tuple(x for x in he_mesh.edges if id(x) < id(x.pair)):
            orig, dest = edge.orig, edge.dest
            face = he_mesh.remove_edge(edge)
            validate_mesh(he_mesh)
            he_mesh.insert_edge(orig, dest, face)
            validate_mesh(he_mesh)


class TestRemoveVert:
    def test_remove_vert_corner(self, he_grid) -> None:
        """Remove outside verts to get five-face plus sign."""
        corners = {x for x in he_grid.verts if x.valence == 2}
        for corner in corners:
            he_grid.remove_vert(corner)
        assert sorted(len(x.edges) for x in he_grid.faces) == [4, 4, 4, 4, 4]
        assert sorted(len(x.edges) for x in he_grid.holes) == [12]

    def test_interior_verts(self, he_grid) -> None:
        """Remove interior verts from grid leaves one big face."""
        corners = {x for x in he_grid.verts if x.valence == 4}
        for corner in corners:
            he_grid.remove_vert(corner)
        assert sorted(len(x.edges) for x in he_grid.faces) == [12]
        assert sorted(len(x.edges) for x in he_grid.holes) == [12]

    @pytest.mark.parametrize("_", range(10))
    def test_remove_missing_edge(self, he_grid, _) -> None:
        """Raise ValueError if vert is not in mesh"""
        vert = random.choice(tuple(he_grid.verts))
        _ = he_grid.remove_vert(vert)
        with pytest.raises(ValueError, match="vert is not in mesh"):
            _ = he_grid.remove_vert(vert)

    @pytest.mark.parametrize(
        "i, j", chain(*(permutations(x) for x in combinations(range(4), 2)))
    )
    def test_remove_vert_bridge(self, i, j, he_grid) -> None:
        """Raise ManifoldMeshError if vert has a bridge edge."""
        # create plus
        corners = {x for x in he_grid.verts if x.valence == 2}
        for corner in corners:
            he_grid.remove_vert(corner)

        # remove any two valence four verts to break manifold
        vl = [x for x in he_grid.vl if x.valence == 4]
        he_grid.remove_vert(vl[i])
        with pytest.raises(ManifoldMeshError) as err:
            he_grid.remove_vert(vl[j])
        assert "removing vert would create non-manifold mesh" in err.value.args[0]

    def test_peninsulas(self) -> None:
        """Remove a vert with peninsulas and regular edges"""
        mesh = HalfEdges.from_vlvi(
            [Vert(Coordinate(x)) for x in range(8)],
            fi={(5, 6, 2, 1)},
            hi={(6, 5, 4, 7, 4, 3, 4, 0, 4, 5, 1, 2)},
        )
        vert = next(x for x in mesh.verts if x.valence == 4)
        mesh.remove_vert(vert)
        assert [x.sides for x in mesh.faces] == [4]
        assert [x.sides for x in mesh.holes] == [4]

    @pytest.mark.parametrize("_", range(10))
    def test_remove_then_insert(self, he_cube, _) -> None:
        """Remove then replace any vert (not against a hole) and keep mesh intact"""
        for vert in tuple(he_cube.verts):
            new_face = he_cube.remove_vert(vert)
            he_cube.insert_vert(new_face)


class TestRemoveFace:
    def test_remove_face(self, mesh_faces) -> None:
        """A poor test, running to see it it works.
        TODO: assert exception
        TODO: assert hole fills edge when face on boundary
        TODO: test returns face"""
        mesh, face = mesh_faces
        mesh.remove_face(face)
        validate_mesh(mesh)


class TestSplitEdge:
    def test_vert_attributes_passed_to_new_vert(self, mesh_edges) -> None:
        """New vert inherits common attributes of orig and dest verts."""
        mesh, edge = mesh_edges
        edge.orig.set_attrib(NamedAttribute('black'))
        edge.dest.set_attrib(NamedAttribute('black'))
        new_vert = mesh.split_edge(edge)
        assert new_vert.get_attrib(NamedAttribute) == "black"

    def test_geometry(self, mesh_edges) -> None:
        """Add one to number of face edges."""
        mesh, edge = mesh_edges
        edge_face = edge.face
        len_edge_face = len(edge_face.edges)
        pair_face = edge.pair.face
        len_pair_face = len(pair_face.edges)
        mesh.split_edge(edge)
        assert len(edge_face.edges) == len_edge_face + 1
        assert len(pair_face.edges) == len_pair_face + 1


class TestFlipEdge:
    def test_flip(self) -> None:
        """Flip edge in adjacent triangles"""
        # class MyVert(Vert["MyVert", "MyEdge", "MyFace"]):
        # nnum: int

        # class MyEdge(Edge["MyVert", "MyEdge", "MyFace"]):
        # pass

        # class MyFace(Face["MyVert", "MyEdge", "MyFace"]):
        # pass

        # class MyHalfEdges(HalfEdges["MyVert", "MyEdge", "MyFace"]):
        # vert = MyVert
        # edge = MyEdge
        # face = MyFace

        vl = [Vert(Coordinate(x)) for x in range(4)]
        vi = {(0, 1, 2), (0, 2, 3)}
        mesh = HalfEdges.from_vlvi(vl, vi)
        split = next(x for x in mesh.edges if x.orig.valence == 3 and x.pair.orig.valence == 3)
        new_edge = mesh.flip_edge(split)
        assert split not in mesh.edges
        assert new_edge.orig.valence == 3
        assert new_edge.dest.valence == 3
        validate_mesh(mesh)


class TestCollapseEdge:
    @pytest.mark.parametrize("repeat", range(100))
    def test_collapse_to_empty(self, he_mesh, repeat) -> None:
        """Collapse edge till mesh is empty"""
        while he_mesh.edges:
            edges = list(he_mesh.edges)
            random.shuffle(edges)
            for edge in edges:
                with suppress(ValueError):
                    he_mesh.collapse_edge(edge)

    @pytest.mark.parametrize("repeat", range(100))
    def test_drum(self, repeat) -> None:
        """Collapse edge with large faces till mesh is empty"""
        top = tuple(range(10))
        bot = tuple(range(10, 20))
        legs = tuple(zip(top, bot))
        sides = {x + tuple(reversed(y)) for x, y in zip(legs, (legs * 2)[1:])}
        vl = [Vert(Coordinate(x)) for x in range(20)]
        fi = {top} | {tuple(reversed(bot))} | sides
        drum = HalfEdges().from_vlvi(vl, fi)
        while drum.edges:
            edges = list(drum.edges)
            random.shuffle(edges)
            for edge in edges:
                with suppress(ValueError):
                    drum.collapse_edge(edge)

    def test_collapse_dart(self) -> None:
        """Create a double slit face by collapsing on side of a triangle inside a dart

        Function will remove both slits.

        Start with a mesh with two faces, a triange and a "dart" (two adjacent
        triangles). An example would be:

         * start with a triangle
         * place a new point in the center and re-triangulate the triangle into three
           smaller triangles
         * remove one interior edge, so two faces remain, a triangle and a dart

        If the exterior edge of the triangle were collapsed, the triangle would
        become a 2-edge face, which collapse_edge would remove. The dart would still
        be a 4-edge face, but the 1st and 3rd (or 2nd and 4th) points would be
        identical so each half of the face would be unambiguously linear.
        collapse_edge will remove these as well.
        """
        vl = [Vert(Coordinate(x)) for x in range(4)]
        mesh = HalfEdges.from_vlvi(vl, {(0, 1, 3), (1, 2, 0, 3)})
        edge = next(
            x for x in mesh.edges if x.orig.get_attrib(Coordinate) == 0 and x.dest.get_attrib(Coordinate) == 1
        )
        mesh.collapse_edge(edge)
        assert not mesh.verts
