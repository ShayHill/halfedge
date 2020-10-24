#!/usr/bin/env python3
# Last modified: 181125 10:23:30
""" test halfedge.operations

created: 181121 13:38:46

"""

import random
from typing import Any, Dict

import pytest
from ..halfedge import operations as ops
from ..halfedge.half_edge_elements import ManifoldMeshError, Vert, Hole
from ..halfedge.half_edge_querries import StaticHalfEdges
from ..halfedge.constructors import mesh_from_vlvi
from ..halfedge.validations import validate_mesh
from . import are_equivalent_meshes
from operator import attrgetter


def sorted_by_sn(elements):
    return sorted(elements, key=attrgetter("sn"))


def test_full_edges_only(he_meshes: Dict[str, Any]) -> None:
    """12 in grid. All on face. No pairs."""
    int_edges = he_meshes["grid"].interior_edges
    len_int_edges = len(int_edges)
    edges = tuple(ops.full_edges_only(int_edges))
    assert len(edges) == 12
    assert all(x.pair not in edges for x in edges)
    assert not any(isinstance(x.face, Hole) for x in edges)
    # no side effects
    assert len(int_edges) == len_int_edges


def test_remove_edge_increases_sns(he_meshes: Dict[str, Any]) -> None:
    """Altered elements will have new serial numbers."""
    for mesh in he_meshes.values():
        max_sn = mesh.last_issued_sn
        edge = next(iter(mesh.edges))
        orig, dest = edge.orig, edge.dest
        edge_face, pair_face = edge.face, edge.pair.face
        ops.remove_edge(mesh, edge)
        validate_mesh(mesh)

        # other sns not altered
        assert all(x.sn <= max_sn for x in mesh.elements)


def test_remove_edge_bridge(meshes_vlvi: Dict[str, Any]) -> None:
    """Raise an exception if mesh is separated into 'islands'.

    The surrounding hole face will be disjoint (referenced edge will never
    "next" to other edges). See remove_edge docstring.
    """
    row_vl = meshes_vlvi["grid_vl"][:8]
    row_vi = {x for x in meshes_vlvi["grid_vi"] if not any(y > 7 for y in x)}
    mesh = mesh_from_vlvi(row_vl, row_vi)
    outer_center_edges = [
        x for x in mesh.edges
        if x.orig.valence == 3 and x.dest.valence == 3 and x.pair.face in mesh.holes
    ]
    ops.remove_edge(mesh, outer_center_edges[0])
    with pytest.raises(ManifoldMeshError) as err:
        ops.remove_edge(mesh, outer_center_edges[1])
    assert "would create non-manifold" in err.value.args[0]


def test_remove_edge_to_empty_mesh(he_meshes: Dict[str, Any]) -> None:
    """Mesh can be reduced to nothing.

    Can only be reduced this far if edges are removed in proper order
    (never disjoint)
    """
    for mesh in he_meshes.values():
        while mesh.edges:
            edges = list(mesh.edges)
            random.shuffle(edges)
            for edge in edges:
                try:
                    ops.remove_edge(mesh, edge)
                except (ValueError, ManifoldMeshError):
                    pass
                validate_mesh(mesh)
        assert mesh.edges == set()


def test_remove_vert_corner(he_meshes: Dict[str, Any]) -> None:
    """Remove outside verts to get five-face plus sign."""
    test = he_meshes["grid"]

    # mesh with faces removed
    vl = [x.coordinate for x in test.vl]
    vi = test.fi
    ctrl = mesh_from_vlvi(vl, {x for x in vi if not {0, 3, 12, 15} & set(x)})

    for corner in tuple(x for x in test.verts if x.valence == 2):
        ops.remove_vert(test, corner)

    assert are_equivalent_meshes(test, ctrl)


def test_remove_vert_interior(he_meshes: Dict[str, Any]) -> None:
    """Remove interior grid verts for one big box."""
    test = he_meshes["grid"]

    for interior_vert in tuple(x for x in test.verts if x.valence == 4):
        ops.remove_vert(test, interior_vert)

    assert [len(x.edges) for x in test.faces] == [12]
    assert [len(x.edges) for x in test.holes] == [12]


def test_remove_vert_bridge(he_meshes: Dict[str, Any]) -> None:
    """Raise exception if edge on vert is a bridge edge."""
    mesh = he_meshes["grid"]
    verts = sorted_by_sn(mesh.verts)
    for i in (0, 5, 10):
        ops.remove_vert(mesh, verts[i])
        validate_mesh(mesh)
    snapshot = StaticHalfEdges(mesh.edges)
    with pytest.raises(ManifoldMeshError) as err:
        ops.remove_vert(mesh, verts[15])
    assert "would create non-manifold" in err.value.args[0]

    # fails before altering mesh
    assert are_equivalent_meshes(mesh, snapshot)


def test_remove_vert_peninsulas(he_meshes: Dict[str, Any]) -> None:
    """Remove vert with peninsula edges.

    Bypass exception for vert edges that do not split a face
    """
    mesh = he_meshes["grid"]
    interior_vert = sorted_by_sn(mesh.verts)[5]
    for edge in [x for x in mesh.edges if x.pair.face in mesh.holes]:
        ops.remove_edge(mesh, edge)
    ops.remove_vert(mesh, interior_vert)  # assert NOT raises
    assert len(mesh.edges) == 16


# TODO: restore this test
# def test_remove_face(he_meshes: Dict[str, Any]) -> None:
#     """Remove center face for a plus-sign-shaped center face."""
#     test = he_meshes["grid"]
#
#     vi = test.fi
#     vi = [vi[x] for x in (0, 2, 6, 8)]
#     vi.append([1, 2, 6, 7, 11, 10, 14, 13, 9, 8, 4, 5])
#     ctrl = mesh_from_vlvi([x.coordinate for x in test.vl], vi)
#
#     ops.remove_face(test, sorted_by_sn(test.faces)[4])
#     assert are_equivalent_meshes(test, ctrl)


# TODO: restore this test
# def test_remove_face_fail(he_meshes: Dict[str, Any]) -> None:
#     """Raise exception when removing center face would leave a disjoint face."""
#     mesh = he_meshes["grid"]
#
#     center_face = sorted_by_sn(mesh.faces)[4]
#     leave_kissing = [sorted_by_sn(mesh.edges)[x] for x in (21, 30)]
#     for edge in leave_kissing:
#         ops.remove_edge(mesh, edge)
#
#     with pytest.raises(ManifoldMeshError) as err:
#         ops.remove_face(mesh, center_face)
#     assert "would create non-manifold" in err.value.args[0]


def test_insert_will_not_overwrite(he_meshes: Dict[str, Any]) -> None:
    """Raise exception if attempting to overwrite existing edge."""
    grid = he_meshes["grid"]
    face = sorted_by_sn(grid.faces)[0]
    edge = sorted_by_sn(grid.edges)[0]
    orig, dest = edge.orig, edge.dest
    with pytest.raises(ManifoldMeshError) as err:
        ops.insert_edge(grid, orig, dest, face)
    assert "overwriting existing edge" in err.value.args[0]


def test_insert_edge_marks_changes(he_meshes: Dict[str, Any]) -> None:
    """Insert_edge affected faces and new edge have new sns."""
    grid = he_meshes["grid"]
    max_sn = grid.last_issued_sn
    face = random.choice(tuple(grid.faces))
    orig, dest = face.verts[0], face.verts[2]
    ops.insert_edge(grid, orig, dest, face)

    assert len([x for x in grid.edges if x.sn > max_sn]) == 2
    assert len([x for x in grid.faces if x.sn > max_sn]) == 1
    assert len([x for x in grid.elements if x.sn > max_sn]) == 3


def test_insert_edge_0to1() -> None:
    """Creates a valid mesh from either direction."""
    mesh = mesh_from_vlvi(
        [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1)], {(0, 1, 2, 3, 4, 5)}
    )
    face = next(iter(mesh.faces))
    verts = sorted_by_sn(mesh.verts)[1::3]
    ops.insert_edge(mesh, verts[0], verts[1], face)
    validate_mesh(mesh)


def test_insert_edge_1to0() -> None:
    mesh = mesh_from_vlvi(
        [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1)], {(0, 1, 2, 3, 4, 5)}
    )
    face = next(iter(mesh.faces))
    verts = sorted_by_sn(mesh.verts)[1::3]
    ops.insert_edge(mesh, verts[1], verts[0], face)
    validate_mesh(mesh)


def test_remove_then_insert(meshes_vlvi: Dict[str, Any]) -> None:
    """Remove an edge, then add it back."""
    for vl, vi in (
        (meshes_vlvi["cube_vl"], meshes_vlvi["cube_vi"]),
        (meshes_vlvi["grid_vl"], meshes_vlvi["grid_vi"]),
    ):
        ctrl = mesh_from_vlvi(vl, vi)
        test = mesh_from_vlvi(vl, vi)

        for edge in tuple(e for e in test.edges if e.sn < e.pair.sn):
            ops.remove_edge(test, edge)
            ops.insert_edge(test, edge.orig, edge.dest, edge.pair.face)
            # try:
            #     assert are_equivalent_meshes(test, ctrl)
            # except:
            #     breakpoint()


def test_insert_edge_new_vert() -> None:
    """Vert object added to face results in two additional face edges."""
    corners = [(0.0, 0), (1, 0), (1, 1), (0, 1)]
    faces = {(0, 1, 2, 3)}
    mesh = mesh_from_vlvi(corners, faces)
    max_sn = mesh.last_issued_sn

    face = sorted_by_sn(mesh.faces)[0]
    orig = sorted_by_sn(mesh.verts)[0]
    new_vert = Vert(mesh=mesh, coordinate=(0.5, 0.5))
    ops.insert_edge(mesh, orig, new_vert, face)
    validate_mesh(mesh)

    # TODO: restore vlvi checks
    # assert mesh.vl == corners[1:] + [corners[0], (0.5, 0.5)]
    # assert len(mesh.vi) == 1
    # assert len(mesh.vi[0]) == 6
    # assert str(mesh.vi[0])[1:-1] in str([3, 0, 1, 2, 3, 4] * 2)

    assert new_vert.sn > max_sn
    assert len([x for x in mesh.edges if x.sn > max_sn]) == 2

    elements = sorted_by_sn(mesh.verts | mesh.edges | mesh.faces | mesh.holes)
    assert all(x.sn <= max_sn for x in elements[:-3])


# TODO: restore test
# def test_split_edge_works() -> None:
#     """Algorithmic edge split matches explicit edge split."""
#
#     # explicitly defined with edge split
#     vl = [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1), (1, 0.5)]
#     vi = {(0, 1, 6, 4, 5), (1, 2, 3, 4, 6)}
#     explicit = mesh_from_vlvi(vl, vi)
#     validate_mesh(explicit)
#
#     # adjacent faces, split with method
#     vl = vl[:-1]
#     vi = {(0, 1, 4, 5), (1, 2, 3, 4)}
#     algorithmic = mesh_from_vlvi(vl, vi)
#     validate_mesh(algorithmic)
#
#     edge = sorted_by_sn(algorithmic.edges)[1]
#     vert = Vert(coordinate=(1, 0.5))
#     ops.split_edge(algorithmic, edge, vert)
#     validate_mesh(algorithmic)
#
#     assert are_equivalent_meshes(algorithmic, explicit)


def test_split_edge_updates_sns(he_meshes: Dict[str, Any]) -> None:
    """One new vert, four "new" edges, two "new" faces."""
    for mesh in he_meshes.values():

        for edge in tuple(filter(lambda x: x.sn < x.pair.sn, mesh.edges)):

            max_sn = mesh.last_issued_sn
            ops.split_edge(mesh, edge)
            validate_mesh(mesh)

            # updated serial numbers
            assert len([x for x in mesh.verts if x.sn >= max_sn]) == 1
            assert len([x for x in mesh.edges if x.sn >= max_sn]) == 4
            assert len([x for x in mesh.elements if x.sn >= max_sn]) == 5


def test_split_edge_destroys_old_edges(he_meshes: Dict[str, Any]) -> None:
    """Cannot split an edge twice.

    Old edge will have disappeared.
    """
    mesh = he_meshes["cube"]
    edge = next(iter(mesh.edges))
    pair = edge.pair

    ops.split_edge(mesh, edge)

    with pytest.raises(ValueError):
        ops.split_edge(mesh, edge)

    with pytest.raises(ValueError):
        ops.split_edge(mesh, pair)


def test_add_edge_vert_passes_all_attrs() -> None:
    """Every new edge inherits from edge or edge.pair."""
    vl = [(0, 0, 0)] * 6
    vi = {(0, 1, 4, 5), (1, 2, 3, 4)}
    mesh = mesh_from_vlvi(vl, vi)
    validate_mesh(mesh)
    max_sn = mesh.last_issued_sn

    edge = sorted_by_sn(mesh.edges)[1]
    edge.orig.mark = "orig of edge"
    edge.pass_this = "edge attr"  # type: ignore
    edge.pair.pass_this = "pair attr"  # type: ignore

    ops.split_edge(mesh, edge)
    new_edges = {x for x in mesh.edges if x.sn > max_sn}
    assert len(new_edges) == 4
    edge = next(x for x in new_edges if hasattr(x.orig, "mark"))
    assert edge.pass_this == "edge attr"  # type: ignore
    assert edge.next.pass_this == "edge attr"  # type: ignore
    assert edge.pair.pass_this == "pair attr"  # type: ignore
    assert edge.next.pair.pass_this == "pair attr"  # type: ignore
