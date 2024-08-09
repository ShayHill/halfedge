"""Test exceptions and details in the elements module.

Most of the code in the half_edge_element module is run to accomplish setup for tests
in other test modules. This module tests the exceptions and any pieces that were not
tested elsewhere.

:author: Shay Hill
:created: 2024-08-09
"""

from typing import Tuple

import pytest

from halfedge.half_edge_elements import Face, Vert
from halfedge.half_edge_object import HalfEdges


class TestVert:
    def test_return_empty_edges_if_no_edge_set(self) -> None:
        """Return an empty list of edges if no edge has been set."""
        assert Vert().edges == []

    def test_holes(self) -> None:
        """Test that holes are correctly identified."""
        vl = [Vert() for _ in range(3)]
        fi: list[Tuple[int, ...]] = [(0, 1, 2)]
        mesh = HalfEdges.from_vlfi(vl, fi)
        assert len(mesh.holes) == 1

    def test_return_empty_neighbors_if_no_edge_set(self) -> None:
        """Return an empty list of neighbors if no edge has been set."""
        vert = Vert()
        assert vert.neighbors == []


class TestEdge:
    def test_raise_attribute_error_if_orig_not_set(self) -> None:
        """Raise an AttributeError if orig is not set."""
        edge = HalfEdges().new_edge()
        with pytest.raises(AttributeError) as err:
            edge.orig
        assert ".orig" in str(err.value)

    def test_raise_attribute_error_if_face_not_set(self) -> None:
        """Raise an AttributeError if face is not set."""
        edge = HalfEdges().new_edge()
        with pytest.raises(AttributeError) as err:
            edge.face
        assert ".face" in str(err.value)

    @pytest.mark.parametrize("edge_index", range(3))
    def test_vert_faces(self, edge_index: int) -> None:
        """Test that vert_faces returns the correct faces."""
        vl = [Vert() for _ in range(3)]
        fi: list[Tuple[int, ...]] = [(0, 1, 2)]
        mesh = HalfEdges.from_vlfi(vl, fi)
        assert len(mesh.faces) == 1
        (face,) = mesh.faces
        vert = face.verts[edge_index]
        assert vert.faces == [face]

    @pytest.mark.parametrize("edge_index", range(3))
    def test_vert_holes(self, edge_index: int) -> None:
        """Test that vert_holes returns the correct holes."""
        vl = [Vert() for _ in range(3)]
        fi: list[Tuple[int, ...]] = [(0, 1, 2)]
        mesh = HalfEdges.from_vlfi(vl, fi)
        assert len(mesh.holes) == 1
        (hole,) = mesh.holes
        vert = hole.verts[edge_index]
        assert vert.holes == [hole]


class TestFace:
    def test_return_empty_edges_if_no_edge_set(self) -> None:
        """Return an empty list of edges if no edge has been set."""
        assert Face().edges == []
