"""Test Attrib access and functionality.

:author: Shay Hill
:created: 2026-08-23
"""

from halfedge.half_edge_constructors import BlindHalfEdges
from halfedge.half_edge_elements import MeshElementBase
from halfedge.type_attrib import Attrib, StaticAttrib


class MyStaticAttrib(StaticAttrib[int]):
    """A test StaticAttrib for testing purposes."""


class MyNestedStaticAttrib(StaticAttrib[MyStaticAttrib]):
    """A test StaticAttrib that is not set in the mesh for testing purposes."""

class MyAttrib(Attrib[int]):
    """A test Attrib for testing purposes."""


class MyNestedAttrib(Attrib[MyAttrib]):
    """A test Attrib that is not set in the mesh for testing purposes."""

class TestBlindHalfEdges:
    def test_attrib_is_a_copy(self):
        """Test setting and getting an attribute."""
        mesh = BlindHalfEdges()
        expect = MyStaticAttrib(0)
        mesh.set_attrib(expect)
        result = mesh.get_attrib(MyStaticAttrib)
        assert result.value == expect.value
        assert type(result) is type(expect)
        assert result is not expect

    def test_value_is_not_a_copy(self):
        """Test setting and getting an attribute."""
        mesh = BlindHalfEdges()
        expect = MyNestedStaticAttrib(MyStaticAttrib(0))
        mesh.set_attrib(expect)
        result = mesh.get_attrib(MyNestedStaticAttrib)
        assert result.value is expect.value

    def test_has_attrib(self):
        """Test setting and getting an attribute."""
        mesh = BlindHalfEdges()
        assert mesh.has_attrib(MyStaticAttrib) is False
        mesh.set_attrib(MyStaticAttrib(0))
        assert mesh.has_attrib(MyStaticAttrib) is True

    def test_attrib_val(self):
        """Test setting and getting an attribute."""
        mesh = BlindHalfEdges()
        expect = MyStaticAttrib(0)
        mesh.set_attrib(expect)
        result = mesh.attrib_val(MyStaticAttrib)
        assert result == expect.value


class TestMeshElementBase:

    def test_attrib_is_a_copy(self):
        """Test setting and getting an attribute."""
        mesh = MeshElementBase()
        expect = MyAttrib(0)
        mesh.set_attrib(expect)
        result = mesh.get_attrib(MyAttrib)
        assert result.value == expect.value
        assert type(result) is type(expect)
        assert result is not expect

    def test_value_is_not_a_copy(self):
        """Test setting and getting an attribute."""
        mesh = MeshElementBase()
        expect = MyNestedAttrib(MyAttrib(0))
        mesh.set_attrib(expect)
        result = mesh.get_attrib(MyNestedAttrib)
        assert result.value is expect.value

    def test_has_attrib(self):
        """Test setting and getting an attribute."""
        mesh = MeshElementBase()
        assert mesh.has_attrib(MyAttrib) is False
        mesh.set_attrib(MyAttrib(0))
        assert mesh.has_attrib(MyAttrib) is True

    def test_attrib_val(self):
        """Test setting and getting an attribute."""
        mesh = MeshElementBase()
        expect = MyAttrib(0)
        mesh.set_attrib(expect)
        result = mesh.attrib_val(MyAttrib)
        assert result == expect.value

    def test_try_get_attrib(self):
        """Test try_get_attrib returns None if not found."""
        mesh = MeshElementBase()
        result = mesh.try_attrib(MyAttrib)
        assert result is None

    def test_try_and_fail_get_attrib(self):
        """Test try_get_attrib returns None if not found."""
        mesh = MeshElementBase()
        result = mesh.try_attrib(MyAttrib)
        assert result is None
