"""Attribute values that know how to merge with each other.

As a mesh is transformed, Verts, Edges, and Faces will be split or combined with each
other. Different attributes will combine in different ways, for instance:

    * a new vert between two boundary verts might produce another boundary vert,
      while a new vert between a boundary vert and a non-boundary vert might produce a
      non-boundary vert.
    * a new vert between two verts with defined vectors might hold the average of
      those two vectors
    * two faces with defined areas might combine into a new face with the sum of those
      two areas
    * expensive attributes might be cached

The MeshElement classes in this library don't support inheritance (with proper
typing), because the mess involved adds too much complication. If you need to extend
the Vert, Edge, Face, or HalfEdges classes with additional attributes, you will need
to define each attribute as a descendent of Attrib.

    class MyAttrib(Attrib):
        ...

    vert = Vert()
    vert.set_attrib(MyAttrib('value'))
    assert vert.get_attrib(MyAttrib) == 'value'

These attributes are held in an AttribHolder __dict__ keyed to the class name of the
attribute.

    assert vert.MyAttrib.value == 'value'
    assert vert.__dict__['MyAttrib'].value == 'value'

Rules governing combination of these properties are defined in the Attrib classes
themselves.

:author: Shay Hill
:created: 2022-06-14
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any, Generic, Optional, Protocol, Type, TypeVar, cast

if TYPE_CHECKING:
    from halfedge.half_edge_elements import MeshElementBase

_TAttribHolder = TypeVar("_TAttribHolder", bound="AttribHolder")
_TAttrib = TypeVar("_TAttrib", bound="Attrib[Any]")
_T = TypeVar("_T")


class AttribHolder:
    """Hold AttribBase instances and retrieve values"""

    def set_attrib(self: _TAttribHolder, *attribs: Attrib[Any]) -> _TAttribHolder:
        """Set attribute with an Attrib instance.

        type(attrib).__name__ : attrib

        :param attribs: Attrib instances, presumably with a None element
        attribute.
        """
        for attrib in attribs:
            attrib.element = self
            self.__dict__[type(attrib).__name__] = attrib
        return self

    def _maybe_set_attrib(self, *attribs: Attrib[Any] | None) -> None:
        """Set attribute if attrib is an Attrib. Pass silently if None"""
        self.set_attrib(*[x for x in attribs if isinstance(x, Attrib)])

    def get_attrib(self, type_: type[_TAttrib]) -> _TAttrib:
        name = type_.__name__
        try:
            return getattr(self, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' has no Attrib '{name}'")

    def try_attrib(self, type_: type[_TAttrib]) -> _TAttrib | None:
        """Try to get an attribute, None if attrib is not set.

        :param type_: type of Attrib to seek in the attrib dictionary. This
            takes a type instead of a string to eliminate any possibility of getting
            a None value just because an attrib dictionary key was mistyped.
        """
        try:
            return self.get_attrib(type_)
        except AttributeError:
            return None

    def get_attrib_value(self, type_: type[Attrib[_T]]) -> _T:
        """Get attrib value. Will fail if attrib is not set

        :param type_: type of Attrib to seek in the attrib dictionary. This
            takes a type instead of a string to eliminate any possibility of getting
            a None value just because an attrib dictionary key was mistyped.
        """
        return self.get_attrib(type_).value

    def try_attrib_value(self, type_: type[Attrib[_T]]) -> _T | None:
        """Try to get an attribute value, None if attrib is not set.

        :param type_: type of Attrib to seek in the attrib dictionary. This
            takes a type instead of a string to eliminate any possibility of getting
            a None value just because an attrib dictionary key was mistyped.
        """
        try:
            return self.get_attrib_value(type_)
        except AttributeError:
            return None

    def cached_attrib_value(self, type_: type[Attrib[_T]]) -> _T | None:
        # TODO: docstring
        attrib = self.try_attrib(type_)
        if attrib is not None and hasattr(attrib, "_value"):
            return attrib.value
        return None


class _SupportsEqual(Protocol):
    def __eq__(self: _T, other: _T) -> bool:
        pass


class _SupportsAverage(Protocol):
    def __add__(self, other):
        pass

    def __div__(self, other):
        pass


_TSupportsAverage = TypeVar("_TSupportsAverage", bound=_SupportsAverage)
_TSupportsEqual = TypeVar("_TSupportsEqual", bound=_SupportsEqual)
_TAttribValue = TypeVar("_TAttribValue")
_TElemAttrib = TypeVar("_TElemAttrib", bound="Attrib[Any]")


class Attrib(Generic[_TAttribValue]):
    """Base class for element attributes.

    MeshElementBase has methods set_attrib and get_attrib that will store
    Attrib instances in the MeshElemenBase __dict__. The Attrib class
    defines how these attributes behave when mesh elements are merged and allows a
    value (e.g., edge length) to be inferred from the Attrib.element property
    when and if needed, allowing us to cache (and potentially never access) slow
    attributes.

    Do not overload `__init__` or `value`. For the most part, treat as an ABC with
    abstract methods `merge`, `slice`, and `_infer_value`--although the base methods
    are marginally useful and instructive, so you will not need to overload both in
    every case.
    """

    __slots__ = ("_value", "element")

    def __init__(
        self,
        value: _TAttribValue | None = None,
        element: AttribHolder | None = None,
    ) -> None:
        self._value: _TAttribValue
        self.element: AttribHolder
        if value is not None:
            self._value = value
        if element is not None:
            self.element = element

    @property
    def value(self) -> _TAttribValue:
        """Return value if set, else try to infer a value"""
        if not hasattr(self, "_value"):
            value = self._infer_value()
            if value is None:
                raise TypeError(f"no value set and failed to infer from {self.element}")
            else:
                self._value = value
        return self._value

    @classmethod
    def merge(
        cls: type[_TElemAttrib], *merge_from: _TElemAttrib | None
    ) -> _TElemAttrib | None:
        """Get value of self from self._merge_from

        Use merge_from values to determine a value. If no value can be determined,
        return None. No element attribute will be set for a None return value.
        Attrib attributes are assumed None if not defined and are never defined
        if their value is None.

        This base method will not merge attributes, which is desirable in some cases.
        For example, a triangle circumcenter that will be meaningless when the
        triangle is merged.
        """
        _ = merge_from
        return None

    @classmethod
    def slice(
        cls: type[_TElemAttrib], slice_from: _TElemAttrib
    ) -> _TElemAttrib | None:
        """Define how attribute will be passed when dividing self.element.

        When an element is divided (face divided by an edge, edge divided by a vert,
        etc.) or altered, define how, if at all, this attribute will be passed to the
        altered element or pieces of the divided element. If a face with a color is
        divided, you might want to give the divided pieces the same color. If an
        attribute is lazy (e.g., edge norm), you might want to unset _value for each
        piece of a split edge.

        This base method will not pass an attribute when dividing or altering.
        """
        _ = slice_from
        return None

    def _infer_value(self) -> _TAttribValue | None:
        """Get value of self from self._element

        Use the containing element to determine a value for self. If no value can be
        determined, return None.

        The purpose is to allow lazy attributes like edge norm and face area. Use
        caution, however. These need to be calculated before merging since the method
        may not support the new shape. For instance, this method might calculate the
        area of a triangle, but would fail if two triangles were merged into a
        square. To keep this safe, the _value is colculated *before* any merging. In
        the "area of a triangle" example,

            * The area calculation is deferred until the first merge.
            * At the first merge, the area of each merged triangle is calculated. The
              implication here is that calculation *cannot* be deferred till after a
              merge.
            * the merged method areas of the merged triangles at the first and
              subsequent mergers, so further triangle area calculations (which
              wouldn't work on the merged shapes anyway) are not required.

        If you infer a value, cache it by setting self._value.

        If you do not intend to infer values, raise an exception. This exception
        should occur *before* an AttributeError is raised for a potentially missing
        element attribute. It should be clear that _infer_value failed because there
        is no provision for inferring this Attrib.value, *not* because the
        user failed to set the Attrib property attribute.
        """
        raise NotImplementedError(
            f"'{type(self).__name__}' has no provision "
            + "for inferring a value from 'self.element'"
        )


class ContagionAttrib(Attrib[_TSupportsEqual]):
    """Spread value when combining with anything.

    This is for element properties like 'IsHole' that are always passed when combining
    elements. The value of the attribute is always True. If any element in a group of
    to-be-merged elements has a ContagionAttributeBase attribute, then the merged
    element will have that attribute.
    """

    def __init__(
        self,
        value: _TSupportsEqual | None = None,
        element: MeshElementBase | None = None,
    ) -> None:
        super().__init__(cast("_TSupportsEqual", True), element)

    @classmethod
    def merge(
        cls: type[_TAttrib], *merge_from: _TAttrib | None
    ) -> _TAttrib | None:
        """If any element has a ContagionAttributeBase attribute, return a new
        instance with that attribute. Otherwise None.
        """
        with suppress(AttributeError):
            if any(getattr(x, "value", None) for x in merge_from):
                return cls()
        return None

    @classmethod
    def slice(cls: type[_TAttrib], slice_from: _TAttrib) -> _TAttrib | None:
        """Copy attribute to slices.

        Holes are defined with IsHole(ContagionAttributeBase), so this will split a
        non-face hole into two non-face holes and a hole (is_face == True) into two
        holes.
        """
        if getattr(slice_from, "value", None):
            return cls()
        return None

    def _infer_value(self):
        raise RuntimeError(
            "This will only be called if self._value is None, "
            + "which should not happen."
        )


class IncompatibleAttrib(Attrib[_TSupportsEqual]):
    """Keep value when all merge_from values are the same

    This class in intended for flags like IsEdge or Hardness.
    """

    @classmethod
    def merge(cls, *merge_from):
        """If all values match and every contributing element has an analog, return
        a new instance with that value. Otherwise None.
        """
        with suppress(AttributeError):
            values = [x.value for x in merge_from]  # type: ignore
            if values and all(values[0] == x for x in values[1:]):
                return cls(values[0], None)
        return None

    @classmethod
    def slice(cls, split_from):
        """Pass the value on."""
        if value := split_from.value:
            return cls(value)
        return None

    def _infer_value(self) -> _TSupportsEqual | None:
        """No way to infer a value. If value is not set in init or merged from init
        arg merge_from, fail (i.e., return None). This should never happen."""
        return None


class NumericAttrib(Attrib[_TSupportsAverage]):
    """Average merge_from values"""

    @classmethod
    def merge(cls, *merge_from):
        """Average values if every contributor has a value. Otherwise None"""
        with suppress(AttributeError):
            values = [x.value for x in merge_from]  # type: ignore
            return cls(sum(values) / len(values))
        return None

    def _infer_value(self) -> _TSupportsAverage | None:
        """No way to infer a value. If value is not set in init or merged from init
        arg merge_from, fail (i.e., return None). This should never happen."""
        return None
