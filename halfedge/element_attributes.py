#!/usr/bin/env python3
# last modified: 220730 12:47:25
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
typing), because the mess involved adds too much complication. Properties like
vertex, is_boundary, area, etc are held in each element's `attrib` attribute (similar
to lxml). Rules governing combination of these properties are defined in the property
classes themselves.

ElementBase.attrib is a dictionary, the keys of which are the class names of the
ElemAttrib instances they hold, so a new ElemAttrib child class is needed for every
element property, even if the behavior is the same.

:author: Shay Hill
:created: 2022-06-14
"""

from __future__ import annotations

from contextlib import suppress
from typing import Generic, Optional, Protocol, TYPE_CHECKING, Type, TypeVar, cast

if TYPE_CHECKING:
    from .half_edge_elements import MeshElementBase


class _SupportsEqual(Protocol):
    def __eq__(self, other):
        pass


class _SupportsAverage(Protocol):
    def __add__(self, other):
        pass

    def __div__(self, other):
        pass


_TSupportsAverage = TypeVar("_TSupportsAverage", bound=_SupportsAverage)
_TSupportsEqual = TypeVar("_TSupportsEqual", bound=_SupportsEqual)
_TAttribValue = TypeVar("_TAttribValue")
_TElemAttrib = TypeVar("_TElemAttrib", bound="ElemAttribBase")


class ElemAttribBase(Generic[_TAttribValue]):
    """Base class for element attributes.

    MeshElementBase has methods set_attrib and get_attrib that will store
    ElemAttribBase instances in the MeshElemenBase __dict__. The ElemAttribBase class
    defines how these attributes behave when mesh elements are merged and allows a
    value (e.g., edge length) to be inferred from the ElemAttribBase.element property
    when and if needed, allowing us to cache (and potentially never access) slow
    attributes.

    Do not overload `__init__` or `value`. For the most part, treat as an ABC with
    abstract methods `merged` and `_infer_value`--although the base methods are
    marginally useful and instructive, so you will not need to overload both in every
    case.
    """

    __slots__ = ("_value", "element")

    def __init__(
        self,
        value: Optional[_TAttribValue] = None,
        element: Optional[MeshElementBase] = None,
    ) -> None:
        self.element: MeshElementBase
        self._value: _TAttribValue
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
    def merged(
        cls: Type[_TElemAttrib], *merge_from: Optional[_TElemAttrib]
    ) -> Optional[_TElemAttrib]:
        """Get value of self from self._merge_from

        Use merge_from values to determine a value. If no value can be determined,
        return None. No element attribute will be set for a None return value.
        ElemAttribBase attributes are assumed None if not defined and are never defined
        if their value is None.

        This base method will not merge attributes, which is desirable in some cases.
        For example, a triangle circumcenter that will be meaningless when the
        triangle is merged.
        """
        _ = merge_from
        return None

    def _infer_value(self) -> Optional[_TAttribValue]:
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
        is no provision for inferring this ElemAttribBase.value, *not* because the
        user failed to set the ElemAttribBase property attribute.
        """
        raise NotImplementedError(
            f"'{type(self).__name__}' has no provision "
            "for inferring a value from 'self.element'"
        )


class ContagionAttributeBase(ElemAttribBase[_TSupportsEqual]):
    """Spread value when combining with anything.

    This is for element properties like 'IsHole' that are always passed when combining
    elements. The value of the attribute is always True. If any element in a group of
    to-be-merged elements has a ContagionAttributeBase attribute, then the merged
    element will have that attribute.
    """

    def __init__(self, value=None, element=None) -> None:
        super().__init__(cast("_TSupportsEqual", True), element)

    @classmethod
    def merged(cls, *merge_from):
        """If any element has a ContagionAttributeBase attribute, return a new
        instance with that attribute. Otherwise None.
        """
        with suppress(AttributeError):
            if any(getattr(x, "value", None) for x in merge_from):
                return cls()
        return None

    def _infer_value(self):
        raise RuntimeError(
            "This will only be called if self._value is None, "
            "which should not happen."
        )


class IncompatibleAttributeBase(ElemAttribBase[_TSupportsEqual]):
    """Keep value when all merge_from values are the same"""

    @classmethod
    def merged(cls, *merge_from):
        """If all values match and every contributing element has an analog, return
        a new instance with that value. Otherwise None.
        """
        with suppress(AttributeError):
            values = [x.value for x in merge_from]  # type: ignore
            if values and all(values[0] == x for x in values[1:]):
                return cls(values[0], None)
        return None

    def _infer_value(self) -> Optional[_TSupportsEqual]:
        """No way to infer a value. If value is not set in init or merged from init
        arg merge_from, fail (i.e., return None). This should never happen."""
        return None


class NumericAttributeBase(ElemAttribBase[_TSupportsAverage]):
    """Average merge_from values"""

    @classmethod
    def merged(cls, *merge_from):
        """Average values if every contributor has a value. Otherwise None"""
        with suppress(AttributeError):
            values = [x.value for x in merge_from]  # type: ignore
            return cls(sum(values) / len(values))
        return None

    def _infer_value(self) -> Optional[_TSupportsAverage]:
        """No way to infer a value. If value is not set in init or merged from init
        arg merge_from, fail (i.e., return None). This should never happen."""
        return None
