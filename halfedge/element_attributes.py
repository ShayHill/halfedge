#!/usr/bin/env python3
# last modified: 220727 11:16:56
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

from abc import ABC, abstractmethod
from contextlib import suppress
from typing import TYPE_CHECKING, Generic, Optional, Protocol, Type, TypeVar, cast

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
_T = TypeVar("_T")


def find_optional_arg_type(*args: Optional[_T]) -> Type[_T]:
    """Find the type of non-None optional args

    :param args: either None or some number of args of the same type
    :return: type of non-None args if all are the same type

    :raises: Value error if all args are None
    :raises: Value error if non-None args are of different types

    If any optional args are given, assert that they are of the same type and return
    that type. The purpose is to identify the type of ElemAttribBase values pulled
    from multiple dictionaries with x.get(key).
    """
    iter_args = iter(args)
    try:
        arg_type = next(type(x) for x in iter_args if x is not None)
    except StopIteration:
        raise ValueError("no non-None argument in args")
    if any(arg_type != type(x) for x in iter_args if x is not None):
        raise ValueError("non-None argument types do not match")
    return arg_type


class ElemAttribBase(ABC, Generic[_TAttribValue]):
    __slots__ = ("_value",)

    def __init__(
        self,
        value: Optional[_TAttribValue] = None,
        element: Optional[MeshElementBase] = None,
    ) -> None:
        self._value = value
        self.element = element

    @property
    def value(self) -> _TAttribValue:
        if self._value is None:
            self._value = self._infer_value()
            if self._value is None:
                raise TypeError(f"no value set and failed to infer from {self.element}")
        return self._value

    @classmethod
    @abstractmethod
    def merged(
        cls: Type[_TElemAttrib], *merge_from: Optional[_TElemAttrib]
    ) -> Optional[_TElemAttrib]:
        """Get value of self from self._merge_from

        Use merge_from values to determine a value. If no value can be determined,
        return None.
        """

    @abstractmethod
    def _infer_value(self) -> Optional[_TAttribValue]:
        """Get value of self from self._element

        Use the containing element to determine a value for self. If no value can be
        determined, return None.

        The purpose is to allow lazy attributes like edge norm and face area. Use
        caution, however, these need to be calculated before merging if the method
        may not support the new shape. For instance, this method might calculate the
        area of a triangle, but would fail if two triangles were merged into a
        square.
        """


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
