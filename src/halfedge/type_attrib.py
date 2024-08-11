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

    class MyAttrib(Attrib[something]):
        ...
        def merge [define how multiple instances of this attribute will combine]
        def split [define how this attribute will be passed when the element is split]
        def _infer_value [define how to calculate the value if not set]

    vert = Vert()
    vert.set_attrib(MyAttrib('value'))
    assert vert.get_attrib(MyAttrib).value == 'value'

These attributes are held in an instance attribute dict, `attrib` keyed to the class
name of the attribute.

    assert vert.get_attrib(MyAttrib).value == 'value'
    assert vert.attrib['MyAttrib'].value == 'value'

Rules governing combination of these properties are defined in the Attrib classes
themselves.

There is a base class, Attrib, here, plus some subclasses modelling common cases. Do
not use these classes directly. Instead, subclass one of these classes for each
attribute you need to define.

:author: Shay Hill
:created: 2022-06-14
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from paragraphs import par

if TYPE_CHECKING:
    from halfedge.half_edge_elements import MeshElementBase

_T = TypeVar("_T")


class Attrib(Generic[_T]):
    """Base class for element attributes.

    MeshElementBase has methods set_attrib and get_attrib that will store
    Attrib instances in the MeshElemenBase __dict__. The Attrib class
    defines how these attributes behave when mesh elements are merged and allows a
    value (e.g., edge length) to be inferred from the Attrib.element property
    when and if needed, allowing us to cache (and potentially never access) slow
    attributes.

    Do not overload `__init__` or `value`. For the most part, treat as an ABC with
    abstract methods `merge`, `split`, and `_infer_value`--although the base methods
    are marginally useful and instructive, so you will not need to overload both in
    every case.
    """

    __slots__ = ("_value", "element")

    def __new__(
        cls: type[_TAttrib],
        value: _T | None = None,
        element: MeshElementBase | None = None,
    ) -> _TAttrib:
        """Raise an exception if the attribute is not subclassed."""
        del value
        del element
        if cls is Attrib:
            msg = "Attrib is an abstract class and cannot be instantiated."
            raise TypeError(msg)
        return object.__new__(cls)

    def __init__(
        self, value: _T | None = None, element: MeshElementBase | None = None
    ) -> None:
        """Set value and element."""
        self._value = value
        self.element = element

    @property
    def value(self) -> _T:
        """Return value if set, else try to infer a value.

        :return: Value of the attribute
        :raises AttributeError: If no value is set and _infer_value fails
        """
        if self._value is not None:
            return self._value
        with suppress(NotImplementedError, ValueError):
            value = self._infer_value()
            self._value = value
            return self._value
        msg = f"no value set and failed to infer from {self.element}"
        raise AttributeError(msg)

    def copy_to_element(self: Attrib[_T], element: MeshElementBase) -> Attrib[_T]:
        """Return a new instance with the same value, assigned to a new element.

        :param element: New element
        :return: Attrib instance
        """
        return type(self)(self._value, element)

    @classmethod
    def merge(cls, *merge_from: _TAttrib | None) -> _TAttrib | None:
        """Get value of self from self._merge_from.

        :param merge_from: Attrib instances to merge (all of the same class)
        :return: Attrib instance or None

        Attrib instance with merged value or None. It is fine to return one of the
        merge_from arguments if it represents what a new merged element should be.
        Eventually, it will be passed through MeshElementBase.set_attrib, which will
        *copy* the Attrib instance to the new element.

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

    def split(self: _TAttrib) -> _TAttrib | None:
        """Define how attribute will be passed when dividing self.element.

        :return: Attrib instance or None

        Attrib instance to be set on any element created by dividing and element with
        this attribute. It is fine to return one of the merge_from arguments if it
        represents what a new merged element should be.  Eventually, it will be
        passed through MeshElementBase.set_attrib, which will *copy* the Attrib
        instance to the new element.

        When an element is divided (face divided by an edge, edge divided by a vert,
        etc.) or altered, define how, if at all, this attribute will be passed to the
        altered element or pieces of the divided element. If a face with a color is
        divided, you might want to give the divided pieces the same color. If an
        attribute is lazy (e.g., edge norm), you might want to unset _value for each
        piece of a split edge.

        This base method will not pass an attribute when dividing or altering.
        """
        return None

    def _infer_value(self) -> _T:
        """Get value of self from self._element.

        Use the containing element to determine a value for self. If no value can be
        determined, return None.

        The purpose is to allow lazy attributes like edge norm and face area. Use
        caution, however. These need to be calculated before merging since the method
        may not support the new shape. For instance, this method might calculate the
        area of a triangle, but would fail if two triangles were merged into a
        square. To keep this safe, the _value is calculated *before* any merging. In
        the "area of a triangle" example,

            * The area calculation is deferred until the first merge.
            * At the first merge, the area of each merged triangle is calculated. The
              implication here is that calculation *cannot* be deferred till after a
              merge.
            * The merged method areas of the merged triangles at the first and
              subsequent mergers, so further triangle area calculations (which
              wouldn't work on the merged shapes anyway) are not required.

        If you infer a value, cache it by setting self._value.

        If you do not intend to infer values, raise an exception. This exception
        should occur *before* an AttributeError is raised for a potentially missing
        element attribute. It should be clear that _infer_value failed because there
        is no provision for inferring this Attrib.value, *not* because the
        user failed to set the Attrib property attribute.
        """
        msg = par(
            f"""'{type(self).__name__}' has no provision for inferring a value from
            'self.element'"""
        )
        raise NotImplementedError(msg)


_TAttrib = TypeVar("_TAttrib", bound=Attrib[Any])


class ContagionAttrib(Attrib[Literal[True]]):
    """Spread value when combining with anything.

    This is for element properties like 'IsHole' that are always passed when combining
    elements. The value of the attribute is always True. If any element in a group of
    to-be-merged elements has a ContagionAttributeBase attribute, then the merged
    element will have that attribute.

    The value is always True, even if something else is passed to __init__.
    """

    def __new__(
        cls: type[_TAttrib],
        value: Literal[True] | None = None,
        element: MeshElementBase | None = None,
    ) -> _TAttrib:
        """Raise an exception if the attribute is not subclassed."""
        del value
        del element
        if cls is ContagionAttrib:
            msg = "ContagionAttrib is an abstract class and cannot be instantiated."
            raise TypeError(msg)
        return object.__new__(cls)

    def __init__(
        self, value: Literal[True] | None = None, element: MeshElementBase | None = None
    ) -> None:
        """Set value and element."""
        super().__init__(value or True, element)

    @classmethod
    def merge(cls, *merge_from: _TAttrib | None) -> _TAttrib | None:
        """Merge values.

        :param merge_from: Attrib instances to merge (all of the same class)
        :return: self if any element has a ContagionAttributeBase attribute.

        If any element has a ContagionAttributeBase attribute, return a new instance
        with that attribute. Otherwise None.
        """
        attribs = [x for x in merge_from if x is not None]
        if attribs:
            return attribs[0]
        return None

    def split(self: _TAttrib) -> _TAttrib | None:
        """Copy attribute to splits.

        :return: self

        Holes are defined with IsHole(ContagionAttributeBase), so this will split a
        non-face hole into two non-face holes and a hole (is_face == True) into two
        holes.
        """
        return self


class IncompatibleAttrib(Attrib[_T]):
    """Keep value when all merge_from values are the same.

    This class in intended for flags like IsEdge or Hardness.
    """

    def __new__(
        cls: type[_TAttrib],
        value: _T | None = None,
        element: MeshElementBase | None = None,
    ) -> _TAttrib:
        """Raise an exception if the attribute is not subclassed."""
        del value
        del element
        if cls is IncompatibleAttrib:
            msg = "IncompatibleAttrib is an abstract class and cannot be instantiated."
            raise TypeError(msg)
        return object.__new__(cls)

    @classmethod
    def merge(cls, *merge_from: _TAttrib | None) -> _TAttrib | None:
        """Merge values.

        :param merge_from: Attrib instances to merge (all of the same class)
        :return: self if all values match and every contributing element has an analog.

        If all values match and every contributing element has an analog, return
        a new instance with that value. Otherwise None.
        """
        if not merge_from or merge_from[0] is None:
            return None

        first_value = merge_from[0].value
        for x in merge_from[1:]:
            if x is None or x.value != first_value:
                return None
        return merge_from[0]

    def split(self: _TAttrib) -> _TAttrib | None:
        """Pass the value on.

        :return: self
        """
        return self


class NumericAttrib(Attrib[_T]):
    """Average merge_from values."""

    def __new__(
        cls: type[_TAttrib],
        value: _T | None = None,
        element: MeshElementBase | None = None,
    ) -> _TAttrib:
        """Raise an exception if the attribute is not subclassed."""
        del value
        del element
        if cls is NumericAttrib:
            msg = "NumericAttrib is an abstract class and cannot be instantiated."
            raise TypeError(msg)
        return object.__new__(cls)

    @classmethod
    def merge(cls, *merge_from: _TAttrib | None) -> _TAttrib | None:
        """Average values if every contributor has a value. Otherwise None.

        :param merge_from: Attrib instances to merge (all of the same class)
        :return: Attrib instance with merged value or None
        """
        have_values = [x for x in merge_from if x is not None]
        if not have_values:
            return None
        values = [x.value for x in have_values]
        return type(have_values[0])(sum(values) / len(values))
