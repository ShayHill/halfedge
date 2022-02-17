#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Test inheritance with new vert, edge, face, and hole types

:author: Shay Hill
:created: 12/1/2021
"""
from typing import Tuple, Union

from ..halfedge.half_edge_elements import Vert, Edge, Face


# TODO: delete this module
# class TestNewElement:
#     def test_new_element_types(self) -> None:
#         """"""
#         MyVert = element_type("MyVert", Vert, {"coordinate": str})
# def element_type(
#     name: str, base: MeshElementBase, *new_attributes: Union[str, Tuple[str, object]]
# ) -> MeshElementBase:
#     """
#     # TODO: complete docstring
#     Inherit from parent and add attributes
#
#     :param name:
#     :param parent:
#     :param attributes:
#     :return:
#     """
#
#     class MyVert(Vert):
#         coordinate: Tuple[float, ...]
#
#     class MyHalfEdges(HalfEdges):
#         vert_type = MyVert
#
#     vert = MyHalfEdges().new_vert()
