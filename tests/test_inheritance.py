#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""Test inheritance with new vert, edge, face, and hole types

:author: Shay Hill
:created: 12/1/2021
"""

from ..halfedge.half_edge_elements import Vert, Edge, Face, Hole


# TODO: delete this module
# class TestInheritance:
#     def test_new_element_types(self) -> None:
#         """"""
#         class MyVert(Vert):
#             def __init__(self, inherit_from, coordinate, **kwargs) -> None:
#         vert_type = type(
#             MyVert,
#             (vert,),
#         )
