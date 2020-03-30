#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# last modified: 12/22/2018
"""MyPy types for half_edges.

created: 12/22/2018
"""

from typing import Union, Tuple, List

"""Coordinate input and output."""
Coordinate = Union[Tuple[float, float], Tuple[float, float, float]]

"""A list of vert coordinates."""
VertexList = List[Coordinate]

"""A list of vert coordinates indices."""
ListIndices = List[int]
