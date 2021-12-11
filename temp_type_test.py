#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""

:author: Shay Hill
:created: 12/10/2021
"""
from typing import TypeVar, Generic

_T = TypeVar("_T")
_S = TypeVar("_S")


class Base(Generic[_T, _S]):
    def __init__(self, some_t: _T) -> None:
        self.some_t = some_t

    def get_some_t(self) -> _T:
        return self.some_t


class Child(Generic[_T, _S], Base[_T, _S]):
    pass


class GrandChild(Generic[_T, _S], Child[_T, _S]):
    pass


MyChild = GrandChild[int, int]


aaa = MyChild(3)

def fail():
    return alphabet

fail()

breakpoint()

bbb = MyChild("2")


