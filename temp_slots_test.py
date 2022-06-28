
from sys import getsizeof
from dataclasses import dataclass
from typing import Dict, Any

def init(self):
    self.a = 1
    self.b = 2
    self.c = 3
    self.d = 4
    self.e = 5
    self.f = 6

class FullSlots:
    __slots__ = 'a', 'b', 'c', 'd', 'e', 'f'
    __init__ = init

class HalfSlots:
    __slots__ = 'a', 'b', 'c', 'd', 'e', '__dict__'
    __init__ = init


class No__Slots:
    __init__ = init

class DC__Slots:
    __slots__ = 'a', 'b', 'c', 'd', 'e', 'f', '__dict__'
    a: int
    b: int
    c: int
    d: int
    e: int
    f: int
    __dict__: Dict[str, Any]

    __init__ = init

class DC2(DC__Slots):
    __slots__ = 'g'

breakpoint()


import sys
from numbers import Number
from collections import deque
from collections.abc import Set, Mapping


ZERO_DEPTH_BASES = (str, bytes, Number, range, bytearray)


def getsize(obj_0):
    """Recursively iterate to sum size of object & members."""
    _seen_ids = set()
    def inner(obj):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, ZERO_DEPTH_BASES):
            pass # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, 'items'):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, 'items')())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size
    return inner(obj_0)

for type_ in (FullSlots, HalfSlots, No__Slots, DC__Slots):
    instances = [type_() for i in range(10000)]
    print(f'{type_.__name__} => {getsize(instances)}')
