# A typical halfedges data structure with some padding

## Use

You will most likely want to create meshes with the `from_vlfi` constructor. It's got a few tricks, so be sure to read the docstring.

```python
from halfedges import HalfEdges, Vert

vertices = [Vert() for _ in range(4)]
face_indices: list[tuple[int, ...]] = [(0, 1, 2), (0, 2, 3)]

mesh = HalfEdges.from_vlfi(vertices, face_indices)

edge = next(iter(mesh.edges))
mesh.remove_edge(edge)
```

## Particulars

The idea (for 19 years and counting) has been to create an interface that is neither too complex, too verbose, nor too magical. I've been all over the place as to where that line should be. This is my current thinking.

### Reflection

If you set `edge_a.orig = vert_a`, then the `Edge.vert` setter will *automagically* set `vert_a.edge = edge_a`. This is true for any setter that might otherwise break the mesh. Sometimes, this reflection will happen when it isn't strictly necessary. Imagine you have `face_a` with three edges: `edge_a`, `edge_b` and `edge_c`. `face_a` has a pointer to `edge_a`, but it could point to any of the three edges and still be correct per the requirements of the halfedge data structure. If you directly set `edge_b.face = face_a`, everything would still be correct (`face_a.edge` would still be `edge_a` and that would still be correct), but the `Edge.edge` setter will nevertheless "reflectively" set `face_a.edge = edge_b`.

### id

Each Vert, Edge, and Face instance has an `id` attribute. This is a unique, sequential identifier for the instance. Sorted vert, edge, and face lists (`HalfEdges.vl`, `HalfEdges.el`, `HalfEdges.fl`) are sorted by `id`. All Vert, Edge, and Face instances across all HalfEdges instances will share the id counter, so don't expect the id of your verts to start at 0.

### Holes

A triangle mesh in 2D will never be entirely triangular (and also manifold). There will be a boundary around the triangles. This library treats the outside of that boundary as a face (with an `is_hole == True` attribute). This way, every edge has a pair and a face, and the `is_hole` flags can be used to keep these hole faces out of your way.

Holes are useful for more than just 2D mesh boundaries, you can explicitly create holes (`Face` instances with an as `is_hole == True` property) to maintain manifold mesh conditions in many circumstances. The constructor `HalfEdges.from_vlfi()` will try to insert hole faces to maintain manifold conditions.

Four main types: Vert, Edge, Face, and HalfEdges. Vert and Face instances have `*.faces` properties which will return all adjacent faces. These properties will *not* return faces flagged as holes. The holes are there to keep things simple and manifold, but otherwise stay out of your way.

```python
Face(
    *attributes: Attrib[Any],
    mesh: BlindHalfEdges | None = None,
    edge: Edge | None = None,
    is_hole: bool = False,
) -> None:
```

The `is_hole` `__init__` kwarg is shorthand for

```python
class IsHole(ContagionAttribute):
    pass

face = Face()
face.add_attrib(IsHole())
```


The `face_instance.is_hole` property getter is shorthand for

```python
vert.get_attrib(IsHole())
```

More on `Attrib` classes below and in `type_attrib.py`.

### Element Attributes

By halfedge convention

- each Vert instance holds a pointer to one Edge instance;
- each Face instance holds a pointer to one Edge instance; and
- each Edge instance holds four pointers (orig, pair, face, next).

These describe the geometry of a mesh, but there may be other attributes you would like to assign to these instances. For example, each Face instance might have a color. There is no objectively correct way to define a face color, nor to merge colors when two faces are merged, nor to split a color when faces are split. Do a red and a blue face behave like paint and combine to make a purple face? Or do they behave like DNA to make a red *or* blue face depending on which is dominant? This library will not guess.

For each such attribute, you will need to define `merge` and `split` methods to explicate how the attribute 1) combines when elements are merged; and 2) behaves when an element is split. Do this by creating a new descendent of `Attrib` or one of `Attrib`'s children defined in `type_attrib.py`. See the docstring in that file for more information.

```python
# `Vector2Attrib` is a child of `Attrib` defined in `type_attrib.py`.
# It defines suitable (YMMV) `merge` (average) and `split` (fail) methods for
# an (x, y) coordinate.
class Coordinate(Vector2Attrib):
    pass

vert = Vert()
vert.add_attrib(Coordinate((1, 2)))
assert vert.get_attrib(Coordinate).value == (1, 2)
```

You cannot assign or access these attributes with `vert.attribute`. Instead assign with `vert.add_attrib(attrib_instance)`. Retrieve the value with `vert.get_attrib(attrib_class)`. Everything will be keyed to the class name, so you will need a new ElemAttribBase descendant for each attribute type.

These element attributes can also be passed at `__init__`

```python
vert = Vert(Coordinate(1, 2))
assert vert.get_attrib(Coordinate).value == (1, 2)
```

### You Should Know

A canonical half-edge data structure stores:

- a set of verts (redundant)
- for each vert, a pointer to an edge
- a set of edges
- for each edge, pointers to vert, pair, face, next
- a set of faces (redundant)
- for each face, a pointer to an edge

This implementation only stores a set of edges. Sets of verts and faces are generated by iterating through references in edge instances. This makes for slower code, but does not violate DRY and makes for dramatically cleaner code.

This means that verts and faces disappear from a mesh when the edges referencing them are removed.

Also means that this won't be useful for more than hundreds (maybe thousands) of edges. Good enough for my purposes.

## Project Structure

The HalfEdges class is defined across three modules:

### half_edge_constructors.py

class BlindHalfEdges defines the from_vlfi constructor and just enough methods to allow it to work. These methods define how pointers and Attrib instances are assigned.

### half_edge_querries.py

class StaticHalfEdges(BlindHalfEdges) defines all the look-up tricks of the halfedge data structure. What faces are adjacent to an edge? What edges radiate from a vert? etc.

### half_edge_object.py

class HalfEdges(StaticHalfEdges) defines all of the methods that change the structure of the mesh: insert_edge, collapse_edge, etc.
