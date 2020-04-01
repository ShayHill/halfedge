#!/usr/bin/env python3
# Last modified: 181126 16:07:44
""" half edge testing methods

created: 181121 13:14:06
"""

from typing import Any, List, Set, Tuple, cast

from halfedge import classes

def _canon_face_rep(face: classes.Face) -> List[Any]:
    """Canonical face representation: value tuples starting at min."""
    coordinates = [x.coordinate for x in face.verts]
    idx_min = coordinates.index(min(coordinates))
    return coordinates[idx_min:] + coordinates[:idx_min]


def _canon_he_rep(edges: Set[classes.Edge]) -> Tuple[List[Any], List[Any]]:
    """Canonical mesh representation [faces, holes].

    faces or holes = [canon_face_rep(face), ...]
    """
    faces = set(x.face for x in edges if not isinstance(x.face, classes.Hole))
    holes = set(x.face for x in edges if isinstance(x.face, classes.Hole))
    face_reps = cast(List[Any], [_canon_face_rep(x) for x in faces])
    hole_reps = cast(List[Any], [_canon_face_rep(x) for x in holes])

    return sorted(face_reps), sorted(hole_reps)


def are_equivalent_edges(
    edges_a: Set[classes.Edge], edges_b: Set[classes.Edge]
) -> bool:
    """Do edges lay on the same vert values with same geometry?"""
    faces_a, holes_a = _canon_he_rep(edges_a)
    faces_b, holes_b = _canon_he_rep(edges_b)

    are_same = len(faces_a) == len(faces_b) and len(holes_a) == len(holes_b)

    for a, b in zip(faces_a + holes_a, faces_b + holes_b):
        are_same = are_same and a == b

    return are_same


def are_equivalent_meshes(mesh_a: classes.HalfEdges, mesh_b: classes.HalfEdges) -> bool:
    """Do meshes lay on the same vert values with same geometry?"""
    return are_equivalent_edges(mesh_a.edges, mesh_b.edges)
