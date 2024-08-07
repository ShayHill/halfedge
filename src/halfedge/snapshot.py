"""Take a picture of mesh elements for debugging.

:author: Shay Hill
:created: 11/1/2020
"""

# from svg_ultralight import new_svg_root, write_svg
# from svg_ultralight.constructors import new_sub_element
# from svg_ultralight.strings import svg_color_tuple, svg_float_tuples

# from .half_edge_elements import Edge, Face, Vert


# class Snapshot:
#     """A picture of the mesh elements for debugging."""

#     def __init__(self):
#         """Create a snapshot."""
#         self.bbox = [0, 0, 1, 1]
#         self.verts = []
#         self.edges = []
#         self.faces = []

#     def segregate_elements(self, elements):
#         """TODO: Docstring for segregate_elements."""
#         verts = {x for x in elements if isinstance(x, Vert)}
#         edges = {x for x in elements if isinstance(x, Edge)}
#         faces = {x for x in elements if isinstance(x, Face)}
#         for face in faces:
#             edges.update(set(face.edges))
#         for edge in edges:
#             verts.update({edge.orig, edge.dest})
#         return verts, edges, faces

#     def add_svg_elements(self, elements, color=None):
#         """TODO: Docstring for add_svg_elements."""
#         verts, edges, faces = self.segregate_elements(elements)
#         coordinates = [x.coordinate for x in verts]

#         self.bbox[0] = min(self.bbox[0], min(x[0] for x in coordinates))
#         self.bbox[1] = min(self.bbox[1], min(x[1] for x in coordinates))
#         self.bbox[2] = max(self.bbox[2], max(x[0] for x in coordinates))
#         self.bbox[3] = max(self.bbox[3], max(x[1] for x in coordinates))

#         for vert in verts:
#             coord = vert.coordinate
#             color_ = color if color and vert in elements else (0, 0, 0)
#             self.verts.append((coord, color_))
#         for edge in edges:
#             coords = (edge.orig.coordinate, edge.dest.coordinate)
#             color_ = color if color and edge in elements else (100, 100, 100)
#             self.edges.append(coords + (color_,))
#         for face in faces:
#             coords = tuple(x.coordinate for x in face.verts)
#             color_ = color if color and face in elements else (200, 200, 200)
#             self.faces.append((coords, color_))

#     def draw(self, filename="snapshot"):
#         """TODO: Docstring for draw."""
#         coordinates = {tuple(x[0]) for x in self.verts}
#         bbox = [
#             min(x for x, y in coordinates),
#             min(y for x, y in coordinates),
#             max(x for x, y in coordinates),
#             max(y for x, y in coordinates),
#         ]
#         width = bbox[2] - bbox[0]
#         height = bbox[3] - bbox[1]
#         scale = min(1000 / width, 1000 / height)
#         screen = new_svg_root(
#             bbox[0], bbox[1], width, height, pad_=15 / scale, dpu_=scale
#         )
#         circle_rad = max(0.08, 3 / scale)
#         line_stroke = max(0.03, 2 / scale)
#         for coos, col in self.faces:
#             new_sub_element(
#                 screen,
#                 "polygon",
#                 points=svg_float_tuples(coos),
#                 fill=svg_color_tuple(col),
#             )
#         for coo_a, coo_b, col in self.edges:
#             new_sub_element(
#                 screen,
#                 "line",
#                 x1=coo_a[0],
#                 y1=coo_a[1],
#                 x2=coo_b[0],
#                 y2=coo_b[1],
#                 stroke_width=line_stroke,
#                 stroke=svg_color_tuple(col),
#             )
#         for coo, col in self.verts:
#             new_sub_element(
#                 screen,
#                 "circle",
#                 cx=coo[0],
#                 cy=coo[1],
#                 r=circle_rad,
#                 fill=svg_color_tuple(col),
#             )

#         write_svg(filename + ".svg", screen)
