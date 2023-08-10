### Image loading

import pyvips

image = pyvips.Image.new_from_file("input.png")

# Remove alpha
if image.hasalpha():
    image = image.flatten(background=[255])

# Convert to black and white.
image = image.colourspace("b-w")

# Apply the threshold. Any pixels that are darker than 50% (less than 127) are
# considered filled areas, any pixels lighter than that are considered
# background. This threshold can be adjusted and inverted as needed.
image = image < 127

### Image tracing

import potracecffi

bitmap = image.numpy()

trace_result = potracecffi.trace(bitmap)

### Bezier approximation

import math
import numpy

# Type alias for a point
point = tuple[float, float]


def bezier_to_points(p1: point, p2: point, p3: point, p4: point, segments: int = 10):
    for t in numpy.linspace(0, 1, num=segments):
        x = (
            p1[0] * math.pow(1 - t, 3)
            + 3 * p2[0] * math.pow(1 - t, 2) * t
            + 3 * p3[0] * (1 - t) * math.pow(t, 2)
            + p4[0] * math.pow(t, 3)
        )
        y = (
            p1[1] * math.pow(1 - t, 3)
            + 3 * p2[1] * math.pow(1 - t, 2) * t
            + 3 * p3[1] * (1 - t) * math.pow(t, 2)
            + p4[1] * math.pow(t, 3)
        )
        yield (x, y)


### Polygon conversion

import gdstk

# A list that contains lists where the first entry is a polygon and
# any subsequent entries in the list are holes in the polygon.
polygons_and_holes: list[list[gdstk.Polygon]] = []

# Go through each path and pull out polygons and holes
for path in potracecffi.iter_paths(trace_result):

    # Go through each segment in the path and put together a list of points
    # that make up the polygon/hole.
    points = [potracecffi.curve_start_point(path.curve)]
    for segment in potracecffi.iter_curve(path.curve):

        # Corner segments are simple lines from c1 to c2
        if segment.tag == potracecffi.CORNER:
            points.append(segment.c1)
            points.append(segment.c2)

        # Curveto segments are cubic bezier curves
        if segment.tag == potracecffi.CURVETO:
            points.extend(
                list(
                    bezier_to_points(
                        points[-1],
                        segment.c0,
                        segment.c1,
                        segment.c2,
                    )
                )
            )

    polygon = gdstk.Polygon(points)

    # Check the sign of the path, + means its a polygon and - means its a hole.
    if path.sign == ord("+"):
        # If it's a polygon, insert a new list with the polygon.
        polygons_and_holes.append([polygon])
    else:
        # If it's a hole, append it to the last polygon's list
        polygons_and_holes[-1].append(polygon)


# Now take the list of polygons and holes and simplify them into a final list
# of simple polygons using boolean operations.
polygons: list[gdstk.Polygon] = []

for polygon, *holes in polygons_and_holes:
    # This polygon has no holes, so it's ready to go
    if not holes:
        polygons.append(polygon)
        continue

    # Use boolean "not" to subtract all of the holes from the polygon.
    results: list[gdstk.Polygon] = gdstk.boolean(polygon, holes, "not")

    # Gdstk will return more than one polygon if the result can not be
    # represented with a simple polygon, so extend the list with the results.
    polygons.extend(results)

### Footprint generation

dots_per_inch = 300
dots_per_millimeter = 25.4 / dots_per_inch


def fp_poly(points: list[point]) -> str:
    points_mm = (
        (x * dots_per_millimeter, y * dots_per_millimeter) for (x, y) in points
    )
    points_sexpr = "\n".join((f"(xy {x:.4f} {y:.4f})" for (x, y) in points_mm))
    return f"""
    (fp_poly
        (pts {points_sexpr})
        (layer "F.SilkS")
        (width 0)
        (fill solid)
        (tstamp "7a7d51f6-24ac-11ed-8354-7a0c86e760e0")
    )
    """


poly_sexprs = "\n".join(fp_poly(polygon.points) for polygon in polygons)

footprint = f"""
(footprint "Library:Name"
  (layer "F.SilkS")
  (at 0 0)
  (attr board_only exclude_from_pos_files exclude_from_bom)
  (tstamp "7a7d5548-24ac-11ed-8354-7a0c86e760e0")
  (tedit "7a7d5552-24ac-11ed-8354-7a0c86e760e0")
  {poly_sexprs}
)
"""

import pathlib

pathlib.Path("footprint.kicad_mod").write_text(footprint)
