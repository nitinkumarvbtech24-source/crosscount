"""Counting-line geometry in pixel coordinates (y grows downward)."""
from typing import Optional, Sequence, Tuple

Point = Tuple[float, float]
Box = Sequence[float]  # x1, y1, x2, y2


def anchor(box: Box) -> Point:
    """Where a road user touches the ground: the bottom centre of its box.
    Counting this point (not the box centre) keeps a tall bus and a short
    scooter crossing the line at the same place on the road."""
    return ((box[0] + box[2]) / 2.0, float(box[3]))


def signed_distance(a: Point, b: Point, p: Point) -> float:
    """Distance of p from the infinite line a->b, positive on one side.
    With a->b pointing right on screen, positive is below the line."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = (dx * dx + dy * dy) ** 0.5 or 1.0
    return (dx * (p[1] - a[1]) - dy * (p[0] - a[0])) / length


def along(a: Point, b: Point, p: Point) -> float:
    """Position of p's projection on a->b: 0 at a, 1 at b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    return ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / ((dx * dx + dy * dy) or 1.0)


def crossing_point(a: Point, b: Point, p0: Point, p1: Point) -> Optional[Point]:
    """Where the step p0->p1 meets the line through a and b, or None when
    both points are on the same side."""
    d0, d1 = signed_distance(a, b, p0), signed_distance(a, b, p1)
    if d0 == d1 or (d0 > 0) == (d1 > 0):
        return None
    u = d0 / (d0 - d1)
    return (p0[0] + u * (p1[0] - p0[0]), p0[1] + u * (p1[1] - p0[1]))


def iou(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def overlap_share(inner: Box, outer: Box) -> float:
    """Share of `inner`'s area that lies inside `outer`."""
    w = min(inner[2], outer[2]) - max(inner[0], outer[0])
    h = min(inner[3], outer[3]) - max(inner[1], outer[1])
    if w <= 0 or h <= 0:
        return 0.0
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return w * h / area
