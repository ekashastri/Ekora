"""
Shape recognizer module – Ekora v1.4.

Analyses a completed freehand stroke and attempts to classify it as one of:
    circle, rectangle, square, triangle, line, arrow

Returns a (shape_type, confidence, params) tuple.
confidence < CONFIDENCE_THRESHOLD → keep original stroke unchanged.

Recognition runs ONLY after stroke completion; it never interrupts drawing.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Minimum confidence to accept a recognition result (0‒1).
CONFIDENCE_THRESHOLD = 0.72

# Minimum points in a stroke before we attempt recognition.
MIN_STROKE_POINTS = 10

ShapeType = str  # "circle" | "rectangle" | "square" | "triangle" | "line" | "arrow" | None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recognize(
    points: List[Tuple[int, int]],
) -> Tuple[Optional[ShapeType], float, dict]:
    """
    Analyse *points* (the pixel path of a completed stroke) and return
    (shape_type, confidence, params).

    params keys depend on shape_type:
        circle    : cx, cy, radius
        rectangle : x, y, w, h
        square    : x, y, side
        triangle  : pts  (list of 3 (x,y))
        line      : p0, p1
        arrow     : p0, p1  (p1 is the arrow head)
    """
    if len(points) < MIN_STROKE_POINTS:
        return None, 0.0, {}

    pts = np.array(points, dtype=np.float32)

    results = [
        _try_line(pts),
        _try_arrow(pts),
        _try_circle(pts),
        _try_ellipse(pts),
        _try_rectangle(pts),   # also covers square
        _try_triangle(pts),
    ]

    best_shape, best_conf, best_params = None, 0.0, {}
    for shape, conf, params in results:
        if conf > best_conf:
            best_shape, best_conf, best_params = shape, conf, params

    if best_conf >= CONFIDENCE_THRESHOLD:
        return best_shape, best_conf, best_params
    return None, best_conf, {}


# ---------------------------------------------------------------------------
# Individual shape testers
# ---------------------------------------------------------------------------

def _try_line(pts: np.ndarray) -> Tuple[ShapeType, float, dict]:
    """
    A stroke is a line if it is essentially straight (low aspect deviation)
    and does NOT close back on itself.
    """
    p0 = pts[0]
    p1 = pts[-1]
    chord = np.linalg.norm(p1 - p0)
    if chord < 10:
        return "line", 0.0, {}

    # Max perpendicular distance from all points to the chord line p0→p1
    direction = (p1 - p0) / chord
    perp = pts - p0
    # Scalar projection onto perpendicular
    cross = np.abs(perp[:, 0] * direction[1] - perp[:, 1] * direction[0])
    max_dev = cross.max()
    # Normalise by chord length
    linearity = 1.0 - min(1.0, (max_dev / chord) * 4)

    # Penalise if the endpoints are close (closed shape)
    close_penalty = np.linalg.norm(p0 - p1) / max(1, _stroke_length(pts))
    if close_penalty < 0.15:
        linearity *= 0.2

    return "line", float(linearity), {
        "p0": tuple(pts[0].astype(int)),
        "p1": tuple(pts[-1].astype(int)),
    }


def _try_arrow(pts: np.ndarray) -> Tuple[ShapeType, float, dict]:
    """
    Arrow = straight shaft + a small V-notch at one end (the head).
    Strategy:
    1. Split the stroke at ~80% of its arc length.
    2. Check shaft linearity.
    3. Check that the last ~20% makes a sharp angle (the barbs).
    """
    n = len(pts)
    split = int(n * 0.80)
    if split < 5:
        return "arrow", 0.0, {}

    shaft = pts[:split]
    tail  = pts[split:]

    # Shaft must be straight
    _, shaft_conf, shaft_params = _try_line(shaft)
    if shaft_conf < 0.60:
        return "arrow", 0.0, {}

    # The full stroke must not close
    start_end_dist = np.linalg.norm(pts[0] - pts[-1])
    chord = np.linalg.norm(pts[0] - pts[split])
    if chord < 5:
        return "arrow", 0.0, {}
    if start_end_dist < chord * 0.25:
        return "arrow", 0.0, {}

    # Check for direction change in the tail (arrowhead barb)
    shaft_dir = shaft[-1] - shaft[0]
    shaft_dir /= max(1e-6, np.linalg.norm(shaft_dir))
    tail_dir  = tail[-1]  - tail[0]
    t_len = np.linalg.norm(tail_dir)
    if t_len < 2:
        return "arrow", 0.0, {}
    tail_dir /= t_len

    dot = np.dot(shaft_dir, tail_dir)
    # Arrowhead barb goes backward relative to shaft direction.
    # Score is high when the barb points somewhere between 90° and 180° from shaft.
    # dot close to -1 (perfectly backwards) or with a large perpendicular component.
    # We want: NOT the same direction (dot near +1 is bad).
    angle_score = max(0.0, (1.0 - dot) / 2.0)  # 0 when same dir, 1 when opposite

    conf = shaft_conf * 0.55 + angle_score * 0.45
    return "arrow", float(conf), {
        "p0": tuple(pts[0].astype(int)),
        "p1": tuple(pts[split].astype(int)),
    }


def _try_circle(pts: np.ndarray) -> Tuple[ShapeType, float, dict]:
    """
    Fit a minimum enclosing circle and measure how well all points lie on it.
    Also verify the stroke is roughly closed.
    """
    center, radius = cv2.minEnclosingCircle(pts.astype(np.float32).reshape(-1, 1, 2))
    cx, cy = center
    if radius < 5:
        return "circle", 0.0, {}

    # Radial deviation of each point from the fitted circle
    dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    mean_dist = dists.mean()
    radial_err = np.abs(dists - mean_dist).mean() / max(1, mean_dist)
    circularity = 1.0 - min(1.0, radial_err * 6)

    # Stroke must close (end near start)
    closure = np.linalg.norm(pts[0] - pts[-1]) / max(1, 2 * radius)
    closure_score = max(0.0, 1.0 - closure * 2)

    # Angular coverage: the stroke should sweep ~360° around the centre
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    angle_range = _angle_range(angles)
    coverage = min(1.0, angle_range / (1.6 * math.pi))

    conf = circularity * 0.45 + closure_score * 0.25 + coverage * 0.30

    return "circle", float(conf), {
        "cx": int(cx),
        "cy": int(cy),
        "radius": int(mean_dist),
    }



def _try_ellipse(pts: np.ndarray) -> Tuple[ShapeType, float, dict]:
    """
    Fit a rotated ellipse and measure how well the points match.
    Uses cv2.fitEllipse.
    """
    if len(pts) < 5:
        return "ellipse", 0.0, {}

    try:
        (cx, cy), (axes_a, axes_b), angle = cv2.fitEllipse(pts.astype(np.float32))
    except Exception:
        return "ellipse", 0.0, {}

    if axes_a < 5 or axes_b < 5:
        return "ellipse", 0.0, {}

    # To check how well the points fit, we transform points to the ellipse's local coordinate system.
    # Translate by (-cx, -cy), rotate by -angle
    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    
    # Rotate points
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    
    # Equation of ellipse: (x / (a/2))^2 + (y / (b/2))^2 = 1
    rx, ry = axes_a / 2, axes_b / 2
    
    # Calculate radial distances in transformed space
    # (x/rx)^2 + (y/ry)^2 should be close to 1
    # distance to center in a distorted space where ellipse is a circle of radius 1:
    dist_sq = (lx / rx)**2 + (ly / ry)**2
    dists = np.sqrt(dist_sq)
    
    # The expected distance is 1.0
    mean_dist = dists.mean()
    radial_err = np.abs(dists - mean_dist).mean() / max(0.1, mean_dist)
    ellipticity = 1.0 - min(1.0, radial_err * 6)

    # Stroke must close (end near start)
    peri = math.pi * (3*(rx + ry) - math.sqrt((3*rx + ry)*(rx + 3*ry)))
    closure = np.linalg.norm(pts[0] - pts[-1]) / max(1, peri)
    closure_score = max(0.0, 1.0 - closure * 2)

    # Angular coverage: the stroke should sweep ~360° around the centre
    angles = np.arctan2(dy, dx)
    angle_range = _angle_range(angles)
    coverage = min(1.0, angle_range / (1.6 * math.pi))

    conf = ellipticity * 0.45 + closure_score * 0.25 + coverage * 0.30
    
    # If axes are very similar, it might be a circle. We penalise ellipse score if aspect ratio ~ 1
    aspect_ratio = min(rx, ry) / max(rx, ry)
    if aspect_ratio > 0.85:
        conf *= 0.8  # Let circle take over if it's really circular

    return "ellipse", float(conf), {
        "cx": int(cx),
        "cy": int(cy),
        "axes_a": int(axes_a),
        "axes_b": int(axes_b),
        "angle": float(angle),
    }

def _try_rectangle(pts: np.ndarray) -> Tuple[ShapeType, float, dict]:
    """
    Fit a rotated minimum-area bounding rectangle and score based on:
    - how much of the stroke lies on/near the four edges
    - closure (start ≈ end)
    - corner count (convex hull should have ~4 dominant corners)
    Returns either "square" or "rectangle".
    """
    hull = cv2.convexHull(pts.astype(np.int32))
    if hull is None or len(hull) < 3:
        return "rectangle", 0.0, {}

    # Approximate the hull polygon
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.04 * peri, True)

    corner_score = 0.0
    if len(approx) == 4:
        corner_score = 1.0
    elif len(approx) in (3, 5):
        corner_score = 0.4

    if corner_score < 0.3:
        return "rectangle", 0.0, {}

    # Min-area bounding rect
    rect = cv2.minAreaRect(pts.astype(np.float32))
    (_, _), (bw, bh), _ = rect
    if bw < 5 or bh < 5:
        return "rectangle", 0.0, {}

    # Distance from each point to the nearest rect edge
    box = cv2.boxPoints(rect).astype(np.float32)
    edge_dists = _dist_to_rect_edges(pts, box)
    edge_err = edge_dists.mean() / max(1, (bw + bh) / 2)
    edge_score = 1.0 - min(1.0, edge_err * 8)

    # Closure
    closure = np.linalg.norm(pts[0] - pts[-1]) / max(1, peri)
    closure_score = max(0.0, 1.0 - closure * 4)

    conf = corner_score * 0.40 + edge_score * 0.35 + closure_score * 0.25

    # Classify as square vs rectangle
    ratio = min(bw, bh) / max(bw, bh, 1)
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))

    if ratio > 0.82:
        side = int((bw + bh) / 2)
        cx_r = x + w // 2
        cy_r = y + h // 2
        half = side // 2
        return "square", float(conf), {
            "x": cx_r - half,
            "y": cy_r - half,
            "side": side,
        }
    else:
        return "rectangle", float(conf), {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        }


def _try_triangle(pts: np.ndarray) -> Tuple[ShapeType, float, dict]:
    """
    Convex hull approx with ~3 corners → triangle.
    """
    hull = cv2.convexHull(pts.astype(np.int32))
    if hull is None or len(hull) < 3:
        return "triangle", 0.0, {}

    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.05 * peri, True)

    if len(approx) != 3:
        return "triangle", 0.0, {}

    tri_pts = approx.reshape(3, 2)

    # Interior angles should all be < 170° (not degenerate)
    angles = _triangle_angles(tri_pts)
    if max(angles) > 170:
        return "triangle", 0.0, {}

    # How well do the stroke points lie near the three sides?
    edge_dists = _dist_to_polygon_edges(pts, tri_pts)
    edge_err = edge_dists.mean() / max(1, peri / 3)
    edge_score = 1.0 - min(1.0, edge_err * 6)

    # Closure
    closure = np.linalg.norm(pts[0] - pts[-1]) / max(1, peri)
    closure_score = max(0.0, 1.0 - closure * 4)

    conf = edge_score * 0.60 + closure_score * 0.40

    return "triangle", float(conf), {
        "pts": [tuple(p) for p in tri_pts.tolist()],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stroke_length(pts: np.ndarray) -> float:
    diffs = np.diff(pts, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def _angle_range(angles: np.ndarray) -> float:
    """Total angular arc swept by a sequence of angles (handles wrap-around)."""
    unwrapped = np.unwrap(angles)
    return float(abs(unwrapped[-1] - unwrapped[0]))


def _dist_point_to_segment(
    p: np.ndarray, a: np.ndarray, b: np.ndarray
) -> float:
    ab = b - a
    ap = p - a
    t  = np.clip(np.dot(ap, ab) / max(1e-6, np.dot(ab, ab)), 0, 1)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _dist_to_rect_edges(pts: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Min distance from each point to the four edges of a box (4×2)."""
    n_edges = len(box)
    all_dists = []
    for i in range(n_edges):
        a = box[i]
        b = box[(i + 1) % n_edges]
        segs = [_dist_point_to_segment(p, a, b) for p in pts]
        all_dists.append(segs)
    return np.min(all_dists, axis=0)


def _dist_to_polygon_edges(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    n_edges = len(poly)
    all_dists = []
    for i in range(n_edges):
        a = poly[i].astype(np.float32)
        b = poly[(i + 1) % n_edges].astype(np.float32)
        segs = [_dist_point_to_segment(p.astype(np.float32), a, b) for p in pts]
        all_dists.append(segs)
    return np.min(all_dists, axis=0)


def _triangle_angles(tri_pts: np.ndarray) -> List[float]:
    angles = []
    for i in range(3):
        a = tri_pts[(i - 1) % 3].astype(float)
        b = tri_pts[i].astype(float)
        c = tri_pts[(i + 1) % 3].astype(float)
        ba = a - b
        bc = c - b
        cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angles.append(math.degrees(math.acos(np.clip(cos_a, -1, 1))))
    return angles


# ---------------------------------------------------------------------------
# Canvas rendering helpers (used by DrawingCanvas)
# ---------------------------------------------------------------------------

def draw_clean_shape(
    canvas: "np.ndarray",
    shape_type: ShapeType,
    params: dict,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    """
    Render a clean vector shape onto *canvas* (in-place).
    Called by DrawingCanvas after recognition succeeds.
    """
    t = max(2, thickness)
    if shape_type == "circle":
        cv2.circle(canvas, (params["cx"], params["cy"]), params["radius"],
                   color, t, cv2.LINE_AA)
                   
    elif shape_type == "ellipse":
        center = (params["cx"], params["cy"])
        axes = (params["axes_a"] // 2, params["axes_b"] // 2)
        cv2.ellipse(canvas, center, axes, params["angle"], 0, 360,
                    color, t, cv2.LINE_AA)

    elif shape_type in ("rectangle", "square"):
        if shape_type == "square":
            x, y, w, h = params["x"], params["y"], params["side"], params["side"]
        else:
            x, y, w, h = params["x"], params["y"], params["w"], params["h"]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, t, cv2.LINE_AA)

    elif shape_type == "triangle":
        tri = np.array(params["pts"], dtype=np.int32)
        cv2.polylines(canvas, [tri], isClosed=True, color=color,
                      thickness=t, lineType=cv2.LINE_AA)

    elif shape_type == "line":
        cv2.line(canvas, params["p0"], params["p1"], color, t, cv2.LINE_AA)

    elif shape_type == "arrow":
        p0, p1 = params["p0"], params["p1"]
        cv2.arrowedLine(canvas, p0, p1, color, t,
                        cv2.LINE_AA, tipLength=0.25)
