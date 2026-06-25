"""
Air annotation system – draw in 3-D space projected onto the camera overlay.

Features:
  • Multi-colour brush via peace-sign colour cycle
  • Pinch / point to draw
  • Fist eraser mode
  • Undo stack (swipe down or keyboard)
  • Save annotations as PNG
  • Laser pointer mode with animated trail
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

import settings


@dataclass
class Stroke:
    """One continuous drawing stroke."""

    points: List[Tuple[int, int]] = field(default_factory=list)
    color: Tuple[int, int, int] = (255, 60, 60)
    size: int = settings.ANNOTATION_BRUSH_SIZE
    eraser: bool = False


@dataclass
class AnnotationManager:
    """
    Manages transparent annotation layer composited over the webcam feed.

    Supports drawing, erasing, undo, colour cycling, and laser pointer.
    """

    width: int = settings.PROCESS_WIDTH
    height: int = settings.PROCESS_HEIGHT
    color_index: int = settings.DEFAULT_ANNOTATION_COLOR_INDEX
    brush_size: int = settings.ANNOTATION_BRUSH_SIZE

    _layer: np.ndarray = field(init=False, repr=False)
    _strokes: List[Stroke] = field(default_factory=list, init=False)
    _undo_stack: List[List[Stroke]] = field(default_factory=list, init=False)
    _active_stroke: Optional[Stroke] = field(default=None, init=False)
    _last_point: Optional[Tuple[int, int]] = field(default=None, init=False)

    # Laser pointer trail
    _laser_trail: Deque[Tuple[int, int]] = field(
        default_factory=lambda: deque(maxlen=settings.LASER_TRAIL_LENGTH), init=False
    )
    _laser_phase: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._reset_layer()

    @property
    def brush_color(self) -> Tuple[int, int, int]:
        return settings.ANNOTATION_COLORS[self.color_index % len(settings.ANNOTATION_COLORS)]

    def resize(self, width: int, height: int) -> None:
        """Reallocate layer when frame size changes."""
        if width == self.width and height == self.height:
            return
        self.width = width
        self.height = height
        self._reset_layer()
        self.clear()

    def _reset_layer(self) -> None:
        self._layer = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def _push_undo(self) -> None:
        """Snapshot current strokes for undo."""
        snapshot = [
            Stroke(
                points=list(s.points),
                color=s.color,
                size=s.size,
                eraser=s.eraser,
            )
            for s in self._strokes
        ]
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > settings.MAX_UNDO_STEPS:
            self._undo_stack.pop(0)

    def start_stroke(self, point: Tuple[int, int], eraser: bool = False) -> None:
        """Begin a new stroke at the given pixel."""
        self._active_stroke = Stroke(
            points=[point],
            color=self.brush_color,
            size=self.brush_size * (2 if eraser else 1),
            eraser=eraser,
        )
        self._last_point = point

    def continue_stroke(self, point: Tuple[int, int]) -> None:
        """Extend the active stroke with smoothing."""
        if self._active_stroke is None:
            self.start_stroke(point)
            return

        if self._last_point is not None:
            alpha = 1.0 - settings.ANNOTATION_SMOOTHING
            sx = int(self._last_point[0] + alpha * (point[0] - self._last_point[0]))
            sy = int(self._last_point[1] + alpha * (point[1] - self._last_point[1]))
            point = (sx, sy)

        self._active_stroke.points.append(point)
        self._last_point = point
        self._draw_segment(self._active_stroke, point)

    def end_stroke(self) -> None:
        """Finalize the active stroke and push to history."""
        if self._active_stroke is not None and len(self._active_stroke.points) > 0:
            self._push_undo()
            self._strokes.append(self._active_stroke)
        self._active_stroke = None
        self._last_point = None

    def _draw_segment(self, stroke: Stroke, point: Tuple[int, int]) -> None:
        """Draw the latest segment onto the layer."""
        if len(stroke.points) < 2:
            cv2.circle(self._layer, point, stroke.size, stroke.color, -1, cv2.LINE_AA)
            return

        p1 = stroke.points[-2]
        p2 = point
        if stroke.eraser:
            cv2.line(self._layer, p1, p2, (0, 0, 0), stroke.size * 2, cv2.LINE_AA)
        else:
            cv2.line(self._layer, p1, p2, stroke.color, stroke.size, cv2.LINE_AA)

    def undo(self) -> bool:
        """Restore previous stroke state."""
        if not self._undo_stack:
            return False
        self._strokes = self._undo_stack.pop()
        self._rebuild_layer()
        return True

    def clear(self) -> None:
        """Remove all annotations."""
        self._push_undo()
        self._strokes.clear()
        self._reset_layer()
        self.end_stroke()

    def cycle_color(self) -> Tuple[int, int, int]:
        """Advance to the next brush colour."""
        self.color_index = (self.color_index + 1) % len(settings.ANNOTATION_COLORS)
        return self.brush_color

    def set_color_index(self, index: int) -> None:
        self.color_index = index % len(settings.ANNOTATION_COLORS)

    def _rebuild_layer(self) -> None:
        """Redraw all strokes from scratch (after undo)."""
        self._reset_layer()
        for stroke in self._strokes:
            for i in range(1, len(stroke.points)):
                if stroke.eraser:
                    cv2.line(
                        self._layer,
                        stroke.points[i - 1],
                        stroke.points[i],
                        (0, 0, 0),
                        stroke.size * 2,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.line(
                        self._layer,
                        stroke.points[i - 1],
                        stroke.points[i],
                        stroke.color,
                        stroke.size,
                        cv2.LINE_AA,
                    )

    def composite(self, frame: np.ndarray) -> np.ndarray:
        """Blend annotation layer onto the camera frame."""
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            self.resize(w, h)

        # Only copy non-black pixels for performance
        mask = cv2.cvtColor(self._layer, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        fg = cv2.bitwise_and(self._layer, self._layer, mask=mask)
        return cv2.add(bg, fg)

    def update_laser(self, point: Optional[Tuple[int, int]]) -> None:
        """Track index finger position for laser pointer overlay."""
        if point is not None:
            self._laser_trail.append(point)
        self._laser_phase += settings.LASER_PULSE_SPEED * 0.016

    def draw_laser(self, frame: np.ndarray) -> None:
        """Render animated red laser pointer on the frame."""
        if not self._laser_trail:
            return

        pulse = 0.6 + 0.4 * abs(np.sin(self._laser_phase))
        radius = int(settings.LASER_RADIUS * pulse)

        # Draw trail fading outward
        trail = list(self._laser_trail)
        for i, (x, y) in enumerate(trail):
            alpha = (i + 1) / len(trail)
            r = max(2, int(radius * alpha * 0.6))
            color = (
                int(settings.LASER_COLOR[0] * alpha),
                int(settings.LASER_COLOR[1] * alpha * 0.3),
                int(settings.LASER_COLOR[2] * alpha * 0.3),
            )
            cv2.circle(frame, (x, y), r, color, -1, cv2.LINE_AA)

        # Bright center dot
        cx, cy = trail[-1]
        cv2.circle(frame, (cx, cy), radius, settings.LASER_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), radius + 3, (255, 255, 255), 2, cv2.LINE_AA)

    def save(self) -> str:
        """Save the annotation layer to a timestamped PNG."""
        path = settings.ANNOTATIONS_DIR / f"annotation_{datetime.now():%Y%m%d_%H%M%S}.png"
        cv2.imwrite(str(path), self._layer)
        return str(path)

    def get_layer(self) -> np.ndarray:
        """Return a copy of the raw annotation layer."""
        return self._layer.copy()
