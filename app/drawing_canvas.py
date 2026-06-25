"""
Drawing canvas module – virtual paint layer and stroke management.

The canvas is a separate numpy array the same size as the camera frame.
Drawing happens only inside the canvas region (excluding the sidebar).
Strokes are anti-aliased line segments between consecutive fingertip positions.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

import config


class DrawingCanvas:
    """
    Off-screen buffer that accumulates brush strokes.

    Attributes
    ----------
    width, height : canvas dimensions in pixels
    brush_color   : current BGR colour
    brush_size    : line thickness for draw mode
    eraser_size   : line thickness for eraser mode
    """

    def __init__(
        self,
        width: int = config.CANVAS_WIDTH,
        height: int = config.CANVAS_HEIGHT,
        brush_color: Tuple[int, int, int] = config.DEFAULT_BRUSH_COLOR,
        brush_size: int = config.DEFAULT_BRUSH_SIZE,
        eraser_size: int = config.ERASER_SIZE,
    ) -> None:
        self.width = width
        self.height = height
        self.brush_color = brush_color
        self.brush_size = brush_size
        self.eraser_size = eraser_size

        # Transparent-black canvas (black strokes on black = invisible until composited)
        self._canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self._prev_point: Optional[Tuple[int, int]] = None
        self._color_index = 0

    # ------------------------------------------------------------------
    # Drawing operations
    # ------------------------------------------------------------------

    def start_stroke(self, point: Tuple[int, int]) -> None:
        """Begin a new stroke at the given pixel coordinate."""
        self._prev_point = self._clamp(point)

    def continue_stroke(self, point: Tuple[int, int], eraser: bool = False) -> None:
        """
        Extend the current stroke to a new point.

        Parameters
        ----------
        point  : current fingertip position
        eraser : when True, paint with black (appears as erasing on dark bg)
        """
        clamped = self._clamp(point)
        if self._prev_point is None:
            self._prev_point = clamped
            return

        thickness = self.eraser_size if eraser else self.brush_size
        color = (0, 0, 0) if eraser else self.brush_color

        cv2.line(
            self._canvas,
            self._prev_point,
            clamped,
            color,
            thickness,
            cv2.LINE_AA,
        )
        self._prev_point = clamped

    def end_stroke(self) -> None:
        """Finish the current stroke (lift finger / change gesture)."""
        self._prev_point = None

    def clear(self) -> None:
        """Wipe the canvas clean."""
        self._canvas[:] = 0
        self._prev_point = None

    def cycle_color(self) -> Tuple[int, int, int]:
        """Advance to the next palette colour and return it."""
        self._color_index = (self._color_index + 1) % len(config.COLOR_PALETTE)
        self.brush_color = config.COLOR_PALETTE[self._color_index]
        return self.brush_color

    def set_color(self, color: Tuple[int, int, int]) -> None:
        """Set brush colour directly (e.g. from sidebar click)."""
        self.brush_color = color
        try:
            self._color_index = config.COLOR_PALETTE.index(color)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Compositing
    # ------------------------------------------------------------------

    def composite(self, frame: np.ndarray, sidebar_width: int = config.SIDEBAR_WIDTH) -> np.ndarray:
        """
        Blend the canvas onto the camera frame.

        Only the region to the right of the sidebar is composited so the
        sidebar stays crisp and unaffected by drawing.
        """
        output = frame.copy()
        x_start = sidebar_width
        roi = output[:, x_start:]
        canvas_roi = self._canvas[:, x_start:]

        # Additive blend: coloured strokes glow over the camera feed
        mask = cv2.cvtColor(canvas_roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(canvas_roi, canvas_roi, mask=mask)

        # Slight glow effect for a modern look
        glow = cv2.GaussianBlur(fg, (0, 0), sigmaX=3, sigmaY=3)
        blended = cv2.addWeighted(bg, 1.0, glow, 0.85, 0)
        blended = cv2.add(blended, fg)

        output[:, x_start:] = blended
        return output

    def get_canvas(self) -> np.ndarray:
        """Return a copy of the raw canvas buffer (for saving)."""
        return self._canvas.copy()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clamp(self, point: Tuple[int, int]) -> Tuple[int, int]:
        """Keep coordinates inside canvas bounds."""
        x = max(0, min(point[0], self.width - 1))
        y = max(0, min(point[1], self.height - 1))
        return x, y
