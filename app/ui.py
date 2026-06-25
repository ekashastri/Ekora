"""
UI renderer module – modern HUD, sidebar, and visual overlays.

Design language
---------------
  • Dark glass-morphism sidebar on the left
  • Rounded colour swatches with active highlight
  • Live FPS counter and gesture status pill
  • Subtle vignette on the drawing area
  • Crosshair cursor when in hover/point mode
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

import config
from app.gesture_detector import GESTURE_LABELS, Gesture


class UIRenderer:
    """Draws all non-canvas UI chrome onto the output frame."""

    def __init__(self) -> None:
        self._fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.time()
        self._gesture_flash_timer = 0.0
        self._gesture_flash_text = ""

    # ------------------------------------------------------------------
    # Frame-level render
    # ------------------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        gesture: Gesture,
        brush_color: Tuple[int, int, int],
        brush_size: int,
        hand_detected: bool,
        cursor_pos: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Apply all UI layers to the frame and return the result.

        Parameters
        ----------
        frame          : composited camera + canvas frame
        gesture        : currently detected gesture
        brush_color    : active brush BGR colour
        brush_size     : active brush diameter
        hand_detected  : whether a hand is in frame
        cursor_pos     : index fingertip for crosshair overlay
        """
        self._update_fps()
        output = frame.copy()

        self._draw_sidebar(output, brush_color, brush_size, gesture, hand_detected)
        self._draw_vignette(output)
        self._draw_status_bar(output, gesture, hand_detected)

        if cursor_pos and gesture in (Gesture.POINT, Gesture.PINCH, Gesture.FIST):
            self._draw_cursor(output, cursor_pos, gesture, brush_color)

        if self._gesture_flash_timer > 0:
            self._draw_flash_banner(output)
            self._gesture_flash_timer -= 1.0 / max(self._fps, 1)

        return output

    def flash_message(self, text: str, duration_sec: float = 1.5) -> None:
        """Show a temporary banner (e.g. 'Canvas Cleared!')."""
        self._gesture_flash_text = text
        self._gesture_flash_timer = duration_sec

    @property
    def fps(self) -> float:
        return self._fps

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------

    def _draw_sidebar(
        self,
        frame: np.ndarray,
        brush_color: Tuple[int, int, int],
        brush_size: int,
        gesture: Gesture,
        hand_detected: bool,
    ) -> None:
        """Render the left sidebar with title, palette, and tool info."""
        sw = config.SIDEBAR_WIDTH
        h = frame.shape[0]

        # Semi-transparent dark panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (sw, h), (18, 18, 28), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        # Accent stripe
        cv2.rectangle(frame, (0, 0), (4, h), (0, 200, 140), -1)

        # Title
        self._put_text(frame, "HAND PAINT", (20, 42), 0.75, (0, 220, 160), 2)
        self._put_text(frame, "Virtual Finger Drawing", (20, 68), 0.42, (160, 170, 190), 1)

        # Divider
        cv2.line(frame, (16, 88), (sw - 16, 88), (50, 55, 75), 1)

        # Colour palette
        self._put_text(frame, "COLORS", (20, 118), 0.45, (120, 130, 155), 1)
        self._draw_palette(frame, brush_color, y_start=135)

        # Brush preview
        cv2.line(frame, (16, 310), (sw - 16, 310), (50, 55, 75), 1)
        self._put_text(frame, "BRUSH", (20, 340), 0.45, (120, 130, 155), 1)
        cx, cy = sw // 2, 385
        cv2.circle(frame, (cx, cy), brush_size + 4, brush_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), brush_size + 4, (80, 90, 110), 2, cv2.LINE_AA)

        # Gesture guide
        cv2.line(frame, (16, 430), (sw - 16, 430), (50, 55, 75), 1)
        self._put_text(frame, "GESTURES", (20, 460), 0.45, (120, 130, 155), 1)
        guides = [
            ("Pinch", "Draw"),
            ("Point", "Hover"),
            ("Fist", "Erase"),
            ("Peace", "Color"),
            ("Palm", "Clear"),
        ]
        gy = 485
        for key, action in guides:
            self._put_text(frame, f"{key}", (20, gy), 0.38, (0, 190, 140), 1)
            self._put_text(frame, f"{action}", (sw - 20, gy), 0.38, (150, 160, 180), 1, align_right=True, right_x=sw - 20)
            gy += 28

        # Hand status indicator at bottom
        status_color = (0, 200, 120) if hand_detected else (60, 65, 80)
        status_text = "Hand detected" if hand_detected else "Show your hand"
        cv2.circle(frame, (28, h - 35), 6, status_color, -1, cv2.LINE_AA)
        self._put_text(frame, status_text, (44, h - 30), 0.42, status_color, 1)

        # Active gesture pill
        label = GESTURE_LABELS.get(gesture, "")
        if label and hand_detected:
            self._draw_pill(frame, label, (20, h - 75), (0, 180, 130))

    def _draw_palette(self, frame: np.ndarray, active_color: Tuple[int, int, int], y_start: int) -> None:
        """Draw colour swatches in a 4×2 grid."""
        cols, swatch_size, gap = 4, 36, 12
        x0 = 20
        for i, color in enumerate(config.COLOR_PALETTE):
            row, col = divmod(i, cols)
            x = x0 + col * (swatch_size + gap)
            y = y_start + row * (swatch_size + gap)
            is_active = color == active_color
            radius = swatch_size // 2

            if is_active:
                cv2.circle(frame, (x + radius, y + radius), radius + 4, (0, 220, 160), 2, cv2.LINE_AA)

            cv2.circle(frame, (x + radius, y + radius), radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x + radius, y + radius), radius, (60, 70, 90), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Overlays
    # ------------------------------------------------------------------

    def _draw_status_bar(self, frame: np.ndarray, gesture: Gesture, hand_detected: bool) -> None:
        """Top-right FPS and mode indicator."""
        h, w = frame.shape[:2]
        fps_text = f"{self._fps:.0f} FPS"
        self._draw_pill(frame, fps_text, (w - 110, 28), (50, 60, 80))

        if hand_detected and gesture != Gesture.NONE:
            label = GESTURE_LABELS[gesture]
            self._draw_pill(frame, label, (w - 110, 62), (0, 150, 110))

    def _draw_cursor(
        self,
        frame: np.ndarray,
        pos: Tuple[int, int],
        gesture: Gesture,
        brush_color: Tuple[int, int, int],
    ) -> None:
        """Crosshair + brush ring at the fingertip."""
        x, y = pos
        color = (100, 100, 120) if gesture == Gesture.POINT else brush_color
        size = 12

        cv2.circle(frame, (x, y), size, color, 1, cv2.LINE_AA)
        cv2.line(frame, (x - size - 6, y), (x - 4, y), color, 1, cv2.LINE_AA)
        cv2.line(frame, (x + 4, y), (x + size + 6, y), color, 1, cv2.LINE_AA)
        cv2.line(frame, (x, y - size - 6), (x, y - 4), color, 1, cv2.LINE_AA)
        cv2.line(frame, (x, y + 4), (x, y + size + 6), color, 1, cv2.LINE_AA)

    def _draw_vignette(self, frame: np.ndarray) -> None:
        """Subtle edge darkening on the drawing area."""
        h, w = frame.shape[:2]
        sw = config.SIDEBAR_WIDTH
        # Only apply to the drawing region
        roi = frame[:, sw:]
        rows, cols = roi.shape[:2]
        X = cv2.getGaussianKernel(cols, cols * 0.8)
        Y = cv2.getGaussianKernel(rows, rows * 0.8)
        kernel = Y * X.T
        mask = kernel / kernel.max()
        mask = (mask * 0.25 + 0.75)  # Subtle – don't crush brightness
        for c in range(3):
            roi[:, :, c] = (roi[:, :, c] * mask).astype(np.uint8)

    def _draw_flash_banner(self, frame: np.ndarray) -> None:
        """Centre-screen temporary notification."""
        h, w = frame.shape[:2]
        text = self._gesture_flash_text
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.9, 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        bx, by = (w - tw) // 2 - 20, (h + th) // 2
        cv2.rectangle(frame, (bx, by - th - 14), (bx + tw + 40, by + 14), (20, 25, 35), -1)
        cv2.rectangle(frame, (bx, by - th - 14), (bx + tw + 40, by + 14), (0, 200, 140), 2)
        cv2.putText(frame, text, (bx + 20, by), font, scale, (0, 230, 170), thickness, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    def _draw_pill(
        self,
        frame: np.ndarray,
        text: str,
        centre: Tuple[int, int],
        bg_color: Tuple[int, int, int],
    ) -> None:
        """Rounded-rectangle badge with centred text."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.45, 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        pad_x, pad_y = 14, 8
        cx, cy = centre
        x1 = cx - tw // 2 - pad_x
        y1 = cy - th // 2 - pad_y
        x2 = cx + tw // 2 + pad_x
        y2 = cy + th // 2 + pad_y + baseline
        cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 80, 100), 1)
        cv2.putText(frame, text, (cx - tw // 2, cy + th // 2), font, scale, (220, 225, 235), thickness, cv2.LINE_AA)

    def _put_text(
        self,
        frame: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        scale: float,
        color: Tuple[int, int, int],
        thickness: int,
        align_right: bool = False,
        right_x: int = 0,
    ) -> None:
        """Helper to draw text with optional right alignment."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        if align_right:
            (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
            pos = (right_x - tw, pos[1])
        cv2.putText(frame, text, pos, font, scale, color, thickness, cv2.LINE_AA)

    def _update_fps(self) -> None:
        """Compute rolling FPS every FPS_UPDATE_INTERVAL seconds."""
        self._frame_count += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= config.FPS_UPDATE_INTERVAL:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer = time.time()
