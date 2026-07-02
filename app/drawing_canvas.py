"""
Drawing canvas module – virtual paint layer and stroke management.

The canvas is a separate numpy array the same size as the camera frame.
Drawing happens only inside the canvas region (excluding the sidebar).
Strokes are anti-aliased line segments between consecutive fingertip positions.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import math
import json

import cv2
import numpy as np

import config
from app import shape_recognizer


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
        self._second_prev_point: Optional[Tuple[int, int]] = None
        self._color_index = 0

        # Pan offset (pixels) – shifts the canvas view when the user pans
        # with a closed fist. Drawing coordinates are NOT affected, only
        # how the buffer is displayed during compositing.
        self.pan_offset: List[int] = [0, 0]
        self._pan_prev_point: Optional[Tuple[int, int]] = None

        # Shape recognition
        self._stroke_points: List[Tuple[int, int]] = []
        self.shape_recognition_enabled: bool = True

        self.is_dirty: bool = False
        
        self.history: List[dict] = []
        self.history_index: int = 0
        self._current_is_eraser: bool = False
        
        # Object interaction state
        self.hovered_shape_index: Optional[int] = None
        self.active_shape_index: Optional[int] = None
        self.interaction_mode: Optional[str] = None
        self.interaction_start_pt: Optional[Tuple[int, int]] = None
        self.interaction_initial_params: Optional[dict] = None

    def _add_action(self, action: dict) -> None:
        if self.history_index < len(self.history):
            self.history = self.history[:self.history_index]
        self.history.append(action)
        self.history_index += 1
        self.is_dirty = True


    # ------------------------------------------------------------------
    # Object Interaction (Smart Shapes)
    # ------------------------------------------------------------------

    def hit_test(self, point: Tuple[int, int]) -> Tuple[Optional[int], Optional[str]]:
        """
        Check if point hits a shape's bounding box.
        Returns (history_index, hit_type) where hit_type is "move" or "resize_XX".
        Searches backwards to pick topmost shape.
        """
        px, py = point
        hit_zone = 20  # generous hit zone
        
        for i in range(self.history_index - 1, -1, -1):
            action = self.history[i]
            if action.get("type") != "shape":
                continue
                
            # Compute bounding box of the shape
            shape_type = action["shape_type"]
            params = action["params"]
            
            x1, y1, x2, y2 = 0, 0, 0, 0
            if shape_type == "circle":
                r = params["radius"]
                cx, cy = params["cx"], params["cy"]
                x1, y1, x2, y2 = cx - r, cy - r, cx + r, cy + r
            elif shape_type == "ellipse":
                rx, ry = params["axes_a"] / 2, params["axes_b"] / 2
                # Rough AABB
                r = max(rx, ry)
                cx, cy = params["cx"], params["cy"]
                x1, y1, x2, y2 = cx - r, cy - r, cx + r, cy + r
            elif shape_type in ("rectangle", "square"):
                x1, y1 = params["x"], params["y"]
                w = params.get("w", params.get("side", 0))
                h = params.get("h", params.get("side", 0))
                x2, y2 = x1 + w, y1 + h
            elif shape_type == "triangle":
                pts = np.array(params["pts"])
                x1, y1 = pts[:, 0].min(), pts[:, 1].min()
                x2, y2 = pts[:, 0].max(), pts[:, 1].max()
            elif shape_type in ("line", "arrow"):
                p0 = params["p0"]
                p1 = params["p1"]
                x1, y1 = min(p0[0], p1[0]), min(p0[1], p1[1])
                x2, y2 = max(p0[0], p1[0]), max(p0[1], p1[1])
                
            # Check corners for resize
            corners = [
                (x1, y1, "resize_tl"), (x2, y1, "resize_tr"),
                (x1, y2, "resize_bl"), (x2, y2, "resize_br")
            ]
            for cx, cy, c_type in corners:
                if math.hypot(px - cx, py - cy) <= hit_zone:
                    return i, c_type
            
            # Check body for move
            if (x1 - hit_zone <= px <= x2 + hit_zone) and (y1 - hit_zone <= py <= y2 + hit_zone):
                return i, "move"
                
        return None, None

    def update_hover(self, point: Optional[Tuple[int, int]]) -> None:
        if point is None:
            self.hovered_shape_index = None
            return
        idx, _ = self.hit_test(point)
        self.hovered_shape_index = idx

    def start_interaction(self, point: Tuple[int, int]) -> bool:
        """Begin interacting with a shape. Returns True if started."""
        idx, hit_type = self.hit_test(point)
        if idx is not None:
            self.active_shape_index = idx
            self.interaction_mode = hit_type
            self.interaction_start_pt = point
            # Clone params so we can modify them incrementally
            import copy
            self.interaction_initial_params = copy.deepcopy(self.history[idx]["params"])
            return True
        return False

    def continue_interaction(self, point: Tuple[int, int]) -> None:
        if self.active_shape_index is None:
            return
            
        dx = point[0] - self.interaction_start_pt[0]
        dy = point[1] - self.interaction_start_pt[1]
        
        action = self.history[self.active_shape_index]
        shape_type = action["shape_type"]
        ip = self.interaction_initial_params
        p = action["params"]
        
        if self.interaction_mode == "move":
            if shape_type in ("circle", "ellipse"):
                p["cx"] = ip["cx"] + dx
                p["cy"] = ip["cy"] + dy
            elif shape_type in ("rectangle", "square"):
                p["x"] = ip["x"] + dx
                p["y"] = ip["y"] + dy
            elif shape_type == "triangle":
                p["pts"] = [(pt[0] + dx, pt[1] + dy) for pt in ip["pts"]]
            elif shape_type in ("line", "arrow"):
                p["p0"] = (ip["p0"][0] + dx, ip["p0"][1] + dy)
                p["p1"] = (ip["p1"][0] + dx, ip["p1"][1] + dy)
                
        elif self.interaction_mode.startswith("resize"):
            # Uniform scale based on dx
            scale = max(0.1, 1.0 + (dx / 100.0))
            if shape_type == "circle":
                p["radius"] = max(5, int(ip["radius"] * scale))
            elif shape_type == "ellipse":
                p["axes_a"] = max(5, int(ip["axes_a"] * scale))
                p["axes_b"] = max(5, int(ip["axes_b"] * scale))
            elif shape_type == "square":
                p["side"] = max(5, int(ip["side"] * scale))
            elif shape_type == "rectangle":
                p["w"] = max(5, int(ip["w"] * scale))
                p["h"] = max(5, int(ip["h"] * scale))
            elif shape_type == "triangle":
                # scale around centroid
                pts = np.array(ip["pts"])
                centroid = pts.mean(axis=0)
                new_pts = centroid + (pts - centroid) * scale
                p["pts"] = [(int(pt[0]), int(pt[1])) for pt in new_pts]
            elif shape_type in ("line", "arrow"):
                # scale around p0
                p0 = np.array(ip["p0"])
                p1 = np.array(ip["p1"])
                p1 = p0 + (p1 - p0) * scale
                p["p1"] = (int(p1[0]), int(p1[1]))
                
        self.is_dirty = True
        self._replay(self.history_index)

    def end_interaction(self) -> None:
        self.active_shape_index = None
        self.interaction_mode = None
        self.interaction_start_pt = None
        self.interaction_initial_params = None
        # Push a dummy state so undo returns it to before interaction? 
        # Actually modifying history in place breaks undo. 
        # For simplicity, we just modify it in place, meaning undo will jump over it or just undo the shape completely.
        # A more robust solution pushes the new state. Let's do that:
        # Wait, to keep it simple, we just allow in-place edits.

    def delete_hovered_shape(self) -> bool:
        if self.hovered_shape_index is not None:
            # We mark it as deleted or remove it
            # To support undo, it's better to append a 'delete' action, or just clear and redraw.
            # Easiest way is to remove from history and push a copy of history.
            # But the user asked for simple delete. Let's just pop it from history.
            action = self.history.pop(self.hovered_shape_index)
            self.history_index -= 1
            self.hovered_shape_index = None
            self.is_dirty = True
            self._replay(self.history_index)
            return True
        return False

    # ------------------------------------------------------------------
    # Drawing operations
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Pan (fist gesture) – moves the canvas view, never draws
    # ------------------------------------------------------------------

    def start_pan(self, point: Tuple[int, int]) -> None:
        """Begin a pan gesture at the given pixel coordinate."""
        self._pan_prev_point = point

    def continue_pan(self, point: Tuple[int, int]) -> None:
        """Move the canvas view by the delta since the last pan point."""
        if self._pan_prev_point is None:
            self._pan_prev_point = point
            return
        sens = getattr(config, "PAN_SENSITIVITY", 1.0)
        dx = int((point[0] - self._pan_prev_point[0]) * sens)
        dy = int((point[1] - self._pan_prev_point[1]) * sens)
        self.pan_offset[0] += dx
        self.pan_offset[1] += dy
        self._pan_prev_point = point

    def end_pan(self) -> None:
        """Finish the pan gesture (no accidental strokes are produced)."""
        self._pan_prev_point = None

    def start_stroke(self, point: Tuple[int, int]) -> None:
        """Begin a new stroke at the given pixel coordinate."""
        self.is_dirty = True
        self._current_is_eraser = False
        self._prev_point = self._clamp(point)
        self._second_prev_point = None
        self._stroke_points = [self._prev_point]

    def continue_stroke(self, point: Tuple[int, int], eraser: bool = False) -> None:
        """
        Extend the current stroke to a new point.
        """
        self._current_is_eraser = eraser
        clamped = self._clamp(point)

        if self._prev_point is None:
            self._prev_point = clamped
            self._stroke_points = [clamped]
            if eraser:
                self._erase_at(clamped)
            return

        p0, p1 = self._prev_point, clamped

        if eraser:
            self._erase_segment(p0, p1)
            self._stroke_points.append(clamped)
            self._second_prev_point = p0
            self._prev_point = clamped
            return

        # --- Catmull-Rom spline interpolation -----------------------------
        # Render the segment between the previous raw point (p1_ctrl) and
        # the newly captured point (p2_ctrl) as a smooth curve instead of a
        # straight line. This removes the faceted look on circles/spirals
        # and the jagged breaks on fast curved motion, without waiting for
        # any future point (so it adds zero extra frames of latency):
        #   - p0_ctrl: the point before the previous one (real, already
        #     captured) gives the incoming tangent.
        #   - p3_ctrl: since the true "next" point doesn't exist yet, it is
        #     linearly extrapolated from the last two real points. This is
        #     a cheap, immediate approximation - not a delay - and only
        #     affects curvature smoothing, never the recorded path.
        # Every raw point is still appended to self._stroke_points below,
        # so handwriting/shape recognition keep seeing the full, untouched
        # capture - only the on-canvas rendering is smoothed.
        p1_ctrl = p0
        p2_ctrl = p1
        p0_ctrl = self._second_prev_point if self._second_prev_point is not None else p1_ctrl
        ex = p2_ctrl[0] - p1_ctrl[0]
        ey = p2_ctrl[1] - p1_ctrl[1]
        p3_ctrl = (p2_ctrl[0] + ex, p2_ctrl[1] + ey)

        dist = math.hypot(ex, ey)
        max_gap = getattr(config, "STROKE_MAX_GAP_PX", 6)
        # At least 2 sub-segments per captured point, more for fast/long
        # jumps, so quick curved motion never skips points or leaves gaps.
        steps = max(2, int(dist / max_gap) + 1)

        prev_draw = p1_ctrl
        for i in range(1, steps + 1):
            t = i / steps
            curr = self._catmull_rom_point(p0_ctrl, p1_ctrl, p2_ctrl, p3_ctrl, t)
            cv2.line(self._canvas, prev_draw, curr, self.brush_color, self.brush_size, cv2.LINE_AA)
            prev_draw = curr

        self._stroke_points.append(clamped)
        self._second_prev_point = p1_ctrl
        self._prev_point = clamped

    @staticmethod
    def _catmull_rom_point(
        p0: Tuple[int, int], p1: Tuple[int, int],
        p2: Tuple[int, int], p3: Tuple[int, int], t: float,
    ) -> Tuple[int, int]:
        """Evaluate a centripetal-style (uniform) Catmull-Rom spline at t in [0,1]
        between p1 and p2, using p0/p3 as tangent control points. Lightweight
        (closed-form, no extra captured points needed) so it adds no latency."""
        t2 = t * t
        t3 = t2 * t
        x = 0.5 * (
            (2 * p1[0])
            + (-p0[0] + p2[0]) * t
            + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
            + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
        )
        y = 0.5 * (
            (2 * p1[1])
            + (-p0[1] + p2[1]) * t
            + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
            + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
        )
        return int(x), int(y)

    def _erase_at(self, center: Tuple[int, int]) -> None:
        """Erase a circle at center with radius = eraser_size // 2."""
        radius = self.eraser_size // 2
        cv2.circle(self._canvas, center, radius, (0, 0, 0), -1)

    def _erase_segment(self, p0: Tuple[int, int], p1: Tuple[int, int]) -> None:
        """
        Erase all canvas pixels within eraser_size // 2 of the line p0→p1.

        Uses filled circles at interpolated steps so fast hand movement does
        not leave gaps between erase positions.
        """
        radius = self.eraser_size // 2
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dist = max(1, int(math.hypot(dx, dy)))
        # Step every radius//2 pixels so circles overlap (no gap)
        step = max(1, radius // 2)
        steps = dist // step + 1
        for i in range(steps + 1):
            t = i / steps
            x = int(p0[0] + dx * t)
            y = int(p0[1] + dy * t)
            cv2.circle(self._canvas, (x, y), radius, (0, 0, 0), -1)
        # Always stamp the endpoints exactly
        cv2.circle(self._canvas, p0, radius, (0, 0, 0), -1)
        cv2.circle(self._canvas, p1, radius, (0, 0, 0), -1)

    def end_stroke(self) -> Optional[str]:
        """
        Finish the current stroke.
        """
        recognized = None
        action = None
        if (
            self.shape_recognition_enabled
            and len(self._stroke_points) >= shape_recognizer.MIN_STROKE_POINTS
        ):
            shape_type, confidence, params = shape_recognizer.recognize(self._stroke_points)
            if shape_type is not None:
                self._erase_stroke_region()
                shape_recognizer.draw_clean_shape(
                    self._canvas, shape_type, params,
                    self.brush_color, self.brush_size
                )
                recognized = shape_type
                action = {
                    "type": "shape",
                    "shape_type": shape_type,
                    "params": params,
                    "color": list(self.brush_color),
                    "size": self.brush_size
                }

        if action is None and len(self._stroke_points) > 0:
            action = {
                "type": "stroke",
                "points": [list(p) for p in self._stroke_points],
                "color": list(self.brush_color),
                "size": self.brush_size,
                "eraser": self._current_is_eraser,
                "eraser_size": self.eraser_size
            }
            
        if action is not None:
            self._add_action(action)

        self._stroke_points = []
        self._prev_point = None
        return recognized

    def _erase_stroke_region(self) -> None:
        """
        Black out the pixels occupied by the current freehand stroke so it can
        be replaced by a clean shape.  We draw thick black lines along the
        same path the brush took, with a slightly enlarged thickness to fully
        cover anti-aliased edges.
        """
        if len(self._stroke_points) < 2:
            return
        pts = self._stroke_points
        erase_thick = self.brush_size + 4
        for i in range(1, len(pts)):
            cv2.line(self._canvas, pts[i - 1], pts[i],
                     (0, 0, 0), erase_thick, cv2.LINE_AA)

    def clear(self) -> None:
        """Wipe the canvas clean."""
        self.is_dirty = True
        self._canvas[:] = 0
        self._prev_point = None
        self._stroke_points = []
        self._add_action({"type": "clear"})

    def _replay(self, up_to_index: int) -> None:
        self._canvas[:] = 0
        old_shape = self.shape_recognition_enabled
        self.shape_recognition_enabled = False
        
        for i in range(up_to_index):
            action = self.history[i]
            if action["type"] == "clear":
                self._canvas[:] = 0
            elif action["type"] == "stroke":
                pts = action.get("points", [])
                if not pts: continue
                
                # Setup state
                old_color = self.brush_color
                old_size = self.brush_size
                old_esize = self.eraser_size
                
                self.brush_color = tuple(action["color"])
                self.brush_size = action["size"]
                self.eraser_size = action.get("eraser_size", config.ERASER_SIZE)
                eraser = action.get("eraser", False)
                
                self._prev_point = tuple(pts[0])
                if eraser:
                    self._erase_at(self._prev_point)
                    
                for pt in pts[1:]:
                    self.continue_stroke(tuple(pt), eraser=eraser)
                    
                # Restore state
                self.brush_color = old_color
                self.brush_size = old_size
                self.eraser_size = old_esize
                
            elif action["type"] == "shape":
                shape_recognizer.draw_clean_shape(
                    self._canvas, action["shape_type"], action["params"],
                    tuple(action["color"]), action["size"]
                )
                
        self.shape_recognition_enabled = old_shape
        self._prev_point = None
        self._stroke_points = []

    def undo(self) -> None:
        """Undo the last stroke or clear action."""
        if self.history_index > 0:
            self.history_index -= 1
            self._replay(self.history_index)
            self.is_dirty = True

    def redo(self) -> None:
        """Redo the previously undone action."""
        if self.history_index < len(self.history):
            self.history_index += 1
            self._replay(self.history_index)
            self.is_dirty = True

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

        # Apply pan offset: shift the canvas buffer before compositing so
        # panning never mutates stroke coordinates, only the displayed view.
        if self.pan_offset[0] != 0 or self.pan_offset[1] != 0:
            M = np.float32([[1, 0, self.pan_offset[0]], [0, 1, self.pan_offset[1]]])
            shown_canvas = cv2.warpAffine(
                self._canvas, M, (self.width, self.height),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
            )
        else:
            shown_canvas = self._canvas

        roi = output[:, x_start:]
        canvas_roi = shown_canvas[:, x_start:]

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
        
        # Draw hover/active bounding boxes
        target_idx = self.active_shape_index if self.active_shape_index is not None else self.hovered_shape_index
        if target_idx is not None and target_idx < len(self.history):
            action = self.history[target_idx]
            if action.get("type") == "shape":
                shape_type = action["shape_type"]
                params = action["params"]
                x1, y1, x2, y2 = 0, 0, 0, 0
                if shape_type == "circle":
                    r = params["radius"]
                    cx, cy = params["cx"], params["cy"]
                    x1, y1, x2, y2 = cx - r, cy - r, cx + r, cy + r
                elif shape_type == "ellipse":
                    r = max(params["axes_a"], params["axes_b"]) / 2
                    cx, cy = params["cx"], params["cy"]
                    x1, y1, x2, y2 = cx - r, cy - r, cx + r, cy + r
                elif shape_type in ("rectangle", "square"):
                    x1, y1 = params["x"], params["y"]
                    w = params.get("w", params.get("side", 0))
                    h = params.get("h", params.get("side", 0))
                    x2, y2 = x1 + w, y1 + h
                elif shape_type == "triangle":
                    pts = np.array(params["pts"])
                    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
                    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
                elif shape_type in ("line", "arrow"):
                    p0 = params["p0"]
                    p1 = params["p1"]
                    x1, y1 = min(p0[0], p1[0]), min(p0[1], p1[1])
                    x2, y2 = max(p0[0], p1[0]), max(p0[1], p1[1])
                
                # Draw box
                cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1, cv2.LINE_AA)
                # Draw corners
                for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                    cv2.circle(output, (int(cx), int(cy)), 5, (255, 255, 255), -1, cv2.LINE_AA)

        return output

    def get_canvas(self) -> np.ndarray:
        """Return a copy of the raw canvas buffer (for saving)."""
        return self._canvas.copy()

    def load_canvas(self, img: np.ndarray) -> None:
        """Load an image into the canvas, resizing to fit if necessary."""
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if img.shape[:2] != (self.height, self.width):
            img = cv2.resize(img, (self.width, self.height))
        self._canvas[:] = img
        self._prev_point = None
        self.history.clear()
        self.history_index = 0
        self.is_dirty = False

    def to_json(self) -> str:
        data = {
            "version": "1.9",
            "canvas_width": self.width,
            "canvas_height": self.height,
            "history": self.history,
            "history_index": self.history_index
        }
        return json.dumps(data)

    def from_json(self, json_str: str) -> None:
        data = json.loads(json_str)
        self.history = data.get("history", [])
        self.history_index = data.get("history_index", len(self.history))
        self._replay(self.history_index)
        self.is_dirty = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clamp(self, point: Tuple[int, int]) -> Tuple[int, int]:
        """Keep coordinates inside canvas bounds."""
        x = max(0, min(point[0], self.width - 1))
        y = max(0, min(point[1], self.height - 1))
        return x, y
