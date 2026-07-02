"""
Stateless Drawing Engine for rendering the Document to a NumPy canvas.
"""
import cv2
import numpy as np
import math
from typing import Tuple, List, Optional

from app.model import Document, Stroke, Shape, Layer, Viewport
from app import shape_recognizer

class DrawingEngine:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        
    def render(self, document: Document, viewport: Viewport, active_stroke: Optional[Stroke] = None) -> np.ndarray:
        """Render the entire document onto a new canvas array."""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Render grid if needed
        # self._render_grid(canvas, viewport)
        
        for layer in document.layers:
            if not layer.visible:
                continue
            for item in layer.items:
                if isinstance(item, Stroke):
                    self._render_stroke(canvas, item, viewport)
                elif isinstance(item, Shape):
                    self._render_shape(canvas, item, viewport)
                    
        # Render the transient active stroke currently being drawn
        if active_stroke is not None:
            self._render_stroke(canvas, active_stroke, viewport)
            
        return canvas
        
    def _render_stroke(self, canvas: np.ndarray, stroke: Stroke, viewport: Viewport) -> None:
        pts = [viewport.doc_to_screen(pt) for pt in stroke.points]
        # Convert to int tuples
        int_pts = [(int(p[0]), int(p[1])) for p in pts]
        
        color = (0, 0, 0) if stroke.is_eraser else stroke.color
        
        if len(int_pts) < 2:
            if int_pts:
                cv2.circle(canvas, int_pts[0], stroke.thickness // 2, color, -1)
            return
            
        # Optional: Catmull-Rom interpolation could happen here, or be pre-calculated
        # For simplicity and speed in full-redraws, we just draw polylines.
        cv2.polylines(canvas, [np.array(int_pts)], isClosed=False, color=color, thickness=stroke.thickness, lineType=cv2.LINE_AA)
        
        # Add round caps
        cv2.circle(canvas, int_pts[0], stroke.thickness // 2, color, -1)
        cv2.circle(canvas, int_pts[-1], stroke.thickness // 2, color, -1)

    def _render_shape(self, canvas: np.ndarray, shape: Shape, viewport: Viewport) -> None:
        # Scale shape parameters to screen space
        params = shape.params.copy()
        
        # For shapes, we need to transform their key parameters
        stype = shape.shape_type
        if stype in ("circle", "ellipse"):
            c = viewport.doc_to_screen((params["cx"], params["cy"]))
            params["cx"], params["cy"] = int(c[0]), int(c[1])
            params["radius"] = int(params.get("radius", 0) * viewport.zoom)
            params["axes_a"] = int(params.get("axes_a", 0) * viewport.zoom)
            params["axes_b"] = int(params.get("axes_b", 0) * viewport.zoom)
            
        elif stype in ("rectangle", "square"):
            p = viewport.doc_to_screen((params["x"], params["y"]))
            params["x"], params["y"] = int(p[0]), int(p[1])
            params["w"] = int(params.get("w", 0) * viewport.zoom)
            params["h"] = int(params.get("h", 0) * viewport.zoom)
            params["side"] = int(params.get("side", 0) * viewport.zoom)
            
        elif stype == "triangle":
            pts = [viewport.doc_to_screen(pt) for pt in params["pts"]]
            params["pts"] = [(int(p[0]), int(p[1])) for p in pts]
            
        elif stype in ("line", "arrow"):
            p0 = viewport.doc_to_screen(params["p0"])
            p1 = viewport.doc_to_screen(params["p1"])
            params["p0"] = (int(p0[0]), int(p0[1]))
            params["p1"] = (int(p1[0]), int(p1[1]))
            
        elif stype == "text":
            p = viewport.doc_to_screen((params.get("x", 100), params.get("y", 100)))
            params["x"], params["y"] = int(p[0]), int(p[1])
            params["scale"] = params.get("scale", 1.0) * viewport.zoom

        shape_recognizer.draw_clean_shape(canvas, stype, params, shape.color, shape.thickness)
