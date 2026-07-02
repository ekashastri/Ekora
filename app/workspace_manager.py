"""
Workspace Manager orchestrates the Document, Viewport, and Command history.
"""
from typing import Tuple, Optional, List
import math
import numpy as np

from app.model import Document, Stroke, Shape, Layer, Viewport
from app.commands import CommandManager, AddItemCommand, RemoveItemCommand, ModifyShapeCommand, ClearDocumentCommand
import config
from app import shape_recognizer
from app.drawing_engine import DrawingEngine
import cv2




class WorkspaceManager:
    """Central repository for the drawing state, managing history and viewport."""
    def __init__(self):
        self.document = Document()
        self.commands = CommandManager(self.document)
        self.viewport = Viewport()
        self.engine = DrawingEngine(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
        self.is_dirty = True
        
        # Tool state
        self.brush_color = config.DEFAULT_BRUSH_COLOR
        self.brush_size = config.DEFAULT_BRUSH_SIZE
        self.eraser_size = config.ERASER_SIZE
        self.shape_recognition_enabled = True
        
        # Stroke drawing state
        self._current_stroke: Optional[Stroke] = None
        
        # Interaction state
        self.hovered_item: Optional[Shape] = None
        self.active_item: Optional[Shape] = None
        self.interaction_mode: Optional[str] = None
        self.interaction_start_pt: Optional[Tuple[float, float]] = None
        self.interaction_initial_params: Optional[dict] = None
        
        # Panning state
        self._pan_prev_point: Optional[Tuple[float, float]] = None
        
    # ------------------------------------------------------------------
    # Strokes
    # ------------------------------------------------------------------
    def start_stroke(self, point: Tuple[int, int]) -> None:
        self.is_dirty = True
        doc_pt = self.viewport.screen_to_doc(point)
        self._current_raw_points = [doc_pt]
        self._current_stroke = Stroke([doc_pt], self.brush_color, self.brush_size, is_eraser=False)
        
    def continue_stroke(self, point: Tuple[int, int], eraser: bool = False) -> None:
        if not self._current_stroke:
            self.start_stroke(point)
            self._current_stroke.is_eraser = eraser
            return
            
        doc_pt = self.viewport.screen_to_doc(point)
        
        # Avoid extremely dense raw points
        last_raw = self._current_raw_points[-1]
        dist_raw = math.hypot(doc_pt[0] - last_raw[0], doc_pt[1] - last_raw[1])
        if dist_raw < 2.0 / self.viewport.zoom:
            return
            
        self._current_raw_points.append(doc_pt)
        
        raw = self._current_raw_points
        interpolated = []
        
        if len(raw) < 2:
            interpolated = list(raw)
        else:
            for i in range(len(raw) - 1):
                p1 = raw[i]
                p2 = raw[i+1]
                p0 = raw[i-1] if i > 0 else p1
                p3 = raw[i+2] if i < len(raw) - 2 else (p2[0] + (p2[0]-p1[0]), p2[1] + (p2[1]-p1[1]))
                
                if i == 0:
                    interpolated.append(p1)
                    
                dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                max_gap = getattr(config, "STROKE_MAX_GAP_PX", 6) / self.viewport.zoom
                steps = max(2, int(dist / max_gap) + 1)
                
                for step in range(1, steps + 1):
                    t = step / steps
                    t2 = t * t
                    t3 = t2 * t
                    x = 0.5 * (
                        (2 * p1[0]) + (-p0[0] + p2[0]) * t +
                        (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                        (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
                    )
                    y = 0.5 * (
                        (2 * p1[1]) + (-p0[1] + p2[1]) * t +
                        (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                        (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
                    )
                    interpolated.append((x, y))
                    
        self._current_stroke.points = interpolated
        self.is_dirty = True
        
    def end_stroke(self) -> Optional[str]:
        if not self._current_stroke:
            return None
            
        recognized = None
        if self.shape_recognition_enabled and len(self._current_stroke.points) >= shape_recognizer.MIN_STROKE_POINTS:
            # Need to convert points back to integer screen coordinates for shape recognizer (legacy compatibility)
            # Actually, shape recognizer works on relative coordinates mostly, but let's pass screen coords
            screen_pts = [(int(p[0]*self.viewport.zoom + self.viewport.pan_x), 
                           int(p[1]*self.viewport.zoom + self.viewport.pan_y)) for p in self._current_stroke.points]
            
            shape_type, confidence, params = shape_recognizer.recognize(screen_pts)
            if shape_type is not None:
                # Convert params back to doc coordinates
                self._convert_params_to_doc(shape_type, params)
                shape = Shape(shape_type, params, self.brush_color, self.brush_size)
                self.commands.execute(AddItemCommand(shape, self.document.active_layer_index))
                recognized = shape_type
                
        if recognized is None and len(self._current_stroke.points) > 1:
            self.commands.execute(AddItemCommand(self._current_stroke, self.document.active_layer_index))
            
        self._current_stroke = None
        self.is_dirty = True
        return recognized

    def _convert_params_to_doc(self, shape_type: str, params: dict) -> None:
        if shape_type in ("circle", "ellipse"):
            cx, cy = self.viewport.screen_to_doc((params["cx"], params["cy"]))
            params["cx"], params["cy"] = cx, cy
            params["radius"] = params.get("radius", 0) / self.viewport.zoom
            params["axes_a"] = params.get("axes_a", 0) / self.viewport.zoom
            params["axes_b"] = params.get("axes_b", 0) / self.viewport.zoom
        elif shape_type in ("rectangle", "square"):
            x, y = self.viewport.screen_to_doc((params["x"], params["y"]))
            params["x"], params["y"] = x, y
            params["w"] = params.get("w", 0) / self.viewport.zoom
            params["h"] = params.get("h", 0) / self.viewport.zoom
            params["side"] = params.get("side", 0) / self.viewport.zoom
        elif shape_type == "triangle":
            params["pts"] = [self.viewport.screen_to_doc(pt) for pt in params["pts"]]
        elif shape_type in ("line", "arrow"):
            params["p0"] = self.viewport.screen_to_doc(params["p0"])
            params["p1"] = self.viewport.screen_to_doc(params["p1"])
        elif shape_type == "text":
            x, y = self.viewport.screen_to_doc((params["x"], params["y"]))
            params["x"], params["y"] = x, y
            params["scale"] = params.get("scale", 1.0) / self.viewport.zoom

    # ------------------------------------------------------------------
    # Panning
    # ------------------------------------------------------------------
    def start_pan(self, point: Tuple[int, int]) -> None:
        self._pan_prev_point = point
        
    def continue_pan(self, point: Tuple[int, int]) -> None:
        if not self._pan_prev_point:
            self._pan_prev_point = point
            return
        dx = point[0] - self._pan_prev_point[0]
        dy = point[1] - self._pan_prev_point[1]
        self.viewport.pan_x += dx
        self.viewport.pan_y += dy
        self._pan_prev_point = point
        self.is_dirty = True
        
    def end_pan(self) -> None:
        self._pan_prev_point = None

    # ------------------------------------------------------------------
    # Object Interaction
    # ------------------------------------------------------------------
    def hit_test(self, point: Tuple[int, int]) -> Tuple[Optional[Shape], Optional[str]]:
        doc_pt = self.viewport.screen_to_doc(point)
        px, py = doc_pt
        hit_zone = 20 / self.viewport.zoom
        
        # Search backwards (top to bottom)
        for layer in reversed(self.document.layers):
            if not layer.visible or layer.locked: continue
            for item in reversed(layer.items):
                if isinstance(item, Stroke):
                    if not item.points: continue
                    pts = np.array(item.points)
                    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
                    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
                    
                    if (x1 - hit_zone <= px <= x2 + hit_zone) and (y1 - hit_zone <= py <= y2 + hit_zone):
                        return item, "move"
                    continue
                    
                if not isinstance(item, Shape):
                    continue
                    
                shape_type = item.shape_type
                params = item.params
                x1, y1, x2, y2 = 0, 0, 0, 0
                
                if shape_type == "circle":
                    r = params["radius"]
                    cx, cy = params["cx"], params["cy"]
                    x1, y1, x2, y2 = cx - r, cy - r, cx + r, cy + r
                elif shape_type == "ellipse":
                    r = max(params["axes_a"]/2, params["axes_b"]/2)
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
                    p0, p1 = params["p0"], params["p1"]
                    x1, y1 = min(p0[0], p1[0]), min(p0[1], p1[1])
                    x2, y2 = max(p0[0], p1[0]), max(p0[1], p1[1])
                elif shape_type == "text":
                    scale = params.get("scale", 1.0)
                    text_len = len(params.get("text", ""))
                    x1, y1 = params["x"], params["y"] - 20 * scale
                    x2, y2 = params["x"] + text_len * 15 * scale, params["y"] + 5 * scale
                    
                corners = [
                    (x1, y1, "resize_tl"), (x2, y1, "resize_tr"),
                    (x1, y2, "resize_bl"), (x2, y2, "resize_br")
                ]
                for cx, cy, c_type in corners:
                    if math.hypot(px - cx, py - cy) <= hit_zone:
                        return item, c_type
                        
                if (x1 - hit_zone <= px <= x2 + hit_zone) and (y1 - hit_zone <= py <= y2 + hit_zone):
                    return item, "move"
                    
        return None, None

    def update_hover(self, point: Optional[Tuple[int, int]]) -> None:
        if point is None:
            if self.hovered_item is not None:
                self.hovered_item = None
                self.is_dirty = True
            return
            
        if self.active_item is not None:
            return
            
        hit_item, hit_type = self.hit_test(point)
        if hit_item != self.hovered_item:
            self.hovered_item = hit_item
            self.is_dirty = True

    def start_interaction(self, point: Tuple[int, int]) -> None:
        hit_item, hit_type = self.hit_test(point)
        if hit_item is not None:
            self.active_item = hit_item
            self.interaction_mode = hit_type
            self.interaction_start_pt = self.viewport.screen_to_doc(point)
            import copy
            if isinstance(hit_item, Shape):
                self.interaction_initial_params = copy.deepcopy(hit_item.params)
            else:
                self.interaction_initial_params = {"points": copy.deepcopy(hit_item.points)}
            
    def continue_interaction(self, point: Tuple[int, int]) -> None:
        if not self.active_item or not self.interaction_start_pt or not self.interaction_initial_params:
            return
            
        doc_pt = self.viewport.screen_to_doc(point)
        dx = doc_pt[0] - self.interaction_start_pt[0]
        dy = doc_pt[1] - self.interaction_start_pt[1]
        
        if isinstance(self.active_item, Stroke):
            if self.interaction_mode == "move":
                orig_points = self.interaction_initial_params["points"]
                self.active_item.points = [(pt[0] + dx, pt[1] + dy) for pt in orig_points]
            self.is_dirty = True
            return
            
        stype = self.active_item.shape_type
        ip = self.interaction_initial_params
        
        new_params = ip.copy()
        
        if self.interaction_mode == "move":
            if stype in ("circle", "ellipse"):
                new_params["cx"] = ip["cx"] + dx
                new_params["cy"] = ip["cy"] + dy
            elif stype in ("rectangle", "square"):
                new_params["x"] = ip["x"] + dx
                new_params["y"] = ip["y"] + dy
            elif stype == "triangle":
                new_params["pts"] = [(pt[0] + dx, pt[1] + dy) for pt in ip["pts"]]
            elif stype in ("line", "arrow"):
                new_params["p0"] = (ip["p0"][0] + dx, ip["p0"][1] + dy)
                new_params["p1"] = (ip["p1"][0] + dx, ip["p1"][1] + dy)
            elif stype == "text":
                new_params["x"] = ip["x"] + dx
                new_params["y"] = ip["y"] + dy
                
        elif self.interaction_mode.startswith("resize"):
            # Zoom dependent scale, here dx is in document coordinates
            # A rough scale based on interaction dx
            scale = max(0.1, 1.0 + (dx / 100.0))
            if stype == "circle":
                new_params["radius"] = max(1.0, ip["radius"] * scale)
            elif stype == "ellipse":
                new_params["axes_a"] = max(1.0, ip["axes_a"] * scale)
                new_params["axes_b"] = max(1.0, ip["axes_b"] * scale)
            elif stype == "square":
                new_params["side"] = max(1.0, ip["side"] * scale)
            elif stype == "rectangle":
                new_params["w"] = max(1.0, ip["w"] * scale)
                new_params["h"] = max(1.0, ip["h"] * scale)
            elif stype == "triangle":
                pts = np.array(ip["pts"])
                centroid = pts.mean(axis=0)
                new_pts = centroid + (pts - centroid) * scale
                new_params["pts"] = [(pt[0], pt[1]) for pt in new_pts]
            elif stype in ("line", "arrow"):
                p0 = np.array(ip["p0"])
                p1 = np.array(ip["p1"])
                p1 = p0 + (p1 - p0) * scale
                new_params["p1"] = (p1[0], p1[1])
            elif stype == "text":
                new_params["scale"] = max(0.5, ip.get("scale", 1.0) * scale)
                
        self.active_item.params = new_params
        self.is_dirty = True
        
    def end_interaction(self) -> None:
        if self.active_item and self.interaction_initial_params:
            if isinstance(self.active_item, Stroke):
                from app.commands import ModifyStrokeCommand
                final_points = self.active_item.points.copy()
                self.active_item.points = self.interaction_initial_params["points"]
                cmd = ModifyStrokeCommand(self.active_item, self.interaction_initial_params["points"], final_points)
                self.commands.execute(cmd)
            else:
                # Revert the temporary change and push formal command
                final_params = self.active_item.params.copy()
                self.active_item.params = self.interaction_initial_params
                cmd = ModifyShapeCommand(self.active_item, self.interaction_initial_params, final_params)
                self.commands.execute(cmd)
            
        self.active_item = None
        self.interaction_mode = None
        self.interaction_start_pt = None
        self.interaction_initial_params = None
        self.is_dirty = True

    def delete_hovered_shape(self) -> bool:
        if self.hovered_item is not None:
            self.remove_item(self.hovered_item)
            self.hovered_item = None
            return True
        return False
        
    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def remove_item(self, item) -> None:
        layer_idx = 0
        for i, layer in enumerate(self.document.layers):
            if item in layer.items:
                layer_idx = i
                break
        cmd = RemoveItemCommand(item, layer_idx)
        self.commands.execute(cmd)
        self.is_dirty = True
        
    def clear(self) -> None:
        cmd = ClearDocumentCommand()
        self.commands.execute(cmd)
        self.is_dirty = True
        
    def undo(self) -> None:
        if self.commands.undo():
            self.is_dirty = True
            
    def redo(self) -> None:
        if self.commands.redo():
            self.is_dirty = True
            
    def cycle_color(self) -> None:
        try:
            idx = config.COLOR_PALETTE.index(self.brush_color)
            idx = (idx + 1) % len(config.COLOR_PALETTE)
        except ValueError:
            idx = 0
        self.brush_color = config.COLOR_PALETTE[idx]

    def set_color(self, color: Tuple[int, int, int]) -> None:
        self.brush_color = color

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_json(self) -> str:
        import json
        
        layers_data = []
        for layer in self.document.layers:
            items_data = []
            for item in layer.items:
                if isinstance(item, Stroke):
                    items_data.append({
                        "type": "stroke",
                        "points": item.points,
                        "color": item.color,
                        "thickness": item.thickness,
                        "is_eraser": item.is_eraser
                    })
                elif isinstance(item, Shape):
                    items_data.append({
                        "type": "shape",
                        "shape_type": item.shape_type,
                        "params": item.params,
                        "color": item.color,
                        "thickness": item.thickness
                    })
            layers_data.append({
                "name": layer.name,
                "visible": layer.visible,
                "locked": layer.locked,
                "items": items_data
            })
            
        data = {
            "version": 2,
            "layers": layers_data,
            "active_layer_index": self.document.active_layer_index,
            "viewport": {
                "pan_x": self.viewport.pan_x,
                "pan_y": self.viewport.pan_y,
                "zoom": self.viewport.zoom
            }
        }
        return json.dumps(data)

    def from_json(self, json_str: str) -> None:
        import json
        try:
            data = json.loads(json_str)
            self.document.layers = []
            self.commands.clear_history()
            
            for layer_data in data.get("layers", []):
                layer = Layer(layer_data.get("name", "Layer"))
                layer.visible = layer_data.get("visible", True)
                layer.locked = layer_data.get("locked", False)
                for item_data in layer_data.get("items", []):
                    if item_data.get("type") == "stroke":
                        pts = [(float(p[0]), float(p[1])) for p in item_data["points"]]
                        color = tuple(item_data["color"])
                        layer.items.append(Stroke(pts, color, item_data["thickness"], item_data.get("is_eraser", False)))
                    elif item_data.get("type") == "shape":
                        color = tuple(item_data["color"])
                        layer.items.append(Shape(item_data["shape_type"], item_data["params"], color, item_data["thickness"]))
                self.document.layers.append(layer)
                
            if not self.document.layers:
                self.document.layers.append(Layer("Layer 1"))
                
            self.document.active_layer_index = data.get("active_layer_index", 0)
            
            vp = data.get("viewport", {})
            self.viewport.pan_x = vp.get("pan_x", 0.0)
            self.viewport.pan_y = vp.get("pan_y", 0.0)
            self.viewport.zoom = vp.get("zoom", 1.0)
            
            self.is_dirty = True
        except Exception as e:
            print(f"Error loading Workspace JSON: {e}")
            self.clear()

    def get_canvas(self) -> np.ndarray:
        return self.engine.render(self.document, self.viewport, active_stroke=self._current_stroke)

    def load_canvas(self, image: np.ndarray) -> None:
        # A purely vector-based engine cannot "load" a raster image as strokes.
        # But to prevent crashing, we ignore or import as a BackgroundImage layer in the future.
        pass

    def composite(self, display_frame: np.ndarray) -> np.ndarray:
        overlay = self.get_canvas()
        mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        # In case dimensions differ, resize overlay
        if overlay.shape[:2] != display_frame.shape[:2]:
            overlay = cv2.resize(overlay, (display_frame.shape[1], display_frame.shape[0]))
            mask = cv2.resize(mask, (display_frame.shape[1], display_frame.shape[0]))
            mask_inv = cv2.resize(mask_inv, (display_frame.shape[1], display_frame.shape[0]))
            
        bg = cv2.bitwise_and(display_frame, display_frame, mask=mask_inv)
        fg = cv2.bitwise_and(overlay, overlay, mask=mask)
        return cv2.add(bg, fg)
