"""
Ekora v2.2 – main entry point with decoupled cursor tracking and new gesture model.
"""

from __future__ import annotations

import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config
from app.camera import create_camera, ThreadedCamera
from app.workspace_manager import WorkspaceManager
from app.gesture_detector import Gesture, GestureDetector
from app.hand_tracker import HandLandmarks, HandTracker, ThreadedTracker
from app.cursor_tracker import CursorTracker
from app.virtual_pen import VirtualPen
from app.ui import UIRenderer, StartupResult
from app.event_bus import EventBus, EventType
from app.plugin_system import PluginManager


class Application:
    def __init__(self) -> None:
        self.app_settings = config.load_settings()
        self.is_running = True
        self.current_workspace_path: Optional[str] = None
        self.last_frame: Optional[np.ndarray] = None
        self.app_mode: str = "Idle"
        
        # Stroke persistence state
        self.stroke_loss_start_time: Optional[float] = None
        self.lost_frame_counter: int = 0
        
        self.event_bus = EventBus()
        self._register_events()

        print("=" * 50)
        print("  Ekora – Virtual Finger Drawing v2.2")
        print("=" * 50)
        print("\nStarting camera and hand tracker…\n")

        self.camera = ThreadedCamera(create_camera())
        self.tracker = ThreadedTracker(HandTracker())
        self.cursor_tracker = CursorTracker()
        self.virtual_pen = VirtualPen()
        self.gestures = GestureDetector()
        self.canvas = WorkspaceManager()
        self.ui = UIRenderer(self.event_bus)
        self.event_bus.subscribe(EventType.FLASH_MESSAGE, self.ui.flash_message)
        
        self.plugin_manager = PluginManager(self.event_bus)
        self.plugin_manager.discover_and_load()
        
        self.window_name = config.WINDOW_NAME
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

        self._recover_autosave()

        self.ui.set_camera_mode(self.app_settings.get("camera_mode", config.CAMERA_MODE_DARK_OVERLAY))
        shape_rec = self.app_settings.get("shape_recognition", True)
        self.ui.set_shape_recognition(shape_rec)
        self.canvas.shape_recognition_enabled = shape_rec

    def _register_events(self) -> None:
        self.event_bus.subscribe(EventType.WORKSPACE_NEW, self.handle_new_workspace)
        self.event_bus.subscribe(EventType.WORKSPACE_OPEN, self.handle_open_workspace)
        self.event_bus.subscribe(EventType.WORKSPACE_SAVE, self.handle_save_workspace)
        self.event_bus.subscribe(EventType.WORKSPACE_SAVE_AS, self.handle_save_as)
        self.event_bus.subscribe(EventType.WORKSPACE_SHOW_RECENT, self.handle_recent)
        self.event_bus.subscribe(EventType.WORKSPACE_EXPORT, self.handle_export)
        self.event_bus.subscribe(EventType.SETTINGS_OPEN, self.handle_settings)
        self.event_bus.subscribe(EventType.HELP_REQUESTED, self.handle_help)
        self.event_bus.subscribe(EventType.ABOUT_REQUESTED, self.handle_about)
        self.event_bus.subscribe(EventType.BRUSH_SIZE_CHANGED, self.handle_brush_size)
        self.event_bus.subscribe(EventType.SHAPE_RECOGNITION_TOGGLED, self.handle_shape_recognition)
        self.event_bus.subscribe(EventType.PLUGIN_ADD_SHAPE, self.handle_plugin_add_shape)

    def _recover_autosave(self) -> None:
        autosave_path = Path(__file__).parent / ".autosave.ekora"
        if autosave_path.exists():
            ok, frame = self.camera.read()
            if ok and frame is not None:
                base = cv2.resize(frame, (config.CAMERA_WIDTH, config.CAMERA_HEIGHT))
                if self.ui.show_recovery_prompt(base):
                    with open(autosave_path, "r", encoding="utf-8") as f:
                        self.canvas.from_json(f.read())
                    self.canvas.is_dirty = True
                    self.ui.flash_message("Recovered workspace!")
            autosave_path.unlink()

    def _add_recent_file(self, path: str) -> None:
        recent = self.app_settings.get("recent_files", [])
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        self.app_settings["recent_files"] = recent[:5]
        config.save_settings(self.app_settings)

    def _check_unsaved(self) -> bool:
        if self.canvas.is_dirty:
            res = self.ui.show_unsaved_prompt(self.last_frame)
            if res == "save":
                self.handle_save_workspace()
                return True
            elif res == "discard":
                return True
            return False
        return True

    def handle_new_workspace(self, *args, **kwargs) -> None:
        if not self._check_unsaved(): return
        self.current_workspace_path = None
        self.canvas.clear()
        self.canvas.is_dirty = False
        self.ui.flash_message("New Workspace")

    def handle_open_workspace(self, path: str = None, *args, **kwargs) -> None:
        if not self._check_unsaved(): return
        
        if not path:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(
                title="Open Workspace",
                filetypes=[("Ekora Workspace", "*.ekora"), ("PNG Image", "*.png"), ("All files", "*.*")],
            )
            root.destroy()
            
        if path:
            try:
                if path.endswith(".ekora"):
                    with open(path, "r", encoding="utf-8") as file:
                        self.canvas.from_json(file.read())
                else:
                    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        self.canvas.load_canvas(img)
                    else:
                        self.ui.flash_message("Could not open image")
                        return
                self.current_workspace_path = path
                self._add_recent_file(path)
                self.ui.flash_message(f"Opened: {Path(path).name}")
            except Exception as e:
                self.ui.flash_message(f"Load failed: {e}")

    def handle_save_workspace(self, *args, **kwargs) -> None:
        if self.current_workspace_path:
            if self.current_workspace_path.endswith(".ekora"):
                with open(self.current_workspace_path, "w", encoding="utf-8") as f:
                    f.write(self.canvas.to_json())
            else:
                cv2.imwrite(self.current_workspace_path, self.canvas.get_canvas())
            self.canvas.is_dirty = False
            self.ui.flash_message(f"Saved: {Path(self.current_workspace_path).name}")
        else:
            self.handle_save_as()

    def handle_save_as(self, *args, **kwargs) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.asksaveasfilename(
                title="Save As",
                defaultextension=".ekora",
                filetypes=[("Ekora Workspace", "*.ekora"), ("PNG Image", "*.png"), ("All files", "*.*")],
            )
            root.destroy()
            if path:
                if path.endswith(".ekora"):
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(self.canvas.to_json())
                else:
                    cv2.imwrite(path, self.canvas.get_canvas())
                self.current_workspace_path = path
                self.canvas.is_dirty = False
                self._add_recent_file(path)
                self.ui.flash_message(f"Saved: {Path(path).name}")
        except Exception as e:
            self.ui.flash_message(f"Save failed: {e}")

    def handle_recent(self, *args, **kwargs) -> None:
        recent = self.app_settings.get("recent_files", [])
        path = self.ui.show_recent_files(self.last_frame, recent)
        if path:
            self.handle_open_workspace(path)

    def handle_export(self, *args, **kwargs) -> None:
        res = self.ui.show_export_prompt(self.last_frame)
        if not res: return
        
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        ext = res["format"]
        path = filedialog.asksaveasfilename(
            title="Export As",
            defaultextension=f".{ext}",
            filetypes=[(f"{ext.upper()} Image", f"*.{ext}")],
        )
        root.destroy()
        if path:
            if res["bg"]:
                sw = config.SIDEBAR_WIDTH
                comp = self.last_frame.copy()
                roi = comp[:, sw:]
                canvas_roi = self.canvas.get_canvas()[:, sw:]
                mask = cv2.cvtColor(canvas_roi, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                fg = cv2.bitwise_and(canvas_roi, canvas_roi, mask=mask)
                blended = cv2.add(bg, fg)
                comp[:, sw:] = blended
                cv2.imwrite(path, comp)
            else:
                cv2.imwrite(path, self.canvas.get_canvas())
            self.ui.flash_message(f"Exported: {Path(path).name}")

    def handle_settings(self, *args, **kwargs) -> None:
        settings = self.ui.show_settings_prompt(self.last_frame)
        self.app_settings.update(settings)
        self.ui.set_camera_mode(settings.get("camera_mode", config.CAMERA_MODE_DARK_OVERLAY))
        self.ui.set_shape_recognition(settings.get("shape_recognition", True))
        self.canvas.shape_recognition_enabled = settings.get("shape_recognition", True)
        self.ui.flash_message("Settings Updated")

    def handle_help(self, *args, **kwargs) -> None:
        shortcuts = [
            "Ctrl+N : New Workspace",
            "Ctrl+O : Open Workspace",
            "Ctrl+S : Save Workspace",
            "Ctrl+Shift+S : Save As",
            "Ctrl+Z : Undo",
            "Ctrl+Y : Redo",
            "Delete : Clear Canvas",
            "Esc    : Quit",
        ]
        self.ui.show_info_dialog(self.last_frame, "Keyboard Shortcuts", shortcuts)

    def handle_about(self, *args, **kwargs) -> None:
        lines = [
            "Ekora",
            "Think. Sketch. Create.",
            "Version 2.2",
            "Developer: AI Assistant",
            "GitHub repository: (Local)",
            "License: MIT",
        ]
        self.ui.show_info_dialog(self.last_frame, "About", lines)

    def handle_brush_size(self, new_size: int) -> None:
        self.canvas.brush_size = new_size
        self.canvas.eraser_size = new_size

    def handle_shape_recognition(self, enabled: bool) -> None:
        self.canvas.shape_recognition_enabled = enabled
        
    def handle_canvas_clear(self, *args, **kwargs) -> None:
        self.canvas.clear()
        self.ui.flash_message("Canvas Cleared")

    def handle_plugin_add_shape(self, shape_type: str, params: dict, color: tuple, size: int) -> None:
        from app.model import Shape
        from app.commands import AddItemCommand
        shape = Shape(shape_type, params, color, size)
        self.canvas.commands.execute(AddItemCommand(shape, self.canvas.document.active_layer_index))
        self.canvas.is_dirty = True

    def run(self) -> None:
        startup_result = self.ui.show_startup(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
        if startup_result == StartupResult.QUIT:
            self.shutdown()
            return

        if startup_result == StartupResult.OPEN:
            self.handle_open_workspace()
        elif startup_result not in (StartupResult.NEW, StartupResult.QUIT):
            self.handle_open_workspace(startup_result)

        def _mouse_handler(event, mx, my, flags, param):
            self.ui.handle_mouse(event, mx, my, flags, self.canvas, self.canvas.brush_size)

        cv2.setMouseCallback(self.window_name, _mouse_handler)

        print("Controls:")
        print("  Point (index finger)   → Draw")
        print("  Pinch (thumb + index)  → Interact/Move")
        print("  Fist                   → Pause drawing")
        print("  Peace sign             → Next colour")
        print("  Open palm              → Erase canvas")
        print("  Q / ESC                → Quit")
        print("  C                      → Clear canvas")
        print("  S                      → Quick save\n")

        prev_gesture = Gesture.NONE
        
        try:
            while self.is_running:
                ok, frame = self.camera.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue

                if frame.shape[1] != config.CAMERA_WIDTH or frame.shape[0] != config.CAMERA_HEIGHT:
                    frame = cv2.resize(frame, (config.CAMERA_WIDTH, config.CAMERA_HEIGHT))

                self.tracker.update_frame(frame)
                hand = self.tracker.get_latest_hand()
                # Update smooth cursor tracking
                raw_pos, tracker_pos, velocity = self.cursor_tracker.update(hand)
                
                if raw_pos is not None:
                    pen_pos = self.virtual_pen.process(raw_pos)
                else:
                    self.virtual_pen.reset()
                    pen_pos = tracker_pos
                
                gesture = Gesture.NONE
                confidence = 0.0
                finger_states = {}

                if hand is not None:
                    detected_gesture, confidence, finger_states = self.gestures.detect(hand, frame.shape[:2])
                else:
                    detected_gesture = Gesture.NONE
                    
                # Stroke Persistence Logic
                if self.gestures.stroke_active and detected_gesture != Gesture.POINT:
                    if self.stroke_loss_start_time is None:
                        self.stroke_loss_start_time = time.time()
                    self.lost_frame_counter += 1
                    
                    if time.time() - self.stroke_loss_start_time < 0.250:
                        gesture = Gesture.POINT
                    else:
                        gesture = detected_gesture
                        self.stroke_loss_start_time = None
                else:
                    gesture = detected_gesture
                    if gesture == Gesture.POINT:
                        self.stroke_loss_start_time = None
                        self.lost_frame_counter = 0

                if hand is not None or pen_pos is not None:
                    self.canvas.update_hover(pen_pos)
                    self._handle_gesture(gesture, prev_gesture, hand, pen_pos)
                    prev_gesture = gesture
                else:
                    # Complete loss of tracking
                    self.canvas.end_stroke()
                    self.gestures.stroke_active = False
                    prev_gesture = Gesture.NONE

                display_frame = self.ui.apply_camera_mode(frame)

                if hand is not None:
                    self.tracker.draw_skeleton(display_frame, hand)

                composited = self.canvas.composite(display_frame)

                stroke_id = id(self.canvas._current_stroke) if self.canvas._current_stroke else 0
                
                output = self.ui.render(
                    composited,
                    gesture       = gesture,
                    brush_color   = self.canvas.brush_color,
                    brush_size    = self.canvas.brush_size,
                    hand_detected = hand is not None or pen_pos is not None,
                    cursor_pos    = pen_pos,
                    eraser_size   = self.canvas.eraser_size,
                    confidence    = confidence,
                    finger_states = finger_states,
                    app_mode      = self.app_mode,
                    raw_pos       = raw_pos,
                    velocity      = velocity,
                    raw_gesture   = self.gestures._candidate_raw,
                    stroke_active = self.gestures.stroke_active,
                    stroke_id     = stroke_id,
                    lost_frame_counter = self.lost_frame_counter,
                    prev_gesture_name = prev_gesture.name,
                    pinch_distance = self.gestures.last_pinch_distance,
                    selected_object = self.canvas.active_item.__class__.__name__ if getattr(self.canvas, 'active_item', None) else getattr(self.canvas, 'hovered_item', None).__class__.__name__ if getattr(self.canvas, 'hovered_item', None) else "None",
                )

                cv2.imshow(self.window_name, output)
                self.last_frame = output.copy()
                
                if self.app_settings.get("auto_save", True):
                    if not hasattr(self.camera, "_last_autosave"):
                        self.camera._last_autosave = time.time()
                    if time.time() - self.camera._last_autosave > self.app_settings.get("auto_save_interval", 300):
                        if self.canvas.is_dirty:
                            def do_autosave(json_data, path):
                                try:
                                    with open(path, "w", encoding="utf-8") as f:
                                        f.write(json_data)
                                except Exception:
                                    pass
                            
                            autosave_path = Path(__file__).parent / ".autosave.ekora"
                            json_data = self.canvas.to_json()
                            threading.Thread(target=do_autosave, args=(json_data, autosave_path), daemon=True).start()
                            
                            self.ui.flash_message("Autosave completed")
                        self.camera._last_autosave = time.time()

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    if self._check_unsaved():
                        self.is_running = False
                elif key == 14:
                    self.handle_new_workspace()
                elif key == 15:
                    self.handle_open_workspace()
                elif key == 19:
                    self.handle_save_workspace()
                elif key == 26:
                    self.canvas.undo()
                elif key == 25:
                    self.canvas.redo()
                elif key == 127:
                    if not self.canvas.delete_hovered_shape():
                        self.handle_canvas_clear()
                elif key in (ord("c"), ord("C")):
                    self.handle_canvas_clear()
                elif key in (ord("o"), ord("O")):
                    self.event_bus.publish(EventType.PLUGIN_OCR_REQUESTED, image=self.canvas.get_canvas())
                elif key in (ord("d"), ord("D")):
                    self.event_bus.publish(EventType.PLUGIN_DIAGRAM_REQUESTED, image=self.canvas.get_canvas())
                elif key in (ord("s"), ord("S")):
                    pass
                elif key in (ord("r"), ord("R")):
                    self.canvas.set_color(config.DEFAULT_BRUSH_COLOR)
                elif key == ord("="): # Plus key for zoom in
                    self.canvas.viewport.zoom *= 1.1
                    self.canvas.is_dirty = True
                elif key == ord("-"): # Minus key for zoom out
                    self.canvas.viewport.zoom /= 1.1
                    self.canvas.is_dirty = True
                # Arrow keys for pan (OpenCV keys vary by OS, but let's use WASD for reliability)
                elif key in (ord("w"), ord("W")):
                    self.canvas.viewport.pan_y += 50
                    self.canvas.is_dirty = True
                elif key in (ord("s"), ord("S")):
                    self.canvas.viewport.pan_y -= 50
                    self.canvas.is_dirty = True
                elif key in (ord("a"), ord("A")):
                    self.canvas.viewport.pan_x += 50
                    self.canvas.is_dirty = True
                elif key in (ord("d"), ord("D")):
                    self.canvas.viewport.pan_x -= 50
                    self.canvas.is_dirty = True

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            self.shutdown()

    def _handle_gesture(
        self,
        gesture: Gesture,
        prev_gesture: Gesture,
        hand: Optional[HandLandmarks],
        cursor_pos: Optional[tuple[int, int]],
    ) -> None:
        if not cursor_pos:
            return

        # 0. Global state cleanup on transition
        if prev_gesture != gesture:
            # Exiting PINCH
            if prev_gesture == Gesture.PINCH:
                if getattr(self.canvas, 'active_item', None) is not None:
                    self.canvas.end_interaction()
                else:
                    self.canvas.end_pan()
            # Exiting POINT (Stroke ends unless gesture locked by persistence)
            if prev_gesture == Gesture.POINT:
                self.canvas.end_stroke()
                self.gestures.stroke_active = False

        # 1. PINCH -> MOVE MODE (Strictly no drawing)
        if gesture == Gesture.PINCH:
            self.app_mode = "Move"
            if prev_gesture != Gesture.PINCH:
                if getattr(self.canvas, 'hovered_item', None) is not None:
                    self.canvas.start_interaction(cursor_pos)
                else:
                    self.canvas.start_pan(cursor_pos)
            else:
                if getattr(self.canvas, 'active_item', None) is not None:
                    self.canvas.continue_interaction(cursor_pos)
                else:
                    self.canvas.continue_pan(cursor_pos)

        # 2. OPEN PALM -> CLEAR
        elif gesture == Gesture.OPEN_PALM:
            self.app_mode = "Erase"
            if prev_gesture != Gesture.OPEN_PALM:
                self.handle_canvas_clear()
                self.ui.flash_message("Canvas Cleared!")
                
        # 3. FIST -> PAUSE MODE (Strictly pause, no erase/delete)
        elif gesture == Gesture.FIST:
            self.app_mode = "Pause"

        # 4. POINT -> DRAW MODE (Strictly index finger only)
        elif gesture == Gesture.POINT:
            self.app_mode = "Brush"
            if prev_gesture != Gesture.POINT:
                self.canvas.start_stroke(cursor_pos)
                self.gestures.stroke_active = True
            else:
                self.canvas.continue_stroke(cursor_pos, eraser=False)
                
        # 5. PEACE (Cycle Color)
        elif gesture == Gesture.PEACE:
            self.app_mode = "Idle"
            if prev_gesture != Gesture.PEACE:
                self.canvas.cycle_color()
                self.ui.flash_message("Color Changed!")
                
        # 6. IDLE / NONE
        else:
            self.app_mode = "Idle"

    def shutdown(self) -> None:
        print("Shutting down…")
        self.plugin_manager.shutdown_all()
        self.tracker.close()
        self.camera.release()
        cv2.destroyAllWindows()
        print("Goodbye!")


def main() -> None:
    app = Application()
    app.run()


if __name__ == "__main__":
    main()
