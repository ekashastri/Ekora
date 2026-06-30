"""
Ekora v1.3 – main entry point.

Performance changes in this file vs the original v1.3
------------------------------------------------------
- Camera frames are grabbed in a background thread so the main loop
  never blocks waiting for the next frame; latency drops by one frame.
- Removed the redundant frame.copy() before gesture detection — the
  original frame is only read (not written) during tracking.
- apply_camera_mode() receives the frame and returns a modified copy
  internally only when a copy is actually needed.
"""

from __future__ import annotations

import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

import config
from app.camera import create_camera
from app.drawing_canvas import DrawingCanvas
from app.gesture_detector import Gesture, GestureDetector
from app.hand_tracker import HandLandmarks, HandTracker, INDEX_TIP, INDEX_MCP
from app.ui import UIRenderer, StartupResult


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Globals shared with callbacks
# ---------------------------------------------------------------------------
_canvas: DrawingCanvas
_ui: UIRenderer
_current_workspace_path = None
_last_frame = None

def _add_recent_file(path: str):
    settings = config.load_settings()
    recent = settings.get("recent_files", [])
    if path in recent:
        recent.remove(path)
    recent.insert(0, path)
    settings["recent_files"] = recent[:5]
    config.save_settings(settings)

def _check_unsaved() -> bool:
    """Returns True if we can proceed (saved or discarded), False to cancel."""
    if _canvas.is_dirty:
        res = _ui.show_unsaved_prompt(_last_frame)
        if res == "save":
            _cb_save()
            return True
        elif res == "discard":
            return True
        return False
    return True

def _cb_new() -> None:
    if not _check_unsaved(): return
    global _current_workspace_path
    _current_workspace_path = None
    _canvas.clear()
    _canvas.is_dirty = False
    _ui.flash_message("New Workspace")

def _cb_open(path: str = None) -> None:
    if not _check_unsaved(): return
    global _current_workspace_path
    
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
                    _canvas.from_json(file.read())
            else:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    _canvas.load_canvas(img)
                else:
                    _ui.flash_message("Could not open image")
                    return
            _current_workspace_path = path
            _add_recent_file(path)
            _ui.flash_message(f"Opened: {Path(path).name}")
        except Exception as e:
            _ui.flash_message(f"Load failed: {e}")

def _cb_save() -> None:
    global _current_workspace_path
    if _current_workspace_path:
        if _current_workspace_path.endswith(".ekora"):
            with open(_current_workspace_path, "w", encoding="utf-8") as f:
                f.write(_canvas.to_json())
        else:
            cv2.imwrite(_current_workspace_path, _canvas.get_canvas())
        _canvas.is_dirty = False
        _ui.flash_message(f"Saved: {Path(_current_workspace_path).name}")
    else:
        _cb_save_as()

def _cb_save_as() -> None:
    global _current_workspace_path
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
                    f.write(_canvas.to_json())
            else:
                cv2.imwrite(path, _canvas.get_canvas())
            _current_workspace_path = path
            _canvas.is_dirty = False
            _add_recent_file(path)
            _ui.flash_message(f"Saved: {Path(path).name}")
    except Exception as e:
        _ui.flash_message(f"Save failed: {e}")

def _cb_recent() -> None:
    settings = config.load_settings()
    recent = settings.get("recent_files", [])
    path = _ui.show_recent_files(_last_frame, recent)
    if path:
        _cb_open(path)

def _cb_export() -> None:
    res = _ui.show_export_prompt(_last_frame)
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
            # export composited frame
            sw = config.SIDEBAR_WIDTH
            comp = _last_frame.copy()
            roi = comp[:, sw:]
            canvas_roi = _canvas._canvas[:, sw:]
            # Simple composite for export
            mask = cv2.cvtColor(canvas_roi, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            fg = cv2.bitwise_and(canvas_roi, canvas_roi, mask=mask)
            blended = cv2.add(bg, fg)
            comp[:, sw:] = blended
            cv2.imwrite(path, comp)
        else:
            cv2.imwrite(path, _canvas.get_canvas())
        _ui.flash_message(f"Exported: {Path(path).name}")

def _cb_settings() -> None:
    global _app_settings
    settings = _ui.show_settings_prompt(_last_frame)
    _app_settings.update(settings)
    _ui.set_camera_mode(settings.get("camera_mode", config.CAMERA_MODE_DARK_OVERLAY))
    _ui.set_shape_recognition(settings.get("shape_recognition", True))
    _canvas.shape_recognition_enabled = settings.get("shape_recognition", True)
    _ui.flash_message("Settings Updated")

def _cb_help() -> None:
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
    _ui.show_info_dialog(_last_frame, "Keyboard Shortcuts", shortcuts)

def _cb_about() -> None:
    lines = [
        "Ekora",
        "Think. Sketch. Create.",
        "Version 1.8",
        "Developer: AI Assistant",
        "GitHub repository: (Local)",
        "License: MIT",
    ]
    _ui.show_info_dialog(_last_frame, "About", lines)

def _cb_brush_size(new_size: int) -> None:
    pass  # canvas already updated inside handle_mouse


def _cb_shape_recognition(enabled: bool) -> None:
    _canvas.shape_recognition_enabled = enabled


def _save_canvas_auto() -> None:
    output_dir = Path(__file__).parent / "saves"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath  = output_dir / f"drawing_{timestamp}.png"
    cv2.imwrite(str(filepath), _canvas.get_canvas())
    _ui.flash_message(f"Saved → {filepath.name}")
    print(f"Drawing saved → {filepath}")



# ---------------------------------------------------------------------------
# Threaded Tracker
# ---------------------------------------------------------------------------
class _ThreadedTracker:
    """
    Runs MediaPipe inference in a background thread to prevent blocking 
    the main OpenCV render loop.
    """
    def __init__(self, tracker) -> None:
        self.tracker = tracker
        self._frame = None
        self._hand = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._new_frame_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def update_frame(self, frame):
        with self._lock:
            # Drop frame if one is already being processed to keep latency low
            self._frame = frame.copy()
        self._new_frame_event.set()

    def get_latest_hand(self):
        with self._lock:
            return self._hand

    def _worker(self):
        while not self._stop.is_set():
            if self._new_frame_event.wait(timeout=0.1):
                self._new_frame_event.clear()
                with self._lock:
                    if self._frame is None:
                        continue
                    frame_to_process = self._frame
                
                hand = self.tracker.process(frame_to_process)
                
                with self._lock:
                    self._hand = hand

    def close(self):
        self._stop.set()
        self._new_frame_event.set()
        self._thread.join(timeout=1.0)
        self.tracker.close()

# ---------------------------------------------------------------------------
# Threaded camera reader
# ---------------------------------------------------------------------------
class _ThreadedCamera:
    """
    Reads camera frames in a daemon thread so the main loop always gets
    the most recent frame without blocking on VideoCapture.read().
    """

    def __init__(self, camera) -> None:
        self._cam   = camera
        self._frame = None
        self._ok    = False
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cam.read()
            if ok:
                with self._lock:
                    self._frame = frame
                    self._ok    = True
            else:
                import time
                time.sleep(0.01)

    def read(self):
        with self._lock:
            return self._ok, (self._frame.copy() if self._frame is not None else None)

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._cam.release()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


_app_settings = {}

def main() -> None:
    global _app_settings
    _app_settings = config.load_settings()
    global _canvas, _ui

    print("=" * 50)
    print("  Ekora – Virtual Finger Drawing v1.5")
    print("=" * 50)
    print("\nStarting camera and hand tracker…\n")

    camera   = _ThreadedCamera(create_camera())
    tracker  = HandTracker()
    gestures = GestureDetector()
    _canvas  = DrawingCanvas()
    _ui      = UIRenderer()
    
    autosave_path = Path(__file__).parent / ".autosave.ekora"
    if autosave_path.exists():
        # Read a dummy frame to show the modal
        ok, frame = camera.read()
        if ok and frame is not None:
            base = cv2.resize(frame, (config.CAMERA_WIDTH, config.CAMERA_HEIGHT))
            if _ui.show_recovery_prompt(base):
                with open(autosave_path, "r", encoding="utf-8") as f:
                    _canvas.from_json(f.read())
                _canvas.is_dirty = True
                _ui.flash_message("Recovered workspace!")
        autosave_path.unlink() # remove after prompt

    settings = config.load_settings()
    _ui.set_camera_mode(settings.get("camera_mode", config.CAMERA_MODE_DARK_OVERLAY))
    shape_rec = settings.get("shape_recognition", True)
    _ui.set_shape_recognition(shape_rec)
    _canvas.shape_recognition_enabled = shape_rec

    _ui.register_callbacks(
        on_new        = _cb_new,
        on_open       = _cb_open,
        on_save       = _cb_save,
        on_save_as    = _cb_save_as,
        on_brush_size = _cb_brush_size,
        on_shape_recognition = _cb_shape_recognition,
        on_recent     = _cb_recent,
        on_export     = _cb_export,
        on_settings   = _cb_settings,
        on_help       = _cb_help,
        on_about      = _cb_about,
    )

    prev_gesture = Gesture.NONE
    window = config.WINDOW_NAME
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

    # Startup screen
    startup_result = _ui.show_startup(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    if startup_result == StartupResult.QUIT:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()
        return

    if startup_result == StartupResult.OPEN:
        _cb_open()
    elif startup_result not in (StartupResult.NEW, StartupResult.QUIT):
        _cb_open(startup_result)

    def _mouse_handler(event, mx, my, flags, param):
        _ui.handle_mouse(event, mx, my, flags, _canvas, _canvas.brush_size)

    cv2.setMouseCallback(window, _mouse_handler)

    print("Controls:")
    print("  Pinch (thumb + index)  → Draw")
    print("  Point (index finger)   → Hover")
    print("  Fist                   → Erase")
    print("  Peace sign             → Next colour")
    print("  Open palm              → Clear canvas")
    print("  Q / ESC                → Quit")
    print("  C                      → Clear canvas")
    print("  S                      → Quick save\n")

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            if frame.shape[1] != config.CAMERA_WIDTH or frame.shape[0] != config.CAMERA_HEIGHT:
                frame = cv2.resize(frame, (config.CAMERA_WIDTH, config.CAMERA_HEIGHT))

            # Gesture detection runs asynchronously
            tracker.update_frame(frame)
            hand = tracker.get_latest_hand()
            gesture  = Gesture.NONE
            cursor_pos = None

            if hand is not None:
                gesture    = gestures.detect(hand, frame.shape[:2])
                # Inform the state machine whether a stroke is in progress
                gestures.stroke_active = (prev_gesture == Gesture.PINCH)
                cursor_pos = (hand.landmarks[INDEX_MCP] if gesture == Gesture.FIST
                              else hand.landmarks[INDEX_TIP])
                _handle_gesture(gesture, prev_gesture, hand, _canvas, _ui)
                prev_gesture = gesture
            else:
                # Only end the stroke once persistence has fully expired
                # (tracker returns None only after the timeout window closes)
                _canvas.end_stroke()
                gestures.stroke_active = False
                prev_gesture = Gesture.NONE

            # Build display frame according to camera mode
            display_frame = _ui.apply_camera_mode(frame)

            # Draw skeleton onto display (after mode transform)
            if hand is not None:
                tracker.draw_skeleton(display_frame, hand)

            # Composite drawing strokes onto display frame
            composited = _canvas.composite(display_frame)

            # Render all UI chrome
            output = _ui.render(
                composited,
                gesture       = gesture,
                brush_color   = _canvas.brush_color,
                brush_size    = _canvas.brush_size,
                hand_detected = hand is not None,
                cursor_pos    = cursor_pos,
                eraser_size   = _canvas.eraser_size,
            )

            cv2.imshow(window, output)

            global _last_frame
            _last_frame = output.copy()
            
            # Autosave logic
            if _app_settings.get("auto_save", True):
                if not hasattr(camera, "_last_autosave"):
                    camera._last_autosave = time.time()
                if time.time() - camera._last_autosave > _app_settings.get("auto_save_interval", 300):
                    if _canvas.is_dirty:
                        def do_autosave(json_data, path):
                            try:
                                with open(path, "w", encoding="utf-8") as f:
                                    f.write(json_data)
                            except Exception:
                                pass
                        
                        autosave_path = Path(__file__).parent / ".autosave.ekora"
                        # Generate JSON synchronously (takes <10ms), but write async
                        json_data = _canvas.to_json()
                        threading.Thread(target=do_autosave, args=(json_data, autosave_path), daemon=True).start()
                        
                        _ui.flash_message("Autosave completed")
                    camera._last_autosave = time.time()

            key = cv2.waitKey(1) & 0xFF
            if key == 27: # Esc
                if _check_unsaved():
                    break
            elif key == 14: # Ctrl+N
                _cb_new()
            elif key == 15: # Ctrl+O
                _cb_open()
            elif key == 19: # Ctrl+S
                _cb_save()
            elif key == 26: # Ctrl+Z
                _canvas.undo()
            elif key == 25: # Ctrl+Y
                _canvas.redo()
            elif key == 127: # Delete
                _canvas.clear()
                _ui.flash_message("Canvas Cleared")
            elif key in (ord("c"), ord("C")):
                _canvas.clear()
            elif key in (ord("s"), ord("S")): # keeping S for quicksave if they want, but Ctrl+S is save
                pass
            elif key in (ord("r"), ord("R")):
                _canvas.set_color(config.DEFAULT_BRUSH_COLOR)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("Shutting down…")
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()
        print("Goodbye!")


def _handle_gesture(
    gesture:      Gesture,
    prev_gesture: Gesture,
    hand:         HandLandmarks,
    canvas:       DrawingCanvas,
    ui:           UIRenderer,
) -> None:
    tip         = hand.landmarks[INDEX_TIP]
    erase_point = hand.landmarks[INDEX_MCP]

    if gesture == Gesture.PINCH:
        if prev_gesture != Gesture.PINCH:
            # If we are hovering a shape, start interaction instead of stroke
            if cursor_pos and canvas.hovered_shape_index is not None:
                canvas.start_interaction(cursor_pos)
            else:
                canvas.start_stroke(tip)
        elif not getattr(hand, "is_ghost", False):
            if canvas.active_shape_index is not None and cursor_pos:
                canvas.continue_interaction(cursor_pos)
            else:
                canvas.continue_stroke(tip, eraser=False)

    elif gesture == Gesture.FIST:
        if prev_gesture == Gesture.PINCH:
            # Transitioning from drawing to erasing: end the draw stroke cleanly
            if canvas.active_shape_index is not None:
                canvas.end_interaction()
            else:
                canvas.end_stroke()
                
        if prev_gesture != Gesture.FIST:
            if cursor_pos and canvas.hovered_shape_index is not None:
                canvas.delete_hovered_shape()
                ui.flash_message("Shape Deleted")
            else:
                canvas.start_stroke(erase_point)
        elif not getattr(hand, "is_ghost", False):
            if canvas.active_shape_index is None and canvas.hovered_shape_index is None:
                canvas.continue_stroke(erase_point, eraser=True)

    elif gesture == Gesture.OPEN_PALM:
        # OPEN_PALM fires only once per confirmation cycle (state machine
        # auto-resets to IDLE the frame after emitting it).
        if prev_gesture != Gesture.OPEN_PALM:
            canvas.clear()
            ui.flash_message("Canvas Cleared!")

    elif gesture == Gesture.PEACE:
        if prev_gesture == Gesture.PINCH:
            canvas.end_stroke()
        if prev_gesture != Gesture.PEACE:
            canvas.cycle_color()
            ui.flash_message("Color Changed!")

    else:
        # NONE / HOVER: end any active stroke cleanly but don't
        # call end_stroke on every hover frame — only on transitions.
        if prev_gesture in (Gesture.PINCH, Gesture.FIST):
            if canvas.active_shape_index is not None:
                canvas.end_interaction()
            else:
                canvas.end_stroke()


if __name__ == "__main__":
    main()
