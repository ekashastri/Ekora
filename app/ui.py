"""
UI renderer module – Ekora v1.3.

Changes from v1.2
-----------------
- Startup: solid dark background, no grid, no "Press ESC to quit",
  perfectly centred logo + buttons.
- Topbar: all menu items (New, Open, Save, Save As, Settings) are
  clickable, trigger callbacks, and track hover state.
- Brush slider: mouse-draggable across the full track; immediately
  updates DrawingCanvas.brush_size; displayed value always matches.
- Camera Modes: Normal / Dark Overlay / Black Canvas implemented.
  Camera mode moved into a modal Settings panel.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

import config
from app.gesture_detector import GESTURE_LABELS, Gesture

# ---------------------------------------------------------------------------
# Design tokens (BGR throughout)
# ---------------------------------------------------------------------------
BG_DEEP        = (20, 15, 13)
BG_SIDEBAR     = (34, 26, 22)
BG_CARD        = (48, 35, 30)
BG_CARD_HOVER  = (62, 47, 40)
ACCENT         = (150, 200, 0)
ACCENT_DIM     = (100, 155, 0)
TEXT_PRIMARY   = (240, 234, 232)
TEXT_SECONDARY = (195, 157, 139)
TEXT_MUTED     = (130, 105, 95)
DIVIDER        = (75, 60, 55)
TOAST_BG       = (42, 34, 30)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


# ---------------------------------------------------------------------------
# Toast dataclass
# ---------------------------------------------------------------------------
@dataclass
class Toast:
    text: str
    ttl: float
    max_ttl: float


# ---------------------------------------------------------------------------
# Startup screen result
# ---------------------------------------------------------------------------
class StartupResult:
    NEW  = "new"
    OPEN = "open"
    QUIT = "quit"


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------
class UIRenderer:
    """Draws all non-canvas UI chrome onto the output frame."""

    def __init__(self) -> None:
        self._fps         = 0.0
        self._frame_count = 0
        self._fps_timer   = time.time()
        self._toasts: deque[Toast] = deque()

        # Brush slider drag state
        self._slider_dragging  = False
        self._slider_track     = (0, 0, 0, 0)   # x1, y1, x2, y2

        # Topbar button hit-rects: list of (x1,y1,x2,y2, label)
        self._topbar_rects: list = []
        self._topbar_hover: str  = ""

        # Settings panel
        
        self._settings_rects:  list = []   # (x1,y1,x2,y2, mode_key)
        self._settings_close_rect = (0,0,0,0)
        self._settings_toggle_rect = (0,0,0,0)
        self._shape_recognition_enabled: bool = True

        # Callbacks – set by main.py via register_*
        self._cb_new:       Optional[Callable] = None
        self._cb_open:      Optional[Callable] = None
        self._cb_save:      Optional[Callable] = None
        self._cb_save_as:   Optional[Callable] = None
        self._cb_brush_size: Optional[Callable[[int], None]] = None
        self._cb_shape_recognition: Optional[Callable[[bool], None]] = None
        self._camera_mode   = config.CAMERA_MODE_DARK_OVERLAY

        # Vignette mask is expensive to compute; cache it per frame size
        self._vignette_mask  = None   # np.ndarray or None
        self._vignette_shape = (0, 0) # (rows, cols) of cached ROI

    # ------------------------------------------------------------------
    # Public registration hooks (called by main.py)
    # ------------------------------------------------------------------

    def register_callbacks(
        self,
        on_new:      Callable,
        on_open:     Callable,
        on_save:     Callable,
        on_save_as:  Callable,
        on_brush_size: Callable[[int], None],
        on_shape_recognition: Optional[Callable[[bool], None]] = None,
        on_recent: Optional[Callable] = None,
        on_export: Optional[Callable] = None,
        on_help: Optional[Callable] = None,
        on_about: Optional[Callable] = None,
        on_settings: Optional[Callable] = None,
    ) -> None:
        self._cb_new       = on_new
        self._cb_open      = on_open
        self._cb_save      = on_save
        self._cb_save_as   = on_save_as
        self._cb_brush_size = on_brush_size
        self._cb_shape_recognition = on_shape_recognition
        self._cb_recent    = on_recent
        self._cb_export    = on_export
        self._cb_help      = on_help
        self._cb_about     = on_about
        self._cb_settings  = on_settings

    def set_camera_mode(self, mode: str) -> None:
        self._camera_mode = mode

    @property
    def camera_mode(self) -> str:
        return self._camera_mode

    def set_shape_recognition(self, enabled: bool) -> None:
        self._shape_recognition_enabled = enabled

    @property
    def shape_recognition_enabled(self) -> bool:
        return self._shape_recognition_enabled

    # ------------------------------------------------------------------
    # Startup screen
    # ------------------------------------------------------------------

    def show_startup(self, frame_w: int, frame_h: int) -> str:
        """
        Blocking startup screen loop.
        Returns StartupResult.NEW, StartupResult.OPEN, or StartupResult.QUIT.
        """
        window = config.WINDOW_NAME
        result = [StartupResult.NEW]

        btn_w, btn_h = 200, 48
        cx = frame_w // 2
        # Centre the two buttons with a 32 px gap between them
        total_btns_w = btn_w * 2 + 32
        btn_left_x   = cx - total_btns_w // 2
        btn_new_rect  = (btn_left_x,            frame_h // 2 + 60,
                         btn_left_x + btn_w,     frame_h // 2 + 60 + btn_h)
        btn_open_rect = (btn_left_x + btn_w + 32, frame_h // 2 + 60,
                         btn_left_x + btn_w + 32 + btn_w, frame_h // 2 + 60 + btn_h)

        hover_btn = -1

        def on_mouse(event, mx, my, flags, param):
            nonlocal hover_btn, result
            in_new  = (btn_new_rect[0]  <= mx <= btn_new_rect[2]  and
                       btn_new_rect[1]  <= my <= btn_new_rect[3])
            in_open = (btn_open_rect[0] <= mx <= btn_open_rect[2] and
                       btn_open_rect[1] <= my <= btn_open_rect[3])
            hover_btn = 0 if in_new else (1 if in_open else -1)
            settings = config.load_settings()
            recent_files = settings.get("recent_files", [])
            
            if hover_btn == -1 and recent_files:
                ry = frame_h // 2 + 130
                for i, rpath in enumerate(recent_files[:3]):
                    y_pos = ry + 40 + i * 35
                    if cx - 150 <= mx <= cx + 150 and y_pos - 15 <= my <= y_pos + 15:
                        hover_btn = 10 + i
                        break
                        
            if event == cv2.EVENT_LBUTTONDOWN:
                if in_new:
                    result[0] = StartupResult.NEW
                    cv2.setMouseCallback(window, lambda *a: None)
                elif in_open:
                    result[0] = StartupResult.OPEN
                    cv2.setMouseCallback(window, lambda *a: None)
                elif hover_btn >= 10:
                    result[0] = recent_files[hover_btn - 10]
                    cv2.setMouseCallback(window, lambda *a: None)
                else:
                    return
                hover_btn = 99

        cv2.setMouseCallback(window, on_mouse)

        alpha  = 0.0
        phase  = "fade_in"
        sel_a  = 0.0

        while True:
            # Solid dark background – no grid, no gradient variation
            canvas = np.full((frame_h, frame_w, 3), (18, 14, 12), dtype=np.uint8)

            # Subtle teal orb behind logo
            orb_center = (cx, frame_h // 2 - 80)
            for r_step in range(120, 0, -4):
                intensity = int(12 * (1 - r_step / 120))
                cv2.circle(canvas, orb_center, r_step,
                           (intensity * 5, intensity * 8, intensity * 3), -1)

            # Fade-in
            if phase == "fade_in":
                alpha = min(1.0, alpha + 0.04)
                if alpha >= 1.0:
                    phase = "idle"
            elif phase in ("selected", "fade_out"):
                sel_a = min(1.0, sel_a + 0.07)

            frame_out = (canvas * alpha).astype(np.uint8)

            # Logo – perfectly centred
            logo_y = frame_h // 2 - 70
            self._put_text_centered(frame_out, "EKORA", cx, logo_y,
                                    1.8, TEXT_PRIMARY, 3, font=FONT_BOLD)
            uw = 80
            cv2.line(frame_out, (cx - uw, logo_y + 14), (cx + uw, logo_y + 14),
                     ACCENT, 2, cv2.LINE_AA)
            self._put_text_centered(frame_out, "Think. Sketch. Create.",
                                    cx, logo_y + 42, 0.52, TEXT_SECONDARY, 1)

            # Buttons
            self._draw_startup_btn(frame_out, "New Workspace",
                                   btn_new_rect,  hover_btn == 0, primary=True)
            self._draw_startup_btn(frame_out, "Open Workspace",
                                   btn_open_rect, hover_btn == 1, primary=False)

            # Recent Projects
            settings = config.load_settings()
            recent_files = settings.get("recent_files", [])
            if recent_files:
                ry = frame_h // 2 + 130
                self._put_text_centered(frame_out, "Recent Projects", cx, ry, 0.5, TEXT_SECONDARY, 1, font=FONT_BOLD)
                cv2.line(frame_out, (cx - 100, ry + 15), (cx + 100, ry + 15), DIVIDER, 1)
                
                # Check recent hover
                def check_recent_hover(mx, my):
                    for i, rpath in enumerate(recent_files[:3]):
                        y_pos = ry + 40 + i * 35
                        if cx - 150 <= mx <= cx + 150 and y_pos - 15 <= my <= y_pos + 15:
                            return i
                    return -1

                for i, rpath in enumerate(recent_files[:3]):
                    y_pos = ry + 40 + i * 35
                    is_h = (hover_btn == 10 + i)
                    color = ACCENT if is_h else TEXT_PRIMARY
                    self._put_text_centered(frame_out, str(Path(rpath).name), cx, y_pos, 0.45, color, 1)

            # Transition fade-to-black
            if phase in ("selected", "fade_out") and sel_a > 0:
                dark = np.zeros_like(frame_out)
                frame_out = cv2.addWeighted(frame_out, 1.0 - sel_a, dark, sel_a, 0)

            cv2.imshow(window, frame_out)
            key = cv2.waitKey(16) & 0xFF

            if hover_btn == 99 or key in (ord('\r'), ord('\n'), ord(' ')):
                phase    = "selected"
                hover_btn = 98

            if phase in ("selected", "fade_out") and sel_a >= 1.0:
                break

            if key in (27, ord('q'), ord('Q')):
                result[0] = StartupResult.QUIT
                break

        cv2.setMouseCallback(window, lambda *a: None)
        return result[0]

    def _draw_startup_btn(
        self,
        frame: np.ndarray,
        label: str,
        rect: Tuple[int, int, int, int],
        hovered: bool,
        primary: bool,
    ) -> None:
        x1, y1, x2, y2 = rect
        if primary:
            bg = (180, 230, 20) if hovered else ACCENT
            fg = BG_DEEP
        else:
            bg = BG_CARD_HOVER if hovered else BG_CARD
            fg = TEXT_PRIMARY
        r = 8
        cv2.rectangle(frame, (x1 + r, y1), (x2 - r, y2), bg, -1)
        cv2.rectangle(frame, (x1, y1 + r), (x2, y2 - r), bg, -1)
        for cx_, cy_ in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
            cv2.circle(frame, (cx_, cy_), r, bg, -1)
        if not primary:
            cv2.rectangle(frame, (x1, y1), (x2, y2), DIVIDER, 1)
        cxb = (x1 + x2) // 2
        cyb = (y1 + y2) // 2 + 5
        self._put_text_centered(frame, label, cxb, cyb, 0.52, fg, 1)

    # ------------------------------------------------------------------
    # Mouse handler (registered by main.py on the main window)
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Modals (Blocking)
    # ------------------------------------------------------------------


    def show_recovery_prompt(self, base_frame) -> bool:
        """Returns True to recover, False to discard."""
        window = config.WINDOW_NAME
        result = [False]
        hover_btn = -1
        h, w = base_frame.shape[:2]
        pw, ph = 400, 200
        px = (w - pw) // 2
        py = (h - ph) // 2

        btn_recover = (px + 40, py + 120, px + 180, py + 160)
        btn_discard = (px + 220, py + 120, px + 360, py + 160)

        def on_mouse(event, mx, my, flags, param):
            nonlocal hover_btn
            in_rec = btn_recover[0] <= mx <= btn_recover[2] and btn_recover[1] <= my <= btn_recover[3]
            in_dis = btn_discard[0] <= mx <= btn_discard[2] and btn_discard[1] <= my <= btn_discard[3]
            
            hover_btn = 0 if in_rec else (1 if in_dis else -1)
            
            if event == cv2.EVENT_LBUTTONDOWN:
                if in_rec:
                    result[0] = True
                    hover_btn = 99
                elif in_dis:
                    result[0] = False
                    hover_btn = 99

        cv2.setMouseCallback(window, on_mouse)

        dim = base_frame.copy()
        cv2.rectangle(dim, (0, 0), (w, h), (0, 0, 0), -1)
        bg = cv2.addWeighted(dim, 0.6, base_frame, 0.4, 0)

        while True:
            frame = bg.copy()
            self._filled_rounded_rect(frame, px, py, px + pw, py + ph, 12, BG_CARD)
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), DIVIDER, 1)

            self._put_text(frame, "Crash Recovery", (px + 20, py + 36), 0.70, TEXT_PRIMARY, 1, font=FONT_BOLD)
            cv2.line(frame, (px + 16, py + 50), (px + pw - 16, py + 50), DIVIDER, 1)
            self._put_text(frame, "Recover previous session?", (px + 20, py + 80), 0.45, TEXT_SECONDARY, 1)

            for i, (rect, label, primary) in enumerate([(btn_recover, "Recover", True), (btn_discard, "Discard", False)]):
                bg_c = ACCENT if primary else (BG_CARD_HOVER if hover_btn == i else BG_CARD)
                if primary and hover_btn == i: bg_c = (180, 230, 20)
                fg_c = BG_DEEP if primary else TEXT_PRIMARY
                self._filled_rounded_rect(frame, rect[0], rect[1], rect[2], rect[3], 6, bg_c)
                cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), DIVIDER, 1)
                self._put_text_centered(frame, label, (rect[0]+rect[2])//2, (rect[1]+rect[3])//2 + 5, 0.45, fg_c, 1)

            cv2.imshow(window, frame)
            key = cv2.waitKey(16) & 0xFF
            if key == 27 or hover_btn == 99:
                break

        cv2.setMouseCallback(window, lambda *a: None)
        return result[0]

    def show_unsaved_prompt(self, base_frame) -> str:
        """Returns 'save', 'discard', or 'cancel'."""
        window = config.WINDOW_NAME
        result = ["cancel"]
        hover_btn = -1
        h, w = base_frame.shape[:2]
        pw, ph = 400, 200
        px = (w - pw) // 2
        py = (h - ph) // 2

        btn_save = (px + 40, py + 120, px + 130, py + 160)
        btn_discard = (px + 140, py + 120, px + 250, py + 160)
        btn_cancel = (px + 260, py + 120, px + 360, py + 160)

        def on_mouse(event, mx, my, flags, param):
            nonlocal hover_btn
            in_save = btn_save[0] <= mx <= btn_save[2] and btn_save[1] <= my <= btn_save[3]
            in_discard = btn_discard[0] <= mx <= btn_discard[2] and btn_discard[1] <= my <= btn_discard[3]
            in_cancel = btn_cancel[0] <= mx <= btn_cancel[2] and btn_cancel[1] <= my <= btn_cancel[3]
            
            hover_btn = 0 if in_save else (1 if in_discard else (2 if in_cancel else -1))
            
            if event == cv2.EVENT_LBUTTONDOWN:
                if in_save:
                    result[0] = "save"
                    hover_btn = 99
                elif in_discard:
                    result[0] = "discard"
                    hover_btn = 99
                elif in_cancel:
                    result[0] = "cancel"
                    hover_btn = 99

        cv2.setMouseCallback(window, on_mouse)

        dim = base_frame.copy()
        cv2.rectangle(dim, (0, 0), (w, h), (0, 0, 0), -1)
        bg = cv2.addWeighted(dim, 0.6, base_frame, 0.4, 0)

        while True:
            frame = bg.copy()
            self._filled_rounded_rect(frame, px, py, px + pw, py + ph, 12, BG_CARD)
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), DIVIDER, 1)

            self._put_text(frame, "Unsaved Changes", (px + 20, py + 36), 0.70, TEXT_PRIMARY, 1, font=FONT_BOLD)
            cv2.line(frame, (px + 16, py + 50), (px + pw - 16, py + 50), DIVIDER, 1)
            self._put_text(frame, "You have unsaved changes. Save?", (px + 20, py + 80), 0.45, TEXT_SECONDARY, 1)

            for i, (rect, label, primary) in enumerate([(btn_save, "Save", True), (btn_discard, "Discard", False), (btn_cancel, "Cancel", False)]):
                bg_c = ACCENT if primary else (BG_CARD_HOVER if hover_btn == i else BG_CARD)
                if primary and hover_btn == i: bg_c = (180, 230, 20)
                fg_c = BG_DEEP if primary else TEXT_PRIMARY
                self._filled_rounded_rect(frame, rect[0], rect[1], rect[2], rect[3], 6, bg_c)
                cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), DIVIDER, 1)
                self._put_text_centered(frame, label, (rect[0]+rect[2])//2, (rect[1]+rect[3])//2 + 5, 0.45, fg_c, 1)

            cv2.imshow(window, frame)
            key = cv2.waitKey(16) & 0xFF
            if key == 27 or hover_btn == 99:
                break

        cv2.setMouseCallback(window, lambda *a: None)
        return result[0]

    def show_export_prompt(self, base_frame) -> dict:
        """Returns {'format': 'png'|'jpg', 'bg': bool} or None."""
        window = config.WINDOW_NAME
        result = None
        hover_btn = -1
        h, w = base_frame.shape[:2]
        pw, ph = 400, 300
        px = (w - pw) // 2
        py = (h - ph) // 2
        
        fmt = "png"
        bg_inc = False
        
        btn_png = (px + 40, py + 100, px + 180, py + 140)
        btn_jpg = (px + 220, py + 100, px + 360, py + 140)
        btn_bg = (px + 40, py + 160, px + 360, py + 200)
        
        btn_export = (px + 80, py + 240, px + 200, py + 280)
        btn_cancel = (px + 220, py + 240, px + 340, py + 280)

        def on_mouse(event, mx, my, flags, param):
            nonlocal hover_btn, fmt, bg_inc, result
            in_png = btn_png[0] <= mx <= btn_png[2] and btn_png[1] <= my <= btn_png[3]
            in_jpg = btn_jpg[0] <= mx <= btn_jpg[2] and btn_jpg[1] <= my <= btn_jpg[3]
            in_bg = btn_bg[0] <= mx <= btn_bg[2] and btn_bg[1] <= my <= btn_bg[3]
            in_exp = btn_export[0] <= mx <= btn_export[2] and btn_export[1] <= my <= btn_export[3]
            in_can = btn_cancel[0] <= mx <= btn_cancel[2] and btn_cancel[1] <= my <= btn_cancel[3]
            
            hover_btn = 0 if in_png else (1 if in_jpg else (2 if in_bg else (3 if in_exp else (4 if in_can else -1))))
            
            if event == cv2.EVENT_LBUTTONDOWN:
                if in_png: fmt = "png"
                elif in_jpg: fmt = "jpg"
                elif in_bg: bg_inc = not bg_inc
                elif in_exp:
                    result = {"format": fmt, "bg": bg_inc}
                    hover_btn = 99
                elif in_can:
                    hover_btn = 99

        cv2.setMouseCallback(window, on_mouse)

        dim = base_frame.copy()
        cv2.rectangle(dim, (0, 0), (w, h), (0, 0, 0), -1)
        bg_frame = cv2.addWeighted(dim, 0.6, base_frame, 0.4, 0)

        while True:
            frame = bg_frame.copy()
            self._filled_rounded_rect(frame, px, py, px + pw, py + ph, 12, BG_CARD)
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), DIVIDER, 1)

            self._put_text(frame, "Export Canvas", (px + 20, py + 36), 0.70, TEXT_PRIMARY, 1, font=FONT_BOLD)
            cv2.line(frame, (px + 16, py + 50), (px + pw - 16, py + 50), DIVIDER, 1)
            
            self._put_text(frame, "Format:", (px + 20, py + 80), 0.45, TEXT_SECONDARY, 1)
            
            for rect, label, is_sel, i in [(btn_png, "PNG", fmt=="png", 0), (btn_jpg, "JPG", fmt=="jpg", 1)]:
                bg_c = BG_CARD_HOVER if is_sel else (BG_CARD_HOVER if hover_btn == i else BG_CARD)
                self._filled_rounded_rect(frame, rect[0], rect[1], rect[2], rect[3], 6, bg_c)
                if is_sel: cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), ACCENT, 1)
                else: cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), DIVIDER, 1)
                self._put_text_centered(frame, label, (rect[0]+rect[2])//2, (rect[1]+rect[3])//2 + 5, 0.45, TEXT_PRIMARY, 1)

            bg_c = BG_CARD_HOVER if bg_inc else (BG_CARD_HOVER if hover_btn == 2 else BG_CARD)
            self._filled_rounded_rect(frame, btn_bg[0], btn_bg[1], btn_bg[2], btn_bg[3], 6, bg_c)
            if bg_inc: cv2.rectangle(frame, (btn_bg[0], btn_bg[1]), (btn_bg[2], btn_bg[3]), ACCENT, 1)
            else: cv2.rectangle(frame, (btn_bg[0], btn_bg[1]), (btn_bg[2], btn_bg[3]), DIVIDER, 1)
            lbl = "Include Camera Background" if bg_inc else "Drawing Only (Transparent/Black)"
            self._put_text_centered(frame, lbl, (btn_bg[0]+btn_bg[2])//2, (btn_bg[1]+btn_bg[3])//2 + 5, 0.45, TEXT_PRIMARY, 1)

            for rect, label, primary, i in [(btn_export, "Export", True, 3), (btn_cancel, "Cancel", False, 4)]:
                bg_c = ACCENT if primary else (BG_CARD_HOVER if hover_btn == i else BG_CARD)
                if primary and hover_btn == i: bg_c = (180, 230, 20)
                fg_c = BG_DEEP if primary else TEXT_PRIMARY
                self._filled_rounded_rect(frame, rect[0], rect[1], rect[2], rect[3], 6, bg_c)
                cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), DIVIDER, 1)
                self._put_text_centered(frame, label, (rect[0]+rect[2])//2, (rect[1]+rect[3])//2 + 5, 0.45, fg_c, 1)

            cv2.imshow(window, frame)
            key = cv2.waitKey(16) & 0xFF
            if key == 27 or hover_btn == 99:
                break

        cv2.setMouseCallback(window, lambda *a: None)
        return result

    def show_info_dialog(self, base_frame, title: str, lines: list) -> None:
        """Generic modal for Help/About."""
        window = config.WINDOW_NAME
        hover_btn = -1
        h, w = base_frame.shape[:2]
        pw, ph = 440, max(200, 100 + len(lines)*30)
        px = (w - pw) // 2
        py = (h - ph) // 2
        
        btn_close = (px + pw//2 - 60, py + ph - 60, px + pw//2 + 60, py + ph - 20)

        def on_mouse(event, mx, my, flags, param):
            nonlocal hover_btn
            in_can = btn_close[0] <= mx <= btn_close[2] and btn_close[1] <= my <= btn_close[3]
            hover_btn = 0 if in_can else -1
            if event == cv2.EVENT_LBUTTONDOWN and in_can:
                hover_btn = 99

        cv2.setMouseCallback(window, on_mouse)

        dim = base_frame.copy()
        cv2.rectangle(dim, (0, 0), (w, h), (0, 0, 0), -1)
        bg = cv2.addWeighted(dim, 0.6, base_frame, 0.4, 0)

        while True:
            frame = bg.copy()
            self._filled_rounded_rect(frame, px, py, px + pw, py + ph, 12, BG_CARD)
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), DIVIDER, 1)

            self._put_text(frame, title, (px + 20, py + 36), 0.70, TEXT_PRIMARY, 1, font=FONT_BOLD)
            cv2.line(frame, (px + 16, py + 50), (px + pw - 16, py + 50), DIVIDER, 1)
            
            for i, line in enumerate(lines):
                self._put_text(frame, line, (px + 20, py + 80 + i*30), 0.45, TEXT_SECONDARY, 1)

            bg_c = BG_CARD_HOVER if hover_btn == 0 else BG_CARD
            self._filled_rounded_rect(frame, btn_close[0], btn_close[1], btn_close[2], btn_close[3], 6, bg_c)
            cv2.rectangle(frame, (btn_close[0], btn_close[1]), (btn_close[2], btn_close[3]), DIVIDER, 1)
            self._put_text_centered(frame, "Close", (btn_close[0]+btn_close[2])//2, (btn_close[1]+btn_close[3])//2 + 5, 0.45, TEXT_PRIMARY, 1)

            cv2.imshow(window, frame)
            key = cv2.waitKey(16) & 0xFF
            if key == 27 or hover_btn == 99:
                break
                
        cv2.setMouseCallback(window, lambda *a: None)

    def show_recent_files(self, base_frame, recent_files: list) -> str:
        """Modal for recent files, returns selected path or None."""
        window = config.WINDOW_NAME
        result = None
        hover_btn = -1
        h, w = base_frame.shape[:2]
        pw, ph = 440, max(240, 120 + len(recent_files)*40)
        px = (w - pw) // 2
        py = (h - ph) // 2
        
        btn_close = (px + pw//2 - 60, py + ph - 60, px + pw//2 + 60, py + ph - 20)
        file_btns = [(px + 20, py + 70 + i*40, px + pw - 20, py + 70 + i*40 + 34, f) for i, f in enumerate(recent_files)]

        def on_mouse(event, mx, my, flags, param):
            nonlocal hover_btn, result
            in_can = btn_close[0] <= mx <= btn_close[2] and btn_close[1] <= my <= btn_close[3]
            curr_hover = -1
            if in_can: curr_hover = 98
            else:
                for i, (r1,r2,r3,r4,_) in enumerate(file_btns):
                    if r1 <= mx <= r3 and r2 <= my <= r4:
                        curr_hover = i
                        break
            hover_btn = curr_hover
            
            if event == cv2.EVENT_LBUTTONDOWN:
                if in_can: hover_btn = 99
                elif curr_hover >= 0 and curr_hover < 90:
                    result = file_btns[curr_hover][4]
                    hover_btn = 99

        cv2.setMouseCallback(window, on_mouse)

        dim = base_frame.copy()
        cv2.rectangle(dim, (0, 0), (w, h), (0, 0, 0), -1)
        bg = cv2.addWeighted(dim, 0.6, base_frame, 0.4, 0)

        while True:
            frame = bg.copy()
            self._filled_rounded_rect(frame, px, py, px + pw, py + ph, 12, BG_CARD)
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), DIVIDER, 1)

            self._put_text(frame, "Recent Files", (px + 20, py + 36), 0.70, TEXT_PRIMARY, 1, font=FONT_BOLD)
            cv2.line(frame, (px + 16, py + 50), (px + pw - 16, py + 50), DIVIDER, 1)
            
            if not recent_files:
                self._put_text(frame, "No recent files found.", (px + 20, py + 90), 0.45, TEXT_SECONDARY, 1)
            else:
                for i, (r1,r2,r3,r4,f) in enumerate(file_btns):
                    bg_c = BG_CARD_HOVER if hover_btn == i else BG_CARD
                    self._filled_rounded_rect(frame, r1, r2, r3, r4, 4, bg_c)
                    from pathlib import Path
                    self._put_text(frame, str(Path(f).name), (r1 + 10, r2 + 22), 0.45, TEXT_PRIMARY, 1)

            bg_c = BG_CARD_HOVER if hover_btn == 98 else BG_CARD
            self._filled_rounded_rect(frame, btn_close[0], btn_close[1], btn_close[2], btn_close[3], 6, bg_c)
            cv2.rectangle(frame, (btn_close[0], btn_close[1]), (btn_close[2], btn_close[3]), DIVIDER, 1)
            self._put_text_centered(frame, "Close", (btn_close[0]+btn_close[2])//2, (btn_close[1]+btn_close[3])//2 + 5, 0.45, TEXT_PRIMARY, 1)

            cv2.imshow(window, frame)
            key = cv2.waitKey(16) & 0xFF
            if key == 27 or hover_btn == 99:
                break

        cv2.setMouseCallback(window, lambda *a: None)
        return result


    def show_settings_prompt(self, base_frame) -> dict:
        """Blocking settings dialog."""
        window = config.WINDOW_NAME
        h, w = base_frame.shape[:2]
        pw, ph = 600, 480
        px = (w - pw) // 2
        py = (h - ph) // 2
        
        settings = config.load_settings()
        
        tabs = ["Drawing", "Display", "Performance", "General"]
        active_tab = 0
        hover_btn = -1
        
        btn_close = (px + pw - 100, py + ph - 50, px + pw - 20, py + ph - 16)
        
        # We will use simple toggles and cycles for settings
        # Drawing
        b_size = settings.get("default_brush_size", config.DEFAULT_BRUSH_SIZE)
        
        # Display
        c_mode = settings.get("camera_mode", config.CAMERA_MODE_DARK_OVERLAY)
        s_landmarks = settings.get("show_landmarks", True)
        
        # Performance
        f_limit = settings.get("fps_limit", 60)
        c_res = settings.get("camera_resolution", "1280x720")
        
        # General
        t_theme = settings.get("theme", "dark")
        a_save = settings.get("auto_save", True)

        def on_mouse(event, mx, my, flags, param):
            nonlocal hover_btn, active_tab, b_size, c_mode, s_landmarks, f_limit, c_res, t_theme, a_save
            
            in_close = btn_close[0] <= mx <= btn_close[2] and btn_close[1] <= my <= btn_close[3]
            
            curr_hover = -1
            if in_close: curr_hover = 99
            
            # Check tabs
            for i in range(len(tabs)):
                tx1 = px + 20 + i*120
                tx2 = tx1 + 110
                ty1 = py + 60
                ty2 = ty1 + 36
                if tx1 <= mx <= tx2 and ty1 <= my <= ty2:
                    curr_hover = i
                    if event == cv2.EVENT_LBUTTONDOWN:
                        active_tab = i
            
            # Check options based on tab
            opt_rects = []
            if active_tab == 0:
                opt_rects = [(px+20, py+140, px+140, py+170, "b_size_dec"), (px+160, py+140, px+280, py+170, "b_size_inc")]
            elif active_tab == 1:
                opt_rects = [
                    (px+20, py+140, px+300, py+170, "c_mode"),
                    (px+20, py+200, px+300, py+230, "s_land")
                ]
            elif active_tab == 2:
                opt_rects = [
                    (px+20, py+140, px+300, py+170, "f_limit"),
                    (px+20, py+200, px+300, py+230, "c_res")
                ]
            elif active_tab == 3:
                opt_rects = [
                    (px+20, py+140, px+300, py+170, "t_theme"),
                    (px+20, py+200, px+300, py+230, "a_save")
                ]
            
            for i, (r1,r2,r3,r4, action) in enumerate(opt_rects):
                if r1 <= mx <= r3 and r2 <= my <= r4:
                    curr_hover = 100 + i
                    if event == cv2.EVENT_LBUTTONDOWN:
                        if action == "b_size_dec": b_size = max(2, b_size - 2)
                        elif action == "b_size_inc": b_size = min(40, b_size + 2)
                        elif action == "c_mode":
                            modes = [config.CAMERA_MODE_NORMAL, config.CAMERA_MODE_DARK_OVERLAY, config.CAMERA_MODE_BLACK_CANVAS]
                            c_mode = modes[(modes.index(c_mode) + 1) % len(modes)]
                        elif action == "s_land": s_landmarks = not s_landmarks
                        elif action == "f_limit": f_limit = 30 if f_limit == 60 else 60
                        elif action == "c_res": c_res = "640x480" if c_res == "1280x720" else "1280x720"
                        elif action == "t_theme": t_theme = "light" if t_theme == "dark" else "dark"
                        elif action == "a_save": a_save = not a_save
            
            hover_btn = curr_hover
            if event == cv2.EVENT_LBUTTONDOWN and in_close:
                hover_btn = 999

        cv2.setMouseCallback(window, on_mouse)

        dim = base_frame.copy()
        cv2.rectangle(dim, (0, 0), (w, h), (0, 0, 0), -1)
        bg = cv2.addWeighted(dim, 0.6, base_frame, 0.4, 0)

        while True:
            frame = bg.copy()
            self._filled_rounded_rect(frame, px, py, px + pw, py + ph, 12, BG_CARD)
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), DIVIDER, 1)

            self._put_text(frame, "Settings", (px + 20, py + 36), 0.70, TEXT_PRIMARY, 1, font=FONT_BOLD)
            
            # Draw Tabs
            for i, t in enumerate(tabs):
                tx1 = px + 20 + i*120
                tx2 = tx1 + 110
                ty1 = py + 60
                ty2 = ty1 + 36
                is_active = (active_tab == i)
                bg_c = BG_CARD_HOVER if is_active or hover_btn == i else BG_CARD
                self._filled_rounded_rect(frame, tx1, ty1, tx2, ty2, 4, bg_c)
                if is_active: cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), ACCENT, 1)
                self._put_text_centered(frame, t, (tx1+tx2)//2, (ty1+ty2)//2 + 5, 0.45, TEXT_PRIMARY, 1)
            
            cv2.line(frame, (px + 16, py + 106), (px + pw - 16, py + 106), DIVIDER, 1)

            # Draw Options based on Tab
            if active_tab == 0:
                self._put_text(frame, f"Default Brush Size: {b_size}px", (px+20, py+130), 0.45, TEXT_SECONDARY, 1)
                for i, (r1,r2,r3,r4,lbl) in enumerate([(px+20, py+140, px+140, py+170, "-2px"), (px+160, py+140, px+280, py+170, "+2px")]):
                    bg_c = BG_CARD_HOVER if hover_btn == 100+i else BG_CARD
                    self._filled_rounded_rect(frame, r1, r2, r3, r4, 6, bg_c)
                    cv2.rectangle(frame, (r1, r2), (r3, r4), DIVIDER, 1)
                    self._put_text_centered(frame, lbl, (r1+r3)//2, (r2+r4)//2 + 5, 0.45, TEXT_PRIMARY, 1)
            elif active_tab == 1:
                self._put_text(frame, f"Camera Mode:", (px+20, py+130), 0.45, TEXT_SECONDARY, 1)
                bg_c = BG_CARD_HOVER if hover_btn == 100 else BG_CARD
                self._filled_rounded_rect(frame, px+20, py+140, px+300, py+170, 6, bg_c)
                cv2.rectangle(frame, (px+20, py+140), (px+300, py+170), DIVIDER, 1)
                self._put_text_centered(frame, c_mode.replace("_", " ").title(), (px+20+px+300)//2, (py+140+py+170)//2 + 5, 0.45, TEXT_PRIMARY, 1)
                
                self._put_text(frame, f"Show Hand Landmarks:", (px+20, py+190), 0.45, TEXT_SECONDARY, 1)
                bg_c = BG_CARD_HOVER if hover_btn == 101 else BG_CARD
                self._filled_rounded_rect(frame, px+20, py+200, px+300, py+230, 6, bg_c)
                cv2.rectangle(frame, (px+20, py+200), (px+300, py+230), DIVIDER, 1)
                self._put_text_centered(frame, "ON" if s_landmarks else "OFF", (px+20+px+300)//2, (py+200+py+230)//2 + 5, 0.45, TEXT_PRIMARY, 1)
            elif active_tab == 2:
                self._put_text(frame, f"FPS Limit:", (px+20, py+130), 0.45, TEXT_SECONDARY, 1)
                bg_c = BG_CARD_HOVER if hover_btn == 100 else BG_CARD
                self._filled_rounded_rect(frame, px+20, py+140, px+300, py+170, 6, bg_c)
                cv2.rectangle(frame, (px+20, py+140), (px+300, py+170), DIVIDER, 1)
                self._put_text_centered(frame, str(f_limit), (px+20+px+300)//2, (py+140+py+170)//2 + 5, 0.45, TEXT_PRIMARY, 1)
                
                self._put_text(frame, f"Camera Resolution:", (px+20, py+190), 0.45, TEXT_SECONDARY, 1)
                bg_c = BG_CARD_HOVER if hover_btn == 101 else BG_CARD
                self._filled_rounded_rect(frame, px+20, py+200, px+300, py+230, 6, bg_c)
                cv2.rectangle(frame, (px+20, py+200), (px+300, py+230), DIVIDER, 1)
                self._put_text_centered(frame, c_res, (px+20+px+300)//2, (py+200+py+230)//2 + 5, 0.45, TEXT_PRIMARY, 1)
            elif active_tab == 3:
                self._put_text(frame, f"Theme:", (px+20, py+130), 0.45, TEXT_SECONDARY, 1)
                bg_c = BG_CARD_HOVER if hover_btn == 100 else BG_CARD
                self._filled_rounded_rect(frame, px+20, py+140, px+300, py+170, 6, bg_c)
                cv2.rectangle(frame, (px+20, py+140), (px+300, py+170), DIVIDER, 1)
                self._put_text_centered(frame, t_theme.title(), (px+20+px+300)//2, (py+140+py+170)//2 + 5, 0.45, TEXT_PRIMARY, 1)
                
                self._put_text(frame, f"Auto Save:", (px+20, py+190), 0.45, TEXT_SECONDARY, 1)
                bg_c = BG_CARD_HOVER if hover_btn == 101 else BG_CARD
                self._filled_rounded_rect(frame, px+20, py+200, px+300, py+230, 6, bg_c)
                cv2.rectangle(frame, (px+20, py+200), (px+300, py+230), DIVIDER, 1)
                self._put_text_centered(frame, "ON" if a_save else "OFF", (px+20+px+300)//2, (py+200+py+230)//2 + 5, 0.45, TEXT_PRIMARY, 1)

            # Close
            bg_c = BG_CARD_HOVER if hover_btn == 99 else BG_CARD
            self._filled_rounded_rect(frame, btn_close[0], btn_close[1], btn_close[2], btn_close[3], 6, bg_c)
            cv2.rectangle(frame, (btn_close[0], btn_close[1]), (btn_close[2], btn_close[3]), DIVIDER, 1)
            self._put_text_centered(frame, "Close & Save", (btn_close[0]+btn_close[2])//2, (btn_close[1]+btn_close[3])//2 + 5, 0.45, TEXT_PRIMARY, 1)

            cv2.imshow(window, frame)
            key = cv2.waitKey(16) & 0xFF
            if key == 27 or hover_btn == 999:
                break

        cv2.setMouseCallback(window, lambda *a: None)
        
        settings["default_brush_size"] = b_size
        settings["camera_mode"] = c_mode
        settings["show_landmarks"] = s_landmarks
        settings["fps_limit"] = f_limit
        settings["camera_resolution"] = c_res
        settings["theme"] = t_theme
        settings["auto_save"] = a_save
        config.save_settings(settings)
        return settings

    def handle_mouse(
        self,
        event: int,
        mx: int,
        my: int,
        flags: int,
        canvas_obj,       # DrawingCanvas – for brush size
        brush_size: int,
    ) -> None:
        """
        Single mouse handler for the main loop window.
        Handles topbar clicks, slider drag, and settings panel.
        """
        # ── Settings panel ──────────────────────────────────────────────
        if getattr(self, "_settings_open", False):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Check mode buttons
                for (x1, y1, x2, y2, mode_key) in self._settings_rects:
                    if x1 <= mx <= x2 and y1 <= my <= y2:
                        self._camera_mode = mode_key
                        settings = config.load_settings()
                        settings["camera_mode"] = mode_key
                        config.save_settings(settings)
                        self.flash_message(f"Camera: {mode_key.replace('_', ' ').title()}")
                        return
                # Shape recognition toggle
                tx1, ty1, tx2, ty2 = self._settings_toggle_rect
                if tx1 <= mx <= tx2 and ty1 <= my <= ty2:
                    self._shape_recognition_enabled = not self._shape_recognition_enabled
                    settings = config.load_settings()
                    settings["shape_recognition"] = self._shape_recognition_enabled
                    config.save_settings(settings)
                    state = "ON" if self._shape_recognition_enabled else "OFF"
                    self.flash_message(f"Shape Recognition {state}")
                    if self._cb_shape_recognition:
                        self._cb_shape_recognition(self._shape_recognition_enabled)
                    return
                # Close button
                cx1, cy1, cx2, cy2 = self._settings_close_rect
                if cx1 <= mx <= cx2 and cy1 <= my <= cy2:
                    self._settings_open = False
                    return
                # Click outside panel → close
                self._settings_open = False
            return

        # ── Topbar clicks ───────────────────────────────────────────────
        if event == cv2.EVENT_MOUSEMOVE:
            self._topbar_hover = ""
            for (x1, y1, x2, y2, label) in self._topbar_rects:
                if x1 <= mx <= x2 and y1 <= my <= y2:
                    self._topbar_hover = label
                    break

        if event == cv2.EVENT_LBUTTONDOWN:
            for (x1, y1, x2, y2, label) in self._topbar_rects:
                if x1 <= mx <= x2 and y1 <= my <= y2:
                    self._dispatch_topbar(label)
                    return

        # ── Brush slider ────────────────────────────────────────────────
        sx1, sy1, sx2, sy2 = self._slider_track
        track_y = (sy1 + sy2) // 2
        hit_zone = 12   # px above/below track centre counts as a hit

        if event == cv2.EVENT_LBUTTONDOWN:
            if sx1 <= mx <= sx2 and abs(my - track_y) <= hit_zone:
                self._slider_dragging = True
                self._update_brush_from_mouse(mx, canvas_obj)

        elif event == cv2.EVENT_MOUSEMOVE and self._slider_dragging:
            self._update_brush_from_mouse(mx, canvas_obj)

        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_LBUTTONUP):
            self._slider_dragging = False

    def _update_brush_from_mouse(self, mx: int, canvas_obj) -> None:
        sx1, sy1, sx2, sy2 = self._slider_track
        prop  = max(0.0, min(1.0, (mx - sx1) / max(1, sx2 - sx1)))
        new_size = int(2 + prop * (40 - 2))
        canvas_obj.brush_size = new_size
        canvas_obj.eraser_size = new_size   # mirror to eraser
        if self._cb_brush_size:
            self._cb_brush_size(new_size)

    def _dispatch_topbar(self, label: str) -> None:
        if label == "New":
            if self._cb_new:
                self._cb_new()
        elif label == "Open":
            if self._cb_open:
                self._cb_open()
        elif label == "Save":
            if self._cb_save:
                self._cb_save()
        elif label == "Save As":
            if self._cb_save_as:
                self._cb_save_as()
        elif label == "Settings":
            if getattr(self, "_cb_settings", None): self._cb_settings()
        elif label == "Recent":
            if getattr(self, "_cb_recent", None): self._cb_recent()
        elif label == "Export":
            if getattr(self, "_cb_export", None): self._cb_export()
        elif label == "Help":
            if getattr(self, "_cb_help", None): self._cb_help()
        elif label == "About":
            if getattr(self, "_cb_about", None): self._cb_about()

    # ------------------------------------------------------------------
    # Frame-level render (main loop)
    # ------------------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        gesture: Gesture,
        brush_color: Tuple[int, int, int],
        brush_size: int,
        hand_detected: bool,
        cursor_pos: Optional[Tuple[int, int]] = None,
        eraser_size: int = config.ERASER_SIZE,
    ) -> np.ndarray:
        self._update_fps()
        output = frame.copy()

        self._draw_sidebar(output, brush_color, brush_size, gesture, hand_detected)
        self._draw_topbar(output)
        self._draw_vignette(output)

        if cursor_pos and gesture in (Gesture.POINT, Gesture.PINCH, Gesture.FIST):
            self._draw_cursor(output, cursor_pos, gesture, brush_color, eraser_size)



        self._draw_toasts(output)
        return output

    def flash_message(self, text: str, duration_sec: float = 2.5) -> None:
        self._toasts = deque(t for t in self._toasts if t.text != text)
        self._toasts.append(Toast(text=text, ttl=duration_sec, max_ttl=duration_sec))
        while len(self._toasts) > 3:
            self._toasts.popleft()

    @property
    def fps(self) -> float:
        return self._fps

    # ------------------------------------------------------------------
    # Camera overlay / mode
    # ------------------------------------------------------------------

    def apply_camera_mode(
        self,
        frame: np.ndarray,
        hand_landmarks_frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return a display frame according to the current camera mode.
        Gesture detection must already have run on the original `frame`.

        Modes
        -----
        normal       – unchanged camera feed
        dark_overlay – 65-70 % black overlay on camera area
        black_canvas – fully black; only hand skeleton / cursor / drawings
                       are visible (compositing happens in main.py)
        """
        sw = config.SIDEBAR_WIDTH
        if self._camera_mode == config.CAMERA_MODE_NORMAL:
            return frame.copy()

        elif self._camera_mode == config.CAMERA_MODE_DARK_OVERLAY:
            out = frame.copy()
            roi  = out[:, sw:]
            dark = np.zeros_like(roi)
            out[:, sw:] = cv2.addWeighted(roi, 0.32, dark, 0.68, 0)
            return out

        elif self._camera_mode == config.CAMERA_MODE_BLACK_CANVAS:
            out = frame.copy()
            # Black out the camera area; drawings + skeleton composited on top
            out[:, sw:] = 0
            return out

        return frame.copy()

    # ------------------------------------------------------------------
    # Top bar
    # ------------------------------------------------------------------

    def _draw_topbar(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        sw   = config.SIDEBAR_WIDTH
        bar_h = 36

        overlay = frame.copy()
        cv2.rectangle(overlay, (sw, 0), (w, bar_h), (20, 15, 13), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        cv2.line(frame, (sw, bar_h), (w, bar_h), DIVIDER, 1)

        # FPS pill
        fps_text = f"{self._fps:.0f} fps"
        self._draw_badge(frame, fps_text, (w - 16, 18), anchor="right",
                         bg=BG_CARD, fg=TEXT_MUTED)

        # Menu items
        actions = ["New", "Open", "Recent", "Save", "Save As", "Export", "Settings", "Help", "About"]
        ax = sw + 16
        self._topbar_rects = []
        for act in actions:
            (tw, th), _ = cv2.getTextSize(act, FONT, 0.42, 1)
            cy = bar_h // 2 + th // 2
            hovered = self._topbar_hover == act
            col = TEXT_PRIMARY if hovered else TEXT_SECONDARY

            # Hover pill
            if hovered:
                pad = 6
                cv2.rectangle(frame,
                              (ax - pad, 4),
                              (ax + tw + pad, bar_h - 4),
                              BG_CARD_HOVER, -1)

            cv2.putText(frame, act, (ax, cy), FONT, 0.42, col, 1, cv2.LINE_AA)
            self._topbar_rects.append((ax - 6, 4, ax + tw + 6, bar_h - 4, act))
            ax += tw + 28

    # ------------------------------------------------------------------
    # Settings panel
    # ------------------------------------------------------------------

    def _draw_settings_panel(self, frame: np.ndarray, brush_size: int) -> None:
        h, w = frame.shape[:2]

        # Semi-transparent dimming overlay
        dim = frame.copy()
        cv2.rectangle(dim, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(dim, 0.55, frame, 0.45, 0, frame)

        # Panel geometry
        pw, ph = 420, 360
        px = (w - pw) // 2
        py = (h - ph) // 2

        # Panel background
        self._filled_rounded_rect(frame, px, py, px + pw, py + ph, 12, (30, 24, 20))
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), DIVIDER, 1)

        # Title
        self._put_text(frame, "Settings", (px + 20, py + 36), 0.70,
                       TEXT_PRIMARY, 1, font=FONT_BOLD)
        cv2.line(frame, (px + 16, py + 50), (px + pw - 16, py + 50), DIVIDER, 1)

        # Camera Mode label
        self._put_text(frame, "Camera Mode", (px + 20, py + 80), 0.46, TEXT_SECONDARY, 1)

        modes = [
            (config.CAMERA_MODE_NORMAL,       "Normal"),
            (config.CAMERA_MODE_DARK_OVERLAY,  "Dark Overlay  (default)"),
            (config.CAMERA_MODE_BLACK_CANVAS,  "Black Canvas"),
        ]

        self._settings_rects = []
        my_start = py + 96
        for i, (mode_key, mode_label) in enumerate(modes):
            bx1 = px + 20
            by1 = my_start + i * 46
            bx2 = px + pw - 20
            by2 = by1 + 36
            is_active = self._camera_mode == mode_key

            bg = BG_CARD_HOVER if is_active else BG_CARD
            self._filled_rounded_rect(frame, bx1, by1, bx2, by2, 6, bg)
            if is_active:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), ACCENT, 1)
                # Accent dot
                cv2.circle(frame, (bx2 - 16, (by1 + by2) // 2), 5, ACCENT, -1, cv2.LINE_AA)

            col = TEXT_PRIMARY if is_active else TEXT_SECONDARY
            self._put_text(frame, mode_label, (bx1 + 14, (by1 + by2) // 2 + 5),
                           0.44, col, 1)
            self._settings_rects.append((bx1, by1, bx2, by2, mode_key))

        # Close button
        cx1 = px + pw - 80
        cy1 = py + ph - 44
        cx2 = px + pw - 20
        cy2 = py + ph - 16
        self._filled_rounded_rect(frame, cx1, cy1, cx2, cy2, 6, BG_CARD)
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), DIVIDER, 1)
        self._put_text_centered(frame, "Close",
                                (cx1 + cx2) // 2, (cy1 + cy2) // 2 + 5,
                                0.44, TEXT_PRIMARY, 1)
        self._settings_close_rect = (cx1, cy1, cx2, cy2)

        # ── Smart Shape Recognition toggle ──────────────────────────────
        sep_y = my_start + len(modes) * 46 + 12
        cv2.line(frame, (px + 16, sep_y), (px + pw - 16, sep_y), DIVIDER, 1)
        self._put_text(frame, "Smart Shape Recognition",
                       (px + 20, sep_y + 22), 0.46, TEXT_SECONDARY, 1)

        toggle_on = self._shape_recognition_enabled
        tog_x2 = px + pw - 20
        tog_x1 = tog_x2 - 46
        tog_y1 = sep_y + 8
        tog_y2 = tog_y1 + 24
        tog_bg = ACCENT if toggle_on else (60, 55, 50)
        self._filled_rounded_rect(frame, tog_x1, tog_y1, tog_x2, tog_y2, 12, tog_bg)
        knob_x = tog_x2 - 14 if toggle_on else tog_x1 + 14
        knob_cy = (tog_y1 + tog_y2) // 2
        cv2.circle(frame, (knob_x, knob_cy), 9, TEXT_PRIMARY, -1, cv2.LINE_AA)
        self._settings_toggle_rect = (tog_x1, tog_y1, tog_x2, tog_y2)

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
        sw = config.SIDEBAR_WIDTH
        h  = frame.shape[0]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (sw, h), BG_SIDEBAR, -1)
        cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)
        cv2.rectangle(frame, (0, 0), (3, h), ACCENT, -1)
        cv2.line(frame, (sw, 0), (sw, h), DIVIDER, 1)

        self._put_text(frame, "EKORA", (16, 38), 0.80, TEXT_PRIMARY, 2, font=FONT_BOLD)
        cv2.line(frame, (16, 46), (54, 46), ACCENT, 2)
        self._put_text(frame, "v1.4", (16, 62), 0.36, TEXT_MUTED, 1)

        y = 88

        y = self._draw_card_header(frame, "COLORS", y)
        y = self._draw_palette(frame, brush_color, y)
        y += 8

        y = self._draw_card_header(frame, "BRUSH SIZE", y)
        y = self._draw_brush_card(frame, brush_color, brush_size, y)

        y = self._draw_card_header(frame, "TOOL", y)
        tool_name, tool_color = self._gesture_to_tool(gesture)
        y = self._draw_tool_card(frame, tool_name, tool_color, y)

        y = self._draw_card_header(frame, "GESTURE STATUS", y)
        y = self._draw_gesture_card(frame, gesture, hand_detected, y)

        self._draw_hand_indicator(frame, hand_detected, h)

    def _draw_card_header(self, frame: np.ndarray, label: str, y: int) -> int:
        sw = config.SIDEBAR_WIDTH
        cv2.line(frame, (12, y), (sw - 12, y), DIVIDER, 1)
        self._put_text(frame, label, (16, y + 18), 0.36, TEXT_MUTED, 1)
        return y + 26

    def _draw_palette(
        self,
        frame: np.ndarray,
        active_color: Tuple[int, int, int],
        y_start: int,
    ) -> int:
        sw_size = 34
        gap     = 10
        cols    = 4
        x0      = 16
        rows    = (len(config.COLOR_PALETTE) + cols - 1) // cols

        for i, color in enumerate(config.COLOR_PALETTE):
            row, col = divmod(i, cols)
            x = x0 + col * (sw_size + gap)
            y = y_start + row * (sw_size + gap)
            is_active = (color == active_color)
            r = 6
            self._filled_rounded_rect(frame, x, y, x + sw_size, y + sw_size, r, color)
            border_col = ACCENT if is_active else (60, 50, 46)
            border_w   = 2 if is_active else 1
            self._rounded_rect_outline(frame, x, y, x + sw_size, y + sw_size, r,
                                       border_col, border_w)
            if is_active:
                cv2.circle(frame, (x + sw_size - 7, y + 7), 4, TEXT_PRIMARY, -1, cv2.LINE_AA)

        return y_start + rows * (sw_size + gap) + 4

    def _draw_brush_card(
        self,
        frame: np.ndarray,
        brush_color: Tuple[int, int, int],
        brush_size: int,
        y: int,
    ) -> int:
        sw = config.SIDEBAR_WIDTH
        pad = 16
        card_y1 = y
        card_y2 = y + 80
        self._filled_rounded_rect(frame, pad, card_y1, sw - pad, card_y2, 6, BG_CARD)

        # Live preview
        preview_cx = pad + 30
        preview_cy = (card_y1 + card_y2) // 2
        max_r = 22
        r = max(2, min(brush_size, max_r))
        cv2.circle(frame, (preview_cx, preview_cy), max_r, BG_CARD_HOVER, -1)
        cv2.circle(frame, (preview_cx, preview_cy), r, brush_color, -1, cv2.LINE_AA)

        # Size label
        self._put_text(frame, f"{brush_size}px", (pad + 64, preview_cy - 10),
                       0.55, TEXT_PRIMARY, 1)
        self._put_text(frame, "Brush diameter", (pad + 64, preview_cy + 12),
                       0.35, TEXT_MUTED, 1)

        # Interactive slider track
        track_x1 = pad + 64
        track_x2 = sw - pad - 8
        track_y  = preview_cy + 32
        track_h  = 4

        # Store track bounds for mouse handler
        self._slider_track = (track_x1, track_y - track_h - 8,
                               track_x2, track_y + track_h + 8)

        cv2.rectangle(frame,
                      (track_x1, track_y - track_h // 2),
                      (track_x2, track_y + track_h // 2),
                      BG_CARD_HOVER, -1)

        # Filled portion — brush_size range 2–40
        prop   = (brush_size - 2) / max(1, 40 - 2)
        fill_x = int(track_x1 + prop * (track_x2 - track_x1))
        fill_x = max(track_x1, min(fill_x, track_x2))

        cv2.rectangle(frame,
                      (track_x1, track_y - track_h // 2),
                      (fill_x,   track_y + track_h // 2),
                      ACCENT, -1)
        # Thumb
        cv2.circle(frame, (fill_x, track_y), 7, ACCENT, -1, cv2.LINE_AA)
        cv2.circle(frame, (fill_x, track_y), 7, BG_CARD, 1, cv2.LINE_AA)

        # Min / max labels
        self._put_text(frame, "2",  (track_x1,     track_y + 16), 0.30, TEXT_MUTED, 1)
        self._put_text(frame, "40", (track_x2 - 14, track_y + 16), 0.30, TEXT_MUTED, 1)

        return card_y2 + 10

    def _draw_tool_card(
        self,
        frame: np.ndarray,
        tool_name: str,
        tool_color: Tuple[int, int, int],
        y: int,
    ) -> int:
        sw  = config.SIDEBAR_WIDTH
        pad = 16
        card_y1 = y
        card_y2 = y + 44
        self._filled_rounded_rect(frame, pad, card_y1, sw - pad, card_y2, 6, BG_CARD)
        cv2.circle(frame, (pad + 14, (card_y1 + card_y2) // 2), 6,
                   tool_color, -1, cv2.LINE_AA)
        self._put_text(frame, tool_name, (pad + 28, (card_y1 + card_y2) // 2 + 5),
                       0.48, TEXT_PRIMARY, 1)
        return card_y2 + 10

    def _draw_gesture_card(
        self,
        frame: np.ndarray,
        gesture: Gesture,
        hand_detected: bool,
        y: int,
    ) -> int:
        sw  = config.SIDEBAR_WIDTH
        pad = 16
        card_y1 = y
        card_y2 = y + 44
        self._filled_rounded_rect(frame, pad, card_y1, sw - pad, card_y2, 6, BG_CARD)
        
        display_gesture = gesture if hand_detected else Gesture.NONE
        label = GESTURE_LABELS.get(display_gesture, "Idle")
        
        status_color = ACCENT if (hand_detected and display_gesture != Gesture.NONE) else TEXT_MUTED
        cv2.circle(frame, (pad + 14, (card_y1 + card_y2) // 2), 6, status_color, -1, cv2.LINE_AA)
        
        self._put_text(frame, label, (pad + 28, (card_y1 + card_y2) // 2 + 5), 0.48, TEXT_PRIMARY, 1)

        return card_y2 + 10

    def _draw_hand_indicator(self, frame: np.ndarray, hand_detected: bool, h: int) -> None:
        status_color = ACCENT if hand_detected else TEXT_MUTED
        status_text  = "Hand detected" if hand_detected else "No hand"
        dot_x, dot_y = 16, h - 26
        cv2.circle(frame, (dot_x, dot_y), 5, status_color, -1, cv2.LINE_AA)
        if hand_detected:
            cv2.circle(frame, (dot_x, dot_y), 9, ACCENT_DIM, 1, cv2.LINE_AA)
        self._put_text(frame, status_text, (dot_x + 14, dot_y + 5), 0.38, status_color, 1)

    # ------------------------------------------------------------------
    # Cursor
    # ------------------------------------------------------------------

    def _draw_cursor(
        self,
        frame: np.ndarray,
        pos: Tuple[int, int],
        gesture: Gesture,
        brush_color: Tuple[int, int, int],
        eraser_size: int = config.ERASER_SIZE,
    ) -> None:
        x, y = pos
        if gesture == Gesture.FIST:
            r = eraser_size // 2
            cv2.circle(frame, (x, y), r, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), max(1, r // 4), (200, 200, 200), 1, cv2.LINE_AA)
            tick = 6
            for dx_, dy_ in [(r+3,0),(-(r+3),0),(0,r+3),(0,-(r+3))]:
                cv2.line(frame, (x+dx_, y+dy_),
                         (x+dx_+(tick if dx_>0 else (-tick if dx_<0 else 0)),
                          y+dy_+(tick if dy_>0 else (-tick if dy_<0 else 0))),
                         (255,255,255), 1, cv2.LINE_AA)
        elif gesture == Gesture.POINT:
            size = 10
            cv2.circle(frame, (x, y), size, TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.line(frame, (x-size-4,y),(x-3,y), TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.line(frame, (x+3,y),(x+size+4,y), TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.line(frame, (x,y-size-4),(x,y-3), TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.line(frame, (x,y+3),(x,y+size+4), TEXT_MUTED, 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (x, y), 12, brush_color, 1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 2,  brush_color, -1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Vignette
    # ------------------------------------------------------------------

    def _draw_vignette(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        sw   = config.SIDEBAR_WIDTH
        roi  = frame[:, sw:]
        rows, cols = roi.shape[:2]

        # Rebuild mask only when the frame size changes (once in practice)
        if self._vignette_mask is None or self._vignette_shape != (rows, cols):
            X = cv2.getGaussianKernel(cols, cols * 0.8)
            Y = cv2.getGaussianKernel(rows, rows * 0.8)
            kernel = Y * X.T
            mask = kernel / kernel.max()
            self._vignette_mask  = (mask * 0.18 + 0.82).astype(np.float32)
            self._vignette_shape = (rows, cols)

        mask = self._vignette_mask
        for c in range(3):
            roi[:, :, c] = np.clip(roi[:, :, c] * mask, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Toast notifications
    # ------------------------------------------------------------------

    def _draw_toasts(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        dt = 1.0 / max(self._fps, 15)
        toast_w  = 240
        toast_h  = 36
        margin_r = 20
        margin_b = 20
        gap      = 8

        living = [t for t in self._toasts if t.ttl > 0]
        for idx, toast in enumerate(reversed(living)):
            alpha_f = min(1.0, toast.ttl / 0.5)
            x1 = w - toast_w - margin_r
            y1 = h - margin_b - toast_h - idx * (toast_h + gap)
            x2 = w - margin_r
            y2 = y1 + toast_h
            if x1 < 0 or y1 < 0:
                continue
            panel   = frame[y1:y2, x1:x2].copy()
            bg_rect = np.full_like(panel, TOAST_BG)
            blended = cv2.addWeighted(bg_rect, alpha_f, panel, 1.0 - alpha_f, 0)
            frame[y1:y2, x1:x2] = blended

            bar_alpha = int(255 * alpha_f)
            if bar_alpha > 0:
                cv2.rectangle(frame, (x1, y1), (x1 + 3, y2),
                              tuple(int(c * alpha_f) for c in ACCENT), -1)
            text_col = tuple(int(c * alpha_f) for c in TEXT_PRIMARY)
            cv2.putText(frame, toast.text,
                        (x1 + 12, y1 + toast_h // 2 + 5),
                        FONT, 0.42, text_col, 1, cv2.LINE_AA)

        for toast in self._toasts:
            toast.ttl -= dt
        self._toasts = deque(t for t in self._toasts if t.ttl > 0)

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    def _filled_rounded_rect(self, frame, x1, y1, x2, y2, r, color):
        cv2.rectangle(frame, (x1+r, y1), (x2-r, y2), color, -1)
        cv2.rectangle(frame, (x1, y1+r), (x2, y2-r), color, -1)
        for cx_, cy_ in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
            cv2.circle(frame, (cx_, cy_), r, color, -1)

    def _rounded_rect_outline(self, frame, x1, y1, x2, y2, r, color, thickness=1):
        cv2.line(frame, (x1+r,y1),(x2-r,y1), color, thickness)
        cv2.line(frame, (x1+r,y2),(x2-r,y2), color, thickness)
        cv2.line(frame, (x1,y1+r),(x1,y2-r), color, thickness)
        cv2.line(frame, (x2,y1+r),(x2,y2-r), color, thickness)
        cv2.ellipse(frame,(x1+r,y1+r),(r,r),180,0,90,color,thickness)
        cv2.ellipse(frame,(x2-r,y1+r),(r,r),270,0,90,color,thickness)
        cv2.ellipse(frame,(x1+r,y2-r),(r,r), 90,0,90,color,thickness)
        cv2.ellipse(frame,(x2-r,y2-r),(r,r),  0,0,90,color,thickness)

    def _draw_badge(self, frame, text, pos, anchor="left",
                    bg=BG_CARD, fg=TEXT_SECONDARY):
        (tw, th), baseline = cv2.getTextSize(text, FONT, 0.40, 1)
        pad_x, pad_y = 10, 5
        if anchor == "right":
            x2 = pos[0]; x1 = x2 - tw - pad_x * 2
        else:
            x1 = pos[0]; x2 = x1 + tw + pad_x * 2
        cy = pos[1]
        y1 = cy - th // 2 - pad_y
        y2 = cy + th // 2 + pad_y + baseline
        self._filled_rounded_rect(frame, x1, y1, x2, y2, 4, bg)
        cv2.putText(frame, text, (x1 + pad_x, cy + th // 2),
                    FONT, 0.40, fg, 1, cv2.LINE_AA)

    def _put_text_centered(self, frame, text, cx, cy, scale, color,
                           thickness, font=FONT):
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.putText(frame, text, (cx - tw // 2, cy + th // 2),
                    font, scale, color, thickness, cv2.LINE_AA)

    def _put_text(self, frame, text, pos, scale, color, thickness,
                  align_right=False, right_x=0, font=FONT):
        if align_right:
            (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
            pos = (right_x - tw, pos[1])
        cv2.putText(frame, text, pos, font, scale, color, thickness, cv2.LINE_AA)

    def _gesture_to_tool(self, gesture):
        mapping = {
            Gesture.PINCH:     ("Brush",  ACCENT),
            Gesture.FIST:      ("Eraser", (100, 100, 220)),
            Gesture.POINT:     ("Hover",  TEXT_SECONDARY),
            Gesture.PEACE:     ("Color",  (100, 200, 255)),
            Gesture.OPEN_PALM: ("Clear",  (50, 80, 220)),
        }
        return mapping.get(gesture, ("Idle", TEXT_MUTED))

    def _update_fps(self):
        self._frame_count += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= config.FPS_UPDATE_INTERVAL:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer = time.time()
