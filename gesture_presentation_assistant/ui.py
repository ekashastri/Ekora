"""
Modern Pygame UI with dark glassmorphism theme.

Renders:
  • Live webcam feed (converted from OpenCV BGR)
  • Gesture status panel with confidence bar
  • Presentation timer & slide counter
  • FPS monitor
  • Mode indicators (annotation, laser, voice)
  • Flash messages for user feedback
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pygame

import settings
from gesture_detector import GESTURE_LABELS, Gesture, GestureResult


def _bgr_to_surface(frame_bgr: np.ndarray) -> pygame.Surface:
    """Convert an OpenCV BGR numpy array to a Pygame surface."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    return pygame.image.frombuffer(frame_rgb.tobytes(), (w, h), "RGB")


def _draw_rounded_rect(
    surface: pygame.Surface,
    rect: pygame.Rect,
    color: Tuple[int, ...],
    radius: int = settings.CARD_RADIUS,
    border_color: Optional[Tuple[int, ...]] = None,
    border_width: int = 1,
) -> None:
    """Draw a filled rounded rectangle with optional border."""
    if len(color) == 4:
        temp = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(temp, color, temp.get_rect(), border_radius=radius)
        surface.blit(temp, rect.topleft)
    else:
        pygame.draw.rect(surface, color, rect, border_radius=radius)

    if border_color:
        pygame.draw.rect(surface, border_color, rect, border_width, border_radius=radius)


@dataclass
class UIRenderer:
    """
    Pygame-based HUD overlay for the Gesture Presentation Assistant.

    Call init() once before the main loop, then render() each frame.
    """

    width: int = settings.WINDOW_WIDTH
    height: int = settings.WINDOW_HEIGHT
    _screen: Optional[pygame.Surface] = field(default=None, init=False)
    _clock: Optional[pygame.time.Clock] = field(default=None, init=False)
    _fonts: Dict[str, pygame.font.Font] = field(default_factory=dict, init=False)
    _fps_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=settings.FPS_HISTORY_LEN), init=False
    )
    _flash_message: str = field(default="", init=False)
    _flash_until: float = field(default=0.0, init=False)
    _session_start: float = field(default_factory=time.time, init=False)

    def init(self) -> None:
        """Initialise Pygame window and fonts."""
        pygame.init()
        pygame.display.set_caption(settings.WINDOW_TITLE)
        self._screen = pygame.display.set_mode((self.width, self.height))
        self._clock = pygame.time.Clock()
        self._fonts = {
            "title": pygame.font.SysFont(settings.FONT_NAME, settings.FONT_SIZE_TITLE, bold=True),
            "body": pygame.font.SysFont(settings.FONT_NAME, settings.FONT_SIZE_BODY),
            "small": pygame.font.SysFont(settings.FONT_NAME, settings.FONT_SIZE_SMALL),
        }
        self._session_start = time.time()

    def handle_events(self) -> List[str]:
        """
        Process Pygame events.

        Returns a list of action strings for the main loop:
          'quit', 'toggle_annotation', 'toggle_laser', 'toggle_voice',
          'save_annotation', 'undo', 'sensitivity_up', 'sensitivity_down'
        """
        actions: List[str] = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                actions.append("quit")
            elif event.type == pygame.KEYDOWN:
                actions.extend(self._handle_key(event.key))
        return actions

    def _handle_key(self, key: int) -> List[str]:
        mapping = {
            pygame.K_ESCAPE: ["quit"],
            pygame.K_q: ["quit"],
            pygame.K_a: ["toggle_annotation"],
            pygame.K_l: ["toggle_laser"],
            pygame.K_v: ["toggle_voice"],
            pygame.K_s: ["save_annotation"],
            pygame.K_z: ["undo"],
            pygame.K_EQUALS: ["sensitivity_up"],
            pygame.K_PLUS: ["sensitivity_up"],
            pygame.K_MINUS: ["sensitivity_down"],
            pygame.K_LEFTBRACKET: ["sensitivity_down"],
        }
        return mapping.get(key, [])

    def flash(self, message: str, duration: float = 2.0) -> None:
        """Show a temporary toast message."""
        self._flash_message = message
        self._flash_until = time.time() + duration

    def render(
        self,
        frame_bgr: np.ndarray,
        gesture_result: GestureResult,
        fps: float,
        slide_index: int,
        total_slides: int,
        presentation_running: bool,
        presentation_paused: bool,
        hands_count: int = 0,
        voice_active: bool = False,
        recording_gesture: bool = False,
    ) -> float:
        """
        Composite the full UI and flip the display.

        Returns the clock tick delta (for FPS limiting).
        """
        assert self._screen is not None and self._clock is not None

        self._fps_history.append(fps)

        # Background
        self._screen.fill(settings.COLOR_BG)

        # Webcam feed – centred, scaled to fit left portion
        feed_rect = pygame.Rect(20, 20, self.width - 340, self.height - 40)
        self._blit_frame(frame_bgr, feed_rect)

        # Glass panels on the right
        panel_x = self.width - 310
        self._draw_gesture_panel(panel_x, 20, gesture_result)
        self._draw_status_panel(panel_x, 200, fps, hands_count, voice_active, recording_gesture)
        self._draw_presentation_panel(
            panel_x, 380, slide_index, total_slides, presentation_running, presentation_paused
        )
        self._draw_controls_hint(panel_x, 560)

        # Title bar
        self._draw_title_bar()

        # Flash toast
        if self._flash_message and time.time() < self._flash_until:
            self._draw_toast(self._flash_message)
        elif self._flash_message:
            self._flash_message = ""

        pygame.display.flip()
        return self._clock.tick(settings.FPS_TARGET) / 1000.0

    def _blit_frame(self, frame_bgr: np.ndarray, dest_rect: pygame.Rect) -> None:
        """Scale and blit the camera frame into dest_rect with rounded clip."""
        assert self._screen is not None
        surf = _bgr_to_surface(frame_bgr)
        scaled = pygame.transform.smoothscale(surf, (dest_rect.width, dest_rect.height))

        # Rounded border frame
        _draw_rounded_rect(
            self._screen,
            dest_rect.inflate(4, 4),
            settings.COLOR_CARD_BORDER[:3],
            radius=settings.CARD_RADIUS + 2,
        )
        clip_surf = pygame.Surface((dest_rect.width, dest_rect.height), pygame.SRCALPHA)
        clip_surf.blit(scaled, (0, 0))
        mask = pygame.Surface((dest_rect.width, dest_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=settings.CARD_RADIUS)
        clip_surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        self._screen.blit(clip_surf, dest_rect.topleft)

    def _draw_title_bar(self) -> None:
        assert self._screen is not None
        title = self._fonts["title"].render(settings.WINDOW_TITLE, True, settings.COLOR_TEXT)
        self._screen.blit(title, (24, 2))

        # Mode badges
        badges = []
        if settings.runtime.annotation_mode:
            badges.append(("ANNOTATION", settings.COLOR_WARNING))
        if settings.runtime.laser_mode:
            badges.append(("LASER", settings.COLOR_DANGER))
        if settings.runtime.voice_enabled:
            badges.append(("VOICE", settings.COLOR_SUCCESS))

        bx = self.width - 20
        for label, color in reversed(badges):
            surf = self._fonts["small"].render(label, True, color)
            bx -= surf.get_width() + 12
            self._screen.blit(surf, (bx, 6))

    def _draw_gesture_panel(self, x: int, y: int, result: GestureResult) -> None:
        assert self._screen is not None
        rect = pygame.Rect(x, y, 290, 170)
        _draw_rounded_rect(self._screen, rect, settings.COLOR_CARD, border_color=settings.COLOR_CARD_BORDER)

        pad = settings.CARD_PADDING
        self._screen.blit(
            self._fonts["title"].render("Gesture", True, settings.COLOR_ACCENT),
            (x + pad, y + pad),
        )

        gesture = result.gesture
        label = GESTURE_LABELS.get(gesture, "Idle")
        if gesture == Gesture.CUSTOM and result.custom_name:
            label = f"Custom: {result.custom_name}"

        color = settings.COLOR_GESTURE_ACTIVE if gesture != Gesture.NONE else settings.COLOR_TEXT_DIM
        self._screen.blit(
            self._fonts["body"].render(label, True, color),
            (x + pad, y + pad + 30),
        )

        # Confidence bar
        conf = result.confidence
        bar_rect = pygame.Rect(x + pad, y + pad + 60, 260, 14)
        pygame.draw.rect(self._screen, (40, 44, 58), bar_rect, border_radius=7)
        if conf > 0:
            fill_w = int(260 * min(1.0, conf))
            fill_rect = pygame.Rect(x + pad, y + pad + 60, fill_w, 14)
            bar_color = settings.COLOR_SUCCESS if conf >= settings.runtime.effective_confidence_threshold() else settings.COLOR_WARNING
            pygame.draw.rect(self._screen, bar_color, fill_rect, border_radius=7)

        conf_text = self._fonts["small"].render(f"Confidence: {conf:.0%}", True, settings.COLOR_TEXT_DIM)
        self._screen.blit(conf_text, (x + pad, y + pad + 82))

        threshold = settings.runtime.effective_confidence_threshold()
        thresh_text = self._fonts["small"].render(f"Threshold: {threshold:.0%}", True, settings.COLOR_TEXT_DIM)
        self._screen.blit(thresh_text, (x + pad, y + pad + 100))

        sens_text = self._fonts["small"].render(
            f"Sensitivity: {settings.runtime.sensitivity:.1f}x  (+/-)", True, settings.COLOR_TEXT_DIM
        )
        self._screen.blit(sens_text, (x + pad, y + pad + 118))

    def _draw_status_panel(
        self,
        x: int,
        y: int,
        fps: float,
        hands_count: int,
        voice_active: bool,
        recording: bool,
    ) -> None:
        assert self._screen is not None
        rect = pygame.Rect(x, y, 290, 170)
        _draw_rounded_rect(self._screen, rect, settings.COLOR_CARD, border_color=settings.COLOR_CARD_BORDER)

        pad = settings.CARD_PADDING
        self._screen.blit(
            self._fonts["title"].render("Status", True, settings.COLOR_ACCENT),
            (x + pad, y + pad),
        )

        elapsed = time.time() - self._session_start
        mins, secs = divmod(int(elapsed), 60)

        avg_fps = sum(self._fps_history) / max(len(self._fps_history), 1)
        fps_color = settings.COLOR_SUCCESS if avg_fps >= settings.FPS_TARGET else settings.COLOR_WARNING

        lines = [
            (f"FPS: {fps:.0f}  (avg {avg_fps:.0f})", fps_color),
            (f"Elapsed: {mins:02d}:{secs:02d}", settings.COLOR_TEXT),
            (f"Hands: {hands_count}", settings.COLOR_TEXT),
            (f"Voice: {'ON' if voice_active else 'OFF'}", settings.COLOR_SUCCESS if voice_active else settings.COLOR_TEXT_DIM),
        ]
        if recording:
            lines.append(("Recording gesture…", settings.COLOR_DANGER))

        for i, (text, color) in enumerate(lines):
            self._screen.blit(
                self._fonts["body"].render(text, True, color),
                (x + pad, y + pad + 32 + i * 26),
            )

    def _draw_presentation_panel(
        self,
        x: int,
        y: int,
        slide_index: int,
        total_slides: int,
        running: bool,
        paused: bool,
    ) -> None:
        assert self._screen is not None
        rect = pygame.Rect(x, y, 290, 170)
        _draw_rounded_rect(self._screen, rect, settings.COLOR_CARD, border_color=settings.COLOR_CARD_BORDER)

        pad = settings.CARD_PADDING
        self._screen.blit(
            self._fonts["title"].render("Presentation", True, settings.COLOR_ACCENT),
            (x + pad, y + pad),
        )

        if running:
            state = "PAUSED" if paused else "RUNNING"
            state_color = settings.COLOR_WARNING if paused else settings.COLOR_SUCCESS
        else:
            state = "IDLE"
            state_color = settings.COLOR_TEXT_DIM

        self._screen.blit(
            self._fonts["body"].render(f"State: {state}", True, state_color),
            (x + pad, y + pad + 32),
        )

        slide_text = f"Slide: {slide_index}"
        if total_slides > 0:
            slide_text += f" / {total_slides}"
            progress = min(1.0, slide_index / total_slides)
            bar_rect = pygame.Rect(x + pad, y + pad + 90, 260, 10)
            pygame.draw.rect(self._screen, (40, 44, 58), bar_rect, border_radius=5)
            fill = pygame.Rect(x + pad, y + pad + 90, int(260 * progress), 10)
            pygame.draw.rect(self._screen, settings.COLOR_ACCENT, fill, border_radius=5)

        self._screen.blit(
            self._fonts["body"].render(slide_text, True, settings.COLOR_TEXT),
            (x + pad, y + pad + 60),
        )

    def _draw_controls_hint(self, x: int, y: int) -> None:
        assert self._screen is not None
        rect = pygame.Rect(x, y, 290, self.height - y - 20)
        _draw_rounded_rect(self._screen, rect, settings.COLOR_CARD, border_color=settings.COLOR_CARD_BORDER)

        pad = settings.CARD_PADDING
        self._screen.blit(
            self._fonts["title"].render("Controls", True, settings.COLOR_ACCENT),
            (x + pad, y + pad),
        )

        hints = [
            "Swipe → / ←  : Slides",
            "Open palm   : Start",
            "Fist        : Pause",
            "Thumbs up   : Resume",
            "Peace       : Annotate",
            "A / L / V   : Modes",
            "S / Z       : Save / Undo",
            "Q / Esc     : Quit",
        ]
        for i, hint in enumerate(hints):
            self._screen.blit(
                self._fonts["small"].render(hint, True, settings.COLOR_TEXT_DIM),
                (x + pad, y + pad + 28 + i * 18),
            )

    def _draw_toast(self, message: str) -> None:
        assert self._screen is not None
        surf = self._fonts["body"].render(message, True, settings.COLOR_TEXT)
        pad_x, pad_y = 20, 12
        w, h = surf.get_size()
        rect = pygame.Rect(
            (self.width - w) // 2 - pad_x,
            self.height - 80,
            w + pad_x * 2,
            h + pad_y * 2,
        )
        _draw_rounded_rect(self._screen, rect, (*settings.COLOR_ACCENT, 200), radius=10)
        self._screen.blit(surf, (rect.x + pad_x, rect.y + pad_y))

    def quit(self) -> None:
        pygame.quit()
