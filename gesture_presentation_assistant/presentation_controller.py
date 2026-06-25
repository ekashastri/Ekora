"""
Presentation automation via PyAutoGUI keyboard shortcuts.

Sends standard keys that work with:
  • Microsoft PowerPoint (F5 slideshow, arrows, B for black screen)
  • Google Slides (in browser – arrows / fullscreen)
  • PDF viewers (Adobe, Edge, Chrome – arrow keys)
  • LibreOffice Impress

Includes slide counter tracking and session analytics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pyautogui

import settings

# Prevent PyAutoGUI from pausing between actions
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True  # Move mouse to corner to abort


@dataclass
class SessionStats:
    """Presentation session analytics."""

    session_start: float = field(default_factory=time.time)
    slides_advanced: int = 0
    slides_reversed: int = 0
    gestures_triggered: Dict[str, int] = field(default_factory=dict)
    voice_commands: int = 0
    annotations_saved: int = 0
    pauses: int = 0

    @property
    def elapsed_sec(self) -> float:
        return time.time() - self.session_start

    @property
    def net_slides(self) -> int:
        return self.slides_advanced - self.slides_reversed

    def record_gesture(self, name: str) -> None:
        self.gestures_triggered[name] = self.gestures_triggered.get(name, 0) + 1

    def summary(self) -> str:
        elapsed = int(self.elapsed_sec)
        mins, secs = divmod(elapsed, 60)
        top_gestures = sorted(
            self.gestures_triggered.items(), key=lambda x: x[1], reverse=True
        )[:5]
        lines = [
            f"Session duration: {mins:02d}:{secs:02d}",
            f"Slides forward: {self.slides_advanced}",
            f"Slides back: {self.slides_reversed}",
            f"Voice commands: {self.voice_commands}",
            f"Pauses: {self.pauses}",
            f"Annotations saved: {self.annotations_saved}",
        ]
        if top_gestures:
            lines.append("Top gestures: " + ", ".join(f"{g}({c})" for g, c in top_gestures))
        return "\n".join(lines)

    def save(self) -> str:
        """Persist analytics to a timestamped text file."""
        path = settings.ANALYTICS_DIR / f"session_{datetime.now():%Y%m%d_%H%M%S}.txt"
        path.write_text(self.summary(), encoding="utf-8")
        return str(path)


@dataclass
class PresentationController:
    """
    Sends keyboard shortcuts to control external presentation software.

    Maintains internal state for slide counter and pause/resume tracking.
    """

    slide_index: int = 1
    total_slides: int = 0  # Unknown unless user sets it
    is_running: bool = False
    is_paused: bool = False
    stats: SessionStats = field(default_factory=SessionStats)
    _last_key_time: float = 0.0

    def next_slide(self, source: str = "gesture") -> None:
        """Advance to the next slide."""
        if not self._can_send_key():
            return
        pyautogui.press(settings.KEY_NEXT_SLIDE)
        self.slide_index += 1
        self.stats.slides_advanced += 1
        self.stats.record_gesture(source)
        settings.runtime.presentation_running = True

    def previous_slide(self, source: str = "gesture") -> None:
        """Go to the previous slide."""
        if not self._can_send_key():
            return
        pyautogui.press(settings.KEY_PREV_SLIDE)
        self.slide_index = max(1, self.slide_index - 1)
        self.stats.slides_reversed += 1
        self.stats.record_gesture(source)
        settings.runtime.presentation_running = True

    def start_presentation(self, source: str = "gesture") -> None:
        """Enter slideshow / presentation mode (F5)."""
        if not self._can_send_key():
            return
        pyautogui.press(settings.KEY_START_PRESENTATION)
        self.is_running = True
        self.is_paused = False
        self.slide_index = 1
        settings.runtime.presentation_running = True
        settings.runtime.presentation_paused = False
        self.stats.record_gesture(source)

    def pause_presentation(self, source: str = "gesture") -> None:
        """Pause via black screen (PowerPoint 'B' key)."""
        if not self._can_send_key():
            return
        pyautogui.press(settings.KEY_PAUSE_PRESENTATION)
        self.is_paused = True
        settings.runtime.presentation_paused = True
        self.stats.pauses += 1
        self.stats.record_gesture(source)

    def resume_presentation(self, source: str = "gesture") -> None:
        """Resume from pause (toggle black screen off)."""
        if not self._can_send_key():
            return
        pyautogui.press(settings.KEY_RESUME_PRESENTATION)
        self.is_paused = False
        settings.runtime.presentation_paused = False
        self.stats.record_gesture(source)

    def stop_presentation(self, source: str = "voice") -> None:
        """Exit slideshow (Escape)."""
        if not self._can_send_key():
            return
        pyautogui.press("escape")
        self.is_running = False
        self.is_paused = False
        settings.runtime.presentation_running = False
        settings.runtime.presentation_paused = False
        self.stats.record_gesture(source)

    def set_total_slides(self, count: int) -> None:
        """Allow user to set total slide count for progress display."""
        self.total_slides = max(0, count)

    def reset_session(self) -> None:
        """Start a fresh analytics session."""
        self.stats = SessionStats()
        self.slide_index = 1
        self.is_running = False
        self.is_paused = False

    def _can_send_key(self) -> bool:
        """Rate-limit key sends to avoid double-firing."""
        now = time.time()
        if now - self._last_key_time < settings.PRESENTATION_KEY_DELAY:
            return False
        self._last_key_time = now
        return True
