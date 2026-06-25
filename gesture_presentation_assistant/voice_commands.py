"""
Background voice command listener using SpeechRecognition.

Supported phrases (see settings.VOICE_PHRASES):
  • "next slide"
  • "previous slide" / "go back"
  • "start presentation" / "begin presentation"
  • "stop presentation" / "end presentation"

Runs in a daemon thread so it never blocks the camera loop.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import settings

# SpeechRecognition is optional at import time – graceful fallback
try:
    import speech_recognition as sr

    _SR_AVAILABLE = True
except ImportError:
    _SR_AVAILABLE = False


@dataclass
class VoiceCommandHandler:
    """
    Listens for voice commands in a background thread.

    Recognised commands are placed in a thread-safe queue for the
    main loop to consume.
    """

    on_next: Optional[Callable[[], None]] = None
    on_previous: Optional[Callable[[], None]] = None
    on_start: Optional[Callable[[], None]] = None
    on_stop: Optional[Callable[[], None]] = None

    _command_queue: queue.Queue = field(default_factory=queue.Queue, init=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False)
    _running: bool = field(default=False, init=False)
    _recognizer: object = field(default=None, init=False)
    _microphone: object = field(default=None, init=False)
    _last_error: str = field(default="", init=False)
    _available: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._available = _SR_AVAILABLE
        if _SR_AVAILABLE:
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = 300
            self._recognizer.dynamic_energy_threshold = True
            self._recognizer.pause_threshold = 0.6

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def last_error(self) -> str:
        return self._last_error

    def start(self) -> bool:
        """Start the background listening thread."""
        if not self._available or self._running:
            return False
        if not settings.runtime.voice_enabled:
            return False

        try:
            self._microphone = sr.Microphone()
        except Exception as exc:
            self._last_error = str(exc)
            self._available = False
            return False

        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop the background listener."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def poll(self) -> Optional[str]:
        """
        Non-blocking check for a pending voice command.

        Returns the action name ('next', 'previous', 'start', 'stop') or None.
        """
        try:
            return self._command_queue.get_nowait()
        except queue.Empty:
            return None

    def _listen_loop(self) -> None:
        """Continuous listen → recognise → enqueue loop."""
        assert _SR_AVAILABLE and self._recognizer is not None

        # Calibrate for ambient noise once
        try:
            with self._microphone as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.8)
        except Exception as exc:
            self._last_error = f"Mic calibration failed: {exc}"
            self._running = False
            return

        while self._running:
            if not settings.runtime.voice_enabled:
                time.sleep(0.2)
                continue

            try:
                with self._microphone as source:
                    audio = self._recognizer.listen(source, timeout=1.0, phrase_time_limit=4)
            except sr.WaitTimeoutError:
                continue
            except Exception as exc:
                self._last_error = str(exc)
                time.sleep(0.5)
                continue

            try:
                text = self._recognizer.recognize_google(
                    audio, language=settings.VOICE_LANGUAGE
                ).lower().strip()
            except sr.UnknownValueError:
                continue
            except sr.RequestError as exc:
                self._last_error = f"Speech API error: {exc}"
                time.sleep(1.0)
                continue

            action = self._match_phrase(text)
            if action:
                self._command_queue.put(action)

    def _match_phrase(self, text: str) -> Optional[str]:
        """Map spoken text to an action using settings.VOICE_PHRASES."""
        for phrase, action in settings.VOICE_PHRASES.items():
            if phrase in text:
                return action
        return None

    def dispatch(self, action: str) -> None:
        """Execute the callback for a recognised action."""
        callbacks = {
            "next": self.on_next,
            "previous": self.on_previous,
            "start": self.on_start,
            "stop": self.on_stop,
        }
        cb = callbacks.get(action)
        if cb is not None:
            cb()
