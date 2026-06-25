"""
Smart gesture recognition with confidence scoring and cooldown management.

Static gestures (pose-based):
  OPEN_PALM   → Start presentation
  FIST        → Pause
  THUMBS_UP   → Resume
  PEACE       → Toggle annotation mode
  POINT       → Laser pointer / annotation draw
  PINCH       → Draw (annotation mode)

Dynamic gestures (motion-based):
  SWIPE_RIGHT → Next slide
  SWIPE_LEFT  → Previous slide
  SWIPE_DOWN  → Undo (annotation mode)

Custom gestures can be loaded from JSON templates in custom_gestures/.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import settings
from hand_tracker import (
    HandLandmarks,
    INDEX_DIP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_DIP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_DIP,
    PINKY_PIP,
    PINKY_TIP,
    RING_DIP,
    RING_PIP,
    RING_TIP,
    THUMB_IP,
    THUMB_TIP,
    WRIST,
)


class Gesture(Enum):
    """All recognised gestures mapped to application actions."""

    NONE = auto()
    SWIPE_RIGHT = auto()
    SWIPE_LEFT = auto()
    SWIPE_DOWN = auto()
    OPEN_PALM = auto()
    FIST = auto()
    THUMBS_UP = auto()
    PEACE = auto()
    POINT = auto()
    PINCH = auto()
    CUSTOM = auto()


GESTURE_LABELS: Dict[Gesture, str] = {
    Gesture.NONE: "Idle",
    Gesture.SWIPE_RIGHT: "Swipe Right → Next",
    Gesture.SWIPE_LEFT: "Swipe Left → Previous",
    Gesture.SWIPE_DOWN: "Swipe Down → Undo",
    Gesture.OPEN_PALM: "Open Palm → Start",
    Gesture.FIST: "Fist → Pause",
    Gesture.THUMBS_UP: "Thumbs Up → Resume",
    Gesture.PEACE: "Peace → Annotation",
    Gesture.POINT: "Point → Laser / Draw",
    Gesture.PINCH: "Pinch → Draw",
    Gesture.CUSTOM: "Custom Gesture",
}

# Cooldown durations per gesture type (seconds)
GESTURE_COOLDOWNS: Dict[Gesture, float] = {
    Gesture.SWIPE_RIGHT: settings.SWIPE_COOLDOWN_SEC,
    Gesture.SWIPE_LEFT: settings.SWIPE_COOLDOWN_SEC,
    Gesture.SWIPE_DOWN: settings.SWIPE_COOLDOWN_SEC,
    Gesture.OPEN_PALM: settings.GESTURE_COOLDOWN_SEC,
    Gesture.FIST: settings.GESTURE_COOLDOWN_SEC,
    Gesture.THUMBS_UP: settings.GESTURE_COOLDOWN_SEC,
    Gesture.PEACE: settings.GESTURE_COOLDOWN_SEC,
}


@dataclass
class GestureResult:
    """Output of one detection cycle."""

    gesture: Gesture = Gesture.NONE
    confidence: float = 0.0
    custom_name: str = ""
    hand: Optional[HandLandmarks] = None


@dataclass
class _PositionSample:
    x: float
    y: float
    t: float


@dataclass
class GestureDetector:
    """
    Combines static pose classification with dynamic swipe tracking.

    Uses a sliding window of palm positions for swipe detection and
    geometric finger-state analysis for static poses.
    """

    position_history: Deque[_PositionSample] = field(
        default_factory=lambda: deque(maxlen=settings.SWIPE_MAX_FRAMES)
    )
    _last_gesture_time: Dict[Gesture, float] = field(default_factory=dict)
    _custom_templates: Dict[str, List[List[Tuple[float, float]]]] = field(
        default_factory=dict, init=False
    )
    _recording_buffer: List[List[Tuple[float, float]]] = field(
        default_factory=list, init=False
    )
    _is_recording: bool = False
    _recording_name: str = ""

    def __post_init__(self) -> None:
        self.load_custom_gestures()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self, hands: List[HandLandmarks], frame_shape: Tuple[int, int]
    ) -> GestureResult:
        """
        Analyse hands and return the highest-confidence actionable gesture.

        Priority: swipes > static gestures > custom > none.
        """
        if not hands:
            self.position_history.clear()
            return GestureResult()

        # Primary hand = first detected (dominant for gestures)
        hand = hands[0]
        h, w = frame_shape
        norm = [(x / w, y / h) for x, y in hand.landmarks]

        # Record custom gesture template if active
        if self._is_recording:
            self._recording_buffer.append(norm)
            return GestureResult(gesture=Gesture.NONE, confidence=0.0, hand=hand)

        # Track palm for swipe detection
        px, py = hand.palm_center
        now = time.time()
        self.position_history.append(_PositionSample(px / w, py / h, now))

        # --- Dynamic swipes -------------------------------------------
        swipe, swipe_conf = self._detect_swipe()
        if swipe != Gesture.NONE and self._cooldown_ok(swipe):
            if swipe_conf >= settings.runtime.effective_confidence_threshold():
                self._mark_triggered(swipe)
                return GestureResult(gesture=swipe, confidence=swipe_conf, hand=hand)

        # --- Static pose gestures -------------------------------------
        static, static_conf = self._detect_static(norm)
        if static != Gesture.NONE and self._cooldown_ok(static):
            if static_conf >= settings.runtime.effective_confidence_threshold():
                self._mark_triggered(static)
                return GestureResult(gesture=static, confidence=static_conf, hand=hand)

        # --- Custom gesture templates ---------------------------------
        custom_name, custom_conf = self._match_custom(norm)
        if custom_name and self._cooldown_ok(Gesture.CUSTOM):
            if custom_conf >= settings.runtime.effective_confidence_threshold():
                self._mark_triggered(Gesture.CUSTOM)
                return GestureResult(
                    gesture=Gesture.CUSTOM,
                    confidence=custom_conf,
                    custom_name=custom_name,
                    hand=hand,
                )

        # Return best static match for UI display even if below threshold
        if static != Gesture.NONE:
            return GestureResult(gesture=static, confidence=static_conf, hand=hand)

        return GestureResult(hand=hand)

    def start_recording(self, name: str) -> None:
        """Begin capturing landmark frames for custom gesture training."""
        self._is_recording = True
        self._recording_name = name
        self._recording_buffer.clear()

    def stop_recording(self) -> bool:
        """
        Finish recording and save template to disk.

        Returns True if saved successfully.
        """
        self._is_recording = False
        if len(self._recording_buffer) < 5:
            self._recording_buffer.clear()
            return False

        # Average down to 10 keyframes for compact storage
        n = len(self._recording_buffer)
        step = max(1, n // 10)
        keyframes = self._recording_buffer[::step][:10]
        path = settings.CUSTOM_GESTURES_DIR / f"{self._recording_name}.json"
        data = {"name": self._recording_name, "frames": keyframes}
        path.write_text(json.dumps(data), encoding="utf-8")
        self.load_custom_gestures()
        self._recording_buffer.clear()
        return True

    def load_custom_gestures(self) -> None:
        """Load all JSON gesture templates from the custom_gestures folder."""
        self._custom_templates.clear()
        for path in settings.CUSTOM_GESTURES_DIR.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                name = data.get("name", path.stem)
                frames = data.get("frames", [])
                if frames:
                    self._custom_templates[name] = frames
            except (json.JSONDecodeError, OSError):
                continue

    def get_cooldown_remaining(self, gesture: Gesture) -> float:
        """Seconds until a gesture can fire again (for UI display)."""
        last = self._last_gesture_time.get(gesture, 0.0)
        cooldown = GESTURE_COOLDOWNS.get(gesture, settings.GESTURE_COOLDOWN_SEC)
        remaining = cooldown - (time.time() - last)
        return max(0.0, remaining)

    # ------------------------------------------------------------------
    # Swipe detection
    # ------------------------------------------------------------------

    def _detect_swipe(self) -> Tuple[Gesture, float]:
        """Analyse position history for horizontal or vertical swipes."""
        if len(self.position_history) < 4:
            return Gesture.NONE, 0.0

        samples = list(self.position_history)
        start = samples[0]
        end = samples[-1]
        dx = end.x - start.x
        dy = end.y - start.y
        dt = max(end.t - start.t, 0.001)

        # Require dominant horizontal or vertical axis
        min_dist = settings.runtime.effective_swipe_distance()
        velocity = math.hypot(dx, dy) / dt

        if velocity < settings.SWIPE_MIN_VELOCITY * settings.runtime.sensitivity:
            return Gesture.NONE, 0.0

        if abs(dx) > abs(dy):
            if dx > min_dist:
                conf = min(1.0, dx / (min_dist * 1.5))
                return Gesture.SWIPE_RIGHT, conf
            if dx < -min_dist:
                conf = min(1.0, abs(dx) / (min_dist * 1.5))
                return Gesture.SWIPE_LEFT, conf
        else:
            if dy > settings.UNDO_SWIPE_DOWN_DISTANCE:
                conf = min(1.0, dy / (settings.UNDO_SWIPE_DOWN_DISTANCE * 1.5))
                return Gesture.SWIPE_DOWN, conf

        return Gesture.NONE, 0.0

    # ------------------------------------------------------------------
    # Static pose detection
    # ------------------------------------------------------------------

    def _detect_static(self, norm: List[Tuple[float, float]]) -> Tuple[Gesture, float]:
        """Classify finger configuration into a static gesture."""
        index_up = self._finger_extended(norm, INDEX_TIP, INDEX_PIP, INDEX_DIP, WRIST)
        middle_up = self._finger_extended(norm, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_DIP, WRIST)
        ring_up = self._finger_extended(norm, RING_TIP, RING_PIP, RING_DIP, WRIST)
        pinky_up = self._finger_extended(norm, PINKY_TIP, PINKY_PIP, PINKY_DIP, WRIST)
        thumb_up = self._thumb_extended(norm)

        fingers_up = sum([index_up, middle_up, ring_up, pinky_up])
        extended_count = fingers_up + (1 if thumb_up else 0)

        pinch_dist = self._distance(norm[THUMB_TIP], norm[INDEX_TIP])
        if pinch_dist < settings.PINCH_THRESHOLD:
            conf = 1.0 - pinch_dist / settings.PINCH_THRESHOLD
            return Gesture.PINCH, min(1.0, conf)

        # Thumbs up: thumb extended, all fingers curled
        if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
            return Gesture.THUMBS_UP, 0.9

        # Peace: index + middle up, others down
        if index_up and middle_up and not ring_up and not pinky_up:
            return Gesture.PEACE, 0.85

        # Open palm: 4+ fingers extended
        if fingers_up >= settings.PALM_OPEN_FINGER_COUNT:
            conf = fingers_up / 4.0
            return Gesture.OPEN_PALM, min(1.0, conf * 0.9)

        # Fist: no fingers extended
        if extended_count == 0:
            return Gesture.FIST, 0.88

        # Point: only index extended
        if index_up and not middle_up and not ring_up and not pinky_up:
            return Gesture.POINT, 0.8

        return Gesture.NONE, 0.0

    # ------------------------------------------------------------------
    # Custom gesture matching
    # ------------------------------------------------------------------

    def _match_custom(
        self, norm: List[Tuple[float, float]]
    ) -> Tuple[str, float]:
        """Compare current pose against saved templates (Procrustes-lite)."""
        best_name = ""
        best_score = 0.0

        for name, frames in self._custom_templates.items():
            for template in frames:
                if len(template) != len(norm):
                    continue
                dist = sum(
                    self._distance(norm[i], template[i]) for i in range(len(norm))
                ) / len(norm)
                score = max(0.0, 1.0 - dist * 8.0)
                if score > best_score:
                    best_score = score
                    best_name = name

        return best_name, best_score

    # ------------------------------------------------------------------
    # Cooldown helpers
    # ------------------------------------------------------------------

    def _cooldown_ok(self, gesture: Gesture) -> bool:
        last = self._last_gesture_time.get(gesture, 0.0)
        cooldown = GESTURE_COOLDOWNS.get(gesture, settings.GESTURE_COOLDOWN_SEC)
        cooldown /= max(settings.runtime.sensitivity, 0.5)
        return (time.time() - last) >= cooldown

    def _mark_triggered(self, gesture: Gesture) -> None:
        self._last_gesture_time[gesture] = time.time()
        self.position_history.clear()

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _finger_extended(
        norm: List[Tuple[float, float]],
        tip_idx: int,
        pip_idx: int,
        dip_idx: int,
        wrist_idx: int,
    ) -> bool:
        tip_y = norm[tip_idx][1]
        pip_y = norm[pip_idx][1]
        dip_y = norm[dip_idx][1]
        wrist_y = norm[wrist_idx][1]
        ratio = settings.FINGER_EXTENDED_RATIO
        tip_above_pip = tip_y < pip_y - ratio * abs(pip_y - wrist_y) * 0.15
        tip_above_dip = tip_y < dip_y
        return tip_above_pip and tip_above_dip

    @staticmethod
    def _thumb_extended(norm: List[Tuple[float, float]]) -> bool:
        tip = norm[THUMB_TIP]
        ip = norm[THUMB_IP]
        return abs(tip[0] - ip[0]) > settings.THUMB_UP_ANGLE_THRESHOLD
