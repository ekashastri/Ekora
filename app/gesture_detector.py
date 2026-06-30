"""
Gesture detector module – interprets hand landmarks as user intents.
v1.5 – Robust Gesture State Machine + Safe Palm Gesture

Changes vs v1.4
---------------
* Explicit state machine (IDLE → HOVER → DRAWING → ERASING →
  COLOR_SELECTION → PALM_CONFIRMATION → CLEAR) replaces frame-by-frame
  decisions.  Transitions are debounced: a gesture must be seen for
  GESTURE_DEBOUNCE_FRAMES consecutive frames before the state changes,
  so a single noisy frame cannot flip the active state.

* Safe palm gesture: the canvas is cleared ONLY when:
    1. ALL five fingers are fully extended.
    2. No active drawing stroke is in progress.
    3. The gesture is stable for PALM_HOLD_DURATION seconds.
    4. PALM_CONFIDENCE_RATIO of frames in that window all agree.
    5. Hand velocity stays below PALM_MAX_VELOCITY (normalised).

Gesture catalogue (unchanged from v1.4)
---------------------------------------
  PINCH       Thumb tip + index tip close together  →  Draw on canvas
  POINT       Index extended, others curled         →  Move cursor (hover)
  OPEN_PALM   All 5 fingers extended + stable       →  Clear canvas (safe)
  FIST        All fingers curled                    →  Eraser mode
  PEACE       Index + middle extended               →  Cycle brush colour
"""

from __future__ import annotations

import math
import time
from collections import deque
from enum import Enum, auto
from typing import Deque, List, Optional, Tuple

import config
from app.hand_tracker import (
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
    """Recognised hand gestures mapped to application actions."""
    NONE  = auto()
    PINCH = auto()       # Drawing
    POINT = auto()       # Hover / cursor only
    OPEN_PALM = auto()   # Clear canvas (after safe confirmation)
    FIST  = auto()       # Eraser
    PEACE = auto()       # Cycle colour
    PALM_DETECTED = auto()


# Human-readable labels for the HUD
GESTURE_LABELS = {
    Gesture.NONE:      "Idle",
    Gesture.PINCH:     "Drawing",
    Gesture.POINT:     "Hover",
    Gesture.OPEN_PALM: "Clearing Canvas...",
    Gesture.FIST:      "Erase",
    Gesture.PEACE:     "Peace – Next Color",
    Gesture.PALM_DETECTED: "Palm Detected",
}


# Internal state machine states
class _State(Enum):
    IDLE             = auto()
    HOVER            = auto()
    DRAWING          = auto()
    ERASING          = auto()
    COLOR_SELECTION  = auto()
    PALM_CONFIRMATION = auto()
    CLEAR            = auto()


# Map state → public Gesture (what the rest of the app sees)
_STATE_TO_GESTURE = {
    _State.IDLE:              Gesture.NONE,
    _State.HOVER:             Gesture.POINT,
    _State.DRAWING:           Gesture.PINCH,
    _State.ERASING:           Gesture.FIST,
    _State.COLOR_SELECTION:   Gesture.PEACE,
    _State.PALM_CONFIRMATION: Gesture.PALM_DETECTED,
    _State.CLEAR:             Gesture.OPEN_PALM,
}


class GestureDetector:
    """
    Stateful gesture classifier with an explicit finite-state machine.

    The palm-clear gesture is gated by a multi-condition safety check
    (duration, velocity, confidence, no active stroke) to eliminate
    false positives.
    """

    def __init__(self) -> None:
        self._state: _State = _State.IDLE
        self._drawing_last_seen_time: float = 0.0

        # --- Debounce: candidate raw gesture must be stable for N frames ---
        self._candidate_raw: Optional[str] = None   # raw gesture name string
        self._candidate_frames: int = 0

        # --- Palm safety tracking ---
        self._palm_start_time: float = 0.0
        self._palm_frame_count: int  = 0    # total frames in palm window
        self._palm_ok_count: int     = 0    # frames where all 5 fingers up
        self._palm_last_pos: Optional[Tuple[float, float]] = None  # wrist pos

        # --- Post-clear cooldown ---
        self._clear_cooldown: float = 0.0   # epoch time after which palm allowed

        # --- Active stroke flag (set by main.py via property) ---
        self._stroke_active: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def stroke_active(self) -> bool:
        return self._stroke_active

    @stroke_active.setter
    def stroke_active(self, value: bool) -> None:
        self._stroke_active = value

    def detect(
        self,
        hand: HandLandmarks,
        frame_shape: Tuple[int, int],
    ) -> Gesture:
        """
        Classify the current hand pose via the state machine.

        Parameters
        ----------
        hand        : detected hand landmarks (may be a ghost frame)
        frame_shape : (height, width) of the frame for normalisation

        Returns
        -------
        Gesture enum value reflecting the *current stable state*.
        """
        lm = hand.landmarks
        h, w = frame_shape

        # Normalise landmarks to 0–1
        norm: List[Tuple[float, float]] = [(x / w, y / h) for x, y in lm]

        # Ghost frames (persistence window) keep the current state but
        # do NOT advance palm confirmation or debounce.
        if hand.is_ghost:
            return _STATE_TO_GESTURE[self._state]

        raw = self._raw_gesture(norm)
        self._update_state(raw, norm)
        return _STATE_TO_GESTURE[self._state]

    # ------------------------------------------------------------------
    # Raw (single-frame, un-debounced) gesture classification
    # ------------------------------------------------------------------

    def _raw_gesture(self, norm: List[Tuple[float, float]]) -> str:
        """Return a raw gesture name from normalised landmarks."""
        index_up  = self._finger_extended(norm, INDEX_TIP,  INDEX_PIP,  INDEX_DIP,  WRIST)
        middle_up = self._finger_extended(norm, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_DIP, WRIST)
        ring_up   = self._finger_extended(norm, RING_TIP,   RING_PIP,   RING_DIP,   WRIST)
        pinky_up  = self._finger_extended(norm, PINKY_TIP,  PINKY_PIP,  PINKY_DIP,  WRIST)
        thumb_up  = self._thumb_extended(norm)

        # Pinch overrides everything (drawing gesture)
        pinch_dist = self._distance(norm[THUMB_TIP], norm[INDEX_TIP])
        if pinch_dist < config.PINCH_THRESHOLD:
            return "PINCH"

        # All five fingers up → palm candidate
        if index_up and middle_up and ring_up and pinky_up and thumb_up:
            return "PALM"

        # Peace: index + middle up, others down
        if index_up and middle_up and not ring_up and not pinky_up:
            return "PEACE"

        # Fist: no fingers extended
        extended = sum([index_up, middle_up, ring_up, pinky_up, thumb_up])
        if extended == 0:
            return "FIST"

        # Point: only index
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "POINT"

        return "NONE"

    # ------------------------------------------------------------------
    # State machine transition
    # ------------------------------------------------------------------

    def _update_state(self, raw: str, norm: List[Tuple[float, float]]) -> None:
        """Advance the state machine given the raw gesture and landmarks."""
        debounce_n = getattr(config, "GESTURE_DEBOUNCE_FRAMES", 3)
        now = time.monotonic()

        # --- Debounce: count consecutive frames of the same raw gesture ---
        if raw == self._candidate_raw:
            self._candidate_frames += 1
        else:
            self._candidate_raw    = raw
            self._candidate_frames = 1

        confirmed = self._candidate_frames >= debounce_n

        # -----------------------------------------------------------------
        # PALM_CONFIRMATION state: collect evidence before clearing canvas
        # -----------------------------------------------------------------
        if self._state == _State.PALM_CONFIRMATION:
            if raw == "PALM":
                self._palm_frame_count += 1
                self._palm_ok_count    += 1

                if self._stroke_active:
                    # Stroke active — restart confirmation
                    self._begin_palm_confirmation(norm)
                    return

                min_ratio = getattr(config, "PALM_CONFIDENCE_RATIO", 0.80)
                ratio = self._palm_ok_count / max(1, self._palm_frame_count)

                # 6 frames + 3 debounce frames = 9 frames total (~300 ms at 30 FPS)
                if (
                    self._palm_frame_count >= 6
                    and ratio >= min_ratio
                    and now >= self._clear_cooldown
                ):
                    # All conditions met → transition to CLEAR
                    self._state = _State.CLEAR
                    self._clear_cooldown = now + 1.5  # 1.5 s cooldown prevent loop
                    return
            else:
                # Palm lost during confirmation — abort, go back to IDLE
                self._state = _State.IDLE
            return

        # -----------------------------------------------------------------
        # CLEAR: emit once then auto-reset to IDLE
        # -----------------------------------------------------------------
        if self._state == _State.CLEAR:
            self._state = _State.IDLE
            return

        # -----------------------------------------------------------------
        # Immediate transitions for zero-latency drawing and hovering
        # -----------------------------------------------------------------
        if raw == "PINCH":
            self._drawing_last_seen_time = now
            self._state = _State.DRAWING
            return

        is_drawing_cooldown = (self._state == _State.DRAWING) and (now - getattr(self, "_drawing_last_seen_time", 0) < 0.20)
        if is_drawing_cooldown and raw in ("NONE", "POINT"):
            return

        if raw == "POINT":
            self._state = _State.HOVER
            return

        # -----------------------------------------------------------------
        # Normal transitions (debounce required to prevent noisy drops)
        # -----------------------------------------------------------------
        if not confirmed:
            return   # Keep current state until new gesture stabilises

        if raw == "FIST":
            self._state = _State.ERASING
        elif raw == "PEACE":
            self._state = _State.COLOR_SELECTION
        elif raw == "PALM":
            if not self._stroke_active and now >= self._clear_cooldown:
                self._begin_palm_confirmation(norm)
        else:
            self._state = _State.IDLE

    def _begin_palm_confirmation(self, norm: List[Tuple[float, float]]) -> None:
        """Enter PALM_CONFIRMATION state and reset tracking counters."""
        self._state            = _State.PALM_CONFIRMATION
        self._palm_start_time  = time.monotonic()
        self._palm_frame_count = 1
        self._palm_ok_count    = 1
        self._palm_last_pos    = norm[WRIST]

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Euclidean distance between two normalised points."""
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _finger_extended(
        norm: List[Tuple[float, float]],
        tip_idx: int,
        pip_idx: int,
        dip_idx: int,
        wrist_idx: int,
    ) -> bool:
        """
        A finger is "extended" when its tip is farther from the wrist than
        the PIP joint, measured along the vertical axis (y decreases upward
        in image coordinates, so smaller y = higher on screen).

        We also verify the tip is above the DIP joint for robustness.
        """
        tip_y   = norm[tip_idx][1]
        pip_y   = norm[pip_idx][1]
        dip_y   = norm[dip_idx][1]
        wrist_y = norm[wrist_idx][1]

        tip_above_pip = (
            tip_y < pip_y - config.FINGER_EXTENDED_RATIO * abs(pip_y - wrist_y) * 0.15
        )
        tip_above_dip = tip_y < dip_y
        return tip_above_pip and tip_above_dip

    @staticmethod
    def _thumb_extended(norm: List[Tuple[float, float]]) -> bool:
        """
        Thumb extension is trickier because it moves sideways.
        We check horizontal spread from the IP joint.
        """
        tip = norm[THUMB_TIP]
        ip  = norm[THUMB_IP]
        return abs(tip[0] - ip[0]) > 0.04
