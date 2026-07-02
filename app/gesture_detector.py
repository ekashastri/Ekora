"""
Gesture detector module - interprets hand landmarks as user intents.
v2.0 - Simplified Production Gesture System

This version replaces the previous multi-gesture, multi-state mapping
with a small, predictable set of four gestures, each tied to exactly
one user-facing action:

  POINT      (one index finger)        -> Draw continuously
  FIST       (closed fist)             -> Pause drawing (freeze stroke)
  OPEN_PALM  (five fingers, held)      -> Clear canvas (once per pose)
  PINCH      (thumb + index touching)  -> Move / drag the entire drawing

Design goals
------------
* Index-finger drawing starts the instant the pose is seen (zero
  confirmation delay) and stops the instant the pose changes.
* A closed fist freezes the stroke in place (no new points are added)
  but cursor tracking keeps running; returning to the index-finger
  pose resumes the SAME stroke immediately, with no gap or restart.
* The open-palm "clear" gesture requires a short, stable hold
  (PALM_HOLD_DURATION, ~200-300ms) before it fires, and fires only
  once per pose - the user must lower the palm before it can clear
  again. This prevents accidental wipes from a single noisy frame.
* Pinch (thumb + index) drags the whole drawing. It never creates new
  strokes, and only does anything once the canvas is non-empty.
* All non-instant transitions (palm, pinch) use consecutive-frame
  confirmation (GESTURE_DEBOUNCE_FRAMES, 3-5 frames) so one noisy
  frame can't flip the active gesture. If tracking is briefly lost
  (a "ghost" frame) the previously confirmed gesture is held steady
  rather than dropping to NONE.
"""

from __future__ import annotations

import math
import time
from enum import Enum, auto
from typing import List, Optional, Tuple

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
    """
    Recognised hand gestures mapped to application actions.

    NOTE: enum members are kept compatible with the rest of the app
    (app/ui.py, main.py) which import Gesture by name. PEACE and
    PALM_DETECTED are retained only so existing imports/dict lookups
    elsewhere don't break; this detector never emits them.
    """
    NONE  = auto()
    PINCH = auto()       # Move / drag the whole drawing
    POINT = auto()       # Draw
    OPEN_PALM = auto()   # Clear canvas (after brief stable hold)
    FIST  = auto()       # Pause drawing
    PEACE = auto()       # Unused in v2 (kept for compatibility)
    PALM_DETECTED = auto()  # Unused in v2 (kept for compatibility)


# Human-readable labels for the HUD
GESTURE_LABELS = {
    Gesture.NONE:      "Idle",
    Gesture.PINCH:     "Moving Drawing",
    Gesture.POINT:     "Drawing",
    Gesture.OPEN_PALM: "Clearing...",
    Gesture.FIST:      "Paused",
    Gesture.PEACE:     "Palette",
    Gesture.PALM_DETECTED: "Clearing...",
}


class GestureDetector:
    """
    Stateful gesture classifier for the simplified v2 interaction model.

    Four gestures only:
        POINT (draw) / FIST (pause) / OPEN_PALM (clear) / PINCH (move)

    Draw and Pause are instantaneous (no confirmation delay) so
    handwriting feels natural and pausing/resuming is immediate.
    Clear and Move are gated by short consecutive-frame confirmation
    to avoid accidental triggers from single noisy frames.
    """

    def __init__(self) -> None:
        self._gesture: Gesture = Gesture.NONE

        # --- Debounce: candidate raw gesture must be stable for N frames ---
        self._candidate_raw: Optional[str] = None
        self._candidate_frames: int = 0

        # --- Open-palm hold tracking ---
        self._palm_hold_start: Optional[float] = None
        self._palm_armed: bool = True   # re-armed once the pose is left
        self._is_pinching = False
        self.last_pinch_distance = 0.0

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
    ) -> Tuple[Gesture, float, dict]:
        """
        Classify the current hand pose.

        Parameters
        ----------
        hand        : detected hand landmarks (may be a ghost frame)
        frame_shape : (height, width) of the frame for normalisation

        Returns
        -------
        Tuple containing:
        - Gesture enum value reflecting the current confirmed gesture.
        - Confidence float (0.0 to 1.0)
        - Finger states dict
        """
        lm = hand.landmarks
        h, w = frame_shape

        # Normalise landmarks to 0-1
        norm: List[Tuple[float, float]] = [(x / w, y / h) for x, y in lm]

        raw, finger_states = self._raw_gesture(norm)
        confidence = self._update_gesture(raw)
        return self._gesture, confidence, finger_states

    # ------------------------------------------------------------------
    # Raw (single-frame, un-debounced) gesture classification
    # ------------------------------------------------------------------

    def _raw_gesture(self, norm: List[Tuple[float, float]]) -> Tuple[str, dict]:
        """Return a raw gesture name and finger states from normalised landmarks."""
        thumb_up  = self._thumb_extended(norm)
        index_up  = self._finger_extended(norm, INDEX_TIP,  INDEX_PIP,  INDEX_DIP,  WRIST)
        middle_up = self._finger_extended(norm, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_DIP, WRIST)
        ring_up   = self._finger_extended(norm, RING_TIP,   RING_PIP,   RING_DIP,   WRIST)
        pinky_up  = self._finger_extended(norm, PINKY_TIP,  PINKY_PIP,  PINKY_DIP,  WRIST)
        
        finger_states = {
            "Thumb": thumb_up,
            "Index": index_up,
            "Middle": middle_up,
            "Ring": ring_up,
            "Pinky": pinky_up
        }
        
        # Pinch (thumb + index touching) takes priority -> Move gesture
        # Normalise pinch distance to hand size (wrist to index PIP)
        hand_size = self._distance(norm[WRIST], norm[INDEX_PIP])
        pinch_dist_raw = self._distance(norm[THUMB_TIP], norm[INDEX_TIP])
        self.last_pinch_distance = pinch_dist_raw / (hand_size + 1e-6)

        PINCH_START_THRESH = 0.15
        PINCH_RELEASE_THRESH = 0.30

        if self._is_pinching:
            if self.last_pinch_distance > PINCH_RELEASE_THRESH:
                self._is_pinching = False
        else:
            if self.last_pinch_distance < PINCH_START_THRESH:
                self._is_pinching = True

        if self._is_pinching:
            return "PINCH", finger_states

        # Open Palm (all 5 fingers extended) -> Clear
        if thumb_up and index_up and middle_up and ring_up and pinky_up:
            return "PALM", finger_states

        # Gesture Locking: If currently drawing, ignore transitional noise
        if self._stroke_active and index_up:
            return "POINT", finger_states

        # Fist (no fingers extended) -> Pause
        if not index_up and not middle_up and not ring_up and not pinky_up:
            return "FIST", finger_states

        # Peace (index + middle extended, ring + pinky curled) -> Palette
        if index_up and middle_up and not ring_up and not pinky_up:
            return "PEACE", finger_states

        # Point: only the index finger extended -> Draw
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "POINT", finger_states

        return "NONE", finger_states

    # ------------------------------------------------------------------
    # Gesture confirmation / transition logic
    # ------------------------------------------------------------------

    def _update_gesture(self, raw: str) -> float:
        """Advance the confirmed gesture given the latest raw reading and return confidence."""
        debounce_n = getattr(config, "GESTURE_DEBOUNCE_FRAMES", 3)

        # --- Debounce bookkeeping: consecutive frames of the same raw value ---
        if raw == self._candidate_raw:
            self._candidate_frames += 1
        else:
            self._candidate_raw = raw
            self._candidate_frames = 1

        confirmed_n_frames = self._candidate_frames >= debounce_n
        
        confidence = min(1.0, self._candidate_frames / max(1, debounce_n))

        # -----------------------------------------------------------------
        # Re-arm the palm/clear trigger as soon as the palm pose is left,
        # so the next full palm hold can clear again.
        # -----------------------------------------------------------------
        if raw != "PALM":
            self._palm_armed = True
            self._palm_hold_start = None

        # -----------------------------------------------------------------
        # OPEN_PALM -> Clear. Requires a short stable hold (1 second)
        # before firing, and fires only once until the pose is released.
        # -----------------------------------------------------------------
        if raw == "PALM":
            now = time.monotonic()
            if self._palm_hold_start is None:
                self._palm_hold_start = now

            hold_duration = 1.0  # Explicitly 1.0s as per requirements
            elapsed = now - self._palm_hold_start
            
            confidence = min(1.0, elapsed / hold_duration)

            if elapsed >= hold_duration and self._palm_armed:
                self._gesture = Gesture.OPEN_PALM
                self._palm_armed = False  # don't fire again until pose is left
            # While still building up the hold (or already fired this pose),
            # keep showing the palm-clearing feedback state rather than
            # snapping back to the previous gesture.
            elif not self._palm_armed:
                self._gesture = Gesture.OPEN_PALM
            return confidence

        # -----------------------------------------------------------------
        # PEACE -> Open colour palette. Immediate, zero-delay like POINT
        # and FIST, so the palette appears the instant the pose is seen.
        # -----------------------------------------------------------------
        if raw == "PEACE":
            self._gesture = Gesture.PEACE
            return confidence

        # -----------------------------------------------------------------
        # POINT -> Draw. Immediate, zero-delay: drawing must start the
        # instant the pose is seen and stop the instant it changes.
        # -----------------------------------------------------------------
        if raw == "POINT":
            self._gesture = Gesture.POINT
            return confidence

        # -----------------------------------------------------------------
        # FIST -> Pause. Also immediate, so pausing/resuming feels instant
        # and a stroke can resume the moment the index finger returns.
        # -----------------------------------------------------------------
        if raw == "FIST":
            self._gesture = Gesture.FIST
            return confidence

        # -----------------------------------------------------------------
        # PINCH -> Move drawing. Small debounce to avoid a momentary
        # finger-tap from yanking the whole drawing around.
        # -----------------------------------------------------------------
        if raw == "PINCH":
            if confirmed_n_frames:
                self._gesture = Gesture.PINCH
            return confidence

        # -----------------------------------------------------------------
        # NONE / Transitional: The user relaxed their hand.
        # Debounce it slightly to avoid flickering, then drop to idle.
        # -----------------------------------------------------------------
        if raw == "NONE":
            if confirmed_n_frames:
                self._gesture = Gesture.NONE
            return confidence
            
        return confidence

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
        the PIP joint. Using Euclidean distance makes this check invariant
        to hand orientation (rotation).
        """
        tip = norm[tip_idx]
        pip = norm[pip_idx]
        wrist = norm[wrist_idx]

        dist_tip = math.hypot(tip[0] - wrist[0], tip[1] - wrist[1])
        dist_pip = math.hypot(pip[0] - wrist[0], pip[1] - wrist[1])

        # For a fully extended finger, the tip is significantly farther from the wrist than the PIP.
        return dist_tip > dist_pip * 1.1

    @staticmethod
    def _thumb_extended(norm: List[Tuple[float, float]]) -> bool:
        """
        Thumb extension is trickier because it moves sideways.
        We check its distance from the wrist compared to the IP joint.
        """
        tip = norm[THUMB_TIP]
        ip  = norm[THUMB_IP]
        wrist = norm[WRIST]
        
        dist_tip = math.hypot(tip[0] - wrist[0], tip[1] - wrist[1])
        dist_ip  = math.hypot(ip[0] - wrist[0], ip[1] - wrist[1])
        return dist_tip > dist_ip
