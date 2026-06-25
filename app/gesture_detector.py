"""
Gesture detector module – interprets hand landmarks as user intents.

Gesture catalogue for Hand Paint
-------------------------------
  PINCH       Thumb tip + index tip close together  →  Draw on canvas
  POINT       Index extended, others curled         →  Move cursor (hover)
  OPEN_PALM   4+ fingers extended                   →  Clear canvas
  FIST        All fingers curled                    →  Eraser mode
  PEACE       Index + middle extended               →  Cycle brush colour

Detection is purely geometric: we compare distances and joint angles in
normalised image space, so it works regardless of camera resolution.
"""

from __future__ import annotations

import math
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
    """Recognised hand gestures mapped to application actions."""
    NONE = auto()
    PINCH = auto()       # Drawing
    POINT = auto()       # Hover / cursor only
    OPEN_PALM = auto()   # Clear canvas
    FIST = auto()        # Eraser
    PEACE = auto()       # Cycle colour


# Human-readable labels for the HUD
GESTURE_LABELS = {
    Gesture.NONE: "No gesture",
    Gesture.PINCH: "Pinch – Drawing",
    Gesture.POINT: "Point – Hover",
    Gesture.OPEN_PALM: "Open Palm – Clear",
    Gesture.FIST: "Fist – Eraser",
    Gesture.PEACE: "Peace – Next Color",
}


class GestureDetector:
    """
    Stateless gesture classifier (except for debounce on OPEN_PALM).

    The palm-clear gesture is debounced so a brief open-hand moment
    doesn't accidentally wipe the canvas.
    """

    def __init__(self) -> None:
        self._palm_clear_cooldown = 0  # Frames remaining before palm can clear again

    def detect(self, hand: HandLandmarks, frame_shape: Tuple[int, int]) -> Gesture:
        """
        Classify the current hand pose.

        Parameters
        ----------
        hand : detected hand landmarks
        frame_shape : (height, width) of the frame for normalisation

        Returns
        -------
        Gesture enum value (highest-priority match wins).
        """
        lm = hand.landmarks
        h, w = frame_shape

        # Normalise all landmarks to 0–1 for scale-invariant math
        norm = [(x / w, y / h) for x, y in lm]

        # Decrement cooldown each frame
        if self._palm_clear_cooldown > 0:
            self._palm_clear_cooldown -= 1

        # --- Individual finger states --------------------------------
        index_up = self._finger_extended(norm, INDEX_TIP, INDEX_PIP, INDEX_DIP, WRIST)
        middle_up = self._finger_extended(norm, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_DIP, WRIST)
        ring_up = self._finger_extended(norm, RING_TIP, RING_PIP, RING_DIP, WRIST)
        pinky_up = self._finger_extended(norm, PINKY_TIP, PINKY_PIP, PINKY_DIP, WRIST)
        thumb_up = self._thumb_extended(norm)

        extended_count = sum([index_up, middle_up, ring_up, pinky_up, thumb_up])

        # --- Pinch: thumb + index tips close -------------------------
        pinch_dist = self._distance(norm[THUMB_TIP], norm[INDEX_TIP])
        if pinch_dist < config.PINCH_THRESHOLD:
            return Gesture.PINCH

        # --- Peace sign: index + middle up, ring + pinky down --------
        if index_up and middle_up and not ring_up and not pinky_up:
            return Gesture.PEACE

        # --- Open palm: most fingers extended → clear (with cooldown) -
        fingers_up = sum([index_up, middle_up, ring_up, pinky_up])
        if fingers_up >= config.PALM_OPEN_FINGER_COUNT and self._palm_clear_cooldown == 0:
            self._palm_clear_cooldown = 45  # ~1.5 s at 30 fps
            return Gesture.OPEN_PALM

        # --- Fist: no fingers extended -------------------------------
        if extended_count == 0:
            return Gesture.FIST

        # --- Point: only index extended --------------------------------
        if index_up and not middle_up and not ring_up and not pinky_up:
            return Gesture.POINT

        return Gesture.NONE

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
        tip_y = norm[tip_idx][1]
        pip_y = norm[pip_idx][1]
        dip_y = norm[dip_idx][1]
        wrist_y = norm[wrist_idx][1]

        # Finger must point generally upward relative to wrist
        tip_above_pip = tip_y < pip_y - config.FINGER_EXTENDED_RATIO * abs(pip_y - wrist_y) * 0.15
        tip_above_dip = tip_y < dip_y
        return tip_above_pip and tip_above_dip

    @staticmethod
    def _thumb_extended(norm: List[Tuple[float, float]]) -> bool:
        """
        Thumb extension is trickier because it moves sideways.
        We check horizontal spread from the IP joint.
        """
        tip = norm[THUMB_TIP]
        ip = norm[THUMB_IP]
        return abs(tip[0] - ip[0]) > 0.04
