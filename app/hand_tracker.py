"""
Hand tracker module – MediaPipe Hand Landmarker (Tasks API) with smoothing.

MediaPipe detects up to 21 3-D landmarks per hand.  This module:
  • Runs inference on each camera frame using the Tasks API (MediaPipe ≥ 0.10).
  • Converts normalised landmarks to pixel coordinates.
  • Draws the hand skeleton for visual feedback.
"""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import threading

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

import config

# ---------------------------------------------------------------------------
# MediaPipe landmark indices (standard 21-point hand model)
# ---------------------------------------------------------------------------
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

HAND_CONNECTIONS = [
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (WRIST, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (WRIST, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (WRIST, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, RING_MCP), (RING_MCP, PINKY_MCP),
]

# Official Google-hosted model (downloaded once on first run)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


def ensure_model() -> str:
    """
    Return the path to the hand landmarker model, downloading it if missing.

    The model is ~3 MB and is cached in the project's models/ folder so
    subsequent runs start instantly.
    """
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return str(MODEL_PATH)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmarker model → {MODEL_PATH}")
    print("(One-time download, ~3 MB. Please wait…)")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model ready.\n")
    return str(MODEL_PATH)


@dataclass
class HandLandmarks:
    """
    Container for one detected hand.

    Attributes
    ----------
    landmarks : list of (x, y) pixel tuples – length 21
    handedness : 'Left' or 'Right'
    """
    landmarks: List[Tuple[int, int]]
    handedness: str = "Right"


@dataclass
class HandTracker:
    """
    Wraps MediaPipe HandLandmarker (Tasks API) with smoothed output.

    Uses VIDEO running mode so tracking is reused between frames,
    which is faster than re-detecting every frame.
    """

    max_num_hands: int = config.MAX_NUM_HANDS
    min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE
    min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE
    _landmarker: vision.HandLandmarker = field(init=False, repr=False)
    _timestamp_ms: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        model_path = ensure_model()
        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_tracking_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame_bgr: np.ndarray) -> Optional[HandLandmarks]:
        """
        Detect hand landmarks in a BGR frame.

        Returns HandLandmarks or None if no hand is detected.
        """
        h, w = frame_bgr.shape[:2]

        # MediaPipe Tasks API expects RGB uint8 numpy array
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Monotonically increasing timestamp required for VIDEO mode
        now = time.time()
        self._timestamp_ms = int((now - self._start_time) * 1000)
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not result.hand_landmarks:
            return None

        # Use the first detected hand (single-hand drawing mode)
        hand_lm = result.hand_landmarks[0]
        handedness = "Right"
        if result.handedness:
            categories = result.handedness[0]
            if categories:
                handedness = categories[0].category_name or "Right"

        pixel_landmarks: List[Tuple[int, int]] = []
        for i, lm in enumerate(hand_lm):
            raw_x = lm.x * w
            raw_y = lm.y * h
            pixel_landmarks.append((round(raw_x), round(raw_y)))

        hand = HandLandmarks(landmarks=pixel_landmarks, handedness=handedness)
        return hand

    def draw_skeleton(
        self,
        frame: np.ndarray,
        hand: HandLandmarks,
        point_color: Tuple[int, int, int] = (0, 255, 180),
        line_color: Tuple[int, int, int] = (0, 200, 140),
    ) -> None:
        """Overlay the hand skeleton onto the frame (in-place)."""
        pts = hand.landmarks

        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, pts[start_idx], pts[end_idx], line_color, 2, cv2.LINE_AA)

        for x, y in pts:
            cv2.circle(frame, (x, y), 4, point_color, -1, cv2.LINE_AA)

        ix, iy = pts[INDEX_TIP]
        cv2.circle(frame, (ix, iy), 8, (255, 255, 255), 2, cv2.LINE_AA)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()


# ---------------------------------------------------------------------------
# Threaded Tracker
# ---------------------------------------------------------------------------

class ThreadedTracker:
    """
    Runs MediaPipe inference in a background thread to prevent blocking 
    the main OpenCV render loop.
    """
    def __init__(self, tracker: HandTracker) -> None:
        self.tracker = tracker
        self._frame = None
        self._hand: Optional[HandLandmarks] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._new_frame_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def update_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            # Drop frame if one is already being processed to keep latency low
            self._frame = frame.copy()
        self._new_frame_event.set()

    def get_latest_hand(self) -> Optional[HandLandmarks]:
        with self._lock:
            return self._hand

    def _worker(self) -> None:
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

    def close(self) -> None:
        self._stop.set()
        self._new_frame_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.tracker.close()

    def draw_skeleton(self, frame: np.ndarray, hand: HandLandmarks) -> None:
        self.tracker.draw_skeleton(frame, hand)
