"""
Real-time hand tracking using MediaPipe Hand Landmarker (Tasks API).

Features:
  • One or two hands
  • Exponential smoothing on all 21 landmarks
  • Hand skeleton overlay
  • FPS-friendly VIDEO running mode
"""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

import settings

# ---------------------------------------------------------------------------
# Landmark indices (MediaPipe 21-point hand model)
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


def ensure_model() -> str:
    """Download the hand landmarker model once if not cached locally."""
    if settings.MODEL_PATH.exists() and settings.MODEL_PATH.stat().st_size > 0:
        return str(settings.MODEL_PATH)

    settings.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmarker model → {settings.MODEL_PATH}")
    print("(One-time download, ~3 MB)")
    urllib.request.urlretrieve(settings.MODEL_URL, settings.MODEL_PATH)
    print("Model ready.\n")
    return str(settings.MODEL_PATH)


@dataclass
class HandLandmarks:
    """One detected hand with pixel-space landmarks."""

    landmarks: List[Tuple[int, int]]
    handedness: str = "Right"
    detection_confidence: float = 1.0

    @property
    def palm_center(self) -> Tuple[int, int]:
        """Approximate palm center from wrist and MCP joints."""
        pts = [
            self.landmarks[WRIST],
            self.landmarks[INDEX_MCP],
            self.landmarks[MIDDLE_MCP],
            self.landmarks[RING_MCP],
            self.landmarks[PINKY_MCP],
        ]
        cx = int(sum(p[0] for p in pts) / len(pts))
        cy = int(sum(p[1] for p in pts) / len(pts))
        return cx, cy

    @property
    def index_tip(self) -> Tuple[int, int]:
        return self.landmarks[INDEX_TIP]


@dataclass
class HandTracker:
    """
    MediaPipe HandLandmarker wrapper with per-hand smoothing buffers.

    VIDEO mode reuses tracking state between frames for lower latency.
    """

    max_num_hands: int = settings.MAX_NUM_HANDS
    min_detection_confidence: float = settings.MIN_DETECTION_CONFIDENCE
    min_tracking_confidence: float = settings.MIN_TRACKING_CONFIDENCE
    smoothing: float = settings.LANDMARK_SMOOTHING

    _landmarker: vision.HandLandmarker = field(init=False, repr=False)
    _timestamp_ms: int = field(default=0, init=False, repr=False)
    _start_time: float = field(default=0.0, init=False, repr=False)
    # One smoothing buffer per possible hand slot
    _smooth_buffers: List[List[Optional[Tuple[float, float]]]] = field(
        default_factory=list, init=False, repr=False
    )

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
        self._init_buffers()

    def _init_buffers(self) -> None:
        self._smooth_buffers = [
            [None] * 21 for _ in range(self.max_num_hands)
        ]

    def update_max_hands(self, count: int) -> None:
        """Recreate landmarker when hand count changes."""
        if count == self.max_num_hands:
            return
        self.close()
        self.max_num_hands = count
        self.__post_init__()

    def process(self, frame_bgr: np.ndarray) -> List[HandLandmarks]:
        """
        Detect hands in a BGR frame.

        Returns a list of HandLandmarks (0–2 items).
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        self._timestamp_ms = int((time.time() - self._start_time) * 1000)
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not result.hand_landmarks:
            self._reset_smoothing()
            return []

        hands: List[HandLandmarks] = []
        for hand_idx, hand_lm in enumerate(result.hand_landmarks):
            if hand_idx >= self.max_num_hands:
                break

            handedness = "Right"
            confidence = 1.0
            if result.handedness and hand_idx < len(result.handedness):
                cats = result.handedness[hand_idx]
                if cats:
                    handedness = cats[0].category_name or "Right"
                    confidence = cats[0].score or 1.0

            pixel_landmarks: List[Tuple[int, int]] = []
            for i, lm in enumerate(hand_lm):
                sx, sy = self._smooth_point(hand_idx, i, lm.x * w, lm.y * h)
                pixel_landmarks.append((int(sx), int(sy)))

            hands.append(
                HandLandmarks(
                    landmarks=pixel_landmarks,
                    handedness=handedness,
                    detection_confidence=float(confidence),
                )
            )

        # Reset unused hand buffers
        for idx in range(len(hands), self.max_num_hands):
            self._smooth_buffers[idx] = [None] * 21

        return hands

    def draw_skeleton(
        self,
        frame: np.ndarray,
        hand: HandLandmarks,
        point_color: Tuple[int, int, int] = (0, 255, 180),
        line_color: Tuple[int, int, int] = (0, 200, 140),
        highlight_index: bool = True,
    ) -> None:
        """Draw hand skeleton overlay in-place on a BGR frame."""
        pts = hand.landmarks
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, pts[start_idx], pts[end_idx], line_color, 2, cv2.LINE_AA)

        for x, y in pts:
            cv2.circle(frame, (x, y), 4, point_color, -1, cv2.LINE_AA)

        if highlight_index:
            ix, iy = pts[INDEX_TIP]
            cv2.circle(frame, (ix, iy), 8, (255, 255, 255), 2, cv2.LINE_AA)

    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, "_landmarker") and self._landmarker is not None:
            self._landmarker.close()

    def _smooth_point(
        self, hand_idx: int, lm_idx: int, x: float, y: float
    ) -> Tuple[float, float]:
        """Exponential moving average for one landmark."""
        buf = self._smooth_buffers[hand_idx]
        prev = buf[lm_idx]
        if prev is None:
            buf[lm_idx] = (x, y)
            return x, y

        alpha = 1.0 - self.smoothing
        sx = prev[0] + alpha * (x - prev[0])
        sy = prev[1] + alpha * (y - prev[1])
        buf[lm_idx] = (sx, sy)
        return sx, sy

    def _reset_smoothing(self) -> None:
        for idx in range(self.max_num_hands):
            self._smooth_buffers[idx] = [None] * 21
