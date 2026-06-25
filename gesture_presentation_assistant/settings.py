"""
Central configuration for the Gesture Presentation Assistant.

All tunable parameters live here so users can adjust sensitivity,
performance, and behaviour without touching application logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = PROJECT_ROOT / "models"
SAVES_DIR = PROJECT_ROOT / "saves"
ANNOTATIONS_DIR = SAVES_DIR / "annotations"
ANALYTICS_DIR = SAVES_DIR / "analytics"
CUSTOM_GESTURES_DIR = PROJECT_ROOT / "custom_gestures"

for _d in (ASSETS_DIR, MODELS_DIR, SAVES_DIR, ANNOTATIONS_DIR, ANALYTICS_DIR, CUSTOM_GESTURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Camera & display
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS_TARGET = 30
WINDOW_TITLE = "Gesture Presentation Assistant"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Processing resolution (lower = faster; upscaled for display)
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 480

# ---------------------------------------------------------------------------
# MediaPipe hand tracking
# ---------------------------------------------------------------------------
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.5
LANDMARK_SMOOTHING = 0.35  # 0 = no smoothing, 0.9 = heavy smoothing

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = MODELS_DIR / "hand_landmarker.task"

# ---------------------------------------------------------------------------
# Gesture detection thresholds
# ---------------------------------------------------------------------------
FINGER_EXTENDED_RATIO = 0.12
PINCH_THRESHOLD = 0.05
PALM_OPEN_FINGER_COUNT = 4
THUMB_UP_ANGLE_THRESHOLD = 0.06

# Swipe: minimum horizontal displacement (normalised 0–1) and frames
SWIPE_MIN_DISTANCE = 0.18
SWIPE_MAX_FRAMES = 18
SWIPE_MIN_VELOCITY = 0.012

# Confidence & cooldown
GESTURE_CONFIDENCE_THRESHOLD = 0.65
GESTURE_COOLDOWN_SEC = 1.2
SWIPE_COOLDOWN_SEC = 0.8

# Default sensitivity multiplier (1.0 = normal; user-adjustable at runtime)
DEFAULT_SENSITIVITY = 1.0
SENSITIVITY_MIN = 0.5
SENSITIVITY_MAX = 2.0

# ---------------------------------------------------------------------------
# Presentation automation (PyAutoGUI key bindings)
# ---------------------------------------------------------------------------
KEY_NEXT_SLIDE = "right"
KEY_PREV_SLIDE = "left"
KEY_START_PRESENTATION = "f5"
KEY_PAUSE_PRESENTATION = "b"       # PowerPoint black screen
KEY_RESUME_PRESENTATION = "b"      # Toggle black screen off
KEY_ANNOTATION_TOGGLE = "ctrl_l"

# Delay after sending a key (seconds) – prevents double-firing
PRESENTATION_KEY_DELAY = 0.05

# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------
ANNOTATION_COLORS: List[Tuple[int, int, int]] = [
    (255, 60, 60),    # Red
    (60, 180, 255),   # Blue
    (60, 255, 120),   # Green
    (255, 220, 60),   # Yellow
    (255, 120, 255),  # Magenta
    (255, 255, 255),  # White
]
DEFAULT_ANNOTATION_COLOR_INDEX = 0
ANNOTATION_BRUSH_SIZE = 4
ANNOTATION_SMOOTHING = 0.4
MAX_UNDO_STEPS = 30

# Gesture: pinch to draw, fist to erase region, open palm + swipe down = undo
UNDO_SWIPE_DOWN_DISTANCE = 0.15

# ---------------------------------------------------------------------------
# Laser pointer
# ---------------------------------------------------------------------------
LASER_COLOR = (255, 40, 40)
LASER_RADIUS = 12
LASER_TRAIL_LENGTH = 8
LASER_PULSE_SPEED = 4.0

# ---------------------------------------------------------------------------
# Voice commands
# ---------------------------------------------------------------------------
VOICE_ENABLED = True
VOICE_LANGUAGE = "en-US"
VOICE_PHRASES: Dict[str, str] = {
    "next slide": "next",
    "previous slide": "previous",
    "go back": "previous",
    "start presentation": "start",
    "begin presentation": "start",
    "stop presentation": "stop",
    "end presentation": "stop",
}

# ---------------------------------------------------------------------------
# UI theme (dark glassmorphism)
# ---------------------------------------------------------------------------
COLOR_BG = (18, 20, 28)
COLOR_CARD = (32, 36, 50, 180)
COLOR_CARD_BORDER = (80, 90, 130, 100)
COLOR_ACCENT = (90, 160, 255)
COLOR_SUCCESS = (80, 220, 140)
COLOR_WARNING = (255, 190, 60)
COLOR_DANGER = (255, 80, 80)
COLOR_TEXT = (230, 235, 245)
COLOR_TEXT_DIM = (140, 150, 170)
COLOR_GESTURE_ACTIVE = (100, 200, 255)

FONT_NAME = None  # Pygame default
FONT_SIZE_TITLE = 22
FONT_SIZE_BODY = 16
FONT_SIZE_SMALL = 13
CARD_RADIUS = 14
CARD_PADDING = 12

# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
FPS_TARGET = 30
FPS_HISTORY_LEN = 60
CAMERA_RETRY_MAX = 5
CAMERA_RETRY_DELAY = 0.5
WEBCAM_RECONNECT_INTERVAL = 2.0

# Skip voice processing every N frames to save CPU
VOICE_PROCESS_EVERY_N_FRAMES = 3


@dataclass
class RuntimeSettings:
    """Mutable settings adjusted at runtime via the UI."""

    sensitivity: float = DEFAULT_SENSITIVITY
    max_num_hands: int = MAX_NUM_HANDS
    voice_enabled: bool = VOICE_ENABLED
    annotation_mode: bool = False
    laser_mode: bool = False
    presentation_running: bool = False
    presentation_paused: bool = False
    gesture_cooldowns: Dict[str, float] = field(default_factory=dict)

    def effective_swipe_distance(self) -> float:
        """Lower sensitivity → easier swipes (smaller distance required)."""
        return SWIPE_MIN_DISTANCE / max(self.sensitivity, 0.1)

    def effective_confidence_threshold(self) -> float:
        """Higher sensitivity → lower confidence bar."""
        return GESTURE_CONFIDENCE_THRESHOLD / max(self.sensitivity, 0.1)


# Global runtime singleton (imported by modules)
runtime = RuntimeSettings()
