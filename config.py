"""
Global configuration for the Ekora application.
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS_TARGET = 30

# ---------------------------------------------------------------------------
# MediaPipe Hands
# ---------------------------------------------------------------------------
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.6

# ---------------------------------------------------------------------------
# Drawing canvas
# ---------------------------------------------------------------------------
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720
DEFAULT_BRUSH_COLOR = (80, 180, 255)   # BGR – sky blue
DEFAULT_BRUSH_SIZE = 8
ERASER_SIZE = 40

# Exponential smoothing for landmark positions (0 = raw/instant, 1 = frozen).
# Lower value = snappier cursor, less jitter damping.
# 0.25 is a good balance between responsiveness and steadiness.
LANDMARK_SMOOTHING = 0.25

# Maximum pixel gap between consecutive stroke points before sub-segment
# interpolation kicks in to prevent broken lines during fast movement.
STROKE_MAX_GAP_PX = 6

# ---------------------------------------------------------------------------
# Gesture detection thresholds
# ---------------------------------------------------------------------------
PINCH_THRESHOLD = 0.05
FINGER_EXTENDED_RATIO = 0.65
PALM_OPEN_FINGER_COUNT = 4

# Pan (fist) sensitivity – multiplies raw fingertip delta in pixels.
PAN_SENSITIVITY = 1.0

# Floating colour palette (peace gesture)
PALETTE_RADIUS_PX = 150
PALETTE_SWATCH_RADIUS_PX = 26

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
WINDOW_NAME = "Ekora"
SIDEBAR_WIDTH = 210
FPS_UPDATE_INTERVAL = 0.5

# Camera display overlay (0 = off, 1 = fully black)
CAMERA_OVERLAY_ALPHA = 0.65

# Color palette shown in the sidebar (BGR tuples)
COLOR_PALETTE = [
    (80, 180, 255),    # Sky blue
    (100, 100, 255),   # Coral red
    (80, 220, 120),    # Mint green
    (50, 200, 255),    # Amber
    (220, 160, 80),    # Lavender
    (255, 200, 80),    # Cyan
    (180, 120, 255),   # Pink
    (200, 200, 200),   # White
]

# ---------------------------------------------------------------------------
# Camera modes
# ---------------------------------------------------------------------------
CAMERA_MODE_NORMAL       = "normal"        # Raw camera feed
CAMERA_MODE_DARK_OVERLAY = "dark_overlay"  # 65-70% black overlay (default)
CAMERA_MODE_BLACK_CANVAS = "black_canvas"  # Fully black, landmarks only

# ---------------------------------------------------------------------------
# Stroke persistence (v1.5)
# ---------------------------------------------------------------------------
# How long (seconds) to keep a stroke alive after landmarks are momentarily lost.
STROKE_PERSISTENCE_TIMEOUT = 0.20   # 200 ms  (150–250 ms spec)

# ---------------------------------------------------------------------------
# Palm-clear safety (v1.5)
# ---------------------------------------------------------------------------
# Seconds the open-palm gesture must remain stable before canvas is cleared.
PALM_HOLD_DURATION = 0.15           # 150 ms hold required (v2.3 spec)

# Maximum normalised hand velocity (per second) allowed during palm hold.
# Anything above this cancels the confirmation.
PALM_MAX_VELOCITY = 0.25            # normalised units / second

# Minimum fraction of frames within the hold window where ALL 5 fingers
# must be extended (robustness against single noisy frames).
PALM_CONFIDENCE_RATIO = 0.80        # 80 % of frames in the window must agree

# ---------------------------------------------------------------------------
# Gesture state-machine (v1.5)
# ---------------------------------------------------------------------------
# Minimum consecutive frames a new gesture must be detected before the
# state machine transitions — prevents single noisy frames from switching state.
GESTURE_DEBOUNCE_FRAMES = 3

# ---------------------------------------------------------------------------
# Curve smoothing (v1.5)
# ---------------------------------------------------------------------------
# Number of historical stroke points used for Catmull-Rom look-back.
# Higher = smoother curves at speed, at the cost of a tiny (~1 frame) lag.
STROKE_SMOOTH_LOOKBACK = 4

# Sub-steps per interpolated segment (higher = smoother arcs).
STROKE_INTERP_STEPS = 8

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
_SETTINGS_PATH = Path(__file__).parent / "ekora_settings.json"

# ---------------------------------------------------------------------------
# Smart Shape Recognition
# ---------------------------------------------------------------------------
SHAPE_RECOGNITION_ENABLED = True   # default; overridden by persisted settings

_DEFAULTS = {
    "camera_mode": CAMERA_MODE_DARK_OVERLAY,
    "shape_recognition": True,
    "show_landmarks": True,
    "fps_limit": 60,
    "camera_resolution": "1280x720",
    "theme": "dark",
    "auto_save": True,
    "auto_save_interval": 300,
    "default_brush_size": DEFAULT_BRUSH_SIZE,
    "default_brush_color": DEFAULT_BRUSH_COLOR,
    "recent_files": [],
}

def load_settings() -> dict:
    """Load persisted settings, merging with defaults."""
    try:
        data = json.loads(_SETTINGS_PATH.read_text())
        merged = {**_DEFAULTS, **data}
        return merged
    except Exception:
        return dict(_DEFAULTS)

def save_settings(settings: dict) -> None:
    """Persist settings to disk."""
    try:
        _SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
    except Exception as e:
        print(f"[Warning] Could not save settings: {e}")
