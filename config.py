"""
Global configuration for the Hand Paint application.

Centralizing settings here makes it easy to tune performance and
appearance without hunting through multiple source files.
"""

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0          # Default webcam (0 = first camera)
CAMERA_WIDTH = 1280       # Capture resolution (width)
CAMERA_HEIGHT = 720       # Capture resolution (height)
CAMERA_FPS_TARGET = 30    # Requested frames per second from the camera

# ---------------------------------------------------------------------------
# MediaPipe Hands
# ---------------------------------------------------------------------------
MAX_NUM_HANDS = 1         # Single-hand mode keeps drawing intuitive
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.6
# Note: MODEL_COMPLEXITY was used by the legacy MediaPipe API.
# The Tasks API uses a single bundled model (float16) optimised for speed.

# ---------------------------------------------------------------------------
# Drawing canvas
# ---------------------------------------------------------------------------
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720
DEFAULT_BRUSH_COLOR = (80, 180, 255)   # BGR – sky blue
DEFAULT_BRUSH_SIZE = 8
ERASER_SIZE = 40

# Smoothing factor for landmark positions (0 = no smoothing, 1 = frozen)
# Lower values react faster; higher values produce steadier strokes.
LANDMARK_SMOOTHING = 0.45

# ---------------------------------------------------------------------------
# Gesture detection thresholds (normalized 0–1 image coordinates)
# ---------------------------------------------------------------------------
PINCH_THRESHOLD = 0.05       # Thumb–index distance to trigger pinch/draw
FINGER_EXTENDED_RATIO = 0.65 # Tip must be above PIP joint by this fraction
PALM_OPEN_FINGER_COUNT = 4   # Fingers that must be extended for "open palm"

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
WINDOW_NAME = "Hand Paint – Virtual Finger Drawing"
SIDEBAR_WIDTH = 220
FPS_UPDATE_INTERVAL = 0.5    # Seconds between FPS label refreshes

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
