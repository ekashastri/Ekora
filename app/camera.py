"""
Camera module – safe webcam capture with graceful error handling.

Responsibilities:
  • Open and configure the webcam.
  • Read frames in a loop-friendly way.
  • Mirror frames horizontally so the preview feels natural (like a mirror).
  • Release resources cleanly on shutdown.
"""

from __future__ import annotations

import sys
from typing import Optional, Tuple

import cv2
import numpy as np

import config


class CameraError(Exception):
    """Raised when the webcam cannot be opened or read."""


class Camera:
    """
    Thin wrapper around OpenCV's VideoCapture.

    Usage
    -----
    camera = Camera()
    if camera.is_opened():
        ok, frame = camera.read()
    camera.release()
    """

    def __init__(
        self,
        index: int = config.CAMERA_INDEX,
        width: int = config.CAMERA_WIDTH,
        height: int = config.CAMERA_HEIGHT,
        fps: int = config.CAMERA_FPS_TARGET,
        mirror: bool = True,
    ) -> None:
        self.index = index
        self.mirror = mirror
        self._cap: Optional[cv2.VideoCapture] = None
        self._open(width, height, fps)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _open(self, width: int, height: int, fps: int) -> None:
        """Attempt to open the camera and apply preferred settings."""
        self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)

        if not self._cap.isOpened():
            # Fallback: try the default backend (works on macOS / Linux)
            self._cap = cv2.VideoCapture(self.index)

        if not self._cap.isOpened():
            raise CameraError(
                f"Cannot open camera at index {self.index}.\n"
                "Check that a webcam is connected and not used by another app."
            )

        # Request resolution / FPS – the driver may round to nearest supported value.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

        # Reduce internal buffer to minimise latency (smoother real-time feel).
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def is_opened(self) -> bool:
        """Return True if the capture device is ready."""
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Grab the next frame.

        Returns
        -------
        (success, frame)
            frame is a BGR numpy array mirrored horizontally when configured.
            Returns (False, None) on read failure.
        """
        if not self.is_opened():
            return False, None

        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None

        if self.mirror:
            frame = cv2.flip(frame, 1)

        return True, frame

    def release(self) -> None:
        """Release the capture device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def create_camera() -> Camera:
    """
    Factory that creates a Camera or prints a helpful message and exits.

    Call this from main() so startup errors are user-friendly.
    """
    try:
        return Camera()
    except CameraError as exc:
        print("\n[Camera Error]", exc, file=sys.stderr)
        print(
            "\nTroubleshooting tips:\n"
            "  • Close other apps that may be using the webcam (Zoom, Teams…)\n"
            "  • Try changing CAMERA_INDEX in config.py (0, 1, 2…)\n"
            "  • Reconnect the webcam and restart the application\n",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Threaded camera reader
# ---------------------------------------------------------------------------
import threading
import time

class ThreadedCamera:
    """
    Reads camera frames in a daemon thread so the main loop always gets
    the most recent frame without blocking on VideoCapture.read().
    """

    def __init__(self, camera: Camera) -> None:
        self._cam   = camera
        self._frame = None
        self._ok    = False
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cam.read()
            if ok:
                with self._lock:
                    self._frame = frame
                    self._ok    = True
            else:
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            return self._ok, (self._frame.copy() if self._frame is not None else None)

    def release(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._cam.release()
