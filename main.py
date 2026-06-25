"""
Hand Paint – main entry point.

Orchestrates the camera loop:
  1. Capture a webcam frame
  2. Detect hand landmarks (MediaPipe)
  3. Classify gesture
  4. Update drawing canvas
  5. Composite + render UI
  6. Display result

Keyboard shortcuts
------------------
  Q / ESC  – Quit
  C        – Clear canvas
  S        – Save canvas as PNG
  R        – Reset brush to default colour
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

import config
from app.camera import create_camera
from app.drawing_canvas import DrawingCanvas
from app.gesture_detector import Gesture, GestureDetector
from app.hand_tracker import HandLandmarks, HandTracker, INDEX_TIP
from app.ui import UIRenderer


def main() -> None:
    """Run the Hand Paint application."""
    print("=" * 50)
    print("  Hand Paint – Virtual Finger Drawing")
    print("=" * 50)
    print("\nStarting camera and hand tracker…\n")

    camera = create_camera()
    tracker = HandTracker()
    gestures = GestureDetector()
    canvas = DrawingCanvas()
    ui = UIRenderer()

    # Track previous gesture to detect transitions (e.g. palm → clear once)
    prev_gesture = Gesture.NONE

    window = config.WINDOW_NAME
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

    print("Controls:")
    print("  Pinch (thumb + index)  → Draw")
    print("  Point (index finger)   → Hover")
    print("  Fist                   → Erase")
    print("  Peace sign             → Next colour")
    print("  Open palm              → Clear canvas")
    print("  Q / ESC                → Quit")
    print("  C                      → Clear canvas (keyboard)")
    print("  S                      → Save drawing\n")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("[Warning] Failed to read frame – retrying…", file=sys.stderr)
                time.sleep(0.05)
                continue

            # Resize frame to expected dimensions if the camera returned something else
            if frame.shape[1] != config.CAMERA_WIDTH or frame.shape[0] != config.CAMERA_HEIGHT:
                frame = cv2.resize(frame, (config.CAMERA_WIDTH, config.CAMERA_HEIGHT))

            hand = tracker.process(frame)
            gesture = Gesture.NONE
            cursor_pos = None

            if hand is not None:
                gesture = gestures.detect(hand, frame.shape[:2])
                cursor_pos = hand.landmarks[INDEX_TIP]
                _handle_gesture(gesture, prev_gesture, hand, canvas, ui)
                tracker.draw_skeleton(frame, hand)
                prev_gesture = gesture
            else:
                canvas.end_stroke()
                prev_gesture = Gesture.NONE

            # Composite canvas onto camera feed
            composited = canvas.composite(frame)

            # Draw UI chrome
            output = ui.render(
                composited,
                gesture=gesture,
                brush_color=canvas.brush_color,
                brush_size=canvas.brush_size,
                hand_detected=hand is not None,
                cursor_pos=cursor_pos,
            )

            cv2.imshow(window, output)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                break
            elif key in (ord("c"), ord("C")):
                canvas.clear()
                ui.flash_message("Canvas Cleared!")
            elif key in (ord("s"), ord("S")):
                _save_canvas(canvas)
            elif key in (ord("r"), ord("R")):
                canvas.set_color(config.DEFAULT_BRUSH_COLOR)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("Shutting down…")
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()
        print("Goodbye!")


def _handle_gesture(
    gesture: Gesture,
    prev_gesture: Gesture,
    hand: HandLandmarks,
    canvas: DrawingCanvas,
    ui: UIRenderer,
) -> None:
    """
    Map the detected gesture to canvas actions.

    Stroke lifecycle
    ----------------
    PINCH / FIST  → start or continue a stroke at the index fingertip
    Other gestures → end any active stroke
    OPEN_PALM (on transition) → clear canvas once
    PEACE (on transition) → cycle brush colour once
    """
    tip = hand.landmarks[INDEX_TIP]

    if gesture == Gesture.PINCH:
        if prev_gesture != Gesture.PINCH:
            canvas.start_stroke(tip)
        else:
            canvas.continue_stroke(tip, eraser=False)

    elif gesture == Gesture.FIST:
        if prev_gesture != Gesture.FIST:
            canvas.start_stroke(tip)
        else:
            canvas.continue_stroke(tip, eraser=True)

    elif gesture == Gesture.OPEN_PALM and prev_gesture != Gesture.OPEN_PALM:
        canvas.clear()
        ui.flash_message("Canvas Cleared!")

    elif gesture == Gesture.PEACE and prev_gesture != Gesture.PEACE:
        canvas.cycle_color()
        ui.flash_message("Color Changed!")

    else:
        canvas.end_stroke()


def _save_canvas(canvas: DrawingCanvas) -> None:
    """Save the current canvas to a timestamped PNG in the project folder."""
    output_dir = Path(__file__).parent / "saves"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"drawing_{timestamp}.png"
    cv2.imwrite(str(filepath), canvas.get_canvas())
    print(f"Drawing saved → {filepath}")


if __name__ == "__main__":
    main()
