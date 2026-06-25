"""
Gesture Presentation Assistant – main application entry point.

Orchestrates:
  1. Webcam capture with error recovery
  2. Hand tracking (MediaPipe)
  3. Gesture recognition → presentation control
  4. Air annotation & laser pointer
  5. Voice commands (background thread)
  6. Pygame UI rendering
"""

from __future__ import annotations

import sys
import time
from typing import List, Optional

import cv2
import numpy as np

import settings
from annotation_manager import AnnotationManager
from gesture_detector import Gesture, GestureDetector, GestureResult
from hand_tracker import HandLandmarks, HandTracker
from presentation_controller import PresentationController
from ui import UIRenderer
from voice_commands import VoiceCommandHandler


class Camera:
    """Webcam wrapper with automatic reconnect on failure."""

    def __init__(self, index: int = settings.CAMERA_INDEX) -> None:
        self.index = index
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_reconnect = 0.0
        self.open()

    def open(self) -> bool:
        """Open or reopen the camera device."""
        if self._cap is not None:
            self._cap.release()

        self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.index)

        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_HEIGHT)
            self._cap.set(cv2.CAP_PROP_FPS, settings.CAMERA_FPS_TARGET)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return True
        return False

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read a frame; attempt reconnect on failure."""
        if self._cap is None or not self._cap.isOpened():
            now = time.time()
            if now - self._last_reconnect > settings.WEBCAM_RECONNECT_INTERVAL:
                self._last_reconnect = now
                self.open()
            return False, None

        ok, frame = self._cap.read()
        if not ok or frame is None:
            now = time.time()
            if now - self._last_reconnect > settings.WEBCAM_RECONNECT_INTERVAL:
                self._last_reconnect = now
                print("[Camera] Reconnecting…", file=sys.stderr)
                self.open()
            return False, None

        # Ensure consistent processing size
        if frame.shape[1] != settings.PROCESS_WIDTH or frame.shape[0] != settings.PROCESS_HEIGHT:
            frame = cv2.resize(frame, (settings.PROCESS_WIDTH, settings.PROCESS_HEIGHT))
        return True, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class GesturePresentationApp:
    """Main application class tying all modules together."""

    def __init__(self) -> None:
        self.camera = Camera()
        self.tracker = HandTracker(max_num_hands=settings.runtime.max_num_hands)
        self.gestures = GestureDetector()
        self.presentation = PresentationController()
        self.annotations = AnnotationManager()
        self.ui = UIRenderer()
        self.voice = VoiceCommandHandler(
            on_next=lambda: self.presentation.next_slide("voice"),
            on_previous=lambda: self.presentation.previous_slide("voice"),
            on_start=lambda: self.presentation.start_presentation("voice"),
            on_stop=lambda: self.presentation.stop_presentation("voice"),
        )

        self._prev_gesture = Gesture.NONE
        self._prev_result = GestureResult()
        self._fps = 0.0
        self._fps_frame_count = 0
        self._fps_timer = time.time()
        self._running = True
        self._recording_gesture = False

    def run(self) -> None:
        """Main event loop."""
        self._print_banner()
        self.ui.init()

        if self.voice.is_available:
            started = self.voice.start()
            if started:
                print("Voice commands: active")
            else:
                print("Voice commands: unavailable (check microphone)")
        else:
            print("Voice commands: SpeechRecognition not installed")

        try:
            while self._running:
                self._process_events()
                self._process_voice()
                frame = self._capture_frame()
                if frame is None:
                    time.sleep(0.03)
                    continue

                hands, result = self._process_frame(frame)
                self._handle_gesture(result)
                self._update_modes(hands, result)
                output = self._compose_output(frame, hands, result)
                self._fps = self._update_fps()
                self.ui.render(
                    output,
                    gesture_result=result,
                    fps=self._fps,
                    slide_index=self.presentation.slide_index,
                    total_slides=self.presentation.total_slides,
                    presentation_running=self.presentation.is_running,
                    presentation_paused=self.presentation.is_paused,
                    hands_count=len(hands),
                    voice_active=self.voice.is_available and settings.runtime.voice_enabled,
                    recording_gesture=self._recording_gesture,
                )
        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # Frame pipeline
    # ------------------------------------------------------------------

    def _capture_frame(self) -> Optional[np.ndarray]:
        ok, frame = self.camera.read()
        return frame if ok else None

    def _process_frame(self, frame: np.ndarray) -> tuple[List[HandLandmarks], GestureResult]:
        hands = self.tracker.process(frame)
        result = self.gestures.detect(hands, frame.shape[:2])
        self._prev_result = result
        return hands, result

    def _compose_output(
        self,
        frame: np.ndarray,
        hands: List[HandLandmarks],
        result: GestureResult,
    ) -> np.ndarray:
        output = frame.copy()

        # Draw skeletons for all detected hands
        for i, hand in enumerate(hands):
            color = (0, 255, 180) if i == 0 else (255, 180, 0)
            self.tracker.draw_skeleton(output, hand, point_color=color, line_color=color)

        # Annotation layer
        if settings.runtime.annotation_mode:
            output = self.annotations.composite(output)

        # Laser pointer (when not in annotation mode)
        if settings.runtime.laser_mode and not settings.runtime.annotation_mode:
            self.annotations.draw_laser(output)

        return output

    def _update_modes(self, hands: List[HandLandmarks], result: GestureResult) -> None:
        """Update laser trail and annotation drawing based on current pose."""
        if not hands:
            self.annotations.end_stroke()
            self.annotations.update_laser(None)
            return

        hand = hands[0]
        tip = hand.index_tip

        if settings.runtime.laser_mode and not settings.runtime.annotation_mode:
            self.annotations.update_laser(tip)

        if settings.runtime.annotation_mode:
            gesture = result.gesture
            if gesture == Gesture.PINCH:
                if self._prev_gesture != Gesture.PINCH:
                    self.annotations.start_stroke(tip, eraser=False)
                else:
                    self.annotations.continue_stroke(tip)
            elif gesture == Gesture.FIST:
                if self._prev_gesture != Gesture.FIST:
                    self.annotations.start_stroke(tip, eraser=True)
                else:
                    self.annotations.continue_stroke(tip)
            elif gesture == Gesture.POINT:
                if self._prev_gesture != Gesture.POINT:
                    self.annotations.start_stroke(tip, eraser=False)
                else:
                    self.annotations.continue_stroke(tip)
            else:
                self.annotations.end_stroke()

    # ------------------------------------------------------------------
    # Gesture → action dispatch
    # ------------------------------------------------------------------

    def _handle_gesture(self, result: GestureResult) -> None:
        """Map confirmed gestures to presentation / annotation actions."""
        gesture = result.gesture
        conf = result.confidence
        threshold = settings.runtime.effective_confidence_threshold()

        # Only act on gestures that passed cooldown + confidence in detector
        if conf < threshold:
            self._prev_gesture = gesture
            return

        # Avoid re-triggering the same static gesture every frame
        triggered = gesture != self._prev_gesture or gesture in (
            Gesture.SWIPE_RIGHT,
            Gesture.SWIPE_LEFT,
            Gesture.SWIPE_DOWN,
        )

        if gesture == Gesture.SWIPE_RIGHT and triggered:
            self.presentation.next_slide()
            self.ui.flash("Next slide →")

        elif gesture == Gesture.SWIPE_LEFT and triggered:
            self.presentation.previous_slide()
            self.ui.flash("← Previous slide")

        elif gesture == Gesture.OPEN_PALM and triggered:
            self.presentation.start_presentation()
            self.ui.flash("Presentation started")

        elif gesture == Gesture.FIST and triggered:
            self.presentation.pause_presentation()
            self.ui.flash("Paused")

        elif gesture == Gesture.THUMBS_UP and triggered:
            self.presentation.resume_presentation()
            self.ui.flash("Resumed")

        elif gesture == Gesture.PEACE and triggered:
            settings.runtime.annotation_mode = not settings.runtime.annotation_mode
            settings.runtime.laser_mode = False
            state = "ON" if settings.runtime.annotation_mode else "OFF"
            self.ui.flash(f"Annotation mode {state}")

        elif gesture == Gesture.SWIPE_DOWN and triggered and settings.runtime.annotation_mode:
            if self.annotations.undo():
                self.ui.flash("Undo")

        elif gesture == Gesture.CUSTOM and triggered:
            self.ui.flash(f"Custom: {result.custom_name}")

        self._prev_gesture = gesture

    # ------------------------------------------------------------------
    # Events & voice
    # ------------------------------------------------------------------

    def _process_events(self) -> None:
        for action in self.ui.handle_events():
            if action == "quit":
                self._running = False
            elif action == "toggle_annotation":
                settings.runtime.annotation_mode = not settings.runtime.annotation_mode
                if settings.runtime.annotation_mode:
                    settings.runtime.laser_mode = False
                self.ui.flash(f"Annotation {'ON' if settings.runtime.annotation_mode else 'OFF'}")
            elif action == "toggle_laser":
                settings.runtime.laser_mode = not settings.runtime.laser_mode
                if settings.runtime.laser_mode:
                    settings.runtime.annotation_mode = False
                self.ui.flash(f"Laser {'ON' if settings.runtime.laser_mode else 'OFF'}")
            elif action == "toggle_voice":
                settings.runtime.voice_enabled = not settings.runtime.voice_enabled
                self.ui.flash(f"Voice {'ON' if settings.runtime.voice_enabled else 'OFF'}")
            elif action == "save_annotation":
                path = self.annotations.save()
                self.presentation.stats.annotations_saved += 1
                self.ui.flash(f"Saved → {path}")
            elif action == "undo":
                if self.annotations.undo():
                    self.ui.flash("Undo")
            elif action == "sensitivity_up":
                settings.runtime.sensitivity = min(
                    settings.SENSITIVITY_MAX,
                    settings.runtime.sensitivity + 0.1,
                )
                self.ui.flash(f"Sensitivity: {settings.runtime.sensitivity:.1f}x")
            elif action == "sensitivity_down":
                settings.runtime.sensitivity = max(
                    settings.SENSITIVITY_MIN,
                    settings.runtime.sensitivity - 0.1,
                )
                self.ui.flash(f"Sensitivity: {settings.runtime.sensitivity:.1f}x")

    def _process_voice(self) -> None:
        action = self.voice.poll()
        if action:
            self.presentation.stats.voice_commands += 1
            self.voice.dispatch(action)
            labels = {
                "next": "Voice: Next slide",
                "previous": "Voice: Previous slide",
                "start": "Voice: Start",
                "stop": "Voice: Stop",
            }
            self.ui.flash(labels.get(action, f"Voice: {action}"))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_fps(self) -> float:
        self._fps_frame_count += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= 1.0:
            self._fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_timer = time.time()
        return self._fps

    def _shutdown(self) -> None:
        print("Saving session analytics…")
        path = self.presentation.stats.save()
        print(f"Analytics → {path}")
        self.voice.stop()
        self.tracker.close()
        self.camera.release()
        self.ui.quit()
        print("Goodbye!")

    @staticmethod
    def _print_banner() -> None:
        print("=" * 56)
        print("  Gesture Presentation Assistant")
        print("=" * 56)
        print("\nGesture controls:")
        print("  Swipe right      → Next slide")
        print("  Swipe left       → Previous slide")
        print("  Open palm        → Start presentation")
        print("  Closed fist      → Pause")
        print("  Thumbs up        → Resume")
        print("  Peace sign       → Toggle annotation")
        print("  Pinch / Point    → Draw (annotation mode)")
        print("  Swipe down       → Undo annotation")
        print()


def main() -> None:
    app = GesturePresentationApp()
    app.run()


if __name__ == "__main__":
    main()
