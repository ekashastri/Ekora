# Gesture Presentation Assistant

Control PowerPoint, Google Slides, PDFs, and any arrow-key-driven presentation entirely with hand gestures and voice commands.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Webcam    │────▶│ Hand Tracker │────▶│ Gesture Detector│
│  (OpenCV)   │     │ (MediaPipe)  │     │ (pose + swipe)  │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                     ┌─────────────────────────────┼─────────────────────────────┐
                     ▼                             ▼                             ▼
           ┌─────────────────┐         ┌──────────────────┐           ┌─────────────────┐
           │  Presentation   │         │   Annotation     │           │  Voice Commands │
           │  Controller     │         │   Manager        │           │  (background)   │
           │  (PyAutoGUI)    │         │  (air drawing)   │           │                 │
           └────────┬────────┘         └────────┬─────────┘           └────────┬────────┘
                    │                           │                              │
                    └───────────────────────────┼──────────────────────────────┘
                                                ▼
                                      ┌──────────────────┐
                                      │   Pygame UI      │
                                      │ (glassmorphism)  │
                                      └──────────────────┘
```

| Module | Responsibility |
|---|---|
| `app.py` | Main loop, event routing, module orchestration |
| `hand_tracker.py` | MediaPipe hand landmarks, smoothing, skeleton overlay |
| `gesture_detector.py` | Static pose + dynamic swipe recognition, cooldowns |
| `presentation_controller.py` | Keyboard automation via PyAutoGUI, session analytics |
| `annotation_manager.py` | Air drawing, laser pointer, undo, save |
| `voice_commands.py` | Background speech recognition thread |
| `ui.py` | Dark glassmorphism Pygame HUD |
| `settings.py` | Central configuration and runtime toggles |

---

## Gesture Recognition Pipeline

1. **Capture** – Webcam frame at 640×480 (configurable).
2. **Detect** – MediaPipe Hand Landmarker returns up to 21 landmarks per hand in VIDEO mode.
3. **Smooth** – Exponential moving average per landmark reduces jitter.
4. **Classify static pose** – Finger extension geometry identifies palm, fist, thumbs up, peace, point, pinch.
5. **Classify dynamic swipe** – Palm-centre position history over ~18 frames detects horizontal/vertical motion.
6. **Score confidence** – Distance ratios, finger clarity, and swipe velocity produce a 0–1 score.
7. **Filter** – Per-gesture cooldown timers and sensitivity-adjusted threshold prevent false positives.
8. **Dispatch** – Confirmed gestures trigger presentation keys, mode toggles, or annotation actions.

---

## Presentation Automation

PyAutoGUI sends standard keyboard shortcuts to the active window:

| Action | Key | Works with |
|---|---|---|
| Next slide | `→` | PowerPoint, Slides, PDF viewers |
| Previous slide | `←` | PowerPoint, Slides, PDF viewers |
| Start slideshow | `F5` | PowerPoint, Impress |
| Pause (black screen) | `B` | PowerPoint |
| Resume | `B` (toggle) | PowerPoint |
| Stop | `Esc` | Most slideshow apps |

**Tip:** Click on your presentation window before gesturing so it receives the keystrokes.

---

## Annotation System

- **Enable:** Peace sign gesture or press `A`.
- **Draw:** Pinch, point, or index finger extended.
- **Erase:** Closed fist (wide eraser stroke).
- **Colours:** Six preset colours; peace sign cycles in annotation mode.
- **Undo:** Swipe down or press `Z`.
- **Save:** Press `S` → PNG saved to `saves/annotations/`.
- **Laser pointer:** Press `L` – red animated dot follows index fingertip.

---

## Performance Optimizations

- **640×480 processing** – Lower resolution for MediaPipe; upscaled in UI.
- **VIDEO running mode** – Reuses tracking state between frames (faster than IMAGE mode).
- **Buffer size 1** – Minimises webcam latency.
- **Selective compositing** – Annotation layer uses mask-based blending.
- **Background voice thread** – Never blocks the camera loop.
- **Gesture cooldowns** – Prevent duplicate key sends.
- **Camera auto-reconnect** – Graceful recovery from device errors.

Target: **≥ 30 FPS** on a modern laptop webcam.

---

## Setup

### Prerequisites

- Python 3.10 – 3.13 recommended (3.14+ uses `pygame-ce`, a drop-in Pygame replacement)
- Webcam
- Windows, macOS, or Linux
- Microphone (optional, for voice commands)

### Install

```bash
cd gesture_presentation_assistant
pip install -r requirements.txt
python app.py
```

### PyAudio on Windows

If `pip install PyAudio` fails:

```bash
pip install pipwin
pipwin install pyaudio
```

Or download a wheel from [https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio).

Voice commands are optional – the app runs without PyAudio; gesture control still works.

---

## Gesture Reference

| Gesture | Action |
|---|---|
| Swipe right | Next slide |
| Swipe left | Previous slide |
| Open palm | Start presentation |
| Closed fist | Pause |
| Thumbs up | Resume |
| Peace sign | Toggle annotation mode |
| Pinch / Point | Draw (annotation mode) |
| Fist | Erase (annotation mode) |
| Swipe down | Undo annotation |

## Voice Commands

| Say | Action |
|---|---|
| "Next slide" | Next slide |
| "Previous slide" / "Go back" | Previous slide |
| "Start presentation" | Start slideshow |
| "Stop presentation" | Exit slideshow |

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `A` | Toggle annotation mode |
| `L` | Toggle laser pointer |
| `V` | Toggle voice commands |
| `S` | Save annotation |
| `Z` | Undo annotation |
| `+` / `-` | Increase / decrease sensitivity |
| `Q` / `Esc` | Quit |

---

## Custom Gesture Training

1. Programmatically call `gestures.start_recording("my_gesture")` while performing the pose repeatedly.
2. Call `gestures.stop_recording()` to save a template to `custom_gestures/`.
3. The detector will match against saved templates at runtime.

---

## Project Structure

```
gesture_presentation_assistant/
├── app.py                      # Main entry point
├── hand_tracker.py             # MediaPipe hand tracking
├── gesture_detector.py         # Gesture recognition
├── presentation_controller.py  # PyAutoGUI automation
├── annotation_manager.py       # Air drawing & laser
├── voice_commands.py           # Speech recognition
├── ui.py                       # Pygame HUD
├── settings.py                 # Configuration
├── requirements.txt
├── README.md
├── assets/                     # Static assets
├── models/                     # Auto-downloaded MediaPipe model
├── saves/
│   ├── annotations/            # Saved annotation PNGs
│   └── analytics/              # Session statistics
└── custom_gestures/            # User-trained gesture JSON
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Camera not found | Change `CAMERA_INDEX` in `settings.py` (try 0, 1, 2) |
| Low FPS | Close other apps; reduce `PROCESS_WIDTH/HEIGHT` |
| Gestures not detected | Increase sensitivity with `+` key; improve lighting |
| Keys not reaching slides | Click the presentation window first |
| Voice not working | Install PyAudio; check microphone permissions |
| Model download fails | Check internet; model caches in `models/` |

---

## License

MIT – use freely for presentations, meetings, and demos.
