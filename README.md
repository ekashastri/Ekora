# Hand Paint – Virtual Finger Drawing

A beginner-friendly, real-time hand-tracking drawing app built with **Python**, **MediaPipe**, and **OpenCV**. Pinch to draw, make a fist to erase, show an open palm to clear the canvas, and flash a peace sign to change colours — all in the air, using your webcam.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-orange)

---

## Features

- Real-time hand skeleton overlay with smoothed landmarks
- Pinch-to-draw with anti-aliased brush strokes
- Fist gesture eraser mode
- Open-palm gesture to clear the canvas
- Peace sign to cycle through 8 colours
- Modern dark UI with sidebar, FPS counter, and gesture HUD
- Graceful camera error handling with troubleshooting tips
- Save drawings as PNG (`S` key)
- Optimised for smooth performance on normal laptops

---

## Project Structure

```
hand-paint/
├── main.py                  # Application entry point
├── config.py                # Centralised settings (camera, gestures, UI)
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── saves/                   # Auto-created when you save a drawing
└── app/
    ├── __init__.py
    ├── camera.py            # Webcam capture with error handling
    ├── hand_tracker.py      # MediaPipe Hands wrapper + smoothing
    ├── gesture_detector.py  # Pinch, fist, palm, peace detection
    ├── drawing_canvas.py    # Virtual paint layer
    └── ui.py                # Sidebar, HUD, and visual overlays
```

---

## Requirements

- **Python 3.8+** (tested on Python 3.11–3.14)
- A working webcam
- Windows, macOS, or Linux

---

## Setup

### 1. Clone or download the project

```bash
cd hand-paint
```

### 2. Create a virtual environment (recommended)

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python main.py
```

The camera window opens automatically. Hold your hand in front of the webcam to begin.

---

## Gesture Controls

| Gesture | How to perform | Action |
|---------|----------------|--------|
| **Pinch** | Touch thumb tip to index tip | Draw on canvas |
| **Point** | Extend index finger only | Hover / move cursor |
| **Fist** | Curl all fingers | Erase |
| **Peace** | Index + middle fingers up | Cycle brush colour |
| **Open palm** | Extend all fingers | Clear canvas |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit |
| `C` | Clear canvas |
| `S` | Save drawing to `saves/` |
| `R` | Reset brush to default colour |

---

## Configuration

Edit `config.py` to tune the app without touching core logic:

| Setting | Default | Description |
|---------|---------|-------------|
| `CAMERA_INDEX` | `0` | Webcam device index (try `1`, `2` if default fails) |
| `CAMERA_WIDTH` / `HEIGHT` | `1280` / `720` | Capture resolution |
| `MAX_NUM_HANDS` | `1` | Number of hands to track simultaneously |
| `LANDMARK_SMOOTHING` | `0.45` | Jitter reduction (0 = off, 1 = frozen) |
| `PINCH_THRESHOLD` | `0.05` | Sensitivity for pinch detection |
| `DEFAULT_BRUSH_SIZE` | `8` | Brush stroke thickness |

---

## Troubleshooting

**"Cannot open camera at index 0"**
- Close other apps using the webcam (Zoom, Teams, browser tabs).
- Change `CAMERA_INDEX` in `config.py` to `1` or `2`.
- Reconnect the webcam and restart.

**Low FPS / laggy tracking**
- Lower `CAMERA_WIDTH` and `CAMERA_HEIGHT` to `640` and `480` in `config.py`.
- Ensure `MODEL_COMPLEXITY = 0`.
- Improve lighting so your hand is clearly visible.

**Hand not detected**
- Use a plain background.
- Keep your hand 30–80 cm from the camera.
- Lower `MIN_DETECTION_CONFIDENCE` to `0.5` in `config.py`.

**`pip install mediapipe` fails**
- Confirm Python version: `python --version` (3.8+ required).
- Upgrade pip: `python -m pip install --upgrade pip`

**Model download on first run**
- The hand landmarker model (~3 MB) downloads automatically to `models/` on first launch.
- Ensure you have an internet connection for the first run only.

---

## Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Camera    │────▶│ Hand Tracker │────▶│ Gesture Detector│
│  (OpenCV)   │     │ (MediaPipe)  │     │  (geometry)     │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                     ┌──────────────┐              ▼
                     │  UI Renderer │◀──── ┌───────────────┐
                     │  (HUD/HUD)   │     │ Drawing Canvas │
                     └──────┬───────┘     └───────────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │  Display    │
                     │  (OpenCV)   │
                     └─────────────┘
```

1. **Camera** captures mirrored webcam frames at the configured resolution.
2. **Hand Tracker** runs MediaPipe Hands, converts 21 landmarks to pixel coordinates, and smooths them.
3. **Gesture Detector** classifies the hand pose using geometric rules (finger angles, tip distances).
4. **Drawing Canvas** records brush strokes on an off-screen buffer and composites them over the camera feed.
5. **UI Renderer** draws the sidebar, colour palette, FPS counter, and gesture status.
6. **Main loop** ties everything together at ~30 FPS.

---

## How Hand Landmarks Work

MediaPipe Hands detects **21 3-D landmarks** per hand:

```
        8   12  16  20       ← fingertips
        |   |   |   |
        7   11  15  19
        |   |   |   |
        6   10  14  18
        |   |   |   |
    4   5   9   13  17
     \  |   |   |   /
      \ |   |   |  /
       \|   |   | /
        0 (wrist)
   thumb: 1–4
```

Each landmark has normalised `(x, y, z)` coordinates. This app:
- Converts them to pixel `(x, y)` based on frame size.
- Applies exponential smoothing to reduce jitter in drawn lines.
- Uses **index fingertip (landmark 8)** as the primary cursor/draw point.
- Uses **thumb tip (4) + index tip (8) distance** for pinch detection.

---

## Gesture Detection Logic

All gestures are detected with **scale-invariant geometry** (normalised 0–1 coordinates):

| Gesture | Detection rule |
|---------|----------------|
| **Pinch** | Euclidean distance between thumb tip and index tip `< PINCH_THRESHOLD` |
| **Peace** | Index and middle extended; ring and pinky curled |
| **Open palm** | ≥ 4 fingers extended (triggers once with cooldown) |
| **Fist** | Zero fingers extended |
| **Point** | Only index finger extended |

Finger "extended" is determined by comparing the tip's Y position to the PIP and DIP joints relative to the wrist — if the tip is higher on screen than both joints, the finger is up.

---

## Licence

MIT – free to use, modify, and share.
