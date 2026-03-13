# Real-Time Indian Classical Mudra Detection System

A computer vision system that detects and classifies eight classical Indian hand mudras in real-time using webcam feed.

## Installation

Install the required dependencies:

```bash
pip install mediapipe opencv-python
```

## Usage

Run the mudra detection system:

```bash
python main.py
```

Press 'q' to quit the application.

## Supported Mudras

The system recognizes the following classical Indian mudras:

### Original Mudras
**Pataka**
All five fingers extended.

**Tripataka**
Ring finger bent, others extended.

**Alapadma**
All fingers extended and spread widely (fingers noticeably apart).

**Ardhachandra**
All fingers extended, thumb stretched far outward.

**Shikhara**
All fingers folded into a fist, thumb extended upward.

### New Mudras
**Hamsasya**
Thumb touching index finger.

**Mukula**
All fingertips touching together.

**Mushti**
Closed fist (all fingers bent).

## Project Structure

```
project/
├── main.py              # Main application with webcam integration
├── hand_tracker.py      # MediaPipe hand detection and landmark tracking
├── mudra_recognizer.py  # Rule-based mudra recognition logic
└── README.md           # This file
```

## Features

- **Real-time webcam processing** at ~30 FPS
- **Mirror view** for natural interaction
- **Support for both hands** (Left/Right) with proper handedness detection
- **Visual hand landmark overlay**
- **Temporal smoothing** to reduce flickering
- **Color-coded mudra display** (green for known, red for unknown)
- **Debug overlay** showing finger states
- **FPS counter**
- **Clean, modular code structure**

## Improvements Made

### Enhanced Detection Accuracy
- Improved finger state detection with better geometric analysis
- Proper left/right hand classification using MediaPipe handedness
- More robust finger extension detection rules

### Additional Mudras Support
- Added Hamsasya (thumb touching index finger)
- Added Mukula (all fingertips touching)
- Added Mushti (closed fist)

### Stability Improvements
- **Temporal smoothing**: Maintains 5-frame history buffer for each hand
- **Confidence-based detection**: Requires consistent detection over multiple frames
- **Reduced flickering**: Mudra names only change when consistently detected

### Enhanced Display
- **Color-coded text**: Green for recognized mudras, red for unknown
- **Debug overlay**: Shows individual finger states (Extended/Bent)
- **Better UI**: Lists supported mudras and current mode

## System Requirements

- Python 3.7+
- Webcam
- OpenCV-compatible operating system (Windows, macOS, Linux)

## How It Works

1. **Hand Detection**: Uses MediaPipe Hands to detect 21 hand landmarks
2. **Handedness Classification**: Correctly identifies left vs right hands
3. **Finger State Analysis**: Determines if each finger is extended or bent
4. **Rule-Based Recognition**: Applies heuristic rules to identify mudras
5. **Temporal Smoothing**: Stabilizes detection using frame history
6. **Real-time Display**: Shows detected mudra names with debug info

## Debug Mode

The system includes a debug overlay that displays:
- Individual finger states (Extended/Bent)
- Helps tune recognition rules
- Can be toggled on/off in the code

## Expected Performance

- **Accuracy**: Improved detection of all 8 mudras
- **Stability**: Reduced flickering with temporal smoothing
- **Left-hand reliability**: Better left-hand detection accuracy
- **Real-time performance**: Maintains ~30 FPS with enhanced features
