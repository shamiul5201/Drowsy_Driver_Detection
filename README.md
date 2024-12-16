# Drowsiness Detection System

This project implements a **drowsiness detection system** using OpenCV, MediaPipe, Streamlit, and Pygame. The application monitors eye movement and triggers an alarm if drowsiness is detected. The program is designed to run in real-time using a webcam feed.

## Features
- **Eye Aspect Ratio (EAR):** Tracks eye landmarks to compute the EAR and determine drowsiness.
- **MediaPipe Integration:** Utilizes MediaPipe's FaceMesh solution for landmark detection.
- **Real-Time Detection:** Processes webcam feed in real-time to monitor user behavior.
- **Alarm System:** Plays an alarm sound when drowsiness is detected.
- **Streamlit Interface:** Offers an interactive UI for deploying the application as a web app.

---

## Prerequisites
Ensure the following dependencies are installed:

- Python 3.7+
- OpenCV
- MediaPipe
- Streamlit
- Pygame

### Installation
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install the required libraries:
   ```bash
   pip install opencv-python mediapipe streamlit pygame
   ```

3. Ensure you have the alarm sound file `wake-up.mp3` in the root directory.

---

## How It Works

1. **Landmark Detection:**
   The program uses MediaPipe’s FaceMesh to detect face landmarks. Specific landmarks around the eyes are tracked to calculate the Eye Aspect Ratio (EAR).

2. **Drowsiness Detection:**
   - If the EAR drops below a threshold for a significant period, it indicates that the eyes are closed.
   - This triggers the alarm to alert the user.

3. **Alarm System:**
   The alarm is implemented using Pygame’s mixer module. The alarm will keep playing until the user opens their eyes.

---

## Code Overview

### Imports and Constants
```python
import cv2
import time
import mediapipe as mp
import streamlit as st
from pygame import mixer
from mediapipe.python.solutions.drawing_utils import (
    _normalized_to_pixel_coordinates as denormalize_coordinates,
)

# Constants for landmarks and colors
left_eye_idxs = [362, 385, 387, 263, 373, 380]
right_eye_idxs = [33, 160, 158, 133, 153, 144]
RED = (255, 0, 0)
GREEN = (0, 255, 0)
COLOR = GREEN
alarm_file_path = "wake-up.mp3"

# Prepare alarm sound
mixer.init()
mixer.music.load(alarm_file_path)
mixer.music.play(-1)
mixer.music.pause()
```

### Caching MediaPipe Resources
```python
@st.cache_resource
def get_mediapipe_app(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
):
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )
    return face_mesh
```

### Eye Aspect Ratio (EAR) Calculation
```python
def distance(point_1, point_2):
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist

def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    try:
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        # Calculate distances
        p2_p6 = distance(coords_points[1], coords_points[5])
        p3_p5 = distance(coords_points[2], coords_points[4])
        p1_p4 = distance(coords_points[0], coords_points[3])

        # Compute EAR
        ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points
```

### Real-Time Detection
```python
def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=1.0, thickness=2):
    image = cv2.putText(image, text, origin, font, fntScale, color, thickness)
    return image

try:
    cap = st.session_state["camera_cap"]
    if cap is not None:
        cap.release()

    st.session_state["camera_cap"] = None
except KeyError:
    cap = None
```

---

## Running the Application

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Allow access to the webcam when prompted.

3. The app will monitor your eyes in real-time and trigger an alarm if drowsiness is detected.


---

## Future Improvements
- Add a customizable EAR threshold and alarm delay.
- Incorporate additional facial features for better accuracy.
- Enhance UI for user configuration and feedback.

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.



