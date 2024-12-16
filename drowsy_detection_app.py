import cv2
import time
import mediapipe as mp
import streamlit as st
from pygame import mixer
from mediapipe.python.solutions.drawing_utils import(
    _normalized_to_pixel_coordinates as denormalize_coordinates,
)

print("hello imports")

# CONSTANTS

# left and right eye choosen landmark
left_eye_idxs = [362, 385, 387, 263, 373, 380]
right_eye_idxs = [33, 160, 158, 133, 153, 144]

all_idxs = left_eye_idxs + right_eye_idxs

RED = (255, 0, 0)
GREEN = (0, 255, 0)

COLOR = GREEN

alarm_file_path = f"wake-up.mp3"

# PREPARE ALARM SOUND
mixer.init()
mixer.music.load(alarm_file_path)
mixer.music.play(-1)
mixer.music.pause()

print('ok')

@st.cache(allow_output_mutation=True)
def get_mediapipe_app(
    max_num_faces = 1,
    refine_landmarks=True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
):
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces = max_num_faces,
        refine_landmarks = refine_landmarks,
        min_detection_confidence = min_detection_confidence,
        min_tracking_confidence = min_tracking_confidence
    )

    return face_mesh

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

        
        p2_p6 = distance(coords_points[1], coords_points[5])
        p3_p5 = distance(coords_points[2], coords_points[4])
        p1_p4 = distance(coords_points[0], coords_points[3])

        # compute eye espect ratio
        ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)

    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points


def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale = 1.0, thickness = 2):
    image = cv2.putText(image, text, origin, font, fntScale, color, thickness)
    return image

print("ok till now")


try:
    cap = st.session_state["camera_cap"]
    if cap is not None:
        cap.release()

    st.session_state["camera_cap"] = None
except KeyError:
    cap = None

