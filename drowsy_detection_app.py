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

st.title("Drowsiness Detection!")

# Set button style.
m = st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #00ff00;
        color:#000000;
        font-weight: bold;
        font-size: 22px;
        padding: 4px 35px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize face mesh model.
face_mesh = get_mediapipe_app()

# Click if you want to run the application.
run_btn_container = st.empty()
run = run_btn_container.button("START")

# Lowest valid value of Eye Aspect Ratio. Ideal values [0.15, 0.2].
EAR_THRESH = st.sidebar.slider("Eye Aspect Ratio threshold:", 0.0, 0.4, 0.18, 0.01)

# The amount to time (in seconds) to wait before sounding the alarm.
WAIT_TIME = st.sidebar.slider("Seconds to wait before sounding alarm:", 0.0, 5.0, 1.0, 0.25)

# Track the amount of seconds it has been with Eye aspect ratio < threshold
DROWSY_TIME = 0.0

# Collect frames to display
FRAME_WINDOW = st.image([])

if run:
    # Start Webcam
    cap = cv2.VideoCapture(0)
    cam_w = int(cap.get(3))  # Camera frame width
    cam_h = int(cap.get(4))  # Camera frame height

    st.session_state["camera_cap"] = cap  # save current session. Useful for releasing the camera resouce.

    # Calculate coordinates for plotting text.
    EAR_txt_pos = (int(cam_w // 2 * 1.3), 30)
    ALM_txt_pos = (10, cam_h - 50)
    DROWSY_TIME_txt_pos = (10, cam_h - 100)

    # Change button style.
    m = st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            background-color: #dd0000;
            color:#ffffff;
            font-weight: bold;
            font-size: 22px;
            padding: 4px 40px;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
    # Change button text.
    run_btn_container.empty()
    run_btn_container.button("STOP")

    start_time = time.perf_counter()
    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # To improve performance, optionally mark the frame as not writeable to pass by reference.
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform Inference.
        results = face_mesh.process(frame)

        # If detections are available.
        if results.multi_face_landmarks:

            # As we have set max_num_faces=1, we can direcly access
            # the landmarks from the list instead of iterating.
            landmarks = results.multi_face_landmarks[0].landmark

            left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, cam_w, cam_h)
            right_ear, right_lm_coordinates = get_ear(landmarks, right_eye_idxs, cam_w, cam_h)

            EAR = (left_ear + right_ear) / 2.0

            # Draw landmarks
            for lm_coordinates in [left_lm_coordinates, right_lm_coordinates]:
                if lm_coordinates:
                    for coord in lm_coordinates:
                        cv2.circle(frame, coord, 2, COLOR, -1)

            # Flip the frame horizontally for a selfie-view display.
            frame = cv2.flip(frame, 1)

            # Check whether current EAR is below the valid threshold value
            if EAR < EAR_THRESH:
                COLOR = RED

                # Increase DROWSY_TIME to track the time period with EAR less than threshold
                # and reset the start_time for the next iteration.
                end_time = time.perf_counter()
                DROWSY_TIME += end_time - start_time
                start_time = end_time

                # If the counter is above the frame patience limit, sound the alarm
                if DROWSY_TIME >= WAIT_TIME:
                    plot_text(frame, "WAKE UP! WAKE UP", ALM_txt_pos, COLOR)
                    mixer.music.unpause()

            else:  # If not;
                mixer.music.pause()
                DROWSY_TIME = 0.0
                COLOR = GREEN
                start_time = time.perf_counter()

            plot_text(frame, f"EAR: {round(EAR, 2)}", EAR_txt_pos, COLOR)
            plot_text(frame, f"DROWSY: {round(DROWSY_TIME, 3)} Secs", DROWSY_TIME_txt_pos, COLOR)

        else:  # Just Flip the frame horizontally for a selfie-view display.
            start_time = time.perf_counter()
            frame = cv2.flip(frame, 1)

        FRAME_WINDOW.image(frame)
