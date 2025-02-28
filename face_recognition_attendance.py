import cv2
import dlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load Haar Cascade files for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Load Dlib's face detection and recognition models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Known face encodings and names
known_face_encodings = []
known_face_names = []

# Function to load known faces
def load_known_faces():
    image_paths = [
        "D:/face_recognition/known_faces/1.jpg",
        "D:/face_recognition/known_faces/2.jpg",
        "D:/face_recognition/known_faces/3.jpg",
        "D:/face_recognition/known_faces/4.jpg",
        "D:/face_recognition/known_faces/5.jpg"
    ]
    names = ["KHEM BORANA", "NOU CHANRY", "PONG SOPHEAVY", "KHEM SOVANARA", "SIN SEYHA"]

    for path, name in zip(image_paths, names):
        image = cv2.imread(path)

        if image is None:
            print(f"Error: Unable to load image at {path}. Please check the file path.")
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector(rgb_image)

        if faces:
            shape = sp(rgb_image, faces[0])
            encoding = np.array(facerec.compute_face_descriptor(rgb_image, shape))
            known_face_encodings.append(encoding)
            known_face_names.append(name)
            print(f"Loaded and encoded face for {name}.")
        else:
            print(f"No face found in {path}. Skipping this image.")

# Load known faces before starting video capture
load_known_faces()

# Function to get the current date
def get_current_date():
    return datetime.now().strftime('%Y-%m-%d')

# Store the last recorded date
last_recorded_date = get_current_date()

# Load existing attendance and track registered names
registered_names = set()
try:
    df = pd.read_excel('./attendance.xlsx')
    registered_names.update(df[df['DateTime'].str.startswith(last_recorded_date)]['Name'].tolist())  
except FileNotFoundError:
    print("Attendance file not found, creating a new one.")
    df = pd.DataFrame(columns=['Name', 'DateTime'])

# Function to mark attendance
def mark_attendance(name):
    global last_recorded_date, registered_names
    current_date = get_current_date()

    # Reset registered names if a new day starts
    if current_date != last_recorded_date:
        registered_names.clear()
        last_recorded_date = current_date
        print("New day detected! Resetting attendance list.")

    if name not in registered_names:  # Prevent duplicate recording for the same day
        now = datetime.now()
        dt_string = now.strftime('%Y-%m-%d %H:%M:%S')

        # Append the new record to the existing Excel file without overwriting
        new_entry = pd.DataFrame([[name, dt_string]], columns=['Name', 'DateTime'])
        df_new = pd.concat([df, new_entry], ignore_index=True)
        df_new.to_excel('./attendance.xlsx', index=False)

        registered_names.add(name)  # Add to registered list
        print(f"Attendance marked for {name} at {dt_string}")
    else:
        print(f"{name} has already been recorded today.")

# Main program
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert frame to RGB (Dlib uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = detector(rgb_frame)

    for face in faces:
        shape = sp(rgb_frame, face)
        face_encoding = np.array(facerec.compute_face_descriptor(rgb_frame, shape))

        # Check if known faces list is not empty
        if known_face_encodings:
            # Compare with known faces
            matches = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
            min_match_index = np.argmin(matches)

            if matches[min_match_index] < 0.6:  # Threshold for matching
                name = known_face_names[min_match_index]
                mark_attendance(name)
            else:
                name = "Unknown"
        else:
            print("No known faces loaded for comparison.")
            name = "Unknown"

        # Draw a rectangle and label around the face
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the video frame
    cv2.imshow('Attendance System', frame)

    # Exit on 'q' key press  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
