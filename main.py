import cv2
import dlib
import numpy as np
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

# Load Dlib's face detection and recognition models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Load known faces
known_face_encodings = []
known_face_names = []

def load_known_faces():
    image_paths = [
        "D:/face_recognition/known_faces/1.jpg",
        "D:/face_recognition/known_faces/2.jpg"
    ]
    names = ["KHEM BORANA", "NOU CHANRY"]

    for path, name in zip(image_paths, names):
        image = cv2.imread(path)
        if image is None:
            print(f"Error loading {path}")
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector(rgb_image)
        if faces:
            shape = sp(rgb_image, faces[0])
            encoding = np.array(facerec.compute_face_descriptor(rgb_image, shape))
            known_face_encodings.append(encoding)
            known_face_names.append(name)
            print(f"Loaded {name}")
        else:
            print(f"No face found in {path}")

load_known_faces()

# Function to get current date
def get_current_date():
    return datetime.now().strftime('%Y-%m-%d')

# Load attendance
registered_names = set()
try:
    df = pd.read_excel('./attendance.xlsx')
    registered_names.update(df[df['DateTime'].str.startswith(get_current_date())]['Name'].tolist())  
except FileNotFoundError:
    df = pd.DataFrame(columns=['Name', 'DateTime'])

# Function to mark attendance
def mark_attendance(name):
    current_date = get_current_date()
    if name not in registered_names:
        now = datetime.now()
        dt_string = now.strftime('%Y-%m-%d %H:%M:%S')

        # Append new record
        new_entry = pd.DataFrame([[name, dt_string]], columns=['Name', 'DateTime'])
        df_new = pd.concat([df, new_entry], ignore_index=True)
        df_new.to_excel('./attendance.xlsx', index=False)

        registered_names.add(name)
        status_label.config(text=f"Attendance marked: {name}")
    else:
        status_label.config(text=f"{name} already recorded today")

# Function to scan attendance
def scan_attendance():
    cap = cv2.VideoCapture(0)
    status_label.config(text="Scanning...")
    root.update_idletasks()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector(rgb_frame)

        for face in faces:
            shape = sp(rgb_frame, face)
            face_encoding = np.array(facerec.compute_face_descriptor(rgb_frame, shape))

            if known_face_encodings:
                matches = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
                min_match_index = np.argmin(matches)

                if matches[min_match_index] < 0.6:
                    name = known_face_names[min_match_index]
                    mark_attendance(name)
                else:
                    name = "Unknown"

            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Attendance Scanner', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Create GUI
root = tk.Tk()
root.title("Smart Attendance System")
root.geometry("400x300")

tk.Label(root, text="Face Recognition Attendance", font=("Arial", 14)).pack(pady=10)
status_label = tk.Label(root, text="Press 'Scan Attendance' to start", font=("Arial", 12))
status_label.pack(pady=10)

scan_button = tk.Button(root, text="Scan Attendance", font=("Arial", 12), command=scan_attendance, bg="green", fg="white")
scan_button.pack(pady=20)

exit_button = tk.Button(root, text="Exit", font=("Arial", 12), command=root.quit, bg="red", fg="white")
exit_button.pack(pady=10)

root.mainloop()
