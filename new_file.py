import cv2
import dlib
import numpy as np
import pandas as pd
from datetime import datetime

# Load the Haar Cascade files for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Capture video from the webcam or use an image
cap = cv2.VideoCapture(0)  # Set 0 for webcam; you can replace '0' with a video file path

# Load pre-trained face detector and recognizer
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Sample known faces encoding and names (replace these with actual data)
known_face_encodings = []  # This should be populated with encodings
known_face_names = []  # Corresponding names of the faces


def load_known_faces():
    image_paths = ["D:/face_recognition/known_faces/1.jpg", "D:/face_recognition/known_faces/2.jpg", "D:/face_recognition/known_faces/3.jpg", "D:/face_recognition/known_faces/4.jpg", "D:/face_recognition/known_faces/5.jpg"]
    names = ["KHEM BORANA", "NOU CHANRY", "PONG SOPHEAVY", "KHEM SOVANARA", "SIN SEYHA" ]

    for path, name in zip(image_paths, names):
        image = cv2.imread(path)

        # Check if the image was loaded successfully
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


# Call this function to load the known faces before starting the video capture
load_known_faces()

# Initialize attendance list
attendance = []


# Function to mark attendance
def mark_attendance(name):
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
    attendance.append([name, dt_string])
    print(f"ATTENDANCE LIST {attendance}")
    print(f"Attendance marked for {name} at {dt_string}")
    # Save attendance to Excel
    if attendance:
        df = pd.DataFrame(attendance, columns=['Name', 'DateTime'])

# match name
        print(f" match name {name}")

# satement register attandant 
        df.to_excel('./attendance.xlsx', index=False)


        # df.to_excel('./attendance.xlsx', index=False)
        print("Attendance saved to attendance.xlsx")
    else:
        print("No attendance recorded.")



# Main program starting point
# Capture video from webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert frame to RGB (dlib uses RGB)
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

    # Display the video frame with boxes
    cv2.imshow('Attendance System', frame)

    # Break loop on 'q' key press  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
video_capture.release()
cv2.destroyAllWindows()


