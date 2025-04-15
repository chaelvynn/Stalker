import os
import cv2
import numpy as np
import face_recognition
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Directory containing known faces
FACE_DIR = "orb_track/faces"

# Load known faces
def load_known_faces(directory):
    known_faces = []
    known_names = []

    for file_name in os.listdir(directory):
        if file_name.endswith((".jpg", ".png")):
            image_path = os.path.join(directory, file_name)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_faces.append(encoding[0])
                known_names.append(file_name.split(".")[0])
    
    return known_faces, known_names

# ORB Tracker
class ORBTracker:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_kp = None
        self.prev_desc = None
        self.prev_box = None

    def track(self, frame, face_box):
        (x, y, w, h) = face_box
        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        kp, desc = self.orb.detectAndCompute(gray, None)
        if desc is None:
            return face_box  # If no features detected, keep the previous box

        if self.prev_desc is not None:
            matches = self.bf.match(self.prev_desc, desc)
            matches = sorted(matches, key=lambda x: x.distance)
            
            if len(matches) > 10:
                dx = np.mean([kp[m.trainIdx].pt[0] - self.prev_kp[m.queryIdx].pt[0] for m in matches])
                dy = np.mean([kp[m.trainIdx].pt[1] - self.prev_kp[m.queryIdx].pt[1] for m in matches])
                x, y = int(x + dx), int(y + dy)

        self.prev_kp = kp
        self.prev_desc = desc
        self.prev_box = (x, y, w, h)

        return (x, y, w, h)

# Create UI Root **SEBELUM inisialisasi variable Tk**
root = tk.Tk()
root.title("Face Tracking ORB")
root.geometry("400x200")

# Dropdown variable (Setelah Tk dibuat)
selected_name = tk.StringVar()

# Load known faces
known_face_encodings, known_face_names = load_known_faces(FACE_DIR)
tracker = ORBTracker()

# Global variables
cap = None
running = False

# Function to start tracking
def start_tracking():
    global cap, running

    if not selected_name.get():
        return  # Do nothing if no name is selected

    running = True
    cap = cv2.VideoCapture(0)

    def process_video():
        global running

        while running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

                if True in matches:
                    match_idx = matches.index(True)
                    name = known_face_names[match_idx]

                    # Hanya track jika sesuai pilihan dropdown
                    if name != selected_name.get():
                        continue

                    # Konversi koordinat ke ukuran asli
                    top, right, bottom, left = [i * 2 for i in face_location]
                    face_box = (left, top, right - left, bottom - top)

                    # Track face menggunakan ORB
                    face_box = tracker.track(frame, face_box)

                    # Gambar bounding box
                    x, y, w, h = face_box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("Face Tracking ORB", frame)

            # End program when pressing 'q' or 'Esc'
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 27 = Esc key
                stop_tracking()
                break

    threading.Thread(target=process_video, daemon=True).start()

# Function to stop tracking
def stop_tracking():
    global cap, running

    running = False
    if cap is not None:
        cap.release()
        cap = None

    cv2.destroyAllWindows()

# Dropdown untuk memilih wajah
tk.Label(root, text="Pilih Wajah untuk Tracking:", font=("Arial", 12)).pack(pady=5)
dropdown = ttk.Combobox(root, textvariable=selected_name, values=known_face_names, state="readonly", font=("Arial", 12))
dropdown.pack(pady=5)
if known_face_names:
    dropdown.current(0)

# Tombol Start Tracking
start_button = tk.Button(root, text="Start Tracking", font=("Arial", 12), bg="green", fg="white", command=start_tracking)
start_button.pack(pady=5)

# Tombol Stop Tracking
stop_button = tk.Button(root, text="Stop Tracking", font=("Arial", 12), bg="red", fg="white", command=stop_tracking)
stop_button.pack(pady=5)

# Start UI loop
root.mainloop()
