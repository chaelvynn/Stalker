import os
import cv2
import numpy as np
import face_recognition
from tkinter import Tk, Label, Button, Frame, StringVar, OptionMenu
from PIL import Image, ImageTk

class FaceRecognition:
    def __init__(self, faces_dir):
        self.faces_dir = faces_dir
        self.known_face_encodings, self.known_face_names = self.load_known_faces()
        self.locked_face_name = None
        self.FOCAL_LENGTH = 800
        self.KNOWN_FACE_WIDTH = 16  # Rata-rata lebar wajah manusia dalam cm

    def load_known_faces(self):
        known_face_encodings = []
        known_face_names = []
        for file_name in os.listdir(self.faces_dir):
            if file_name.endswith((".jpg", ".png")):
                image_path = os.path.join(self.faces_dir, file_name)
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(os.path.splitext(file_name)[0])
        return known_face_encodings, known_face_names

    @staticmethod
    def calculate_confidence(face_distance, face_match_threshold=0.7):
        if face_distance > face_match_threshold:
            linear_val = (1.0 - face_distance) / (0.1 - face_match_threshold)
            return max(0.0, min(1.0, linear_val)) * 100
        else:
            linear_val = (1.0 - face_distance) / (face_match_threshold - 0.1)
            return max(0.0, min(1.0, linear_val)) * 100

    def calculate_distance(self, face_width_pixels):
        if face_width_pixels == 0:
            return 0.0
        return (self.KNOWN_FACE_WIDTH * self.FOCAL_LENGTH) / face_width_pixels

    def recognize_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        face_confidences = []
        face_distances = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            confidence = 0.0
            distance = 0.0

            face_distances_current = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances_current)
            if matches[best_match_index]:
                confidence = self.calculate_confidence(face_distances_current[best_match_index])
                if confidence > 95:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)
            face_confidences.append(confidence)
            face_distances.append(distance)

        return face_locations, face_names, face_confidences, face_distances

    def lock_face(self, name):
        if name in self.known_face_names:
            self.locked_face_name = name
            print(f"Locked face: {self.locked_face_name}")
        else:
            self.locked_face_name = None
            print("Face recognition is enable")

    def display_results(self, frame, face_locations, face_names, face_confidences, face_distances):
        for (top, right, bottom, left), name, confidence, distance in zip(face_locations, face_names, face_confidences, face_distances):
            if self.locked_face_name is None or name == self.locked_face_name:
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                face_width_pixels = right - left
                distance = self.calculate_distance(face_width_pixels)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                label = f"{name} ({confidence:.2f}%) Distance: {distance:.2f} cm"
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                fontScale = 0.5 
                cv2.putText(frame, label, (left + 6, bottom - 6), font, fontScale, (255, 255, 255), 1)

        return frame

class WebcamController:
    def __init__(self, face_recognition_system):
        self.root = Tk()
        self.root.title("Webcam Controller - Tkinter")
        self.root.minsize(800, 600)

        self.face_recognition_system = face_recognition_system
        self.cap = cv2.VideoCapture(0)

        self.input_frame = Frame(self.root)
        self.cap_lbl = Label(self.root)
        self.button_frame = Frame(self.root)

        self.demo_button = Button(self.button_frame, text="Demo Button", command=self.demo_function)
        self.demo_button.pack(side='left', padx=10)

        self.face_detection_var = StringVar(self.root)
        self.face_detection_var.set("Disable")
        self.face_detection_menu = OptionMenu(self.button_frame, self.face_detection_var, "Disable", "Enable All", *self.face_recognition_system.known_face_names, command=self.on_dropdown_select)
        self.face_detection_menu.pack(side='left')

        self.button_frame.pack(anchor="center", pady=10)

    def demo_function(self):
        print("Demo Button clicked!")

    def on_dropdown_select(self, selection):
        print(f"{selection} is clicked")
        if selection == "Disable":
            self.face_recognition_system.lock_face(None)
        elif selection == "Enable All":
            self.face_recognition_system.lock_face(None)
        else:
            self.face_recognition_system.lock_face(selection)

    def run_app(self):
        try:
            self.input_frame.pack()
            self.input_frame.focus_set()
            self.cap_lbl.pack(anchor="center", pady=15)
            self.video_stream()
            self.button_frame.pack(anchor="s", pady=10) 
            self.root.mainloop()
        except Exception as e:
            print(f"Error running the application: {e}")
        finally:
            self.cleanup()

    def video_stream(self):
        h, w = 480, 720
        ret, frame = self.cap.read()

        if ret:
            frame = cv2.resize(frame, (w, h))
            face_locations, face_names, face_confidences, face_distances = [], [], [], []

            if self.face_detection_var.get() != "Disable":
                face_locations, face_names, face_confidences, face_distances = self.face_recognition_system.recognize_faces(frame)

            frame = self.face_recognition_system.display_results(frame, face_locations, face_names, face_confidences, face_distances)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
            imgtk = ImageTk.PhotoImage(image=img)
            self.cap_lbl.imgtk = imgtk
            self.cap_lbl.configure(image=imgtk)

        self.cap_lbl.after(10, self.video_stream)

    def cleanup(self):
        try:
            print("Cleaning up resources...")
            self.cap.release()
            self.root.quit()
        except Exception as e:
            print(f"Error performing cleanup: {e}")

if __name__ == "__main__":
    faces_dir = "faces"
    face_recognition_system = FaceRecognition(faces_dir)
    gui = WebcamController(face_recognition_system)
    gui.run_app()
