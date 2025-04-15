import os
import cv2
import numpy as np
import face_recognition
from tkinter import Tk, Label, Button, Frame, StringVar, OptionMenu
from PIL import Image, ImageTk
from djitellopy import Tello
import threading

class FaceRecognition:
    def __init__(self, faces_dir):
        self.faces_dir = faces_dir
        self.known_face_encodings, self.known_face_names = self.load_known_faces()
        self.locked_face_name = None
        self.FOCAL_LENGTH = 800
        self.KNOWN_FACE_WIDTH = 16 

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
                if confidence > 80:
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
            print("Face recognition is enabled")

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


class DroneController:
    def __init__(self, face_recognition_system):
        self.root = Tk()
        self.root.title("Drone Controller - Tkinter")
        self.root.minsize(800, 600)

        self.face_recognition_system = face_recognition_system

        # Initialize the Tello drone
        self.drone = Tello()
        self.drone.connect()
        self.drone.streamon()

        self.input_frame = Frame(self.root)
        self.cap_lbl = Label(self.root)
        self.button_frame = Frame(self.root)

        self.takeoff_land_button = Button(self.button_frame, text="Takeoff/Land", command=self.takeoff_land)
        self.takeoff_land_button.pack(side='left', padx=10)

        self.stop_following_button = Button(self.button_frame, text="Stop Following", command=self.stop_following)
        self.stop_following_button.pack(side='left', padx=10)

        self.face_detection_var = StringVar(self.root)
        self.face_detection_var.set("Disable")
        self.face_detection_menu = OptionMenu(self.button_frame, self.face_detection_var, "Disable", "Enable All", *self.face_recognition_system.known_face_names, command=self.on_dropdown_select)
        self.face_detection_menu.pack(side='left')

        self.button_frame.pack(anchor="center", pady=10)

    def takeoff_land(self):
        if self.drone.is_flying:
            threading.Thread(target=self.drone.land).start()
        else:
            threading.Thread(target=self.drone.takeoff).start()

    def stop_following(self):
        self.face_recognition_system.lock_face(None)
        print("Stopped Following")

    def on_dropdown_select(self, selection):
        print(f"{selection} selected")
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
            threading.Thread(self.video_stream()).start() #change this
            self.button_frame.pack(anchor="s", pady=10)
            self.root.mainloop()
        except Exception as e:
            print(f"Error running the application: {e}")
        finally:
            self.cleanup()

    def video_stream(self):
        h, w = 480, 720

        # Capture frame from Tello drone
        frame = self.drone.get_frame_read(max_queue_len=0).frame
        frame = cv2.resize(frame, (w, h))

        face_locations, face_names, face_confidences, face_distances = [], [], [], []

        if self.face_detection_var.get() != "Disable":
            face_locations, face_names, face_confidences, face_distances = self.face_recognition_system.recognize_faces(frame)

            # Drone following logic
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                if name == self.face_recognition_system.locked_face_name:
                    self.follow_person(top, right, bottom, left)

        frame = self.face_recognition_system.display_results(frame, face_locations, face_names, face_confidences, face_distances)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
        imgtk = ImageTk.PhotoImage(image=img)
        self.cap_lbl.imgtk = imgtk
        self.cap_lbl.configure(image=imgtk)

        self.cap_lbl.after(10, self.video_stream)

    def follow_person(self, top, right, bottom, left):
        frame_center_x = 360
        frame_center_y = 240
        face_center_x = (left + right) // 2
        face_center_y = (top + bottom) // 2

        error_x = face_center_x - frame_center_x
        error_y = face_center_y - frame_center_y

        if abs(error_x) > 20:
            if error_x > 0:
                self.drone.send_rc_control(0, 0, 0, 20)  # Turn right
            else:
                self.drone.send_rc_control(0, 0, 0, -20)  # Turn left

        if abs(error_y) > 20:
            if error_y > 0:
                self.drone.send_rc_control(0, -20, 0, 0)  # Move down
            else:
                self.drone.send_rc_control(0, 20, 0, 0)  # Move up

    def cleanup(self):
        try:
            print("Cleaning up resources...")
            self.drone.streamoff()
            self.root.quit()
        except Exception as e:
            print(f"Error performing cleanup: {e}")

if __name__ == "__main__":
    faces_dir = "faces"
    face_recognition_system = FaceRecognition(faces_dir)
    drone_controller = DroneController(face_recognition_system)
    drone_controller.run_app()
