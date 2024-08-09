from tkinter import Tk, Label, Button, Frame, StringVar, OptionMenu
import cv2
from PIL import Image, ImageTk
from djitellopy import tello
import threading
import face_recognition
import os
import numpy as np

from flight_commands import start_flying, stop_flying

# Load known faces and their encodings from a directory
def load_known_faces(directory):
    known_face_encodings = []
    known_face_names = []

    for file_name in os.listdir(directory):
        if file_name.endswith((".jpg", ".png")):
            image_path = os.path.join(directory, file_name)
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(os.path.splitext(file_name)[0])

    return known_face_encodings, known_face_names

# Function to calculate confidence from face distance
def calculate_confidence(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        linear_val = (1.0 - face_distance) / (0.1 - face_match_threshold)
        return max(0.0, min(1.0, linear_val)) * 100
    else:
        linear_val = (1.0 - face_distance) / (face_match_threshold - 0.1)
        return max(0.0, min(1.0, linear_val)) * 100

# Function to perform face recognition on a single frame
def recognize_faces(frame, known_face_encodings, known_face_names):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    face_confidences = []
    face_distances = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        confidence = 0.0
        distance = 0.0  

        face_distances_current = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances_current)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            confidence = calculate_confidence(face_distances_current[best_match_index])
            distance = face_distances_current[best_match_index]  

        face_names.append(name)
        face_confidences.append(confidence)
        face_distances.append(distance)

    return face_locations, face_names, face_confidences, face_distances


class DroneController:
    def __init__(self):
        self.root = Tk()
        self.root.title("Drone Keyboard Controller - Tkinter")
        self.root.minsize(800, 600)

        self.input_frame = Frame(self.root)

        self.drone = tello.Tello()
        self.drone.connect()
        self.drone.streamon()
        self.frame = self.drone.get_frame_read()

        self.drone.speed = 50

        self.cap_lbl = Label(self.root)
        self.button_frame = Frame(self.root)

        self.takeoff_land_button = Button(self.button_frame, text="Takeoff/Land", command=self.takeoff_land)
        self.stop_following_button = Button(self.button_frame, text="Stop Following", command=self.stop_following)

        
        faces_dir = "faces"
        self.known_face_encodings, self.known_face_names = load_known_faces(faces_dir)
        self.dropdown_var = StringVar(self.root)
        self.dropdown_var.set("Disable")
        self.dropdown_menu = OptionMenu(self.button_frame, self.dropdown_var, "Disable", *self.known_face_names)

        
        self.FOCAL_LENGTH = 800  
        self.KNOWN_FACE_WIDTH = 16  

    def takeoff_land(self):
        if self.drone.is_flying:
            threading.Thread(target=self.drone.land).start()
        else:
            threading.Thread(target=self.drone.takeoff).start()

    def stop_following(self):
        print("Stop Following Button clicked!")

    def calculate_distance(self, face_width_pixels):
        """Calculate distance based on face width in pixels."""
        if face_width_pixels == 0:
            return 0.0
        return (self.KNOWN_FACE_WIDTH * self.FOCAL_LENGTH) / face_width_pixels

    def run_app(self):
        try:
            self.input_frame.pack()
            self.input_frame.focus_set()

            self.input_frame.bind('<KeyPress-w>', lambda event: start_flying(event, 'upward', self.drone, self.drone.speed))
            self.input_frame.bind('<KeyRelease-w>', lambda event: stop_flying(event, self.drone))

            self.input_frame.bind('<KeyPress-a>', lambda event: start_flying(event, 'yaw_left', self.drone, self.drone.speed))
            self.input_frame.bind('<KeyRelease-a>', lambda event: stop_flying(event, self.drone))

            self.input_frame.bind('<KeyPress-s>', lambda event: start_flying(event, 'downward', self.drone, self.drone.speed))
            self.input_frame.bind('<KeyRelease-s>', lambda event: stop_flying(event, self.drone))

            self.input_frame.bind('<KeyPress-d>', lambda event: start_flying(event, 'yaw_right', self.drone, self.drone.speed))
            self.input_frame.bind('<KeyRelease-d>', lambda event: stop_flying(event, self.drone))

            self.input_frame.bind('<KeyPress-Up>', lambda event: start_flying(event, 'forward', self.drone, self.drone.speed))
            self.input_frame.bind('<KeyRelease-Up>', lambda event: stop_flying(event, self.drone))

            self.input_frame.bind('<KeyPress-Down>', lambda event: start_flying(event, 'backward', self.drone, self.drone.speed))
            self.input_frame.bind('<KeyRelease-Down>', lambda event: stop_flying(event, self.drone))

            self.input_frame.bind('<KeyPress-Left>', lambda event: start_flying(event, 'left', self.drone, self.drone.speed))
            self.input_frame.bind('<KeyRelease-Left>', lambda event: stop_flying(event, self.drone))

            self.input_frame.bind('<KeyPress-Right>', lambda event: start_flying(event, 'right', self.drone, self.drone.speed))
            self.input_frame.bind('<KeyRelease-Right>', lambda event: stop_flying(event, self.drone))

            self.cap_lbl.pack(anchor="center", pady=15)
            self.takeoff_land_button.pack(side='left', padx=10)
            self.stop_following_button.pack(side='left', padx=10)
            self.dropdown_menu.pack(side='left', padx=10)
            self.button_frame.pack(anchor="center", pady=10)

            self.video_stream()

            self.root.mainloop()

        except Exception as e:
            print(f"Error running the application: {e}")
        finally:
            self.cleanup()

    def video_stream(self):
        h, w = 480, 720

        
        frame = self.frame.frame

        if frame is not None:
            frame = cv2.resize(frame, (w, h))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            selected_name = self.face_detection_var.get()
            if selected_name != "Disable":
                if selected_name in self.known_face_names:
                    idx = self.known_face_names.index(selected_name)
                    target_encodings = [self.known_face_encodings[idx]]
                    target_names = [self.known_face_names[idx]]
                else:
                    target_encodings = self.known_face_encodings
                    target_names = self.known_face_names

                face_locations, face_names, face_confidences, face_distances = recognize_faces(frame, target_encodings, target_names)
                for (top, right, bottom, left), name, confidence, distance in zip(face_locations, face_names, face_confidences, face_distances):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    if name != "Unknown":
                        face_width_pixels = right - left
                        distance = self.calculate_distance(face_width_pixels)

                        
                        if distance > 200:
                            self.drone.move_forward(20)  
                        elif distance < 190:
                            self.drone.move_back(20)  

                        
                        face_center_x = (left + right) // 2
                        frame_center_x = w // 2
                        if face_center_x < frame_center_x - 50:
                            self.drone.move_left(20)  
                        elif face_center_x > frame_center_x + 50:
                            self.drone.move_right(20)  

                        
                        face_center_y = (top + bottom) // 2
                        frame_center_y = h // 2
                        if face_center_y < frame_center_y - 50:
                            self.drone.move_up(20)  
                        elif face_center_y > frame_center_y + 50:
                            self.drone.move_down(20)  

                        # Display the bounding box and label
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        label = f"{name} ({confidence:.2f}%) Distance: {distance:.2f} cm"
                        cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
            imgtk = ImageTk.PhotoImage(image=img)
            self.cap_lbl.imgtk = imgtk
            self.cap_lbl.configure(image=imgtk)

        self.cap_lbl.after(10, self.video_stream)


    def cleanup(self):
        try:
            print("Cleaning up resources...")
            self.drone.end()
            self.root.quit()
            exit()
        except Exception as e:
            print(f"Error performing cleanup: {e}")

if __name__ == "__main__":
    gui = DroneController()
    gui.run_app()
