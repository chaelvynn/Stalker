import os
import cv2
import numpy as np
import face_recognition
from tkinter import Tk, Label, Button, Frame, StringVar, OptionMenu
from PIL import Image, ImageTk
import threading as thread
import time





# Function to load known faces and their encodings from a directory
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
        distance = 0.0  # Initialize distance

        face_distances_current = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances_current)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            confidence = calculate_confidence(face_distances_current[best_match_index])
            distance = face_distances_current[best_match_index]  # Set distance

        face_names.append(name)
        face_confidences.append(confidence)
        face_distances.append(distance)

    return face_locations, face_names, face_confidences, face_distances

# Class for controlling the webcam video stream via Tkinter GUI
class WebcamController:
    def __init__(self):
        self.root = Tk()
        self.root.title("Webcam Controller - Tkinter")
        self.root.minsize(800, 600)


        self.prev_tick = cv2.getTickCount()
        self.curr_fps = 0.0


        # ===== Evaluation Metrics =====
        self.evaluation_active = False

        self.fps_log = []
        self.total_frames = 0
        self.success_frames = 0

        self.point_motion = []
        self.prev_point = None




        self.input_frame = Frame(self.root)
        self.cap = cv2.VideoCapture(0)

        faces_dir = "faces"
        self.known_face_encodings, self.known_face_names = load_known_faces(faces_dir)

        self.cap_lbl = Label(self.root)
        self.button_frame = Frame(self.root)

        self.demo_button = Button(self.button_frame, text="Demo Button", command=self.demo_function)

        self.face_detection_var = StringVar(self.root)
        self.face_detection_var.set("Disable")
        self.face_detection_menu = OptionMenu(self.button_frame, self.face_detection_var, "Disable", *self.known_face_names)

        # Camera and face parameters
        self.FOCAL_LENGTH = 800  # Adjust this value based on your camera
        self.KNOWN_FACE_WIDTH = 16  # Average width of a human face in cm

        self.prev_frame_gray = None
        self.tracked_points = None
        self.current_name = "Disable"
        
        
        self.lk_params = dict(
                    winSize=(15, 15),  # Increased window size for better tracking
                    maxLevel=3,  # Number of pyramid levels
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )

    def start_evaluation(self):
        self.fps_log.clear()
        self.total_frames = 0
        self.success_frames = 0
        self.point_motion.clear()
        self.prev_point = None
        self.evaluation_active = True
        print("Evaluation started.")
        self.eval_start_time = time.time()



    def demo_function(self):
        print("Button clicked!")

    def calculate_distance(self, face_width_pixels):
        if face_width_pixels == 0:
            return 0.0
        return (self.KNOWN_FACE_WIDTH * self.FOCAL_LENGTH) / face_width_pixels

    def run_app(self):
        try:
            self.input_frame.pack()
            self.input_frame.focus_set()
            self.cap_lbl.pack(anchor="center", pady=15)
            self.demo_button.pack(side='left', padx=10)
            self.face_detection_menu.pack(side='left')
            self.button_frame.pack(anchor="center", pady=10)
            self.video_stream()
            self.root.mainloop()
        except Exception as e:
            print(f"Error running the application: {e}")
        finally:
            self.cleanup()

        
    def video_stream(self):
        h, w = 480, 720
        # Start a thread to read a frame from the video capture
        ret, frame = None, None

         # Hitung FPS
        curr_tick = cv2.getTickCount()
        elapsed = (curr_tick - self.prev_tick) / cv2.getTickFrequency()
        self.curr_fps = 1.0 / elapsed if elapsed > 0 else 0
        self.prev_tick = curr_tick

        if self.evaluation_active:
            self.fps_log.append(self.curr_fps)


        if self.prev_frame_gray is None:
            ret, frame = self.cap.read()
            self.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        def read_frame():
            nonlocal ret, frame
            ret, frame = self.cap.read()
        
        t1 = thread.Thread(target=read_frame)
        t1.start()
        t1.join()

        if ret:
            if self.evaluation_active:
                self.total_frames += 1
                if self.tracked_points is not None:
                    self.success_frames += 1
            frame = cv2.resize(frame, (w, h))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cur_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Tampilkan FPS di frame
            cv2.putText(frame, f"FPS: {self.curr_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if self.tracked_points is None:
                selected_name = self.face_detection_var.get()

                if selected_name != "Disable" and not self.evaluation_active:
                    self.start_evaluation()

                self.current_name = selected_name



                if selected_name != "Disable":
                    if selected_name in self.known_face_names:
                        idx = self.known_face_names.index(selected_name)
                        target_encodings = [self.known_face_encodings[idx]]
                        target_names = [self.known_face_names[idx]]
                    else:
                        target_encodings = self.known_face_encodings
                        target_names = self.known_face_names

                    # Start a thread to perform face recognition
                    face_locations, face_names, face_confidences, face_distances = [], [], [], []
                    def recognize_faces_thread():
                        nonlocal face_locations, face_names, face_confidences, face_distances
                        face_locations, face_names, face_confidences, face_distances = recognize_faces(frame, target_encodings, target_names)
                    
                    t2 = thread.Thread(target=recognize_faces_thread)
                    t2.start()
                    t2.join()

                    for (top, right, bottom, left), name, confidence, distance in zip(face_locations, face_names, face_confidences, face_distances):
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4

                        # Only draw bounding box if the face is known
                        if name != "Unknown":
                            face_width_pixels = right - left
                            distance = self.calculate_distance(face_width_pixels)  # Calculate distance based on face width in pixels

                            self.tracked_points = np.array([[[(left + right) // 2, (top + bottom) // 2]]], dtype=np.float32)
                            x, y = self.tracked_points[0].ravel()
                            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

                            # Determine movement based on distance and position
                            movement = []
                            if distance > 200:
                                movement.append("Move forward")
                            elif distance < 190:
                                movement.append("Move backward")

                            # Check for horizontal alignment
                            face_center_x = (left + right) // 2
                            frame_center_x = w // 2
                            if face_center_x < frame_center_x - 50:
                                movement.append("Move left")
                            elif face_center_x > frame_center_x + 50:
                                movement.append("Move right")

                            # Check for vertical alignment
                            face_center_y = (top + bottom) // 2
                            frame_center_y = h // 2
                            if face_center_y < frame_center_y - 50:
                                movement.append("Move upward")
                            elif face_center_y > frame_center_y + 50:
                                movement.append("Move downward")

                            if movement:
                                print(f"Actions: {', '.join(movement)}")
                            else:
                                print("Target is centered and at the correct distance")

                        # Display the bounding box and label
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        label = f"{name} ({confidence:.2f}%) Distance: {distance:.2f} cm"
                        cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            
            elif self.tracked_points is not None:
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_frame_gray, cur_gray_frame, self.tracked_points, None, **self.lk_params
            )

                # Ensure valid points are returned
                if next_points is not None and len(next_points) > 0:
                    x, y = next_points[0].ravel()
                    if self.evaluation_active and self.prev_point is not None:
                        dist = np.linalg.norm([x - self.prev_point[0], y - self.prev_point[1]]) #calculate euclidean distance
                        self.point_motion.append(dist)

                    self.prev_point = (x, y)

                    # Check if the point is out of bounds
                    if 0 <= x < w and 0 <= y < h:
                        # Point is within bounds
                        self.tracked_points = next_points
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                    else:
                        # Point is out of bounds, reset tracking
                        print("Tracked point out of bounds. Resetting tracking.")
                        self.tracked_points = None

            else:
                # Invalid points, reset tracking
                print("Optical flow returned invalid points. Resetting tracking.")
                self.tracked_points = None


            self.prev_frame_gray = cur_gray_frame.copy()
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

        if hasattr(self, "total_frames") and self.total_frames > 0:
            print("\n=== MP + Optical Flow Evaluation ===")
            print(f"Avg FPS : {sum(self.fps_log)/len(self.fps_log):.2f}")
            print(f"TSR (%) : {self.success_frames/self.total_frames*100:.2f}")
            if self.point_motion:
                print(f"Tracking Stability : {sum(self.point_motion)/len(self.point_motion):.2f}")

        duration = time.time() - self.eval_start_time
        print(f"Duration (s) : {duration:.2f}")



if __name__ == "__main__":
    gui = WebcamController()
    gui.run_app()
    
