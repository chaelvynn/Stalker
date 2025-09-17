import os
import cv2
import numpy as np
import face_recognition
from tkinter import Tk, Label, Button, Frame, StringVar, OptionMenu
from PIL import Image, ImageTk
import threading as thread
import facerec_mediapipe as mp

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

def calculate_confidence(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        linear_val = (1.0 - face_distance) / (0.1 - face_match_threshold)
        return max(0.0, min(1.0, linear_val)) * 100
    else:
        linear_val = (1.0 - face_distance) / (face_match_threshold - 0.1)
        return max(0.0, min(1.0, linear_val)) * 100

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

class WebcamController:
    def __init__(self):
        self.root = Tk()
        self.root.title("Webcam Controller - Tkinter")
        self.root.minsize(800, 600)


        self.prev_tick = cv2.getTickCount()
        self.curr_fps = 0.0

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

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)



        self.FOCAL_LENGTH = 800  
        self.KNOWN_FACE_WIDTH = 16 

        self.prev_frame_gray = None
        self.tracked_points = None
        self.current_name = "Disable"
        
        
        self.lk_params = dict(
                    winSize=(15, 15), 
                    maxLevel=3, 
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )

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
        ret, frame = None, None

         # Hitung FPS
        curr_tick = cv2.getTickCount()
        elapsed = (curr_tick - self.prev_tick) / cv2.getTickFrequency()
        self.curr_fps = 1.0 / elapsed if elapsed > 0 else 0
        self.prev_tick = curr_tick


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
            frame = cv2.resize(frame, (w, h))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cur_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Tampilkan FPS di frame
            cv2.putText(frame, f"FPS: {self.curr_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if self.tracked_points is None:
                selected_name = self.face_detection_var.get()
                self.current_name = selected_name
                if selected_name != "Disable":
                    if selected_name in self.known_face_names:
                        idx = self.known_face_names.index(selected_name)
                        target_encodings = [self.known_face_encodings[idx]]
                        target_names = [self.known_face_names[idx]]
                    else:
                        target_encodings = self.known_face_encodings
                        target_names = self.known_face_names

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

                    
                        if name != "Unknown":

                         
                            mediapipe_results = self.face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                            if mediapipe_results.detections:
                                for detection in mediapipe_results.detections:
                                    bbox = detection.location_data.relative_bounding_box
                                    mp_x, mp_y = int(bbox.xmin * w), int(bbox.ymin * h)
                                    mp_w, mp_h = int(bbox.width * w), int(bbox.height * h)
                                    mp_center = (mp_x + mp_w // 2, mp_y + mp_h // 2)

                                    # Bandingkan titik tengah mediapipe dan face recognition
                                    if abs(mp_center[0] - (left + right)//2) < 50 and abs(mp_center[1] - (top + bottom)//2) < 50:
                                        # Hanya track jika overlap dengan wajah yang dikenali
                                        self.tracked_points = np.array([[[mp_center[0], mp_center[1]]]], dtype=np.float32)
                                        cv2.circle(frame, mp_center, 5, (255, 0, 255), -1)  



                            face_width_pixels = right - left
                            distance = self.calculate_distance(face_width_pixels) 

                            self.tracked_points = np.array([[[(left + right) // 2, (top + bottom) // 2]]], dtype=np.float32)
                            x, y = self.tracked_points[0].ravel()
                            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

                    
                            movement = []
                            if distance > 200:
                                movement.append("Move forward")
                            elif distance < 190:
                                movement.append("Move backward")

             
                            face_center_x = (left + right) // 2
                            frame_center_x = w // 2
                            if face_center_x < frame_center_x - 50:
                                movement.append("Move left")
                            elif face_center_x > frame_center_x + 50:
                                movement.append("Move right")

                      
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


                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        label = f"{name} ({confidence:.2f}%) Distance: {distance:.2f} cm"
                        cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            
            elif self.tracked_points is not None:
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_frame_gray, cur_gray_frame, self.tracked_points, None, **self.lk_params
            )


                if next_points is not None and len(next_points) > 0:
                    x, y = next_points[0].ravel()


                    if 0 <= x < w and 0 <= y < h:
                        self.tracked_points = next_points
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                    else:
                        print("Tracked point out of bounds. Resetting tracking.")
                        self.tracked_points = None

            else:
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

if __name__ == "__main__":
    gui = WebcamController()
    gui.run_app()
