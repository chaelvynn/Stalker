from tkinter import Tk, Label, Button, Frame, StringVar, OptionMenu
import traceback
import cv2, face_recognition, os, pygame, time
from PIL import Image, ImageTk
from djitellopy import tello
from flight_commands import start_flying, stop_flying
import numpy as np
import threading as thread
import mediapipe as mp


class FaceRecognition:
    def __init__(self):
        pass

    def load_known_faces(self, directory):
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

    def calculate_confidence(self, face_distance, face_match_threshold=0.6):
        if face_distance > face_match_threshold:  # kalau ga mirip 0%
            linear_val = (1.0 - face_distance) / (0.1 - face_match_threshold)
            return max(0.0, min(1.0, linear_val)) * 100
        else:  # kalau mirip 100%
            linear_val = (1.0 - face_distance) / (face_match_threshold - 0.1)
            return max(0.0, min(1.0, linear_val)) * 100

    def recognize_faces(self, frame, known_face_encodings, known_face_names):
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
                confidence = self.calculate_confidence(face_distances_current[best_match_index])
                distance = face_distances_current[best_match_index]  # Set distance

            face_names.append(name)
            face_confidences.append(confidence)
            face_distances.append(distance)

        return face_locations, face_names, face_confidences, face_distances

class JoystickController:
    def __init__(self, drone):
        self.drone = drone
        self.is_flying = False
        self.is_landing = False
        self.joystick_running = False
        self.available = False

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            print("[INFO] No joystick connected.")
            return
            # raise Exception("No joystick connected")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.available = True
        print(f"[INFO] Joystick connected: {self.joystick.get_name()}")

    def scale_axis(self, value, deadzone=0.1):
        if abs(value) < deadzone:
            return 0
        return int(value * 100)
    
    def joystick_control(self):
        last_roll, last_pitch, last_throttle, last_yaw = 0, 0, 0, 0
        self.joystick_running = True

        while self.joystick_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

                if event.type == pygame.JOYBUTTONDOWN:
                    # Button START to take off
                    if event.button == 7 and not self.is_flying:
                        print("Takeoff")
                        self.drone.takeoff()
                        self.is_flying = True
                        self.is_landing = False
                    
                    # Button SELECT to land
                    elif event.button == 6 and self.is_flying and not self.is_landing:
                        print("Landing")
                        self.drone.land()
                        self.is_landing = True
                        self.is_flying = False

                    # Button X to emergency stop the drone
                    elif event.button == 0:
                        print("EMERGENCY STOP!")
                        self.drone.emergency()
                        self.is_flying = False
                        self.is_landing = False
                
            #Axis control
            # Right axis:
            roll     = self.scale_axis(self.joystick.get_axis(2))    # left/right (X)
            pitch    = -self.scale_axis(self.joystick.get_axis(3))   # forward/ backward (Z)
            # Left axis:       
            throttle = -self.scale_axis(self.joystick.get_axis(1))   # up/down (Y)
            yaw      = self.scale_axis(self.joystick.get_axis(0))    # yaw (turn left/ turn right)


            if roll!= 0 or pitch !=0 or throttle != 0 or yaw != 0:

                self.drone.send_rc_control(roll, pitch, throttle, yaw)
                last_roll, last_pitch, last_throttle, last_yaw = roll, pitch, throttle, yaw

                # Print commands
                if roll != 0:
                    if roll > 0:
                        print("[C]Moving Right:", roll)
                    elif roll < 0:
                        print("[C]Moving Left:", roll)
                if pitch != 0:
                    if pitch > 0:
                        print("[C]Moving Forward:", pitch)
                    elif roll < 0:
                        print("[C]Moving Backward:", pitch)
                if throttle != 0:
                    if throttle > 0:
                        print("[C]Moving Up:", throttle)
                    elif throttle < 0:
                        print("[C]Moving Down:", throttle)
                if yaw != 0:
                    if yaw > 0:
                        print("[C]Yaw Right:", yaw)
                    elif yaw < 0:
                        print("[C]Yaw Left:", yaw)
            elif roll == pitch == throttle == yaw == 0 and (last_roll != 0 or last_pitch != 0 or last_throttle != 0 or last_yaw != 0):
                self.drone.send_rc_control(0, 0, 0, 0)
                last_roll, last_pitch, last_throttle, last_yaw = 0, 0, 0, 0
                print("Stopping movement")

            time.sleep(0.1)

    def start(self):
        if not self.available:
            print("[INFO] Joystick not available.")
            return
        
        if not self.joystick_running:
            t5 = thread.Thread(target=self.joystick_control, daemon=True)
            t5.start()
            self.joystick_running = True
            print("[INFO] Joystick control thread started.")

    def stop(self):
        self.joystick_running = False
        print("[INFO] Joystick control thread stopped.")

class KeyboardController:
    def __init__(self, drone, input_frame):
        self.drone = drone
        self.input_frame = input_frame
    
    def bind_keys(self):
        f = self.input_frame

        """CONTROL BY KEYBOARD"""
        # Bind the key presses with to the flight commands by associating them with a direction to travel.
        f.bind('<KeyPress-w>', lambda event: start_flying(event, 'forward', self.drone, 100))
        f.bind('<KeyRelease-w>', lambda event: stop_flying(event, self.drone))

        f.bind('<KeyPress-a>', lambda event: start_flying(event, 'left', self.drone, 100))
        f.bind('<KeyRelease-a>', lambda event: stop_flying(event, self.drone))

        f.bind('<KeyPress-s>', lambda event: start_flying(event, 'backward', self.drone, 100))
        f.bind('<KeyRelease-s>', lambda event: stop_flying(event, self.drone))

        f.bind('<KeyPress-d>', lambda event: start_flying(event, 'right', self.drone, 100))
        f.bind('<KeyRelease-d>', lambda event: stop_flying(event, self.drone))

        f.bind('<KeyPress-Up>', lambda event: start_flying(event, 'upward', self.drone, 100))
        f.bind('<KeyRelease-Up>', lambda event: stop_flying(event, self.drone))

        f.bind('<KeyPress-Down>', lambda event: start_flying(event, 'downward', self.drone, 100))
        f.bind('<KeyRelease-Down>', lambda event: stop_flying(event, self.drone))

        f.bind('<KeyPress-q>', lambda event: start_flying(event, 'yaw_left', self.drone, 100))
        f.bind('<KeyRelease-q>', lambda event: stop_flying(event, self.drone))

        f.bind('<KeyPress-e>', lambda event: start_flying(event, 'yaw_right', self.drone, 100))
        f.bind('<KeyRelease-e>', lambda event: stop_flying(event, self.drone))

class Autonomus:
    def __init__(self, drone, face_recognizer, known_face_encodings, known_face_names):
        self.drone = drone
        self.face_recognizer = face_recognizer
        self.known_face_encodings = known_face_encodings
        self.known_face_names = known_face_names

        self.FOCAL_LENGTH = 800
        self.KNOWN_FACE_WIDTH = 16

        #States
        self.prev_frame_gray = None
        self.tracked_points = None
        self.current_name = "Disable"
        self.old_distance = 0.0
        self.isPositionRight = 2
        self.prev_move = ""
        self.movement = []

        #Metrics
        self.total_frames = 0
        self.success_frames = 0
        self.prev_point = None
        self.point_motion = [] 
        self.tracking_start_time = None
        self.tracking_durations = []
        self.is_tracking_active = False
        self.evaluation_active = False
        self.tracking_attempt_active = False
        self.autonomous_start_time = None
        self.autonomous_duration = 0.0
        self.autonomous_active = False



        #Mediapipe setup
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)

        self.lk_params = dict(
            winSize=(25, 25),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def calculate_distance(self, face_width_pixels):
        if face_width_pixels == 0:
            return 0.0
        return (self.KNOWN_FACE_WIDTH * self.FOCAL_LENGTH) / face_width_pixels
    
    def process_frame(self, frame, cur_gray_frame, selected_name, w, h):
        self._autonomous_logic(frame, cur_gray_frame, selected_name, w, h)
        self.prev_frame_gray = cur_gray_frame.copy()

    def _autonomous_logic(self, frame, cur_gray_frame, selected_name, w, h):
        # selected_name = self.face_detection_var.get()
        
        #Metric
        if self.tracking_attempt_active:
            self.total_frames += 1

        def read_movement():
            if self.movement:
                for movement in self.movement:
                    if self.isPositionRight == 0:
                        if self.prev_move == "Move forward":
                            start_flying(None, 'backward', self.drone, 10)
                            self.drone.send_control_command("rc 0 0 0 0")
                        elif self.prev_move == "Move backward":
                            start_flying(None, 'forward', self.drone, 10)
                            self.drone.send_control_command("rc 0 0 0 0")
                        elif self.prev_move == "Move left":
                            start_flying(None, 'right', self.drone, 10)
                            self.drone.send_control_command("rc 0 0 0 0")
                        elif self.prev_move == "Move right":
                            start_flying(None, 'left', self.drone, 10)
                            self.drone.send_control_command("rc 0 0 0 0")
                        elif self.prev_move == "Move upward":
                            start_flying(None, 'downward', self.drone, 10)
                            self.drone.send_control_command("rc 0 0 0 0")
                        elif self.prev_move == "Move downward":
                            start_flying(None, 'upward', self.drone, 10)
                            self.drone.send_control_command("rc 0 0 0 0")
                        self.prev_move = ""
                        break

                    if movement == "Move forward1":
                        start_flying(None, 'forward', self.drone, 20)
                    if movement == "Move forward2":
                        start_flying(None, 'forward', self.drone, 30)    
                    elif movement == "Move backward1":
                        start_flying(None, 'backward', self.drone, 20)
                    elif movement == "Move backward2":
                        start_flying(None, 'backward', self.drone, 30)    
                    elif movement == "Move left1":
                        start_flying(None, 'left', self.drone, 20)
                    elif movement == "Move left2":
                        start_flying(None, 'left', self.drone, 30)
                    elif movement == "Move right1":
                        start_flying(None, 'right', self.drone, 20)
                    elif movement == "Move right2":
                        start_flying(None, 'right', self.drone, 30)
                    elif movement == "Move upward":
                        start_flying(None, 'upward', self.drone, 30)
                    elif movement == "Move downward":
                        start_flying(None, 'downward', self.drone, 30)
                    self.prev_move = movement
                self.movement.clear()


        t3 = thread.Thread(target=read_movement)
        t3.start()
        t3.join()

        # --- FACE RECOGNITION PHASE ---
        if self.tracked_points is None:
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
                face_locations, face_names, face_confidences, face_distances = \
                    self.face_recognizer.recognize_faces(frame, target_encodings, target_names)

            t2 = thread.Thread(target=recognize_faces_thread)
            t2.start()
            t2.join()

            for (top, right, bottom, left), name, confidence, distance in zip(
                    face_locations, face_names, face_confidences, face_distances):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                if name != "Unknown":
                    #Metric
                    self.tracking_attempt_active = True
                    if not self.is_tracking_active:
                        self.tracking_start_time = time.time()
                        self.is_tracking_active = True
                    
                    # #Metric TSR
                    if not self.evaluation_active:
                        self.evaluation_active = True
                        # self.total_frames += 1 #Metric TSR
                        # self.success_frames += 1

                    self.movement.clear()
                    face_width_pixels = right - left
                    distance = self.calculate_distance(face_width_pixels)
                    self.old_distance = distance
                    self.tracked_points = np.array(
                        [[[(left + right) // 2, (top + bottom) // 2]]], dtype=np.float32
                    )
                    x, y = self.tracked_points[0].ravel()
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

                    # Distance
                    if distance > 85 and distance < 95:
                        self.movement.append("Move forward1")
                        self.isPositionRight = 1
                    elif distance > 95:
                        self.movement.append("Move forward2")
                        self.isPositionRight = 1
                    elif distance < 60 and distance > 50:
                        self.movement.append("Move backward1")
                        self.isPositionRight = 1
                    elif distance <= 50:
                        self.movement.append("Move backward2")
                        self.isPositionRight = 1

                    # Horizontal alignment
                    face_center_x = (left + right) // 2
                    frame_center_x = w // 2
                    if face_center_x < frame_center_x - 120:
                        self.movement.append("Move left")
                        self.isPositionRight = 1
                    elif face_center_x > frame_center_x + 120:
                        self.movement.append("Move right")
                        self.isPositionRight = 1

                    # Vertical alignment
                    face_center_y = (top + bottom) // 2
                    frame_center_y = h // 2
                    if face_center_y < frame_center_y - 90:
                        self.movement.append("Move upward")
                        self.isPositionRight = 1
                    elif face_center_y > frame_center_y + 90:
                        self.movement.append("Move downward")
                        self.isPositionRight = 1
                    
                    if self.isPositionRight == 0:
                        self.movement.clear()

                    if self.isPositionRight == 1:
                        self.isPositionRight = 2
                
                    if self.isPositionRight == 2:
                        self.isPositionRight = 0
                        

                    if self.isPositionRight == 0:
                        self.isPositionRight = 2

                # Draw box + label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                label = f"{name} ({confidence:.2f}%) Dist: {distance:.2f} cm"
                cv2.putText(frame, label, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # --- OPTICAL FLOW PHASE ---
        elif self.tracked_points is not None:
            #Metric
            if self.evaluation_active:
                self.total_frames += 1 
                self.success_frames += 1

            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_frame_gray, cur_gray_frame, self.tracked_points, None, **self.lk_params
            )
            self.tracked_points = next_points

            if next_points is not None and len(next_points) > 0:
                x, y = next_points[0].ravel()
                self.success_frames += 1 #Metric

                #Metric
                if self.prev_point is not None:
                    # dx = x - self.prev_point[0]
                    # dy = y - self.prev_point[1]
                    # dist = np.sqrt(dx**2 + dy**2)  # EUCLIDEAN
                    dist = np.linalg.norm([x - self.prev_point[0], y - self.prev_point[1]])
                    self.point_motion.append(dist)

                self.prev_point = (x, y)



                results = self.face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                faces = []
                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        xmin, ymin = int(bboxC.xmin * w), int(bboxC.ymin * h)
                        width, height = int(bboxC.width * w), int(bboxC.height * h)
                        faces.append((xmin, ymin, xmin + width, ymin + height))

                matched_face = None
                for face in faces:
                    left, top, right, bottom = face
                    if left <= x <= right and top <= y <= bottom:
                        matched_face = face
                        break

                if matched_face:
                    face_width = matched_face[2] - matched_face[0]
                    distance = self.calculate_distance(face_width)

                    print(f"Estimated Distance: {distance:.2f} cm")
                    self.movement.clear()
                    
                    if distance > 85 and distance < 95:
                        self.movement.append("Move forward1")
                        self.isPositionRight = 1
                    elif distance > 95:
                        self.movement.append("Move forward2")
                        self.isPositionRight = 1
                    elif distance < 60 and distance > 50:
                        self.movement.append("Move backward1")
                        self.isPositionRight = 1
                    elif distance <= 50:
                        self.movement.append("Move backward2")
                        self.isPositionRight = 1

                    # Alignment x-axis
                    face_center_x = (left + right) // 2
                    frame_center_x = w // 2
                    if x < frame_center_x - 120 and x > frame_center_x - 150:
                        self.movement.append("Move left1")
                        self.isPositionRight = 1
                    elif x < frame_center_x - 150:
                        self.movement.append("Move left2")
                        self.isPositionRight = 1
                    elif face_center_x > frame_center_x + 120 and x < frame_center_x + 150:
                        self.movement.append("Move right1")
                        self.isPositionRight = 1
                    elif face_center_x > frame_center_x + 150:
                        self.movement.append("Move right2")
                        self.isPositionRight = 1

                    # Alignment y-axis
                    face_center_y = (top + bottom) // 2
                    frame_center_y = h // 2
                    if y < frame_center_y - 90:
                        self.movement.append("Move upward")
                        self.isPositionRight = 1
                    elif face_center_y > frame_center_y + 90:
                        self.movement.append("Move downward")
                        self.isPositionRight = 1

                    if self.isPositionRight == 1:
                        self.isPositionRight = 2
                    
                    if self.isPositionRight == 2:
                        self.isPositionRight = 0
                        

                    if self.isPositionRight == 0:
                        self.isPositionRight = 2  

                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

                else:
                    print("Tracked point lost. Resetting...")
                    self.tracked_points = None
                    self.movement.clear()

                    #Metric
                    if self.is_tracking_active:
                        duration = time.time() - self.tracking_start_time
                        self.tracking_durations.append(duration)
                        self.is_tracking_active = False
                        self.evaluation_active = False


            else:
                print("Invalid optical flow points. Resetting...")
                self.tracked_points = None
                self.movement.clear()

                #Metric
                self.tracking_attempt_active = False
                self.evaluation_active = False


class VideoStream:
    def __init__(self, cap, cap_lbl, face_detection_var, joystick_controller, autonomous):
        self.cap = cap
        self.cap_lbl = cap_lbl
        self.face_detection_var = face_detection_var
        self.joystick_controller = joystick_controller
        self.autonomous = autonomous

        self.is_running = False

        # Metrics init
        self.prev_time = time.time()
        self.fps = 0.0
        self.fps_list = []
        self.autonomous_active = False

    def start(self):
        self.is_running = True
        self.cap_lbl.after(0, self.video_stream)

    def stop(self):
        self.is_running = False

    def video_stream(self):
        try:
            if not self.is_running:
                return
            h, w = 480, 720
            frame = None

            # FRAME READING
            if self.autonomous.prev_frame_gray is None:
                frame = self.cap.frame
                if frame is None:
                    self.cap_lbl.after(10, self.video_stream)
                    return
                self.autonomous.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            def read_frame():
                nonlocal frame
                frame = self.cap.frame

            # t1 = thread.Thread(target=read_frame)
            # t1.start()
            # t1.join()
            thread.Thread(target=read_frame, daemon=True).start()

            if frame is None:
                self.cap_lbl.after(10, self.video_stream)
                return

            if frame is not None:
                frame = cv2.resize(frame, (w, h))
                cur_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                selected_name = self.face_detection_var.get()

                if selected_name == "Disable":

                    #Metric
                    if self.autonomous.autonomous_active:
                        self.autonomous.autonomous_duration += (time.time() - self.autonomous.autonomous_start_time)
                        self.autonomous.autonomous_active = False
                        self.autonomous.autonomous_start_time = None
                    self.autonomous_active = False


                    self.autonomous_active = False #Metric
                    if self.joystick_controller.available:
                        if not self.joystick_controller.joystick_running:
                            self.joystick_controller.start()

                else:
                    #Metric
                    self.autonomous_active = True #Metric
                    if not self.autonomous.autonomous_active:
                        self.autonomous.autonomous_start_time = time.time()
                        self.autonomous.autonomous_active = True

                    if selected_name != self.autonomous.current_name:
                        self.autonomous.current_name = selected_name
                        self.autonomous.tracked_points = None
                        self.autonomous.movement.clear()

                    # Run autonomous logic
                    self.autonomous.process_frame(frame, cur_gray_frame, selected_name, w, h)

                #Metric
                cv2.putText(frame, f"FPS: {self.fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)


                # Render ke GUI
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.cap_lbl.imgtk = imgtk
                self.cap_lbl.configure(image=imgtk)

            #Metric
            current_time = time.time()
            self.fps = 1 / (current_time - self.prev_time)
            self.prev_time = current_time
            if self.autonomous_active:
                self.fps_list.append(self.fps)

            # Schedule next frame
            self.cap_lbl.after(10, self.video_stream)
        
        except Exception as e:
            print(f"Error in video stream: {e}")
            self.stop()
            self.drone.send_control_command("rc 0 0 0 0")


class DroneController:
    def __init__(self):
        # GUI Setup
        self.root = Tk()
        self.root.title("Drone Keyboard Controller - Tkinter")
        self.root.minsize(800, 600)
        self.input_frame = Frame(self.root)

        # Drone Setup
        self.drone = tello.Tello()
        self.drone.connect()
        self.drone.streamon()
        self.cap = self.drone.get_frame_read()
        print("Battery: ", self.drone.get_battery(), "%")
        self.drone.speed = 100

        # Controllers
        self.keyboard_controller = KeyboardController(self.drone, self.input_frame)
        self.joystick_controller = JoystickController(self.drone)

        self.joystick_running = True
        
        # Drone state
        self.is_flying = False
        self.is_landing = False

        # Face Recognition Setup
        faces_dir = "faces"
        self.face_recognizer = FaceRecognition()
        self.known_face_encodings, self.known_face_names = self.face_recognizer.load_known_faces(faces_dir)

        # GUI Components
        self.cap_lbl = Label(self.root)
        self.button_frame = Frame(self.root)
        self.demo_button = Button(self.button_frame, text="Takeoff/Land", command=self.takeoff_land)

        self.face_detection_var = StringVar(self.root)
        self.face_detection_var.set("Disable")
        self.face_detection_menu = OptionMenu(self.button_frame, self.face_detection_var, "Disable", *self.known_face_names)


        # Autonomous Setup        
        self.autonomous = Autonomus(
            drone=self.drone,
            face_recognizer=self.face_recognizer,
            known_face_encodings=self.known_face_encodings,
            known_face_names=self.known_face_names
        )
        
        self.video_stream = VideoStream(
            cap=self.cap,
            cap_lbl=self.cap_lbl,
            face_detection_var=self.face_detection_var,
            joystick_controller=self.joystick_controller,
            autonomous=self.autonomous
        )

    
    def takeoff_land(self):
        if not self.is_flying:
            print("Takeoff")
            self.drone.takeoff()
            self.is_flying = True
            self.is_landing = False
        else:
            print("Landing")
            self.drone.land()
            self.is_landing = True
            self.is_flying = False


    def run_app(self):
        try:
            self.keyboard_controller.bind_keys()

            self.input_frame.pack()
            self.input_frame.focus_set()
            self.cap_lbl.pack(anchor="center", pady=15)
            self.demo_button.pack(side='left', padx=10)
            self.face_detection_menu.pack(side='left')
            self.button_frame.pack(anchor="center", pady=10)
            
            self.joystick_controller.start()
            self.video_stream.start()
            self.root.mainloop()

        except Exception as e:
            print(f"Error running the application: {e}")
            traceback.print_exc()
            self.video_stream.stop()
        finally:
            self.cleanup()
            print("Cleaning up...")



  
    def dummy_function(self, event, key):
        print(f"Key {key} pressed")


    def cleanup(self) -> None:
        self.video_stream.stop()
        self.drone.streamoff()

        auto = self.autonomous
        if auto.autonomous_active and auto.autonomous_start_time is not None:
            auto.autonomous_duration += time.time() - auto.autonomous_start_time
            auto.autonomous_active = False


        print("\n=== EVALUATION METRICS ===")

        # FPS
        if self.video_stream.fps_list:
            avg_fps = sum(self.video_stream.fps_list) / len(self.video_stream.fps_list)
            print(f"Average FPS       : {avg_fps:.2f}")

        # TSR
        auto = self.autonomous
        if auto.total_frames > 0:
            tsr = (auto.success_frames / auto.total_frames) * 100
            print(f"Tracking Success  : {tsr:.2f}%")

        # Tracking Stability
        if auto.point_motion:
            stability = sum(auto.point_motion) / len(auto.point_motion)
            print(f"Tracking Stability: {stability:.2f} px")

        # Tracking Duration
        print(f"Tracking Duration: {auto.autonomous_duration:.2f} sec")

        print("===== END METRICS =====\n")
        print("Cleanup complete.")


if __name__ == "__main__":
    gui = DroneController()
    gui.run_app()
